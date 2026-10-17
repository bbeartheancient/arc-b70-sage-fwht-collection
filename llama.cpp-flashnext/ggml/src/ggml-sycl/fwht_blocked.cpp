#include "fwht_blocked.hpp"

#include <cmath>

namespace ggml {

// One sub-group per (row, block). M is the INNER transform size, M = block/2.
//
// The two halves are processed in sequence through a compile-time-unrolled
// loop, so only M/WARP_SIZE registers are live at once -- never block/WARP_SIZE.
// `x` is therefore read twice, which is the deliberate cost of bounding the
// register footprint (see the header).
template <int M>
static void fwht_blocked_kernel(const float * __restrict__ src,
                                float * __restrict__ dst,
                                const int8_t * __restrict__ signs,
                                const int64_t n_units,   // nrows * nblocks
                                const float scale,
                                const sycl::nd_item<2> & item) {
    static_assert(M % WARP_SIZE == 0, "inner size must be a whole number of sub-group widths");

    const sycl::sub_group sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();

    const int64_t unit = item.get_global_id(0);   // one (row, block) per sub-group
    if (unit >= n_units) {
        return;
    }

    constexpr int el_w = M / WARP_SIZE;

    const float * xb = src + unit * (2 * M);
    float       * yb = dst + unit * (2 * M);
    const int8_t * sb = signs + unit * (2 * M);

    float reg[el_w];

    // half == 0 -> lo = H.M (x_lo + x_hi) -> dst[0..M)
    // half == 1 -> hi = H.M (x_lo - x_hi) -> dst[M..2M)
#pragma unroll
    for (int half = 0; half < 2; half++) {
        const float sgn = half == 0 ? 1.0f : -1.0f;

#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const int i = j * WARP_SIZE + lane;
            // Multiplying by +-1 is a sign-bit flip; the compiler emits a cheap
            // conditional negate for `* (float)sb[i]`, not a general multiply.
            const float a = xb[i]              * (float) sb[i];
            const float c = xb[i + M]          * (float) sb[i + M];
            reg[j] = a + sgn * c;
        }

        // Stage 1: butterflies INSIDE the sub-group. These exchange values
        // between lanes, so they need the sub-group permute -- without this half
        // of the network the transform is incomplete and the result is wrong by
        // O(1), not by rounding. (Omitting it was a real bug here: it produced
        // |d| = 20.4 against the CPU reference while a pre-sized-reference bug
        // in the test was hiding it.)
#pragma unroll
        for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
            for (int j = 0; j < el_w; ++j) {
                const float val  = reg[j];
                const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);
                reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }

        // Stage 2: butterflies ACROSS registers.
#pragma unroll
        for (int h = WARP_SIZE; h < M; h *= 2) {
            const int step = h / WARP_SIZE;
#pragma unroll
            for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
                for (int k = 0; k < step; ++k) {
                    const float x = reg[j + k];
                    const float y = reg[j + k + step];
                    reg[j + k]        = x + y;
                    reg[j + k + step] = x - y;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            yb[half * M + j * WARP_SIZE + lane] = reg[j] * scale;
        }
    }
}

template <int M>
static void launch_fwht_blocked(
                                const float * src, float * dst,
                                const int8_t * signs,
                                const int64_t n_units,
                                const float scale,
                                dpct::queue_ptr stream) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_units + rows_per_block - 1) / rows_per_block;

    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_blocked_kernel<M>(src, dst, signs, n_units, scale, item);
                         });
}

bool sycl_fwht_blocked(sycl::queue & q,
                       const float * src, float * dst,
                       const int8_t * signs,
                       int64_t nrows, int nblocks, int block) {
    if (block <= 0 || nblocks <= 0 || nrows <= 0) {
        return false;
    }
    // A Sylvester WHT is only defined for a power-of-two length. block must be
    // one; the TOTAL width deliberately need not be, which is the whole point.
    if ((block & (block - 1)) != 0) {
        return false;
    }
    if (block < 2 * WARP_SIZE) {
        return false;
    }

    const int      m    = block / 2;
    const float    scale = 1.0f / std::sqrt((float) block);
    const int64_t  n_units = nrows * (int64_t) nblocks;
    dpct::queue_ptr stream = &q;

    switch (m) {
        // The inner size is what actually gets compiled, so dispatch here.
        // 512 is the case Ternary-Bonsai needs (block 1024) and the one the
        // existing production path already exercises; the smaller ones make the
        // kernel testable at shapes the CPU oracle can also compute directly.
        case 512:  launch_fwht_blocked<512>(src, dst, signs, n_units, scale, stream); return true;
        case 256:  launch_fwht_blocked<256>(src, dst, signs, n_units, scale, stream); return true;
        case 128:  launch_fwht_blocked<128>(src, dst, signs, n_units, scale, stream); return true;
        case 64:   launch_fwht_blocked<64>(src, dst, signs, n_units, scale, stream);  return true;
        case 32:   launch_fwht_blocked<32>(src, dst, signs, n_units, scale, stream);  return true;
        case 16:   launch_fwht_blocked<16>(src, dst, signs, n_units, scale, stream);  return true;
        default:
            // Explicitly refusing is the whole point. Returning true here, or
            // falling through to a plain mat-mul, is the silent-wrong-answer bug.
            return false;
    }
}

} // namespace ggml

#ifndef FWHT_BLOCKED_NO_GGML_WRAPPER
bool ggml_sycl_op_fwht_blocked(ggml_backend_sycl_context & ctx,
                               const ggml_tensor * src, ggml_tensor * dst,
                               int nblocks, const int8_t * signs) {
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src, dst)) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (signs == nullptr) {
        return false;
    }
    const int block = (int) src->ne[0];
    if (block <= 0 || block % nblocks != 0) {
        return false;
    }
    return ggml::sycl_fwht_blocked(*ctx.stream(), (const float *) src->data, (float *) dst->data,
                                   signs, ggml_nrows(src), nblocks, block);
}
#endif // FWHT_BLOCKED_NO_GGML_WRAPPER
