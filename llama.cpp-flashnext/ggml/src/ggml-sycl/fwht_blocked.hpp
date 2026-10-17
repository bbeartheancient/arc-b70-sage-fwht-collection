#ifndef GGML_SYCL_FWHT_BLOCKED_HPP
#define GGML_SYCL_FWHT_BLOCKED_HPP

#include "common.hpp"

// Blockwise SIGNED normalised Sylvester Walsh-Hadamard, for ternary weights
// whose stored form is pre-rotated by H.D (D a random +-1 diagonal supplied in
// GGUF metadata). See SIMULATION-PLAN.md 4.1.1 and R0.7.1/R0.7.3.
//
//     y[b] = (1/sqrt(BLOCK)) * H_BLOCK * (signs[b] .* x[b])   for each block b
//
// Why this is separate from ggml_sycl_op_fwht (fwht.hpp), which it complements
// rather than replaces:
//
//   * fwht() serves KV-cache rotation. Its n is a head dimension -- a power of
//     two, 64..512 -- and it transforms the WHOLE row in one N-wide butterfly.
//   * This serves weight dequant. Its n is the *input width* of a weight
//     (5120 / 6144 / 17408 for Ternary-Bonsai-2-27B), which is NOT a power of
//     two, so a single N-wide WHT is not even defined. The model therefore
//     declares a block size (1024) and the transform is BLOCKWISE, at one
//     sign vector per block.
//
// Conflating the two is the mistake that would produce a plausible wrong
// answer: fwht()'s `default: return false` is documented as "fall through to
// the ordinary mat-mul", so a width it cannot serve is silently not applied.
//
// Register pressure. fwht_kernel<N> holds N/WARP_SIZE floats per lane. On this
// build WARP_SIZE is 16 (GGML_SYCL_TARGET=INTEL), so a whole 1024 block would
// be 64 registers per lane and spill. H.1024 = H.512 (x) H.2, so each 1024
// block is done as two independent 512 transforms:
//
//     y_lo = H.512 (x_lo + x_hi)      y_hi = H.512 (x_lo - x_hi)
//
// which peaks at 512/16 = 32 registers per lane -- the same figure the
// existing, production 512 path already uses. x is read twice; that is the
// trade, and it is the same trade the register analysis predicted.

namespace ggml {

// Core, free of any ggml context so it can be unit-tested against a CPU
// reference with a plain sycl::queue. `block` must be a power of two and
// `nrows * nblocks * block` elements are processed, one independent transform
// per (row, block). Returns false if the request is not one this can serve.
bool sycl_fwht_blocked(sycl::queue & q,
                       const float * src, float * dst,
                       const int8_t * signs,
                       int64_t nrows, int nblocks, int block);

} // namespace ggml

// The ggml-facing wrapper below is a thin adapter that resolves the stream from
// the backend context and does the shape/contiguity guards. It is compiled out
// for the standalone test (which defines FWHT_BLOCKED_NO_GGML_WRAPPER) so the
// kernel can be proved against a CPU reference without linking all of libggml.
// The part worth testing is the core, not the adapter.
#ifndef FWHT_BLOCKED_NO_GGML_WRAPPER
bool ggml_sycl_op_fwht_blocked(ggml_backend_sycl_context & ctx,
                               const ggml_tensor * src, ggml_tensor * dst,
                               int nblocks, const int8_t * signs);
#endif

#endif // GGML_SYCL_FWHT_BLOCKED_HPP
