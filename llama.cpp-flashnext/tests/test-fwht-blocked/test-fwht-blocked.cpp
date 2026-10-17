// Standalone validation of the blockwise signed Hadamard SYCL kernel against a
// CPU reference. Deliberately does NOT go through llama-server: the failure this
// guards against is silent (a transform that is simply not applied), so the
// kernel has to be provable on its own before it is wired into a dispatch.
//
// The CPU reference is the STRAIGHTFORWARD blocked form; the kernel uses the
// Kronecker form (two 512 halves). Their agreement is what validates the
// register-saving decomposition on real hardware, not just on paper.
//
// Optionally takes a file of +1/-1 integers to use as the sign vector, so the
// real Ternary-Bonsai sign block can be replayed:
//     ./test-fwht-blocked <signs-file> <block> <nblocks> <nrows>

#include "fwht_blocked.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0, g_checks = 0;
static void check(bool ok, const char *what, double d = 0.0) {
    g_checks++;
    if (ok) printf("PASS  %s  (worst |d|=%.3g)\n", what, d);
    else    { g_fail++; printf("FAIL  %s  (worst |d|=%.3g)\n", what, d); }
}

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }

// CPU reference: per block, sign-multiply then a full in-place FWHT, normalised.
static void ref_blocked(const std::vector<float> & x, std::vector<float> & y,
                        const std::vector<int8_t> & s, int nrows, int nblocks, int block) {
    // Refuse rather than write out of bounds: a short destination is exactly the
    // bug that made this file lie for an hour.
    if ((int) x.size() < nrows * nblocks * block || (int) y.size() < nrows * nblocks * block ||
        (int) s.size() < nrows * nblocks * block) {
        fprintf(stderr, "ref_blocked: size mismatch\n");
        exit(2);
    }
    const float scale = (float)(1.0 / std::sqrt((double) block));
    for (int r = 0; r < nrows; r++) {
        for (int b = 0; b < nblocks; b++) {
            const size_t base = (size_t)r * nblocks * block + (size_t)b * block;
            std::vector<float> t(block);
            for (int i = 0; i < block; i++) t[i] = x[base + i] * (float) s[base + i];
            for (int h = 1; h < block; h *= 2)
                for (int i = 0; i < block; i += 2 * h)
                    for (int j = i; j < i + h; j++) {
                        const float u = t[j], v = t[j + h];
                        t[j] = u + v; t[j + h] = u - v;
                    }
            for (int i = 0; i < block; i++) y[base + i] = t[i] * scale;
        }
    }
}

static bool read_signs(const char *path, std::vector<int8_t> & out, size_t want) {
    FILE * f = fopen(path, "r");
    if (!f) return false;
    // clear() is load-bearing. Without it a pre-sized caller's vector makes
    // `out.size() < want` false on entry, the loop never runs, the function
    // returns TRUE having read nothing, and the caller believes it has the real
    // signs. That is not hypothetical: it happened, and the parity check passed
    // vacuously on an all-zero sign vector until the checksum was compared
    // against an independent one.
    out.clear();
    long v = 0;
    while ((size_t)out.size() < want) {
        if (fscanf(f, "%ld", &v) != 1) break;
        out.push_back((int8_t)(v > 0 ? 1 : -1));
    }
    fclose(f);
    return out.size() == want;
}

int main(int argc, char ** argv) {
    int  block   = 1024, nblocks = 5, nrows = 7;
    const char * signs_path = nullptr;
    if (argc >= 5) {
        signs_path = argv[1]; block = atoi(argv[2]); nblocks = atoi(argv[3]); nrows = atoi(argv[4]);
    } else if (argc >= 2) {
        signs_path = argv[1];
    }

    sycl::queue q;
    printf("=== test-fwht-blocked ===\n");
    printf("  device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    {
        const auto sgs = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
        printf("  sub-group sizes supported: %u ->", (unsigned) sgs.size());
        for (auto v : sgs) printf(" %u", (unsigned) v);
        printf("\n");
    }
    printf("  block=%d nblocks=%d nrows=%d total=%d\n", block, nblocks, nrows, block * nblocks * nrows);

    const int n = block * nblocks * nrows;

    std::vector<int8_t> signs((size_t)n, 0);
    bool from_file = false;
    if (signs_path && read_signs(signs_path, signs, (size_t)n)) {
        from_file = true;
        printf("  signs: %d values read from %s\n", n, signs_path);
    } else {
        if (signs_path) {
            printf("  signs: COULD NOT READ %d values from %s -- falling back to synthetic\n",
                   n, signs_path);
        }
        uint32_t s = 0x9E3779B9u;
        for (int i = 0; i < n; i++) signs[(size_t)i] = (lcg(&s) & 1u) ? (int8_t) 1 : (int8_t) -1;
    }
    {
        // checksum + head/tail, so "which signs ran" is answerable from outside
        uint32_t h = 2166136261u; int pos = 0, neg = 0;
        for (int i = 0; i < n; i++) {
            h = (h ^ (uint32_t)(int)(signs[(size_t)i] + 1)) * 16777619u;
            if (signs[(size_t)i] > 0) pos++; else neg++;
        }
        printf("  signs: source=%s first=%d last=%d  +1:%d -1:%d  fnv1a=%08x\n",
               from_file ? "FILE" : "SYNTHETIC", (int) signs[0], (int) signs[(size_t)n - 1],
               pos, neg, h);
    }

    // A degenerate sign vector would make the parity comparison below trivially
    // true (both sides produce zero), so it is asserted rather than assumed.
    {
        int pos = 0, neg = 0;
        for (int i = 0; i < n; i++) (signs[(size_t)i] > 0 ? pos : neg)++;
        check(pos > 0 && neg > 0, "sign vector is non-degenerate (has BOTH +1 and -1)");
    }

    std::vector<float> x((size_t)n), y((size_t)n, 0.0f);
    uint32_t s2 = 12345u;
    for (int i = 0; i < n; i++) x[(size_t)i] = (float)((double)((int)(lcg(&s2) % 2001) - 1000) / 1000.0);

    // SIZED. Declaring this empty and then writing y[base+i] into it is
    // out-of-bounds, and the resulting undefined behaviour made the headline
    // parity check compare the kernel against junk: it passed on a correct
    // kernel and MISSED three of four injected faults. The check that did catch
    // a fault was the only one not of the form kernel-vs-reference. Every
    // reference buffer in this file is sized explicitly for that reason.
    std::vector<float> ref((size_t)n, 0.0f);
    ref_blocked(x, ref, signs, nrows, nblocks, block);

    bool served = ggml::sycl_fwht_blocked(q, x.data(), y.data(), signs.data(), nrows, nblocks, block);
    check(served, "kernel accepted the request (did not silently refuse)");
    if (!served) { printf("  cannot continue\n"); return 1; }
    q.wait_and_throw();

    double worst = 0.0; int worst_i = -1;
    for (int i = 0; i < n; i++) {
        const double d = std::fabs((double) y[(size_t)i] - (double) ref[(size_t)i]);
        if (d > worst) { worst = d; worst_i = i; }
    }
    // float32 accumulate over `block` terms; 1e-4 is generous but still ~4
    // orders of magnitude below what any real fault produces.
    check(worst < 1e-4, "SYCL Kronecker form matches the CPU blocked reference", worst);
    if (worst >= 1e-4 && worst_i >= 0) printf("      worst at i=%d  got %.9g want %.9g\n",
                                               worst_i, y[(size_t)worst_i], ref[(size_t)worst_i]);

    // Bisection: with an all-+1 sign vector the kernel reduces to the plain
    // normalised H_1024, so this isolates {halves, butterfly, layout} from the
    // sign path. If this passes and the signed one fails, the bug is in how the
    // signs are applied; if it fails too, the bug is structural.
    {
        std::vector<int8_t> allpos((size_t)n, 1);
        std::vector<float> ry((size_t)n, 0.0f), rref((size_t)n, 0.0f);
        ref_blocked(x, rref, allpos, nrows, nblocks, block);
        bool ok = ggml::sycl_fwht_blocked(q, x.data(), ry.data(), allpos.data(), nrows, nblocks, block);
        q.wait_and_throw();
        double w2 = 0.0;
        for (int i = 0; i < n; i++) w2 = std::max(w2, std::fabs((double) ry[(size_t)i] - (double) rref[(size_t)i]));
        check(ok && w2 < 1e-4, "UNSIGNED parity (isolates halves/butterfly from signs)", w2);
        if (!(w2 < 1e-4)) {
            printf("      (unsigned worst at i=0: got %.9g want %.9g)\n",
                   (double) ry[0], (double) rref[0]);
        }
    }

    // And nblocks=1: a single 1024 block, so the half-split is the only thing
    // under test with no repetition to average over.
    {
        const int nb1 = 1;
        std::vector<float> y1((size_t)block * nrows, 0.0f);
        std::vector<float> r1((size_t)block * nrows, 0.0f);
        std::vector<int8_t> s1(signs.begin(), signs.begin() + block * nrows);
        ref_blocked(x, r1, s1, nrows, nb1, block);
        bool ok = ggml::sycl_fwht_blocked(q, x.data(), y1.data(), s1.data(), nrows, nb1, block);
        q.wait_and_throw();
        double w3 = 0.0;
        for (int i = 0; i < block * nrows; i++)
            w3 = std::max(w3, std::fabs((double) y1[(size_t)i] - (double) r1[(size_t)i]));
        check(ok && w3 < 1e-4, "single-block parity (nblocks=1)", w3);
    }

    // The omission test: without the signs the answer must differ materially.
    std::vector<int8_t> ones((size_t)n, 1);
    std::vector<float> y2((size_t)n, 0.0f);
    bool ok2 = ggml::sycl_fwht_blocked(q, x.data(), y2.data(), ones.data(), nrows, nblocks, block);
    q.wait_and_throw();
    if (ok2) {
        double dworst = 0.0;
        for (int i = 0; i < n; i++) {
            const double d = std::fabs((double) y[(size_t)i] - (double) y2[(size_t)i]);
            if (d > dworst) dworst = d;
        }
        check(dworst > 1.0, "dropping the sign vector changes the on-device result", dworst);
    }

    // Non-power-of-two block must be REFUSED, never quietly wrong.
    check(!ggml::sycl_fwht_blocked(q, x.data(), y.data(), signs.data(), nrows, nblocks, 1000),
          "a non-power-of-two block is refused rather than silently mishandled");

    printf("--- %d checks, %d failures ---\n", g_checks, g_fail);
    printf(g_fail ? "FWHT_BLOCKED_TEST_FAIL\n" : "FWHT_BLOCKED_TEST_PASS\n");
    return g_fail ? 1 : 0;
}
