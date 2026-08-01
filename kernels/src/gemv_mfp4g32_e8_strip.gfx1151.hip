// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 decode-GEMV COMPUTE-STRIP probe, gfx1151-only.
//
// NOT A CORRECT GEMV. This is a diagnostic kernel: it keeps the EXACT memory
// access pattern of gemv_mfp4g32_e8.gfx1151.hip (same header read, same per-
// block E4M3 scale byte, same codeword reads, same all-8 x[] reads) but
// replaces the ~50-op E8 lattice decode + 8 per-element MACs with a trivial
// accumulate. A/B'd against the full kernel at identical shapes, the delta is
// the COMPUTE CEILING — the most any "cute math" (v_perm decode + DP4A) could
// buy. If this is ~= the full kernel, the GEMV is memory-bound and DP4A can't
// help; if it's much faster, decode-compute is the gate.
//
// Output is garbage — for tok/s timing only. gfx1151-gated; env HIPFIRE_E8_STRIP=1.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#if defined(__gfx1200__) || defined(__gfx1201__)
// gfx12 native OCP FP8 (radiowave Fp8Format::E4M3Ocp). Bit-identical to the
// scalar idiom over codes 1..=126 — the entire range e4m3_encode_roundup can
// emit; the full-domain mismatches are the NaN slot 0x7F and the negative
// half, neither reachable for a SCALE. See
// benchmarks/kernel-probes/e4m3-native-decode/.
__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    __hip_fp8_e4m3 v;
    v.__x = b;
    return (float)v;
}
#else
__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) {
        return 0.015625f * ((float)mant) * 0.125f;
    }
    if (exp == 0xF && mant == 7u) {
        return 448.0f; // defensive: NaN code -> max finite (never emitted)
    }
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}
#endif

__launch_bounds__(32, 7)
extern "C" __global__ void gemv_mfp4g32_e8_strip_gfx1151(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_block_bytes = (K / 32) * 17;
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const int block_in_iter = tid >> 2;
    const int cw_off = (tid & 3) * 4;
    const int thread_block_off = block_in_iter * 17;

    float acc0 = 0.0f, acc1 = 0.0f;
    const int pairs = groups_per_row >> 1;
    const int tail  = groups_per_row & 1;

    for (int p = 0; p < pairs; p++) {
        const int g = p << 1;
        const char* gp0 = row_ptr + (long long)g * 136 + thread_block_off;
        const char* gp1 = gp0 + 136;

        // SAME memory reads as the real kernel: scale byte + codeword + 8 x[].
        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp0) * 0.88f;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp1) * 0.88f;
        const unsigned int ew0 = *(const unsigned int*)(gp0 + 1 + cw_off);
        const unsigned int ew1 = *(const unsigned int*)(gp1 + 1 + cw_off);
        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256;

        // STRIPPED compute: read all 8 x[] (sum), fold in codeword+scale trivially.
        float sx0 = x[base0]+x[base0+1]+x[base0+2]+x[base0+3]+x[base0+4]+x[base0+5]+x[base0+6]+x[base0+7];
        float sx1 = x[base1]+x[base1+1]+x[base1+2]+x[base1+3]+x[base1+4]+x[base1+5]+x[base1+6]+x[base1+7];
        acc0 += ssc0 * sx0 * (float)(ew0 & 0xFu);
        acc1 += ssc1 * sx1 * (float)(ew1 & 0xFu);
    }

    if (tail) {
        const int g = pairs << 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float sx = x[base]+x[base+1]+x[base+2]+x[base+3]+x[base+4]+x[base+5]+x[base+6]+x[base+7];
        acc0 += ssc * sx * (float)(ew & 0xFu);
    }

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[row] = acc;
}
