// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 decode-GEMV with LDS-staged x + 4 rows/block, gfx1151-only.
//
// Memory-side experiment (the rocprof-supported lever): the per-row kernel is
// memory-bound (MemUnitBusy ~84%, eff BW ~33% of peak — busy-but-inefficient).
// This variant processes R=4 output rows per workgroup (128 threads = 4 wave32),
// staging the x vector into LDS once and reusing it across all 4 rows. Two aims:
//   1. cut redundant x re-reads (x loaded once per block, not once per row);
//   2. more independent weight-load streams in flight per block → better
//      memory-level parallelism / latency hiding on the busy memory pipe.
// Weight decode is the byte-identical E8 lattice path (bit-exact per row).
//
// Dynamic LDS = K floats (set by host launch). gfx1151-gated; HIPFIRE_E8_LDSX=1.
// Bit-exact with gemv_mfp4g32_e8_gfx1151 (same per-row math, x via LDS).

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

__device__ __forceinline__ void e8_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) { e[i] = (idx >> (unsigned)(4 * i)) & 0xFu; sl += e[i]; }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    const unsigned int lsb = (sl + p7) & 1u;
    e[7] = p7 | lsb;
    for (int i = 0; i < 8; i++) { float c = (float)((int)e[i] - 7); p[i] = coset ? c + 0.5f : c; }
}

// Per-row E8 GEMV reading x from LDS. Bit-exact with the global-x version.
__device__ __forceinline__ void gemv_e8_row_lds(
    const char* __restrict__ A, const float* __restrict__ xl,
    float* __restrict__ y, int row, int K, int lane
) {
    const int groups_per_row = K / 256;
    const int row_block_bytes = (K / 32) * 17;
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const int block_in_iter = lane >> 2;
    const int cw_off = (lane & 3) * 4;
    const int thread_block_off = block_in_iter * 17;

    float acc0 = 0.0f, acc1 = 0.0f;
    const int pairs = groups_per_row >> 1;
    const int tail  = groups_per_row & 1;

    for (int p = 0; p < pairs; p++) {
        const int g = p << 1;
        const char* gp0 = row_ptr + (long long)g * 136 + thread_block_off;
        const char* gp1 = gp0 + 136;
        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp0) * 0.88f;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp1) * 0.88f;
        const unsigned int ew0 = *(const unsigned int*)(gp0 + 1 + cw_off);
        const unsigned int ew1 = *(const unsigned int*)(gp1 + 1 + cw_off);
        const int base0 = g * 256 + lane * 8;
        const int base1 = base0 + 256;
        float pe[8];
        e8_decode_index(ew0, pe);
        acc0 += ssc0*pe[0]*xl[base0  ] + ssc0*pe[1]*xl[base0+1] + ssc0*pe[2]*xl[base0+2] + ssc0*pe[3]*xl[base0+3]
              + ssc0*pe[4]*xl[base0+4] + ssc0*pe[5]*xl[base0+5] + ssc0*pe[6]*xl[base0+6] + ssc0*pe[7]*xl[base0+7];
        e8_decode_index(ew1, pe);
        acc1 += ssc1*pe[0]*xl[base1  ] + ssc1*pe[1]*xl[base1+1] + ssc1*pe[2]*xl[base1+2] + ssc1*pe[3]*xl[base1+3]
              + ssc1*pe[4]*xl[base1+4] + ssc1*pe[5]*xl[base1+5] + ssc1*pe[6]*xl[base1+6] + ssc1*pe[7]*xl[base1+7];
    }
    if (tail) {
        const int g = pairs << 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + lane * 8;
        float pe[8];
        e8_decode_index(ew, pe);
        acc0 += ssc*pe[0]*xl[base  ] + ssc*pe[1]*xl[base+1] + ssc*pe[2]*xl[base+2] + ssc*pe[3]*xl[base+3]
              + ssc*pe[4]*xl[base+4] + ssc*pe[5]*xl[base+5] + ssc*pe[6]*xl[base+6] + ssc*pe[7]*xl[base+7];
    }
    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) y[row] = acc;
}

#define E8_LDSX_ROWS 4

__launch_bounds__(128, 4)
extern "C" __global__ void gemv_mfp4g32_e8_ldsx_gfx1151(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    extern __shared__ float xl[];  // K floats
    const int tid = threadIdx.x;       // 0..127
    // Cooperative stage of x into LDS (once per block, reused by all 4 rows).
    for (int i = tid; i < K; i += 128) xl[i] = x[i];
    __syncthreads();

    const int lane = tid & 31;
    const int row_in_block = tid >> 5;             // 0..3
    const int row = blockIdx.x * E8_LDSX_ROWS + row_in_block;
    if (row < M) {
        gemv_e8_row_lds(A, xl, y, row, K, lane);
    }
}
