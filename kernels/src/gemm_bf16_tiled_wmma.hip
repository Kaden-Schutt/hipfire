// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Register-tiled, ZERO-LDS BF16×BF16→F32 WMMA GEMM for gfx1103 (RDNA3, wave32).
//
// The baseline gemm_bf16_x_bf16_wmma computes ONE 16×16 output tile per wave via
// a single dependent v_wmma chain down K — no ILP to hide the WMMA's fixed
// latency, and each 16×16 A/B fragment feeds only one tile (2 frag-loads per
// output tile). On gfx1103 that pins the kernel at ~3% of peak regardless of
// dtype (int8 WMMA runs at the same rate on RDNA3, so oq8 doesn't help).
//
// This kernel gives each wave an MB×NB grid of 16×16 output subtiles with
// MB×NB independent f32 accumulators. Per K-step it loads MB A-fragments and NB
// B-fragments and issues MB×NB WMMAs, so:
//   * MB×NB independent WMMA chains → deep ILP hides the WMMA latency,
//   * each A-frag is reused NB times and each B-frag MB times → global frag
//     traffic drops to (MB+NB)/(MB·NB) loads per output tile (0.5 at 4×4 vs the
//     baseline's 2.0), all in registers, no LDS (respects the gfx1103 LDS-wedge
//     hazard).
//
//   A : [M, K] BF16 row-major (raw uint16_t)   X : [B, K] BF16 row-major
//   Y : [B, M] F32 row-major
//
// grid = (ceil(M/(16·MB)), ceil(B/(16·NB))), block = 32 (one wave).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(8)))  float8_t;

// One entrypoint per (MB,NB) tiling so a single compiled module can be swept.
#define DEFINE_TILED_GEMM(NAME, TILE_MB, TILE_NB, MINWAVES)                     \
__launch_bounds__(32, MINWAVES)                                                 \
extern "C" __global__ void NAME(                                               \
    const unsigned short* __restrict__ A,                                       \
    const unsigned short* __restrict__ X,                                       \
    float*               __restrict__ Y,                                        \
    int M, int K, int B                                                         \
) {                                                                             \
    const int row_start   = blockIdx.x * (16 * TILE_MB);                        \
    const int batch_start = blockIdx.y * (16 * TILE_NB);                        \
    const int tid  = threadIdx.x;                                               \
    const int lane = tid & 15;                                                  \
    if (row_start >= M || batch_start >= B) return;                             \
    const unsigned short* w_row[TILE_MB];                                       \
    const unsigned short* x_row[TILE_NB];                                       \
    _Pragma("unroll")                                                           \
    for (int mi = 0; mi < TILE_MB; ++mi) {                                      \
        int m = row_start + mi * 16 + lane;                                     \
        w_row[mi] = A + (long long)((m < M) ? m : (M - 1)) * K;                 \
    }                                                                           \
    _Pragma("unroll")                                                           \
    for (int ni = 0; ni < TILE_NB; ++ni) {                                      \
        int b = batch_start + ni * 16 + lane;                                   \
        x_row[ni] = X + (long long)((b < B) ? b : (B - 1)) * K;                 \
    }                                                                           \
    float8_t acc[TILE_MB][TILE_NB];                                             \
    _Pragma("unroll")                                                           \
    for (int mi = 0; mi < TILE_MB; ++mi)                                        \
        _Pragma("unroll")                                                       \
        for (int ni = 0; ni < TILE_NB; ++ni)                                    \
            acc[mi][ni] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};             \
    for (int k = 0; k < K; k += 16) {                                          \
        b16x16_t a_frag[TILE_MB];                                               \
        b16x16_t x_frag[TILE_NB];                                               \
        _Pragma("unroll")                                                       \
        for (int mi = 0; mi < TILE_MB; ++mi)                                    \
            a_frag[mi] = *(const b16x16_t*)(w_row[mi] + k);                     \
        _Pragma("unroll")                                                       \
        for (int ni = 0; ni < TILE_NB; ++ni)                                    \
            x_frag[ni] = *(const b16x16_t*)(x_row[ni] + k);                     \
        _Pragma("unroll")                                                       \
        for (int mi = 0; mi < TILE_MB; ++mi)                                    \
            _Pragma("unroll")                                                   \
            for (int ni = 0; ni < TILE_NB; ++ni)                                \
                acc[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(      \
                    a_frag[mi], x_frag[ni], acc[mi][ni]);                       \
    }                                                                           \
    _Pragma("unroll")                                                           \
    for (int ni = 0; ni < TILE_NB; ++ni) {                                      \
        const int out_col = batch_start + ni * 16 + lane;                       \
        if (out_col >= B) continue;                                             \
        _Pragma("unroll")                                                       \
        for (int mi = 0; mi < TILE_MB; ++mi) {                                  \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < 8; ++j) {                                       \
                const int out_row = row_start + mi * 16 + 2 * j + (tid >> 4);   \
                if (out_row < M) {                                              \
                    Y[(long long)out_col * M + out_row] = acc[mi][ni][j];       \
                }                                                               \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}

DEFINE_TILED_GEMM(gemm_bf16_tiled_wmma_2x2, 2, 2, 4)
DEFINE_TILED_GEMM(gemm_bf16_tiled_wmma_4x2, 4, 2, 2)
DEFINE_TILED_GEMM(gemm_bf16_tiled_wmma_4x4, 4, 4, 2)
