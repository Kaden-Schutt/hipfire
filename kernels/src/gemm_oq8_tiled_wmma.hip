// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus-Quant W8A8, register-tiled: the int8 generalization of
// gemm_bf16_tiled_wmma and the tiled generalization of gemm_oq8_grouped_wmma.
// gfx1103 (RDNA3) wave32, ZERO LDS.
//
// gemm_oq8_grouped_wmma computes one 16×16 output tile per wave (single
// dependent v_wmma_i32 chain, no ILP) — same latency-bound ~0.5 TF/s ceiling as
// the naive bf16 kernel. This gives each wave an MB×NB grid of 16×16 output
// subtiles with MB×NB independent int32 accumulators (per group) rescaled into
// MB×NB f32 accumulators, so the WMMA latency is hidden by ILP and each loaded
// int8 fragment is reused across the tile. On RDNA3 int8 WMMA runs at the same
// rate as bf16 WMMA, so warm throughput matches gemm_bf16_tiled_wmma — but the
// weights are int8 (half the bytes ⇒ half the UMA footprint and cold-load).
//
//   W : [M, K] int8            row stride = K bytes
//   Ws: [M, K/G] f32           per-group weight scales
//   X : [B, K] int8            row stride = K bytes
//   Xs: [B, K/G] f32           per-group activation scales
//   Y : [B, M] f32             Y[b*M + m]
//   Y[b,m] = Σ_g sw[m,g]·sx[b,g] · Σ_{k∈g} qw[m,k]·qx[b,k]
// Requires K % group == 0 and group % 16 == 0.
//
// grid = (ceil(M/(16·MB)), ceil(B/(16·NB))), block = 32 (one wave).
//
// Register budget note: each subtile needs a live int32×8 group accumulator AND
// a live f32×8 cross-group accumulator, so accumulator pressure is ~2× the bf16
// tiled kernel. 4×4 would spill; 4×2 / 2×4 are the usable large tilings.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

#define DEFINE_TILED_OQ8_GEMM(NAME, TILE_MB, TILE_NB, MINWAVES)                 \
__launch_bounds__(32, MINWAVES)                                                \
extern "C" __global__ void NAME(                                               \
    const int8_t* __restrict__ W, const float* __restrict__ Ws,                \
    const int8_t* __restrict__ X, const float* __restrict__ Xs,                \
    float* __restrict__ Y,                                                      \
    int M, int K, int B, int group                                             \
) {                                                                            \
    const int tid  = threadIdx.x;                                              \
    const int lane = tid & 15;                                                 \
    const int row_start   = blockIdx.x * (16 * TILE_MB);                       \
    const int batch_start = blockIdx.y * (16 * TILE_NB);                       \
    if (row_start >= M || batch_start >= B) return;                           \
    const int n_groups = K / group;                                           \
    const int tiles_per_group = group / 16;                                   \
    const int8_t* w_row[TILE_MB];                                             \
    const int8_t* x_row[TILE_NB];                                             \
    _Pragma("unroll")                                                          \
    for (int mi = 0; mi < TILE_MB; ++mi) {                                     \
        int m = row_start + mi * 16 + lane;                                    \
        w_row[mi] = W + (long long)((m < M) ? m : (M - 1)) * K;                \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int ni = 0; ni < TILE_NB; ++ni) {                                     \
        int b = batch_start + ni * 16 + lane;                                  \
        x_row[ni] = X + (long long)((b < B) ? b : (B - 1)) * K;                \
    }                                                                          \
    float8_t facc[TILE_MB][TILE_NB];                                          \
    _Pragma("unroll")                                                          \
    for (int mi = 0; mi < TILE_MB; ++mi)                                       \
        _Pragma("unroll")                                                      \
        for (int ni = 0; ni < TILE_NB; ++ni)                                   \
            facc[mi][ni] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};                  \
    for (int g = 0; g < n_groups; ++g) {                                       \
        int32x8_t iacc[TILE_MB][TILE_NB];                                     \
        _Pragma("unroll")                                                      \
        for (int mi = 0; mi < TILE_MB; ++mi)                                   \
            _Pragma("unroll")                                                  \
            for (int ni = 0; ni < TILE_NB; ++ni)                               \
                iacc[mi][ni] = {0,0,0,0,0,0,0,0};                              \
        const int g_off = g * group;                                          \
        for (int kt = 0; kt < tiles_per_group; ++kt) {                        \
            int32x4_t a_frag[TILE_MB];                                        \
            int32x4_t x_frag[TILE_NB];                                        \
            _Pragma("unroll")                                                  \
            for (int mi = 0; mi < TILE_MB; ++mi)                               \
                a_frag[mi] = *(const int32x4_t*)(w_row[mi] + g_off + kt * 16); \
            _Pragma("unroll")                                                  \
            for (int ni = 0; ni < TILE_NB; ++ni)                               \
                x_frag[ni] = *(const int32x4_t*)(x_row[ni] + g_off + kt * 16); \
            _Pragma("unroll")                                                  \
            for (int mi = 0; mi < TILE_MB; ++mi)                               \
                _Pragma("unroll")                                              \
                for (int ni = 0; ni < TILE_NB; ++ni)                           \
                    iacc[mi][ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32( \
                        true, a_frag[mi], true, x_frag[ni], iacc[mi][ni], false); \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int ni = 0; ni < TILE_NB; ++ni) {                                 \
            const int b_col = batch_start + ni * 16 + lane;                    \
            const float sx = (b_col < B) ? Xs[(long long)b_col * n_groups + g] : 0.f; \
            _Pragma("unroll")                                                  \
            for (int mi = 0; mi < TILE_MB; ++mi) {                             \
                _Pragma("unroll")                                              \
                for (int j = 0; j < 8; ++j) {                                  \
                    const int m_row = row_start + mi * 16 + 2 * j + (tid >> 4);\
                    const float sw = (m_row < M) ? Ws[(long long)m_row * n_groups + g] : 0.f; \
                    facc[mi][ni][j] += (float)iacc[mi][ni][j] * sw * sx;       \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int ni = 0; ni < TILE_NB; ++ni) {                                     \
        const int out_col = batch_start + ni * 16 + lane;                      \
        if (out_col >= B) continue;                                            \
        _Pragma("unroll")                                                      \
        for (int mi = 0; mi < TILE_MB; ++mi) {                                 \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 8; ++j) {                                      \
                const int out_row = row_start + mi * 16 + 2 * j + (tid >> 4);  \
                if (out_row < M)                                              \
                    Y[(long long)out_col * M + out_row] = facc[mi][ni][j];     \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_TILED_OQ8_GEMM(gemm_oq8_tiled_wmma_2x2, 2, 2, 4)
DEFINE_TILED_OQ8_GEMM(gemm_oq8_tiled_wmma_4x2, 4, 2, 2)
DEFINE_TILED_OQ8_GEMM(gemm_oq8_tiled_wmma_2x4, 2, 4, 2)
