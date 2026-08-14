// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned batch-tiled gfx1100 HFQ4-G256 residual GEMM, deterministic
// K-split phase 1 (partials only).
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_ksplit_det.hip (Apache-2.0) with the
// batch-tile structure of gemm_hfq4g256_residual_wmma_gfx1100_muse_bt.hip
// (Apache-2.0).
//
// Keeps K_SPLITS=4 and the same g_start/g_end partition as ksplit_det. Each
// independent BV accumulator sees the same per-split g/kt A-then-B WMMA order
// as production ksplit_det; weight dequant is shared once across BV 16-column
// batch tiles. Writes plain partials to the existing [K_SPLITS][batch][M]
// scratch layout — no atomicAdd, no residual. Finalize remains
// gemm_ksplit_det_finalize (not in this source).
//
// FULL-TILE PRECONDITION. Host guarantees batch % (16*BV) == 0 for every
// launched tile, so batch indexing is affine (no tail clamp). M row bounds
// are still checked on output.
//
// Grid: [ceil(M/16), batch/(16*BV), K_SPLITS]. Block: [32]. LDS: 0.
// Instantiated widths: 4, 6, 12 (exact-gate measurement candidates).
// K_SPLITS here MUST match K_SPLITS in gemm_ksplit_det_finalize.hip.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define K_SPLITS 4

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define GEN_MUSE_KS_BT_FULL(BV, SUFFIX)                                                           \
    __launch_bounds__(32, 2) extern "C" __global__ void                                           \
        gemm_hfq4g256_residual_wmma_gfx1100_muse_ks_##SUFFIX(                                     \
            const char* __restrict__ A, const _Float16* __restrict__ X,                           \
            float* __restrict__ partials, int M, int K, int batch_size) {                         \
        const int tid = threadIdx.x;                                                              \
        const int row_start = blockIdx.x * 16;                                                    \
        const int batch_start = blockIdx.y * (16 * BV);                                           \
        const int ksplit = blockIdx.z;                                                            \
        if (row_start >= M) return;                                                               \
        const int ml = tid & 15;                                                                  \
        const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);                   \
        const int groups_per_row = K / 256;                                                       \
        const int groups_per_split = groups_per_row / K_SPLITS;                                   \
        const int g_start = ksplit * groups_per_split;                                            \
        const int g_end = (ksplit == K_SPLITS - 1) ? groups_per_row                               \
                                                   : g_start + groups_per_split;                  \
        const char* row_base = A + (long long)safe_row * groups_per_row * 136;                    \
        const long long xstride = (long long)16 * K;                                              \
        float8_t acc[BV];                                                                         \
        for (int b = 0; b < BV; b++) acc[b] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};                   \
        for (int g = g_start; g < g_end; g++) {                                                   \
            const char* gp = row_base + g * 136;                                                  \
            _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));      \
            _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));  \
            const _Float16* xg0 = X + (long long)(batch_start + ml) * K + g * 256;                \
            for (int kt = 0; kt < 16; kt += 2) {                                                  \
                const int k_off_a = kt * 16;                                                      \
                unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);                 \
                unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);             \
                const int k_off_b = (kt + 1) * 16;                                                \
                unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);                 \
                unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);             \
                half16_t a_a;                                                                     \
                a_a[0] = sc_h * (_Float16)(float)((pk0a >> 0) & 0xFu) + zp_h;                     \
                a_a[1] = sc_h * (_Float16)(float)((pk0a >> 4) & 0xFu) + zp_h;                     \
                a_a[2] = sc_h * (_Float16)(float)((pk0a >> 8) & 0xFu) + zp_h;                     \
                a_a[3] = sc_h * (_Float16)(float)((pk0a >> 12) & 0xFu) + zp_h;                    \
                a_a[4] = sc_h * (_Float16)(float)((pk0a >> 16) & 0xFu) + zp_h;                    \
                a_a[5] = sc_h * (_Float16)(float)((pk0a >> 20) & 0xFu) + zp_h;                    \
                a_a[6] = sc_h * (_Float16)(float)((pk0a >> 24) & 0xFu) + zp_h;                    \
                a_a[7] = sc_h * (_Float16)(float)((pk0a >> 28) & 0xFu) + zp_h;                    \
                a_a[8] = sc_h * (_Float16)(float)((pk1a >> 0) & 0xFu) + zp_h;                     \
                a_a[9] = sc_h * (_Float16)(float)((pk1a >> 4) & 0xFu) + zp_h;                     \
                a_a[10] = sc_h * (_Float16)(float)((pk1a >> 8) & 0xFu) + zp_h;                    \
                a_a[11] = sc_h * (_Float16)(float)((pk1a >> 12) & 0xFu) + zp_h;                   \
                a_a[12] = sc_h * (_Float16)(float)((pk1a >> 16) & 0xFu) + zp_h;                   \
                a_a[13] = sc_h * (_Float16)(float)((pk1a >> 20) & 0xFu) + zp_h;                   \
                a_a[14] = sc_h * (_Float16)(float)((pk1a >> 24) & 0xFu) + zp_h;                   \
                a_a[15] = sc_h * (_Float16)(float)((pk1a >> 28) & 0xFu) + zp_h;                   \
                _Pragma("unroll") for (int b = 0; b < BV; b++) {                                  \
                    half16_t b_a = *(const half16_t*)(xg0 + k_off_a + b * xstride);               \
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc[b]);        \
                }                                                                                 \
                a_a[0] = sc_h * (_Float16)(float)((pk0b >> 0) & 0xFu) + zp_h;                     \
                a_a[1] = sc_h * (_Float16)(float)((pk0b >> 4) & 0xFu) + zp_h;                     \
                a_a[2] = sc_h * (_Float16)(float)((pk0b >> 8) & 0xFu) + zp_h;                     \
                a_a[3] = sc_h * (_Float16)(float)((pk0b >> 12) & 0xFu) + zp_h;                    \
                a_a[4] = sc_h * (_Float16)(float)((pk0b >> 16) & 0xFu) + zp_h;                    \
                a_a[5] = sc_h * (_Float16)(float)((pk0b >> 20) & 0xFu) + zp_h;                    \
                a_a[6] = sc_h * (_Float16)(float)((pk0b >> 24) & 0xFu) + zp_h;                    \
                a_a[7] = sc_h * (_Float16)(float)((pk0b >> 28) & 0xFu) + zp_h;                    \
                a_a[8] = sc_h * (_Float16)(float)((pk1b >> 0) & 0xFu) + zp_h;                     \
                a_a[9] = sc_h * (_Float16)(float)((pk1b >> 4) & 0xFu) + zp_h;                     \
                a_a[10] = sc_h * (_Float16)(float)((pk1b >> 8) & 0xFu) + zp_h;                    \
                a_a[11] = sc_h * (_Float16)(float)((pk1b >> 12) & 0xFu) + zp_h;                   \
                a_a[12] = sc_h * (_Float16)(float)((pk1b >> 16) & 0xFu) + zp_h;                   \
                a_a[13] = sc_h * (_Float16)(float)((pk1b >> 20) & 0xFu) + zp_h;                   \
                a_a[14] = sc_h * (_Float16)(float)((pk1b >> 24) & 0xFu) + zp_h;                   \
                a_a[15] = sc_h * (_Float16)(float)((pk1b >> 28) & 0xFu) + zp_h;                   \
                _Pragma("unroll") for (int b = 0; b < BV; b++) {                                  \
                    half16_t b_b = *(const half16_t*)(xg0 + k_off_b + b * xstride);               \
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc[b]);        \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (tid>>4)][tid & 15]. */                            \
        /* Plain store into this ksplit's partial slice (no atomic, no residual). */              \
        const long long split_base = (long long)ksplit * batch_size * M;                          \
        _Pragma("unroll") for (int b = 0; b < BV; b++) {                                          \
            const int out_col = batch_start + b * 16 + ml;                                        \
            _Pragma("unroll") for (int j = 0; j < 8; j++) {                                       \
                int out_row = row_start + 2 * j + (tid >> 4);                                     \
                if (out_row < M) {                                                                \
                    partials[split_base + (long long)out_col * M + out_row] = acc[b][j];          \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
    }

GEN_MUSE_KS_BT_FULL(4, full4)
GEN_MUSE_KS_BT_FULL(6, full6)
GEN_MUSE_KS_BT_FULL(12, full12)
