// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned multi-wave batch-tiled gfx1100 HFQ4-G256 residual GEMM
// (BT4 / K2). Candidate for exact gate/up projection on gfx1100.
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx1100_muse_bt.hip (Apache-2.0).
//
// Groups NWAVES identical BT4 row waves into one block. Each wave independently
// owns one adjacent 16-row tile and the same 64-column batch tile (BV=4).
// No LDS, no barriers: per-wave early return is safe. Activation (X) traffic
// is identical across waves within a block, exposing L1 reuse / dual-issue
// scheduling without changing the per-accumulator WMMA order.
//
// Preserves the gfx1100 K2 arithmetic exactly per wave: each independent
// accumulator sees the same g/kt pair sequence and
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 calls as the standalone BT4
// muse kernel. Weight dequant is shared once across BV=4 independent 16-column
// batch tiles; each chain loads its own X.
//
// FULL-TILE PRECONDITION. Host guarantees batch % 64 == 0 for every launched
// tile, so batch indexing is affine (no tail clamp). M row bounds are still
// checked on output. Output is Y += acc (caller pre-inits Y).
//
// Grid:  [ceil(ceil(M/16)/NWAVES), batch/64, 1]
// Block: [32*NWAVES, 1, 1]. LDS: 0.
// Instantiated: mw2_bt4, mw4_bt4, mw8_bt4.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// BV fixed at 4 (64 batch columns per block). NWAVES = 2, 4, or 8.
#define GEN_MUSE_MW_BT4(NWAVES, SUFFIX)                                                           \
    __launch_bounds__(32 * (NWAVES), 1) extern "C" __global__ void                                \
        gemm_hfq4g256_residual_wmma_gfx1100_muse_##SUFFIX(                                        \
            const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y,    \
            int M, int K, int batch_size) {                                                       \
        const int lane = threadIdx.x & 31;                                                        \
        const int wave_id = threadIdx.x >> 5;                                                     \
        const int row_start = (blockIdx.x * (NWAVES) + wave_id) * 16;                             \
        const int batch_start = blockIdx.y * 64;                                                  \
        if (row_start >= M) return;                                                               \
        const int ml = lane & 15;                                                                 \
        const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);                   \
        const int groups_per_row = K / 256;                                                       \
        const char* row_base = A + (long long)safe_row * groups_per_row * 136;                    \
        const long long xstride = (long long)16 * K;                                              \
        float8_t acc[4];                                                                          \
        for (int b = 0; b < 4; b++) acc[b] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};                    \
        for (int g = 0; g < groups_per_row; g++) {                                                \
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
                _Pragma("unroll") for (int b = 0; b < 4; b++) {                                   \
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
                _Pragma("unroll") for (int b = 0; b < 4; b++) {                                   \
                    half16_t b_b = *(const half16_t*)(xg0 + k_off_b + b * xstride);               \
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc[b]);        \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (lane>>4)][lane & 15]. */                          \
        _Pragma("unroll") for (int b = 0; b < 4; b++) {                                           \
            const int out_col = batch_start + b * 16 + ml;                                        \
            _Pragma("unroll") for (int j = 0; j < 8; j++) {                                       \
                int out_row = row_start + 2 * j + (lane >> 4);                                    \
                if (out_row < M) Y[(long long)out_col * M + out_row] += acc[b][j];                \
            }                                                                                     \
        }                                                                                         \
    }

GEN_MUSE_MW_BT4(2, mw2_bt4)
GEN_MUSE_MW_BT4(4, mw4_bt4)
GEN_MUSE_MW_BT4(8, mw8_bt4)
