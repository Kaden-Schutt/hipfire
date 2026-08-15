// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned batch-tiled gfx1100 HFQ4-G256 residual GEMM (K2) with
// per-group LDS affine codebook.
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx1100_muse_bt.hip (Apache-2.0)
// and the MQ4-Lloyd residual LDS-codebook pattern in
// gemm_mq4g256_lloyd_residual_wmma.hip (Apache-2.0).
//
// Instead of reconstructing sc*q+zp for every nibble inside the K-tile loop,
// each workgroup cooperatively fills __shared__ _Float16 cb_lds[16*16] once
// per (output-row tile, group) with the 16 affine HFQ values:
//   cb_lds[row*16 + q] = sc_h * (_Float16)(float)q + zp_h
// then A-tile half16 entries are row-local LDS lookups. BV independent
// accumulator chains and the g/kt A-then-B WMMA order match the K2 Muse BT
// candidate bit-for-bit on A/B application order.
//
// Sync discipline: exactly two __syncthreads() per group (after fill, before
// next-group overwrite). ZERO barriers inside the K-tile loop.
//
// FULL-TILE PRECONDITION. Host guarantees batch % (16*BV) == 0 for every
// launched tile, so batch indexing is affine (no tail clamp). M row bounds
// are still checked on output. Output is Y += acc (caller pre-inits Y).
//
// Grid: [ceil(M/16), batch/(16*BV)]. Block: [32]. LDS: 512 B.
// Instantiated widths: 4, 6, 12 (exact-gate measurement candidates).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define GEN_MUSE_CB_BT_FULL(BV, SUFFIX)                                                           \
    __launch_bounds__(32, 2) extern "C" __global__ void                                           \
        gemm_hfq4g256_residual_wmma_gfx1100_muse_cb_##SUFFIX(                                     \
            const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y,    \
            int M, int K, int batch_size) {                                                       \
        const int tid = threadIdx.x;                                                              \
        const int row_start = blockIdx.x * 16;                                                    \
        const int batch_start = blockIdx.y * (16 * BV);                                           \
        if (row_start >= M) return;                                                               \
        const int ml = tid & 15;                                                                  \
        const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);                   \
        const int groups_per_row = K / 256;                                                       \
        const long long row_stride = (long long)groups_per_row * 136;                             \
        const char* row_base = A + (long long)safe_row * row_stride;                              \
        const long long xstride = (long long)16 * K;                                              \
        /* Cooperative-load mapping: 32 threads × 8 fp16 = 256 entries (16 rows × 16 q). */     \
        const int load_tile_row = tid >> 1;                                                       \
        const int load_lo = (tid & 1) * 8;                                                        \
        const int safe_load_row =                                                                 \
            (row_start + load_tile_row < M) ? (row_start + load_tile_row) : (M - 1);              \
        const char* load_row_base = A + (long long)safe_load_row * row_stride;                    \
        __shared__ _Float16 cb_lds[16 * 16];                                                      \
        float8_t acc[BV];                                                                         \
        for (int b = 0; b < BV; b++) acc[b] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};                   \
        for (int g = 0; g < groups_per_row; g++) {                                                \
            const char* load_gp = load_row_base + g * 136;                                        \
            _Float16 sc_h =                                                                       \
                (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(load_gp));             \
            _Float16 zp_h =                                                                       \
                (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(load_gp + 4));         \
            _Pragma("unroll") for (int i = 0; i < 8; i++) {                                       \
                const int q = load_lo + i;                                                        \
                cb_lds[load_tile_row * 16 + q] = sc_h * (_Float16)(float)q + zp_h;                \
            }                                                                                     \
            __syncthreads();                                                                      \
            const char* gp = row_base + g * 136;                                                  \
            const _Float16* xg0 = X + (long long)(batch_start + ml) * K + g * 256;                \
            const int cb_base = ml * 16;                                                          \
            for (int kt = 0; kt < 16; kt += 2) {                                                  \
                const int k_off_a = kt * 16;                                                      \
                unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);                 \
                unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);             \
                const int k_off_b = (kt + 1) * 16;                                                \
                unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);                 \
                unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);             \
                half16_t a_a;                                                                     \
                a_a[0] = cb_lds[cb_base + ((pk0a >> 0) & 0xFu)];                                  \
                a_a[1] = cb_lds[cb_base + ((pk0a >> 4) & 0xFu)];                                  \
                a_a[2] = cb_lds[cb_base + ((pk0a >> 8) & 0xFu)];                                  \
                a_a[3] = cb_lds[cb_base + ((pk0a >> 12) & 0xFu)];                                 \
                a_a[4] = cb_lds[cb_base + ((pk0a >> 16) & 0xFu)];                                 \
                a_a[5] = cb_lds[cb_base + ((pk0a >> 20) & 0xFu)];                                 \
                a_a[6] = cb_lds[cb_base + ((pk0a >> 24) & 0xFu)];                                 \
                a_a[7] = cb_lds[cb_base + ((pk0a >> 28) & 0xFu)];                                 \
                a_a[8] = cb_lds[cb_base + ((pk1a >> 0) & 0xFu)];                                  \
                a_a[9] = cb_lds[cb_base + ((pk1a >> 4) & 0xFu)];                                  \
                a_a[10] = cb_lds[cb_base + ((pk1a >> 8) & 0xFu)];                                 \
                a_a[11] = cb_lds[cb_base + ((pk1a >> 12) & 0xFu)];                                \
                a_a[12] = cb_lds[cb_base + ((pk1a >> 16) & 0xFu)];                                \
                a_a[13] = cb_lds[cb_base + ((pk1a >> 20) & 0xFu)];                                \
                a_a[14] = cb_lds[cb_base + ((pk1a >> 24) & 0xFu)];                                \
                a_a[15] = cb_lds[cb_base + ((pk1a >> 28) & 0xFu)];                                \
                _Pragma("unroll") for (int b = 0; b < BV; b++) {                                  \
                    half16_t b_a = *(const half16_t*)(xg0 + k_off_a + b * xstride);               \
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc[b]);        \
                }                                                                                 \
                a_a[0] = cb_lds[cb_base + ((pk0b >> 0) & 0xFu)];                                  \
                a_a[1] = cb_lds[cb_base + ((pk0b >> 4) & 0xFu)];                                  \
                a_a[2] = cb_lds[cb_base + ((pk0b >> 8) & 0xFu)];                                  \
                a_a[3] = cb_lds[cb_base + ((pk0b >> 12) & 0xFu)];                                 \
                a_a[4] = cb_lds[cb_base + ((pk0b >> 16) & 0xFu)];                                 \
                a_a[5] = cb_lds[cb_base + ((pk0b >> 20) & 0xFu)];                                 \
                a_a[6] = cb_lds[cb_base + ((pk0b >> 24) & 0xFu)];                                 \
                a_a[7] = cb_lds[cb_base + ((pk0b >> 28) & 0xFu)];                                 \
                a_a[8] = cb_lds[cb_base + ((pk1b >> 0) & 0xFu)];                                  \
                a_a[9] = cb_lds[cb_base + ((pk1b >> 4) & 0xFu)];                                  \
                a_a[10] = cb_lds[cb_base + ((pk1b >> 8) & 0xFu)];                                 \
                a_a[11] = cb_lds[cb_base + ((pk1b >> 12) & 0xFu)];                                \
                a_a[12] = cb_lds[cb_base + ((pk1b >> 16) & 0xFu)];                                \
                a_a[13] = cb_lds[cb_base + ((pk1b >> 20) & 0xFu)];                                \
                a_a[14] = cb_lds[cb_base + ((pk1b >> 24) & 0xFu)];                                \
                a_a[15] = cb_lds[cb_base + ((pk1b >> 28) & 0xFu)];                                \
                _Pragma("unroll") for (int b = 0; b < BV; b++) {                                  \
                    half16_t b_b = *(const half16_t*)(xg0 + k_off_b + b * xstride);               \
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc[b]);        \
                }                                                                                 \
            }                                                                                     \
            __syncthreads();                                                                      \
        }                                                                                         \
        /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (tid>>4)][tid & 15]. */                            \
        _Pragma("unroll") for (int b = 0; b < BV; b++) {                                          \
            const int out_col = batch_start + b * 16 + ml;                                        \
            _Pragma("unroll") for (int j = 0; j < 8; j++) {                                       \
                int out_row = row_start + 2 * j + (tid >> 4);                                     \
                if (out_row < M) Y[(long long)out_col * M + out_row] += acc[b][j];                \
            }                                                                                     \
        }                                                                                         \
    }

GEN_MUSE_CB_BT_FULL(4, full4)
GEN_MUSE_CB_BT_FULL(6, full6)
GEN_MUSE_CB_BT_FULL(12, full12)
