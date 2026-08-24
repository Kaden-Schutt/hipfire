// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batch-tiled gfx1100 WMMA residual for MQ4G256V2 prefill.
// Arithmetic / header / interleaved-C authority:
//   gemm_mq4g256v2_residual_wmma.hip  (base gfx11 half16 K2 kernel)
// B-accumulator / X-loop structure only:
//   gemm_mq4g256v2_residual_wmma_gfx12_bt.hip
//
// Each block (1 wave) computes 16 rows x (16*B) batch with B independent
// accumulator chains. Weights + dequant load once per K-tile and reuse across
// the B N-tiles. Same ABI and Y += W@X (residual add) semantics as the base.
// Caller pre-inits Y with the residual stream.
//
// Grid: [ceil(M/16), ceil(N/(16*B))]. Block: [32]. LDS: 0.
// Instantiated: bt4/6/8 (64/96/128 batch columns). Production policy on exact
// gfx1100 selects among these by batch size; screen-only 10/12/14/16 removed.
// Exact gfx1100 launch_bounds(32,2) — no gfx1151 admission branch.
// Output: acc[j] = C[2*j + (tid>>4)][tid & 15] (interleaved).
// Headers: dual FP16 half headers split at kt=8, 136-byte groups.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define GEN_RESID_BT(BV) \
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq4g256v2_residual_wmma_gfx11_bt##BV( \
    const char* __restrict__ A, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y, \
    int M, int K, int N \
) { \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * BV); \
    const int tid = threadIdx.x; \
    if (row_start >= M) return; \
    if (batch_start >= N) return; \
    const int ml = tid & 15; \
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1); \
    const int groups_per_row = K / 256; \
    const char* row_base = A + (long long)safe_row * groups_per_row * 136; \
    int sb[BV]; \
    for (int b = 0; b < BV; b++) { \
        int t = batch_start + b * 16 + ml; \
        sb[b] = (t < N) ? t : 0; \
    } \
    float8_t acc[BV]; \
    for (int b = 0; b < BV; b++) \
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        const char* gp = row_base + g * 136; \
        const unsigned int hA = *(const unsigned int*)(gp); \
        const unsigned int hB = *(const unsigned int*)(gp + 4); \
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16))); \
        const _Float16* xg[BV]; \
        for (int b = 0; b < BV; b++) \
            xg[b] = X + (long long)sb[b] * K + g * 256; \
        _Pragma("unroll") \
        for (int kt = 0; kt < 16; kt++) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off = kt * 16; \
            unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2); \
            unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4); \
            half16_t a_reg; \
            a_reg[0]  = sc_h * (_Float16)(float)((pk0 >>  0) & 0xFu) + zp_h; \
            a_reg[1]  = sc_h * (_Float16)(float)((pk0 >>  4) & 0xFu) + zp_h; \
            a_reg[2]  = sc_h * (_Float16)(float)((pk0 >>  8) & 0xFu) + zp_h; \
            a_reg[3]  = sc_h * (_Float16)(float)((pk0 >> 12) & 0xFu) + zp_h; \
            a_reg[4]  = sc_h * (_Float16)(float)((pk0 >> 16) & 0xFu) + zp_h; \
            a_reg[5]  = sc_h * (_Float16)(float)((pk0 >> 20) & 0xFu) + zp_h; \
            a_reg[6]  = sc_h * (_Float16)(float)((pk0 >> 24) & 0xFu) + zp_h; \
            a_reg[7]  = sc_h * (_Float16)(float)((pk0 >> 28) & 0xFu) + zp_h; \
            a_reg[8]  = sc_h * (_Float16)(float)((pk1 >>  0) & 0xFu) + zp_h; \
            a_reg[9]  = sc_h * (_Float16)(float)((pk1 >>  4) & 0xFu) + zp_h; \
            a_reg[10] = sc_h * (_Float16)(float)((pk1 >>  8) & 0xFu) + zp_h; \
            a_reg[11] = sc_h * (_Float16)(float)((pk1 >> 12) & 0xFu) + zp_h; \
            a_reg[12] = sc_h * (_Float16)(float)((pk1 >> 16) & 0xFu) + zp_h; \
            a_reg[13] = sc_h * (_Float16)(float)((pk1 >> 20) & 0xFu) + zp_h; \
            a_reg[14] = sc_h * (_Float16)(float)((pk1 >> 24) & 0xFu) + zp_h; \
            a_reg[15] = sc_h * (_Float16)(float)((pk1 >> 28) & 0xFu) + zp_h; \
            _Pragma("unroll") \
            for (int b = 0; b < BV; b++) { \
                half16_t b_reg = *(const half16_t*)(xg[b] + k_off); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[b]); \
            } \
        } \
    } \
    /* RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15]. */ \
    _Pragma("unroll") \
    for (int b = 0; b < BV; b++) { \
        const int out_col = batch_start + b * 16 + ml; \
        if (out_col < N) { \
            _Pragma("unroll") \
            for (int j = 0; j < 8; j++) { \
                int out_row = row_start + 2 * j + (tid >> 4); \
                if (out_row < M) { \
                    Y[(long long)out_col * M + out_row] += acc[b][j]; \
                } \
            } \
        } \
    } \
}

GEN_RESID_BT(4)
GEN_RESID_BT(6)
GEN_RESID_BT(8)
