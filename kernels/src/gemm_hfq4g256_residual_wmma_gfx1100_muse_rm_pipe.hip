// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned RM2/BV6 gfx1100 HFQ4-G256 residual GEMM (K2) — two-slot
// X-fragment software pipeline over the muse_rm_bt topology.
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_bt.hip (Apache-2.0).
// Packed A dequant copied from muse_rm_pk (bit-identical measured path).
//
// Rationale: baseline RM2 loads B after A is ready. This candidate keeps the
// same g0/kt0..g25/kt15 order, acc[2][6] layout, and Y += mapping, but
// pipelines X loads against A dequant with an explicit contracted two-slot
// schedule (at most two B half16s live):
//
//   issue B0, B1
//   dequant A row0 / row1
//   consume B0; issue B2
//   consume B1; issue B3
//   consume B2; issue B4
//   consume B3; issue B5
//   consume B4; consume B5
//
// Physical slots s0/s1 hold the live pair. Fully unrolled per kt half.
// No WMMA crosses a kt boundary. No LDS, no lane-broadcast, no altered bounds.
//
// FULL-TILE PRECONDITION. Host: M=19968, K=6656, batch=192 → grid [624,2,1].
// Output is Y += acc (caller pre-inits Y). Block [32,1,1]. LDS 0.
// Symbols: rm2_bv6_pipe_scalar, rm2_bv6_pipe_pk2 (measurement-only).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Packed-half2 A dequant — exact copy of muse_rm_pk.
#define DEQUANT_PK_H2(pk, sc_h, zp_h, h01, h23, h45, h67)                                         \
    do {                                                                                          \
        const __half2 sc2 = __half2half2(sc_h);                                                   \
        const __half2 zp2 = __half2half2(zp_h);                                                   \
        const __half2 q01 = __floats2half2_rn((float)(((pk) >> 0) & 0xFu),                        \
                                              (float)(((pk) >> 4) & 0xFu));                       \
        const __half2 q23 = __floats2half2_rn((float)(((pk) >> 8) & 0xFu),                        \
                                              (float)(((pk) >> 12) & 0xFu));                      \
        const __half2 q45 = __floats2half2_rn((float)(((pk) >> 16) & 0xFu),                       \
                                              (float)(((pk) >> 20) & 0xFu));                      \
        const __half2 q67 = __floats2half2_rn((float)(((pk) >> 24) & 0xFu),                       \
                                              (float)(((pk) >> 28) & 0xFu));                      \
        (h01) = __hfma2(q01, sc2, zp2);                                                           \
        (h23) = __hfma2(q23, sc2, zp2);                                                           \
        (h45) = __hfma2(q45, sc2, zp2);                                                           \
        (h67) = __hfma2(q67, sc2, zp2);                                                           \
    } while (0)

#define DEQUANT_A_FRAG_PK(pk0, pk1, sc_h, zp_h, frag)                                             \
    do {                                                                                          \
        __half2 _h[8];                                                                            \
        DEQUANT_PK_H2((pk0), (sc_h), (zp_h), _h[0], _h[1], _h[2], _h[3]);                        \
        DEQUANT_PK_H2((pk1), (sc_h), (zp_h), _h[4], _h[5], _h[6], _h[7]);                        \
        (frag) = *reinterpret_cast<const half16_t*>(_h);                                          \
    } while (0)

// Scalar f16 dequant — exact muse_rm_bt expression.
#define DEQUANT_A_FRAG_SCALAR(pk0, pk1, sc_h, zp_h, frag)                                         \
    do {                                                                                          \
        half16_t a_a;                                                                             \
        a_a[0] = (sc_h) * (_Float16)(float)(((pk0) >> 0) & 0xFu) + (zp_h);                        \
        a_a[1] = (sc_h) * (_Float16)(float)(((pk0) >> 4) & 0xFu) + (zp_h);                        \
        a_a[2] = (sc_h) * (_Float16)(float)(((pk0) >> 8) & 0xFu) + (zp_h);                        \
        a_a[3] = (sc_h) * (_Float16)(float)(((pk0) >> 12) & 0xFu) + (zp_h);                       \
        a_a[4] = (sc_h) * (_Float16)(float)(((pk0) >> 16) & 0xFu) + (zp_h);                       \
        a_a[5] = (sc_h) * (_Float16)(float)(((pk0) >> 20) & 0xFu) + (zp_h);                       \
        a_a[6] = (sc_h) * (_Float16)(float)(((pk0) >> 24) & 0xFu) + (zp_h);                       \
        a_a[7] = (sc_h) * (_Float16)(float)(((pk0) >> 28) & 0xFu) + (zp_h);                       \
        a_a[8] = (sc_h) * (_Float16)(float)(((pk1) >> 0) & 0xFu) + (zp_h);                        \
        a_a[9] = (sc_h) * (_Float16)(float)(((pk1) >> 4) & 0xFu) + (zp_h);                        \
        a_a[10] = (sc_h) * (_Float16)(float)(((pk1) >> 8) & 0xFu) + (zp_h);                       \
        a_a[11] = (sc_h) * (_Float16)(float)(((pk1) >> 12) & 0xFu) + (zp_h);                      \
        a_a[12] = (sc_h) * (_Float16)(float)(((pk1) >> 16) & 0xFu) + (zp_h);                      \
        a_a[13] = (sc_h) * (_Float16)(float)(((pk1) >> 20) & 0xFu) + (zp_h);                      \
        a_a[14] = (sc_h) * (_Float16)(float)(((pk1) >> 24) & 0xFu) + (zp_h);                      \
        a_a[15] = (sc_h) * (_Float16)(float)(((pk1) >> 28) & 0xFu) + (zp_h);                      \
        (frag) = a_a;                                                                             \
    } while (0)

// Two-slot rolling consume for BV=6. s0/s1 already hold B0/B1; A is ready.
// Overwrites slots so at most two B fragments are live. WMMA order: cols 0..5.
#define PIPE_BV6_CONSUME(a0, a1, xg0, k_off, xstride, acc, s0, s1)                                \
    do {                                                                                          \
        /* consume B0; issue B2 into s0 */                                                        \
        (acc)[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a0), (s0), (acc)[0][0]);       \
        (acc)[1][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a1), (s0), (acc)[1][0]);       \
        (s0) = *(const half16_t*)((xg0) + (k_off) + 2 * (xstride)); /* B2 */                     \
        /* consume B1; issue B3 into s1 */                                                        \
        (acc)[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a0), (s1), (acc)[0][1]);       \
        (acc)[1][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a1), (s1), (acc)[1][1]);       \
        (s1) = *(const half16_t*)((xg0) + (k_off) + 3 * (xstride)); /* B3 */                     \
        /* consume B2; issue B4 into s0 */                                                        \
        (acc)[0][2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a0), (s0), (acc)[0][2]);       \
        (acc)[1][2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a1), (s0), (acc)[1][2]);       \
        (s0) = *(const half16_t*)((xg0) + (k_off) + 4 * (xstride)); /* B4 */                     \
        /* consume B3; issue B5 into s1 */                                                        \
        (acc)[0][3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a0), (s1), (acc)[0][3]);       \
        (acc)[1][3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a1), (s1), (acc)[1][3]);       \
        (s1) = *(const half16_t*)((xg0) + (k_off) + 5 * (xstride)); /* B5 */                     \
        /* consume B4 then B5 (drain) */                                                          \
        (acc)[0][4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a0), (s0), (acc)[0][4]);       \
        (acc)[1][4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a1), (s0), (acc)[1][4]);       \
        (acc)[0][5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a0), (s1), (acc)[0][5]);       \
        (acc)[1][5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a1), (s1), (acc)[1][5]);       \
    } while (0)

// ---------------------------------------------------------------------------
// Scalar pipe: issue B0/B1 → scalar-dequant A row0/row1 → rolling consume.
// ---------------------------------------------------------------------------
__launch_bounds__(32, 2) extern "C" __global__ void
gemm_hfq4g256_residual_wmma_gfx1100_muse_rm2_bv6_pipe_scalar(
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, int M,
    int K, int batch_size) {
    constexpr int RM = 2;
    constexpr int BV = 6;
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * (16 * RM);
    const int batch_start = blockIdx.y * (16 * BV);
    const int ml = tid & 15;
    const int groups_per_row = K / 256;
    const long long xstride = (long long)16 * K;
    const long long a_row_stride = (long long)groups_per_row * 136;

    const char* row_base[RM];
    _Pragma("unroll") for (int r = 0; r < RM; r++) {
        const int rs = row_start + r * 16;
        const int safe_row = (rs + ml < M) ? (rs + ml) : (M - 1);
        row_base[r] = A + (long long)safe_row * a_row_stride;
    }

    float8_t acc[RM][BV];
    _Pragma("unroll") for (int r = 0; r < RM; r++) {
        _Pragma("unroll") for (int b = 0; b < BV; b++) {
            acc[r][b] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    for (int g = 0; g < groups_per_row; g++) {
        _Float16 sc_h[RM];
        _Float16 zp_h[RM];
        const char* gp[RM];
        _Pragma("unroll") for (int r = 0; r < RM; r++) {
            gp[r] = row_base[r] + g * 136;
            sc_h[r] = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp[r]));
            zp_h[r] = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp[r] + 4));
        }
        const _Float16* xg0 = X + (long long)(batch_start + ml) * K + g * 256;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            // --- kt even half ---
            {
                half16_t s0 = *(const half16_t*)(xg0 + k_off_a + 0 * xstride); // B0
                half16_t s1 = *(const half16_t*)(xg0 + k_off_a + 1 * xstride); // B1

                half16_t a0, a1;
                {
                    unsigned int pk0 = *(const unsigned int*)(gp[0] + 8 + k_off_a / 2);
                    unsigned int pk1 = *(const unsigned int*)(gp[0] + 8 + k_off_a / 2 + 4);
                    DEQUANT_A_FRAG_SCALAR(pk0, pk1, sc_h[0], zp_h[0], a0);
                }
                {
                    unsigned int pk0 = *(const unsigned int*)(gp[1] + 8 + k_off_a / 2);
                    unsigned int pk1 = *(const unsigned int*)(gp[1] + 8 + k_off_a / 2 + 4);
                    DEQUANT_A_FRAG_SCALAR(pk0, pk1, sc_h[1], zp_h[1], a1);
                }

                PIPE_BV6_CONSUME(a0, a1, xg0, k_off_a, xstride, acc, s0, s1);
            }

            // --- kt odd half: no WMMA crosses kt ---
            {
                half16_t s0 = *(const half16_t*)(xg0 + k_off_b + 0 * xstride); // B0
                half16_t s1 = *(const half16_t*)(xg0 + k_off_b + 1 * xstride); // B1

                half16_t a0, a1;
                {
                    unsigned int pk0 = *(const unsigned int*)(gp[0] + 8 + k_off_b / 2);
                    unsigned int pk1 = *(const unsigned int*)(gp[0] + 8 + k_off_b / 2 + 4);
                    DEQUANT_A_FRAG_SCALAR(pk0, pk1, sc_h[0], zp_h[0], a0);
                }
                {
                    unsigned int pk0 = *(const unsigned int*)(gp[1] + 8 + k_off_b / 2);
                    unsigned int pk1 = *(const unsigned int*)(gp[1] + 8 + k_off_b / 2 + 4);
                    DEQUANT_A_FRAG_SCALAR(pk0, pk1, sc_h[1], zp_h[1], a1);
                }

                PIPE_BV6_CONSUME(a0, a1, xg0, k_off_b, xstride, acc, s0, s1);
            }
        }
    }

    // RDNA3 wave32 WMMA: acc[j] = C[2*j + (tid>>4)][tid & 15].
    _Pragma("unroll") for (int r = 0; r < RM; r++) {
        const int rs = row_start + r * 16;
        _Pragma("unroll") for (int b = 0; b < BV; b++) {
            const int out_col = batch_start + b * 16 + ml;
            _Pragma("unroll") for (int j = 0; j < 8; j++) {
                int out_row = rs + 2 * j + (tid >> 4);
                if (out_row < M) Y[(long long)out_col * M + out_row] += acc[r][b][j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Packed-half2 pipe: same two-slot B schedule; A dequant = muse_rm_pk exact.
// ---------------------------------------------------------------------------
__launch_bounds__(32, 2) extern "C" __global__ void
gemm_hfq4g256_residual_wmma_gfx1100_muse_rm2_bv6_pipe_pk2(
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, int M,
    int K, int batch_size) {
    constexpr int RM = 2;
    constexpr int BV = 6;
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * (16 * RM);
    const int batch_start = blockIdx.y * (16 * BV);
    const int ml = tid & 15;
    const int groups_per_row = K / 256;
    const long long xstride = (long long)16 * K;
    const long long a_row_stride = (long long)groups_per_row * 136;

    const char* row_base[RM];
    _Pragma("unroll") for (int r = 0; r < RM; r++) {
        const int rs = row_start + r * 16;
        const int safe_row = (rs + ml < M) ? (rs + ml) : (M - 1);
        row_base[r] = A + (long long)safe_row * a_row_stride;
    }

    float8_t acc[RM][BV];
    _Pragma("unroll") for (int r = 0; r < RM; r++) {
        _Pragma("unroll") for (int b = 0; b < BV; b++) {
            acc[r][b] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    for (int g = 0; g < groups_per_row; g++) {
        __half sc_h[RM];
        __half zp_h[RM];
        const char* gp[RM];
        _Pragma("unroll") for (int r = 0; r < RM; r++) {
            gp[r] = row_base[r] + g * 136;
            sc_h[r] = (__half)__builtin_bit_cast(float, *(const unsigned int*)(gp[r]));
            zp_h[r] = (__half)__builtin_bit_cast(float, *(const unsigned int*)(gp[r] + 4));
        }
        const _Float16* xg0 = X + (long long)(batch_start + ml) * K + g * 256;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            // --- kt even half ---
            {
                half16_t s0 = *(const half16_t*)(xg0 + k_off_a + 0 * xstride); // B0
                half16_t s1 = *(const half16_t*)(xg0 + k_off_a + 1 * xstride); // B1

                half16_t a0, a1;
                {
                    unsigned int pk0 = *(const unsigned int*)(gp[0] + 8 + k_off_a / 2);
                    unsigned int pk1 = *(const unsigned int*)(gp[0] + 8 + k_off_a / 2 + 4);
                    DEQUANT_A_FRAG_PK(pk0, pk1, sc_h[0], zp_h[0], a0);
                }
                {
                    unsigned int pk0 = *(const unsigned int*)(gp[1] + 8 + k_off_a / 2);
                    unsigned int pk1 = *(const unsigned int*)(gp[1] + 8 + k_off_a / 2 + 4);
                    DEQUANT_A_FRAG_PK(pk0, pk1, sc_h[1], zp_h[1], a1);
                }

                PIPE_BV6_CONSUME(a0, a1, xg0, k_off_a, xstride, acc, s0, s1);
            }

            // --- kt odd half ---
            {
                half16_t s0 = *(const half16_t*)(xg0 + k_off_b + 0 * xstride); // B0
                half16_t s1 = *(const half16_t*)(xg0 + k_off_b + 1 * xstride); // B1

                half16_t a0, a1;
                {
                    unsigned int pk0 = *(const unsigned int*)(gp[0] + 8 + k_off_b / 2);
                    unsigned int pk1 = *(const unsigned int*)(gp[0] + 8 + k_off_b / 2 + 4);
                    DEQUANT_A_FRAG_PK(pk0, pk1, sc_h[0], zp_h[0], a0);
                }
                {
                    unsigned int pk0 = *(const unsigned int*)(gp[1] + 8 + k_off_b / 2);
                    unsigned int pk1 = *(const unsigned int*)(gp[1] + 8 + k_off_b / 2 + 4);
                    DEQUANT_A_FRAG_PK(pk0, pk1, sc_h[1], zp_h[1], a1);
                }

                PIPE_BV6_CONSUME(a0, a1, xg0, k_off_b, xstride, acc, s0, s1);
            }
        }
    }

    // RDNA3 wave32 WMMA: acc[j] = C[2*j + (tid>>4)][tid & 15].
    _Pragma("unroll") for (int r = 0; r < RM; r++) {
        const int rs = row_start + r * 16;
        _Pragma("unroll") for (int b = 0; b < BV; b++) {
            const int out_col = batch_start + b * 16 + ml;
            _Pragma("unroll") for (int j = 0; j < 8; j++) {
                int out_row = rs + 2 * j + (tid >> 4);
                if (out_row < M) Y[(long long)out_col * M + out_row] += acc[r][b][j];
            }
        }
    }
}
