// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned multi-row batch-tiled gfx1100 HFQ4-G256 residual GEMM
// (K2). Measurement-only family probing the gfx11 operand-ABI bottleneck:
// baseline muse_bt reloads the same B half16 once per row tile; this kernel
// keeps RM row tiles in one wave and reuses each B fragment across RM WMMA
// calls so X traffic drops by ~RM while A dequant grows linearly with RM.
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx1100_muse_bt.hip (Apache-2.0).
//
// gfx11 B-fragment reuse: RDNA3 wave32 WMMA takes A and B as VGPR half16
// fragments. A is unique per 16-row weight tile; B is the same 16-col X tile
// for every row tile that shares the batch window. Loading B once per batch
// tile and feeding it to RM successive WMMAs tests whether the operand path
// can hold B live across multiple matrix ops (vs re-streaming X from L1/L2
// once per row as muse_bt does). No LDS — pure register reuse.
//
// Preserves the gfx1100 K2 arithmetic exactly per accumulator: each acc[r][b]
// sees the same g/kt pair sequence and
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 calls as muse_bt / k2 baseline.
// Within every g/kt half-step: build each row tile's A fragment, load each B
// half16 once per batch tile, reuse that B across all RM WMMAs.
//
// FULL-TILE PRECONDITION. Host guarantees M % (16*RM) == 0 and
// batch % (16*BV) == 0 for every launched tile (Muse gate/up: M=19968,
// RM*BV=12). Batch indexing is affine. Row-safe A loads and output guards
// remain for defense in depth. Output is Y += acc (caller pre-inits Y).
//
// Grid:  [M/(16*RM), batch/(16*BV), 1]
// Block: [32, 1, 1]. LDS: 0.
// Instantiated: rm2_bv6, rm3_bv4, rm4_bv3, rm6_bv2 (RM*BV = 12).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define GEN_MUSE_RM_BT(RM, BV, SUFFIX)                                                            \
    __launch_bounds__(32, 2) extern "C" __global__ void                                           \
        gemm_hfq4g256_residual_wmma_gfx1100_muse_##SUFFIX(                                        \
            const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y,    \
            int M, int K, int batch_size) {                                                       \
        const int tid = threadIdx.x;                                                              \
        const int row_start = blockIdx.x * (16 * (RM));                                           \
        const int batch_start = blockIdx.y * (16 * (BV));                                         \
        const int ml = tid & 15;                                                                  \
        const int groups_per_row = K / 256;                                                       \
        const long long xstride = (long long)16 * K;                                              \
        const long long a_row_stride = (long long)groups_per_row * 136;                           \
                                                                                                  \
        const char* row_base[RM];                                                                 \
        _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                        \
            const int rs = row_start + r * 16;                                                    \
            const int safe_row = (rs + ml < M) ? (rs + ml) : (M - 1);                             \
            row_base[r] = A + (long long)safe_row * a_row_stride;                                 \
        }                                                                                         \
                                                                                                  \
        float8_t acc[RM][BV];                                                                     \
        _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                        \
            _Pragma("unroll") for (int b = 0; b < (BV); b++) {                                    \
                acc[r][b] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};                                     \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        for (int g = 0; g < groups_per_row; g++) {                                                \
            _Float16 sc_h[RM];                                                                    \
            _Float16 zp_h[RM];                                                                    \
            const char* gp[RM];                                                                   \
            _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                    \
                gp[r] = row_base[r] + g * 136;                                                    \
                sc_h[r] = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp[r]));     \
                zp_h[r] =                                                                         \
                    (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp[r] + 4));       \
            }                                                                                     \
            const _Float16* xg0 = X + (long long)(batch_start + ml) * K + g * 256;                \
                                                                                                  \
            for (int kt = 0; kt < 16; kt += 2) {                                                  \
                const int k_off_a = kt * 16;                                                      \
                const int k_off_b = (kt + 1) * 16;                                                \
                                                                                                  \
                /* --- kt even half: dequant each row A, load each B once, RM reuse --- */        \
                half16_t a_frag[RM];                                                              \
                _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                \
                    unsigned int pk0a = *(const unsigned int*)(gp[r] + 8 + k_off_a / 2);          \
                    unsigned int pk1a = *(const unsigned int*)(gp[r] + 8 + k_off_a / 2 + 4);      \
                    half16_t a_a;                                                                 \
                    a_a[0] = sc_h[r] * (_Float16)(float)((pk0a >> 0) & 0xFu) + zp_h[r];           \
                    a_a[1] = sc_h[r] * (_Float16)(float)((pk0a >> 4) & 0xFu) + zp_h[r];           \
                    a_a[2] = sc_h[r] * (_Float16)(float)((pk0a >> 8) & 0xFu) + zp_h[r];           \
                    a_a[3] = sc_h[r] * (_Float16)(float)((pk0a >> 12) & 0xFu) + zp_h[r];          \
                    a_a[4] = sc_h[r] * (_Float16)(float)((pk0a >> 16) & 0xFu) + zp_h[r];          \
                    a_a[5] = sc_h[r] * (_Float16)(float)((pk0a >> 20) & 0xFu) + zp_h[r];          \
                    a_a[6] = sc_h[r] * (_Float16)(float)((pk0a >> 24) & 0xFu) + zp_h[r];          \
                    a_a[7] = sc_h[r] * (_Float16)(float)((pk0a >> 28) & 0xFu) + zp_h[r];          \
                    a_a[8] = sc_h[r] * (_Float16)(float)((pk1a >> 0) & 0xFu) + zp_h[r];           \
                    a_a[9] = sc_h[r] * (_Float16)(float)((pk1a >> 4) & 0xFu) + zp_h[r];           \
                    a_a[10] = sc_h[r] * (_Float16)(float)((pk1a >> 8) & 0xFu) + zp_h[r];          \
                    a_a[11] = sc_h[r] * (_Float16)(float)((pk1a >> 12) & 0xFu) + zp_h[r];         \
                    a_a[12] = sc_h[r] * (_Float16)(float)((pk1a >> 16) & 0xFu) + zp_h[r];         \
                    a_a[13] = sc_h[r] * (_Float16)(float)((pk1a >> 20) & 0xFu) + zp_h[r];         \
                    a_a[14] = sc_h[r] * (_Float16)(float)((pk1a >> 24) & 0xFu) + zp_h[r];         \
                    a_a[15] = sc_h[r] * (_Float16)(float)((pk1a >> 28) & 0xFu) + zp_h[r];         \
                    a_frag[r] = a_a;                                                              \
                }                                                                                 \
                _Pragma("unroll") for (int b = 0; b < (BV); b++) {                                \
                    half16_t b_a = *(const half16_t*)(xg0 + k_off_a + b * xstride);               \
                    _Pragma("unroll") for (int r = 0; r < (RM); r++) {                            \
                        acc[r][b] =                                                               \
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag[r], b_a, acc[r][b]);\
                    }                                                                             \
                }                                                                                 \
                                                                                                  \
                /* --- kt odd half: same B-reuse pattern --- */                                   \
                _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                \
                    unsigned int pk0b = *(const unsigned int*)(gp[r] + 8 + k_off_b / 2);          \
                    unsigned int pk1b = *(const unsigned int*)(gp[r] + 8 + k_off_b / 2 + 4);      \
                    half16_t a_a;                                                                 \
                    a_a[0] = sc_h[r] * (_Float16)(float)((pk0b >> 0) & 0xFu) + zp_h[r];           \
                    a_a[1] = sc_h[r] * (_Float16)(float)((pk0b >> 4) & 0xFu) + zp_h[r];           \
                    a_a[2] = sc_h[r] * (_Float16)(float)((pk0b >> 8) & 0xFu) + zp_h[r];           \
                    a_a[3] = sc_h[r] * (_Float16)(float)((pk0b >> 12) & 0xFu) + zp_h[r];          \
                    a_a[4] = sc_h[r] * (_Float16)(float)((pk0b >> 16) & 0xFu) + zp_h[r];          \
                    a_a[5] = sc_h[r] * (_Float16)(float)((pk0b >> 20) & 0xFu) + zp_h[r];          \
                    a_a[6] = sc_h[r] * (_Float16)(float)((pk0b >> 24) & 0xFu) + zp_h[r];          \
                    a_a[7] = sc_h[r] * (_Float16)(float)((pk0b >> 28) & 0xFu) + zp_h[r];          \
                    a_a[8] = sc_h[r] * (_Float16)(float)((pk1b >> 0) & 0xFu) + zp_h[r];           \
                    a_a[9] = sc_h[r] * (_Float16)(float)((pk1b >> 4) & 0xFu) + zp_h[r];           \
                    a_a[10] = sc_h[r] * (_Float16)(float)((pk1b >> 8) & 0xFu) + zp_h[r];          \
                    a_a[11] = sc_h[r] * (_Float16)(float)((pk1b >> 12) & 0xFu) + zp_h[r];         \
                    a_a[12] = sc_h[r] * (_Float16)(float)((pk1b >> 16) & 0xFu) + zp_h[r];         \
                    a_a[13] = sc_h[r] * (_Float16)(float)((pk1b >> 20) & 0xFu) + zp_h[r];         \
                    a_a[14] = sc_h[r] * (_Float16)(float)((pk1b >> 24) & 0xFu) + zp_h[r];         \
                    a_a[15] = sc_h[r] * (_Float16)(float)((pk1b >> 28) & 0xFu) + zp_h[r];         \
                    a_frag[r] = a_a;                                                              \
                }                                                                                 \
                _Pragma("unroll") for (int b = 0; b < (BV); b++) {                                \
                    half16_t b_b = *(const half16_t*)(xg0 + k_off_b + b * xstride);               \
                    _Pragma("unroll") for (int r = 0; r < (RM); r++) {                            \
                        acc[r][b] =                                                               \
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag[r], b_b, acc[r][b]);\
                    }                                                                             \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (tid>>4)][tid & 15]. */                            \
        _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                        \
            const int rs = row_start + r * 16;                                                    \
            _Pragma("unroll") for (int b = 0; b < (BV); b++) {                                    \
                const int out_col = batch_start + b * 16 + ml;                                    \
                _Pragma("unroll") for (int j = 0; j < 8; j++) {                                   \
                    int out_row = rs + 2 * j + (tid >> 4);                                        \
                    if (out_row < M) Y[(long long)out_col * M + out_row] += acc[r][b][j];         \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
    }

GEN_MUSE_RM_BT(2, 6, rm2_bv6)
GEN_MUSE_RM_BT(3, 4, rm3_bv4)
GEN_MUSE_RM_BT(4, 3, rm4_bv3)
GEN_MUSE_RM_BT(6, 2, rm6_bv2)
