// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned multi-row batch-tiled gfx1100 HFQ4-G256 residual GEMM
// (K2) — half-broadcast (hb) variant of muse_rm_bt.
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_bt.hip (Apache-2.0).
//
// Rationale: RDNA3 wave32 WMMA A-fragment layout duplicates the same 16-row
// weight tile across lane pairs (lane i and i+16 hold identical A VGPRs for
// the two C-row parities). muse_rm_bt therefore loads+dequants A twice per
// unique weight row. This candidate keeps the RM/BV B-reuse structure and
// exact g/kt WMMA order, but only lanes 0..15 touch A: they load packed
// nibbles and run the scalar f16 dequant, then bit-broadcast the raw 8×u32
// image of each half16 to the paired upper lane via wave32 __shfl from
// source lane (tid&15). No arithmetic re-dequant on the upper half, no LDS.
// B fragments stay directly loaded (batch-unique). Y += semantics unchanged.
//
// Preserves per-accumulator K2 order: same g/kt half-step sequence and
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 calls as muse_rm_bt.
//
// FULL-TILE PRECONDITION. Host guarantees M % (16*RM) == 0 and
// batch % (16*BV) == 0 for every launched tile (Muse gate/up: M=19968,
// RM*BV=12). Batch indexing is affine. Row-safe A bases and output guards
// remain for defense in depth. Output is Y += acc (caller pre-inits Y).
//
// Grid:  [M/(16*RM), batch/(16*BV), 1]
// Block: [32, 1, 1]. LDS: 0.
// Instantiated: rm2_bv6_hb, rm4_bv3_hb (best row-reuse shapes only).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Bit-broadcast half16 from lane (tid&15) to its paired upper lane.
// eight u32 words cover the 32-byte half16; no value reconstruction.
#define HB_BROADCAST_HALF16(frag)                                                                 \
    do {                                                                                          \
        unsigned int* _hb_w = reinterpret_cast<unsigned int*>(&(frag));                           \
        _Pragma("unroll") for (int _hb_i = 0; _hb_i < 8; _hb_i++) {                               \
            _hb_w[_hb_i] = __shfl(_hb_w[_hb_i], ml, 32);                                          \
        }                                                                                         \
    } while (0)

#define GEN_MUSE_RM_HB(RM, BV, SUFFIX)                                                            \
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
            }                                                                                     \
            /* Scale/zp only needed on lower half (sole dequant producers). */                    \
            if (tid < 16) {                                                                       \
                _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                \
                    sc_h[r] =                                                                     \
                        (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp[r]));       \
                    zp_h[r] =                                                                     \
                        (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp[r] + 4));   \
                }                                                                                 \
            }                                                                                     \
            const _Float16* xg0 = X + (long long)(batch_start + ml) * K + g * 256;                \
                                                                                                  \
            for (int kt = 0; kt < 16; kt += 2) {                                                  \
                const int k_off_a = kt * 16;                                                      \
                const int k_off_b = (kt + 1) * 16;                                                \
                                                                                                  \
                /* --- kt even half: lower-lane A dequant + bit-broadcast, B reuse --- */         \
                half16_t a_frag[RM];                                                              \
                _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                \
                    half16_t a_a;                                                                 \
                    if (tid < 16) {                                                               \
                        unsigned int pk0a = *(const unsigned int*)(gp[r] + 8 + k_off_a / 2);      \
                        unsigned int pk1a =                                                       \
                            *(const unsigned int*)(gp[r] + 8 + k_off_a / 2 + 4);                  \
                        a_a[0] = sc_h[r] * (_Float16)(float)((pk0a >> 0) & 0xFu) + zp_h[r];       \
                        a_a[1] = sc_h[r] * (_Float16)(float)((pk0a >> 4) & 0xFu) + zp_h[r];       \
                        a_a[2] = sc_h[r] * (_Float16)(float)((pk0a >> 8) & 0xFu) + zp_h[r];       \
                        a_a[3] = sc_h[r] * (_Float16)(float)((pk0a >> 12) & 0xFu) + zp_h[r];      \
                        a_a[4] = sc_h[r] * (_Float16)(float)((pk0a >> 16) & 0xFu) + zp_h[r];      \
                        a_a[5] = sc_h[r] * (_Float16)(float)((pk0a >> 20) & 0xFu) + zp_h[r];      \
                        a_a[6] = sc_h[r] * (_Float16)(float)((pk0a >> 24) & 0xFu) + zp_h[r];      \
                        a_a[7] = sc_h[r] * (_Float16)(float)((pk0a >> 28) & 0xFu) + zp_h[r];      \
                        a_a[8] = sc_h[r] * (_Float16)(float)((pk1a >> 0) & 0xFu) + zp_h[r];       \
                        a_a[9] = sc_h[r] * (_Float16)(float)((pk1a >> 4) & 0xFu) + zp_h[r];       \
                        a_a[10] = sc_h[r] * (_Float16)(float)((pk1a >> 8) & 0xFu) + zp_h[r];      \
                        a_a[11] = sc_h[r] * (_Float16)(float)((pk1a >> 12) & 0xFu) + zp_h[r];     \
                        a_a[12] = sc_h[r] * (_Float16)(float)((pk1a >> 16) & 0xFu) + zp_h[r];     \
                        a_a[13] = sc_h[r] * (_Float16)(float)((pk1a >> 20) & 0xFu) + zp_h[r];     \
                        a_a[14] = sc_h[r] * (_Float16)(float)((pk1a >> 24) & 0xFu) + zp_h[r];     \
                        a_a[15] = sc_h[r] * (_Float16)(float)((pk1a >> 28) & 0xFu) + zp_h[r];     \
                    }                                                                             \
                    HB_BROADCAST_HALF16(a_a);                                                     \
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
                /* --- kt odd half: same A half-broadcast + B-reuse pattern --- */                \
                _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                \
                    half16_t a_a;                                                                 \
                    if (tid < 16) {                                                               \
                        unsigned int pk0b = *(const unsigned int*)(gp[r] + 8 + k_off_b / 2);      \
                        unsigned int pk1b =                                                       \
                            *(const unsigned int*)(gp[r] + 8 + k_off_b / 2 + 4);                  \
                        a_a[0] = sc_h[r] * (_Float16)(float)((pk0b >> 0) & 0xFu) + zp_h[r];       \
                        a_a[1] = sc_h[r] * (_Float16)(float)((pk0b >> 4) & 0xFu) + zp_h[r];       \
                        a_a[2] = sc_h[r] * (_Float16)(float)((pk0b >> 8) & 0xFu) + zp_h[r];       \
                        a_a[3] = sc_h[r] * (_Float16)(float)((pk0b >> 12) & 0xFu) + zp_h[r];      \
                        a_a[4] = sc_h[r] * (_Float16)(float)((pk0b >> 16) & 0xFu) + zp_h[r];      \
                        a_a[5] = sc_h[r] * (_Float16)(float)((pk0b >> 20) & 0xFu) + zp_h[r];      \
                        a_a[6] = sc_h[r] * (_Float16)(float)((pk0b >> 24) & 0xFu) + zp_h[r];      \
                        a_a[7] = sc_h[r] * (_Float16)(float)((pk0b >> 28) & 0xFu) + zp_h[r];      \
                        a_a[8] = sc_h[r] * (_Float16)(float)((pk1b >> 0) & 0xFu) + zp_h[r];       \
                        a_a[9] = sc_h[r] * (_Float16)(float)((pk1b >> 4) & 0xFu) + zp_h[r];       \
                        a_a[10] = sc_h[r] * (_Float16)(float)((pk1b >> 8) & 0xFu) + zp_h[r];      \
                        a_a[11] = sc_h[r] * (_Float16)(float)((pk1b >> 12) & 0xFu) + zp_h[r];     \
                        a_a[12] = sc_h[r] * (_Float16)(float)((pk1b >> 16) & 0xFu) + zp_h[r];     \
                        a_a[13] = sc_h[r] * (_Float16)(float)((pk1b >> 20) & 0xFu) + zp_h[r];     \
                        a_a[14] = sc_h[r] * (_Float16)(float)((pk1b >> 24) & 0xFu) + zp_h[r];     \
                        a_a[15] = sc_h[r] * (_Float16)(float)((pk1b >> 28) & 0xFu) + zp_h[r];     \
                    }                                                                             \
                    HB_BROADCAST_HALF16(a_a);                                                     \
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

GEN_MUSE_RM_HB(2, 6, rm2_bv6_hb)
GEN_MUSE_RM_HB(4, 3, rm4_bv3_hb)
