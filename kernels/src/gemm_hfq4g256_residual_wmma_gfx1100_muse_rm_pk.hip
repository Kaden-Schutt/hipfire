// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned multi-row batch-tiled gfx1100 HFQ4-G256 residual GEMM
// (K2) — packed-half2 (pk) A-dequant variant of muse_rm_bt.
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_bt.hip (Apache-2.0).
// Packed A dequant follows gemm_hfq4g256_residual_fp16.hip (__hfma2 / half2).
//
// Rationale: scalar dequant does 16 independent f16 mul-adds per A fragment.
// This candidate packs each pair of nibbles into __half2 and uses v_pk_fma_f16
// via __hfma2 so eight packed FMAs cover the 16-nibble fragment. sc/zp are
// broadcast with __half2half2. Completed eight __half2 values are bit-cast
// (no value change) to the half16 WMMA A operand.
//
// EXACTNESS IS ORACLE-GATED. __hfma2 is a fused packed FMA; the scalar path
// is separate f16 mul then add. Bit-identity vs muse_rm_bt is NOT assumed —
// full-output bitdiff against the existing oracle decides contraction.
//
// Preserves g/kt WMMA call order, X/B layout, output mapping, and Y +=.
// No lane broadcast, no LDS, no altered launch_bounds / scheduling flags,
// no reordered WMMA calls. B is still loaded once per batch tile and reused
// across RM row WMMAs (RM2 keeps one B load for two row WMMAs).
//
// FULL-TILE PRECONDITION. Host guarantees M % (16*RM) == 0 and
// batch % (16*BV) == 0 for every launched tile (Muse gate/up: M=19968 K=6656,
// RM*BV=12). Batch indexing is affine. Row-safe A bases and output guards
// remain for defense in depth. Output is Y += acc (caller pre-inits Y).
//
// Grid:  [M/(16*RM), batch/(16*BV), 1]
//   rm1: [M/16,  B/192, 1]
//   rm2: [M/32,  B/96,  1]
// Block: [32, 1, 1]. LDS: 0.
// Instantiated: rm1_bv12_pk, rm2_bv6_pk (measurement-only; no production select).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Dequant 8 packed nibbles (one u32) → 4 × __half2 via fused packed FMA.
// Pair shifts: (0,4), (8,12), (16,20), (24,28). q via __floats2half2_rn;
// sc2/zp2 broadcast from scalar halves; __hfma2(q2, sc2, zp2).
// Exactness vs scalar mul+add is oracle-gated (fused rounding).
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

// Build half16 A fragment from pk0 (pairs 0..3) + pk1 (pairs 4..7).
// Eight __half2 values are reinterpreted as half16 without changing bits.
#define DEQUANT_A_FRAG_PK(pk0, pk1, sc_h, zp_h, frag)                                             \
    do {                                                                                          \
        __half2 _h[8];                                                                            \
        DEQUANT_PK_H2((pk0), (sc_h), (zp_h), _h[0], _h[1], _h[2], _h[3]);                        \
        DEQUANT_PK_H2((pk1), (sc_h), (zp_h), _h[4], _h[5], _h[6], _h[7]);                        \
        (frag) = *reinterpret_cast<const half16_t*>(_h);                                          \
    } while (0)

#define GEN_MUSE_RM_PK(RM, BV, SUFFIX)                                                            \
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
            __half sc_h[RM];                                                                      \
            __half zp_h[RM];                                                                      \
            const char* gp[RM];                                                                   \
            _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                    \
                gp[r] = row_base[r] + g * 136;                                                    \
                sc_h[r] = (__half)__builtin_bit_cast(float, *(const unsigned int*)(gp[r]));       \
                zp_h[r] =                                                                         \
                    (__half)__builtin_bit_cast(float, *(const unsigned int*)(gp[r] + 4));         \
            }                                                                                     \
            const _Float16* xg0 = X + (long long)(batch_start + ml) * K + g * 256;                \
                                                                                                  \
            for (int kt = 0; kt < 16; kt += 2) {                                                  \
                const int k_off_a = kt * 16;                                                      \
                const int k_off_b = (kt + 1) * 16;                                                \
                                                                                                  \
                /* --- kt even half: packed-half2 A dequant, load B once, RM reuse --- */         \
                half16_t a_frag[RM];                                                              \
                _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                \
                    unsigned int pk0a = *(const unsigned int*)(gp[r] + 8 + k_off_a / 2);          \
                    unsigned int pk1a = *(const unsigned int*)(gp[r] + 8 + k_off_a / 2 + 4);      \
                    DEQUANT_A_FRAG_PK(pk0a, pk1a, sc_h[r], zp_h[r], a_frag[r]);                   \
                }                                                                                 \
                _Pragma("unroll") for (int b = 0; b < (BV); b++) {                                \
                    half16_t b_a = *(const half16_t*)(xg0 + k_off_a + b * xstride);               \
                    _Pragma("unroll") for (int r = 0; r < (RM); r++) {                            \
                        acc[r][b] =                                                               \
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag[r], b_a, acc[r][b]);\
                    }                                                                             \
                }                                                                                 \
                                                                                                  \
                /* --- kt odd half: same packed A + B-reuse pattern --- */                        \
                _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                \
                    unsigned int pk0b = *(const unsigned int*)(gp[r] + 8 + k_off_b / 2);          \
                    unsigned int pk1b = *(const unsigned int*)(gp[r] + 8 + k_off_b / 2 + 4);      \
                    DEQUANT_A_FRAG_PK(pk0b, pk1b, sc_h[r], zp_h[r], a_frag[r]);                   \
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

// RM1 matches full-12 batch tiling (BV=12); single row tile, no multi-row B reuse.
GEN_MUSE_RM_PK(1, 12, rm1_bv12_pk)
// RM2 retains one B load reused across two row WMMAs.
GEN_MUSE_RM_PK(2, 6, rm2_bv6_pk)
