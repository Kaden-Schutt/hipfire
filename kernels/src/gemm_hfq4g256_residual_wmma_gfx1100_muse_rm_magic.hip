// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned multi-row batch-tiled gfx1100 HFQ4-G256 residual GEMM
// (K2) — FP16 magic-bit A-dequant probe of muse_rm_pk (RM2/BV6 only).
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_pk.hip (Apache-2.0).
//
// Rationale: muse_rm_pk builds each nibble pair via int→float→half
// (__floats2half2_rn) before packed FMA. This probe constructs the same
// half2 quant values with pure integer magic bits + bitcast, removing those
// conversions. Two variants share topology:
//
//   exact  — bits 0x64006400 | q0 | (q1<<12) → half2(1024+q0,1024+q1),
//            packed sub half2(1024), then __hfma2(q, sc2, zp2).
//            Oracle target: bitdiff0 vs scalar/pk dequant.
//
//   fold16 — bits 0x4C004C00 | (q0<<6) | (q1<<18) → half2(16+q0,16+q1);
//            per-group folded bias = __hfma2(sc2, half2(-16), zp2);
//            per pair one __hfma2(qmagic, sc2, folded_bias).
//            Relaxed: nonzero maxabs/maxrel allowed.
//
// Preserves g/kt WMMA call order, X/B layout, output mapping, Y +=, and
// RM2/BV6 B reuse (one B load for two row WMMAs). No LDS, no full FP16
// staging, no new weight layout, no scheduler barriers.
//
// FULL-TILE PRECONDITION. Host guarantees M % (16*RM) == 0 and
// batch % (16*BV) == 0 for every launched tile (Muse gate/up: M=19968 K=6656
// B=192, RM*BV=12). Grid[624,2,1], block[32], LDS0 for that shape.
// Batch indexing is affine. Row-safe A bases and output guards remain for
// defense in depth. Output is Y += acc (caller pre-inits Y).
//
// Grid:  [M/(16*RM), batch/(16*BV), 1]  → rm2: [M/32, B/96, 1]
// Block: [32, 1, 1]. LDS: 0.
// Instantiated: rm2_bv6_magic_exact, rm2_bv6_magic_fold16 (measurement-only).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Exact: one byte x=q0|(q1<<4) → half2 via 1024-bias magic, sub 1024, FMA.
#define MAGIC_EXACT_BYTE_TO_H2(x, sc2, zp2, h1024, out_h2)                                        \
    do {                                                                                          \
        const unsigned _mb =                                                                      \
            0x64006400u | ((unsigned)(x) & 0x0Fu) | (((unsigned)(x) & 0xF0u) << 12);             \
        __half2 _q = __builtin_bit_cast(__half2, _mb);                                            \
        _q = __hsub2(_q, (h1024));                                                                \
        (out_h2) = __hfma2(_q, (sc2), (zp2));                                                     \
    } while (0)

// Fold16: one byte → half2 via 16-bias magic; single packed FMA with folded bias.
#define MAGIC_FOLD_BYTE_TO_H2(x, sc2, bias, out_h2)                                               \
    do {                                                                                          \
        const unsigned _mb =                                                                      \
            0x4C004C00u | (((unsigned)(x) & 0x0Fu) << 6) | (((unsigned)(x) & 0xF0u) << 18);      \
        const __half2 _q = __builtin_bit_cast(__half2, _mb);                                      \
        (out_h2) = __hfma2(_q, (sc2), (bias));                                                    \
    } while (0)

// Four bytes of one u32 pk → four __half2 (exact path).
#define DEQUANT_PK_MAGIC_EXACT(pk, sc2, zp2, h1024, h01, h23, h45, h67)                           \
    do {                                                                                          \
        MAGIC_EXACT_BYTE_TO_H2((pk) & 0xFFu, (sc2), (zp2), (h1024), (h01));                       \
        MAGIC_EXACT_BYTE_TO_H2(((pk) >> 8) & 0xFFu, (sc2), (zp2), (h1024), (h23));                \
        MAGIC_EXACT_BYTE_TO_H2(((pk) >> 16) & 0xFFu, (sc2), (zp2), (h1024), (h45));               \
        MAGIC_EXACT_BYTE_TO_H2(((pk) >> 24) & 0xFFu, (sc2), (zp2), (h1024), (h67));               \
    } while (0)

// Four bytes of one u32 pk → four __half2 (fold16 path).
#define DEQUANT_PK_MAGIC_FOLD(pk, sc2, bias, h01, h23, h45, h67)                                  \
    do {                                                                                          \
        MAGIC_FOLD_BYTE_TO_H2((pk) & 0xFFu, (sc2), (bias), (h01));                                \
        MAGIC_FOLD_BYTE_TO_H2(((pk) >> 8) & 0xFFu, (sc2), (bias), (h23));                         \
        MAGIC_FOLD_BYTE_TO_H2(((pk) >> 16) & 0xFFu, (sc2), (bias), (h45));                        \
        MAGIC_FOLD_BYTE_TO_H2(((pk) >> 24) & 0xFFu, (sc2), (bias), (h67));                        \
    } while (0)

// pk0 + pk1 → half16 A fragment (exact).
#define DEQUANT_A_FRAG_MAGIC_EXACT(pk0, pk1, sc2, zp2, h1024, frag)                               \
    do {                                                                                          \
        __half2 _h[8];                                                                            \
        DEQUANT_PK_MAGIC_EXACT((pk0), (sc2), (zp2), (h1024), _h[0], _h[1], _h[2], _h[3]);         \
        DEQUANT_PK_MAGIC_EXACT((pk1), (sc2), (zp2), (h1024), _h[4], _h[5], _h[6], _h[7]);         \
        (frag) = *reinterpret_cast<const half16_t*>(_h);                                          \
    } while (0)

// pk0 + pk1 → half16 A fragment (fold16).
#define DEQUANT_A_FRAG_MAGIC_FOLD(pk0, pk1, sc2, bias, frag)                                      \
    do {                                                                                          \
        __half2 _h[8];                                                                            \
        DEQUANT_PK_MAGIC_FOLD((pk0), (sc2), (bias), _h[0], _h[1], _h[2], _h[3]);                  \
        DEQUANT_PK_MAGIC_FOLD((pk1), (sc2), (bias), _h[4], _h[5], _h[6], _h[7]);                  \
        (frag) = *reinterpret_cast<const half16_t*>(_h);                                          \
    } while (0)

// IS_FOLD: 0 = exact (sub-1024 + FMA), 1 = fold16 (base-16 bias FMA).
#define GEN_MUSE_RM_MAGIC(RM, BV, SUFFIX, IS_FOLD)                                                \
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
        /* Hoisted magic constants (raw u32 bitcast — no conversions). */                        \
        const __half2 h1024 = __builtin_bit_cast(__half2, 0x64006400u);                           \
        const __half2 h_neg16 = __builtin_bit_cast(__half2, 0xCC00CC00u);                         \
        (void)h1024;                                                                              \
        (void)h_neg16;                                                                            \
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
            __half2 sc2[RM];                                                                      \
            __half2 zp2[RM];                                                                      \
            __half2 fold_bias[RM];                                                                \
            const char* gp[RM];                                                                   \
            _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                    \
                gp[r] = row_base[r] + g * 136;                                                    \
                const __half sc_h =                                                               \
                    (__half)__builtin_bit_cast(float, *(const unsigned int*)(gp[r]));             \
                const __half zp_h =                                                               \
                    (__half)__builtin_bit_cast(float, *(const unsigned int*)(gp[r] + 4));         \
                sc2[r] = __half2half2(sc_h);                                                      \
                zp2[r] = __half2half2(zp_h);                                                      \
                if (IS_FOLD) {                                                                    \
                    fold_bias[r] = __hfma2(sc2[r], h_neg16, zp2[r]);                              \
                }                                                                                 \
            }                                                                                     \
            const _Float16* xg0 = X + (long long)(batch_start + ml) * K + g * 256;                \
                                                                                                  \
            for (int kt = 0; kt < 16; kt += 2) {                                                  \
                const int k_off_a = kt * 16;                                                      \
                const int k_off_b = (kt + 1) * 16;                                                \
                                                                                                  \
                /* --- kt even half: magic A dequant, load B once, RM reuse --- */                \
                half16_t a_frag[RM];                                                              \
                _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                \
                    unsigned int pk0a = *(const unsigned int*)(gp[r] + 8 + k_off_a / 2);          \
                    unsigned int pk1a = *(const unsigned int*)(gp[r] + 8 + k_off_a / 2 + 4);      \
                    if (IS_FOLD) {                                                                \
                        DEQUANT_A_FRAG_MAGIC_FOLD(pk0a, pk1a, sc2[r], fold_bias[r], a_frag[r]);   \
                    } else {                                                                      \
                        DEQUANT_A_FRAG_MAGIC_EXACT(pk0a, pk1a, sc2[r], zp2[r], h1024, a_frag[r]); \
                    }                                                                             \
                }                                                                                 \
                _Pragma("unroll") for (int b = 0; b < (BV); b++) {                                \
                    half16_t b_a = *(const half16_t*)(xg0 + k_off_a + b * xstride);               \
                    _Pragma("unroll") for (int r = 0; r < (RM); r++) {                            \
                        acc[r][b] =                                                               \
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag[r], b_a, acc[r][b]);\
                    }                                                                             \
                }                                                                                 \
                                                                                                  \
                /* --- kt odd half: same magic A + B-reuse pattern --- */                         \
                _Pragma("unroll") for (int r = 0; r < (RM); r++) {                                \
                    unsigned int pk0b = *(const unsigned int*)(gp[r] + 8 + k_off_b / 2);          \
                    unsigned int pk1b = *(const unsigned int*)(gp[r] + 8 + k_off_b / 2 + 4);      \
                    if (IS_FOLD) {                                                                \
                        DEQUANT_A_FRAG_MAGIC_FOLD(pk0b, pk1b, sc2[r], fold_bias[r], a_frag[r]);   \
                    } else {                                                                      \
                        DEQUANT_A_FRAG_MAGIC_EXACT(pk0b, pk1b, sc2[r], zp2[r], h1024, a_frag[r]); \
                    }                                                                             \
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

// RM2 retains one B load reused across two row WMMAs (BV=6 → B tile 96).
GEN_MUSE_RM_MAGIC(2, 6, rm2_bv6_magic_exact, 0)
GEN_MUSE_RM_MAGIC(2, 6, rm2_bv6_magic_fold16, 1)
