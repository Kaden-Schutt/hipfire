// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned multi-row batch-tiled gfx1100 HFQ4-G256 residual GEMM
// (K2) — fragment-major (fg) A weight-layout candidate of muse_rm_bt RM2/BV6.
//
// MEASUREMENT-ONLY. Not wired into production selectors/forward/loaders.
// Hypothesis: row-major A makes same-kt nibble loads stride by
// groups_per_row*136 (=3536 B at K=6656) across the 16 rows of a WMMA tile
// (poor coalescing). Fragment-major packs each kt's 16×8 B row strip
// contiguously so the 16 active ml lanes issue a single 128 B coalesced
// transaction per fragment. Byte size is unchanged:
// 16 rows × 136 B/group = 2176 B per (tile, group).
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_bt.hip (Apache-2.0).
// Scalar f16 affine dequant, B load/reuse, g/kt traversal, WMMA calls,
// accumulator shape, output mapping, and Y += are byte-for-byte equivalent
// to RM2/BV6 muse_rm_bt; only A address calculation changes.
//
// Device layout per 16-row tile t and K-group g (groups_per_row = K/256):
//   group_base = A + (t * groups_per_row + g) * 2176
//   bytes 0..127:   16 row headers; row ml at ml*8  (f32 scale, f32 zero)
//   bytes 128..2175: 16 K fragments; fragment kt has 16 rows × 8 packed
//                    bytes, row ml at 128 + kt*128 + ml*8
// Source mapping from original row-major 136 B groups:
//   row=(t*16+ml), group g, fragment kt
//     ← row*gpr*136 + g*136 + 8 + kt*8
// Lanes tid&15 retain paired-duplicate semantics (both wave halves load the
// same A row / header / q strip via ml = tid & 15).
//
// FULL-TILE PRECONDITION. Host hard-gates M=19968 (M%32==0) and batch%96==0
// for every launched tile (Muse gate/up: M=19968, K=6656, B=192). Batch
// indexing is affine; tile_id is exact (no safe-row clamp/div). Output
// guards remain for defense in depth. Output is Y += acc (caller pre-inits Y).
//
// Grid:  [M/32, batch/96, 1] = [624, 2, 1] at the Muse B192 gate/up shape.
// Block: [32, 1, 1]. LDS: 0.
// Instantiated: rm2_bv6_fg only (measurement candidate).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// RM=2, BV=6 — sole fragment-major measurement symbol.
__launch_bounds__(32, 2) extern "C" __global__ void
gemm_hfq4g256_residual_wmma_gfx1100_muse_rm2_bv6_fg(
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, int M,
    int K, int batch_size) {
    constexpr int RM = 2;
    constexpr int BV = 6;
    constexpr long long kTileGroupBytes = 2176LL; // 16 * 136

    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * (16 * RM);
    const int batch_start = blockIdx.y * (16 * BV);
    const int ml = tid & 15;
    const int groups_per_row = K / 256;
    const long long xstride = (long long)16 * K;

    // Host gates full tiles: tile t = blockIdx.x*RM + r; in-tile row = ml.
    int tile_id[RM];
    _Pragma("unroll") for (int r = 0; r < RM; r++) {
        tile_id[r] = blockIdx.x * RM + r;
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
        const char* gb[RM];
        _Pragma("unroll") for (int r = 0; r < RM; r++) {
            // group_base = A + (t * gpr + g) * 2176
            gb[r] = A + ((long long)tile_id[r] * groups_per_row + g) * kTileGroupBytes;
            // header row ml at ml*8: [f32 scale][f32 zero]
            const char* hdr = gb[r] + ml * 8;
            sc_h[r] = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(hdr));
            zp_h[r] = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(hdr + 4));
        }
        const _Float16* xg0 = X + (long long)(batch_start + ml) * K + g * 256;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            /* --- kt even half: dequant each row A, load each B once, RM reuse --- */
            half16_t a_frag[RM];
            _Pragma("unroll") for (int r = 0; r < RM; r++) {
                // fragment kt, row ml at 128 + kt*128 + ml*8 (8 packed bytes)
                const char* qp = gb[r] + 128 + kt * 128 + ml * 8;
                unsigned int pk0a = *(const unsigned int*)(qp);
                unsigned int pk1a = *(const unsigned int*)(qp + 4);
                half16_t a_a;
                a_a[0] = sc_h[r] * (_Float16)(float)((pk0a >> 0) & 0xFu) + zp_h[r];
                a_a[1] = sc_h[r] * (_Float16)(float)((pk0a >> 4) & 0xFu) + zp_h[r];
                a_a[2] = sc_h[r] * (_Float16)(float)((pk0a >> 8) & 0xFu) + zp_h[r];
                a_a[3] = sc_h[r] * (_Float16)(float)((pk0a >> 12) & 0xFu) + zp_h[r];
                a_a[4] = sc_h[r] * (_Float16)(float)((pk0a >> 16) & 0xFu) + zp_h[r];
                a_a[5] = sc_h[r] * (_Float16)(float)((pk0a >> 20) & 0xFu) + zp_h[r];
                a_a[6] = sc_h[r] * (_Float16)(float)((pk0a >> 24) & 0xFu) + zp_h[r];
                a_a[7] = sc_h[r] * (_Float16)(float)((pk0a >> 28) & 0xFu) + zp_h[r];
                a_a[8] = sc_h[r] * (_Float16)(float)((pk1a >> 0) & 0xFu) + zp_h[r];
                a_a[9] = sc_h[r] * (_Float16)(float)((pk1a >> 4) & 0xFu) + zp_h[r];
                a_a[10] = sc_h[r] * (_Float16)(float)((pk1a >> 8) & 0xFu) + zp_h[r];
                a_a[11] = sc_h[r] * (_Float16)(float)((pk1a >> 12) & 0xFu) + zp_h[r];
                a_a[12] = sc_h[r] * (_Float16)(float)((pk1a >> 16) & 0xFu) + zp_h[r];
                a_a[13] = sc_h[r] * (_Float16)(float)((pk1a >> 20) & 0xFu) + zp_h[r];
                a_a[14] = sc_h[r] * (_Float16)(float)((pk1a >> 24) & 0xFu) + zp_h[r];
                a_a[15] = sc_h[r] * (_Float16)(float)((pk1a >> 28) & 0xFu) + zp_h[r];
                a_frag[r] = a_a;
            }
            _Pragma("unroll") for (int b = 0; b < BV; b++) {
                half16_t b_a = *(const half16_t*)(xg0 + k_off_a + b * xstride);
                _Pragma("unroll") for (int r = 0; r < RM; r++) {
                    acc[r][b] =
                        __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag[r], b_a, acc[r][b]);
                }
            }

            /* --- kt odd half: same B-reuse pattern --- */
            _Pragma("unroll") for (int r = 0; r < RM; r++) {
                const char* qp = gb[r] + 128 + (kt + 1) * 128 + ml * 8;
                unsigned int pk0b = *(const unsigned int*)(qp);
                unsigned int pk1b = *(const unsigned int*)(qp + 4);
                half16_t a_a;
                a_a[0] = sc_h[r] * (_Float16)(float)((pk0b >> 0) & 0xFu) + zp_h[r];
                a_a[1] = sc_h[r] * (_Float16)(float)((pk0b >> 4) & 0xFu) + zp_h[r];
                a_a[2] = sc_h[r] * (_Float16)(float)((pk0b >> 8) & 0xFu) + zp_h[r];
                a_a[3] = sc_h[r] * (_Float16)(float)((pk0b >> 12) & 0xFu) + zp_h[r];
                a_a[4] = sc_h[r] * (_Float16)(float)((pk0b >> 16) & 0xFu) + zp_h[r];
                a_a[5] = sc_h[r] * (_Float16)(float)((pk0b >> 20) & 0xFu) + zp_h[r];
                a_a[6] = sc_h[r] * (_Float16)(float)((pk0b >> 24) & 0xFu) + zp_h[r];
                a_a[7] = sc_h[r] * (_Float16)(float)((pk0b >> 28) & 0xFu) + zp_h[r];
                a_a[8] = sc_h[r] * (_Float16)(float)((pk1b >> 0) & 0xFu) + zp_h[r];
                a_a[9] = sc_h[r] * (_Float16)(float)((pk1b >> 4) & 0xFu) + zp_h[r];
                a_a[10] = sc_h[r] * (_Float16)(float)((pk1b >> 8) & 0xFu) + zp_h[r];
                a_a[11] = sc_h[r] * (_Float16)(float)((pk1b >> 12) & 0xFu) + zp_h[r];
                a_a[12] = sc_h[r] * (_Float16)(float)((pk1b >> 16) & 0xFu) + zp_h[r];
                a_a[13] = sc_h[r] * (_Float16)(float)((pk1b >> 20) & 0xFu) + zp_h[r];
                a_a[14] = sc_h[r] * (_Float16)(float)((pk1b >> 24) & 0xFu) + zp_h[r];
                a_a[15] = sc_h[r] * (_Float16)(float)((pk1b >> 28) & 0xFu) + zp_h[r];
                a_frag[r] = a_a;
            }
            _Pragma("unroll") for (int b = 0; b < BV; b++) {
                half16_t b_b = *(const half16_t*)(xg0 + k_off_b + b * xstride);
                _Pragma("unroll") for (int r = 0; r < RM; r++) {
                    acc[r][b] =
                        __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag[r], b_b, acc[r][b]);
                }
            }
        }
    }

    /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (tid>>4)][tid & 15]. */
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
