// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned RM2/BV6 gfx1100 HFQ4-G256 residual GEMM accuracy probe:
// packs the two row tiles into one half16 accumulator via the tied FP16-output
// WMMA builtin (OPSEL low then high). Measurement-only; no production select.
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_bt.hip GEN_MUSE_RM_BT
// (2,6,rm2_bv6) (Apache-2.0). Preserves g/kt order, A dequant, B loads, grid,
// residual Y+=. Only accumulator representation and WMMA/epilogue change.
//
// Accumulator: half16_t acc_pair[6]. Low halves own r=0, high halves r=1.
// Updates use __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32 with literal
// false then true. Ordinary non-tied FP16 builtin and inline ISA are forbidden.
//
// FULL-TILE PRECONDITION. Host guarantees exact M=19968, K=6656, batch=192.
// Grid:  [624, 2, 1]  (= M/32, batch/96)
// Block: [32, 1, 1]. LDS: 0.
// Symbol: gemm_hfq4g256_residual_wmma_gfx1100_muse_rm2_bv6_f16acc_pair

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;

__launch_bounds__(32, 2) extern "C" __global__ void
gemm_hfq4g256_residual_wmma_gfx1100_muse_rm2_bv6_f16acc_pair(
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

    // Six paired half16 accumulators: low half = r0, high half = r1.
    half16_t acc_pair[BV];
    _Pragma("unroll") for (int b = 0; b < BV; b++) {
        acc_pair[b] = half16_t{(_Float16)0, (_Float16)0, (_Float16)0, (_Float16)0,
                               (_Float16)0, (_Float16)0, (_Float16)0, (_Float16)0,
                               (_Float16)0, (_Float16)0, (_Float16)0, (_Float16)0,
                               (_Float16)0, (_Float16)0, (_Float16)0, (_Float16)0};
    }

    for (int g = 0; g < groups_per_row; g++) {
        _Float16 sc_h[RM];
        _Float16 zp_h[RM];
        const char* gp[RM];
        _Pragma("unroll") for (int r = 0; r < RM; r++) {
            gp[r] = row_base[r] + g * 136;
            sc_h[r] = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp[r]));
            zp_h[r] =
                (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp[r] + 4));
        }
        const _Float16* xg0 = X + (long long)(batch_start + ml) * K + g * 256;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            /* --- kt even half: dequant each row A, load each B once, RM reuse --- */
            half16_t a_frag[RM];
            _Pragma("unroll") for (int r = 0; r < RM; r++) {
                unsigned int pk0a = *(const unsigned int*)(gp[r] + 8 + k_off_a / 2);
                unsigned int pk1a = *(const unsigned int*)(gp[r] + 8 + k_off_a / 2 + 4);
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
                // r0 low (OPSEL false), then r1 high (OPSEL true); tied preserves other half.
                acc_pair[b] = __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32(
                    a_frag[0], b_a, acc_pair[b], false);
                acc_pair[b] = __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32(
                    a_frag[1], b_a, acc_pair[b], true);
            }

            /* --- kt odd half: same B-reuse pattern --- */
            _Pragma("unroll") for (int r = 0; r < RM; r++) {
                unsigned int pk0b = *(const unsigned int*)(gp[r] + 8 + k_off_b / 2);
                unsigned int pk1b = *(const unsigned int*)(gp[r] + 8 + k_off_b / 2 + 4);
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
                acc_pair[b] = __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32(
                    a_frag[0], b_b, acc_pair[b], false);
                acc_pair[b] = __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32(
                    a_frag[1], b_b, acc_pair[b], true);
            }
        }
    }

    // Oracle coordinate: out_row = rs + 2*j + (tid>>4), out_col = batch_start + b*16 + ml.
    // Paired layout: even vector slots are r=0 (low), odd are r=1 (high) → index 2*j+r.
    _Pragma("unroll") for (int b = 0; b < BV; b++) {
        const int out_col = batch_start + b * 16 + ml;
        _Pragma("unroll") for (int r = 0; r < RM; r++) {
            const int rs = row_start + r * 16;
            _Pragma("unroll") for (int j = 0; j < 8; j++) {
                int out_row = rs + 2 * j + (tid >> 4);
                if (out_row < M)
                    Y[(long long)out_col * M + out_row] += (float)acc_pair[b][2 * j + r];
            }
        }
    }
}
