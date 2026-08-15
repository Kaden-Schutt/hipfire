// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned multi-wave batch-tiled gfx1100 HFQ4-G256 residual GEMM
// with LDS-staged X groups (BT6 / K2 / MW8). Candidate for exact gate/up on
// gfx1100.
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx1100_muse_bt.hip and
// gemm_hfq4g256_residual_wmma_gfx1100_muse_mw_bt.hip (Apache-2.0).
//
// One 256-thread block = 8 wave32 row tiles (128 M rows) × 6 independent
// 16-column batch tiles (96 N). For each HFQ group g the block cooperatively
// stages X[batch_start..+96, g*256..+256] into __shared__ x_lds[96*256]
// (49,152 B), then all eight waves reuse that group across their row tiles.
// Cuts external X traffic ~16× versus re-reading the same tile per 16-row
// block, while preserving the exact per-accumulator K2 g/kt WMMA order.
//
// LDS layout is tile-major [kt][batch][ki] with offset
//   ((kt * 96 + batch_local) * 16 + ki)
// so 16 simultaneous 32-byte half16 loads across batch lanes hit minimum
// four-way bank pressure (not the 16-way conflict of row-major [batch][256]).
// Cooperative fill: 1,536 aligned half16 chunks, v=tid; v<1536; v+=256 with
//   kt = v / 96, batch_local = v % 96
// copying X[(batch_start+batch)*K + g*256 + kt*16] → LDS[(kt*96+batch)*16].
// Compute loads B half16 at (kt*96 + (b*16+ml))*16. Exactly two
// __syncthreads per group (after stage, before overwrite); none inside the kt
// loop. Barriers are block-uniform — no early returns around them.
//
// FULL-TILE PRECONDITION. Host guarantees M % 128 == 0 and batch % 96 == 0 for
// every launched tile (M=19968, K=6656, B multiple of 96; B192 canonical).
// Safe row/output checks retained. Output is Y += acc (caller pre-inits Y).
//
// Grid:  [M/128, batch/96, 1]
// Block: [256, 1, 1]. LDS: 49152 B static.
// Symbol: gemm_hfq4g256_residual_wmma_gfx1100_muse_lds_g256_mw8_bt6

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(256, 1) extern "C" __global__ void
gemm_hfq4g256_residual_wmma_gfx1100_muse_lds_g256_mw8_bt6(
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, int M,
    int K, int batch_size) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave_id = tid >> 5;
    const int row_start = (blockIdx.x * 8 + wave_id) * 16;
    const int batch_start = blockIdx.y * 96;
    const int ml = lane & 15;
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M > 0 ? M - 1 : 0);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    // Tile-major [kt][batch][ki]: offset ((kt * 96 + batch_local) * 16 + ki).
    __shared__ _Float16 x_lds[96 * 256];

    float8_t acc[6];
    for (int b = 0; b < 6; b++) acc[b] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        // Stage full X group: 16 kt × 96 batch = 1,536 half16 vectors.
        for (int v = tid; v < 1536; v += 256) {
            const int kt = v / 96;
            const int batch_local = v % 96;
            half16_t val =
                *(const half16_t*)(X + (long long)(batch_start + batch_local) * K + g * 256 +
                                   kt * 16);
            *(half16_t*)(x_lds + (long long)(kt * 96 + batch_local) * 16) = val;
        }
        __syncthreads();

        if (row_start < M) {
            const char* gp = row_base + g * 136;
            _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
            _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            for (int kt = 0; kt < 16; kt += 2) {
                const int k_off_a = kt * 16;
                unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
                unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);
                const int k_off_b = (kt + 1) * 16;
                unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
                unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);
                half16_t a_a;
                a_a[0] = sc_h * (_Float16)(float)((pk0a >> 0) & 0xFu) + zp_h;
                a_a[1] = sc_h * (_Float16)(float)((pk0a >> 4) & 0xFu) + zp_h;
                a_a[2] = sc_h * (_Float16)(float)((pk0a >> 8) & 0xFu) + zp_h;
                a_a[3] = sc_h * (_Float16)(float)((pk0a >> 12) & 0xFu) + zp_h;
                a_a[4] = sc_h * (_Float16)(float)((pk0a >> 16) & 0xFu) + zp_h;
                a_a[5] = sc_h * (_Float16)(float)((pk0a >> 20) & 0xFu) + zp_h;
                a_a[6] = sc_h * (_Float16)(float)((pk0a >> 24) & 0xFu) + zp_h;
                a_a[7] = sc_h * (_Float16)(float)((pk0a >> 28) & 0xFu) + zp_h;
                a_a[8] = sc_h * (_Float16)(float)((pk1a >> 0) & 0xFu) + zp_h;
                a_a[9] = sc_h * (_Float16)(float)((pk1a >> 4) & 0xFu) + zp_h;
                a_a[10] = sc_h * (_Float16)(float)((pk1a >> 8) & 0xFu) + zp_h;
                a_a[11] = sc_h * (_Float16)(float)((pk1a >> 12) & 0xFu) + zp_h;
                a_a[12] = sc_h * (_Float16)(float)((pk1a >> 16) & 0xFu) + zp_h;
                a_a[13] = sc_h * (_Float16)(float)((pk1a >> 20) & 0xFu) + zp_h;
                a_a[14] = sc_h * (_Float16)(float)((pk1a >> 24) & 0xFu) + zp_h;
                a_a[15] = sc_h * (_Float16)(float)((pk1a >> 28) & 0xFu) + zp_h;
#pragma unroll
                for (int b = 0; b < 6; b++) {
                    half16_t b_a =
                        *(const half16_t*)(x_lds + (long long)(kt * 96 + (b * 16 + ml)) * 16);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc[b]);
                }
                a_a[0] = sc_h * (_Float16)(float)((pk0b >> 0) & 0xFu) + zp_h;
                a_a[1] = sc_h * (_Float16)(float)((pk0b >> 4) & 0xFu) + zp_h;
                a_a[2] = sc_h * (_Float16)(float)((pk0b >> 8) & 0xFu) + zp_h;
                a_a[3] = sc_h * (_Float16)(float)((pk0b >> 12) & 0xFu) + zp_h;
                a_a[4] = sc_h * (_Float16)(float)((pk0b >> 16) & 0xFu) + zp_h;
                a_a[5] = sc_h * (_Float16)(float)((pk0b >> 20) & 0xFu) + zp_h;
                a_a[6] = sc_h * (_Float16)(float)((pk0b >> 24) & 0xFu) + zp_h;
                a_a[7] = sc_h * (_Float16)(float)((pk0b >> 28) & 0xFu) + zp_h;
                a_a[8] = sc_h * (_Float16)(float)((pk1b >> 0) & 0xFu) + zp_h;
                a_a[9] = sc_h * (_Float16)(float)((pk1b >> 4) & 0xFu) + zp_h;
                a_a[10] = sc_h * (_Float16)(float)((pk1b >> 8) & 0xFu) + zp_h;
                a_a[11] = sc_h * (_Float16)(float)((pk1b >> 12) & 0xFu) + zp_h;
                a_a[12] = sc_h * (_Float16)(float)((pk1b >> 16) & 0xFu) + zp_h;
                a_a[13] = sc_h * (_Float16)(float)((pk1b >> 20) & 0xFu) + zp_h;
                a_a[14] = sc_h * (_Float16)(float)((pk1b >> 24) & 0xFu) + zp_h;
                a_a[15] = sc_h * (_Float16)(float)((pk1b >> 28) & 0xFu) + zp_h;
#pragma unroll
                for (int b = 0; b < 6; b++) {
                    half16_t b_b =
                        *(const half16_t*)(x_lds + (long long)((kt + 1) * 96 + (b * 16 + ml)) * 16);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc[b]);
                }
            }
        }

        __syncthreads();
    }

    /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (lane>>4)][lane & 15]. */
    if (row_start < M) {
#pragma unroll
        for (int b = 0; b < 6; b++) {
            const int out_col = batch_start + b * 16 + ml;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int out_row = row_start + 2 * j + (lane >> 4);
                if (out_row < M) Y[(long long)out_col * M + out_row] += acc[b][j];
            }
        }
    }
}
