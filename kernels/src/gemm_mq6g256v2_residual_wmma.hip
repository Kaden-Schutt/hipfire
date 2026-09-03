// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched MQ6G256V2 GEMM with fused residual add.
//
// gfx11 (RDNA3 / RDNA3.5) variant — sister of gemm_hfq6g256_residual_wmma_k2.hip
// (HFQ6-G256) and gemm_mq6g256v2_residual_wmma.gfx12.hip (RDNA4).
//
// Same geometry as the HFQ6 gfx11 kernel:
//   Each wave32 block computes a 16×16 output tile:
//     C[16 rows, 16 batch] += W[16 rows, 16 K] * X[16 K, 16 batch]
//   iterated over K in chunks of 16, grouped by 256-element weight groups.
//   Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. LDS: 0.
//   K2 unroll (2 K-tiles per iteration) with software pipelining.
//
// Differences from the HFQ6 gfx11 kernel — MQ6V2 header contract only:
//   200 B row groups, dual fp16 headers [s0,z0,s1,z1], payload at +8.
//   Header bytes:
//     [0..4)  : s0 (fp16 lo) | z0 (fp16 hi)  → governs weights 0..127 of the group
//     [4..8)  : s1 (fp16 lo) | z1 (fp16 hi)  → governs weights 128..255 of the group
//     [8..200): 192 bytes of packed 6-bit codes (256 weights).
//   The 16 K-tiles inside a group split at kt=8: tiles 0..7 use (s0,z0), 8..15 use (s1,z1).
//   Implemented as `kt < 8 ? sc0 : sc1` / `kt < 8 ? zp0 : zp1`.
//
// 6-bit packing (identical to HFQ6 / MQ6V2 gfx12): 4 values per 3 bytes.
//   Per K-tile (16 values): 12 bytes. Overlapping uint32 reads at +0,+3,+6,+9;
//   extract with ((dw >> sh) & 63u) at shifts 0,6,12,18.
//
// WMMA contracts:
//   half16_t A/B operands (16 fp16 per lane, full 16-K per lane),
//   __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 (no gfx12 suffix),
//   float8_t accumulator, interleaved C mapping acc[j]=C[2*j+(tid>>4)][tid&15].
//
// Reference: gemm_hfq6g256_residual_wmma_k2.hip (gfx11) and
// gemm_mq6g256v2_residual_wmma.gfx12.hip (gfx12 sibling).
// Compile with `hipcc --offload-arch=gfx1100` (or 1151).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
__launch_bounds__(32, 8)
#else
__launch_bounds__(32, 2)
#endif
extern "C" __global__ void gemm_mq6g256v2_residual_wmma(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= M) return;
    if (batch_start >= N) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < N) ? (batch_start + (tid & 15)) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        const _Float16* xg = x_base + g * 256;

        // Process 2 K-tiles per iteration (8 iterations covers 16 tiles)
        for (int kt = 0; kt < 16; kt += 2) {
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1;

            // === Tile A (kt) ===
            const int byte_off_a = kt * 12;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + byte_off_a);
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = *(const unsigned int*)(dp_a + 3);
            unsigned int d2a = *(const unsigned int*)(dp_a + 6);
                        // Bytes 9,10,11 of the tile, NOT a dword at +9 — see the same fix
            // in gemm_qkv_mq6g256v2_wmma.hip. The four 6-bit values this feeds
            // sit at shifts 0/6/12/18, so only bits 0..23 are ever used and
            // byte 12 is read then discarded. At kt=15 a +9 dword reads one
            // byte past the 200-byte group, and on the last group of the last
            // row that is one byte past the weight tensor. Load at +8 and
            // shift down: identical bits 0..23, never leaves the tile.
            unsigned int d3a = (*(const unsigned int*)(dp_a + 8)) >> 8;

            // === Tile B (kt+1) — start loading early ===
            const int byte_off_b = (kt + 1) * 12;
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + byte_off_b);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = *(const unsigned int*)(dp_b + 3);
            unsigned int d2b = *(const unsigned int*)(dp_b + 6);
            unsigned int d3b = (*(const unsigned int*)(dp_b + 8)) >> 8;  // see d3a

            half16_t b_a = *(const half16_t*)(xg + kt * 16);
            half16_t b_b = *(const half16_t*)(xg + (kt + 1) * 16);

            half16_t a_reg;
            #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
            DQ6(0, d0a, 0); DQ6(1, d0a, 6); DQ6(2, d0a, 12); DQ6(3, d0a, 18);
            DQ6(4, d1a, 0); DQ6(5, d1a, 6); DQ6(6, d1a, 12); DQ6(7, d1a, 18);
            DQ6(8, d2a, 0); DQ6(9, d2a, 6); DQ6(10, d2a, 12); DQ6(11, d2a, 18);
            DQ6(12, d3a, 0); DQ6(13, d3a, 6); DQ6(14, d3a, 12); DQ6(15, d3a, 18);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            DQ6(0, d0b, 0); DQ6(1, d0b, 6); DQ6(2, d0b, 12); DQ6(3, d0b, 18);
            DQ6(4, d1b, 0); DQ6(5, d1b, 6); DQ6(6, d1b, 12); DQ6(7, d1b, 18);
            DQ6(8, d2b, 0); DQ6(9, d2b, 6); DQ6(10, d2b, 12); DQ6(11, d2b, 18);
            DQ6(12, d3b, 0); DQ6(13, d3b, 6); DQ6(14, d3b, 12); DQ6(15, d3b, 18);
            #undef DQ6

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    const int out_col = batch_start + (tid & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
