// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA GEMM: 2x K-tile unroll with software pipelining (HFQ6-G256 variant).
//
// 6-bit variant of gemm_hfq4g256_residual_wmma_k2. HFQ6-G256 packs 256
// weights per group in 200 bytes: [f32 scale][f32 zero][192B packed 6-bit data].
//
// 6-bit packing: 4 values per 3 bytes.
//   byte0 = q0 | (q1 << 6)
//   byte1 = (q1 >> 2) | (q2 << 4)
//   byte2 = (q2 >> 4) | (q3 << 2)
// Unpack:
//   q0 = b0 & 63
//   q1 = (b0 >> 6) | ((b1 & 0xF) << 2)
//   q2 = (b1 >> 4) | ((b2 & 3) << 4)
//   q3 = b2 >> 2
//
// Each K-tile is 16 values = 12 bytes (was 8 bytes for HFQ4).
// 16 K-tiles per 256-element group: 16 x 12 = 192 bytes data region.
//
// Dequant strategy: read 4 overlapping uint32 at 3-byte offsets from
// the data region. Each uint32 contains 4 complete 6-bit values in
// bits [0:5], [6:11], [12:17], [18:23]. Bits [24:31] are partial
// overlap from the next triplet -- masked out by & 63u.
//
// Processes 2 K-tiles (32 K-elements) per loop body:
// - Load weights+X for tile B while WMMA on tile A is executing
// - The compiler can overlap the second set of loads with the first WMMA
//
// Within each 256-element group, there are 16 K-tiles of 16.
// We process them in pairs: (0,1), (2,3), ..., (14,15) = 8 iterations.
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq6g256_residual_wmma_k2(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < batch_size) ? (batch_start + (tid & 15)) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;

        // Process 2 K-tiles per iteration (8 iterations covers 16 tiles)
        for (int kt = 0; kt < 16; kt += 2) {
            // === Tile A (kt) ===
            // Each K-tile of 16 values occupies 12 bytes in 6-bit packing.
            // Read 4 overlapping uint32 at 3-byte strides.
            const int byte_off_a = kt * 12;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + byte_off_a);
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = *(const unsigned int*)(dp_a + 3);
            unsigned int d2a = *(const unsigned int*)(dp_a + 6);
            unsigned int d3a = *(const unsigned int*)(dp_a + 9);

            // === Tile B (kt+1) -- start loading early ===
            const int byte_off_b = (kt + 1) * 12;
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + byte_off_b);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = *(const unsigned int*)(dp_b + 3);
            unsigned int d2b = *(const unsigned int*)(dp_b + 6);
            unsigned int d3b = *(const unsigned int*)(dp_b + 9);

            // Load both X tiles early (consecutive addresses -- prefetcher-friendly)
            half16_t b_a = *(const half16_t*)(xg + kt * 16);
            half16_t b_b = *(const half16_t*)(xg + (kt + 1) * 16);

            // Dequant tile A: 4 uint32 x 4 values each = 16 weights
            half16_t a_a;
            #define DQ6(reg, i, dw, sh) reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
            DQ6(a_a, 0,  d0a, 0);  DQ6(a_a, 1,  d0a, 6);  DQ6(a_a, 2,  d0a, 12); DQ6(a_a, 3,  d0a, 18);
            DQ6(a_a, 4,  d1a, 0);  DQ6(a_a, 5,  d1a, 6);  DQ6(a_a, 6,  d1a, 12); DQ6(a_a, 7,  d1a, 18);
            DQ6(a_a, 8,  d2a, 0);  DQ6(a_a, 9,  d2a, 6);  DQ6(a_a, 10, d2a, 12); DQ6(a_a, 11, d2a, 18);
            DQ6(a_a, 12, d3a, 0);  DQ6(a_a, 13, d3a, 6);  DQ6(a_a, 14, d3a, 12); DQ6(a_a, 15, d3a, 18);

            // WMMA A -- b_b load may still be in flight
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            // Dequant tile B (reuse a_a register)
            DQ6(a_a, 0,  d0b, 0);  DQ6(a_a, 1,  d0b, 6);  DQ6(a_a, 2,  d0b, 12); DQ6(a_a, 3,  d0b, 18);
            DQ6(a_a, 4,  d1b, 0);  DQ6(a_a, 5,  d1b, 6);  DQ6(a_a, 6,  d1b, 12); DQ6(a_a, 7,  d1b, 18);
            DQ6(a_a, 8,  d2b, 0);  DQ6(a_a, 9,  d2b, 6);  DQ6(a_a, 10, d2b, 12); DQ6(a_a, 11, d2b, 18);
            DQ6(a_a, 12, d3b, 0);  DQ6(a_a, 13, d3b, 6);  DQ6(a_a, 14, d3b, 12); DQ6(a_a, 15, d3b, 18);
            #undef DQ6

            // WMMA B
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    // So row = 2*j + (tid>>4), col (batch) = tid & 15.
    const int out_col = batch_start + (tid & 15);  // batch index
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
