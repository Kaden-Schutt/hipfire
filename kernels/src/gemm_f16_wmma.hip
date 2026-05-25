// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 nickfinease
// Copyright (c) 2026 Kevin Read (correctness fix for the lane-cooperative WMMA layout)
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated F16×F32 batched GEMM for the vision encoder.
// gfx1100+ only (RDNA3 wave32 WMMA).
//
// Computes: Y[M, N] = W_f16[M, K] @ X_f32[N, K]^T
//
// Where:
//   W is row-major F16 [M, K]  (weight matrix)
//   X is row-major F32 [N, K]  (batch of input vectors, F32 cast to F16 lane-side)
//   Y is row-major F32 [M, N]  (output)
//
// WMMA intrinsic: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
// computes (per AMD RDNA3 ISA):  D = A @ B^T + C
// where A is 16×16, B is 16×16 (both F16), and C/D are 16×16 F32.
//
// RDNA3 WMMA register layout (wave32, F16 input):
//   For A operand `half16_t a` (16 F16 elements per lane):
//     Lane t in 0..15 holds row t of A (columns 0..15).
//     Lanes 16..31 mirror lanes 0..15 (the same row data, replicated).
//   Same for B: lane t in 0..15 holds row t of B (columns 0..15).
//   For C/D accumulator `float8_t c` (8 F32 elements per lane):
//     c[j] at lane t maps to D[2*j + (t>>4), t & 15].
//     (Lanes 0..15 cover even output rows; lanes 16..31 cover odd rows.)
//
// The OLD version of this kernel had every lane redundantly fill
// `a_reg[0..255]` and `b_reg[0..255]` — writing past the 16-element
// half16_t vector. That's UB; on RDNA3 it manifests as NaN output
// (the input register file gets garbage past the first 16 elements
// and the WMMA intrinsic propagates it). The fix below loads only
// the lane's owned row, matching the working pattern used by
// `kernels/src/gemm_qkv_hfq4g256_wmma.hip` and friends.
//
// Grid: [ceil(M/16), ceil(N/16)]
// Block: [32]
// LDS: 0

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_f16_wmma(
    const _Float16* __restrict__ W,   // [M, K] row-major, F16
    const float*    __restrict__ X,   // [N, K] row-major, F32
    float*          __restrict__ Y,   // [M, N] row-major, F32
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= M || col_start >= N) return;

    // Each lane owns one row in the 16-row tile (tid & 15). Lanes 16..31
    // hold the same data as lanes 0..15 per the wave32 WMMA replication
    // convention.
    const int my_a_row = row_start + (tid & 15);
    const int my_b_row = col_start + (tid & 15);   // "row of B" = column of X^T = position in N axis

    // Out-of-bounds lanes load zeros so they don't read garbage and so
    // their contribution to the WMMA accumulator is well-defined zero.
    const bool a_in_bounds = (my_a_row < M);
    const bool b_in_bounds = (my_b_row < N);

    // WMMA output accumulator: 8 F32 per lane.
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Process K in tiles of 16.
    for (int k0 = 0; k0 < K; k0 += 16) {
        // Load A: lane (tid & 15) holds row `my_a_row` of W, columns k0..k0+15.
        half16_t a_reg;
        if (a_in_bounds) {
            const _Float16* src = W + (long long)my_a_row * K + k0;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                a_reg[j] = (k0 + j < K) ? src[j] : (_Float16)0.0f;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 16; j++) a_reg[j] = (_Float16)0.0f;
        }

        // Load B: lane (tid & 15) holds row `my_b_row` of X, columns k0..k0+15.
        // Down-cast F32 → F16 lane-side (the WMMA intrinsic only consumes F16
        // inputs; the F32 accumulator preserves precision across K-tiles).
        half16_t b_reg;
        if (b_in_bounds) {
            const float* src = X + (long long)my_b_row * K + k0;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                b_reg[j] = (k0 + j < K) ? (_Float16)src[j] : (_Float16)0.0f;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 16; j++) b_reg[j] = (_Float16)0.0f;
        }

        // WMMA: acc += A @ B^T  (16×16×16 F16-input F32-acc matrix multiply).
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
    }

    // Output: acc[j] at lane t maps to (row = 2*j + (t>>4), col = t & 15).
    const int out_col = col_start + (tid & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_row * N + out_col] = acc[j];
            }
        }
    }
}
