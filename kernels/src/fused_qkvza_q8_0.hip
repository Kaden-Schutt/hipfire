// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// 4-way fused Q8_0 GEMV for the Qwen3.5 DeltaNet linear-attention
// layer preamble (DECODE path, n=1). One kernel launch performs all four
// projections from a single shared x_normed input:
//
//   y_qkv   = W_qkv   · x      (M=qkv_m)
//   y_z     = W_z     · x      (M=z_m)
//   y_beta  = W_beta  · x      (M=beta_m)
//   y_alpha = W_alpha · x      (M=alpha_m)
//
// Grid = qkv_m + z_m + beta_m + alpha_m workgroups × 32 threads. Each
// workgroup picks its target matrix/row via blockIdx.x range check, then
// runs the same inner loop as fused_gate_up_q8_0.hip / gemv_q8_0.hip.
//
// Q8_0 block layout: 34 bytes per 32-element block
//   [fp16 scale (2 B) | int8[32] weights (32 B)]
//
// x is pre-normed F32 (the RmsnormAutomatic producer writes it).
// Bit-exact with four sequential gemv_q8_0 calls.
//
// Arch coverage: works on every RDNA generation (same design as
// fused_gate_up_q8_0.hip). No dp4a dependency.

__launch_bounds__(32, 20)
extern "C" __global__ void fused_qkvza_q8_0(
    const unsigned char* __restrict__ A_qkv,
    const unsigned char* __restrict__ A_z,
    const unsigned char* __restrict__ A_beta,
    const unsigned char* __restrict__ A_alpha,
    const float*         __restrict__ x,
    float*               __restrict__ y_qkv,
    float*               __restrict__ y_z,
    float*               __restrict__ y_beta,
    float*               __restrict__ y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K
) {
    const int gid = blockIdx.x;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    // Route this workgroup to its matrix + output row.
    const unsigned char* A;
    float* y;
    int row;
    if (gid < qkv_m) {
        A = A_qkv;   y = y_qkv;   row = gid;
    } else if (gid < qkv_m + z_m) {
        A = A_z;     y = y_z;     row = gid - qkv_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta;  y = y_beta;  row = gid - (qkv_m + z_m);
    } else {
        A = A_alpha; y = y_alpha; row = gid - (qkv_m + z_m + beta_m);
    }

    const int tid = threadIdx.x;
    const int nblocks = K / 32;
    const unsigned char* row_data = A + (size_t)row * nblocks * 34;

    float sum = 0.0f;

    // Process 8 Q8_0 blocks (256 elements) per outer iteration.
    // Matches gemv_q8_0.hip's 8-block unroll for bit-exact equivalence.
    const int outer_iters = nblocks / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;

        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }

    // Remainder blocks (when K is not a multiple of 256).
    for (int bi = outer_iters * 8; bi < nblocks; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (tid == 0) y[row] = sum;
}
