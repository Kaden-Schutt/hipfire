// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// F16-weight × F32-input GEMV: y[m] = W_f16[m, k] @ x_f32[k].
//
// Use case: full-precision MTP weights. The WMMA F16 path converts
// F32 input to F16 (10-bit) before the matmul — that input precision
// loss makes F16 weights worse than Q8 (which keeps F32 input) in
// some downstream tasks (e.g. DeepSeek V4 MTP per-HC-row math).
//
// This kernel keeps F32 input precision: each thread reads F16
// weights, casts to F32, multiplies by F32 input, and accumulates
// in F32. Same shape contract as gemv_q8_0 / gemv_f32:
//   grid = [M, 1, 1], block = [32, 1, 1]  (1 warp per output row)
//
// Per-row work: K F16 weight loads + K F32 input loads + K FMA.
// Bandwidth: 2K bytes per row from weight, 4K from input, 4 bytes
// out → bandwidth-bound on weight + input streams.
__launch_bounds__(32)
extern "C" __global__ void gemv_f16_xf32(
    const _Float16* __restrict__ W,   // [M, K] F16
    const float*    __restrict__ x,   // [K] F32
    float*          __restrict__ y,   // [M] F32
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const _Float16* w_row = W + (size_t)row * K;

    float sum = 0.0f;
    // Each warp lane strides by 32 across the K dim.
    for (int i = tid; i < K; i += 32) {
        sum += (float)w_row[i] * x[i];
    }

    // Wave-shuffle reduction (wave32).
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
