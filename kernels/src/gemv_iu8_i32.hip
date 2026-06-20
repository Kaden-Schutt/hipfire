// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Generic kernel library GEMV: signed INT8 weight × INT8 vector → INT32.
// y[m] = sum_k W_i8[m,k] * x_i8[k]. gfx1103 wave32, zero LDS, i32 accumulate.
// grid=[M], block=[32], wave-shuffle reduction.
#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void gemv_iu8_i32(
    const int8_t* __restrict__ W,   // [M, K] int8
    const int8_t* __restrict__ x,   // [K] int8
    int32_t*      __restrict__ y,   // [M] int32
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int8_t* w_row = W + (size_t)row * K;

    int sum = 0;
    for (int i = tid; i < K; i += 32) {
        sum += (int)w_row[i] * (int)x[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
