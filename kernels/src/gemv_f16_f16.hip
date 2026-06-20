// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Generic kernel library GEMV: F16 weight × F16 vector → F16.
// y[m] = sum_k W_f16[m,k] * x_f16[k]. gfx1103 wave32, zero LDS, F32 accumulate,
// F16 round on store. grid=[M], block=[32], wave-shuffle reduction.
#include <hip/hip_runtime.h>

__launch_bounds__(32)
extern "C" __global__ void gemv_f16_f16(
    const _Float16* __restrict__ W,   // [M, K] F16
    const _Float16* __restrict__ x,   // [K] F16
    _Float16*       __restrict__ y,   // [M] F16
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const _Float16* w_row = W + (size_t)row * K;

    float sum = 0.0f;
    for (int i = tid; i < K; i += 32) {
        sum += (float)w_row[i] * (float)x[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = (_Float16)sum;
}
