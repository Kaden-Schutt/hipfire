// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q8hfq(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int row_stride
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const unsigned char* row_base = A + (size_t)row * row_stride;
    const int n_groups = K / 32;

    const unsigned char* scales_ptr = row_base;
    const signed char* values_ptr = (const signed char*)(row_base + n_groups * 2);

    float sum = 0.0f;

    const int outer_iters = n_groups / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            int gi = oi * 8 + sub;
            float d = (float)*((const _Float16*)(scales_ptr + gi * 2));
            signed char qval = values_ptr[gi * 32 + tid];
            sum += d * (float)qval * x[gi * 32 + tid];
        }
    }
    for (int gi = outer_iters * 8; gi < n_groups; gi++) {
        float d = (float)*((const _Float16*)(scales_ptr + gi * 2));
        signed char qval = values_ptr[gi * 32 + tid];
        sum += d * (float)qval * x[gi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (tid == 0) y[row] = sum;
}
