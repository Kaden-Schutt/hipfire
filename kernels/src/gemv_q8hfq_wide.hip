// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_q8hfq_wide(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int row_stride
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const unsigned char* row_base = A + (size_t)row * row_stride;
    const int n_groups = K / 32;

    const unsigned char* scales_ptr = row_base;
    const signed char* values_ptr = (const signed char*)(row_base + n_groups * 2);

    float sum = 0.0f;

    int gi = 0;
    for (; gi + 7 < n_groups; gi += 8) {
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            float d = (float)*((const _Float16*)(scales_ptr + (gi + u) * 2));
            signed char qval = values_ptr[(gi + u) * 32 + lane];
            sum += d * (float)qval * x[(gi + u) * 32 + lane];
        }
    }
    for (; gi < n_groups; gi++) {
        float d = (float)*((const _Float16*)(scales_ptr + gi * 2));
        signed char qval = values_ptr[gi * 32 + lane];
        sum += d * (float)qval * x[gi * 32 + lane];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0) y[row] = sum;
}
