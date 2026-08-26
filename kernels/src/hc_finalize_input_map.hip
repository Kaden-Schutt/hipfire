// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// DeepSeek4 mHC finalization plus its immediate input-map consumer. One
// workgroup keeps the four pre weights resident after sigmoid/Sinkhorn setup,
// then stripes the 4096 hidden dimensions over its 256 lanes.
extern "C" __global__ __launch_bounds__(256, 1)
void hc_finalize_input_map(
    float*        __restrict__ c,
    const __half* __restrict__ alpha,
    const __half* __restrict__ base,
    const float*  __restrict__ streams,
    float*        __restrict__ x_out,
    int hidden,
    float hc_eps,
    float post_scale,
    int sinkhorn_iters
) {
    __shared__ float pre[4];
    const int tid = threadIdx.x;

    if (tid < 24) {
        int segment;
        if      (tid < 4) segment = 0;
        else if (tid < 8) segment = 1;
        else              segment = 2;
        const float a = __half2float(alpha[segment]);
        const float b = __half2float(base[tid]);
        c[tid] = a * c[tid] + (1.0f - a) * b;
    }
    __syncthreads();

    if (tid < 4) {
        const float value = 1.0f / (1.0f + __expf(-c[tid])) + hc_eps;
        c[tid] = value;
        pre[tid] = value;
    } else if (tid < 8) {
        c[tid] = post_scale * (1.0f / (1.0f + __expf(-c[tid])));
    }

    if (tid < 16) {
        float value = c[8 + tid];
        float row_max = value;
        row_max = fmaxf(row_max, __shfl_xor(row_max, 1));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 2));
        value = __expf(value - row_max);
        float row_sum = value;
        row_sum += __shfl_xor(row_sum, 1);
        row_sum += __shfl_xor(row_sum, 2);
        value = value / row_sum + hc_eps;

        float column_sum = value;
        column_sum += __shfl_xor(column_sum, 4);
        column_sum += __shfl_xor(column_sum, 8);
        column_sum += hc_eps;
        value /= column_sum;

        for (int iteration = 0; iteration < sinkhorn_iters - 1; ++iteration) {
            row_sum = value;
            row_sum += __shfl_xor(row_sum, 1);
            row_sum += __shfl_xor(row_sum, 2);
            row_sum += hc_eps;
            value /= row_sum;

            column_sum = value;
            column_sum += __shfl_xor(column_sum, 4);
            column_sum += __shfl_xor(column_sum, 8);
            column_sum += hc_eps;
            value /= column_sum;
        }
        c[8 + tid] = value;
    }
    __syncthreads();

    const float a0 = pre[0];
    const float a1 = pre[1];
    const float a2 = pre[2];
    const float a3 = pre[3];
    for (int d = tid; d < hidden; d += blockDim.x) {
        x_out[d] = a0 * streams[d]
                 + a1 * streams[hidden + d]
                 + a2 * streams[2 * hidden + d]
                 + a3 * streams[3 * hidden + d];
    }
}
