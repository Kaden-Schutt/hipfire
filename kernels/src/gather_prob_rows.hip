// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Gather one probability from each row of a dense [rows, row_stride] matrix.
// `indices` and `out` are tiny persistent device buffers; native MTP uses this
// after its target softmax so the host receives K scalars instead of K full
// vocabulary rows.
extern "C" __global__ void gather_prob_rows_f32(
    const float* __restrict__ probs,
    const int* __restrict__ indices,
    float* __restrict__ out,
    int row_stride,
    int rows
) {
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < rows) {
        int col = indices[row];
        out[row] = (col >= 0 && col < row_stride)
            ? probs[(size_t)row * row_stride + col]
            : 0.0f;
    }
}

extern "C" __global__ void max_value_f32(
    const float* __restrict__ values,
    float* __restrict__ out,
    int n
) {
    extern __shared__ float scratch[];
    int tid = (int)threadIdx.x;
    float best = 0.0f;
    for (int i = tid; i < n; i += (int)blockDim.x) {
        best = fmaxf(best, values[i]);
    }
    scratch[tid] = best;
    __syncthreads();
    for (int step = (int)blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] = fmaxf(scratch[tid], scratch[tid + step]);
        __syncthreads();
    }
    if (tid == 0) out[0] = scratch[0];
}

// Replay-visible replacement for hipMemcpyDtoD inside a retained kernel
// sequence. Redline records kernel launches, not HIP memcpy commands; using a
// plain grid-stride copy here keeps GDN innovation-tape writes in the PM4 IB.
extern "C" __global__ void copy_f32_replayable(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) dst[i] = src[i];
}
