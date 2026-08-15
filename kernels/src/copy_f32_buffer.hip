// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void copy_f32_buffer(float *__restrict__ dst,
                                            const float *__restrict__ src,
                                            int n) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n)
        dst[i] = src[i];
}

extern "C" __global__ void copy_f32_strided_slot_buffer(
    float *__restrict__ dst, const float *__restrict__ src, int rows,
    int row_elems, int dst_row_stride, int dst_slot_offset) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = rows * row_elems;
    if (i >= total)
        return;
    const int row = i / row_elems;
    const int col = i - row * row_elems;
    dst[row * dst_row_stride + dst_slot_offset + col] = src[i];
}
