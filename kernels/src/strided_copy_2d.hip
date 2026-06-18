// SPDX-License-Identifier: Apache-2.0
// hipfire-train — strided 2D copy (fp32), the gather/scatter primitive for GQA.
//
// Copies a [rows, cols] block between buffers with independent row strides and
// base offsets. With accumulate!=0 it adds into the destination (scatter-add),
// needed where several query heads share one key/value head and their grads
// must sum back into that kv head.
//
//   dst[dst_off + r*dst_stride + c] (+=|=) src[src_off + r*src_stride + c]

#include <hip/hip_runtime.h>

extern "C" __global__ void strided_copy_2d(
    const float* __restrict__ src, int src_off, int src_stride,
    float* __restrict__ dst, int dst_off, int dst_stride,
    int rows, int cols, int accumulate
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = src[(size_t)src_off + (size_t)r * src_stride + c];
        size_t di = (size_t)dst_off + (size_t)r * dst_stride + c;
        if (accumulate) dst[di] += v;
        else dst[di] = v;
    }
}
