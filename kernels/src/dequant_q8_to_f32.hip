// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Nous Research
// hipfire — see LICENSE and NOTICE in the project root.
//
// Q8_0 block dequantize: Q8_0 → FP32.
// Reads Q8_0 block format (2-byte f16 scale + 32 int8 values = 34 bytes per 32 elems),
// writes FP32 values.
// Grid = ceil(n_elems / 32), one thread per element.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __launch_bounds__(256, 4)
__global__ void dequant_q8_to_f32(
    const int8_t* __restrict__ src,
    float* __restrict__ dst,
    int n_elems
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_elems) return;

    int block_idx = gid / 32;
    int lane = gid % 32;

    int src_block_base = block_idx * 34;
    __half scale_h = *(__half*)(src + src_block_base);
    float scale = (float)scale_h;
    int8_t qval = src[src_block_base + 2 + lane];

    dst[gid] = (float)qval * scale;
}
