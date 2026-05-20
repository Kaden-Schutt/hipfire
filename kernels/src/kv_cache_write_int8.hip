// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write_int8(
    signed char* __restrict__ dst_vals,   // [max_seq × kv_dim] int8
    float* __restrict__ dst_scales,       // [max_seq × n_kv_heads] f32
    const float* __restrict__ src,        // [kv_dim] FP32
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];
    const int kv_dim = n_kv_heads * head_dim;

    const float* head_src = src + h * head_dim;

    // Find max absolute value via warp shuffle (head_dim=128, 32 threads, 4 per thread)
    float amax = 0.0f;
    for (int i = tid; i < head_dim; i += 32) {
        amax = fmaxf(amax, fabsf(head_src[i]));
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    // Write scale (lane 0)
    if (tid == 0) {
        dst_scales[pos * n_kv_heads + h] = scale;
    }

    // Quantize and write contiguous int8
    signed char* out = dst_vals + pos * kv_dim + h * head_dim;
    for (int i = tid; i < head_dim; i += 32) {
        int q = __float2int_rn(head_src[i] * inv_scale);
        q = max(-127, min(127, q));
        out[i] = (signed char)q;
    }
}
