// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write_q8(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];

    const float* head_src = src + h * head_dim;

    // Find max absolute value for symmetric quantization
    extern __shared__ float smem[];
    float local_max = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(head_src[i]));
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float amax = smem[0];
    float scale = amax / 127.0f;
    float inv_scale = (scale > 0.0f) ? (127.0f / amax) : 0.0f;

    int bytes_per_head = 4 + head_dim;  // f32 scale + int8 values
    int bytes_per_pos = n_kv_heads * bytes_per_head;
    unsigned char* out = dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    if (tid == 0) *(float*)(out) = scale;
    __syncthreads();

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = head_src[i];
        int q = __float2int_rn(v * inv_scale);
        q = max(-127, min(127, q));
        out[4 + i] = (unsigned char)(signed char)q;
    }
}
