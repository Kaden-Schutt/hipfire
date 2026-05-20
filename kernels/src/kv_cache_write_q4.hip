// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write_q4(
    unsigned char* __restrict__ dst,     // quantized cache
    const float* __restrict__ src,       // [kv_dim] FP32 KV vector
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int h = blockIdx.x;  // one block per KV head
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];

    const float* head_src = src + h * head_dim;

    // Shared memory for min/max reduction
    extern __shared__ float smem[];
    float* s_min = smem;
    float* s_max = smem + blockDim.x;

    // Find min/max across head_dim elements
    float local_min = 1e30f, local_max = -1e30f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = head_src[i];
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    float vmin = s_min[0];
    float vmax = s_max[0];

    float scale = (vmax - vmin) / 15.0f;
    float zero = vmin;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // Output layout per head: [4B scale][4B zero][head_dim/2 nibbles]
    int bytes_per_head = 8 + head_dim / 2;
    int bytes_per_pos = n_kv_heads * bytes_per_head;
    unsigned char* out = dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    // Write scale and zero (thread 0 only)
    if (tid == 0) {
        *(float*)(out) = scale;
        *(float*)(out + 4) = zero;
    }
    __syncthreads();

    // Quantize and pack nibbles (2 elements per byte)
    for (int i = tid; i < head_dim / 2; i += blockDim.x) {
        int e0 = i * 2;
        int e1 = i * 2 + 1;
        float v0 = head_src[e0];
        float v1 = head_src[e1];
        int q0 = __float2int_rn((v0 - zero) * inv_scale);
        int q1 = __float2int_rn((v1 - zero) * inv_scale);
        q0 = max(0, min(15, q0));
        q1 = max(0, min(15, q1));
        out[8 + i] = (unsigned char)((q1 << 4) | q0);
    }
}
