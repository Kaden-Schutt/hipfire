// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write_hfq8(
    unsigned char* __restrict__ dst_data,   // [max_seq × kv_dim] uint8
    float* __restrict__ dst_scales,         // [max_seq × n_kv_heads × 2] f32
    const float* __restrict__ src,          // [kv_dim] FP32
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

    // Warp min/max reduction (head_dim=128, 32 threads, 4 per thread)
    float local_min = 1e30f, local_max = -1e30f;
    for (int i = tid; i < head_dim; i += 32) {
        float v = head_src[i];
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_min = fminf(local_min, __shfl_xor(local_min, offset));
        local_max = fmaxf(local_max, __shfl_xor(local_max, offset));
    }

    float scale = (local_max - local_min) / 255.0f;
    float zero = local_min;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // Write scale+zero (lane 0)
    if (tid == 0) {
        dst_scales[(pos * n_kv_heads + h) * 2] = scale;
        dst_scales[(pos * n_kv_heads + h) * 2 + 1] = zero;
    }

    // Quantize and write contiguous uint8
    unsigned char* out = dst_data + pos * kv_dim + h * head_dim;
    for (int i = tid; i < head_dim; i += 32) {
        int q = __float2int_rn((head_src[i] - zero) * inv_scale);
        q = max(0, min(255, q));
        out[i] = (unsigned char)q;
    }
}
