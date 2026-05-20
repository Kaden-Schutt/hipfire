// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write_hfq4(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;  // 0..31
    const int pos = pos_buf[0];

    const float* head_src = src + h * head_dim;

    // Each thread handles 4 elements (32 threads × 4 = 128)
    float v0 = (tid * 4 < head_dim) ? head_src[tid * 4] : 0.0f;
    float v1 = (tid * 4 + 1 < head_dim) ? head_src[tid * 4 + 1] : 0.0f;
    float v2 = (tid * 4 + 2 < head_dim) ? head_src[tid * 4 + 2] : 0.0f;
    float v3 = (tid * 4 + 3 < head_dim) ? head_src[tid * 4 + 3] : 0.0f;

    // Warp min/max
    float local_min = fminf(fminf(v0, v1), fminf(v2, v3));
    float local_max = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_min = fminf(local_min, __shfl_xor(local_min, offset));
        local_max = fmaxf(local_max, __shfl_xor(local_max, offset));
    }

    float scale = (local_max - local_min) / 15.0f;
    float zero = local_min;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // Block: [4B scale][4B zero][head_dim/2 nibbles]
    int bytes_per_block = 8 + head_dim / 2;
    int bytes_per_pos = n_kv_heads * bytes_per_block;
    unsigned char* out = dst + (size_t)pos * bytes_per_pos + h * bytes_per_block;

    if (tid == 0) {
        *(float*)(out) = scale;
        *(float*)(out + 4) = zero;
    }

    // Quantize 4 values → 2 bytes (4 nibbles)
    int q0 = __float2int_rn((v0 - zero) * inv_scale); q0 = max(0, min(15, q0));
    int q1 = __float2int_rn((v1 - zero) * inv_scale); q1 = max(0, min(15, q1));
    int q2 = __float2int_rn((v2 - zero) * inv_scale); q2 = max(0, min(15, q2));
    int q3 = __float2int_rn((v3 - zero) * inv_scale); q3 = max(0, min(15, q3));

    out[8 + tid * 2]     = (unsigned char)((q1 << 4) | q0);
    out[8 + tid * 2 + 1] = (unsigned char)((q3 << 4) | q2);
}
