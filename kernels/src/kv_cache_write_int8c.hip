// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void kv_cache_write_int8c(
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

    float amax = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        amax = fmaxf(amax, fabsf(head_src[i]));
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, o));

    float scale = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    // Padded to 8 + head_dim bytes (136 for head_dim=128) for cache line alignment
    int bph = 8 + head_dim;  // 8-byte header (scale + 4 pad) + data
    int bpp = n_kv_heads * bph;
    unsigned char* out = dst + (size_t)pos * bpp + h * bph;

    if (tid == 0) {
        *(float*)(out) = scale;
        *(unsigned int*)(out + 4) = 0;  // padding
    }

    for (int i = tid; i < head_dim; i += 32) {
        int q = __float2int_rn(head_src[i] * inv);
        out[8 + i] = (unsigned char)(signed char)(q > 127 ? 127 : (q < -127 ? -127 : q));
    }
}
