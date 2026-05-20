// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write(
    float* __restrict__ dst,          // cache: [max_seq * kv_dim]
    const float* __restrict__ src,    // current KV: [kv_dim]
    const int* __restrict__ pos_buf,  // position buffer
    int kv_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    int pos = pos_buf[0];
    dst[pos * kv_dim + i] = src[i];
}
