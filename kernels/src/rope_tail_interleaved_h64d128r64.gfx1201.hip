// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1201 DeepSeek V4 indexer-Q tail RoPE.

#include <hip/hip_runtime.h>

// DeepSeek V4 Flash 0731 indexer shape: H=64, D=128, rotary tail=64.
//
// The portable kernel assigns one 32-lane wave to the 32 rotary pairs and
// loops over all 64 heads.  On gfx1201 that leaves the rest of the device idle
// for a latency-visible bookkeeping launch.  Here each wave owns one head;
// waves in a workgroup share the 32 transcendental results through LDS.  The
// launch geometry controls the number of heads per workgroup (2, 4, or 8)
// without changing arithmetic.
extern "C" __global__ void rope_tail_interleaved_h64d128r64_gfx1201(
    float* __restrict__ q,
    const int* __restrict__ pos_buf,
    int head_dim,
    int n_rot,
    float freq_base
) {
    constexpr int kHeads = 64;
    constexpr int kHalf = 32;

    __shared__ float cos_cache[kHalf];
    __shared__ float sin_cache[kHalf];

    const int pair = threadIdx.x & (kHalf - 1);
    if (threadIdx.x < kHalf) {
        const int pos = *pos_buf;
        const float freq =
            1.0f / powf(freq_base, (float)(2 * pair) / (float)n_rot);
        const float angle = (float)pos * freq;
        cos_cache[pair] = cosf(angle);
        sin_cache[pair] = sinf(angle);
    }
    __syncthreads();

    const int heads_per_block = blockDim.x / kHalf;
    const int head = blockIdx.x * heads_per_block + threadIdx.x / kHalf;
    if (head >= kHeads) return;

    const int d_even = head_dim - n_rot + 2 * pair;
    const int d_odd = d_even + 1;
    const int base = head * head_dim;
    const float x0 = q[base + d_even];
    const float x1 = q[base + d_odd];
    const float cos_a = cos_cache[pair];
    const float sin_a = sin_cache[pair];
    q[base + d_even] = x0 * cos_a - x1 * sin_a;
    q[base + d_odd] = x0 * sin_a + x1 * cos_a;
}
