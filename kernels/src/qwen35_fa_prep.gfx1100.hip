// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_QWEN35_FA_PREP_KERNEL
#define HIPFIRE_QWEN35_FA_PREP_KERNEL qwen35_fa_prep_gfx1100
#endif

// Exact gfx1100 Qwen3.6-35B full-attention preparation. The first 16
// workgroups reproduce deinterleave_f32 plus one 256-wide rmsnorm_f32 Q
// workgroup each; the final two reproduce the K rmsnorm workgroups. Each
// head then applies the same partial half-split RoPE locally. There are no
// cross-workgroup dependencies.
//
// Fixed admitted shape: Q=16 heads, K=2 heads, head_dim=256, n_rot=64.
extern "C" __launch_bounds__(256, 1)
__global__ void HIPFIRE_QWEN35_FA_PREP_KERNEL(
    const float* __restrict__ q_interleaved,
    float* __restrict__ q,
    float* __restrict__ gate,
    float* __restrict__ k,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const int* __restrict__ pos_buf,
    float eps,
    float freq_base)
{
    constexpr int NQ = 16;
    constexpr int HD = 256;
    constexpr int NROT = 64;
    constexpr int HALF = NROT / 2;

    __shared__ float sdata[HD];

    const int head_slot = blockIdx.x;
    const int tid = threadIdx.x;
    const bool is_q = head_slot < NQ;
    const int head = is_q ? head_slot : head_slot - NQ;

    float v;
    const float* weight;
    float* out;
    if (is_q) {
        const int src = head * HD * 2 + tid;
        v = q_interleaved[src];
        gate[head * HD + tid] = q_interleaved[src + HD];
        weight = q_weight;
        out = q + head * HD;
    } else {
        out = k + head * HD;
        v = out[tid];
        weight = k_weight;
    }

    // Match rmsnorm_f32's 256-thread shared-memory reduction exactly.
    sdata[tid] = v * v;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float rms = rsqrtf(sdata[0] / (float)HD + eps);
    const float normed = v * weight[tid] * rms;
    out[tid] = normed;
    // Lagging waves still read sdata[0] as the sum of squares; rsqrtf of the negative value written below returns NaN.
    __syncthreads();
    sdata[tid] = normed;
    __syncthreads();

    // Match rope_partial_halfsplit_f32's source expression and pair mapping.
    if (tid < HALF) {
        const int pos = *pos_buf;
        const float freq = 1.0f / powf(freq_base, (float)(2 * tid) / (float)NROT);
        const float angle = (float)pos * freq;
        const float cos_a = cosf(angle);
        const float sin_a = sinf(angle);
        const float x0 = sdata[tid];
        const float x1 = sdata[tid + HALF];
        out[tid] = x0 * cos_a - x1 * sin_a;
        out[tid + HALF] = x0 * sin_a + x1 * cos_a;
    }
}
