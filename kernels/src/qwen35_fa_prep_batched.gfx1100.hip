// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched gfx1100 Qwen3.5 full-attention preparation. Generalizes the
// single-token qwen35_fa_prep_gfx1100 arithmetic to a [18, N] grid: the first
// 16 workgroup-rows reproduce deinterleave_f32_batched plus one 256-wide
// rmsnorm_f32 Q row each, the final two reproduce the K rmsnorm rows, and
// each head then applies the same partial half-split RoPE locally from the
// per-row positions buffer. There are no cross-workgroup dependencies.
//
// Fixed admitted shape: Q=16 heads, K=2 heads, head_dim=256, n_rot=64.
// Grid: [18, batch_size, 1]. Block: [256, 1, 1]. Static LDS: 1 KiB.
//
// Bit-exactness vs the four-launch sequence it replaces:
// - gate/Q values come from the same interleaved words deinterleave reads;
// - the 256-thread shared-memory reduction tree is statement-identical to
//   rmsnorm_f32 at n == blockDim (one element per thread), so the sum lands
//   in sdata[0] with the same association order, and out = x*w*rms matches;
// - RoPE reuses the halfsplit source expression, pair mapping (i, i+32),
//   and positions[b] + pos_offset phase, per (row, head, i).
extern "C" __launch_bounds__(256, 1)
__global__ void qwen35_fa_prep_batched_gfx1100(
    const float* __restrict__ q_interleaved, // [N x 16 x 256 x 2]
    float* __restrict__ q,                   // [N x 16 x 256]
    float* __restrict__ gate,                // [N x 16 x 256]
    float* __restrict__ k,                   // [N x 2 x 256], in-place in/out
    const float* __restrict__ q_weight,      // [256]
    const float* __restrict__ k_weight,      // [256]
    const int* __restrict__ positions,       // [N] physical KV slots
    float eps,
    float freq_base,
    int pos_offset, // added to positions[b] for the RoPE angle only
    int batch_size)
{
    constexpr int NQ = 16;
    constexpr int NK = 2;
    constexpr int HD = 256;
    constexpr int NROT = 64;
    constexpr int HALF = NROT / 2;

    __shared__ float sdata[HD];

    const int head_slot = blockIdx.x;
    const int b = blockIdx.y;
    if (head_slot >= NQ + NK || b >= batch_size) return;
    const int tid = threadIdx.x;
    const bool is_q = head_slot < NQ;
    const int head = is_q ? head_slot : head_slot - NQ;

    float v;
    const float* weight;
    float* out;
    if (is_q) {
        const long long row = (long long)b * NQ * HD + (long long)head * HD;
        const float* src_row =
            q_interleaved + (long long)b * NQ * HD * 2 + (long long)head * HD * 2;
        v = src_row[tid];
        gate[row + tid] = src_row[tid + HD];
        weight = q_weight;
        out = q + row;
    } else {
        const long long row = (long long)b * NK * HD + (long long)head * HD;
        out = k + row;
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
    // Match rope_partial_halfsplit_batched_f32's source expression and pair mapping.
    // The two outputs MUST use explicit fmaf in the exact formation the old
    // kernel's TU contracts to (probed on gfx1100: o[i] fuses the v0 term as
    // fma(v0, trig, +/-(v1*trig')) on both sides). Default fp-contract in this
    // TU fuses the opposite way and drifts ~2% of words by 1 ULP.
    if (tid < HALF) {
        const int pos = positions[b] + pos_offset;
        const float freq = 1.0f / powf(freq_base, (float)(2 * tid) / (float)NROT);
        const float angle = (float)pos * freq;
        const float cos_a = cosf(angle);
        const float sin_a = sinf(angle);
        const float x0 = sdata[tid];
        const float x1 = sdata[tid + HALF];
        out[tid] = fmaf(x0, cos_a, -(x1 * sin_a));
        out[tid + HALF] = fmaf(x0, sin_a, x1 * cos_a);
    }
}
