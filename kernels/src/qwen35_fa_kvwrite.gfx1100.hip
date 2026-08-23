// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1100 Qwen3.5 full-attention prep with a folded Q8_0 KV-cache epilogue.
// The prep body is identical to qwen35_fa_prep.gfx1100.hip (16 Q workgroups
// reproduce deinterleave + one 256-wide rmsnorm each, the next NK workgroups
// reproduce the K rmsnorm + partial half-split RoPE); two additions remove the
// separate kv_cache_write_q8_0_pair launch per full-attention layer:
//
//   * after RoPE completes, each K workgroup quantizes its finished row into
//     the K cache (one warp per 32-element Q8_0 block);
//   * NK tail workgroups quantize V straight from fa_v (no transform).
//
// The Q8_0 arithmetic is byte-for-byte kv_cache_write_q8_0_pair's: warp-wide
// |.| max via __shfl_xor butterfly, scale = amax/127 stored as _Float16,
// __float2int_rn rounding, symmetric +-127 clamp, and the same
// [pos * total_blocks * 34 + gid * 34] cache layout. The folded K row is the
// same rope output the pair kernel would read, so cache bytes are identical.
//
// Kept in its own translation unit: adding a body beside the portable writer
// or the base prep perturbs LLVM codegen for other targets (see
// kv_cache_write_q8_0_pair.gfx1100.hip).
//
// Fixed admitted shape: Q=16 heads, head_dim=256, n_rot=64; n_kv heads derived
// from the grid (gridDim.x = 16 + 2*n_kv). Single token only (pos_buf[0]).
extern "C" __launch_bounds__(256, 1)
__global__ void qwen35_fa_prep_kvwrite_gfx1100(
    const float* __restrict__ q_interleaved,
    float* __restrict__ q,
    float* __restrict__ gate,
    float* __restrict__ k,
    const float* __restrict__ v_src,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
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

    const int nk = (gridDim.x - NQ) / 2;
    const int blocks_per_head = HD / 32;
    const int total_blocks = nk * blocks_per_head;
    __shared__ float sdata[HD];

    const int head_slot = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid / 32;

    // ── V tail workgroups: quantize fa_v straight into the V cache ──
    if (head_slot >= NQ + nk) {
        const int vh = head_slot - NQ - nk;
        const int lane = tid & 31;
        const int gid = vh * blocks_per_head + warp;
        const float val = v_src[vh * HD + warp * 32 + lane];
        float amax = fabsf(val);
        for (int offset = 16; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor(amax, offset));
        const float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int qv = __float2int_rn(val * inv_scale);
        qv = max(-127, min(127, qv));
        unsigned char* out = v_cache + (size_t)pos_buf[0] * total_blocks * 34 + gid * 34;
        if (lane == 0) {
            *((_Float16*)(out)) = (_Float16)(amax / 127.0f);
        }
        out[2 + lane] = (unsigned char)(signed char)qv;
        return;
    }

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

    // Match rope_partial_halfsplit_f32's source expression, pair mapping, AND
    // FMA contraction (see qwen35_fa_prep.gfx1100.hip for the ulp-level trap).
    if (tid < HALF) {
        const int pos = *pos_buf;
        const float freq = 1.0f / powf(freq_base, (float)(2 * tid) / (float)NROT);
        const float angle = (float)pos * freq;
        const float cos_a = cosf(angle);
        const float sin_a = sinf(angle);
        const float x0 = sdata[tid];
        const float x1 = sdata[tid + HALF];
        out[tid] = x0 * cos_a - x1 * sin_a;
        out[tid + HALF] = __builtin_fmaf(x0, sin_a, x1 * cos_a);
    }

    // ── K epilogue: quantize the finished rope'd row into the K cache ──
    if (!is_q) {
        __syncthreads();
        const int lane = tid & 31;
        const int gid = head * blocks_per_head + warp;
        const float val = k[head * HD + warp * 32 + lane];
        float amax = fabsf(val);
        for (int offset = 16; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor(amax, offset));
        const float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int qv = __float2int_rn(val * inv_scale);
        qv = max(-127, min(127, qv));
        unsigned char* outb = k_cache + (size_t)pos_buf[0] * total_blocks * 34 + gid * 34;
        if (lane == 0) {
            *((_Float16*)(outb)) = (_Float16)(amax / 127.0f);
        }
        outb[2 + lane] = (unsigned char)(signed char)qv;
    }
}
