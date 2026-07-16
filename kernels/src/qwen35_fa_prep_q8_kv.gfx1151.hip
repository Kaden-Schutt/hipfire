// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact gfx1151 Qwen3.6-35B full-attention preparation plus Q8_0 KV write.
//
// Fixed admitted shape: Q=16 heads, K/V=2 heads, head_dim=256, n_rot=64.
// Workgroups 0..15 prepare Q, 16..17 prepare and pack K, and 18..19 pack V.
// Each 256-thread K/V workgroup contains eight independent wave32 Q8_0 block
// quantizers, preserving the arithmetic and byte layout of
// kv_cache_write_q8_0_pair.
extern "C" __launch_bounds__(256, 1)
__global__ void qwen35_fa_prep_q8_kv_gfx1151(
    const float* __restrict__ q_interleaved,
    float* __restrict__ q,
    float* __restrict__ gate,
    const float* __restrict__ k,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const int* __restrict__ pos_buf,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const float* __restrict__ v,
    float eps,
    float freq_base)
{
    constexpr int NQ = 16;
    constexpr int NKV = 2;
    constexpr int HD = 256;
    constexpr int NROT = 64;
    constexpr int HALF = NROT / 2;
    constexpr int Q8_ELEMS = 32;
    constexpr int BLOCKS_PER_HEAD = HD / Q8_ELEMS;
    constexpr int TOTAL_BLOCKS = NKV * BLOCKS_PER_HEAD;
    constexpr int Q8_BLOCK_BYTES = 34;

    __shared__ float sdata[HD];

    const int head_slot = blockIdx.x;
    const int tid = threadIdx.x;

    // V needs no normalization or RoPE. One head-local 256-thread workgroup
    // supplies eight wave32 block quantizers.
    if (head_slot >= NQ + NKV) {
        const int head = head_slot - NQ - NKV;
        if (head >= NKV) return;
        const int block = tid >> 5;
        const int lane = tid & 31;
        const int gid = head * BLOCKS_PER_HEAD + block;
        const float val = v[head * HD + block * Q8_ELEMS + lane];
        float amax = fabsf(val);
        for (int offset = 16; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor(amax, offset, 32));

        const float scale = amax / 127.0f;
        const float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int quantized = __float2int_rn(val * inv_scale);
        quantized = max(-127, min(127, quantized));

        const int pos = pos_buf[0];
        unsigned char* packed = v_cache
            + (size_t)pos * TOTAL_BLOCKS * Q8_BLOCK_BYTES
            + gid * Q8_BLOCK_BYTES;
        if (lane == 0) *((_Float16*)packed) = (_Float16)scale;
        packed[2 + lane] = (unsigned char)(signed char)quantized;
        return;
    }

    const bool is_q = head_slot < NQ;
    const int head = is_q ? head_slot : head_slot - NQ;

    float value;
    const float* weight;
    float* q_out = nullptr;
    if (is_q) {
        const int src = head * HD * 2 + tid;
        value = q_interleaved[src];
        gate[head * HD + tid] = q_interleaved[src + HD];
        weight = q_weight;
        q_out = q + head * HD;
    } else {
        value = k[head * HD + tid];
        weight = k_weight;
    }

    // Match rmsnorm_f32's 256-thread shared-memory reduction exactly.
    sdata[tid] = value * value;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float rms = rsqrtf(sdata[0] / (float)HD + eps);
    const float normed = value * weight[tid] * rms;
    if (is_q) q_out[tid] = normed;
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
        const float y0 = x0 * cos_a - x1 * sin_a;
        const float y1 = x0 * sin_a + x1 * cos_a;
        if (is_q) {
            q_out[tid] = y0;
            q_out[tid + HALF] = y1;
        } else {
            // K is consumed only by the admitted fused Q8 writer. Keep the
            // normalized/rotated value in LDS and never materialize the
            // legacy normalized-K global intermediate.
            sdata[tid] = y0;
            sdata[tid + HALF] = y1;
        }
    }

    if (!is_q) {
        // RoPE rewrites the first 64 LDS lanes. Make every rotated/normalized
        // K element visible before the eight wave32 quantizers pack it.
        __syncthreads();
        const int block = tid >> 5;
        const int lane = tid & 31;
        const int gid = head * BLOCKS_PER_HEAD + block;
        const float val = sdata[block * Q8_ELEMS + lane];
        float amax = fabsf(val);
        for (int offset = 16; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor(amax, offset, 32));

        const float scale = amax / 127.0f;
        const float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int quantized = __float2int_rn(val * inv_scale);
        quantized = max(-127, min(127, quantized));

        const int pos = pos_buf[0];
        unsigned char* packed = k_cache
            + (size_t)pos * TOTAL_BLOCKS * Q8_BLOCK_BYTES
            + gid * Q8_BLOCK_BYTES;
        if (lane == 0) *((_Float16*)packed) = (_Float16)scale;
        packed[2 + lane] = (unsigned char)(signed char)quantized;
    }
}
