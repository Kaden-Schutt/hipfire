// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention tile for asym2: K at givens2 (rotated 2-bit), V at Q8_0.
// Output in normal space → reuses Q8_0 flash reduce.
//
// MAX_HALVES is a compile-time bound on head_dim/128 so the per-thread
// mq/out_vec arrays stay in registers (zero scratch spill at the default
// width). Two instantiations are exported:
//   attention_flash_asym2_tile        — MAX_HALVES=2, head_dim ∈ {128, 256}
//   attention_flash_asym2_tile_hd512  — MAX_HALVES=4, head_dim ∈ {384, 512}
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

template <int MAX_HALVES>
__device__ __forceinline__ void asym2_tile_body(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ pos_buf,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    // Sliding-window lookback. 0 = full causal.
    int window_size,
    // Ring-buffer cache capacity. 0 = no wrap.
    int cache_capacity
) {
    const int seq_len = pos_buf[0] + 1;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tile_id = blockIdx.y;
    const int tile_start = tile_id * tile_size;
    const int tile_end = min(tile_start + tile_size, seq_len);
    if (tile_start >= seq_len) return;
    const int tile_len = tile_end - tile_start;

    const int t_lo = (window_size > 0) ? max(0, seq_len - window_size) : 0;
    if (window_size > 0 && tile_end <= t_lo) {
        if (threadIdx.x == 0) {
            float* p0 = partials + (h * max_tiles + tile_id) * (2 + head_dim);
            p0[0] = -1e30f;
            p0[1] = 0.0f;
        }
        return;
    }

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    // Compile-time clamp keeps mq/out_vec indices provably in range so they
    // stay in VGPRs. Callers guarantee head_dim/128 <= MAX_HALVES.
    const int n_halves = min(head_dim / 128, MAX_HALVES);

    // K geometry (givens2: 1 byte per 4 values)
    const int k_bytes_per_head = 4 + head_dim / 4;        // 36 (hd128) / 68 (hd256)
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    // V geometry (Q8_0)
    const int v_blocks_per_head = head_dim / 32;
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q + rotate (per half)
    const float* q_head = q + h * head_dim;
    float mq[MAX_HALVES * 4];
    for (int half = 0; half < n_halves; half++) {
        const int d0 = half * 128 + tid * 4;
        float a = q_head[d0 + 0];
        float b = q_head[d0 + 1];
        float c = q_head[d0 + 2];
        float d = q_head[d0 + 3];
        const int block_tid = half * 32 + tid;
        givens_forward(a, b, c, d, cos_theta, sin_theta, block_tid);
        mq[half * 4 + 0] = a;
        mq[half * 4 + 1] = b;
        mq[half * 4 + 2] = c;
        mq[half * 4 + 3] = d;
    }

    // Phase A: Q·K (2-bit K)
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const bool out_of_window = (window_size > 0) && (t < t_lo);
        float partial = 0.0f;
        if (!out_of_window) {
            const int k_slot = (cache_capacity > 0) ? (t % cache_capacity) : t;
            const unsigned char* kb = k_cache + (size_t)k_slot * k_bytes_per_pos + k_head_off;
            float cnorm = *(const float*)kb;
            for (int half = 0; half < n_halves; half++) {
                // Byte offset: 4 (cnorm) + half * 32 + tid.
                unsigned char packed = kb[4 + half * 32 + tid];
                partial += mq[half * 4 + 0] * (cnorm * TURBO_C2[(packed)      & 0x3])
                         + mq[half * 4 + 1] * (cnorm * TURBO_C2[(packed >> 2) & 0x3])
                         + mq[half * 4 + 2] * (cnorm * TURBO_C2[(packed >> 4) & 0x3])
                         + mq[half * 4 + 3] * (cnorm * TURBO_C2[(packed >> 6) & 0x3]);
            }
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0)
            scores[t_local] = out_of_window ? -1e30f : (partial * scale_attn);
    }
    __syncthreads();

    // Phase B: tile max
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    // Phase C: exp + sum
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    float tile_sum = local_sum;
    __syncthreads();

    // Phase D: V (Q8_0 normal space)
    float out_vec[MAX_HALVES * 4] = {};
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const int v_slot = (cache_capacity > 0) ? (t % cache_capacity) : t;
        const unsigned char* vb = v_cache + (size_t)v_slot * v_row_stride;
        for (int half = 0; half < n_halves; half++) {
            const int d_base = half * 128 + tid * 4;
            const int v_bi       = d_base / 32;
            const int v_bj_base  = d_base % 32;
            const int v_blk_off  = (v_kv_head_blk + v_bi) * 34;
            const unsigned char* vbh = vb + v_blk_off;
            float vs = (float)*((const _Float16*)vbh);
            out_vec[half * 4 + 0] += w * (vs * (float)((signed char)vbh[2 + v_bj_base + 0]));
            out_vec[half * 4 + 1] += w * (vs * (float)((signed char)vbh[2 + v_bj_base + 1]));
            out_vec[half * 4 + 2] += w * (vs * (float)((signed char)vbh[2 + v_bj_base + 2]));
            out_vec[half * 4 + 3] += w * (vs * (float)((signed char)vbh[2 + v_bj_base + 3]));
        }
    }

    // Write partials — stride by max_tiles to match Q8_0 reduce layout.
    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        p[2 + d_base + 0] = out_vec[half * 4 + 0];
        p[2 + d_base + 1] = out_vec[half * 4 + 1];
        p[2 + d_base + 2] = out_vec[half * 4 + 2];
        p[2 + d_base + 3] = out_vec[half * 4 + 3];
    }
}

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_asym2_tile(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ pos_buf,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_heads, int n_kv_heads, int head_dim, int max_seq,
    float scale_attn, int tile_size, int max_tiles,
    int window_size, int cache_capacity
) {
    asym2_tile_body<2>(q, k_cache, v_cache, partials, pos_buf, cos_theta, sin_theta,
        n_heads, n_kv_heads, head_dim, max_seq, scale_attn, tile_size, max_tiles,
        window_size, cache_capacity);
}

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_asym2_tile_hd512(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ pos_buf,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_heads, int n_kv_heads, int head_dim, int max_seq,
    float scale_attn, int tile_size, int max_tiles,
    int window_size, int cache_capacity
) {
    asym2_tile_body<4>(q, k_cache, v_cache, partials, pos_buf, cos_theta, sin_theta,
        n_heads, n_kv_heads, head_dim, max_seq, scale_attn, tile_size, max_tiles,
        window_size, cache_capacity);
}
