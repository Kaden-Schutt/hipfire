// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention tile for fwht2: K at FWHT-rotated 2-bit, V at Q8_0.
// 2-pass over 128-element halves, mirroring fwht4_tile structure.
//
// K cache byte-identical to asym2 (1 byte per 4 packed 2-bit values).
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_fwht2_tile(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int v_mode          // 8 = legacy Q8_0 V; 2/3/4 = 256-wide lloyd FWHT-rotated centroid V
) {
    const int seq_len = pos_buf[0] + 1;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tile_id = blockIdx.y;
    const int tile_start = tile_id * tile_size;
    const int tile_end = min(tile_start + tile_size, seq_len);
    if (tile_start >= seq_len) return;
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int n_halves = head_dim / 128;

    const int k_bytes_per_head = 4 + head_dim / 4;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    const int v_blocks_per_head = head_dim / 32;
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    const float* q_head = q + h * head_dim;
    float mq[8];
    for (int half = 0; half < n_halves; half++) {
        const int d0 = half * 128 + tid * 4;
        float a = q_head[d0 + 0];
        float b = q_head[d0 + 1];
        float c = q_head[d0 + 2];
        float d = q_head[d0 + 3];
        fwht_shfl_forward(a, b, c, d, signs1, signs2, tid);
        mq[half * 4 + 0] = a;
        mq[half * 4 + 1] = b;
        mq[half * 4 + 2] = c;
        mq[half * 4 + 3] = d;
    }

    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * k_bytes_per_pos + k_head_off;
        float cnorm = *(const float*)kb;
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            unsigned char packed = kb[4 + half * 32 + tid];
            partial += mq[half * 4 + 0] * (cnorm * TURBO_C2[(packed)      & 0x3])
                     + mq[half * 4 + 1] * (cnorm * TURBO_C2[(packed >> 2) & 0x3])
                     + mq[half * 4 + 2] * (cnorm * TURBO_C2[(packed >> 4) & 0x3])
                     + mq[half * 4 + 3] * (cnorm * TURBO_C2[(packed >> 6) & 0x3]);
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0)
            scores[t_local] = partial * scale_attn;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

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

    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }

    if (v_mode == 8) {
        // ── Legacy Q8_0 V (normal space, per-half layout) — unchanged ──
        float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * v_row_stride;
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
        for (int half = 0; half < n_halves; half++) {
            const int d_base = half * 128 + tid * 4;
            p[2 + d_base + 0] = out_vec[half * 4 + 0];
            p[2 + d_base + 1] = out_vec[half * 4 + 1];
            p[2 + d_base + 2] = out_vec[half * 4 + 2];
            p[2 + d_base + 3] = out_vec[half * 4 + 3];
        }
    } else {
        // ── 256-wide lloyd2/3/4 V: FWHT-rotated centroid (head_dim==256 only).
        // Own tid*8 contiguous layout — required by fwht_shfl_inverse_256.
        const int d0 = tid * 8;
        float ov[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        const int v_bytes_per_head = 4 + (head_dim * v_mode) / 8;
        const int v_bytes_per_pos  = n_kv_heads * v_bytes_per_head;
        const int v_head_off       = kv_h * v_bytes_per_head;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * v_bytes_per_pos + v_head_off;
            float cnorm = *(const float*)vb;
            const unsigned char* base = vb + 4 + tid * v_mode;   // v_mode bytes/thread
            unsigned int packed = (unsigned int)base[0];
            if (v_mode >= 2) packed |= ((unsigned int)base[1] << 8);
            if (v_mode >= 3) packed |= ((unsigned int)base[2] << 16);
            if (v_mode >= 4) packed |= ((unsigned int)base[3] << 24);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float c;
                if (v_mode == 2)      c = TURBO_C2_256[(packed >> (i * 2)) & 3];
                else if (v_mode == 3) c = TURBO_C3_256[(packed >> (i * 3)) & 7];
                else                  c = TURBO_C4_256[(packed >> (i * 4)) & 15];
                ov[i] += w * (cnorm * c);
            }
        }
        // V accumulated in ROTATED space (ov ≈ H·Σw·v). The inverse FWHT is
        // deferred to the dedicated lloyd reduce kernel, which applies it once
        // after the cross-tile combine (inverse is linear, so it commutes with
        // the weighted-sum + scalar normalize the reduce performs).
        #pragma unroll
        for (int i = 0; i < 8; i++) p[2 + d0 + i] = ov[i];
    }
}
