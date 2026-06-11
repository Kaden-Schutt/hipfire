// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// gfx906 WAVE64 dp4a flash attention for Q8_0 KV, long context.
//
// Wave64 sibling of attention_flash_q8_0_dp4a.gfx906.hip. rocprofv2
// (2026-05-27) showed the wave32-shaped dp4a/tile kernels run at
// VALUUtilization = 48.6% on gfx906 — the upper 32 lanes of every
// 64-lane wave idle (`__launch_bounds__(32)`, wg=32, shfl over 32). This
// variant uses all 64 lanes: head_dim=256 split as 4 dims/lane × 64
// lanes = 256, one sdot4 per token per lane, wave reduce over 64.
//
// Per-32-block Q scale is now shared by 8 lanes (32 dims / 4 dims-per-lane);
// the q-scale max-reduction runs over those 8 lanes (xor 1,2,4).
//
// Same tiled-partials layout + asym reduce reuse as the wave32 kernels.
// cos/sin accepted for launcher ABI parity, ignored (Q8 K unrotated).
//
// Grid: [n_heads, max_tiles, sub_batch]. Block: [64,1,1].
#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(64, 8)
__global__ void attention_flash_q8_0_dp4a_wave64(
    const float* __restrict__ q,               // [sub_batch × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    const float* __restrict__ /*cos_theta*/,
    const float* __restrict__ /*sin_theta*/,
    const float* __restrict__ tree_bias,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int block_start,
    int block_cols
) {
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;                // 0..63

    const int global_bid = batch_offset + local_bid;
    const bool tree_mode = (tree_bias != nullptr);
    const int seq_len = tree_mode ? (block_start + block_cols) : (positions[global_bid] + 1);
    const int tile_start = tile_id * tile_size;
    if (tile_start >= seq_len) return;
    const int tile_end = min(tile_start + tile_size, seq_len);
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;

    const int blocks_per_head = head_dim / 32;          // 8 for hd=256
    const int total_blocks    = n_kv_heads * blocks_per_head;
    const int kv_head_blk     = kv_h * blocks_per_head;
    const int row_stride      = total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Each of 64 lanes owns 4 consecutive dims (d0 = tid*4) → exactly one
    // sdot4. 8 lanes share each 32-block (32 dims / 4).
    const int d0 = tid * 4;
    const int my_block = d0 / 32;               // 0..blocks_per_head-1
    const int bj = d0 % 32;                     // byte offset within the block
    const float* q_head = q + local_bid * q_dim + h * head_dim;

    float qf[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) qf[i] = q_head[d0 + i];

    // Per-32-block max|q| → scale, reduced across the 8 lanes that share
    // the block (lanes with the same my_block). d0/32 groups 8 consecutive
    // lanes, so xor over 1,2,4 covers exactly those 8.
    float bmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) bmax = fmaxf(bmax, fabsf(qf[i]));
    bmax = fmaxf(bmax, __shfl_xor(bmax, 1));
    bmax = fmaxf(bmax, __shfl_xor(bmax, 2));
    bmax = fmaxf(bmax, __shfl_xor(bmax, 4));     // all 8 lanes in block agree
    const float q_scale = (bmax > 0.0f) ? (bmax / 127.0f) : 1.0f;
    const float inv_q_scale = 1.0f / q_scale;

    // Pack this lane's 4 i8 into one i32.
    int q_packed = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int v = (int)rintf(qf[i] * inv_q_scale);
        v = max(-127, min(127, v));
        q_packed |= (v & 0xFF) << (i * 8);
    }

    // ── Phase A: Q·K via one sdot4 per token per lane, reduce over 64. ──
    for (int t_local = 0; t_local < tile_len; t_local++) {
        const int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * row_stride + (kv_head_blk + my_block) * 34;
        const float k_scale = (float)*((const _Float16*)kb);
        const signed char* kc = (const signed char*)(kb + 2) + bj;
        int k_packed = (kc[0] & 0xFF) | ((kc[1] & 0xFF) << 8)
                     | ((kc[2] & 0xFF) << 16) | ((kc[3] & 0xFF) << 24);
        int acc = __builtin_amdgcn_sdot4(q_packed, k_packed, 0, false);
        float lane_partial = (float)acc * q_scale * k_scale;
        // Wave reduce across all 64 lanes → full head_dim dot.
        for (int off = 32; off > 0; off >>= 1)
            lane_partial += __shfl_xor(lane_partial, off);
        if (tid == 0) {
            float s = lane_partial * scale_attn;
            if (tree_mode && t >= block_start)
                s += tree_bias[global_bid * block_cols + (t - block_start)];
            scores[t_local] = s;
        }
    }
    __syncthreads();

    // Phase B: tile max (64-lane strided)
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 64)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 32; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    // Phase C: exp + sum
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 64) {
        float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 32; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    float tile_sum = local_sum;
    __syncthreads();

    // Phase D: V-weighted accumulation (Q8_0, this lane's 4 dims).
    float out_vec[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const unsigned char* vbh = v_cache + (size_t)t * row_stride + (kv_head_blk + my_block) * 34;
        float vs = (float)*((const _Float16*)vbh);
        const signed char* vc = (const signed char*)(vbh + 2) + bj;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_vec[i] += w * (vs * (float)vc[i]);
    }

    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                        + tile_id * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    #pragma unroll
    for (int i = 0; i < 4; i++)
        p[2 + d0 + i] = out_vec[i];
}
