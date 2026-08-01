// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Q8_0 KV attention, single-token decode, BOUNDED shared memory.
//
// v2 (2026-07-31, long-context fix): v1 kept `scores[seq_len]` in LDS —
// O(seq_len) shared memory — exceeding gfx1100's 64 KiB LDS cap at
// seq_len ~ 15872 (16 KiB * 4 B) and faulting the GPU. v2 processes
// positions in fixed ATT_Q8_TILE-sized tiles with online softmax, so LDS =
// [ATT_Q8_TILE + head_dim + block] floats regardless of seq_len: safe at any
// context, and graph-safe (fixed LDS + position-independent grid).
//
// Grid: [n_heads, 1, 1]. Block: [256, 1, 1] (any power-of-two block works).
#include <hip/hip_runtime.h>

#define ATT_Q8_TILE 2048
#define ATT_Q8_ACC_MAX 16

extern "C" __global__ void attention_q8_0_kv(
    const float* __restrict__ q,         // [n_heads, head_dim]
    const unsigned char* __restrict__ k_cache,  // [max_seq, n_kv_heads*blocks_per_head*34]
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,             // [n_heads, head_dim]
    const int* __restrict__ pos_buf,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn
) {
    const int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];

    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* q_head = q + h * head_dim;
    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;

    float* scores = sdata;                    // [ATT_Q8_TILE]
    float* q_shared = sdata + ATT_Q8_TILE;    // [head_dim]
    float* workspace = sdata + ATT_Q8_TILE + head_dim;  // [nthreads]

    // Load Q into shared memory (once, used for all tiles).
    for (int d = tid; d < head_dim; d += nthreads)
        q_shared[d] = q_head[d];
    __syncthreads();

    // V accumulator: thread tid owns dims d = tid + i*nthreads.
    const int n_acc = (head_dim + nthreads - 1) / nthreads;
    float acc[ATT_Q8_ACC_MAX];
    for (int i = 0; i < n_acc && i < ATT_Q8_ACC_MAX; i++) acc[i] = 0.0f;

    // Online-softmax running stats (per block).
    float m = -1e30f;
    float l = 0.0f;

    for (int tile_start = 0; tile_start < seq_len; tile_start += ATT_Q8_TILE) {
        const int tile_len = (seq_len - tile_start < ATT_Q8_TILE)
                                 ? (seq_len - tile_start)
                                 : ATT_Q8_TILE;

        // Phase A: Q @ K^T for this tile -> LDS scores.
        for (int t = tid; t < tile_len; t += nthreads) {
            const int pos = tile_start + t;
            float dot = 0.0f;
            for (int bi = 0; bi < blocks_per_head; bi++) {
                const unsigned char* block =
                    k_cache + (size_t)pos * total_blocks_per_pos * 34
                            + (kv_head_block_start + bi) * 34;
                const float d = (float)*((const _Float16*)block);
                const float* qb = q_shared + bi * 32;
                for (int j = 0; j < 32; j++)
                    dot += qb[j] * (d * (float)((signed char)block[2 + j]));
            }
            scores[t] = dot * scale_attn;
        }
        __syncthreads();

        // Tile max via LDS tree reduction (block-wide, any power-of-two size).
        float tmax = -1e30f;
        for (int t = tid; t < tile_len; t += nthreads)
            tmax = fmaxf(tmax, scores[t]);
        workspace[tid] = tmax;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
            __syncthreads();
        }
        tmax = workspace[0];
        __syncthreads();  // workspace free for reuse

        // Merge into running stats and rescale the V accumulator.
        const float new_m = fmaxf(m, tmax);
        const float rescale = expf(m - new_m);
        l *= rescale;
        for (int i = 0; i < n_acc && i < ATT_Q8_ACC_MAX; i++) acc[i] *= rescale;

        float tsum = 0.0f;
        for (int t = tid; t < tile_len; t += nthreads) {
            const float e = expf(scores[t] - new_m);
            scores[t] = e;
            tsum += e;
        }
        workspace[tid] = tsum;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) workspace[tid] += workspace[tid + s];
            __syncthreads();
        }
        tsum = workspace[0];
        l += tsum;
        m = new_m;
        __syncthreads();  // protect LDS scores before Phase B reads

        // Phase B: V-weighted accumulation for this tile.
        // Each thread owns dims d = tid + i*nthreads and must fold every
        // position's softmax weight into its private acc (no position stride).
        for (int t = 0; t < tile_len; t++) {
            const float w = scores[t];
            const int pos = tile_start + t;
            const int v_blk_off = kv_head_block_start * 34;
            for (int i = 0; i < n_acc && i < ATT_Q8_ACC_MAX; i++) {
                const int d = tid + i * nthreads;
                if (d >= head_dim) break;
                const int bi = d / 32;
                const int bj = d % 32;
                const unsigned char* block =
                    v_cache + (size_t)pos * total_blocks_per_pos * 34
                            + v_blk_off + bi * 34;
                const float vd = (float)*((const _Float16*)block)
                               * (float)(signed char)block[2 + bj];
                acc[i] += w * vd;
            }
        }
        __syncthreads(); // protect LDS scores before the next tile overwrites
    }

    // Normalize and write out.
    float* out_head = out + h * head_dim;
    const float inv_l = 1.0f / l;
    for (int i = 0; i < n_acc && i < ATT_Q8_ACC_MAX; i++) {
        const int d = tid + i * nthreads;
        if (d >= head_dim) break;
        out_head[d] = acc[i] * inv_l;
    }
}
