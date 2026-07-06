// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention tile kernel — sequential dot product + 2-pass softmax.
//
// Grid: [n_heads, n_tiles, 1]. Block: [32, 1, 1] (one WAVE32).
// LDS: tile_size floats for scores + head_dim floats for Q.
//   head_dim=128 → (128+128)*4 = 1024 B.
//   head_dim=256 → (128+256)*4 = 1536 B.
//
// Wave-cooperative dot product: each thread owns 4 of head_dim K elements
// (per half), wave-reduces via __shfl_xor. For head_dim > 128 the kernel
// iterates over `n_halves = (head_dim + 127) / 128` halves (one of 128
// elements each) inside the per-token loop and accumulates the partial
// before the wave reduction — preserves the exact same mathematical sum
// the baseline kernel produces and avoids any softmax-amplified score
// error from re-ordering FMAs.
//
// V accumulation also loops over halves so all `head_dim` output dims are
// written to partials (the previous version only wrote the first 128 dims,
// leaving the upper half of the partials row uninitialized — fine when the
// buffer was freshly malloc'd-zero but catastrophic when stale prefill data
// later collided with the reduce kernel's full-head_dim read).
//
// Works on RDNA1 (gfx1010), RDNA2 (gfx1030), RDNA3 (gfx1100), RDNA4 (gfx1200+).
#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_q8_0_tile(
    const float* __restrict__ q,         // [n_heads, head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,        // [n_heads, max_tiles, 2 + head_dim]
    const int* __restrict__ pos_buf,     // pos_buf[0] = last position
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int window                           // sliding-window span; <= 0 = full causal
) {
    const int seq_len = pos_buf[0] + 1;
    // Sliding-window lower bound: keys at t < win_lo are outside the window
    // [seq_len-window, seq_len) and masked. window <= 0 = full causal.
    const int win_lo = (window > 0) ? (seq_len - window) : 0;
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

    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;
    const int row_stride = total_blocks_per_pos * 34;

    // Number of 128-element halves spanning head_dim. One pass per half
    // in Phase A (Q·K) and Phase D (V) so a 32-thread wave covers the
    // full head_dim regardless of size.
    const int n_halves = (head_dim + 127) / 128;

    // LDS layout: [tile_size scores][head_dim Q values]
    extern __shared__ float sdata[];
    float* scores = sdata;
    float* q_lds  = sdata + tile_size;

    // Load Q into LDS. 32 threads × 4 values × n_halves = head_dim values.
    const float* q_head = q + h * head_dim;
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        q_lds[d_base + 0] = q_head[d_base + 0];
        q_lds[d_base + 1] = q_head[d_base + 1];
        q_lds[d_base + 2] = q_head[d_base + 2];
        q_lds[d_base + 3] = q_head[d_base + 3];
    }
    __syncthreads();

    // ═══ Phase A: compute tile_len dot products (wave-cooperative) ═══
    // Each thread handles 4 of 128 elements per half, accumulates across
    // halves, then wave-reduces via __shfl_xor. Thread assignment within a
    // half: bi_k = tid/8, offset = (tid%8)*4. Block index across halves:
    // half*4 + bi_k.
    const int k_off = (tid % 8) * 4;

    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        if (t < win_lo) {                  // outside sliding window → mask, skip dot
            if (tid == 0) scores[t_local] = -INFINITY;
            continue;                      // t is wave-uniform, all lanes skip together
        }
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            const int k_bi = half * 4 + tid / 8;
            const int k_blk_off = (kv_head_block_start + k_bi) * 34;
            const unsigned char* kb = k_cache + (size_t)t * row_stride + k_blk_off;
            float ks = (float)*((const _Float16*)kb);
            const float* qb_thread = q_lds + k_bi * 32 + k_off;
            partial += qb_thread[0] * (ks * (float)((signed char)kb[2 + k_off + 0]))
                     + qb_thread[1] * (ks * (float)((signed char)kb[2 + k_off + 1]))
                     + qb_thread[2] * (ks * (float)((signed char)kb[2 + k_off + 2]))
                     + qb_thread[3] * (ks * (float)((signed char)kb[2 + k_off + 3]));
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0)
            scores[t_local] = partial * scale_attn;
    }
    __syncthreads();

    // ═══ Phase B: find tile max ═══
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    // ═══ Phase C: exp + sum ═══
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

    // ═══ Write tile partials ═══
    //
    // Stride by max_tiles (derived from max_seq) so the layout matches the
    // attention_flash_q8_0_reduce kernel, which uses max_tiles for its
    // per-head row stride. Using n_tiles (dynamic, ceil(seq_len/tile_size))
    // here would put head h's data at offset h*n_tiles*stride, while the
    // reduce reads from h*max_tiles*stride — mismatch for h>0 whenever
    // seq_len < max_seq. This is the asym2/3/4 tile-kernel convention.
    // Symptom of the previous mismatch: at long context all heads except
    // h=0 produced NaN/garbage outputs (reduce reading uninitialized or
    // wrong-head stale partials), poisoning every subsequent layer and
    // producing 100% NaN logits at the first single-token decode step
    // after a long prefill (see fix/long-decode-nan).
    const int max_tiles = (max_seq + tile_size - 1) / tile_size;
    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }

    // ═══ Phase D: V-weighted accumulation (loops over halves so all
    //              head_dim output dims are written, not just first 128) ═══
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        const int bi = d_base / 32;
        const int bj_base = d_base % 32;
        const int v_blk_off = (kv_head_block_start + bi) * 34;
        float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * row_stride + v_blk_off;
            float vs = (float)*((const _Float16*)vb);
            out0 += w * (vs * (float)((signed char)vb[2 + bj_base + 0]));
            out1 += w * (vs * (float)((signed char)vb[2 + bj_base + 1]));
            out2 += w * (vs * (float)((signed char)vb[2 + bj_base + 2]));
            out3 += w * (vs * (float)((signed char)vb[2 + bj_base + 3]));
        }
        p[2 + d_base + 0] = out0;
        p[2 + d_base + 1] = out1;
        p[2 + d_base + 2] = out2;
        p[2 + d_base + 3] = out3;
    }
}
