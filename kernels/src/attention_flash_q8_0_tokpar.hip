// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// Token-parallel batched Q8_0 flash attention for long-context prefill.
//
// Replaces the per-token-serial Phase A of attention_flash_q8_0_tile_batched
// (which had all 32 lanes compute one token then __shfl_xor-reduce to a
// single score — 31/32 lanes wasted, serial over tile_len). Here each
// thread owns DISTINCT tokens and computes their full head_dim Q·K dot
// independently; no per-token wave reduction. gfx906 is issue-rate-bound,
// not DRAM-bound (<1% of HBM peak measured), so token-level parallelism +
// a tight per-thread FMA chain is the lever, not GQA K/V reuse.
//
// Online-softmax tiled so LDS stays bounded (scores[TILE] only) regardless
// of seq_len — no 15k cliff. Grid [n_heads, sub_batch]: one block owns the
// whole context for one (head, query-row), runs all tiles with a running
// (m, l, O) and writes the final output directly — no separate reduce.
//
// Both K and V are Q8_0: head_dim/32 blocks/head, 34 B/block (fp16 scale
// + 32 i8). cos/sin accepted for launcher ABI parity, ignored.
//
// Grid: [n_heads, sub_batch_size, 1]. Block: [BLOCK, 1, 1].
#include <hip/hip_runtime.h>

#ifndef Q8_TOKPAR_TILE
#define Q8_TOKPAR_TILE 256
#endif

extern "C" __global__ void attention_flash_q8_0_tokpar(
    const float* __restrict__ q,               // [sub_batch × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,                   // [sub_batch × n_heads × head_dim]
    const int* __restrict__ positions,         // [total_batch]
    const float* __restrict__ /*cos_theta*/,   // unused (Q8 K unrotated)
    const float* __restrict__ /*sin_theta*/,   // unused
    const float* __restrict__ tree_bias,       // optional [total_batch × block_cols]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int batch_offset,
    int block_start,                           // ignored when tree_bias == nullptr
    int block_cols                             // ignored when tree_bias == nullptr
) {
    const int h = blockIdx.x;
    const int local_bid = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int global_bid = batch_offset + local_bid;
    const bool tree_mode = (tree_bias != nullptr);
    const int seq_len = tree_mode ? (block_start + block_cols) : (positions[global_bid] + 1);

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks    = n_kv_heads * blocks_per_head;
    const int kv_head_blk     = kv_h * blocks_per_head;
    const int row_stride      = total_blocks * 34;

    // LDS: scores[TILE] + per-thread reduction workspace ws[nthreads]
    // + q_lds[head_dim] (Q staged once, reused across all tiles/tokens).
    extern __shared__ float sdata[];
    float* scores = sdata;                       // [TILE]
    float* ws     = sdata + Q8_TOKPAR_TILE;      // [nthreads]
    float* q_lds  = ws + nthreads;               // [head_dim]

    const float* q_head = q + local_bid * q_dim + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) q_lds[d] = q_head[d];
    __syncthreads();

    // Running online-softmax state (replicated across threads; updated
    // identically). O accumulated per-thread over the dims it owns, then
    // written out at the end.
    float m_run = -1e30f;
    float l_run = 0.0f;
    // Each thread owns dims [tid, tid+nthreads, ...] of the output.
    // head_dim/nthreads accumulators (<=2 for hd256/bd128, 1 for bd256).
    float o_acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) o_acc[i] = 0.0f;

    for (int tile_start = 0; tile_start < seq_len; tile_start += Q8_TOKPAR_TILE) {
        const int tile_end = min(tile_start + Q8_TOKPAR_TILE, seq_len);
        const int tile_len = tile_end - tile_start;

        // ── Phase A: token-parallel Q·K. Each thread owns whole tokens. ──
        for (int t_local = tid; t_local < tile_len; t_local += nthreads) {
            const int t = tile_start + t_local;
            const unsigned char* krow = k_cache + (size_t)t * row_stride + kv_head_blk * 34;
            float dot = 0.0f;
            // Full head_dim dot: blocks_per_head Q8 blocks of 32.
            for (int bi = 0; bi < blocks_per_head; bi++) {
                const unsigned char* kb = krow + bi * 34;
                float ks = (float)*((const _Float16*)kb);
                const signed char* codes = (const signed char*)(kb + 2);
                const float* ql = q_lds + bi * 32;
                #pragma unroll
                for (int j = 0; j < 32; j++)
                    dot += ql[j] * (ks * (float)codes[j]);
            }
            float s = dot * scale_attn;
            if (tree_mode && t >= block_start)
                s += tree_bias[global_bid * block_cols + (t - block_start)];
            scores[t_local] = s;
        }
        __syncthreads();

        // ── Phase B: tile max (block reduction over scores[0..tile_len]) ──
        float local_max = -1e30f;
        for (int i = tid; i < tile_len; i += nthreads)
            local_max = fmaxf(local_max, scores[i]);
        ws[tid] = local_max;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid + s]);
            __syncthreads();
        }
        const float tile_max = ws[0];
        __syncthreads();

        // ── Phase C: exp + tile sum ──
        float local_sum = 0.0f;
        for (int i = tid; i < tile_len; i += nthreads) {
            float e = expf(scores[i] - tile_max);
            scores[i] = e;
            local_sum += e;
        }
        ws[tid] = local_sum;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) ws[tid] += ws[tid + s];
            __syncthreads();
        }
        const float tile_sum = ws[0];
        __syncthreads();

        // ── Online-softmax merge of this tile into the running state ──
        const float m_new = fmaxf(m_run, tile_max);
        const float corr_old = expf(m_run - m_new);
        const float corr_tile = expf(tile_max - m_new);
        l_run = l_run * corr_old + tile_sum * corr_tile;

        // ── Phase D: V accumulation. Each thread owns output dims
        //    [tid, tid+nthreads, ...]; loops the tile's tokens (scores
        //    already exp'd, relative to tile_max → apply corr_tile). ──
        int acc_i = 0;
        for (int d = tid; d < head_dim; d += nthreads, acc_i++) {
            const int bi = d / 32;
            const int bj = d % 32;
            float acc = 0.0f;
            for (int t_local = 0; t_local < tile_len; t_local++) {
                const int t = tile_start + t_local;
                const unsigned char* vb = v_cache + (size_t)t * row_stride + (kv_head_blk + bi) * 34;
                float vs = (float)*((const _Float16*)vb);
                float vcode = (float)((signed char)vb[2 + bj]);
                acc += scores[t_local] * (vs * vcode);
            }
            o_acc[acc_i] = o_acc[acc_i] * corr_old + acc * corr_tile;
        }
        m_run = m_new;
        __syncthreads();
    }

    // ── Finalize: divide by running sum, write output ──
    const float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    float* out_row = out + local_bid * q_dim + h * head_dim;
    int acc_i = 0;
    for (int d = tid; d < head_dim; d += nthreads, acc_i++)
        out_row[d] = o_acc[acc_i] * inv_l;
}
