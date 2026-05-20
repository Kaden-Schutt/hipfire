// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched causal attention with Q8_0 quantized KV cache. Counterpart of
// attention_q8_0_kv for prefill batching — processes N queries in one launch,
// each with its own causal window read from positions[] array.
//
// Grid: [n_heads, batch_size, 1]
// Block processes (head h, batch row b). The query comes from
// q[b * n_heads * head_dim + h * head_dim]; the causal window extends
// from position 0 through positions[b] inclusive. Output is written to
// out[b * n_heads * head_dim + h * head_dim].
//
// Shared memory layout (host must pass `(max_seq + nthreads + head_dim)*4`
// as the dynamic LDS size, where max_seq is the MAX positions[b]+1 in
// this launch):
//   scores[seq_len] + workspace[nthreads] + q_shared[head_dim]
//
// Byte-exact with attention_q8_0_kv at batch_size=1: same math, same
// reduction order, same shared-memory layout (the workspace and q_shared
// pointers are derived relative to seq_len so different batch rows with
// different seq_len each have their own valid slice within the per-block
// allocation).
//
// ─── Tree-attention mode (DDTree batched verify) ──────────────────────────
// When `tree_bias != nullptr` the kernel runs in tree mode:
//   - seq_len = block_start + block_cols (full cache up to and including
//     the tree block, ignoring `positions[b]` for the causal cutoff).
//   - For keys in [block_start, block_start + block_cols), the score is
//     augmented by `tree_bias[b * block_cols + (t - block_start)]`. The
//     caller uses `+0.0` on ancestor-or-self positions and `-inf` on
//     non-ancestors; this gates within-block attention to the tree's
//     visibility matrix without changing the prompt path.
//   - Keys in [0, block_start) are unmasked (prompt is always visible).
//   - Rope-encoded K entries in the block may have LOGICAL positions that
//     differ from their cache slot (siblings share a depth). That's
//     handled host-side at KV-write time — this kernel just does dot
//     products on whatever is in the cache.
// When `tree_bias == nullptr` the kernel falls back to standard causal
// (byte-identical to the pre-tree path).
extern "C" __global__ void attention_q8_0_kv_batched(
    const float* __restrict__ q,            // [batch_size × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache,  // [max_seq × n_kv_heads × head_dim Q8_0 blocks]
    const unsigned char* __restrict__ v_cache,  // same layout
    float* __restrict__ out,                // [batch_size × n_heads × head_dim]
    const int* __restrict__ positions,      // [batch_size]
    const float* __restrict__ tree_bias,    // optional [batch_size × block_cols] or nullptr
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int block_start,                        // ignored when tree_bias == nullptr
    int block_cols                          // ignored when tree_bias == nullptr
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (h >= n_heads) return;

    const bool tree_mode = (tree_bias != nullptr);
    // Tree mode: iterate over the full cache including the block.
    // Causal mode: attend to 0..positions[b] inclusive (unchanged).
    const int seq_len = tree_mode ? (block_start + block_cols) : (positions[b] + 1);
    extern __shared__ float sdata[];

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int q_stride = n_heads * head_dim;
    const float* q_head = q + (long long)b * q_stride + h * head_dim;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;

    float* scores = sdata;
    float* workspace = sdata + seq_len;
    float* q_shared = workspace + nthreads;

    // Preload this batch row's Q head into shared memory.
    for (int d = tid; d < head_dim; d += nthreads)
        q_shared[d] = q_head[d];
    __syncthreads();

    // Phase 1: Q @ K^T with Q8_0 dequant.
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float dot = 0.0f;
        for (int bi = 0; bi < blocks_per_head; bi++) {
            const unsigned char* blk = k_cache + (size_t)t * total_blocks_per_pos * 34
                                      + (kv_head_block_start + bi) * 34;
            float d = (float)*((const _Float16*)blk);
            const float* qb = q_shared + bi * 32;
            for (int j = 0; j < 32; j++) {
                dot += qb[j] * (d * (float)((signed char)blk[2 + j]));
            }
        }
        float s = dot * scale_attn;
        // Tree-attention bias: add per-row mask for keys inside the block.
        // Prompt keys (t < block_start) stay unmasked. A -inf bias on a
        // non-ancestor key makes expf(s - max) collapse to 0, so that key
        // contributes nothing to the numerator or denominator. Correctness
        // of the running-max reduction requires that we NOT mix -inf into
        // the max when every non-ancestor gets -inf — but the ancestor set
        // always includes the root (slot 0 in the block) so at least one
        // in-block score is finite on every row.
        if (tree_mode && t >= block_start) {
            s += tree_bias[b * block_cols + (t - block_start)];
        }
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    float max_val = workspace[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    float sum_val = workspace[0];
    __syncthreads();

    for (int t = tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // Phase 2: weighted V sum with Q8_0 dequant.
    float* out_head = out + (long long)b * q_stride + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        int bi = d / 32;
        int bj = d % 32;
        for (int t = 0; t < seq_len; t++) {
            const unsigned char* blk = v_cache + (size_t)t * total_blocks_per_pos * 34
                                      + (kv_head_block_start + bi) * 34;
            float vd = (float)*((const _Float16*)blk) * (float)(signed char)blk[2 + bj];
            val += scores[t] * vd;
        }
        out_head[d] = val;
    }
}
