// Batched flash attention tile for asym3: K at 3-bit rotated, V at Q8_0.
// Processes sub_batch_size positions in one launch via blockIdx.z.
// Partials layout: [sub_batch × n_heads × max_tiles × (2+head_dim)].
//
// Grid: [n_heads, max_tiles, sub_batch_size]. Block: [32, 1, 1].
// LDS: tile_size * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

// ─── Tree-attention mode (DDTree batched verify) ──────────────────────────
// When `tree_bias != nullptr`:
//   - Effective seq_len is `block_start + block_cols` (full cache including
//     the tree block; `positions[global_bid]` is ignored as a cutoff).
//   - For in-block keys (t >= block_start) the score receives an additive
//     bias `tree_bias[global_bid * block_cols + (t - block_start)]` before
//     the tile max / exp / sum. Caller supplies -inf on non-ancestors so
//     exp collapses to 0.
//   - Prompt keys (t < block_start) remain unmasked.
// When `tree_bias == nullptr` this matches the legacy causal path exactly.
extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_asym3_tile_batched(
    const float* __restrict__ q,               // [sub_batch × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,         // [total_batch]
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    const float* __restrict__ tree_bias,       // optional [total_batch × block_cols]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int block_start,                           // ignored when tree_bias == nullptr
    int block_cols,                            // ignored when tree_bias == nullptr
    int kv_window                              // 0 = unbounded; else clip to last kv_window positions
) {
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const int global_bid = batch_offset + local_bid;
    const bool tree_mode = (tree_bias != nullptr);
    const int seq_len = tree_mode ? (block_start + block_cols) : (positions[global_bid] + 1);
    // Sliding window applied uniformly to AR and tree-verify modes. In tree
    // mode the in-block keys [block_start, seq_len) are the most recent
    // positions and stay in-window for any non-zero kv_window; only the
    // pre-block prompt at [0, block_start) gets clipped. Tree bias indexing
    // is preserved per-position (unchanged), so even partial tiles spanning
    // the prompt/block boundary apply the bias correctly.
    const int kv_start = (kv_window > 0 && seq_len > kv_window) ? (seq_len - kv_window) : 0;
    const int tile_start_raw = tile_id * tile_size;
    if (tile_start_raw >= seq_len) return;
    const int tile_end = min(tile_start_raw + tile_size, seq_len);
    const int tile_start = max(tile_start_raw, kv_start);
    if (tile_start >= tile_end) {
        // Tile entirely outside window — write sentinel so reduce skips.
        if (tid == 0) {
            float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                                + tile_id * (2 + head_dim);
            p[0] = -1e30f;
            p[1] = 0.0f;
        }
        return;
    }
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;

    // K (3-bit rotated)
    const int k_bytes_per_head = 4 + (head_dim * 3) / 8;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    // V (Q8_0)
    const int v_blocks_per_head = head_dim / 32;
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q for this batch position + per-thread 8 values, forward-rotate.
    const float* q_head = q + local_bid * q_dim + h * head_dim;
    const int d0 = tid * 8;
    float mq0 = q_head[d0 + 0], mq1 = q_head[d0 + 1];
    float mq2 = q_head[d0 + 2], mq3 = q_head[d0 + 3];
    float mq4 = q_head[d0 + 4], mq5 = q_head[d0 + 5];
    float mq6 = q_head[d0 + 6], mq7 = q_head[d0 + 7];
    const int b0 = tid * 4;
    {
        float c = cos_theta[b0 + 0], s = sin_theta[b0 + 0];
        float a2 = mq0 * c - mq1 * s; mq1 = mq0 * s + mq1 * c; mq0 = a2;
    }
    {
        float c = cos_theta[b0 + 1], s = sin_theta[b0 + 1];
        float a2 = mq2 * c - mq3 * s; mq3 = mq2 * s + mq3 * c; mq2 = a2;
    }
    {
        float c = cos_theta[b0 + 2], s = sin_theta[b0 + 2];
        float a2 = mq4 * c - mq5 * s; mq5 = mq4 * s + mq5 * c; mq4 = a2;
    }
    {
        float c = cos_theta[b0 + 3], s = sin_theta[b0 + 3];
        float a2 = mq6 * c - mq7 * s; mq7 = mq6 * s + mq7 * c; mq6 = a2;
    }

    // Phase A: Q·K
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * k_bytes_per_pos + k_head_off;
        float cnorm = *(const float*)kb;
        const unsigned char* base = kb + 4 + tid * 3;
        unsigned int packed = (unsigned int)base[0]
                            | ((unsigned int)base[1] << 8)
                            | ((unsigned int)base[2] << 16);
        float partial = mq0 * (cnorm * TURBO_C3_256[ packed        & 7])
                      + mq1 * (cnorm * TURBO_C3_256[(packed >> 3)  & 7])
                      + mq2 * (cnorm * TURBO_C3_256[(packed >> 6)  & 7])
                      + mq3 * (cnorm * TURBO_C3_256[(packed >> 9)  & 7])
                      + mq4 * (cnorm * TURBO_C3_256[(packed >> 12) & 7])
                      + mq5 * (cnorm * TURBO_C3_256[(packed >> 15) & 7])
                      + mq6 * (cnorm * TURBO_C3_256[(packed >> 18) & 7])
                      + mq7 * (cnorm * TURBO_C3_256[(packed >> 21) & 7]);
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0) {
            float s = partial * scale_attn;
            // Tree-attention bias on in-block keys. -inf bias makes exp(s-max)
            // collapse to 0; the ancestor set always contains the root so at
            // least one finite in-block score exists for every query row.
            if (tree_mode && t >= block_start) {
                s += tree_bias[global_bid * block_cols + (t - block_start)];
            }
            scores[t_local] = s;
        }
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

    // Phase D: V-weighted accumulation (Q8_0 normal space)
    float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const unsigned char* vb = v_cache + (size_t)t * v_row_stride;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int d = d0 + i;
            const int bi = d / 32;
            const int bj = d % 32;
            const int v_blk_off = (v_kv_head_blk + bi) * 34;
            const unsigned char* vbh = vb + v_blk_off;
            float vs = (float)*((const _Float16*)vbh);
            out_vec[i] += w * (vs * (float)((signed char)vbh[2 + bj]));
        }
    }

    // Write partials — layout: [sub_batch × n_heads × max_tiles × stride]
    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                        + tile_id * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        p[2 + d0 + i] = out_vec[i];
    }
}
