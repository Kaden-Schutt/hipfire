// Batched flash attention tile for asym4: K at 4-bit rotated, V at Q8_0.
// Processes sub_batch_size positions via blockIdx.z.
// 2-pass K for head_dim=256 (128-dim per pass).
//
// Partials layout: [sub_batch × n_heads × max_tiles × (2+head_dim)].
// Per-thread writes 2 halves × 4 dims per half.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

// ─── Tree-attention mode (DDTree batched verify) ──────────────────────────
// See attention_flash_asym3_tile_batched.hip for the full contract — same
// semantics apply here. Summary: `tree_bias != nullptr` → seq_len extends
// to `block_start + block_cols` and in-block key scores receive the
// per-row bias (-inf on non-ancestors, 0 on ancestors).
extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_asym4_tile_batched(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
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
    int /*kv_window*/                          // accepted for shared launcher ABI; asym4 ignores (sliding-window FA is asym3-only — PRD docs/plans/sliding-window-fa.prd)
) {
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

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
    const int n_halves = head_dim / 128;

    const int k_bytes_per_head = 4 + head_dim / 2;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    const int v_blocks_per_head = head_dim / 32;
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q + per-half Givens forward
    const float* q_head = q + local_bid * q_dim + h * head_dim;
    float mq[8];
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

    // Phase A: Q·K (4-bit K)
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * k_bytes_per_pos + k_head_off;
        float cnorm = *(const float*)kb;
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            unsigned short pk = *(const unsigned short*)(kb + 4 + half * 64 + tid * 2);
            partial += mq[half * 4 + 0] * (cnorm * TURBO_C4[(pk)       & 0xF])
                     + mq[half * 4 + 1] * (cnorm * TURBO_C4[(pk >> 4)  & 0xF])
                     + mq[half * 4 + 2] * (cnorm * TURBO_C4[(pk >> 8)  & 0xF])
                     + mq[half * 4 + 3] * (cnorm * TURBO_C4[(pk >> 12) & 0xF]);
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0) {
            float s = partial * scale_attn;
            if (tree_mode && t >= block_start) {
                s += tree_bias[global_bid * block_cols + (t - block_start)];
            }
            scores[t_local] = s;
        }
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

    // Phase D: V (Q8_0 normal space, 2 halves for hd=256)
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

    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                        + tile_id * (2 + head_dim);
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
