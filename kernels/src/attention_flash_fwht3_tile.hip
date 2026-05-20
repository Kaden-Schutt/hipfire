// Flash attention tile — fwht3 KV: K at FWHT-256 rotated 3-bit, V at Q8_0 (normal space).
//
// Single-pass 256-element signed-FWHT on Q to match the K-write rotation.
// K cache byte-identical to asym3 (4B cnorm + 32 threads × 3 bytes packed 3-bit).
//
// Tuned for head_dim=256 (Qwen 3.5): 32 threads × 8 values = 256 dims single pass.
// Q8_0 V reduce works as-is (output in normal space).
//
// Grid: [n_heads, n_tiles, 1]. Block: [32, 1, 1]. LDS: tile_size * 4.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_fwht3_tile(
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
    int max_tiles
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

    const int k_bytes_per_head = 4 + (head_dim * 3) / 8;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    const int v_blocks_per_head = head_dim / 32;
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q (8 values per thread) + signed-FWHT-256 forward.
    const float* q_head = q + h * head_dim;
    const int d0 = tid * 8;
    float mq0 = q_head[d0 + 0], mq1 = q_head[d0 + 1];
    float mq2 = q_head[d0 + 2], mq3 = q_head[d0 + 3];
    float mq4 = q_head[d0 + 4], mq5 = q_head[d0 + 5];
    float mq6 = q_head[d0 + 6], mq7 = q_head[d0 + 7];

    fwht_shfl_forward_256(mq0, mq1, mq2, mq3, mq4, mq5, mq6, mq7, signs1, signs2, tid);

    // Phase A: Q·K (3-bit K)
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
        if (tid == 0)
            scores[t_local] = partial * scale_attn;
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

    // Phase D: V (Q8_0 normal space) — 8 output dims per thread.
    float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const unsigned char* vb = v_cache + (size_t)t * v_row_stride;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int d = d0 + i;
            const int bi       = d / 32;
            const int bj       = d % 32;
            const int v_blk_off = (v_kv_head_blk + bi) * 34;
            const unsigned char* vbh = vb + v_blk_off;
            float vs = (float)*((const _Float16*)vbh);
            out_vec[i] += w * (vs * (float)((signed char)vbh[2 + bj]));
        }
    }

    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        p[2 + d0 + i] = out_vec[i];
    }
}
