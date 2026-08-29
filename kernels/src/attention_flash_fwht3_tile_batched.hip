// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched flash attention tile — fwht3 KV V2 (K two-half FWHT-128 3-bit;
// V at Q8_0 OR two-half lloyd2/3/4, gated by v_mode).
// Partials layout: [sub_batch × n_heads × max_tiles × (2+head_dim)].
//
// Grid: [n_heads, max_tiles, sub_batch_size]. Block: [32, 1, 1].
// LDS: tile_size * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Tree-attention mode contract matches attention_flash_asym3_tile_batched.
extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_fwht3_tile_batched(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
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
    int block_cols,
    int v_mode          // 8 = legacy Q8_0 V; 2/3/4 = two-half lloyd FWHT-rotated centroid V
) {
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const int global_bid = batch_offset + local_bid;
    const bool tree_mode = (tree_bias != nullptr);
    // hipGraph-safe block_start: under captured tree-verify replay the
    // `block_start` scalar kernarg is baked at capture time and goes stale as
    // the committed prefix grows. positions[] is re-uploaded (uncaptured) every
    // cycle and the linearized tree root sits at positions[0] == start_pos ==
    // block_start, so derive it from the device buffer instead of the scalar.
    // Eager (non-capture) value is identical, so this is a no-op off-graph.
    const int eff_block_start = tree_mode ? positions[0] : block_start;
    const int seq_len = tree_mode ? (eff_block_start + block_cols) : (positions[global_bid] + 1);
    const int tile_start = tile_id * tile_size;
    if (tile_start >= seq_len) return;
    const int tile_end = min(tile_start + tile_size, seq_len);
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;

    const int k_bytes_per_head = 4 + (head_dim * 3) / 8;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    extern __shared__ float sdata[];
    float* scores = sdata;

    const float* q_head = q + local_bid * q_dim + h * head_dim;
    float mq[8];
    #pragma unroll
    for (int half = 0; half < 2; half++) {
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

    // Phase A: Q·K V2 — two fp16 cnorms; 24-bit pack = half0-four then half1-four.
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * k_bytes_per_pos + k_head_off;
        const float cn0 = (float)((const _Float16*)kb)[0];
        const float cn1 = (float)((const _Float16*)kb)[1];
        const unsigned char* base = kb + 4 + tid * 3;
        unsigned int packed = (unsigned int)base[0]
                            | ((unsigned int)base[1] << 8)
                            | ((unsigned int)base[2] << 16);
        float partial = mq[0] * (cn0 * TURBO_C3[ packed        & 7])
                      + mq[1] * (cn0 * TURBO_C3[(packed >> 3)  & 7])
                      + mq[2] * (cn0 * TURBO_C3[(packed >> 6)  & 7])
                      + mq[3] * (cn0 * TURBO_C3[(packed >> 9)  & 7])
                      + mq[4] * (cn1 * TURBO_C3[(packed >> 12) & 7])
                      + mq[5] * (cn1 * TURBO_C3[(packed >> 15) & 7])
                      + mq[6] * (cn1 * TURBO_C3[(packed >> 18) & 7])
                      + mq[7] * (cn1 * TURBO_C3[(packed >> 21) & 7]);
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0) {
            float s = partial * scale_attn;
            if (tree_mode && t >= eff_block_start) {
                s += tree_bias[global_bid * block_cols + (t - eff_block_start)];
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

    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                        + tile_id * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }

    if (v_mode == 8) {
        // Legacy Q8_0 V (normal space) — unchanged behavior.
        const int d0 = tid * 8;
        float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        const int v_blocks_per_head = head_dim / 32;
        const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
        const int v_kv_head_blk     = kv_h * v_blocks_per_head;
        const int v_row_stride      = v_total_blocks * 34;
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
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            p[2 + d0 + i] = out_vec[i];
        }
    } else if (v_mode == 3) {
        // Dedicated TURBO_C3 four-token-unrolled Lloyd3 V (no dynamic bit-width branch).
        float ov[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        const int v_bytes_per_head = 4 + (head_dim * 3) / 8;
        const int v_bytes_per_pos  = n_kv_heads * v_bytes_per_head;
        const int v_head_off       = kv_h * v_bytes_per_head;
        int tv = 0;
        for (; tv + 3 < tile_len; tv += 4) {
            float w0 = scores[tv + 0], w1 = scores[tv + 1];
            float w2 = scores[tv + 2], w3 = scores[tv + 3];
            const unsigned char* vb0 = v_cache + (size_t)(tile_start + tv + 0) * v_bytes_per_pos + v_head_off;
            const unsigned char* vb1 = v_cache + (size_t)(tile_start + tv + 1) * v_bytes_per_pos + v_head_off;
            const unsigned char* vb2 = v_cache + (size_t)(tile_start + tv + 2) * v_bytes_per_pos + v_head_off;
            const unsigned char* vb3 = v_cache + (size_t)(tile_start + tv + 3) * v_bytes_per_pos + v_head_off;
            float c00 = (float)((const _Float16*)vb0)[0], c01 = (float)((const _Float16*)vb0)[1];
            float c10 = (float)((const _Float16*)vb1)[0], c11 = (float)((const _Float16*)vb1)[1];
            float c20 = (float)((const _Float16*)vb2)[0], c21 = (float)((const _Float16*)vb2)[1];
            float c30 = (float)((const _Float16*)vb3)[0], c31 = (float)((const _Float16*)vb3)[1];
            const unsigned char* base0 = vb0 + 4 + tid * 3;
            const unsigned char* base1 = vb1 + 4 + tid * 3;
            const unsigned char* base2 = vb2 + 4 + tid * 3;
            const unsigned char* base3 = vb3 + 4 + tid * 3;
            unsigned int pk0 = (unsigned int)base0[0] | ((unsigned int)base0[1] << 8) | ((unsigned int)base0[2] << 16);
            unsigned int pk1 = (unsigned int)base1[0] | ((unsigned int)base1[1] << 8) | ((unsigned int)base1[2] << 16);
            unsigned int pk2 = (unsigned int)base2[0] | ((unsigned int)base2[1] << 8) | ((unsigned int)base2[2] << 16);
            unsigned int pk3 = (unsigned int)base3[0] | ((unsigned int)base3[1] << 8) | ((unsigned int)base3[2] << 16);
            #define ACC3(W, CN0, CN1, PK) do { \
                ov[0] += (W) * ((CN0) * TURBO_C3[(PK)        & 7]); \
                ov[1] += (W) * ((CN0) * TURBO_C3[((PK) >> 3)  & 7]); \
                ov[2] += (W) * ((CN0) * TURBO_C3[((PK) >> 6)  & 7]); \
                ov[3] += (W) * ((CN0) * TURBO_C3[((PK) >> 9)  & 7]); \
                ov[4] += (W) * ((CN1) * TURBO_C3[((PK) >> 12) & 7]); \
                ov[5] += (W) * ((CN1) * TURBO_C3[((PK) >> 15) & 7]); \
                ov[6] += (W) * ((CN1) * TURBO_C3[((PK) >> 18) & 7]); \
                ov[7] += (W) * ((CN1) * TURBO_C3[((PK) >> 21) & 7]); \
            } while (0)
            ACC3(w0, c00, c01, pk0);
            ACC3(w1, c10, c11, pk1);
            ACC3(w2, c20, c21, pk2);
            ACC3(w3, c30, c31, pk3);
            #undef ACC3
        }
        for (; tv < tile_len; tv++) {
            float w = scores[tv];
            const unsigned char* vb = v_cache + (size_t)(tile_start + tv) * v_bytes_per_pos + v_head_off;
            const float cn0 = (float)((const _Float16*)vb)[0];
            const float cn1 = (float)((const _Float16*)vb)[1];
            const unsigned char* base = vb + 4 + tid * 3;
            unsigned int packed = (unsigned int)base[0]
                                | ((unsigned int)base[1] << 8)
                                | ((unsigned int)base[2] << 16);
            ov[0] += w * (cn0 * TURBO_C3[ packed        & 7]);
            ov[1] += w * (cn0 * TURBO_C3[(packed >> 3)  & 7]);
            ov[2] += w * (cn0 * TURBO_C3[(packed >> 6)  & 7]);
            ov[3] += w * (cn0 * TURBO_C3[(packed >> 9)  & 7]);
            ov[4] += w * (cn1 * TURBO_C3[(packed >> 12) & 7]);
            ov[5] += w * (cn1 * TURBO_C3[(packed >> 15) & 7]);
            ov[6] += w * (cn1 * TURBO_C3[(packed >> 18) & 7]);
            ov[7] += w * (cn1 * TURBO_C3[(packed >> 21) & 7]);
        }
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const int d_base = half * 128 + tid * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                p[2 + d_base + i] = ov[half * 4 + i];
        }
    } else {
        // lloyd2/4: two-half V2; one-token-at-a-time ok for this screen.
        float ov[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        const int v_bytes_per_head = 4 + (head_dim * v_mode) / 8;
        const int v_bytes_per_pos  = n_kv_heads * v_bytes_per_head;
        const int v_head_off       = kv_h * v_bytes_per_head;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * v_bytes_per_pos + v_head_off;
            const float cn_h0 = (float)((const _Float16*)vb)[0];
            const float cn_h1 = (float)((const _Float16*)vb)[1];
            const unsigned char* base = vb + 4 + tid * v_mode;
            unsigned int packed = (unsigned int)base[0];
            if (v_mode >= 2) packed |= ((unsigned int)base[1] << 8);
            if (v_mode >= 3) packed |= ((unsigned int)base[2] << 16);
            if (v_mode >= 4) packed |= ((unsigned int)base[3] << 24);
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const float cnorm = (half == 0) ? cn_h0 : cn_h1;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int idx = half * 4 + i;
                    const int shift = idx * v_mode;
                    float c = (v_mode == 2)
                        ? TURBO_C2[(packed >> shift) & 3]
                        : TURBO_C4[(packed >> shift) & 15];
                    ov[idx] += w * (cnorm * c);
                }
            }
        }
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const int d_base = half * 128 + tid * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                p[2 + d_base + i] = ov[half * 4 + i];
        }
    }
}
