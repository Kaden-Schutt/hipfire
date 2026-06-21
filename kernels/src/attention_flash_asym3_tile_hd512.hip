// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention tile for asym3, head_dim=512 (Gemma 4 full layers).
//
// Same K-side rotated 3-bit + V-side Q8_0 (normal space) layout as
// `attention_flash_asym3_tile`, but each thread covers 2 chunks of 8
// dims (= 16 dims total) so the 32-thread block still fits a 512-dim
// head with no cross-warp reduce. Per-tile output layout matches the
// hd=256 kernel; the existing `attention_flash_q8_0_reduce` kernel
// already loops over `head_dim/128` halves so it handles hd=512 unchanged.
//
// Geometry:
//   tid ∈ [0, 32)   — single warp
//   each thread:
//     chunk 0  → dims (tid*8) .. (tid*8 + 7)        — covers 0..255
//     chunk 1  → dims (tid*8 + 256) .. (tid*8 + 263) — covers 256..511
//   Givens blocks per thread: 4 + 4 = 8 (each block = 2 dims)
//   K bytes per thread: 3 + 3 = 6 (24 bits = 8 vals × 3 bits per chunk)
//   V values per thread: 8 + 8 = 16
//
// Grid / block / LDS identical to hd=256 variant — call sites need only
// switch the launch target name.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

__device__ __forceinline__ void givens_rot_fwd(
    float& a, float& b, float c, float s)
{
    float a2 = a * c - b * s;
    float b2 = a * s + b * c;
    a = a2; b = b2;
}

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_asym3_tile_hd512(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ pos_buf,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int window_size
) {
    // Hard requirement — see header comment. Other head_dims should land
    // in the original `attention_flash_asym3_tile` kernel (= 256) or a
    // future generalization.
    if (head_dim != 512) return;

    const int seq_len = pos_buf[0] + 1;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tile_id = blockIdx.y;
    const int tile_start = tile_id * tile_size;
    const int tile_end = min(tile_start + tile_size, seq_len);
    if (tile_start >= seq_len) return;
    const int tile_len = tile_end - tile_start;
    const int t_lo = (window_size > 0) ? max(0, seq_len - window_size) : 0;
    if (window_size > 0 && tile_end <= t_lo) {
        // Mark this tile as no-contribution. Each thread zeroes its 16 dims
        // (8 in each half-chunk) so the reduce sees a clean partial.
        float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
        if (threadIdx.x == 0) { p[0] = -1e30f; p[1] = 0.0f; }
        const int d0_lo = threadIdx.x * 8;
        const int d0_hi = d0_lo + 256;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            p[2 + d0_lo + i] = 0.0f;
            p[2 + d0_hi + i] = 0.0f;
        }
        return;
    }

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;

    // K geometry (3-bit rotated). For hd=512: 4 + 192 = 196 bytes per head.
    const int k_bytes_per_head = 4 + (head_dim * 3) / 8;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;
    // Per-thread byte offsets within a head:
    //   chunk 0 starts at byte 4 (after the 4-byte float scale) + tid*3
    //   chunk 1 starts 96 bytes later (192 bits worth = 256 dims of 3-bit packed) + tid*3
    const int k_chunk_bytes = (256 * 3) / 8;  // = 96 — matches the dim-256 split

    // V geometry (Q8_0). Each block holds 32 dims; for hd=512 → 16 blocks/head.
    const int v_blocks_per_head = head_dim / 32;
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q for both chunks (16 values per thread total). Apply Givens
    // rotation per the same per-block (cos, sin) sched used during write.
    const float* q_head = q + h * head_dim;
    const int d0_lo = tid * 8;          // chunk 0 base
    const int d0_hi = d0_lo + 256;      // chunk 1 base
    float mq0  = q_head[d0_lo + 0], mq1  = q_head[d0_lo + 1];
    float mq2  = q_head[d0_lo + 2], mq3  = q_head[d0_lo + 3];
    float mq4  = q_head[d0_lo + 4], mq5  = q_head[d0_lo + 5];
    float mq6  = q_head[d0_lo + 6], mq7  = q_head[d0_lo + 7];
    float mq8  = q_head[d0_hi + 0], mq9  = q_head[d0_hi + 1];
    float mq10 = q_head[d0_hi + 2], mq11 = q_head[d0_hi + 3];
    float mq12 = q_head[d0_hi + 4], mq13 = q_head[d0_hi + 5];
    float mq14 = q_head[d0_hi + 6], mq15 = q_head[d0_hi + 7];
    // Givens block indices: chunk 0 owns blocks tid*4..tid*4+3 (matches
    // hd=256 layout). Chunk 1 owns blocks 128 + tid*4..tid*4+3 (the
    // upper-half blocks; head_dim/2 = 256, so block index 128 = dim 256).
    const int b0_lo = tid * 4;
    const int b0_hi = b0_lo + 128;
    givens_rot_fwd(mq0,  mq1,  cos_theta[b0_lo + 0], sin_theta[b0_lo + 0]);
    givens_rot_fwd(mq2,  mq3,  cos_theta[b0_lo + 1], sin_theta[b0_lo + 1]);
    givens_rot_fwd(mq4,  mq5,  cos_theta[b0_lo + 2], sin_theta[b0_lo + 2]);
    givens_rot_fwd(mq6,  mq7,  cos_theta[b0_lo + 3], sin_theta[b0_lo + 3]);
    givens_rot_fwd(mq8,  mq9,  cos_theta[b0_hi + 0], sin_theta[b0_hi + 0]);
    givens_rot_fwd(mq10, mq11, cos_theta[b0_hi + 1], sin_theta[b0_hi + 1]);
    givens_rot_fwd(mq12, mq13, cos_theta[b0_hi + 2], sin_theta[b0_hi + 2]);
    givens_rot_fwd(mq14, mq15, cos_theta[b0_hi + 3], sin_theta[b0_hi + 3]);

    // Phase A: Q·K  (both chunks accumulated into a single scalar partial).
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const bool out_of_window = (window_size > 0) && (t < t_lo);
        float partial = 0.0f;
        if (!out_of_window) {
            const unsigned char* kb = k_cache + (size_t)t * k_bytes_per_pos + k_head_off;
            float cnorm = *(const float*)kb;
            // Chunk 0: 24 bits at offset 4 + tid*3.
            const unsigned char* base0 = kb + 4 + tid * 3;
            unsigned int packed0 = (unsigned int)base0[0]
                                 | ((unsigned int)base0[1] << 8)
                                 | ((unsigned int)base0[2] << 16);
            partial += mq0 * (cnorm * TURBO_C3_256[ packed0        & 7])
                     + mq1 * (cnorm * TURBO_C3_256[(packed0 >> 3)  & 7])
                     + mq2 * (cnorm * TURBO_C3_256[(packed0 >> 6)  & 7])
                     + mq3 * (cnorm * TURBO_C3_256[(packed0 >> 9)  & 7])
                     + mq4 * (cnorm * TURBO_C3_256[(packed0 >> 12) & 7])
                     + mq5 * (cnorm * TURBO_C3_256[(packed0 >> 15) & 7])
                     + mq6 * (cnorm * TURBO_C3_256[(packed0 >> 18) & 7])
                     + mq7 * (cnorm * TURBO_C3_256[(packed0 >> 21) & 7]);
            // Chunk 1: 24 bits at offset 4 + k_chunk_bytes + tid*3.
            const unsigned char* base1 = kb + 4 + k_chunk_bytes + tid * 3;
            unsigned int packed1 = (unsigned int)base1[0]
                                 | ((unsigned int)base1[1] << 8)
                                 | ((unsigned int)base1[2] << 16);
            partial += mq8  * (cnorm * TURBO_C3_256[ packed1        & 7])
                     + mq9  * (cnorm * TURBO_C3_256[(packed1 >> 3)  & 7])
                     + mq10 * (cnorm * TURBO_C3_256[(packed1 >> 6)  & 7])
                     + mq11 * (cnorm * TURBO_C3_256[(packed1 >> 9)  & 7])
                     + mq12 * (cnorm * TURBO_C3_256[(packed1 >> 12) & 7])
                     + mq13 * (cnorm * TURBO_C3_256[(packed1 >> 15) & 7])
                     + mq14 * (cnorm * TURBO_C3_256[(packed1 >> 18) & 7])
                     + mq15 * (cnorm * TURBO_C3_256[(packed1 >> 21) & 7]);
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0)
            scores[t_local] = out_of_window ? -1e30f : (partial * scale_attn);
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

    // Phase D: V (Q8_0 normal space). 16 output dims per thread = 2 × 8.
    float out_lo[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float out_hi[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const unsigned char* vb = v_cache + (size_t)t * v_row_stride;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int d_lo = d0_lo + i;
            const int bi_lo = d_lo / 32;
            const int bj_lo = d_lo & 31;
            const unsigned char* vbh_lo = vb + (v_kv_head_blk + bi_lo) * 34;
            float vs_lo = (float)*((const _Float16*)vbh_lo);
            out_lo[i] += w * (vs_lo * (float)((signed char)vbh_lo[2 + bj_lo]));

            const int d_hi = d0_hi + i;
            const int bi_hi = d_hi / 32;
            const int bj_hi = d_hi & 31;
            const unsigned char* vbh_hi = vb + (v_kv_head_blk + bi_hi) * 34;
            float vs_hi = (float)*((const _Float16*)vbh_hi);
            out_hi[i] += w * (vs_hi * (float)((signed char)vbh_hi[2 + bj_hi]));
        }
    }

    // Write partials. Layout matches hd=256 reduce — flat per-tile [2 + head_dim].
    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        p[2 + d0_lo + i] = out_lo[i];
        p[2 + d0_hi + i] = out_hi[i];
    }
}
