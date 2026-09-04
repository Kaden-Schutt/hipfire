// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kate
// hipfire — see LICENSE and NOTICE in the project root.

// Batched flash attention tile for asym3, head_dim=512 (Gemma 4 full layers).
//
// Combines the hd=512 per-thread 16-dim Q/K/V handling from
// `attention_flash_asym3_tile_hd512.hip` with the batched scheme from
// `attention_flash_asym3_tile_batched.hip`: blockIdx.z is the batch slot,
// `positions[]` holds per-batch absolute positions, and partials are
// laid out as `[sub_batch × n_heads × max_tiles × (2 + head_dim)]`.
//
// Geometry (matches single-token hd=512):
//   tid ∈ [0, 32)   — single warp
//   each thread:
//     chunk 0  → dims (tid*8) .. (tid*8 + 7)        — covers 0..255
//     chunk 1  → dims (tid*8 + 256) .. (tid*8 + 263) — covers 256..511
//   Givens blocks per thread: 4 + 4 = 8 (each block = 2 dims)
//
// Grid: [n_heads, max_tiles, sub_batch_size]. Block: [32, 1, 1].
// LDS: tile_size * 4 bytes.
#include <hip/hip_runtime.h>
#include <assert.h>
#include "turbo_common.h"
#include "givens_common.h"
#include "kv_slot_desc.h"

__device__ __forceinline__ void givens_rot_fwd_hd512_batched(
    float& a, float& b, float c, float s)
{
    float a2 = a * c - b * s;
    float b2 = a * s + b * c;
    a = a2; b = b2;
}

// Tree-attention mode (DDTree) is supported here in the same way as the
// hd=256 batched kernel — keeps the signature uniform across all asym3
// batched variants and lets the existing launch_asym_flash_batched dispatch
// helper push the same kernarg blob for both. Full-attention layers don't
// currently use tree mode, but keeping the arg avoids ABI divergence.
extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_asym3_tile_hd512_batched(
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
    int v_mode,                               // ABI parity; hd512 asym3 uses Q8 V
    int window_size,                          // 0 = full causal (Gemma 4 full layers)
    const KvSlotDesc* __restrict__ slot_descs, // [n_slots] or nullptr = legacy
    const int* __restrict__ row_slot           // [total_batch] or nullptr = legacy
) {
    (void)v_mode;
    if (head_dim != 512) return;

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

    // One code path for both modes: in legacy mode we synthesise a descriptor
    // with zero bases, so the address arithmetic below is unchanged and the
    // output is bitwise identical to the pre-SP1 kernel. row_slot is indexed
    // by GLOBAL row (batch_offset + blockIdx.z, i.e. global_bid above) — using
    // the local blockIdx.z alone would silently read the wrong slot once the
    // partials buffer forces the launcher into sub-batching.
    const int slot = (row_slot != nullptr) ? row_slot[global_bid] : 0;
    const KvSlotDesc desc = (slot_descs != nullptr)
        ? slot_descs[slot]
        : kv_slot_legacy(seq_len, max_seq);

    // `positions[row] + 1` (== `seq_len` above) is the PER-ROW CAUSAL BOUND:
    // how far this specific query row is allowed to attend. `desc.seq_len` is
    // the SLOT's logical KV length — allocation/capacity metadata, not a
    // per-row quantity. The two coincide only when a slot has exactly one
    // query row (M=1 decode); a slot verifying M>1 draft tokens has rows at
    // positions p, p+1, ..., p+M-1 that legitimately differ, and row 0 must
    // NOT see row 2's key. `desc` supplies the K/V slab BASE ADDRESS (via
    // kv_offset_for_k/v below) — it must never override the row's own
    // causal bound. This also keeps tile-count agreement with
    // attention_flash_asym_reduce_batched, which independently derives its
    // n_tiles from positions[global_bid]+1 and has no way to learn a
    // different eff_seq_len computed here; disagreement would make the
    // reduce iterate tiles this kernel early-returned from without writing a
    // partial, folding stale (possibly another slot's) memory into the
    // output.
    const int eff_seq_len = seq_len;

    const int tile_start = tile_id * tile_size;
    if (tile_start >= eff_seq_len) return;
    const int tile_end = min(tile_start + tile_size, eff_seq_len);
    const int tile_len = tile_end - tile_start;
    const bool sliding = (window_size > 0) && !tree_mode;
    const int t_lo = sliding ? max(0, eff_seq_len - window_size) : 0;
    if (sliding && tile_end <= t_lo) {
        float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                            + tile_id * (2 + head_dim);
        if (tid == 0) { p[0] = -1e30f; p[1] = 0.0f; }
        const int d0_lo_z = tid * 8;
        const int d0_hi_z = d0_lo_z + 256;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            p[2 + d0_lo_z + i] = 0.0f;
            p[2 + d0_hi_z + i] = 0.0f;
        }
        return;
    }

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;

    // K geometry (hd=512: 4 + 192 = 196 bytes per head).
    const int k_bytes_per_head = 4 + (head_dim * 3) / 8;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;
    const int k_chunk_bytes    = (256 * 3) / 8;  // 96 — chunk 1 offset within head

    // V geometry (Q8_0). 16 blocks/head at hd=512.
    const int v_blocks_per_head = head_dim / 32;
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q for both chunks for THIS batch position.
    const float* q_head = q + local_bid * q_dim + h * head_dim;
    const int d0_lo = tid * 8;
    const int d0_hi = d0_lo + 256;
    float mq0  = q_head[d0_lo + 0], mq1  = q_head[d0_lo + 1];
    float mq2  = q_head[d0_lo + 2], mq3  = q_head[d0_lo + 3];
    float mq4  = q_head[d0_lo + 4], mq5  = q_head[d0_lo + 5];
    float mq6  = q_head[d0_lo + 6], mq7  = q_head[d0_lo + 7];
    float mq8  = q_head[d0_hi + 0], mq9  = q_head[d0_hi + 1];
    float mq10 = q_head[d0_hi + 2], mq11 = q_head[d0_hi + 3];
    float mq12 = q_head[d0_hi + 4], mq13 = q_head[d0_hi + 5];
    float mq14 = q_head[d0_hi + 6], mq15 = q_head[d0_hi + 7];
    const int b0_lo = tid * 4;
    const int b0_hi = b0_lo + 128;
    givens_rot_fwd_hd512_batched(mq0,  mq1,  cos_theta[b0_lo + 0], sin_theta[b0_lo + 0]);
    givens_rot_fwd_hd512_batched(mq2,  mq3,  cos_theta[b0_lo + 1], sin_theta[b0_lo + 1]);
    givens_rot_fwd_hd512_batched(mq4,  mq5,  cos_theta[b0_lo + 2], sin_theta[b0_lo + 2]);
    givens_rot_fwd_hd512_batched(mq6,  mq7,  cos_theta[b0_lo + 3], sin_theta[b0_lo + 3]);
    givens_rot_fwd_hd512_batched(mq8,  mq9,  cos_theta[b0_hi + 0], sin_theta[b0_hi + 0]);
    givens_rot_fwd_hd512_batched(mq10, mq11, cos_theta[b0_hi + 1], sin_theta[b0_hi + 1]);
    givens_rot_fwd_hd512_batched(mq12, mq13, cos_theta[b0_hi + 2], sin_theta[b0_hi + 2]);
    givens_rot_fwd_hd512_batched(mq14, mq15, cos_theta[b0_hi + 3], sin_theta[b0_hi + 3]);

    // Phase A: Q·K — both chunks accumulated into a single scalar partial.
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const bool out_of_window = sliding && (t < t_lo);
        float partial = 0.0f;
        if (!out_of_window) {
            const unsigned char* kb = k_cache + kv_offset_for_k(desc, t, k_bytes_per_pos) + k_head_off;
            float cnorm = *(const float*)kb;
            // Chunk 0
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
            // Chunk 1
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
        if (tid == 0) {
            float s = out_of_window ? -1e30f : (partial * scale_attn);
            if (tree_mode && t >= eff_block_start) {
                s += tree_bias[global_bid * block_cols + (t - eff_block_start)];
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

    // Phase D: V — 16 output dims per thread (2 × 8).
    float out_lo[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float out_hi[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const unsigned char* vb = v_cache + kv_offset_for_v(desc, t, v_row_stride);
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

    // Write partials — batched layout: [sub_batch × n_heads × max_tiles × stride].
    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                        + tile_id * (2 + head_dim);
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
