// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Fused KVarN flash attention tile (Phase D2): reads the variance-normalized
// 4-bit K records IN PLACE with inline dequant, instead of v1's "build a full
// f16 shadow K then read it". V stays Q8_0. This preserves KVarN's 4-bit K
// storage (the point of the feature) while removing v1's per-token
// `kvarn_build_kcache` pass + the physical_cap×kv_dim f16 shadow buffer, cutting
// decode K bandwidth ~4× (4-bit records vs f16 shadow) and a kernel launch/layer.
//
// The flash's 128-token tiles align exactly with the KVarN 128-token blocks:
//   tile_id <  n_full_blocks → tokens come from record block `tile_id`
//                              (one [head_dim × GROUP] tile per kv-head, channel-
//                              major; dequant (q*scale_abs[d]+zp_abs[d])*s_col[c]).
//   tile_id == n_full_blocks → the trailing partial block: tokens come from the
//                              f32 recent-window (slot = token - n_full*GROUP).
//
// Phase B/C/D (softmax + Q8 V accumulation) and the partials layout are
// byte-identical to attention_flash_q8_0_tile_batched, so this shares
// attention_flash_asym_reduce_batched. Specialized for head_dim=256 (8 dims ×
// 32 threads), like the rest of this flash family.
//
// Record layout (== kvarn_quantize_tile / pack_kvarn_tile), GROUP cols:
//   [ (head_dim*GROUP+1)/2 ]  4-bit packed q (row-major, channel-major rows)
//   [ head_dim*2 ]            scale_abs[d] fp16   (per channel)
//   [ head_dim*2 ]            zp_abs[d]    fp16
//   [ GROUP*2   ]             s_col[c]     fp16   (per token-in-block)
//
// Grid: [n_heads, max_tiles, sub_batch_size]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_kvarn_tile_batched(
    const float* __restrict__ q,               // [sub_batch × n_heads × head_dim]
    const uint8_t* __restrict__ records,       // block-tiled KVarN K records
    const float* __restrict__ window,          // [GROUP × kv_dim] f32 recent window
    const unsigned char* __restrict__ v_cache, // Q8_0
    float* __restrict__ partials,
    const int* __restrict__ positions,         // [total_batch]
    const float* __restrict__ tree_bias,       // optional [total_batch × block_cols]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,                             // == GROUP (128)
    int max_tiles,
    int batch_offset,
    int block_start,                           // ignored when tree_bias == nullptr
    int block_cols,                            // ignored when tree_bias == nullptr
    int n_full_blocks,                         // record tiles before the window tile
    int rec_bytes,                             // kvarn_record_bytes_bits(head_dim, GROUP, bits)
    int bits                                   // K code bits in {2,4,8}; 8/bits codes/byte
) {
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    // dims-per-thread: 32-thread block covers head_dim (128 -> 4, 256 -> 8). This
    // (not a hardcoded 8) is what makes the kernel correct for head_dim != 256.
    const int dpt = head_dim / 32;
    const int cpb = 8 / bits;                  // K codes per byte
    const int codemask = (1 << bits) - 1;

    const int global_bid = batch_offset + local_bid;
    const bool tree_mode = (tree_bias != nullptr);
    const int seq_len = tree_mode ? (block_start + block_cols) : (positions[global_bid] + 1);
    const int group = tile_size;                   // KVarN GROUP
    const int tile_start = tile_id * group;
    if (tile_start >= seq_len) return;
    const int tile_end = min(tile_start + group, seq_len);
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;

    // V is Q8_0: head_dim/32 blocks/head, 34 bytes/block.
    const int blocks_per_head = head_dim / 32;
    const int total_blocks    = n_kv_heads * blocks_per_head;
    const int kv_head_blk     = kv_h * blocks_per_head;
    const int v_row_stride    = total_blocks * 34;
    const int kv_dim          = n_kv_heads * head_dim;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q for this batch position: `dpt` consecutive dims per thread.
    const float* q_head = q + local_bid * q_dim + h * head_dim;
    const int d0 = tid * dpt;
    float mq[8];                               // dpt <= 8 (head_dim <= 256)
    for (int i = 0; i < dpt; i++) mq[i] = q_head[d0 + i];

    // ── Phase A: Q·K with inline KVarN dequant (records) or f32 window. ──
    const bool is_window = (tile_id >= n_full_blocks);
    if (!is_window) {
        // Record tile `tile_id`, kv-head `kv_h`. Channel-major [head_dim × GROUP].
        const uint8_t* rec = records + (size_t)(tile_id * n_kv_heads + kv_h) * rec_bytes;
        const int off_scale = (head_dim * group + cpb - 1) / cpb;
        const int off_zp = off_scale + head_dim * 2;
        const int off_scol = off_zp + head_dim * 2;
        const __half* scale_abs = (const __half*)(rec + off_scale);
        const __half* zp_abs = (const __half*)(rec + off_zp);
        const __half* s_col = (const __half*)(rec + off_scol);
        // Per-channel scale/zp for this thread's dpt dims — load once per tile.
        float sa[8], za[8];
        for (int i = 0; i < dpt; i++) {
            sa[i] = __half2float(scale_abs[d0 + i]);
            za[i] = __half2float(zp_abs[d0 + i]);
        }
        for (int t_local = 0; t_local < tile_len; t_local++) {
            const int t = tile_start + t_local;
            const float sc = __half2float(s_col[t_local]);
            float partial = 0.0f;
            for (int i = 0; i < dpt; i++) {
                const int idx = (d0 + i) * group + t_local;    // channel-major flat index
                const uint8_t byte = rec[idx / cpb];
                const float qd = (float)((byte >> ((idx % cpb) * bits)) & codemask);
                const float k = (qd * sa[i] + za[i]) * sc;
                partial += mq[i] * k;
            }
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor(partial, off);
            if (tid == 0) {
                float s = partial * scale_attn;
                if (tree_mode && t >= block_start)
                    s += tree_bias[global_bid * block_cols + (t - block_start)];
                scores[t_local] = s;
            }
        }
    } else {
        // Trailing partial block: f32 recent window, slot = token - n_full*GROUP.
        const int win_base = n_full_blocks * group;            // == tile_start
        for (int t_local = 0; t_local < tile_len; t_local++) {
            const int t = tile_start + t_local;
            const float* kt = window + (size_t)(t - win_base) * kv_dim + kv_h * head_dim;
            float partial = 0.0f;
            for (int i = 0; i < dpt; i++) {
                partial += mq[i] * kt[d0 + i];
            }
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor(partial, off);
            if (tid == 0) {
                float s = partial * scale_attn;
                if (tree_mode && t >= block_start)
                    s += tree_bias[global_bid * block_cols + (t - block_start)];
                scores[t_local] = s;
            }
        }
    }
    __syncthreads();

    // ── Phase B: tile max ──
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    // ── Phase C: exp + sum ──
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

    // ── Phase D: V-weighted accumulation (Q8_0) — identical to the Q8 tile. ──
    float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const unsigned char* vb = v_cache + (size_t)t * v_row_stride;
        for (int i = 0; i < dpt; i++) {
            const int d = d0 + i;
            const int bi = d / 32;
            const int bj = d % 32;
            const unsigned char* vbh = vb + (kv_head_blk + bi) * 34;
            float vs = (float)*((const _Float16*)vbh);
            out_vec[i] += w * (vs * (float)((signed char)vbh[2 + bj]));
        }
    }

    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                        + tile_id * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    for (int i = 0; i < dpt; i++)
        p[2 + d0 + i] = out_vec[i];
}
