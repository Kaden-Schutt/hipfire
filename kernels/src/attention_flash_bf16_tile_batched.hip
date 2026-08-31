// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batched flash attention tile for FLAT BF16 KV, with a sliding-window mask.
// Prefill sibling of `attention_flash_bf16_tile.hip`, and a direct port of
// `attention_flash_q8_0_tile_batched.hip` with the Q8_0 block addressing
// replaced by a flat `[pos][kv_head][d]` bf16 read.
//
// Processes sub_batch_size positions per launch via blockIdx.z. Partials
// layout: [sub_batch × n_heads × max_tiles × (2+head_dim)] — identical to the
// asym3/Q8 tiles, so it shares `attention_flash_asym_reduce_batched`
// unmodified (that reduce only ever touches f32 partials).
//
// LDS is tile_size*4 bytes regardless of seq_len, so there is NO context cap.
//
// cos_theta/sin_theta and v_mode_bits are accepted for ABI compatibility with
// the shared `launch_asym_flash_batched` dispatcher but ignored — bf16 K is
// unrotated and there is no separate V tier.
//
// Grid: [n_heads, max_tiles, sub_batch_size]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>
#include "kv_slot_desc.h"

static __device__ __forceinline__ float hipfire_bf16_to_f32(unsigned short b) {
    return __builtin_bit_cast(float, ((unsigned int)b) << 16);
}

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_bf16_tile_batched(
    const float* __restrict__ q,               // [sub_batch × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,         // [total_batch]
    const float* __restrict__ /*cos_theta*/,   // unused (bf16 K unrotated)
    const float* __restrict__ /*sin_theta*/,   // unused
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
    int /*v_mode_bits*/,                       // consumed-but-unused (shared launcher)
    int window,                                // sliding-window span; <= 0 = full causal
    const KvSlotDesc* __restrict__ slot_descs, // [n_slots] or nullptr = legacy
    const int* __restrict__ row_slot           // [total_batch] or nullptr = legacy
) {
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const int global_bid = batch_offset + local_bid;
    const bool tree_mode = (tree_bias != nullptr);
    // hipGraph-safe block_start: under captured tree-verify replay the scalar
    // kernarg is baked at capture time and goes stale as the committed prefix
    // grows. positions[] is re-uploaded every cycle and the linearized tree
    // root sits at positions[0], so derive it from the device buffer instead.
    const int eff_block_start = tree_mode ? positions[0] : block_start;
    const int seq_len = tree_mode ? (eff_block_start + block_cols)
                                  : (positions[global_bid] + 1);

    // row_slot is indexed by GLOBAL row — using local blockIdx.z alone reads
    // the wrong slot once the partials buffer forces the launcher to sub-batch
    // (correct at small batch, silent cross-slot corruption after chunking).
    const int slot = (row_slot != nullptr) ? row_slot[global_bid] : 0;
    const KvSlotDesc desc = (slot_descs != nullptr)
        ? slot_descs[slot]
        : kv_slot_legacy(seq_len, max_seq);

    const int kv_dim = n_kv_heads * head_dim;
    const int per_pos_bytes = kv_dim * 2;

    // `positions[row] + 1` is the PER-ROW causal bound; `desc.seq_len` is the
    // slot's logical KV length (capacity metadata). They coincide only at
    // M=1. The descriptor supplies the slab BASE, never the row's own bound —
    // and the reduce independently derives n_tiles from positions[]+1, so any
    // disagreement here makes it fold tiles this kernel never wrote.
    const int eff_seq_len = seq_len;

    const int win_lo = (window > 0) ? (eff_seq_len - window) : 0;
    const int tile_start = tile_id * tile_size;
    if (tile_start >= eff_seq_len) return;
    const int tile_end = min(tile_start + tile_size, eff_seq_len);
    const int tile_len = tile_end - tile_start;

    // Sliding-window fast path: a tile entirely below win_lo contributes
    // nothing. Write an empty partial (tile_sum=0 → the reduce skips it via
    // its `p[1] > 0` guard) and return BEFORE the Q load and Phases A-D.
    // Phase D loads V for every position even at weight 0, so masked-but-
    // iterated tiles are the dominant per-query cost on sliding layers at long
    // context. Only fires when window>0; window<=0 is byte-identical.
    if (window > 0 && tile_end <= win_lo) {
        if (tid == 0) {
            float* pe = partials
                + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                + tile_id * (2 + head_dim);
            pe[0] = -1e30f;
            pe[1] = 0.0f;
        }
        return;
    }

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;
    const int head_base = kv_h * head_dim;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q for this batch position: dpt consecutive dims per thread (matches
    // the asym3/Q8 tiles so the dim-indexed partials/reduce layout is
    // identical). The block is always 32 lanes, so the __shfl_xor reductions
    // still sum over all head_dim dims; only the per-thread span scales.
    const float* q_head = q + local_bid * q_dim + h * head_dim;
    const int dpt = head_dim / 32;
    const int d0 = tid * dpt;
    float mq[16];
    for (int i = 0; i < dpt; i++) mq[i] = q_head[d0 + i];

    // ── Phase A: Q·K ──
    // Flat bf16 needs no block/scale decomposition, so unlike the Q8 kernel
    // there is no `lane_one_q8_block` fast path to select — the general form
    // IS the fast form here.
    for (int t_local = 0; t_local < tile_len; t_local++) {
        const int t = tile_start + t_local;
        if (t < win_lo) {                  // outside sliding window → mask, skip dot
            if (tid == 0) scores[t_local] = -INFINITY;
            continue;                      // t is wave-uniform; all lanes skip together
        }
        const unsigned short* kb = (const unsigned short*)
            (k_cache + kv_offset_for_k(desc, t, per_pos_bytes)) + head_base + d0;
        float partial = 0.0f;
        for (int i = 0; i < dpt; i++) {
            partial += mq[i] * hipfire_bf16_to_f32(kb[i]);
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0) {
            float s = partial * scale_attn;
            if (tree_mode && t >= eff_block_start)
                s += tree_bias[global_bid * block_cols + (t - eff_block_start)];
            scores[t_local] = s;
        }
    }
    __syncthreads();

    // ── Phase B: tile max ──
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    const float tile_max = local_max;

    // ── Phase C: exp + sum ──
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        const float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    const float tile_sum = local_sum;
    __syncthreads();

    // ── Phase D: V-weighted accumulation ──
    float out_vec[16] = { 0.0f };
    for (int t_local = 0; t_local < tile_len; t_local++) {
        const float w = scores[t_local];
        const int t = tile_start + t_local;
        const unsigned short* vb = (const unsigned short*)
            (v_cache + kv_offset_for_v(desc, t, per_pos_bytes)) + head_base + d0;
        for (int i = 0; i < dpt; i++) {
            out_vec[i] += w * hipfire_bf16_to_f32(vb[i]);
        }
    }

    float* p = partials
        + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
        + tile_id * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    for (int i = 0; i < dpt; i++)
        p[2 + d0 + i] = out_vec[i];
}
