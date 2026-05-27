// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// gfx906 (Vega 20 / GCN5) dp4a flash attention for Q8_0 KV, long context.
//
// The scalar tile_batched kernel is instruction-issue-bound on the Q·K
// dequant-dot (8 waves/SIMD, 0 spills, <1% peak BW — not memory-bound).
// gfx906 has NO matrix cores (MFMA starts at gfx908) but DOES have dp4a:
// v_dot4_i32_i8 (`__builtin_amdgcn_sdot4`), 4 int8 MACs per instruction.
//
// Per the gfx906 dot4/dot8 study (skyne98 wiki, 2026-02-21):
//   - sdot4 dependency-chained: ~22 TOPS; with 4 independent accumulators
//     (ILP4): ~44 TOPS — 2×. So we MUST round-robin ≥4 accumulators.
//   - accumulator is 32-bit, can overflow; we use clamp=false (i8·i8 sums
//     over head_dim=256 stay well within i32: 256·127·127 ≈ 4.1M ≪ 2^31).
//
// Math: K is Q8_0 (per 32-block: fp16 scale + 32 i8). We ALSO quantize Q
// per-32-block to i8 (per-block scale = max|q|/127), ONCE per query head,
// reused across all tokens (amortized). Per-block dot =
//   q_scale · k_scale · Σ_32 (q_i8 · k_i8)   ← the Σ is 8 sdot4 ops.
// head_dim=256 → 8 blocks → 64 sdot4 ops, round-robined over 4 i32 accs.
//
// Phase D (V accumulation) stays scalar f32 — scores aren't int, dp4a
// doesn't apply. Q·K is the dominant issue cost, so accelerating it is
// the lever.
//
// Same tiled online-softmax skeleton + partials layout as
// attention_flash_q8_0_tile_batched → reuses attention_flash_asym_reduce_batched.
// Block [32,1,1] (wave32-shaped reduction; gfx906 runs it in a wave64 slot
// — acceptable, the win is the dot, not the lane count). cos/sin ignored.
#include <hip/hip_runtime.h>

#ifndef DP4A_BLOCKS_MAX
#define DP4A_BLOCKS_MAX 8        // head_dim/32, max 256-dim head
#endif

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_q8_0_dp4a(
    const float* __restrict__ q,               // [sub_batch × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,         // [total_batch]
    const float* __restrict__ /*cos_theta*/,   // unused (Q8 K unrotated)
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
    int block_start,
    int block_cols
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

    const int blocks_per_head = head_dim / 32;          // 8 for hd=256
    const int total_blocks    = n_kv_heads * blocks_per_head;
    const int kv_head_blk     = kv_h * blocks_per_head;
    const int row_stride      = total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // ── Quantize this thread's slice of Q to i8, once. Each of 32 lanes
    //    owns 8 consecutive dims (d0 = tid*8) = part of one 32-block.
    //    We instead need WHOLE 32-blocks packed for sdot4, so each lane
    //    handles its 8 dims as 2 packed i32 (4 i8 each), plus the block's
    //    f32 scale (one per 32-block, shared by 4 lanes). Compute the
    //    per-block q-scale cooperatively. ──
    const int d0 = tid * 8;                  // this lane's 8 dims
    const int my_block = d0 / 32;            // which 32-block (0..blocks-1)
    const float* q_head = q + local_bid * q_dim + h * head_dim;

    float qf[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) qf[i] = q_head[d0 + i];

    // Per-32-block max|q| → scale. 4 lanes share a block; reduce their
    // 8-dim maxes across the 4 lanes (lanes my_block*4 .. +3 within the
    // wave have lane = my_block*4 + (0..3) since d0/32 groups 4 lanes).
    float my_absmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) my_absmax = fmaxf(my_absmax, fabsf(qf[i]));
    // 4-lane reduction: lanes within the same block are tid&~3 .. |3.
    float bmax = my_absmax;
    bmax = fmaxf(bmax, __shfl_xor(bmax, 1));
    bmax = fmaxf(bmax, __shfl_xor(bmax, 2));   // now all 4 lanes in block agree
    const float q_scale = (bmax > 0.0f) ? (bmax / 127.0f) : 1.0f;
    const float inv_q_scale = 1.0f / q_scale;

    // Pack this lane's 8 i8 into 2 i32 (q_lo = dims 0..3, q_hi = dims 4..7).
    int q_lo = 0, q_hi = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int v = (int)rintf(qf[i] * inv_q_scale);
        v = max(-127, min(127, v));
        q_lo |= (v & 0xFF) << (i * 8);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int v = (int)rintf(qf[4 + i] * inv_q_scale);
        v = max(-127, min(127, v));
        q_hi |= (v & 0xFF) << (i * 8);
    }

    // ── Phase A: Q·K via sdot4. Each lane computes the partial dot for
    //    its 8 dims (2 sdot4 ops, 2 independent accs for ILP), then the
    //    wave reduces across the 32 lanes (= full head_dim). The per-block
    //    q_scale·k_scale folds in after the integer dot. ──
    for (int t_local = 0; t_local < tile_len; t_local++) {
        const int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * row_stride + (kv_head_blk + my_block) * 34;
        const float k_scale = (float)*((const _Float16*)kb);
        // K i8 codes for this lane's 8 dims (offset within the 32-block).
        const signed char* kc = (const signed char*)(kb + 2) + (d0 % 32);
        int k_lo = (kc[0] & 0xFF) | ((kc[1] & 0xFF) << 8) | ((kc[2] & 0xFF) << 16) | ((kc[3] & 0xFF) << 24);
        int k_hi = (kc[4] & 0xFF) | ((kc[5] & 0xFF) << 8) | ((kc[6] & 0xFF) << 16) | ((kc[7] & 0xFF) << 24);
        // Two independent sdot4 accumulators (ILP2 within the lane).
        int acc0 = __builtin_amdgcn_sdot4(q_lo, k_lo, 0, false);
        int acc1 = __builtin_amdgcn_sdot4(q_hi, k_hi, 0, false);
        // Lane partial in real units: (acc0+acc1) is the i8·i8 sum over 8
        // dims; scale by q_scale·k_scale for THIS lane's block.
        float lane_partial = (float)(acc0 + acc1) * q_scale * k_scale;
        // Wave reduce across all 32 lanes → full head_dim dot.
        for (int off = 16; off > 0; off >>= 1)
            lane_partial += __shfl_xor(lane_partial, off);
        if (tid == 0) {
            float s = lane_partial * scale_attn;
            if (tree_mode && t >= block_start)
                s += tree_bias[global_bid * block_cols + (t - block_start)];
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

    // Phase D: V-weighted accumulation (Q8_0, scalar — scores are f32).
    float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const unsigned char* vb = v_cache + (size_t)t * row_stride;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
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
    #pragma unroll
    for (int i = 0; i < 8; i++)
        p[2 + d0 + i] = out_vec[i];
}
