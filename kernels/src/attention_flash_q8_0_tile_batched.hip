// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched flash attention tile for Q8_0 KV (both K and V at Q8_0, 34-byte
// blocks of 32 elements: fp16 scale + 32 signed bytes). Processes
// sub_batch_size positions per launch via blockIdx.z. Partials layout:
// [sub_batch × n_heads × max_tiles × (2+head_dim)] — identical to the asym3
// tile so it shares attention_flash_asym_reduce_batched.
//
// LDS is tile_size*4 bytes regardless of seq_len, so this has NO context
// cap. It replaces the per-position attention_flash_q8_0 fallback that
// attention_q8_0_kv_batched_masked degenerated to above 15000 tokens
// (that kernel stages scores[max_ctx_len] in LDS → overflows ~64 KB at
// 16k+). See fix/q8-batched-masked-no-lds-cap.
//
// cos_theta/sin_theta are accepted for ABI compatibility with the shared
// launch_asym_flash_batched dispatcher but ignored — Q8 K is unrotated.
//
// Grid: [n_heads, max_tiles, sub_batch_size]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>

static __device__ __forceinline__ unsigned int q8_tile_load_u32(
    const unsigned char* p
) {
    unsigned int packed;
    __builtin_memcpy(&packed, p, sizeof(packed));
    return packed;
}

static __device__ __forceinline__ unsigned long long q8_tile_load_u64(
    const unsigned char* p
) {
    unsigned long long packed;
    __builtin_memcpy(&packed, p, sizeof(packed));
    return packed;
}

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_q8_0_tile_batched(
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
    int block_start,                           // ignored when tree_bias == nullptr
    int block_cols,                            // ignored when tree_bias == nullptr
    int /*v_mode_bits*/,                       // consumed-but-unused: the shared
                                               // launcher pushes V_MODE_* here for
                                               // the asym tile kernels; Q8 ignores it.
    int window                                 // sliding-window span; <= 0 = full causal
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
    // Sliding-window lower bound: a query at position seq_len-1 attends only to
    // keys in [seq_len-window, seq_len); keys at t < win_lo are masked out.
    // window <= 0 = full causal (legacy behavior, no lower bound).
    const int win_lo = (window > 0) ? (seq_len - window) : 0;
    const int tile_start = tile_id * tile_size;
    if (tile_start >= seq_len) return;
    const int tile_end = min(tile_start + tile_size, seq_len);
    const int tile_len = tile_end - tile_start;

    // Sliding-window fast path: a tile entirely below win_lo has no in-window
    // keys, so it contributes nothing to the output. Write an empty partial
    // (tile_sum=0 → the reduce skips it via its `p[1] > 0` guard) and return
    // BEFORE the Q load and Phase A-D. This matters because Phase D below loads
    // V for every position even when its weight is 0, so the masked-but-iterated
    // tiles are the dominant per-query cost on sliding layers at long context.
    // Only fires when window>0 (cohere2moe sliding layers); window<=0 (full
    // causal, all other arches + the global NoPE layers) is byte-identical.
    if (window > 0 && tile_end <= win_lo) {
        if (tid == 0) {
            float* pe = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                                 + tile_id * (2 + head_dim);
            pe[0] = -1e30f;
            pe[1] = 0.0f;
        }
        return;
    }

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;

    // Both K and V are Q8_0: head_dim/32 blocks/head, 34 bytes/block.
    const int blocks_per_head = head_dim / 32;
    const int total_blocks    = n_kv_heads * blocks_per_head;
    const int kv_head_blk     = kv_h * blocks_per_head;
    const int row_stride      = total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q for this batch position: 8 consecutive dims per thread (matches
    // asym3 tile so the dim-indexed partials/reduce layout is identical).
    const float* q_head = q + local_bid * q_dim + h * head_dim;
    // Dims-per-thread = head_dim/32 (4/8/16 for hd=128/256/512). The block is
    // always 32 lanes, so the __shfl_xor reductions below still sum over all
    // head_dim dims; only the per-thread span (d0) scales with head_dim.
    const int dpt = head_dim / 32;
    const int d0 = tid * dpt;
    // dpt is a runtime kernarg-derived trip count, so LLVM cannot prove that
    // d0=tid*dpt is block-aligned and does not LICM the Q8 address/scale loads.
    // A lane's span shares one Q8_0 block only when the head has complete
    // 32-element blocks and the span does not cross a block boundary. The
    // production hd=128/256 shapes satisfy this with dpt=4/8; retain the
    // original per-dimension addressing as the fallback for every other shape.
    const bool lane_one_q8_block = head_dim % 32 == 0 && dpt <= 32
                                && (d0 % 32) + dpt <= 32;
    float mq[16];
    for (int i = 0; i < dpt; i++) mq[i] = q_head[d0 + i];

    // Phase A: Q·K (Q8_0 dequant — 8 dims, spanning bi = d0/32, bj = d0%32).
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        if (t < win_lo) {                  // outside sliding window → mask, skip dot
            if (tid == 0) scores[t_local] = -INFINITY;
            continue;                      // t is wave-uniform, so all lanes skip together
        }
        float partial = 0.0f;
        if (lane_one_q8_block && dpt == 8) {
            const int bi = d0 / 32;
            const int bj = d0 % 32;
            const unsigned char* kb = k_cache + (size_t)t * row_stride
                                    + (kv_head_blk + bi) * 34;
            const float ks = (float)*((const _Float16*)kb);
            const unsigned long long packed = q8_tile_load_u64(kb + 2 + bj);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const signed char quant = (signed char)(packed >> (8 * i));
                partial += mq[i] * (ks * (float)quant);
            }
        } else if (lane_one_q8_block && dpt == 4) {
            const int bi = d0 / 32;
            const int bj = d0 % 32;
            const unsigned char* kb = k_cache + (size_t)t * row_stride
                                    + (kv_head_blk + bi) * 34;
            const float ks = (float)*((const _Float16*)kb);
            const unsigned int packed = q8_tile_load_u32(kb + 2 + bj);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const signed char quant = (signed char)(packed >> (8 * i));
                partial += mq[i] * (ks * (float)quant);
            }
        } else {
            for (int i = 0; i < dpt; i++) {
                const int d = d0 + i;
                const int bi = d / 32;
                const int bj = d % 32;
                const unsigned char* kb = k_cache + (size_t)t * row_stride
                                        + (kv_head_blk + bi) * 34;
                const float ks = (float)*((const _Float16*)kb);
                partial += mq[i] * (ks * (float)((signed char)kb[2 + bj]));
            }
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

    // Phase D: V-weighted accumulation (Q8_0)
    float out_vec[16] = { 0.0f };
    if (lane_one_q8_block && dpt == 8) {
        const int bi = d0 / 32;
        const int bj = d0 % 32;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            const float w = scores[t_local];
            const int t = tile_start + t_local;
            const unsigned char* vbh = v_cache + (size_t)t * row_stride
                                     + (kv_head_blk + bi) * 34;
            const float vs = (float)*((const _Float16*)vbh);
            const unsigned long long packed = q8_tile_load_u64(vbh + 2 + bj);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const signed char quant = (signed char)(packed >> (8 * i));
                out_vec[i] += w * (vs * (float)quant);
            }
        }
    } else if (lane_one_q8_block && dpt == 4) {
        const int bi = d0 / 32;
        const int bj = d0 % 32;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            const float w = scores[t_local];
            const int t = tile_start + t_local;
            const unsigned char* vbh = v_cache + (size_t)t * row_stride
                                     + (kv_head_blk + bi) * 34;
            const float vs = (float)*((const _Float16*)vbh);
            const unsigned int packed = q8_tile_load_u32(vbh + 2 + bj);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const signed char quant = (signed char)(packed >> (8 * i));
                out_vec[i] += w * (vs * (float)quant);
            }
        }
    } else {
        for (int t_local = 0; t_local < tile_len; t_local++) {
            const float w = scores[t_local];
            const int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * row_stride;
            for (int i = 0; i < dpt; i++) {
                const int d = d0 + i;
                const int bi = d / 32;
                const int bj = d % 32;
                const unsigned char* vbh = vb + (kv_head_blk + bi) * 34;
                const float vs = (float)*((const _Float16*)vbh);
                out_vec[i] += w * (vs * (float)((signed char)vbh[2 + bj]));
            }
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
