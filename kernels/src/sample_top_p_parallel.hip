// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Multi-workgroup parallel rewrite of `sample_top_p` (see sample_top_p.hip).
//
// The single-block kernel runs the entire 150K-vocab top-K scan on ONE
// workgroup (one CU) while the other ~95 CUs idle — measured at ~305 us/token
// on gfx1100 A3B decode (the largest isolated decode sink). This splits the
// vocab scan across N_BLOCKS workgroups, each producing a block-local top-20,
// then a single finalize block merges the N_BLOCKS×20 candidates and runs the
// IDENTICAL softmax / sort / top-p / RNG phase as the original.
//
// Correctness: the global top-20 is a subset of the union of the per-block
// top-20s (a global top-20 element lives in some block's strip and is therefore
// in that strip's top-20), so the merged result is the EXACT same top-20 as the
// single-block kernel for distinct logit values — hence a byte-identical token.
// Ties on exactly-equal f32 logits may pick a different representative index,
// which is semantically equivalent (equal model preference).
//
// Phase 0 (repeat penalty) modifies logits in place and must complete before
// any block reads logits, so it is a separate prepass kernel launched only when
// a penalty is active.

#define TOP_K 20

// ── Phase 0: repeat / presence / frequency penalty, in place (1 block) ──
// Byte-identical to sample_top_p.hip Phase 0.
extern "C" __global__ void sample_apply_repeat_penalty(
    float* __restrict__ logits,
    const unsigned int* __restrict__ repeat_tokens,
    int vocab_size,
    int repeat_window,
    float repeat_penalty,
    float presence_penalty,
    float frequency_penalty
) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    if (repeat_window <= 0) return;
    float window_len = (float)repeat_window;
    for (int i = tid; i < repeat_window; i += nthreads) {
        unsigned int tok = repeat_tokens[i];
        if ((int)tok >= vocab_size) continue;
        bool has_later = false;
        int count = 1;
        for (int j = 0; j < repeat_window; j++) {
            if (j == i) continue;
            if (repeat_tokens[j] == tok) {
                if (j > i) { has_later = true; break; }
                count++;
            }
        }
        if (has_later) continue;
        if (repeat_penalty > 1.0f) {
            float recency = ((float)i + 1.0f) / window_len;
            float effective = powf(repeat_penalty, (float)count * recency);
            if (effective > 1.5f) effective = 1.5f;
            float v = logits[tok];
            logits[tok] = (v > 0.0f) ? (v / effective) : (v * effective);
        }
        if (presence_penalty > 0.0f || frequency_penalty > 0.0f) {
            logits[tok] -= frequency_penalty * (float)count + presence_penalty;
        }
    }
}

// ── shared block-local top-K reduction ──
// Each thread keeps a private top-K over `vals[i]` for i in [lo,hi) (strided),
// carrying an explicit index `idx_of(i)`; then a tree reduction over the block
// produces the block top-K in smem[0..TOP_K]. Writes the block top-K to
// out_val/out_idx (TOP_K each). Caller provides smem of
// nthreads*TOP_K*(4+4) bytes.
__device__ __forceinline__ void block_topk_strided(
    char* smem,                          // dynamic LDS: nthreads*TOP_K*(4+4) bytes
    const float* __restrict__ vals,
    const int*   __restrict__ idx_src,   // null → index = element position
    int lo, int hi,
    float* out_val, int* out_idx
) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    float* topk_val = (float*)smem;
    int*   topk_idx = (int*)(topk_val + nthreads * TOP_K);

    float my_vals[TOP_K];
    int   my_idx[TOP_K];
    #pragma unroll
    for (int k = 0; k < TOP_K; k++) { my_vals[k] = -1e30f; my_idx[k] = 0; }
    int   min_pos = 0;
    float min_val = -1e30f;
    for (int i = lo + tid; i < hi; i += nthreads) {
        float v = vals[i];
        if (v > min_val) {
            my_vals[min_pos] = v;
            my_idx[min_pos]  = idx_src ? idx_src[i] : i;
            min_val = my_vals[0]; min_pos = 0;
            #pragma unroll
            for (int k = 1; k < TOP_K; k++) {
                if (my_vals[k] < min_val) { min_val = my_vals[k]; min_pos = k; }
            }
        }
    }
    #pragma unroll
    for (int k = 0; k < TOP_K; k++) {
        topk_val[tid * TOP_K + k] = my_vals[k];
        topk_idx[tid * TOP_K + k] = my_idx[k];
    }
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float* a_val = topk_val + tid * TOP_K;
            int*   a_idx = topk_idx + tid * TOP_K;
            float* b_val = topk_val + (tid + stride) * TOP_K;
            int*   b_idx = topk_idx + (tid + stride) * TOP_K;
            #pragma unroll
            for (int j = 0; j < TOP_K; j++) {
                float bv = b_val[j]; int bi = b_idx[j];
                float amin = a_val[0]; int amin_pos = 0;
                #pragma unroll
                for (int k = 1; k < TOP_K; k++) {
                    if (a_val[k] < amin) { amin = a_val[k]; amin_pos = k; }
                }
                if (bv > amin) { a_val[amin_pos] = bv; a_idx[amin_pos] = bi; }
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) { out_val[k] = topk_val[k]; out_idx[k] = topk_idx[k]; }
    }
}

// ── Phase 1: per-block partial top-K over a contiguous vocab strip ──
extern "C" __global__ void sample_topk_partial(
    const float* __restrict__ logits,
    int vocab_size,
    int n_blocks,
    float* __restrict__ partial_val,   // [n_blocks * TOP_K]
    int*   __restrict__ partial_idx    // [n_blocks * TOP_K]
) {
    extern __shared__ char smem[];
    const int blk = blockIdx.x;
    if (blk >= n_blocks) return;
    long long lo = (long long)blk * vocab_size / n_blocks;
    long long hi = (long long)(blk + 1) * vocab_size / n_blocks;
    block_topk_strided(smem, logits, nullptr, (int)lo, (int)hi,
                       partial_val + blk * TOP_K, partial_idx + blk * TOP_K);
}

// ── Phase 2+3: merge partials → global top-K, then softmax/sort/top-p/RNG ──
// The merge produces the same global top-K as the single-block kernel; the
// sampling tail below is copied verbatim from sample_top_p.hip Phase 3.
extern "C" __global__ void sample_topk_finalize(
    const float* __restrict__ partial_val,  // [n_cand]
    const int*   __restrict__ partial_idx,   // [n_cand]
    int n_cand,                              // n_blocks * TOP_K
    unsigned int* __restrict__ result,       // [2]: token_id, rng_state
    float temperature,
    float top_p,
    unsigned int rand_seed
) {
    extern __shared__ char smem[];
    const int tid = threadIdx.x;

    // Block-merge the candidate pool into a single top-K in smem.
    float merged_val[TOP_K]; int merged_idx[TOP_K];
    block_topk_strided(smem, partial_val, partial_idx, 0, n_cand, merged_val, merged_idx);

    if (tid != 0) return;

    // ── Phase 3 (verbatim from sample_top_p.hip) ──
    float vals[TOP_K]; int idx[TOP_K];
    #pragma unroll
    for (int k = 0; k < TOP_K; k++) { vals[k] = merged_val[k]; idx[k] = merged_idx[k]; }

    float max_val = vals[0];
    int   max_pos = 0;
    for (int k = 1; k < TOP_K; k++) { if (vals[k] > max_val) { max_val = vals[k]; max_pos = k; } }
    if (max_val <= -1e29f) { result[0] = 0; result[1] = rand_seed; return; }

    if (temperature <= 1e-6f) {
        result[0] = (unsigned int)idx[max_pos];
        result[1] = rand_seed;
        return;
    }

    float inv_temp = 1.0f / temperature;
    float sum = 0.0f;
    float probs[TOP_K];
    #pragma unroll
    for (int k = 0; k < TOP_K; k++) {
        if (vals[k] <= -1e29f) { probs[k] = 0.0f; continue; }
        float p = expf((vals[k] - max_val) * inv_temp);
        probs[k] = p;
        sum += p;
    }
    if (sum <= 0.0f) { result[0] = (unsigned int)idx[0]; result[1] = rand_seed; return; }

    int order[TOP_K];
    #pragma unroll
    for (int k = 0; k < TOP_K; k++) order[k] = k;
    for (int i = 1; i < TOP_K; i++) {
        int cur = order[i];
        float cv = probs[cur];
        int j = i;
        while (j > 0 && probs[order[j - 1]] < cv) { order[j] = order[j - 1]; j--; }
        order[j] = cur;
    }

    unsigned int rng = rand_seed;
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;

    float p_thresh = top_p * sum;
    float trunc_sum = 0.0f;
    int trunc_k = TOP_K;
    for (int i = 0; i < TOP_K; i++) {
        trunc_sum += probs[order[i]];
        if (trunc_sum >= p_thresh) { trunc_k = i + 1; break; }
    }

    float r = (float)rng / 4294967295.0f * trunc_sum;
    float cum = 0.0f;
    for (int i = 0; i < trunc_k; i++) {
        cum += probs[order[i]];
        if (cum >= r) { result[0] = (unsigned int)idx[order[i]]; result[1] = rng; return; }
    }
    result[0] = (unsigned int)idx[order[trunc_k - 1]];
    result[1] = rng;
}
