// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// MoE router top-8 + (optional) renormalize, given PRE-SOFTMAXED probs.
// Companion to gpu.softmax_f32(router_logits); the dispatch site runs that
// regular softmax kernel first, then this kernel for top-K + renorm. This
// matches the CPU fallback path's softmax math exactly and avoids the
// 1-ULP precision divergence seen when softmax + topk + renorm were all
// fused into one kernel.
//
// Reads [n_exp] PROBS in [0, 1] summing to 1, writes [8] i32 indices and
// [8] f32 weights to device buffers. One workgroup, 256 threads. Supports
// n_exp up to 1024. For n_exp=256 (A3B/A10B) the loops degenerate to one
// slot per thread.

#define BLOCK_SIZE 256
#define MAX_NEXP   1024
#define K_TOP      8

__device__ __forceinline__ void r4_topk_pair_reduce(float& v, int& idx) {
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down(v, off);
        int oi = __shfl_down(idx, off);
        if (ov > v) {
            v = ov;
            idx = oi;
        }
    }
}

extern "C" __launch_bounds__(BLOCK_SIZE, 2)
__global__ void moe_topk_renorm_k8(
    const float* __restrict__ probs_in,    // [n_exp] pre-softmaxed
    int*         __restrict__ topk_idx,    // [8]
    float*       __restrict__ topk_w,      // [8]
    int n_exp,
    int norm_topk
) {
    __shared__ float smem_v[MAX_NEXP];
    __shared__ int   smem_i[MAX_NEXP];
    __shared__ float reduce_v[BLOCK_SIZE];
    __shared__ int   reduce_i[BLOCK_SIZE];
    __shared__ int   picked_idx[K_TOP];
    __shared__ float picked_w[K_TOP];
    __shared__ float warp_v[8];
    __shared__ int   warp_i[8];

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    if (n_exp == BLOCK_SIZE && norm_topk) {
        // R10c2_topk_normexact: A3B routes over exactly 256 experts and
        // renormalizes the top-k weights. Keep the R9 exact-256 register
        // fast path, then remove the hot-path norm_topk/s>0 predicates.
        float cur_v = probs_in[tid];
        int cur_i = tid;

        for (int k = 0; k < K_TOP; k++) {
            float best_v = cur_v;
            int best_i = cur_i;
            r4_topk_pair_reduce(best_v, best_i);
            if (lane_id == 0) {
                warp_v[warp_id] = best_v;
                warp_i[warp_id] = best_i;
            }
            __syncthreads();

            float final_v = (warp_id == 0 && lane_id < 8) ? warp_v[lane_id] : -INFINITY;
            int final_i = (warp_id == 0 && lane_id < 8) ? warp_i[lane_id] : -1;
            if (warp_id == 0) {
                r4_topk_pair_reduce(final_v, final_i);
                if (lane_id == 0) {
                    warp_i[0] = final_i;
                    warp_v[0] = final_v;
                    topk_idx[k] = final_i;
                    topk_w[k] = final_v;
                }
            }
            __syncthreads();

            const int winner = warp_i[0];
            if (cur_i == winner) {
                cur_v = -INFINITY;
                cur_i = -1;
            }
        }

        if (tid == 0) {
            float s = 0.0f;
            for (int k = 0; k < K_TOP; k++) s += topk_w[k];
            for (int k = 0; k < K_TOP; k++) topk_w[k] /= s;
        }
        return;
    }

    if (n_exp <= BLOCK_SIZE) {
        float cur_v = (tid < n_exp) ? probs_in[tid] : -INFINITY;
        int cur_i = (tid < n_exp) ? tid : -1;

        for (int k = 0; k < K_TOP; k++) {
            float best_v = cur_v;
            int best_i = cur_i;
            r4_topk_pair_reduce(best_v, best_i);
            if (lane_id == 0) {
                warp_v[warp_id] = best_v;
                warp_i[warp_id] = best_i;
            }
            __syncthreads();

            float final_v = (warp_id == 0 && lane_id < 8) ? warp_v[lane_id] : -INFINITY;
            int final_i = (warp_id == 0 && lane_id < 8) ? warp_i[lane_id] : -1;
            if (warp_id == 0) {
                r4_topk_pair_reduce(final_v, final_i);
                if (lane_id == 0) {
                    warp_i[0] = final_i;
                    warp_v[0] = final_v;
                    int safe_idx = final_i;
                    float safe_w = final_v;
                    if (safe_idx < 0 || safe_idx >= n_exp) {
                        safe_idx = 0;
                        safe_w = 0.0f;
                    }
                    topk_idx[k] = safe_idx;
                    topk_w[k] = safe_w;
                }
            }
            __syncthreads();

            const int winner = warp_i[0];
            if (cur_i == winner) {
                cur_v = -INFINITY;
                cur_i = -1;
            }
        }

        // R7c2_topk_regout: write each picked expert directly from lane 0,
        // then renormalize the tiny global topk_w buffer in place. This keeps
        // the R6 fast path's barrier structure but removes the picked_idx /
        // picked_w shared-memory round trip from the hot n_exp<=256 case.
        if (tid == 0) {
            float s = 0.0f;
            for (int k = 0; k < K_TOP; k++) s += topk_w[k];
            if (norm_topk && s > 0.0f) {
                for (int k = 0; k < K_TOP; k++) topk_w[k] /= s;
            }
        }
        return;
    }

    for (int i = tid; i < MAX_NEXP; i += BLOCK_SIZE) {
        smem_v[i] = (i < n_exp) ? probs_in[i] : -INFINITY;
        smem_i[i] = i;
    }
    __syncthreads();

    for (int k = 0; k < K_TOP; k++) {
        float best_v = -INFINITY;
        int   best_i = -1;
        for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
            float v = smem_v[i];
            if (v > best_v) { best_v = v; best_i = smem_i[i]; }
        }
        reduce_v[tid] = best_v;
        reduce_i[tid] = best_i;
        __syncthreads();

        for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                float va = reduce_v[tid];
                float vb = reduce_v[tid + s];
                if (vb > va) {
                    reduce_v[tid] = vb;
                    reduce_i[tid] = reduce_i[tid + s];
                }
            }
            __syncthreads();
        }
        const int   winner = reduce_i[0];
        const float winner_w = reduce_v[0];
        if (tid == 0) {
            picked_idx[k] = winner;
            picked_w[k]   = winner_w;
        }
        if (winner >= 0 && winner < n_exp) {
            if ((winner & (BLOCK_SIZE - 1)) == tid) {
                smem_v[winner] = -INFINITY;
            }
        }
        __syncthreads();
    }

    // Renormalize using direct division (matches CPU fallback path
    // *w /= sum exactly; mul-by-reciprocal would add 1 ULP error per
    // element which the hyper-quantized MoE layer stack amplifies into
    // a structural attractor).
    if (tid == 0) {
        float s = 0.0f;
        for (int k = 0; k < K_TOP; k++) s += picked_w[k];
        if (norm_topk && s > 0.0f) {
            for (int k = 0; k < K_TOP; k++) picked_w[k] /= s;
        }
    }
    __syncthreads();

    // Write output. Clamp invalid winner (NaN/OOB from corrupted probs)
    // to expert 0 with weight 0 — same defense as moe_softmax_topk_k8
    // (closes #152). With pre-softmaxed probs this should never fire,
    // but keep the safety net.
    if (tid < K_TOP) {
        int   safe_idx = picked_idx[tid];
        float safe_w   = picked_w[tid];
        if (safe_idx < 0 || safe_idx >= n_exp) {
            safe_idx = 0;
            safe_w   = 0.0f;
        }
        topk_idx[tid] = safe_idx;
        topk_w[tid]   = safe_w;
    }
}
