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
    // R12c2_topk_exact_nolds_lastskip: keep only the small hot-path state in
    // LDS. The n_exp>256 fallback rescans probs_in and skips already-picked
    // indices instead of staging all MAX_NEXP values into shared memory.
    __shared__ int   picked_idx[K_TOP];
    __shared__ float picked_w[K_TOP];
    __shared__ float warp_v[8];
    __shared__ int   warp_i[8];

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    if (n_exp == BLOCK_SIZE && norm_topk) {
        // R20c1_topk_finalskip: keep the exact-256 register/vector-output
        // fast path, but avoid publishing the final winner through LDS. Only
        // iterations that feed the next invalidation pass need warp_i[0].
        float cur_v = probs_in[tid];
        int cur_i = tid;
        float raw0 = 0.0f, raw1 = 0.0f, raw2 = 0.0f, raw3 = 0.0f;
        float raw4 = 0.0f, raw5 = 0.0f, raw6 = 0.0f, raw7 = 0.0f;
        int idx0 = 0, idx1 = 0, idx2 = 0, idx3 = 0;
        int idx4 = 0, idx5 = 0, idx6 = 0, idx7 = 0;
        float sum = 0.0f;

        // R79c2_topk_lane0final8: exact same unrolled pick/invalidation order
        // as R78c2, but replace the second-stage warp0 shuffle reduction over
        // eight warp winners with a lane0 serial compare of warp_v[0..7].
        // This trades five shuffles for eight LDS reads in the tiny router
        // kernel while preserving strict "first larger wins" tie behavior.
        #define PICK_EXACT(dst_idx, dst_raw, publish_next) do { \
            float best_v = cur_v; \
            int best_i = cur_i; \
            r4_topk_pair_reduce(best_v, best_i); \
            if (lane_id == 0) { \
                warp_v[warp_id] = best_v; \
                warp_i[warp_id] = best_i; \
            } \
            __syncthreads(); \
            if (tid == 0) { \
                float final_v = warp_v[0]; \
                int final_i = warp_i[0]; \
                for (int wi = 1; wi < 8; wi++) { \
                    float ov = warp_v[wi]; \
                    int oi = warp_i[wi]; \
                    if (ov > final_v) { \
                        final_v = ov; \
                        final_i = oi; \
                    } \
                } \
                if (publish_next) warp_i[0] = final_i; \
                sum += final_v; \
                dst_idx = final_i; \
                dst_raw = final_v; \
            } \
            if (publish_next) { \
                __syncthreads(); \
                const int winner = warp_i[0]; \
                if (cur_i == winner) { \
                    cur_v = -INFINITY; \
                    cur_i = -1; \
                } \
                /* REQUIRED. `warp_i[0]` is read here by every thread as this \
                   iteration's winner, and rewritten by warp 0 lane 0 at the TOP \
                   of the next iteration (`warp_i[warp_id]`) with no barrier in \
                   between. Without this, a fast warp 0 clobbers the winner \
                   before a slow warp has invalidated against it; that warp \
                   keeps the picked expert live and it is selected AGAIN, so the \
                   top-8 comes back with a DUPLICATE and a distinct expert \
                   dropped. Measured 1 occurrence per ~46k router calls before \
                   this barrier, always from bit-identical input logits. */ \
                __syncthreads(); \
            } \
        } while (0)

        PICK_EXACT(idx0, raw0, true);
        PICK_EXACT(idx1, raw1, true);
        PICK_EXACT(idx2, raw2, true);
        PICK_EXACT(idx3, raw3, true);
        PICK_EXACT(idx4, raw4, true);
        PICK_EXACT(idx5, raw5, true);
        PICK_EXACT(idx6, raw6, true);
        PICK_EXACT(idx7, raw7, false);
        #undef PICK_EXACT

        if (tid == 0) {
            *reinterpret_cast<int4*>(topk_idx) =
                make_int4(idx0, idx1, idx2, idx3);
            *reinterpret_cast<int4*>(topk_idx + 4) =
                make_int4(idx4, idx5, idx6, idx7);
            *reinterpret_cast<float4*>(topk_w) =
                make_float4(raw0 / sum, raw1 / sum, raw2 / sum, raw3 / sum);
            *reinterpret_cast<float4*>(topk_w + 4) =
                make_float4(raw4 / sum, raw5 / sum, raw6 / sum, raw7 / sum);
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
                    if (k + 1 < K_TOP) warp_i[0] = final_i;
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
            if (k + 1 < K_TOP) {
                __syncthreads();
                const int winner = warp_i[0];
                if (cur_i == winner) {
                    cur_v = -INFINITY;
                    cur_i = -1;
                }
                // REQUIRED, same hazard as the exact-256 path above: the next
                // iteration's `warp_i[warp_id]` store races this read of
                // `warp_i[0]`, and losing it yields a duplicated expert.
                __syncthreads();
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

    for (int k = 0; k < K_TOP; k++) {
        float best_v = -INFINITY;
        int   best_i = -1;
        for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
            bool already_picked = false;
            for (int j = 0; j < k; j++) {
                if (picked_idx[j] == i) already_picked = true;
            }
            float v = already_picked ? -INFINITY : probs_in[i];
            if (v > best_v) { best_v = v; best_i = i; }
        }
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
                picked_idx[k] = final_i;
                picked_w[k]   = final_v;
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
