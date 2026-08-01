// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 MoE router: GPU-side bias-aware top-K + weight normalization.
//
// Replaces the per-layer D2H scores → CPU top-K → H2D indices/weights
// round-trip with a single in-place GPU kernel. Cuts ~3 syncs per
// MoE layer × ~40 routed layers = ~120 host/device syncs per token
// at ~5-7 µs each (≈ 600-800 µs/token saved).
//
// Semantics match `bias_aware_topk_weights` in forward.rs:
//   biased[i]     = scores[i] + bias[i]
//   for k in 0..k_top:
//     pick i = argmax(biased)   # unique
//     indices[k] = i
//     biased[i] = -inf
//   weights[k]    = scores[indices[k]]
//   weights      /= sum(weights)
//   weights      *= route_scale  # caller's HIPFIRE_DEEPSEEK4_ROUTE_SCALE override
//
// Hash-routed layers have NO bias tensor — caller passes a zero bias
// view to keep this kernel's signature uniform. Layout:
//   scores, bias: [n_exp] fp32 — n_exp is 256 on DeepSeek V4.
//   indices:      [k_top]      i32 (written as i32 little-endian raw)
//   weights:      [k_top]      fp32
//
// Launch: grid = [1, 1, 1], block = [256, 1, 1]. n_exp must equal blockDim.x.
// k_top ≤ 32 (DeepSeek V4: k_top=6).

#include <hip/hip_runtime.h>
#include <math.h>

#define MAX_K_TOP 32
#define WARP_SIZE 32

extern "C" __global__ void deepseek4_moe_topk_bias_aware_f32(
    const float* __restrict__ scores,      // [n_exp]
    const float* __restrict__ bias,        // [n_exp] — zero if hash-routed
    int*         __restrict__ indices,     // [k_top]
    float*       __restrict__ weights,     // [k_top]
    int   n_exp,
    int   k_top,
    float route_scale
) {
    const int tid = threadIdx.x;

    __shared__ float s_score[1024];        // scores[i] (raw, used for weights)
    __shared__ float s_biased[1024];       // working biased values
    __shared__ int   s_idx[MAX_K_TOP];     // top-K indices
    __shared__ float s_w_unsum[MAX_K_TOP]; // scores at top-K indices (pre-norm)
    __shared__ float s_sum;                // sum of s_w_unsum

    // Phase 1: cooperative load. Bounds check for n_exp ≤ blockDim.x.
    if (tid < n_exp) {
        const float sc = scores[tid];
        s_score[tid]  = sc;
        s_biased[tid] = sc + bias[tid];
    }
    __syncthreads();

    // Phase 2: K rounds of argmax. Each round runs a block-wide reduction
    // over s_biased to find (max value, lane that holds it). After picking,
    // poison that lane's value with -inf so the next round skips it.
    //
    // Single-warp setup: since blockDim.x = 256 = 8 warps, we reduce
    // per-warp via __shfl_down then cross-warp via LDS scratch.
    __shared__ float s_warp_max[WARP_SIZE];
    __shared__ int   s_warp_arg[WARP_SIZE];

    for (int k = 0; k < k_top; ++k) {
        // Each thread loads its slot's biased value (or -inf if out of range).
        float v   = (tid < n_exp) ? s_biased[tid] : -1e30f;
        int   ix  = tid;

        // Warp reduce.
        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down(v,  off);
            int   oi = __shfl_down(ix, off);
            if (ov > v) { v = ov; ix = oi; }
        }
        if (lane_id == 0) {
            s_warp_max[warp_id] = v;
            s_warp_arg[warp_id] = ix;
        }
        __syncthreads();

        // Cross-warp reduce on first warp.
        if (warp_id == 0) {
            const int n_warps = (blockDim.x + 31) >> 5;
            float ww = (lane_id < n_warps) ? s_warp_max[lane_id] : -1e30f;
            int   wi = (lane_id < n_warps) ? s_warp_arg[lane_id] : 0;
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down(ww, off);
                int   oi = __shfl_down(wi, off);
                if (ov > ww) { ww = ov; wi = oi; }
            }
            if (lane_id == 0) {
                s_idx[k] = wi;
                s_w_unsum[k] = s_score[wi];
                s_biased[wi] = -1e30f;  // poison so next round skips it
            }
        }
        __syncthreads();
    }

    // Phase 3: sum the top-K unsumed weights (single thread; k_top ≤ 32).
    // Clamp with a tiny epsilon so the divide below doesn't NaN-out when
    // the router selects experts whose unbiased scores all happen to be
    // zero (degenerate-but-possible case under sqrt_softplus → 0).
    if (tid == 0) {
        float sum = 0.0f;
        for (int k = 0; k < k_top; ++k) sum += s_w_unsum[k];
        s_sum = fmaxf(sum, 1e-12f);
    }
    __syncthreads();

    // Phase 4: write outputs. Normalize by sum then apply route_scale.
    // First k_top threads each write one slot.
    if (tid < k_top) {
        const float w = (s_w_unsum[tid] / s_sum) * route_scale;
        indices[tid] = s_idx[tid];
        weights[tid] = w;
    }
}
