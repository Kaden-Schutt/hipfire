// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Nine-path fused MoE down + weighted combine (routed k=8, decode T=1).
//
// Replaces gemv_hfq4g256_moe_down_k8_indexed_batched_expanded +
// moe_down_combine_k8_batched: the expanded [8 × down_m] intermediate never
// leaves LDS. One 256-thread CTA stages the rotated activation for all 8
// experts once; warp w owns expert topk_indices[w] and dots its down rows;
// after a barrier, warp 0 folds the 8 partials with topk_weights in the
// exact k=0..7 order of moe_down_combine_k8_batched and accumulates into
// the residual (single owner per row — no atomics, deterministic under
// graph replay, matching the atomic-free contract of the expanded+combine
// pair).
//
// BYTE-EXACT with the expanded+combine pair at down_k=512: per-row dot uses
// the same two live pairwise slots (group0 → acc0, group1 → acc1;
// acc = acc0 + acc1) and the same shuffle-down tree; the fold applies
// weights in ascending krank order then += into out, same as the combine.
//
// Requirements: down_k == 512 (2 groups), down_m % RPB == 0,
// blockDim == 256 (8 warps × 32).

__launch_bounds__(256, 4)
extern "C" __global__ void gemv_hfq4g256_moe_ninepath_d4(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [8]
    const float* __restrict__ topk_weights,              // [8]
    const float* __restrict__ act,                       // [8 × 512] rotated silu out
    float* __restrict__ out,                             // [down_m] accumulated
    int down_m, int down_k
) {
    constexpr int RPB = 16;
    __shared__ float act_s[8][512];
    __shared__ float part[8][RPB];
    const int tid = threadIdx.x;
    for (int i = tid; i < 8 * 512; i += 256)
        (&act_s[0][0])[i] = act[i];
    __syncthreads();

    const int warp = tid >> 5;   // 0..7 = krank
    const int lane = tid & 31;
    const int expert_id = topk_indices[warp];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const int r0 = blockIdx.x * RPB;
    const int boff = lane * 4;
    const float* __restrict__ xa = act_s[warp];

    #define DOG_X8(gp, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp)); \
        float zp = __builtin_bit_cast(float, *(const unsigned int*)((gp) + 4)); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * (x0) \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (x1) \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (x2) \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (x3) \
             + (sc * (float)(((pk) >> 16)  & 0xFu) + zp) * (x4) \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (x5) \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (x6) \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (x7); \
    } while (0)

    for (int r = 0; r < RPB; r++) {
        const int row = r0 + r;
        const char* row_ptr = A + (long long)row * 2 * 136;

        const float4 xv0a = *reinterpret_cast<const float4*>(xa + lane * 8);
        const float4 xv0b = *reinterpret_cast<const float4*>(xa + lane * 8 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(xa + lane * 8 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(xa + lane * 8 + 260);

        float acc0 = 0.0f, acc1 = 0.0f;
        // Group 0 → pairwise slot 0 (matches baseline tail g=0 → acc0).
        DOG_X8(row_ptr + 0 * 136, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        // Group 1 → pairwise slot 1 (matches baseline tail g=1 → acc1).
        DOG_X8(row_ptr + 1 * 136, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);

        float acc = acc0 + acc1;
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (lane == 0) part[warp][r] = acc;
    }
    __syncthreads();

    // Deterministic weighted fold: ascending krank, single owner per row.
    if (warp == 0 && lane < RPB) {
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++)
            a += topk_weights[w] * part[w][lane];
        out[r0 + lane] += a;
    }

    #undef DOG_X8
}
