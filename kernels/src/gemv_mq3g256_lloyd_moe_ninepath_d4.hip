// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Nine-path fused MoE down + weighted combine for MQ3-Lloyd (routed k=8,
// decode T=1). Codebook port of gemv_hfq4g256_moe_ninepath_d4.hip.
//
// WHY — measured per-dispatch PM4 attribution, a3b decode, gfx1201, 693
// launches/token on BOTH SKUs:
//
//   projection   mq2r (MQ3-Lloyd)              mq4r (HFQ4)
//   gate_up      347.6 us  grid 1024x8         649.7 us  grid 1024x8
//   down         882.2 us  grid 2048x8         585.3 us  grid 128x1  <- ninepath
//   routed total 1229.8 us                     1235.0 us
//
// mq2r's 2.25-bpw gate_up already WINS by 302 us. It gives all of it back on
// the down projection, and the cause is decomposition, not dtype: the
// incumbent launches one 32-thread block per (row, krank) — 16,384 single-wave
// workgroups, EIGHT fills of a 2048-slot part — and every one of them re-reads
// the same rotated activation. 166 GB/s effective against gate_up's 543.
//
// This kernel adopts the HFQ4 ninepath structure:
//   * ONE 256-thread CTA (8 warps) per row tile. Warp w owns expert
//     topk_indices[w]; grid.x tiles the down rows.
//   * The rotated activation for all 8 experts is staged into LDS ONCE and
//     shared by every warp, instead of being re-read 16,384 times.
//   * The expanded [8 x down_m] intermediate never leaves LDS: after a single
//     barrier, warp 0 folds the 8 partials with topk_weights in ascending
//     krank order and accumulates into the residual. SINGLE OWNER PER ROW —
//     no atomicAdd, so unlike both the incumbent and the R2 row tile this is
//     deterministic under graph replay.
//
// CODEBOOK HANDLING — the one real difference from the HFQ4 parent. HFQ4
// decodes a nibble with a (scale, zero-point) pair held in registers; MQ3-Lloyd
// indexes an 8-entry fp16 codebook per group, and a DYNAMIC index into a local
// array spills to scratch on AMDGPU. So the codebooks must live in LDS. They
// are staged for ALL RPB rows UP FRONT behind the same single barrier as the
// activation, which keeps the row loop completely barrier-free — the per-row
// staging alternative costs two __syncthreads() per row (32 per CTA).
//
// LDS / occupancy, measured by compiling for gfx1201 (hipcc 7.14):
//   HFQ4 reference    lds 16896  occ 14 waves/SIMD  (no codebook to stage)
//   MQ3 RPB=4         lds 18560  occ 14   <- default, matches the reference tier
//   MQ3 RPB=8         lds 20736  occ 12
//   MQ3 RPB=16        lds 25088  occ 10
// The codebook staging is what costs occupancy at larger RPB, so RPB=4 is the
// default. grid = down_m / RPB = 2048/4 = 512 CTAs = 4096 waves (two fills),
// against the incumbent's 16,384 single-wave workgroups (eight fills).
//
// NUMERICS — per row this matches the incumbent's tail path exactly: group 0
// into pairwise slot 0, group 1 into slot 1, acc = acc0 + acc1, the same
// 5-step __shfl_down tree, the same product/sum order inside a group. What
// CHANGES is the cross-expert combine: the incumbent atomicAdds 8 wave results
// into the residual in hardware-arrival (non-deterministic) order; this folds
// them in ascending krank order. That is a DIFFERENT and deterministic
// summation order, so the result is not bit-identical to the incumbent — it is
// bit-identical in the same sense the HFQ4 ninepath is to its expanded+combine
// pair, which is the contract that kernel already ships under.
//
// Requirements: down_k == 512 (2 groups), down_m % RPB == 0, blockDim == 256.
// The host must not select this kernel for any other down_k.
//
// STATIC ONLY at time of writing: compiled for gfx1201, never executed.

#ifndef MQ3_NINEPATH_RPB
#define MQ3_NINEPATH_RPB 4
#endif

__launch_bounds__(256, 4)
extern "C" __global__ void gemv_mq3g256_lloyd_moe_ninepath_d4(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int*   __restrict__ topk_indices,             // [8]
    const float* __restrict__ topk_weights,             // [8]
    const float* __restrict__ act,                      // [8 x 512] rotated silu out
    float*       __restrict__ out,                      // [down_m] accumulated
    int down_m, int down_k
) {
    constexpr int RPB = MQ3_NINEPATH_RPB;
    __shared__ float act_s[8][512];
    __shared__ float cb_s[8][RPB][2][8];
    __shared__ float part[8][RPB];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;   // 0..7 = krank
    const int lane = tid & 31;

    // ---- stage the activation for all 8 experts, once ----
    for (int i = tid; i < 8 * 512; i += 256)
        (&act_s[0][0])[i] = act[i];

    const int expert_id = topk_indices[warp];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const int r0 = blockIdx.x * RPB;

    // ---- stage every row's codebooks for THIS warp's expert, once ----
    // Each warp fills its own cb_s[warp] slice: RPB rows x 2 groups x 8 entries
    // = 16*RPB values, covered by 32 lanes in ceil(16*RPB/32) passes.
    {
        const int per_warp = RPB * 2 * 8;
        for (int i = lane; i < per_warp; i += 32) {
            const int e = i & 7;          // codebook entry
            const int g = (i >> 3) & 1;   // group
            const int r = i >> 4;         // row within the tile
            const char* gp = A + (long long)(r0 + r) * 2 * 112 + (long long)g * 112;
            cb_s[warp][r][g][e] = __half2float(((const __half*)gp)[e]);
        }
    }
    __syncthreads();

    const int boff = lane * 3;
    const float* __restrict__ xa = act_s[warp];

    // Identical product/sum tree to the incumbent's DOG3_LDS / TAIL_LOAD_AND_DOT.
    #define DOG3_X8(cb, gp, a, x0, x1, x2, x3, x4, x5, x6, x7) do {            \
        const unsigned char* d = (const unsigned char*)((gp) + 16) + boff;      \
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8)        \
                                             | ((unsigned int)d[2] << 16);      \
        (a) += (cb)[ pk        & 7u] * (x0)                                     \
             + (cb)[(pk >>  3) & 7u] * (x1)                                     \
             + (cb)[(pk >>  6) & 7u] * (x2)                                     \
             + (cb)[(pk >>  9) & 7u] * (x3)                                     \
             + (cb)[(pk >> 12) & 7u] * (x4)                                     \
             + (cb)[(pk >> 15) & 7u] * (x5)                                     \
             + (cb)[(pk >> 18) & 7u] * (x6)                                     \
             + (cb)[(pk >> 21) & 7u] * (x7);                                    \
    } while (0)

    for (int r = 0; r < RPB; r++) {
        const char* row_ptr = A + (long long)(r0 + r) * 2 * 112;

        const float4 xv0a = *reinterpret_cast<const float4*>(xa + lane * 8);
        const float4 xv0b = *reinterpret_cast<const float4*>(xa + lane * 8 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(xa + lane * 8 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(xa + lane * 8 + 260);

        float acc0 = 0.0f, acc1 = 0.0f;
        // Group 0 -> pairwise slot 0 (matches the incumbent tail g=0 -> acc0).
        DOG3_X8(cb_s[warp][r][0], row_ptr + 0 * 112, acc0,
                xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        // Group 1 -> pairwise slot 1 (matches the incumbent tail g=1 -> acc1).
        DOG3_X8(cb_s[warp][r][1], row_ptr + 1 * 112, acc1,
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

    #undef DOG3_X8
}
