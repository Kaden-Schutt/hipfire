// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>  // __ushort_as_half / __half2float for the qt44 f16 header

// MQ4G256V2 (qt=44) sister of `gemv_hfq4g256_moe_ninepath_d4`.
//
// Transcribed from that kernel: the LDS staging, warp-per-krank assignment,
// per-row dot, shuffle tree and deterministic weighted fold are identical line
// for line. The ONLY difference is the group header decode.
//
// Why: qt44 matched neither `ninepath_hfq4` (qt13 on both projections) nor
// `ninepath_mq3l` (MQ2/MQ3-Lloyd), so it was denied the fused path even though
// it clears the shape gate. The PUBLISHED Ornith 1.5 artifact hits
// `ninepath_shape_ok` exactly -- hidden_size 2048, moe_intermediate_size 512,
// num_experts_per_tok 8 -- so it was paying the expanded+combine cost on the
// precise shape this path was tuned for.
//
// MEASURED: no decode win on single-stream. gfx1151, shipped artifact,
// 4 alternations x 12 runs, cold run discarded:
//   OFF (expanded+combine)  medians 71.50 71.50 71.80 71.40  -> 71.55 tok/s
//   ON  (this kernel)       medians 71.40 72.00 71.90 71.60  -> 71.73 tok/s
// +0.18 tok/s (+0.25%) against a within-arm spread of 71.33-71.88 -- i.e. the
// difference is smaller than the noise and is NOT a win.
//
// Routing was verified independently rather than inferred from the timings:
// a probe inside the dispatch arm fires in the ON arm and neither it nor the
// qt13 fallthrough fires in the OFF arm. (Two weaker checks misled first --
// the JIT cache is populated eagerly in BOTH arms, so cache presence proves
// nothing about dispatch.)
//
// The "+0.8% on the A3B serve battery" figure attached to qt13's D4 is from a
// CONCURRENT/multiturn workload, not this single-stream bench -- which is the
// least favourable case for D4, whose benefit is removing the expanded
// [8 x down_m] intermediate and a combine launch. That workload is untested
// here. Keep this kernel for structural parity, lower register pressure
// (46 VGPRs vs qt13's 48) and one less launch; do NOT quote a speedup for it
// until the concurrent arm is measured.
//
// Affine-grid selection is LANE-CONSTANT: lane reads the u32 at
// `boff = lane*4`, covering weights [lane*8, lane*8+8) -- wholly inside one
// 128-weight half, the same half in every group. So the header offset hoists
// out of the row loop:
//   lane  0..15 -> weights   0..127 -> grid 0 -> header at gp+0
//   lane 16..31 -> weights 128..255 -> grid 1 -> header at gp+4
// qt44 therefore reads ONE header u32 where qt13 reads two.
//
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
// BYTE-EXACT with the qt44 expanded+combine pair at down_k=512: per-row dot uses
// the same two live pairwise slots (group0 → acc0, group1 → acc1;
// acc = acc0 + acc1) and the same shuffle-down tree; the fold applies
// weights in ascending krank order then += into out, same as the combine.
//
// Requirements: down_k == 512 (2 groups), down_m % RPB == 0,
// blockDim == 256 (8 warps × 32).
//
// Kernel-name / RPB macros: defaults keep the incumbent exported symbol and
// RPB=16. Candidate modules redefine HIPFIRE_MQ4V2_NINEPATH_KERNEL and
// HIPFIRE_MQ4V2_NINEPATH_RPB before including this TU (e.g. RPB=8 gfx1100).
#ifndef HIPFIRE_MQ4V2_NINEPATH_KERNEL
#define HIPFIRE_MQ4V2_NINEPATH_KERNEL gemv_mq4g256v2_moe_ninepath_d4
#endif
#ifndef HIPFIRE_MQ4V2_NINEPATH_RPB
#define HIPFIRE_MQ4V2_NINEPATH_RPB 16
#endif

__launch_bounds__(256, 4)
extern "C" __global__ void HIPFIRE_MQ4V2_NINEPATH_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [8]
    const float* __restrict__ topk_weights,              // [8]
    const float* __restrict__ act,                       // [8 × 512] rotated silu out
    float* __restrict__ out,                             // [down_m] accumulated
    int down_m, int down_k
) {
    constexpr int RPB = HIPFIRE_MQ4V2_NINEPATH_RPB;
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
    // Lane-constant affine-grid header offset -- see the note above.
    const int hoff = (lane < 16) ? 0 : 4;
    const float* __restrict__ xa = act_s[warp];

    #define DOG_X8(gp, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        unsigned int h = *(const unsigned int*)((gp) + hoff); \
        float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16))); \
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
