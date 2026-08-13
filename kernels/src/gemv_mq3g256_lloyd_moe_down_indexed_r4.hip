// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Row-tiled variants of
// `gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed`
// (kernels/src/gemv_mq3g256_lloyd_moe_down_indexed.hip).
//
// Emits two kernels from one templated body so the R=2 and R=4 tilings cannot
// drift apart:
//   gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r2  (grid ceil(M/2))
//   gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r4  (grid ceil(M/4))
//
// WHY (static analysis at the a3b decode shape M=2048, K=512, k_top=8, gfx1201):
//
//  1. WAVE COUNT. The incumbent launches grid [M, k_top] = 2048x8 = 16384
//     single-wave workgroups per layer. A 64-CU RDNA4 part has 128 SIMDs x 16
//     waves/SIMD = 2048 wave slots, so that is EIGHT full machine fills to move
//     2 groups x 112 B = 224 B of weight per wave. Its gate_up sibling
//     (gemv_mq2g256_lloyd_moe_gate_up_k8_indexed) launches 1024x8 = 8192 waves
//     — FOUR fills — each moving 8 groups x 72 B = 576 B. From the per-dispatch
//     PM4 attribution: gate_up 543 GB/s effective, down 166 GB/s. The down
//     kernel is NOT bandwidth-bound; it is per-wave-latency bound, and it pays
//     that per-wave floor twice as many times for a quarter of the work.
//
//  2. DEAD MAIN LOOP + SERIALIZED TAIL. `quads = (K/256) >> 2` is ZERO when
//     K = 512, so the incumbent's K4-unrolled main loop — the one with four
//     disjoint LDS codebook slots and four independent accumulators — never
//     executes. Both groups fall through TAIL_LOAD_AND_DOT, which reloads the
//     codebook into the SAME `cb_lds[0..8)` slots. gfx1201 ISA confirms the
//     resulting WAR hazard: each tail block opens with
//     `s_wait_loadcnt_dscnt 0x0`, a FULL VMEM + LDS drain, so the two groups
//     cannot overlap. The quad path pays that drain once per FOUR groups; the
//     tail path pays it once per group.
//
//  3. Only 8 of 32 lanes participate in the tail codebook load (`if (tid < 8)`).
//
// WHAT THESE KERNELS CHANGE (dispatch geometry + codebook staging, NOT the math):
//
//  * R rows per wave. grid.x becomes ceil(M/R), cutting the wave count Rx. The
//    prologue's two DEPENDENT scalar loads (topk_indices[krank], then
//    expert_ptrs[expert_id]) are paid once per R rows instead of once per row.
//  * `x` (= rot_batch + krank*K) depends only on `krank`, never on the row, so
//    each group's 8 x elements are loaded once (2 x float4) and reused by all R
//    rows — the x-reuse row-tile shape that paid +3.5% on the fused MQ4 MoE
//    down and +4.24% on multirow R=2. x is loaded per group rather than hoisted
//    across a whole quad so register pressure stays bounded for every K.
//  * The codebook for a whole quad-step — R rows x 4 groups x 8 entries — is
//    filled by ALL 32 lanes into DISJOINT slots behind a SINGLE __syncthreads.
//    At K=512 that is one full drain per wave covering 2R (row, group)
//    products, versus the incumbent's one drain per 1.
//  * 4R independent accumulators restore the ILP the K4 path was written for.
//  * gfx12 `s_prefetch_data` on the row bases, copied from the proven
//    gemv_hfq4g256_moe_down_k8_indexed_batched_expanded template.
//
// NUMERICS — BIT-EXACT PER ROW vs the incumbent, by construction:
//  * Group g accumulates into slot `g & 3`. The incumbent does the same: its
//    quad body sends 4q+j to accj and its tail sends (quads*4)+j to accj. Here
//    the group loop advances in steps of 4 from g=0, so the group at offset j
//    inside a step has (g+j) & 3 == j.
//  * Per (row, group) the 8 products are summed in the identical expression
//    shape (same operand order, same tree) as DOG3_LDS / TAIL_LOAD_AND_DOT.
//  * The final fold is the identical `(s0 + s1) + (s2 + s3)`, followed by the
//    identical 5-step `__shfl_down` reduction and the identical
//    `atomicAdd(&x_residual[row], topk_weights[krank] * acc)`.
//  Cross-wave atomic ARRIVAL ORDER changes (fewer, wider waves), but that order
//  is already hardware-defined and non-reproducible in the incumbent, so these
//  kernels introduce no new non-determinism.
//
// OCCUPANCY (static, gfx1201 wave32, 1536 VGPR/SIMD, granularity 8, cap 16):
//  R=2 → 45 VGPR, R=4 → 77 VGPR, both zero spill and zero scratch, both at the
//  16-wave/SIMD hardware cap (needs <= 96 VGPR). The incumbent is 74 VGPR. So
//  the row tile is free on occupancy: it buys Rx fewer waves at unchanged
//  waves-per-SIMD, and keeps the scratch-free property the PM4 replay path
//  requires.
//
// STATIC ONLY: nothing above was measured on a GPU. Every figure is a
// compile-time / ISA / PM4-attribution count, not a performance claim.
//
// Grid: [ceil(M/R), k_top, 1], block [32]. Kernargs identical to the incumbent.

#ifndef MQ3_DOWN_R2_WAVES_PER_EU
#define MQ3_DOWN_R2_WAVES_PER_EU 16
#endif
#ifndef MQ3_DOWN_R4_WAVES_PER_EU
#define MQ3_DOWN_R4_WAVES_PER_EU 16
#endif

// This lane's 3 packed 3-bit-index bytes for one (row, group).
__device__ __forceinline__ unsigned int mq3_down_pk3(const char* gp, int boff) {
    const unsigned char* d = (const unsigned char*)(gp + 16) + boff;
    return (unsigned int)d[0] | ((unsigned int)d[1] << 8)
                              | ((unsigned int)d[2] << 16);
}

// Identical product/sum tree to DOG3_LDS / TAIL_LOAD_AND_DOT in the incumbent.
__device__ __forceinline__ float mq3_down_dog3(const float* __restrict__ cb,
                                               unsigned int pk,
                                               float4 xa, float4 xb) {
    return cb[ pk        & 7u] * xa.x
         + cb[(pk >>  3) & 7u] * xa.y
         + cb[(pk >>  6) & 7u] * xa.z
         + cb[(pk >>  9) & 7u] * xa.w
         + cb[(pk >> 12) & 7u] * xb.x
         + cb[(pk >> 15) & 7u] * xb.y
         + cb[(pk >> 18) & 7u] * xb.z
         + cb[(pk >> 21) & 7u] * xb.w;
}

template <int R>
__device__ __forceinline__ void mq3_down_rowtile_body(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float*       __restrict__ x_residual,
    int M, int K,
    float* cb_lds)   // R * 4 * 8 floats
{
    const int row_base = blockIdx.x * R;
    if (row_base >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = rot_batch + (long long)krank * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 112;
    const int boff = tid * 3;

    // Out-of-range rows are clamped onto row_base so their weight reads stay in
    // bounds; their atomicAdd is masked off separately below.
    const char* rp[R];
    bool live[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        live[r] = (row_base + r) < M;
        const int rr = live[r] ? (row_base + r) : row_base;
        rp[r] = A + (long long)rr * row_bytes;
    }

#if defined(__gfx1200__) || defined(__gfx1201__)
    #pragma unroll
    for (int r = 0; r < R; r++)
        __builtin_amdgcn_s_prefetch_data((const void*)rp[r], 0);
#endif

    // acc[r][slot]; slot == g & 3, matching the incumbent's assignment.
    float acc[R][4];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int s = 0; s < 4; s++) acc[r][s] = 0.0f;
    }

    for (int g = 0; g < groups_per_row; g += 4) {
        const int n = (groups_per_row - g) < 4 ? (groups_per_row - g) : 4;

        // Cooperative codebook fill: all 32 lanes, disjoint slots, ONE barrier.
        // Pass r covers slots [32r, 32r+32) → row r, group-in-quad tid>>3,
        // codebook entry tid&7.
        {
            const int j_ = tid >> 3;
            const int e_ = tid & 7;
            if (j_ < n) {
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    const __half* cb_h =
                        (const __half*)(rp[r] + (long long)(g + j_) * 112);
                    cb_lds[r * 32 + j_ * 8 + e_] = __half2float(cb_h[e_]);
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= n) continue;
            const int xb = (g + j) * 256 + tid * 8;
            const float4 xv0 = *reinterpret_cast<const float4*>(x + xb);
            const float4 xv1 = *reinterpret_cast<const float4*>(x + xb + 4);
            #pragma unroll
            for (int r = 0; r < R; r++) {
                const unsigned int pk =
                    mq3_down_pk3(rp[r] + (long long)(g + j) * 112, boff);
                acc[r][j] +=
                    mq3_down_dog3(cb_lds + r * 32 + j * 8, pk, xv0, xv1);
            }
        }

        // WAR against the NEXT step's cooperative fill. Guarded so the
        // single-step case (K <= 1024, which includes the a3b decode shape
        // K=512) pays exactly ONE full VMEM+LDS drain per wave instead of two.
        // `g` and `groups_per_row` are block-uniform, so the guard is uniform.
        // Guarding at the loop TAIL rather than hoisting to the head matters:
        // the head-guarded form makes LLVM peel the first iteration, which cost
        // +17 VGPR on R=2 and +17 on R=4 (94 of the 96-VGPR 16-wave budget) for
        // no drain benefit.
        if (g + 4 < groups_per_row) __syncthreads();
    }

    // Identical pairwise fold and identical 5-step reduction, per row.
    float red[R];
    #pragma unroll
    for (int r = 0; r < R; r++)
        red[r] = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int r = 0; r < R; r++) red[r] += __shfl_down(red[r], offset);
    }

    if (tid == 0) {
        const float scale = topk_weights[krank];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (live[r]) atomicAdd(&x_residual[row_base + r], scale * red[r]);
        }
    }
}

__launch_bounds__(32, MQ3_DOWN_R2_WAVES_PER_EU)
extern "C" __global__ void gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r2(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float*       __restrict__ x_residual,
    int M, int K
) {
    __shared__ float cb_lds[2 * 4 * 8];
    mq3_down_rowtile_body<2>(expert_ptrs, topk_indices, topk_weights,
                             rot_batch, x_residual, M, K, cb_lds);
}

__launch_bounds__(32, MQ3_DOWN_R4_WAVES_PER_EU)
extern "C" __global__ void gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r4(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float*       __restrict__ x_residual,
    int M, int K
) {
    __shared__ float cb_lds[4 * 4 * 8];
    mq3_down_rowtile_body<4>(expert_ptrs, topk_indices, topk_weights,
                             rot_batch, x_residual, M, K, cb_lds);
}
