// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Row-tiled variants of
// `gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed`
// (kernels/src/gemv_mq2g256_lloyd_moe_down_indexed.hip).
//
// Direct port of gemv_mq3g256_lloyd_moe_down_indexed_r4.hip to the MQ2-Lloyd
// container. Only the container constants and the decode differ:
//   MQ3: 112 B/group, 8-entry codebook (16 B header), 3 packed bytes/lane
//   MQ2:  72 B/group, 4-entry codebook ( 8 B header), 2 packed bytes/lane
//
// Emits two kernels from one templated body so R=2 and R=4 cannot drift apart:
//   gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r2  (grid ceil(M/2))
//   gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r4  (grid ceil(M/4))
//
// WHY — the incumbent has the SAME two defects the MQ3 sibling had, and the MQ3
// row tile measured +3.8% end-to-end on PM4 (193.8 -> 201.1 tok/s, a3b P1,
// gfx1201, tools.redline bench, route proof valid):
//
//  1. WAVE COUNT. Incumbent grid [M, k_top] = 2048x8 = 16384 single-wave
//     workgroups per layer on a part with 2048 wave slots — EIGHT machine fills
//     to move 2 groups x 72 B = 144 B of weight per wave.
//
//  2. DEAD MAIN LOOP at the a3b decode shape. `quads = groups_per_row >> 2` is
//     ZERO when K = moe_intermediate = 512 (groups_per_row = 2), so the
//     K4-unrolled body — four disjoint LDS codebook slots, four independent
//     accumulators — NEVER EXECUTES. Both groups fall through
//     TAIL_LOAD_AND_DOT, which restages the codebook with only `tid < 4`
//     (4 of 32 lanes) behind its OWN __syncthreads, once PER GROUP. Two full
//     barriers + two 4-lane restages to cover two groups, where the quad path
//     would pay one of each for four.
//
//     This is the shape-dependent trap, not a coding error: gate_up is safe
//     because its K is `hidden` (2048 -> 8 groups). Any MoE with
//     moe_intermediate < 1024 hits it on the DOWN projection with a G256
//     codebook format. ds4 (moe_intermediate 2048) is unaffected.
//
// FIX (geometry + codebook staging only, never the math):
//  * R rows per wave on a ceil(M/R) grid: 16384 waves -> 8192 (R2) / 4096 (R4).
//    The two DEPENDENT prologue scalar loads (topk_indices[krank] then
//    expert_ptrs[expert_id]) are paid once per R rows instead of once per row.
//  * x = rot_batch + krank*K depends only on krank, never on the row, so each
//    group's 8 x elements load once (2x float4) and are reused by all R rows.
//  * The codebook for a whole quad-step — R rows x 4 groups x 4 entries — is
//    filled by all 32 lanes into DISJOINT slots behind a SINGLE __syncthreads.
//  * 4R independent accumulators restore the ILP the K4 path was written for.
//
// BIT-EXACTNESS: NOT bit-exact, and deliberately so — the MQ3 sibling measured
// +0.40% wt2 KLD (0.235561 -> 0.236503) from exactly this change. Per (row,
// group) the 8 products use the identical operand order and sum tree, and the
// final fold is the identical `(s0+s1)+(s2+s3)` followed by the identical
// 5-step __shfl_down and `atomicAdd(&x_residual[row], topk_weights[krank]*acc)`
// — but distributing groups across 4R accumulators changes the ACCUMULATION
// ORDER versus the incumbent's tail path, which is a real FP difference.
// Keep OPT-IN; do not flip the default on speed alone.
//
// Group g accumulates into slot `g & 3`, matching the incumbent (its quad body
// sends 4q+j to accj, its tail sends (quads*4)+j to accj; here the group loop
// steps by 4 from g=0 so the group at offset j has (g+j) & 3 == j).
//
// STATIC ONLY: nothing here was measured on a GPU by the author of this file.
// The +3.8% / +0.40% KLD figures above are the measured MQ3 sibling, not this.
//
// Grid: [ceil(M/R), k_top, 1], block [32]. Kernargs identical to the incumbent.

#ifndef MQ2_DOWN_R2_WAVES_PER_EU
#define MQ2_DOWN_R2_WAVES_PER_EU 16
#endif
#ifndef MQ2_DOWN_R4_WAVES_PER_EU
#define MQ2_DOWN_R4_WAVES_PER_EU 16
#endif

// This lane's 2 packed 2-bit-index bytes for one (row, group).
// MQ2 header is 8 B (4 fp16 codebook entries); data starts at +8.
__device__ __forceinline__ unsigned int mq2_down_pk2(const char* gp, int boff) {
    const unsigned char* d = (const unsigned char*)(gp + 8) + boff;
    return (unsigned int)d[0] | ((unsigned int)d[1] << 8);
}

// Identical product/sum tree to DOG2_LDS / TAIL_LOAD_AND_DOT in the incumbent.
__device__ __forceinline__ float mq2_down_dog2(const float* __restrict__ cb,
                                               unsigned int pk,
                                               float4 xa, float4 xb) {
    return cb[ pk        & 3u] * xa.x
         + cb[(pk >>  2) & 3u] * xa.y
         + cb[(pk >>  4) & 3u] * xa.z
         + cb[(pk >>  6) & 3u] * xa.w
         + cb[(pk >>  8) & 3u] * xb.x
         + cb[(pk >> 10) & 3u] * xb.y
         + cb[(pk >> 12) & 3u] * xb.z
         + cb[(pk >> 14) & 3u] * xb.w;
}

template <int R>
__device__ __forceinline__ void mq2_down_rowtile_body(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float*       __restrict__ x_residual,
    int M, int K,
    float* cb_lds)   // R * 4 * 4 floats
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
    const int row_bytes = groups_per_row * 72;
    const int boff = tid * 2;

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
        // MQ2's codebook is 4 entries, so one pass covers 8 (row, group) pairs
        // per 32 lanes: lane -> group-in-quad tid>>2, codebook entry tid&3, and
        // the R rows are unrolled inside.
        {
            const int j_ = tid >> 2;
            const int e_ = tid & 3;
            if (j_ < n) {
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    const __half* cb_h =
                        (const __half*)(rp[r] + (long long)(g + j_) * 72);
                    cb_lds[r * 16 + j_ * 4 + e_] = __half2float(cb_h[e_]);
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
                    mq2_down_pk2(rp[r] + (long long)(g + j) * 72, boff);
                acc[r][j] +=
                    mq2_down_dog2(cb_lds + r * 16 + j * 4, pk, xv0, xv1);
            }
        }

        // WAR against the NEXT step's cooperative fill. Guarded so the
        // single-step case (K <= 1024, which includes the a3b decode shape
        // K=512) pays exactly ONE full VMEM+LDS drain per wave instead of two.
        // Guarding at the loop TAIL rather than the head is deliberate: the
        // head-guarded form makes LLVM peel the first iteration, which cost
        // +17 VGPR on the MQ3 sibling for no drain benefit.
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

__launch_bounds__(32, MQ2_DOWN_R2_WAVES_PER_EU)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r2(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float*       __restrict__ x_residual,
    int M, int K
) {
    __shared__ float cb_lds[2 * 4 * 4];
    mq2_down_rowtile_body<2>(expert_ptrs, topk_indices, topk_weights,
                             rot_batch, x_residual, M, K, cb_lds);
}

__launch_bounds__(32, MQ2_DOWN_R4_WAVES_PER_EU)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r4(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float*       __restrict__ x_residual,
    int M, int K
) {
    __shared__ float cb_lds[4 * 4 * 4];
    mq2_down_rowtile_body<4>(expert_ptrs, topk_indices, topk_weights,
                             rot_batch, x_residual, M, K, cb_lds);
}
