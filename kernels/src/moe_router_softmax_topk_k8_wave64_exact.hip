// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

#include <hip/hip_runtime.h>

// gfx1100 exact wave64 router for the 256-expert decode path.
//
// One physical wave emulates the arithmetic and selection order of the
// production two-launch path:
//
//   softmax_f32 (256 wave32 threads) -> moe_topk_renorm_k8
//
// Each physical lane owns virtual threads lane + {0,64,128,192}. The max and
// sum helpers first reproduce the legacy 256 -> 128 -> 64 tree in registers,
// then complete its 64 -> 1 tail with wave64 shuffles. Top-k reduces the same
// eight consecutive wave32 chunks and compares their winners in chunk order.
// Direct division is intentional at both normalization sites: reciprocal
// multiplication changes routing weights by one ULP on MQ4 MoE models.

#define ROUTER_EXACT_PER_LANE 4
#define ROUTER_EXACT_K 8

__device__ __forceinline__ float router_exact_max_256(
    float v0, float v1, float v2, float v3, int lane)
{
    // Legacy s=128, then s=64.
    float v02 = fmaxf(v0, v2);
    float v13 = fmaxf(v1, v3);
    float v = fmaxf(v02, v13);

    // Legacy s=32. An indexed wave64 shuffle cannot cross gfx1100's physical
    // wave32 halves; XOR 32 is the native cross-half exchange.
    float other = __shfl_xor(v, 32);
    if (lane < 32) v = fmaxf(v, other);

    // Legacy s=16..1 within the low physical half.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        other = __shfl_down(v, off, 32);
        if (lane < off) v = fmaxf(v, other);
    }

    // Broadcast the low-half lane-0 result to both physical halves.
    float half_leader = __shfl(v, 0, 32);
    float other_half = __shfl_xor(half_leader, 32);
    return lane < 32 ? half_leader : other_half;
}

__device__ __forceinline__ float router_exact_sum_256(
    float v0, float v1, float v2, float v3, int lane)
{
    // Preserve the legacy association:
    //   s=128: (0+128), (64+192)
    //   s=64:  (0+128) + (64+192)
    float v02 = v0 + v2;
    float v13 = v1 + v3;
    float v = v02 + v13;

    float other = __shfl_xor(v, 32);
    if (lane < 32) v += other;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        other = __shfl_down(v, off, 32);
        if (lane < off) v += other;
    }

    float half_leader = __shfl(v, 0, 32);
    float other_half = __shfl_xor(half_leader, 32);
    return lane < 32 ? half_leader : other_half;
}

__device__ __forceinline__ float router_exact_half_leader_f32(float v, int lane) {
    float own_half = __shfl(v, 0, 32);
    float other_half = __shfl_xor(own_half, 32);
    return lane < 32 ? own_half : other_half;
}

__device__ __forceinline__ float router_exact_other_half_leader_f32(float v, int lane) {
    float own_half = __shfl(v, 0, 32);
    float other_half = __shfl_xor(own_half, 32);
    return lane < 32 ? other_half : own_half;
}

__device__ __forceinline__ int router_exact_half_leader_i32(int v, int lane) {
    int own_half = __shfl(v, 0, 32);
    int other_half = __shfl_xor(own_half, 32);
    return lane < 32 ? own_half : other_half;
}

__device__ __forceinline__ int router_exact_other_half_leader_i32(int v, int lane) {
    int own_half = __shfl(v, 0, 32);
    int other_half = __shfl_xor(own_half, 32);
    return lane < 32 ? other_half : own_half;
}

__device__ __forceinline__ void router_exact_wave32_chunk_argmax(
    float& v, int& idx)
{
    // width=32 partitions the wave64 into the same eight logical wave32
    // chunks used by moe_topk_renorm_k8's 256-thread workgroup.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down(v, off, 32);
        int oi = __shfl_down(idx, off, 32);
        if (ov > v) {
            v = ov;
            idx = oi;
        }
    }
}

extern "C" __launch_bounds__(64, 4)
__global__ void moe_router_softmax_topk_k8_wave64_exact(
    const float* __restrict__ logits,
    int* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int n_exp,
    int norm_topk)
{
    const int lane = threadIdx.x;
    float prob[ROUTER_EXACT_PER_LANE];

    // This kernel is dispatched only for n_exp==256. Keep the bounds handling
    // defensive so an accidental direct call fails safely rather than reading
    // outside the logits tensor.
    #pragma unroll
    for (int slot = 0; slot < ROUTER_EXACT_PER_LANE; ++slot) {
        const int expert = lane + slot * 64;
        prob[slot] = expert < n_exp ? logits[expert] : -INFINITY;
    }

    // softmax_f32 starts every virtual thread's max at -1e30 before folding
    // its one n_exp==256 element.
    const float max0 = fmaxf(-1.0e30f, prob[0]);
    const float max1 = fmaxf(-1.0e30f, prob[1]);
    const float max2 = fmaxf(-1.0e30f, prob[2]);
    const float max3 = fmaxf(-1.0e30f, prob[3]);
    const float row_max = router_exact_max_256(max0, max1, max2, max3, lane);

    #pragma unroll
    for (int slot = 0; slot < ROUTER_EXACT_PER_LANE; ++slot) {
        const int expert = lane + slot * 64;
        prob[slot] = expert < n_exp ? expf(prob[slot] - row_max) : 0.0f;
    }
    const float row_sum = router_exact_sum_256(
        prob[0], prob[1], prob[2], prob[3], lane);

    // Match softmax_f32's direct `/= sum`, including its rounding point.
    #pragma unroll
    for (int slot = 0; slot < ROUTER_EXACT_PER_LANE; ++slot)
        prob[slot] /= row_sum;

    // Distribute the eight selected pairs over lanes 0..7. This needs one
    // value/index register per lane instead of an eight-element array per lane.
    float picked_v = 0.0f;
    int picked_i = -1;

    #pragma unroll 1
    for (int round = 0; round < ROUTER_EXACT_K; ++round) {
        float chunk_v0 = prob[0];
        float chunk_v1 = prob[1];
        float chunk_v2 = prob[2];
        float chunk_v3 = prob[3];
        int chunk_i0 = lane;
        int chunk_i1 = lane + 64;
        int chunk_i2 = lane + 128;
        int chunk_i3 = lane + 192;

        router_exact_wave32_chunk_argmax(chunk_v0, chunk_i0);
        router_exact_wave32_chunk_argmax(chunk_v1, chunk_i1);
        router_exact_wave32_chunk_argmax(chunk_v2, chunk_i2);
        router_exact_wave32_chunk_argmax(chunk_v3, chunk_i3);

        // Reproduce tid 0's strict `>` scan of warp_v[0..7]. Broadcasts make
        // every lane execute the same serial scan, so no extra barrier or LDS
        // publication is needed.
        float final_v = router_exact_half_leader_f32(chunk_v0, lane);
        int final_i = router_exact_half_leader_i32(chunk_i0, lane);

        #define ROUTER_EXACT_CONSIDER(candidate_v_expr, candidate_i_expr) do { \
            const float candidate_v = (candidate_v_expr); \
            const int candidate_i = (candidate_i_expr); \
            if (candidate_v > final_v) { \
                final_v = candidate_v; \
                final_i = candidate_i; \
            } \
        } while (0)

        ROUTER_EXACT_CONSIDER(
            router_exact_other_half_leader_f32(chunk_v0, lane),
            router_exact_other_half_leader_i32(chunk_i0, lane));
        ROUTER_EXACT_CONSIDER(
            router_exact_half_leader_f32(chunk_v1, lane),
            router_exact_half_leader_i32(chunk_i1, lane));
        ROUTER_EXACT_CONSIDER(
            router_exact_other_half_leader_f32(chunk_v1, lane),
            router_exact_other_half_leader_i32(chunk_i1, lane));
        ROUTER_EXACT_CONSIDER(
            router_exact_half_leader_f32(chunk_v2, lane),
            router_exact_half_leader_i32(chunk_i2, lane));
        ROUTER_EXACT_CONSIDER(
            router_exact_other_half_leader_f32(chunk_v2, lane),
            router_exact_other_half_leader_i32(chunk_i2, lane));
        ROUTER_EXACT_CONSIDER(
            router_exact_half_leader_f32(chunk_v3, lane),
            router_exact_half_leader_i32(chunk_i3, lane));
        ROUTER_EXACT_CONSIDER(
            router_exact_other_half_leader_f32(chunk_v3, lane),
            router_exact_other_half_leader_i32(chunk_i3, lane));
        #undef ROUTER_EXACT_CONSIDER

        if (lane == round) {
            picked_v = final_v;
            picked_i = final_i;
        }

        #pragma unroll
        for (int slot = 0; slot < ROUTER_EXACT_PER_LANE; ++slot) {
            if (lane + slot * 64 == final_i) prob[slot] = -INFINITY;
        }
    }

    // Match moe_topk_renorm_k8's lane-0 accumulation order exactly.
    float picked_sum = 0.0f;
    #pragma unroll
    for (int round = 0; round < ROUTER_EXACT_K; ++round)
        picked_sum += __shfl(picked_v, round, 64);

    if (lane < ROUTER_EXACT_K) {
        int safe_i = picked_i;
        float safe_w = norm_topk && picked_sum > 0.0f
            ? picked_v / picked_sum
            : picked_v;
        if (safe_i < 0 || safe_i >= n_exp) {
            safe_i = 0;
            safe_w = 0.0f;
        }
        topk_idx[lane] = safe_i;
        topk_w[lane] = safe_w;
    }
}
