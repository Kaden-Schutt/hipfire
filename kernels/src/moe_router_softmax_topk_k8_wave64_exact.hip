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

#ifndef HIPFIRE_ROUTER_EXACT_KERNEL
#define HIPFIRE_ROUTER_EXACT_KERNEL moe_router_softmax_topk_k8_wave64_exact
#endif

__device__ __forceinline__ float router_exact_cross_half_f32(float v) {
    return __uint_as_float(__builtin_amdgcn_permlane64(__float_as_uint(v)));
}

__device__ __forceinline__ int router_exact_cross_half_i32(int v) {
    return (int)__builtin_amdgcn_permlane64((unsigned)v);
}

__device__ __forceinline__ float router_exact_max_256(
    float v0, float v1, float v2, float v3, int lane)
{
    // Legacy s=128, then s=64.
    float v02 = fmaxf(v0, v2);
    float v13 = fmaxf(v1, v3);
    float v = fmaxf(v02, v13);

    // Legacy s=32. gfx1100's ds_bpermute-based HIP shuffles are confined to a
    // physical wave32 half even in wave64 mode; V_PERMLANE64 is the gfx11
    // instruction which exchanges matching lanes across those halves.
    float other = router_exact_cross_half_f32(v);
    if (lane < 32) v = fmaxf(v, other);

    // Legacy s=16..1 within the low physical half.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        other = __shfl_down(v, off, 32);
        if (lane < off) v = fmaxf(v, other);
    }

    // Broadcast the low-half lane-0 result to both physical halves.
    float half_leader = __shfl(v, 0, 32);
    float other_half = router_exact_cross_half_f32(half_leader);
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

    float other = router_exact_cross_half_f32(v);
    if (lane < 32) v += other;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        other = __shfl_down(v, off, 32);
        if (lane < off) v += other;
    }

    float half_leader = __shfl(v, 0, 32);
    float other_half = router_exact_cross_half_f32(half_leader);
    return lane < 32 ? half_leader : other_half;
}

__device__ __forceinline__ float router_exact_half_leader_f32(float v, int lane) {
    float own_half = __shfl(v, 0, 32);
    float other_half = router_exact_cross_half_f32(own_half);
    return lane < 32 ? own_half : other_half;
}

__device__ __forceinline__ float router_exact_other_half_leader_f32(float v, int lane) {
    float own_half = __shfl(v, 0, 32);
    float other_half = router_exact_cross_half_f32(own_half);
    return lane < 32 ? other_half : own_half;
}

__device__ __forceinline__ int router_exact_half_leader_i32(int v, int lane) {
    int own_half = __shfl(v, 0, 32);
    int other_half = router_exact_cross_half_i32(own_half);
    return lane < 32 ? own_half : other_half;
}

__device__ __forceinline__ int router_exact_other_half_leader_i32(int v, int lane) {
    int own_half = __shfl(v, 0, 32);
    int other_half = router_exact_cross_half_i32(own_half);
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

#ifdef HIPFIRE_ROUTER_SHARED_SILU_MQ_ROTATE
// Workgroups after block 0 execute the independent shared-expert SwiGLU + MQ
// rotation.  A physical wave64 carries two original wave32 workgroups, one in
// each half, so K=512 retains the original two-way group parallelism while the
// command processor sees one heterogeneous dispatch instead of two kernels.
__device__ __forceinline__ void router_shared_silu_mq_rotate(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K)
{
    const int lane = threadIdx.x;
    const int tid = lane & 31;
    const int group = ((int)blockIdx.x - 1) * 2 + (lane >> 5);
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    const int d0 = tid * 8;
    const int base = group * 256 + d0;

    #define ROUTER_SHARED_SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
    float g0 = gate[base];     float u0 = up[base];
    float g1 = gate[base + 1]; float u1 = up[base + 1];
    float g2 = gate[base + 2]; float u2 = up[base + 2];
    float g3 = gate[base + 3]; float u3 = up[base + 3];
    float g4 = gate[base + 4]; float u4 = up[base + 4];
    float g5 = gate[base + 5]; float u5 = up[base + 5];
    float g6 = gate[base + 6]; float u6 = up[base + 6];
    float g7 = gate[base + 7]; float u7 = up[base + 7];

    float v0 = ROUTER_SHARED_SILU_MUL(g0, u0) * signs1[d0];
    float v1 = ROUTER_SHARED_SILU_MUL(g1, u1) * signs1[d0 + 1];
    float v2 = ROUTER_SHARED_SILU_MUL(g2, u2) * signs1[d0 + 2];
    float v3 = ROUTER_SHARED_SILU_MUL(g3, u3) * signs1[d0 + 3];
    float v4 = ROUTER_SHARED_SILU_MUL(g4, u4) * signs1[d0 + 4];
    float v5 = ROUTER_SHARED_SILU_MUL(g5, u5) * signs1[d0 + 5];
    float v6 = ROUTER_SHARED_SILU_MUL(g6, u6) * signs1[d0 + 6];
    float v7 = ROUTER_SHARED_SILU_MUL(g7, u7) * signs1[d0 + 7];
    #undef ROUTER_SHARED_SILU_MUL

    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    #define ROUTER_SHARED_HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)
    #define ROUTER_SHARED_HBFLY8(pat, str) \
        ROUTER_SHARED_HBFLY(v0,pat,str); ROUTER_SHARED_HBFLY(v1,pat,str); \
        ROUTER_SHARED_HBFLY(v2,pat,str); ROUTER_SHARED_HBFLY(v3,pat,str); \
        ROUTER_SHARED_HBFLY(v4,pat,str); ROUTER_SHARED_HBFLY(v5,pat,str); \
        ROUTER_SHARED_HBFLY(v6,pat,str); ROUTER_SHARED_HBFLY(v7,pat,str)

    ROUTER_SHARED_HBFLY8(0x041F, 1);
    ROUTER_SHARED_HBFLY8(0x081F, 2);
    ROUTER_SHARED_HBFLY8(0x101F, 4);
    ROUTER_SHARED_HBFLY8(0x201F, 8);
    ROUTER_SHARED_HBFLY8(0x401F, 16);
    #undef ROUTER_SHARED_HBFLY8
    #undef ROUTER_SHARED_HBFLY

    const float s = 0.0625f;
    x_rot[base]     = v0 * s * signs2[d0];
    x_rot[base + 1] = v1 * s * signs2[d0 + 1];
    x_rot[base + 2] = v2 * s * signs2[d0 + 2];
    x_rot[base + 3] = v3 * s * signs2[d0 + 3];
    x_rot[base + 4] = v4 * s * signs2[d0 + 4];
    x_rot[base + 5] = v5 * s * signs2[d0 + 5];
    x_rot[base + 6] = v6 * s * signs2[d0 + 6];
    x_rot[base + 7] = v7 * s * signs2[d0 + 7];
}
#endif

extern "C" __launch_bounds__(64, 4)
__global__ void HIPFIRE_ROUTER_EXACT_KERNEL(
    const float* __restrict__ logits,
    int* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int n_exp,
    int norm_topk
#ifdef HIPFIRE_ROUTER_SHARED_SILU_MQ_ROTATE
    , const float* __restrict__ shared_gate
    , const float* __restrict__ shared_up
    , const float* __restrict__ signs1
    , const float* __restrict__ signs2
    , float* __restrict__ shared_x_rot
    , int shared_k
#endif
)
{
#ifdef HIPFIRE_ROUTER_SHARED_SILU_MQ_ROTATE
    if (blockIdx.x != 0) {
        router_shared_silu_mq_rotate(
            shared_gate, shared_up, signs1, signs2, shared_x_rot, shared_k);
        return;
    }
#endif
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
