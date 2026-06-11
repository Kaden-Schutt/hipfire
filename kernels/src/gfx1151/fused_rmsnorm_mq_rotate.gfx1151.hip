// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1151 fused RMSNorm + FWHT rotation for MagnumQuant inputs.
//
// Same math contract as the generic fused_rmsnorm_mq_rotate kernel, but the
// RMS reduction avoids the 256-float shared-memory halving ladder. Each wave
// reduces its strided sum with shuffles, then wave 0 reduces the 8 wave sums.
// The FWHT phase is intentionally identical to the generic kernel.

#if defined(__gfx1151__)
#define HIPFIRE_GFX1151 1
#else
#define HIPFIRE_GFX1151 0
#endif

__device__ __forceinline__ float wave_sum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 32);
    }
    return v;
}

extern "C" __launch_bounds__(256, 1)
__global__ void fused_rmsnorm_mq_rotate_gfx1151(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K,
    float eps
) {
#if HIPFIRE_GFX1151
    extern __shared__ float smem[];
    float* x_shared = smem;         // [K]
    float* warp_sums = smem + K;    // [8]

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    const long long batch_off = (long long)blockIdx.x * K;
    const float* x_b = x + batch_off;
    float* x_rot_b = x_rot + batch_off;

    float local_sum = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        const float v = x_b[i];
        x_shared[i] = v;
        local_sum += v * v;
    }

    const float wsum = wave_sum32(local_sum);
    if (lane_id == 0) {
        warp_sums[warp_id] = wsum;
    }
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        total = wave_sum32(total);
        if (lane_id == 0) {
            warp_sums[0] = total;
        }
    }
    __syncthreads();

    const float rms = rsqrtf(warp_sums[0] / (float)K + eps);

    for (int i = tid; i < K; i += blockDim.x) {
        x_shared[i] = x_shared[i] * weight[i] * rms;
    }
    __syncthreads();

    const int groups_total = K / 256;
    const int groups_per_warp = (groups_total + num_warps - 1) / num_warps;
    const int warp_group_start = warp_id * groups_per_warp;

    for (int gi = 0; gi < groups_per_warp; gi++) {
        const int group = warp_group_start + gi;
        if (group >= groups_total) break;

        const int d0 = lane_id * 8;
        const int base = group * 256 + d0;

        float v0 = x_shared[base]     * signs1[d0];
        float v1 = x_shared[base + 1] * signs1[d0 + 1];
        float v2 = x_shared[base + 2] * signs1[d0 + 2];
        float v3 = x_shared[base + 3] * signs1[d0 + 3];
        float v4 = x_shared[base + 4] * signs1[d0 + 4];
        float v5 = x_shared[base + 5] * signs1[d0 + 5];
        float v6 = x_shared[base + 6] * signs1[d0 + 6];
        float v7 = x_shared[base + 7] * signs1[d0 + 7];

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

        #define HBFLY(v, pat, str) do { \
            float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
            if (lane_id & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
        } while (0)
        #define HBFLY8(pat, str) \
            HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
            HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

        HBFLY8(0x041F, 1);
        HBFLY8(0x081F, 2);
        HBFLY8(0x101F, 4);
        HBFLY8(0x201F, 8);
        HBFLY8(0x401F, 16);
        #undef HBFLY8
        #undef HBFLY

        const float s = 0.0625f;
        x_rot_b[base]     = v0 * s * signs2[d0];
        x_rot_b[base + 1] = v1 * s * signs2[d0 + 1];
        x_rot_b[base + 2] = v2 * s * signs2[d0 + 2];
        x_rot_b[base + 3] = v3 * s * signs2[d0 + 3];
        x_rot_b[base + 4] = v4 * s * signs2[d0 + 4];
        x_rot_b[base + 5] = v5 * s * signs2[d0 + 5];
        x_rot_b[base + 6] = v6 * s * signs2[d0 + 6];
        x_rot_b[base + 7] = v7 * s * signs2[d0 + 7];
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && K == 0) {
        x_rot[0] = x[0] + weight[0] + signs1[0] + signs2[0] + eps;
    }
#endif
}
