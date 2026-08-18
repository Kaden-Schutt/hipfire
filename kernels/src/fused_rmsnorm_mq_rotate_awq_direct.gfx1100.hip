// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1100 AWQ direct path. Preserve the baseline's 256-thread RMS
// reduction tree, IEEE divide, FWHT order, and output ABI, but do not stage the
// full activation and normalized vector through LDS. The second x read should
// hit cache; each FWHT wave instead consumes x/weight/AWQ directly in float4s.
// Qwen3.6-27B K=5120 certification: +2.44% graph-on A/B/B/A, 14.859 ->
// 9.116 us/launch, and exact tokens through a 1025-token greedy replay.
//
// Grid: [batch]. Block: [256]. LDS: 256 floats for the exact RMS tree.
__launch_bounds__(256, 1)
extern "C" __global__ void fused_rmsnorm_mq_rotate_awq(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ awq_scale,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K,
    float eps
) {
    extern __shared__ float reduce[];

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    const long long batch_off = (long long)blockIdx.x * K;
    const float* x_b = x + batch_off;
    float* x_rot_b = x_rot + batch_off;

    // Keep the baseline's scalar accumulation and descending-offset LDS tree.
    float local_sum = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        const float v = x_b[i];
        local_sum += v * v;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    const float rms = rsqrtf(reduce[0] / (float)K + eps);

    const int groups_total = K / 256;
    const int groups_per_warp = (groups_total + num_warps - 1) / num_warps;
    const int warp_group_start = warp_id * groups_per_warp;
    const int d0 = lane_id * 8;
    const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    for (int gi = 0; gi < groups_per_warp; gi++) {
        const int group = warp_group_start + gi;
        if (group >= groups_total) break;

        const int base = group * 256 + d0;
        const float4 x0 = *reinterpret_cast<const float4*>(x_b + base);
        const float4 x1 = *reinterpret_cast<const float4*>(x_b + base + 4);
        const float4 w0 = *reinterpret_cast<const float4*>(weight + base);
        const float4 w1 = *reinterpret_cast<const float4*>(weight + base + 4);
        const float4 a0 = *reinterpret_cast<const float4*>(awq_scale + base);
        const float4 a1 = *reinterpret_cast<const float4*>(awq_scale + base + 4);

        // Match the baseline expression and its IEEE division exactly:
        // ((x * weight) * rms / awq_scale) * sign1.
        float v0 = x0.x * w0.x * rms / a0.x * s10.x;
        float v1 = x0.y * w0.y * rms / a0.y * s10.y;
        float v2 = x0.z * w0.z * rms / a0.z * s10.z;
        float v3 = x0.w * w0.w * rms / a0.w * s10.w;
        float v4 = x1.x * w1.x * rms / a1.x * s11.x;
        float v5 = x1.y * w1.y * rms / a1.y * s11.y;
        float v6 = x1.z * w1.z * rms / a1.z * s11.z;
        float v7 = x1.w * w1.w * rms / a1.w * s11.w;

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
            float _p = __int_as_float( \
                __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
            if (lane_id & (str)) { (v) = _p - (v); } \
            else { (v) = (v) + _p; } \
        } while (0)
        #define HBFLY8(pat, str) \
            HBFLY(v0,pat,str); HBFLY(v1,pat,str); \
            HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
            HBFLY(v4,pat,str); HBFLY(v5,pat,str); \
            HBFLY(v6,pat,str); HBFLY(v7,pat,str)

        HBFLY8(0x041F, 1);
        HBFLY8(0x081F, 2);
        HBFLY8(0x101F, 4);
        HBFLY8(0x201F, 8);
        HBFLY8(0x401F, 16);
        #undef HBFLY8
        #undef HBFLY

        const float scale = 0.0625f;
        float4 out0;
        float4 out1;
        out0.x = v0 * scale * s20.x;
        out0.y = v1 * scale * s20.y;
        out0.z = v2 * scale * s20.z;
        out0.w = v3 * scale * s20.w;
        out1.x = v4 * scale * s21.x;
        out1.y = v5 * scale * s21.y;
        out1.z = v6 * scale * s21.z;
        out1.w = v7 * scale * s21.w;
        *reinterpret_cast<float4*>(x_rot_b + base) = out0;
        *reinterpret_cast<float4*>(x_rot_b + base + 4) = out1;
    }
}
