// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_MOE_COMBINE_RMSNORM_MQ_KERNEL
#define HIPFIRE_MOE_COMBINE_RMSNORM_MQ_KERNEL moe_down_combine_rmsnorm_mq_rotate_vecsum
#endif

// gfx1100/K=2048 decode experiment. Fold the previous layer's atomic-free
// routed-MoE output into the residual and immediately consume the completed
// row for the next layer's RMSNorm + MagnumQuant FWHT rotation.
//
// One 256-thread workgroup owns the full row: each of its eight wave32s owns
// one 256-value FWHT group, and every lane owns eight adjacent columns. This
// is the same ownership/reduction/rotation schedule as
// fused_rmsnorm_mq_rotate_vecsum; only the x values are produced in registers
// by the byte-exact moe_down_combine_k8_batched arithmetic first.

__launch_bounds__(256, 1)
extern "C" __global__ void HIPFIRE_MOE_COMBINE_RMSNORM_MQ_KERNEL(
    const float* __restrict__ expert_outputs, // [K_TOP, M]
    const float* __restrict__ topk_weights,   // [K_TOP]
    float*       __restrict__ x_residual,     // [M], accumulated in place
    const float* __restrict__ norm_weight,    // [M]
    const float* __restrict__ signs1,         // [256]
    const float* __restrict__ signs2,         // [256]
    float*       __restrict__ x_rot,          // [M]
    int M,
    int K_TOP,
    float eps
) {
    if (M != 2048 || K_TOP != 8) return;

    __shared__ float reduce[8];

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int d0 = lane_id * 8;
    const int base = warp_id * 256 + d0;

    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    float a4 = 0.0f;
    float a5 = 0.0f;
    float a6 = 0.0f;
    float a7 = 0.0f;

    // Preserve the standalone combine's krank order and per-column FMA chain.
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float w = topk_weights[k];
        const float4 e0 = *reinterpret_cast<const float4*>(
            expert_outputs + (long long)k * M + base);
        const float4 e1 = *reinterpret_cast<const float4*>(
            expert_outputs + (long long)k * M + base + 4);
        a0 += w * e0.x;
        a1 += w * e0.y;
        a2 += w * e0.z;
        a3 += w * e0.w;
        a4 += w * e1.x;
        a5 += w * e1.y;
        a6 += w * e1.z;
        a7 += w * e1.w;
    }

    const float4 residual0 = *reinterpret_cast<const float4*>(x_residual + base);
    const float4 residual1 = *reinterpret_cast<const float4*>(x_residual + base + 4);
    float4 x0 = {
        residual0.x + a0,
        residual0.y + a1,
        residual0.z + a2,
        residual0.w + a3,
    };
    float4 x1 = {
        residual1.x + a4,
        residual1.y + a5,
        residual1.z + a6,
        residual1.w + a7,
    };
    *reinterpret_cast<float4*>(x_residual + base) = x0;
    *reinterpret_cast<float4*>(x_residual + base + 4) = x1;

    const float4 w0 = *reinterpret_cast<const float4*>(norm_weight + base);
    const float4 w1 = *reinterpret_cast<const float4*>(norm_weight + base + 4);
    const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    float warp_sum = x0.x * x0.x;
    warp_sum += x0.y * x0.y;
    warp_sum += x0.z * x0.z;
    warp_sum += x0.w * x0.w;
    warp_sum += x1.x * x1.x;
    warp_sum += x1.y * x1.y;
    warp_sum += x1.z * x1.z;
    warp_sum += x1.w * x1.w;
    warp_sum += __shfl_down(warp_sum, 16);
    warp_sum += __shfl_down(warp_sum, 8);
    warp_sum += __shfl_down(warp_sum, 4);
    warp_sum += __shfl_down(warp_sum, 2);
    warp_sum += __shfl_down(warp_sum, 1);
    if (lane_id == 0) reduce[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 32) {
        warp_sum = tid < 8 ? reduce[tid] : 0.0f;
        warp_sum += __shfl_down(warp_sum, 4);
        warp_sum += __shfl_down(warp_sum, 2);
        warp_sum += __shfl_down(warp_sum, 1);
        if (tid == 0) reduce[0] = rsqrtf(warp_sum / (float)M + eps);
    }
    __syncthreads();
    const float rms = reduce[0];

    // Exact rotation multiply order: ((x * weight) * rms) * sign.
    float v0 = x0.x * w0.x * rms * s10.x;
    float v1 = x0.y * w0.y * rms * s10.y;
    float v2 = x0.z * w0.z * rms * s10.z;
    float v3 = x0.w * w0.w * rms * s10.w;
    float v4 = x1.x * w1.x * rms * s11.x;
    float v5 = x1.y * w1.y * rms * s11.y;
    float v6 = x1.z * w1.z * rms * s11.z;
    float v7 = x1.w * w1.w * rms * s11.w;

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

    const float scale = 0.0625f;
    float4 out0 = {
        v0 * scale * s20.x,
        v1 * scale * s20.y,
        v2 * scale * s20.z,
        v3 * scale * s20.w,
    };
    float4 out1 = {
        v4 * scale * s21.x,
        v5 * scale * s21.y,
        v6 * scale * s21.z,
        v7 * scale * s21.w,
    };
    *reinterpret_cast<float4*>(x_rot + base) = out0;
    *reinterpret_cast<float4*>(x_rot + base + 4) = out1;
}
