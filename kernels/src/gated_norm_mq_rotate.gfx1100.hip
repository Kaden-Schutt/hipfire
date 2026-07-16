// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL
#define HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL gated_norm_mq_rotate_gfx1100
#endif

// gfx1100 Qwen3.6 DeltaNet decode specialization. Two wave32s reproduce two
// independent head_dim=128 gated_norm_f32 workgroups, publish their normalized
// 256-value group through LDS, then wave 0 performs the exact mq_rotate_x
// butterfly. This removes the normalized global-memory round trip and one
// dependent dispatch without changing either operation's arithmetic order.
extern "C" __launch_bounds__(64, 1)
__global__ void HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int n_heads,
    int head_dim,
    float eps)
{
    if (n_heads != 32 || head_dim != 128) return;

    __shared__ float normalized[256];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int group = blockIdx.x;
    const int head = group * 2 + wave;

    const float* xh = x + head * head_dim;
    const float* zh = z + head * head_dim;
    float* oh = normalized + wave * head_dim;

    // Match gated_norm_f32's per-lane accumulation and XOR reduction exactly.
    float sq_sum = 0.0f;
    for (int i = lane; i < head_dim; i += 32)
        sq_sum += xh[i] * xh[i];
    for (int o = 16; o > 0; o >>= 1)
        sq_sum += __shfl_xor(sq_sum, o);
    const float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);

    for (int i = lane; i < head_dim; i += 32) {
        const float normed = xh[i] * inv_rms * weight[i];
        const float z_val = zh[i];
        const float silu_z = z_val / (1.0f + expf(-z_val));
        oh[i] = normed * silu_z;
    }
    __syncthreads();

    // Each 256-value MQ group spans exactly four normalized heads. One wave
    // consumes the LDS row using mq_rotate_x's original lane ownership.
    if (wave != 0) return;
    const int d0 = lane * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(normalized + d0);
    const float4 x1 = *reinterpret_cast<const float4*>(normalized + d0 + 4);
    const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    float v0 = x0.x * s10.x;
    float v1 = x0.y * s10.y;
    float v2 = x0.z * s10.z;
    float v3 = x0.w * s10.w;
    float v4 = x1.x * s11.x;
    float v5 = x1.y * s11.y;
    float v6 = x1.z * s11.z;
    float v7 = x1.w * s11.w;

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
        if (lane & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
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
    float* out = x_rot + group * 256 + d0;
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
    *reinterpret_cast<float4*>(out) = out0;
    *reinterpret_cast<float4*>(out + 4) = out1;
}
