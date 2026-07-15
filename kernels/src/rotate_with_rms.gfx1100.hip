// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Second half of the gfx1100 decode split. One wave32 owns each original
// 256-element FWHT group, preserving the fused kernel's load and arithmetic
// order without any inter-workgroup synchronization.

__launch_bounds__(32, 16)
extern "C" __global__ void rotate_with_rms_gfx1100(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    const float* __restrict__ rms_in,
    float* __restrict__ x_rot,
    int K
) {
    const int lane_id = threadIdx.x;
    const int group = blockIdx.x;
    const int d0 = lane_id * 8;
    const int base = group * 256 + d0;
    const float rms = rms_in[0];

    const float4 x0 = *reinterpret_cast<const float4*>(x + base);
    const float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
    const float4 w0 = *reinterpret_cast<const float4*>(weight + base);
    const float4 w1 = *reinterpret_cast<const float4*>(weight + base + 4);
    const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    // Exact multiply order: ((x * weight) * rms) * sign.
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
    *reinterpret_cast<float4*>(x_rot + base) = out0;
    *reinterpret_cast<float4*>(x_rot + base + 4) = out1;
}
