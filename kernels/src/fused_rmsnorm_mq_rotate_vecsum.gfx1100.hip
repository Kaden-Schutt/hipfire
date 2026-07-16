// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_RDNA3_RMSNORM_SIGN_LDS
#define HIPFIRE_RDNA3_RMSNORM_SIGN_LDS 0
#endif

#ifndef HIPFIRE_RDNA3_RMSNORM_SIGN_CONST
#define HIPFIRE_RDNA3_RMSNORM_SIGN_CONST 0
#endif

#ifndef HIPFIRE_RDNA3_RMSNORM_VECSUM_KERNEL
#define HIPFIRE_RDNA3_RMSNORM_VECSUM_KERNEL fused_rmsnorm_mq_rotate_vecsum
#endif

#if HIPFIRE_RDNA3_RMSNORM_SIGN_CONST
// Bit i is one for +1 and zero for -1. These are the exact 256-value LCG
// streams generated from the product MQ seeds 42 and 1042.
__device__ __constant__ unsigned int hipfire_mq_sign1_words[8] = {
    0x7e1f436fu, 0x7dcca4dbu, 0x656132f7u, 0x3461ea31u,
    0x3f8ff8acu, 0x57873038u, 0x3f939d88u, 0x408e25a3u,
};
__device__ __constant__ unsigned int hipfire_mq_sign2_words[8] = {
    0x615eb361u, 0x8eab84a4u, 0x2481c03bu, 0x511cc661u,
    0x3ad419b8u, 0xbef3ec93u, 0xee3a4271u, 0x390fafe9u,
};
#endif

// gfx1100 K=2048 experiment: each of the eight wave32s owns exactly one FWHT
// group and derives its RMS partial from the same float4 x values already held
// for rotation. This removes the baseline's eight scalar x loads per thread;
// the only semantic difference is the sum-of-squares association order.

__launch_bounds__(256, 1)
extern "C" __global__ void HIPFIRE_RDNA3_RMSNORM_VECSUM_KERNEL(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K,
    float eps
) {
    if (K != 2048) return;
    __shared__ float reduce[8];
#if HIPFIRE_RDNA3_RMSNORM_SIGN_LDS
    // All eight waves use the same two 256-float sign tables. Stage each
    // float4 once per workgroup instead of issuing the same global reads from
    // every wave; the existing reduction barrier makes the stage visible.
    __shared__ float4 signs1_lds[64];
    __shared__ float4 signs2_lds[64];
#endif

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int d0 = lane_id * 8;
    const int base = warp_id * 256 + d0;

#if HIPFIRE_RDNA3_RMSNORM_SIGN_CONST
    const unsigned int sign_shift = (unsigned int)(lane_id & 3) * 8u;
    const unsigned int sign1_bits =
        hipfire_mq_sign1_words[lane_id >> 2] >> sign_shift;
    const unsigned int sign2_bits =
        hipfire_mq_sign2_words[lane_id >> 2] >> sign_shift;
#endif

#if HIPFIRE_RDNA3_RMSNORM_SIGN_LDS
    if (tid < 64) {
        signs1_lds[tid] = reinterpret_cast<const float4*>(signs1)[tid];
        signs2_lds[tid] = reinterpret_cast<const float4*>(signs2)[tid];
    }
#endif

    const float4 x0 = *reinterpret_cast<const float4*>(x + base);
    const float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
    const float4 w0 = *reinterpret_cast<const float4*>(weight + base);
    const float4 w1 = *reinterpret_cast<const float4*>(weight + base + 4);
#if !HIPFIRE_RDNA3_RMSNORM_SIGN_LDS && !HIPFIRE_RDNA3_RMSNORM_SIGN_CONST
    const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);
#endif

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
        if (tid == 0) reduce[0] = rsqrtf(warp_sum / (float)K + eps);
    }
    __syncthreads();
    const float rms = reduce[0];

#if HIPFIRE_RDNA3_RMSNORM_SIGN_LDS
    const float4 s10 = signs1_lds[lane_id * 2];
    const float4 s11 = signs1_lds[lane_id * 2 + 1];
    const float4 s20 = signs2_lds[lane_id * 2];
    const float4 s21 = signs2_lds[lane_id * 2 + 1];
#endif

    // Exact rotation multiply order: ((x * weight) * rms) * sign.
#if HIPFIRE_RDNA3_RMSNORM_SIGN_CONST
    #define APPLY_PACKED_SIGN(value, bits, bit) \
        __builtin_bit_cast(float, __builtin_bit_cast(unsigned int, (value)) ^ \
            ((((~(bits)) >> (bit)) & 1u) << 31))
    float v0 = APPLY_PACKED_SIGN(x0.x * w0.x * rms, sign1_bits, 0);
    float v1 = APPLY_PACKED_SIGN(x0.y * w0.y * rms, sign1_bits, 1);
    float v2 = APPLY_PACKED_SIGN(x0.z * w0.z * rms, sign1_bits, 2);
    float v3 = APPLY_PACKED_SIGN(x0.w * w0.w * rms, sign1_bits, 3);
    float v4 = APPLY_PACKED_SIGN(x1.x * w1.x * rms, sign1_bits, 4);
    float v5 = APPLY_PACKED_SIGN(x1.y * w1.y * rms, sign1_bits, 5);
    float v6 = APPLY_PACKED_SIGN(x1.z * w1.z * rms, sign1_bits, 6);
    float v7 = APPLY_PACKED_SIGN(x1.w * w1.w * rms, sign1_bits, 7);
#else
    float v0 = x0.x * w0.x * rms * s10.x;
    float v1 = x0.y * w0.y * rms * s10.y;
    float v2 = x0.z * w0.z * rms * s10.z;
    float v3 = x0.w * w0.w * rms * s10.w;
    float v4 = x1.x * w1.x * rms * s11.x;
    float v5 = x1.y * w1.y * rms * s11.y;
    float v6 = x1.z * w1.z * rms * s11.z;
    float v7 = x1.w * w1.w * rms * s11.w;
#endif

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
#if HIPFIRE_RDNA3_RMSNORM_SIGN_CONST
    out0.x = APPLY_PACKED_SIGN(v0 * scale, sign2_bits, 0);
    out0.y = APPLY_PACKED_SIGN(v1 * scale, sign2_bits, 1);
    out0.z = APPLY_PACKED_SIGN(v2 * scale, sign2_bits, 2);
    out0.w = APPLY_PACKED_SIGN(v3 * scale, sign2_bits, 3);
    out1.x = APPLY_PACKED_SIGN(v4 * scale, sign2_bits, 4);
    out1.y = APPLY_PACKED_SIGN(v5 * scale, sign2_bits, 5);
    out1.z = APPLY_PACKED_SIGN(v6 * scale, sign2_bits, 6);
    out1.w = APPLY_PACKED_SIGN(v7 * scale, sign2_bits, 7);
    #undef APPLY_PACKED_SIGN
#else
    out0.x = v0 * scale * s20.x;
    out0.y = v1 * scale * s20.y;
    out0.z = v2 * scale * s20.z;
    out0.w = v3 * scale * s20.w;
    out1.x = v4 * scale * s21.x;
    out1.y = v5 * scale * s21.y;
    out1.z = v6 * scale * s21.z;
    out1.w = v7 * scale * s21.w;
#endif
    *reinterpret_cast<float4*>(x_rot + base) = out0;
    *reinterpret_cast<float4*>(x_rot + base + 4) = out1;
}

#undef HIPFIRE_RDNA3_RMSNORM_VECSUM_KERNEL
