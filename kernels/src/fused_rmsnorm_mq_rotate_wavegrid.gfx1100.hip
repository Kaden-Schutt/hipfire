// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1100 K=2048 experiment: preserve the exact arithmetic assignment of the
// baseline 256-thread block while allowing its eight wave32s to occupy eight
// CUs. Each workgroup computes one original warp partial. The last arrival
// performs the same 8-value reduction and publishes RMS through persistent
// global scratch; all eight waves then execute one independent FWHT group.
//
// Scratch layout (64 bytes, zero-initialized and restored after every launch):
//   f32 partials[8], f32 rms, u32 arrivals, u32 ready, u32 done, padding.

__launch_bounds__(32, 16)
extern "C" __global__ void fused_rmsnorm_mq_rotate_wavegrid(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    float* __restrict__ scratch,
    int K,
    float eps
) {
    if (K != 2048 || blockDim.x != 32 || gridDim.x != 8) return;

    volatile float* partials = scratch;
    volatile float* rms_slot = scratch + 8;
    unsigned int* arrivals = reinterpret_cast<unsigned int*>(scratch + 9);
    unsigned int* ready = reinterpret_cast<unsigned int*>(scratch + 10);
    unsigned int* done = reinterpret_cast<unsigned int*>(scratch + 11);

    const int lane_id = threadIdx.x;
    const int warp_id = blockIdx.x;
    const int original_tid = warp_id * 32 + lane_id;
    const int d0 = lane_id * 8;
    const int base = warp_id * 256 + d0;

    // Match the baseline thread's i=tid; i<K; i+=256 accumulation exactly.
    float local_sum = 0.0f;
    for (int i = original_tid; i < K; i += 256) {
        const float v = x[i];
        local_sum += v * v;
    }

    // Match the baseline first-group prefetch window while the reduction runs.
    const float4 x0 = *reinterpret_cast<const float4*>(x + base);
    const float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
    const float4 w0 = *reinterpret_cast<const float4*>(weight + base);
    const float4 w1 = *reinterpret_cast<const float4*>(weight + base + 4);
    const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    float warp_sum = local_sum;
    warp_sum += __shfl_down(warp_sum, 16);
    warp_sum += __shfl_down(warp_sum, 8);
    warp_sum += __shfl_down(warp_sum, 4);
    warp_sum += __shfl_down(warp_sum, 2);
    warp_sum += __shfl_down(warp_sum, 1);

    unsigned int arrival = 0;
    if (lane_id == 0) {
        partials[warp_id] = warp_sum;
        __threadfence();
        arrival = atomicAdd(arrivals, 1u);
    }
    arrival = __shfl(arrival, 0);

    // The eighth arrival performs the baseline tid<32 second-stage tree.
    if (arrival == 7u) {
        float total = lane_id < 8 ? partials[lane_id] : 0.0f;
        total += __shfl_down(total, 4);
        total += __shfl_down(total, 2);
        total += __shfl_down(total, 1);
        if (lane_id == 0) {
            *rms_slot = rsqrtf(total / (float)K + eps);
            __threadfence();
            atomicExch(ready, 1u);
        }
    }

    if (lane_id == 0) {
        while (atomicAdd(ready, 0u) == 0u) {
            __builtin_amdgcn_s_sleep(1);
        }
        __threadfence();
    }
    const float rms = __shfl(lane_id == 0 ? *rms_slot : 0.0f, 0);

    // Exact multiply order: ((x * weight) * rms) * sign.
    float v0 = x0.x * w0.x * rms * s10.x;
    float v1 = x0.y * w0.y * rms * s10.y;
    float v2 = x0.z * w0.z * rms * s10.z;
    float v3 = x0.w * w0.w * rms * s10.w;
    float v4 = x1.x * w1.x * rms * s11.x;
    float v5 = x1.y * w1.y * rms * s11.y;
    float v6 = x1.z * w1.z * rms * s11.z;
    float v7 = x1.w * w1.w * rms * s11.w;

    // Local butterfly: strides 1, 2, 4.
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

    // Last completion restores the persistent epoch before the next launch.
    __threadfence();
    if (lane_id == 0 && atomicAdd(done, 1u) == 7u) {
        atomicExch(arrivals, 0u);
        atomicExch(ready, 0u);
        atomicExch(done, 0u);
    }
}
