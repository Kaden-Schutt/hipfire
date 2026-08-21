// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1100 AWQ RMSNorm+MQ-rotate wavegrid (exact-tree variant).
//
// The `_awq` and `_awq_direct` baselines run the whole kernel in ONE 256-thread
// workgroup: per-thread strided sum-of-squares, then a descending-offset LDS
// tree (s = 128, 64, ..., 1), then the FWHT tail. One workgroup means one CU —
// every VRAM load serializes behind a single wave's latency chain.
//
// This variant keeps every float operation bit-identical to that baseline but
// spreads the K/256 FWHT groups across K/256 wave32 workgroups:
//
//   1. Lane t of workgroup w (logical thread index t = w*32+lane) reproduces
//      the baseline's strided accumulation exactly: local = ((x[t]^2 +
//      x[t+256]^2) + x[t+512]^2) ... in the same order. Published as r[t].
//   2. One arrival barrier. Workgroup 0's warp then folds the tree with the
//      baseline's exact association:
//        a_i = r_i + r_{i+128};  b_i = a_i + a_{i+64};  c_i = b_i + b_{i+32}
//      so lane l<32 computes
//        c_l = (a_l + a_{l+64}) + (a_{l+32} + a_{l+96})
//      from r directly (same operands, same order, same floats), then
//      shfl_down 16/8/4/2/1 reproduces the s = 16..1 stages. rms = rsqrtf(
//      total/K + eps) is bit-identical to the baseline's reduce[0].
//   3. Every workgroup runs its own 256-element FWHT group with the exact
//      `_awq_direct` element expressions: ((x * weight) * rms / awq_scale) *
//      sign1 → butterfly + swizzle stages → * 0.0625 * sign2.
//
// Scratch layout (2048 B, zero-initialized once, self-restoring epoch):
//   f32 r[256], f32 rms, u32 arrivals, u32 ready, u32 done, padding.
//
// Grid: [K/256]. Block: [32]. Host gates K % 256 == 0 and K/256 <= 32 so all
// waves stay co-resident (spin-wait is safe).

__launch_bounds__(32, 16)
extern "C" __global__ void fused_rmsnorm_mq_rotate_awq_wavegrid(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ awq_scale,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    float* __restrict__ scratch,
    int K,
    float eps
) {
    volatile float* r = scratch;
    volatile float* rms_slot = scratch + 256;
    unsigned int* arrivals = reinterpret_cast<unsigned int*>(scratch + 257);
    unsigned int* ready = reinterpret_cast<unsigned int*>(scratch + 258);
    unsigned int* done = reinterpret_cast<unsigned int*>(scratch + 259);

    const int lane_id = threadIdx.x;
    const int group_id = blockIdx.x;
    const int n_groups = gridDim.x;
    const int tid = group_id * 32 + lane_id;   // baseline logical thread index
    const int d0 = lane_id * 8;
    const int base = group_id * 256 + d0;

    // Baseline per-thread accumulation, exact order: i = tid; i < K; i += 256.
    // Only logical threads 0..255 exist in the baseline; workgroups 8+ have no
    // reduction role (their lanes skip straight to the ready spin).
    if (tid < 256) {
        float local_sum = 0.0f;
        for (int i = tid; i < K; i += 256) {
            const float v = x[i];
            local_sum += v * v;
        }
        // Every lane publishes its own baseline-thread partial.
        r[tid] = local_sum;
        __threadfence();
    }
    __syncthreads();
    if (group_id < 8 && lane_id == 0) {
        atomicAdd(arrivals, 1u);
    }

    // Prefetch this group's slice while the reduction publishes.
    const float4 x0 = *reinterpret_cast<const float4*>(x + base);
    const float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
    const float4 w0 = *reinterpret_cast<const float4*>(weight + base);
    const float4 w1 = *reinterpret_cast<const float4*>(weight + base + 4);
    const float4 a0 = *reinterpret_cast<const float4*>(awq_scale + base);
    const float4 a1 = *reinterpret_cast<const float4*>(awq_scale + base + 4);
    const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);


    // Workgroup 0 folds the tree once all per-thread partials are visible.
    if (group_id == 0) {
        if (lane_id == 0) {
            while (atomicAdd(arrivals, 0u) != 8u) {
                __builtin_amdgcn_s_sleep(1);
            }
            __threadfence();
        }
        // Broadcast the arrival signal to the whole warp.
        const int all_in = __shfl(lane_id == 0 ? 1 : 0, 0);
        if (all_in) {
            // Stages a + b + c with the baseline's exact association. Lane l
            // needs r_l, r_{l+32}, r_{l+64}, r_{l+96}, r_{l+128}, r_{l+160},
            // r_{l+192}, r_{l+224}.
            const int l = lane_id;
            const float a_l = r[l] + r[l + 128];
            const float a_l32 = r[l + 32] + r[l + 160];
            const float a_l64 = r[l + 64] + r[l + 192];
            const float a_l96 = r[l + 96] + r[l + 224];
            const float b_l = a_l + a_l64;
            const float b_l32 = a_l32 + a_l96;
            const float c_l = b_l + b_l32;

            // Stages s = 16..1: shfl_down with the same operand pairing.
            float v = c_l;
            v += __shfl_down(v, 16);
            v += __shfl_down(v, 8);
            v += __shfl_down(v, 4);
            v += __shfl_down(v, 2);
            v += __shfl_down(v, 1);
            if (lane_id == 0) {
                *rms_slot = rsqrtf(v / (float)K + eps);
                __threadfence();
                atomicExch(ready, 1u);
            }
        }
    }

    if (lane_id == 0) {
        while (atomicAdd(ready, 0u) == 0u) {
            __builtin_amdgcn_s_sleep(1);
        }
        __threadfence();
    }
    const float rms = __shfl(lane_id == 0 ? *rms_slot : 0.0f, 0);

    // Exact multiply/divide order of the `_awq_direct` twin:
    // ((x * weight) * rms / awq_scale) * sign1.
    float v0 = x0.x * w0.x * rms / a0.x * s10.x;
    float v1 = x0.y * w0.y * rms / a0.y * s10.y;
    float v2 = x0.z * w0.z * rms / a0.z * s10.z;
    float v3 = x0.w * w0.w * rms / a0.w * s10.w;
    float v4 = x1.x * w1.x * rms / a1.x * s11.x;
    float v5 = x1.y * w1.y * rms / a1.y * s11.y;
    float v6 = x1.z * w1.z * rms / a1.z * s11.z;
    float v7 = x1.w * w1.w * rms / a1.w * s11.w;

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
    if (lane_id == 0 && atomicAdd(done, 1u) == static_cast<unsigned int>(n_groups - 1)) {
        atomicExch(arrivals, 0u);
        atomicExch(ready, 0u);
        atomicExch(done, 0u);
    }
}
