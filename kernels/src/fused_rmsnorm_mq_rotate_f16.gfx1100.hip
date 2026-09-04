// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Exact-FP16 variants of the fused RMSNorm + FWHT rotation producers for
// S3-f16-projection-inputs (DFlash launch fusion, gfx1100 only).
//
// CONTRACT: each kernel below is an operation-order-exact clone of its F32
// baseline, differing ONLY in the final store, which converts with the same
// `(_Float16)` round-to-nearest-even cast the `convert_f32_to_f16` /
// `cast_f32_to_f16` kernels apply. After: every F16 element is bit-identical
// to old F32-producer output followed by `convert_f32_to_f16`. No FP32
// reduction reassociation, no changed load/multiply order, no FMA exposure
// beyond what the baseline source already contains (the store expressions
// are textual copies of the baseline stores wrapped in the cast).
//
// - `fused_rmsnorm_mq_rotate_f16` clones `fused_rmsnorm_mq_rotate`
//   (kernels/src/fused_rmsnorm_mq_rotate.hip, R29c1 rms_p1_pref_xw):
//   stride sum-of-squares, first-group float4 prefetch under the reduction,
//   warp-shuffle reduction tree, rsqrt, `((x*w)*rms)*s1`, local +
//   ds_swizzle butterflies, `(v*s)*s2` stores.
// - `fused_rmsnorm_mq_rotate_awq_f16` clones the gfx1100 AWQ path op order
//   shared bit-exactly by `fused_rmsnorm_mq_rotate_awq` and
//   `fused_rmsnorm_mq_rotate_awq_direct_gfx1100`: scalar stride
//   sum-of-squares, descending-offset LDS reduction tree, rsqrt,
//   `((x*w)*rms/awq)*s1`, identical butterflies, `(v*scale)*s2` stores.
//   (The two AWQ baselines differ only in LDS staging, never in value
//   operation order, so one F16 clone matches whichever baseline runs.)
//
// Grid: [batch_size, 1, 1]. Block: [256]. The plain kernel takes the same
// oversized `(K+256)*4` dynamic-shared reservation as its baseline launcher
// (the R29c1 kernel only needs `reduce[256]`); the AWQ kernel takes
// `256*4` exactly like the direct baseline launcher.

#include <hip/hip_runtime.h>

__launch_bounds__(256, 1)
extern "C" __global__ void fused_rmsnorm_mq_rotate_f16(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    _Float16* __restrict__ x_rot_f16,
    int K,
    float eps
) {
    extern __shared__ float smem[];
    float* reduce = smem;              // [256]

    const int tid = threadIdx.x;
    const int block_size = blockDim.x; // 256
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;      // 0..7
    const int num_warps = block_size >> 5;

    const long long batch_off = (long long)blockIdx.x * K;
    const float* x_b     = x     + batch_off;
    _Float16* x_rot_f16_b = x_rot_f16 + batch_off;

    // Phase 1a: accumulate sum-of-squares (same stride/order as baseline).
    float local_sum = 0.0f;
    for (int i = tid; i < K; i += block_size) {
        float v = x_b[i];
        local_sum += v * v;
    }

    // Prefetch first FWHT group's x/weight (+ lane-local signs) so the
    // loads ride under Phase-1b barriers. groups layout is wave-uniform
    // (depends only on K and warp_id), so this is safe before rms exists.
    const int groups_total = K / 256;
    const int groups_per_warp = (groups_total + num_warps - 1) / num_warps;
    const int warp_group_start = warp_id * groups_per_warp;
    const int d0 = lane_id * 8;

    float4 pref_x0, pref_x1, pref_w0, pref_w1;
    float4 s10, s11, s20, s21;
    const int pref_group = warp_group_start;
    const bool have_pref = (groups_per_warp > 0 && pref_group < groups_total);
    if (have_pref) {
        const int base = pref_group * 256 + d0;
        pref_x0 = *reinterpret_cast<const float4*>(x_b + base);
        pref_x1 = *reinterpret_cast<const float4*>(x_b + base + 4);
        pref_w0 = *reinterpret_cast<const float4*>(weight + base);
        pref_w1 = *reinterpret_cast<const float4*>(weight + base + 4);
    }
    // signs1/2 are indexed only by lane (not group) — load once for the warp.
    s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    // Phase 1b: block reduction -> RMS (unchanged tree).
    float warp_sum = local_sum;
    warp_sum += __shfl_down(warp_sum, 16);
    warp_sum += __shfl_down(warp_sum, 8);
    warp_sum += __shfl_down(warp_sum, 4);
    warp_sum += __shfl_down(warp_sum, 2);
    warp_sum += __shfl_down(warp_sum, 1);
    if (lane_id == 0) reduce[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 32) {
        warp_sum = (tid < num_warps) ? reduce[tid] : 0.0f;
        warp_sum += __shfl_down(warp_sum, 4);
        warp_sum += __shfl_down(warp_sum, 2);
        warp_sum += __shfl_down(warp_sum, 1);
        if (tid == 0) reduce[0] = rsqrtf(warp_sum / (float)K + eps);
    }
    __syncthreads();
    const float rms = reduce[0];

    // Phase 2: FWHT. First group uses prefetched x/w; later groups (if any)
    // load as before. signs already in registers.
    for (int gi = 0; gi < groups_per_warp; gi++) {
        const int group = warp_group_start + gi;
        if (group >= groups_total) break;

        const int base = group * 256 + d0;

        float4 x0, x1, w0, w1;
        if (gi == 0 && have_pref) {
            x0 = pref_x0; x1 = pref_x1;
            w0 = pref_w0; w1 = pref_w1;
        } else {
            x0 = *reinterpret_cast<const float4*>(x_b + base);
            x1 = *reinterpret_cast<const float4*>(x_b + base + 4);
            w0 = *reinterpret_cast<const float4*>(weight + base);
            w1 = *reinterpret_cast<const float4*>(weight + base + 4);
        }

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

        // F16 store: the parenthesized F32 expression is a textual copy of
        // the baseline store; the outer (_Float16) cast is the same
        // round-to-nearest-even conversion convert_f32_to_f16 applies.
        const float s = 0.0625f;
        x_rot_f16_b[base]     = (_Float16)(v0 * s * s20.x);
        x_rot_f16_b[base + 1] = (_Float16)(v1 * s * s20.y);
        x_rot_f16_b[base + 2] = (_Float16)(v2 * s * s20.z);
        x_rot_f16_b[base + 3] = (_Float16)(v3 * s * s20.w);
        x_rot_f16_b[base + 4] = (_Float16)(v4 * s * s21.x);
        x_rot_f16_b[base + 5] = (_Float16)(v5 * s * s21.y);
        x_rot_f16_b[base + 6] = (_Float16)(v6 * s * s21.z);
        x_rot_f16_b[base + 7] = (_Float16)(v7 * s * s21.w);
    }
}

// AWQ exact-FP16 producer. Clones the gfx1100 AWQ value operation order
// (scalar stride sum, descending-offset LDS reduction tree, IEEE divide
// before the FWHT, float4 group loop) shared bit-exactly by
// fused_rmsnorm_mq_rotate_awq and fused_rmsnorm_mq_rotate_awq_direct_gfx1100.
//
// Grid: [batch]. Block: [256]. LDS: 256 floats for the exact RMS tree.
__launch_bounds__(256, 1)
extern "C" __global__ void fused_rmsnorm_mq_rotate_awq_f16(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ awq_scale,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    _Float16* __restrict__ x_rot_f16,
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
    _Float16* x_rot_f16_b = x_rot_f16 + batch_off;

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

        #define HBFLY_A16(v, pat, str) do { \
            float _p = __int_as_float( \
                __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
            if (lane_id & (str)) { (v) = _p - (v); } \
            else { (v) = (v) + _p; } \
        } while (0)
        #define HBFLY8_A16(pat, str) \
            HBFLY_A16(v0,pat,str); HBFLY_A16(v1,pat,str); \
            HBFLY_A16(v2,pat,str); HBFLY_A16(v3,pat,str); \
            HBFLY_A16(v4,pat,str); HBFLY_A16(v5,pat,str); \
            HBFLY_A16(v6,pat,str); HBFLY_A16(v7,pat,str)

        HBFLY8_A16(0x041F, 1);
        HBFLY8_A16(0x081F, 2);
        HBFLY8_A16(0x101F, 4);
        HBFLY8_A16(0x201F, 8);
        HBFLY8_A16(0x401F, 16);
        #undef HBFLY8_A16
        #undef HBFLY_A16

        const float scale = 0.0625f;
        x_rot_f16_b[base]     = (_Float16)(v0 * scale * s20.x);
        x_rot_f16_b[base + 1] = (_Float16)(v1 * scale * s20.y);
        x_rot_f16_b[base + 2] = (_Float16)(v2 * scale * s20.z);
        x_rot_f16_b[base + 3] = (_Float16)(v3 * scale * s20.w);
        x_rot_f16_b[base + 4] = (_Float16)(v4 * scale * s21.x);
        x_rot_f16_b[base + 5] = (_Float16)(v5 * scale * s21.y);
        x_rot_f16_b[base + 6] = (_Float16)(v6 * scale * s21.z);
        x_rot_f16_b[base + 7] = (_Float16)(v7 * scale * s21.w);
    }
}
