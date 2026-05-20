// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused RMSNorm + AWQ per-channel divide + FWHT rotation for MQ4-AWQ inputs.
// Variant of fused_rmsnorm_mq_rotate.hip with an additional `awq_scale`
// argument: after the RMSNorm step computes `x_norm[i] = x[i] * γ[i] * rms`,
// divide elementwise by `awq_scale[i]` BEFORE the FWHT rotation. Completes
// the AWQ math `(W·s) · (x/s) = W·x` where W·s is baked at quantize time
// (see hipfire-quantize's `awq_pre_scale_weights`).
//
// Phase A Stage A — see docs/plans/awq_hipfire.md §6 for the integration
// pattern and the rationale for putting the divide here (before the FWHT)
// rather than in the GEMM kernel.
//
// Backward compatibility: this kernel is dispatched ONLY when the upcoming
// linear layer carries an `awq_scale` GpuTensor (loaded from a sidecar
// `<weight>.awq_scale.weight` tensor in the .hfq file). Otherwise the
// standard fused_rmsnorm_mq_rotate kernel runs. Both produce identical
// output on .hfq files that don't carry AWQ scales.
//
// Layout / phases: identical to fused_rmsnorm_mq_rotate.hip aside from the
// extra Phase 1d divide. See that kernel's preamble for the FWHT details.

__launch_bounds__(256, 1)
extern "C" __global__ void fused_rmsnorm_mq_rotate_awq(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ awq_scale,    // [K], FP32 (host-side F16→F32)
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K,
    float eps
) {
    extern __shared__ float smem[];
    float* x_shared = smem;            // [K]
    float* reduce   = smem + K;        // [256]

    const int tid = threadIdx.x;
    const int block_size = blockDim.x; // 256
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;      // 0..7
    const int num_warps = block_size >> 5;

    // Batch row: this block processes x[blockIdx.x * K .. (blockIdx.x+1) * K].
    const long long batch_off = (long long)blockIdx.x * K;
    const float* x_b     = x     + batch_off;
          float* x_rot_b = x_rot + batch_off;

    // Phase 1a: gather → cache to LDS and accumulate sum-of-squares.
    float local_sum = 0.0f;
    for (int i = tid; i < K; i += block_size) {
        float v = x_b[i];
        x_shared[i] = v;
        local_sum += v * v;
    }
    reduce[tid] = local_sum;
    __syncthreads();

    // Phase 1b: block reduction → RMS.
    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    const float rms = rsqrtf(reduce[0] / (float)K + eps);

    // Phase 1c + AWQ: normalize + per-channel divide. The single FMA-ish
    // expression `x_shared[i] * γ[i] * rms / awq_scale[i]` collapses the
    // two element-wise multiplies of the non-AWQ kernel plus the AWQ
    // divide. Numerically: division is slower than multiplication on
    // RDNA — but this loop is bandwidth-bound (K * 4 bytes streaming in
    // from γ + awq_scale + x_shared), not arithmetic-bound, so the perf
    // cost of the divide is negligible at typical K values (4096-12288).
    //
    // Alternative considered: precompute reciprocal at upload time and
    // multiply here. ~Same wall-time, marginally cleaner kernel. Defer
    // to a perf-tuning follow-up.
    for (int i = tid; i < K; i += block_size) {
        x_shared[i] = x_shared[i] * weight[i] * rms / awq_scale[i];
    }
    __syncthreads();

    // Phase 2: FWHT per 256-element group. Identical to fused_rmsnorm_mq_rotate.
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

        // Wave butterfly via ds_swizzle (wave-local, 5 stages of stride 1-16).
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
}
