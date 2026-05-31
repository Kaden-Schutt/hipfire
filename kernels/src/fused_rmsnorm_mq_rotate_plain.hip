// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused RMSNorm + (FWHT rotation AND plain RMSNorm output).
//
// Strict superset of `fused_rmsnorm_mq_rotate` (which only writes the
// FWHT-rotated output). Many DeepSeek V4 call sites compute the SAME RMSNorm
// and need both:
//   - the rotated output (`x_rot`) for MQ4-style weights
//   - the plain RMSNormed output (`x_plain`) for Q8/F16/F32 weights
// The current pattern fires `fused_rmsnorm_mq_rotate` + `rmsnorm_f32`
// back-to-back, doing the same sum-of-squares reduction twice and the
// same weight-multiply twice. This kernel saves the second pass.
//
// Layout: identical to `fused_rmsnorm_mq_rotate`.
//   - 1 workgroup of 256 threads per batch row (block.x = 256).
//   - Grid: [N, 1, 1] where N = batch_size (1 for decode).
//   - LDS:  `x_shared[K] + reduce[256]` (K ≤ 12288 fits in 64 KB).
//
// Phase 1 (RMSNorm pre-pass): identical to the rotate-only kernel.
//   - Stride through x, cache to LDS, accumulate local sum-of-squares.
//   - Block reduction → rms.
//   - In-place normalize: x_shared[i] = x[i] * weight[i] * rms.
//   - **NEW**: copy x_shared[] out to x_plain[] before phase 2.
//
// Phase 2 (FWHT rotation): identical to the rotate-only kernel.
//   - 8 waves of 32 threads, group-of-256 butterflies, ds_swizzle for the
//     stride-1-to-16 cross-wave hops, 1/16 scale, sign-flip with signs2.

__launch_bounds__(256, 1)
extern "C" __global__ void fused_rmsnorm_mq_rotate_plain(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,    // [N, K] FWHT-rotated
    float* __restrict__ x_plain,  // [N, K] plain RMSNormed
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

    const long long batch_off = (long long)blockIdx.x * K;
    const float* x_b       = x       + batch_off;
          float* x_rot_b   = x_rot   + batch_off;
          float* x_plain_b = x_plain + batch_off;

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

    // Phase 1c: normalize in place. x_shared now holds rmsnormalized
    // values; ALSO mirror them out to x_plain[] before phase 2 clobbers
    // x_shared via the butterfly writeback path. NB: x_rot writes go to
    // x_rot_b, NOT x_shared, so x_shared is in fact preserved through
    // phase 2 — but mirroring here means x_plain is ready as soon as the
    // following __syncthreads completes, with no extra LDS pressure.
    for (int i = tid; i < K; i += block_size) {
        float v = x_shared[i] * weight[i] * rms;
        x_shared[i] = v;
        x_plain_b[i] = v;
    }
    __syncthreads();

    // Phase 2: FWHT per group, identical to fused_rmsnorm_mq_rotate.
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
}
