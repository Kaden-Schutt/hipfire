// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Fused RMSNorm + PARO4G128T (engine layout) per-group Givens rotation.
//
// Replaces the rmsnorm_f32 + paro4g128t_rotate kernel pair with a single
// launch. Mirrors fused_rmsnorm_mq_rotate.hip structure (whole-K LDS),
// but the rotation step uses PARO's pair/sincos Givens algorithm instead
// of the FWHT butterfly.
//
// Tensor byte layout (engine `--layout engine`, qtype 29 / PARO4G128T):
//   qweight:        int32 [K, M/8] or [M/8, K] (transposed engine layout)
//   qzeros:         int32 [K/128, M/8]
//   scales:         f16   [K/128, M]
//   pairs:          int16 [8, K]
//   sincos:         f32   [8, K/2, 2]    (PRECOMPUTED_TRIG=true engine layout)
//   channel_scales: f16   [K]
//
// Math identity with the separated `rmsnorm_f32 + paro4g128t_rotate` path:
//   x_norm = (x / rms(x)) * weight     (rmsnorm_f32)
//   xg[i]  = x_norm[i] * channel_scales[i]
//   for r in 0..8:
//     for each pair (i, j) at round r:
//       xg[i], xg[j] = c*xi + s*xj, c*xj - s*xi   (Givens rotation)
//   x_rot = xg
//
// The fused kernel folds rmsnorm into the rotate's gather pass, multiplying
// by `weight[i] * rms * channel_scales[i]` in one step before the Givens
// rounds. Numerically equivalent modulo float-mul reorder — output matches
// the separated path within FP16 epsilon on every K (tested in the example
// `bench_fused_rmsnorm_paro_rotate`).
//
// Layout / sizing:
//   grid  = [1, 1, 1]   (single workgroup per token; decode path)
//   block = [256, 1, 1] (8 wave32 warps)
//   smem  = (K + 256) * 4 bytes
//           x_shared[K] + reduce[256]
//
// Per arch the kernel is wave32. K must be a multiple of 128 (paro group size),
// and (K + 256) * 4 must fit in LDS (64 KiB on RDNA → K ≤ 16128).

// Lever 1 fused kernel.
//
// Outputs:
//   x_rot:  [K] rotated activation, consumed by the immediately following
//           gemv_paro4g128t_prerotated. Equivalent to the output of the
//           standalone `paro4g128t_rotate` kernel.
//   x_norm: [K] post-rmsnorm activation (pre-channel-scale, pre-rotation),
//           materialized so subsequent paro linears in the same residual
//           block can apply their OWN rotation (each linear has different
//           pairs/theta/channel_scales). If null, this write is skipped.

__launch_bounds__(256, 1)
extern "C" __global__ void fused_rmsnorm_paro4g128t_rotate(
    const char* __restrict__ A,           // PARO weights blob
    const float* __restrict__ x,          // [K] pre-rmsnorm input
    const float* __restrict__ weight,     // [K] rmsnorm weight
    float* __restrict__ x_rot,            // [K] output rotated activation
    float* __restrict__ x_norm,           // [K] output post-rmsnorm (may be null)
    int M,
    int K,
    float eps
) {
    extern __shared__ float smem[];
    float* x_shared = smem;            // [K]
    float* reduce   = smem + K;        // [256]

    constexpr int GROUP_SIZE = 128;
    constexpr int KROT = 8;
    constexpr int PACK = 8;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;     // 256
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;          // 0..7
    const int num_warps = block_size >> 5; // 8

    // Same offset math as paro4g128_rotate_impl<true (engine), true (precomp trig)>.
    const int groups = K / GROUP_SIZE;
    const int m_pack = M / PACK;
    const long long qweight_off = 0;
    const long long qzeros_off = qweight_off + (long long)K * m_pack * (long long)sizeof(int);
    const long long scales_off = qzeros_off + (long long)groups * m_pack * (long long)sizeof(int);
    const long long pairs_off = scales_off + (long long)groups * M * (long long)sizeof(_Float16);
    const long long theta_off = pairs_off + (long long)KROT * K * (long long)sizeof(short);
    // PRECOMPUTED_TRIG engine layout — theta_elems = KROT * K/2 * 2 floats.
    const long long channel_scales_off = theta_off + (long long)KROT * (K / 2) * 2LL * (long long)sizeof(float);

    const short* __restrict__ pairs = (const short*)(A + pairs_off);
    const float* __restrict__ trig = (const float*)(A + theta_off);          // sincos_f32
    const _Float16* __restrict__ channel_scales = (const _Float16*)(A + channel_scales_off);

    // Phase 1a: gather x → x_shared, accumulate per-thread sum-of-squares.
    float local_sum = 0.0f;
    for (int i = tid; i < K; i += block_size) {
        const float v = x[i];
        x_shared[i] = v;
        local_sum += v * v;
    }
    reduce[tid] = local_sum;
    __syncthreads();

    // Phase 1b: block reduction → global sum-of-squares → rms.
    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    const float rms = rsqrtf(reduce[0] / (float)K + eps);

    // Phase 1c: normalize, optionally emit pure x_norm, then fold
    // channel_scales for the rotation step.
    //
    // x_shared[i] := x[i] * weight[i] * rms                  (pure rmsnorm output)
    // x_norm[i]   := x_shared[i]                              (emit if non-null)
    // x_shared[i] := x_shared[i] * channel_scales[i]         (fold for rotation)
    //
    // Mirrors the original paro_rotate Phase 0:
    //   xg[lane] = (x_norm[lane]) * channel_scales[lane]
    for (int i = tid; i < K; i += block_size) {
        const float w_i = weight[i];
        const float xn = x_shared[i] * w_i * rms;
        x_shared[i] = xn;
    }
    __syncthreads();

    // Optional x_norm emit. Coalesced K-float write; skipped if null pointer.
    if (x_norm != nullptr) {
        for (int i = tid; i < K; i += block_size) {
            x_norm[i] = x_shared[i];
        }
    }

    // Phase 1d: fold channel_scales for rotation.
    for (int i = tid; i < K; i += block_size) {
        const float cs_i = (float)channel_scales[i];
        x_shared[i] *= cs_i;
    }
    __syncthreads();

    // Phase 2: PARO per-group Givens rotation, 8 rounds (KROT=8).
    //
    // Each warp owns `groups_per_warp` groups and processes them sequentially.
    // Within a warp, 32 lanes cooperatively apply 64 pair operations per
    // round (2 per lane). All warps __syncthreads() between rounds so that
    // each warp's group is in a consistent state before the next round
    // begins. The PARO calibration guarantees the 64 pair entries in one
    // round form a non-overlapping permutation of [0, 128) — no intra-round
    // race within a group.
    const int groups_per_warp = (groups + num_warps - 1) / num_warps;

    #pragma unroll 1
    for (int r = 0; r < KROT; ++r) {
        for (int gi = 0; gi < groups_per_warp; ++gi) {
            const int g = warp_id * groups_per_warp + gi;
            if (g >= groups) break;

            const int base = g * GROUP_SIZE;
            const int pair_base = r * K + base;
            const int theta_base = r * (K / 2) + g * (GROUP_SIZE / 2);

            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int pair_idx = lane_id + p * 32;
                const int i = (int)pairs[pair_base + 2 * pair_idx];
                const int j = (int)pairs[pair_base + 2 * pair_idx + 1];
                const int trig_idx = (theta_base + pair_idx) * 2;
                const float s = trig[trig_idx];
                const float c = trig[trig_idx + 1];
                const float xi = x_shared[base + i];
                const float xj = x_shared[base + j];
                x_shared[base + i] = fmaf(c, xi, s * xj);
                x_shared[base + j] = fmaf(c, xj, -s * xi);
            }
        }
        __syncthreads();
    }

    // Phase 3: stream x_shared → x_rot global memory.
    for (int i = tid; i < K; i += block_size) {
        x_rot[i] = x_shared[i];
    }
}
