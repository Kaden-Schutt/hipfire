// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Mamba-2 RMSNormGated (nemotron_h `MambaRMSNormGated`) — multi-token PREFILL
// form of `mamba2_gated_norm_f32`. GATE then GROUP-RMSNorm, per position. Each
// position is independent (no recurrence), so this is the decode kernel with a
// position dimension on the grid: one block per (position, group). y/z/out are
// [seq * d_inner] (position-major); weight is [d_inner] (shared across
// positions). One launch instead of `seq` decode launches.
//
//   gated = y * silu(z)
//   out[g] = gated[g] * rsqrt(mean(gated[g]^2) + eps) * weight[g]   (per group)

#include <hip/hip_runtime.h>

extern "C" __global__ void mamba2_gated_norm_seq_f32(
    float* __restrict__ out,          // [seq * d_inner]
    const float* __restrict__ y,      // [seq * d_inner]
    const float* __restrict__ z,      // [seq * d_inner]
    const float* __restrict__ weight, // [d_inner] (= n_groups * group_size)
    int seq,
    int n_groups,
    int group_size,
    float eps
) {
    int g = blockIdx.x;
    int p = blockIdx.y;
    if (g >= n_groups || p >= seq) return;
    int tid = threadIdx.x;
    int base = (p * n_groups + g) * group_size; // into [seq * d_inner]
    int wbase = g * group_size;                 // into [d_inner]

    __shared__ float sh[256];
    float ss = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        float yi = y[base + i];
        float zi = z[base + i];
        float gated = yi * (zi / (1.0f + expf(-zi))); // y * silu(z)
        ss += gated * gated;
    }
    sh[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sh[0] / (float)group_size + eps);

    for (int i = tid; i < group_size; i += blockDim.x) {
        float yi = y[base + i];
        float zi = z[base + i];
        float gated = yi * (zi / (1.0f + expf(-zi)));
        out[base + i] = gated * inv_rms * weight[wbase + i];
    }
}
