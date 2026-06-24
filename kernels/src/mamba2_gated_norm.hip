// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Mamba-2 RMSNormGated (nemotron_h `MambaRMSNormGated`, norm_before_gate=False):
// GATE then GROUP-RMSNorm. Distinct from qwen35's gated_norm.hip (which is
// norm-then-gate, per-head). Confirmed from modeling_nemotron_h.py:
//   gated = y * silu(z)                         # gate first
//   per group g of `group_size` elements (group_size = d_inner / n_groups):
//     out[g] = gated[g] * rsqrt(mean(gated[g]^2) + eps) * weight[g]
// `y`/`z`/`out`/`weight` are length n_groups*group_size (= d_inner). One block
// per group; weight is per-element (length d_inner).

#include <hip/hip_runtime.h>

extern "C" __global__ void mamba2_gated_norm_f32(
    float* __restrict__ out,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    int n_groups,
    int group_size,
    float eps
) {
    int g = blockIdx.x;
    if (g >= n_groups) return;
    int tid = threadIdx.x;
    int base = g * group_size;

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
        out[base + i] = gated * inv_rms * weight[base + i];
    }
}
