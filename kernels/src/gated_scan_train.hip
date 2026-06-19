// SPDX-License-Identifier: Apache-2.0
// hipfire — gated linear-recurrence scan (fp32 training), the token-mixer for the
// GLA-lite / minimal-selective-SSM PFlash drafter.
//
//   h[t] = g[t] * h[t-1] + (1 - g[t]) * u[t],   h[-1] = 0
//
// g (input-dependent forget gate in (0,1)) is the selectivity; u is the input
// projection. Per-channel diagonal recurrence — one thread per channel, sequential
// over time, NO shared memory (gfx1103 LDS kernels wedge the GPU; see
// project-gfx1103-lds-gpu-reset / gemm_f32_train header). g,u,h: [seq * D].

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(256)
__global__ void gated_scan_fwd(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ h_out,
    int seq,
    int D
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float h = 0.0f;
    for (int t = 0; t < seq; t++) {
        size_t i = (size_t)t * D + c;
        float gt = g[i];
        h = gt * h + (1.0f - gt) * u[i];
        h_out[i] = h;
    }
}

// Backward. Given d_hout = dL/dh[t] for every t, produce d_g, d_u. Reverse scan:
// dL/dh[t] = d_hout[t] + g[t+1]*dL/dh[t+1]; then
//   d_u[t] = dL/dh[t] * (1 - g[t])
//   d_g[t] = dL/dh[t] * (h[t-1] - u[t])
extern "C" __launch_bounds__(256)
__global__ void gated_scan_bwd(
    const float* __restrict__ g,
    const float* __restrict__ u,
    const float* __restrict__ h_out,
    const float* __restrict__ d_hout,
    float* __restrict__ d_g,
    float* __restrict__ d_u,
    int seq,
    int D
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float dh = 0.0f; // dL/dh[t], accumulated from downstream
    for (int t = seq - 1; t >= 0; t--) {
        size_t i = (size_t)t * D + c;
        dh += d_hout[i];
        float gt = g[i];
        float h_prev = (t > 0) ? h_out[i - D] : 0.0f;
        float ut = u[i];
        d_u[i] = dh * (1.0f - gt);
        d_g[i] = dh * (h_prev - ut);
        dh = dh * gt; // propagate to dL/dh[t-1]
    }
}
