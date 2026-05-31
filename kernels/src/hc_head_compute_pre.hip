// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 head HC mix — compute the per-stream pre weights.
//
// Upstream DeepSeek V4 (model.py:ParallelHead.hc_head):
//   x_flat = streams.flatten()                                  # [hc_mult * hidden]
//   rsqrt = torch.rsqrt(x_flat.square().mean(-1) + norm_eps)
//   mixes = F.linear(x_flat, hc_head_fn) * rsqrt                # [hc_mult]
//   pre = sigmoid(mixes * hc_head_scale + hc_head_base) + hc_eps  # [hc_mult]
//
// Then: y[d] = sum_h pre[h] * streams[h, d]  (uses hc_input_map_4stream).
//
// This kernel computes pre[hc_mult] from streams. One block per output
// entry. Each block does its own rsqrt computation (cheap given x_dim=
// 16384 vs hc_mult=4 blocks). Saves d2h sync vs computing rsqrt on host.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define THREADS_PER_BLOCK 256

__launch_bounds__(THREADS_PER_BLOCK)
extern "C" __global__ void hc_head_compute_pre(
    const float*  __restrict__ x_flat,   // [x_dim = hc_mult * hidden] F32
    const __half* __restrict__ w_fn,     // [hc_mult, x_dim] F16
    const __half* __restrict__ base,     // [hc_mult] F16
    float*        __restrict__ pre,      // [hc_mult] F32 output
    int hc_mult,
    int x_dim,
    float scale,                          // hc_head_scale (scalar)
    float norm_eps,
    float hc_eps                          // eps added after sigmoid
) {
    const int h = blockIdx.x;
    if (h >= hc_mult) return;
    const int tid = threadIdx.x;

    // Accumulate dot product AND sum-of-squares in one pass.
    float acc_dot = 0.0f;
    float acc_sq  = 0.0f;
    for (int d = tid; d < x_dim; d += THREADS_PER_BLOCK) {
        float xv = x_flat[d];
        float wv = __half2float(w_fn[h * x_dim + d]);
        acc_dot += xv * wv;
        acc_sq  += xv * xv;
    }

    // Wave-shuffle reduction (wave32).
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_dot += __shfl_down(acc_dot, offset);
        acc_sq  += __shfl_down(acc_sq,  offset);
    }

    __shared__ float partial_dot[THREADS_PER_BLOCK / 32];
    __shared__ float partial_sq [THREADS_PER_BLOCK / 32];
    if (tid % 32 == 0) {
        partial_dot[tid / 32] = acc_dot;
        partial_sq [tid / 32] = acc_sq;
    }
    __syncthreads();
    if (tid == 0) {
        float total_dot = 0.0f;
        float total_sq  = 0.0f;
        for (int i = 0; i < THREADS_PER_BLOCK / 32; ++i) {
            total_dot += partial_dot[i];
            total_sq  += partial_sq [i];
        }
        float rsqrt = rsqrtf(total_sq / (float)x_dim + norm_eps);
        float mixes_h = total_dot * rsqrt;
        float score = mixes_h * scale + __half2float(base[h]);
        pre[h] = 1.0f / (1.0f + __expf(-score)) + hc_eps;
    }
}
