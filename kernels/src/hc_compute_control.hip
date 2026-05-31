// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Hyper-Connections control-vector computation (DeepSeek V4 Phase 3).
//
// Upstream DeepSeek V4 (model.py:Block.hc_pre):
//   x = x.flatten(2).float()                                    # [hidden * hc_mult]
//   rsqrt = torch.rsqrt(x.square().mean(-1) + norm_eps)         # SCALAR
//   mixes = F.linear(x, hc_fn) * rsqrt                          # then + base in sinkhorn
//
// We fuse the (X·W) GEMV with the rsqrt + base-add into one launch:
//   1. Each block computes one output entry's dot product (X·W)_ctrl
//      AND accumulates sum-of-squares for the rsqrt in the same pass.
//   2. Reduce both, compute rsqrt = 1/sqrt(mean(x^2) + eps).
//   3. c[ctrl] = acc_dot * rsqrt + base[ctrl].
//
// DeepSeek V4 per-layer `hc_attn_fn` / `hc_ffn_fn` are [24, 16384] (n_ctrl=24,
// x_dim=hidden*hc_mult=16384). hc_base is [24] F16; output c is F32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define THREADS_PER_BLOCK 256

__launch_bounds__(THREADS_PER_BLOCK)
extern "C" __global__ void hc_compute_control(
    const float*  __restrict__ x_flat,   // [x_dim] F32 (DeepSeek V4 residual convention)
    const __half* __restrict__ w_fn,     // [n_ctrl, x_dim] F16 (HFQ-converted)
    const __half* __restrict__ base,     // [n_ctrl] F16
    float*        __restrict__ c,         // [n_ctrl] F32
    int n_ctrl,
    int x_dim
) {
    const int ctrl = blockIdx.x;
    if (ctrl >= n_ctrl) return;
    const int tid = threadIdx.x;

    // Compute BOTH dot product AND sum-of-squares in one pass.
    float acc_dot = 0.0f;
    float acc_sq  = 0.0f;
    for (int d = tid; d < x_dim; d += THREADS_PER_BLOCK) {
        float xv = x_flat[d];
        float wv = __half2float(w_fn[ctrl * x_dim + d]);
        acc_dot += xv * wv;
        acc_sq  += xv * xv;
    }

    // Wave-shuffle reduction (wave32 on gfx11xx).
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
        // DeepSeek V4 upstream: mixes = F.linear(x, hc_fn) * rsqrt(mean(x^2) + eps)
        float rsqrt = rsqrtf(total_sq / (float)x_dim + 1e-6f);
        c[ctrl] = total_dot * rsqrt + __half2float(base[ctrl]);
    }
}
