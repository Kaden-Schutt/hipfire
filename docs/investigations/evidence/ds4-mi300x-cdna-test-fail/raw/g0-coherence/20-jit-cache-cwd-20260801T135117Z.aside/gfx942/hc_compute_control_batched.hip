// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// HC control-vector computation — BATCHED variant (Phase B2, 2026-05-18).
//
// Batched twin of `hc_compute_control`. Per batch position b:
//   c[b, ctrl] = (sum_d x_flat[b, d] * w_fn[ctrl, d]) * rsqrt(mean(x_flat[b]^2) + eps)
//              + base[ctrl]
//
// w_fn and base are shared across batch (model weights). x_flat is
// per-batch `[B, x_dim]`. Output c is `[B, n_ctrl]`.
//
// Grid:  [n_ctrl, batch_size, 1]
// Block: [256, 1, 1]
//
// Byte-identical to hc_compute_control at batch_size == 1.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define THREADS_PER_BLOCK 256

__launch_bounds__(THREADS_PER_BLOCK)
extern "C" __global__ void hc_compute_control_batched(
    const float*  __restrict__ x_flat,   // [B, x_dim]
    const __half* __restrict__ w_fn,     // [n_ctrl, x_dim]
    const __half* __restrict__ base,     // [n_ctrl]
    float*        __restrict__ c,         // [B, n_ctrl]
    int n_ctrl,
    int x_dim,
    int batch_size
) {
    const int ctrl = blockIdx.x;
    const int b    = blockIdx.y;
    if (ctrl >= n_ctrl || b >= batch_size) return;
    const int tid = threadIdx.x;

    const float* x_b = x_flat + (size_t)b * x_dim;

    float acc_dot = 0.0f;
    float acc_sq  = 0.0f;
    for (int d = tid; d < x_dim; d += THREADS_PER_BLOCK) {
        float xv = x_b[d];
        float wv = __half2float(w_fn[(size_t)ctrl * x_dim + d]);
        acc_dot += xv * wv;
        acc_sq  += xv * xv;
    }

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
        float rsqrt = rsqrtf(total_sq / (float)x_dim + 1e-6f);
        c[(size_t)b * n_ctrl + ctrl] = total_dot * rsqrt + __half2float(base[ctrl]);
    }
}
