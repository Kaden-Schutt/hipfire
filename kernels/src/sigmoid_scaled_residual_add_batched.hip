// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — batched sigmoid-scaled residual add (generic f32).
//
// y[t,:] += sigmoid(scalars[t]) * x[t,:]   for t in [0, N), over `dim` cols.
//
// Used by the Q8 shared-expert down step of `prefill_moe_ffn_body_batched`:
// after a plain Q8 GEMM produces the shared-expert down output x = W_down·h
// into a [N × dim] temp, this folds it into the residual stream x_batch with
// the per-token sigmoid(shared_expert_gate logit) gate — the batched analog of
// the decode `scaled_add_inplace_gpu_scalar_f32` after `sigmoid_f32`. Arch-
// agnostic (pure f32 elementwise); only ever dispatched from the E8 MoE path.

#include <hip/hip_runtime.h>

extern "C" __global__ void sigmoid_scaled_residual_add_batched_f32(
    float* __restrict__ y,             // [N × dim] residual (in/out)
    const float* __restrict__ x,       // [N × dim] down output
    const float* __restrict__ scalars, // [N] pre-sigmoid logits
    int n, int dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * dim;
    if (idx >= total) return;
    const int t = idx / dim;
    const float s = scalars[t];
    const float sig = 1.0f / (1.0f + __expf(-s));
    y[idx] += sig * x[idx];
}
