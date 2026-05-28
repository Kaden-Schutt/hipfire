// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 MoE routing affinity activation: sqrt(softplus(x)) = sqrt(log(1 + exp(x))).
//
// Per DeepSeek_V4.pdf §2.1, DeepSeek V4 replaces V3's sigmoid affinity with
// sqrt(softplus(·)). Element-wise activation, in-place.
//
// Numerical stability: for large x, log(1 + exp(x)) ~ x. For very
// negative x, log(1 + exp(x)) ~ exp(x) → 0. Use the standard
// branched formulation to avoid overflow:
//   softplus(x) = x + log(1 + exp(-x))   if x > 0
//               = log(1 + exp(x))        if x <= 0

#include <hip/hip_runtime.h>
#include <math.h>

extern "C" __global__ void sqrt_softplus_f32(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float sp;
    if (v > 0.0f) {
        sp = v + logf(1.0f + expf(-v));
    } else {
        sp = logf(1.0f + expf(v));
    }
    x[i] = sqrtf(sp);
}
