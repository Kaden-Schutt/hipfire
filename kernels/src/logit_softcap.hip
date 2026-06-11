// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Logit soft-capping (Gemma 2/3/4 final-logit softcap): out[i] = tanh(x[i]/cap)*cap
// In-place over the flat logits vector. Gemma 4 has NO per-attention-layer softcap.
#include <hip/hip_runtime.h>
extern "C" __global__ void logit_softcap_f32(float* __restrict__ x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float inv_cap = 1.0f / cap;
    x[i] = tanhf(x[i] * inv_cap) * cap;
}
