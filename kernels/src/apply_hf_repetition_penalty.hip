// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void apply_hf_repetition_penalty_f32(
    float* __restrict__ logits,
    const unsigned int* __restrict__ unique_tokens,
    int unique_count,
    int vocab_size,
    float penalty
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= unique_count) return;

    const unsigned int token = unique_tokens[index];
    if (token >= static_cast<unsigned int>(vocab_size)) return;

    const float logit = logits[token];
    logits[token] = logit > 0.0f ? logit / penalty : logit * penalty;
}
