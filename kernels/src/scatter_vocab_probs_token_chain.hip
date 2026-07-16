// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Embed one compressed-vocabulary probability row into a zero-filled
// full-vocabulary row and remap its sampled index into the resident MTP token
// chain. vocab_map is injective for the compressed draft sidecar.
extern "C" __global__ void scatter_vocab_probs_token_chain_f32(
    const float* __restrict__ compact_probs,
    float* __restrict__ full_probs,
    const unsigned int* __restrict__ vocab_map,
    const int* __restrict__ sampled_compact_token,
    int* __restrict__ token_chain,
    int compact_vocab,
    int dst_slot
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < compact_vocab) {
        full_probs[vocab_map[i]] = compact_probs[i];
    }
    if (i == 0) {
        token_chain[dst_slot] = (int)vocab_map[sampled_compact_token[0]];
    }
}
