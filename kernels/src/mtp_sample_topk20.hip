// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Apply repeat/presence/frequency penalties to one compressed draft-logit row.
// `base_tokens[0..base_len)` is the committed generated-token history already
// mapped into compressed-vocabulary indices. `draft_tokens[0..draft_prefix)`
// is the device-resident speculative prefix produced earlier in this graph.
// The combined sequence is clipped to its most recent `repeat_window` entries,
// exactly like the host prepass used before this graph path.
extern "C" __global__ void mtp_apply_chain_penalties_f32(
    float* __restrict__ logits,
    const unsigned int* __restrict__ base_tokens,
    const int* __restrict__ base_len_ptr,
    const int* __restrict__ draft_tokens,
    int draft_prefix,
    int vocab_size,
    int repeat_window,
    float repeat_penalty,
    float presence_penalty,
    float frequency_penalty
) {
    int base_len = max(0, base_len_ptr[0]);
    int combined = base_len + draft_prefix;
    int window = min(max(0, repeat_window), combined);
    if (window == 0) return;
    int start = combined - window;

    for (int wi = (int)threadIdx.x; wi < window; wi += (int)blockDim.x) {
        int src = start + wi;
        unsigned int tok = src < base_len
            ? base_tokens[src]
            : (unsigned int)draft_tokens[src - base_len];
        if ((int)tok >= vocab_size) continue;

        bool has_later = false;
        int count = 1;
        for (int wj = 0; wj < window; ++wj) {
            if (wj == wi) continue;
            int other_src = start + wj;
            unsigned int other = other_src < base_len
                ? base_tokens[other_src]
                : (unsigned int)draft_tokens[other_src - base_len];
            if (other == tok) {
                if (wj > wi) {
                    has_later = true;
                    break;
                }
                ++count;
            }
        }
        if (has_later) continue;

        if (repeat_penalty > 1.0f) {
            float recency = ((float)wi + 1.0f) / (float)window;
            float effective = powf(repeat_penalty, (float)count * recency);
            effective = min(effective, 1.5f);
            float value = logits[tok];
            logits[tok] = value > 0.0f ? value / effective : value * effective;
        }
        if (presence_penalty > 0.0f || frequency_penalty > 0.0f) {
            logits[tok] -= presence_penalty + frequency_penalty * (float)count;
        }
    }
}

// Sample exactly from top-k -> temperature -> min-p -> top-p. The top-k
// indices/values are already sorted descending by `topk20_batched_f32`.
// The selected compressed index feeds both the next proposal's penalty history
// and, through vocab_map, the full-token device chain consumed by embedding.
extern "C" __global__ void mtp_sample_topk20_token_chain_f32(
    const int* __restrict__ top_indices,
    const float* __restrict__ top_values,
    int top_k,
    float temperature,
    float top_p,
    float min_p,
    const float* __restrict__ uniforms,
    int uniform_slot,
    int* __restrict__ draft_indices,
    float* __restrict__ draft_probabilities,
    int* __restrict__ token_chain,
    int token_slot,
    const unsigned int* __restrict__ vocab_map
) {
    if (threadIdx.x != 0) return;

    double weights[20];
    double max_value = (double)top_values[0];
    double inv_temp = 1.0 / (double)temperature;
    double mass = 0.0;
    for (int i = 0; i < top_k; ++i) {
        weights[i] = exp(((double)top_values[i] - max_value) * inv_temp);
        mass += weights[i];
    }
    for (int i = 0; i < top_k; ++i) weights[i] /= mass;

    int retained = top_k;
    if (min_p > 0.0f) {
        double threshold = (double)min_p * weights[0];
        for (int i = 1; i < retained; ++i) {
            if (weights[i] < threshold) {
                retained = i;
                break;
            }
        }
        mass = 0.0;
        for (int i = 0; i < retained; ++i) mass += weights[i];
        for (int i = 0; i < retained; ++i) weights[i] /= mass;
    }

    if (top_p < 1.0f) {
        double cumulative = 0.0;
        for (int i = 0; i < retained; ++i) {
            cumulative += weights[i];
            if (cumulative >= (double)top_p) {
                retained = i + 1;
                break;
            }
        }
        mass = 0.0;
        for (int i = 0; i < retained; ++i) mass += weights[i];
        for (int i = 0; i < retained; ++i) weights[i] /= mass;
    }

    double uniform = min(max((double)uniforms[uniform_slot], 0.0), 0.9999999999999999);
    double cumulative = 0.0;
    int picked = retained - 1;
    for (int i = 0; i < retained; ++i) {
        cumulative += weights[i];
        if (uniform < cumulative) {
            picked = i;
            break;
        }
    }

    int draft_index = top_indices[picked];
    draft_indices[uniform_slot] = draft_index;
    draft_probabilities[uniform_slot] = (float)weights[picked];
    token_chain[token_slot] = vocab_map ? (int)vocab_map[draft_index] : draft_index;
}
