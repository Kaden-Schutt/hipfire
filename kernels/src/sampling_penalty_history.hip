// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Apply Hipfire's sampling penalties to one or more speculative-distribution
// rows. Row r sees the committed generation history plus the first
// `prefix_base + r * prefix_step` tokens from `prefix_tokens`. This is the
// history an autoregressive sampler would see at that exact speculative
// position.
//
// `full_to_local` maps full-vocabulary token IDs into a compressed draft
// vocabulary. Pass map_vocab_size=0 for an identity (full-vocab) mapping.
#include <hip/hip_runtime.h>

extern "C" __global__ void sampling_penalty_history_f32(
    float* __restrict__ logits,
    const unsigned int* __restrict__ history_tokens,
    const int* __restrict__ history_len_ptr,
    const unsigned int* __restrict__ prefix_tokens,
    const int* __restrict__ full_to_local,
    int map_vocab_size,
    int vocab_size,
    int prefix_base,
    int prefix_step,
    int repeat_window,
    float repeat_penalty,
    float presence_penalty,
    float frequency_penalty
) {
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int nthreads = (int)blockDim.x;
    const int history_len = max(*history_len_ptr, 0);
    const int prefix_len = max(prefix_base + row * prefix_step, 0);
    const int combined_len = history_len + prefix_len;
    const int window_len = min(max(repeat_window, 0), combined_len);
    if (window_len <= 0) return;

    const int window_start = combined_len - window_len;
    float* row_logits = logits + (size_t)row * (size_t)vocab_size;

    // Match sample_apply_repeat_penalty: only the newest occurrence of a
    // token writes its logit, while its scan counts every occurrence in the
    // effective oldest-to-newest window.
    for (int rel = tid; rel < window_len; rel += nthreads) {
        const int combined_i = window_start + rel;
        const unsigned int tok = combined_i < history_len
            ? history_tokens[combined_i]
            : prefix_tokens[combined_i - history_len];

        bool has_later = false;
        int count = 1;
        for (int rel_j = 0; rel_j < window_len; ++rel_j) {
            if (rel_j == rel) continue;
            const int combined_j = window_start + rel_j;
            const unsigned int other = combined_j < history_len
                ? history_tokens[combined_j]
                : prefix_tokens[combined_j - history_len];
            if (other == tok) {
                if (rel_j > rel) {
                    has_later = true;
                    break;
                }
                ++count;
            }
        }
        if (has_later) continue;

        int local_tok;
        if (map_vocab_size > 0) {
            if ((int)tok >= map_vocab_size) continue;
            local_tok = full_to_local[tok];
        } else {
            local_tok = (int)tok;
        }
        if (local_tok < 0 || local_tok >= vocab_size) continue;

        if (repeat_penalty > 1.0f) {
            const float recency = ((float)rel + 1.0f) / (float)window_len;
            float effective = powf(repeat_penalty, (float)count * recency);
            effective = min(effective, 1.5f);
            const float v = row_logits[local_tok];
            row_logits[local_tok] = v > 0.0f ? v / effective : v * effective;
        }
        if (presence_penalty > 0.0f || frequency_penalty > 0.0f) {
            row_logits[local_tok] -=
                frequency_penalty * (float)count + presence_penalty;
        }
    }
}
