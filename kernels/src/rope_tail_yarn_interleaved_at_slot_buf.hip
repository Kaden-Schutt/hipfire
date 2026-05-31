// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// In-place YaRN-aware tail RoPE at a device-buffer-indexed slot of a base
// buffer. HIP-graphs-safe twin of `rope_tail_yarn_interleaved` applied to
// a `kv_cache.sub_offset(slot*head_dim, ...)` view. Slot index comes from
// `slot_buf[0]`; -1 sentinel → no-op.
//
// Single-tensor convention: rotates one row of length `head_dim` at
// `base + slot * head_dim`. Caller drops `n_heads_q`/`n_heads_k` (they
// are always 1/0 for the compressor commit path).
//
// Set `freq_scale=1.0, ext_factor=0.0, attn_factor=1.0, corr_low=0.0,
// corr_high=0.0` to recover plain `rope_tail_interleaved` semantics
// (indexer compressor case). Set the DeepSeek V4 per-layer YaRN constants for
// the main compressor case.

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_tail_yarn_interleaved_at_slot_buf_f32(
    float* __restrict__ base,           // [max_slots, head_dim]
    const int*   __restrict__ pos_buf,  // [N] — read pos_buf[0]
    const int*   __restrict__ slot_buf, // [1] — -1 sentinel = no-op
    int head_dim,
    int n_rot,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float corr_low,
    float corr_high
) {
    const int slot = slot_buf[0];
    if (slot < 0) return;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n_rot / 2;
    if (i >= half) return;

    int pos = pos_buf[0];
    int idx_pair = i * 2;

    float freq = 1.0f / powf(freq_base, (float)idx_pair / (float)n_rot);
    float theta_extrap = (float)pos * freq;
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;

    if (ext_factor != 0.0f) {
        float y = ((float)i - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        float ramp_mix = ramp * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    float cos_a = cosf(theta) * mscale;
    float sin_a = sinf(theta) * mscale;

    int tail_off = head_dim - n_rot;
    int d_even = tail_off + idx_pair;
    int d_odd  = d_even + 1;

    float* row = base + (size_t)slot * head_dim;
    float x0 = row[d_even];
    float x1 = row[d_odd];
    row[d_even] = x0 * cos_a - x1 * sin_a;
    row[d_odd]  = x0 * sin_a + x1 * cos_a;
}
