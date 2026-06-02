// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched partial/interleaved RoPE. Counterpart of rope_partial_interleaved_f32
// for prefill batching — each batch row has its own absolute position read
// from a positions[] array.
//
// Grid: [ceil(n_rot/2 / block), batch_size, 1]
// Each thread handles one (pair, batch) combination: rotates dims (2p, 2p+1)
// of every Q and K head for batch row b, using angle = positions[b] * freq.
//
// Byte-exact with rope_partial_interleaved_f32 for batch_size=1 + pos=positions[0].
// Q/K are [batch_size × n_heads × head_dim] row-major.

extern "C" __global__ void rope_partial_interleaved_batched_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,          // number of dims to rotate (64 for Qwen3.5)
    float freq_base,
    int batch_size,
    int pos_offset      // added to positions[b] for the RoPE angle (NOT the KV slot);
                        // carries kv_cache.compact_offset so post-eviction batched
                        // Q/K rotate at absolute phase matching cached keys. 0 = no-op.
) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int n_pairs = n_rot / 2;
    if (pair >= n_pairs || b >= batch_size) return;

    int pos = positions[b] + pos_offset;

    // Interleaved: pair i rotates dims (2*i, 2*i+1)
    int d0 = pair * 2;
    int d1 = pair * 2 + 1;

    float freq = 1.0f / powf(freq_base, (float)(2 * pair) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int q_stride = n_heads_q * head_dim;
    int k_stride = n_heads_k * head_dim;

    for (int h = 0; h < n_heads_q; h++) {
        int base = b * q_stride + h * head_dim;
        float v0 = q[base + d0];
        float v1 = q[base + d1];
        q[base + d0] = v0 * cos_a - v1 * sin_a;
        q[base + d1] = v0 * sin_a + v1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = b * k_stride + h * head_dim;
        float v0 = k[base + d0];
        float v1 = k[base + d1];
        k[base + d0] = v0 * cos_a - v1 * sin_a;
        k[base + d1] = v0 * sin_a + v1 * cos_a;
    }
}
