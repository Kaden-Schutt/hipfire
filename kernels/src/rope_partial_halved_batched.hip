// Batched partial RoPE using the HuggingFace "rotate_half" pairing convention.
// See rope_partial_halved.hip for the math; this is the N-token batched
// counterpart used by Gemma 4 batched prefill (Phase 5). Reads positions
// from a device array (one i32 per token) so no per-token H2D is required.
//
// Layout: q [N × n_heads_q × head_dim], k [N × n_heads_k × head_dim].
// Grid: [n_rot_pairs blocks, batch]; threadIdx.x indexes pairs.

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_partial_halved_f32_batched(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,    // [batch] i32
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot_pairs,
    float freq_base,
    int batch
) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= n_rot_pairs) return;
    const int b = blockIdx.y;
    if (b >= batch) return;

    const int half = head_dim / 2;
    const int pos = positions[b];
    const float freq = 1.0f / powf(freq_base, (float)(2 * pair) / (float)head_dim);
    const float angle = (float)pos * freq;
    const float cos_a = cosf(angle);
    const float sin_a = sinf(angle);

    float* __restrict__ q_b = q + (long long)b * n_heads_q * head_dim;
    float* __restrict__ k_b = k + (long long)b * n_heads_k * head_dim;

    for (int h = 0; h < n_heads_q; h++) {
        const int base = h * head_dim;
        const float v0 = q_b[base + pair];
        const float v1 = q_b[base + pair + half];
        q_b[base + pair]        = v0 * cos_a - v1 * sin_a;
        q_b[base + pair + half] = v0 * sin_a + v1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; h++) {
        const int base = h * head_dim;
        const float v0 = k_b[base + pair];
        const float v1 = k_b[base + pair + half];
        k_b[base + pair]        = v0 * cos_a - v1 * sin_a;
        k_b[base + pair + half] = v0 * sin_a + v1 * cos_a;
    }
}
