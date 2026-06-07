// Partial RoPE using the HuggingFace "rotate_half" pairing convention.
//
// This is the Gemma 4 full-attention layer RoPE: head_dim=512 but only the
// first `n_rot_pairs` pairs rotate (pair i = dim i paired with dim i+head_dim/2),
// the rest of the head (pairs n_rot_pairs..head_dim/2) are NoPE (pass-through).
//
// Matches transformers' `_compute_proportional_rope_parameters`:
//   rope_angles = int(partial_rotary_factor * head_dim / 2)   # = n_rot_pairs
//   inv_freq_rotated[i] = 1 / base^(2i/head_dim)              for i in [0, rope_angles)
//   inv_freq[i] = 0                                            for i in [rope_angles, head_dim/2)
//
// and its application via rotate_half:
//   new_x[i]            = x[i]            * cos(pos·inv_freq[i]) - x[i+d/2] * sin(pos·inv_freq[i])
//   new_x[i+head_dim/2] = x[i+head_dim/2] * cos(pos·inv_freq[i]) + x[i]     * sin(pos·inv_freq[i])
//
// For pairs with inv_freq[i]=0 (i >= n_rot_pairs), cos=1 / sin=0, i.e. identity — we
// skip those with an early-return.
//
// Distinct from `rope_partial_interleaved`:
//   • Interleaved pairs (2i, 2i+1); this one pairs (i, i+head_dim/2).
//   • Interleaved uses `n_rot` as the count of rotated DIMS (128 for Qwen3.5);
//     this kernel uses `n_rot_pairs` as the count of rotated PAIRS (64 for Gemma 4 full).

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_partial_halved_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf,  // device buffer, single i32 — graph-safe
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot_pairs,    // number of pairs (i, i+head_dim/2) to rotate
    float freq_base
) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= n_rot_pairs) return;

    const int half = head_dim / 2;
    const int pos = *pos_buf;
    // `(2 * pair) / head_dim` — the division uses the FULL head_dim (not the
    // rotated portion), per the proportional-RoPE spec.
    const float freq = 1.0f / powf(freq_base, (float)(2 * pair) / (float)head_dim);
    const float angle = (float)pos * freq;
    const float cos_a = cosf(angle);
    const float sin_a = sinf(angle);

    for (int h = 0; h < n_heads_q; h++) {
        const int base = h * head_dim;
        const float v0 = q[base + pair];
        const float v1 = q[base + pair + half];
        q[base + pair]        = v0 * cos_a - v1 * sin_a;
        q[base + pair + half] = v0 * sin_a + v1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; h++) {
        const int base = h * head_dim;
        const float v0 = k[base + pair];
        const float v1 = k[base + pair + half];
        k[base + pair]        = v0 * cos_a - v1 * sin_a;
        k[base + pair + half] = v0 * sin_a + v1 * cos_a;
    }
}
