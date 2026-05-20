#include <hip/hip_runtime.h>

// Partial RoPE, half-split convention (HF / GGUF / rotate_half style).
//
// Rotates only the first `n_rot` dims of each head. Pairs are (i, i + n_rot/2)
// — matches HF Qwen2 / Qwen3 / Qwen3.5 `apply_rotary_pos_emb` which uses
// `rotate_half`:
//     q_embed = q * cos + rotate_half(q) * sin
//     where rotate_half(x) = concat(-x[..., half:], x[..., :half])
//
// Dims [n_rot, head_dim) are pass-through.
//
// Twin of `rope_partial_interleaved_f32` for arches that store Q/K weights in
// HF's native half-split layout (Qwen3.5 family — see
// docs/plans/qwen35-mq4-quality-gap.md §"RoPE convention probe"). The
// interleaved variant remains available for arches whose Q/K weight loader
// already permutes half-split storage into interleaved layout at quantize time.
extern "C" __global__ void rope_partial_halfsplit_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    float freq_base
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n_rot / 2;
    if (i >= half) return;

    int pos = *pos_buf;
    // Frequency for dim i — same indexing as HF's compute_default_rope_parameters,
    // restricted to the partial-rotary region (n_rot = head_dim * partial_rotary_factor).
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    for (int h = 0; h < n_heads_q; h++) {
        int base = h * head_dim;
        float q0 = q[base + i];
        float q1 = q[base + i + half];
        q[base + i]        = q0 * cos_a - q1 * sin_a;
        q[base + i + half] = q0 * sin_a + q1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = h * head_dim;
        float k0 = k[base + i];
        float k1 = k[base + i + half];
        k[base + i]        = k0 * cos_a - k1 * sin_a;
        k[base + i + half] = k0 * sin_a + k1 * cos_a;
    }
}
