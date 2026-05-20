#include <hip/hip_runtime.h>

// Batched partial RoPE, half-split convention (HF rotate_half) — see
// kernels/src/rope_partial_halfsplit.hip for the per-position twin and
// docs/plans/qwen35-mq4-quality-gap.md §"RoPE convention probe / halfsplit fix"
// for the rationale. Pairs are (i, i + n_rot/2). Dispatched from
// rope_partial_interleaved_f32_batched as the default since 2026-05-12;
// the interleaved batched kernel remains available for legacy reproducibility
// behind HIPFIRE_ROPE_INTERLEAVED_LEGACY=1.
//
// Grid: [ceil(n_rot/2 / block), batch_size, 1]
// Q/K are [batch_size × n_heads × head_dim] row-major.

extern "C" __global__ void rope_partial_halfsplit_batched_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    float freq_base,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int half = n_rot / 2;
    if (i >= half || b >= batch_size) return;

    int pos = positions[b];

    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int q_stride = n_heads_q * head_dim;
    int k_stride = n_heads_k * head_dim;

    for (int h = 0; h < n_heads_q; h++) {
        int base = b * q_stride + h * head_dim;
        float v0 = q[base + i];
        float v1 = q[base + i + half];
        q[base + i]        = v0 * cos_a - v1 * sin_a;
        q[base + i + half] = v0 * sin_a + v1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = b * k_stride + h * head_dim;
        float v0 = k[base + i];
        float v1 = k[base + i + half];
        k[base + i]        = v0 * cos_a - v1 * sin_a;
        k[base + i + half] = v0 * sin_a + v1 * cos_a;
    }
}
