#include <hip/hip_runtime.h>

// Fused L2-norm Q + L2-norm K + scale Q. Replaces the sequence
//
//   gpu.l2_norm_f32(&q_raw, n_heads, head_dim, eps);
//   gpu.l2_norm_f32(&k_raw, n_heads, head_dim, eps);
//   gpu.scale_f32(&q_raw, 1.0 / sqrt(head_dim));
//
// in the DeltaNet attention path. Three launches → one launch per layer.
// On Qwen3.5 0.8B/4B/9B this saves 2 launches × 18-24 linear-attention
// layers = 36-48 launches per forward, each costing ~9 µs of dispatch
// latency on gfx1100 → ~300 µs shaved off forward time.
//
// Grid: [n_heads, batch, 1]. Each block handles one head's Q then one
// head's K for a single batch row. Q gets normalized AND scaled. For
// decode grid.y=1 and blockIdx.y=0 (byte-identical to pre-batching);
// for prefill grid.y=N and q/k are laid out as [N × (n_heads*head_dim)]
// row-major.
//
// Uses wave-reduction over head_dim, same pattern as the single l2_norm
// kernel. head_dim must fit "reasonable" sizes (≤ 4096, easily).

extern "C" __global__ void fused_qk_l2_norm_scale_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    int n_heads,
    int head_dim,
    float q_scale,   // e.g. 1.0f / sqrtf((float)head_dim)
    float eps
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const long long batch_off = (long long)blockIdx.y * n_heads * head_dim;
    float* q_head = q + batch_off + h * head_dim;
    float* k_head = k + batch_off + h * head_dim;

    // ── Q: l2-normalize, then scale ──
    // IMPORTANT: the legacy un-fused path applied l2_norm and scale as two
    // independent in-place multiplications. Folding them into one
    // `q[i] *= rsqrt * q_scale` multiply changes the rounding order
    // (`(q*rsqrt)*scale` vs `q*(rsqrt*scale)`) which, over 2k+ greedy tokens,
    // picks a different argmax and breaks the quality gate. We keep the
    // two-multiply sequence to match the original bit-for-bit.
    float q_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        q_sq += q_head[i] * q_head[i];
    for (int o = 16; o > 0; o >>= 1)
        q_sq += __shfl_xor(q_sq, o);
    float q_inv_norm = rsqrtf(q_sq + eps);
    for (int i = tid; i < head_dim; i += 32)
        q_head[i] *= q_inv_norm;
    for (int i = tid; i < head_dim; i += 32)
        q_head[i] *= q_scale;

    // ── K: l2-normalize (no scale) ──
    float k_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        k_sq += k_head[i] * k_head[i];
    for (int o = 16; o > 0; o >>= 1)
        k_sq += __shfl_xor(k_sq, o);
    float k_inv_norm = rsqrtf(k_sq + eps);
    for (int i = tid; i < head_dim; i += 32)
        k_head[i] *= k_inv_norm;
}
