#include <hip/hip_runtime.h>

// Fused L2-norm(Q) + scale(Q) + L2-norm(K) + repeat-interleave(Q,K).
//
// Replaces the back-to-back `fused_qk_l2_norm_scale_f32_batched` +
// `repeat_interleave_qk_f32_batched` calls in the LA prefill body
// (qwen35.rs:4809-4824). One launch instead of two — saves ~200µs per
// MoE layer × 30 LA layers ≈ 6ms per A3B prefill.
//
// Algorithm: per (key_head, batch) block, compute the L2 norm of q and
// k for the source key head, apply q_scale to q, then write all
// `ratio` destination value-head slots in q_dst/k_dst with the
// normalized (and for q, scaled) values. The norm reduction runs once
// per source key head — saved O(ratio) redundant reduction work
// relative to a per-dst-head approach.
//
// Shapes:
//   q_src      : [N × n_key_heads × head_dim] f32 (read; also valid to
//                                                  alias with q_dst when
//                                                  the source layout is
//                                                  the same as dst — not
//                                                  here, dst is wider)
//   k_src      : [N × n_key_heads × head_dim] f32
//   q_dst      : [N × n_value_heads × head_dim] f32 (written; n_value =
//                                                   n_key * ratio)
//   k_dst      : [N × n_value_heads × head_dim] f32 (written)
//
// Grid: [n_key_heads, N, 1]. Block: [32] (wave32 reduction).

extern "C" __global__ void fused_qk_l2_norm_scale_interleave_f32_batched(
    const float* __restrict__ q_src,
    const float* __restrict__ k_src,
    float*       __restrict__ q_dst,
    float*       __restrict__ k_dst,
    int n_key_heads,
    int ratio,
    int head_dim,
    float q_scale,
    float eps
) {
    const int kh = blockIdx.x;
    if (kh >= n_key_heads) return;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;

    const long long src_batch_off = (long long)b * n_key_heads * head_dim;
    const float* q_head = q_src + src_batch_off + kh * head_dim;
    const float* k_head = k_src + src_batch_off + kh * head_dim;

    // ── L2 norm of Q (single reduction, wave32) ──
    float q_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        q_sq += q_head[i] * q_head[i];
    for (int o = 16; o > 0; o >>= 1)
        q_sq += __shfl_xor(q_sq, o);
    float q_inv_norm = rsqrtf(q_sq + eps);

    // ── L2 norm of K (single reduction) ──
    float k_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        k_sq += k_head[i] * k_head[i];
    for (int o = 16; o > 0; o >>= 1)
        k_sq += __shfl_xor(k_sq, o);
    float k_inv_norm = rsqrtf(k_sq + eps);

    // ── Replicate-and-write across `ratio` value-head slots ──
    // Match the legacy q[i] *= inv_norm; q[i] *= scale order — the two
    // multiplies aren't associative under FP32 and combining into one
    // breaks the quality-gated byte-identity vs the un-fused path
    // (per the comment in fused_qk_l2_norm_scale.hip:43).
    const int n_value_heads = n_key_heads * ratio;
    const long long dst_batch_off = (long long)b * n_value_heads * head_dim;
    for (int r = 0; r < ratio; r++) {
        const int vh = kh * ratio + r;
        float* q_out = q_dst + dst_batch_off + vh * head_dim;
        float* k_out = k_dst + dst_batch_off + vh * head_dim;
        for (int i = tid; i < head_dim; i += 32) {
            float qv = q_head[i] * q_inv_norm;
            qv *= q_scale;
            q_out[i] = qv;
            k_out[i] = k_head[i] * k_inv_norm;
        }
    }
}
