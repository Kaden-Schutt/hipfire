#include <hip/hip_runtime.h>

// Path 2 Stage 3 (down half): fan Y_down_grouped back to per-token
// x_residual with topk_weights applied.
//
// Pairs with `gemm_hfq4g256_moe_grouped_wmma_k2` invoked with
// `x_row_div = 1`, which writes `Y_down_grouped[m_total × dim]` —
// each slot row holds the down-projection of one routed (token,
// k_rank). This kernel inverts the scatter using `inverse_perm`:
//   for each (token, m_col):
//     acc = sum_k_rank in 0..K_TOP: topk_weights[token, k_rank] * Y_down_grouped[inverse_perm[token*K_TOP + k_rank], m_col]
//     x_residual[token, m_col] += acc
//
// Each (token, m_col) is owned by exactly one thread — no atomic
// contention. Companion to Path 1's `moe_down_combine_k8_batched`,
// which used a non-permuted `[N × K_TOP × dim]` source instead of the
// grouped layout.
//
// Shapes:
//   Y_down_grouped : [m_total × dim] f32
//   inverse_perm   : [N * K_TOP] i32 (flat → sorted_pos in Y_down_grouped)
//   topk_weights   : [N × K_TOP] f32
//   x_residual     : [N × dim] f32 (in-place +=)
//
// Grid: (ceil(dim/256), N). Block: [256]. LDS: 0.

extern "C" __global__ void moe_down_combine_grouped_k8(
    const float* __restrict__ Y_down_grouped,
    const int*   __restrict__ inverse_perm,
    const float* __restrict__ topk_weights,
    float*       __restrict__ x_residual,
    int dim, int K_TOP
) {
    const int token = blockIdx.y;
    const int m_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (m_col >= dim) return;

    const int weights_base = token * K_TOP;
    const int flat_base    = token * K_TOP;

    // K_TOP = 8 in production. Hard-coded unroll so the compiler can
    // overlap the 8 (inverse_perm, Y, weights) loads. The k < K_TOP
    // guard preserves correctness if a future caller picks K_TOP < 8.
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < K_TOP) {
            const int slot = inverse_perm[flat_base + k];
            const float w  = topk_weights[weights_base + k];
            const float v  = Y_down_grouped[(long long)slot * dim + m_col];
            acc += w * v;
        }
    }

    x_residual[(long long)token * dim + m_col] += acc;
}
