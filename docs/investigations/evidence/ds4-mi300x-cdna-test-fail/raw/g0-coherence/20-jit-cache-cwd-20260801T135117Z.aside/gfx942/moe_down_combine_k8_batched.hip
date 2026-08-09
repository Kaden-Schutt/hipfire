#include <hip/hip_runtime.h>

// Combine step for the atomic-free MoE down path. Sums over K_TOP
// per-(token, krank) expert outputs weighted by topk_weights, then
// adds the result into the per-token residual buffer `x_residual`.
//
// Pairs with `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` — that
// kernel writes [N × K_TOP × M] into `expert_outputs` with no atomic
// contention; this kernel folds the K_TOP slots into [N × M] with the
// topk_weights scale applied, then accumulates into x_residual.
//
// Shapes:
//   expert_outputs : [N × K_TOP × M] (f32)
//   topk_weights   : [N × K_TOP]     (f32)
//   x_residual     : [N × M]         (f32, in-place += result)
//
// Each token writes to its own M-column slice of x_residual, so no
// cross-token atomicAdd is needed. Each thread owns one output cell.
//
// Grid: ((M + BLOCK_M - 1) / BLOCK_M, N). Block: BLOCK_M = 256 threads.

extern "C" __global__ void moe_down_combine_k8_batched(
    const float* __restrict__ expert_outputs, // [N × K_TOP × M]
    const float* __restrict__ topk_weights,   // [N × K_TOP]
    float*       __restrict__ x_residual,     // [N × M] accumulated in place
    int M, int K_TOP
) {
    const int bid = blockIdx.y;          // token id
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // m column
    if (col >= M) return;

    const long long token_base   = (long long)bid * K_TOP * M;
    const int       weights_base = bid * K_TOP;
    const long long residual_off = (long long)bid * M + col;

    // K_TOP is statically 8 per the indexed-batched kernel family contract.
    // Hard-coding the unroll lets the compiler issue 8 independent loads of
    // (expert_outputs, topk_weights) without a loop branch, and keeps the
    // accumulator chain short for FP-stable reductions.
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < K_TOP) {
            float w  = topk_weights[weights_base + k];
            float v  = expert_outputs[token_base + (long long)k * M + col];
            acc += w * v;
        }
    }

    x_residual[residual_off] += acc;
}
