#include <hip/hip_runtime.h>

// Phase 1 of the SGLang-style MoE scatter pipeline. Computes the raw
// per-expert histogram over the flattened topk_indices stream.
//
// Single workgroup; LDS atomicAdd on `num_experts` counters (256 for A3B
// = 1 KiB LDS). LDS atomics avoid the global-memory atomic traffic that
// a naïve global-histogram path would issue. The final write-back is a
// coalesced store, so global BW pressure is O(num_experts) ints.
//
// Pairs with `moe_scatter_offsets_k8` (Phase 2) and
// `moe_scatter_permute_k8` (Phase 3). See
// docs/methodology/perf-benchmarking.md and the path-2 design notes in
// memory `project_a3b_prefill_path1_path2_state` for the SGLang/vLLM
// scatter-then-grouped-GEMM rationale.
//
// Shapes:
//   topk_indices         : [total_slots] i32 (= N × K_TOP)
//   expert_token_counts  : [num_experts] i32 (out, RAW counts; phase 2
//                                            rewrites to PADDED counts)
//
// Grid: (1,1,1). Block: 256 threads. LDS: num_experts * 4 bytes.

extern "C" __global__ void moe_scatter_histogram_k8(
    const int* __restrict__ topk_indices,
    int*       __restrict__ expert_token_counts,
    int total_slots,
    int num_experts
) {
    extern __shared__ int s_hist[];
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // Init LDS counters.
    for (int i = tid; i < num_experts; i += bdim) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Accumulate. Out-of-range expert ids (negative or >= num_experts)
    // are skipped — caller invariants guarantee in-range, but the guard
    // costs nothing and rules out silent OOB writes if a future router
    // misbehaves.
    for (int i = tid; i < total_slots; i += bdim) {
        const int e = topk_indices[i];
        if (e >= 0 && e < num_experts) {
            atomicAdd(&s_hist[e], 1);
        }
    }
    __syncthreads();

    // Coalesced write-back.
    for (int i = tid; i < num_experts; i += bdim) {
        expert_token_counts[i] = s_hist[i];
    }
}
