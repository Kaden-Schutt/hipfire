#include <hip/hip_runtime.h>

// Phase 2: pad raw histogram counts up to a multiple of `block_m` (the
// WMMA tile row count) and compute the exclusive prefix-sum offsets.
//
// In-place rewrites `expert_token_counts` from RAW → PADDED so the
// downstream scatter (Phase 3) and grouped-GEMM (Stage 2) can iterate
// expert-by-expert without re-deriving padded counts.
//
// `expert_offsets[E+1]` is the exclusive prefix sum of padded counts;
// `expert_offsets[E]` equals `M_total`, the total padded slot count.
//
// Single workgroup; serial scan by thread 0. For E ≤ 256 (A3B has 256
// experts) the serial scan is well below the cost of a single global
// atomic on Phase 1, so block-scan via __shfl is over-engineering at
// this size. Phase 3's scatter dominates wall time anyway.
//
// Grid: (1,1,1). Block: 256 threads. LDS: num_experts * 4 bytes.

extern "C" __global__ void moe_scatter_offsets_k8(
    int*  __restrict__ expert_token_counts, // [E] in: raw, out: padded
    int*  __restrict__ expert_offsets,      // [E+1] out: exclusive scan
    int   num_experts,
    int   block_m
) {
    extern __shared__ int s_padded[];
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // Pad each expert's count and stage in LDS for the scan below.
    for (int i = tid; i < num_experts; i += bdim) {
        int raw = expert_token_counts[i];
        int padded = ((raw + block_m - 1) / block_m) * block_m;
        s_padded[i] = padded;
        expert_token_counts[i] = padded;
    }
    __syncthreads();

    // Serial scan by thread 0. E ≤ 256 fits in a tight loop; the
    // dependent-add chain isn't a bottleneck vs the scatter that
    // follows.
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < num_experts; i++) {
            const int v = s_padded[i];
            expert_offsets[i] = acc;
            acc += v;
        }
        expert_offsets[num_experts] = acc; // = M_total
    }
}
