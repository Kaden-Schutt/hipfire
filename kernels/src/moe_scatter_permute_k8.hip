#include <hip/hip_runtime.h>

// Phase 3: permutation + tile_ids generation.
//
// Writes `sorted_slot_index[M_total]` with each flat input slot index
// (= n * K_TOP + k) at position `expert_offsets[e] + bucket_fill[e]`.
// Padding slots within an expert's range (those between raw_count and
// padded_count) stay at the -1 sentinel that the GEMM kernel uses to
// short-circuit the row load.
//
// Also writes `expert_tile_ids[M_total / block_m]` = expert id for each
// WMMA tile. Tiles that span both real and padding slots within a given
// expert still carry that expert's id; the GEMM kernel handles the -1
// rows by emitting zero contributions.
//
// LDS holds `num_experts` bucket-fill counters (1 KiB for E=256). The
// scatter pass is a single workgroup with bdim=256 threads — for A3B's
// total_slots = N * K_TOP ≤ 2048 this is one or two iterations per
// thread, dominated by global stores to `sorted_slot_index`.
//
// Grid: (1,1,1). Block: 256 threads. LDS: num_experts * 4 bytes.

extern "C" __global__ void moe_scatter_permute_k8(
    const int* __restrict__ topk_indices,
    const int* __restrict__ expert_offsets,    // [E+1] exclusive padded scan
    int*       __restrict__ sorted_slot_index, // [M_total] out (-1 for padding)
    int*       __restrict__ expert_tile_ids,   // [M_total / block_m] out
    int*       __restrict__ inverse_perm,      // [total_slots] out: flat → sorted_pos
    int        total_slots,
    int        num_experts,
    int        m_total,
    int        block_m
) {
    extern __shared__ int s_bucket[];
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // Init LDS bucket-fill counters.
    for (int i = tid; i < num_experts; i += bdim) {
        s_bucket[i] = 0;
    }

    // Init sorted_slot_index to -1 so padding slots within each
    // expert's range carry the sentinel. Stride-coalesced; runs in
    // parallel with the LDS init above.
    for (int i = tid; i < m_total; i += bdim) {
        sorted_slot_index[i] = -1;
    }
    __syncthreads();

    // Scatter: each flat slot gets its bucket position from an LDS
    // atomic, then written to its destination row in sorted_slot_index.
    // Per-expert bucket ordering is not stable (atomics race), but the
    // downstream grouped-GEMM doesn't depend on the order — it only
    // requires that the same expert's slots land contiguously.
    for (int i = tid; i < total_slots; i += bdim) {
        const int e = topk_indices[i];
        if (e >= 0 && e < num_experts) {
            const int pos_in_bucket = atomicAdd(&s_bucket[e], 1);
            const int base = expert_offsets[e];
            const int sorted_pos = base + pos_in_bucket;
            sorted_slot_index[sorted_pos] = i;
            // inverse_perm[flat] = sorted_pos enables the non-atomic
            // down-combine: per (token, m), iterate K_TOP slots and read
            // inverse_perm[token*K_TOP + k] to locate each contribution.
            inverse_perm[i] = sorted_pos;
        }
    }
    __syncthreads();

    // Tile ids: walk by expert (not by tile) so each thread emits a
    // contiguous run of tile_ids. Avoids the O(E) per-tile search the
    // tile-walked variant would need.
    for (int e = tid; e < num_experts; e += bdim) {
        const int start_slot = expert_offsets[e];
        const int end_slot = expert_offsets[e + 1];
        // start_slot and end_slot are both multiples of block_m
        // (guaranteed by Phase 2's pad-then-scan).
        const int tile_start = start_slot / block_m;
        const int tile_end = end_slot / block_m;
        for (int t = tile_start; t < tile_end; t++) {
            expert_tile_ids[t] = e;
        }
    }
}
