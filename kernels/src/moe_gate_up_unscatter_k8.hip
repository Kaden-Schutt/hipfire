#include <hip/hip_runtime.h>

// Path 2 Stage 3 (gate_up half): fan Y_grouped back to per-token
// gate_batch / up_batch streams.
//
// Pairs with `gemm_hfq4g256_moe_grouped_wmma_k2` which writes
// `Y_grouped[m_total × (2*mi)]` (rows 0..mi → gate logits, rows mi..2*mi
// → up logits) for each routed slot. This kernel inverts the scatter
// permutation: for each valid slot, recover (token, k_rank) from
// `sorted_slot_index[slot]` and write the gate+up halves to their
// per-token positions.
//
// Padding slots (sorted_slot_index = -1) are skipped — the producer
// wrote zeros into Y_grouped for those, but the consumer (SwiGLU+FWHT)
// only reads valid (token, k_rank) tuples, so the writes don't need
// to land.
//
// Shapes:
//   Y_grouped         : [m_total × (2*mi)] f32
//   sorted_slot_index : [m_total] i32 (flat = token*K_TOP + k_rank, or -1)
//   y_gate            : [N × K_TOP × mi] f32 (written)
//   y_up              : [N × K_TOP × mi] f32 (written)
//
// Grid: (ceil(mi/256), m_total). Block: [256]. LDS: 0.

extern "C" __global__ void moe_gate_up_unscatter_k8(
    const float* __restrict__ Y_grouped,
    const int*   __restrict__ sorted_slot_index,
    float*       __restrict__ y_gate,
    float*       __restrict__ y_up,
    int mi, int K_TOP, int m_total
) {
    const int slot = blockIdx.y;
    if (slot >= m_total) return;
    const int flat = sorted_slot_index[slot];
    if (flat < 0) return; // padding slot — producer wrote 0; downstream
                          // doesn't consume this (token, k_rank).

    // Recover the routed-token addressing.
    const int token = flat / K_TOP;
    const int krank = flat - token * K_TOP;

    const int m_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (m_col >= mi) return;

    const long long y_grp_base = (long long)slot * 2 * mi;
    const long long out_base   = (long long)token * K_TOP * mi + (long long)krank * mi;

    y_gate[out_base + m_col] = Y_grouped[y_grp_base + m_col];
    y_up  [out_base + m_col] = Y_grouped[y_grp_base + mi + m_col];
}
