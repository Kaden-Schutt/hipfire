#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// SGLang-style grouped-WMMA-GEMM for MoE prefill. Adapted from
// `gemm_hfq4g256_residual_wmma_k2`: same 2× K-tile pipeline + WMMA
// 16×16×16 F16 layout, but the X operand is GATHERED via
// `sorted_slot_index` and the A operand is INDEXED by expert via
// `expert_tile_ids[tile_y]`.
//
// Pairs with the scatter pipeline at
//   kernels/src/moe_scatter_{histogram,offsets,permute}_k8.hip
// and the unscatter combine that fans Y_grouped back into the
// per-token gate_batch/up_batch streams.
//
// Shapes:
//   expert_weight_ptrs  : [E] u64 (device pointers to per-expert HFQ4-G256 weights,
//                                  each weight is [M × K] in the usual flat layout)
//   expert_tile_ids     : [m_total / 16] i32 (per-tile expert id)
//   sorted_slot_index   : [m_total] i32 (flat slot = token * K_TOP + k_rank, or -1 padding)
//   X_norm              : [N × K] f16 (per-token activations, NOT permuted)
//   Y_grouped           : [m_total × M] f32 (per-slot output rows; padded slots are written
//                                            to but never read by the combine)
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.
//
// Per-block contract: each block owns a 16-row × 16-slot output tile.
// All 16 slots in this tile belong to the same expert (the scatter
// pipeline guarantees expert_offsets[E] is a multiple of 16). The
// kernel:
//   1. Looks up expert id from expert_tile_ids[blockIdx.y].
//   2. Loads its A operand base = expert_weight_ptrs[expert_id].
//   3. For each WMMA lane (tid & 15), reads sorted_slot_index[tile_y*16 + lane]
//      and gathers X_norm[flat / K_TOP, :]. If sorted_slot_index = -1, the
//      lane contributes a zero row to B.
//   4. Runs the standard wmma_k2 inner loop and stores Y_grouped[slot, row].

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_moe_grouped_wmma_k2(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows × K]
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    // Sentinel early-return: expert_tile_ids[tile_y] == -1 marks tiles
    // beyond the live m_total (the dispatcher may launch up to
    // m_total_max/16 tiles to skip the m_total dtoh sync).
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X gather: each lane (tid & 15) gets its own routed slot, recovers
    // the source row via x_row_div (K_TOP for gate_up, 1 for down).
    // Padding lanes (sorted_slot_index == -1) get a NULL gather so the
    // X loads below can branch on the sentinel and substitute zeros.
    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    // A-row selection: pad the OOB rows to the last valid one (same trick
    // as wmma_k2 — keeps the loads in-range, the masked output write
    // discards the result).
    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        // 8 iterations × 2 K-tiles each = 16 K-tiles per group.
        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            // Weight: 4-bit packs.
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            // X loads — substitute zero vector when the lane has no source token.
            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            // Dequantize tile A.
            half16_t a_a;
            #define DQ(reg, i, pk, sh) reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(a_a, 0,  pk0a,  0); DQ(a_a, 1,  pk0a,  4); DQ(a_a, 2,  pk0a,  8); DQ(a_a, 3,  pk0a, 12);
            DQ(a_a, 4,  pk0a, 16); DQ(a_a, 5,  pk0a, 20); DQ(a_a, 6,  pk0a, 24); DQ(a_a, 7,  pk0a, 28);
            DQ(a_a, 8,  pk1a,  0); DQ(a_a, 9,  pk1a,  4); DQ(a_a, 10, pk1a,  8); DQ(a_a, 11, pk1a, 12);
            DQ(a_a, 12, pk1a, 16); DQ(a_a, 13, pk1a, 20); DQ(a_a, 14, pk1a, 24); DQ(a_a, 15, pk1a, 28);

            // WMMA A — b_b may still be in flight.
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            // Dequantize tile B (reuse a_a).
            DQ(a_a, 0,  pk0b,  0); DQ(a_a, 1,  pk0b,  4); DQ(a_a, 2,  pk0b,  8); DQ(a_a, 3,  pk0b, 12);
            DQ(a_a, 4,  pk0b, 16); DQ(a_a, 5,  pk0b, 20); DQ(a_a, 6,  pk0b, 24); DQ(a_a, 7,  pk0b, 28);
            DQ(a_a, 8,  pk1b,  0); DQ(a_a, 9,  pk1b,  4); DQ(a_a, 10, pk1b,  8); DQ(a_a, 11, pk1b, 12);
            DQ(a_a, 12, pk1b, 16); DQ(a_a, 13, pk1b, 20); DQ(a_a, 14, pk1b, 24); DQ(a_a, 15, pk1b, 28);
            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (M-dim), B operand = slot lane (m_total-dim).
    // We use direct store (=) not += because Y_grouped is freshly allocated
    // per-call. Padding lanes (slot_idx >= m_total) are silently dropped.
    const int out_col = slot_start + lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
