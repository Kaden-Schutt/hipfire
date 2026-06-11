#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) HFQ6/MQ6 sister of
// `gemm_hfq4g256_moe_grouped_wmma_gfx12`. Pure composition of two
// already-validated patterns:
//   - HFQ4 grouped dispatch contract (expert_tile_ids sentinel,
//     sorted_slot_index X-gather, x_row_div for gate_up vs down),
//     validated in gemm_hfq4g256_moe_grouped_wmma.gfx12.hip.
//   - HFQ6-G256 dequant inner loop (200 B/group: 4 B scale + 4 B zero
//     + 192 B packed 6-bit), validated in gemm_qkv_hfq6g256_wmma.gfx12.hip
//     and gemm_qkvza_hfq6g256_wmma.gfx12.hip.
//
// HFQ6 vs HFQ4 deltas in the inner loop (everything else is identical):
//   - Group stride: 200 B instead of 136 B.
//   - Per-K-tile (16 K values) the packed payload is 12 B instead of 8 B
//     (16 × 6 bits = 96 bits = 12 B), so the kt stride is 12 and each
//     k_grp's half-tile is 6 B (8 × 6 bits = 48 bits = 6 B).
//   - Read 6 B as two overlapping unsigned ints (d0 = uint32 at +0,
//     d1 = uint32 at +3) so 8 values cleanly factor into 4 shifts × 6
//     bits per uint. Same trick used in the qkv/qkvza variants.
//
// MQ6G256 weights use the IDENTICAL on-disk layout (same 200 B/group);
// the FWHT rotation is applied by the caller to X before dispatch, so
// the kernel itself is dtype-agnostic between HFQ6 and MQ6.
//
// Per-block contract (unchanged from the HFQ4 grouped variant):
//   - Tile (blockIdx.x, blockIdx.y) owns a 16-row × 16-slot output tile.
//   - All 16 slots share the same expert (Phase 2's pad-then-scan
//     guarantees expert_offsets aligned to BLOCK_M = 16).
//   - Each lane m_lane = tid & 15 corresponds to one slot AND one M-row.
//   - X is gathered via sorted_slot_index[slot_start + m_lane]; -1
//     padding lanes substitute zero rows for the B operand.
//   - expert_tile_ids[tile_y] == -1 → early-return (matches the
//     sync-free dispatch convention).
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq6g256_moe_grouped_wmma_gfx12(
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
    // m_total_max/16 tiles to skip the m_total dtoh sync, since the
    // scatter pre-fills the upper-bound range with -1).
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X gather (per m_lane; both k_grp lanes for the same m_lane read the
    // same source row). -1 padding lanes contribute a zero B row.
    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    // A-row selection: pad OOB lanes to the last valid M-row (gfx11 trick
    // — the masked output write below discards the result anyway).
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        // HFQ6 dequant: 8 values from 6 packed bytes via 2 overlapping
        // unsigned-int loads (d0 = bytes [0..3], d1 = bytes [3..6]).
        // Shifts 0/6/12/18 × mask 63u extract 4 × 6-bit values per uint.
        #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
        #define DQ6_8(d0, d1) \
            DQ6(0, (d0), 0);  DQ6(1, (d0), 6);  DQ6(2, (d0), 12); DQ6(3, (d0), 18); \
            DQ6(4, (d1), 0);  DQ6(5, (d1), 6);  DQ6(6, (d1), 12); DQ6(7, (d1), 18)

        for (int kt = 0; kt < 16; kt += 4) {
            // 4 weight half-tiles, each 6 bytes (8 × 6-bit). HFQ6
            // K-tile stride is 12 B (vs HFQ4's 8 B); per-k_grp half is 6 B.
            const int bb_a = (kt + 0) * 12 + k_grp * 6;
            const int bb_b = (kt + 1) * 12 + k_grp * 6;
            const int bb_c = (kt + 2) * 12 + k_grp * 6;
            const int bb_d = (kt + 3) * 12 + k_grp * 6;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + bb_a);
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + bb_b);
            const unsigned char* dp_c = (const unsigned char*)(gp + 8 + bb_c);
            const unsigned char* dp_d = (const unsigned char*)(gp + 8 + bb_d);
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = *(const unsigned int*)(dp_a + 3);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = *(const unsigned int*)(dp_b + 3);
            unsigned int d0c = *(const unsigned int*)(dp_c);
            unsigned int d1c = *(const unsigned int*)(dp_c + 3);
            unsigned int d0d = *(const unsigned int*)(dp_d);
            unsigned int d1d = *(const unsigned int*)(dp_d + 3);

            half8_t b_a, b_b, b_c, b_d;
            if (xg != nullptr) {
                b_a = *(const half8_t*)(xg + (kt + 0) * 16 + k_grp * 8);
                b_b = *(const half8_t*)(xg + (kt + 1) * 16 + k_grp * 8);
                b_c = *(const half8_t*)(xg + (kt + 2) * 16 + k_grp * 8);
                b_d = *(const half8_t*)(xg + (kt + 3) * 16 + k_grp * 8);
            } else {
                b_a = zero_b; b_b = zero_b; b_c = zero_b; b_d = zero_b;
            }

            half8_t a_reg;
            DQ6_8(d0a, d1a);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ6_8(d0b, d1b);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ6_8(d0c, d1c);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ6_8(d0d, d1d);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
        #undef DQ6_8
        #undef DQ6
    }

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    // Use direct store (=) since Y_grouped is fresh per call.
    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
