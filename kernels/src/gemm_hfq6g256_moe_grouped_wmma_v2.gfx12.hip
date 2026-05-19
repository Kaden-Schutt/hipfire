#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) M-direction 2×1 reg-blocked sister of
// `gemm_hfq6g256_moe_grouped_wmma_gfx12` (v1, commit b2284a11).
//
// **STATUS: FALSIFIED on R9700/gfx1201 AWQ A3B prefill 2026-05-19.**
//   - Hypothesis: HFQ6 is BW-bound, so the dequant cost the HFQ4 m2
//     falsification suffered from would be hidden behind memory waits.
//   - Result: -2.4% prefill regression (v1 2400.5 → v2 2343.1 tok/s on
//     /mnt/nas/kaden/hipfire/mi300x-v3/qwen3-35b-a3b.mq4-awq with the
//     standard merge-sort code prompt at max_tokens=150, dn_quant=q8).
//   - Likely cause: same HFQ4-m2-falsification pattern (dequant adds
//     register pressure / serialization beyond the BW-bound ceiling),
//     plus possible VGPR-spill from the doubled accumulator + dual
//     safe_row state at __launch_bounds__(32, 2).
//
// Kept as opt-in research artifact under HIPFIRE_MOE_HFQ6_V2=1; do not
// flip default on. See feedback_v2_sgpr_lut_falsified_2026_05_10
// (rdna4 SGPR-LUT) and project_mq3_mq4_fdot2_gfx1030_2026_05_18 for
// patterns: in this codebase + ROCm 7.2, the only levers that cross
// zero in production are launch-reduction / fusion / format changes;
// reg-blocking with extra dequant rarely survives synthetic→prod.
//
// Original hypothesis below — kept verbatim for future cross-reference.
//
// Why this lever for HFQ6 specifically (vs the HFQ4 m2 which was
// FALSIFIED on MQ4 grouped due to dequant serialization):
//   - HFQ6 is BW-BOUND (200 B/group, 47% more than HFQ4's 136 B); v1
//     measures ~half the per-row throughput of the MQ4 sister.
//   - In a BW-bound kernel the dequant cost is hidden behind memory
//     waits — adding 2× dequants per K-substep doesn't serialize the
//     way it did for the ALU-bound MQ4 case.
//   - Halving the X-gather BW per output element directly reduces the
//     L2 read pressure that the v1 measures as the dominant cost.
//
// Lever (extends `gemm_hfq4g256_moe_grouped_wmma_m2.gfx12.hip`):
//   - Each warp covers a 32-row × 16-slot output tile (was 16×16).
//   - The B operand (X-gather) is loaded ONCE per K-substep and
//     reused across two M-blocks; A doubles.
//   - Per K-substep: 2 A-dequants + 1 B-load + 2 WMMAs (was 1+1+1).
//   - WMMA-per-load ratio: 0.67 (was 0.5) → ~33% more compute density
//     on the dominant memory pattern when not VGPR-bound.
//
// Expert-boundary safety: this kernel block-strides ONLY in the M
// (output-row) direction, not the N (slot) direction. The scatter
// alignment to BLOCK_M=16 carries through unchanged: every 16-slot
// sub-tile maps to exactly one expert, same as v1. No expert-mismatch
// branch needed.
//
// Per-block contract:
//   - Tile (blockIdx.x, blockIdx.y) owns a 32-row × 16-slot output tile.
//   - All 16 slots share the same expert (BLOCK_M=16 scatter contract).
//   - Each lane m_lane = tid & 15 maps to ONE slot AND TWO M-rows
//     (row_start + m_lane for acc0, row_start + 16 + m_lane for acc1).
//   - X is gathered via sorted_slot_index[slot_start + m_lane]; -1
//     padding lanes substitute zero rows for the B operand.
//   - expert_tile_ids[tile_y] == -1 → early-return (matches the
//     sync-free dispatch convention).
//
// Grid: [ceil(M/32), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq6g256_moe_grouped_wmma_v2_gfx12(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows × K]
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 32;   // 2× wider M-stride
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // B operand: single X-gather row per lane, shared across both M-blocks.
    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    // A operand: two M-block rows per lane. Clamp OOB rows to last valid
    // row (same trick as base kernel; output mask discards the result).
    const int safe_row_0 = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int safe_row_1 = (row_start + 16 + m_lane < M) ? (row_start + 16 + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base_0 = A + (long long)safe_row_0 * groups_per_row * 200;
    const char* row_base_1 = A + (long long)safe_row_1 * groups_per_row * 200;

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};

    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp0 = row_base_0 + g * 200;
        const char* gp1 = row_base_1 + g * 200;
        _Float16 sc0 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp0));
        _Float16 zp0 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        _Float16 sc1 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp1));
        _Float16 zp1 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        // HFQ6 dequant: 8 values from 6 packed bytes via 2 overlapping
        // unsigned-int loads (d0 = bytes [0..3], d1 = bytes [3..6]).
        // Shifts 0/6/12/18 × mask 63u extract 4 × 6-bit values per uint.
        #define DQ6_TO(reg, dw0, dw1, sc, zp) \
            reg[0] = sc * (_Float16)(float)(((dw0) >>  0) & 63u) + zp; \
            reg[1] = sc * (_Float16)(float)(((dw0) >>  6) & 63u) + zp; \
            reg[2] = sc * (_Float16)(float)(((dw0) >> 12) & 63u) + zp; \
            reg[3] = sc * (_Float16)(float)(((dw0) >> 18) & 63u) + zp; \
            reg[4] = sc * (_Float16)(float)(((dw1) >>  0) & 63u) + zp; \
            reg[5] = sc * (_Float16)(float)(((dw1) >>  6) & 63u) + zp; \
            reg[6] = sc * (_Float16)(float)(((dw1) >> 12) & 63u) + zp; \
            reg[7] = sc * (_Float16)(float)(((dw1) >> 18) & 63u) + zp

        #pragma unroll
        for (int kt = 0; kt < 16; kt += 4) {
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                const int k_off = (kt + sub) * 16;
                // HFQ6 K-tile stride = 12 B; per-k_grp half = 6 B.
                const int bb = (kt + sub) * 12 + k_grp * 6;
                const unsigned char* dp0 = (const unsigned char*)(gp0 + 8 + bb);
                const unsigned char* dp1 = (const unsigned char*)(gp1 + 8 + bb);
                unsigned int d0_a = *(const unsigned int*)(dp0);
                unsigned int d1_a = *(const unsigned int*)(dp0 + 3);
                unsigned int d0_b = *(const unsigned int*)(dp1);
                unsigned int d1_b = *(const unsigned int*)(dp1 + 3);

                half8_t b_reg;
                if (xg != nullptr) {
                    b_reg = *(const half8_t*)(xg + k_off + k_grp * 8);
                } else {
                    b_reg = zero_b;
                }

                half8_t a_reg_0, a_reg_1;
                DQ6_TO(a_reg_0, d0_a, d1_a, sc0, zp0);
                DQ6_TO(a_reg_1, d0_b, d1_b, sc1, zp1);
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg_0, b_reg, acc0);
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg_1, b_reg, acc1);
            }
        }
        #undef DQ6_TO
    }

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row_0 = row_start + 8 * k_grp + j;
            if (out_row_0 < M) {
                Y_grouped[(long long)out_col * M + out_row_0] = acc0[j];
            }
            int out_row_1 = row_start + 16 + 8 * k_grp + j;
            if (out_row_1 < M) {
                Y_grouped[(long long)out_col * M + out_row_1] = acc1[j];
            }
        }
    }
}
