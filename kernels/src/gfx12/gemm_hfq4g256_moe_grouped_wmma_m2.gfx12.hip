#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// 2×1 M-direction reg-blocked sister of
// `gemm_hfq4g256_moe_grouped_wmma_gfx12`.
//
// Lever attributed to glovepost/wmma_ops (`wmma_kernels_optimized.hpp`):
// each warp covers a 32-row × 16-slot output tile (vs 16×16 in the base).
// The B operand (X-gather, the expensive side in grouped-GEMM) is loaded
// ONCE per K-substep and reused across two M-blocks; A doubles. Per
// K-substep: 2 packed4 dequants + 1 b_reg load + 2 WMMA (was 1+1+1).
// WMMA-per-load ratio: 1.0 (was 0.5) → ~2× compute density when not
// VGPR-bound. Expected: −1 tile in Y per CU, plus halved X-gather BW
// per output, → +20-40% on the grouped GEMM kernel time.
//
// Per-block contract:
//   - Tile (blockIdx.x, blockIdx.y) owns a 32-row × 16-slot output tile.
//   - All 16 slots share the same expert (the scatter still pads to
//     BLOCK_M = 16; we do NOT change scatter alignment).
//   - Each lane m_lane = tid & 15 maps to ONE slot AND TWO M-rows
//     (row_start + m_lane for acc0, row_start + 16 + m_lane for acc1).
//   - X is gathered via sorted_slot_index[slot_start + m_lane]; -1
//     padding lanes substitute zero rows for the B operand.
//
// Grid: [ceil(M/32), m_total_max/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_moe_grouped_wmma_m2_gfx12(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total_max / 16]
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
    // row (same trick as base kernel; output write below masks the result).
    const int safe_row_0 = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int safe_row_1 = (row_start + 16 + m_lane < M) ? (row_start + 16 + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base_0 = A + (long long)safe_row_0 * groups_per_row * 136;
    const char* row_base_1 = A + (long long)safe_row_1 * groups_per_row * 136;

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};

    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp0 = row_base_0 + g * 136;
        const char* gp1 = row_base_1 + g * 136;
        _Float16 sc0 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp0));
        _Float16 zp0 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        _Float16 sc1 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp1));
        _Float16 zp1 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        #define DQ8_TO(reg, pk, sc, zp) \
            reg[0] = sc * (_Float16)(float)(((pk) >>  0) & 0xFu) + zp; \
            reg[1] = sc * (_Float16)(float)(((pk) >>  4) & 0xFu) + zp; \
            reg[2] = sc * (_Float16)(float)(((pk) >>  8) & 0xFu) + zp; \
            reg[3] = sc * (_Float16)(float)(((pk) >> 12) & 0xFu) + zp; \
            reg[4] = sc * (_Float16)(float)(((pk) >> 16) & 0xFu) + zp; \
            reg[5] = sc * (_Float16)(float)(((pk) >> 20) & 0xFu) + zp; \
            reg[6] = sc * (_Float16)(float)(((pk) >> 24) & 0xFu) + zp; \
            reg[7] = sc * (_Float16)(float)(((pk) >> 28) & 0xFu) + zp

        #pragma unroll
        for (int kt = 0; kt < 16; kt += 4) {
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                const int k_off = (kt + sub) * 16;

                unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + k_off / 2 + k_grp * 4);
                unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + k_off / 2 + k_grp * 4);

                half8_t b_reg;
                if (xg != nullptr) {
                    b_reg = *(const half8_t*)(xg + k_off + k_grp * 8);
                } else {
                    b_reg = zero_b;
                }

                half8_t a_reg_0, a_reg_1;
                DQ8_TO(a_reg_0, pk0, sc0, zp0);
                DQ8_TO(a_reg_1, pk1, sc1, zp1);
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg_0, b_reg, acc0);
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg_1, b_reg, acc1);
            }
        }
        #undef DQ8_TO
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
