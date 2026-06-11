#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1151 (RDNA3.5) M-direction 2x1 reg-blocked sister of
// `gemm_hfq6g256_moe_grouped_wmma_gfx1151`.
//
// Each warp covers a 32-row x 16-slot output tile. The gathered X tile is
// loaded once per K-substep and reused across two HFQ6 rows. This is the
// gfx11/RDNA3.5-specific port of the gfx12 v2 experiment, using the gfx11
// WMMA operand shape and C-fragment mapping.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq6g256_moe_grouped_wmma_v2_gfx1151(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows x K]
    float*                    __restrict__ Y_grouped,          // [m_total x M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 32;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const int lane = tid & 15;
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    const int safe_row_0 = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int safe_row_1 = (row_start + 16 + lane < M) ? (row_start + 16 + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base_0 = A + (long long)safe_row_0 * groups_per_row * 200;
    const char* row_base_1 = A + (long long)safe_row_1 * groups_per_row * 200;

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    #define DQ6_TO(reg, i, dw, sh, sc, zp) reg[i] = sc * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp
    #define DQ6_8_TO(reg, base, d0, d1, sc, zp) \
        DQ6_TO(reg, (base) + 0, (d0), 0,  sc, zp); DQ6_TO(reg, (base) + 1, (d0), 6,  sc, zp); \
        DQ6_TO(reg, (base) + 2, (d0), 12, sc, zp); DQ6_TO(reg, (base) + 3, (d0), 18, sc, zp); \
        DQ6_TO(reg, (base) + 4, (d1), 0,  sc, zp); DQ6_TO(reg, (base) + 5, (d1), 6,  sc, zp); \
        DQ6_TO(reg, (base) + 6, (d1), 12, sc, zp); DQ6_TO(reg, (base) + 7, (d1), 18, sc, zp)

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp0 = row_base_0 + g * 200;
        const char* gp1 = row_base_1 + g * 200;
        _Float16 sc0 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp0));
        _Float16 zp0 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        _Float16 sc1 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp1));
        _Float16 zp1 = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        #pragma unroll
        for (int kt = 0; kt < 16; kt += 2) {
            const unsigned char* dpa0 = (const unsigned char*)(gp0 + 8 + (kt + 0) * 12);
            const unsigned char* dpa1 = (const unsigned char*)(gp1 + 8 + (kt + 0) * 12);
            const unsigned char* dpb0 = (const unsigned char*)(gp0 + 8 + (kt + 1) * 12);
            const unsigned char* dpb1 = (const unsigned char*)(gp1 + 8 + (kt + 1) * 12);

            unsigned int d0a0 = *(const unsigned int*)(dpa0);
            unsigned int d1a0 = *(const unsigned int*)(dpa0 + 3);
            unsigned int d2a0 = *(const unsigned int*)(dpa0 + 6);
            unsigned int d3a0 = *(const unsigned int*)(dpa0 + 9);
            unsigned int d0a1 = *(const unsigned int*)(dpa1);
            unsigned int d1a1 = *(const unsigned int*)(dpa1 + 3);
            unsigned int d2a1 = *(const unsigned int*)(dpa1 + 6);
            unsigned int d3a1 = *(const unsigned int*)(dpa1 + 9);

            unsigned int d0b0 = *(const unsigned int*)(dpb0);
            unsigned int d1b0 = *(const unsigned int*)(dpb0 + 3);
            unsigned int d2b0 = *(const unsigned int*)(dpb0 + 6);
            unsigned int d3b0 = *(const unsigned int*)(dpb0 + 9);
            unsigned int d0b1 = *(const unsigned int*)(dpb1);
            unsigned int d1b1 = *(const unsigned int*)(dpb1 + 3);
            unsigned int d2b1 = *(const unsigned int*)(dpb1 + 6);
            unsigned int d3b1 = *(const unsigned int*)(dpb1 + 9);

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + (kt + 0) * 16);
                b_b = *(const half16_t*)(xg + (kt + 1) * 16);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            half16_t a0, a1;
            DQ6_8_TO(a0, 0, d0a0, d1a0, sc0, zp0);
            DQ6_8_TO(a0, 8, d2a0, d3a0, sc0, zp0);
            DQ6_8_TO(a1, 0, d0a1, d1a1, sc1, zp1);
            DQ6_8_TO(a1, 8, d2a1, d3a1, sc1, zp1);
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b_a, acc0);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b_a, acc1);

            DQ6_8_TO(a0, 0, d0b0, d1b0, sc0, zp0);
            DQ6_8_TO(a0, 8, d2b0, d3b0, sc0, zp0);
            DQ6_8_TO(a1, 0, d0b1, d1b1, sc1, zp1);
            DQ6_8_TO(a1, 8, d2b1, d3b1, sc1, zp1);
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b_b, acc0);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b_b, acc1);
        }
    }

    #undef DQ6_8_TO
    #undef DQ6_TO

    const int out_col = slot_start + lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row_0 = row_start + 2 * j + (tid >> 4);
            if (out_row_0 < M) {
                Y_grouped[(long long)out_col * M + out_row_0] = acc0[j];
            }
            int out_row_1 = row_start + 16 + 2 * j + (tid >> 4);
            if (out_row_1 < M) {
                Y_grouped[(long long)out_col * M + out_row_1] = acc1[j];
            }
        }
    }
}
