#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1151 (RDNA3.5) HFQ6/MQ6 sister of `gemm_hfq4g256_moe_grouped_wmma_k2`.
// Same MoE Path-2 grouped dispatch contract as HFQ4/HFQ3; only the
// dequant layout changes to HFQ6-G256's 200 B/group.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq6g256_moe_grouped_wmma_gfx1151(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows x K]
    float*                    __restrict__ Y_grouped,          // [m_total x M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
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

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
    #define DQ6_8(base, d0, d1) \
        DQ6((base) + 0, (d0), 0);  DQ6((base) + 1, (d0), 6);  DQ6((base) + 2, (d0), 12); DQ6((base) + 3, (d0), 18); \
        DQ6((base) + 4, (d1), 0);  DQ6((base) + 5, (d1), 6);  DQ6((base) + 6, (d1), 12); DQ6((base) + 7, (d1), 18)

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        #pragma unroll
        for (int kt = 0; kt < 16; kt += 2) {
            const unsigned char* dpa = (const unsigned char*)(gp + 8 + (kt + 0) * 12);
            const unsigned char* dpb = (const unsigned char*)(gp + 8 + (kt + 1) * 12);
            unsigned int d0a0 = *(const unsigned int*)(dpa);
            unsigned int d1a0 = *(const unsigned int*)(dpa + 3);
            unsigned int d0a1 = *(const unsigned int*)(dpa + 6);
            unsigned int d1a1 = *(const unsigned int*)(dpa + 9);
            unsigned int d0b0 = *(const unsigned int*)(dpb);
            unsigned int d1b0 = *(const unsigned int*)(dpb + 3);
            unsigned int d0b1 = *(const unsigned int*)(dpb + 6);
            unsigned int d1b1 = *(const unsigned int*)(dpb + 9);

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + (kt + 0) * 16);
                b_b = *(const half16_t*)(xg + (kt + 1) * 16);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            half16_t a_reg;
            DQ6_8(0, d0a0, d1a0);
            DQ6_8(8, d0a1, d1a1);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            DQ6_8(0, d0b0, d1b0);
            DQ6_8(8, d0b1, d1b1);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }

    #undef DQ6_8
    #undef DQ6

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
