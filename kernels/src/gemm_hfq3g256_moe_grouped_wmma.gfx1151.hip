#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1151 (RDNA3.5) HFQ3/MQ3 sister of `gemm_hfq4g256_moe_grouped_wmma_k2`.
// Same MoE Path-2 contract as the HFQ4 grouped kernel: pad-then-scan
// guarantees that all 16 slots in a (blockIdx.x, blockIdx.y) tile share
// one expert, selected through expert_tile_ids[tile_y].
//
// HFQ3-G256 storage layout (per 256-element group, 104 B total):
//   [0..4]    fp32 scale
//   [4..8]    fp32 zero
//   [8..104]  96 B packed 3-bit weights: 16 K-tiles * 6 B/tile.
//
// gfx1151 uses the gfx11 wave32 WMMA ABI:
//   - half16_t operands, builtin __builtin_amdgcn_wmma_*_w32
//   - C mapping: acc[j] = C[2*j + (tid >> 4)][tid & 15]
//
// X gather:
//   gate_up: X_src = x_rot_batch [N x K], x_row_div = K_TOP
//   down:    X_src = rot_batch   [N*K_TOP x K], x_row_div = 1

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq3g256_moe_grouped_wmma_gfx1151(
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
    const char* row_base = A + (long long)safe_row * groups_per_row * 104;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    #define UNPACK8(dst, base, b0, b1, b2)                                      \
        do {                                                                    \
            unsigned int _a0 = (b0), _a1 = (b1), _a2 = (b2);                    \
            (dst)[(base) + 0] = sc_h * (_Float16)(float)(_a0 & 7u) + zp_h;      \
            (dst)[(base) + 1] = sc_h * (_Float16)(float)((_a0 >> 3) & 7u) + zp_h; \
            (dst)[(base) + 2] = sc_h * (_Float16)(float)(((_a0 >> 6) | (_a1 << 2)) & 7u) + zp_h; \
            (dst)[(base) + 3] = sc_h * (_Float16)(float)((_a1 >> 1) & 7u) + zp_h; \
            (dst)[(base) + 4] = sc_h * (_Float16)(float)((_a1 >> 4) & 7u) + zp_h; \
            (dst)[(base) + 5] = sc_h * (_Float16)(float)(((_a1 >> 7) | (_a2 << 1)) & 7u) + zp_h; \
            (dst)[(base) + 6] = sc_h * (_Float16)(float)((_a2 >> 2) & 7u) + zp_h; \
            (dst)[(base) + 7] = sc_h * (_Float16)(float)((_a2 >> 5) & 7u) + zp_h; \
        } while (0)

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 104;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;
        const unsigned char* dp = (const unsigned char*)(gp + 8);

        #pragma unroll
        for (int kt = 0; kt < 16; kt += 2) {
            const unsigned char* dpa = dp + (kt + 0) * 6;
            const unsigned char* dpb = dp + (kt + 1) * 6;
            const int k_off_a = (kt + 0) * 16;
            const int k_off_b = (kt + 1) * 16;

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            half16_t a_reg;
            UNPACK8(a_reg, 0, dpa[0], dpa[1], dpa[2]);
            UNPACK8(a_reg, 8, dpa[3], dpa[4], dpa[5]);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            UNPACK8(a_reg, 0, dpb[0], dpb[1], dpb[2]);
            UNPACK8(a_reg, 8, dpb[3], dpb[4], dpb[5]);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }
    #undef UNPACK8

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
