#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1151 HFQ6/MQ6 grouped-MoE WMMA experiment.
//
// Four warps cover a 64-row x 16-slot output tile for one expert tile.
// The 16-slot X slice for each 256-K HFQ6 group is staged once in LDS and
// reused by the four row-warps. This targets the v1 kernel's repeated X
// global loads across adjacent M-row tiles without changing the grouped
// scatter contract or the gfx11 WMMA C mapping.

#if defined(__gfx1151__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE 64
#define SLOT_TILE 16
#define WARPS 4
#define M_PER_WARP 16
#define GROUP_K 256

__launch_bounds__(128, 1)
extern "C" __global__ void gemm_hfq6g256_moe_grouped_wmma_4w_gfx1151(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [x_src_rows x K]
    float*                    __restrict__ Y_grouped,          // [m_total x M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;

    const int row_start = blockIdx.x * M_TILE;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * SLOT_TILE;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int warp_row_start = row_start + warp_id * M_PER_WARP;
    const int safe_row = (warp_row_start + lane16 < M) ? (warp_row_start + lane16) : (M - 1);
    const int groups_per_row = K / GROUP_K;
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    __shared__ _Float16 lds_x[SLOT_TILE * GROUP_K];

    #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
    #define DQ6_8(base, d0, d1) \
        DQ6((base) + 0, (d0), 0);  DQ6((base) + 1, (d0), 6);  DQ6((base) + 2, (d0), 12); DQ6((base) + 3, (d0), 18); \
        DQ6((base) + 4, (d1), 0);  DQ6((base) + 5, (d1), 6);  DQ6((base) + 6, (d1), 12); DQ6((base) + 7, (d1), 18)

    for (int g = 0; g < groups_per_row; g++) {
        // Stage 16 routed slots x 256 K-elements. There are 256 half16
        // vectors; 128 threads load two vectors each.
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            const int vec = pass * 128 + tid;
            const int slot_lane = vec >> 4;
            const int k_vec = vec & 15;
            const int slot_idx = slot_start + slot_lane;
            int x_row = -1;
            if (slot_idx < m_total) {
                const int flat = sorted_slot_index[slot_idx];
                if (flat >= 0) {
                    x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
                }
            }
            half16_t v;
            if (x_row >= 0) {
                v = *(const half16_t*)(X_src + (long long)x_row * K + g * GROUP_K + k_vec * 16);
            } else {
                v = (half16_t){
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0
                };
            }
            *(half16_t*)(lds_x + slot_lane * GROUP_K + k_vec * 16) = v;
        }
        __syncthreads();

        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

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

            half16_t b_a = *(const half16_t*)(lds_x + lane16 * GROUP_K + (kt + 0) * 16);
            half16_t b_b = *(const half16_t*)(lds_x + lane16 * GROUP_K + (kt + 1) * 16);

            half16_t a_reg;
            DQ6_8(0, d0a0, d1a0);
            DQ6_8(8, d0a1, d1a1);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            DQ6_8(0, d0b0, d1b0);
            DQ6_8(8, d0b1, d1b1);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
        __syncthreads();
    }

    #undef DQ6_8
    #undef DQ6

    const int out_col = slot_start + lane16;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = warp_row_start + 2 * j + (lane >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#else

extern "C" __global__ void gemm_hfq6g256_moe_grouped_wmma_4w_gfx1151(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int
) {}

#endif
