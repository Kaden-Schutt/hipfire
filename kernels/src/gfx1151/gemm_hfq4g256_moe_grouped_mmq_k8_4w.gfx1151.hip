#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// gfx1151 HFQ4/MQ4 grouped-MoE i8 MMQ k8, four-warp X-staged variant.
//
// Each block covers 64 M-rows x 16 routed slots for one expert tile. For
// each Q8_1 activation block, the 16 routed-slot blocks are staged once in
// LDS and reused by four row-warps. The per-warp math is the same k8 path as
// gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151.

#if defined(__gfx1151__)

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int8_t  __attribute__((ext_vector_type(16))) int8x16_t;
typedef int32_t __attribute__((ext_vector_type(4)))  int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE 64
#define SLOT_TILE 16
#define WARPS 4
#define M_PER_WARP 16

__launch_bounds__(128, 1)
extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_k8_4w_gfx1151(
    const unsigned long long*  __restrict__ expert_weight_ptrs,
    const int*                 __restrict__ expert_tile_ids,
    const int*                 __restrict__ sorted_slot_index,
    const block_q8_1_mmq*      __restrict__ X_q8_src,
    float*                     __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int x_src_rows
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
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int blocks128 = K / 128;

    __shared__ block_q8_1_mmq lds_x[SLOT_TILE];

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int src_lane = 2 * j + (lane >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        for (int kb_in_group = 0; kb_in_group < 2; kb_in_group++) {
            const int kb_q8 = 2 * g + kb_in_group;
            if (kb_q8 >= blocks128) break;

            // Cooperative copy: 16 slots x 144 B = 576 u32 words.
            unsigned int* lds_u32 = reinterpret_cast<unsigned int*>(lds_x);
            #pragma unroll
            for (int pass = 0; pass < 5; pass++) {
                const int word = pass * 128 + tid;
                if (word < SLOT_TILE * 36) {
                    const int slot_lane = word / 36;
                    const int word_in_block = word - slot_lane * 36;
                    const int slot_idx = slot_start + slot_lane;
                    int x_row = -1;
                    if (slot_idx < m_total) {
                        const int flat = sorted_slot_index[slot_idx];
                        if (flat >= 0) {
                            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
                        }
                    }
                    if (x_row >= 0) {
                        const block_q8_1_mmq* xb_global =
                            X_q8_src + (long long)kb_q8 * x_src_rows + x_row;
                        const unsigned int* src_u32 = reinterpret_cast<const unsigned int*>(xb_global);
                        lds_u32[word] = src_u32[word_in_block];
                    } else {
                        lds_u32[word] = 0u;
                    }
                }
            }
            __syncthreads();

            const block_q8_1_mmq* xb = lds_x + lane16;

            float d_x0 = 0.0f, sum_x0 = 0.0f;
            float d_x1 = 0.0f, sum_x1 = 0.0f;
            float d_x2 = 0.0f, sum_x2 = 0.0f;
            float d_x3 = 0.0f, sum_x3 = 0.0f;
            half2 dm0 = xb->ds4[0];
            half2 dm1 = xb->ds4[1];
            half2 dm2 = xb->ds4[2];
            half2 dm3 = xb->ds4[3];
            float2 dm0_f = __half22float2(dm0);
            float2 dm1_f = __half22float2(dm1);
            float2 dm2_f = __half22float2(dm2);
            float2 dm3_f = __half22float2(dm3);
            d_x0 = dm0_f.x; sum_x0 = dm0_f.y;
            d_x1 = dm1_f.x; sum_x1 = dm1_f.y;
            d_x2 = dm2_f.x; sum_x2 = dm2_f.y;
            d_x3 = dm3_f.x; sum_x3 = dm3_f.y;

            int32x4_t a_i32[8];
            int32x4_t b_i32[8];

            #pragma unroll
            for (int kt_idx = 0; kt_idx < 8; kt_idx++) {
                const int sb = kt_idx >> 1;
                const int kt = kt_idx & 1;
                const int k_off = kb_in_group * 128 + sb * 32 + kt * 16;

                const unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2);
                const unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4);

                int8x16_t a_vec;
                a_vec[0]  = (int8_t)((pk0 >>  0) & 0xFu);
                a_vec[1]  = (int8_t)((pk0 >>  4) & 0xFu);
                a_vec[2]  = (int8_t)((pk0 >>  8) & 0xFu);
                a_vec[3]  = (int8_t)((pk0 >> 12) & 0xFu);
                a_vec[4]  = (int8_t)((pk0 >> 16) & 0xFu);
                a_vec[5]  = (int8_t)((pk0 >> 20) & 0xFu);
                a_vec[6]  = (int8_t)((pk0 >> 24) & 0xFu);
                a_vec[7]  = (int8_t)((pk0 >> 28) & 0xFu);
                a_vec[8]  = (int8_t)((pk1 >>  0) & 0xFu);
                a_vec[9]  = (int8_t)((pk1 >>  4) & 0xFu);
                a_vec[10] = (int8_t)((pk1 >>  8) & 0xFu);
                a_vec[11] = (int8_t)((pk1 >> 12) & 0xFu);
                a_vec[12] = (int8_t)((pk1 >> 16) & 0xFu);
                a_vec[13] = (int8_t)((pk1 >> 20) & 0xFu);
                a_vec[14] = (int8_t)((pk1 >> 24) & 0xFu);
                a_vec[15] = (int8_t)((pk1 >> 28) & 0xFu);

                int8x16_t b_vec;
                const int8_t* qs_p = xb->qs + sb * 32 + kt * 16;
                b_vec = *(const int8x16_t*)qs_p;

                a_i32[kt_idx] = *(const int32x4_t*)&a_vec;
                b_i32[kt_idx] = *(const int32x4_t*)&b_vec;
            }

            int32x8_t cacc0 = {0, 0, 0, 0, 0, 0, 0, 0};
            int32x8_t cacc1 = {0, 0, 0, 0, 0, 0, 0, 0};
            int32x8_t cacc2 = {0, 0, 0, 0, 0, 0, 0, 0};
            int32x8_t cacc3 = {0, 0, 0, 0, 0, 0, 0, 0};

            cacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[0], true, b_i32[0], cacc0, true);
            cacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[1], true, b_i32[1], cacc0, true);
            cacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[2], true, b_i32[2], cacc1, true);
            cacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[3], true, b_i32[3], cacc1, true);
            cacc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[4], true, b_i32[4], cacc2, true);
            cacc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[5], true, b_i32[5], cacc2, true);
            cacc3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[6], true, b_i32[6], cacc3, true);
            cacc3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[7], true, b_i32[7], cacc3, true);

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                acc[j] += sc_row[j] * d_x0 * (float)cacc0[j] + zp_row[j] * sum_x0;
                acc[j] += sc_row[j] * d_x1 * (float)cacc1[j] + zp_row[j] * sum_x1;
                acc[j] += sc_row[j] * d_x2 * (float)cacc2[j] + zp_row[j] * sum_x2;
                acc[j] += sc_row[j] * d_x3 * (float)cacc3[j] + zp_row[j] * sum_x3;
            }
            __syncthreads();
        }
    }

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

extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_k8_4w_gfx1151(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
