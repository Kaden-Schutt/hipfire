// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// k8 (deepest K-tile pipeline) sibling of
// `gemm_paro_q4g128_moe_grouped_mmq.gfx1151.hip`. Processes all 4
// sub-blocks of the (single) Q8_1 mmq block per HFQ4G128 group in a
// single inner iteration — 8 WMMAs into 4 independent int32
// accumulators before the per-sub-block scale FMA resolves. Same
// kernarg signature + grid + block geometry as the k2 sibling; only
// the unroll depth differs.
//
// Rationale: i8 WMMA has ~2× FP16 WMMA throughput on gfx11 so per-tile
// latency is short — a deeper pipeline hides more dequant + (d, sum)
// load latency than the k2 baseline. The G256 sister's k8 variant was
// the "natural next experiment after k4 hit +4.6%". For G128 with
// only 4 sub-blocks per HFQ4 group, k8 = "all in one pass" (k4 wouldn't
// add new pipelining beyond k2 since only 2 outer iterations either way).
//
// X must be pre-quantized to Q8_1 via `quantize_q8_1_mmq_ds4` before
// dispatch. gfx1151-only; stub on other archs.

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

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
    const unsigned long long*  __restrict__ expert_weight_ptrs, // [E]
    const int*                 __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                 __restrict__ sorted_slot_index,  // [m_total]
    const block_q8_1_mmq*      __restrict__ X_q8_src,           // [K/128 × x_src_rows]
    float*                     __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total, int x_src_rows
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 128;
    const char* row_base = A + (long long)safe_row * groups_per_row * 72;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 72;
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        // Broadcast sc/zp per output row (output row r = row_start + 2*j + (tid>>4),
        // source lane = 2*j + (tid>>4), j in [0,8)).
        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int src_lane = 2 * j + (tid >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        // HFQ4G128: 1 Q8_1 mmq block per HFQ4 group.
        const int kb_q8 = g;
        const block_q8_1_mmq* xb = (x_row >= 0)
            ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
            : nullptr;

        // 4 (d_x, sum_x) pairs for the 4 sub-blocks.
        float d_x0 = 0.0f, sum_x0 = 0.0f;
        float d_x1 = 0.0f, sum_x1 = 0.0f;
        float d_x2 = 0.0f, sum_x2 = 0.0f;
        float d_x3 = 0.0f, sum_x3 = 0.0f;
        if (xb != nullptr) {
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
        }

        // Pre-load all 8 K-tiles' operands. 4 sub-blocks × 2 kt = 8 K-tiles.
        int32x4_t a_i32[8];
        int32x4_t b_i32[8];

        #pragma unroll
        for (int kt_idx = 0; kt_idx < 8; kt_idx++) {
            const int sb = kt_idx >> 1;
            const int kt = kt_idx & 1;
            const int k_off = sb * 32 + kt * 16;

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
            if (xb != nullptr) {
                const int8_t* qs_p = xb->qs + sb * 32 + kt * 16;
                b_vec = *(const int8x16_t*)qs_p;
            } else {
                b_vec = (int8x16_t){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            }

            a_i32[kt_idx] = *(const int32x4_t*)&a_vec;
            b_i32[kt_idx] = *(const int32x4_t*)&b_vec;
        }

        // Issue 8 WMMAs into 4 independent accumulators (2 per sub-block).
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

        // Scale FMA per sub-block.
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float dot0 = (float)cacc0[j];
            float dot1 = (float)cacc1[j];
            float dot2 = (float)cacc2[j];
            float dot3 = (float)cacc3[j];
            acc[j] += sc_row[j] * d_x0 * dot0 + zp_row[j] * sum_x0;
            acc[j] += sc_row[j] * d_x1 * dot1 + zp_row[j] * sum_x1;
            acc[j] += sc_row[j] * d_x2 * dot2 + zp_row[j] * sum_x2;
            acc[j] += sc_row[j] * d_x3 * dot3 + zp_row[j] * sum_x3;
        }
    } // for g

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

#else

extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
