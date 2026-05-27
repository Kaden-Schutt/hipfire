// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// i8 WMMA MMQ MoE grouped-GEMM for HFQ4G128 (ParoQuant) on gfx1151.
// Port of `gemm_hfq4g256_moe_grouped_mmq.gfx1151.hip` with the HFQ4G128
// 72 B/group stride (vs 136 B/group G256) and 128 K-elements per HFQ4
// group (vs 256). One Q8_1 mmq block (128 K-elements) per HFQ4 group
// instead of two.
//
// Pairs with the same scatter pipeline + `ensure_q8_1_mmq_x` quantization
// helper as the G256 sister. ~2× FP16 WMMA throughput on gfx1151 (~140
// vs ~71 TFLOPS theoretical) — the FP16 grouped WMMA kernel measured
// 68.5% of GPU time in Phase 4 (rocprofv3 attribution 2026-05-21,
// 150ms/220ms, 25.8 GiB/s — clearly compute-bound). Doubling compute
// throughput should yield 30-50% wall reduction on the routed-expert
// portion.
//
// X must be pre-quantized to Q8_1 (block_q8_1_mmq) via
// `quantize_q8_1_mmq_ds4` before dispatch — same as the G256 sister.
//
// Shapes:
//   expert_weight_ptrs : [E] u64 (HFQ4-G128 per-expert weights)
//   expert_tile_ids    : [m_total / 16] i32
//   sorted_slot_index  : [m_total] i32
//   X_q8_src           : [(K/128) × x_src_rows] block_q8_1_mmq
//   Y_grouped          : [m_total × M] f32
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.
//
// Per HFQ4G128 layout (kernels/src/gemv_hfq4g128.hip):
//   - 128 elements per group (4 Q8_1 sub-blocks of 32 each)
//   - 8-byte header: scale F32 + zero F32
//   - 64-byte payload: 64 packed 4-bit nibbles

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

// `x_row_div` controls how sorted_slot_index maps to X_q8_src rows:
//   gate_up: X_src_f32 was [N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src_f32 was [N*K_TOP × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
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
        // HFQ4 per-group scale (sc) and zero (zp) for THIS lane's row.
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        // Broadcast sc/zp via __shfl: output row r = row_start + 2*j + (tid>>4),
        // source lane = 2*j + (tid>>4), j in [0,8).
        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int src_lane = 2 * j + (tid >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        // HFQ4G128: 1 Q8_1 mmq block per HFQ4 group (128 K-elements / 128 = 1).
        // No outer kb_in_group loop needed — directly map g to the mmq block.
        const int kb_q8 = g;
        const block_q8_1_mmq* xb = (x_row >= 0)
            ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
            : nullptr;

        // 4 sub-blocks of 32 K each. Per sub-block: 2 WMMA calls (K=16
        // each), int32 accumulator, then scale FMA.
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            float d_x_self  = 0.0f;
            float sum_x_self = 0.0f;
            if (xb != nullptr) {
                half2 dm = xb->ds4[sb];
                float2 dm_f = __half22float2(dm);
                d_x_self   = dm_f.x;
                sum_x_self = dm_f.y;
            }

            int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};

            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                // K-offset within the HFQ4G128 group: sb*32 + kt*16.
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

                int32x4_t a_i32 = *(const int32x4_t*)&a_vec;
                int32x4_t b_i32 = *(const int32x4_t*)&b_vec;

                cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, a_i32, true, b_i32, cacc, true
                );
            } // for kt

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float dot_f = (float)cacc[j];
                acc[j] += sc_row[j] * d_x_self * dot_f
                        + zp_row[j] * sum_x_self;
            }
        } // for sb
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

// Stub for non-gfx1151 archs.
extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
