// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Non-grouped i8 WMMA MMQ port of `gemm_hfq4g128.hip` for gfx1151 (Strix Halo).
//
// Why this exists (rocprof 2026-05-22 attribution on A3B-PARO pp256 HEAD):
// `gemm_hfq4g128` = **66.31%** of prefill wall time, the single largest
// kernel slice. It's the catch-all batched GEMM for ParoQ4G128 weights NOT
// covered by the routed-MoE grouped MMQ k8 path — shared-expert gate/up/
// down, LA wqkv/wz, FullAttn QKV/wo. The G256 sister has a full arch zoo
// (RDNA2/CDNA1/RDNA3 MMQ + RDNA3/RDNA4 WMMA in dispatch.rs:13756-13830);
// G128 has nothing but the baseline non-WMMA wave32 tiled kernel. This
// closes that gap by reusing the i8 WMMA pattern from
// `gemm_paro_q4g128_moe_grouped_mmq_k8.gfx1151.hip` minus the expert
// scatter/gather, so it can replace the baseline directly.
//
// Activation contract: X must be pre-quantized to Q8_1 mmq DS4 layout via
// `quantize_q8_1_mmq_ds4` before dispatch (same helper used by the routed
// grouped path). The dispatcher in dispatch.rs:gemm_hfq4g128 handles that
// step transparently when this fast-path is selected.
//
// Output contract: y[batch_idx * M + row] = dot(A_row, x_batch) — direct
// store, NOT residual. Matches the baseline `gemm_hfq4g128` contract; all
// in-tree callers (qwen35.rs) write into a fresh scratch buffer.
//
// Shape requirements (caller-enforced; we assert without bounds-checking
// the inner loop hot path):
//   M % 16 == 0   — output rows are 16-tile aligned
//   N % 16 == 0   — batch rows are 16-tile aligned (a3b prefill always)
//   K % 128 == 0  — HFQ4-G128 group size
//
// Grid: [M/16, N/16, 1]. Block: [32]. One wave32 per 16×16 output tile.

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
extern "C" __global__ void gemm_hfq4g128_mmq_gfx1151(
    const char*                __restrict__ A,            // [M × (K/128) × 72] HFQ4-G128 packed
    const block_q8_1_mmq*      __restrict__ X_q8_src,     // [K/128 × N] Q8_1 mmq DS4 (pre-quantized)
    float*                     __restrict__ Y,            // [N × M] direct store
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int slot_start = blockIdx.y * 16;
    if (row_start >= M || slot_start >= N) return;

    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    const int x_row = (slot_idx < N) ? slot_idx : -1;

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 128;
    const char* row_base = A + (long long)safe_row * groups_per_row * 72;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 72;
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        // Broadcast sc/zp per output row. The 16×16 WMMA tile maps each
        // accumulator slot acc[j] to output row r = row_start + 2*j + (tid>>4)
        // (see RDNA3 wave32 WMMA C-layout); the (sc, zp) pair lives in the
        // lane that loaded weight row r — namely src_lane = 2*j + (tid>>4).
        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int src_lane = 2 * j + (tid >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        // HFQ4G128: exactly 1 Q8_1 mmq block per HFQ4 group.
        const int kb_q8 = g;
        const block_q8_1_mmq* xb = (x_row >= 0)
            ? (X_q8_src + (long long)kb_q8 * N + x_row)
            : nullptr;

        // 4 (d_x, sum_x) pairs for the 4 Q8_1 sub-blocks within this 128-K group.
        float d_x0 = 0.0f, sum_x0 = 0.0f;
        float d_x1 = 0.0f, sum_x1 = 0.0f;
        float d_x2 = 0.0f, sum_x2 = 0.0f;
        float d_x3 = 0.0f, sum_x3 = 0.0f;
        if (xb != nullptr) {
            float2 dm0_f = __half22float2(xb->ds4[0]);
            float2 dm1_f = __half22float2(xb->ds4[1]);
            float2 dm2_f = __half22float2(xb->ds4[2]);
            float2 dm3_f = __half22float2(xb->ds4[3]);
            d_x0 = dm0_f.x; sum_x0 = dm0_f.y;
            d_x1 = dm1_f.x; sum_x1 = dm1_f.y;
            d_x2 = dm2_f.x; sum_x2 = dm2_f.y;
            d_x3 = dm3_f.x; sum_x3 = dm3_f.y;
        }

        // Pre-load all 8 K-tiles' operands (4 sub-blocks × 2 K-tiles = 8 tiles).
        int32x4_t a_i32[8];
        int32x4_t b_i32[8];

        #pragma unroll
        for (int kt_idx = 0; kt_idx < 8; kt_idx++) {
            const int sb = kt_idx >> 1;          // sub-block 0..3
            const int kt = kt_idx & 1;           // K-tile within sub-block: 0 or 1
            const int k_off = sb * 32 + kt * 16; // K-offset in the 128-K group

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

        // 8 i8 WMMAs into 4 independent int32 accumulators (2 per sub-block).
        // Independent accumulators let the scale FMA close per-sub-block on
        // 4 narrow chains rather than one long serial dep.
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

        // Scale FMA per sub-block: combine (sc, zp, d_x, sum_x).
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
    }

    // RDNA3 wave32 WMMA C-layout store: acc[j] sits at output
    // Y[out_col=slot_start+lane][out_row=row_start + 2*j + (tid>>4)].
    // Output Y is laid out [N × M] row-major (matches baseline
    // `gemm_hfq4g128`: `y[(batch_start + b) * M + row]`).
    const int out_col = slot_start + lane;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#else

// Stub for non-gfx1151 archs (initial port; broader gfx11 enablement
// requires per-arch channel-test on each target).
extern "C" __global__ void gemm_hfq4g128_mmq_gfx1151(
    const char*, const void*, float*, int, int, int
) {}

#endif
