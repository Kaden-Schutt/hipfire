// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// SGLang-style grouped-WMMA-GEMM for MoE prefill — MQ2-Lloyd port.
// Mirrors `gemm_hfq4g256_moe_grouped_wmma_k2.hip` exactly except for the
// weight decode: HFQ4G256 (136 B/group, affine `sc*q+zp` on 4-bit packs)
// becomes MQ2-Lloyd-G256 (72 B/group, codebook lookup `cb[idx]` on 2-bit
// packs). Same scatter pipeline (moe_scatter_fused_k8), same unscatter
// (moe_gate_up_unscatter_k8 / moe_down_combine_grouped_k8), same grid.
//
// MQ2-Lloyd block layout (72 B/group):
//   bytes [0..8)   : 4 × fp16 codebook entries (sorted ascending at quant-time)
//   bytes [8..72)  : 64 bytes packed 2-bit indices (4 per byte, LSB-first)
//
// Per K-tile (16 weights) we read 1 × uint32 (16 × 2 bits = 32 bits = 4 bytes).
// HFQ4 read 2 × uint32 per K-tile (16 × 4 bits = 64 bits = 8 bytes).
//
// Decode (per weight):
//   HFQ4:   w = sc_h * (q & 0xF)  + zp_h          (q ∈ 0..15)
//   MQ2L:   w = cb[(packed >> sh) & 3]            (idx ∈ 0..3)
//
// MQ2-Lloyd has HALF the weight-read bandwidth of HFQ4 (72 vs 136 B/group)
// at the cost of higher decode-ALU (codebook select vs single FMA).
// At DeepSeek V4 shape the GEMM is decode-bound on scalar K4 — Gate 2 measured
// 1.71× speedup of HFQ4-grouped over MQ2-Lloyd-scalar. Porting MQ2-Lloyd
// onto the grouped path is the structural lever; expected ~1.4–1.6× over
// MQ2-Lloyd scalar K4 (some headroom is consumed by the 4-way codebook
// select vs HFQ4's single FMA).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
// Bumping to (32, 8) was measured neutral / slightly worse for this
// kernel at PP_BATCH=1024 — the Lloyd codebook decode loop is ALU-bound
// enough that more concurrent waves don't help (vs Q8's simpler dequant
// where (32, 8) wins +1%). Kept at (32, 2) — see gemm_q8_0_wmma.hip
// for the contrasting case.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows × K]
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total
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
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 72;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 72;
        // 4 fp16 codebook entries → 4 half registers.
        const __half* cb_h = reinterpret_cast<const __half*>(gp);
        _Float16 cb0 = (_Float16)__half2float(cb_h[0]);
        _Float16 cb1 = (_Float16)__half2float(cb_h[1]);
        _Float16 cb2 = (_Float16)__half2float(cb_h[2]);
        _Float16 cb3 = (_Float16)__half2float(cb_h[3]);
        // Bilinear-interpolation decomposition of cb[q] for q ∈ {0,1,2,3}:
        //   q = qhi*2 + qlo, qhi, qlo ∈ {0,1}
        //   cb[q] = cb0 + qlo*(cb1-cb0) + qhi*(cb2-cb0) + qhi*qlo*(cb3-cb2-cb1+cb0)
        //         = cb0 + qlo*d01 + qhi*d02 + (qhi*qlo)*d_xor
        // 3 FMA per weight, fully vectorizable. Critical: avoids the
        // ternary-chain cb[q] formulation which the compiler refused to
        // vectorize → no WMMA emitted (verified via objdump 2026-05-22).
        const _Float16 d01   = cb1 - cb0;                 // qlo bit
        const _Float16 d02   = cb2 - cb0;                 // qhi bit
        const _Float16 d_xor = cb3 - cb2 - cb1 + cb0;     // qhi & qlo
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            // Index loads: 1 uint32 per K-tile (16 weights × 2 bits = 32 bits).
            unsigned int pa = *(const unsigned int*)(gp + 8 + k_off_a / 4);
            unsigned int pb = *(const unsigned int*)(gp + 8 + k_off_b / 4);

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            // Vectorizable DQ: extract qlo/qhi bits per lane, FMA chain.
            // `(_Float16)(int)` is element-wise convert; compiler vectorizes.
            #define DQ(reg, i, pk, sh) do { \
                _Float16 qlo = (_Float16)(int)(((pk) >> (sh))     & 0x1u); \
                _Float16 qhi = (_Float16)(int)(((pk) >> ((sh)+1)) & 0x1u); \
                reg[i] = cb0 + qlo * d01 + qhi * d02 + (qlo * qhi) * d_xor; \
            } while (0)

            half16_t a_a;
            DQ(a_a, 0,  pa,  0); DQ(a_a, 1,  pa,  2); DQ(a_a, 2,  pa,  4); DQ(a_a, 3,  pa,  6);
            DQ(a_a, 4,  pa,  8); DQ(a_a, 5,  pa, 10); DQ(a_a, 6,  pa, 12); DQ(a_a, 7,  pa, 14);
            DQ(a_a, 8,  pa, 16); DQ(a_a, 9,  pa, 18); DQ(a_a, 10, pa, 20); DQ(a_a, 11, pa, 22);
            DQ(a_a, 12, pa, 24); DQ(a_a, 13, pa, 26); DQ(a_a, 14, pa, 28); DQ(a_a, 15, pa, 30);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            DQ(a_a, 0,  pb,  0); DQ(a_a, 1,  pb,  2); DQ(a_a, 2,  pb,  4); DQ(a_a, 3,  pb,  6);
            DQ(a_a, 4,  pb,  8); DQ(a_a, 5,  pb, 10); DQ(a_a, 6,  pb, 12); DQ(a_a, 7,  pb, 14);
            DQ(a_a, 8,  pb, 16); DQ(a_a, 9,  pb, 18); DQ(a_a, 10, pb, 20); DQ(a_a, 11, pb, 22);
            DQ(a_a, 12, pb, 24); DQ(a_a, 13, pb, 26); DQ(a_a, 14, pb, 28); DQ(a_a, 15, pb, 30);

            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
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
