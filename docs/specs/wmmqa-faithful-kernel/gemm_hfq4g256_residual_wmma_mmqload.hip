// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// wmmqa — int8-WMMA GEMM for the HFQ4-G256 dense residual on gfx11 (RDNA3).
//
// Motivation: the residual prefill GEMM is the dominant prefill cost and the
// default path runs scalar sdot4 MMQ (no matrix cores). A confirmed A/B showed
// fp16-WMMA is only ~15% slower than MMQ despite 2× the activation bytes — the
// GEMM is partially compute-bound and matrix cores help. wmmqa combines MMQ's
// light int8 activation traffic (Q8_1) with WMMA's matrix cores.
//
// Construction:
//   - Tiling: lifted verbatim from gemm_hfq4g256_residual_wmma_k2.hip — the
//     16×16×16 WMMA tiling, A=weight-rows / B=X-cols mapping, and the
//     acc[j] = C[2*j + (tid>>4)][tid & 15] output convention carry over to
//     int8 unchanged on RDNA3 (only dtype half→int8, accumulator
//     float→int32 change).
//   - Load: the Q8_1 activation (B operand) is the pre-quantized
//     block_q8_1_mmq scratch produced by quantize_q8_1_mmq_ds4 (same buffer the
//     MMQ residual path uses), and the HFQ4 4-bit nibble unpack of the weight
//     (A operand) is grafted from the MMQ body.
//   - Matmul: __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 (RDNA3 wave32, int8 in
//     / int32 accumulate). signed_a=true/signed_b=true; HFQ4 nibbles are raw
//     [0,15] (MSB always zero) so signed interpretation is safe.
//   - Epilogue: dequant per 32-K sub-block — w = sc*nibble + zp, so
//     acc[j] += sc_row[j] * d_x * (float)cacc[j] + zp_row[j] * sum_x  (this is
//     algebraically identical to the MMQ body's biased-nibble + zp_eff form:
//     raw nibble n with plain zp == (n-8) with zp_eff=zp+8*sc). Y += sum
//     (residual add internal; Y is never reused as scratch).
//
// CRITICAL CORRECTNESS SEAM — per-32-element Q8_1 scaling.
//   The Q8_1 activation scale (d_x) and float-sum (sum_x) are PER-32-ELEMENT
//   sub-block (QK8_1 = 32). One WMMA i8 tile consumes 16 K, so 32 K = 2 WMMA
//   tiles. The int32 accumulator MUST be flushed (multiplied by sc*d_x and
//   biased by zp*sum_x) every 32 K — i.e. after every PAIR of WMMA tiles —
//   to match the MMQ body's dequant granularity EXACTLY. A coarser flush
//   (e.g. once per 256-K group) would apply one block's scale to another
//   block's products and wreck NRMSE. We reset a fresh int32 cacc at the top
//   of each sub-block, run exactly 2 WMMAs into it, then resolve the float FMA.
//
// int32 accumulate is order-independent, so any K-split is deterministic for
// free (no _ksplit_det machinery needed).
//
// Grid: [ceil(M/16), ceil(batch_size/16)]. Block: [32]. LDS: 0.
// X_q8_src layout: block_q8_1_mmq[ (k/128) * batch_size + col ]  (1 block = 128 K
// = 4 sub-blocks of 32). HFQ4 weight group stride = 136 B (4 B sc, 4 B zp,
// 128 B nibble body).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

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
extern "C" __global__ void gemm_hfq4g256_residual_wmma_mmqload(
    const char*           __restrict__ A,          // HFQ4 weights, 136 B/group
    const block_q8_1_mmq* __restrict__ Xq,         // Q8_1 activations [K/128 × batch_size]
    float*                __restrict__ Y,          // [batch_size × M], col-major-ish (col*M+row)
    int M, int K, int batch_size
) {
    const int tid        = threadIdx.x;
    const int row_start  = blockIdx.x * 16;   // weight rows (A operand, M-dim)
    const int batch_start = blockIdx.y * 16;  // X cols (B operand, batch-dim)

    if (row_start >= M || batch_start >= batch_size) return;

    const int lane   = tid & 31;
    const int lane16 = tid & 15;

    // Each lane (0..15) owns one weight row in the A 16×16 tile; lanes 16..31
    // mirror lanes 0..15. Pad OOB rows to the last valid row (same trick as the
    // fp16 wmma_k2 sibling) so the WMMA load stays in-bounds; the bad rows are
    // masked at store time.
    const int safe_row = (row_start + lane16 < M) ? (row_start + lane16) : (M - 1);

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    // Each lane (0..15) owns one X column (batch slot). OOB cols substitute a
    // null Q8_1 block (zero contribution). lanes 16..31 mirror 0..15.
    const int x_col = batch_start + lane16;
    const bool x_valid = (x_col < batch_size);

    const int blocks128 = K / 128;   // number of Q8_1 128-K blocks across the row

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;

        // HFQ4 per-group scale/zero for THIS lane's row. Broadcast each output
        // row's sc/zp to the lanes that will resolve that row's FMA. Output row
        // r = row_start + 2*j + (tid>>4)  ⇒  source lane = 2*j + (tid>>4).
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int src_lane = 2 * j + (tid >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        // 2 Q8_1 128-K blocks per 256-K group.
        #pragma unroll
        for (int kb_in_group = 0; kb_in_group < 2; kb_in_group++) {
            const int kb_q8 = 2 * g + kb_in_group;
            if (kb_q8 >= blocks128) break;

            // Q8_1 block for this lane's X column (or null for padding).
            const block_q8_1_mmq* xb = x_valid
                ? (Xq + (long long)kb_q8 * batch_size + x_col)
                : nullptr;

            // ── 4 sub-blocks of 32 K, each its own (d_x, sum_x) scale. ──────
            // SEAM: a fresh int32 cacc per sub-block, exactly 2 WMMAs (=32 K)
            // into it, then resolve the float FMA against that sub-block's
            // (d_x, sum_x). This makes the dequant boundary land at exactly
            // 32 K, identical to the MMQ body.
            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                float d_x = 0.0f, sum_x = 0.0f;
                if (xb != nullptr) {
                    const float2 dm = __half22float2(xb->ds4[sb]);
                    d_x   = dm.x;
                    sum_x = dm.y;
                }

                int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};

                // 2 K-tiles of 16 = one 32-K sub-block.
                #pragma unroll
                for (int kt = 0; kt < 2; kt++) {
                    const int k_off = kb_in_group * 128 + sb * 32 + kt * 16;

                    // ── A operand: HFQ4 nibbles → int8 (raw [0,15]). ──
                    // 16 nibbles = 8 bytes = 2 packed uint32 at gp+8+k_off/2.
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

                    // ── B operand: Q8_1 int8 (16 contiguous of this sub-block). ──
                    int8x16_t b_vec;
                    if (xb != nullptr) {
                        const int8_t* qs_p = xb->qs + sb * 32 + kt * 16;
                        b_vec = *(const int8x16_t*)qs_p;
                    } else {
                        b_vec = (int8x16_t){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
                    }

                    const int32x4_t a_i32 = *(const int32x4_t*)&a_vec;
                    const int32x4_t b_i32 = *(const int32x4_t*)&b_vec;

                    // signed_a = signed_b = true (values fit signed int8 range).
                    cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, a_i32, true, b_i32, cacc, true
                    );
                }

                // ── Per-32-K flush: int32 → float dequant + residual-bound FMA.
                //   w = sc*nibble + zp  ⇒  contribution =
                //     sc_row[j] * d_x * Σ(nibble · q_int8) + zp_row[j] * Σ x
                //   = sc_row[j] * d_x * cacc[j] + zp_row[j] * sum_x
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[j] += sc_row[j] * d_x * (float)cacc[j]
                            + zp_row[j] * sum_x;
                }
            } // sb
        } // kb_in_group
    } // g

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (M-dim), B operand = batch X cols (batch-dim).
    // Residual add (+=) — Y holds the residual; never reuse Y as scratch.
    const int out_col = batch_start + lane16;  // batch index
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] += acc[j];
            }
        }
    }
}
