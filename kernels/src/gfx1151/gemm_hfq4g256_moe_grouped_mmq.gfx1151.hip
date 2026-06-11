#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// i8 WMMA MMQ MoE grouped-GEMM port for gfx1151 (Strix Halo iGPU).
//
// This combines two existing patterns:
//   - `gemm_hfq4g256_residual_mmq.hip`     : i8 WMMA (`wmma_i32_16x16x16_iu8_w32`)
//                                            on Q8_1-quantized activations, with
//                                            HFQ4 scale/zero applied via Q8_1
//                                            scale/sum correction.
//   - `gemm_hfq4g256_moe_grouped_wmma_k2.hip`: SGLang-style grouped MoE dispatch
//                                              with expert_tile_ids + sorted_slot_index
//                                              scatter and a 16×16 output tile per WG.
//
// Rationale: on gfx1151 the FP16 WMMA grouped path is compute-bound at
// ~71 TFLOPS (near peak). The i8 WMMA path on gfx11 runs at ~2× the
// FLOP rate (~140 TFLOPS theoretical) thanks to the `i32_16x16x16_iu8`
// intrinsic. The compute-bound regime is the prefill grouped-GEMM (gate_up
// and down) for MoE A3B at batch ≥ 256.
//
// X must be pre-quantized to Q8_1 (block_q8_1_mmq) via `quantize_q8_1_mmq_ds4`
// before dispatch. The kernarg layout is identical to the FP16 sister
// kernel except the `X_src` pointer is the Q8_1 packed buffer (NOT the
// FP16 conversion buffer).
//
// Shapes:
//   expert_weight_ptrs : [E] u64 (HFQ4-G256 per-expert weights)
//   expert_tile_ids    : [m_total / 16] i32 (per-tile expert id, -1 sentinel)
//   sorted_slot_index  : [m_total] i32 (flat slot = token * K_TOP + k_rank,
//                                       or -1 padding)
//   X_q8_src           : [(K/128) × x_src_rows] block_q8_1_mmq (Q8_1-packed
//                                                                 activations)
//   Y_grouped          : [m_total × M] f32 (direct store, per-slot output rows)
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.
//
// Per-block contract: each block owns a 16-row × 16-slot output tile.
// All 16 slots in this tile belong to the same expert (scatter pipeline
// guarantees expert_offsets[E] is a multiple of 16). The kernel:
//   1. Looks up expert id from expert_tile_ids[blockIdx.y]; -1 ⇒ skip.
//   2. Loads A operand base = expert_weight_ptrs[expert_id].
//   3. Per lane (tid & 15): reads sorted_slot_index[tile_y*16 + lane],
//      recovers the source x_row via x_row_div. If sorted_slot_index = -1,
//      the lane substitutes zero Q8_1 blocks.
//   4. Iterates over groups (256 K-elements each = 2 Q8_1 blocks). Per
//      Q8_1 sub-block of 32 K-elements:
//        - 2 i8 WMMA calls (K=16 each) accumulate int32 → C_frag
//        - Apply scale FMA: sum_f32 += sc * d_x * C_frag + zp * d_x * sum_x
//      where (sc, zp) are HFQ4 per-group scales and (d_x, sum_x) are
//      Q8_1 per-sub-block scale/sum.
//
// Compile gate: gfx1151 only for the initial port. The other gfx11 archs
// (gfx1100/1101/1102/1103/1150) can be enabled later after channel testing
// on each target.

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
//   gate_up: X_src_f32 had [N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src_f32 had [N*K_TOP × K], x_row_div = 1 → x_row = flat
//
// X_q8_src is the result of running `quantize_q8_1_mmq_ds4` on a buffer of
// `x_src_rows × K` floats. Layout: X_q8_src[kb_q8 * x_src_rows + x_row].
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_gfx1151(
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

    // Sentinel early-return: tile_y beyond live m_total.
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X gather: each lane (tid & 15) gets its own routed slot. Padding lanes
    // (sorted_slot_index == -1) substitute zero Q8_1 contributions.
    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }

    // A-row selection: pad OOB to last valid row (same trick as the FP16
    // sibling). Each lane owns ONE row in the WMMA A operand (rows 0-15
    // for the 16×16 A-tile), with lanes 16-31 mirroring lanes 0-15.
    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    // Output accumulator in float (RDNA3 wave32 layout:
    // acc[j] = Y[2*j + (tid>>4)][tid & 15], with j ∈ [0,8) covering 16
    // M-rows × 2 lane-halves).
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const int blocks128 = K / 128;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        // HFQ4 per-group scale (sc) and zero (zp) for THIS lane's row.
        // We need to broadcast each output row's sc/zp via __shfl below.
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        // 8 j-positions × 2 lane halves = 16 output rows. Lane `m` (in [0,16))
        // has sc/zp for row `row_start + m`. Output row r = row_start + 2*j + (tid>>4)
        // ⇒ source lane = 2*j + (tid>>4).
        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int src_lane = 2 * j + (tid >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        // 2 Q8_1 blocks per group (256 K-elements / 128 = 2). Each block
        // holds 4 sub-blocks of 32 K = 4 (d_x, sum_x) pairs.
        for (int kb_in_group = 0; kb_in_group < 2; kb_in_group++) {
            const int kb_q8 = 2 * g + kb_in_group;
            if (kb_q8 >= blocks128) break;

            // Pull the Q8_1 block for this lane's slot (or null for padding).
            const block_q8_1_mmq* xb = (x_row >= 0)
                ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
                : nullptr;

            // 4 sub-blocks of 32 K each. Per sub-block we do 2 WMMA calls
            // (K=16 each) on an int32 accumulator that resets at the start
            // of the sub-block, then apply the scale FMA.
            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                // (d_x, sum_x) for this slot's sub-block.
                float d_x_self  = 0.0f;
                float sum_x_self = 0.0f;
                if (xb != nullptr) {
                    half2 dm = xb->ds4[sb];
                    float2 dm_f = __half22float2(dm);
                    d_x_self   = dm_f.x;
                    sum_x_self = dm_f.y;
                }

                // The output-element ownership puts slot s = tid & 15 ⇒ lane = s.
                // d_x/sum_x for each output element belongs to the slot it
                // is associated with, which is exactly THIS lane's slot.
                // So no shuffle needed — every lane uses its own d_x/sum_x.

                // Two WMMA i8 calls per sub-block. K offsets within sub-block:
                //   kt 0: K[0..15]  in sub-block
                //   kt 1: K[16..31] in sub-block
                int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};

                #pragma unroll
                for (int kt = 0; kt < 2; kt++) {
                    // ----- A operand (HFQ4 nibbles → int8) -----
                    // Position within the 256-element group:
                    //   K-offset in group = kb_in_group*128 + sb*32 + kt*16
                    // Nibbles are packed 2-per-byte starting at gp+8.
                    // 16 nibbles = 8 bytes = 2 packed uint32.
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

                    // ----- B operand (Q8_1 int8) -----
                    // qs offset: sub-block sb starts at qs[sb*32]. K-offset
                    // within sub-block is kt*16.
                    int8x16_t b_vec;
                    if (xb != nullptr) {
                        const int8_t* qs_p = xb->qs + sb * 32 + kt * 16;
                        // qs is naturally 16-byte aligned in the int4 sub-region
                        // because block_q8_1_mmq starts on 8-byte alignment and
                        // ds4[4] = 16 bytes precedes qs.
                        b_vec = *(const int8x16_t*)qs_p;
                    } else {
                        b_vec = (int8x16_t){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
                    }

                    int32x4_t a_i32 = *(const int32x4_t*)&a_vec;
                    int32x4_t b_i32 = *(const int32x4_t*)&b_vec;

                    // signed_a = true is safe because HFQ4 nibble values fit
                    // in [0,15] (MSB always zero, so sign vs unsigned interp
                    // produces the same int8).
                    cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, a_i32, true, b_i32, cacc, true
                    );
                } // for kt

                // Scale FMA: per output element (j in [0,8))
                //   acc[j] += sc_row[j] * d_x * (float)cacc[j]
                //           + zp_row[j] * sum_x_float
                // The sc/zp are per-row (gathered above via shuffle).
                // d_x and sum_x_float are per-slot, stored in this lane's locals.
                // Note: sum_x_float (= dsB.y from Q8_1) is the FLOAT sum of x
                // values in the sub-block, NOT d_x * sum_of_int_x. This matches
                // the residual MMQ kernel convention exactly.
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float dot_f = (float)cacc[j];
                    acc[j] += sc_row[j] * d_x_self * dot_f
                            + zp_row[j] * sum_x_self;
                }
            } // for sb
        } // for kb_in_group
    } // for g

    // RDNA3 wave32 WMMA output store: acc[j] = Y[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (M-dim), B operand = slot lane (m_total-dim).
    // Direct store (=) because Y_grouped is fresh per-call.
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

// Stub for non-gfx1151 archs (the kernel is gfx1151-gated as the initial
// port; broader gfx11 enablement requires per-arch channel testing).
extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_gfx1151(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
