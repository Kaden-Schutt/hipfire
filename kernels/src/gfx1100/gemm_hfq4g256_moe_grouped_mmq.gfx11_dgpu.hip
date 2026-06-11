#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// i8 WMMA MMQ MoE grouped-GEMM port for gfx11 dGPUs
// (gfx1100/1101/1102/1103 — 7900 XTX, 7800/7700, 7600, Phoenix mobile).
//
// Combines two existing patterns:
//   - `gemm_hfq4g256_residual_mmq.hip`     : i8 WMMA (`wmma_i32_16x16x16_iu8_w32`)
//                                            on Q8_1-quantized activations, with
//                                            HFQ4 scale/zero applied via Q8_1
//                                            scale/sum correction.
//   - `gemm_hfq4g256_moe_grouped_wmma_k2.hip`: SGLang-style grouped MoE dispatch
//                                              with expert_tile_ids + sorted_slot_index
//                                              scatter and a 16×16 output tile per WG.
//
// Rationale: gfx11 dGPUs are compute-bound on the grouped MoE GEMM path.
// The i8 WMMA path runs at ~2× the FLOP rate of FP16 WMMA via the
// `i32_16x16x16_iu8` intrinsic (RDNA3 native).
//
// X must be pre-quantized to Q8_1 (block_q8_1_mmq) via `quantize_q8_1_mmq_ds4`
// before dispatch. The kernarg layout matches the gfx1151 i8 sister — the
// FP16 grouped's 9 kernargs + 1 extra `x_src_rows` so the kernel can
// compute the K-block stride into the Q8_1 buffer.
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
// Per-block contract: each block owns a 16-row × 16-slot output tile. All
// 16 slots share the same expert (Phase 2 scatter pad-then-scan guarantees
// expert_offsets aligned to BLOCK_M=16). Per K=16 WMMA call (RDNA3 wave32),
// lane m_lane = tid & 15 holds the K=16 strip for A-row m_lane (=
// row_start + m_lane) AND for B-col m_lane (= this lane's slot). Lanes 0..15
// and 16..31 receive replicated operands; the WMMA acc[j] = C[2*j +
// (tid>>4)][tid & 15] for j=0..7 covers 16 output M-rows × 16 output slots.
//
// HFQ4 sc/zp correction is applied AFTER the WMMA in float at the 32-K Q8_1
// sub-block boundary:
//   Y[m, slot] += sc[m] × d_sub × dot_int  +  zp[m] × sum_floats_sub
// where `sum_floats_sub = ds4[ksub].y` (precomputed sum of ORIGINAL floats,
// NOT sum of quants — see `quantize_q8_1_mmq_ds4`). The sc/zp lookup uses a
// per-row __shfl: each lane reads its own row's (sc, zp) from global memory
// once per HFQ4 group, then broadcasts to the 8 output rows it owns via
// warp shuffle (8 single-source shuffles per group, no LDS).

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
#define HIPFIRE_GFX11_DGPU 1
#else
#define HIPFIRE_GFX11_DGPU 0
#endif

#define QK8_1 32
#define WARP_SIZE 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int8_t  __attribute__((ext_vector_type(16))) int8x16_t;
typedef int32_t __attribute__((ext_vector_type(4)))  int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;

#if !HIPFIRE_GFX11_DGPU

// Compile-only stub on non-target archs. The dispatcher gates on arch
// (gfx1100/1101/1102/1103) and never launches on anything else.
extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu(
    const unsigned long long*,
    const int*,
    const int*,
    const block_q8_1_mmq*,
    float*,
    int, int, int, int, int
) {}

#else

// `x_row_div` controls how sorted_slot_index maps to X_q8_src rows:
//   gate_up: X_src_f32 was [N × K],       x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src_f32 was [N*K_TOP × K], x_row_div = 1     → x_row = flat
//
// X_q8_src layout from `quantize_q8_1_mmq_ds4`: [K/128, x_src_rows] of
// block_q8_1_mmq. Block address: X_q8_src + kb_q8 * x_src_rows + x_row.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const block_q8_1_mmq*     __restrict__ X_q8_src,           // [K/128 × x_src_rows]
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total, int x_src_rows
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    // Sentinel early-return: expert_tile_ids[tile_y] == -1 marks tiles
    // beyond the live m_total (mirrors the FP16 grouped kernel).
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int lane = tid & 15;
    const int k_grp = tid >> 4;

    // X gather: each lane (tid & 15) gets its own routed slot. Padding lanes
    // (sorted_slot_index == -1) contribute zero Q8_1 blocks for B.
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const bool has_x = (x_row >= 0);

    // A-row selection for the WMMA A operand contribution. Each lane carries
    // ONE A-row's K=16 strip; lanes 16-31 mirror lanes 0-15. Pad OOB to last
    // valid row (the masked output write below discards padded results).
    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    // Final float accumulator. RDNA3 wave32 WMMA C-layout:
    //   acc[j] = Y[row_start + 2*j + k_grp][slot_start + lane]
    // for j ∈ [0, 8) and k_grp ∈ {0, 1}. Each lane writes 8 M-rows × 1 slot.
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; ++g) {
        const char* gp = row_base + g * 136;

        // Lane reads its OWN row's (sc, zp). Each lane in [0, 16) has the
        // header for row (row_start + lane). The 8 output M-rows owned by
        // this lane are (row_start + 2*j + k_grp); the matching source lane
        // is (2*j + k_grp). Lanes 16..31 mirror lanes 0..15, so __shfl
        // grabs the right value regardless of k_grp.
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 0));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int src_lane = 2 * j + k_grp;
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        // Two 128-K Q8_1 blocks per HFQ4 group (256 / 128 = 2).
        #pragma unroll 1
        for (int kb_in_group = 0; kb_in_group < 2; ++kb_in_group) {
            const int kb_q8 = 2 * g + kb_in_group;
            if (kb_q8 >= blocks128) break;

            const block_q8_1_mmq* xb = has_x
                ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
                : (const block_q8_1_mmq*)nullptr;

            // 4 × 32-K Q8_1 sub-blocks per 128-K block. Each sub-block has
            // its own (d, sum) and is covered by 2 WMMA K=16 calls.
            #pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                // (d, sum_floats) for this lane's slot's sub-block. No
                // shuffle needed — d/sum is per-slot, and slot = this lane.
                float d_x = 0.0f;
                float sum_x = 0.0f;
                if (xb != nullptr) {
                    half2 dm = xb->ds4[sb];
                    float2 dm_f = __half22float2(dm);
                    d_x   = dm_f.x;
                    sum_x = dm_f.y; // sum of ORIGINAL FLOATS over 32 K-elements
                }

                // int32 accumulator for the 32-K sub-block dot product.
                int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};

                #pragma unroll
                for (int kt = 0; kt < 2; ++kt) {
                    // ----- A operand: 16 raw HFQ4 nibbles widened to i8 -----
                    // K offset within the 256-element HFQ4 group:
                    //   kb_in_group * 128  (Q8_1 block within group)
                    // + sb * 32             (sub-block within Q8_1 block)
                    // + kt * 16             (WMMA call within sub-block).
                    // Nibbles are packed 2-per-byte starting at gp+8.
                    const int k_off = kb_in_group * 128 + sb * 32 + kt * 16;
                    const unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2 + 0);
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

                    // ----- B operand: 16 Q8_1 int8 quantized values -----
                    int8x16_t b_vec;
                    if (xb != nullptr) {
                        const int8_t* qs_p = xb->qs + sb * 32 + kt * 16;
                        // 16-byte aligned: block_q8_1_mmq starts on >=8B
                        // align and ds4[4]=16B precedes qs; sub-block offset
                        // sb*32 is 16-aligned; kt*16 is 16-aligned.
                        b_vec = *(const int8x16_t*)qs_p;
                    } else {
                        b_vec = (int8x16_t){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
                    }

                    int32x4_t a_i32 = *(const int32x4_t*)&a_vec;
                    int32x4_t b_i32 = *(const int32x4_t*)&b_vec;

                    // modA=true (signed): HFQ4 nibbles ∈ [0,15] — MSB always
                    // zero — so signed and unsigned i8 interpretations match.
                    // modB=true (signed): Q8_1 i8 ∈ [-127, 127].
                    cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, a_i32, true, b_i32, cacc, true
                    );
                } // kt

                // Apply HFQ4 sc/zp + Q8_1 d/sum correction in float at the
                // 32-K sub-block boundary. Math: per-output-element
                //   Y[m, slot] += sc[m] × d_x × dot_int + zp[m] × sum_floats
                // where sum_floats = Σ X over the 32-K sub-block (precomputed
                // in Q8_1 ds4.y as sum-of-original-floats).
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float dot_f = (float)cacc[j];
                    acc[j] += sc_row[j] * d_x * dot_f
                            + zp_row[j] * sum_x;
                }
            } // sb
        } // kb_in_group
    } // g

    // Output write: WMMA wave32 layout — acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (M-dim), B operand = slot lane (m_total-dim).
    // Direct store (=) because Y_grouped is freshly allocated per call.
    const int out_col = slot_start + lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int out_row = row_start + 2 * j + k_grp;
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#endif // HIPFIRE_GFX11_DGPU
