#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// i8 WMMA MMQ MoE grouped-GEMM port for gfx12 (RDNA4 — R9700/gfx1201, gfx1200).
//
// This is the gfx12 sister of `gemm_hfq4g256_moe_grouped_mmq.gfx1151.hip`
// and shares its dispatch contract:
//   - X pre-quantized to Q8_1 (block_q8_1_mmq) via `quantize_q8_1_mmq_ds4`
//     before launch (X_q8_src layout = [K/128 × x_src_rows]).
//   - Per-tile (blockIdx.x, blockIdx.y) owns a 16-row × 16-slot output tile.
//   - All 16 slots in this tile share the same expert (scatter pad guarantees
//     expert_offsets aligned to BLOCK_M = 16).
//   - Per lane (tid & 15) routes to ONE slot via sorted_slot_index; -1 padding
//     lanes contribute zero Q8_1 to the B operand.
//   - HFQ4 scale/zero applied per-row via Q8_1 (d, sum) correction at the FMA
//     step (`sum_f32 += sc * d_x * C_frag + zp * d_x * sum_x`).
//
// === KEY GFX12 DIFFERENCES vs gfx11/gfx1151 i8 MMQ ===
//
// 1. **Intrinsic name**: `_gfx12` suffix.
//    gfx11:  __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32
//    gfx12:  __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12
//
// 2. **Operand width per lane**: gfx12 wave32 packs HALF as many K elements
//    per lane (8 int8 / lane) compared to gfx11 (16 int8 / lane). Each lane
//    holds 2 int32 (= 8 int8) instead of 4 int32 (= 16 int8).
//      gfx11 A/B per lane: int32x4 (16 int8)
//      gfx12 A/B per lane: int32x2 ( 8 int8)
//
// 3. **K-tile lane split**: Each WMMA call still covers K=16 of the K-dim,
//    but on gfx12 wave32 the 16 K-elements are split across 2 lane groups
//    (k_grp = tid >> 4). Lanes 0-15 carry K[0..7], lanes 16-31 carry K[8..15].
//    On gfx11 wave32 BOTH lane halves carry the same K=16 in each lane.
//
// 4. **C-output mapping**: gfx12 wave32 stores `acc[j] = C[8*k_grp + j][m_lane]`
//    (k_grp = tid >> 4, m_lane = tid & 15). gfx11 uses `acc[j] = C[2*j + k_grp][m_lane]`.
//    The 8 acc[j] slots are *contiguous M-rows* on gfx12 (j=0..7 → rows 0..7
//    for k_grp=0, rows 8..15 for k_grp=1).
//
// 5. **Single WMMA per Q8_1 sub-block**: A Q8_1 sub-block covers K=32; on
//    gfx12 we need TWO WMMA calls per sub-block (each covers K=16). The B
//    operand for each call is half of the sub-block's qs[].
//
// 6. **HFQ4 scale/zero broadcast**: The per-row sc/zp shuffle mapping differs
//    because the C-output layout differs. Output row r = row_start + 8*k_grp + j
//    is owned by lane (tid) where tid&15 controls m_lane (slot) and tid>>16
//    controls k_grp. So src_lane for row r's sc/zp = (r - row_start) - 8*k_grp = j
//    when k_grp = (r - row_start) >> 3. Each lane needs sc_row[j] = sc broadcast
//    from src_lane = 8*k_grp + j (the lane that holds row r as its "own row").
//
// Compile gate: gfx1200 / gfx1201 only.
//
// Per-block contract (mirrors gfx1151):
//   1. Look up expert id from expert_tile_ids[blockIdx.y]; -1 ⇒ skip.
//   2. Load A operand base = expert_weight_ptrs[expert_id].
//   3. Per lane: routed slot via sorted_slot_index[tile_y*16 + (tid&15)].
//   4. Iterate over groups (256 K-elements each = 2 Q8_1 blocks). Per Q8_1
//      sub-block of K=32:
//        - 2 i8 WMMA calls (K=16 each) accumulate int32 → cacc
//        - Apply HFQ4 + Q8_1 scale FMA: acc[j] += sc_row[j] * d_x * cacc[j]
//                                              + zp_row[j] * d_x * sum_x
//   5. Direct-store output (= not +=); MoE combine fans Y_grouped per token.

#if defined(__gfx1200__) || defined(__gfx1201__)

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int8_t  __attribute__((ext_vector_type(8)))  int8x8_t;
typedef int32_t __attribute__((ext_vector_type(2)))  int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_q8_src rows:
//   gate_up: X_src_f32 had [N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src_f32 had [N*K_TOP × K], x_row_div = 1 → x_row = flat
//
// X_q8_src is the result of running `quantize_q8_1_mmq_ds4` on a buffer of
// `x_src_rows × K` floats. Layout: X_q8_src[kb_q8 * x_src_rows + x_row].
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_gfx12(
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

    const int m_lane = tid & 15;   // 0..15: slot index within tile
    const int k_grp  = tid >> 4;   // 0/1: which K-half this lane carries

    // X gather: each lane (m_lane) gets its own routed slot. Padding lanes
    // (sorted_slot_index == -1) substitute zero Q8_1 contributions.
    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }

    // A-row selection: pad OOB to last valid row (same trick as FP16 sister).
    // Each lane (m_lane) owns ONE row of the WMMA A operand. Both lane halves
    // (k_grp 0 and 1) read from the SAME row (different K-halves).
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    // Output accumulator (gfx12 wave32 layout): acc[j] = Y[8*k_grp + j][m_lane].
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const int blocks128 = K / 128;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        // HFQ4 per-group scale (sc) and zero (zp) for THIS lane's row.
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        // gfx12 acc[j] = Y[8*k_grp + j][m_lane]. Output row r = row_start + 8*k_grp + j
        // is owned by THIS lane (m_lane is the slot/N-dim). The sc/zp for row r
        // belongs to the lane whose own row is r — i.e. lane with m_lane = 8*k_grp + j.
        // Since sc_self/zp_self at lane L holds row L's value, we shuffle from
        // src_lane = 8*k_grp + j (within the same lane group of 16).
        //
        // Lanes 16-31 mirror lanes 0-15 in the wave32 WMMA register layout for the
        // 16x16 A-operand; we use __shfl over the full wave32 with src_lane in
        // [0, 16), so the broadcast happens within each k_grp lane half. With wave32
        // __shfl, src_lane < 32 — using src_lane in [0,16) works the same on both
        // halves because each k_grp half is a "mirror" of the 16-row A-operand.
        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int src_lane = 8 * k_grp + j;
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        // 2 Q8_1 blocks per group (256 K-elements / 128 = 2). Each block holds
        // 4 sub-blocks of 32 K = 4 (d_x, sum_x) pairs.
        for (int kb_in_group = 0; kb_in_group < 2; kb_in_group++) {
            const int kb_q8 = 2 * g + kb_in_group;
            if (kb_q8 >= blocks128) break;

            // Pull the Q8_1 block for THIS lane's slot (or null for padding).
            // Per-slot lookup: only m_lane controls the slot; both k_grp lanes
            // for the same m_lane access the same Q8_1 block (different K halves
            // of qs[]).
            const block_q8_1_mmq* xb = (x_row >= 0)
                ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
                : nullptr;

            // K-iteration optimization: issue ALL 8 (sb=0..4, kt=0..2) A+B
            // loads upfront to overlap memory latency with WMMA execution.
            // Then sub-block FMA chain is fed by already-resident registers.
            // This mirrors the FP16 sister's 4-WMMA-bundle pattern.
            unsigned int pk_arr[8];
            int8x8_t b_arr[8];
            float d_x_arr[4];
            float sum_x_arr[4];

            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                if (xb != nullptr) {
                    half2 dm = xb->ds4[sb];
                    float2 dm_f = __half22float2(dm);
                    d_x_arr[sb]   = dm_f.x;
                    sum_x_arr[sb] = dm_f.y;
                } else {
                    d_x_arr[sb]   = 0.0f;
                    sum_x_arr[sb] = 0.0f;
                }
                #pragma unroll
                for (int kt = 0; kt < 2; kt++) {
                    const int slot = sb * 2 + kt;
                    // HFQ4 packed nibbles, 1 uint32 = 8 nibbles per lane.
                    // K-offset within group:
                    //   kb_in_group * 128  (Q8_1 block within group)
                    // + sb * 32             (sub-block)
                    // + kt * 16             (WMMA call within sub-block)
                    // + k_grp * 8           (lane-group's K half).
                    const int k_off = kb_in_group * 128 + sb * 32 + kt * 16 + k_grp * 8;
                    pk_arr[slot] = *(const unsigned int*)(gp + 8 + k_off / 2);
                    // Q8_1 quants: 8 int8 per lane.
                    if (xb != nullptr) {
                        const int8_t* qs_p = xb->qs + sb * 32 + kt * 16 + k_grp * 8;
                        b_arr[slot] = *(const int8x8_t*)qs_p;
                    } else {
                        b_arr[slot] = (int8x8_t){0, 0, 0, 0, 0, 0, 0, 0};
                    }
                }
            }

            // 4 sub-blocks of 32 K each. Per sub-block we do TWO i8 WMMA calls
            // (K=16 each) on an int32 accumulator that resets at the start of the
            // sub-block, then apply the scale FMA across (sc_row, zp_row, d_x, sum_x).
            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};

                #pragma unroll
                for (int kt = 0; kt < 2; kt++) {
                    const int slot = sb * 2 + kt;
                    const unsigned int pk = pk_arr[slot];

                    int8x8_t a_vec;
                    a_vec[0] = (int8_t)((pk >>  0) & 0xFu);
                    a_vec[1] = (int8_t)((pk >>  4) & 0xFu);
                    a_vec[2] = (int8_t)((pk >>  8) & 0xFu);
                    a_vec[3] = (int8_t)((pk >> 12) & 0xFu);
                    a_vec[4] = (int8_t)((pk >> 16) & 0xFu);
                    a_vec[5] = (int8_t)((pk >> 20) & 0xFu);
                    a_vec[6] = (int8_t)((pk >> 24) & 0xFu);
                    a_vec[7] = (int8_t)((pk >> 28) & 0xFu);

                    int32x2_t a_i32 = *(const int32x2_t*)&a_vec;
                    int32x2_t b_i32 = *(const int32x2_t*)&b_arr[slot];

                    // signed_a = false (HFQ4 nibbles are 0..15, unsigned-safe);
                    // signed_b = true (Q8_1 is signed -127..127). clamp=true
                    // saturates the int32 accumulator at INT_MAX/INT_MIN (safe
                    // for K=16 chunks: max abs dot = 16 * 15 * 127 ≈ 30k).
                    cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        false, a_i32, true, b_i32, cacc, true
                    );
                } // for kt

                // Scale FMA: per output element (j in [0,8)).
                //   dot_real = sum_k(a_real[k] * b_real[k])
                //            = sum_k((sc*q4[k]+zp) * d_x*q8[k])
                //            = sc * d_x * sum_k(q4*q8) + zp * d_x * sum_k(q8)
                //            = sc * d_x * cacc[j] + zp * (d_x*sum_k(q8))
                //            = sc * d_x * cacc[j] + zp * sum_real_x
                // Note: d_x * sum_q8_int ≈ sum_real_x (Q8_1 stores sum of REAL
                // x values directly in `ds4[].y`, not sum of int quants), so we
                // multiply by `sum_x_self` only, NOT by `d_x_self * sum_x_self`.
                // Matches `gemm_hfq4g256_residual_mmq` line 330-331 convention.
                // sc/zp per-row (gathered above via shuffle).
                // d_x, sum_x per-slot (this lane's own).
                const float d_x_self   = d_x_arr[sb];
                const float sum_x_self = sum_x_arr[sb];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float dot_f = (float)cacc[j];
                    acc[j] += sc_row[j] * d_x_self * dot_f
                            + zp_row[j] * sum_x_self;
                }
            } // for sb
        } // for kb_in_group
    } // for g

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    // Direct store (=) because Y_grouped is fresh per call.
    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#else

// Stub for non-gfx12 archs (the kernel is gfx12-gated; gfx11/gfx1151 have
// their own MMQ ports with different WMMA intrinsics + operand widths).
extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_gfx12(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
