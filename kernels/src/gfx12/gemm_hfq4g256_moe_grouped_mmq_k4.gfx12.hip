#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// i8 WMMA MMQ MoE grouped-GEMM port for gfx12 (RDNA4 — R9700/gfx1201,
// gfx1200) — k4 (deeper K-tile pipeline) variant.
//
// This is the gfx12 sister of `gemm_hfq4g256_moe_grouped_mmq_k4.gfx1151.hip`
// AND the deeper-pipeline sibling of `gemm_hfq4g256_moe_grouped_mmq.gfx12.hip`
// (the k2 kernel).
//
// Structural diagnosis from the prior gfx12 work: the k2 kernel runs at
// 279µs (FP16 ref) → 408µs (i8) = +46% kernel slowdown despite correctness
// PASS (NRMSE ~0.4% on a3b-slice). Root cause hypothesis: per-sub-block
// scale FMA serial dependency chain dominates the 16×16 output tile —
// after each pair of WMMAs (= one Q8_1 sub-block), 8 INT32→F32 conversions
// + 16 FMAs must serialize before the next sub-block's WMMA pair can
// retire its accumulator.
//
// k4 amortizes that chain over more WMMA dispatches per scale-FMA boundary:
// process 4 sub-blocks (sb=0,1,2,3) per inner iteration with 4 independent
// int32 accumulators, batch all 8 WMMAs (2 per sub-block) before resolving
// any scale FMA. This is the same lever that gave +4.6% on gfx1151 and
// +2.8% on gfx1100/gfx11_dgpu. R9700/gfx1201 has ~40-48 CUs (vs 7900 XTX's
// 96 CUs), so each CU eats more of the FMA serial cost — gfx12 should
// proportionally benefit MORE if the diagnosis is correct.
//
// === KEY GFX12 DIFFERENCES vs gfx11/gfx1151 k4 ===
// (Same differences as the k2 sibling — see
// `gemm_hfq4g256_moe_grouped_mmq.gfx12.hip` for the full prose.)
//
// 1. Intrinsic: `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`
//    (vs `_w32` without `_gfx12` on gfx11).
// 2. Per-lane operand width: int32x2 (8 int8 / lane) vs int32x4 on gfx11.
// 3. K-tile split across 2 lane groups (k_grp = tid >> 4):
//    lanes 0-15 carry K[0..7], lanes 16-31 carry K[8..15].
// 4. C-output: `acc[j] = C[8*k_grp + j][m_lane]` (8 contiguous M-rows per
//    lane half). Per-row sc/zp broadcast: `src_lane = 8*k_grp + j`.
// 5. TWO WMMA calls per Q8_1 sub-block (each WMMA covers K=16, sub-block
//    covers K=32).
//
// === K4 PIPELINE STRUCTURE ===
//
// Per Q8_1 block (kb_in_group ∈ {0, 1}):
//   - Pre-load 8 (sb, kt) operand pairs into `pk_arr[8]`, `b_arr[8]`,
//     `d_x_arr[4]`, `sum_x_arr[4]`. This matches the k2's existing
//     "issue all 8 loads upfront" pattern.
//   - Issue 8 WMMAs into 4 independent cacc accumulators
//     (cacc[sb], one per sub-block). Each cacc takes 2 WMMAs (kt=0,1).
//   - Resolve 4 scale FMAs (one per sub-block) after all 8 WMMAs complete.
//
// The scale FMA must remain per-sub-block because each Q8_1 sub-block
// owns its own (d_x, sum_x). We just delay the FMA past more WMMAs to
// hide latency. The output is numerically equivalent to k2 modulo
// floating-point summation order — same `acc[j] += sc*d_x*dot + zp*sum_x`
// pattern, same per-sub-block resolution boundary.
//
// `__launch_bounds__(32, 2)` — same as k2 sibling, proven safe there.
//
// Compile gate: gfx1200 / gfx1201 only.

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

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_k4_gfx12(
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

        // Per-row sc/zp broadcast: gfx12 acc[j] = Y[8*k_grp + j][m_lane], so
        // output row r = row_start + 8*k_grp + j is owned by the lane whose
        // own row is r — i.e. m_lane = 8*k_grp + j. Shuffle from src_lane
        // in [0,16) (within the k_grp lane half) — wave32 __shfl with src
        // < 16 broadcasts the same value to both lane halves because the
        // 16-row A-operand is mirrored across lanes 0-15 and 16-31.
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
            const block_q8_1_mmq* xb = (x_row >= 0)
                ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
                : nullptr;

            // K-tile prefetch: issue ALL 8 (sb=0..3, kt=0..1) A+B operand
            // loads upfront so the compiler can hoist them above the WMMA
            // chain. Same pattern as k2; just keeps all 4 sub-blocks live
            // simultaneously instead of resolving one before loading the next.
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
                    // K-offset within group:
                    //   kb_in_group * 128  (Q8_1 block within group)
                    // + sb * 32             (sub-block)
                    // + kt * 16             (WMMA call within sub-block)
                    // + k_grp * 8           (lane-group's K half).
                    const int k_off = kb_in_group * 128 + sb * 32 + kt * 16 + k_grp * 8;
                    pk_arr[slot] = *(const unsigned int*)(gp + 8 + k_off / 2);
                    if (xb != nullptr) {
                        const int8_t* qs_p = xb->qs + sb * 32 + kt * 16 + k_grp * 8;
                        b_arr[slot] = *(const int8x8_t*)qs_p;
                    } else {
                        b_arr[slot] = (int8x8_t){0, 0, 0, 0, 0, 0, 0, 0};
                    }
                }
            }

            // k4 pipeline: 4 independent int32 accumulators, all 8 WMMAs
            // issued before any scale FMA resolves. This is the lever —
            // amortize the INT→F32 + scale FMA serial chain over a larger
            // batch of WMMA dispatches per chain step.
            int32x8_t cacc0 = {0, 0, 0, 0, 0, 0, 0, 0};
            int32x8_t cacc1 = {0, 0, 0, 0, 0, 0, 0, 0};
            int32x8_t cacc2 = {0, 0, 0, 0, 0, 0, 0, 0};
            int32x8_t cacc3 = {0, 0, 0, 0, 0, 0, 0, 0};

            // Materialize a_i32 per (sb, kt) slot. The nibble unpack is the
            // same as k2; we just hoist all 8 unpacks before the WMMA chain
            // so the compiler can interleave them with the prefetch loads.
            int32x2_t a_i32_arr[8];
            int32x2_t b_i32_arr[8];
            #pragma unroll
            for (int slot = 0; slot < 8; slot++) {
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
                a_i32_arr[slot] = *(const int32x2_t*)&a_vec;
                b_i32_arr[slot] = *(const int32x2_t*)&b_arr[slot];
            }

            // Issue 8 WMMAs into 4 independent accumulators. The pattern
            // intentionally interleaves sub-blocks (cacc0/1/2/3) instead of
            // grouping by sub-block — this maximises the scheduler's ability
            // to overlap because consecutive WMMAs target distinct
            // accumulator registers.
            //
            // signed_a = false: HFQ4 nibbles are 0..15 (unsigned-safe);
            // signed_b = true: Q8_1 is signed -127..127.
            // clamp = true: saturates int32 acc at INT_MAX/INT_MIN; max abs
            // per WMMA = 16 * 15 * 127 ≈ 30k, well within int32 range, so
            // clamp shouldn't fire — but matches k2 convention.
            cacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                false, a_i32_arr[0], true, b_i32_arr[0], cacc0, true);
            cacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                false, a_i32_arr[2], true, b_i32_arr[2], cacc1, true);
            cacc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                false, a_i32_arr[4], true, b_i32_arr[4], cacc2, true);
            cacc3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                false, a_i32_arr[6], true, b_i32_arr[6], cacc3, true);
            cacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                false, a_i32_arr[1], true, b_i32_arr[1], cacc0, true);
            cacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                false, a_i32_arr[3], true, b_i32_arr[3], cacc1, true);
            cacc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                false, a_i32_arr[5], true, b_i32_arr[5], cacc2, true);
            cacc3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                false, a_i32_arr[7], true, b_i32_arr[7], cacc3, true);

            // Resolve 4 scale FMAs in order. Each sub-block applies:
            //   acc[j] += sc_row[j] * d_x_sb * (float)cacc_sb[j]
            //           + zp_row[j] * sum_x_sb
            // sc/zp per-row (gathered above), d_x/sum_x per-slot.
            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                const float d_x_sb   = d_x_arr[sb];
                const float sum_x_sb = sum_x_arr[sb];
                const int32x8_t cacc_sb = (sb == 0) ? cacc0
                                        : (sb == 1) ? cacc1
                                        : (sb == 2) ? cacc2
                                                    : cacc3;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float dot_f = (float)cacc_sb[j];
                    acc[j] += sc_row[j] * d_x_sb * dot_f
                            + zp_row[j] * sum_x_sb;
                }
            }
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
extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_k4_gfx12(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
