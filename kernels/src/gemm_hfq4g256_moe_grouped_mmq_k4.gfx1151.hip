#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// i8 WMMA MMQ MoE grouped-GEMM port for gfx1151 (Strix Halo iGPU) —
// k4 (deeper K-tile pipeline) variant.
//
// This is a 1:1 deeper-pipeline sibling of
// `gemm_hfq4g256_moe_grouped_mmq.gfx1151.hip`. The k2 kernel processes
// 2 K-tiles per inner iteration (one Q8_1 sub-block = 2 × WMMA(K=16)).
// This k4 variant processes 4 K-tiles per inner iteration by pairing
// two adjacent sub-blocks (sb=0,1 then sb=2,3) and keeping two
// independent int32 accumulators in flight.
//
// Rationale: i8 WMMA on gfx11 runs at ~2× FP16 WMMA throughput, so the
// per-tile latency is shorter than FP16 and a deeper pipeline can hide
// more of the dequant + Q8_1 (d, sum) load chain. Each Q8_1 sub-block
// still owns its own (d_x, sum_x) scale, so the FMA boundary stays
// per-32-K — we just batch two sub-blocks of WMMA loads/dispatches
// together before resolving the scale FMA for each.
//
// Per-block contract: identical to the k2 kernel. Same grid
// [ceil(M/16), m_total/16], same block [32], same kernarg layout,
// same scatter / sentinel semantics. Numerically equivalent to k2
// modulo int32 summation-order ambiguity within a sub-block (which is
// non-existent here because the two WMMAs per cacc are commutative in
// integer addition — the only floating-point reduction is the scale
// FMA, and that happens at the same boundaries as k2).
//
// Compile gate: gfx1151 only for the initial port. Mirror the k2
// kernel exactly here so the broader-gfx11 ports can follow the same
// pattern once channel-tested on each target.

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
extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_k4_gfx1151(
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

            // k4 pipeline: process 2 adjacent sub-blocks per inner iteration.
            // Each pair contributes 4 K-tiles (4 WMMAs) into 2 independent
            // int32 accumulators. The scale FMA still applies per sub-block,
            // so cacc0/cacc1 stay separate and resolve at the end of the pair.
            #pragma unroll
            for (int sb_pair = 0; sb_pair < 2; sb_pair++) {
                const int sb0 = 2 * sb_pair;
                const int sb1 = sb0 + 1;

                // (d_x, sum_x) for each sub-block in the pair.
                float d_x0   = 0.0f, sum_x0 = 0.0f;
                float d_x1   = 0.0f, sum_x1 = 0.0f;
                if (xb != nullptr) {
                    half2 dm0 = xb->ds4[sb0];
                    half2 dm1 = xb->ds4[sb1];
                    float2 dm0_f = __half22float2(dm0);
                    float2 dm1_f = __half22float2(dm1);
                    d_x0   = dm0_f.x; sum_x0 = dm0_f.y;
                    d_x1   = dm1_f.x; sum_x1 = dm1_f.y;
                }

                // Pre-load all 4 K-tiles' operands so the compiler can hoist
                // loads above the WMMA chain. Each kt is 16 K-elements;
                // sub-block has 2 kt; pair = 2 sb × 2 kt = 4 K-tiles.
                int32x4_t a_i32[4];
                int32x4_t b_i32[4];

                #pragma unroll
                for (int kt_idx = 0; kt_idx < 4; kt_idx++) {
                    const int sb = (kt_idx < 2) ? sb0 : sb1;
                    const int kt = kt_idx & 1;
                    const int k_off = kb_in_group * 128 + sb * 32 + kt * 16;

                    // ----- A operand (HFQ4 nibbles → int8) -----
                    // Nibbles are packed 2-per-byte starting at gp+8.
                    // 16 nibbles = 8 bytes = 2 packed uint32.
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
                    int8x16_t b_vec;
                    if (xb != nullptr) {
                        const int8_t* qs_p = xb->qs + sb * 32 + kt * 16;
                        // qs is naturally 16-byte aligned in the int4 sub-region.
                        b_vec = *(const int8x16_t*)qs_p;
                    } else {
                        b_vec = (int8x16_t){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
                    }

                    a_i32[kt_idx] = *(const int32x4_t*)&a_vec;
                    b_i32[kt_idx] = *(const int32x4_t*)&b_vec;
                }

                // Issue 4 WMMAs against 2 independent accumulators.
                // signed_a = true is safe because HFQ4 nibble values fit
                // in [0,15] (MSB always zero).
                int32x8_t cacc0 = {0, 0, 0, 0, 0, 0, 0, 0};
                int32x8_t cacc1 = {0, 0, 0, 0, 0, 0, 0, 0};

                cacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, a_i32[0], true, b_i32[0], cacc0, true
                );
                cacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, a_i32[1], true, b_i32[1], cacc0, true
                );
                cacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, a_i32[2], true, b_i32[2], cacc1, true
                );
                cacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, a_i32[3], true, b_i32[3], cacc1, true
                );

                // Scale FMA per sub-block.
                //   acc[j] += sc_row[j] * d_x_sb * (float)cacc_sb[j]
                //           + zp_row[j] * sum_x_sb
                // sc/zp are per-row (broadcast above via __shfl), d_x/sum_x
                // are per-slot for THIS lane (so no shuffle for them).
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float dot0 = (float)cacc0[j];
                    float dot1 = (float)cacc1[j];
                    acc[j] += sc_row[j] * d_x0 * dot0
                            + zp_row[j] * sum_x0;
                    acc[j] += sc_row[j] * d_x1 * dot1
                            + zp_row[j] * sum_x1;
                }
            } // for sb_pair
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

// Stub for non-gfx1151 archs (initial port is gfx1151-gated; broader
// gfx11 enablement requires per-arch channel testing).
extern "C" __global__ void gemm_hfq4g256_moe_grouped_mmq_k4_gfx1151(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
