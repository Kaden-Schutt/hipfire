#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) HFP4G32 sister of `gemm_hfq4g256_moe_grouped_wmma_gfx12`.
// Same WMMA tile geometry + same expert_tile_ids/-1 sentinel pattern; the
// only differences are the dequant (HFP4G32 18 B/group: 1 B UE8M0 + 16 B
// packed FP4 via E2M1 LUT, with a per-row FP16 row_scale prefix) and the
// G32 inner loop (K/32 blocks; the loop is unrolled as K/256 "groups" of
// 8 blocks each — same shape the residual+qkvza HFP4G32 kernels use).
//
// Levers (mirroring the validated gfx12 ports):
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
//   2. A/B operands: half8_t (8 fp16/lane) instead of half16_t.
//   3. K dim split across 2 lane-groups (k_grp = tid >> 4); each carries
//      a half of the 16-K tile.
//   4. C-output mapping: acc[j] = C[8*k_grp + j][m_lane].
//
// Per-block contract (mirrors the HFQ4 grouped variant):
//   - Tile (blockIdx.x, blockIdx.y) owns a 16-row × 16-slot output tile.
//   - All 16 slots share the same expert (Phase 2's pad-then-scan
//     guarantees expert_offsets aligned to BLOCK_M = 16).
//   - Each lane m_lane = tid & 15 corresponds to one slot AND one M-row.
//   - X is gathered via sorted_slot_index[slot_start + m_lane]; -1
//     padding lanes substitute zero rows for the B operand.
//   - expert_tile_ids[tile_y] < 0 → tile beyond live m_total, early-return.
//
// MFP4G32 shares HFP4G32's storage layout — rotation applied by the caller
// (in x_src). No kernel-side rotate distinction needed.
//
// HFP4G32 row layout (bytes per row):
//   [0..2]     fp16 row_scale_h
//   [2..16]    padding/unused (consumes header alignment)
//   [16..16+blocks*17]  blocks_per_row × 17-byte blocks:
//       [0]    UE8M0 exponent (1 byte)
//       [1..17] 16 bytes = 32 packed FP4 nibbles
//   row_stride = 16 + (K/32) * 17 bytes
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 32 B (16 × fp16 LUT).

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfp4g32_moe_grouped_wmma_gfx12(
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

    // Sentinel early-return: expert_tile_ids[tile_y] == -1 marks tiles
    // beyond the live m_total (the dispatcher may launch up to
    // m_total_max/16 tiles to skip the m_total dtoh sync, since the
    // scatter pre-fills the upper-bound range with -1).
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    // E2M1 dequant LUT (16 fp16 values). Shared across the CTA.
    __shared__ _Float16 lut[16];
    if (tid < 16) {
        const float E2M1[16] = {
            +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        lut[tid] = (_Float16)E2M1[tid];
    }
    __syncthreads();

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X gather (per m_lane; both k_grp lanes for the same m_lane read the
    // same source row). -1 padding lanes contribute a zero B row.
    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    // A-row selection: pad OOB lanes to the last valid M-row (gfx11 trick
    // — the masked output write below discards the result anyway).
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);

    const int blocks_per_row = K / 32;
    const long long row_stride = 16 + (long long)blocks_per_row * 17;
    const char* hdr = A + (long long)safe_row * row_stride;
    const char* row_ptr = hdr + 16;

    const unsigned short rs_bits = *(const unsigned short*)hdr;
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, rs_bits);

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    // Outer K loop: K/256 "groups", each 8 blocks of 17 B (136 B total).
    // Inner unroll mirrors gemm_qkvza_hfp4g32_wmma.gfx12 / residual_wmma.gfx12.
    const int groups_per_row = K / 256;
    for (int g = 0; g < groups_per_row; g++) {
        const char* group_base = row_ptr + (long long)g * 136;
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        for (int kt = 0; kt < 16; kt += 4) {
            const int blk_a = kt >> 1;
            const int blk_c = blk_a + 1;
            const char* bpa = group_base + blk_a * 17;
            const char* bpc = group_base + blk_c * 17;

            const unsigned int e_a = *(const unsigned char*)bpa;
            const unsigned int e_c = *(const unsigned char*)bpc;
            const float sc_block_a_f = __builtin_bit_cast(float, e_a << 23);
            const float sc_block_c_f = __builtin_bit_cast(float, e_c << 23);
            const _Float16 sc_h_ab = row_scale_h * (_Float16)sc_block_a_f;
            const _Float16 sc_h_cd = row_scale_h * (_Float16)sc_block_c_f;

            unsigned int pka = *(const unsigned int*)(bpa + 1 + k_grp * 4);
            unsigned int pkb = *(const unsigned int*)(bpa + 9 + k_grp * 4);
            unsigned int pkc = *(const unsigned int*)(bpc + 1 + k_grp * 4);
            unsigned int pkd = *(const unsigned int*)(bpc + 9 + k_grp * 4);

            const int k_off_a = (kt + 0) * 16 + k_grp * 8;
            const int k_off_b = (kt + 1) * 16 + k_grp * 8;
            const int k_off_c = (kt + 2) * 16 + k_grp * 8;
            const int k_off_d = (kt + 3) * 16 + k_grp * 8;

            half8_t b_a, b_b, b_c, b_d;
            if (xg != nullptr) {
                b_a = *(const half8_t*)(xg + k_off_a);
                b_b = *(const half8_t*)(xg + k_off_b);
                b_c = *(const half8_t*)(xg + k_off_c);
                b_d = *(const half8_t*)(xg + k_off_d);
            } else {
                b_a = zero_b; b_b = zero_b; b_c = zero_b; b_d = zero_b;
            }

            half8_t a_reg;
            #define DQ8(pk, sc) \
                a_reg[0] = (sc) * lut[((pk) >>  0) & 0xFu]; \
                a_reg[1] = (sc) * lut[((pk) >>  4) & 0xFu]; \
                a_reg[2] = (sc) * lut[((pk) >>  8) & 0xFu]; \
                a_reg[3] = (sc) * lut[((pk) >> 12) & 0xFu]; \
                a_reg[4] = (sc) * lut[((pk) >> 16) & 0xFu]; \
                a_reg[5] = (sc) * lut[((pk) >> 20) & 0xFu]; \
                a_reg[6] = (sc) * lut[((pk) >> 24) & 0xFu]; \
                a_reg[7] = (sc) * lut[((pk) >> 28) & 0xFu]

            DQ8(pka, sc_h_ab);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ8(pkb, sc_h_ab);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ8(pkc, sc_h_cd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ8(pkd, sc_h_cd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);

            #undef DQ8
        }
    }

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    // Use direct store (=) since Y_grouped is fresh per call.
    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
