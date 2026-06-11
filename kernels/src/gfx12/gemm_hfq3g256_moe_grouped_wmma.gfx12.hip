#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) HFQ3/MQ3 sister of `gemm_hfq4g256_moe_grouped_wmma_gfx12`.
// Same MoE Path-2 contract: pad-then-scan guarantees that all 16 slots
// in a (blockIdx.x, blockIdx.y) tile share the same expert, picked via
// `expert_tile_ids[tile_y]` (-1 sentinel → early return).
//
// HFQ3-G256 storage layout (per 256-element group, 104 B total):
//   [0..4]   fp32 scale
//   [4..8]   fp32 zero
//   [8..104] 96 B of packed 3-bit weights — 32 chunks × 3 B, each chunk
//            encodes 8 weights across 24 bits (`(pk >> (3*i)) & 7u`).
//
// gfx12 wave32 WMMA mapping (mirrors qkvza/qkv/gate_up/residual gfx12
// HFQ3 ports + the HFQ4 grouped sister):
//   - half8_t operands (8 fp16/lane), K split across 2 lane-groups
//     (k_grp = tid >> 4). Each lane handles 1 chunk per K-tile (8
//     weights = 3 bytes) at byte offset 8 + kt*6 + k_grp*3.
//   - C-output mapping: acc[j] = C[8 * k_grp + j][m_lane].
//   - K4 unroll: 4 K-tiles per inner body, all weight+X loads up-front.
//
// X gather is identical to the HFQ4 grouped sister:
//   gate_up: X_src = x_rot_batch [N × K], x_row_div = K_TOP
//   down:    X_src = rot_batch   [n_rows × K], x_row_div = 1
// -1 padding lanes contribute a zero B row.
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq3g256_moe_grouped_wmma_gfx12(
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

    // Sentinel early-return: dispatcher may launch up to m_total_max/16
    // tiles to skip the m_total dtoh sync; scatter pre-fills the upper-
    // bound range of expert_tile_ids with -1.
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X-row gather (per m_lane; both k_grp lanes for the same m_lane read
    // the same source row). -1 padding lanes contribute a zero B row.
    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    // A-row selection: pad OOB lanes to the last valid M-row (the
    // masked output write below discards the result anyway).
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 104;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    // Unpack 8 × 3-bit weights from a 3-byte chunk (a0, a1, a2) into
    // a_reg. Cross-byte pattern matches `gemm_qkvza_hfq3g256_wmma.gfx12`
    // line 78-89.
    #define UNPACK_CHUNK_DQ(b0, b1, b2)                          \
        do {                                                      \
            unsigned int _a0 = (b0), _a1 = (b1), _a2 = (b2);     \
            a_reg[0] = sc_h * (_Float16)(float)(_a0 & 7u) + zp_h; \
            a_reg[1] = sc_h * (_Float16)(float)((_a0 >> 3) & 7u) + zp_h; \
            a_reg[2] = sc_h * (_Float16)(float)(((_a0 >> 6) | (_a1 << 2)) & 7u) + zp_h; \
            a_reg[3] = sc_h * (_Float16)(float)((_a1 >> 1) & 7u) + zp_h; \
            a_reg[4] = sc_h * (_Float16)(float)((_a1 >> 4) & 7u) + zp_h; \
            a_reg[5] = sc_h * (_Float16)(float)(((_a1 >> 7) | (_a2 << 1)) & 7u) + zp_h; \
            a_reg[6] = sc_h * (_Float16)(float)((_a2 >> 2) & 7u) + zp_h; \
            a_reg[7] = sc_h * (_Float16)(float)((_a2 >> 5) & 7u) + zp_h; \
        } while (0)

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 104;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        // K4 unroll: 4 K-tiles per inner body. Each lane reads 1 chunk
        // (3 bytes) per K-tile at offset kt*6 + k_grp*3 within the
        // packed-weight section (gp + 8 ... gp + 104).
        const unsigned char* dp = (const unsigned char*)(gp + 8);

        for (int kt = 0; kt < 16; kt += 4) {
            const unsigned char* dpa = dp + (kt + 0) * 6 + k_grp * 3;
            const unsigned char* dpb = dp + (kt + 1) * 6 + k_grp * 3;
            const unsigned char* dpc = dp + (kt + 2) * 6 + k_grp * 3;
            const unsigned char* dpd = dp + (kt + 3) * 6 + k_grp * 3;

            const int k_off_a = (kt + 0) * 16;
            const int k_off_b = (kt + 1) * 16;
            const int k_off_c = (kt + 2) * 16;
            const int k_off_d = (kt + 3) * 16;

            half8_t b_a, b_b, b_c, b_d;
            if (xg != nullptr) {
                b_a = *(const half8_t*)(xg + k_off_a + k_grp * 8);
                b_b = *(const half8_t*)(xg + k_off_b + k_grp * 8);
                b_c = *(const half8_t*)(xg + k_off_c + k_grp * 8);
                b_d = *(const half8_t*)(xg + k_off_d + k_grp * 8);
            } else {
                b_a = zero_b; b_b = zero_b; b_c = zero_b; b_d = zero_b;
            }

            unsigned int a0 = dpa[0], a1 = dpa[1], a2 = dpa[2];
            unsigned int b0 = dpb[0], b1 = dpb[1], b2 = dpb[2];
            unsigned int c0 = dpc[0], c1 = dpc[1], c2 = dpc[2];
            unsigned int d0 = dpd[0], d1 = dpd[1], d2 = dpd[2];

            half8_t a_reg;
            UNPACK_CHUNK_DQ(a0, a1, a2);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            UNPACK_CHUNK_DQ(b0, b1, b2);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            UNPACK_CHUNK_DQ(c0, c1, c2);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            UNPACK_CHUNK_DQ(d0, d1, d2);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
    }
    #undef UNPACK_CHUNK_DQ

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    // Direct write (=) — Y_grouped is fresh per call.
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
