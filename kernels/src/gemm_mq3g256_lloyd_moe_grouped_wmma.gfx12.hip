#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) MQ3-Lloyd grouped-WMMA GEMM for MoE prefill.
// T3-3L cold-tier building block: gfx12 sister of
// `gemm_mq3g256_lloyd_moe_grouped_wmma_k2`.
//
// DISTINCT from `gemm_hfq3g256_moe_grouped_wmma_gfx12` (that kernel is
// HFQ3-G256: affine `sc*q+zp`, 104 B/group). This kernel is MQ3-Lloyd:
// 8-entry fp16 codebook + 3-bit direct-lookup, 112 B/group.
//
// MQ3-Lloyd-G256 storage layout (112 B/group):
//   bytes [0..16)  : 8 × fp16 codebook entries (sorted ascending at quant-time)
//   bytes [16..112): 96 B packed 3-bit indices — 256 values × 3 bits, LSB-first.
//                    Each K-tile (16 weights) uses 6 B (48 bits = 16 × 3-bit).
//                    Each k_grp half owns 8 values = 3 bytes at kt*6 + k_grp*3.
//                    idx_i = (pk >> (3*i)) & 7u for i in 0..7.
//
// gfx12 wave32 WMMA mapping (mirrors gemm_hfq6g256_moe_grouped_wmma.gfx12):
//   - half8_t operands (8 fp16/lane), K split across 2 lane-groups
//     (k_grp = tid >> 4). Each lane handles 1 chunk per K-tile (8 weights =
//     3 bytes) at byte offset 16 + kt*6 + k_grp*3.
//   - C-output mapping: acc[j] = C[8*k_grp + j][m_lane].
//   - K4 unroll: 4 K-tiles per inner body.
//
// Dequant (3D bilinear decomposition, see gfx11 sister for derivation):
//   Per-lane codebook: each lane uses its OWN row's codebook (safe_row =
//   row_start + m_lane). The __shared__ LDS approach is WRONG for the grouped
//   MoE contract because each m_lane lane uses a different A-row and thus a
//   different codebook. The register-based 3D bilinear form is correct and
//   vectorizes into WMMA (unlike the ternary-chain / LDS form which
//   suppresses WMMA emission — verified 2026-05-22 for MQ2).
//
// 16 K-tiles × 6 B/tile = 96 B ≡ bytes [16..112). ✓
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq3g256_lloyd_moe_grouped_wmma_gfx12(
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
    const char* row_base = A + (long long)safe_row * groups_per_row * 112;   // 112 B/group

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 112;
        // Per-lane codebook: 8 fp16 entries from this lane's weight row.
        // Each m_lane has a DIFFERENT safe_row → different codebook.
        // Register-based 3D bilinear decomposition (no LDS needed).
        const __half* cb_h = reinterpret_cast<const __half*>(gp);
        _Float16 cb0 = (_Float16)__half2float(cb_h[0]);
        _Float16 cb1 = (_Float16)__half2float(cb_h[1]);
        _Float16 cb2 = (_Float16)__half2float(cb_h[2]);
        _Float16 cb3 = (_Float16)__half2float(cb_h[3]);
        _Float16 cb4 = (_Float16)__half2float(cb_h[4]);
        _Float16 cb5 = (_Float16)__half2float(cb_h[5]);
        _Float16 cb6 = (_Float16)__half2float(cb_h[6]);
        _Float16 cb7 = (_Float16)__half2float(cb_h[7]);

        // 3D bilinear coefficients (exact for any 8-entry table).
        const _Float16 coA = cb0;
        const _Float16 coB = cb1 - cb0;
        const _Float16 coC = cb2 - cb0;
        const _Float16 coD = cb4 - cb0;
        const _Float16 coE = cb3 - cb2 - cb1 + cb0;
        const _Float16 coF = cb5 - cb4 - cb1 + cb0;
        const _Float16 coG = cb6 - cb4 - cb2 + cb0;
        const _Float16 coH = cb7 - cb6 - cb5 + cb4 - cb3 + cb2 + cb1 - cb0;

        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        // K4 unroll: 4 K-tiles per inner body.
        // Each lane reads 1 chunk (3 bytes) per K-tile at offset
        // 16 + kt*6 + k_grp*3 within the group.
        for (int kt = 0; kt < 16; kt += 4) {
            const unsigned char* dp_a = (const unsigned char*)(gp + 16 + (kt+0)*6 + k_grp*3);
            const unsigned char* dp_b = (const unsigned char*)(gp + 16 + (kt+1)*6 + k_grp*3);
            const unsigned char* dp_c = (const unsigned char*)(gp + 16 + (kt+2)*6 + k_grp*3);
            const unsigned char* dp_d = (const unsigned char*)(gp + 16 + (kt+3)*6 + k_grp*3);
            unsigned int pa = (unsigned int)dp_a[0]|((unsigned int)dp_a[1]<<8)|((unsigned int)dp_a[2]<<16);
            unsigned int pb = (unsigned int)dp_b[0]|((unsigned int)dp_b[1]<<8)|((unsigned int)dp_b[2]<<16);
            unsigned int pc = (unsigned int)dp_c[0]|((unsigned int)dp_c[1]<<8)|((unsigned int)dp_c[2]<<16);
            unsigned int pd = (unsigned int)dp_d[0]|((unsigned int)dp_d[1]<<8)|((unsigned int)dp_d[2]<<16);

            half8_t b_a, b_b, b_c, b_d;
            if (xg != nullptr) {
                b_a = *(const half8_t*)(xg + (kt+0)*16 + k_grp*8);
                b_b = *(const half8_t*)(xg + (kt+1)*16 + k_grp*8);
                b_c = *(const half8_t*)(xg + (kt+2)*16 + k_grp*8);
                b_d = *(const half8_t*)(xg + (kt+3)*16 + k_grp*8);
            } else { b_a = zero_b; b_b = zero_b; b_c = zero_b; b_d = zero_b; }

            // 3D bilinear decode per element: b0=(idx>>0)&1, b1=(idx>>1)&1, b2=(idx>>2)&1.
            // Vectorizes to WMMA (mirrors MQ2 approach extended to 3 bits).
            #define DQ3(d, pk, sh) do { \
                _Float16 b0 = (_Float16)(int)(((pk) >> (sh))     & 1u); \
                _Float16 b1 = (_Float16)(int)(((pk) >> ((sh)+1)) & 1u); \
                _Float16 b2 = (_Float16)(int)(((pk) >> ((sh)+2)) & 1u); \
                (d) = coA + b0*coB + b1*coC + b2*coD + (b0*b1)*coE + (b0*b2)*coF + (b1*b2)*coG + (b0*b1*b2)*coH; \
            } while (0)

            // Each half8_t covers 8 weights (one k_grp's half of a K-tile).
            #define DQ3_8(d, pk) \
                DQ3((d)[0], (pk),  0); DQ3((d)[1], (pk),  3); \
                DQ3((d)[2], (pk),  6); DQ3((d)[3], (pk),  9); \
                DQ3((d)[4], (pk), 12); DQ3((d)[5], (pk), 15); \
                DQ3((d)[6], (pk), 18); DQ3((d)[7], (pk), 21)

            half8_t a_reg;
            DQ3_8(a_reg, pa); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ3_8(a_reg, pb); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ3_8(a_reg, pc); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ3_8(a_reg, pd); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
            #undef DQ3_8
            #undef DQ3
        }
    }

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
