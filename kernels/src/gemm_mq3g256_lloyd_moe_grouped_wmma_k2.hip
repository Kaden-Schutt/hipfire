#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx11 (RDNA3/3.5) MQ3-Lloyd grouped-WMMA GEMM for MoE prefill.
// T3-3L cold-tier building block: same MoE Path-2 scatter contract as
// the HFQ4/HFQ6/MQ2-Lloyd grouped sisters, with MQ3-Lloyd weight decode.
//
// MQ3-Lloyd-G256 storage layout (112 B/group):
//   bytes [0..16)  : 8 × fp16 codebook entries (sorted ascending at quant-time)
//   bytes [16..112): 96 B packed 3-bit indices — 256 values × 3 bits, LSB-first.
//                    Each K-tile (16 weights) uses 6 B (= 48 bits = 16 × 3-bit).
//                    The 6 B splits into two 3-byte packs (p0: weights 0..7,
//                    p1: weights 8..15); idx_i = (pk >> (3*i)) & 7u.
//
// Dequant per weight (3D bilinear expansion, exact for any 8-entry table):
//   b0 = (idx >> 0) & 1,  b1 = (idx >> 1) & 1,  b2 = (idx >> 2) & 1
//   w = A + b0*B + b1*C + b2*D + b0b1*E + b0b2*F + b1b2*G + b0b1b2*H
//   where:
//     A=cb[0], B=cb[1]-cb[0], C=cb[2]-cb[0], D=cb[4]-cb[0],
//     E=cb[3]-cb[2]-cb[1]+cb[0], F=cb[5]-cb[4]-cb[1]+cb[0],
//     G=cb[6]-cb[4]-cb[2]+cb[0],
//     H=cb[7]-cb[6]-cb[5]+cb[4]-cb[3]+cb[2]+cb[1]-cb[0]
//
// This mirrors the MQ2-Lloyd bilinear decomposition (gemm_mq2g256_lloyd_moe_grouped_wmma_k2)
// extended to 3 bits. Key property: all 8 fp16 codebook values collapse into
// 8 fp16 coefficients (A..H) computed ONCE per group per lane; the per-weight
// decode is then 7 FMAs on bit-extracted scalars. The compiler vectorizes
// this into WMMA fp16 instructions — per the 2026-05-22 analysis, the
// ternary-chain / register-array LDS-indexed form suppressed WMMA emission
// while the FMA-decomposed form vectorizes correctly.
//
// Per-lane codebook: each lane uses its OWN row_base / gp (safe_row =
// row_start + (tid & 15)), so each lane's codebook is from a DIFFERENT
// weight row. The __shared__ approach used by non-grouped kernels (e.g.
// gemm_gate_up_mq3g256_lloyd_wmma.hip) is WRONG here — LDS is per-block,
// not per-lane, so a broadcast would overwrite one lane's codebook with
// another's. The register-based bilinear decomposition is the correct
// approach for the grouped GEMM contract.
//
// Inner loop: 2×K-tile pipeline matching gemm_hfq6g256_moe_grouped_wmma_k2.
// 16 K-tiles per group × 6 B/tile = 96 B ≡ bytes [16..112). ✓
//
// Per-block contract (unchanged from HFQ4/HFQ6 grouped variant):
//   - Tile (blockIdx.x, blockIdx.y) owns a 16-row × 16-slot output tile.
//   - All 16 slots share one expert (Phase-2 pad-then-scan, expert_offsets
//     aligned to BLOCK_M = 16).
//   - expert_tile_ids[tile_y] == -1 → early-return (sync-free dispatch).
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq3g256_lloyd_moe_grouped_wmma_k2(
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
    const char* row_base = A + (long long)safe_row * groups_per_row * 112;   // 112 B/group

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 112;
        // Per-lane codebook: 8 fp16 entries from bytes [0..16) of this lane's
        // weight row group. Each lane reads its OWN codebook from its OWN row.
        const __half* cb_h = reinterpret_cast<const __half*>(gp);
        _Float16 cb0 = (_Float16)__half2float(cb_h[0]);
        _Float16 cb1 = (_Float16)__half2float(cb_h[1]);
        _Float16 cb2 = (_Float16)__half2float(cb_h[2]);
        _Float16 cb3 = (_Float16)__half2float(cb_h[3]);
        _Float16 cb4 = (_Float16)__half2float(cb_h[4]);
        _Float16 cb5 = (_Float16)__half2float(cb_h[5]);
        _Float16 cb6 = (_Float16)__half2float(cb_h[6]);
        _Float16 cb7 = (_Float16)__half2float(cb_h[7]);

        // 3D bilinear decomposition coefficients (computed once per group).
        // Exact for any 8-entry table; vectorizable by the compiler.
        const _Float16 coA = cb0;
        const _Float16 coB = cb1 - cb0;                          // b0 contribution
        const _Float16 coC = cb2 - cb0;                          // b1 contribution
        const _Float16 coD = cb4 - cb0;                          // b2 contribution
        const _Float16 coE = cb3 - cb2 - cb1 + cb0;             // b0*b1
        const _Float16 coF = cb5 - cb4 - cb1 + cb0;             // b0*b2
        const _Float16 coG = cb6 - cb4 - cb2 + cb0;             // b1*b2
        const _Float16 coH = cb7 - cb6 - cb5 + cb4 - cb3 + cb2 + cb1 - cb0; // b0*b1*b2

        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;
            // Index payload at [16..112): each K-tile = 6 B (16 × 3-bit).
            const unsigned char* wa = (const unsigned char*)(gp + 16 + kt       * 6);
            const unsigned char* wb = (const unsigned char*)(gp + 16 + (kt + 1) * 6);
            // 6 B → two 24-bit packs (8 × 3-bit indices each), LSB-first.
            unsigned int p0a = (unsigned int)wa[0] | ((unsigned int)wa[1] << 8) | ((unsigned int)wa[2] << 16);
            unsigned int p1a = (unsigned int)wa[3] | ((unsigned int)wa[4] << 8) | ((unsigned int)wa[5] << 16);
            unsigned int p0b = (unsigned int)wb[0] | ((unsigned int)wb[1] << 8) | ((unsigned int)wb[2] << 16);
            unsigned int p1b = (unsigned int)wb[3] | ((unsigned int)wb[4] << 8) | ((unsigned int)wb[5] << 16);

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            // 3D bilinear decode: extract b0/b1/b2 from 3-bit index, apply FMAs.
            // Compiler vectorizes this into WMMA (verified pattern mirrors MQ2).
            #define DQ3(reg, i, pk, sh) do { \
                _Float16 b0 = (_Float16)(int)(((pk) >> (sh))     & 1u); \
                _Float16 b1 = (_Float16)(int)(((pk) >> ((sh)+1)) & 1u); \
                _Float16 b2 = (_Float16)(int)(((pk) >> ((sh)+2)) & 1u); \
                (reg)[i] = coA + b0*coB + b1*coC + b2*coD + (b0*b1)*coE + (b0*b2)*coF + (b1*b2)*coG + (b0*b1*b2)*coH; \
            } while (0)

            half16_t a_a;
            DQ3(a_a, 0,  p0a, 0); DQ3(a_a, 1,  p0a, 3); DQ3(a_a, 2,  p0a, 6);  DQ3(a_a, 3,  p0a, 9);
            DQ3(a_a, 4,  p0a, 12);DQ3(a_a, 5,  p0a, 15);DQ3(a_a, 6,  p0a, 18); DQ3(a_a, 7,  p0a, 21);
            DQ3(a_a, 8,  p1a, 0); DQ3(a_a, 9,  p1a, 3); DQ3(a_a, 10, p1a, 6);  DQ3(a_a, 11, p1a, 9);
            DQ3(a_a, 12, p1a, 12);DQ3(a_a, 13, p1a, 15);DQ3(a_a, 14, p1a, 18); DQ3(a_a, 15, p1a, 21);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            DQ3(a_a, 0,  p0b, 0); DQ3(a_a, 1,  p0b, 3); DQ3(a_a, 2,  p0b, 6);  DQ3(a_a, 3,  p0b, 9);
            DQ3(a_a, 4,  p0b, 12);DQ3(a_a, 5,  p0b, 15);DQ3(a_a, 6,  p0b, 18); DQ3(a_a, 7,  p0b, 21);
            DQ3(a_a, 8,  p1b, 0); DQ3(a_a, 9,  p1b, 3); DQ3(a_a, 10, p1b, 6);  DQ3(a_a, 11, p1b, 9);
            DQ3(a_a, 12, p1b, 12);DQ3(a_a, 13, p1b, 15);DQ3(a_a, 14, p1b, 18); DQ3(a_a, 15, p1b, 21);
            #undef DQ3
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
