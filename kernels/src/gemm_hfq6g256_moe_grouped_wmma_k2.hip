#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx11 (RDNA3) HFQ6/MQ6 grouped-WMMA GEMM for MoE prefill — the
// `_w32` sister of `gemm_hfq6g256_moe_grouped_wmma.gfx12.hip`. Until
// now MQ6 grouped prefill existed ONLY on gfx12, so MQ6 (and any
// graded blend with an MQ6 hot tier) fell back to per-token Path-1 on
// gfx11/gfx1151. This closes that gap.
//
// Pure composition of two validated patterns (identical to how the
// gfx12 sister was built):
//   - gfx11 HFQ4 grouped dispatch + WMMA layout (half16_t fragments,
//     __builtin_amdgcn_wmma_f32_16x16x16_f16_w32, the 2×K-tile inner
//     pipeline, the 2*j+(tid>>4) output mapping) from
//     gemm_hfq4g256_moe_grouped_wmma_k2.hip.
//   - HFQ6-G256 dequant (200 B/group: 4 B scale + 4 B zero + 192 B
//     packed 6-bit) from gemm_hfq6g256_moe_grouped_wmma.gfx12.hip.
//
// HFQ6 vs HFQ4 inner-loop deltas (everything else identical):
//   - Group stride 200 B (vs 136 B).
//   - Per K-tile (16 K-values) the packed payload is 12 B (16×6=96 bits)
//     vs HFQ4's 8 B. The 16 values factor into FOUR overlapping uint
//     reads (at +0/+3/+6/+9), each yielding 4 × 6-bit values via shifts
//     0/6/12/18 & mask 63u. This is the half16_t (16-wide) form of the
//     gfx12 sister's DQ6_8 (which unpacks 8 at a time per k_grp half).
//
// MQ6G256 and HFQ6G256 share the IDENTICAL on-disk layout; the FWHT
// rotation is applied by the caller to X before dispatch, so the kernel
// is dtype-agnostic between HFQ6 and MQ6 (same as the gfx12 sister).
//
// Per-block contract (unchanged from the HFQ4 grouped variant):
//   - Tile (blockIdx.x, blockIdx.y) owns a 16-row × 16-slot output tile;
//     all 16 slots share one expert (scatter pads expert_offsets to 16).
//   - expert_tile_ids[tile_y] == -1 → early-return (sync-free dispatch).
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq6g256_moe_grouped_wmma_k2(
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
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        // 8 iterations × 2 K-tiles each = 16 K-tiles per group.
        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            // HFQ6 weight: 12 B per 16-value K-tile, read as four
            // overlapping uints (+0/+3/+6/+9) for clean 6-bit factoring.
            const unsigned char* wa = (const unsigned char*)(gp + 8 + kt * 12);
            const unsigned char* wb = (const unsigned char*)(gp + 8 + (kt + 1) * 12);
            // a3/c3 use only bits 0..23 (3 bytes); a *(uint*)(w+9) fetches a 4th
            // byte (w+12) that is 1 B past the group on the last tile of the last
            // group → GPU VM read fault on tightly-packed expert buffers. Read 3 B.
            unsigned int a0 = *(const unsigned int*)(wa);
            unsigned int a1 = *(const unsigned int*)(wa + 3);
            unsigned int a2 = *(const unsigned int*)(wa + 6);
            unsigned int a3 = (unsigned int)wa[9] | ((unsigned int)wa[10] << 8) | ((unsigned int)wa[11] << 16);
            unsigned int c0 = *(const unsigned int*)(wb);
            unsigned int c1 = *(const unsigned int*)(wb + 3);
            unsigned int c2 = *(const unsigned int*)(wb + 6);
            unsigned int c3 = (unsigned int)wb[9] | ((unsigned int)wb[10] << 8) | ((unsigned int)wb[11] << 16);

            // X loads — substitute zero vector when the lane has no source token.
            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            // Dequantize tile A (16 × 6-bit, mask 63u, shifts 0/6/12/18).
            half16_t a_a;
            #define DQ6(reg, i, dw, sh) reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
            DQ6(a_a, 0,  a0, 0); DQ6(a_a, 1,  a0, 6); DQ6(a_a, 2,  a0, 12); DQ6(a_a, 3,  a0, 18);
            DQ6(a_a, 4,  a1, 0); DQ6(a_a, 5,  a1, 6); DQ6(a_a, 6,  a1, 12); DQ6(a_a, 7,  a1, 18);
            DQ6(a_a, 8,  a2, 0); DQ6(a_a, 9,  a2, 6); DQ6(a_a, 10, a2, 12); DQ6(a_a, 11, a2, 18);
            DQ6(a_a, 12, a3, 0); DQ6(a_a, 13, a3, 6); DQ6(a_a, 14, a3, 12); DQ6(a_a, 15, a3, 18);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            // Dequantize tile B (reuse a_a).
            DQ6(a_a, 0,  c0, 0); DQ6(a_a, 1,  c0, 6); DQ6(a_a, 2,  c0, 12); DQ6(a_a, 3,  c0, 18);
            DQ6(a_a, 4,  c1, 0); DQ6(a_a, 5,  c1, 6); DQ6(a_a, 6,  c1, 12); DQ6(a_a, 7,  c1, 18);
            DQ6(a_a, 8,  c2, 0); DQ6(a_a, 9,  c2, 6); DQ6(a_a, 10, c2, 12); DQ6(a_a, 11, c2, 18);
            DQ6(a_a, 12, c3, 0); DQ6(a_a, 13, c3, 6); DQ6(a_a, 14, c3, 12); DQ6(a_a, 15, c3, 18);
            #undef DQ6

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
