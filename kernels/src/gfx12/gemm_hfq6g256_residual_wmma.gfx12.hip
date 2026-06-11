#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched HFQ6-G256 GEMM with fused residual add.
//
// **gfx12 (RDNA4) variant** — sister of gemm_hfq6g256_residual_wmma_k2.hip
// (gfx11 / RDNA3). Compile with `hipcc --offload-arch=gfx1200` (or 1201).
//
// Y[N, M] += X[N, K] @ A_hfq6[M, K]^T
//
// Caller seeds Y with the residual; the kernel fuses the `+=` (no atomics —
// see non-overlapping-write invariant below).
//
// HFQ6-G256 group layout (unchanged from gfx11): 200 bytes / 256 elements
//   bytes [0..4)   f32 scale
//   bytes [4..8)   f32 zero
//   bytes [8..200) 192B packed 6-bit values
//   Per K-tile (16 values): 12 bytes (16 × 6/8). Per K half-tile (8
//   values): 6 bytes.
//
// gfx12 differences from the gfx11 source — same recipe as
// gemm_qkv_hfq6g256_wmma.gfx12.hip and gemm_q8_0_residual_wmma.gfx12.hip:
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12.
//   2. A and B operands: half8_t (8 fp16 per lane) instead of half16_t.
//      Accumulator C is unchanged at float8_t.
//   3. K dimension is split across 2 lane-groups (k_grp = tid >> 4):
//        k_grp = 0  -> K = [0..7]  of the 16-K tile
//        k_grp = 1  -> K = [8..15] of the 16-K tile
//      Each lane reads 6 bytes (8 × 6-bit) of weights per K-tile.
//   4. C-output mapping (validated on R9700 / gfx1201 across the
//      hfq4/hfq6 QKV scaffolds + q8_0 residual sister):
//        acc[j] = C[8 * (tid >> 4) + j][tid & 15]
//
// === Non-overlapping-write invariant (DO NOT BREAK) ===
// The `Y[i] += acc[j]` at the end is **NOT atomic**. Race-free only
// because each (out_row, out_col) cell is written by exactly one lane
// in exactly one block under this kernel's grid. With the gfx12 mapping
//   acc[j] → C[8*(tid>>4) + j][tid & 15]
// lane group 0 writes rows 0..7, lane group 1 writes rows 8..15 — also
// non-overlapping by construction. Any K-split sibling would still need
// atomicAdd.
//
// HFQ6 nibble-extract trick: 8 packed 6-bit values = 48 bits = 6 bytes.
// Two unaligned uint reads at byte offsets {0, 3} (relative to the
// half-tile base) cover all 8 values cleanly:
//   k_grp = 0 -> base = byte_off + 0   (covers values 0..7)
//   k_grp = 1 -> base = byte_off + 6   (covers values 8..15)
//
// K4 unroll (issue #65 lever 2): 4 K-tiles per body, all loads up-front.
// Same lift pattern as the qkv hfq6 sister and the q8_0 residual sister.
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq6g256_residual_wmma_gfx12(
    const char*     __restrict__ A,
    const _Float16* __restrict__ X,   // [N x K], FP16
    float*          __restrict__ Y,   // [N x M], caller seeds with residual
    int M, int K, int batch_size
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid         = threadIdx.x;

    if (row_start   >= M)          return;
    if (batch_start >= batch_size) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row   = row_start + m_lane;
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    const int groups_per_row   = K / 256;
    const long long row_stride = (long long)groups_per_row * 200;
    const char* row_base       = A + (long long)safe_row * row_stride;

    const int safe_batch = ((batch_start + m_lane) < batch_size)
                           ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
        #define DQ6_8(d0, d1) \
            DQ6(0, (d0), 0);  DQ6(1, (d0), 6);  DQ6(2, (d0), 12); DQ6(3, (d0), 18); \
            DQ6(4, (d1), 0);  DQ6(5, (d1), 6);  DQ6(6, (d1), 12); DQ6(7, (d1), 18)

        // 16 K-tiles per 256-element group, processed 4 at a time.
        for (int kt = 0; kt < 16; kt += 4) {
            // 4 weight half-tiles, each 6 bytes (8 × 6-bit).
            const int bb_a = (kt + 0) * 12 + k_grp * 6;
            const int bb_b = (kt + 1) * 12 + k_grp * 6;
            const int bb_c = (kt + 2) * 12 + k_grp * 6;
            const int bb_d = (kt + 3) * 12 + k_grp * 6;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + bb_a);
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + bb_b);
            const unsigned char* dp_c = (const unsigned char*)(gp + 8 + bb_c);
            const unsigned char* dp_d = (const unsigned char*)(gp + 8 + bb_d);
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = *(const unsigned int*)(dp_a + 3);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = *(const unsigned int*)(dp_b + 3);
            unsigned int d0c = *(const unsigned int*)(dp_c);
            unsigned int d1c = *(const unsigned int*)(dp_c + 3);
            unsigned int d0d = *(const unsigned int*)(dp_d);
            unsigned int d1d = *(const unsigned int*)(dp_d + 3);

            half8_t b_a = *(const half8_t*)(x_base + (kt + 0) * 16 + k_grp * 8);
            half8_t b_b = *(const half8_t*)(x_base + (kt + 1) * 16 + k_grp * 8);
            half8_t b_c = *(const half8_t*)(x_base + (kt + 2) * 16 + k_grp * 8);
            half8_t b_d = *(const half8_t*)(x_base + (kt + 3) * 16 + k_grp * 8);

            half8_t a_reg;
            DQ6_8(d0a, d1a);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ6_8(d0b, d1b);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ6_8(d0c, d1c);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ6_8(d0d, d1d);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
        #undef DQ6_8
        #undef DQ6
    }

    // gfx12 C-output mapping: acc[j] = C[8*(tid>>4) + j][tid & 15]
    // `+=` semantics fuses the residual (caller seeded Y).
    const int out_col = batch_start + m_lane;  // batch index
    if (out_col >= batch_size) return;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 8 * k_grp + j;
        if (out_row < M) {
            Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
