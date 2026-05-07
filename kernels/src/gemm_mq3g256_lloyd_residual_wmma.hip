#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched MQ3-Lloyd-G256 GEMM with fused residual add.
// gfx1100+ only (RDNA3 wave32 WMMA). Phase A MVP — see
// docs/plans/mq3-lloyd-wmma-prefill.md.
//
// Lloyd port of gemm_hfq3g256_residual_wmma.hip. Group is 112 B
// (16 B fp16 codebook + 96 B 3-bit indices) instead of HFQ3's 104 B
// (8 B affine header + 96 B). Decode is per-(row, group) codebook
// lookup `cb_lds[my_row * 8 + q]` instead of HFQ3's `sc * q + zp`.
//
// LDS layout (Phase A choice — fp16 storage, no conversion at decode):
//   __shared__ _Float16 cb_lds[16 * 8];  // 256 B per workgroup.
// Each row of the 16-row tile has its own 8-entry codebook for the
// current group; LDS is restaged once per group iteration.
//
// Cooperative load: 32 threads × 4 fp16 each = 128 entries per group.
// Thread tid covers tile-row (tid >> 1), entries (tid & 1) * 4 + [0..4).
// This load mapping is independent of the WMMA my_tile_row mapping
// (tid & 15) — they're orthogonal.
//
// Sync discipline (acceptance constraint): exactly two __syncthreads()
// per group (one after cooperative load, one before next iteration's
// load to guard write-after-read on cb_lds), ZERO inside the K-tile
// loop. Per-K-tile barriers would crater perf — see plan §Phase A.
//
// Output convention (matches HFQ3 residual): lane (tid & 15) covers
// column (batch) tid&15, with 8 rows per lane via
//   acc[j] = C[2*j + (tid>>4)][tid & 15].
// Y += acc[j] (residual add — caller must initialize Y with the
// residual stream before launching).
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32].
// LDS: 256 B (cb_lds).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq3g256_lloyd_residual_wmma(
    const char*    __restrict__ A,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    // WMMA work-row mapping: tid & 15 is the tile-internal row [0..16).
    const int my_tile_row = tid & 15;
    const int safe_row = (row_start + my_tile_row < M) ? (row_start + my_tile_row) : (M - 1);
    const int safe_batch = (batch_start + my_tile_row < batch_size) ? (batch_start + my_tile_row) : 0;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 112;
    const char* row_base = A + safe_row * row_stride;
    const _Float16* x_base = X + (long long)safe_batch * K;

    // Cooperative-load mapping (32 threads × 4 fp16 each = 128 fp16 entries).
    // Independent of my_tile_row — load_tile_row spans 0..15 over tid 0..31.
    const int load_tile_row = tid >> 1;        // 0..15
    const int load_lo       = (tid & 1) * 4;    // 0 or 4
    const int safe_load_row = (row_start + load_tile_row < M) ? (row_start + load_tile_row) : (M - 1);
    const char* load_row_base = A + (long long)safe_load_row * row_stride;

    __shared__ _Float16 cb_lds[16 * 8];

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        // Cooperative codebook load for group g (16 row-codebooks × 8 entries).
        // 4 scalar fp16 reads per thread; alignment is guaranteed because
        // codebook header sits at offset 0 of the 112 B group, and 112 % 8 == 0.
        const __half* cb_h_load = (const __half*)(load_row_base + g * 112);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            cb_lds[load_tile_row * 8 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 112;
        const _Float16* xg = x_base + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 16);  // skip 16 B codebook header
        const int cb_base = my_tile_row * 8;

        // K-tile loop with K2 unroll. ZERO __syncthreads inside.
        for (int kt = 0; kt < 16; kt += 2) {
            // === Tile A (kt) ===
            const int k_off_a = kt * 16;
            const unsigned char* dpa = dp + kt * 6;

            // === Tile B (kt+1) — start loading early ===
            const int k_off_b = (kt + 1) * 16;
            const unsigned char* dpb = dp + (kt + 1) * 6;

            // Load both X tiles early (consecutive addresses — prefetcher-friendly).
            half16_t b_a = *(const half16_t*)(xg + k_off_a);
            half16_t b_b = *(const half16_t*)(xg + k_off_b);

            // 6 bytes for tile A = 2 chunks of 3 bytes each, packing 16 weights as
            // q0..q15 via the same cross-byte layout as HFQ3.
            unsigned int a0 = dpa[0], a1 = dpa[1], a2 = dpa[2];
            unsigned int a3 = dpa[3], a4 = dpa[4], a5 = dpa[5];

            unsigned int q0 = a0 & 7u;
            unsigned int q1 = (a0 >> 3) & 7u;
            unsigned int q2 = ((a0 >> 6) | (a1 << 2)) & 7u;
            unsigned int q3 = (a1 >> 1) & 7u;
            unsigned int q4 = (a1 >> 4) & 7u;
            unsigned int q5 = ((a1 >> 7) | (a2 << 1)) & 7u;
            unsigned int q6 = (a2 >> 2) & 7u;
            unsigned int q7 = (a2 >> 5) & 7u;
            unsigned int q8  = a3 & 7u;
            unsigned int q9  = (a3 >> 3) & 7u;
            unsigned int q10 = ((a3 >> 6) | (a4 << 2)) & 7u;
            unsigned int q11 = (a4 >> 1) & 7u;
            unsigned int q12 = (a4 >> 4) & 7u;
            unsigned int q13 = ((a4 >> 7) | (a5 << 1)) & 7u;
            unsigned int q14 = (a5 >> 2) & 7u;
            unsigned int q15 = (a5 >> 5) & 7u;

            half16_t a_a;
            #define DQ(i, q) a_a[i] = cb_lds[cb_base + (q)]
            DQ(0, q0);   DQ(1, q1);   DQ(2, q2);   DQ(3, q3);
            DQ(4, q4);   DQ(5, q5);   DQ(6, q6);   DQ(7, q7);
            DQ(8, q8);   DQ(9, q9);   DQ(10, q10); DQ(11, q11);
            DQ(12, q12); DQ(13, q13); DQ(14, q14); DQ(15, q15);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            // Tile B decode (reuse cb_lds — same group).
            unsigned int B0 = dpb[0], B1 = dpb[1], B2 = dpb[2];
            unsigned int B3 = dpb[3], B4 = dpb[4], B5 = dpb[5];

            unsigned int Q0 = B0 & 7u;
            unsigned int Q1 = (B0 >> 3) & 7u;
            unsigned int Q2 = ((B0 >> 6) | (B1 << 2)) & 7u;
            unsigned int Q3 = (B1 >> 1) & 7u;
            unsigned int Q4 = (B1 >> 4) & 7u;
            unsigned int Q5 = ((B1 >> 7) | (B2 << 1)) & 7u;
            unsigned int Q6 = (B2 >> 2) & 7u;
            unsigned int Q7 = (B2 >> 5) & 7u;
            unsigned int Q8  = B3 & 7u;
            unsigned int Q9  = (B3 >> 3) & 7u;
            unsigned int Q10 = ((B3 >> 6) | (B4 << 2)) & 7u;
            unsigned int Q11 = (B4 >> 1) & 7u;
            unsigned int Q12 = (B4 >> 4) & 7u;
            unsigned int Q13 = ((B4 >> 7) | (B5 << 1)) & 7u;
            unsigned int Q14 = (B5 >> 2) & 7u;
            unsigned int Q15 = (B5 >> 5) & 7u;

            DQ(0, Q0);   DQ(1, Q1);   DQ(2, Q2);   DQ(3, Q3);
            DQ(4, Q4);   DQ(5, Q5);   DQ(6, Q6);   DQ(7, Q7);
            DQ(8, Q8);   DQ(9, Q9);   DQ(10, Q10); DQ(11, Q11);
            DQ(12, Q12); DQ(13, Q13); DQ(14, Q14); DQ(15, Q15);
            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }

        // Guard write-after-read on cb_lds before next iteration's cooperative load.
        // Harmless on the final iteration.
        __syncthreads();
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    const int out_col = batch_start + (tid & 15);
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
