#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched MQ4-Lloyd-G256 GEMM with fused residual add.
// gfx1100+ only (RDNA3 wave32 WMMA). Phase A MVP — see
// docs/plans/mq4-lloyd-wmma-prefill.md.
//
// MQ4-Lloyd port of gemm_hfq4g256_residual_wmma.hip. Group is 160 B
// (32 B fp16 codebook header [16 entries] + 128 B 4-bit nibble-pair
// indices) instead of HFQ4's 136 B (8 B affine header + 128 B). Decode
// is per-(row, group) codebook lookup `cb_lds[my_row * 16 + q]` instead
// of HFQ4's `sc * q + zp`.
//
// Per-row codebook + LDS-staging pattern adopted from MQ3-Lloyd Phase A
// (kernels/src/gemm_mq3g256_lloyd_residual_wmma.hip, commit 869236d).
// fp16-LDS storage settled by MQ3 Phase A's empirical bench (fp16 wins
// 7.15% aggregate over fp32, bit-identical numerical output across all
// shapes — see benchmarks/results/devlog_20260507_lloyd_wmma_phase_a.md).
//
// LDS layout (fp16, no conversion at decode):
//   __shared__ _Float16 cb_lds[16 * 16];  // 512 B per workgroup.
// Each row of the 16-row tile has its own 16-entry codebook for the
// current group; LDS is restaged once per group iteration.
//
// Cooperative load: 32 threads × 8 fp16 each = 256 entries per group.
// Thread tid covers tile-row (tid >> 1), entries (tid & 1) * 8 + [0..8).
// This load mapping is independent of the WMMA my_tile_row mapping
// (tid & 15) — they're orthogonal. (MQ3 Phase A used tid × 4; MQ4
// doubles to × 8 because the codebook is 2× wider.)
//
// Sync discipline (acceptance constraint): exactly two __syncthreads()
// per group (one after cooperative load, one before next iteration's
// load to guard write-after-read on cb_lds), ZERO inside the K-tile
// loop. Per-K-tile barriers would crater perf — see plan §Phase A.
//
// K-tile decode uses uint32 reads (the 4-byte alignment is guaranteed:
// codebook header is 32 B → data section starts at offset 32, and the
// group stride 160 % 4 == 0). Mirrors HFQ4's nibble-pair decode pattern:
// 16 weights = 8 bytes = 2 uint32 = 16 nibbles via `(pk >> sh) & 0xF`.
//
// Output convention (matches MQ3 Phase A): lane (tid & 15) covers
// column (batch) tid&15, with 8 rows per lane via
//   acc[j] = C[2*j + (tid>>4)][tid & 15].
// Y += acc[j] (residual add — caller must initialize Y with the
// residual stream before launching).
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32].
// LDS: 512 B (cb_lds).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256_lloyd_residual_wmma(
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
    const long long row_stride = (long long)groups_per_row * 160;
    const char* row_base = A + safe_row * row_stride;
    const _Float16* x_base = X + (long long)safe_batch * K;

    // Cooperative-load mapping (32 threads × 8 fp16 each = 256 fp16 entries).
    // Independent of my_tile_row — load_tile_row spans 0..15 over tid 0..31.
    const int load_tile_row = tid >> 1;        // 0..15
    const int load_lo       = (tid & 1) * 8;    // 0 or 8
    const int safe_load_row = (row_start + load_tile_row < M) ? (row_start + load_tile_row) : (M - 1);
    const char* load_row_base = A + (long long)safe_load_row * row_stride;

    __shared__ _Float16 cb_lds[16 * 16];  // 512 B

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        // Cooperative codebook load for group g (16 row-codebooks × 16 entries).
        // 8 scalar fp16 reads per thread; alignment is guaranteed because
        // codebook header sits at offset 0 of the 160 B group, and
        // 160 % 16 == 0. (Vectorized `half8_t` loads possible later as a
        // Phase C optimization if needed.)
        const __half* cb_h_load = (const __half*)(load_row_base + g * 160);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            cb_lds[load_tile_row * 16 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 160;
        const _Float16* xg = x_base + g * 256;
        const char* dp = gp + 32;  // skip 32 B codebook header
        const int cb_base = my_tile_row * 16;

        // K-tile loop with K2 unroll. ZERO __syncthreads inside.
        // Per K-tile: 16 weights = 8 bytes (nibble pair) = 2 uint32 reads.
        for (int kt = 0; kt < 16; kt += 2) {
            // === Tile A (kt) ===
            const int k_off_a = kt * 16;
            unsigned int pk0a = *(const unsigned int*)(dp + kt * 8);
            unsigned int pk1a = *(const unsigned int*)(dp + kt * 8 + 4);

            // === Tile B (kt+1) — start loading early ===
            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0b = *(const unsigned int*)(dp + (kt + 1) * 8);
            unsigned int pk1b = *(const unsigned int*)(dp + (kt + 1) * 8 + 4);

            // Load both X tiles early (consecutive addresses — prefetcher-friendly).
            half16_t b_a = *(const half16_t*)(xg + k_off_a);
            half16_t b_b = *(const half16_t*)(xg + k_off_b);

            // Tile A decode: 16 nibbles via 2 uint32 with shifts 0,4,...,28.
            // Same pattern as HFQ4 nibble decode at gemm_hfq4g256_residual_wmma.hip:53,
            // but with codebook lookup instead of affine reconstruction.
            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = cb_lds[cb_base + (((pk) >> (sh)) & 0xFu)]
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            // Tile B decode (reuse cb_lds — same group). Overwrite a_reg.
            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
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
