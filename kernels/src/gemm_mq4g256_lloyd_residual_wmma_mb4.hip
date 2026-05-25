#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MQ4-Lloyd WMMA residual GEMM with 4× batch-tile fanout per workgroup.
// Phase D-A — see docs/plans/mq4-lloyd-batch-fanout-phase-d.md.
//
// Direct fork of gemm_mq4g256_lloyd_residual_wmma.hip (Phase A, single 16×16
// output tile per WG). The change: each WG now produces a 16×64 output tile
// by emitting 4 WMMA dispatches per K-tile that share the same A_reg decode.
// 4× weight reuse per WG → 4× the per-byte arithmetic intensity for weights →
// closes the batch ≥ 128 gap diagnosed in
// benchmarks/results/devlog_20260509_mq4_lloyd_gfx1151_bench.md (Lloyd-MQ4
// at 11 GiB/s vs HFQ4-mmq at 26 GiB/s on gfx1151).
//
// Grid: [ceil(M/16), ceil(batch/64)]. Block: [32].
// LDS: 512 B (cb_lds, unchanged from Phase A — codebook reuse is per-row,
// independent of batch fanout).
//
// Per K-tile inner loop:
//   decode A_reg                          (16 nibbles via cb_lds lookup)
//   for sub-tile s ∈ [0..4):
//     load b_s = X[batch_start + 16*s + my_tile_row, kt..kt+16]
//     wmma(A_reg, b_s, acc[s])
// 4 distinct B registers (b0..b3) declared as named variables so the compiler
// allocates separate VGPR windows. Streaming b through one variable causes
// the compiler to recycle the same physical register and emit
// `s_waitcnt vmcnt(0)` between every load+WMMA — sequential, not pipelined
// (verified via disassembly on a recycle-prone earlier draft of this file).
//
// Output per lane: 4 sub-tiles × 8 outputs = 32 floats per lane.
// Same Phase A output convention per sub-tile: acc[s][j] writes to
//   Y[(batch_start + 16*s + (tid & 15)) * M + (row_start + 2*j + (tid >> 4))].

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256_lloyd_residual_wmma_mb4(
    const char*    __restrict__ A,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 64;  // 4 sub-tiles × 16 = 64

    if (row_start >= M || batch_start >= batch_size) return;

    // WMMA work-row mapping: tid & 15 is the tile-internal row [0..16).
    const int my_tile_row = tid & 15;
    const int safe_row = (row_start + my_tile_row < M) ? (row_start + my_tile_row) : (M - 1);

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 160;
    const char* row_base = A + safe_row * row_stride;

    // Per-sub-tile safe batch indices for X loads. Out-of-range sub-tiles
    // fall back to row 0 (dummy load); their WMMA output is discarded at
    // the write guard below.
    long long safe_batch_s[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int b_idx = batch_start + 16 * s + my_tile_row;
        safe_batch_s[s] = (b_idx < batch_size) ? (long long)b_idx : 0LL;
    }

    // Cooperative-load mapping (unchanged from Phase A): 32 threads × 8 fp16
    // = 256 entries per group. Independent of WMMA my_tile_row mapping.
    const int load_tile_row = tid >> 1;        // 0..15
    const int load_lo       = (tid & 1) * 8;   // 0 or 8
    const int safe_load_row = (row_start + load_tile_row < M) ? (row_start + load_tile_row) : (M - 1);
    const char* load_row_base = A + (long long)safe_load_row * row_stride;

    __shared__ _Float16 cb_lds[16 * 16];  // 512 B

    float8_t acc[4] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    for (int g = 0; g < groups_per_row; g++) {
        // Cooperative codebook load — identical to Phase A.
        const __half* cb_h_load = (const __half*)(load_row_base + g * 160);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            cb_lds[load_tile_row * 16 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 160;
        const char* dp = gp + 32;  // skip 32 B codebook header
        const int cb_base = my_tile_row * 16;

        // X bases per sub-tile for this group. Each lane reads K bytes of
        // its own batch row from each of the 4 sub-tiles.
        const _Float16* xg_s[4];
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            xg_s[s] = X + safe_batch_s[s] * (long long)K + g * 256;
        }

        // K-tile loop with K2 unroll. ZERO __syncthreads inside.
        for (int kt = 0; kt < 16; kt += 2) {
            // === Tile A (kt) ===
            unsigned int pk0a = *(const unsigned int*)(dp + kt * 8);
            unsigned int pk1a = *(const unsigned int*)(dp + kt * 8 + 4);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = cb_lds[cb_base + (((pk) >> (sh)) & 0xFu)]
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);

            const int k_off_a = kt * 16;
            // Load all 4 B-tiles into distinct named VGPRs first, then issue
            // the 4 WMMAs back-to-back. This lets the compiler emit one
            // `s_waitcnt vmcnt(0)` covering all four loads, instead of one
            // per load+WMMA pair.
            half16_t b0a = *(const half16_t*)(xg_s[0] + k_off_a);
            half16_t b1a = *(const half16_t*)(xg_s[1] + k_off_a);
            half16_t b2a = *(const half16_t*)(xg_s[2] + k_off_a);
            half16_t b3a = *(const half16_t*)(xg_s[3] + k_off_a);
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b0a, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b1a, acc[1]);
            acc[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b2a, acc[2]);
            acc[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b3a, acc[3]);

            // === Tile B (kt+1) — reuse cb_lds (same group) ===
            unsigned int pk0b = *(const unsigned int*)(dp + (kt + 1) * 8);
            unsigned int pk1b = *(const unsigned int*)(dp + (kt + 1) * 8 + 4);

            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            #undef DQ

            const int k_off_b = (kt + 1) * 16;
            half16_t b0b = *(const half16_t*)(xg_s[0] + k_off_b);
            half16_t b1b = *(const half16_t*)(xg_s[1] + k_off_b);
            half16_t b2b = *(const half16_t*)(xg_s[2] + k_off_b);
            half16_t b3b = *(const half16_t*)(xg_s[3] + k_off_b);
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b0b, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b1b, acc[1]);
            acc[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b2b, acc[2]);
            acc[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b3b, acc[3]);
        }

        // Guard write-after-read on cb_lds before next iteration's load.
        __syncthreads();
    }

    // Output writes: 4 sub-tiles × 8 outputs/lane.
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int out_col = batch_start + 16 * s + (tid & 15);
        if (out_col < batch_size) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int out_row = row_start + 2 * j + (tid >> 4);
                if (out_row < M) {
                    Y[(long long)out_col * M + out_row] += acc[s][j];
                }
            }
        }
    }
}
