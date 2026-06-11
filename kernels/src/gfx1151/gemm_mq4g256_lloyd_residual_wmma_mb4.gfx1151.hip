#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MQ4-Lloyd WMMA residual GEMM, 4× batch-tile fanout — gfx1151 K4 variant.
//
// K4 unroll: 4 K-tiles (64 elements) per inner loop iteration vs the generic
// kernel's K2 (32 elements). Front-loads all 8 nibble-pack reads before any DQ
// or WMMA, giving the memory subsystem (LPDDR5x unified on Strix Halo) a wider
// window to prefetch per-row weight data before it's needed.
//
// All other structure identical to gemm_mq4g256_lloyd_residual_wmma_mb4.hip:
// 16×64 output tile, 4 batch sub-tiles, cooperative codebook LDS load.
// Bench: benchmarks/results/devlog_20260509_mq4_lloyd_gfx1151_bench.md
// (K-schedule lever — "cheap to bisect, marginal; needs bench").
//
// Grid: [ceil(M/16), ceil(batch/64)]. Block: [32]. LDS: 512 B.

#if defined(__gfx1151__)

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
    const int batch_start = blockIdx.y * 64;

    if (row_start >= M || batch_start >= batch_size) return;

    const int my_tile_row = tid & 15;
    const int safe_row = (row_start + my_tile_row < M) ? (row_start + my_tile_row) : (M - 1);

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 160;
    const char* row_base = A + safe_row * row_stride;

    long long safe_batch_s[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int b_idx = batch_start + 16 * s + my_tile_row;
        safe_batch_s[s] = (b_idx < batch_size) ? (long long)b_idx : 0LL;
    }

    const int load_tile_row = tid >> 1;
    const int load_lo       = (tid & 1) * 8;
    const int safe_load_row = (row_start + load_tile_row < M) ? (row_start + load_tile_row) : (M - 1);
    const char* load_row_base = A + (long long)safe_load_row * row_stride;

    __shared__ _Float16 cb_lds[16 * 16];

    float8_t acc[4] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    for (int g = 0; g < groups_per_row; g++) {
        const __half* cb_h_load = (const __half*)(load_row_base + g * 160);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            cb_lds[load_tile_row * 16 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 160;
        const char* dp = gp + 32;
        const int cb_base = my_tile_row * 16;

        const _Float16* xg_s[4];
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            xg_s[s] = X + safe_batch_s[s] * (long long)K + g * 256;
        }

        // K4 unroll: 4 tiles per iteration (4 × 8 B = 32 B of nibble data).
        // All 8 nibble-pack reads are front-loaded to overlap with preceding WMMAs.
        for (int kt = 0; kt < 16; kt += 4) {
            const unsigned char* dp4 = (const unsigned char*)dp + kt * 8;

            unsigned int pk0a = *(const unsigned int*)dp4;
            unsigned int pk1a = *(const unsigned int*)(dp4 + 4);
            unsigned int pk0b = *(const unsigned int*)(dp4 + 8);
            unsigned int pk1b = *(const unsigned int*)(dp4 + 12);
            unsigned int pk0c = *(const unsigned int*)(dp4 + 16);
            unsigned int pk1c = *(const unsigned int*)(dp4 + 20);
            unsigned int pk0d = *(const unsigned int*)(dp4 + 24);
            unsigned int pk1d = *(const unsigned int*)(dp4 + 28);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = cb_lds[cb_base + (((pk) >> (sh)) & 0xFu)]

            // === Tile kt+0 ===
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);
            half16_t b0a = *(const half16_t*)(xg_s[0] + kt * 16);
            half16_t b1a = *(const half16_t*)(xg_s[1] + kt * 16);
            half16_t b2a = *(const half16_t*)(xg_s[2] + kt * 16);
            half16_t b3a = *(const half16_t*)(xg_s[3] + kt * 16);
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b0a, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b1a, acc[1]);
            acc[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b2a, acc[2]);
            acc[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b3a, acc[3]);

            // === Tile kt+1 ===
            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            half16_t b0b = *(const half16_t*)(xg_s[0] + (kt + 1) * 16);
            half16_t b1b = *(const half16_t*)(xg_s[1] + (kt + 1) * 16);
            half16_t b2b = *(const half16_t*)(xg_s[2] + (kt + 1) * 16);
            half16_t b3b = *(const half16_t*)(xg_s[3] + (kt + 1) * 16);
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b0b, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b1b, acc[1]);
            acc[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b2b, acc[2]);
            acc[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b3b, acc[3]);

            // === Tile kt+2 ===
            DQ(0,  pk0c,  0); DQ(1,  pk0c,  4); DQ(2,  pk0c,  8); DQ(3,  pk0c, 12);
            DQ(4,  pk0c, 16); DQ(5,  pk0c, 20); DQ(6,  pk0c, 24); DQ(7,  pk0c, 28);
            DQ(8,  pk1c,  0); DQ(9,  pk1c,  4); DQ(10, pk1c,  8); DQ(11, pk1c, 12);
            DQ(12, pk1c, 16); DQ(13, pk1c, 20); DQ(14, pk1c, 24); DQ(15, pk1c, 28);
            half16_t b0c = *(const half16_t*)(xg_s[0] + (kt + 2) * 16);
            half16_t b1c = *(const half16_t*)(xg_s[1] + (kt + 2) * 16);
            half16_t b2c = *(const half16_t*)(xg_s[2] + (kt + 2) * 16);
            half16_t b3c = *(const half16_t*)(xg_s[3] + (kt + 2) * 16);
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b0c, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b1c, acc[1]);
            acc[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b2c, acc[2]);
            acc[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b3c, acc[3]);

            // === Tile kt+3 ===
            DQ(0,  pk0d,  0); DQ(1,  pk0d,  4); DQ(2,  pk0d,  8); DQ(3,  pk0d, 12);
            DQ(4,  pk0d, 16); DQ(5,  pk0d, 20); DQ(6,  pk0d, 24); DQ(7,  pk0d, 28);
            DQ(8,  pk1d,  0); DQ(9,  pk1d,  4); DQ(10, pk1d,  8); DQ(11, pk1d, 12);
            DQ(12, pk1d, 16); DQ(13, pk1d, 20); DQ(14, pk1d, 24); DQ(15, pk1d, 28);
            #undef DQ
            half16_t b0d = *(const half16_t*)(xg_s[0] + (kt + 3) * 16);
            half16_t b1d = *(const half16_t*)(xg_s[1] + (kt + 3) * 16);
            half16_t b2d = *(const half16_t*)(xg_s[2] + (kt + 3) * 16);
            half16_t b3d = *(const half16_t*)(xg_s[3] + (kt + 3) * 16);
            acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b0d, acc[0]);
            acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b1d, acc[1]);
            acc[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b2d, acc[2]);
            acc[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b3d, acc[3]);
        }

        __syncthreads();
    }

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

#else

// Stub: this kernel is gfx1151-only. Other archs route through the generic mb4 kernel.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256_lloyd_residual_wmma_mb4(
    const char*, const _Float16*, float*, int, int, int) {}

#endif
