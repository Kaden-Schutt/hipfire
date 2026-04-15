#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Optimized WMMA F16×F32 GEMM with shared memory tiling for the vision encoder.
// gfx1100+ only (RDNA3 wave32 WMMA).
//
// Computes: Y[M, N] = W_f16[M, K] @ X_f32[N, K]^T
//
// Optimization over gemm_f16_wmma:
//   - Shared memory tiling: loads K-tiles into LDS once, reuses across WMMA
//   - Double-buffering: overlaps LDS loads with WMMA compute
//   - Coalesced global loads: 32 threads load contiguous 16x16 tiles
//
// Grid: [ceil(M/16), ceil(N/16)]
// Block: [32]
// LDS: 2 * 16 * 16 * sizeof(half) * 2 (double-buffered A + B) = 2048 bytes

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define TILE_K 16
#define TILE_MN 16
#define LDS_TILE_SIZE (TILE_K * TILE_MN)  // 256 halfs = 512 bytes per tile

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_f16_wmma_tiled(
    const _Float16* __restrict__ W,   // [M, K] row-major, F16
    const float*    __restrict__ X,   // [N, K] row-major, F32
    float*          __restrict__ Y,   // [M, N] row-major, F32
    int M, int K, int N
) {
    const int row_start = blockIdx.x * TILE_MN;
    const int col_start = blockIdx.y * TILE_MN;
    const int tid = threadIdx.x;

    if (row_start >= M || col_start >= N) return;

    // Shared memory: double-buffered A and B tiles
    // Each tile is 16x16 halfs = 256 halfs = 512 bytes
    // 4 tiles total = 2048 bytes
    __shared__ _Float16 smem_a[2][LDS_TILE_SIZE];
    __shared__ _Float16 smem_b[2][LDS_TILE_SIZE];

    // WMMA output accumulator: 8 floats per thread (16x16 tile, wave32)
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const int n_k_tiles = (K + TILE_K - 1) / TILE_K;

    // ─── Helper: load one 16x16 tile from W (F16) into shared memory ───
    // 32 threads cooperatively load 256 elements (8 per thread)
    auto load_a_tile = [&](int k_tile, int buf) {
        int k0 = k_tile * TILE_K;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int linear = tid * 8 + i;
            int r = linear / TILE_K;
            int c = linear % TILE_K;
            int global_r = row_start + r;
            int global_k = k0 + c;
            _Float16 val = (_Float16)0.0f;
            if (global_r < M && global_k < K) {
                val = W[(long long)global_r * K + global_k];
            }
            smem_a[buf][linear] = val;
        }
    };

    // ─── Helper: load one 16x16 tile from X (F32→F16) into shared memory ───
    auto load_b_tile = [&](int k_tile, int buf) {
        int k0 = k_tile * TILE_K;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int linear = tid * 8 + i;
            int r = linear / TILE_K;
            int c = linear % TILE_K;
            int global_c = col_start + r;
            int global_k = k0 + c;
            _Float16 val = (_Float16)0.0f;
            if (global_c < N && global_k < K) {
                val = (_Float16)X[(long long)global_c * K + global_k];
            }
            smem_b[buf][linear] = val;
        }
    };

    // ─── Prefetch first tile ───
    load_a_tile(0, 0);
    load_b_tile(0, 0);
    __syncthreads();

    // ─── Main K-tile loop with double buffering ───
    for (int kt = 0; kt < n_k_tiles; kt++) {
        int cur = kt & 1;
        int nxt = cur ^ 1;

        // Prefetch next tile while computing current
        if (kt + 1 < n_k_tiles) {
            load_a_tile(kt + 1, nxt);
            load_b_tile(kt + 1, nxt);
        }

        // Load A and B from shared memory into registers for WMMA
        half16_t a_reg, b_reg;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                a_reg[i * 16 + j] = smem_a[cur][i * TILE_K + j];
                b_reg[i * 16 + j] = smem_b[cur][i * TILE_K + j];
            }
        }

        // WMMA: acc += A_tile @ B_tile^T (16x16x16 F16 matrix multiply)
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);

        __syncthreads();
    }

    // ─── Write output ───
    // RDNA3 wave32 WMMA layout: acc[j] maps to row = 2*j + (tid>>4), col = tid & 15
    const int out_col = col_start + (tid & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_row * N + out_col] = acc[j];
            }
        }
    }
}
