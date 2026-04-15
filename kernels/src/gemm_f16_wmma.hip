#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated F16×F32 batched GEMM for the vision encoder.
// gfx1100+ only (RDNA3 wave32 WMMA).
//
// Computes: Y[M, N] = W_f16[M, K] @ X_f32[N, K]^T
//
// Where:
//   W is row-major F16 [M, K]  (weight matrix)
//   X is row-major F32 [N, K]  (batch of input vectors)
//   Y is row-major F32 [M, N]  (output)
//
// WMMA layout: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
// computes  D[i][j] = sum_k a[i][k] * b[j][k] + c[i][j]
// which is  D = A @ B^T + C.
//
// So we load A from W (row tile) and B from X (row tile at same K range),
// and get Y_tile = W_tile @ X_tile^T.
//
// Grid: [ceil(M/16), ceil(N/16)]
// Block: [32]
// LDS: 0

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_f16_wmma(
    const _Float16* __restrict__ W,   // [M, K] row-major, F16
    const float*    __restrict__ X,   // [N, K] row-major, F32
    float*          __restrict__ Y,   // [M, N] row-major, F32
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= M || col_start >= N) return;

    // WMMA output accumulator: 8 floats per thread (16x16 tile, wave32)
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Process K in tiles of 16
    for (int k0 = 0; k0 < K; k0 += 16) {
        // Load A from W: 16 consecutive K values for each of 16 rows.
        // A_reg[i][j] = W[row_start + i, k0 + j]  (row tile of W)
        half16_t a_reg;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int r = row_start + i;
            if (r < M) {
                // Fast path: contiguous K load
                const _Float16* src = W + (long long)r * K + k0;
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    a_reg[i * 16 + j] = (k0 + j < K) ? src[j] : (_Float16)0.0f;
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) a_reg[i * 16 + j] = (_Float16)0.0f;
            }
        }

        // Load B from X: 16 consecutive K values for each of 16 rows (batch elements).
        // B_reg[i][j] = X[col_start + i, k0 + j]  (row tile of X, same K range as A)
        // WMMA computes A @ B^T, so output[m,n] = sum_k A[m,k] * B[n,k] = W[m,k]*X[n,k]
        half16_t b_reg;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int c = col_start + i;
            if (c < N) {
                const float* src = X + (long long)c * K + k0;
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    b_reg[i * 16 + j] = (k0 + j < K) ? (_Float16)src[j] : (_Float16)0.0f;
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) b_reg[i * 16 + j] = (_Float16)0.0f;
            }
        }

        // WMMA: acc += a_reg @ b_reg^T  (16x16x16 F16 matrix multiply)
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
    }

    // Write output
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
