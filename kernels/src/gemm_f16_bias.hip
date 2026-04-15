#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Fused F16 GEMM + bias for the vision encoder.
// Computes: Y[N, M] = X[N, K] @ W_f16[M, K]^T + bias[M]
//
// This is equivalent to:
//   1. gemm_f16: Y_t[M, N] = W[M, K] @ X[N, K]^T
//   2. transpose: Y[N, M] = Y_t^T
//   3. bias_add: Y += bias
//
// But done in ONE kernel launch, saving ~7MB of memory roundtrips
// (the intermediate Y_t doesn't need to be allocated or written).
//
// Grid: [N, 1] (one block per row of X / row of output Y)
// Block: [256] (256 threads process M columns via strided loop)
//
// Each thread handles M/256 output columns:
//   for col in my_columns:
//     sum = 0
//     for k in 0..K:
//       sum += X[row, k] * W[col, k]   (W accessed by column)
//     Y[row, col] = sum + bias[col]

__launch_bounds__(256, 4)
extern "C" __global__ void gemm_f16_bias(
    const _Float16* __restrict__ W,   // [M, K] row-major, F16
    const float*    __restrict__ X,   // [N, K] row-major, F32
    const float*    __restrict__ bias,// [M] bias vector
    float*          __restrict__ Y,   // [N, M] row-major, F32
    int M, int K, int N
) {
    const int row = blockIdx.x;  // row of X and Y
    const int tid = threadIdx.x;

    if (row >= N) return;

    const float* x_row = X + (long long)row * K;

    // Each thread processes columns [tid, tid+256, tid+512, ...]
    for (int col = tid; col < M; col += blockDim.x) {
        const _Float16* w_row = W + (long long)col * K;
        float sum = 0.0f;

        // 8-way unrolled dot product
        int k = 0;
        for (; k + 7 < K; k += 8) {
            sum += (float)w_row[k]     * x_row[k]
                 + (float)w_row[k + 1] * x_row[k + 1]
                 + (float)w_row[k + 2] * x_row[k + 2]
                 + (float)w_row[k + 3] * x_row[k + 3]
                 + (float)w_row[k + 4] * x_row[k + 4]
                 + (float)w_row[k + 5] * x_row[k + 5]
                 + (float)w_row[k + 6] * x_row[k + 6]
                 + (float)w_row[k + 7] * x_row[k + 7];
        }
        for (; k < K; k++) {
            sum += (float)w_row[k] * x_row[k];
        }

        // Add bias and write output directly (no intermediate Y_t)
        Y[(long long)row * M + col] = sum + bias[col];
    }
}
