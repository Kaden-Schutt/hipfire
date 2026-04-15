#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Optimized F16xF32 GEMM for the vision encoder.
// Computes: Y[M, N] = W_f16[M, K] @ X_f32[N, K]^T
//
// Same grid as naive gemm_f16: Grid=[M, N], Block=[32]
// Optimizations over naive:
//   1. 8-way unrolled inner loop (ILP)
//   2. Stride-256 instead of stride-32 (better cache line utilization)
//   3. Registers over shared memory (avoids occupancy penalty)
//
// Shared memory tiling doesn't help here because:
//   - K is small (1152-4304), fits in L2 cache
//   - LDS allocation reduces occupancy too much
//   - The naive kernel's bottleneck is ILP, not memory bandwidth

__launch_bounds__(32, 20)
extern "C" __global__ void gemm_f16_tiled(
    const _Float16* __restrict__ W,   // [M, K] row-major, F16
    const float*    __restrict__ X,   // [N, K] row-major, F32
    float*          __restrict__ Y,   // [M, N] row-major, F32
    int M, int K, int N
) {
    const int m = blockIdx.x;
    const int n = blockIdx.y;
    const int tid = threadIdx.x;

    if (m >= M || n >= N) return;

    const _Float16* w_row = W + (long long)m * K;
    const float* x_row = X + (long long)n * K;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    // 4-way accumulator with stride-128 (4x32=128 elements per iteration)
    // Better ILP than naive's single accumulator
    int k = tid * 4;
    for (; k + 511 < K; k += 512) {
        sum0 += (float)w_row[k]       * x_row[k]
              + (float)w_row[k + 1]   * x_row[k + 1]
              + (float)w_row[k + 2]   * x_row[k + 2]
              + (float)w_row[k + 3]   * x_row[k + 3]
              + (float)w_row[k + 4]   * x_row[k + 4]
              + (float)w_row[k + 5]   * x_row[k + 5]
              + (float)w_row[k + 6]   * x_row[k + 6]
              + (float)w_row[k + 7]   * x_row[k + 7]
              + (float)w_row[k + 8]   * x_row[k + 8]
              + (float)w_row[k + 9]   * x_row[k + 9]
              + (float)w_row[k + 10]  * x_row[k + 10]
              + (float)w_row[k + 11]  * x_row[k + 11]
              + (float)w_row[k + 12]  * x_row[k + 12]
              + (float)w_row[k + 13]  * x_row[k + 13]
              + (float)w_row[k + 14]  * x_row[k + 14]
              + (float)w_row[k + 15]  * x_row[k + 15];

        int k2 = k + 128;
        sum1 += (float)w_row[k2]       * x_row[k2]
              + (float)w_row[k2 + 1]   * x_row[k2 + 1]
              + (float)w_row[k2 + 2]   * x_row[k2 + 2]
              + (float)w_row[k2 + 3]   * x_row[k2 + 3]
              + (float)w_row[k2 + 4]   * x_row[k2 + 4]
              + (float)w_row[k2 + 5]   * x_row[k2 + 5]
              + (float)w_row[k2 + 6]   * x_row[k2 + 6]
              + (float)w_row[k2 + 7]   * x_row[k2 + 7]
              + (float)w_row[k2 + 8]   * x_row[k2 + 8]
              + (float)w_row[k2 + 9]   * x_row[k2 + 9]
              + (float)w_row[k2 + 10]  * x_row[k2 + 10]
              + (float)w_row[k2 + 11]  * x_row[k2 + 11]
              + (float)w_row[k2 + 12]  * x_row[k2 + 12]
              + (float)w_row[k2 + 13]  * x_row[k2 + 13]
              + (float)w_row[k2 + 14]  * x_row[k2 + 14]
              + (float)w_row[k2 + 15]  * x_row[k2 + 15];

        int k3 = k + 256;
        sum2 += (float)w_row[k3]       * x_row[k3]
              + (float)w_row[k3 + 1]   * x_row[k3 + 1]
              + (float)w_row[k3 + 2]   * x_row[k3 + 2]
              + (float)w_row[k3 + 3]   * x_row[k3 + 3]
              + (float)w_row[k3 + 4]   * x_row[k3 + 4]
              + (float)w_row[k3 + 5]   * x_row[k3 + 5]
              + (float)w_row[k3 + 6]   * x_row[k3 + 6]
              + (float)w_row[k3 + 7]   * x_row[k3 + 7]
              + (float)w_row[k3 + 8]   * x_row[k3 + 8]
              + (float)w_row[k3 + 9]   * x_row[k3 + 9]
              + (float)w_row[k3 + 10]  * x_row[k3 + 10]
              + (float)w_row[k3 + 11]  * x_row[k3 + 11]
              + (float)w_row[k3 + 12]  * x_row[k3 + 12]
              + (float)w_row[k3 + 13]  * x_row[k3 + 13]
              + (float)w_row[k3 + 14]  * x_row[k3 + 14]
              + (float)w_row[k3 + 15]  * x_row[k3 + 15];

        int k4 = k + 384;
        sum3 += (float)w_row[k4]       * x_row[k4]
              + (float)w_row[k4 + 1]   * x_row[k4 + 1]
              + (float)w_row[k4 + 2]   * x_row[k4 + 2]
              + (float)w_row[k4 + 3]   * x_row[k4 + 3]
              + (float)w_row[k4 + 4]   * x_row[k4 + 4]
              + (float)w_row[k4 + 5]   * x_row[k4 + 5]
              + (float)w_row[k4 + 6]   * x_row[k4 + 6]
              + (float)w_row[k4 + 7]   * x_row[k4 + 7]
              + (float)w_row[k4 + 8]   * x_row[k4 + 8]
              + (float)w_row[k4 + 9]   * x_row[k4 + 9]
              + (float)w_row[k4 + 10]  * x_row[k4 + 10]
              + (float)w_row[k4 + 11]  * x_row[k4 + 11]
              + (float)w_row[k4 + 12]  * x_row[k4 + 12]
              + (float)w_row[k4 + 13]  * x_row[k4 + 13]
              + (float)w_row[k4 + 14]  * x_row[k4 + 14]
              + (float)w_row[k4 + 15]  * x_row[k4 + 15];
    }

    // Handle remaining with single accumulator
    for (; k < K; k += 128) {
        int end = k + 128;
        if (end > K) end = K;
        for (int j = k; j < end; j++) {
            sum0 += (float)w_row[j] * x_row[j];
        }
    }

    float sum = sum0 + sum1 + sum2 + sum3;

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (tid == 0)
        Y[(long long)m * N + n] = sum;
}
