// SPDX-License-Identifier: Apache-2.0
// hipfire — general fp32 GEMM for the training path (hipfire-train).
//
// The inference `gemm_f32_batched` is NT-only; the two backward matmuls contract
// a different axis, so the training crate uses ONE general GEMM with per-operand
// transpose flags. Computes  C[M,N] = op(A) · op(B)   (C row-major, C[m*N+n]):
//   op(A)[m,k] = transA ? A[k*lda + m] : A[m*lda + k]   (op(A) is [M,K])
//   op(B)[k,n] = transB ? B[n*ldb + k] : B[k*ldb + n]   (op(B) is [K,N])
//
// The three linear-layer matmuls (X:[M,K] tokens×in, W:[Nout,K]) map as:
//   forward  Y[M,Nout] = X·Wᵀ : gemm(X, W,  Y,  M, Nout, K,  K, K, transA=0, transB=1)
//   dX[M,K]            = dY·W  : gemm(dY,W,  dX, M, K,   Nout, Nout, K, 0, 0)
//   dW[Nout,K]         = dYᵀ·X : gemm(dY,X,  dW, Nout, K, M,   Nout, K, 1, 0)
//
// SHARED-MEMORY TILED (TILE×TILE, one output element per thread). Replaces the
// original one-wave-per-output-element kernel (~17 GFLOP/s, zero data reuse): an
// A/B tile is staged in shared memory and reused TILE× across the block, ~10-20×
// faster on the training shapes. Summation order differs from the old wave-shuffle
// form, so fp results differ in the last bits — fine, the training path validates
// against finite differences with atol+rtol, not bit-exactness.

#include <hip/hip_runtime.h>

#define TILE 16

__device__ __forceinline__ float load_a(
    const float* __restrict__ A, int m, int ak, int M, int K, int lda, int transA) {
    if (m >= M || ak >= K) return 0.0f;
    return transA ? A[(size_t)ak * lda + m] : A[(size_t)m * lda + ak];
}
__device__ __forceinline__ float load_b(
    const float* __restrict__ B, int bk, int n, int N, int K, int ldb, int transB) {
    if (bk >= K || n >= N) return 0.0f;
    return transB ? B[(size_t)n * ldb + bk] : B[(size_t)bk * ldb + n];
}

extern "C" __launch_bounds__(TILE * TILE, 4)
__global__ void gemm_f32_train(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb,
    int transA, int transB
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int ty = threadIdx.y, tx = threadIdx.x;
    int m = blockIdx.y * TILE + ty;
    int n = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int kt = 0; kt < K; kt += TILE) {
        As[ty][tx] = load_a(A, m, kt + tx, M, K, lda, transA);
        Bs[ty][tx] = load_b(B, kt + ty, n, N, K, ldb, transB);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; kk++) sum += As[ty][kk] * Bs[kk][tx];
        __syncthreads();
    }
    if (m < M && n < N) C[(size_t)m * N + n] = sum;
}

// Accumulating variant: C = beta*C + op(A)·op(B). Used where a gradient lands on
// a buffer that already holds a partial (residual-merged dX, LoRA dX onto base).
extern "C" __launch_bounds__(TILE * TILE, 4)
__global__ void gemm_f32_train_accum(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb,
    int transA, int transB,
    float beta
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int ty = threadIdx.y, tx = threadIdx.x;
    int m = blockIdx.y * TILE + ty;
    int n = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int kt = 0; kt < K; kt += TILE) {
        As[ty][tx] = load_a(A, m, kt + tx, M, K, lda, transA);
        Bs[ty][tx] = load_b(B, kt + ty, n, N, K, ldb, transB);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; kk++) sum += As[ty][kk] * Bs[kk][tx];
        __syncthreads();
    }
    if (m < M && n < N) {
        size_t idx = (size_t)m * N + n;
        C[idx] = beta * C[idx] + sum;
    }
}
