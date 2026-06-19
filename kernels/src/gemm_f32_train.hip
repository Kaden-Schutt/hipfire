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
// 16x16 2D BLOCK, one output element per thread, NO shared memory. ~6× the naive
// one-wave-per-output kernel (~15 → ~92 GFLOP/s on gfx1103): 256 independent
// threads per block beat 32-thread waves doing a shuffle-reduce per output.
//
// WHY NO LDS: a shared-memory tiled variant is ~3× faster again (~288 GFLOP/s) but
// triggers an intermittent HIP-719 "unspecified launch failure" on the FIRST
// launch on gfx1103 (Phoenix APU) — measured ~70% fault rate, halved by a
// post-module-load device sync, but never eliminated; the no-LDS form is 0/N.
// Isolated to LDS (the 2D block alone is fine). See gemm_f32_train_bench. Re-land
// the LDS version only with a register-tiled (no-LDS) reuse scheme or once the
// driver issue is resolved.

#include <hip/hip_runtime.h>

#define TILE 16

extern "C" __launch_bounds__(TILE * TILE)
__global__ void gemm_f32_train(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb,
    int transA, int transB
) {
    int m = blockIdx.y * TILE + threadIdx.y;
    int n = blockIdx.x * TILE + threadIdx.x;
    if (m >= M || n >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = transA ? A[(size_t)k * lda + m] : A[(size_t)m * lda + k];
        float b = transB ? B[(size_t)n * ldb + k] : B[(size_t)k * ldb + n];
        sum += a * b;
    }
    C[(size_t)m * N + n] = sum;
}

// Accumulating variant: C = beta*C + op(A)·op(B). Used where a gradient lands on
// a buffer that already holds a partial (residual-merged dX, LoRA dX onto base).
extern "C" __launch_bounds__(TILE * TILE)
__global__ void gemm_f32_train_accum(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb,
    int transA, int transB,
    float beta
) {
    int m = blockIdx.y * TILE + threadIdx.y;
    int n = blockIdx.x * TILE + threadIdx.x;
    if (m >= M || n >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = transA ? A[(size_t)k * lda + m] : A[(size_t)m * lda + k];
        float b = transB ? B[(size_t)n * ldb + k] : B[(size_t)k * ldb + n];
        sum += a * b;
    }
    size_t idx = (size_t)m * N + n;
    C[idx] = beta * C[idx] + sum;
}
