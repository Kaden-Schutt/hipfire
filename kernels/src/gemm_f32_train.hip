// SPDX-License-Identifier: Apache-2.0
// hipfire — general fp32 GEMM for the training path (hipfire-train).
//
// Phase 0 (docs/plans/2026-06-17-hipfire-train-phase0.md): the inference
// `gemm_f32_batched` is NT-only (Y[n,m] = Σ_k A[m,k]·B[n,k]), which matches the
// forward linear `y = x·Wᵀ` but NOT the two backward matmuls, which contract a
// different axis. Rather than juggle transpose kernels, the training crate uses
// ONE general GEMM with per-operand transpose flags. Perf is irrelevant at
// Phase-0 scale (50M params, correctness-first); this is a deliberately simple,
// obviously-correct kernel — a wave32 reduction over the contraction dim, one
// block per output element. A tiled/WMMA version is a later optimization.
//
// Computes  C[M,N] = op(A) · op(B)   with C row-major (C[m*N + n]):
//   op(A)[m,k] = transA ? A[k*lda + m] : A[m*lda + k]   (op(A) is [M,K])
//   op(B)[k,n] = transB ? B[n*ldb + k] : B[k*ldb + n]   (op(B) is [K,N])
//
// The three linear-layer matmuls (X:[M,K] tokens×in, W:[Nout,K]) map as:
//   forward  Y[M,Nout] = X·Wᵀ : gemm(X, W,  Y,  M, Nout, K,  K, K, transA=0, transB=1)
//   dX[M,K]            = dY·W  : gemm(dY,W,  dX, M, K,   Nout, Nout, K, 0, 0)
//   dW[Nout,K]         = dYᵀ·X : gemm(dY,X,  dW, Nout, K, M,   Nout, K, 1, 0)

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void gemm_f32_train(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb,
    int transA, int transB
) {
    int m = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;
    if (m >= M || n >= N) return;

    float sum = 0.0f;
    for (int k = tid; k < K; k += 32) {
        float a = transA ? A[(size_t)k * lda + m] : A[(size_t)m * lda + k];
        float b = transB ? B[(size_t)n * ldb + k] : B[(size_t)k * ldb + n];
        sum += a * b;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (tid == 0)
        C[(size_t)m * N + n] = sum;
}

// Variant that accumulates into C (C += op(A)·op(B)) instead of overwriting.
// Needed where a gradient flows into a buffer that already holds a partial
// (e.g. residual-merged dX, or LoRA dX added onto the base-path dX). beta is
// applied to the existing C value: C = beta*C + op(A)·op(B).
extern "C" __launch_bounds__(32, 16)
__global__ void gemm_f32_train_accum(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb,
    int transA, int transB,
    float beta
) {
    int m = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;
    if (m >= M || n >= N) return;

    float sum = 0.0f;
    for (int k = tid; k < K; k += 32) {
        float a = transA ? A[(size_t)k * lda + m] : A[(size_t)m * lda + k];
        float b = transB ? B[(size_t)n * ldb + k] : B[(size_t)k * ldb + n];
        sum += a * b;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (tid == 0) {
        size_t idx = (size_t)m * N + n;
        C[idx] = beta * C[idx] + sum;
    }
}
