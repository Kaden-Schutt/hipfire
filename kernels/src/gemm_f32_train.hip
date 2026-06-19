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
// REGISTER-TILED, NO SHARED MEMORY. Each thread computes a TM×TN (4×4) output
// micro-tile, holding accumulators in registers and reusing each loaded A/B value
// across the micro-tile — arithmetic intensity TM·TN/(TM+TN)=2 vs 0.5 for a
// one-output-per-thread kernel, so meaningfully faster while reading from global.
//
// WHY NO LDS: a shared-memory tiled variant is faster still (~288 GFLOP/s) but
// triggers a STICKY, UNRECOVERABLE intermittent HIP-719 on gfx1103 (Phoenix APU)
// — a per-launch fault (~few %) that accumulates to ~100% over many launches and
// cannot be cleared/retried. Isolated to LDS use (the no-LDS forms are 0/N). See
// gemm_f32_train_bench / gemm_f32_train_recover. Re-enable LDS only on hardware/
// drivers where it's reliable.

#include <hip/hip_runtime.h>

#define TM 4
#define TN 4
#define TPB 16            // 16x16 = 256 threads/block
#define BM (TPB * TM)     // 64 output rows per block
#define BN (TPB * TN)     // 64 output cols per block

__device__ __forceinline__ float lda_(
    const float* __restrict__ A, int m, int k, int M, int lda, int transA) {
    if (m >= M) return 0.0f;
    return transA ? A[(size_t)k * lda + m] : A[(size_t)m * lda + k];
}
__device__ __forceinline__ float ldb_(
    const float* __restrict__ B, int k, int n, int N, int ldb, int transB) {
    if (n >= N) return 0.0f;
    return transB ? B[(size_t)n * ldb + k] : B[(size_t)k * ldb + n];
}

extern "C" __launch_bounds__(TPB * TPB)
__global__ void gemm_f32_train(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb,
    int transA, int transB
) {
    int row0 = blockIdx.y * BM + threadIdx.y * TM;
    int col0 = blockIdx.x * BN + threadIdx.x * TN;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    for (int k = 0; k < K; k++) {
        float a[TM], b[TN];
        #pragma unroll
        for (int i = 0; i < TM; i++) a[i] = lda_(A, row0 + i, k, M, lda, transA);
        #pragma unroll
        for (int j = 0; j < TN; j++) b[j] = ldb_(B, k, col0 + j, N, ldb, transB);
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int m = row0 + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int n = col0 + j;
            if (n < N) C[(size_t)m * N + n] = acc[i][j];
        }
    }
}

// Accumulating variant: C = beta*C + op(A)·op(B).
extern "C" __launch_bounds__(TPB * TPB)
__global__ void gemm_f32_train_accum(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb,
    int transA, int transB,
    float beta
) {
    int row0 = blockIdx.y * BM + threadIdx.y * TM;
    int col0 = blockIdx.x * BN + threadIdx.x * TN;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    for (int k = 0; k < K; k++) {
        float a[TM], b[TN];
        #pragma unroll
        for (int i = 0; i < TM; i++) a[i] = lda_(A, row0 + i, k, M, lda, transA);
        #pragma unroll
        for (int j = 0; j < TN; j++) b[j] = ldb_(B, k, col0 + j, N, ldb, transB);
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int m = row0 + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int n = col0 + j;
            if (n < N) {
                size_t idx = (size_t)m * N + n;
                C[idx] = beta * C[idx] + acc[i][j];
            }
        }
    }
}
