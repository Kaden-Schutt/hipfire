// SPDX-License-Identifier: Apache-2.0
// hipfire — calibration activation reductions (Tier-1 native Hessian/imatrix).
//
// Two accumulate-in-place reductions over a captured activation block
// x[N, K] (row-major, F32; N = tokens this call, K = linear input dim):
//
//   calib_sumsq_reduce_f32      acc[c]   += Σ_n x[n,c]²          (imatrix / diag(H))
//   calib_hessian_outer_f32     H[i,j]   += Σ_n x[n,i]·x[n,j]    (GPTQ Hessian, K×K)
//
// Both ADD into their destination so the caller accumulates across the whole
// calibration corpus (one call per captured linear per forward). Destinations
// are caller-zeroed once at start.

#include <hip/hip_runtime.h>

// acc[c] += Σ_n x[n,c]².  One thread per column; the inner loop walks rows.
// Threads in a warp cover consecutive columns → coalesced reads of x[n, c..c+31].
extern "C" __global__ void calib_sumsq_reduce_f32(
    const float* __restrict__ x,
    float* __restrict__ acc,
    int N, int K
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    float s = 0.0f;
    for (int n = 0; n < N; n++) {
        float v = x[(long long)n * K + c];
        s += v * v;
    }
    acc[c] += s;
}

// H[i,j] += Σ_n x[n,i]·x[n,j].  Tiled GEMM of A=xᵀ (Aᵢₙ = x[n,i]) by B=x
// (Bₙⱼ = x[n,j]) → C[i,j]. TILE×TILE block computes one output tile, streaming
// the contraction dim N in TILE-row chunks through shared memory.
#define CALIB_TILE 16
extern "C" __global__ void calib_hessian_outer_f32(
    const float* __restrict__ x,
    float* __restrict__ H,
    int N, int K
) {
    __shared__ float As[CALIB_TILE][CALIB_TILE]; // As[t][i] = x[n0+t, i_tile+i]
    __shared__ float Bs[CALIB_TILE][CALIB_TILE]; // Bs[t][j] = x[n0+t, j_tile+j]

    int i = blockIdx.y * CALIB_TILE + threadIdx.y; // output row (channel i)
    int j = blockIdx.x * CALIB_TILE + threadIdx.x; // output col (channel j)

    float acc = 0.0f;
    for (int n0 = 0; n0 < N; n0 += CALIB_TILE) {
        // Cooperative load of the [CALIB_TILE rows × CALIB_TILE cols] tiles.
        int load_i = blockIdx.y * CALIB_TILE + threadIdx.x; // col index for As
        int load_j = blockIdx.x * CALIB_TILE + threadIdx.x; // col index for Bs
        int rA = n0 + threadIdx.y;
        int rB = n0 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (rA < N && load_i < K) ? x[(long long)rA * K + load_i] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (rB < N && load_j < K) ? x[(long long)rB * K + load_j] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < CALIB_TILE; t++) {
            // As[t][threadIdx.y] = x[n0+t, i] ; Bs[t][threadIdx.x] = x[n0+t, j]
            acc += As[t][threadIdx.y] * Bs[t][threadIdx.x];
        }
        __syncthreads();
    }
    if (i < K && j < K) {
        H[(long long)i * K + j] += acc;
    }
}

// acc[c] += Σ_n w[n]·x[n,c]².  Per-row-weighted column sum-of-squares = the
// diagonal of the weighted Hessian (diag(H̄)). One thread per column.
extern "C" __global__ void calib_sumsq_weighted_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ acc,
    int N, int K
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    float s = 0.0f;
    for (int n = 0; n < N; n++) {
        float v = x[(long long)n * K + c];
        s += w[n] * v * v;
    }
    acc[c] += s;
}

// w[n] += (1/K) Σ_c d[n,c]².  One thread per ROW (token); the inner loop walks
// the K columns. Used by GuidedQuant calibration to turn a linear's output-grad
// block d[N,K] (∂ℓ/∂z) into the per-token Fisher weight w[n] that reweights the
// Hessian. ADDS into w (caller-zeroed) so it can accumulate if needed; for the
// per-batch use the caller passes a fresh w.
extern "C" __global__ void calib_row_meansq_f32(
    const float* __restrict__ d,
    float* __restrict__ w,
    int N, int K
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float s = 0.0f;
    for (int c = 0; c < K; c++) {
        float v = d[(long long)n * K + c];
        s += v * v;
    }
    w[n] += s / (float)K;
}

// H[i,j] += Σ_n w[n]·x[n,i]·x[n,j].  The per-row-weighted Hessian (GuidedQuant /
// empirical-Fisher): identical tiling to calib_hessian_outer_f32, but each
// rank-1 contribution x[n,:]x[n,:]ᵀ is scaled by the token weight w[n]. With
// w≡1 this reduces exactly to the unweighted outer product.
extern "C" __global__ void calib_hessian_outer_weighted_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ H,
    int N, int K
) {
    __shared__ float As[CALIB_TILE][CALIB_TILE];
    __shared__ float Bs[CALIB_TILE][CALIB_TILE];

    int i = blockIdx.y * CALIB_TILE + threadIdx.y;
    int j = blockIdx.x * CALIB_TILE + threadIdx.x;

    float acc = 0.0f;
    for (int n0 = 0; n0 < N; n0 += CALIB_TILE) {
        int load_i = blockIdx.y * CALIB_TILE + threadIdx.x;
        int load_j = blockIdx.x * CALIB_TILE + threadIdx.x;
        int rA = n0 + threadIdx.y;
        int rB = n0 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (rA < N && load_i < K) ? x[(long long)rA * K + load_i] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (rB < N && load_j < K) ? x[(long long)rB * K + load_j] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < CALIB_TILE; t++) {
            int row = n0 + t;
            float wt = (row < N) ? w[row] : 0.0f;
            acc += wt * As[t][threadIdx.y] * Bs[t][threadIdx.x];
        }
        __syncthreads();
    }
    if (i < K && j < K) {
        H[(long long)i * K + j] += acc;
    }
}
