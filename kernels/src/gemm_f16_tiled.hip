// SPDX-License-Identifier: MIT
// Copyright (c) 2026 nickfinease
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Optimized F16xF32 GEMM (originally for the vision encoder).
// Computes: Y[M, N] = W_f16[M, K] @ X_f32[N, K]^T
//
// Same grid as naive gemm_f16: Grid=[M, N], Block=[32], LDS=0.
// One wave computes one output dot product Y[m, n] = W[m, :] · X[n, :].
//
// K partition (disjoint, exact): lane `tid` owns elements
// {tid + 32*j : tid + 32*j < K}. The main loop walks 8 of those per
// iteration (256 elements per wave) with 8 independent accumulators
// (8-way ILP); consecutive lanes read consecutive addresses, so every
// load is coalesced. The stride-32 tail loop finishes any K remainder
// (arbitrary K supported, including K % 32 != 0).
//
// History: the original version of this kernel started each lane at
// k = tid*4 but then read 16 consecutive elements per sub-block and
// stepped by 512, so lanes overlapped 4x inside each 128-chunk, and a
// per-lane contiguous tail loop re-summed the entire trailing K
// remainder once per lane (~31x multi-count) — output came back ~10x
// too large (f16_gemv_parity: l2 ratio 10.42 at m=256,k=4096,n=1).
// This rewrite keeps the launch contract and the performance shape
// (register accumulators, no LDS) but covers K exactly once.
//
// Shared memory tiling doesn't help here because:
//   - K is small (1152-4304), fits in L2 cache
//   - LDS allocation reduces occupancy too much
//   - The bottleneck is ILP, not memory bandwidth

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
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;

    // Main loop: 256 elements per wave iteration, lane-coalesced
    // stride-32 reads, 8 independent accumulators (8-way ILP).
    int k = tid;
    for (; k + 224 < K; k += 256) {
        sum0 += (float)w_row[k]       * x_row[k];
        sum1 += (float)w_row[k +  32] * x_row[k +  32];
        sum2 += (float)w_row[k +  64] * x_row[k +  64];
        sum3 += (float)w_row[k +  96] * x_row[k +  96];
        sum4 += (float)w_row[k + 128] * x_row[k + 128];
        sum5 += (float)w_row[k + 160] * x_row[k + 160];
        sum6 += (float)w_row[k + 192] * x_row[k + 192];
        sum7 += (float)w_row[k + 224] * x_row[k + 224];
    }

    // Tail: stride-32, single accumulator. Picks up exactly where the
    // main loop left this lane off, so coverage stays disjoint + exact.
    for (; k < K; k += 32) {
        sum0 += (float)w_row[k] * x_row[k];
    }

    float sum = ((sum0 + sum1) + (sum2 + sum3))
              + ((sum4 + sum5) + (sum6 + sum7));

    // Wave-shuffle reduction (wave32) — same idiom as gemm_f16 /
    // gemv_f16_xf32.
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (tid == 0)
        Y[(long long)m * N + n] = sum;
}
