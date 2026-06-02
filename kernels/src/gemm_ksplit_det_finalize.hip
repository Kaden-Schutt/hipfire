// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// DETERMINISTIC K-split, phase 2 (finalize): sum the K_SPLITS partials
// produced by gemm_hfq4g256_residual_wmma_ksplit_det into Y, in FIXED
// index order (z = 0,1,2,3), on top of the residual already in Y.
//
//   Y[idx] = Y[idx](residual) + sum_z partials[z*batch_size*M + idx]
//
// The z-stride is batch_size*M — the SAME literal expression the partial
// kernel uses for split_base (ksplit*batch_size*M), so the layout contract
// is self-documenting and cannot be broken by passing a wrong cell count.
// Fixed loop order → bit-reproducible (unlike the race-ordered atomicAdd in
// the non-det ksplit). partials is [K_SPLITS][batch_size][M] contiguous.
//
// K_SPLITS here MUST match K_SPLITS in
// gemm_hfq4g256_residual_wmma_ksplit_det.hip.
//
// Grid: [ceil(batch_size*M/256)]. Block: [256]. LDS: 0.

#define K_SPLITS 4

extern "C" __global__ void gemm_ksplit_det_finalize(
    float* __restrict__ Y,
    const float* __restrict__ partials,
    int batch_size, int M
) {
    const long long N = (long long)batch_size * M;
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= N) return;
    float s = Y[idx];
    #pragma unroll
    for (int z = 0; z < K_SPLITS; z++) {
        s += partials[(long long)z * N + idx];
    }
    Y[idx] = s;
}
