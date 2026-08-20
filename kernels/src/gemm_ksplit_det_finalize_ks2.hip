// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
//
// K_SPLITS=2 variant of the deterministic K-split finalize.
// Sums 2 partials + residual into Y in fixed index order.
//
// K_SPLITS here MUST match K_SPLITS in
// gemm_hfq4g256_residual_wmma_ksplit_det_bt2_ks2.hip.
//

#include <hip/hip_runtime.h>

#define K_SPLITS 2

extern "C" __global__ void gemm_ksplit_det_finalize_ks2(
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
