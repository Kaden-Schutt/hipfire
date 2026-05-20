// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched argmax over F32 rows. Grid.x is the batch dim (one block per row);
// each block reduces `n` values via a per-block shared-memory tournament.
//
// Used in the DFlash spec-decode verify path to fold B argmaxes into a
// single launch that lands just B × 4 bytes on the host — avoiding the
// ~15 MB-per-verify D2H of full logits on the 4B Q8 lm_head.
extern "C" __global__ void argmax_f32_batched(
    const float* __restrict__ data,   // [batch, n] row-major
    int* __restrict__ result,          // [batch]
    int n
) {
    extern __shared__ float s[];
    int* si = (int*)(s + blockDim.x);
    const int batch_idx = blockIdx.x;
    const float* row = data + (size_t)batch_idx * n;

    float lmax = -1e30f;
    int lidx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
        if (v > lmax) { lmax = v; lidx = i; }
    }
    s[threadIdx.x] = lmax;
    si[threadIdx.x] = lidx;
    __syncthreads();

    for (int sz = blockDim.x / 2; sz > 0; sz >>= 1) {
        if (threadIdx.x < sz && s[threadIdx.x + sz] > s[threadIdx.x]) {
            s[threadIdx.x] = s[threadIdx.x + sz];
            si[threadIdx.x] = si[threadIdx.x + sz];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) result[batch_idx] = si[0];
}
