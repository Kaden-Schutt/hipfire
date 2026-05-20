// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
extern "C" __global__ void argmax_f32(
    const float* __restrict__ data, int* __restrict__ result, int n
) {
    extern __shared__ float s[];
    int* si = (int*)(s + blockDim.x);
    float lmax = -1e30f; int lidx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (data[i] > lmax) { lmax = data[i]; lidx = i; }
    }
    s[threadIdx.x] = lmax; si[threadIdx.x] = lidx;
    __syncthreads();
    for (int sz = blockDim.x/2; sz > 0; sz >>= 1) {
        if (threadIdx.x < sz && s[threadIdx.x + sz] > s[threadIdx.x]) {
            s[threadIdx.x] = s[threadIdx.x + sz]; si[threadIdx.x] = si[threadIdx.x + sz];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) result[0] = si[0];
}
