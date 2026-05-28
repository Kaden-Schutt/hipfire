// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Single-row F32 argmax that writes the next MTP token directly into a
// device-resident token chain. `vocab_map == nullptr` means the argmax index
// is already the full-vocab token id; otherwise it maps compressed draft-vocab
// index -> full-vocab token id.
extern "C" __global__ void argmax_token_chain_f32(
    const float* __restrict__ data,
    int* __restrict__ argmax_out,
    int* __restrict__ token_chain,
    const unsigned int* __restrict__ vocab_map,
    int n,
    int dst_slot,
    int use_vocab_map
) {
    extern __shared__ float s[];
    int* si = (int*)(s + blockDim.x);

    float lmax = -1e30f;
    int lidx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = data[i];
        if (v > lmax) {
            lmax = v;
            lidx = i;
        }
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

    if (threadIdx.x == 0) {
        int idx = si[0];
        argmax_out[0] = idx;
        token_chain[dst_slot] = use_vocab_map ? (int)vocab_map[idx] : idx;
    }
}
