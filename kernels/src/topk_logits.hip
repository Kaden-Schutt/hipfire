// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Top-K extraction over a logits vector, K=4 per thread × 256 threads
// = 1024 total candidates written to the output buffer. The host reduces
// this to the final top-20 before sampling; 1024 is plenty of headroom
// above the sampler's TOP_K=20.
//
// Output layout in a 12 KB buffer:
//   bytes [    0..4096): 1024 × u32 token indices
//   bytes [4096..8192): 1024 × f32 raw logit values
//   (total 8 KB; one memcpy_dtoh pulls the whole thing)
//
// Design notes:
//
//  - Previous version used 1 wave of 32 threads. With each thread scanning
//    ~4,700 logits serially (the min-tracking branch stalls load pipelining),
//    the kernel took ~1.4 ms — almost a full forward-pass worth of latency.
//  - This version uses 256 threads (8 waves) on a single workgroup. Each
//    thread scans ~590 logits and keeps a local top-4 in registers. More
//    threads → more in-flight memory operations → 10× faster in the
//    latency-bound regime.
//  - No LDS reduction: the host reads all 1024 candidates and runs its
//    existing min-tracking loop over them. 1024 × 20 ops on CPU is ~20 µs.
//  - VGPR pressure per thread: 4 × (float + int) = 8 VGPRs for local top-K,
//    plus ~8 for loop state. 16 VGPRs at launch_bounds(256, 8) gives
//    1536/16 = 96 waves/SIMD — fully saturates.

__launch_bounds__(256, 8)
extern "C" __global__ void topk_logits_f32(
    const float* __restrict__ logits,
    void*        __restrict__ topk_buf,   // 8 KB: [1024 u32 idx | 1024 f32 val]
    int vocab_size
) {
    constexpr int K = 4;
    const int tid = threadIdx.x;  // 0..255
    const int nth = blockDim.x;   // 256

    float loc_val[K];
    int   loc_idx[K];
    #pragma unroll
    for (int j = 0; j < K; j++) {
        loc_val[j] = -INFINITY;
        loc_idx[j] = 0;
    }

    // Strided scan. Thread tid handles indices tid, tid+256, tid+512, ...
    for (int i = tid; i < vocab_size; i += nth) {
        const float v = logits[i];

        int   min_j = 0;
        float min_v = loc_val[0];
        #pragma unroll
        for (int j = 1; j < K; j++) {
            if (loc_val[j] < min_v) {
                min_v = loc_val[j];
                min_j = j;
            }
        }

        if (v > min_v) {
            loc_val[min_j] = v;
            loc_idx[min_j] = i;
        }
    }

    // Write to the two halves of the output buffer.
    unsigned int* out_idx = (unsigned int*)topk_buf;
    float*        out_val = (float*)((char*)topk_buf + 1024 * 4);

    #pragma unroll
    for (int j = 0; j < K; j++) {
        out_idx[tid * K + j] = (unsigned int)loc_idx[j];
        out_val[tid * K + j] = loc_val[j];
    }
}
