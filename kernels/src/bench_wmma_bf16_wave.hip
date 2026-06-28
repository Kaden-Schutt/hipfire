// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Microbench: v_wmma_f32_16x16x16_bf16 in wave32 vs wave64 form. Each kernel is
// a pure matrix-core throughput loop — load one bf16 16x16 A/B fragment per lane
// from a tiny buffer, then chain `iters` WMMA accumulations into the acc fragment
// (the chain creates a real dependency so the loop cannot be elided; a large grid
// of independent waves hides the per-wave latency, leaving aggregate WGP
// throughput). acc is written out at the end to keep it live.
//
// One 16x16x16 WMMA produces the same one output tile (8192 FLOPs) in both modes;
// the difference being measured is how the WGP schedules a 32-lane vs a 64-lane
// wave for that tile (RDNA3 is wave32-native), plus the acc register footprint
// (wave32: 8 f32/lane, wave64: 4 f32/lane).

#include <hip/hip_runtime.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(8)))  float8_t;
typedef float __attribute__((ext_vector_type(4)))  float4_t;

extern "C" __global__ void bench_wmma_bf16_w32(
    const short* __restrict__ A, // [16*16] bf16 payload
    const short* __restrict__ Bm,// [16*16] bf16 payload
    float* __restrict__ Y,       // [gridDim.x * blockDim.x * 8]
    int iters
) {
    const int tid = threadIdx.x;       // 0..31, one wave32
    const int lane = tid & 15;
    b16x16_t a = *(const b16x16_t*)(A + lane * 16);
    b16x16_t b = *(const b16x16_t*)(Bm + lane * 16);
    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < iters; ++i) {
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    }
    const int base = (blockIdx.x * blockDim.x + tid) * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) Y[base + j] = acc[j];
}
