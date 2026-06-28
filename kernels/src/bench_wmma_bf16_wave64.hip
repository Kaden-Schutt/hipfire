// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Wave64 sibling of bench_wmma_bf16_wave.hip. The `_w64` WMMA builtin requires
// the kernel be compiled in wavefront-size-64 mode (target features
// wavefrontsize64 + wmma-256b-insts) — RDNA3/3.5 is wave32-native, so this whole
// translation unit is built wave64 via the flag below. See
// bench_wmma_wave32_vs_wave64.rs.
//
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

#include <hip/hip_runtime.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(4)))  float4_t;

extern "C" __global__ void bench_wmma_bf16_w64(
    const short* __restrict__ A, // [16*16] bf16 payload
    const short* __restrict__ Bm,// [16*16] bf16 payload
    float* __restrict__ Y,       // [gridDim.x * blockDim.x * 4]
    int iters
) {
    const int tid = threadIdx.x;       // 0..63, one wave64
    const int lane = tid & 15;
    b16x16_t a = *(const b16x16_t*)(A + lane * 16);
    b16x16_t b = *(const b16x16_t*)(Bm + lane * 16);
    float4_t acc = {0, 0, 0, 0};
    for (int i = 0; i < iters; ++i) {
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64(a, b, acc);
    }
    const int base = (blockIdx.x * blockDim.x + tid) * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) Y[base + j] = acc[j];
}
