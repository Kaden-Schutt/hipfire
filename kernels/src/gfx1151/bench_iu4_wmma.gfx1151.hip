// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// gfx1151 IU4/IU8 WMMA probe.
//
// One wave32 block issues a dependent chain of WMMA instructions using packed
// all-ones operands. Every C/D accumulator cell should be 16 * iters. This is
// intentionally synthetic: it proves instruction availability, operand vector
// widths, accumulator layout, and rough IU4-vs-IU8 issue throughput before any
// Q4 activation-scratch model path is attempted.

#if defined(__gfx1151__)

typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;

__device__ __forceinline__ uint32_t pack_iu4_ones() {
    return 0x11111111u;
}

__device__ __forceinline__ uint32_t pack_iu8_ones() {
    return 0x01010101u;
}

extern "C" __launch_bounds__(32, 8)
__global__ void bench_iu4_wmma_gfx1151(int* __restrict__ out, int iters) {
    const int lane = threadIdx.x;

    const int32x2_t a = {
        (int32_t)pack_iu4_ones(),
        (int32_t)pack_iu4_ones(),
    };
    const int32x2_t b = {
        (int32_t)pack_iu4_ones(),
        (int32_t)pack_iu4_ones(),
    };
    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < iters; ++i) {
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, acc, true);
    }

    int* dst = out + ((long long)blockIdx.x * 32 + lane) * 8;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        dst[r] = acc[r];
    }
}

extern "C" __launch_bounds__(32, 8)
__global__ void bench_iu8_wmma_gfx1151(int* __restrict__ out, int iters) {
    const int lane = threadIdx.x;

    const int32x4_t a = {
        (int32_t)pack_iu8_ones(),
        (int32_t)pack_iu8_ones(),
        (int32_t)pack_iu8_ones(),
        (int32_t)pack_iu8_ones(),
    };
    const int32x4_t b = {
        (int32_t)pack_iu8_ones(),
        (int32_t)pack_iu8_ones(),
        (int32_t)pack_iu8_ones(),
        (int32_t)pack_iu8_ones(),
    };
    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < iters; ++i) {
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc, true);
    }

    int* dst = out + ((long long)blockIdx.x * 32 + lane) * 8;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        dst[r] = acc[r];
    }
}

#else

extern "C" __global__ void bench_iu4_wmma_gfx1151(int* out, int) {
    if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = 0;
}

extern "C" __global__ void bench_iu8_wmma_gfx1151(int* out, int) {
    if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = 0;
}

#endif
