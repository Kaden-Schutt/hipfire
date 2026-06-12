// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// gfx1151 S4 x S4 -> I32 WMMA tile probe.
//
// This is a channel-test/prototype kernel for the Q4 activation-scratch path.
// It intentionally does not implement HFQ scale/zero corrections or route into
// model code. One wave32 computes a 16-row x 16-token output tile using
// v_wmma_i32_16x16x16_iu4. A and X are packed row-major/token-major signed
// 4-bit two's-complement values:
//   A: [M, K/2]     byte = k_even | (k_odd << 4)
//   X: [N, K/2]     byte = k_even | (k_odd << 4)
//   Y: [N, M] i32

#if defined(__gfx1151__)

typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;

static __device__ __forceinline__ int32x2_t load_u4x16(const uint8_t* p) {
    const uint32_t lo = *(const uint32_t*)(p + 0);
    const uint32_t hi = *(const uint32_t*)(p + 4);
    return (int32x2_t){(int32_t)lo, (int32_t)hi};
}

extern "C" __launch_bounds__(32, 8)
__global__ void gemm_s4s4_wmma_tile_gfx1151(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ X,
    int32_t* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int tid = threadIdx.x;
    const int lane = tid & 15;
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * 16;
    if (row_start >= M || col_start >= N) return;

    const int a_row = row_start + lane;
    const int x_col = col_start + lane;
    const int safe_a_row = (a_row < M) ? a_row : (M - 1);
    const int safe_x_col = (x_col < N) ? x_col : (N - 1);
    const int k_tiles = K / 16;
    const int a_stride = K / 2;
    const int x_stride = K / 2;

    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int kt = 0; kt < k_tiles; ++kt) {
        const uint8_t* ap = A + (long long)safe_a_row * a_stride + kt * 8;
        const uint8_t* xp = X + (long long)safe_x_col * x_stride + kt * 8;
        const int32x2_t a = load_u4x16(ap);
        const int32x2_t b = load_u4x16(xp);
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, acc, true);
    }

    if (x_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)x_col * M + out_row] = acc[j];
            }
        }
    }
}

#else

extern "C" __global__ void gemm_s4s4_wmma_tile_gfx1151(
    const uint8_t*, const uint8_t*, int32_t*, int, int, int
) {}

#endif
