// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Generic kernel library: WMMA GEMM, signed INT4 inputs -> INT32 output.
// gfx1103 (RDNA3 UMA) wave32, register-tiled, ZERO LDS. Generalises the
// gfx1151-only gemm_s4s4_wmma_tile probe into a portable RDNA3 kernel with
// clamp=false (no i32 saturation mid-accumulation). Packing:
//   A : [M, K/2] bytes, byte = k_even | (k_odd << 4)  (signed 4-bit two's-comp)
//   X : [B, K/2] bytes, same packing
//   Y : [B, M] int32 row-major
//
// Each lane loads its 16 int4 K-elements as one int32x2_t (8 bytes).
// acc[j] at lane t maps to (row=2*j+(t>>4), col=t&15). Requires K % 16 == 0.
// Unsigned int4 is a future variant (flip the signedness flags).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;

static __device__ __forceinline__ int32x2_t load_u4x16(const uint8_t* p) {
    const uint32_t lo = *(const uint32_t*)(p + 0);
    const uint32_t hi = *(const uint32_t*)(p + 4);
    return (int32x2_t){(int32_t)lo, (int32_t)hi};
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_iu4_i32_wmma(
    const uint8_t* __restrict__ A,   // [M, K/2] packed signed int4
    const uint8_t* __restrict__ X,   // [B, K/2] packed signed int4
    int32_t*       __restrict__ Y,   // [B, M] int32
    int M, int K, int B
) {
    const int tid  = threadIdx.x;
    const int lane = tid & 15;
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    if (row_start >= M || batch_start >= B) return;

    const int a_row = row_start + lane;
    const int x_row = batch_start + lane;
    const int safe_a = (a_row < M) ? a_row : (M - 1);
    const int safe_x = (x_row < B) ? x_row : (B - 1);
    const int k_tiles = K / 16;
    const int stride  = K / 2; // bytes per row

    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int kt = 0; kt < k_tiles; ++kt) {
        const uint8_t* ap = A + (long long)safe_a * stride + kt * 8;
        const uint8_t* xp = X + (long long)safe_x * stride + kt * 8;
        const int32x2_t a = load_u4x16(ap);
        const int32x2_t b = load_u4x16(xp);
        // signed A, signed B, clamp=false.
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, acc, false);
    }

    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
