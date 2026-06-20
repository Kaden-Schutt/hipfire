// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Generic kernel library GEMV: signed INT4 weight × INT4 vector → INT32.
// y[m] = sum_k W_i4[m,k] * x_i4[k]. gfx1103 wave32, zero LDS, i32 accumulate.
// Packing: byte = k_even | (k_odd << 4), signed 4-bit two's-complement.
// W is [M, K/2] bytes, x is [K/2] bytes. grid=[M], block=[32]. Lanes stride
// over the K/2 bytes; each byte contributes two products.
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ int s4_lo(uint8_t b) {
    int v = (int)(b & 0xf);
    return v - ((v & 0x8) << 1); // sign-extend 4-bit
}
__device__ __forceinline__ int s4_hi(uint8_t b) {
    int v = (int)(b >> 4);
    return v - ((v & 0x8) << 1);
}

__launch_bounds__(32)
extern "C" __global__ void gemv_iu4_i32(
    const uint8_t* __restrict__ W,   // [M, K/2] packed signed int4
    const uint8_t* __restrict__ x,   // [K/2] packed signed int4
    int32_t*       __restrict__ y,   // [M] int32
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int kb = K / 2; // bytes per row
    const uint8_t* w_row = W + (size_t)row * kb;

    int sum = 0;
    for (int i = tid; i < kb; i += 32) {
        const uint8_t wb = w_row[i];
        const uint8_t xb = x[i];
        sum += s4_lo(wb) * s4_lo(xb);
        sum += s4_hi(wb) * s4_hi(xb);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
