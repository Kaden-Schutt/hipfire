// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// BF16-weight × F32-input GEMV: y[m] = W_bf16[m, k] @ x_f32[k].
//
// Native-bf16 reference path (the KLD/PPL oracle). The model ships in bf16;
// this keeps the EXACT downloaded weight values rather than re-quantizing to
// f16 (a different exponent/mantissa split). bf16 is the high 16 bits of an
// f32, so the widen is a lossless 16-bit left shift — no conversion error.
// F32 input precision is preserved (input is read as f32, accumulate in f32),
// matching gemv_f16_xf32 / gemv_q8_0 / gemv_f32.
//   grid = [M, 1, 1], block = [32, 1, 1]  (1 warp per output row)
__launch_bounds__(32)
extern "C" __global__ void gemv_bf16_xf32(
    const unsigned short* __restrict__ W,  // [M, K] bf16 (raw u16)
    const float*          __restrict__ x,  // [K] F32
    float*                __restrict__ y,  // [M] F32
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const unsigned short* w_row = W + (size_t)row * K;

    float sum = 0.0f;
    for (int i = tid; i < K; i += 32) {
        // bf16 → f32: widen to the high 16 bits of the f32 (lossless).
        float w = __uint_as_float(((unsigned int)w_row[i]) << 16);
        sum += w * x[i];
    }

    // Wave-shuffle reduction (wave32).
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
