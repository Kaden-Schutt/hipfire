// SPDX-License-Identifier: Apache-2.0
// hipfire — reference kernel layer: per-token symmetric int8 activation quant.
//
// X[B,K] f32 → Xq[B,K] int8 (signed, levels [-127,127]) + Xs[B] f32 per-row scale.
// xq[b,k] = clamp(round(x[b,k] / s_b), -127, 127), s_b = max_k|x[b,k]| / 127.
// This is the A8 input to the iu8 WMMA W8A8/W4A8 path; dequant applies s_b · w_scale
// to the int32 accumulator AFTER the matmul (scales never enter the WMMA).
//
// One wave (32 lanes) per row, zero LDS (gfx1103-safe): strided per-lane amax then a
// __shfl_xor butterfly reduce, then strided quantize. Lane 0 writes the row scale.
//
// Grid: [B]. Block: [32].

#include <hip/hip_runtime.h>

__launch_bounds__(32)
extern "C" __global__ void quantize_act_int8_per_token(
    const float* __restrict__ X,   // [B, K] f32
    signed char* __restrict__ XQ,  // [B, K] int8
    float* __restrict__ XS,        // [B] f32 per-row scale
    int b_rows, int k
) {
    const int row = blockIdx.x;
    if (row >= b_rows) return;
    const int lane = threadIdx.x;
    const float* xr = X + (long long)row * k;

    // Per-lane strided amax, then butterfly all-reduce → every lane holds row amax.
    float amax = 0.0f;
    for (int c = lane; c < k; c += 32) {
        amax = fmaxf(amax, fabsf(xr[c]));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, off));
    }
    const float scale = fmaxf(amax / 127.0f, 1e-8f);
    const float inv = 1.0f / scale;

    signed char* xqr = XQ + (long long)row * k;
    for (int c = lane; c < k; c += 32) {
        float q = roundf(xr[c] * inv);
        q = fminf(fmaxf(q, -127.0f), 127.0f);
        xqr[c] = (signed char)q;
    }
    if (lane == 0) XS[row] = scale;
}
