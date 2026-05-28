// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 mHC α-scaling post-step — BATCHED variant (Phase B2, 2026-05-18).
//
// Batched twin of `hc_apply_alpha`. Per batch position b:
//   c_new[b, i] = α[seg(i)] · c[b, i] + (1 - α[seg(i)]) · base[i]
//
// α and base are shared across batch (model weights). c is per-batch
// `[B, 24]`. Same segment layout as the sequential kernel:
//   [0..4]   PRE   uses α[0]
//   [4..8]   POST  uses α[1]
//   [8..24]  COMB  uses α[2]
//
// Grid:  [batch_size, 1, 1]
// Block: [24, 1, 1]
//
// Byte-identical to hc_apply_alpha at batch_size == 1.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void hc_apply_alpha_batched(
    float*        __restrict__ c,        // [B, 24] in/out
    const __half* __restrict__ alpha,    // [3]
    const __half* __restrict__ base,     // [24]
    int batch_size
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int i = threadIdx.x;
    if (i >= 24) return;

    int seg;
    if      (i < 4) seg = 0;
    else if (i < 8) seg = 1;
    else            seg = 2;

    float a = __half2float(alpha[seg]);
    float bv = __half2float(base[i]);
    size_t idx = (size_t)b * 24 + i;
    float ci = c[idx];
    c[idx] = a * ci + (1.0f - a) * bv;
}
