// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Round-trip an F32 array through bf16 in-place. Mimics PyTorch's
// `tensor.to(torch.bfloat16).to(torch.float32)` — keeps only the
// top 7 bits of the F32 mantissa (round-to-nearest-even).
//
// Used by the dots.ocr vision encoder to match HF's bf16 forward
// path. The dots.ocr model was loaded with `dtype=torch.bfloat16`,
// so EVERY residual-stream value HF computes is bf16-precision.
// Our F32 residual stream keeps higher precision; for some queries
// near a softmax saturation boundary, this extra precision flips
// the attention "winner" K, and the divergence compounds over the
// 42-block stack into a 20× norm explosion on a handful of patches.
//
// Layout: in-place, single 1-D buffer of `n` F32 elements.
// Grid:  ceil(n / 256), Block: 256.
extern "C" __global__ void bf16_round_trip_f32(
    float* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (!isfinite(v)) return;
    union { float f; unsigned int u; } box;
    box.f = v;
    unsigned int lsb = (box.u >> 16) & 1u;
    box.u = (box.u + 0x7FFFu + lsb) & 0xFFFF0000u;
    x[i] = box.f;
}
