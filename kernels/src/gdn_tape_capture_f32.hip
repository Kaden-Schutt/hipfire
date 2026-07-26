// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Capture one Qwen3.5/3.6 GatedDeltaNet innovation row-set into its rollback
// tape. Retained Redline records kernel launches, not HIP DMA copies, so the
// verifier must express these three contiguous D2D copies as a dispatch.

#include <hip/hip_runtime.h>

extern "C" __global__ void gdn_tape_capture_f32(
    const float* __restrict__ qkv_src,
    const float* __restrict__ alpha_src,
    const float* __restrict__ beta_src,
    float* __restrict__ qkv_dst,
    float* __restrict__ alpha_dst,
    float* __restrict__ beta_dst,
    int qkv_count,
    int alpha_count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < qkv_count) {
        qkv_dst[i] = qkv_src[i];
    }
    if (i < alpha_count) {
        alpha_dst[i] = alpha_src[i];
        beta_dst[i] = beta_src[i];
    }
}
