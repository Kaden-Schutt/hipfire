// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// HIP-graphs-safe twin of deepseek4_topk_kv_gather_f32: reads `K` and
// `N_compressed` from device buffers instead of i32 kernargs. Launch
// the kernel with a FIXED grid sized to the maximum K (= topk_window
// = out_stride); blocks beyond `K_buf[0]` early-return.
//
// Body math is byte-identical to deepseek4_topk_kv_gather_f32 — only the
// kernarg layout changes.
extern "C" __global__ void deepseek4_topk_kv_gather_f32_buf(
    const float* __restrict__ kv_cache,            // [N_compressed, head_dim]
    const int*   __restrict__ topk_idx,            // [K]
    float*       __restrict__ out,                 // [n_kv=1, head_dim, out_stride]
    const int*   __restrict__ K_buf,               // [1]
    const int*   __restrict__ N_compressed_buf,    // [1]
    int head_dim,
    int out_stride,
    int col_offset,
    float scale
) {
    const int K = K_buf[0];
    const int N_compressed = N_compressed_buf[0];
    const int k = blockIdx.x;
    const int d = threadIdx.x;
    if (k >= K || d >= head_dim) return;

    const int idx = topk_idx[k];
    float val = 0.0f;
    if (idx >= 0 && idx < N_compressed) {
        val = kv_cache[idx * head_dim + d] * scale;
    }
    out[d * out_stride + col_offset + k] = val;
}
