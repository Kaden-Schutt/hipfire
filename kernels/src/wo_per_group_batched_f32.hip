// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 per-group O-LoRA batched GEMV — F32 weights (Phase B2, 2026-05-18).
//
// Block-diagonal GEMV: wo_a in DeepSeek V4 has natural per-group structure
//   wo_a: [G, M, K]   row-major  (G=n_groups, M=o_lora_rank, K=per_group_in)
// Input x_in is per-(batch, group) chunks of the attention output:
//   x_in: [B, G, K]
// Output y_out: [B, G, M]
//
// Per (b, g, r): y[b, g, r] = sum_k wo_a[g, r, k] * x_in[b, g, k]
//
// Single launch in place of `B * G` separate gemv_f32 calls. Each
// (g, r, b) block does one K-dim dot product via warp-shuffle reduction.
//
// Grid:  [M, B, G]
// Block: [32, 1, 1]
//
// DeepSeek V4 config: G=8, M=o_lora_rank=1024, K=per_group_in=heads_per_group*
//             head_dim=8*512=4096, B≤max_batch.

#include <hip/hip_runtime.h>

extern "C" __global__ void wo_per_group_batched_f32(
    const float* __restrict__ wo_a,   // [G, M, K]
    const float* __restrict__ x_in,   // [B, G, K]
    float*       __restrict__ y_out,  // [B, G, M]
    int G,
    int M,
    int K,
    int batch_size
) {
    const int r = blockIdx.x;     // output row within group
    const int b = blockIdx.y;     // batch index
    const int g = blockIdx.z;     // group index
    if (r >= M || b >= batch_size || g >= G) return;
    const int tid = threadIdx.x;

    const float* w_row = wo_a + ((size_t)g * M + r) * K;
    const float* x_row = x_in + ((size_t)b * G + g) * K;

    float acc = 0.0f;
    for (int k = tid; k < K; k += 32) {
        acc += w_row[k] * x_row[k];
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (tid == 0) {
        y_out[((size_t)b * G + g) * M + r] = acc;
    }
}
