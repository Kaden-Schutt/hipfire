// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 per-group O-LoRA batched GEMV — Q8_0 weights (Phase D, 2026-05-21).
//
// Block-diagonal GEMV. wo_a in DeepSeek V4 has natural per-group structure:
//   wo_a: [G, M, K]   Q8_0-packed   (each [M, K] block is independent)
//   x:    [B, G, K]   plain F32     (Q8 contract: NO FWHT pre-rotation)
//   y:    [B, G, M]
//
// Per (b, g, r):
//   y[b, g, r] = Σ_k dequant(W[g, r, k]) · x[b, g, k]
//
// Single launch in place of `B × G` separate `gemv_q8_0` dispatches —
// collapses the per-position fallback loop in `attention_block_batched_*`
// (forward.rs:4209-4235 / 4752-4779) for the deepseek4-mq2lloyd-q8 build where
// wo_a is Q8_0.
//
// Q8_0 layout (per 32-element block):
//   bytes 0..1   = scale d (F16)
//   bytes 2..33  = 32 × signed int8 quantized weights
// Total: 34 bytes per block.
//
// Grid:  [M, B, G]
// Block: [32, 1, 1]
//
// Each thread handles one byte index within each Q8 block. K must be a
// multiple of 32. The 8× unrolled inner loop matches the per-position
// `gemv_q8_0` kernel for cache-warm reuse along the M axis.
//
// DeepSeek V4 config: G=8, M=o_lora_rank, K=per_group_in=(n_heads/G)*head_dim,
// B ≤ max_batch.

#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void wo_per_group_batched_q8_0(
    const unsigned char* __restrict__ wo_a,  // [G * M * K / 32 * 34] bytes
    const float* __restrict__ x_in,          // [B, G, K]
    float*       __restrict__ y_out,         // [B, G, M]
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

    const int blocks_per_row = K / 32;
    const size_t per_group_bytes = (size_t)M * blocks_per_row * 34;
    const size_t row_stride = (size_t)blocks_per_row * 34;
    const unsigned char* row_ptr =
        wo_a + (size_t)g * per_group_bytes + (size_t)r * row_stride;
    const float* x_row = x_in + ((size_t)b * G + g) * K;

    float sum = 0.0f;

    // 8 blocks (256 elements) per outer iteration.
    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_ptr + oi * 8 * 34;
        const float* xb = x_row + oi * 256;
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }

    // Tail (K not a multiple of 256).
    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_ptr + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x_row[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (tid == 0) {
        y_out[((size_t)b * G + g) * M + r] = sum;
    }
}
