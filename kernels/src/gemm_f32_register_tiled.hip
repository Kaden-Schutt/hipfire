// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Register-tiled F32 batched GEMM. Mirrors gemm_hfq4g256.hip's pattern:
// each block holds `BATCH_TILE` accumulators in registers and reuses
// each loaded weight tile across all BATCH_TILE batch positions —
// amortizing weight bandwidth which is the prefill bottleneck on
// unified-memory Strix Halo.
//
// Replaces gemm_f32_batched (in gemm_f32.hip), which puts batch in
// blockIdx.y and reloads the weight row N times. For DeepSeek V4's
// F16→F32 attention projections (Q-LoRA, KV-joint, wo_b, compressor
// wkv/wgate, indexer wq_b/weights_proj, gate.weight), this is the
// dominant cost in batched prefill.
//
// Layout:
//   A : [M, K]                   row-major F32 weights
//   x : [batch_size, K]          row-major F32 inputs
//   y : [batch_size, M]          row-major F32 outputs
//
// Grid:  [M, (batch_size + BATCH_TILE - 1) / BATCH_TILE, 1]
// Block: [32, 1, 1]    — one wave per (row, batch_tile) pair
//
// At batch_size == 1 (single tile, local_bs==1) produces the same
// output as gemm_f32 / gemv_f32. Math: FMA order matches a
// per-thread strided sum followed by warp-shuffle reduction —
// identical to gemm_f32_batched modulo accumulator placement.

#include <hip/hip_runtime.h>

#define BATCH_TILE 8

extern "C" __launch_bounds__(32, 16)
__global__ void gemm_f32_register_tiled(
    const float* __restrict__ A,    // [M, K]
    const float* __restrict__ x,    // [batch_size, K]
    float* __restrict__ y,          // [batch_size, M]
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;

    // How many batch elements this block handles (the last tile may
    // be partial when batch_size % BATCH_TILE != 0).
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const float* a_row = A + (size_t)row * K;

    // Per-batch accumulator held in registers — the whole point of
    // register tiling is that A is read once and amortized into
    // BATCH_TILE accumulators.
    float acc[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    // Stride 32: thread t handles K indices {t, t+32, t+64, …}. Each
    // thread loads one weight element and multiplies it against
    // BATCH_TILE x rows in registers — this is where the amortization
    // happens.
    for (int k = tid; k < K; k += 32) {
        const float w = a_row[k];
        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b < local_bs) {
                acc[b] += w * x[(size_t)(batch_start + b) * K + k];
            }
        }
    }

    // Wave-shuffle reduce each batch's accumulator, then write.
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) continue;
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down(val, offset);
        }
        if (tid == 0) {
            y[(size_t)(batch_start + b) * M + row] = val;
        }
    }
}
