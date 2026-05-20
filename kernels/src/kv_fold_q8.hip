// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// CASK m-folding merge kernel for Q8_0 KV cache.
//
// Folds `m` source rows into 1 destination row per output slot via
// weighted average + per-block requantization. Core singletons are
// handled uniformly: caller sets src_indices[slot × m + 0] to the
// source position and src_weights[slot × m + 0] = 1.0, rest = 0.
//
// Q8_0 layout: each row is [n_kv × n_blocks × 34 bytes]. Each 34-byte
// block = 2-byte f16 scale + 32 int8 values (so head_dim must be
// divisible by 32).
//
// Grid: (budget, n_kv, n_blocks).  Block: (32, 1, 1).
//   blockIdx.x: output slot in [0, budget)
//   blockIdx.y: KV head in [0, n_kv)
//   blockIdx.z: Q8 block within head in [0, n_blocks)
//   threadIdx.x: int8 lane in [0, 32)
//
// Per (slot, head, block):
//   1. Each lane accumulates Σ_i w_i · scale_i · int8_i in f32
//   2. Wave32 reduction of max |acc| for the new block scale
//   3. Quantize and write: lane 0 writes the f16 scale; all lanes write
//      their int8 value
#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void kv_fold_q8(
    const unsigned char* __restrict__ src,    // [physical × n_kv × n_blocks × 34]
    unsigned char* __restrict__ dst,           // [budget × n_kv × n_blocks × 34]
    const int* __restrict__ src_indices,       // [budget × m]
    const float* __restrict__ src_weights,     // [budget × m]
    int n_kv,
    int n_blocks,
    int m                                      // fold factor (per-slot)
) {
    const int out_slot = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int blk = blockIdx.z;
    const int lane = threadIdx.x;

    const int head_block_stride = n_blocks * 34;
    const int row_bytes = n_kv * head_block_stride;

    // 1. Weighted-average accumulate in f32.
    float acc = 0.f;
    for (int i = 0; i < m; ++i) {
        const float w = src_weights[out_slot * m + i];
        if (w == 0.f) continue;
        const int p = src_indices[out_slot * m + i];
        const unsigned char* sp = src
            + (size_t)p * row_bytes
            + (size_t)kv_head * head_block_stride
            + (size_t)blk * 34;
        // Load f16 block scale (bytes 0-1).
        const float scale = (float)(*(const _Float16*)sp);
        const int qv = (signed char)sp[2 + lane];
        acc += w * scale * (float)qv;
    }

    // 2. Warp-reduce max |acc| to get the new block scale.
    float my_abs = fabsf(acc);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        my_abs = fmaxf(my_abs, __shfl_xor(my_abs, off));
    }
    const float max_abs = my_abs;
    const float new_scale = max_abs / 127.f;
    const float inv_scale = new_scale > 1e-10f ? 1.f / new_scale : 0.f;

    // 3. Quantize and write.
    unsigned char* dp = dst
        + (size_t)out_slot * row_bytes
        + (size_t)kv_head * head_block_stride
        + (size_t)blk * 34;
    if (lane == 0) {
        *(_Float16*)dp = (_Float16)new_scale;
    }
    int qi = __float2int_rn(acc * inv_scale);
    qi = max(-127, min(127, qi));
    dp[2 + lane] = (unsigned char)(signed char)qi;
}
