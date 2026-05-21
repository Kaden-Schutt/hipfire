// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// HFQ4-G128 GEMV with fused per-token SIGMOID-SCALED residual add, batched
// over N tokens.
//
// y_batch[token, row] += sigmoid(c_batch[token]) * (A[row] · x_batch[token])
//
// Layout:
//   A:        [M × groups_per_row × 72 bytes]  (HFQ4G128: 8 B header + 64 B
//                                               packed nibbles per 128-elem group)
//   x_batch:  [N × K]   F32
//   y_batch:  [N × M]   F32 (residual accumulator, += in place)
//   c_batch:  [N]       F32 (per-token logits; kernel applies sigmoid)
//
// Grid:  (M, N, 1)        — one block per (row, token)
// Block: (32, 1, 1)       — one warp per output cell
//
// Per HFQ4G128 layout (see kernels/src/gemv_hfq4g128.hip):
//   - 128 elements per group
//   - 8-byte header: scale F32 + zero F32
//   - 64-byte payload: 64 packed 4-bit nibbles
//   - 32 threads × 4 elements per thread per group = 128 ✓
//
// Used by prefill_moe_ffn_body_batched's shared-expert down projection on
// PARO checkpoints (DType::ParoQ4G128). The HFQ4-G256 sister kernel is
// gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched (kernels/src/
// gemv_hfq4g256_residual_scaled.hip). Same epilogue, different group stride.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched(
    const char* __restrict__ A,
    const float* __restrict__ x_batch,    // [N × K]
    float* __restrict__ y_batch,          // [N × M]
    const float* __restrict__ c_batch,    // [N]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int bid = blockIdx.y;
    const int tid = threadIdx.x;

    const float* __restrict__ x = x_batch + (long long)bid * K;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        // 128 weights / 32 threads = 4 weights per thread
        int base_idx = g * 128 + tid * 4;
        int byte_off = tid * 2;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];

        float v0 = scale * (float)(b0 & 0xF) + zero;
        float v1 = scale * (float)(b0 >> 4)  + zero;
        float v2 = scale * (float)(b1 & 0xF) + zero;
        float v3 = scale * (float)(b1 >> 4)  + zero;

        acc += v0 * x[base_idx]     + v1 * x[base_idx + 1]
             + v2 * x[base_idx + 2] + v3 * x[base_idx + 3];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        float gate = 1.0f / (1.0f + __expf(-c_batch[bid]));
        y_batch[(long long)bid * M + row] += gate * acc;
    }
}
