// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Dequantize an HFQ4G256 matrix [M × K] into FP16 [M × K] row-major. One-shot
// utility kernel used when the downstream compute path (rocBLAS / hipBLASLt
// GEMM on CDNA3) wants the weight in straight FP16 rather than packed
// 4-bit. The output buffer is 4× the size of the MQ4 input, so callers only
// use this path on large-VRAM GPUs (MI300X 192 GB).
//
// HFQ4G256 group layout (136 bytes per group, K=256 elements):
//   bytes   0..3    scale (f32)
//   bytes   4..7    zero  (f32)
//   bytes   8..135  128 bytes of packed nibbles (256 × 4-bit values,
//                    2 per byte, low-nibble first)
//
// Launch: grid = (M, K/256, 1), block = (128, 1, 1). Each thread writes two
// output elements (one byte's worth of nibbles).
extern "C" __global__ void hfq4g256_dequantize_to_f16(
    const char*   __restrict__ W_mq4,
    _Float16*     __restrict__ W_fp16,
    int M, int K
) {
    const int row = blockIdx.x;
    const int grp = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= M) return;

    const int groups_per_row = K / 256;
    const char* row_ptr = W_mq4 + (long long)row * groups_per_row * 136;
    const char* gptr = row_ptr + grp * 136;

    float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
    float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));

    const unsigned char* nibbles = (const unsigned char*)(gptr + 8);
    unsigned char byte = nibbles[tid];

    float w0 = scale * (float)(byte & 0xFu) + zero;
    float w1 = scale * (float)(byte >> 4)   + zero;

    _Float16* out = W_fp16 + (long long)row * K + (long long)grp * 256 + tid * 2;
    out[0] = (_Float16)w0;
    out[1] = (_Float16)w1;
}
