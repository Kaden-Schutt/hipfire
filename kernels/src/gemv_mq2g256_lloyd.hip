// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// MagnumQuant 2-bit GEMV with per-block 4-entry fp16 Lloyd-Max codebook.
//
// Block layout (72 B/group, bandwidth-identical to uniform MQ2):
//   bytes [0..8)   : 4 × fp16 codebook entries (sorted ascending at quant-time)
//   bytes [8..72)  : 64 bytes packed 2-bit indices (4 per byte, little-endian within byte)
//
// Per-thread work: 8 weights / thread × 32 threads = 256 weights / block.
// Reconstruction is `cb[q]` via LDS-indexed lookup (branchless).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    __shared__ float cbt[4];

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;

        if (tid < 4) {
            cbt[tid] = __half2float(((const __half*)gptr)[tid]);
        }
        __syncthreads();

        const unsigned char* data = (const unsigned char*)(gptr + 8);

        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 2;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];

        acc += cbt[b0 & 3]        * x[base_idx]
             + cbt[(b0 >> 2) & 3] * x[base_idx + 1]
             + cbt[(b0 >> 4) & 3] * x[base_idx + 2]
             + cbt[b0 >> 6]       * x[base_idx + 3]
             + cbt[b1 & 3]        * x[base_idx + 4]
             + cbt[(b1 >> 2) & 3] * x[base_idx + 5]
             + cbt[(b1 >> 4) & 3] * x[base_idx + 6]
             + cbt[b1 >> 6]       * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
