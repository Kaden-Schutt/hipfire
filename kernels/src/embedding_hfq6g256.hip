// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// HFQ6-G256 embedding lookup: dequantize one row to F32.
// Group layout (byte-identical to gemv_hfq6g256.hip): 200 B / 256 elems
//   [f32 scale][f32 zero][192 B packed 6-bit]
// Packing: 4 weights per 3 bytes (same unpack as gemv_hfq6g256).
extern "C" __global__ void embedding_hfq6g256(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    int token_id, int dim
) {
    const int tid = threadIdx.x;
    const int groups_per_row = dim / 256;
    const int bytes_per_row = groups_per_row * 200;
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int group = elem / 256;
        int within = elem % 256;
        const unsigned char* gptr = row + group * 200;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = gptr + 8;

        // 4 weights per 3 bytes — bit-exact with gemv_hfq6g256.hip
        int pack = within / 4;
        int slot = within % 4;
        const unsigned char* p = data + pack * 3;
        unsigned char b0 = p[0];
        unsigned char b1 = p[1];
        unsigned char b2 = p[2];
        int q;
        if (slot == 0)      q = b0 & 63;
        else if (slot == 1) q = (b0 >> 6) | ((b1 & 0xF) << 2);
        else if (slot == 2) q = (b1 >> 4) | ((b2 & 3) << 4);
        else                q = b2 >> 2;

        output[elem] = scale * (float)q + zero;
    }
}
