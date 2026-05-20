// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_q4k(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    int token_id, int dim
) {
    const int tid = threadIdx.x;
    const int blocks_per_row = dim / 256;
    const int bytes_per_row = blocks_per_row * 144;
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int block_idx = elem / 256;
        int within = elem % 256;
        const unsigned char* block = row + block_idx * 144;

        float d = (float)*((const _Float16*)block);
        float dmin = (float)*((const _Float16*)(block + 2));
        const unsigned char* sc = block + 4;

        int group = within / 64;
        int in_group = within % 64;
        int sub = in_group / 32;
        int l = in_group % 32;
        int sb_idx = group * 2 + sub;

        float scale, mn;
        if (sb_idx < 4) {
            scale = d * (float)(sc[sb_idx] & 63);
            mn = dmin * (float)(sc[4 + sb_idx] & 63);
        } else {
            int i = sb_idx - 4;
            scale = d * (float)((sc[8 + i] & 0xF) | ((sc[i] >> 6) << 4));
            mn = dmin * (float)((sc[8 + i] >> 4) | ((sc[4 + i] >> 6) << 4));
        }

        unsigned char qbyte = block[16 + group * 32 + l];
        float nibble = (sub == 0) ? (float)(qbyte & 0xF) : (float)(qbyte >> 4);
        output[elem] = scale * nibble - mn;
    }
}
