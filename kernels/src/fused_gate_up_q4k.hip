// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void fused_gate_up_q4k(
    const unsigned char* __restrict__ A_gate,
    const unsigned char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;

    const unsigned char* A;
    float* y;
    int local_row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; local_row = gid;
    } else {
        A = A_up; y = y_up; local_row = gid - gate_m;
    }

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A + (size_t)local_row * blocks_per_row * 144;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 144;
        float d = (float)*((const _Float16*)block);
        float dmin = (float)*((const _Float16*)(block + 2));
        const unsigned char* sc = block + 4;
        const float* xb = x + bi * 256;

        {
            unsigned char qbyte = block[16 + tid];
            sum += (d * (float)(sc[0] & 63) * (float)(qbyte & 0xF) - dmin * (float)(sc[4] & 63)) * xb[tid];
            sum += (d * (float)(sc[1] & 63) * (float)(qbyte >> 4) - dmin * (float)(sc[5] & 63)) * xb[32 + tid];
        }
        {
            unsigned char qbyte = block[48 + tid];
            sum += (d * (float)(sc[2] & 63) * (float)(qbyte & 0xF) - dmin * (float)(sc[6] & 63)) * xb[64 + tid];
            sum += (d * (float)(sc[3] & 63) * (float)(qbyte >> 4) - dmin * (float)(sc[7] & 63)) * xb[96 + tid];
        }
        {
            unsigned char qbyte = block[80 + tid];
            sum += (d * (float)((sc[8] & 0xF) | ((sc[0] >> 6) << 4)) * (float)(qbyte & 0xF) - dmin * (float)((sc[8] >> 4) | ((sc[4] >> 6) << 4))) * xb[128 + tid];
            sum += (d * (float)((sc[9] & 0xF) | ((sc[1] >> 6) << 4)) * (float)(qbyte >> 4) - dmin * (float)((sc[9] >> 4) | ((sc[5] >> 6) << 4))) * xb[160 + tid];
        }
        {
            unsigned char qbyte = block[112 + tid];
            sum += (d * (float)((sc[10] & 0xF) | ((sc[2] >> 6) << 4)) * (float)(qbyte & 0xF) - dmin * (float)((sc[10] >> 4) | ((sc[6] >> 6) << 4))) * xb[192 + tid];
            sum += (d * (float)((sc[11] & 0xF) | ((sc[3] >> 6) << 4)) * (float)(qbyte >> 4) - dmin * (float)((sc[11] >> 4) | ((sc[7] >> 6) << 4))) * xb[224 + tid];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[local_row] = sum;
}
