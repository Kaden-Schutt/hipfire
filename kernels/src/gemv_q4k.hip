// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float half_to_float(unsigned short h) {
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp  = (h >> 10) & 0x1F;
    unsigned int frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) return __int_as_float(sign << 31);
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
        exp = 127 - 15 + 1 + exp;
        return __int_as_float((sign << 31) | (exp << 23) | (frac << 13));
    }
    if (exp == 31) return __int_as_float((sign << 31) | (0xFF << 23) | (frac ? (frac << 13 | 1) : 0));
    return __int_as_float((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13));
}

// Q4_K GEMV v6: 32 threads (single warp), 8 elements per thread per Q4K block.
// No shared memory needed — warp shuffle only. Higher occupancy.
// launch_bounds: 32 threads, target 20 blocks/CU (10 waves/SIMD) for max occupancy.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4k(
    const unsigned char* __restrict__ A_q4k,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A_q4k + (size_t)row * blocks_per_row * 144;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 144;
        float d = (float)*((const _Float16*)block);
        float dmin = (float)*((const _Float16*)(block + 2));
        const unsigned char* sc = block + 4;
        const float* xb = x + bi * 256;

        // Group 0: elements 0-63
        {
            unsigned char qbyte = block[16 + tid];
            float s0 = d * (float)(sc[0] & 63);
            float m0 = dmin * (float)(sc[4] & 63);
            sum += (s0 * (float)(qbyte & 0xF) - m0) * xb[tid];

            float s1 = d * (float)(sc[1] & 63);
            float m1 = dmin * (float)(sc[5] & 63);
            sum += (s1 * (float)(qbyte >> 4) - m1) * xb[32 + tid];
        }

        // Group 1: elements 64-127
        {
            unsigned char qbyte = block[48 + tid];
            float s2 = d * (float)(sc[2] & 63);
            float m2 = dmin * (float)(sc[6] & 63);
            sum += (s2 * (float)(qbyte & 0xF) - m2) * xb[64 + tid];

            float s3 = d * (float)(sc[3] & 63);
            float m3 = dmin * (float)(sc[7] & 63);
            sum += (s3 * (float)(qbyte >> 4) - m3) * xb[96 + tid];
        }

        // Group 2: elements 128-191 (different scale encoding)
        {
            unsigned char qbyte = block[80 + tid];
            float s4 = d * (float)((sc[8] & 0xF) | ((sc[0] >> 6) << 4));
            float m4 = dmin * (float)((sc[8] >> 4) | ((sc[4] >> 6) << 4));
            sum += (s4 * (float)(qbyte & 0xF) - m4) * xb[128 + tid];

            float s5 = d * (float)((sc[9] & 0xF) | ((sc[1] >> 6) << 4));
            float m5 = dmin * (float)((sc[9] >> 4) | ((sc[5] >> 6) << 4));
            sum += (s5 * (float)(qbyte >> 4) - m5) * xb[160 + tid];
        }

        // Group 3: elements 192-255
        {
            unsigned char qbyte = block[112 + tid];
            float s6 = d * (float)((sc[10] & 0xF) | ((sc[2] >> 6) << 4));
            float m6 = dmin * (float)((sc[10] >> 4) | ((sc[6] >> 6) << 4));
            sum += (s6 * (float)(qbyte & 0xF) - m6) * xb[192 + tid];

            float s7 = d * (float)((sc[11] & 0xF) | ((sc[3] >> 6) << 4));
            float m7 = dmin * (float)((sc[11] >> 4) | ((sc[7] >> 6) << 4));
            sum += (s7 * (float)(qbyte >> 4) - m7) * xb[224 + tid];
        }
    }

    // Warp shuffle reduction (5 steps, no shared memory)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
