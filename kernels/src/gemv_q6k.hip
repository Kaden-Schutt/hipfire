// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float half_to_float_q6(unsigned short h) {
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

extern "C" __global__ void gemv_q6k(
    const unsigned char* __restrict__ A_q6k,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    if (row >= M) return;

    const int blocks_per_row = K / 256;
    const int bytes_per_block = 210;
    const unsigned char* row_data = A_q6k + (size_t)row * blocks_per_row * bytes_per_block;

    float sum = 0.0f;

    for (int elem = threadIdx.x; elem < K; elem += blockDim.x) {
        int block_idx = elem / 256;
        int within_block = elem % 256;

        const unsigned char* block = row_data + block_idx * bytes_per_block;
        const unsigned char* ql = block;           // 128 bytes
        const unsigned char* qh = block + 128;     // 64 bytes
        const signed char* sc = (const signed char*)(block + 192); // 16 bytes
        unsigned short d_bits = block[208] | (block[209] << 8);
        float d = half_to_float_q6(d_bits);

        // Which group (0 or 1) of 128 elements
        int group = within_block / 128;
        int in_group = within_block % 128;

        // Advance pointers for group
        const unsigned char* g_ql = ql + group * 64;
        const unsigned char* g_qh = qh + group * 32;
        const signed char* g_sc = sc + group * 8;

        // Which of 4 sub-positions within group
        int sub = in_group / 32;  // 0,1,2,3
        int l = in_group % 32;
        int is = l / 16;  // scale sub-index

        int q;
        switch (sub) {
            case 0: q = ((g_ql[l] & 0xF) | (((g_qh[l] >> 0) & 3) << 4)) - 32; break;
            case 1: q = ((g_ql[l + 32] & 0xF) | (((g_qh[l] >> 2) & 3) << 4)) - 32; break;
            case 2: q = ((g_ql[l] >> 4) | (((g_qh[l] >> 4) & 3) << 4)) - 32; break;
            case 3: q = ((g_ql[l + 32] >> 4) | (((g_qh[l] >> 6) & 3) << 4)) - 32; break;
        }

        float scale = (float)g_sc[is + sub * 2];
        float w = d * scale * (float)q;
        sum += w * x[elem];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = sdata[0];
    }
}
