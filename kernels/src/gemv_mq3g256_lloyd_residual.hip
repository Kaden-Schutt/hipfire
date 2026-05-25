// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// MagnumQuant 3-bit GEMV (Lloyd codebook) with fused residual add.
// y[row] += A[row] · x.
//
// Same body as gemv_mq3g256_lloyd.hip (8-way ternary lookup; baseline for
// non-RDNA3 archs). Only the final write differs (+= instead of =).
// Mirrors gemv_hfq3g256_residual.hip.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq3g256_lloyd_residual(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 112;
    const int boff = tid * 3;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 112;

        const __half* cb_h = reinterpret_cast<const __half*>(gptr);
        float cb0 = __half2float(cb_h[0]);
        float cb1 = __half2float(cb_h[1]);
        float cb2 = __half2float(cb_h[2]);
        float cb3 = __half2float(cb_h[3]);
        float cb4 = __half2float(cb_h[4]);
        float cb5 = __half2float(cb_h[5]);
        float cb6 = __half2float(cb_h[6]);
        float cb7 = __half2float(cb_h[7]);

        const unsigned char* data = (const unsigned char*)(gptr + 16);

        unsigned int pk = (unsigned int)data[boff]
                        | ((unsigned int)data[boff + 1] << 8)
                        | ((unsigned int)data[boff + 2] << 16);

        unsigned int q0 = pk & 7u;
        unsigned int q1 = (pk >> 3) & 7u;
        unsigned int q2 = (pk >> 6) & 7u;
        unsigned int q3 = (pk >> 9) & 7u;
        unsigned int q4 = (pk >> 12) & 7u;
        unsigned int q5 = (pk >> 15) & 7u;
        unsigned int q6 = (pk >> 18) & 7u;
        unsigned int q7 = (pk >> 21) & 7u;

        int base_idx = g * 256 + tid * 8;

        #define CB_LOOKUP3(q) (                                          \
            (q) == 0 ? cb0 : (q) == 1 ? cb1 : (q) == 2 ? cb2 : (q) == 3  \
            ? cb3 : (q) == 4 ? cb4 : (q) == 5 ? cb5 : (q) == 6 ? cb6     \
            : cb7                                                         \
        )

        acc += CB_LOOKUP3(q0) * x[base_idx]
             + CB_LOOKUP3(q1) * x[base_idx + 1]
             + CB_LOOKUP3(q2) * x[base_idx + 2]
             + CB_LOOKUP3(q3) * x[base_idx + 3]
             + CB_LOOKUP3(q4) * x[base_idx + 4]
             + CB_LOOKUP3(q5) * x[base_idx + 5]
             + CB_LOOKUP3(q6) * x[base_idx + 6]
             + CB_LOOKUP3(q7) * x[base_idx + 7];

        #undef CB_LOOKUP3
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += acc;   // fused residual add
}
