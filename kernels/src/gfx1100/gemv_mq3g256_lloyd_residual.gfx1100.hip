// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1100 (RDNA3) MQ3G256-Lloyd GEMV with fused residual add: y[row] += A[row] · x.
// Same K4 unroll + LDS-resident codebook structure as
// gemv_mq3g256_lloyd.gfx1100.hip; only the final write differs (+=
// instead of =). Used by `weight_gemv_residual` MQ3-Lloyd arm to
// eliminate the alloc + gemv + add_inplace_f32 + free fallback chain
// (saves the 4.4% add_inplace_f32 cost from the 2026-05-06 decode profile).

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

    __shared__ float cb_lds[32];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 112;
        const char* gp1 = gp0 + 112;
        const char* gp2 = gp1 + 112;
        const char* gp3 = gp2 + 112;

        const __half* cb_h = (const __half*)(gp0 + (tid >> 3) * 112);
        cb_lds[tid] = __half2float(cb_h[tid & 7]);
        __syncthreads();

        const unsigned char* d0 = (const unsigned char*)(gp0 + 16) + boff;
        const unsigned char* d1 = (const unsigned char*)(gp1 + 16) + boff;
        const unsigned char* d2 = (const unsigned char*)(gp2 + 16) + boff;
        const unsigned char* d3 = (const unsigned char*)(gp3 + 16) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8) | ((unsigned int)d0[2] << 16);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8) | ((unsigned int)d1[2] << 16);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8) | ((unsigned int)d2[2] << 16);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8) | ((unsigned int)d3[2] << 16);

        int base = g * 256 + tid * 8;

        #define DOG3_LDS(pk, lds, b, a) \
            (a) += cb_lds[(lds) + ((pk) & 7u)]        * x[(b)]     \
                 + cb_lds[(lds) + (((pk) >> 3)  & 7u)] * x[(b) + 1] \
                 + cb_lds[(lds) + (((pk) >> 6)  & 7u)] * x[(b) + 2] \
                 + cb_lds[(lds) + (((pk) >> 9)  & 7u)] * x[(b) + 3] \
                 + cb_lds[(lds) + (((pk) >> 12) & 7u)] * x[(b) + 4] \
                 + cb_lds[(lds) + (((pk) >> 15) & 7u)] * x[(b) + 5] \
                 + cb_lds[(lds) + (((pk) >> 18) & 7u)] * x[(b) + 6] \
                 + cb_lds[(lds) + (((pk) >> 21) & 7u)] * x[(b) + 7]

        DOG3_LDS(pk0,  0, base,        acc0);
        DOG3_LDS(pk1,  8, base + 256,  acc1);
        DOG3_LDS(pk2, 16, base + 512,  acc2);
        DOG3_LDS(pk3, 24, base + 768,  acc3);
        #undef DOG3_LDS
    }

    #define TAIL_LOAD_AND_DOT(g_expr, acc) do {                                 \
        const int g = (g_expr);                                                  \
        const char* gp = row_ptr + g * 112;                                      \
        if (tid < 8) {                                                           \
            cb_lds[tid] = __half2float(((const __half*)gp)[tid]);                \
        }                                                                        \
        __syncthreads();                                                         \
        const unsigned char* d = (const unsigned char*)(gp + 16) + boff;         \
        unsigned int pk = (unsigned int)d[0]                                     \
                        | ((unsigned int)d[1] << 8)                              \
                        | ((unsigned int)d[2] << 16);                            \
        int base = g * 256 + tid * 8;                                            \
        (acc) += cb_lds[ (pk)        & 7u] * x[base]                             \
               + cb_lds[((pk) >> 3)  & 7u] * x[base + 1]                         \
               + cb_lds[((pk) >> 6)  & 7u] * x[base + 2]                         \
               + cb_lds[((pk) >> 9)  & 7u] * x[base + 3]                         \
               + cb_lds[((pk) >> 12) & 7u] * x[base + 4]                         \
               + cb_lds[((pk) >> 15) & 7u] * x[base + 5]                         \
               + cb_lds[((pk) >> 18) & 7u] * x[base + 6]                         \
               + cb_lds[((pk) >> 21) & 7u] * x[base + 7];                        \
    } while (0)

    if (tail >= 1) TAIL_LOAD_AND_DOT( quads << 2,       acc0);
    if (tail >= 2) TAIL_LOAD_AND_DOT((quads << 2) + 1,  acc1);
    if (tail >= 3) TAIL_LOAD_AND_DOT((quads << 2) + 2,  acc2);
    #undef TAIL_LOAD_AND_DOT

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += acc;   // fused residual add
}
