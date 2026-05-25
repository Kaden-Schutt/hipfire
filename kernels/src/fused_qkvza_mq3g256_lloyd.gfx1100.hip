// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1100 (RDNA3) Fused QKVZA MQ3-Lloyd: 4 GEMVs in one launch (saves
// 3 launch submits per LA layer vs separate gemv_mq3g256_lloyd calls).
// Mirrors fused_qkvza_hfq4g256.hip's routing (grid = qkv_m + z_m +
// beta_m + alpha_m, conditional A/y selection by gid range), with the
// K4 + LDS-resident codebook body from gemv_mq3g256_lloyd.gfx1100.hip.
//
// Why fuse for B=1 decode (per the 2026-05-06 profile findings):
//   1. Three launches eliminated × 24 LA layers per token: ~72 launches
//      per token of overhead avoided.
//   2. The 16-row w_beta and w_alpha launches are catastrophically
//      under-subscribed — they pay full launch fixed cost for ~28 KB
//      of weight data each (~9 µs/call vs ~1 µs of actual GPU work).
//      Co-scheduling with the 6144-row wqkv body lets the wave
//      scheduler keep real work in flight during those tiny bodies.
//   3. fused_gate_up_mq3g256_lloyd already validated this pattern at
//      57.6% peak BW (vs 40.8% for standalone gemv_mq3g256_lloyd) —
//      the per-launch BW amortization is real and measurable.
//
// Bit-exact with four sequential gemv_mq3g256_lloyd calls — the inner
// loop is a verbatim copy of gemv_mq3g256_lloyd.gfx1100.hip.

__launch_bounds__(32, 16)
extern "C" __global__ void fused_qkvza_mq3g256_lloyd(
    const char*  __restrict__ A_qkv,
    const char*  __restrict__ A_z,
    const char*  __restrict__ A_beta,
    const char*  __restrict__ A_alpha,
    const float* __restrict__ x,
    float*       __restrict__ y_qkv,
    float*       __restrict__ y_z,
    float*       __restrict__ y_beta,
    float*       __restrict__ y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K
) {
    const int gid = blockIdx.x;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    // Route this workgroup to its matrix + output row.
    const char* A;
    float* y;
    int row;
    if (gid < qkv_m) {
        A = A_qkv;   y = y_qkv;   row = gid;
    } else if (gid < qkv_m + z_m) {
        A = A_z;     y = y_z;     row = gid - qkv_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta;  y = y_beta;  row = gid - (qkv_m + z_m);
    } else {
        A = A_alpha; y = y_alpha; row = gid - (qkv_m + z_m + beta_m);
    }

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

    if (tid == 0) y[row] = acc;
}
