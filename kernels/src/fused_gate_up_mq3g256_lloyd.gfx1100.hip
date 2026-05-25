// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1100 (RDNA3) Fused Gate+Up MQ3-Lloyd: two GEMVs in one launch (saves
// 1 launch per FFN layer vs separate gate/up calls). Grid = [gate_m + up_m]
// — each block processes one row from either A_gate or A_up based on its
// index. Per-row math is identical to gemv_mq3g256_lloyd.gfx1100.hip:
// K4 unroll + LDS-resident codebook + clamped tail. See that file for the
// full design rationale.
//
// Why fused (vs two separate launches): primarily symmetry with the MQ4
// `fused_gate_up_hfq4g256` family — pattern consistency for the future
// MQ4-Lloyd port. On graph-capture-enabled decode the wall-time savings
// from launch-count reduction are bounded (≤1% empirically; small
// kernels already pipeline-overlap the dominant GEMV).

__launch_bounds__(32, 16)
extern "C" __global__ void fused_gate_up_mq3g256_lloyd(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;

    // Pick which projection this block computes (gate or up).
    const char* A;
    float* y;
    int local_row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; local_row = gid;
    } else {
        A = A_up; y = y_up; local_row = gid - gate_m;
    }

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)local_row * groups_per_row * 112;
    const int boff = tid * 3;

    __shared__ float cb_lds[32];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // ----- main K4 loop -----
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 112;
        const char* gp1 = gp0 + 112;
        const char* gp2 = gp1 + 112;
        const char* gp3 = gp2 + 112;

        // Cooperative load: 32 threads × 1 fp16 → 32 fp32 entries in LDS.
        // thread tid → group (tid >> 3), entry (tid & 7).
        const __half* cb_h = (const __half*)(gp0 + (tid >> 3) * 112);
        cb_lds[tid] = __half2float(cb_h[tid & 7]);
        __syncthreads();

        // Load 4 packed-uint24 indices (one per group, this thread's 3 bytes).
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

    // ----- tail loop -----
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

    if (tid == 0) y[local_row] = acc;
}
