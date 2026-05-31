// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE down GEMV with scaled residual for MQ2-Lloyd
// (2-bit + per-block 4-entry fp16 codebook, 72 B/group). Sibling of
// gemv_hfq4g256_moe_down_residual_scaled_k8_indexed — same expert-ptrs
// dispatch + atomicAdd residual, but the per-group decode is the 4-entry
// codebook lookup.
//
// `rot_batch[krank * K .. krank * K + K]` must be FWHT-pre-rotated for the
// down projection (each top-k expert has its own post-gate-up activation;
// each needs an independent rotation before this kernel sees it).
//
// atomicAdd into x_residual accumulates topk_weights[krank] * (W_down · rot_batch[krank])
// across all top-k experts and rows. One launch replaces 8 per-expert
// scaled-gemv-residual launches.
//
// 2026-05-18: K4-unrolled + LDS-resident codebook port from MQ3 gfx1100
// pattern. See gemv_mq2g256_lloyd_moe_gate_up_indexed.hip for design notes.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int*   __restrict__ topk_indices,              // [k_top]
    const float* __restrict__ topk_weights,              // [k_top]
    const float* __restrict__ rot_batch,                 // [k_top × K] FWHT-rotated
    float*       __restrict__ x_residual,                // [M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = rot_batch + (long long)krank * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 2;

    __shared__ float cb_lds[16];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;

        if (tid < 16) {
            const __half* cb_h =
                (const __half*)(gp0 + (long long)(tid >> 2) * 72);
            cb_lds[tid] = __half2float(cb_h[tid & 3]);
        }
        __syncthreads();

        const unsigned char* d0 = (const unsigned char*)(gp0 + 8) + boff;
        const unsigned char* d1 = (const unsigned char*)(gp1 + 8) + boff;
        const unsigned char* d2 = (const unsigned char*)(gp2 + 8) + boff;
        const unsigned char* d3 = (const unsigned char*)(gp3 + 8) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8);

        int base = g * 256 + tid * 8;

        #define DOG2_LDS(pk, lds, b, a)                              \
            (a) += cb_lds[(lds) + ( (pk)        & 3u)] * x[(b)]      \
                 + cb_lds[(lds) + (((pk) >>  2) & 3u)] * x[(b) + 1]  \
                 + cb_lds[(lds) + (((pk) >>  4) & 3u)] * x[(b) + 2]  \
                 + cb_lds[(lds) + (((pk) >>  6) & 3u)] * x[(b) + 3]  \
                 + cb_lds[(lds) + (((pk) >>  8) & 3u)] * x[(b) + 4]  \
                 + cb_lds[(lds) + (((pk) >> 10) & 3u)] * x[(b) + 5]  \
                 + cb_lds[(lds) + (((pk) >> 12) & 3u)] * x[(b) + 6]  \
                 + cb_lds[(lds) + (((pk) >> 14) & 3u)] * x[(b) + 7]

        DOG2_LDS(pk0,  0, base,        acc0);
        DOG2_LDS(pk1,  4, base + 256,  acc1);
        DOG2_LDS(pk2,  8, base + 512,  acc2);
        DOG2_LDS(pk3, 12, base + 768,  acc3);
        #undef DOG2_LDS
    }

    #define TAIL_LOAD_AND_DOT(g_expr, acc) do {                                 \
        const int g = (g_expr);                                                  \
        const char* gp = row_ptr + g * 72;                                       \
        if (tid < 4) {                                                           \
            cb_lds[tid] = __half2float(((const __half*)gp)[tid]);                \
        }                                                                        \
        __syncthreads();                                                         \
        const unsigned char* d = (const unsigned char*)(gp + 8) + boff;          \
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8);        \
        int base = g * 256 + tid * 8;                                            \
        (acc) += cb_lds[ (pk)        & 3u] * x[base]                             \
               + cb_lds[((pk) >>  2) & 3u] * x[base + 1]                         \
               + cb_lds[((pk) >>  4) & 3u] * x[base + 2]                         \
               + cb_lds[((pk) >>  6) & 3u] * x[base + 3]                         \
               + cb_lds[((pk) >>  8) & 3u] * x[base + 4]                         \
               + cb_lds[((pk) >> 10) & 3u] * x[base + 5]                         \
               + cb_lds[((pk) >> 12) & 3u] * x[base + 6]                         \
               + cb_lds[((pk) >> 14) & 3u] * x[base + 7];                        \
    } while (0)

    if (tail >= 1) TAIL_LOAD_AND_DOT( quads << 2,       acc0);
    if (tail >= 2) TAIL_LOAD_AND_DOT((quads << 2) + 1,  acc1);
    if (tail >= 3) TAIL_LOAD_AND_DOT((quads << 2) + 2,  acc2);
    #undef TAIL_LOAD_AND_DOT

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const float scale = topk_weights[krank];
        atomicAdd(&x_residual[row], scale * acc);
    }
}
