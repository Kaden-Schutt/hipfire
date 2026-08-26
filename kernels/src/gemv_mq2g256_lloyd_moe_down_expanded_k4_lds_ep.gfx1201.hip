// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Exact-gfx1201 DS4 TP/EP deterministic down projection with cooperative
// codebook staging. This preserves the incumbent output and reduction order;
// only the four half codebook loads per group move from every lane to LDS.
// Grid: [M, K_TOP, batch], block [32].

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_expanded_k4_lds_gfx1201_ep(
    const unsigned long long* __restrict__ expert_ptrs,
    const unsigned long long* __restrict__ ownership_ptrs,
    const void* nonowned_dummy,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs,
    int M,
    int K,
    int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const long long output_index =
        (long long)bid * K_TOP * M + (long long)krank * M + row;
    if (reinterpret_cast<const void*>(ownership_ptrs[expert_id]) == nonowned_dummy) {
        if (tid == 0) {
            expert_outputs[output_index] = 0.0f;
        }
        return;
    }

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 2;

    __shared__ float cb_lds[16];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
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
                reinterpret_cast<const __half*>(gp0 + (long long)(tid >> 2) * 72);
            cb_lds[tid] = __half2float(cb_h[tid & 3]);
        }
        __syncthreads();

        const unsigned char* d0 = reinterpret_cast<const unsigned char*>(gp0 + 8) + boff;
        const unsigned char* d1 = reinterpret_cast<const unsigned char*>(gp1 + 8) + boff;
        const unsigned char* d2 = reinterpret_cast<const unsigned char*>(gp2 + 8) + boff;
        const unsigned char* d3 = reinterpret_cast<const unsigned char*>(gp3 + 8) + boff;
        const unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8);
        const unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8);
        const unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8);
        const unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8);
        const int base = g * 256 + tid * 8;

#define DOG2_LDS(pk, lds, b, a)                                      \
        (a) += cb_lds[(lds) + ((pk) & 3u)] * x[(b)]                   \
             + cb_lds[(lds) + (((pk) >> 2) & 3u)] * x[(b) + 1]       \
             + cb_lds[(lds) + (((pk) >> 4) & 3u)] * x[(b) + 2]       \
             + cb_lds[(lds) + (((pk) >> 6) & 3u)] * x[(b) + 3]       \
             + cb_lds[(lds) + (((pk) >> 8) & 3u)] * x[(b) + 4]       \
             + cb_lds[(lds) + (((pk) >> 10) & 3u)] * x[(b) + 5]     \
             + cb_lds[(lds) + (((pk) >> 12) & 3u)] * x[(b) + 6]      \
             + cb_lds[(lds) + (((pk) >> 14) & 3u)] * x[(b) + 7]

        DOG2_LDS(pk0, 0, base, acc0);
        DOG2_LDS(pk1, 4, base + 256, acc1);
        DOG2_LDS(pk2, 8, base + 512, acc2);
        DOG2_LDS(pk3, 12, base + 768, acc3);
#undef DOG2_LDS
    }

#define TAIL_LOAD_AND_DOT(g_expr, acc) do {                            \
        const int g = (g_expr);                                         \
        const char* gp = row_ptr + g * 72;                              \
        if (tid < 4) {                                                   \
            cb_lds[tid] = __half2float(                                 \
                reinterpret_cast<const __half*>(gp)[tid]);              \
        }                                                                \
        __syncthreads();                                                 \
        const unsigned char* d =                                        \
            reinterpret_cast<const unsigned char*>(gp + 8) + boff;      \
        const unsigned int pk =                                         \
            (unsigned int)d[0] | ((unsigned int)d[1] << 8);             \
        const int base = g * 256 + tid * 8;                             \
        (acc) += cb_lds[(pk) & 3u] * x[base]                            \
               + cb_lds[((pk) >> 2) & 3u] * x[base + 1]                \
               + cb_lds[((pk) >> 4) & 3u] * x[base + 2]                \
               + cb_lds[((pk) >> 6) & 3u] * x[base + 3]                \
               + cb_lds[((pk) >> 8) & 3u] * x[base + 4]                \
               + cb_lds[((pk) >> 10) & 3u] * x[base + 5]               \
               + cb_lds[((pk) >> 12) & 3u] * x[base + 6]               \
               + cb_lds[((pk) >> 14) & 3u] * x[base + 7];              \
    } while (0)

    if (tail >= 1) TAIL_LOAD_AND_DOT(quads << 2, acc0);
    if (tail >= 2) TAIL_LOAD_AND_DOT((quads << 2) + 1, acc1);
    if (tail >= 3) TAIL_LOAD_AND_DOT((quads << 2) + 2, acc2);
#undef TAIL_LOAD_AND_DOT

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    if (tid == 0) {
        expert_outputs[output_index] = acc;
    }
}
