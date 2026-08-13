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

// One wave computes one row for two routed ranks before issuing a single
// atomic residual update. DeepSeek4 uses top-k=6, so this halves both the
// number of scheduled workgroups and contending atomics while retaining
// 3*M workgroups (ample to fill gfx1151). Weight traffic and per-rank dot
// arithmetic are unchanged.
__device__ __forceinline__ float mq2_down_dot_k4(
    const char* __restrict__ row_ptr,
    const float* __restrict__ x,
    int K,
    float* cb_lds
) {
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const int boff = tid * 2;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;

        if (tid < 16) {
            const __half* cb_h =
                reinterpret_cast<const __half*>(
                    gp0 + (long long)(tid >> 2) * 72);
            cb_lds[tid] = __half2float(cb_h[tid & 3]);
        }
        __syncthreads();

        const unsigned int pk0 =
            *reinterpret_cast<const unsigned short*>(gp0 + 8 + boff);
        const unsigned int pk1 =
            *reinterpret_cast<const unsigned short*>(gp1 + 8 + boff);
        const unsigned int pk2 =
            *reinterpret_cast<const unsigned short*>(gp2 + 8 + boff);
        const unsigned int pk3 =
            *reinterpret_cast<const unsigned short*>(gp3 + 8 + boff);
        const int base = g * 256 + tid * 8;

        #define MQ2_DOT8(pk, lds, b, a)                              \
            (a) += cb_lds[(lds) + ( (pk)        & 3u)] * x[(b)]      \
                 + cb_lds[(lds) + (((pk) >>  2) & 3u)] * x[(b) + 1]  \
                 + cb_lds[(lds) + (((pk) >>  4) & 3u)] * x[(b) + 2]  \
                 + cb_lds[(lds) + (((pk) >>  6) & 3u)] * x[(b) + 3]  \
                 + cb_lds[(lds) + (((pk) >>  8) & 3u)] * x[(b) + 4]  \
                 + cb_lds[(lds) + (((pk) >> 10) & 3u)] * x[(b) + 5]  \
                 + cb_lds[(lds) + (((pk) >> 12) & 3u)] * x[(b) + 6]  \
                 + cb_lds[(lds) + (((pk) >> 14) & 3u)] * x[(b) + 7]
        MQ2_DOT8(pk0,  0, base,       acc0);
        MQ2_DOT8(pk1,  4, base + 256, acc1);
        MQ2_DOT8(pk2,  8, base + 512, acc2);
        MQ2_DOT8(pk3, 12, base + 768, acc3);
        #undef MQ2_DOT8
    }

    for (int t = 0; t < tail; ++t) {
        const int g = (quads << 2) + t;
        const char* gp = row_ptr + g * 72;
        if (tid < 4) {
            cb_lds[tid] =
                __half2float(reinterpret_cast<const __half*>(gp)[tid]);
        }
        __syncthreads();
        const unsigned int pk =
            *reinterpret_cast<const unsigned short*>(gp + 8 + boff);
        const int base = g * 256 + tid * 8;
        float tail_acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            tail_acc += cb_lds[(pk >> (j * 2)) & 3u] * x[base + j];
        }
        acc0 += tail_acc;
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    return acc;
}

__launch_bounds__(32, 20)
extern "C" __global__
void gemv_mq2g256_lloyd_moe_down_residual_scaled_rankpair_indexed(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float* __restrict__ x_residual,
    int M,
    int K,
    int k_top
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int pair = blockIdx.y;
    const int rank0 = pair * 2;
    const int rank1 = rank0 + 1;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    __shared__ float cb_lds[16];

    float combined = 0.0f;
    if (rank0 < k_top) {
        const int expert0 = topk_indices[rank0];
        const char* row0 =
            reinterpret_cast<const char*>(expert_ptrs[expert0])
            + (long long)row * row_bytes;
        const float* x0 = rot_batch + (long long)rank0 * K;
        const float dot0 = mq2_down_dot_k4(row0, x0, K, cb_lds);
        if (tid == 0) combined += topk_weights[rank0] * dot0;
    }
    if (rank1 < k_top) {
        const int expert1 = topk_indices[rank1];
        const char* row1 =
            reinterpret_cast<const char*>(expert_ptrs[expert1])
            + (long long)row * row_bytes;
        const float* x1 = rot_batch + (long long)rank1 * K;
        const float dot1 = mq2_down_dot_k4(row1, x1, K, cb_lds);
        if (tid == 0) combined += topk_weights[rank1] * dot1;
    }
    if (tid == 0) {
        atomicAdd(&x_residual[row], combined);
    }
}

// One wave computes two adjacent output rows for one routed rank.  Both rows
// consume the same rotated activation, so explicitly keeping each x value in a
// register avoids issuing the activation load twice.  Weight traffic and the
// per-row accumulation order remain identical to the one-row K4 kernel.
__launch_bounds__(32, 16)
extern "C" __global__
void gemv_mq2g256_lloyd_moe_down_residual_scaled_rowtile2_indexed(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float* __restrict__ x_residual,
    int M,
    int K
) {
    const int row0 = blockIdx.x * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool row1_valid = row1 < M;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = rot_batch + (long long)krank * K;
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr0 = A + (long long)row0 * row_bytes;
    // Reusing row0 for the inactive tail row keeps all lanes converged.
    const char* row_ptr1 =
        A + (long long)(row1_valid ? row1 : row0) * row_bytes;
    const int boff = tid * 2;

    __shared__ float cb_lds[32];
    float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define MQ2_DOT8_ROWS(gp0, gp1, cb0, cb1, b, acc0, acc1) do {   \
        const unsigned int pk0 =                                    \
            *reinterpret_cast<const unsigned short*>((gp0) + 8 + boff); \
        const unsigned int pk1 =                                    \
            *reinterpret_cast<const unsigned short*>((gp1) + 8 + boff); \
        float dot0 = 0.0f;                                          \
        float dot1 = 0.0f;                                          \
        float xv = x[(b)];                                          \
        dot0 += cb_lds[(cb0) + ((pk0) & 3u)] * xv;                  \
        dot1 += cb_lds[(cb1) + ((pk1) & 3u)] * xv;                  \
        xv = x[(b) + 1];                                            \
        dot0 += cb_lds[(cb0) + (((pk0) >> 2) & 3u)] * xv;           \
        dot1 += cb_lds[(cb1) + (((pk1) >> 2) & 3u)] * xv;           \
        xv = x[(b) + 2];                                            \
        dot0 += cb_lds[(cb0) + (((pk0) >> 4) & 3u)] * xv;           \
        dot1 += cb_lds[(cb1) + (((pk1) >> 4) & 3u)] * xv;           \
        xv = x[(b) + 3];                                            \
        dot0 += cb_lds[(cb0) + (((pk0) >> 6) & 3u)] * xv;           \
        dot1 += cb_lds[(cb1) + (((pk1) >> 6) & 3u)] * xv;           \
        xv = x[(b) + 4];                                            \
        dot0 += cb_lds[(cb0) + (((pk0) >> 8) & 3u)] * xv;           \
        dot1 += cb_lds[(cb1) + (((pk1) >> 8) & 3u)] * xv;           \
        xv = x[(b) + 5];                                            \
        dot0 += cb_lds[(cb0) + (((pk0) >> 10) & 3u)] * xv;          \
        dot1 += cb_lds[(cb1) + (((pk1) >> 10) & 3u)] * xv;          \
        xv = x[(b) + 6];                                            \
        dot0 += cb_lds[(cb0) + (((pk0) >> 12) & 3u)] * xv;          \
        dot1 += cb_lds[(cb1) + (((pk1) >> 12) & 3u)] * xv;          \
        xv = x[(b) + 7];                                            \
        dot0 += cb_lds[(cb0) + (((pk0) >> 14) & 3u)] * xv;          \
        dot1 += cb_lds[(cb1) + (((pk1) >> 14) & 3u)] * xv;          \
        (acc0) += dot0;                                             \
        (acc1) += dot1;                                             \
    } while (0)

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const char* gp00 = row_ptr0 + g * 72;
        const char* gp10 = row_ptr1 + g * 72;
        if (tid < 16) {
            const __half* cb =
                reinterpret_cast<const __half*>(
                    gp00 + (long long)(tid >> 2) * 72);
            cb_lds[tid] = __half2float(cb[tid & 3]);
        } else {
            const int t = tid - 16;
            const __half* cb =
                reinterpret_cast<const __half*>(
                    gp10 + (long long)(t >> 2) * 72);
            cb_lds[tid] = __half2float(cb[t & 3]);
        }
        __syncthreads();

        const char* gp01 = gp00 + 72;
        const char* gp02 = gp01 + 72;
        const char* gp03 = gp02 + 72;
        const char* gp11 = gp10 + 72;
        const char* gp12 = gp11 + 72;
        const char* gp13 = gp12 + 72;
        const int base = g * 256 + tid * 8;
        MQ2_DOT8_ROWS(gp00, gp10, 0, 16, base,       a00, a10);
        MQ2_DOT8_ROWS(gp01, gp11, 4, 20, base + 256, a01, a11);
        MQ2_DOT8_ROWS(gp02, gp12, 8, 24, base + 512, a02, a12);
        MQ2_DOT8_ROWS(gp03, gp13, 12, 28, base + 768, a03, a13);
    }

    for (int t = 0; t < tail; ++t) {
        const int g = (quads << 2) + t;
        const char* gp0 = row_ptr0 + g * 72;
        const char* gp1 = row_ptr1 + g * 72;
        if (tid < 4) {
            cb_lds[tid] =
                __half2float(reinterpret_cast<const __half*>(gp0)[tid]);
        } else if (tid >= 16 && tid < 20) {
            const int lane = tid - 16;
            cb_lds[16 + lane] =
                __half2float(reinterpret_cast<const __half*>(gp1)[lane]);
        }
        __syncthreads();
        const int base = g * 256 + tid * 8;
        MQ2_DOT8_ROWS(gp0, gp1, 0, 16, base, a00, a10);
    }
    #undef MQ2_DOT8_ROWS

    float sum0 = (a00 + a01) + (a02 + a03);
    float sum1 = (a10 + a11) + (a12 + a13);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }
    if (tid == 0) {
        const float scale = topk_weights[krank];
        atomicAdd(&x_residual[row0], scale * sum0);
        if (row1_valid) {
            atomicAdd(&x_residual[row1], scale * sum1);
        }
    }
}

// DeepSeek4's routed down projection is exactly K=2048 (8 MQ2 groups).  Load
// every codebook before a single barrier, then expose eight independent dot
// chains to the gfx1151 scheduler.  The K4 kernel needs two codebook barriers
// and revisits four accumulator chains; this form trades a small VGPR increase
// for deeper within-wave memory/FMA ILP.
__launch_bounds__(32, 16)
extern "C" __global__
void gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float* __restrict__ x_residual,
    int M,
    int K
) {
    const int row = blockIdx.x;
    if (row >= M || K != 2048) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;
    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = rot_batch + (long long)krank * K;
    const char* row_ptr = A + (long long)row * (8 * 72);
    const int boff = tid * 2;

    __shared__ float cb_lds[32];
    const int cb_group = tid >> 2;
    const __half* cb =
        reinterpret_cast<const __half*>(row_ptr + cb_group * 72);
    cb_lds[tid] = __half2float(cb[tid & 3]);
    __syncthreads();

    const unsigned int p0 =
        *reinterpret_cast<const unsigned short*>(row_ptr + 0 * 72 + 8 + boff);
    const unsigned int p1 =
        *reinterpret_cast<const unsigned short*>(row_ptr + 1 * 72 + 8 + boff);
    const unsigned int p2 =
        *reinterpret_cast<const unsigned short*>(row_ptr + 2 * 72 + 8 + boff);
    const unsigned int p3 =
        *reinterpret_cast<const unsigned short*>(row_ptr + 3 * 72 + 8 + boff);
    const unsigned int p4 =
        *reinterpret_cast<const unsigned short*>(row_ptr + 4 * 72 + 8 + boff);
    const unsigned int p5 =
        *reinterpret_cast<const unsigned short*>(row_ptr + 5 * 72 + 8 + boff);
    const unsigned int p6 =
        *reinterpret_cast<const unsigned short*>(row_ptr + 6 * 72 + 8 + boff);
    const unsigned int p7 =
        *reinterpret_cast<const unsigned short*>(row_ptr + 7 * 72 + 8 + boff);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

    #define MQ2_DOT8_GROUP(pk, lds, b, acc)                         \
        (acc) += cb_lds[(lds) + ((pk) & 3u)] * x[(b)]               \
               + cb_lds[(lds) + (((pk) >> 2) & 3u)] * x[(b) + 1]    \
               + cb_lds[(lds) + (((pk) >> 4) & 3u)] * x[(b) + 2]    \
               + cb_lds[(lds) + (((pk) >> 6) & 3u)] * x[(b) + 3]    \
               + cb_lds[(lds) + (((pk) >> 8) & 3u)] * x[(b) + 4]    \
               + cb_lds[(lds) + (((pk) >> 10) & 3u)] * x[(b) + 5]   \
               + cb_lds[(lds) + (((pk) >> 12) & 3u)] * x[(b) + 6]   \
               + cb_lds[(lds) + (((pk) >> 14) & 3u)] * x[(b) + 7]

    const int lane_base = tid * 8;
    MQ2_DOT8_GROUP(p0, 0,  lane_base + 0 * 256, a0);
    MQ2_DOT8_GROUP(p1, 4,  lane_base + 1 * 256, a1);
    MQ2_DOT8_GROUP(p2, 8,  lane_base + 2 * 256, a2);
    MQ2_DOT8_GROUP(p3, 12, lane_base + 3 * 256, a3);
    MQ2_DOT8_GROUP(p4, 16, lane_base + 4 * 256, a4);
    MQ2_DOT8_GROUP(p5, 20, lane_base + 5 * 256, a5);
    MQ2_DOT8_GROUP(p6, 24, lane_base + 6 * 256, a6);
    MQ2_DOT8_GROUP(p7, 28, lane_base + 7 * 256, a7);
    #undef MQ2_DOT8_GROUP

    const float b0 = a0 + a4;
    const float b1 = a1 + a5;
    const float b2 = a2 + a6;
    const float b3 = a3 + a7;
    float acc = (b0 + b1) + (b2 + b3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) {
        atomicAdd(&x_residual[row], topk_weights[krank] * acc);
    }
}
