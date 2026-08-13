// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Position-batched form of the shipping K=2048 K8-all MQ2-Lloyd routed-down
// kernel. The ordinary AR path already uses this fixed-shape body; speculative
// verify previously fell back to the generic looped K4 kernel despite having
// the same K=2048 shape. Arithmetic and pairwise reduction order are unchanged.
//
// Grid: (M, K_TOP, N). Block: 32 threads.

__launch_bounds__(32, 16)
extern "C" __global__ void
gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float* __restrict__ x_residual,
    int M,
    int K,
    int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M || K != 2048) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;
    const int routing_base = bid * K_TOP;
    const int routing_index = routing_base + krank;
    const int expert_id = topk_indices[routing_index];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + (long long)routing_index * K;
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

    #define MQ2_DOT8_GROUP(pk, lds, base, acc)                         \
        (acc) += cb_lds[(lds) + ((pk) & 3u)] * x[(base)]               \
               + cb_lds[(lds) + (((pk) >> 2) & 3u)] * x[(base) + 1]   \
               + cb_lds[(lds) + (((pk) >> 4) & 3u)] * x[(base) + 2]   \
               + cb_lds[(lds) + (((pk) >> 6) & 3u)] * x[(base) + 3]   \
               + cb_lds[(lds) + (((pk) >> 8) & 3u)] * x[(base) + 4]   \
               + cb_lds[(lds) + (((pk) >> 10) & 3u)] * x[(base) + 5]  \
               + cb_lds[(lds) + (((pk) >> 12) & 3u)] * x[(base) + 6]  \
               + cb_lds[(lds) + (((pk) >> 14) & 3u)] * x[(base) + 7]

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
        atomicAdd(
            &x_residual[(long long)bid * M + row],
            topk_weights[routing_index] * acc
        );
    }
}
