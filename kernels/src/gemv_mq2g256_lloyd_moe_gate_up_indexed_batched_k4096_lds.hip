// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Fixed-K=4096 position-batched twin of the shipping LDS-codebook routed
// gate/up kernel. DeepSeek V4 Flash has exactly 16 MQ2G256 groups per expert
// row. Load all 16 four-entry codebooks cooperatively once, then retain the
// shipping K4 accumulator and reduction order without a loop or repeated
// codebook loads.
//
// Grid: (M, K_TOP, N). Block: 32 threads.

__launch_bounds__(32, 20)
extern "C" __global__ void
gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4096_lds(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x_rot,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M,
    int K,
    int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M || K != 4096) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;
    const int routing_index = bid * K_TOP + krank;

    const int expert_id = topk_indices[routing_index];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x_rot + (long long)bid * K;
    const char* row_ptr = A + (long long)row * (16 * 72);
    const int boff = tid * 2;
    const int xoff = tid * 8;

    __shared__ float cb_lds[64];
    const int cb_group = tid >> 1;
    const int cb_entry = (tid & 1) << 1;
    const __half* cb =
        reinterpret_cast<const __half*>(row_ptr + cb_group * 72);
    cb_lds[tid * 2] = __half2float(cb[cb_entry]);
    cb_lds[tid * 2 + 1] = __half2float(cb[cb_entry + 1]);
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    #define MQ2_DOT_GROUP(group, acc) do {                                \
        const unsigned int packed =                                       \
            *reinterpret_cast<const unsigned short*>(                     \
                row_ptr + (group) * 72 + 8 + boff);                        \
        const int lds = (group) * 4;                                       \
        const int base = (group) * 256 + xoff;                             \
        (acc) += cb_lds[lds + ((packed) & 3u)] * x_row[base]               \
               + cb_lds[lds + (((packed) >> 2) & 3u)] * x_row[base + 1]   \
               + cb_lds[lds + (((packed) >> 4) & 3u)] * x_row[base + 2]   \
               + cb_lds[lds + (((packed) >> 6) & 3u)] * x_row[base + 3]   \
               + cb_lds[lds + (((packed) >> 8) & 3u)] * x_row[base + 4]   \
               + cb_lds[lds + (((packed) >> 10) & 3u)] * x_row[base + 5]  \
               + cb_lds[lds + (((packed) >> 12) & 3u)] * x_row[base + 6]  \
               + cb_lds[lds + (((packed) >> 14) & 3u)] * x_row[base + 7]; \
    } while (0)

    MQ2_DOT_GROUP(0, acc0);
    MQ2_DOT_GROUP(1, acc1);
    MQ2_DOT_GROUP(2, acc2);
    MQ2_DOT_GROUP(3, acc3);
    MQ2_DOT_GROUP(4, acc0);
    MQ2_DOT_GROUP(5, acc1);
    MQ2_DOT_GROUP(6, acc2);
    MQ2_DOT_GROUP(7, acc3);
    MQ2_DOT_GROUP(8, acc0);
    MQ2_DOT_GROUP(9, acc1);
    MQ2_DOT_GROUP(10, acc2);
    MQ2_DOT_GROUP(11, acc3);
    MQ2_DOT_GROUP(12, acc0);
    MQ2_DOT_GROUP(13, acc1);
    MQ2_DOT_GROUP(14, acc2);
    MQ2_DOT_GROUP(15, acc3);
    #undef MQ2_DOT_GROUP

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    if (tid == 0) {
        const int mi = M >> 1;
        const long long stride = (long long)K_TOP * mi;
        if (row < mi) {
            y_gate[(long long)bid * stride + krank * mi + row] = acc;
        } else {
            y_up[(long long)bid * stride + krank * mi + (row - mi)] = acc;
        }
    }
}
