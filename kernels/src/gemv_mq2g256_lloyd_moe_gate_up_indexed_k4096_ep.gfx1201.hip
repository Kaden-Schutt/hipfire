// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Exact-gfx1201 DS4 TP/EP decode gate/up specialized for the model's fixed
// K=4096. The incumbent stages four codebooks at a time and synchronizes four
// times. This sister stages all sixteen codebooks once, then preserves the
// incumbent four-accumulator arithmetic and wave32 reduction order.
// Grid: [M, k_top, 1], block [32].

__launch_bounds__(32, 20)
extern "C" __global__ void
gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_k4096_gfx1201_ep(
    const unsigned long long* __restrict__ expert_ptrs,
    const void* nonowned_dummy,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x_rot,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M,
    int K
) {
    const int row = blockIdx.x;
    if (row >= M || K != 4096) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;
    const int mi = M >> 1;

    const int expert_id = topk_indices[krank];
    const unsigned long long expert_address = expert_ptrs[expert_id];
    if (reinterpret_cast<const void*>(expert_address) == nonowned_dummy) {
        if (tid == 0) {
            if (row < mi) {
                y_gate[krank * mi + row] = 0.0f;
            } else {
                y_up[krank * mi + (row - mi)] = 0.0f;
            }
        }
        return;
    }

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_address);
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

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

#define MQ2_DOT_GROUP(group, acc) do {                                 \
        const unsigned int packed =                                    \
            *reinterpret_cast<const unsigned short*>(                  \
                row_ptr + (group) * 72 + 8 + boff);                     \
        const int lds = (group) * 4;                                    \
        const int base = (group) * 256 + xoff;                          \
        (acc) += cb_lds[lds + ((packed) & 3u)] * x_rot[base]            \
               + cb_lds[lds + (((packed) >> 2) & 3u)] * x_rot[base + 1] \
               + cb_lds[lds + (((packed) >> 4) & 3u)] * x_rot[base + 2] \
               + cb_lds[lds + (((packed) >> 6) & 3u)] * x_rot[base + 3] \
               + cb_lds[lds + (((packed) >> 8) & 3u)] * x_rot[base + 4] \
               + cb_lds[lds + (((packed) >> 10) & 3u)] * x_rot[base + 5] \
               + cb_lds[lds + (((packed) >> 12) & 3u)] * x_rot[base + 6] \
               + cb_lds[lds + (((packed) >> 14) & 3u)] * x_rot[base + 7]; \
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
        if (row < mi) {
            y_gate[krank * mi + row] = acc;
        } else {
            y_up[krank * mi + (row - mi)] = acc;
        }
    }
}
