// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// CDNA3 wave64 lowering of the MQ2-Lloyd routed-expert gate/up GEMV.
// Each lane consumes one packed byte (four weights) from every 256-weight
// group. The full wave therefore covers one group without relying on wave32
// execution, while retaining the shipping 72-byte group layout.

extern "C" __global__ __launch_bounds__(64)
void gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_gfx942(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x_rot,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M,
    int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    __shared__ float cb_lds[16];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define MQ2_DOT4(pk, lds, base, acc) do {                         \
        (acc) += cb_lds[(lds) + ( (pk)       & 3u)] * x_rot[(base)]   \
               + cb_lds[(lds) + (((pk) >> 2) & 3u)] * x_rot[(base)+1] \
               + cb_lds[(lds) + (((pk) >> 4) & 3u)] * x_rot[(base)+2] \
               + cb_lds[(lds) + (((pk) >> 6) & 3u)] * x_rot[(base)+3]; \
    } while (0)

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;

        if (tid < 16) {
            const __half* cb = reinterpret_cast<const __half*>(
                gp0 + (long long)(tid >> 2) * 72);
            cb_lds[tid] = __half2float(cb[tid & 3]);
        }
        __syncthreads();

        const unsigned int pk0 =
            *(reinterpret_cast<const unsigned char*>(gp0 + 8) + tid);
        const unsigned int pk1 =
            *(reinterpret_cast<const unsigned char*>(gp1 + 8) + tid);
        const unsigned int pk2 =
            *(reinterpret_cast<const unsigned char*>(gp2 + 8) + tid);
        const unsigned int pk3 =
            *(reinterpret_cast<const unsigned char*>(gp3 + 8) + tid);
        const int base = g * 256 + tid * 4;

        MQ2_DOT4(pk0, 0, base, acc0);
        MQ2_DOT4(pk1, 4, base + 256, acc1);
        MQ2_DOT4(pk2, 8, base + 512, acc2);
        MQ2_DOT4(pk3, 12, base + 768, acc3);
    }

    for (int t = 0; t < tail; ++t) {
        const int g = (quads << 2) + t;
        const char* gp = row_ptr + (long long)g * 72;
        if (tid < 4) {
            cb_lds[tid] = __half2float(
                reinterpret_cast<const __half*>(gp)[tid]);
        }
        __syncthreads();
        const unsigned int pk =
            *(reinterpret_cast<const unsigned char*>(gp + 8) + tid);
        const int base = g * 256 + tid * 4;
        MQ2_DOT4(pk, 0, base, acc0);
    }
    #undef MQ2_DOT4

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 32; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    if (tid == 0) {
        const int mi = M >> 1;
        if (row < mi) {
            y_gate[krank * mi + row] = acc;
        } else {
            y_up[krank * mi + (row - mi)] = acc;
        }
    }
}
