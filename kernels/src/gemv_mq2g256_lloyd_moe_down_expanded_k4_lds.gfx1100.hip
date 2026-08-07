// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Exact-gfx1100 micro candidate for deterministic DS4 MQ2-Lloyd routed-down.
// The incumbent redundantly issues four fp16 loads and conversions in every
// lane for every group. This version has lanes 0..15 cooperatively materialize
// the four K4 codebooks once in LDS. Index loads, K4 accumulator partition,
// FMA expression order, wave reduction, and output layout are unchanged.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_expanded_k4_lds_gfx1100_candidate(
    const unsigned long long* __restrict__ expert_ptrs,
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
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 2;
    const int xoff = tid * 8;
    __shared__ float cb_lds[16];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define GROUP_FMA_LDS(gptr, lds, base, acc) do {                         \
        const unsigned char* data =                                          \
            reinterpret_cast<const unsigned char*>((gptr) + 8);              \
        const unsigned char b0 = data[boff];                                  \
        const unsigned char b1 = data[boff + 1];                              \
        (acc) += cb_lds[(lds) + (b0 & 3)]        * x[(base)]                  \
              + cb_lds[(lds) + ((b0 >> 2) & 3)] * x[(base) + 1]              \
              + cb_lds[(lds) + ((b0 >> 4) & 3)] * x[(base) + 2]              \
              + cb_lds[(lds) + (b0 >> 6)]       * x[(base) + 3]              \
              + cb_lds[(lds) + (b1 & 3)]        * x[(base) + 4]              \
              + cb_lds[(lds) + ((b1 >> 2) & 3)] * x[(base) + 5]              \
              + cb_lds[(lds) + ((b1 >> 4) & 3)] * x[(base) + 6]              \
              + cb_lds[(lds) + (b1 >> 6)]       * x[(base) + 7];             \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;
        if (tid < 16) {
            const __half* codebook = reinterpret_cast<const __half*>(
                gp0 + (long long)(tid >> 2) * 72);
            cb_lds[tid] = __half2float(codebook[tid & 3]);
        }
        __syncthreads();
        const int base = g * 256 + xoff;
        GROUP_FMA_LDS(gp0, 0, base,       acc0);
        GROUP_FMA_LDS(gp1, 4, base + 256, acc1);
        GROUP_FMA_LDS(gp2, 8, base + 512, acc2);
        GROUP_FMA_LDS(gp3, 12, base + 768, acc3);
    }

    const int tail_start = quads << 2;
    for (int g = tail_start; g < tail_start + tail; g++) {
        const char* gp = row_ptr + g * 72;
        if (tid < 4) {
            cb_lds[tid] = __half2float(reinterpret_cast<const __half*>(gp)[tid]);
        }
        __syncthreads();
        const int base = g * 256 + xoff;
        GROUP_FMA_LDS(gp, 0, base, acc0);
    }
    #undef GROUP_FMA_LDS

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) {
        expert_outputs[(long long)bid * K_TOP * M + (long long)krank * M + row] = acc;
    }
}
