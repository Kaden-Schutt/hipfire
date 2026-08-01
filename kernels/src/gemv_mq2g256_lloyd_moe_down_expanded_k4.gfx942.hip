// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Native wave64 lowering of the deterministic MQ2-Lloyd MoE down projection.
// Each 32-lane stripe retains the portable kernel's lane-to-weight mapping and
// reduction order, while one scheduled wave computes two output rows. Per-rank
// results remain expanded; the existing fixed-order combine stays unchanged.

extern "C" __global__ __launch_bounds__(64)
void gemv_mq2g256_lloyd_moe_down_expanded_k4_gfx942(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs,
    int M,
    int K,
    int K_TOP
) {
    const int tid = threadIdx.x;
    const int stripe = tid >> 5;
    const int lane = tid & 31;
    const int row = (int)blockIdx.x * 2 + stripe;
    if ((int)blockIdx.x * 2 >= M) return;

    const int safe_row = row < M ? row : M - 1;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)safe_row * row_bytes;
    const int boff = lane * 2;
    const int xoff = lane * 8;
    const int cb_base = stripe * 16;
    __shared__ float cb_lds[32];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define MQ2_DOT8(gptr, lds, base, acc) do {                         \
        const unsigned char* data =                                     \
            reinterpret_cast<const unsigned char*>((gptr) + 8);         \
        const unsigned char b0 = data[boff];                             \
        const unsigned char b1 = data[boff + 1];                         \
        (acc) += cb_lds[(lds) + (b0 & 3u)] * x[(base)]                   \
               + cb_lds[(lds) + ((b0 >> 2) & 3u)] * x[(base) + 1]       \
               + cb_lds[(lds) + ((b0 >> 4) & 3u)] * x[(base) + 2]       \
               + cb_lds[(lds) + (b0 >> 6)] * x[(base) + 3]              \
               + cb_lds[(lds) + (b1 & 3u)] * x[(base) + 4]              \
               + cb_lds[(lds) + ((b1 >> 2) & 3u)] * x[(base) + 5]       \
               + cb_lds[(lds) + ((b1 >> 4) & 3u)] * x[(base) + 6]       \
               + cb_lds[(lds) + (b1 >> 6)] * x[(base) + 7];             \
    } while (0)

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;
        if (lane < 16) {
            const __half* cb = reinterpret_cast<const __half*>(
                gp0 + (long long)(lane >> 2) * 72);
            cb_lds[cb_base + lane] = __half2float(cb[lane & 3]);
        }
        __syncthreads();

        const int base = g * 256 + xoff;
        MQ2_DOT8(gp0, cb_base, base, acc0);
        MQ2_DOT8(gp1, cb_base + 4, base + 256, acc1);
        MQ2_DOT8(gp2, cb_base + 8, base + 512, acc2);
        MQ2_DOT8(gp3, cb_base + 12, base + 768, acc3);
    }

    for (int t = 0; t < tail; ++t) {
        const int g = (quads << 2) + t;
        const char* gp = row_ptr + (long long)g * 72;
        if (lane < 4) {
            cb_lds[cb_base + lane] =
                __half2float(reinterpret_cast<const __half*>(gp)[lane]);
        }
        __syncthreads();
        MQ2_DOT8(gp, cb_base, g * 256 + xoff, acc0);
    }
    #undef MQ2_DOT8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset, 32);
    }

    if (row < M && lane == 0) {
        const long long out =
            (long long)bid * K_TOP * M + (long long)krank * M + row;
        expert_outputs[out] = acc;
    }
}
