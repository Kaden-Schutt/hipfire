// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Exact DeepSeek-V4 MQ2R decode shape for gfx942:
//   top-k = 6, M = 4096 (2048 gate + 2048 up), K = 4096.
//
// Eight native wave64s own eight consecutive output rows.  All eight rows
// share the same routed-rank activation, so the workgroup stages each
// four-group (1024-float) x tile once in LDS instead of rereading it once per
// row.  Each wave retains a disjoint Lloyd-codebook slice and the incumbent
// four-accumulator / width-64 reduction order.

extern "C" __global__ __launch_bounds__(512)
void gemv_mq2g256_lloyd_moe_gate_up_wave64x8_gfx942_candidate(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x_rot,
    float* __restrict__ y_gate,
    float* __restrict__ y_up
) {
    constexpr int M = 4096;
    constexpr int MI = 2048;
    constexpr int K = 4096;
    constexpr int GROUP_BYTES = 72;
    constexpr int GROUPS_PER_ROW = K / 256;
    constexpr int ROW_BYTES = GROUPS_PER_ROW * GROUP_BYTES;

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int row = (int)blockIdx.x * 8 + wave;
    const int krank = blockIdx.y;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ weights =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const char* __restrict__ row_ptr = weights + (long long)row * ROW_BYTES;
    const float* __restrict__ x = x_rot;

    // 4 KiB activation tile shared by all eight rows, plus 512 B of disjoint
    // per-wave Lloyd codebooks (four codebooks per unrolled iteration).
    __shared__ float x_lds[1024];
    __shared__ float cb_lds[8][16];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    #define MQ2_DOT4(pk, cb, xbase, acc) do {                         \
        (acc) += cb_lds[wave][(cb) + ( (pk)       & 3u)] *           \
                     x_lds[(xbase)]                                   \
               + cb_lds[wave][(cb) + (((pk) >> 2) & 3u)] *           \
                     x_lds[(xbase) + 1]                               \
               + cb_lds[wave][(cb) + (((pk) >> 4) & 3u)] *           \
                     x_lds[(xbase) + 2]                               \
               + cb_lds[wave][(cb) + (((pk) >> 6) & 3u)] *           \
                     x_lds[(xbase) + 3];                              \
    } while (0)

    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int g = q * 4;
        const char* gp0 = row_ptr + (long long)g * GROUP_BYTES;
        const char* gp1 = gp0 + GROUP_BYTES;
        const char* gp2 = gp1 + GROUP_BYTES;
        const char* gp3 = gp2 + GROUP_BYTES;

        // 512 threads cooperatively stage 1024 floats: two elements/thread.
        x_lds[tid] = x[g * 256 + tid];
        x_lds[tid + 512] = x[g * 256 + tid + 512];

        // Lanes 0..15 in each wave stage four adjacent codebooks for that
        // wave's row. No wave can overwrite another wave's slice.
        if (lane < 16) {
            const __half* cb = reinterpret_cast<const __half*>(
                gp0 + (long long)(lane >> 2) * GROUP_BYTES);
            cb_lds[wave][lane] = __half2float(cb[lane & 3]);
        }
        __syncthreads();

        const unsigned int pk0 =
            *(reinterpret_cast<const unsigned char*>(gp0 + 8) + lane);
        const unsigned int pk1 =
            *(reinterpret_cast<const unsigned char*>(gp1 + 8) + lane);
        const unsigned int pk2 =
            *(reinterpret_cast<const unsigned char*>(gp2 + 8) + lane);
        const unsigned int pk3 =
            *(reinterpret_cast<const unsigned char*>(gp3 + 8) + lane);
        const int base = lane * 4;

        MQ2_DOT4(pk0, 0, base, acc0);
        MQ2_DOT4(pk1, 4, 256 + base, acc1);
        MQ2_DOT4(pk2, 8, 512 + base, acc2);
        MQ2_DOT4(pk3, 12, 768 + base, acc3);

        // Every wave must finish reading the shared activation tile before
        // any thread overwrites it for the next four-group iteration.
        __syncthreads();
    }
    #undef MQ2_DOT4

    float acc = (acc0 + acc1) + (acc2 + acc3);
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    if (lane == 0) {
        if (row < MI) {
            y_gate[(long long)krank * MI + row] = acc;
        } else {
            y_up[(long long)krank * MI + (row - MI)] = acc;
        }
    }
}
