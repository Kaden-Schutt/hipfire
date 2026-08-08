// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Expert-grouped twin of gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4096_lds.
//
// The incumbent resolves its expert per verify position
// (topk_indices[bid * K_TOP + krank]) on a (M, K_TOP, N) grid, so an expert's
// weight row streams from memory once per position that routes to it. Measured
// on the golden k6 fixture, 37.4% of routed-expert references are duplicates,
// and the duplicate fraction GROWS with the block (ratio 0.554 at B=6 against
// ~0.61 at B=2). That redundancy is the marginal per-position verify cost that
// pins the DSpark block controller at B=2.
//
// This variant keeps the grid, the launch shape, and the buffers identical.
// Each block instead deduplicates itself: only the FIRST routing index that
// references a given expert does any work, and it loops over every later index
// sharing that expert. Duplicate blocks exit before touching memory. Total
// arithmetic is unchanged; weight TRAFFIC drops to one read per distinct
// (row, expert).
//
// Bit-identical to the incumbent by construction. Grouping changes only which
// block loads a weight, never how an output is summed: every output element
// keeps its own accumulator set, the same 16 fully-unrolled MQ2_DOT_GROUP
// steps in the same order, and the same shuffle reduction tree. Reduction
// order is a property of the member loop body, which is a verbatim copy of the
// incumrent's, not of the grouping.
//
// The block is a single wave32, so __any() is a block-wide reduction and no
// barrier is needed for the dedup vote.
//
// The 16 packed code words per thread are staged into LDS rather than re-read
// per member. x_row is 16 KiB per member and would otherwise evict the 1152 B
// weight row from L0 between members, which would forfeit the entire point.
// LDS cost is 1024 B of codes + 256 B of codebooks.
//
// Grid: (M, K_TOP, N). Block: 32 threads.

__launch_bounds__(32, 20)
extern "C" __global__ void
gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4096_lds_grouped(
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

    const int n_route = K_TOP * (int)gridDim.z;
    const int my_idx = bid * K_TOP + krank;
    const int my_expert = topk_indices[my_idx];

    // Dedup vote: bail unless this is the first routing index for my_expert.
    // 32 threads stride the prefix; the block is one wave so __any covers it.
    int dup = 0;
    for (int i = tid; i < my_idx; i += 32) {
        if (topk_indices[i] == my_expert) dup = 1;
    }
    if (__any(dup)) return;

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[my_expert]);
    const char* row_ptr = A + (long long)row * (16 * 72);
    const int boff = tid * 2;
    const int xoff = tid * 8;

    __shared__ float cb_lds[64];
    __shared__ unsigned short code_lds[16 * 32];

    const int cb_group = tid >> 1;
    const int cb_entry = (tid & 1) << 1;
    const __half* cb =
        reinterpret_cast<const __half*>(row_ptr + cb_group * 72);
    cb_lds[tid * 2] = __half2float(cb[cb_entry]);
    cb_lds[tid * 2 + 1] = __half2float(cb[cb_entry + 1]);

    // One HBM read of this (row, expert)'s packed codes, reused by every member.
    #pragma unroll
    for (int g = 0; g < 16; ++g) {
        code_lds[g * 32 + tid] =
            *reinterpret_cast<const unsigned short*>(row_ptr + g * 72 + 8 + boff);
    }
    __syncthreads();

    const int mi = M >> 1;
    const long long stride = (long long)K_TOP * mi;

    for (int j = my_idx; j < n_route; ++j) {
        if (topk_indices[j] != my_expert) continue;

        const int m_bid = j / K_TOP;
        const int m_krank = j - m_bid * K_TOP;
        const float* __restrict__ x_row = x_rot + (long long)m_bid * K;

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        #define MQ2_DOT_GROUP(group, acc) do {                                \
            const unsigned int packed = code_lds[(group) * 32 + tid];         \
            const int lds = (group) * 4;                                      \
            const int base = (group) * 256 + xoff;                            \
            (acc) += cb_lds[lds + ((packed) & 3u)] * x_row[base]              \
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
            if (row < mi) {
                y_gate[(long long)m_bid * stride + m_krank * mi + row] = acc;
            } else {
                y_up[(long long)m_bid * stride + m_krank * mi + (row - mi)] = acc;
            }
        }
    }
}
