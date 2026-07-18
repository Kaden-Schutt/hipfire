// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// MODEL CONTRACT: LiquidAI LFM2.5-8B-A1B routed-expert down projection only.
//   M=2048, K=1792, K_TOP=4, HFQ4-G256 affine weights (136 bytes/group).
// HARDWARE CONTRACT: wave32 RDNA on gfx1010/gfx1030/gfx1100/gfx1151/gfx1201.
// OWNERSHIP CONTRACT: compiled and launched only by hipfire-arch-lfm2moe.
//   Do not register this symbol in rdna-compute shared MoE dispatch.

#include <hip/hip_runtime.h>

namespace {
constexpr int LFM2_M = 2048;
constexpr int LFM2_K = 1792;
constexpr int LFM2_K_TOP = 4;
constexpr int GROUP_SIZE = 256;
constexpr int GROUP_BYTES = 136;
constexpr int GROUPS_PER_ROW = LFM2_K / GROUP_SIZE;
}

__launch_bounds__(32, 16)
extern "C" __global__ void lfm2_a1b_moe_down_hfq4g256_k1792_wave32(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs
) {
    const int row = blockIdx.x;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;
    if (row >= LFM2_M || krank >= LFM2_K_TOP) return;

    const int expert_id = topk_indices[bid * LFM2_K_TOP + krank];
    const char* __restrict__ expert =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const char* __restrict__ row_ptr =
        expert + (long long)row * GROUPS_PER_ROW * GROUP_BYTES;
    const float* __restrict__ x =
        rot_batch + ((long long)bid * LFM2_K_TOP + krank) * LFM2_K;
    const int packed_offset = tid * 4;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

#define ACCUMULATE_GROUP(group, accumulator) do {                                      \
        const char* gp = row_ptr + (group) * GROUP_BYTES;                              \
        const float scale = __builtin_bit_cast(float, *(const unsigned int*)(gp));      \
        const float zero = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));   \
        const unsigned int packed =                                                     \
            *(const unsigned int*)(gp + 8 + packed_offset);                             \
        const int base = (group) * GROUP_SIZE + tid * 8;                                \
        (accumulator) += (scale * (float)(packed & 0xFu) + zero) * x[base]               \
            + (scale * (float)((packed >> 4) & 0xFu) + zero) * x[base + 1]              \
            + (scale * (float)((packed >> 8) & 0xFu) + zero) * x[base + 2]              \
            + (scale * (float)((packed >> 12) & 0xFu) + zero) * x[base + 3]             \
            + (scale * (float)((packed >> 16) & 0xFu) + zero) * x[base + 4]             \
            + (scale * (float)((packed >> 20) & 0xFu) + zero) * x[base + 5]             \
            + (scale * (float)((packed >> 24) & 0xFu) + zero) * x[base + 6]             \
            + (scale * (float)((packed >> 28) & 0xFu) + zero) * x[base + 7];            \
    } while (0)

    // Seven groups: preserve the generic kernel's four-way accumulation tree.
    ACCUMULATE_GROUP(0, acc0);
    ACCUMULATE_GROUP(1, acc1);
    ACCUMULATE_GROUP(2, acc2);
    ACCUMULATE_GROUP(3, acc3);
    ACCUMULATE_GROUP(4, acc0);
    ACCUMULATE_GROUP(5, acc1);
    ACCUMULATE_GROUP(6, acc2);
#undef ACCUMULATE_GROUP

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const long long out =
            ((long long)bid * LFM2_K_TOP + krank) * LFM2_M + row;
        expert_outputs[out] = acc;
    }
}
