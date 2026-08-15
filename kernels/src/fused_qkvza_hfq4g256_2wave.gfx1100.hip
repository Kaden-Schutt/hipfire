// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// RDNA3 decode experiment: two wave32s cooperate on one output row. Each
// wave owns alternating HFQ4 groups, keeping four independent accumulators
// while halving the per-wave K-loop. A two-float LDS join replaces a second
// global dispatch and gives gfx1100 enough light waves to approach its
// occupancy-sensitive bandwidth ceiling.
__launch_bounds__(64, 16)
extern "C" __global__ void fused_qkvza_hfq4g256_2wave(
    const char* __restrict__ A_qkv,
    const char* __restrict__ A_z,
    const char* __restrict__ A_beta,
    const char* __restrict__ A_alpha,
    const float* __restrict__ x,
    float* __restrict__ y_qkv,
    float* __restrict__ y_z,
    float* __restrict__ y_beta,
    float* __restrict__ y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K
) {
    const int tid = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane = tid & 31;
    const int gid = blockIdx.x;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    const char* A;
    float* y;
    int row;
    if (gid < qkv_m) {
        A = A_qkv; y = y_qkv; row = gid;
    } else if (gid < qkv_m + z_m) {
        A = A_z; y = y_z; row = gid - qkv_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta; y = y_beta; row = gid - qkv_m - z_m;
    } else {
        A = A_alpha; y = y_alpha; row = gid - qkv_m - z_m - beta_m;
    }

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = lane * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto row_rsrc = HIPFIRE_WEIGHT_RSRC(row_ptr);
#endif

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

#define DOG_GROUP(group_index, accum) do { \
        const int dog_g = (group_index); \
        const char* dog_gp = row_ptr + dog_g * 136; \
        const float dog_sc = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, dog_gp, dog_g * 136)); \
        const float dog_zp = __builtin_bit_cast(float, \
            HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, dog_gp + 4, dog_g * 136 + 4)); \
        const unsigned int dog_pk = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            row_rsrc, dog_gp + 8 + boff, dog_g * 136 + 8 + boff); \
        const int dog_b = dog_g * 256 + lane * 8; \
        (accum) += (dog_sc * (float)((dog_pk) & 0xFu) + dog_zp) * x[dog_b] \
                 + (dog_sc * (float)(((dog_pk) >> 4) & 0xFu) + dog_zp) * x[dog_b + 1] \
                 + (dog_sc * (float)(((dog_pk) >> 8) & 0xFu) + dog_zp) * x[dog_b + 2] \
                 + (dog_sc * (float)(((dog_pk) >> 12) & 0xFu) + dog_zp) * x[dog_b + 3] \
                 + (dog_sc * (float)(((dog_pk) >> 16) & 0xFu) + dog_zp) * x[dog_b + 4] \
                 + (dog_sc * (float)(((dog_pk) >> 20) & 0xFu) + dog_zp) * x[dog_b + 5] \
                 + (dog_sc * (float)(((dog_pk) >> 24) & 0xFu) + dog_zp) * x[dog_b + 6] \
                 + (dog_sc * (float)(((dog_pk) >> 28) & 0xFu) + dog_zp) * x[dog_b + 7]; \
    } while (0)

    for (int g = wave_id; g < groups_per_row; g += 8) {
        DOG_GROUP(g, acc0);
        if (g + 2 < groups_per_row) DOG_GROUP(g + 2, acc1);
        if (g + 4 < groups_per_row) DOG_GROUP(g + 4, acc2);
        if (g + 6 < groups_per_row) DOG_GROUP(g + 6, acc3);
    }
#undef DOG_GROUP

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    __shared__ float wave_partial[2];
    if (lane == 0) wave_partial[wave_id] = acc;
    __syncthreads();
    if (tid == 0) y[row] = wave_partial[0] + wave_partial[1];
}
