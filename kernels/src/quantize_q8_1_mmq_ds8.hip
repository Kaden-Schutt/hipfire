// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

#define WARP_SIZE 32
#define QK8_1 32

struct block_q8_1_mmq_ds8 {
    half2 ds8[8];          // one (d, mean_x) pair per 16-value WMMA K tile
    int8_t qs[4 * QK8_1];  // 128 int8 values
};

static_assert(sizeof(block_q8_1_mmq_ds8) == 160, "bad block_q8_1_mmq_ds8 size");

extern "C" __global__ void quantize_q8_1_mmq_ds8(
    const float* __restrict__ X,
    block_q8_1_mmq_ds8* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Four lanes cover one 16-value WMMA K tile. Keep reductions inside
    // that 4-lane subgroup so each tile gets its own centered d/mean pair.
    float sum = xi.x + xi.y + xi.z + xi.w;
    sum += __shfl_xor(sum, 2, WARP_SIZE);
    sum += __shfl_xor(sum, 1, WARP_SIZE);
    const float mean = sum * 0.0625f;

    float amax = fmaxf(
        fmaxf(fabsf(xi.x - mean), fabsf(xi.y - mean)),
        fmaxf(fabsf(xi.z - mean), fabsf(xi.w - mean))
    );
    amax = fmaxf(amax, __shfl_xor(amax, 2, WARP_SIZE));
    amax = fmaxf(amax, __shfl_xor(amax, 1, WARP_SIZE));

    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, (xi.x - mean) * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, (xi.y - mean) * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, (xi.z - mean) * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, (xi.w - mean) * id)));

    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_mmq_ds8* out = Y + (long long)kblock * N + token;
    ((char4*)out->qs)[iqs / 4] = q;

    if ((iqs & 15) == 0) {
        out->ds8[iqs / 16] = make_half2((_Float16)d, (_Float16)mean);
    }
}
