// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Out-of-place variant of `givens_rotate_f32`. Reads from `x_in`, writes
// rotated activations to `x_out`. The in-place variant requires a
// preceding `copy_d2d(x_in -> x_out)` when the caller needs to preserve
// `x_in` for downstream consumers (the ParoQuant decode path runs
// multiple GEMVs against the same un-rotated x_norm). Combining the
// copy into the rotation kernel removes one graph node + one inter-node
// stream dependency, eliminating a class of hipGraph dependency-analysis
// hazards observed on gfx1151 / HIP 7.13.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <assert.h>

#define CTA_M 4
#define GROUP_SIZE 128
#define HALF_GROUP 64

extern "C" __global__ void givens_rotate_to_f32(
    const float* __restrict__ x_in,          // [seq_len, hidden_dim]
    float*       __restrict__ x_out,         // [seq_len, hidden_dim]
    const short* __restrict__ pairs,         // [krot, hidden_dim]
    const half*  __restrict__ theta,         // [krot, hidden_dim/2]
    const half*  __restrict__ channel_scales,// [hidden_dim]
    const int seq_len,
    const int hidden_dim,
    const int krot
) {
    const int row_block = blockIdx.x;
    const int group_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int ch_base = group_idx * GROUP_SIZE;
    if (ch_base >= hidden_dim) return;

    extern __shared__ float smem[];

    const float s0 = __half2float(channel_scales[ch_base + tid * 2]);
    const float s1 = __half2float(channel_scales[ch_base + tid * 2 + 1]);

    for (int r = 0; r < CTA_M; r++) {
        const int row = row_block * CTA_M + r;
        if (row >= seq_len) {
            smem[r * GROUP_SIZE + tid * 2]     = 0.0f;
            smem[r * GROUP_SIZE + tid * 2 + 1] = 0.0f;
            continue;
        }
        const int gx = row * hidden_dim + ch_base;
        smem[r * GROUP_SIZE + tid * 2]     = x_in[gx + tid * 2]     * s0;
        smem[r * GROUP_SIZE + tid * 2 + 1] = x_in[gx + tid * 2 + 1] * s1;
    }
    __syncthreads();

    for (int rot = 0; rot < krot; rot++) {
        const int i = (int)pairs[rot * hidden_dim + ch_base + tid * 2];
        const int j = (int)pairs[rot * hidden_dim + ch_base + tid * 2 + 1];
#ifdef NDEBUG
        __builtin_assume(i >= 0 && i < GROUP_SIZE);
        __builtin_assume(j >= 0 && j < GROUP_SIZE);
#else
        assert(i >= 0 && i < GROUP_SIZE);
        assert(j >= 0 && j < GROUP_SIZE);
#endif

        const float angle = __half2float(theta[rot * (hidden_dim / 2) + group_idx * HALF_GROUP + tid]);
        const float c = cosf(angle);
        const float s = sinf(angle);

        for (int r = 0; r < CTA_M; r++) {
            const int row = row_block * CTA_M + r;
            if (row >= seq_len) continue;

            float* row_smem = &smem[r * GROUP_SIZE];
            const float xi = row_smem[i];
            const float xj = row_smem[j];

            row_smem[i] = xi * c + xj * s;
            row_smem[j] = xj * c - xi * s;
        }
        __syncthreads();
    }

    for (int r = 0; r < CTA_M; r++) {
        const int row = row_block * CTA_M + r;
        if (row >= seq_len) continue;
        const int gx = row * hidden_dim + ch_base;
        x_out[gx + tid * 2]     = smem[r * GROUP_SIZE + tid * 2];
        x_out[gx + tid * 2 + 1] = smem[r * GROUP_SIZE + tid * 2 + 1];
    }
}
