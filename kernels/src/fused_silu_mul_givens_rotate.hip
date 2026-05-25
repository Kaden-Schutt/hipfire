// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Fused silu(gate) * up + per-channel scale + krot rounds of Givens
// rotation. Single-kernel replacement for the
// `silu_mul_f32 + givens_rotate` pair that the ParoQuant MoE decode
// path used between the routed-expert gate_up GEMV and the routed-
// expert down GEMV.
//
// Structural mirror of MQ4's `fused_silu_mul_rotate_mq` (which fuses
// silu+mul+FWHT in one launch). Empirically the 2-launch decomposition
// produced a small but reproducible numerical delta between direct
// stream execution and hipGraph replay on ParoQuant routed experts
// (gfx1151 / HIP 7.13). Fusing the silu/mul into the same kernel
// closes that gap by eliminating the inter-launch state and matching
// the MQ4 single-kernel reduction pattern.
//
// Grid:  (ceil(seq_len / CTA_M), groups_per_row)
// Block: (GROUP_SIZE / 2) = (64) for GROUP_SIZE=128
// Smem:  CTA_M * GROUP_SIZE * sizeof(float)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <assert.h>

#define CTA_M 4        // rows per block (seq_len dimension)
#define GROUP_SIZE 128 // channels per quantization group
#define HALF_GROUP 64  // threads per group = GROUP_SIZE / 2

extern "C" __global__ void fused_silu_mul_givens_rotate_f32(
    const float* __restrict__ gate,          // [seq_len, hidden_dim]
    const float* __restrict__ up,            // [seq_len, hidden_dim]
    float*       __restrict__ out,           // [seq_len, hidden_dim]
    const short* __restrict__ pairs,         // [krot, hidden_dim]
    const half*  __restrict__ theta,         // [krot, hidden_dim/2]
    const half*  __restrict__ channel_scales,// [hidden_dim]
    const int seq_len,
    const int hidden_dim,
    const int krot
) {
    const int row_block = blockIdx.x;
    const int group_idx = blockIdx.y;
    const int tid = threadIdx.x;  // 0..63

    const int ch_base = group_idx * GROUP_SIZE;
    if (ch_base >= hidden_dim) return;

    extern __shared__ float smem[];  // [CTA_M][GROUP_SIZE]

    const float s0 = __half2float(channel_scales[ch_base + tid * 2]);
    const float s1 = __half2float(channel_scales[ch_base + tid * 2 + 1]);

    // Load gate/up, fuse silu(gate)*up, apply per-channel scale, into smem.
    for (int r = 0; r < CTA_M; r++) {
        const int row = row_block * CTA_M + r;
        if (row >= seq_len) {
            smem[r * GROUP_SIZE + tid * 2]     = 0.0f;
            smem[r * GROUP_SIZE + tid * 2 + 1] = 0.0f;
            continue;
        }
        const int gx = row * hidden_dim + ch_base;
        const float g0 = gate[gx + tid * 2];
        const float g1 = gate[gx + tid * 2 + 1];
        const float u0 = up  [gx + tid * 2];
        const float u1 = up  [gx + tid * 2 + 1];
        const float sm0 = (g0 / (1.0f + expf(-g0))) * u0;
        const float sm1 = (g1 / (1.0f + expf(-g1))) * u1;
        smem[r * GROUP_SIZE + tid * 2]     = sm0 * s0;
        smem[r * GROUP_SIZE + tid * 2 + 1] = sm1 * s1;
    }
    __syncthreads();

    // Apply krot sequential Givens rotation rounds (identical math to
    // `givens_rotate_f32`).
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

    // Store rotated activations to global output buffer.
    for (int r = 0; r < CTA_M; r++) {
        const int row = row_block * CTA_M + r;
        if (row >= seq_len) continue;
        const int gx = row * hidden_dim + ch_base;
        out[gx + tid * 2]     = smem[r * GROUP_SIZE + tid * 2];
        out[gx + tid * 2 + 1] = smem[r * GROUP_SIZE + tid * 2 + 1];
    }
}
