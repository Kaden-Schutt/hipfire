// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// CASK m-folding merge kernel for asym3 K cache (givens-rotated 3-bit).
//
// Each source K row: [4B cnorm][32 threads × 3 bytes packed 3-bit indices]
// = 4 + (head_dim × 3)/8 bytes per head.
//
// Key observation: Givens rotation is orthogonal/linear, so the weighted
// average of unrotated K equals invGivens of the weighted average of
// rotated K. Since we store pre-Givens-applied (i.e. "rotated") values
// and we want to re-store the same format, we can operate ENTIRELY in
// rotated space — no inv-Givens or re-Givens pass needed:
//
//   stuff_i = cnorm_i · rotated_i       (the unnormalized rotated vector)
//   sum     = Σ w_i · stuff_i
//   new_cnorm = ||sum||_2
//   new_rotated = sum / new_cnorm
//   new_idx = turbo_quantize(new_rotated)   // nearest codebook entry
//
// Thread layout (head_dim=256): 32 threads × 8 dims/thread × 4 bands/thread.
// Grid: (budget, n_kv_heads). Block: 32 threads.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void kv_fold_asym3(
    const unsigned char* __restrict__ src,    // [physical × n_kv × bytes_per_head]
    unsigned char* __restrict__ dst,           // [budget × n_kv × bytes_per_head]
    const int* __restrict__ src_indices,       // [budget × m]
    const float* __restrict__ src_weights,     // [budget × m]
    int n_kv,
    int head_dim,
    int m                                      // fold factor
) {
    const int out_slot = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;

    const int bytes_per_head = 4 + (head_dim * 3) / 8;
    const int bytes_per_pos = n_kv * bytes_per_head;

    // Each thread owns 8 consecutive dims (tid*8 .. tid*8+7).
    // Per thread: 4 bands (tid*4 .. tid*4+3).
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // 1. Accumulate Σ_i w_i · cnorm_i · rotated_i in f32.
    for (int i = 0; i < m; ++i) {
        const float w = src_weights[out_slot * m + i];
        if (w == 0.f) continue;
        const int p = src_indices[out_slot * m + i];

        const unsigned char* src_head = src + (size_t)p * bytes_per_pos
                                      + (size_t)kv_head * bytes_per_head;
        const float cnorm = *(const float*)src_head;
        const unsigned char* base = src_head + 4 + tid * 3;
        const unsigned int packed = (unsigned int)base[0]
                                  | ((unsigned int)base[1] << 8)
                                  | ((unsigned int)base[2] << 16);
        const float wc = w * cnorm;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int idx = (packed >> (j * 3)) & 7;
            acc[j] += wc * TURBO_C3_256[idx];
        }
    }

    // 2. Compute new cnorm = ||acc||_2 across all 256 dims via wave32 reduce.
    float lsq = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) lsq += acc[j] * acc[j];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) lsq += __shfl_xor(lsq, off);
    const float new_cnorm = sqrtf(lsq);
    const float inv_cnorm = new_cnorm > 1e-10f ? 1.f / new_cnorm : 0.f;

    // 3. Quantize & pack 8 × 3-bit indices.
    int idx_out[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        idx_out[j] = turbo_quantize_3bit_256(acc[j] * inv_cnorm);
    }

    // 4. Write. Lane 0 writes the cnorm; all lanes write their 3-byte chunk.
    unsigned char* dst_head = dst + (size_t)out_slot * bytes_per_pos
                                  + (size_t)kv_head * bytes_per_head;
    if (tid == 0) {
        *(float*)dst_head = new_cnorm;
    }
    unsigned int packed_out = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        packed_out |= ((unsigned int)(idx_out[j] & 7)) << (j * 3);
    }
    unsigned char* out_base = dst_head + 4 + tid * 3;
    out_base[0] = (unsigned char)(packed_out       & 0xFF);
    out_base[1] = (unsigned char)((packed_out >> 8)  & 0xFF);
    out_base[2] = (unsigned char)((packed_out >> 16) & 0xFF);
}
