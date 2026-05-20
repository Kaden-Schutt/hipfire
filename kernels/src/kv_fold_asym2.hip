// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// CASK m-folding merge kernel for asym2 K cache (givens-rotated 2-bit).
//
// Each source K row: [4B cnorm][head_dim/4 packed bytes, 4 values per byte]
//   hd=128 → 36 B/head.  hd=256 → 68 B/head.
//
// Per asym3/asym4 note: weighted average commutes with Givens, so fold
// in rotated space directly.
//
// Codebook: uses TURBO_C2_256 + turbo_quantize_2bit_256 to match what
// `triattn_score_asym2` reads. (The writer `kv_cache_write_asym_k_givens2`
// uses the non-_256 variant; that inconsistency is outside the scope of
// this kernel and tracked separately — fold aligns with the reader side.)
//
// Thread layout: 32 threads, 4 values/thread/half, n_halves = head_dim/128.
// Each thread writes 1 byte per half (4 × 2-bit).
//
// Grid: (budget, n_kv_heads). Block: 32 threads.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void kv_fold_asym2(
    const unsigned char* __restrict__ src,    // [physical × n_kv × bytes_per_head]
    unsigned char* __restrict__ dst,           // [budget × n_kv × bytes_per_head]
    const int* __restrict__ src_indices,       // [budget × m]
    const float* __restrict__ src_weights,     // [budget × m]
    int n_kv,
    int head_dim,
    int m
) {
    const int out_slot = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;

    const int bytes_per_head = 4 + head_dim / 4;
    const int bytes_per_pos = n_kv * bytes_per_head;
    const int n_halves = head_dim / 128;

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // 1. Accumulate Σ_i w_i · cnorm_i · rotated_i.
    for (int i = 0; i < m; ++i) {
        const float w = src_weights[out_slot * m + i];
        if (w == 0.f) continue;
        const int p = src_indices[out_slot * m + i];

        const unsigned char* src_head = src + (size_t)p * bytes_per_pos
                                      + (size_t)kv_head * bytes_per_head;
        const float cnorm = *(const float*)src_head;
        const float wc = w * cnorm;

        for (int half = 0; half < n_halves; ++half) {
            const int read_off = 4 + half * 32 + tid;
            const unsigned char byte = src_head[read_off];
            const int i0 = byte & 0x3;
            const int i1 = (byte >> 2) & 0x3;
            const int i2 = (byte >> 4) & 0x3;
            const int i3 = (byte >> 6) & 0x3;
            const int base = half * 4;
            acc[base + 0] += wc * TURBO_C2_256[i0];
            acc[base + 1] += wc * TURBO_C2_256[i1];
            acc[base + 2] += wc * TURBO_C2_256[i2];
            acc[base + 3] += wc * TURBO_C2_256[i3];
        }
    }

    // 2. New cnorm via wave32 lsq reduce.
    float lsq = 0.f;
    for (int half = 0; half < n_halves; ++half) {
        const int base = half * 4;
        lsq += acc[base + 0] * acc[base + 0];
        lsq += acc[base + 1] * acc[base + 1];
        lsq += acc[base + 2] * acc[base + 2];
        lsq += acc[base + 3] * acc[base + 3];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) lsq += __shfl_xor(lsq, off);
    const float new_cnorm = sqrtf(lsq);
    const float inv_cnorm = new_cnorm > 1e-10f ? 1.f / new_cnorm : 0.f;

    // 3. Quantize against TURBO_C2_256.
    int idx_out[8];
    for (int half = 0; half < n_halves; ++half) {
        const int base = half * 4;
        idx_out[base + 0] = turbo_quantize_2bit_256(acc[base + 0] * inv_cnorm);
        idx_out[base + 1] = turbo_quantize_2bit_256(acc[base + 1] * inv_cnorm);
        idx_out[base + 2] = turbo_quantize_2bit_256(acc[base + 2] * inv_cnorm);
        idx_out[base + 3] = turbo_quantize_2bit_256(acc[base + 3] * inv_cnorm);
    }

    // 4. Write.
    unsigned char* dst_head = dst + (size_t)out_slot * bytes_per_pos
                                  + (size_t)kv_head * bytes_per_head;
    if (tid == 0) {
        *(float*)dst_head = new_cnorm;
    }
    for (int half = 0; half < n_halves; ++half) {
        const int base = half * 4;
        const int write_off = 4 + half * 32 + tid;
        dst_head[write_off] = (unsigned char)(
            (idx_out[base + 3] << 6) |
            (idx_out[base + 2] << 4) |
            (idx_out[base + 1] << 2) |
            (idx_out[base + 0] & 0x3)
        );
    }
}
