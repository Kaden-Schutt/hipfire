// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Wider-N variant of `gemm_q8_0_wmma`: 16×64 output tile per workgroup
// instead of 16×16. Same single-warp WMMA structure, but each K-step
// issues 4 back-to-back WMMA tiles sharing one A (weight) fragment
// against 4 different B (batch) fragments. Quadruples per-block weight
// reuse — the missing structural lever exposed by the profiler at
// 7.3 GiB/s effective on gemm_q8_0_wmma (vs 211 GB/s DRAM peak,
// 30× below cap — see commit 84b7630 commit body and llama.cpp issue
// 21284 / pedapudi for the analysis that prompted this rewrite).
//
// Reuse arithmetic (per K-step of 32 K-elements, group of 2 WMMA tiles):
//   old (16×16): 1 weight chunk + 1 X chunk → 256 output cells
//   new (16×64): 1 weight chunk + 4 X chunks → 1024 output cells
//                ⇒ 4× lower weight-bytes-per-output-cell pressure
//
// VGPR cost: 4 × float8_t acc = 32 VGPRs (was 8), 4 × half16_t B_frag
// = 32 VGPRs (was 8). Net ~+48 over the baseline ~70 VGPRs. We drop
// launch_bounds back to (32, 2) (was bumped to (32, 8) in the baseline)
// since the higher VGPR pressure rules out (32, 8) — but the 4× reuse
// per block more than compensates for the lower occupancy.
//
// Grid:  [ceil(M/16), ceil(N/64)]
// Block: [32]  (wave32, single wave per workgroup)
// LDS:   0

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_q8_0_wmma_x64(
    const char*     __restrict__ A_q8,    // [M, K/32 * 34] packed Q8_0
    const _Float16* __restrict__ X,       // [N, K] FP16
    float*          __restrict__ Y,       // [N, M] FP32
    int M, int K, int N
) {
    const int row_start     = blockIdx.x * 16;
    const int batch_start64 = blockIdx.y * 64;
    const int tid           = threadIdx.x;

    if (row_start   >= M) return;
    if (batch_start64 >= N) return;

    const int my_row   = row_start + (tid & 15);
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A_q8 + (long long)safe_row * row_stride;

    // 4 batch tiles per WG. Each WMMA's "lane mapping" puts batch col
    // = (tid & 15) at the local sub-tile, so the 4 tiles cover cols
    // [batch_start64+0..16), [+16..32), [+32..48), [+48..64).
    const int b0 = batch_start64 + 0  + (tid & 15);
    const int b1 = batch_start64 + 16 + (tid & 15);
    const int b2 = batch_start64 + 32 + (tid & 15);
    const int b3 = batch_start64 + 48 + (tid & 15);
    const int sb0 = (b0 < N) ? b0 : 0;
    const int sb1 = (b1 < N) ? b1 : 0;
    const int sb2 = (b2 < N) ? b2 : 0;
    const int sb3 = (b3 < N) ? b3 : 0;

    float8_t acc0 = {0,0,0,0,0,0,0,0};
    float8_t acc1 = {0,0,0,0,0,0,0,0};
    float8_t acc2 = {0,0,0,0,0,0,0,0};
    float8_t acc3 = {0,0,0,0,0,0,0,0};

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const char* bp = row_base + bi * 34;

        _Float16 sc;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const _Float16* x_base0 = X + (long long)sb0 * K + bi * 32;
        const _Float16* x_base1 = X + (long long)sb1 * K + bi * 32;
        const _Float16* x_base2 = X + (long long)sb2 * K + bi * 32;
        const _Float16* x_base3 = X + (long long)sb3 * K + bi * 32;

        // ── A fragment shared across all 4 WMMA tiles in this K-step ──
        // Build twice: once for K[0..16] (a0) and once for K[16..32] (a1).

        // a0 (K-elements 0..15)
        const signed char* w0 = (const signed char*)(bp + 2);
        half16_t a0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
        }

        // Load 4 B fragments for K[0..15], one per batch tile.
        half16_t b0_0 = *(const half16_t*)(x_base0);
        half16_t b0_1 = *(const half16_t*)(x_base1);
        half16_t b0_2 = *(const half16_t*)(x_base2);
        half16_t b0_3 = *(const half16_t*)(x_base3);

        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0_0, acc0);
        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0_1, acc1);
        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0_2, acc2);
        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0_3, acc3);

        // a1 (K-elements 16..31)
        const signed char* w1 = (const signed char*)(bp + 2 + 16);
        half16_t a1;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }

        half16_t b1_0 = *(const half16_t*)(x_base0 + 16);
        half16_t b1_1 = *(const half16_t*)(x_base1 + 16);
        half16_t b1_2 = *(const half16_t*)(x_base2 + 16);
        half16_t b1_3 = *(const half16_t*)(x_base3 + 16);

        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1_0, acc0);
        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1_1, acc1);
        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1_2, acc2);
        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1_3, acc3);
    }

    // Output write. WMMA lane layout: acc[j] on lane L holds C[2*j + (L>>4)][L & 15].
    // For tile k ∈ {0,1,2,3}: output col = batch_start64 + 16*k + (tid & 15).
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const int out_col = batch_start64 + 16 * k + (tid & 15);
        if (out_col >= N) continue;
        const float8_t* acc_k = (k == 0) ? &acc0 : (k == 1) ? &acc1 : (k == 2) ? &acc2 : &acc3;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = (*acc_k)[j];
            }
        }
    }
}
