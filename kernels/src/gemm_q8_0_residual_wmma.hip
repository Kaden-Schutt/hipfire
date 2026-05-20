// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched Q8_0 GEMM with fused residual add.
// gfx1100+ only (RDNA3 wave32 WMMA, gfx11 only — the `_w32` builtin is
// not selectable on gfx12; that arch needs a `.gfx12.hip` sibling).
//
// Y[N, M] += X[N, K] @ A_q8[M, K]^T
//
// Same recipe as gemm_qkv_q8_0_wmma — register-redundant FP16-WMMA, no LDS.
// Output write uses += semantics so the caller's Y supplies the residual.
//
// === Non-overlapping-write invariant (DO NOT BREAK) ===
// The `Y[i] += acc[j]` at the end of this kernel is **NOT atomic**. It is
// only race-free because each (out_row, out_col) cell is written by exactly
// one lane in exactly one block, under this kernel's grid:
//   Grid:  [ceil(M/16), ceil(N/16)]
//   Block: [32] (wave32),  LDS: 0.
// WMMA output mapping: acc[j] → C[row=2*j + (tid>>4)][col=tid & 15].
// Tiles are 16×16, non-overlapping by construction.
//
// If anyone adds a K-split variant (the HFQ4 family has `_ksplit` for
// register-pressure relief; the Q8 plan §Risks calls it out as a possible
// mitigation), this `+=` becomes a data race across blocks for the same
// (out_row, out_col). Switch to `atomicAdd((float*)&Y[...], acc[j])` in
// any K-split sibling, OR split into a scratch + reduce path.
//
// === Output-mapping divergence from HFQ4 residual (note for maintainers) ===
// `gemm_hfq4g256_residual_wmma.hip` uses a different output-row convention
// (its tile maps acc[j] to C[j + (tid>>4)*8][tid & 15]). The Q8 family here
// uses the qkv-style mapping (acc[j] → C[2*j + (tid>>4)][tid & 15]) instead,
// unit-tested against the substrate. Do not assume the two residual kernels
// are interchangeable templates.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_q8_0_residual_wmma(
    const char*     __restrict__ A_q8,
    const _Float16* __restrict__ X,      // [N, K] FP16
    float*          __restrict__ Y,      // [N, M] FP32 — caller seeds with residual
    int M, int K, int N
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid         = threadIdx.x;

    if (row_start >= M)   return;
    if (batch_start >= N) return;

    const int my_row   = row_start + (tid & 15);
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A_q8 + (long long)safe_row * row_stride;

    const int safe_batch = ((batch_start + (tid & 15)) < N)
                           ? (batch_start + (tid & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // TODO(perf): K-tile prefetch — see gemm_qkv_q8_0_wmma.hip's note.
    for (int bi = 0; bi < blocks_per_row; bi++) {
        const char* bp = row_base + bi * 34;

        _Float16 sc;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const _Float16* x_base = X + (long long)safe_batch * K + bi * 32;

        const signed char* w0 = (const signed char*)(bp + 2);
        half16_t a0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
        }
        half16_t b0 = *(const half16_t*)(x_base);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc);

        const signed char* w1 = (const signed char*)(bp + 2 + 16);
        half16_t a1;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }
        half16_t b1 = *(const half16_t*)(x_base + 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc);
    }

    // Output mapping matches gemm_qkv_q8_0_wmma (unit-tested): acc[j] →
    // C[row=2j+(tid>>4)][col=tid&15]. += semantics fuses the residual.
    const int out_col = batch_start + (tid & 15);
    if (out_col >= N) return;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 2 * j + (tid >> 4);
        if (out_row < M) {
            Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
