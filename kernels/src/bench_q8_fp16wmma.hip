// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// EXPERIMENTAL / REGRESSION PROBE — Q8_0 batched GEMM via FP16 WMMA (RDNA3+).
//
// Microbench for the Tier 3 recipe pick (see docs/plans/q8-fused-prefill-kernels.md
// §Element format choice — FP16 vs INT8 WMMA). NOT a production kernel — this
// is the source template the 4 production fused kernels were derived from
// (gemm_qkv/qkvza/gate_up/q8_0_residual_wmma). Kept in tree as a regression
// probe: re-run via `bench_q8_wmma_variants.rs` if anyone touches the dequant
// prologue or WMMA inner loop, to confirm the 11–30× substrate speedup
// hasn't eroded. INT8-WMMA sibling lives in bench_q8_int8wmma.hip (deferred).
//
// Y[N, M] = X[N, K] @ A_q8[M, K]^T
//   A: Q8_0 packed weights. Per K=32 block: [2B fp16 scale | 32B int8 weights],
//      34 bytes/block, K must be a multiple of 32.
//   X: FP16 [N, K] row-major (pre-converted on host or via convert_f32_to_f16).
//   Y: FP32 [N, M] row-major.
//
// Recipe: register-redundant dequant (no LDS), mirrors gemm_qkv_hfq4g256_wmma.
// Per Q8 block (K=32): 1 scale load, 2 WMMA tiles (16 K-elems each), inline
// int8→fp16 cast scaled by the block scale.
//
// Grid:  [ceil(M/16), ceil(N/16)]
// Block: [32] (wave32, single wave per workgroup).
// LDS:   0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void bench_q8_fp16wmma(
    const char*     __restrict__ A_q8,    // [M, K/32 * 34] packed Q8_0
    const _Float16* __restrict__ X,       // [N, K] FP16
    float*          __restrict__ Y,       // [N, M] FP32
    int M, int K, int N
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid         = threadIdx.x;

    if (row_start >= M)   return;
    if (batch_start >= N) return;

    // Each thread owns one row in the 16-row tile (tid & 15).
    const int my_row   = row_start + (tid & 15);
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A_q8 + (long long)safe_row * row_stride;

    // Each thread also owns one batch col (tid & 15) for the X load.
    const int safe_batch = ((batch_start + (tid & 15)) < N)
                           ? (batch_start + (tid & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const char* bp = row_base + bi * 34;

        // fp16 scale (offset 0, 2 bytes — may be unaligned across blocks)
        _Float16 sc;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const _Float16* x_base = X + (long long)safe_batch * K + bi * 32;

        // === WMMA tile 0 (K-elements 0..16) ===
        const signed char* w0 = (const signed char*)(bp + 2);
        half16_t a0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
        }
        half16_t b0 = *(const half16_t*)(x_base);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc);

        // === WMMA tile 1 (K-elements 16..32) ===
        const signed char* w1 = (const signed char*)(bp + 2 + 16);
        half16_t a1;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }
        half16_t b1 = *(const half16_t*)(x_base + 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc);
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    const int out_col = batch_start + (tid & 15);
    if (out_col >= N) return;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 2 * j + (tid >> 4);
        if (out_row < M) {
            Y[(long long)out_col * M + out_row] = acc[j];
        }
    }
}
