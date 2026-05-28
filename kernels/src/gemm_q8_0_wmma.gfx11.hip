// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched Q8_0 GEMM (no residual).
// Y[N, M] = X[N, K] @ A_q8[M, K]^T
//
// gfx11 (RDNA3 / RDNA3.5) variant. Mirrors `gemm_q8_0_residual_wmma`
// (the gfx11 residual sibling) but writes Y directly (= rather than +=) —
// used as a drop-in replacement for the scalar `gemm_q8_0_batched`
// kernel which dominated GPU time in A3B prefill (per the same
// rocprofv3 finding that motivated the gfx12 sibling 2026-05-19).
//
// Sibling of `gemm_q8_0_wmma.gfx12.hip`; that one uses the `_gfx12`
// WMMA builtin + lane-group K split with `half8_t` operands. This
// gfx11 variant uses the `_w32` builtin (no `_gfx12` suffix) and
// `half16_t` operands (register-redundant — the entire WMMA K-tile
// fans out across each lane), matching the convention established by
// the other gfx11 Q8 WMMA siblings (qkv / qkvza / gate_up / residual).
//
// Grid: [ceil(M/16), ceil(N/16), 1]. Block: [32] (wave32). LDS: 0.
//
// Per-block contract: each warp owns a 16-row × 16-batch output tile.
// One Q8 block = 32 K-elements = 2 WMMA-K-tiles (each tile is 16 K).
//
// === Non-overlapping-write invariant (DO NOT BREAK) ===
// The `Y[i] = acc[j]` at the end of this kernel is **not atomic**.
// It is only race-free because each (out_row, out_col) cell is
// written by exactly one lane in exactly one block:
//   Grid:  [ceil(M/16), ceil(N/16)]
//   Block: [32] (wave32),  LDS: 0.
// WMMA output mapping: acc[j] → C[row=2*j + (tid>>4)][col=tid & 15].
// Tiles are 16×16, non-overlapping by construction.
//
// If anyone adds a K-split variant (register-pressure relief), the
// `=` write becomes a data race across blocks for the same
// (out_row, out_col). Switch to a scratch + reduce path, OR to
// `atomicExch` / `atomicAdd` as appropriate.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_q8_0_wmma_gfx11(
    const char*     __restrict__ A_q8,
    const _Float16* __restrict__ X,
    float*          __restrict__ Y,
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
    // Q8 is 2× the BW pressure of HFQ4; this loop already saturates
    // a single tile at a time. If/when this becomes the bottleneck,
    // mirror the K-pair prefetch from the HFQ4 sibling.
    for (int bi = 0; bi < blocks_per_row; bi++) {
        const char* bp = row_base + bi * 34;

        // fp16 scale at byte offset 0 (2 bytes). May be 2-byte aligned but
        // not 4-byte — load via memcpy to avoid unaligned-access UB.
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
    // Direct write (= no residual). Mirrors the qkv/residual sibling
    // mapping, unit-tested against the substrate via the existing
    // `gemm_q8_0_batched_chunked` round-trip.
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
