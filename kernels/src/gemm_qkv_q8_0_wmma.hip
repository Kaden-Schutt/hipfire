// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 3-way fused Q8_0 GEMM for the Qwen3.5
// FullAttention preamble (Q + K + V projections from one normed X).
// gfx1100+ only (RDNA3 wave32 WMMA).
//
// Recipe-selected variant per docs/plans/q8-fused-prefill-kernels.md
// §Element format choice (T3-1a microbench: FP16-WMMA wins 11-30× over
// the Tier-2 substrate; INT8-WMMA deferred until-or-unless headroom needed).
//
// Mirrors gemm_qkv_hfq4g256_wmma.hip's structure: 16x16 output tiles via
// wave32 WMMA, register-redundant dequant (no LDS, no __syncthreads),
// per-row routing to the correct weight + output matrix.
//
// Per Q8_0 block (K=32 elements, 34 bytes): 1 fp16 scale + 32 int8 weights.
// One Q8 block = 2 WMMA-K-tiles (each tile is 16 K-elements).
//
// Grid:  [ceil((q_m + k_m + v_m)/16), ceil(N/16)]
// Block: [32] (wave32).  LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_q8_0_wmma(
    const char*     __restrict__ A_q,    // [q_m, K/32 * 34] Q8_0
    const char*     __restrict__ A_k,    // [k_m, K/32 * 34] Q8_0
    const char*     __restrict__ A_v,    // [v_m, K/32 * 34] Q8_0
    const _Float16* __restrict__ X,      // [N, K] FP16
    float*          __restrict__ Y_q,    // [N, q_m] FP32
    float*          __restrict__ Y_k,    // [N, k_m] FP32
    float*          __restrict__ Y_v,    // [N, v_m] FP32
    int q_m, int k_m, int v_m,
    int K, int N
) {
    const int total_m    = q_m + k_m + v_m;
    const int row_start  = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid        = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N)     return;

    // Each thread owns one row in the 16-row tile (tid & 15).
    const int my_row   = row_start + (tid & 15);
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    // Route to the correct weight matrix for this thread's row.
    const char* A;
    int local_row;
    if (safe_row < q_m) {
        A = A_q;  local_row = safe_row;
    } else if (safe_row < q_m + k_m) {
        A = A_k;  local_row = safe_row - q_m;
    } else {
        A = A_v;  local_row = safe_row - q_m - k_m;
    }

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A + (long long)local_row * row_stride;

    const int safe_batch = ((batch_start + (tid & 15)) < N)
                           ? (batch_start + (tid & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // TODO(perf, BW-bound): the HFQ4 sibling at
    // `kernels/src/gemm_qkv_hfq4g256_wmma.hip:78` runs the K-loop with
    // `kt += 2` and prefetches both X tiles before issuing the WMMA pair —
    // overlaps weight load with compute and avoids `s_waitcnt` stalls. Q8
    // is 2× the BW pressure of HFQ4, so this optimization is more likely
    // to matter here. Not done yet — gfx1100 hit 1069.5 tok/s on the
    // current single-tile-at-a-time form, but profile if/when prefill
    // becomes the bottleneck after lm_head batching lands.
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
    const int out_col = batch_start + (tid & 15);
    if (out_col >= N) return;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 2 * j + (tid >> 4);
        if (out_row < total_m) {
            // Route this row to the correct output matrix.
            float* Y;
            int proj_row;
            int proj_m;
            if (out_row < q_m) {
                Y = Y_q;  proj_row = out_row;                 proj_m = q_m;
            } else if (out_row < q_m + k_m) {
                Y = Y_k;  proj_row = out_row - q_m;            proj_m = k_m;
            } else {
                Y = Y_v;  proj_row = out_row - q_m - k_m;      proj_m = v_m;
            }
            Y[(long long)out_col * proj_m + proj_row] = acc[j];
        }
    }
}
