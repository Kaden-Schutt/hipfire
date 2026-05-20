// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 2-way fused Q8_0 GEMM for the Qwen3.5 FFN
// preamble (w_gate + w_up projections from the same normed X).
// gfx1100+ only (RDNA3 wave32 WMMA).
//
// Same recipe as gemm_qkv_q8_0_wmma — register-redundant FP16-WMMA, no LDS.
// Per-thread output routes to Y_gate or Y_up based on row.
//
// Grid:  [ceil((gate_m + up_m)/16), ceil(N/16)]
// Block: [32] (wave32).  LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_q8_0_wmma(
    const char*     __restrict__ A_gate,
    const char*     __restrict__ A_up,
    const _Float16* __restrict__ X,         // [N, K] FP16
    float*          __restrict__ Y_gate,    // [N, gate_m]
    float*          __restrict__ Y_up,      // [N, up_m]
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m    = gate_m + up_m;
    const int row_start  = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid        = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N)     return;

    const int my_row   = row_start + (tid & 15);
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < gate_m) {
        A = A_gate;  local_row = safe_row;
    } else {
        A = A_up;    local_row = safe_row - gate_m;
    }

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A + (long long)local_row * row_stride;

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

    const int out_col = batch_start + (tid & 15);
    if (out_col >= N) return;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 2 * j + (tid >> 4);
        if (out_row < total_m) {
            float* Y;
            int proj_row;
            int proj_m;
            if (out_row < gate_m) {
                Y = Y_gate; proj_row = out_row;            proj_m = gate_m;
            } else {
                Y = Y_up;   proj_row = out_row - gate_m;   proj_m = up_m;
            }
            Y[(long long)out_col * proj_m + proj_row] = acc[j];
        }
    }
}
