// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 2-way fused HFQ6-G256 GEMM for the Qwen3.5 FFN
// preamble (gate + up projections).  gfx1100+ only (RDNA3 wave32 WMMA).
//
// K2 variant: 2x K-tile unroll with software pipelining.
// Processes 2 K-tiles (32 K-elements) per loop body. Loads weights+X for
// tile B while WMMA on tile A is executing, enabling the compiler to
// overlap loads with compute (target: s_waitcnt vmcnt(2)).
//
// Same structure as gemm_gate_up_hfq6g256 but uses WMMA 16x16x16 tiles
// instead of scalar FMA.  X is pre-converted to FP16.
//
// HFQ6-G256 group layout: [f32 scale][f32 zero][192B packed 6-bit] = 200 bytes
// Per K-tile (16 values): 12 bytes (16 × 6/8 = 12).
//
// Grid: [ceil((gate_m + up_m)/16), ceil(N/16)].  Block: [32].  LDS: 0.
//
// Each 16-row tile may straddle the gate/up boundary.  Per-thread output
// routing resolves which Y pointer (Y_gate or Y_up) each row writes to.
// Overwrite semantics (y = acc, not +=).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_hfq6g256_wmma(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,        // [N x K], FP16
    float*         __restrict__ Y_gate,    // [N x gate_m]
    float*         __restrict__ Y_up,      // [N x up_m]
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    // Each thread owns one row in the 16-row tile (tid & 15).
    const int my_row = row_start + (tid & 15);

    // Clamp row into valid range for weight loads (OOB threads will be
    // masked at output time).  Route to gate or up projection.
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);
    const char* A;
    int local_row;
    if (safe_row < gate_m) {
        A = A_gate;
        local_row = safe_row;
    } else {
        A = A_up;
        local_row = safe_row - gate_m;
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 200;
    const char* row_base = A + local_row * row_stride;

    // Clamp batch index for the B-matrix (X) load.
    const int safe_batch = (batch_start + (tid & 15) < N)
                           ? (batch_start + (tid & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // Process 2 K-tiles per iteration (8 iterations covers 16 tiles)
        for (int kt = 0; kt < 16; kt += 2) {
            // === Tile A (kt) ===
            const int byte_off_a = kt * 12;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + byte_off_a);
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = *(const unsigned int*)(dp_a + 3);
            unsigned int d2a = *(const unsigned int*)(dp_a + 6);
            unsigned int d3a = *(const unsigned int*)(dp_a + 9);

            // === Tile B (kt+1) — start loading early ===
            const int byte_off_b = (kt + 1) * 12;
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + byte_off_b);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = *(const unsigned int*)(dp_b + 3);
            unsigned int d2b = *(const unsigned int*)(dp_b + 6);
            unsigned int d3b = *(const unsigned int*)(dp_b + 9);

            // Load both X tiles early (consecutive addresses — prefetcher-friendly)
            half16_t b_a = *(const half16_t*)(x_base + kt * 16);
            half16_t b_b = *(const half16_t*)(x_base + (kt + 1) * 16);

            // Dequant tile A
            half16_t a_reg;
            #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
            DQ6(0,d0a,0); DQ6(1,d0a,6); DQ6(2,d0a,12); DQ6(3,d0a,18);
            DQ6(4,d1a,0); DQ6(5,d1a,6); DQ6(6,d1a,12); DQ6(7,d1a,18);
            DQ6(8,d2a,0); DQ6(9,d2a,6); DQ6(10,d2a,12); DQ6(11,d2a,18);
            DQ6(12,d3a,0); DQ6(13,d3a,6); DQ6(14,d3a,12); DQ6(15,d3a,18);

            // WMMA A — b_b load may still be in flight
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            // Dequant tile B (reuse a_reg register)
            DQ6(0,d0b,0); DQ6(1,d0b,6); DQ6(2,d0b,12); DQ6(3,d0b,18);
            DQ6(4,d1b,0); DQ6(5,d1b,6); DQ6(6,d1b,12); DQ6(7,d1b,18);
            DQ6(8,d2b,0); DQ6(9,d2b,6); DQ6(10,d2b,12); DQ6(11,d2b,18);
            DQ6(12,d3b,0); DQ6(13,d3b,6); DQ6(14,d3b,12); DQ6(15,d3b,18);
            #undef DQ6

            // WMMA B
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }

    // --- Output ---
    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    // So row = 2*j + (tid>>4), col (batch) = tid & 15.
    const int out_col = batch_start + (tid & 15);  // batch index
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                // Route this row to the correct output matrix.
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < gate_m) {
                    Y = Y_gate;  proj_row = out_row;              out_stride = gate_m;
                } else {
                    Y = Y_up;    proj_row = out_row - gate_m;     out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
