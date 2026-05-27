// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// K4 variant of gemm_gate_up_hfq4g256_wmma. Same kernel contract as the
// base k2 sibling — fused gate+up HFQ4-G256 GEMM via WMMA 16x16x16 —
// but uses a 4-tile interleaved pipeline (B prefetch ↔ DQ+WMMA) to keep
// more memory loads in flight per wave. The k2 pipeline holds 2 K-tiles
// in flight (current production kernel at 305 GB/s = 32% of 960 GB/s
// peak on gfx1100 at M=27648, N=3); the k4 variant holds 3-4 in flight
// to give the memory subsystem more concurrent requests per wave to
// hide GDDR6 latency (~300-400 cycles).
//
// Same WMMA C-mapping as k2 (canonical wave32 layout: acc[j] writes
// row = 2*j + (tid>>4), col = tid & 15). Byte-exact-up-to-fp-rounding
// vs k2 (atomic-free accumulation in FP32; rounding may differ due to
// different per-WMMA accumulator ordering across tiles).
//
// Grid/block/LDS identical to k2: grid [ceil((gate_m + up_m)/16),
// ceil(N/16)], block [32], LDS 0. Each 16-row tile may straddle the
// gate/up boundary; per-thread output routing resolves which Y pointer
// to write.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_hfq4g256_wmma_k4(
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
    // masked at output time). Route to gate or up projection.
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
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + local_row * row_stride;

    // Clamp batch index for the B-matrix (X) load.
    const int safe_batch = (batch_start + (tid & 15) < N)
                           ? (batch_start + (tid & 15)) : 0;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;

        // Inline macro: dequant one K-tile worth of weights into a_reg
        // and issue one WMMA against pre-loaded b_reg. Same dequant
        // pattern as the k2 sibling (4-bit unpack + scale + zero-point
        // add per element).
        #define DQ_WMMA(kt_off, b_reg)                                                \
        {                                                                              \
            unsigned int pk0 = *(const unsigned int*)(gp + 8 + (kt_off) * 8);         \
            unsigned int pk1 = *(const unsigned int*)(gp + 8 + (kt_off) * 8 + 4);     \
            half16_t a_reg;                                                            \
            a_reg[0]  = sc_h * (_Float16)(float)((pk0      ) & 0xFu) + zp_h;          \
            a_reg[1]  = sc_h * (_Float16)(float)((pk0 >>  4) & 0xFu) + zp_h;          \
            a_reg[2]  = sc_h * (_Float16)(float)((pk0 >>  8) & 0xFu) + zp_h;          \
            a_reg[3]  = sc_h * (_Float16)(float)((pk0 >> 12) & 0xFu) + zp_h;          \
            a_reg[4]  = sc_h * (_Float16)(float)((pk0 >> 16) & 0xFu) + zp_h;          \
            a_reg[5]  = sc_h * (_Float16)(float)((pk0 >> 20) & 0xFu) + zp_h;          \
            a_reg[6]  = sc_h * (_Float16)(float)((pk0 >> 24) & 0xFu) + zp_h;          \
            a_reg[7]  = sc_h * (_Float16)(float)((pk0 >> 28) & 0xFu) + zp_h;          \
            a_reg[8]  = sc_h * (_Float16)(float)((pk1      ) & 0xFu) + zp_h;          \
            a_reg[9]  = sc_h * (_Float16)(float)((pk1 >>  4) & 0xFu) + zp_h;          \
            a_reg[10] = sc_h * (_Float16)(float)((pk1 >>  8) & 0xFu) + zp_h;          \
            a_reg[11] = sc_h * (_Float16)(float)((pk1 >> 12) & 0xFu) + zp_h;          \
            a_reg[12] = sc_h * (_Float16)(float)((pk1 >> 16) & 0xFu) + zp_h;          \
            a_reg[13] = sc_h * (_Float16)(float)((pk1 >> 20) & 0xFu) + zp_h;          \
            a_reg[14] = sc_h * (_Float16)(float)((pk1 >> 24) & 0xFu) + zp_h;          \
            a_reg[15] = sc_h * (_Float16)(float)((pk1 >> 28) & 0xFu) + zp_h;          \
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);      \
        }

        // 4 K-tiles per iteration with interleaved B prefetch.
        // Pattern mirrors gemm_hfq4g256_residual_wmma_k4: prefetch B[0]
        // and B[1] upfront, then for each subsequent WMMA issue the
        // dequant-and-WMMA against an already-loaded B while loading
        // the NEXT B. Gives ~3 in-flight B loads at steady state.
        for (int kt = 0; kt < 16; kt += 4) {
            half16_t b0 = *(const half16_t*)(xg + kt * 16);
            half16_t b1 = *(const half16_t*)(xg + (kt + 1) * 16);

            DQ_WMMA(kt, b0);

            half16_t b2 = *(const half16_t*)(xg + (kt + 2) * 16);
            DQ_WMMA(kt + 1, b1);

            half16_t b3 = *(const half16_t*)(xg + (kt + 3) * 16);
            DQ_WMMA(kt + 2, b2);

            DQ_WMMA(kt + 3, b3);
        }
        #undef DQ_WMMA
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
