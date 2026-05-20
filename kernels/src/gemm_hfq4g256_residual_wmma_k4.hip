// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA GEMM: K4 interleaved — 4 WMMAs per group iteration.
// Unlike K4-upfront (all loads first), this version interleaves:
//   load B[n+1] → dequant A[n] → WMMA[n]
// So each WMMA executes while the NEXT B is loading.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 1)
extern "C" __global__ void gemm_hfq4g256_residual_wmma_k4(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < batch_size) ? (batch_start + (tid & 15)) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;

        #define DQ_WMMA(kt_off, b_reg)                                              \
        {                                                                             \
            unsigned int pk0 = *(const unsigned int*)(gp + 8 + (kt_off) * 8);        \
            unsigned int pk1 = *(const unsigned int*)(gp + 8 + (kt_off) * 8 + 4);    \
            half16_t a_reg;                                                           \
            a_reg[0]  = sc_h * (_Float16)(float)((pk0      ) & 0xFu) + zp_h;         \
            a_reg[1]  = sc_h * (_Float16)(float)((pk0 >>  4) & 0xFu) + zp_h;         \
            a_reg[2]  = sc_h * (_Float16)(float)((pk0 >>  8) & 0xFu) + zp_h;         \
            a_reg[3]  = sc_h * (_Float16)(float)((pk0 >> 12) & 0xFu) + zp_h;         \
            a_reg[4]  = sc_h * (_Float16)(float)((pk0 >> 16) & 0xFu) + zp_h;         \
            a_reg[5]  = sc_h * (_Float16)(float)((pk0 >> 20) & 0xFu) + zp_h;         \
            a_reg[6]  = sc_h * (_Float16)(float)((pk0 >> 24) & 0xFu) + zp_h;         \
            a_reg[7]  = sc_h * (_Float16)(float)((pk0 >> 28) & 0xFu) + zp_h;         \
            a_reg[8]  = sc_h * (_Float16)(float)((pk1      ) & 0xFu) + zp_h;         \
            a_reg[9]  = sc_h * (_Float16)(float)((pk1 >>  4) & 0xFu) + zp_h;         \
            a_reg[10] = sc_h * (_Float16)(float)((pk1 >>  8) & 0xFu) + zp_h;         \
            a_reg[11] = sc_h * (_Float16)(float)((pk1 >> 12) & 0xFu) + zp_h;         \
            a_reg[12] = sc_h * (_Float16)(float)((pk1 >> 16) & 0xFu) + zp_h;         \
            a_reg[13] = sc_h * (_Float16)(float)((pk1 >> 20) & 0xFu) + zp_h;         \
            a_reg[14] = sc_h * (_Float16)(float)((pk1 >> 24) & 0xFu) + zp_h;         \
            a_reg[15] = sc_h * (_Float16)(float)((pk1 >> 28) & 0xFu) + zp_h;         \
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);     \
        }

        for (int kt = 0; kt < 16; kt += 4) {
            // Prefetch B[0] and B[1]
            half16_t b0 = *(const half16_t*)(xg + kt * 16);
            half16_t b1 = *(const half16_t*)(xg + (kt+1) * 16);

            // Dequant A[0] + WMMA[0] — while b1 load completes
            DQ_WMMA(kt, b0);

            // Prefetch B[2]
            half16_t b2 = *(const half16_t*)(xg + (kt+2) * 16);
            // Dequant A[1] + WMMA[1] — while b2 load completes
            DQ_WMMA(kt+1, b1);

            // Prefetch B[3]
            half16_t b3 = *(const half16_t*)(xg + (kt+3) * 16);
            // Dequant A[2] + WMMA[2] — while b3 load completes
            DQ_WMMA(kt+2, b2);

            // Dequant A[3] + WMMA[3]
            DQ_WMMA(kt+3, b3);
        }
        #undef DQ_WMMA
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    // So row = 2*j + (tid>>4), col (batch) = tid & 15. Same canonical
    // mapping as gemm_hfq4g256_residual_wmma_k2.hip (case-studies §4 /
    // commit b7ac66a fix). Phase −1 of plans/k4_plan.md tests whether
    // this is the K4 bug.
    const int out_col = batch_start + (tid & 15);  // batch index
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
