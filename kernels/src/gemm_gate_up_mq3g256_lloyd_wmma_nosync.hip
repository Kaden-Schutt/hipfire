// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// Barrier-free variant of gemm_gate_up_mq3g256_lloyd_wmma.
// Each lane loads its own row's codebook directly from global memory
// instead of cooperative LDS staging, eliminating both __syncthreads()
// calls. Same WMMA arithmetic; different memory schedule.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_mq3g256_lloyd_wmma_nosync(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int my_tile_row = tid & 15;
    const int my_row = row_start + my_tile_row;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < gate_m) {
        A = A_gate; local_row = safe_row;
    } else {
        A = A_up;   local_row = safe_row - gate_m;
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 112;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + my_tile_row < N)
                           ? (batch_start + my_tile_row) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        // Each lane loads its own row's codebook (8 FP16 entries) from global.
        // Load as half[8] via half16_t and use only the low 8 entries.
        const __half* cb_h = (const __half*)(row_base + g * 112);
        half16_t cb = {0};
        cb[0] = cb_h[0]; cb[1] = cb_h[1]; cb[2] = cb_h[2]; cb[3] = cb_h[3];
        cb[4] = cb_h[4]; cb[5] = cb_h[5]; cb[6] = cb_h[6]; cb[7] = cb_h[7];

        const char* gp = row_base + g * 112;
        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 16);

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const unsigned char* dpa = dp + kt * 6;
            const int k_off_b = (kt + 1) * 16;
            const unsigned char* dpb = dp + (kt + 1) * 6;

            half16_t b_a = *(const half16_t*)(x_base + k_off_a);
            half16_t b_b = *(const half16_t*)(x_base + k_off_b);

            unsigned int a0 = dpa[0], a1 = dpa[1], a2 = dpa[2];
            unsigned int a3 = dpa[3], a4 = dpa[4], a5 = dpa[5];

            unsigned int q0  = a0 & 7u;
            unsigned int q1  = (a0 >> 3) & 7u;
            unsigned int q2  = ((a0 >> 6) | (a1 << 2)) & 7u;
            unsigned int q3  = (a1 >> 1) & 7u;
            unsigned int q4  = (a1 >> 4) & 7u;
            unsigned int q5  = ((a1 >> 7) | (a2 << 1)) & 7u;
            unsigned int q6  = (a2 >> 2) & 7u;
            unsigned int q7  = (a2 >> 5) & 7u;
            unsigned int q8  = a3 & 7u;
            unsigned int q9  = (a3 >> 3) & 7u;
            unsigned int q10 = ((a3 >> 6) | (a4 << 2)) & 7u;
            unsigned int q11 = (a4 >> 1) & 7u;
            unsigned int q12 = (a4 >> 4) & 7u;
            unsigned int q13 = ((a4 >> 7) | (a5 << 1)) & 7u;
            unsigned int q14 = (a5 >> 2) & 7u;
            unsigned int q15 = (a5 >> 5) & 7u;

            half16_t a_reg;
            #define DQ(i, q) a_reg[i] = cb[(q)]
            DQ(0, q0);   DQ(1, q1);   DQ(2, q2);   DQ(3, q3);
            DQ(4, q4);   DQ(5, q5);   DQ(6, q6);   DQ(7, q7);
            DQ(8, q8);   DQ(9, q9);   DQ(10, q10); DQ(11, q11);
            DQ(12, q12); DQ(13, q13); DQ(14, q14); DQ(15, q15);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            unsigned int B0 = dpb[0], B1 = dpb[1], B2 = dpb[2];
            unsigned int B3 = dpb[3], B4 = dpb[4], B5 = dpb[5];

            unsigned int Q0  = B0 & 7u;
            unsigned int Q1  = (B0 >> 3) & 7u;
            unsigned int Q2  = ((B0 >> 6) | (B1 << 2)) & 7u;
            unsigned int Q3  = (B1 >> 1) & 7u;
            unsigned int Q4  = (B1 >> 4) & 7u;
            unsigned int Q5  = ((B1 >> 7) | (B2 << 1)) & 7u;
            unsigned int Q6  = (B2 >> 2) & 7u;
            unsigned int Q7  = (B2 >> 5) & 7u;
            unsigned int Q8  = B3 & 7u;
            unsigned int Q9  = (B3 >> 3) & 7u;
            unsigned int Q10 = ((B3 >> 6) | (B4 << 2)) & 7u;
            unsigned int Q11 = (B4 >> 1) & 7u;
            unsigned int Q12 = (B4 >> 4) & 7u;
            unsigned int Q13 = ((B4 >> 7) | (B5 << 1)) & 7u;
            unsigned int Q14 = (B5 >> 2) & 7u;
            unsigned int Q15 = (B5 >> 5) & 7u;

            DQ(0, Q0);   DQ(1, Q1);   DQ(2, Q2);   DQ(3, Q3);
            DQ(4, Q4);   DQ(5, Q5);   DQ(6, Q6);   DQ(7, Q7);
            DQ(8, Q8);   DQ(9, Q9);   DQ(10, Q10); DQ(11, Q11);
            DQ(12, Q12); DQ(13, Q13); DQ(14, Q14); DQ(15, Q15);
            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }

    const int out_col = batch_start + my_tile_row;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < gate_m) {
                    Y = Y_gate; proj_row = out_row;            out_stride = gate_m;
                } else {
                    Y = Y_up;   proj_row = out_row - gate_m;   out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
