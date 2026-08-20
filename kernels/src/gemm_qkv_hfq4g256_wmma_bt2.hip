// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
//
// Batch-tiled B=2 gfx11 HFQ4-G256 fused Q+K+V GEMM (Qwen3.5 FullAttention
// preamble). 2 independent acc chains over the batch (N) dim hide the WMMA
// latency that caps the 1-acc gemm_qkv_hfq4g256_wmma; weights+dequant loaded
// once per K-tile, reused across both N-tiles. The per-row Q/K/V output
// routing is unchanged. Byte-exact vs the 1-acc kernel (independent acc
// chains, same FP accumulation order per chain).
//
// Grid: [ceil((q_m + k_m + v_m)/16), ceil(N/32)]. Block: [32]. LDS: 0.
// Derived from gemm_qkv_hfq4g256_wmma.hip (Apache-2.0) with the batch-tile
// structure of gemm_gate_up_hfq4g256_wmma_bt.hip.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

extern "C" __launch_bounds__(32, 2) __global__ void gemm_qkv_hfq4g256_wmma_bt2(
    const char*    __restrict__ A_q,
    const char*    __restrict__ A_k,
    const char*    __restrict__ A_v,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_q,
    float*         __restrict__ Y_k,
    float*         __restrict__ Y_v,
    int q_m, int k_m, int v_m,
    int K, int N
) {
    const int total_m = q_m + k_m + v_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 32;
    const int tid = threadIdx.x;

    if (row_start >= total_m || batch_start >= N) return;

    const int my_row = row_start + (tid & 15);
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < q_m) {
        A = A_q;  local_row = safe_row;
    } else if (safe_row < q_m + k_m) {
        A = A_k;  local_row = safe_row - q_m;
    } else {
        A = A_v;  local_row = safe_row - q_m - k_m;
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + local_row * row_stride;

    int sb0 = batch_start + (tid & 15);
    int sb1 = batch_start + 16 + (tid & 15);
    if (sb0 >= N) sb0 = 0;
    if (sb1 >= N) sb1 = 0;

    const _Float16* xg0_base = X + (long long)sb0 * K;
    const _Float16* xg1_base = X + (long long)sb1 * K;

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg0 = xg0_base + g * 256;
        const _Float16* xg1 = xg1_base + g * 256;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);

            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            half16_t b0_a = *(const half16_t*)(xg0 + k_off_a);
            half16_t b0_b = *(const half16_t*)(xg0 + k_off_b);
            half16_t b1_a = *(const half16_t*)(xg1 + k_off_a);
            half16_t b1_b = *(const half16_t*)(xg1 + k_off_b);

            half16_t a_a;
            #define DQ(reg, i, pk, sh) reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(a_a, 0,  pk0a,  0); DQ(a_a, 1,  pk0a,  4); DQ(a_a, 2,  pk0a,  8); DQ(a_a, 3,  pk0a, 12);
            DQ(a_a, 4,  pk0a, 16); DQ(a_a, 5,  pk0a, 20); DQ(a_a, 6,  pk0a, 24); DQ(a_a, 7,  pk0a, 28);
            DQ(a_a, 8,  pk1a,  0); DQ(a_a, 9,  pk1a,  4); DQ(a_a, 10, pk1a,  8); DQ(a_a, 11, pk1a, 12);
            DQ(a_a, 12, pk1a, 16); DQ(a_a, 13, pk1a, 20); DQ(a_a, 14, pk1a, 24); DQ(a_a, 15, pk1a, 28);

            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b0_a, acc0);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b1_a, acc1);

            DQ(a_a, 0,  pk0b,  0); DQ(a_a, 1,  pk0b,  4); DQ(a_a, 2,  pk0b,  8); DQ(a_a, 3,  pk0b, 12);
            DQ(a_a, 4,  pk0b, 16); DQ(a_a, 5,  pk0b, 20); DQ(a_a, 6,  pk0b, 24); DQ(a_a, 7,  pk0b, 28);
            DQ(a_a, 8,  pk1b,  0); DQ(a_a, 9,  pk1b,  4); DQ(a_a, 10, pk1b,  8); DQ(a_a, 11, pk1b, 12);
            DQ(a_a, 12, pk1b, 16); DQ(a_a, 13, pk1b, 20); DQ(a_a, 14, pk1b, 24); DQ(a_a, 15, pk1b, 28);
            #undef DQ

            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b0_b, acc0);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b1_b, acc1);
        }
    }

    // Output: RDNA3 wave32 WMMA C-map: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // row = 2*j + (tid>>4), col (batch) = tid & 15.
    const int out_col0 = batch_start + (tid & 15);
    const int out_col1 = batch_start + 16 + (tid & 15);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int out_row = row_start + 2 * j + (tid >> 4);
        if (out_row < total_m) {
            float* Y; int proj_row; int out_stride;
            if (out_row < q_m) {
                Y = Y_q;  proj_row = out_row;               out_stride = q_m;
            } else if (out_row < q_m + k_m) {
                Y = Y_k;  proj_row = out_row - q_m;          out_stride = k_m;
            } else {
                Y = Y_v;  proj_row = out_row - q_m - k_m;    out_stride = v_m;
            }
            if (out_col0 < N)
                Y[(long long)out_col0 * out_stride + proj_row] = acc0[j];
            if (out_col1 < N)
                Y[(long long)out_col1 * out_stride + proj_row] = acc1[j];
        }
    }
}
