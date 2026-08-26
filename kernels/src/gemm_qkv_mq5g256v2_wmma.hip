// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 3-way fused MQ5G256V2 GEMM for the Qwen3.5
// FullAttention preamble (Q + K + V projections).
// gfx1100+ only (RDNA3 wave32 WMMA).
//
// Same geometry and WMMA contracts as gemm_qkv_mq4g256v2_wmma.hip (MQ4V2 gfx11):
//   Grid: [ceil((q_m + k_m + v_m)/16), ceil(N/16)]. Block: [32]. LDS: 0.
//   half16_t A/B operands, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32,
//   interleaved C mapping acc[j]=C[2*j+(tid>>4)][tid&15].
//   K2 unroll (2 K-tiles per iteration) with software pipelining.
//
// MQ5V2 group contract (168 B, dual fp16 headers, 5-bit payload):
//   [0..4)   : s0 (fp16 lo) | z0 (fp16 hi) → governs weights 0..127
//   [4..8)   : s1 (fp16 lo) | z1 (fp16 hi) → governs weights 128..255
//   [8..168) : 160 bytes of 5-bit packs (8 weights / 5 bytes)
//   Each K-tile is 16 weights = 10 bytes at payload offset kt*10.
//   Half selection: both tiles in a K2 pair share the same half (kt<8 vs kt>=8).
//
// Reference: gemm_qkv_mq4g256v2_wmma.hip (gfx11 architecture) and
// gemm_qkv_mq5g256v2_wmma.gfx12.hip (gfx12 sister, quant layout/decode).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
__launch_bounds__(32, 8)
#else
__launch_bounds__(32, 2)
#endif
extern "C" __global__ void gemm_qkv_mq5g256v2_wmma(
    const char*    __restrict__ A_q,
    const char*    __restrict__ A_k,
    const char*    __restrict__ A_v,
    const _Float16* __restrict__ X,      // [N x K], FP16
    float*         __restrict__ Y_q,     // [N x q_m]
    float*         __restrict__ Y_k,     // [N x k_m]
    float*         __restrict__ Y_v,     // [N x v_m]
    int q_m, int k_m, int v_m,
    int K, int N
) {
    const int total_m = q_m + k_m + v_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    // Each thread owns one row in the 16-row tile (tid & 15).
    const int my_row = row_start + (tid & 15);
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

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 168;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + (tid & 15) < N)
                           ? (batch_start + (tid & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 168;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // Process 2 K-tiles per iteration (8 iterations covers 16 tiles)
        for (int kt = 0; kt < 16; kt += 2) {
            // half selection: both tiles in the K2 pair share the same half (kt<8 vs kt>=8)
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1;

            // === Tile A (kt) ===
            const int k_off_a = kt * 16;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + kt * 10);
            const uint32_t d0a = *(const uint32_t*)(dp_a);
            const uint8_t  d4a = dp_a[4];
            const uint64_t bits0a = ((uint64_t)d0a) | ((uint64_t)d4a << 32);
            const uint32_t d5a = *(const uint32_t*)(dp_a + 5);
            const uint8_t  d9a = dp_a[9];
            const uint64_t bits1a = ((uint64_t)d5a) | ((uint64_t)d9a << 32);

            // === Tile B (kt+1) — start loading early ===
            const int k_off_b = (kt + 1) * 16;
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + (kt + 1) * 10);
            const uint32_t d0b = *(const uint32_t*)(dp_b);
            const uint8_t  d4b = dp_b[4];
            const uint64_t bits0b = ((uint64_t)d0b) | ((uint64_t)d4b << 32);
            const uint32_t d5b = *(const uint32_t*)(dp_b + 5);
            const uint8_t  d9b = dp_b[9];
            const uint64_t bits1b = ((uint64_t)d5b) | ((uint64_t)d9b << 32);

            // Load both X tiles early (consecutive addresses — prefetcher-friendly)
            half16_t b_a = *(const half16_t*)(x_base + k_off_a);
            half16_t b_b = *(const half16_t*)(x_base + k_off_b);

            // Dequant tile A
            half16_t a_reg;
            #define DQ5(i, bits, sh) a_reg[i] = sc_h * (_Float16)(float)(((bits) >> (sh)) & 31u) + zp_h
            DQ5(0,  bits0a,  0); DQ5(1,  bits0a,  5); DQ5(2,  bits0a, 10); DQ5(3,  bits0a, 15);
            DQ5(4,  bits0a, 20); DQ5(5,  bits0a, 25); DQ5(6,  bits0a, 30); DQ5(7,  bits0a, 35);
            DQ5(8,  bits1a,  0); DQ5(9,  bits1a,  5); DQ5(10, bits1a, 10); DQ5(11, bits1a, 15);
            DQ5(12, bits1a, 20); DQ5(13, bits1a, 25); DQ5(14, bits1a, 30); DQ5(15, bits1a, 35);

            // WMMA A — b_b load may still be in flight
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            // Dequant tile B (reuse a_reg register)
            DQ5(0,  bits0b,  0); DQ5(1,  bits0b,  5); DQ5(2,  bits0b, 10); DQ5(3,  bits0b, 15);
            DQ5(4,  bits0b, 20); DQ5(5,  bits0b, 25); DQ5(6,  bits0b, 30); DQ5(7,  bits0b, 35);
            DQ5(8,  bits1b,  0); DQ5(9,  bits1b,  5); DQ5(10, bits1b, 10); DQ5(11, bits1b, 15);
            DQ5(12, bits1b, 20); DQ5(13, bits1b, 25); DQ5(14, bits1b, 30); DQ5(15, bits1b, 35);
            #undef DQ5

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
                if (out_row < q_m) {
                    Y = Y_q;  proj_row = out_row;               out_stride = q_m;
                } else if (out_row < q_m + k_m) {
                    Y = Y_k;  proj_row = out_row - q_m;          out_stride = k_m;
                } else {
                    Y = Y_v;  proj_row = out_row - q_m - k_m;    out_stride = v_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
