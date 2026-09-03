// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 3-way fused MQ6G256V2 GEMM for the Qwen3.5
// FullAttention preamble (Q + K + V projections).
// gfx1100+ only (RDNA3 wave32 WMMA).
//
// Same geometry and WMMA contracts as gemm_qkv_hfq6g256_wmma.hip (HFQ6-G256):
//   Grid: [ceil((q_m + k_m + v_m)/16), ceil(N/16)]. Block: [32]. LDS: 0.
//   half16_t A/B operands, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32,
//   interleaved C mapping acc[j]=C[2*j+(tid>>4)][tid&15].
//   K2 unroll (2 K-tiles per iteration) with software pipelining.
//
// Differences — MQ6V2 header contract only (200 B groups, dual fp16 headers):
//   [0..4)  : s0 (fp16 lo) | z0 (fp16 hi) → governs weights 0..127 of the group
//   [4..8)  : s1 (fp16 lo) | z1 (fp16 hi) → governs weights 128..255 of the group
//   [8..200): 192 bytes of packed 6-bit codes (256 weights).
//   The 16 K-tiles inside a group split at kt=8: tiles 0..7 use (s0,z0), 8..15 use (s1,z1).
//   Implemented as `kt < 8 ? sc0 : sc1` per K2 iteration (both tiles in a pair share the same half).
//
// 6-bit packing: 4 values per 3 bytes; 12 bytes per K-tile of 16.
// Overlapping uint32 at +0,+3,+6,+9; extract ((dw >> sh) & 63u).
//
// Reference: gemm_qkv_hfq6g256_wmma.hip (gfx11 HFQ6) and
// gemm_qkv_mq6g256v2_wmma.gfx12.hip (gfx12 sister).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
__launch_bounds__(32, 8)
#else
__launch_bounds__(32, 2)
#endif
extern "C" __global__ void gemm_qkv_mq6g256v2_wmma(
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
    const long long row_stride = (long long)groups_per_row * 200;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + (tid & 15) < N)
                           ? (batch_start + (tid & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        for (int kt = 0; kt < 16; kt += 2) {
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1;

            const int byte_off_a = kt * 12;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + byte_off_a);
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = *(const unsigned int*)(dp_a + 3);
            unsigned int d2a = *(const unsigned int*)(dp_a + 6);
            // Bytes 9,10,11 of the tile, NOT a dword at +9. The four 6-bit
            // values this dword feeds live at shifts 0/6/12/18, so only bits
            // 0..23 — bytes 9,10,11 — are ever used; byte 12 is read and
            // discarded. Reading at +9 therefore touches one byte PAST the
            // 12-byte K-tile, and for kt=15 that is one byte past the whole
            // 200-byte group. On the last group of the last row that runs off
            // the end of the weight tensor: measured as an illegal memory
            // access on escha-35b (q_m=8192, k=2048) for any N < 96, which is
            // exactly where this base kernel is reached — N >= 96 routes to
            // the batch-tile path and hid the bug. Load the dword at +8 and
            // shift down instead: same bits 0..23, never leaves the tile.
            unsigned int d3a = (*(const unsigned int*)(dp_a + 8)) >> 8;

            const int byte_off_b = (kt + 1) * 12;
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + byte_off_b);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = *(const unsigned int*)(dp_b + 3);
            unsigned int d2b = *(const unsigned int*)(dp_b + 6);
            unsigned int d3b = (*(const unsigned int*)(dp_b + 8)) >> 8;  // see d3a

            half16_t b_a = *(const half16_t*)(x_base + kt * 16);
            half16_t b_b = *(const half16_t*)(x_base + (kt + 1) * 16);

            half16_t a_reg;
            #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
            DQ6(0, d0a, 0); DQ6(1, d0a, 6); DQ6(2, d0a, 12); DQ6(3, d0a, 18);
            DQ6(4, d1a, 0); DQ6(5, d1a, 6); DQ6(6, d1a, 12); DQ6(7, d1a, 18);
            DQ6(8, d2a, 0); DQ6(9, d2a, 6); DQ6(10, d2a, 12); DQ6(11, d2a, 18);
            DQ6(12, d3a, 0); DQ6(13, d3a, 6); DQ6(14, d3a, 12); DQ6(15, d3a, 18);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            DQ6(0, d0b, 0); DQ6(1, d0b, 6); DQ6(2, d0b, 12); DQ6(3, d0b, 18);
            DQ6(4, d1b, 0); DQ6(5, d1b, 6); DQ6(6, d1b, 12); DQ6(7, d1b, 18);
            DQ6(8, d2b, 0); DQ6(9, d2b, 6); DQ6(10, d2b, 12); DQ6(11, d2b, 18);
            DQ6(12, d3b, 0); DQ6(13, d3b, 6); DQ6(14, d3b, 12); DQ6(15, d3b, 18);
            #undef DQ6

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    const int out_col = batch_start + (tid & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
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
