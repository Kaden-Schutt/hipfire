// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx11 (RDNA3) HFQ4-G256 batched lm_head GEMM.
//
// Overwrite-semantics sibling of gemm_hfq4g256_residual_wmma_k2 (the gfx11
// default-dispatched residual kernel at lm_head shape band, M>=8192). Same
// math, same tile mapping, but writes Y = W·X instead of accumulating
// Y += W·X — DFlash and MTP lm_head call sites only need logits, so the
// residual kernel's pre-zero memset and the per-tile RMW are avoidable work
// at the largest matrix in the decode cycle.
//
// Sister of gemm_hfq4g256_lmhead_wmma.gfx12.hip (RDNA4). Differences from
// the gfx12 sibling (gfx11 WMMA dialect):
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
//      (no _gfx12 suffix).
//   2. A and B operands: half16_t (16 fp16 per lane, K replicated across all
//      32 lanes — gfx11 wave32 WMMA layout) instead of half8_t with K split
//      across 2 lane-groups.
//   3. C-output mapping (rows interleaved by 2):
//        acc[j] = C[2*j + (tid>>4)][tid & 15]
//      A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
//      So row = 2*j + (tid>>4), col (batch) = tid & 15.
//
// **Output-mapping note**: this kernel uses the same C-output mapping as
// `gemm_hfq4g256_residual_wmma_k2`. The unsuffixed base
// `gemm_hfq4g256_residual_wmma` kernel has a KNOWN output-mapping bug
// (see dispatch.rs comments around HIPFIRE_WO_WMMA_VARIANT — flagged as
// "debug only"). Do not copy from the unsuffixed base.
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_lmhead_wmma_gfx11(
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

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;

        #pragma unroll
        for (int kt = 0; kt < 16; kt++) {
            const int k_off = kt * 16;

            unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2);
            unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(0,  pk0,  0); DQ(1,  pk0,  4); DQ(2,  pk0,  8); DQ(3,  pk0, 12);
            DQ(4,  pk0, 16); DQ(5,  pk0, 20); DQ(6,  pk0, 24); DQ(7,  pk0, 28);
            DQ(8,  pk1,  0); DQ(9,  pk1,  4); DQ(10, pk1,  8); DQ(11, pk1, 12);
            DQ(12, pk1, 16); DQ(13, pk1, 20); DQ(14, pk1, 24); DQ(15, pk1, 28);
            #undef DQ

            half16_t b_reg = *(const half16_t*)(xg + k_off);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
    }

    // gfx11 wave32 WMMA output mapping (rows interleaved by 2, same as k2):
    //   acc[j] = C[2*j + (tid>>4)][tid & 15]
    // -> row = 2*j + (tid>>4) (M index), col (batch) = tid & 15 (N index).
    // Y is [N × M] row-major: Y[batch][m] = Y[batch*M + m]. Overwrite
    // semantics (=) — caller does NOT pre-zero Y.
    const int out_col = batch_start + (tid & 15);  // batch index (N)
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);  // M index
            if (out_row < M)
                Y[(long long)out_col * M + out_row] = acc[j];
        }
    }
}
