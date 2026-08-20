// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Batch-tiled gfx11 HFQ4-G256 fused gate+up GEMM (Qwen3.5 FFN preamble).
// B independent acc chains over the batch (N) dim hide the WMMA latency
// that caps the 1-acc gemm_gate_up_hfq4g256_wmma; weights+dequant loaded
// once per K-tile, reused across the B N-tiles. The per-row gate/up output
// routing is unchanged. Byte-exact vs the 1-acc kernel (independent acc
// chains, same FP accumulation order per chain).
//
// Grid: [ceil((gate_m + up_m)/16), ceil(N/(16*B))]. Block: [32]. LDS: 0.
// Derived from gemm_gate_up_hfq4g256_wmma.hip (Apache-2.0) with the
// batch-tile structure of gemm_qkvza_hfq4g256_wmma_gfx12_bt.hip.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#define DQ16(pk0, pk1, sc, zp) \
    a[0]=sc*(_Float16)(float)(((pk0)>>0)&0xFu)+zp;  a[1]=sc*(_Float16)(float)(((pk0)>>4)&0xFu)+zp; \
    a[2]=sc*(_Float16)(float)(((pk0)>>8)&0xFu)+zp;  a[3]=sc*(_Float16)(float)(((pk0)>>12)&0xFu)+zp; \
    a[4]=sc*(_Float16)(float)(((pk0)>>16)&0xFu)+zp; a[5]=sc*(_Float16)(float)(((pk0)>>20)&0xFu)+zp; \
    a[6]=sc*(_Float16)(float)(((pk0)>>24)&0xFu)+zp; a[7]=sc*(_Float16)(float)(((pk0)>>28)&0xFu)+zp; \
    a[8]=sc*(_Float16)(float)(((pk1)>>0)&0xFu)+zp;  a[9]=sc*(_Float16)(float)(((pk1)>>4)&0xFu)+zp; \
    a[10]=sc*(_Float16)(float)(((pk1)>>8)&0xFu)+zp; a[11]=sc*(_Float16)(float)(((pk1)>>12)&0xFu)+zp; \
    a[12]=sc*(_Float16)(float)(((pk1)>>16)&0xFu)+zp;a[13]=sc*(_Float16)(float)(((pk1)>>20)&0xFu)+zp; \
    a[14]=sc*(_Float16)(float)(((pk1)>>24)&0xFu)+zp;a[15]=sc*(_Float16)(float)(((pk1)>>28)&0xFu)+zp

#define GEN_GATE_UP_BT(BV) \
extern "C" __launch_bounds__(32, 2) __global__ void gemm_gate_up_hfq4g256_wmma_bt##BV( \
    const char*    __restrict__ A_gate, \
    const char*    __restrict__ A_up, \
    const _Float16* __restrict__ X, \
    float*         __restrict__ Y_gate, \
    float*         __restrict__ Y_up, \
    int gate_m, int up_m, int K, int N) { \
    const int total_m = gate_m + up_m; \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * BV); \
    const int tid = threadIdx.x; \
    if (row_start >= total_m || batch_start >= N) return; \
    const int my_row = row_start + (tid & 15); \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1); \
    const char* A; int local_row; \
    if (safe_row < gate_m) { A = A_gate; local_row = safe_row; } \
    else { A = A_up; local_row = safe_row - gate_m; } \
    const int groups_per_row = K / 256; \
    const long long row_stride = (long long)groups_per_row * 136; \
    const char* row_base = A + local_row * row_stride; \
    int sb[BV]; \
    for (int b = 0; b < BV; b++) { \
        int t = batch_start + b * 16 + (tid & 15); \
        sb[b] = (t < N) ? t : 0; \
    } \
    float8_t acc[BV]; \
    for (int b = 0; b < BV; b++) \
        acc[b] = (float8_t){0,0,0,0,0,0,0,0}; \
    for (int g = 0; g < groups_per_row; g++) { \
        const char* gp = row_base + g * 136; \
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp)); \
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
        const _Float16* xg[BV]; \
        for (int b = 0; b < BV; b++) \
            xg[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 4) { \
            half16_t b0[BV], b1[BV]; \
            for (int b = 0; b < BV; b++) { \
                b0[b] = *(const half16_t*)(xg[b] + kt * 16); \
                b1[b] = *(const half16_t*)(xg[b] + (kt + 1) * 16); \
            } \
            { unsigned int pk0 = *(const unsigned int*)(gp + 8 + kt * 8); \
              unsigned int pk1 = *(const unsigned int*)(gp + 8 + kt * 8 + 4); \
              half16_t a; DQ16(pk0, pk1, sc_h, zp_h); \
              for (int b = 0; b < BV; b++) \
                  acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0[b], acc[b]); } \
            half16_t b2[BV]; \
            for (int b = 0; b < BV; b++) \
                b2[b] = *(const half16_t*)(xg[b] + (kt + 2) * 16); \
            { unsigned int pk0 = *(const unsigned int*)(gp + 8 + (kt + 1) * 8); \
              unsigned int pk1 = *(const unsigned int*)(gp + 8 + (kt + 1) * 8 + 4); \
              half16_t a; DQ16(pk0, pk1, sc_h, zp_h); \
              for (int b = 0; b < BV; b++) \
                  acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1[b], acc[b]); } \
            half16_t b3[BV]; \
            for (int b = 0; b < BV; b++) \
                b3[b] = *(const half16_t*)(xg[b] + (kt + 3) * 16); \
            { unsigned int pk0 = *(const unsigned int*)(gp + 8 + (kt + 2) * 8); \
              unsigned int pk1 = *(const unsigned int*)(gp + 8 + (kt + 2) * 8 + 4); \
              half16_t a; DQ16(pk0, pk1, sc_h, zp_h); \
              for (int b = 0; b < BV; b++) \
                  acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b2[b], acc[b]); } \
            { unsigned int pk0 = *(const unsigned int*)(gp + 8 + (kt + 3) * 8); \
              unsigned int pk1 = *(const unsigned int*)(gp + 8 + (kt + 3) * 8 + 4); \
              half16_t a; DQ16(pk0, pk1, sc_h, zp_h); \
              for (int b = 0; b < BV; b++) \
                  acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b3[b], acc[b]); } \
        } \
    } \
    for (int b = 0; b < BV; b++) { \
        const int out_col = batch_start + b * 16 + (tid & 15); \
        if (out_col < N) { \
            _Pragma("unroll") \
            for (int j = 0; j < 8; j++) { \
                int out_row = row_start + 2 * j + (tid >> 4); \
                if (out_row < total_m) { \
                    float* Y; int proj_row; int out_stride; \
                    if (out_row < gate_m) { \
                        Y = Y_gate; proj_row = out_row; out_stride = gate_m; \
                    } else { \
                        Y = Y_up; proj_row = out_row - gate_m; out_stride = up_m; \
                    } \
                    Y[(long long)out_col * out_stride + proj_row] = acc[b][j]; \
                } \
            } \
        } \
    } \
}

GEN_GATE_UP_BT(2)
GEN_GATE_UP_BT(4)

#undef DQ16
#undef GEN_GATE_UP_BT
