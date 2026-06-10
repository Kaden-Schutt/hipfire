// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batch-tiled gfx12 HFQ4-G256 residual GEMM (fused residual ADD: Y += W@X).
// B independent acc chains over the batch (N) dim hide the WMMA latency that
// caps the 1-acc gemm_hfq4g256_residual_wmma_gfx12; weights+dequant loaded once
// per K-tile, reused across the B N-tiles. Byte-exact vs the 1-acc kernel.
// Grid: [ceil(M/16), ceil(batch/(16*B))]. Block: [32]. LDS: 0. Output is +=
// (caller pre-inits Y with the residual), same as the 1-acc kernel.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;
#define HFQ4_DQ8(pk, sc, zp) \
    a[0]=sc*(_Float16)(float)(((pk)>>0)&0xFu)+zp; a[1]=sc*(_Float16)(float)(((pk)>>4)&0xFu)+zp; \
    a[2]=sc*(_Float16)(float)(((pk)>>8)&0xFu)+zp; a[3]=sc*(_Float16)(float)(((pk)>>12)&0xFu)+zp; \
    a[4]=sc*(_Float16)(float)(((pk)>>16)&0xFu)+zp; a[5]=sc*(_Float16)(float)(((pk)>>20)&0xFu)+zp; \
    a[6]=sc*(_Float16)(float)(((pk)>>24)&0xFu)+zp; a[7]=sc*(_Float16)(float)(((pk)>>28)&0xFu)+zp

#define GEN_RESID_BT(BV) \
extern "C" __launch_bounds__(32, 2) __global__ void gemm_hfq4g256_residual_wmma_gfx12_bt##BV( \
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
    int M, int K, int batch_size) { \
    const int tid = threadIdx.x, rs = blockIdx.x * 16, bs = blockIdx.y * (16 * BV); \
    if (rs >= M || bs >= batch_size) return; \
    const int ml = tid & 15, kg = tid >> 4; \
    const int sr = (rs + ml < M) ? (rs + ml) : (M - 1); \
    const int gpr = K / 256; const char* rb = A + (long long)sr * gpr * 136; \
    int sb[BV]; for (int b = 0; b < BV; b++) { int t = bs + b * 16 + ml; sb[b] = (t < batch_size) ? t : 0; } \
    float8_t acc[BV]; for (int b = 0; b < BV; b++) acc[b] = float8_t{0,0,0,0,0,0,0,0}; \
    for (int g = 0; g < gpr; g++) { \
        const char* gp = rb + g * 136; \
        _Float16 sc = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp)); \
        _Float16 zp = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
        const _Float16* xg[BV]; for (int b = 0; b < BV; b++) xg[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 4) { \
            unsigned int p[4] = { *(const unsigned int*)(gp+8+(kt+0)*8+kg*4), *(const unsigned int*)(gp+8+(kt+1)*8+kg*4), \
                                  *(const unsigned int*)(gp+8+(kt+2)*8+kg*4), *(const unsigned int*)(gp+8+(kt+3)*8+kg*4) }; \
            _Pragma("unroll") \
            for (int ki = 0; ki < 4; ki++) { half8_t a; HFQ4_DQ8(p[ki], sc, zp); \
                _Pragma("unroll") \
                for (int b = 0; b < BV; b++) { half8_t bb = *(const half8_t*)(xg[b] + (kt+ki)*16 + kg*8); \
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc[b]); } } } \
    } \
    for (int b = 0; b < BV; b++) { const int oc = bs + b * 16 + ml; if (oc < batch_size) for (int j = 0; j < 8; j++) { \
        int orow = rs + 8 * kg + j; if (orow < M) Y[(long long)oc * M + orow] += acc[b][j]; } } \
}

GEN_RESID_BT(4)
GEN_RESID_BT(8)
GEN_RESID_BT(12)
