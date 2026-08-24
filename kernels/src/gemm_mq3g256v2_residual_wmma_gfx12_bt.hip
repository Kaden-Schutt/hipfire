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
#define MQ3_DQ8(b0,b1,b2, sc, zp) \
    a[0]=sc*(_Float16)(float)((b0)&7u)+zp; a[1]=sc*(_Float16)(float)(((b0)>>3)&7u)+zp; \
    a[2]=sc*(_Float16)(float)((((b0)>>6)|((b1)<<2))&7u)+zp; a[3]=sc*(_Float16)(float)(((b1)>>1)&7u)+zp; \
    a[4]=sc*(_Float16)(float)(((b1)>>4)&7u)+zp; a[5]=sc*(_Float16)(float)((((b1)>>7)|((b2)<<1))&7u)+zp; \
    a[6]=sc*(_Float16)(float)(((b2)>>2)&7u)+zp; a[7]=sc*(_Float16)(float)(((b2)>>5)&7u)+zp

// #ifdef cannot appear inside a function-like macro body, so the v2 coexistence
// block lives here and expands at the GEN_RESID_BT header site.
#define HFQ4_MQ4_HDR_LOAD(gp) \
    const unsigned int hA = *(const unsigned int*)(gp); \
    const unsigned int hB = *(const unsigned int*)(gp + 4); \
    const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
    const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
    const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
    const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)))
#define HFQ4_MQ4_HDR_SEL \
    _Float16 sc = ((kt+ki) < 8) ? sc0 : sc1; \
    _Float16 zp = ((kt+ki) < 8) ? zp0 : zp1;

#define GEN_RESID_BT(BV) \
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_residual_wmma_gfx12_bt##BV( \
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
    int M, int K, int batch_size) { \
    const int tid = threadIdx.x, rs = blockIdx.x * 16, bs = blockIdx.y * (16 * BV); \
    if (rs >= M || bs >= batch_size) return; \
    const int ml = tid & 15, kg = tid >> 4; \
    const int sr = (rs + ml < M) ? (rs + ml) : (M - 1); \
    const int gpr = K / 256; const char* rb = A + (long long)sr * gpr * 104; \
    int sb[BV]; for (int b = 0; b < BV; b++) { int t = bs + b * 16 + ml; sb[b] = (t < batch_size) ? t : 0; } \
    float8_t acc[BV]; for (int b = 0; b < BV; b++) acc[b] = float8_t{0,0,0,0,0,0,0,0}; \
    for (int g = 0; g < gpr; g++) { \
        const char* gp = rb + g * 104; \
        HFQ4_MQ4_HDR_LOAD(gp); \
        const _Float16* xg[BV]; for (int b = 0; b < BV; b++) xg[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 4) { \
            const unsigned char* dp = (const unsigned char*)(gp+8); \
            unsigned char b0[4], b1[4], b2[4]; \
            for (int kk=0; kk<4; kk++) { int off = (kt+kk)*6 + kg*3; b0[kk]=dp[off]; b1[kk]=dp[off+1]; b2[kk]=dp[off+2]; } \
            _Pragma("unroll") \
            for (int ki = 0; ki < 4; ki++) { half8_t a; HFQ4_MQ4_HDR_SEL MQ3_DQ8(b0[ki],b1[ki],b2[ki], sc, zp); \
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
