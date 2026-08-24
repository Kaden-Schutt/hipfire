// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batch-tiled gfx12 HFQ4-G256 4-way fused QKVZA GEMM (Qwen3.5 DeltaNet
// q/k/v/z/beta/alpha projections). B independent acc chains over the batch (N)
// dim hide the WMMA latency that caps the 1-acc gemm_qkvza_hfq4g256_wmma_gfx12;
// weights+dequant loaded once per K-tile, reused across the B N-tiles. The
// per-row weight/output routing (A_qkv/A_z/A_beta/A_alpha) is unchanged.
// Byte-exact vs the 1-acc kernel. Grid: [ceil(total_m/16), ceil(N/(16*B))].
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;
#define MQ3_DQ8(b0,b1,b2, sc, zp) \
    a[0]=sc*(_Float16)(float)((b0)&7u)+zp; a[1]=sc*(_Float16)(float)(((b0)>>3)&7u)+zp; \
    a[2]=sc*(_Float16)(float)((((b0)>>6)|((b1)<<2))&7u)+zp; a[3]=sc*(_Float16)(float)(((b1)>>1)&7u)+zp; \
    a[4]=sc*(_Float16)(float)(((b1)>>4)&7u)+zp; a[5]=sc*(_Float16)(float)((((b1)>>7)|((b2)<<1))&7u)+zp; \
    a[6]=sc*(_Float16)(float)(((b2)>>2)&7u)+zp; a[7]=sc*(_Float16)(float)(((b2)>>5)&7u)+zp

// Nested #ifdef inside GEN_* is invalid after expansion; hoist header decode
// into helpers so GEN_QKVZA_BT stays a pure token stream.
#define HFQ4_MQ4_HDR_LOAD(gp) \
    const unsigned int hA = *(const unsigned int*)(gp); \
    const unsigned int hB = *(const unsigned int*)(gp + 4); \
    const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
    const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
    const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
    const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)))
// WMMA packs t=kt*2+k_grp; half splits at pack>=16 i.e. kt>=8 (not tid).
#define HFQ4_MQ4_HDR_SEL \
    _Float16 sc = ((kt+ki) < 8) ? sc0 : sc1; \
    _Float16 zp = ((kt+ki) < 8) ? zp0 : zp1;

#define GEN_QKVZA_BT(BV) \
extern "C" __launch_bounds__(32, 2) __global__ void gemm_qkvza_mq3g256v2_wmma_gfx12_bt##BV( \
    const char* __restrict__ A_qkv, const char* __restrict__ A_z, const char* __restrict__ A_beta, \
    const char* __restrict__ A_alpha, const _Float16* __restrict__ X, \
    float* __restrict__ Y_qkv, float* __restrict__ Y_z, float* __restrict__ Y_beta, float* __restrict__ Y_alpha, \
    int qkv_m, int z_m, int beta_m, int alpha_m, int K, int N) { \
    const int total_m = qkv_m + z_m + beta_m + alpha_m, rs = blockIdx.x * 16, bs = blockIdx.y * (16 * BV), tid = threadIdx.x; \
    if (rs >= total_m || bs >= N) return; \
    const int ml = tid & 15, kg = tid >> 4; \
    const int sr = (rs + ml < total_m) ? (rs + ml) : (total_m - 1); \
    const char* A; int lrow; \
    if (sr < qkv_m) { A = A_qkv; lrow = sr; } \
    else if (sr < qkv_m + z_m) { A = A_z; lrow = sr - qkv_m; } \
    else if (sr < qkv_m + z_m + beta_m) { A = A_beta; lrow = sr - (qkv_m + z_m); } \
    else { A = A_alpha; lrow = sr - (qkv_m + z_m + beta_m); } \
    const int gpr = K / 256; const char* rb = A + (long long)lrow * gpr * 104; \
    int sb[BV]; for (int b = 0; b < BV; b++) { int t = bs + b * 16 + ml; sb[b] = (t < N) ? t : 0; } \
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
    for (int b = 0; b < BV; b++) { const int oc = bs + b * 16 + ml; if (oc < N) for (int j = 0; j < 8; j++) { \
        int orow = rs + 8 * kg + j; if (orow < total_m) { float* Y; int pr, os; \
            if (orow < qkv_m) { Y = Y_qkv; pr = orow; os = qkv_m; } \
            else if (orow < qkv_m + z_m) { Y = Y_z; pr = orow - qkv_m; os = z_m; } \
            else if (orow < qkv_m + z_m + beta_m) { Y = Y_beta; pr = orow - (qkv_m + z_m); os = beta_m; } \
            else { Y = Y_alpha; pr = orow - (qkv_m + z_m + beta_m); os = alpha_m; } \
            Y[(long long)oc * os + pr] = acc[b][j]; } } } \
}

GEN_QKVZA_BT(4)
GEN_QKVZA_BT(8)
GEN_QKVZA_BT(12)
