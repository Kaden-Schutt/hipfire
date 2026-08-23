// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batch-tiled gfx1201 MQ4G256V2 3-way fused QKV GEMM — exact gfx1201 port
// of the QKVZA gfx12 BT authority (gemm_qkvza_mq4g256v2_wmma_gfx12_bt.hip)
// reduced to three Q/K/V outputs. Production-neutral; not wired to
// dispatch/kernels.rs/composer. Composer-wireable via explicit ABI below.
//
// Authority: current gfx12 base/BT
//   base:  gemm_qkv_mq4g256v2_wmma.gfx12.hip (K4, half8, kg=tid>>4, gfx12 builtin, C map 8*kg+j, dual headers)
//   BT:    gemm_qkvza_mq4g256v2_wmma_gfx12_bt.hip (GEN_QKVZA_BT structure, BV tiling, routing/tails/overwrite)
//
// ABI / Symbols / Modules (composer-wireable, production-neutral):
//   Module (proposed, not wired): gemm_qkv_mq4g256v2_wmma_gfx1201_bt
//   Symbols (distinct, non-colliding with production gfx12 symbols):
//     gemm_qkv_mq4g256v2_wmma_gfx1201_bt4  — BT=4  (4 independent acc chains, grid ceil(N/64))
//     gemm_qkv_mq4g256v2_wmma_gfx1201_bt8  — BT=8  (8 chains, ceil(N/128))
//     gemm_qkv_mq4g256v2_wmma_gfx1201_bt12 — BT=12 (12 chains, ceil(N/192))
//   Grid: [ceil((q_m+k_m+v_m)/16), ceil(N/(16*BV))]  Block: [32]
//   Args ABI (identical across BV variants, mirrors QKVZA BT with Q/K/V):
//     (const char* A_q, const char* A_k, const char* A_v,
//      const _Float16* X,
//      float* Y_q, float* Y_k, float* Y_v,
//      int q_m, int k_m, int v_m, int K, int N)
//   Routing: per-row A_q/A_k/A_v and per-output Y_q/Y_k/Y_v via total_m.
//   Tails: row clamp safe_row=(rs+ml<total_m?rs+ml:total_m-1), batch sb[b]=clamped 0,
//          out_col< N and out_row< total_m guards. Overwrite: Y=*/acc (not +=).
//   K ordering: exact dual headers/K4 per HFQ group matching gfx12 base (kt 0..15 step4).
//
// Compile guard: exact-gfx1201. This TU must be compiled ONLY with
//   hipcc --offload-arch=gfx1201 (defines __gfx1201__). gfx1200 alone is insufficient.

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "exact-gfx1201: gemm_qkv_mq4g256v2_wmma_gfx1201_bt.hip requires --offload-arch=gfx1201"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#define HFQ4_DQ8_QKV(pk, sc, zp) \
    a[0]=sc*(_Float16)(float)(((pk)>>0)&0xFu)+zp; a[1]=sc*(_Float16)(float)(((pk)>>4)&0xFu)+zp; \
    a[2]=sc*(_Float16)(float)(((pk)>>8)&0xFu)+zp; a[3]=sc*(_Float16)(float)(((pk)>>12)&0xFu)+zp; \
    a[4]=sc*(_Float16)(float)(((pk)>>16)&0xFu)+zp; a[5]=sc*(_Float16)(float)(((pk)>>20)&0xFu)+zp; \
    a[6]=sc*(_Float16)(float)(((pk)>>24)&0xFu)+zp; a[7]=sc*(_Float16)(float)(((pk)>>28)&0xFu)+zp

// Hoisted header helpers (nested #ifdef inside GEN forbidden)
#define HFQ4_MQ4_HDR_LOAD_QKV(gp) \
    const unsigned int hA = *(const unsigned int*)(gp); \
    const unsigned int hB = *(const unsigned int*)(gp + 4); \
    const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
    const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
    const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
    const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)))
#define HFQ4_MQ4_HDR_SEL_QKV \
    _Float16 sc = (kt < 8) ? sc0 : sc1; \
    _Float16 zp = (kt < 8) ? zp0 : zp1;

// GEN_QKV_BT(BV): port QKVZA BT structure to three Q/K/V outputs.
// Exact routing/tails/overwrite matching authority; BV=4/8/12 instantiation.
#define GEN_QKV_BT(BV) \
extern "C" __launch_bounds__(32, 2) __global__ void gemm_qkv_mq4g256v2_wmma_gfx1201_bt##BV( \
    const char* __restrict__ A_q, const char* __restrict__ A_k, const char* __restrict__ A_v, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_q, float* __restrict__ Y_k, float* __restrict__ Y_v, \
    int q_m, int k_m, int v_m, int K, int N) { \
    const int total_m = q_m + k_m + v_m, rs = blockIdx.x * 16, bs = blockIdx.y * (16 * BV), tid = threadIdx.x; \
    if (rs >= total_m || bs >= N) return; \
    const int ml = tid & 15, kg = tid >> 4; /* lane k_grp=tid>>4 */ \
    const int sr = (rs + ml < total_m) ? (rs + ml) : (total_m - 1); \
    const char* A; int lrow; \
    if (sr < q_m) { A = A_q; lrow = sr; } \
    else if (sr < q_m + k_m) { A = A_k; lrow = sr - q_m; } \
    else { A = A_v; lrow = sr - (q_m + k_m); } \
    const int gpr = K / 256; const char* rb = A + (long long)lrow * gpr * 136; \
    int sb[BV]; for (int b = 0; b < BV; b++) { int t = bs + b * 16 + ml; sb[b] = (t < N) ? t : 0; } \
    float8_t acc[BV]; for (int b = 0; b < BV; b++) acc[b] = float8_t{0,0,0,0,0,0,0,0}; \
    for (int g = 0; g < gpr; g++) { \
        const char* gp = rb + g * 136; \
        HFQ4_MQ4_HDR_LOAD_QKV(gp); \
        const _Float16* xg[BV]; for (int b = 0; b < BV; b++) xg[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 4) { \
            unsigned int p[4] = { *(const unsigned int*)(gp+8+(kt+0)*8+kg*4), *(const unsigned int*)(gp+8+(kt+1)*8+kg*4), \
                                  *(const unsigned int*)(gp+8+(kt+2)*8+kg*4), *(const unsigned int*)(gp+8+(kt+3)*8+kg*4) }; \
            _Pragma("unroll") \
            for (int ki = 0; ki < 4; ki++) { half8_t a; HFQ4_MQ4_HDR_SEL_QKV HFQ4_DQ8_QKV(p[ki], sc, zp); \
                _Pragma("unroll") \
                for (int b = 0; b < BV; b++) { half8_t bb = *(const half8_t*)(xg[b] + (kt+ki)*16 + kg*8); \
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc[b]); } } } \
    } \
    for (int b = 0; b < BV; b++) { const int oc = bs + b * 16 + ml; if (oc < N) for (int j = 0; j < 8; j++) { \
        int orow = rs + 8 * kg + j; /* C map row=8*k_grp+j */ \
        if (orow < total_m) { float* Y; int pr, os; \
            if (orow < q_m) { Y = Y_q; pr = orow; os = q_m; } \
            else if (orow < q_m + k_m) { Y = Y_k; pr = orow - q_m; os = k_m; } \
            else { Y = Y_v; pr = orow - (q_m + k_m); os = v_m; } \
            Y[(long long)oc * os + pr] = acc[b][j]; /* overwrite */ \
        } \
    } } \
}

GEN_QKV_BT(4)
GEN_QKV_BT(8)
GEN_QKV_BT(12)

#undef HFQ4_DQ8_QKV
#undef HFQ4_MQ4_HDR_LOAD_QKV
#undef HFQ4_MQ4_HDR_SEL_QKV
#undef GEN_QKV_BT
