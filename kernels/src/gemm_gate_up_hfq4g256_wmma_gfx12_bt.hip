// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batch-tiled gfx12 (RDNA4) WMMA gate_up for HFQ4-G256 prefill. Each block (1 wave)
// computes 16 rows x (16*B) batch with B INDEPENDENT accumulator chains, hiding the
// WMMA latency that caps the 1-acc production kernel (gemm_gate_up_hfq4g256_wmma_gfx12)
// at ~19% of the 170-TFLOPS fp16-WMMA peak. Weights + dequant are loaded ONCE per
// K-tile and reused across the B N-tiles (dequant /B). Byte-exact vs the 1-acc kernel
// (max_rel 0). B is the host's adaptive pick = clamp(N/16, 1, 12); B=16 spills VGPR so
// 12 is the cap. Measured (gfx1201, N=192): 32.3 -> 59.7 TFLOPS (+85%).
//
// Grid: [ceil((gate_m+up_m)/16), ceil(N/(16*B))]. Block: [32]. LDS: 0.
// Same lane decomposition / C-mapping as the 1-acc gfx12 kernel; only the batch
// dimension is tiled. Overwrite semantics (y = acc).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#define HFQ4_DQ8(pk, sc, zp) \
    a[0]=sc*(_Float16)(float)(((pk)>>0)&0xFu)+zp; a[1]=sc*(_Float16)(float)(((pk)>>4)&0xFu)+zp; \
    a[2]=sc*(_Float16)(float)(((pk)>>8)&0xFu)+zp; a[3]=sc*(_Float16)(float)(((pk)>>12)&0xFu)+zp; \
    a[4]=sc*(_Float16)(float)(((pk)>>16)&0xFu)+zp; a[5]=sc*(_Float16)(float)(((pk)>>20)&0xFu)+zp; \
    a[6]=sc*(_Float16)(float)(((pk)>>24)&0xFu)+zp; a[7]=sc*(_Float16)(float)(((pk)>>28)&0xFu)+zp

#define GEN_GATE_UP_BT(BV) \
extern "C" __launch_bounds__(32, 2) __global__ void gemm_gate_up_hfq4g256_wmma_gfx12_bt##BV( \
    const char* __restrict__ A_gate, const char* __restrict__ A_up, const _Float16* __restrict__ X, \
    float* __restrict__ Y_gate, float* __restrict__ Y_up, int gate_m, int up_m, int K, int N) { \
    const int tm = gate_m + up_m, rs = blockIdx.x * 16, bs = blockIdx.y * (16 * BV), tid = threadIdx.x; \
    if (rs >= tm || bs >= N) return; \
    const int ml = tid & 15, kg = tid >> 4; \
    const int mr = rs + ml, sr = (mr < tm) ? mr : (tm - 1); \
    const char* A; int lr; if (sr < gate_m) { A = A_gate; lr = sr; } else { A = A_up; lr = sr - gate_m; } \
    const int gpr = K / 256; const char* rb = A + (long long)lr * gpr * 136; \
    int sb[BV]; for (int b = 0; b < BV; b++) { int t = bs + b * 16 + ml; sb[b] = (t < N) ? t : 0; } \
    float8_t acc[BV]; for (int b = 0; b < BV; b++) acc[b] = float8_t{0,0,0,0,0,0,0,0}; \
    for (int g = 0; g < gpr; g++) { \
        const char* gp = rb + g * 136; \
        _Float16 sc = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp)); \
        _Float16 zp = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
        const _Float16* xb[BV]; for (int b = 0; b < BV; b++) xb[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 4) { \
            unsigned int p[4] = { *(const unsigned int*)(gp+8+(kt+0)*8+kg*4), *(const unsigned int*)(gp+8+(kt+1)*8+kg*4), \
                                  *(const unsigned int*)(gp+8+(kt+2)*8+kg*4), *(const unsigned int*)(gp+8+(kt+3)*8+kg*4) }; \
            _Pragma("unroll") \
            for (int ki = 0; ki < 4; ki++) { half8_t a; HFQ4_DQ8(p[ki], sc, zp); \
                _Pragma("unroll") \
                for (int b = 0; b < BV; b++) { half8_t bb = *(const half8_t*)(xb[b] + (kt+ki)*16 + kg*8); \
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc[b]); } } } \
    } \
    for (int b = 0; b < BV; b++) { const int oc = bs + b * 16 + ml; if (oc < N) for (int j = 0; j < 8; j++) { \
        int orow = rs + 8 * kg + j; if (orow < tm) { float* Y; int pr, os; \
            if (orow < gate_m) { Y = Y_gate; pr = orow; os = gate_m; } else { Y = Y_up; pr = orow - gate_m; os = up_m; } \
            Y[(long long)oc * os + pr] = acc[b][j]; } } } \
}

GEN_GATE_UP_BT(4)
GEN_GATE_UP_BT(8)
GEN_GATE_UP_BT(12)
