// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned batch-tiled gfx12 HFQ4-G256 residual GEMM.
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx12_bt.hip (MIT OR Apache-2.0).
// The inner math — dequant, WMMA operand packing, C-output mapping — is copied
// verbatim so results stay comparable to the shared kernel. The ONLY changes
// are the batch-tile width BV and the occupancy hint. This file exists so
// Glimmer can tile widths the shared kernel does not instantiate WITHOUT
// touching a kernel that qwen / gemma4 / deepseek depend on.
//
// Why a new BV at all. Each block covers BV*16 batch columns and the host
// launches ceil(batch/(16*BV)) tiles, so throughput depends on whether the
// prefill chunk divides that evenly. Measured on gfx1201, 19968x6656:
//
//     B=192  BT12  1 tile  exact   60.0 TF
//     B=384  BT12  2 tiles exact   62.4 TF
//     B=256  BT8   2 tiles exact   51.9 TF   <-- Glimmer's shipping chunk
//     B=288  BT12  2 tiles 33% waste 53.4 TF
//
// The shared heuristic is CORRECT for the widths it has: at B=256 only BT8
// divides evenly, and forcing BT12 there would waste a third of every block.
// The gap is not a bad choice, it is a missing width. BV=16 covers 256 columns
// per block, so B=256 tiles exactly in ONE block while amortizing the weight
// load and dequant over 16 N-tiles instead of 8. BV=24 does the same for 384.
//
// Occupancy. acc[BV] costs BV*8 VGPRs, so BV=16 needs 128 for accumulators
// alone, plus xg[BV] pointers and sb[BV]. That does not fit the 128-VGPR
// budget implied by __launch_bounds__(32, 2), which is why the shared file
// stops at BV=12. These variants therefore run at occupancy 1 and trade wave
// parallelism for per-block reuse; whether that trade pays is an empirical
// question, so a BV=16 occupancy-2 variant is included to measure it rather
// than assume. Occupancy 2 at BV=16 is EXPECTED to spill — it is a control.
//
// Grid: [ceil(M/16), ceil(batch/(16*BV))]. Block: [32]. LDS: 0.
// Output is += (caller pre-inits Y), identical contract to the shared kernel.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;
#define HFQ4_DQ8(pk, sc, zp) \
    a[0]=sc*(_Float16)(float)(((pk)>>0)&0xFu)+zp; a[1]=sc*(_Float16)(float)(((pk)>>4)&0xFu)+zp; \
    a[2]=sc*(_Float16)(float)(((pk)>>8)&0xFu)+zp; a[3]=sc*(_Float16)(float)(((pk)>>12)&0xFu)+zp; \
    a[4]=sc*(_Float16)(float)(((pk)>>16)&0xFu)+zp; a[5]=sc*(_Float16)(float)(((pk)>>20)&0xFu)+zp; \
    a[6]=sc*(_Float16)(float)(((pk)>>24)&0xFu)+zp; a[7]=sc*(_Float16)(float)(((pk)>>28)&0xFu)+zp

#define GEN_MUSE_BT(BV, OCC, SUFFIX) \
extern "C" __launch_bounds__(32, OCC) __global__ void gemm_hfq4g256_residual_wmma_gfx12_muse_##SUFFIX( \
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

// Register-lean FULL-TILE variants. The host guarantees bs + BV*16 <= batch
// for every launched tile (i.e. batch % (16*BV) == 0), which removes ALL
// clamping and lets the X addresses collapse to one base pointer plus a
// constant b-stride, since sb[b] = bs + b*16 + ml is affine in b when nothing
// is clamped:  xg[b] = X + (bs + b*16 + ml)*K + g*256 = xg[0] + b*(16*K).
// That deletes the xg[BV] pointer array (2*BV VGPRs) and the sb[BV] index
// array (BV VGPRs) — about 48 registers at BV=16, which is the difference
// between fitting in 256 and spilling. Kept as SEPARATE kernels rather than a
// branch inside the general one because the register allocator budgets for the
// worst path, so a fused fast/slow kernel would spill exactly as much as the
// general one and buy nothing. Ragged batches keep using the shared BT kernel.
#define GEN_MUSE_FULL(BV, OCC, SUFFIX) \
extern "C" __launch_bounds__(32, OCC) __global__ void gemm_hfq4g256_residual_wmma_gfx12_muse_##SUFFIX( \
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
    int M, int K, int batch_size) { \
    const int tid = threadIdx.x, rs = blockIdx.x * 16, bs = blockIdx.y * (16 * BV); \
    if (rs >= M) return; \
    const int ml = tid & 15, kg = tid >> 4; \
    const int sr = (rs + ml < M) ? (rs + ml) : (M - 1); \
    const int gpr = K / 256; const char* rb = A + (long long)sr * gpr * 136; \
    const long long xstride = (long long)16 * K; \
    float8_t acc[BV]; for (int b = 0; b < BV; b++) acc[b] = float8_t{0,0,0,0,0,0,0,0}; \
    for (int g = 0; g < gpr; g++) { \
        const char* gp = rb + g * 136; \
        _Float16 sc = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp)); \
        _Float16 zp = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
        const _Float16* xg0 = X + (long long)(bs + ml) * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 4) { \
            unsigned int p[4] = { *(const unsigned int*)(gp+8+(kt+0)*8+kg*4), *(const unsigned int*)(gp+8+(kt+1)*8+kg*4), \
                                  *(const unsigned int*)(gp+8+(kt+2)*8+kg*4), *(const unsigned int*)(gp+8+(kt+3)*8+kg*4) }; \
            _Pragma("unroll") \
            for (int ki = 0; ki < 4; ki++) { half8_t a; HFQ4_DQ8(p[ki], sc, zp); \
                const _Float16* xp = xg0 + (kt+ki)*16 + kg*8; \
                _Pragma("unroll") \
                for (int b = 0; b < BV; b++) { half8_t bb = *(const half8_t*)(xp + b * xstride); \
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc[b]); } } } \
    } \
    for (int b = 0; b < BV; b++) { const int oc = bs + b * 16 + ml; for (int j = 0; j < 8; j++) { \
        int orow = rs + 8 * kg + j; if (orow < M) Y[(long long)oc * M + orow] += acc[b][j]; } } \
}

GEN_MUSE_BT(16, 1, bt16)
GEN_MUSE_BT(16, 2, bt16occ2)
GEN_MUSE_BT(24, 1, bt24)
GEN_MUSE_FULL(16, 1, full16)
GEN_MUSE_FULL(12, 2, full12)
GEN_MUSE_FULL(8, 2, full8)
GEN_MUSE_FULL(4, 4, full4)
