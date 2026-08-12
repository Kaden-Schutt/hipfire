// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Muse Glimmer-owned batch-tiled gfx12 HFQ4-G256 residual GEMM.
//
// DERIVED FROM gemm_hfq4g256_residual_wmma_gfx12_bt.hip (MIT OR Apache-2.0).
// The inner math — dequant, WMMA operand packing, C-output mapping — is copied
// verbatim so results stay comparable to the shared kernel. The ONLY changes
// are the removal of tail clamping and the resulting affine X addressing. This
// file exists so Glimmer can tile widths the shared kernel does not instantiate
// WITHOUT touching a kernel that qwen / gemma4 / deepseek depend on.
//
// FULL-TILE PRECONDITION. The host guarantees bs + BV*16 <= batch for every
// launched tile (batch % (16*BV) == 0), which removes all clamping and lets the
// X addresses collapse to one base pointer plus a constant b-stride, since
// sb[b] = bs + b*16 + ml is affine in b when nothing is clamped:
//     xg[b] = X + (bs + b*16 + ml)*K + g*256 = xg[0] + b*(16*K)
// That deletes the xg[BV] pointer array (2*BV VGPRs) and the sb[BV] index array
// (BV VGPRs). On gfx1201 that is 247 VGPR / occupancy 5 in the shared kernel
// vs 197 / occupancy 7 here, zero spill both. Occupancy, not tile width, is
// the whole win — same BV, same math, more waves in flight.
//
// ONLY BV=12 IS INSTANTIATED, deliberately. Every extra instantiation is
// compile time that every user pays on first load, since ensure_kernel builds
// the whole translation unit. Widths that were implemented, measured, and
// removed:
//   bt=8   138 occ 10 -> 103 occ 12, yet LOST to the shared bt8 at B=256
//   bt=4    79 occ 16 ->  77 occ 16, no gain, shared is already saturated
//   bt=16  spills 53 (57 even in this affine form) — acc[16] is 128 VGPRs
//          before the 16 hoisted half8_t operands, against a 256-VGPR file
//   bt=24  spills 157      bt=20 (full-tile) spills 405
// bt=16 DOES fit under the gcn-iterative-ilp scheduler (245 VGPR, zero spill,
// occupancy 5), found with `radiowave compile --scheduler-profile`. It is
// still not worth it: it only helps a 256-token chunk, and chunk 192 beats
// chunk 256 end-to-end even if gate/up were free there, because 192 also wins
// down_proj. Do not re-add a width on a resource manifest alone.
//
// The nine-kernel version of this file cost 2.89 s to compile versus 1.68 s
// for one kernel. That 1.11 s surfaced as a phantom ~1200 ms "regression"
// — constant across prompt lengths, which is what gave it away — before being
// traced to JIT compilation rather than to any kernel being slow.
//
// Grid: [ceil(M/16), batch/(16*BV)]. Block: [32]. LDS: 0.
// Output is += (caller pre-inits Y), identical contract to the shared kernel.
// Overwrite semantics were considered to drop the caller's memset; measured
// not worth a variant, the memset is 0.8-4.5% of a call (~1.4% of layer time).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;
#define HFQ4_DQ8(pk, sc, zp) \
    a[0]=sc*(_Float16)(float)(((pk)>>0)&0xFu)+zp; a[1]=sc*(_Float16)(float)(((pk)>>4)&0xFu)+zp; \
    a[2]=sc*(_Float16)(float)(((pk)>>8)&0xFu)+zp; a[3]=sc*(_Float16)(float)(((pk)>>12)&0xFu)+zp; \
    a[4]=sc*(_Float16)(float)(((pk)>>16)&0xFu)+zp; a[5]=sc*(_Float16)(float)(((pk)>>20)&0xFu)+zp; \
    a[6]=sc*(_Float16)(float)(((pk)>>24)&0xFu)+zp; a[7]=sc*(_Float16)(float)(((pk)>>28)&0xFu)+zp

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

GEN_MUSE_FULL(12, 2, full12)
