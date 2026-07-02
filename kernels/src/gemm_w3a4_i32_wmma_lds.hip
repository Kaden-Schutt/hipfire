// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64
//
// Tuned W3A4 (signed 3-bit weight × int4 activation → int32) GEMM for the
// prefill/batch path on RDNA3.5 (gfx1151). Same tuned iu4·iu4 matrix core as
// gemm_iu4_i32_wmma_lds (wave64, LDS-staged, double-buffered, N-heavy 2×8), but
// the WEIGHT operand is a 3-bit bit-plane (3/8 B/wt vs int4's 4/8 → 25% less
// weight traffic) unpacked to int4 in LDS during staging via a cheap Morton
// spread. The memory-ceiling lever: wins ~1.3× exactly where weight bandwidth
// dominates (small-batch prefill / spec-decode), neutral at large batch. See
// memory reference_gfx1151_w3a4_gemm.
//   A : [M, 3K/32] u32 bit-plane. Per 32-weight group g (K-major), 3 contiguous
//       u32 (p0,p1,p2); plane b bit i = bit b of the signed two's-comp weight at
//       position 32g+i. This is the Oq3G256 on-disk payload (sans the per-group
//       f16 scale, applied in the dequant epilogue elsewhere).
//   X : [B, K/2] bytes, byte = k_even | (k_odd << 4)  (signed 4-bit two's-comp).
//   Y : [B, M] int32 row-major.  K % 64 == 0 (Oq3G256 guarantees %256).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t i32x2 __attribute__((ext_vector_type(2)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));
#define W64(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64(true, a, true, b, c, false)

#define BK 64
#define BLOCK 256   // 4 wave64 waves
#define WARPS_N 2
#define WMt 2
#define WNt 8
#define BM (2 * WMt * 16)  // 64
#define BN (2 * WNt * 16)  // 256

static __device__ __forceinline__ i32x2 ld8(const uint8_t* p) {
    return (i32x2){(int)(*(const uint32_t*)p), (int)(*(const uint32_t*)(p + 4))};
}
// deposit 8 bits (bit i -> nibble i, i.e. bit 4i) — Morton spread-by-4.
static __device__ __forceinline__ uint32_t spread8(uint32_t b) {
    b = (b ^ (b << 12)) & 0x000F000F;
    b = (b ^ (b << 6)) & 0x03030303;
    b = (b ^ (b << 3)) & 0x11111111;
    return b;
}
// one output u32 = 8 int4 nibbles from byte `w` of each bit-plane, sign-extended
// (3-bit signed -> 4-bit two's-comp: replicate plane-2 bit into nibble bit3, so
// the iu4 signed WMMA reads it correctly).
static __device__ __forceinline__ uint32_t unpack8_w3(uint32_t p0, uint32_t p1, uint32_t p2, int w) {
    uint32_t s2 = spread8((p2 >> (8 * w)) & 0xff);
    return spread8((p0 >> (8 * w)) & 0xff) | (spread8((p1 >> (8 * w)) & 0xff) << 1) | (s2 << 2) | (s2 << 3);
}

__launch_bounds__(BLOCK) extern "C" __global__ void gemm_w3a4_i32_wmma_lds(
    const uint32_t* __restrict__ Aw3, const uint8_t* __restrict__ X, int32_t* __restrict__ Y,
    int M, int K, int B) {
    __shared__ uint8_t As[2][BM * (BK / 2)], Xs[2][BN * (BK / 2)];
    const int bm = blockIdx.x * BM, bn = blockIdx.y * BN;
    const int stw = 3 * K / 32, stx = K / 2;  // weight u32/row, act byte/row
    const int wid = threadIdx.x >> 6, wtid = threadIdx.x & 63, lane = wtid & 15;
    const int wm = wid / WARPS_N, wn = wid % WARPS_N, nstrip = K / BK;
    i32x4 acc[WMt][WNt];
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++) acc[i][j] = (i32x4){0, 0, 0, 0};
    const int WOW = BM * (BK / 32) * 4;         // weight output-u32/strip (=512): 2/thread
    const int NXu = BN * (BK / 2) / 4 / BLOCK;  // act u32/thread (=8)
    // Weight staging: unpack 3-bit bit-planes -> int4 in LDS, all threads active
    // (cheap bit-spread, no scalar 32-loop). Act staging: plain b32 copy to LDS.
#define LOADSTORE(kk, bf) \
    _Pragma("unroll") for (int q = threadIdx.x; q < WOW; q += BLOCK) { \
        int group = q >> 2, w = q & 3, row = group / (BK / 32), grp = group % (BK / 32); \
        int r = bm + row < M ? bm + row : M - 1; \
        const uint32_t* pp = Aw3 + (long long)r * stw + ((kk) / 32) * 3 + grp * 3; \
        ((uint32_t*)As[bf])[row * (BK / 2 / 4) + grp * 4 + w] = unpack8_w3(pp[0], pp[1], pp[2], w); \
    } \
    _Pragma("unroll") for (int l = 0; l < NXu; l++) { int idx = threadIdx.x + l * BLOCK, row = idx / ((BK / 2) / 4), c = idx % ((BK / 2) / 4); int r = bn + row < B ? bn + row : B - 1; ((uint32_t*)Xs[bf])[idx] = *(const uint32_t*)(X + (long long)r * stx + (kk) / 2 + c * 4); }
    LOADSTORE(0, 0);
    __syncthreads();
    for (int k = 0; k < nstrip; k++) {
        int buf = k & 1;
#pragma unroll
        for (int s = 0; s < BK / 16; s++) {
            i32x2 af[WMt], xf[WNt];
#pragma unroll
            for (int i = 0; i < WMt; i++) af[i] = ld8(As[buf] + (wm * WMt * 16 + i * 16 + lane) * (BK / 2) + s * 8);
#pragma unroll
            for (int j = 0; j < WNt; j++) xf[j] = ld8(Xs[buf] + (wn * WNt * 16 + j * 16 + lane) * (BK / 2) + s * 8);
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int j = 0; j < WNt; j++) acc[i][j] = W64(af[i], xf[j], acc[i][j]);
        }
        __syncthreads();
        if (k + 1 < nstrip) { LOADSTORE((k + 1) * BK, (k + 1) & 1); __syncthreads(); }
    }
#undef LOADSTORE
    // Plain (uncoalesced) epilogue — prefill K is large, so the store is amortized.
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++) {
            int oc = bn + wn * WNt * 16 + j * 16 + lane;
            if (oc >= B) continue;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                int r = bm + wm * WMt * 16 + i * 16 + 4 * g + (wtid >> 4);
                if (r < M) Y[(long long)oc * M + r] = acc[i][j][g];
            }
        }
}
