// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64
//
// Tuned W4A4 (signed int4 × int4 → int32) GEMM for the prefill/batch path on
// RDNA3.5 (gfx1151). LDS-staged, double-buffered, wave64. Same contract and
// packing as gemm_iu4_i32_wmma (single-chain), just far faster on large GEMMs:
//   A : [M, K/2] bytes, byte = k_even | (k_odd << 4)  (signed 4-bit two's-comp)
//   X : [B, K/2] bytes, same packing
//   Y : [B, M] int32 row-major.  K % 64 == 0 (Oq4G256 guarantees %256).
//
// ~14× the single-chain kernel / ~50% of the iu4 matrix-core peak on gfx1151.
// Structure (see memory reference_gfx1151_iu4_gemm_tuning):
//   • wave64 WMMA `_w64` → 4 acc GPRs/tile (vs 8 wave32) → affords N-heavy tiling.
//   • block 256 thr = 4 wave64 waves (2×2); block tile BM=64 × BN=256; per wave
//     WMt=2 × WNt=8 = 16 output 16×16 sub-tiles (N-heavy: maximises A-row reuse).
//   • BK=64 K-strip; register-staged double buffer (RDNA3 has no cp.async):
//     prefetch next strip global→regs DURING compute, store regs→alt LDS buffer.
//   • b128 LDS fragment reads; b128 global loads for the (large) X operand.
// Decode (B=1) stays on the bandwidth-bound single-chain gemv; this is prefill.
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

__launch_bounds__(BLOCK) extern "C" __global__ void gemm_iu4_i32_wmma_lds(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ X, int32_t* __restrict__ Y,
    int M, int K, int B) {
    __shared__ uint8_t As[2][BM * (BK / 2)], Xs[2][BN * (BK / 2)];
    const int bm = blockIdx.x * BM, bn = blockIdx.y * BN, st = K / 2;
    const int wid = threadIdx.x >> 6, wtid = threadIdx.x & 63, lane = wtid & 15;
    const int wm = wid / WARPS_N, wn = wid % WARPS_N, nstrip = K / BK;
    const int UA = (BK / 2) / 4;               // u32 per source row (=8)
    const int NA = BM * UA / BLOCK;            // As u32/thread (=2)
    const int NXv = BN * UA / 4 / BLOCK;       // Xs i32x4/thread (=2, GVEC b128)
    i32x4 acc[WMt][WNt];
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++) acc[i][j] = (i32x4){0, 0, 0, 0};
    uint32_t rA[NA];
    i32x4 rX[NXv];
    // global -> registers (As: b32; Xs: contiguous b128)
#define LOADG(kk) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { int idx = threadIdx.x + l * BLOCK, row = idx / UA, c = idx % UA; int r = bm + row < M ? bm + row : M - 1; rA[l] = *(const uint32_t*)(A + (long long)r * st + (kk) / 2 + c * 4); } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { int q = threadIdx.x + l * BLOCK, base = q * 4, row = base / UA, c = base % UA; int r = bn + row < B ? bn + row : B - 1; rX[l] = *(const i32x4*)(X + (long long)r * st + (kk) / 2 + c * 4); }
#define STS(bf) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { int idx = threadIdx.x + l * BLOCK; ((uint32_t*)As[bf])[idx] = rA[l]; } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { int q = threadIdx.x + l * BLOCK; ((i32x4*)Xs[bf])[q] = rX[l]; }
    { LOADG(0); STS(0); }
    __syncthreads();
    for (int k = 0; k < nstrip; k++) {
        int buf = k & 1;
        if (k + 1 < nstrip) { LOADG((k + 1) * BK); }
        // Fragment reads: one i32x2 (8 B) per K-substep from the BK-deep LDS row;
        // adjacent substeps fuse to ds_load_b128 in the compiler.
#pragma unroll
        for (int s = 0; s < BK / 16; s++) {
            i32x2 af[WMt], xf[WNt];
#pragma unroll
            for (int i = 0; i < WMt; i++) af[i] = *(const i32x2*)(As[buf] + (wm * WMt * 16 + i * 16 + lane) * (BK / 2) + s * 8);
#pragma unroll
            for (int j = 0; j < WNt; j++) xf[j] = *(const i32x2*)(Xs[buf] + (wn * WNt * 16 + j * 16 + lane) * (BK / 2) + s * 8);
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int j = 0; j < WNt; j++) acc[i][j] = W64(af[i], xf[j], acc[i][j]);
        }
        if (k + 1 < nstrip) { STS((k + 1) & 1); }
        __syncthreads();
    }
#undef LOADG
#undef STS
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
