// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MFMA-accelerated GEMM: F16 weight × F16 input → F32 output.
// gfx942 (CDNA3 wave64 MFMA, 16×16×16 tiles).
//
// CDNA3 port of `gemm_f16_x_f16_wmma.hip`. That kernel is built on
// `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, an RDNA3 wave32 WMMA
// intrinsic that requires target features `gfx11-insts,wavefrontsize32`
// and therefore cannot compile for gfx942 at all. CDNA3's equivalent
// primitive is `__builtin_amdgcn_mfma_f32_16x16x16f16` (wave64), so this
// is a register-layout port rather than an algorithmic change: identical
// math, identical signature, identical caller contract.
//
// Layout (output is [B, M], matching the WMMA original byte-for-byte):
//   A : [M, K] F16 row-major  (weight)
//   X : [B, K] F16 row-major  (caller F32→F16-converts upstream)
//   Y : [B, M] F32 row-major
//
// Compute: Y[b, m] = sum_k A[m, k] * X[b, k]
//
// MFMA computes D[i][j] = sum_k A'[i][k] * B'[k][j]. Setting i = m (row
// within the tile) and j = b (batch within the tile) gives
//   A'[m][k] = W[m][k]      and      B'[k][b] = X[b][k],
// i.e. B' is X transposed at tile granularity. Both operands therefore
// read 4 CONTIGUOUS halves from the same K offset of their own row, so
// the two fragment loads share one addressing pattern.
//
// v_mfma_f32_16x16x16f16 register layout (wave64, lane l ∈ [0, 64)):
//   A' frag : lane l holds A'[l & 15][4*(l >> 4) + i],  i ∈ [0, 4)
//   B' frag : lane l holds B'[4*(l >> 4) + i][l & 15],  i ∈ [0, 4)
//   D  frag : lane l holds D[4*(l >> 4) + i][l & 15],   i ∈ [0, 4)
// so on write-out row = row_start + 4*(l >> 4) + i and col (= batch) =
// batch_start + (l & 15). Note this differs from the wave32 WMMA original,
// where the accumulator was 8 wide and row = 2*j + (tid >> 4).
//
// K must be a multiple of 16 (same contract as the WMMA original — the K
// loop has no tail guard).
//
// Grid:  [ceil(M / 16), ceil(B / 16)]
// Block: [64]
// LDS:   0

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(4))) half4_t;
typedef float    __attribute__((ext_vector_type(4))) float4_t;

__launch_bounds__(64, 8)
extern "C" __global__ void gemm_f16_x_f16_mfma_gfx942(
    const _Float16* __restrict__ A,    // [M, K] F16
    const _Float16* __restrict__ X,    // [B, K] F16
    float*          __restrict__ Y,    // [B, M] F32
    int M, int K, int B
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int lane = threadIdx.x;          // wave64: [0, 64)

    if (row_start >= M || batch_start >= B) return;

    // Both fragments index their own row by (lane & 15) and their K
    // offset by 4*(lane >> 4) — the MFMA A'/B' layouts coincide here
    // because B' is X transposed.
    const int idx   = lane & 15;
    const int k_off = (lane >> 4) * 4;

    const int my_m = row_start + idx;
    const int my_b = batch_start + idx;
    // Clamp OOB lanes onto a valid row: they still participate in the
    // MFMA (it is a wave-wide instruction) but their results are
    // discarded by the guarded write-out below.
    const int safe_m = (my_m < M) ? my_m : (M - 1);
    const int safe_b = (my_b < B) ? my_b : (B - 1);

    const _Float16* w_row = A + (long long)safe_m * K;
    const _Float16* x_row = X + (long long)safe_b * K;

    float4_t acc = {0.f, 0.f, 0.f, 0.f};

    // K loop: 16 elements per MFMA step, 4 per lane. One mma per iter.
    for (int k = 0; k < K; k += 16) {
        half4_t a_frag = *(const half4_t*)(w_row + k + k_off);
        half4_t x_frag = *(const half4_t*)(x_row + k + k_off);
        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, x_frag, acc, 0, 0, 0);
    }

    // Write 4 outputs per lane following the MFMA D layout.
    const int out_col = batch_start + idx;
    if (out_col < B) {
        const int row_base = row_start + (lane >> 4) * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int out_row = row_base + i;
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[i];
            }
        }
    }
}
