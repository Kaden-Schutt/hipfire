// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// WMMA-accelerated GEMM: F16 weight × F16 input → F32 output.
// gfx1100+ only (RDNA3 / RDNA3.5 wave32 WMMA, 16×16×16 tiles).
//
// Layout (output is [B, M] like gemm_f32_register_tiled):
//   A : [M, K] F16 row-major  (weight)
//   X : [B, K] F16 row-major  (caller F32→F16-converts upstream)
//   Y : [B, M] F32 row-major
//
// Compute: Y[b, m] = sum_k A[m, k] * X[b, k]
//        = A @ X^T at the tile granularity 16×16×16.
//
// Each workgroup is a single wave32 and produces one 16-row × 16-batch
// output tile. Inside the K loop we issue one WMMA mma per K=16 step,
// dramatically reducing instruction count vs the scalar K-loop FMAs.
//
// Thread mapping for loads: tid ∈ [0, 32). lane = tid & 15 indexes
// rows within the 16-row tile (W rows for the A fragment, X rows for
// the B fragment). Threads 0..15 and 16..31 hold duplicate fragments
// per RDNA3 wave32 WMMA spec — that's by design, not waste.
//
// Output write convention (RDNA3 wave32 WMMA):
//   acc[j] = C[2*j + (tid >> 4)][tid & 15]
//   so row = 2*j + (tid >> 4), col (= batch) = tid & 15.
//
// Grid:  [ceil(M / 16), ceil(B / 16)]
// Block: [32]
// LDS:   0

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_f16_x_f16_wmma(
    const _Float16* __restrict__ A,    // [M, K] F16
    const _Float16* __restrict__ X,    // [B, K] F16
    float*          __restrict__ Y,    // [B, M] F32
    int M, int K, int B
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid  = threadIdx.x;
    const int lane = tid & 15;

    if (row_start >= M || batch_start >= B) return;

    // Pick row/batch indices for this thread. Threads 0..15 own rows
    // 0..15 of the A fragment; threads 16..31 also hold rows 0..15
    // (duplicate per wave32 WMMA spec). Same for B fragment.
    int my_m = row_start + lane;
    int my_b = batch_start + lane;
    int safe_m = (my_m < M) ? my_m : (M - 1);
    int safe_b = (my_b < B) ? my_b : (B - 1);

    const _Float16* w_row = A + (long long)safe_m * K;
    const _Float16* x_row = X + (long long)safe_b * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // K loop: 16 elements per WMMA step. Issue ONE mma per iter.
    for (int k = 0; k < K; k += 16) {
        // half16_t load = single 32-byte aligned read per thread.
        half16_t a_frag = *(const half16_t*)(w_row + k);
        half16_t x_frag = *(const half16_t*)(x_row + k);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, x_frag, acc);
    }

    // Write 8 outputs per thread following the WMMA layout.
    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
