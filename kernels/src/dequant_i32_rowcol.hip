// SPDX-License-Identifier: Apache-2.0
// hipfire — reference kernel layer: int32 → f32 dequant by row·col scales.
//
// The iu8/iu4 WMMA cores accumulate int32 with NO scales inside. This applies the
// W8A8/W4A8 dequant: y_f32[b,m] = y_i32[b,m] · x_scale[b] · w_scale[m], where
// x_scale is per-token (row) and w_scale is per-output-channel (col). One thread
// per output element, zero LDS.
//
// Grid: [ceil(B*M / 256)]. Block: [256].

#include <hip/hip_runtime.h>

__launch_bounds__(256)
extern "C" __global__ void dequant_i32_rowcol(
    const int* __restrict__ Y_I32, // [B, M] int32
    const float* __restrict__ X_SCALE, // [B]
    const float* __restrict__ W_SCALE, // [M]
    float* __restrict__ Y_F32,     // [B, M]
    int b_rows, int m
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)b_rows * m;
    if (i >= n) return;
    const int b = (int)(i / m);
    const int col = (int)(i - (long long)b * m);
    Y_F32[i] = (float)Y_I32[i] * X_SCALE[b] * W_SCALE[col];
}
