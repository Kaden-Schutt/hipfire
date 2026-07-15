// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Dequant BQ1G128 packed weights (PrismML Q1_0-layout binary) to row-major FP16.
// Input:  A_packed [M × groups_per_row × 18 bytes] (1-bit sign, FP16 scale at block start)
// Output: W_f16 [M × K] FP16 (row-major). Grid [M, groups_per_row], Block [32].
// bit=1 -> +d, bit=0 -> -d.
extern "C" __launch_bounds__(32, 16)
__global__ void dequant_bq1g128_to_f16(
    const char* __restrict__ A_packed, _Float16* __restrict__ W_f16, int M, int K
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 128;
    const char* gp = A_packed + (long long)row * groups_per_row * 18 + group * 18;
    float d = (float)(*(const _Float16*)gp);
    unsigned char b = (unsigned char)gp[2 + (tid >> 1)];
    int sh = (tid & 1) * 4;
    _Float16* out = W_f16 + (long long)row * K + group * 128 + tid * 4;
    out[0] = (_Float16)(((b >> (sh + 0)) & 1) ? d : -d);
    out[1] = (_Float16)(((b >> (sh + 1)) & 1) ? d : -d);
    out[2] = (_Float16)(((b >> (sh + 2)) & 1) ? d : -d);
    out[3] = (_Float16)(((b >> (sh + 3)) & 1) ? d : -d);
}
