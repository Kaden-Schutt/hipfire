// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware MoE down GEMV for HFP4G32 with route-scaled residual accumulate.
// Sibling of gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed — identical
// dispatch and output contract, HFP4G32 decode instead of a Lloyd codebook.
//
// Serves DeepSeek V4's native FP4 experts without requantization; see
// gemv_hfp4g32_moe_gate_up_indexed for the format notes.
//
// X IS NOT ROTATED. The MQ-family sibling reads an FWHT-rotated per-expert
// activation batch because its format bakes that rotation in; HFP4G32 does not.
// The caller passes the PLAIN per-expert batch. Getting this wrong is silent —
// the results are simply wrong, not an error.
//
// Output contract matches the sibling exactly: each (row, krank) pair
// atomically accumulates topk_weights[krank] * dot into x_residual[row], so the
// caller sees the route-weighted expert sum without a separate combine pass.
//
// Grid: [M, k_top, 1], block [32].

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfp4g32_moe_down_residual_scaled_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int*   __restrict__ topk_indices,             // [k_top]
    const float* __restrict__ topk_weights,             // [k_top]
    const float* __restrict__ plain_batch,              // [k_top × K] PLAIN
    float*       __restrict__ x_residual,               // [M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = plain_batch + (long long)krank * K;

    __shared__ float lut[16];
    if (tid < 16) {
        const float E2M1[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        lut[tid] = E2M1[tid];
    }
    __syncthreads();

    const int n_blocks = K >> 5;
    const int row_bytes = 16 + n_blocks * 17;
    const char* row_ptr = A + (long long)row * row_bytes;

    const float row_scale = __half2float(*(const __half*)row_ptr);

    float acc = 0.0f;
    for (int b = tid; b < n_blocks; b += 32) {
        const unsigned char* bp =
            (const unsigned char*)(row_ptr + 16 + b * 17);
        const float bscale = ldexpf(1.0f, (int)bp[0] - 127);
        const float* xb = x + (b << 5);

        float part = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned int byte = bp[1 + i];
            part += lut[byte & 0x0Fu] * xb[2 * i];
            part += lut[(byte >> 4) & 0x0Fu] * xb[2 * i + 1];
        }
        acc += part * bscale;
    }
    acc *= row_scale;

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        atomicAdd(&x_residual[row], topk_weights[krank] * acc);
    }
}
