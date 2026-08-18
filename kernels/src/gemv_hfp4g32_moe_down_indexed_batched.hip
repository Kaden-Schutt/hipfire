// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Batched (prefill) MoE down GEMV for HFP4G32 with route-scaled residual
// accumulate. Batched sibling of
// gemv_hfp4g32_moe_down_residual_scaled_indexed; same shape and output
// contract as gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4:
//
//   grid [M, K_TOP, N], block [32]
//   topk_indices / topk_weights : [N × K_TOP]
//   plain_batch                 : [N × K_TOP × K]
//   x_residual                  : [N × M]  (atomically accumulated)
//
// X IS NOT ROTATED — HFP4G32 carries no baked FWHT, so the caller passes the
// PLAIN per-expert activation batch (the SwiGLU output BEFORE the MQ rotate).
// The MQ siblings consume the rotated batch; swapping them is silently wrong.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfp4g32_moe_down_residual_scaled_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,             // [N × K_TOP]
    const float* __restrict__ topk_weights,             // [N × K_TOP]
    const float* __restrict__ plain_batch,              // [N × K_TOP × K] PLAIN
    float*       __restrict__ x_residual,               // [N × M]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        plain_batch + ((long long)bid * K_TOP + krank) * K;

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
        const unsigned char* bp = (const unsigned char*)(row_ptr + 16 + b * 17);
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
        const float scale = topk_weights[routing_base + krank];
        atomicAdd(&x_residual[(long long)bid * M + row], scale * acc);
    }
}
