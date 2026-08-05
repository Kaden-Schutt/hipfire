// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Batched (prefill) MoE gate_up GEMV for HFP4G32 — native-FP4 routed experts.
// Batched sibling of gemv_hfp4g32_moe_gate_up_indexed; same shape contract as
// gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4:
//
//   grid [M, K_TOP, N], block [32]
//   topk_indices : [N × K_TOP]
//   x            : [N × K]
//   y_gate/y_up  : [N × K_TOP × MI], MI = M/2
//
// X IS NOT ROTATED — HFP4G32 bakes in no FWHT (format_flags bit 0 clear), so
// the caller passes the PLAIN per-position activations. The MQ-family batched
// kernels take a rotated buffer; mixing them up is silently wrong, not an error.
//
// Row layout: 16 B header (fp16 row_scale_a at 0) + 17 B per 32-element block
// (UE8M0 exponent, then 16 packed nibbles; low nibble = even element).
// Dequant: row_scale_a * 2^(e-127) * E2M1_LUT[nibble].

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfp4g32_moe_gate_up_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,               // [N × K_TOP]
    const float* __restrict__ x,                        // [N × K] PLAIN
    float* __restrict__ y_gate,                         // [N × K_TOP × MI]
    float* __restrict__ y_up,                           // [N × K_TOP × MI]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;

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
        const float* xb = x_row + (b << 5);
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
        const int mi = M >> 1;
        const long long stride = (long long)K_TOP * mi;
        if (row < mi) {
            y_gate[(long long)bid * stride + krank * mi + row] = acc;
        } else {
            y_up[(long long)bid * stride + krank * mi + (row - mi)] = acc;
        }
    }
}
