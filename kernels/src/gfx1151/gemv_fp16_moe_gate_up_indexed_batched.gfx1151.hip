// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Full-precision routed MoE gate/up indexed GEMV for gfx1151.
//
// This is the small-prefill companion to the grouped WMMA BF16/F16 kernels:
// it computes only real N*K_TOP routed slots and avoids grouped-MoE padding
// when the all-expert tile bound dominates runtime. Inputs are staged by the
// Rust wrapper to the same precision class as the weights, matching the WMMA
// grouped path's activation rounding.

static __device__ __forceinline__ float bf16_to_f32(unsigned short x) {
    union {
        unsigned int u;
        float f;
    } v;
    v.u = ((unsigned int)x) << 16;
    return v.f;
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_f16_moe_gate_up_k8_indexed_batched_gfx1151(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices,
    const _Float16*           __restrict__ x,
    float*                    __restrict__ y_gate,
    float*                    __restrict__ y_up,
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const _Float16* __restrict__ A =
        reinterpret_cast<const _Float16*>(expert_ptrs[expert_id]);
    const _Float16* __restrict__ x_row = x + (long long)bid * K;
    const _Float16* __restrict__ w_row = A + (long long)row * K;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int k = tid; k < K; k += 128) {
        acc0 += (float)w_row[k] * (float)x_row[k];
        if (k + 32 < K) acc1 += (float)w_row[k + 32] * (float)x_row[k + 32];
        if (k + 64 < K) acc2 += (float)w_row[k + 64] * (float)x_row[k + 64];
        if (k + 96 < K) acc3 += (float)w_row[k + 96] * (float)x_row[k + 96];
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    if (tid == 0) {
        const int mi = M >> 1;
        const long long base = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base + row] = acc;
        } else {
            y_up[base + (row - mi)] = acc;
        }
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_bf16_moe_gate_up_k8_indexed_batched_gfx1151(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices,
    const unsigned short*     __restrict__ x,
    float*                    __restrict__ y_gate,
    float*                    __restrict__ y_up,
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const unsigned short* __restrict__ A =
        reinterpret_cast<const unsigned short*>(expert_ptrs[expert_id]);
    const unsigned short* __restrict__ x_row = x + (long long)bid * K;
    const unsigned short* __restrict__ w_row = A + (long long)row * K;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int k = tid; k < K; k += 128) {
        acc0 += bf16_to_f32(w_row[k]) * bf16_to_f32(x_row[k]);
        if (k + 32 < K) acc1 += bf16_to_f32(w_row[k + 32]) * bf16_to_f32(x_row[k + 32]);
        if (k + 64 < K) acc2 += bf16_to_f32(w_row[k + 64]) * bf16_to_f32(x_row[k + 64]);
        if (k + 96 < K) acc3 += bf16_to_f32(w_row[k + 96]) * bf16_to_f32(x_row[k + 96]);
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    if (tid == 0) {
        const int mi = M >> 1;
        const long long base = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base + row] = acc;
        } else {
            y_up[base + (row - mi)] = acc;
        }
    }
}
