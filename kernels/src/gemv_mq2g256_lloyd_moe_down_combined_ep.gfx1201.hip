// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1201 TP3 MQ2-Lloyd local down+combine screen.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

static __device__ __forceinline__ float mq2_lloyd_down_row_gfx1201(
    const char* __restrict__ row_ptr,
    const float* __restrict__ x,
    int K,
    int tid) {
    const int boff = tid * 2;
    const int xoff = tid * 8;
    const int groups_per_row = K / 256;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    auto group_fma = [&] (const char* gptr, int base, float& acc) {
        const __half* cb_h = reinterpret_cast<const __half*>(gptr);
        float cb[4] = {
            __half2float(cb_h[0]),
            __half2float(cb_h[1]),
            __half2float(cb_h[2]),
            __half2float(cb_h[3]),
        };
        const unsigned char* data =
            reinterpret_cast<const unsigned char*>(gptr + 8);
        const unsigned char b0 = data[boff];
        const unsigned char b1 = data[boff + 1];
        acc += cb[b0 & 3] * x[base]
            + cb[(b0 >> 2) & 3] * x[base + 1]
            + cb[(b0 >> 4) & 3] * x[base + 2]
            + cb[b0 >> 6] * x[base + 3]
            + cb[b1 & 3] * x[base + 4]
            + cb[(b1 >> 2) & 3] * x[base + 5]
            + cb[(b1 >> 4) & 3] * x[base + 6]
            + cb[b1 >> 6] * x[base + 7];
    };

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;
        const int base = g * 256 + xoff;
        group_fma(gp0, base, acc0);
        group_fma(gp1, base + 256, acc1);
        group_fma(gp2, base + 512, acc2);
        group_fma(gp3, base + 768, acc3);
    }
    const int tail_start = quads << 2;
    for (int g = tail_start; g < tail_start + tail; g++) {
        const char* gp = row_ptr + g * 72;
        const int base = g * 256 + xoff;
        group_fma(gp, base, acc0);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    return acc;
}

// One wave owns one output row. It retains the incumbent per-expert dot and
// six-slot combine order, but never materializes [K_TOP, M] expert outputs.
// Grid [M, 1, 1], block [32]. Decode-only (batch = 1).
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_combined_gfx1201_ep(
    const unsigned long long* __restrict__ expert_ptrs,
    const unsigned long long* __restrict__ ownership_ptrs,
    const void* nonowned_dummy,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float* __restrict__ ffn_out,
    int M,
    int K,
    int K_TOP) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    float combined = 0.0f;

#pragma unroll 1
    for (int krank = 0; krank < 8; krank++) {
        if (krank < K_TOP) {
            const int expert_id = topk_indices[krank];
            float dot = 0.0f;
            if (reinterpret_cast<const void*>(ownership_ptrs[expert_id]) != nonowned_dummy) {
                const char* __restrict__ A =
                    reinterpret_cast<const char*>(expert_ptrs[expert_id]);
                const int row_bytes = (K / 256) * 72;
                const char* row_ptr = A + (long long)row * row_bytes;
                const float* x = rot_batch + (long long)krank * K;
                dot = mq2_lloyd_down_row_gfx1201(row_ptr, x, K, tid);
            }
            if (tid == 0) {
                combined += topk_weights[krank] * dot;
            }
        }
    }

    if (tid == 0) {
        ffn_out[row] += combined;
    }
}
