// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Exact-gfx1201 DS4 TP/EP deterministic down projection. Non-owned down
// pointer-table entries intentionally reuse a valid compact weight pointer,
// so ownership is checked through the matching gate/up table and its zero
// dummy. Owned blocks preserve the portable K4 arithmetic and reduction order.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_expanded_k4_gfx1201_ep(
    const unsigned long long* __restrict__ expert_ptrs,
    const unsigned long long* __restrict__ ownership_ptrs,
    const void* nonowned_dummy,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs,
    int M,
    int K,
    int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const long long output_index =
        (long long)bid * K_TOP * M + (long long)krank * M + row;
    if (reinterpret_cast<const void*>(ownership_ptrs[expert_id]) == nonowned_dummy) {
        if (tid == 0) {
            expert_outputs[output_index] = 0.0f;
        }
        return;
    }

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 2;
    const int xoff = tid * 8;

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

    if (tid == 0) {
        expert_outputs[output_index] = acc;
    }
}
