// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Nous Research
// hipfire — see LICENSE and NOTICE in the project root.
//
// Q8_0 block quantize: FP32 → Q8_0 (2-byte f16 scale + 32 int8 values = 34 bytes per 32 elems).
// Uses warp shuffle for block-level max reduction (no LDS races).
// Grid = n_blocks (= ceil(n_elems / 32)), one block per 32-element group.
// Each block of the grid processes exactly 32 elements.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __launch_bounds__(256, 4)
__global__ void quant_f32_to_q8(
    const float* __restrict__ src,
    int8_t* __restrict__ dst,
    int n_elems
) {
    // One block per 32-element Q8 group, block_size=32
    int block_idx = blockIdx.x;
    int base = block_idx * 32;
    int tid = threadIdx.x;
    if (tid >= 32) return; // only first 32 threads in the block are used

    int gid = base + tid;
    float val = (gid < n_elems) ? src[gid] : 0.0f;
    float abs_val = fabsf(val);

    // Warp shuffle reduction over 32 threads to find max abs
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other = __shfl_xor(abs_val, mask, 32);
        abs_val = fmaxf(abs_val, other);
    }
    // Now thread 0 has the max abs in abs_val (replicated to all threads via shfl)

    float block_absmax = __shfl(abs_val, 0, 32);
    float scale_f32 = block_absmax / 127.0f;
    float inv_scale = (block_absmax > 1e-10f) ? (127.0f / block_absmax) : 0.0f;
    int8_t qval = (gid < n_elems) ? (int8_t)__float2int_rn(val * inv_scale) : 0;

    // Thread 0 writes the f16 scale, all threads write their byte
    int dst_block_base = block_idx * 34;
    if (tid == 0) {
        *(__half*)(dst + dst_block_base) = (__half)scale_f32;
    }
    dst[dst_block_base + 2 + tid] = qval;
}
