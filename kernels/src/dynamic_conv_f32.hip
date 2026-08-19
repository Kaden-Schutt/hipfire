// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Dynamic causal convolution: left-zero-padded, grouped per-position dynamic kernel.
//
// output[row,c] = sum_{off=0..kernel_size-1, row>=off}
//     (base[off, c] + dynamic[row, off, c/group_size]) * input[row-off, c]
//
// Layouts (row-major, F32):
//   input   [rows, hidden]                     -> input[row*hidden + c]
//   base    [kernel_size, hidden]              -> base[off*hidden + c]
//   dynamic compact [rows, kernel_size, groups] -> dynamic[(row*kernel_size + off)*groups + g]
//   dynamic strided [rows, stride] with window  -> dynamic[row*dynamic_row_stride + dynamic_offset + off*groups + g]
//             where groups = hidden / group_size, g = c / group_size
//             compact: dynamic_row_stride = kernel_size*groups, dynamic_offset = 0
//             2-phase (DFlash2 kernel_projection [B,2,kernel_size,groups]): stride = 2*kernel_size*groups
//             phase 0 -> offset 0, phase 1 -> offset kernel_size*groups
//   output  [rows, hidden]                     -> output[row*hidden + c]
//
// kernel_size is runtime; kernel_size==2 is unrolled for DFlash2 (group 16) but the
// generic loop remains correct for any kernel_size when that value is passed.

// ── Strided variant (preferred): supports 2-phase [B,2,K,G] without repacking ──
extern "C" __global__ void dynamic_causal_conv_f32(
    const float* __restrict__ input,
    const float* __restrict__ base,
    const float* __restrict__ dynamic,
    float* __restrict__ output,
    int rows,
    int hidden,
    int kernel_size,
    int groups,
    int group_size,
    int dynamic_row_stride,
    int dynamic_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * hidden;
    if (idx >= total) {
        return;
    }
    int r = idx / hidden;
    int c = idx % hidden;
    int g = c / group_size;

    float acc = 0.0f;

    // Optimized DFlash2 path: kernel_size == 2.
    // Branch is uniform across the grid (kernel_size is scalar), so no divergence.
    if (kernel_size == 2) {
        // off = 0
        float b0 = base[c];
        float d0 = dynamic[r * dynamic_row_stride + dynamic_offset + g];
        float w0 = b0 + d0;
        float x0 = input[r * hidden + c];
        acc += w0 * x0;
        // off = 1, causal guard r >= 1
        if (r >= 1) {
            float b1 = base[hidden + c];
            float d1 = dynamic[r * dynamic_row_stride + dynamic_offset + groups + g];
            float w1 = b1 + d1;
            float x1 = input[(r - 1) * hidden + c];
            acc += w1 * x1;
        }
    } else {
        // Generic causal loop: left-zero-padded.
        for (int off = 0; off < kernel_size; ++off) {
            if (r < off) {
                continue;
            }
            float b = base[off * hidden + c];
            float d = dynamic[r * dynamic_row_stride + dynamic_offset + off * groups + g];
            float w = b + d;
            float x = input[(r - off) * hidden + c];
            acc += w * x;
        }
    }

    output[idx] = acc;
}

// ── Compact variant (original contract): contiguous [rows, kernel_size, groups] ──
// Provided for backwards compatibility and for the compact wrapper. Functionally
// equivalent to dynamic_causal_conv_f32 with stride=kernel_size*groups, offset=0.
extern "C" __global__ void dynamic_conv_f32(
    const float* __restrict__ input,
    const float* __restrict__ base,
    const float* __restrict__ dynamic,
    float* __restrict__ output,
    int rows,
    int hidden,
    int kernel_size,
    int groups,
    int group_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * hidden;
    if (idx >= total) {
        return;
    }
    int r = idx / hidden;
    int c = idx % hidden;
    int g = c / group_size;

    float acc = 0.0f;

    if (kernel_size == 2) {
        float b0 = base[c];
        float d0 = dynamic[(r * 2 + 0) * groups + g];
        float w0 = b0 + d0;
        float x0 = input[r * hidden + c];
        acc += w0 * x0;
        if (r >= 1) {
            float b1 = base[hidden + c];
            float d1 = dynamic[(r * 2 + 1) * groups + g];
            float w1 = b1 + d1;
            float x1 = input[(r - 1) * hidden + c];
            acc += w1 * x1;
        }
    } else {
        for (int off = 0; off < kernel_size; ++off) {
            if (r < off) {
                continue;
            }
            float b = base[off * hidden + c];
            float d = dynamic[(r * kernel_size + off) * groups + g];
            float w = b + d;
            float x = input[(r - off) * hidden + c];
            acc += w * x;
        }
    }

    output[idx] = acc;
}
