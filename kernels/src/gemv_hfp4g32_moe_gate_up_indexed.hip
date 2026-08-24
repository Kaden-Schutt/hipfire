// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE gate_up GEMV for HFP4G32 (OCP E2M1 nibbles + UE8M0
// per-32 block scale + FP16 row scale). Sibling of
// gemv_mq3g256_lloyd_moe_gate_up_indexed — same grid and expert-ptr dispatch,
// but the per-block decode is an E2M1 LUT lookup scaled by 2^(e-127) instead of
// a Lloyd codebook over packed indices.
//
// This exists so DeepSeek V4's NATIVE FP4 experts can be served without
// requantization: the checkpoint is already E2M1 + UE8M0 g32, so a passthrough
// repack (see --format deepseek4-fp4) is bit-exact and this kernel reads it
// directly.
//
// X IS NOT ROTATED. The MQ-family kernels take an FWHT-pre-rotated x because
// their format bakes in that rotation; HFP4G32 does not (format_flags bit 0
// clear), so the caller must pass the PLAIN activation vector. Passing a
// rotated x here silently produces wrong results — the two paths differ only
// in which x buffer they read.
//
// Row layout (16 + 17 * K/32 bytes):
//   [0..2)   : fp16 row_scale_a
//   [2..4)   : fp16 row_scale_b  (reserved, always 0 — not read)
//   [4..6)   : u16 block_count
//   [6]      : u8  format_flags  (0 = no rotation for this path)
//   [7..16)  : reserved
//   then per 32-element block, 17 B: [0] UE8M0 exponent, [1..17) 16 packed
//   nibbles (low nibble = even element, high nibble = odd).
//
// Dequant: value = row_scale_a * 2^(e - 127) * E2M1_LUT[nibble]
// row_scale_a hoists out of the K loop; the block scale multiplies each
// block's partial before accumulation, matching the reference dequant's
// association order closely enough for f32 accumulation.
//
// Grid: [M, k_top, 1], block [32]. Output split: row < M/2 → y_gate, else y_up.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfp4g32_moe_gate_up_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,               // [k_top]
    const float* __restrict__ x,                        // [K] PLAIN (not rotated)
    float* __restrict__ y_gate,                         // [k_top × M/2]
    float* __restrict__ y_up,                           // [k_top × M/2]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    // 16-entry signed E2M1 LUT. Indexed directly by the nibble: bit 3 is the
    // sign, so entries 8..15 mirror 0..7 negated (0x8 is -0.0, which is
    // harmless in an f32 accumulate).
    __shared__ float lut[16];
    if (tid < 16) {
        const float E2M1[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        lut[tid] = E2M1[tid];
    }
    __syncthreads();

    const int n_blocks = K >> 5;               // K / 32
    const int row_bytes = 16 + n_blocks * 17;
    const char* row_ptr = A + (long long)row * row_bytes;

    const float row_scale = __half2float(*(const __half*)row_ptr);

    // Each thread strides over blocks; 32 threads, one 32-element block each.
    float acc = 0.0f;
    for (int b = tid; b < n_blocks; b += 32) {
        const unsigned char* bp =
            (const unsigned char*)(row_ptr + 16 + b * 17);
        const int e = (int)bp[0];
        // 2^(e-127); ldexpf keeps this exact for the whole UE8M0 range.
        const float bscale = ldexpf(1.0f, e - 127);
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
        const int mi = M >> 1;
        if (row < mi) {
            y_gate[krank * mi + row] = acc;
        } else {
            y_up[krank * mi + (row - mi)] = acc;
        }
    }
}
