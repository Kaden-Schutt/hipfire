// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// S7 (dflash draft launch collapse), gfx1100-only. Five symbols that remove
// the draft forward's per-GEMM convert+fill glue, per-sublayer residual
// copies, and finish conv+add pairs. Every symbol is an exact clone of its
// oracle kernel's operation order with ONLY the documented epilogue change:
//
// - mq_rotate_x_f16_dflash_gfx1100: clone of `mq_rotate_x`
//   (kernels/src/gemv_mq4g256.hip) with an F16 store. The stored expression
//   is the identical `v * s * signs2` tree followed by a single rn
//   conversion, matching `rotate f32` + `convert_f32_to_f16` bit-for-bit.
// - gemm_hfq4g256_overwrite_wmma_k2_dflash_gfx1100: clone of
//   `gemm_hfq4g256_residual_wmma_k2` with `Y = acc` instead of `Y += acc`.
//   The caller guarantees Y needs no residual (pure projection), so the
//   pre-zero memset disappears with it.
// - gemm_ksplit_det_overwrite_finalize_dflash_gfx1100: clone of
//   `gemm_ksplit_det_finalize` with the accumulator seeded from +0 instead
//   of the Y residual. Phase 1 reuses the existing
//   `gemm_hfq4g256_residual_wmma_ksplit_det` partial kernel unchanged (it
//   already plain-stores, takes F16 X, and has no residual).
// - rmsnorm_residual_dual_gfx1100: clone of `rmsnorm_f32` (kernels/src/rmsnorm.hip)
//   that additionally writes `residual = x` bitwise in the first pass.
//   Replaces `memcpy_dtod(residual <- x)` + `rmsnorm_batched(x -> x_norm)`.
// - dynamic_conv_residual_gfx1100: clone of `dynamic_causal_conv_f32`
//   (strided DFlash2 variant) with `out = residual + conv(input)`.
//   Replaces `dynamic_causal_conv_f32(input -> tmp)` + `add_f32(residual, tmp -> x)`.
//   `input`, `residual`, and `output` must be pairwise distinct buffers
//   (the draft forward routes conv input through dead conv_temp for this).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ── FWHT rotate with F16 store ──────────────────────────────────────────
// Operation order identical to `mq_rotate_x`: signs1 gather, local
// butterfly (strides 1,2,4), wave butterfly via ds_swizzle (strides
// 1,2,4,8,16), then `v * 0.0625 * signs2`.
// Grid: [groups_total * batch, 1, 1]. Block: [32]. x in/out: [batch × K].
extern "C" __launch_bounds__(32, 16)
__global__ void mq_rotate_x_f16_dflash_gfx1100(
    const float* __restrict__ x_in,
    _Float16* __restrict__ x_out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K
) {
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    const int linear_group = blockIdx.x;
    const int batch = linear_group / groups_total;
    const int group = linear_group - batch * groups_total;

    const long long batch_off = (long long)batch * K;
    const float* x_in_b  = x_in  + batch_off;
    _Float16* x_out_b = x_out + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    float v0 = x_in_b[base]     * signs1[d0];
    float v1 = x_in_b[base + 1] * signs1[d0 + 1];
    float v2 = x_in_b[base + 2] * signs1[d0 + 2];
    float v3 = x_in_b[base + 3] * signs1[d0 + 3];
    float v4 = x_in_b[base + 4] * signs1[d0 + 4];
    float v5 = x_in_b[base + 5] * signs1[d0 + 5];
    float v6 = x_in_b[base + 6] * signs1[d0 + 6];
    float v7 = x_in_b[base + 7] * signs1[d0 + 7];

    // Local butterfly: strides 1, 2, 4
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)

    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float s = 0.0625f;
    x_out_b[base]     = (_Float16)(v0 * s * signs2[d0]);
    x_out_b[base + 1] = (_Float16)(v1 * s * signs2[d0 + 1]);
    x_out_b[base + 2] = (_Float16)(v2 * s * signs2[d0 + 2]);
    x_out_b[base + 3] = (_Float16)(v3 * s * signs2[d0 + 3]);
    x_out_b[base + 4] = (_Float16)(v4 * s * signs2[d0 + 4]);
    x_out_b[base + 5] = (_Float16)(v5 * s * signs2[d0 + 5]);
    x_out_b[base + 6] = (_Float16)(v6 * s * signs2[d0 + 6]);
    x_out_b[base + 7] = (_Float16)(v7 * s * signs2[d0 + 7]);
}

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// ── Overwrite WMMA GEMM (k2 schedule, no residual) ─────────────────────
// Identical dequant + 2× K-tile pipelined WMMA body to
// `gemm_hfq4g256_residual_wmma_k2`; only the epilogue changes from
// `Y += acc` to `Y = acc`.
// Grid: [ceil(M/16), ceil(batch/16)]. Block: [32]. LDS: 0.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_overwrite_wmma_k2_dflash_gfx1100(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < batch_size) ? (batch_start + (tid & 15)) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;

        // Process 2 K-tiles per iteration (8 iterations covers 16 tiles)
        for (int kt = 0; kt < 16; kt += 2) {
            // === Tile A (kt) ===
            const int k_off_a = kt * 16;
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);

            // === Tile B (kt+1) — start loading early ===
            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            // Load both X tiles early (consecutive addresses — prefetcher-friendly)
            half16_t b_a = *(const half16_t*)(xg + k_off_a);
            half16_t b_b = *(const half16_t*)(xg + k_off_b);

            // Dequant tile A
            half16_t a_a;
            #define DQ(reg, i, pk, sh) reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(a_a, 0,  pk0a,  0); DQ(a_a, 1,  pk0a,  4); DQ(a_a, 2,  pk0a,  8); DQ(a_a, 3,  pk0a, 12);
            DQ(a_a, 4,  pk0a, 16); DQ(a_a, 5,  pk0a, 20); DQ(a_a, 6,  pk0a, 24); DQ(a_a, 7,  pk0a, 28);
            DQ(a_a, 8,  pk1a,  0); DQ(a_a, 9,  pk1a,  4); DQ(a_a, 10, pk1a,  8); DQ(a_a, 11, pk1a, 12);
            DQ(a_a, 12, pk1a, 16); DQ(a_a, 13, pk1a, 20); DQ(a_a, 14, pk1a, 24); DQ(a_a, 15, pk1a, 28);

            // WMMA A — b_b load may still be in flight
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            // Dequant tile B (reuse a_a register)
            DQ(a_a, 0,  pk0b,  0); DQ(a_a, 1,  pk0b,  4); DQ(a_a, 2,  pk0b,  8); DQ(a_a, 3,  pk0b, 12);
            DQ(a_a, 4,  pk0b, 16); DQ(a_a, 5,  pk0b, 20); DQ(a_a, 6,  pk0b, 24); DQ(a_a, 7,  pk0b, 28);
            DQ(a_a, 8,  pk1b,  0); DQ(a_a, 9,  pk1b,  4); DQ(a_a, 10, pk1b,  8); DQ(a_a, 11, pk1b, 12);
            DQ(a_a, 12, pk1b, 16); DQ(a_a, 13, pk1b, 20); DQ(a_a, 14, pk1b, 24); DQ(a_a, 15, pk1b, 28);
            #undef DQ

            // WMMA B
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    // So row = 2*j + (tid>>4), col (batch) = tid & 15.
    // OVERWRITE epilogue (S7): pure projection, no Y residual to add.
    const int out_col = batch_start + (tid & 15);  // batch index
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M)
                Y[(long long)out_col * M + out_row] = acc[j];
        }
    }
}

// ── Overwrite ksplit-det finalize (no residual) ────────────────────────
// Identical fixed-order sum to `gemm_ksplit_det_finalize` over the
// [K_SPLITS][batch_size][M] partials, but seeded from +0 instead of the Y
// residual: Y[idx] = sum_z partials[z*N + idx].
// K_SPLITS MUST match the partial kernel (4).
// Grid: [ceil(batch_size*M/256)]. Block: [256]. LDS: 0.
#define K_SPLITS 4

extern "C" __global__ void gemm_ksplit_det_overwrite_finalize_dflash_gfx1100(
    float* __restrict__ Y,
    const float* __restrict__ partials,
    int batch_size, int M
) {
    const long long N = (long long)batch_size * M;
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= N) return;
    float s = 0.0f;
    #pragma unroll
    for (int z = 0; z < K_SPLITS; z++) {
        s += partials[(long long)z * N + idx];
    }
    Y[idx] = s;
}

// ── Dual-output RMSNorm: residual capture + normalize in one launch ───
// Identical two-pass structure and accumulation order to `rmsnorm_f32`:
// pass 1 reduces sum(x^2) with the same thread tiling and the same
// halving tree; pass 2 stores x*weight*rms with the same indexing.
// Additionally pass 1 writes residual = x bitwise.
// Replaces memcpy_dtod(residual <- x) + rmsnorm_batched(x -> out).
// `x` must not alias `residual`; `x` may not alias `out` either (pass 2
// re-reads x after the reduction — the draft call sites use distinct
// scratch planes for both, matching the old kernels' contracts).
// Grid: [batch]. Block: [min(256, n)]. Dynamic shared: block*4 bytes.
extern "C" __global__ void rmsnorm_residual_dual_gfx1100(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ residual,
    float* __restrict__ out,
    int n, float eps
) {
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[blockIdx.x * n + i];
        sum_sq += v * v;
        residual[blockIdx.x * n + i] = v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int idx = blockIdx.x * n + i;
        out[idx] = x[idx] * weight[i] * rms;
    }
}

// ── Fused DFlash2 finish conv + residual add ───────────────────────────
// Identical dynamic causal convolution body to `dynamic_causal_conv_f32`
// (strided DFlash2 variant, kernel_size==2 unrolled), with the epilogue
// `out = residual + acc` replacing the separate `add_f32(residual, tmp)`.
// Add order matches add_f32(a=residual, b=conv) literally.
// `input`, `residual`, and `output` must be pairwise distinct device
// buffers (the draft forward feeds conv input from dead conv_temp).
// Grid: [ceil(rows*hidden/256)]. Block: [256]. LDS: 0.
extern "C" __global__ void dynamic_conv_residual_gfx1100(
    const float* __restrict__ input,
    const float* __restrict__ base,
    const float* __restrict__ dynamic,
    const float* __restrict__ residual,
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

    output[idx] = residual[idx] + acc;
}
