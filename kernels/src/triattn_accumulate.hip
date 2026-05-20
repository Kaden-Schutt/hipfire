// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// TriAttention sidecar calibration: per-band statistics accumulator.
//
// For each (layer, head, band) triple, the sidecar needs:
//   - sum_re    : Σ real(q)            over all sampled tokens
//   - sum_im    : Σ imag(q)
//   - sum_abs   : Σ |q| = Σ sqrt(re² + im²)    ← sqrt was the CPU bottleneck
//   - count     : # samples
//
// CPU version: per-token, per-layer, acquires a mutex, runs a serial loop
// of n_heads × n_bands accumulator updates with f64 sqrt. Measured 99%+
// of wall time on MI300X EPYC host with rayon-per-sample giving only
// ~15% speedup (rayon overhead dominates small work units).
//
// GPU version: one kernel launch per (chunk, layer). Q is already on GPU
// (`fa_q_batch` in qwen35.rs) — no PCIe download. Grid = [n_heads, n_bands, 1].
// Each block handles one (head, band) pair across all n_tokens in the chunk,
// reduces partial sums in shared memory, adds into the persistent device-side
// accumulator with a single writer (tid==0). No cross-block contention: each
// (layer, head, band) is written by exactly one block at a time.
//
// Expected speedup on MI300X: 5-8× vs CPU baseline. Expected speedup on
// gfx1100 (7900 XTX): 4-6×.
extern "C" __global__ void triattn_accumulate_f32(
    const float* __restrict__ q,              // [n_tokens, n_heads, head_dim]
    double* __restrict__ accs_sum_re,         // [n_layers, n_heads, n_bands]
    double* __restrict__ accs_sum_im,         // [n_layers, n_heads, n_bands]
    double* __restrict__ accs_sum_abs,        // [n_layers, n_heads, n_bands]
    unsigned long long* __restrict__ accs_count, // [n_layers, n_heads, n_bands]
    int n_tokens, int n_heads, int head_dim,
    int layer_idx
) {
    const int head = blockIdx.x;
    const int band = blockIdx.y;
    const int tid = threadIdx.x;
    const int tcount = blockDim.x;
    const int n_bands = head_dim / 2;

    // Per-thread partials across the time axis.
    double p_re = 0.0;
    double p_im = 0.0;
    double p_abs = 0.0;
    unsigned long long p_count = 0;

    for (int t = tid; t < n_tokens; t += tcount) {
        const int q_base = t * n_heads * head_dim + head * head_dim + 2 * band;
        const float re = q[q_base];
        const float im = q[q_base + 1];
        p_re    += (double)re;
        p_im    += (double)im;
        p_abs   += sqrt((double)(re * re + im * im));
        p_count += 1;
    }

    // Shared-memory block reduction. Sized for blockDim.x ≤ 64 — one wave on
    // MI300X wave64 or two waves on gfx1100 wave32. Smaller blocks are fine;
    // larger would need external shmem.
    __shared__ double s_re[64];
    __shared__ double s_im[64];
    __shared__ double s_abs[64];
    __shared__ unsigned long long s_count[64];

    s_re[tid]    = p_re;
    s_im[tid]    = p_im;
    s_abs[tid]   = p_abs;
    s_count[tid] = p_count;
    __syncthreads();

    // Tree reduction. Safe for power-of-two tcount ≤ 64.
    for (int stride = tcount / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_re[tid]    += s_re[tid + stride];
            s_im[tid]    += s_im[tid + stride];
            s_abs[tid]   += s_abs[tid + stride];
            s_count[tid] += s_count[tid + stride];
        }
        __syncthreads();
    }

    // Single writer per block — no contention across blocks (each (h, band)
    // is unique), so plain read-modify-write is safe. The accs arrays
    // accumulate across chunks, so we add rather than overwrite.
    if (tid == 0) {
        const int acc_idx = (layer_idx * n_heads + head) * n_bands + band;
        accs_sum_re[acc_idx]    += s_re[0];
        accs_sum_im[acc_idx]    += s_im[0];
        accs_sum_abs[acc_idx]   += s_abs[0];
        accs_count[acc_idx]     += s_count[0];
    }
}
