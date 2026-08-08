// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Atomic-free MQ2-Lloyd K4 MoE down GEMV. Same per-(token, krank) compute
// as `gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4`,
// but writes each (token, krank) result to its own row in
// `expert_outputs[N × K_TOP × M]` instead of `atomicAdd(scale * acc)`-ing
// into a shared `x_residual[N × M]` cell. Pair with
// `moe_down_combine_k8_batched` to fold K_TOP outputs into `x_residual`
// with `topk_weights` applied — non-atomic, deterministic per-cell sum.
//
// Determinism: the original residual_scaled kernel is non-deterministic
// at temp=0 because K_TOP=6 grid blocks contend on the same x_residual
// cell via atomicAdd → FP add order is hw-defined and not reproducible
// across runs. This expanded variant has exactly one writer per
// expert_outputs cell → bit-reproducible. See memory
// `feedback_deepseek4_prefill_nondeterministic_at_temp0` for the discovery
// trail.
//
// Shapes (vs the residual_scaled variant):
//   expert_ptrs    : [n_exp]                — unchanged
//   topk_indices   : [N × K_TOP]            — unchanged
//   rot_batch      : [N × K_TOP × K]        — unchanged
//   expert_outputs : [N × K_TOP × M]        — replaces x_residual; no atomic
// `topk_weights` is NOT consumed here — combine applies it during reduction.
//
// Grid: (M, K_TOP, N). Block = 32 threads. Same as the residual_scaled
// kernel — caller can swap the dispatch with no grid/block changes.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_expanded_k4(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M]
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
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 2;
    const int xoff = tid * 8;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
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
        const unsigned char* data = (const unsigned char*)(gptr + 8);
        unsigned char b0 = data[boff];
        unsigned char b1 = data[boff + 1];
        acc += cb[b0 & 3]        * x[base]
            +  cb[(b0 >> 2) & 3] * x[base + 1]
            +  cb[(b0 >> 4) & 3] * x[base + 2]
            +  cb[b0 >> 6]       * x[base + 3]
            +  cb[b1 & 3]        * x[base + 4]
            +  cb[(b1 >> 2) & 3] * x[base + 5]
            +  cb[(b1 >> 4) & 3] * x[base + 6]
            +  cb[b1 >> 6]       * x[base + 7];
    };

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;
        const int base = g * 256 + xoff;
        group_fma(gp0, base,         acc0);
        group_fma(gp1, base + 256,   acc1);
        group_fma(gp2, base + 512,   acc2);
        group_fma(gp3, base + 768,   acc3);
    }
    const int tail_start = quads << 2;
    for (int g = tail_start; g < tail_start + tail; g++) {
        const char* gp = row_ptr + g * 72;
        const int base = g * 256 + xoff;
        group_fma(gp, base, acc0);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        expert_outputs[(long long)bid * K_TOP * M + (long long)krank * M + row] = acc;
    }
}

// ---------------------------------------------------------------------------
// Reproducible int64-accumulating expanded variant (MQ2-Lloyd)
//
// Same per-cell write as gemv_mq2g256_lloyd_moe_down_expanded_k4 (no atomics,
// one writer per expert_outputs cell), but the inner K-dot uses int64
// fixed-point accumulation (S=2^24) for partition invariance under K-split.
//
// Per-group partial is warp-reduced in FP32 (same as base), then converted to
// i64 before cross-group accumulation — same scheme as the residual i64 variant
// but writing to the expanded output buffer rather than atomicAdd'ing.
//
// expert_outputs_i64: [N × K_TOP × M] raw unsigned long long (signed i64 bits).
// Call `moe_i64_residual_to_f32` element-wise on each cell to convert.
// topk_weights is NOT consumed here — the combine step applies it.
// ---------------------------------------------------------------------------

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_expanded_i64_k4(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    unsigned long long* __restrict__ expert_outputs_i64, // [N × K_TOP × M] int64
    int M, int K, int K_TOP
) {
    const int row   = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid   = blockIdx.z;
    const int tid   = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 2;

    __shared__ float cb_lds[4];

    long long acc_i64 = 0LL;
    const float S = (float)(1 << 24);

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + g * 72;

        if (tid < 4) {
            cb_lds[tid] = __half2float(((const __half*)gp)[tid]);
        }
        __syncthreads();

        const unsigned char* d = (const unsigned char*)(gp + 8) + boff;
        unsigned char b0 = d[0];
        unsigned char b1 = d[1];
        int base = g * 256 + tid * 8;

        float partial =
              cb_lds[ b0        & 3u] * x[base]
            + cb_lds[(b0 >>  2) & 3u] * x[base + 1]
            + cb_lds[(b0 >>  4) & 3u] * x[base + 2]
            + cb_lds[ b0 >>  6      ] * x[base + 3]
            + cb_lds[ b1        & 3u] * x[base + 4]
            + cb_lds[(b1 >>  2) & 3u] * x[base + 5]
            + cb_lds[(b1 >>  4) & 3u] * x[base + 6]
            + cb_lds[ b1 >>  6      ] * x[base + 7];

        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down(partial, offset);

        // Lane 0: accumulate per-group partial into int64 (no topk_w here —
        // weight is applied by the combine step after expand).
        if (tid == 0) {
            acc_i64 += (long long)llroundf(partial * S);
        }
        __syncthreads();
    }

    if (tid == 0) {
        expert_outputs_i64[(long long)bid * K_TOP * M + (long long)krank * M + row] =
            (unsigned long long)acc_i64;
    }
}
