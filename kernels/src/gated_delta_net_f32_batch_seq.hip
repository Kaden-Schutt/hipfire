// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define HD 128
#define TILE_ROWS 4

// Batched-sequential FP32 GDN recurrence — the full-precision counterpart of
// gated_delta_net_q8 (gated_delta_net_q8.hip). Processes n_tokens sequentially
// with the same 32-thread × 32-tile decomposition (1024 threads/head), so it
// has the SAME parallelism/throughput as the Q8 batch kernel — unlike the
// 128-thread/head single-token `gated_delta_net` (gated_delta_net.hip), which
// is fine for n_tokens=1 decode but ~4-5× too slow for the B-token prefill/
// verify batch.
//
// Difference from gated_delta_net_q8: the state lives in FP32 the whole way,
// so there is NO dequant-on-load, NO requant-on-store, no per-row absmax
// scale, and no stochastic-rounding `frame` entropy. S is loaded from the
// persistent FP32 state into LDS once, advanced in place across the batch,
// and written back once. The recurrence body (lines marked "identical to Q8")
// is byte-identical to gated_delta_net_q8's inner loop.
//
// This is the linear (non-tree) sibling of gated_delta_net_f32_tree; the only
// difference there is the per-node tape read/write for tree-DFS parents.

extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f32_batch_seq(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ s_f32,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    __shared__ float S_tile[TILE_ROWS * HD];  // 4 KiB

    // Load FP32 S tile → LDS once (no dequant). Stride-32 traversal is
    // coalesced for the global read and bank-conflict-free for the LDS write.
    float* s_base = s_f32 + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        S_tile[i] = s_base[i];
    }

    // ── Recurrence (identical to Q8 gated_delta_net inner loop) ───────────
    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha = expf(gate[t * n_heads + h]);
        float beta_v = beta[t * n_heads + h];

        float k0 = k_t[tid], k1 = k_t[tid+32], k2 = k_t[tid+64], k3 = k_t[tid+96];
        float q0 = q_t[tid], q1 = q_t[tid+32], q2 = q_t[tid+64], q3 = q_t[tid+96];

        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            float* sr = S_tile + r * HD;
            float s0 = sr[tid], s1 = sr[tid+32], s2 = sr[tid+64], s3 = sr[tid+96];

            float kv = s0*k0 + s1*k1 + s2*k2 + s3*k3;
            for (int o = 16; o > 0; o >>= 1)
                kv += __shfl_down(kv, o);
            kv = __shfl(kv, 0);

            float delta = (v_t[row] - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            sr[tid] = s0; sr[tid+32] = s1; sr[tid+64] = s2; sr[tid+96] = s3;

            float out_v = s0*q0 + s1*q1 + s2*q2 + s3*q3;
            for (int o = 16; o > 0; o >>= 1)
                out_v += __shfl_down(out_v, o);
            if (tid == 0)
                output[t * stride + h * HD + row] = out_v;
        }
    }

    // Write the advanced FP32 S tile back to the persistent state (no requant).
    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        s_base[i] = S_tile[i];
    }
}
