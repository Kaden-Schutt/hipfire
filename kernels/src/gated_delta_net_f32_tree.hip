// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define HD 128
#define TILE_ROWS 4

// FP32 tree-aware variant of gated_delta_net (linear: gated_delta_net.hip,
// Q8 tree: gated_delta_net_q8_tree.hip). Processes tokens in tree-DFS order,
// each token's S-state read from either (a) its in-block parent's slot in
// the F32 s_tape, or (b) the pre-block initial state s_f32_init when
// parent_indices[t] < 0 (root of the tree).
//
// After each token, the updated S-tile is written back to s_tape_f32[t] so
// later tokens whose parent is t read the post-update S — the same
// "persist-write SSM intermediate" pattern as the Q8 tree kernel, which is
// what makes sibling tokens see the parent's post-update S (not the previous
// sibling's) and keeps DDTree tree-verify correct.
//
// Difference from the Q8 tree kernel: the state lives in FP32 the whole way
// through, so there is NO per-token dequant-on-read or requant-on-write and
// NO per-row absmax scale. The recurrence body is byte-identical to the Q8
// kernel; only the LDS load/store of S change. This is the spec-decode
// counterpart of the FP32 `StateQuant` decode path (qwen35.rs
// forward_scratch_layers / forward_prefill_chunk FP32 arms): full-precision
// DeltaNet state, no per-step Q8 rounding noise.
//
// Spine topology (parent_indices = [-1, 0, 1, 2, ...]) produces outputs
// byte-exact with the linear FP32 kernel (gated_delta_net.hip) called with
// n_tokens=1 N times in sequence, using the rolling S as both input and
// output per call.

extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f32_tree(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const float* __restrict__ s_f32_init,
    float* __restrict__ s_tape_f32,
    const int* __restrict__ parent_indices,
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

    // Per-token stride to reach tile (h, row_start..row_start+TILE_ROWS)
    // inside s_tape_f32[t * n_heads * HD * HD + ...]. Same element layout as
    // the Q8 tape, just float instead of signed char and no scales side-table.
    const int head_tile_offset = h * HD * HD + row_start * HD;
    const int token_stride      = n_heads * HD * HD;

    for (int t = 0; t < n_tokens; t++) {
        // Resolve which FP32 tile to read into LDS: parent's tape slot, or
        // the pre-block initial state at the root.
        int parent = parent_indices[t];
        const float* s_src;
        if (parent >= 0) {
            s_src = s_tape_f32 + parent * token_stride + head_tile_offset;
        } else {
            s_src = s_f32_init + head_tile_offset;
        }

        // Load FP32 tile → LDS (no dequant). Stride-32 pattern matches the
        // Q8 variant for bank-conflict-free LDS writes.
        for (int i = tid; i < TILE_ROWS * HD; i += 32) {
            S_tile[i] = s_src[i];
        }

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

        // Persist FP32 LDS → tape slot for token t (no requant, no scales).
        float* dst_tile = s_tape_f32 + t * token_stride + head_tile_offset;
        for (int i = tid; i < TILE_ROWS * HD; i += 32) {
            dst_tile[i] = S_tile[i];
        }
    }
}
