// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define HD 128
#define TILE_ROWS 4

// Tree-aware variant of gated_delta_net_q8. Processes tokens in tree-DFS
// order, with each token's S-state read from either (a) its in-block
// parent's slot in s_tape, or (b) the pre-block initial state s_q8_init
// when parent_indices[t] < 0 (root of the tree).
//
// After each token, the updated S-tile is requantized back to s_tape[t]
// so subsequent tokens whose parent is t can read it. This is the
// "persist-write SSM intermediate" pattern — sibling tokens see the
// parent's post-update S, not the previous sibling's (avoids the
// sibling pollution that breaks DDTree correctness when using the
// linear-seq kernel on a tree-verify forward).
//
// Differences from linear variant:
// - Per-token dequant + requant of S-tile (instead of once at
//   start/end). Adds ~2x LDS↔global byte traffic per token and
//   accumulates Q8 rounding noise at each step rather than only at
//   end-of-batch. This matches what calling the linear kernel with
//   n_tokens=1 N times in sequence already does, so correctness vs
//   that reference is byte-exact modulo the parent-chain read.
// - s_q8_init / s_scales_init are READ-ONLY (pre-block snapshot).
//   Caller runs linear replay on accepted spine post-acceptance to
//   advance the persistent DN state (same pattern as the tree-conv1d
//   kernel).
//
// Spine topology (parent_indices = [-1, 0, 1, 2, ...]) produces
// outputs byte-exact with linear gated_delta_net_q8 called with
// n_tokens=1 N times in sequence (using the rolling s_q8 as both
// input and output per call).

extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_q8_tree(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const signed char* __restrict__ s_q8_init,
    const float* __restrict__ s_scales_init,
    signed char* __restrict__ s_tape_q8,
    float* __restrict__ s_tape_scales,
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
    // inside s_tape_q8[t * n_heads * HD * HD + ...].
    const int head_tile_offset  = h * HD * HD + row_start * HD;
    const int scale_tile_offset = h * HD + row_start;
    const int token_q8_stride   = n_heads * HD * HD;
    const int token_sc_stride   = n_heads * HD;

    for (int t = 0; t < n_tokens; t++) {
        // Resolve which Q8 tile + scales to dequant into LDS.
        int parent = parent_indices[t];
        const signed char* sq_src;
        const float* sc_src;
        if (parent >= 0) {
            sq_src = s_tape_q8 + parent * token_q8_stride + head_tile_offset;
            sc_src = s_tape_scales + parent * token_sc_stride + scale_tile_offset;
        } else {
            sq_src = s_q8_init + head_tile_offset;
            sc_src = s_scales_init + scale_tile_offset;
        }

        // Dequant Q8 tile → FP32 LDS. Same stride-32 pattern as linear
        // variant for bank-conflict-free LDS writes.
        for (int i = tid; i < TILE_ROWS * HD; i += 32) {
            int r = i / HD;
            float scale = sc_src[r];
            S_tile[i] = scale * (float)sq_src[i];
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

        // Requant FP32 LDS → Q8 tape slot for token t. Same stochastic
        // rounding + per-row absmax as the linear variant; seed differs
        // by token index to avoid correlated noise across the tape.
        signed char* dst_tile  = s_tape_q8 + t * token_q8_stride + head_tile_offset;
        float* dst_scales_row  = s_tape_scales + t * token_sc_stride + scale_tile_offset;

        for (int r = 0; r < TILE_ROWS; r++) {
            float* sr = S_tile + r * HD;
            float s0 = sr[tid], s1 = sr[tid+32], s2 = sr[tid+64], s3 = sr[tid+96];
            float my_max = fmaxf(fmaxf(fabsf(s0), fabsf(s1)), fmaxf(fabsf(s2), fabsf(s3)));
            for (int o = 16; o > 0; o >>= 1)
                my_max = fmaxf(my_max, __shfl_xor(my_max, o));

            float inv_s = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;
            signed char* dst = dst_tile + r * HD;

            // PRNG seed MUST match linear kernel's formula so that calling
            // tree kernel with spine parents produces byte-exact output
            // against N sequential linear n_tokens=1 calls. Per-step
            // variation in my_max already decorrelates noise across the
            // tape, so no explicit t-index term needed.
            unsigned int rng = (unsigned int)(h * 1664525u + (row_start + r) * 1013904223u
                             + tid * 214013u + __float_as_uint(my_max) * 2531011u);
            #define STOCH_ROUND(val) ({ \
                rng = rng * 1664525u + 1013904223u; \
                float noise = (float)(rng >> 8) * (1.0f / 16777216.0f); \
                float scaled = (val) * inv_s; \
                (signed char)fminf(fmaxf(floorf(scaled + noise), -128.0f), 127.0f); \
            })
            dst[tid]      = STOCH_ROUND(s0);
            dst[tid+32]   = STOCH_ROUND(s1);
            dst[tid+64]   = STOCH_ROUND(s2);
            dst[tid+96]   = STOCH_ROUND(s3);
            #undef STOCH_ROUND
            if (tid == 0) dst_scales_row[r] = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
        }
    }
}
