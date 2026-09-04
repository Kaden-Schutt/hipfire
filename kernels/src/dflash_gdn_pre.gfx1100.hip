// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// S5-gdn-pre-tape-fusion: single-launch GDN preambles for gfx1100.
//
// Verify side today issues, per LinearAttention layer,
//   fused_sigmoid_alpha_gate_f32_batched          (1 launch)
//   3x tape memcpy_dtod (raw qkv, cooked alpha/beta) (3 launches)
//   conv1d_silu_split_f32_n                       (1 launch)
//   fused_qk_l2_norm_scale_interleave_f32_batched (1 launch, GQA route)
// `dflash_gdn_pre_capture_gfx1100` folds all five into ONE launch: the three
// tape copies vanish (the kernel stores tape rows directly) and the three
// computes fuse. The replay side folds conv1d + in-place QK norm +
// repeat-interleave into `dflash_gdn_pre_replay_gfx1100` (3 -> 1).
// `gated_delta_net_q8_fast` (the recurrence/state owner) is untouched.
//
// Parallel structure: grid = [n_key_heads + v_blocks (+ 1 prep), 1, 1],
// block = [256, 1, 1]. The N token rows are looped INSIDE each block, so the
// causal conv ring state advances row-sequentially exactly like the old
// batched conv kernel (thread c loops t = 0..N-1); no cross-block ordering
// exists. Grid shape is N-independent (N is a kernarg loop bound), which is
// capture-friendly.
//
// Bit-exactness contract (all verified by
// test_dflash_gdn_pre_gfx1100.rs byte-for-byte):
// - sigmoid(beta): verbatim `1/(1+exp(-b))` from fused_sigmoid_alpha_gate.hip.
// - alpha gate: verbatim softplus guards + `sp * (-exp(a_log))` from the same.
// - conv: verbatim 4-tap causal depthwise + SiLU + ring update from
//   conv1d_silu_split.hip, INCLUDING in-place weight[] indexing (hoisting the
//   loads changes FMA fusion and nudges numerics ~1 ULP).
// - Q/K norm: verbatim per-(head,row) strided accumulation + __shfl_xor
//   wave32 tree + the TWO ordered multiplies (`qv *= inv; qv *= scale` —
//   folding them breaks byte-identity per fused_qk_l2_norm_scale.hip:43).
// - Q/K conv results stage in LDS (as in conv1d_silu_split_qknorm.gfx1201.hip:
//   float->float staging is exact); the norm reads LDS, so no extra global
//   round-trip enters the reduction.
//
// Constraints (enforced host-side; anything else stays on the old path):
// - head_dim == 128 (HD), single-lane sequential batch, GQA ratio >= 1
//   (capture requires ratio > 1, i.e. the interleave branch the fixture
//   takes; replay also covers ratio == 1, matching the old memcpy path).
// - conv_state is the single-lane [n_channels x 3] ring; input/output rows
//   are dense [N x stride] row-major with lane 0, exactly like the old
//   `*_f32_n` batched kernels.

#define GDN_PRE_HD 128
#define GDN_PRE_BLOCK 256

// Verbatim causal depthwise 4-tap conv + SiLU + ring update for one
// (row, channel). `W` is conv_w, `S` is conv_state, `C` the channel.
//
// The accumulation uses EXPLICIT fmaf (not the source-ordered `+` chain):
// the old kernel's SASS contracts the sum as
//   acc = w3*x; acc += w2*s0; acc += w1*s1; acc += w0*s2
// with one rounding per step (v_mul + 3x v_fmac), and the backend does NOT
// reliably rediscover that tree from `+` source in every block shape (the V
// region compiled strict and drifted 1 ULP on the parity gate). fmaf pins
// the exact old tree in all six instantiations (capture Q/K/V, replay
// Q/K/V). Weight/state loads stay in-place (see conv1d_silu_split.hip).
#define GDN_PRE_CONV_STEP(X, C, Y)                                            \
    do {                                                                      \
        float x = (X);                                                        \
        float s0 = (S)[(C) * 3];                                              \
        float s1 = (S)[(C) * 3 + 1];                                          \
        float s2 = (S)[(C) * 3 + 2];                                          \
        float y = fmaf((W)[(C) * 4 + 3], x, (W)[(C) * 4 + 2] * s0);            \
        y = fmaf((W)[(C) * 4 + 1], s1, y);                                    \
        y = fmaf((W)[(C) * 4], s2, y);                                        \
        float r = y / (1.0f + expf(-y));                                      \
        (S)[(C) * 3 + 2] = s1;                                                \
        (S)[(C) * 3 + 1] = s0;                                                \
        (S)[(C) * 3] = x;                                                     \
        (Y) = r;                                                              \
    } while (0)

extern "C" __launch_bounds__(GDN_PRE_BLOCK)
__global__ void dflash_gdn_pre_capture_gfx1100(
    float* __restrict__ beta,        // [N x n_v_heads] in/out, sigmoid
    float* __restrict__ alpha,       // [N x n_v_heads] in/out, alpha gate
    const float* __restrict__ dt_bias, // [n_v_heads]
    const float* __restrict__ a_log,   // [n_v_heads]
    const float* __restrict__ qkv_in,  // [N x qkv_dim] raw projection
    const float* __restrict__ conv_w,  // [n_channels x 4]
    float* __restrict__ conv_state,    // [n_channels x 3] single lane
    float* __restrict__ q_raw,   // [N x k_dim] conv Q (as before: conv outputs)
    float* __restrict__ k_raw,   // [N x k_dim] conv K
    float* __restrict__ v_out,   // [N x v_dim]
    float* __restrict__ q_dst,   // [N x n_v_heads*HD] normed+scaled+repeated
    float* __restrict__ k_dst,   // [N x n_v_heads*HD] normed+repeated
    float* __restrict__ tape_qkv,   // [max_n x qkv_dim]
    float* __restrict__ tape_alpha, // [max_n x n_v_heads]
    float* __restrict__ tape_beta,  // [max_n x n_v_heads]
    int n_v_heads,
    int n_key_heads,
    int ratio,
    int k_dim,
    int v_dim,
    int qkv_dim,
    int n_tokens,
    int tape_offset,
    float q_scale,
    float eps) {
    __shared__ float q_s[GDN_PRE_HD];
    __shared__ float k_s[GDN_PRE_HD];

    const float* W = conv_w;
    float* S = conv_state;
    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int v_blocks = (v_dim + GDN_PRE_BLOCK - 1) / GDN_PRE_BLOCK;

    if (bx < n_key_heads) {
        // Q/K head block: owns Q channels [h*HD,(h+1)*HD) and K channels
        // [k_dim+h*HD,k_dim+(h+1)*HD). Threads 0..127 take Q, 128..255 take K.
        const int h = bx;
        for (int t = 0; t < n_tokens; ++t) {
            const long long in_row = (long long)t * qkv_dim;
            const long long tape_row = (long long)(tape_offset + t) * qkv_dim;
            const long long qk_row = (long long)t * k_dim;
            float result = 0.0f;
            if (tid < GDN_PRE_HD) {
                const int c = h * GDN_PRE_HD + tid;
                GDN_PRE_CONV_STEP(qkv_in[in_row + c], c, result);
                q_raw[qk_row + c] = result;
                q_s[tid] = result;
                tape_qkv[tape_row + c] = qkv_in[in_row + c];
            } else {
                const int d = tid - GDN_PRE_HD;
                const int c = k_dim + h * GDN_PRE_HD + d;
                GDN_PRE_CONV_STEP(qkv_in[in_row + c], c, result);
                k_raw[qk_row + h * GDN_PRE_HD + d] = result;
                k_s[d] = result;
                tape_qkv[tape_row + c] = qkv_in[in_row + c];
            }
            __syncthreads();
            // Verbatim interleave-kernel reduction for this single head.
            if (tid < 32) {
                float q_sq = 0.0f;
                for (int d = tid; d < GDN_PRE_HD; d += 32)
                    q_sq += q_s[d] * q_s[d];
                for (int o = 16; o > 0; o >>= 1)
                    q_sq += __shfl_xor(q_sq, o);
                float q_inv_norm = rsqrtf(q_sq + eps);
                float k_sq = 0.0f;
                for (int d = tid; d < GDN_PRE_HD; d += 32)
                    k_sq += k_s[d] * k_s[d];
                for (int o = 16; o > 0; o >>= 1)
                    k_sq += __shfl_xor(k_sq, o);
                float k_inv_norm = rsqrtf(k_sq + eps);
                const long long dst_row = (long long)t * n_v_heads * GDN_PRE_HD;
                for (int d = tid; d < GDN_PRE_HD; d += 32) {
                    float qv = q_s[d] * q_inv_norm;
                    qv *= q_scale;
                    float kv = k_s[d] * k_inv_norm;
                    for (int r = 0; r < ratio; ++r) {
                        const int vh = h * ratio + r;
                        q_dst[dst_row + vh * GDN_PRE_HD + d] = qv;
                        k_dst[dst_row + vh * GDN_PRE_HD + d] = kv;
                    }
                }
            }
            __syncthreads();
        }
        return;
    }

    if (bx < n_key_heads + v_blocks) {
        // V block: owns V channels [vi0, vi0+256).
        const int vi0 = (bx - n_key_heads) * GDN_PRE_BLOCK;
        for (int t = 0; t < n_tokens; ++t) {
            const int vi = vi0 + tid;
            if (vi < v_dim) {
                const int c = 2 * k_dim + vi;
                const long long in_row = (long long)t * qkv_dim;
                float result = 0.0f;
                GDN_PRE_CONV_STEP(qkv_in[in_row + c], c, result);
                v_out[(long long)t * v_dim + vi] = result;
                tape_qkv[(long long)(tape_offset + t) * qkv_dim + c] =
                    qkv_in[in_row + c];
            }
        }
        return;
    }

    if (bx == n_key_heads + v_blocks) {
        // Prep block: sigmoid(beta) + alpha gate, then tape the cooked rows.
        for (int t = 0; t < n_tokens; ++t) {
            if (tid < n_v_heads) {
                const long long ab_row = (long long)t * n_v_heads;
                const long long tape_ab = (long long)(tape_offset + t) * n_v_heads;
                float b = beta[ab_row + tid];
                b = 1.0f / (1.0f + expf(-b));
                beta[ab_row + tid] = b;
                tape_beta[tape_ab + tid] = b;
                float a = alpha[ab_row + tid];
                float biased = a + dt_bias[tid];
                float sp = (biased > 20.0f)
                    ? biased
                    : ((biased < -20.0f) ? expf(biased) : logf(1.0f + expf(biased)));
                a = sp * (-expf(a_log[tid]));
                alpha[ab_row + tid] = a;
                tape_alpha[tape_ab + tid] = a;
            }
        }
    }
    // Any block beyond the prep block idles (launcher sizes the grid exactly).
}

extern "C" __launch_bounds__(GDN_PRE_BLOCK)
__global__ void dflash_gdn_pre_replay_gfx1100(
    const float* __restrict__ qkv_tape, // [max_n x qkv_dim] taped raw qkv
    const float* __restrict__ conv_w,   // [n_channels x 4]
    float* __restrict__ conv_state,     // [n_channels x 3] single lane
    float* __restrict__ q_raw,   // [N x k_dim] NORMED (old in-place parity)
    float* __restrict__ k_raw,   // [N x k_dim] NORMED (old in-place parity)
    float* __restrict__ v_out,   // [N x v_dim]
    float* __restrict__ q_dst,   // [N x n_v_heads*HD] normed+scaled+repeated
    float* __restrict__ k_dst,   // [N x n_v_heads*HD]
    int n_v_heads,
    int n_key_heads,
    int ratio,
    int k_dim,
    int v_dim,
    int qkv_dim,
    int n_steps,
    float q_scale,
    float eps) {
    __shared__ float q_s[GDN_PRE_HD];
    __shared__ float k_s[GDN_PRE_HD];

    const float* W = conv_w;
    float* S = conv_state;
    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int v_blocks = (v_dim + GDN_PRE_BLOCK - 1) / GDN_PRE_BLOCK;

    if (bx < n_key_heads) {
        // Q/K head block. Same conv + reduction as capture, but the raw
        // scratch keeps the OLD in-place-norm postcondition (normed values),
        // matching fused_qk_l2_norm_scale_f32_batched's two ordered
        // multiplies exactly.
        const int h = bx;
        for (int t = 0; t < n_steps; ++t) {
            const long long in_row = (long long)t * qkv_dim;
            const long long qk_row = (long long)t * k_dim;
            float result = 0.0f;
            if (tid < GDN_PRE_HD) {
                const int c = h * GDN_PRE_HD + tid;
                GDN_PRE_CONV_STEP(qkv_tape[in_row + c], c, result);
                q_s[tid] = result;
            } else {
                const int d = tid - GDN_PRE_HD;
                const int c = k_dim + h * GDN_PRE_HD + d;
                GDN_PRE_CONV_STEP(qkv_tape[in_row + c], c, result);
                k_s[d] = result;
            }
            __syncthreads();
            if (tid < 32) {
                float q_sq = 0.0f;
                for (int d = tid; d < GDN_PRE_HD; d += 32)
                    q_sq += q_s[d] * q_s[d];
                for (int o = 16; o > 0; o >>= 1)
                    q_sq += __shfl_xor(q_sq, o);
                float q_inv_norm = rsqrtf(q_sq + eps);
                float k_sq = 0.0f;
                for (int d = tid; d < GDN_PRE_HD; d += 32)
                    k_sq += k_s[d] * k_s[d];
                for (int o = 16; o > 0; o >>= 1)
                    k_sq += __shfl_xor(k_sq, o);
                float k_inv_norm = rsqrtf(k_sq + eps);
                const long long dst_row = (long long)t * n_v_heads * GDN_PRE_HD;
                for (int d = tid; d < GDN_PRE_HD; d += 32) {
                    // Old in-place order: q *= inv, then q *= scale.
                    float qv = q_s[d] * q_inv_norm;
                    qv *= q_scale;
                    float kv = k_s[d] * k_inv_norm;
                    q_raw[qk_row + h * GDN_PRE_HD + d] = qv;
                    k_raw[qk_row + h * GDN_PRE_HD + d] = kv;
                    for (int r = 0; r < ratio; ++r) {
                        const int vh = h * ratio + r;
                        q_dst[dst_row + vh * GDN_PRE_HD + d] = qv;
                        k_dst[dst_row + vh * GDN_PRE_HD + d] = kv;
                    }
                }
            }
            __syncthreads();
        }
        return;
    }

    if (bx < n_key_heads + v_blocks) {
        // V region: every V block runs this striped loop over its own stripe
        // set. V block j (0-based past the Q/K blocks) handles stripes
        // j, j+v_blocks, ...; the vi guard breaks the loop once past v_dim.
        // Union over blocks covers every V channel exactly once.
        for (int t = 0; t < n_steps; ++t) {
            const long long in_row = (long long)t * qkv_dim;
            for (int stripe = bx - n_key_heads; ; stripe += v_blocks) {
                const int vi = stripe * GDN_PRE_BLOCK + tid;
                if (vi >= v_dim)
                    break;
                const int c = 2 * k_dim + vi;
                float result = 0.0f;
                GDN_PRE_CONV_STEP(qkv_tape[in_row + c], c, result);
                v_out[(long long)t * v_dim + vi] = result;
            }
        }
    }
    // Any block beyond the V region idles (launcher sizes the grid exactly).
}
