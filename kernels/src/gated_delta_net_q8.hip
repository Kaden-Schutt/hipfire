// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HD 128
#define TILE_ROWS 4

extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_q8(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    signed char* __restrict__ s_q8,
    float* __restrict__ s_scales,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim,
    unsigned int frame,  // per-launch monotonic counter (host-supplied) — fresh,
                         // data-INDEPENDENT entropy for the stochastic-rounding
                         // dither. Without it the dither was seeded from my_max
                         // (state-derived) → correlated, biased over the recurrence.
    __half* __restrict__ s_ef_residual   // OPTIONAL (nullable) per-element f16
                         // error-feedback residual, [n_heads*HD*HD], same layout
                         // as s_q8. Non-null ⇒ DETERMINISTIC round + sigma-delta
                         // noise-shaping (carry the per-element quant residual,
                         // fold it back before the next requant) instead of
                         // stochastic rounding. DeltaNet's contractive decay damps
                         // the shaped error → ~FP32 state at Q8's byte container
                         // (CPU sweep: 5.7× lower L2 than stochastic at α=0.99).
    ,
    // Requant cadence gate (host-supplied). 0 = single-end: requant S→Q8 ONCE at
    // the last token (MQ4-fast path; intermediate tokens keep FP32 state in LDS).
    // 1 = per-token: Q8 roundtrip after every token (PARO drift-echo correctness,
    // ~1.8× slower in batched). MQ4/HFQ4 weights pass 0; PARO passes 1. For
    // n_tokens==1 (AR decode / DFlash draft) both modes are identical.
    int requant_per_token
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    __shared__ float S_tile[TILE_ROWS * HD];  // 4 KiB

    // Dequant Q8 tile → FP32 in LDS. Stride-32 traversal is coalesced for
    // the global char read AND conflict-free for the LDS float write
    // (consecutive tids hit consecutive banks modulo 32).
    signed char* sq_base = s_q8 + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        int r = i / HD;
        float scale = s_scales[h * HD + row_start + r];
        S_tile[i] = scale * (float)sq_base[i];
    }

    // KEY CORRECTNESS NOTE (phase2-kernel-audit top-10 #6, §MEM-5):
    // The original kernel had each thread own 4 CONSECUTIVE floats per row
    // (col = tid*4). Within a row that gives bank = (tid*4) mod 32, so tid
    // {0, 8, 16, 24} all land on bank 0, tid {1, 9, 17, 25} all on bank 4,
    // etc — a 4-way bank conflict costing ~75 % of LDS bandwidth.
    //
    // Fix: each thread owns 4 STRIDED floats per row (tid, tid+32, tid+64,
    // tid+96). Bank = (tid + 32k) mod 32 = tid mod 32, so all 32 threads
    // land on distinct banks per LDS instruction. k/q/v are re-indexed the
    // same way so the dot-product sums sum the same set of 128 (s_i * k_i)
    // and (s_i * q_i) products — addition is commutative, so the final
    // reduced value is identical modulo fp32 rounding order.

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

        // Q8 requant. Cadence gated by `requant_per_token`:
        //   per_token=1 (PARO): requant S→Q8 after EVERY token and dequant back into
        //     LDS so the next token reads the Q8-roundtripped state (matches decode
        //     cadence; prevents recurrent-state drift → PARO prompt echo).
        //   per_token=0 (MQ4-fast, DEFAULT): requant ONCE at the last token;
        //     intermediate tokens keep FP32 state in LDS. ~1.8× faster in batched
        //     (no n_tokens× requant, VGPR pressure lower). MQ4 stays coherent with
        //     this (the drift echo is PARO-specific; prototype cceb07c0 = 254 tok/s
        //     coherent). For n_tokens==1 both modes are identical.
        // Stochastic rounding: floor(x + uniform) has E[result] = x (unbiased).
        const bool per_token = (requant_per_token != 0);
        const bool last_tok = (t == n_tokens - 1);
        if (!per_token && !last_tok) continue;
        const bool use_ef = (s_ef_residual != nullptr);
        for (int r = 0; r < TILE_ROWS; r++) {
            int rr = row_start + r;
            float* sr = S_tile + r * HD;
            float s0 = sr[tid], s1 = sr[tid+32], s2 = sr[tid+64], s3 = sr[tid+96];

            // EF (sigma-delta): fold the carried per-element quant residual into
            // the state BEFORE requant; efr layout matches s_q8 (h*HD*HD+row*HD+col).
            __half* efr = use_ef ? (s_ef_residual + h * HD * HD + rr * HD) : nullptr;
            if (use_ef) {
                s0 += __half2float(efr[tid]);
                s1 += __half2float(efr[tid+32]);
                s2 += __half2float(efr[tid+64]);
                s3 += __half2float(efr[tid+96]);
            }

            float my_max = fmaxf(fmaxf(fabsf(s0), fabsf(s1)), fmaxf(fabsf(s2), fabsf(s3)));
            for (int o = 16; o > 0; o >>= 1)
                my_max = fmaxf(my_max, __shfl_xor(my_max, o));

            float inv_s = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;
            float scale = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;

            int q0, q1, q2, q3;
            if (use_ef) {
                // Deterministic round-to-nearest; re-store the fresh residual
                // (val − dequant), capturing round AND clamp-saturation error so
                // the feedback corrects both. Deterministic ⇒ `frame` unused here.
                #define EF_ROUND(val, slot) ({ \
                    float qf = fminf(fmaxf(rintf((val) * inv_s), -128.0f), 127.0f); \
                    efr[slot] = __float2half((val) - qf * scale); \
                    (int)qf; \
                })
                q0 = EF_ROUND(s0, tid);    q1 = EF_ROUND(s1, tid+32);
                q2 = EF_ROUND(s2, tid+64); q3 = EF_ROUND(s3, tid+96);
                #undef EF_ROUND
            } else {
                // Stochastic rounding: seed = spatial (h, row, tid) + (frame + t)
                // for fresh per-token entropy (data-independent; the old
                // my_max-derived seed was correlated/biased over the recurrence).
                unsigned int rng = (unsigned int)(h * 1664525u + rr * 1013904223u
                                 + tid * 214013u + (frame + (unsigned int)t) * 2531011u);
                #define STOCH_ROUND(val) ({ \
                    rng = rng * 1664525u + 1013904223u; \
                    float noise = (float)(rng >> 8) * (1.0f / 16777216.0f); \
                    (int)fminf(fmaxf(floorf((val) * inv_s + noise), -128.0f), 127.0f); \
                })
                q0 = STOCH_ROUND(s0); q1 = STOCH_ROUND(s1);
                q2 = STOCH_ROUND(s2); q3 = STOCH_ROUND(s3);
                #undef STOCH_ROUND
            }

            // per_token mode: dequant back into LDS so the next token reads the Q8
            // state. fast mode reaches here only on last_tok, so skip (we're done).
            if (per_token && !last_tok) {
                sr[tid] = q0 * scale; sr[tid+32] = q1 * scale;
                sr[tid+64] = q2 * scale; sr[tid+96] = q3 * scale;
            }

            if (last_tok) {
                signed char* dst = sq_base + r * HD;
                dst[tid]    = (signed char)q0; dst[tid+32] = (signed char)q1;
                dst[tid+64] = (signed char)q2; dst[tid+96] = (signed char)q3;
                if (tid == 0) s_scales[h * HD + rr] = scale;
            }
        }
    }
}
