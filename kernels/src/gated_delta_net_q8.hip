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
    __half* __restrict__ s_ef_residual  // OPTIONAL (nullable) per-element f16
                         // error-feedback residual, same [n_heads*HD*HD] layout
                         // as s_q8. When non-null, the requant uses deterministic
                         // round + sigma-delta noise-shaping instead of stochastic
                         // rounding: the carried quantization error is added back
                         // before requant and the fresh error re-stored. DeltaNet's
                         // contractive decay damps the shaped error, giving
                         // ~FP32-grade state correctness at the Q8 byte container
                         // (see CPU sweep: 5.7× lower L2 than stochastic at α=0.99).
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
    }

    // Requant FP32 → Q8.
    //
    // Two schemes, selected per-launch by whether an error-feedback residual
    // buffer was supplied:
    //
    //  • s_ef_residual != null  → DETERMINISTIC round + sigma-delta noise-shaping
    //    (error feedback). The carried per-element quantization residual is added
    //    back BEFORE requant, then the fresh residual (t − dequant(q)) is re-stored.
    //    Over the recurrence DeltaNet's contractive decay damps the shaped error,
    //    so the long-run state stays ~FP32-accurate at Q8's byte container with no
    //    systematic bias. Beats stochastic rounding on BOTH bias and variance —
    //    CPU sweep: 5.7× lower output-L2 and flat (length-independent) state drift
    //    at α=0.99. Deterministic ⇒ no per-step entropy needed (`frame` unused).
    //
    //  • s_ef_residual == null  → STOCHASTIC rounding (legacy / examples / tests).
    //    Deterministic roundf alone systematically crushes small values to 0,
    //    drifting the state off-distribution; floor(x + uniform) is unbiased in
    //    expectation. Seed = spatial location (h, row, tid) + per-launch `frame`.
    const bool use_ef = (s_ef_residual != nullptr);
    for (int r = 0; r < TILE_ROWS; r++) {
        const int row = row_start + r;
        float* sr = S_tile + r * HD;
        float s0 = sr[tid], s1 = sr[tid+32], s2 = sr[tid+64], s3 = sr[tid+96];

        // EF: fold the carried quantization residual into the state pre-requant.
        __half* efr = use_ef ? (s_ef_residual + h * HD * HD + row * HD) : nullptr;
        if (use_ef) {
            s0 += __half2float(efr[tid]);
            s1 += __half2float(efr[tid+32]);
            s2 += __half2float(efr[tid+64]);
            s3 += __half2float(efr[tid+96]);
        }

        float my_max = fmaxf(fmaxf(fabsf(s0), fabsf(s1)), fmaxf(fabsf(s2), fabsf(s3)));
        for (int o = 16; o > 0; o >>= 1)
            my_max = fmaxf(my_max, __shfl_xor(my_max, o));

        const float scale = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
        const float inv_s = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;
        signed char* dst = sq_base + r * HD;

        if (use_ef) {
            // Round-to-nearest-even (rintf), then re-store the new residual.
            // The residual captures rounding AND clamp-saturation error, so the
            // feedback corrects both.
            #define EF_ROUND(val, slot) do { \
                float qf = fminf(fmaxf(rintf((val) * inv_s), -128.0f), 127.0f); \
                dst[slot] = (signed char)qf; \
                efr[slot] = __float2half((val) - qf * scale); \
            } while (0)
            EF_ROUND(s0, tid);
            EF_ROUND(s1, tid+32);
            EF_ROUND(s2, tid+64);
            EF_ROUND(s3, tid+96);
            #undef EF_ROUND
        } else {
            // PRNG: position-derived LCG — deterministic, no shared state.
            unsigned int rng = (unsigned int)(h * 1664525u + row * 1013904223u
                             + tid * 214013u + frame * 2531011u);
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
        }
        if (tid == 0) s_scales[h * HD + row] = scale;
    }
}
