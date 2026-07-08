// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// On-GPU chain rejection-sampling accept kernel for spec-decode temp>0 path.
//
// Implements Algorithm 1 (Chen & Leviathan 2023) for an ordered chain of
// `b` speculated positions, fully on-device.  Replaces the host accept loop
// in speculative.rs that required two ~9 MB D2H transfers of the full prob
// vectors.  After this kernel the host only downloads 16 bytes.
//
// ── Inputs ────────────────────────────────────────────────────────────────
//   tgt_probs[b * vocab]     — target model probabilities (post-softmax,
//                              NOT truncated; this kernel applies topp in-kernel
//                              using tau_t / z_t).
//   dft_probs[b * vocab]     — draft model probabilities (post-softmax, same
//                              convention).  b here is the DRAFT batch (number
//                              of drafted positions, NOT including the target-
//                              only bonus position at the end).
//   draft_tokens[b]          — i32 drafted token ids.
//   draft_p_at_token[b]      — f32 effective prob of the drafted token under
//                              the draft distribution (AFTER topp truncation),
//                              i.e. p_d used in the acceptance ratio.
//   tau_t[b], z_t[b]         — target topp parameters per position.
//                              tau_t[i] == 0 means no truncation for position i.
//   tau_d[b], z_d[b]         — draft topp parameters per position.
//   b                        — number of speculated positions (0-indexed; the
//                              bonus draw is from tgt_probs[b * vocab] ONLY when
//                              all b positions are accepted; requires the target
//                              forward to produce b+1 rows, last row = bonus row).
//   vocab                    — vocabulary size.
//   rng_seed                 — u32 base seed (LCG convention, same as SWOR walk).
//   cactus_delta             — CACTUS boost parameter δ ≥ 0 (0.0 = plain RS).
//
// ── Outputs ───────────────────────────────────────────────────────────────
//   out[0]: accept_len (i32) — number of tokens accepted from the draft.
//   out[1]: bonus_token (i32) — bonus token drawn from the appropriate
//           distribution (residual on rejection, target[b] on all-accepted).
//   out[2]: rejected_at (i32) — index of first rejected position, or -1 if
//           all b positions accepted.
//   out[3]: new_rng (u32)    — updated RNG state (for host bookkeeping).
//
// ── Algorithm ─────────────────────────────────────────────────────────────
//   One block (blockDim.x = 256).  Thread 0 drives the sequential accept
//   chain; block-parallel sweeps are used for:
//     a) The residual sum Σmax(p_t[j]-p_d[j], 0) (block reduce)
//     b) The residual CDF walk (sequential thread 0 only, same as BCAT)
//     c) The all-accepted bonus draw from tgt_probs[b] (same pattern as BCAT)
//
// ── RNG ───────────────────────────────────────────────────────────────────
//   LCG: s = s * 1664525 + 1013904223; u = (s >> 8) / 2^24.
//   Seeded from rng_seed | 1.  Byte-identical RNG to host xorshift64* is
//   infeasible (different algorithm families); distribution-parity is the bar.
//
// ── CACTUS ────────────────────────────────────────────────────────────────
//   When cactus_delta > 0, the acceptance probability is boosted to:
//     accept_prob = min(p_t + sqrt(2 * delta * p_t * (1 - p_t)), 1)
//   On rejection the target row is replaced with the CACTUS h-distribution
//   BEFORE residual sampling, matching the host fallback exactly
//   (speculative.rs:3660-3677):
//     gamma* = accept_prob          (boosted, clamped value at drafted token t)
//     if p_t >= 1 - 1e-6:  h = one-hot at t
//     else:                h[t] = gamma*,  h[j!=t] = (1-gamma*)/(1-p_t) * p_t[j]
//   The bonus is then drawn from residual(h, p_draft), so the committed
//   rejection token is distribution-identical to the host path. At delta == 0
//   the rewrite is skipped and this reduces to plain residual(p_t, p_draft)
//   (Chen & Leviathan 2023). See casp_h_val() below.

#include <hip/hip_runtime.h>

#define CASP_NTH 256

// Inline effective-prob helper: applies topp truncation.
__device__ __forceinline__ float eff_prob(float p, float tau, float inv_z) {
    if (tau <= 0.0f) return p;  // no truncation
    return (p >= tau) ? (p * inv_z) : 0.0f;
}

// CACTUS h-distribution value at column j on rejection. `pt_j` is the effective
// (topp-truncated) target prob at column j; `t` is the drafted token. When
// use_cx is false this is a no-op (h == raw effective target). Mirrors the host
// h-distribution rewrite in speculative.rs:3660-3677 term-for-term.
__device__ __forceinline__ float casp_h_val(
    int j, int t, float pt_j, bool use_cx, bool onehot, float gamma_star, float scale) {
    if (!use_cx) return pt_j;
    if (onehot)  return (j == t) ? 1.0f : 0.0f;
    return (j == t) ? gamma_star : scale * pt_j;
}

extern "C" __global__ void chain_accept_spec_f32(
    const float* __restrict__ tgt_probs,         // [b * vocab]
    const float* __restrict__ dft_probs,         // [b * vocab]
    const int*   __restrict__ draft_tokens,      // [b] i32
    const float* __restrict__ draft_p_at_token,  // [b] f32
    const float* __restrict__ tau_t,             // [b] target topp tau
    const float* __restrict__ z_t,               // [b] target topp Z
    const float* __restrict__ tau_d,             // [b] draft topp tau
    const float* __restrict__ z_d,               // [b] draft topp Z
    int   b,         // number of speculated positions
    int   vocab,
    unsigned int rng_seed,
    float cactus_delta,
    int*  __restrict__ out   // [4]: accept_len, bonus_token, rejected_at, new_rng
) {
    if (blockIdx.x != 0) return;

    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    __shared__ unsigned int s_rng;
    __shared__ float s_red[CASP_NTH];  // block reduction scratch
    __shared__ int   s_accept_len;
    __shared__ int   s_bonus_token;
    __shared__ int   s_rejected_at;
    __shared__ int   s_done;           // 1 once rejection or all-accepted

    if (tid == 0) {
        s_rng        = rng_seed | 1u;
        s_accept_len = 0;
        s_bonus_token = -1;
        s_rejected_at = -1;
        s_done = 0;
    }
    __syncthreads();

    // ── Sequential accept chain over b positions ───────────────────────────
    for (int i = 0; i < b && !s_done; ++i) {
        const float* trow = tgt_probs + (size_t)i * vocab;
        const float  tau_ti = tau_t[i];
        const float  inv_zt = 1.0f / fmaxf(z_t[i], 1e-30f);

        // Read scalars used in the accept decision (thread 0).
        int   draft_tok;
        float p_d;
        if (tid == 0) {
            draft_tok = draft_tokens[i];
            p_d       = draft_p_at_token[i];
        }
        // Broadcast via shared (reuse s_red[0] slot; sync guarantees visibility).
        __shared__ int   s_dtok;
        __shared__ float s_pd;
        if (tid == 0) { s_dtok = draft_tok; s_pd = p_d; }
        __syncthreads();
        draft_tok = s_dtok;
        p_d       = s_pd;

        // p_t: effective target prob at the drafted token.
        float p_t_raw = trow[draft_tok];
        float p_t = eff_prob(p_t_raw, tau_ti, inv_zt);

        // CACTUS boost: accept_prob = p_t + sqrt(2*delta*p_t*(1-p_t))
        float accept_prob = p_t;
        if (cactus_delta > 0.0f && p_t > 0.0f && p_t < 1.0f) {
            accept_prob = p_t + sqrtf(2.0f * cactus_delta * p_t * (1.0f - p_t));
            if (accept_prob > p_d) accept_prob = p_d;  // clamp: ratio <= 1
        }

        // Thread 0 draws acceptance uniform.
        __shared__ int s_accepted;
        if (tid == 0) {
            s_rng = s_rng * 1664525u + 1013904223u;
            float u = (float)(s_rng >> 8) * (1.0f / 16777216.0f);
            // Accept iff u * p_d <= accept_prob
            // (equivalent to u <= accept_prob / p_d when p_d > 0).
            // Guard: if p_d == 0 we should always reject (draft put zero mass here).
            int accepted;
            if (p_d <= 0.0f) {
                accepted = 0;
            } else {
                accepted = (u * p_d <= accept_prob) ? 1 : 0;
            }
            s_accepted = accepted;
        }
        __syncthreads();
        int accepted = s_accepted;

        if (accepted) {
            if (tid == 0) s_accept_len = i + 1;
            __syncthreads();
            // Continue to next position.
        } else {
            // ── Rejection: sample bonus from residual(h, p_d) ───────────────
            // residual[j] = max(h[j] - p_deff[j], 0), where h is the CACTUS
            // h-distribution (== raw effective target when cactus_delta == 0).
            // p_deff uses tau_d[i] / z_d[i].
            const float* drow = dft_probs + (size_t)i * vocab;
            const float  tau_di = tau_d[i];
            const float  inv_zd = 1.0f / fmaxf(z_d[i], 1e-30f);

            // CACTUS h-distribution scalars (all threads; p_t and draft_tok are
            // in scope). Mirrors host speculative.rs:3660-3677: gamma* is the
            // boosted accept prob at the drafted token, and the rest of the row
            // is rescaled by (1-gamma*)/(1-p_t). No-op when cactus_delta == 0.
            const bool  use_cx = (cactus_delta > 0.0f);
            const float qn = fminf(fmaxf(p_t, 0.0f), 1.0f);
            const float cx_bump = (p_t > 0.0f && p_t < 1.0f)
                ? sqrtf(2.0f * cactus_delta * p_t * (1.0f - p_t)) : 0.0f;
            const float gamma_star = fminf(p_t + cx_bump, 1.0f);
            const bool  cx_onehot = (qn >= 1.0f - 1e-6f);
            const float cx_scale =
                cx_onehot ? 0.0f : (1.0f - gamma_star) / fmaxf(1.0f - qn, 1e-30f);

            // Block-parallel residual sum.
            float local_sum = 0.0f;
            for (int j = tid; j < vocab; j += nth) {
                float pt = casp_h_val(j, draft_tok, eff_prob(trow[j], tau_ti, inv_zt),
                                      use_cx, cx_onehot, gamma_star, cx_scale);
                float pd = eff_prob(drow[j], tau_di, inv_zd);
                float diff = pt - pd;
                if (diff > 0.0f) local_sum += diff;
            }
            s_red[tid] = local_sum;
            __syncthreads();
            for (int s = nth / 2; s > 0; s >>= 1) {
                if (tid < s) s_red[tid] += s_red[tid + s];
                __syncthreads();
            }
            float resid_sum = s_red[0];
            __syncthreads();

            // Thread 0: sequential CDF walk over the residual.
            if (tid == 0) {
                s_rng = s_rng * 1664525u + 1013904223u;
                float u2;
                if (resid_sum > 0.0f) {
                    u2 = (float)(s_rng >> 8) * (1.0f / 16777216.0f) * resid_sum;
                } else {
                    u2 = 0.0f;
                }
                int bonus = vocab - 1;
                if (resid_sum <= 0.0f) {
                    // Degenerate: p_draft >= h everywhere. Fall back to argmax of
                    // the h-distribution (match host sample_residual fallback,
                    // which argmaxes the same rewritten target row).
                    float best = -1.0f;
                    for (int j = 0; j < vocab; ++j) {
                        float pt = casp_h_val(j, draft_tok, eff_prob(trow[j], tau_ti, inv_zt),
                                              use_cx, cx_onehot, gamma_star, cx_scale);
                        if (pt > best) { best = pt; bonus = j; }
                    }
                } else {
                    float acc = 0.0f;
                    for (int j = 0; j < vocab; ++j) {
                        float pt = casp_h_val(j, draft_tok, eff_prob(trow[j], tau_ti, inv_zt),
                                              use_cx, cx_onehot, gamma_star, cx_scale);
                        float pd = eff_prob(drow[j], tau_di, inv_zd);
                        float diff = pt - pd;
                        if (diff > 0.0f) {
                            acc += diff;
                            if (u2 < acc) { bonus = j; break; }
                        }
                    }
                }
                s_bonus_token = bonus;
                s_rejected_at = i;
                s_done = 1;
            }
            __syncthreads();
            // break is handled by the loop guard (!s_done).
        }
    }

    // ── All-accepted path: bonus from tgt_probs[b] ────────────────────────
    // When s_done == 0 after the loop, all b positions accepted.
    // The caller must supply tgt_probs[(b+1)*vocab] where row b is the
    // target output for the last position (the "bonus position").
    // Here b_row points to that last row.
    if (!s_done) {
        const float* trow_bonus = tgt_probs + (size_t)b * vocab;
        const float  tau_tb     = tau_t[b];          // tau for the bonus row
        const float  inv_ztb    = 1.0f / fmaxf(z_t[b], 1e-30f);

        // Block-parallel total mass (should be 1.0; compute for robustness).
        float local_mass = 0.0f;
        for (int j = tid; j < vocab; j += nth) {
            local_mass += eff_prob(trow_bonus[j], tau_tb, inv_ztb);
        }
        s_red[tid] = local_mass;
        __syncthreads();
        for (int s = nth / 2; s > 0; s >>= 1) {
            if (tid < s) s_red[tid] += s_red[tid + s];
            __syncthreads();
        }
        float total_mass = s_red[0];
        __syncthreads();

        // Thread 0: draw and CDF-walk the bonus row.
        if (tid == 0) {
            s_rng = s_rng * 1664525u + 1013904223u;
            float u = (float)(s_rng >> 8) * (1.0f / 16777216.0f) * total_mass;
            float acc = 0.0f;
            int bonus = vocab - 1;
            for (int j = 0; j < vocab; ++j) {
                float pe = eff_prob(trow_bonus[j], tau_tb, inv_ztb);
                if (pe <= 0.0f) continue;
                acc += pe;
                if (u < acc) { bonus = j; break; }
            }
            s_bonus_token = bonus;
            // s_rejected_at stays -1 (all accepted).
        }
        __syncthreads();
    }

    // ── Write outputs ──────────────────────────────────────────────────────
    if (tid == 0) {
        out[0] = s_accept_len;
        out[1] = s_bonus_token;
        out[2] = s_rejected_at;
        out[3] = (int)s_rng;
    }
}
