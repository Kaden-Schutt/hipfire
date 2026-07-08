// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused on-GPU sample+accept for DSpark (deepseek4) temp>0 spec-decode verify.
//
// Replaces the per-position host loop in
// `final_norm_and_sample_all_batched_lazy` (one `sample_top_p_pf` call + one
// 8-byte D2H per position → a host round-trip that serialises ~tau positions
// per window). Here the batched lm-head already left the target logits for all
// `n` verify positions resident in `logits[n * vocab]` (one 565 MB weight read
// via `compute_batched_head_logits`), so this kernel samples them entirely on
// the device and returns only the accepted prefix + the advanced RNG.
//
// Per position `pos` this replays the SINGLE-BLOCK `sample_top_p` algorithm
// (top-K gather → softmax(temperature) → top-p truncation → xorshift32
// categorical draw) over `logits + pos*vocab`. DSpark never uses the
// repeat/presence/frequency/min_p penalties, so the penalty phase is dropped
// and `logits` is read-only. The RNG state is threaded across positions exactly
// as the host loop threads `*rng = new_rng`, so the sampled token sequence is
// byte-identical to calling the single-block `sample_top_p` per position.
//
// LAZY acceptance (a prefix): after sampling position `pos`, if its token
// differs from the drafted token for the next position (`draft[pos+1]`), every
// later position is rejected — the loop stops and the tail is padded with
// u32::MAX (mirrors the host's `while ids.len() < n { push(u32::MAX) }`). The
// last position (pos == n-1) has no `draft[pos+1]` and is always kept (the
// bonus token), matching the host's `if i+1 < n` guard.
//
// Output `out[n + 1]`:
//   out[0..n] — sampled token id per position (u32::MAX after the first
//               mismatch); out[pos] at the mismatch is the corrected token.
//   out[n]    — advanced RNG state (u32), to thread into the next window.
//
// Launch: 1 block × 64 threads (power of two for the tree reduction). Shared
// mem = 64 * 64 * 4 * 2 = 32 KiB. (The single-block `sample_top_p` uses 128
// threads / 64 KiB, but gfx1151's usable dynamic LDS is < 64 KiB, so this kernel
// stays at 64 threads / 32 KiB — byte-identical, since the top-K gather + tree
// reduction select the same global top-64 for any thread count.) The host must
// launch with a power-of-two blockDim.x whose LDS (blockDim.x * 64 * 8) fits.

extern "C" __global__ void dspark_sample_accept_lazy_f32(
    const float* __restrict__ logits, // [n * vocab] resident target logits (read-only)
    const unsigned int* __restrict__ draft, // [n] drafted block tokens (compare against draft[pos+1])
    unsigned int* __restrict__ out,   // [n + 1]: sampled ids (u32::MAX after stop) + out[n]=new rng
    int n,
    int vocab_size,
    float temperature,
    float top_p,
    unsigned int rand_seed,
    int top_k_req,                    // <=0 or >=TOP_K means "no extra cut" (20 → legacy nucleus)
    float cactus_delta                // >0 → CACTUS acceptance boost (bench-only, deliberately lossy)
) {
    const int TOP_K = 64;
    extern __shared__ char smem[];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Per-thread local top-K scratch (same layout as sample_top_p).
    float* topk_val = (float*)smem;
    int*   topk_idx = (int*)(topk_val + nthreads * TOP_K);

    __shared__ unsigned int s_rng; // RNG state threaded across positions
    __shared__ unsigned int s_tok; // this position's sampled token
    __shared__ int          s_stop; // index of first mismatch, or n if none yet

    if (tid == 0) {
        s_rng  = rand_seed;
        s_stop = n;
    }
    __syncthreads();

    for (int pos = 0; pos < n; pos++) {
        const float* row = logits + (size_t)pos * vocab_size;

        // Phase 1: each thread builds its local top-K over its vocab strip.
        float my_vals[64]; int my_idx[64];
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) { my_vals[k] = -1e30f; my_idx[k] = 0; }
        int   min_pos = 0;
        float min_val = -1e30f;
        for (int i = tid; i < vocab_size; i += nthreads) {
            float v = row[i];
            if (v > min_val) {
                my_vals[min_pos] = v;
                my_idx[min_pos] = i;
                min_val = my_vals[0]; min_pos = 0;
                #pragma unroll
                for (int k = 1; k < TOP_K; k++) {
                    if (my_vals[k] < min_val) { min_val = my_vals[k]; min_pos = k; }
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) {
            topk_val[tid * TOP_K + k] = my_vals[k];
            topk_idx[tid * TOP_K + k] = my_idx[k];
        }
        __syncthreads();

        // Phase 2: tree reduction — merge per-thread top-K into a global top-K.
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float* a_val = topk_val + tid * TOP_K;
                int*   a_idx = topk_idx + tid * TOP_K;
                float* b_val = topk_val + (tid + stride) * TOP_K;
                int*   b_idx = topk_idx + (tid + stride) * TOP_K;
                #pragma unroll
                for (int j = 0; j < TOP_K; j++) {
                    float bv = b_val[j]; int bi = b_idx[j];
                    float amin = a_val[0]; int amin_pos = 0;
                    #pragma unroll
                    for (int k = 1; k < TOP_K; k++) {
                        if (a_val[k] < amin) { amin = a_val[k]; amin_pos = k; }
                    }
                    if (bv > amin) { a_val[amin_pos] = bv; a_idx[amin_pos] = bi; }
                }
            }
            __syncthreads();
        }

        // Phase 3: thread 0 owns the final top-K — softmax, sort desc, top-p draw.
        if (tid == 0) {
            float vals[64]; int idx[64];
            #pragma unroll
            for (int k = 0; k < TOP_K; k++) { vals[k] = topk_val[k]; idx[k] = topk_idx[k]; }

            float max_val = vals[0]; int max_pos = 0;
            for (int k = 1; k < TOP_K; k++) { if (vals[k] > max_val) { max_val = vals[k]; max_pos = k; } }

            unsigned int rng_in  = s_rng;
            unsigned int rng_out = rng_in; // default: unchanged (degenerate / greedy)
            unsigned int tok_out = 0;

            if (max_val <= -1e29f) {
                // All candidates are -inf: emit token 0, rng unchanged.
                tok_out = 0u;
            } else if (temperature <= 1e-6f) {
                // Greedy fast path: argmax, no RNG advance (matches sample_top_p).
                tok_out = (unsigned int)idx[max_pos];
            } else {
                float inv_temp = 1.0f / temperature;
                float probs[64];
                #pragma unroll
                for (int k = 0; k < TOP_K; k++) {
                    probs[k] = (vals[k] <= -1e29f) ? 0.0f : expf((vals[k] - max_val) * inv_temp);
                }
                int order[64];
                #pragma unroll
                for (int k = 0; k < TOP_K; k++) order[k] = k;
                for (int i = 1; i < TOP_K; i++) {
                    int cur = order[i];
                    float cv = probs[cur];
                    int j = i;
                    while (j > 0 && probs[order[j - 1]] < cv) { order[j] = order[j - 1]; j--; }
                    order[j] = cur;
                }

                unsigned int rng = rng_in;
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;

                int cap = TOP_K;
                if (top_k_req > 0 && top_k_req < TOP_K) cap = top_k_req;
                if (cap < 1) cap = 1;

                float sum = 0.0f;
                for (int i = 0; i < cap; i++) sum += probs[order[i]];
                if (sum <= 0.0f) {
                    // Degenerate: return top candidate, rng unchanged (== rand_seed).
                    tok_out = (unsigned int)idx[order[0]];
                    rng_out = rng_in;
                } else {
                    float p_thresh = top_p * sum;
                    float trunc_sum = 0.0f;
                    int trunc_k = cap;
                    for (int i = 0; i < cap; i++) {
                        trunc_sum += probs[order[i]];
                        if (trunc_sum >= p_thresh) { trunc_k = i + 1; break; }
                    }
                    if (cactus_delta > 0.0f && pos + 1 < n) {
                        // ── CACTUS boosted accept vs the point-mass draft d ──────
                        // The draft is deterministic (a point mass), so the general
                        // rejection-sampling CACTUS (chain_accept_spec.hip) reduces to:
                        //   accept d with prob min(p_t + sqrt(2·δ·p_t·(1−p_t)), 1);
                        //   on reject, draw the bonus from the target support with d
                        //   removed and renormalized (== the h-distribution residual
                        //   for a point-mass draft — the residual algebra collapses to
                        //   "p_target minus the drafted token"). Deliberately lossy: it
                        //   commits d more often than the true target warrants, trading
                        //   distribution fidelity for higher acceptance τ.
                        unsigned int d = draft[pos + 1];
                        float p_t = 0.0f;
                        for (int i = 0; i < trunc_k; i++) {
                            if ((unsigned int)idx[order[i]] == d) {
                                p_t = probs[order[i]] / trunc_sum;
                                break;
                            }
                        }
                        float bump = (p_t > 0.0f && p_t < 1.0f)
                            ? sqrtf(2.0f * cactus_delta * p_t * (1.0f - p_t)) : 0.0f;
                        float accept_prob = fminf(p_t + bump, 1.0f);
                        // `rng` was advanced once above (line ~161); reuse as u1.
                        float u1 = (float)rng / 4294967295.0f;
                        if (u1 < accept_prob) {
                            // Accept → commit d. tok_out == draft[pos+1] so the stop
                            // check below does NOT fire (the prefix continues).
                            tok_out = d;
                            rng_out = rng;
                        } else {
                            // Reject → residual over the support with d removed,
                            // renormalized. tok_out != d → the stop check fires.
                            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                            float rs = 0.0f;
                            for (int i = 0; i < trunc_k; i++)
                                if ((unsigned int)idx[order[i]] != d) rs += probs[order[i]];
                            float r2 = (float)rng / 4294967295.0f * rs;
                            float cum = 0.0f;
                            int picked = -1;
                            for (int i = 0; i < trunc_k; i++) {
                                if ((unsigned int)idx[order[i]] == d) continue;
                                cum += probs[order[i]];
                                if (cum >= r2) { picked = order[i]; break; }
                            }
                            // picked<0 only if the support is {d} alone (rs≈0); then
                            // there is nothing else to draw — fall back to d.
                            tok_out = (picked >= 0) ? (unsigned int)idx[picked] : d;
                            rng_out = rng;
                        }
                    } else {
                        float r = (float)rng / 4294967295.0f * trunc_sum;
                        float cum = 0.0f;
                        int picked = trunc_k - 1; // FP tail fallback: last top-p entry
                        for (int i = 0; i < trunc_k; i++) {
                            cum += probs[order[i]];
                            if (cum >= r) { picked = i; break; }
                        }
                        tok_out = (unsigned int)idx[order[picked]];
                        rng_out = rng;
                    }
                }
            }

            s_tok = tok_out;
            s_rng = rng_out;
            out[pos] = tok_out;
            // LAZY prefix stop: draft for position pos is draft[pos+1] (pos<n-1).
            if (pos + 1 < n && draft[pos + 1] != tok_out) {
                s_stop = pos;
            }
        }
        __syncthreads();
        if (s_stop == pos) break; // all threads leave together on the first mismatch
    }

    if (tid == 0) {
        for (int p = s_stop + 1; p < n; p++) out[p] = 0xFFFFFFFFu;
        out[n] = s_rng;
    }
}
