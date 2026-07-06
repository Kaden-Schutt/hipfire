// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// GPU-side sampling. Matches CPU llama::sample_top_p + apply_repeat_penalty:
//   - Frequency-weighted + recency-decayed repeat penalty (count + cap at 1.5×)
//   - Top-K (gather budget =64) selection directly from raw logits (no race)
//   - Softmax over the CAPPED candidate set only (matches CPU normalization)
//   - Top-p sampling within the capped distribution
//
// Output: result[0] = token_id, result[1] = new rng state.
//
// W7 P2b: the gather budget TOP_K was widened 20 → 64 so request top_k>20
// (e.g. minimax top_k=40) is honored. BYTE-PRESERVATION: the softmax sum
// feeding p_thresh is computed over the CAPPED set [0,cap) (cap defaults to
// top_k_req=20), so the default path sums the same top-20 probabilities and
// stays byte-identical to the pre-widen kernel.
//
// Launch: 1 block × 128 threads. Shared mem required:
//   topk_val[128 × 64] = 32768 bytes (per-thread top-K buffer)
//   topk_idx[128 × 64] = 32768 bytes
//   = 65536 bytes = 64 KiB, the RDNA wave32 group-segment limit. The host
//   MUST launch with blockDim.x = 128 (256 would request 128 KiB and fail).
//
// Why this replaces the prior "threshold + atomicAdd into cand_buf[512]"
// approach: at temperature >= 1 many vocab entries exceed the threshold, so
// atomicAdd races to fill 512 slots and can DROP the actual top tokens.
// Direct top-K reduction is deterministic and correct at all temperatures.

extern "C" __global__ void sample_top_p(
    float* __restrict__ logits,                     // [vocab_size] (modified in-place for repeat penalty)
    unsigned int* __restrict__ result,              // [2]: token_id, rng_state
    const unsigned int* __restrict__ repeat_tokens, // [repeat_window] recent token IDs (most recent last)
    int vocab_size,
    float temperature,
    float top_p,
    unsigned int rand_seed,
    int repeat_window,
    float repeat_penalty,
    float presence_penalty,                         // OpenAI: flat logit subtraction for any token seen in window
    float frequency_penalty,                        // OpenAI: logit subtraction scaled by occurrence count
    int   top_k_req,                                // request top-K cutoff; <=0 or >=TOP_K means "no extra cut" (default 20 → byte-identical to legacy)
    float min_p                                     // request min-p cutoff; 0.0 disables (byte-identical to legacy)
) {
    const int TOP_K = 64;
    extern __shared__ char smem[];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Shared memory layout (per thread):
    //   topk_val[nthreads][TOP_K]    — each thread's local top-K logit values
    //   topk_idx[nthreads][TOP_K]    — each thread's local top-K indices
    float* topk_val = (float*)smem;
    int*   topk_idx = (int*)(topk_val + nthreads * TOP_K);

    // Phase 0: Apply repetition penalty in parallel.
    // Each thread handles one position in the recent window. The thread
    // determines whether position i holds the LAST occurrence of its token;
    // if so, it counts total occurrences, computes recency = (i+1)/window_len,
    // and applies penalty^(count*recency) (capped at 1.5×) to logits[tok].
    bool any_penalty = (repeat_penalty > 1.0f) || (presence_penalty > 0.0f) || (frequency_penalty > 0.0f);
    if (any_penalty && repeat_window > 0) {
        float window_len = (float)repeat_window;
        for (int i = tid; i < repeat_window; i += nthreads) {
            unsigned int tok = repeat_tokens[i];
            if ((int)tok >= vocab_size) continue;
            bool has_later = false;
            int count = 1;
            for (int j = 0; j < repeat_window; j++) {
                if (j == i) continue;
                if (repeat_tokens[j] == tok) {
                    if (j > i) { has_later = true; break; }
                    count++;
                }
            }
            if (has_later) continue;
            // `count` is now the total occurrences of `tok` in the window (this
            // is the last occurrence). Apply the existing multiplicative,
            // recency-weighted repeat penalty first...
            if (repeat_penalty > 1.0f) {
                float recency = ((float)i + 1.0f) / window_len;
                float effective = powf(repeat_penalty, (float)count * recency);
                if (effective > 1.5f) effective = 1.5f;
                float v = logits[tok];
                logits[tok] = (v > 0.0f) ? (v / effective) : (v * effective);
            }
            // ...then the OpenAI-style subtractive presence/frequency penalties.
            // These are flat in logit space (not recency-weighted) so they
            // suppress a token equally no matter where in the (long) window it
            // recurred — the mechanism that breaks block-level repetition loops
            // a short recency-weighted window cannot see. Matches llama.cpp /
            // Lemonade semantics: logit -= frequency*count + presence.
            if (presence_penalty > 0.0f || frequency_penalty > 0.0f) {
                logits[tok] -= frequency_penalty * (float)count + presence_penalty;
            }
        }
    }
    __syncthreads();

    // Phase 1: Each thread builds its local top-K from its assigned vocab strip.
    //
    // Use a fixed-size local buffer with manual min-tracking (same algorithm
    // as CPU `sample_top_p` but sharded across threads). After this pass,
    // every thread owns TOP_K candidates covering its strip.
    float my_vals[64]; int my_idx[64];
    #pragma unroll
    for (int k = 0; k < TOP_K; k++) { my_vals[k] = -1e30f; my_idx[k] = 0; }
    int min_pos = 0;
    float min_val = -1e30f;
    for (int i = tid; i < vocab_size; i += nthreads) {
        float v = logits[i];
        if (v > min_val) {
            my_vals[min_pos] = v;
            my_idx[min_pos] = i;
            // Recompute min.
            min_val = my_vals[0]; min_pos = 0;
            #pragma unroll
            for (int k = 1; k < TOP_K; k++) {
                if (my_vals[k] < min_val) { min_val = my_vals[k]; min_pos = k; }
            }
        }
    }
    // Stash this thread's local top-K into shared memory.
    #pragma unroll
    for (int k = 0; k < TOP_K; k++) {
        topk_val[tid * TOP_K + k] = my_vals[k];
        topk_idx[tid * TOP_K + k] = my_idx[k];
    }
    __syncthreads();

    // Phase 2: Tree reduction — merge per-thread top-K into a global top-K.
    // Each step: thread `tid` merges its list with thread `tid + stride`.
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float*  a_val = topk_val + tid * TOP_K;
            int*    a_idx = topk_idx + tid * TOP_K;
            float*  b_val = topk_val + (tid + stride) * TOP_K;
            int*    b_idx = topk_idx + (tid + stride) * TOP_K;
            // Merge two sorted-or-unsorted TOP_K lists into top-K of the union.
            // Since neither list is sorted, we do min-replace: for each element
            // in b, try to insert into a.
            #pragma unroll
            for (int j = 0; j < TOP_K; j++) {
                float bv = b_val[j]; int bi = b_idx[j];
                // Find a's min.
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

    // Phase 3: thread 0 owns the final top-K — softmax, sort descending, top-p sample.
    if (tid == 0) {
        float vals[64]; int idx[64];
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) {
            vals[k] = topk_val[k];
            idx[k]  = topk_idx[k];
        }

        // Find max (for numerical stability) and clamp against degenerate cases.
        float max_val = vals[0];
        int   max_pos = 0;
        for (int k = 1; k < TOP_K; k++) { if (vals[k] > max_val) { max_val = vals[k]; max_pos = k; } }
        if (max_val <= -1e29f) { result[0] = 0; result[1] = rand_seed; return; }

        // Greedy fast path: at temperature == 0 (or effectively zero), just
        // return the argmax. Skipping the softmax avoids a 1/temp division
        // that would produce +inf → NaN probabilities and a garbage pick.
        if (temperature <= 1e-6f) {
            result[0] = (unsigned int)idx[max_pos];
            result[1] = rand_seed;
            return;
        }

        // Softmax over the candidate set (matches CPU `sample_top_p`). The sum
        // is taken AFTER the cap is known (see below) so the default path sums
        // only the legacy top-20.
        float inv_temp = 1.0f / temperature;
        float probs[64];
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) {
            if (vals[k] <= -1e29f) { probs[k] = 0.0f; continue; }
            probs[k] = expf((vals[k] - max_val) * inv_temp);
        }

        // Sort indices by probability descending (insertion sort).
        int order[64];
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) order[k] = k;
        for (int i = 1; i < TOP_K; i++) {
            int cur = order[i];
            float cv = probs[cur];
            int j = i;
            while (j > 0 && probs[order[j - 1]] < cv) {
                order[j] = order[j - 1];
                j--;
            }
            order[j] = cur;
        }

        // RNG: xorshift32.
        unsigned int rng = rand_seed;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;

        // Request-driven candidate cap (applied to the sorted order BEFORE
        // top-p). Two cuts:
        //   - top_k_req: keep at most this many candidates. The host passes 20
        //     by default (the legacy nucleus); a request may widen up to TOP_K.
        //   - min_p: drop candidates whose prob < min_p * max_prob. The sort
        //     above is descending, so order[0] holds the max prob; we scan
        //     until a candidate falls below the floor. 0.0 disables.
        int cap = TOP_K;
        if (top_k_req > 0 && top_k_req < TOP_K) cap = top_k_req;
        if (min_p > 0.0f) {
            float floor_p = min_p * probs[order[0]];
            int mp_cap = cap;
            for (int i = 1; i < cap; i++) {
                if (probs[order[i]] < floor_p) { mp_cap = i; break; }
            }
            cap = mp_cap;
        }
        if (cap < 1) cap = 1;

        // BYTE-PRESERVATION: softmax sum over the CAPPED set [0,cap) of the
        // descending-sorted candidates, NOT the full TOP_K=64 gather. With the
        // legacy default cap=20 this sums exactly the same top-20 probabilities
        // as the pre-widen 20-wide kernel, so p_thresh and the truncation are
        // byte-identical. A wider request (top_k=40) sums a wider nucleus.
        float sum = 0.0f;
        for (int i = 0; i < cap; i++) sum += probs[order[i]];
        if (sum <= 0.0f) { result[0] = (unsigned int)idx[order[0]]; result[1] = rand_seed; return; }

        // Top-p filtering: find cutoff where cumulative mass reaches top_p * sum.
        float p_thresh = top_p * sum;
        float trunc_sum = 0.0f;
        int trunc_k = cap;
        for (int i = 0; i < cap; i++) {
            trunc_sum += probs[order[i]];
            if (trunc_sum >= p_thresh) { trunc_k = i + 1; break; }
        }

        // Sample uniformly in [0, trunc_sum).
        float r = (float)rng / 4294967295.0f * trunc_sum;
        float cum = 0.0f;
        for (int i = 0; i < trunc_k; i++) {
            cum += probs[order[i]];
            if (cum >= r) {
                result[0] = (unsigned int)idx[order[i]];
                result[1] = rng;
                return;
            }
        }
        // FP edge case: tail — take the last top-p entry.
        result[0] = (unsigned int)idx[order[trunc_k - 1]];
        result[1] = rng;
    }
}
