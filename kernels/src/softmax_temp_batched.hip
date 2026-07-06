// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batched temperature-scaled softmax, out-of-place.
//
// For each row r (one block per row), computes
//
//     s_i   = logits[r*n + i] * inv_t
//     m     = max_i s_i
//     e_i   = expf(s_i - m)
//     probs[r*n + i] = e_i / sum_i e_i
//
// This MIRRORS the host `softmax_temp_into` in
// crates/hipfire-arch-qwen35/src/speculative.rs: temperature is applied to
// each logit FIRST (v * inv_t), the max is taken over the scaled logits, and
// exp is taken of (scaled - max). The arithmetic ORDER of the reduction is a
// tree reduction here vs. a sequential sum on the host, so the produced
// probabilities are DISTRIBUTION-parity, NOT byte-identical, to the host
// softmax (see HIPFIRE_DFLASH_FAST_SAMPLE gating in speculative.rs).
//
// Out-of-place: the input `logits` buffer is left untouched (the caller still
// needs the raw logits for argmax on the same rows).

#include <hip/hip_runtime.h>

extern "C" __global__ void softmax_temp_batched_f32(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int n,
    float inv_t
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* row_in = logits + (size_t)row * n;
    float* row_out = probs + (size_t)row * n;

    // max over scaled logits
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float s = row_in[i] * inv_t;
        max_val = fmaxf(max_val, s);
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // exp(scaled - max), accumulate sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = expf(row_in[i] * inv_t - max_val);
        row_out[i] = e;
        sum += e;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

// Variant that additionally emits the per-row nucleus (top_p) threshold and
// kept mass, WITHOUT modifying `probs` (host applies the truncation). For each
// row r it writes:
//
//   tau_cut[r] = the inclusive nucleus probability threshold: the value of the
//                crossing token's softmax probability. Host keeps tokens with
//                p_i >= tau_cut[r].
//   Z[r]       = the kept mass = sum of p_i for p_i >= tau_cut[r]. Host renorms
//                kept tokens by 1/Z[r].
//
// When top_p >= 1.0 the nucleus phase is skipped and tau_cut=0, Z=1 (host
// truncation collapses to identity, byte-path unchanged vs the no-top_p path).
//
// tau_cut is found by bisection over [0, p_max] on the monotone predicate
// mass(tau) = sum_{p_i >= tau} p_i, which increases as tau decreases. The AR
// nucleus rule (sort desc, accumulate, cut at the FIRST prefix whose cumulative
// >= top_p, INCLUSIVE crossing token, renorm by kept mass) is reproduced in
// threshold form: find the largest tau whose mass(tau) >= top_p. Because the
// full softmax already sums to 1, the kept-set sum used by AR is 1, so the
// cumulative target is exactly top_p. No sort: each bisection iteration is one
// strided pass over row_out + a block tree-reduce. A final exact pass recomputes
// Z = mass(tau_cut) from the converged threshold rather than trusting the
// interpolated mass. Comparator is `>=` (inclusive crossing token); exact float
// ties at the boundary are kept together (measure-zero divergence from AR's
// sort-position tie-break — see speculative.rs apply_topp_trunc doc).
extern "C" __global__ void softmax_temp_topp_batched_f32(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    float* __restrict__ tau_cut,   // [rows] out
    float* __restrict__ Z,         // [rows] out
    int n,
    float inv_t,
    float top_p,
    int top_k,
    float min_p
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* row_in = logits + (size_t)row * n;
    float* row_out = probs + (size_t)row * n;

    // ── Phase 1: max over scaled logits ────────────────────────────────────
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float s = row_in[i] * inv_t;
        max_val = fmaxf(max_val, s);
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // ── Phase 2: exp(scaled - max), accumulate sum ─────────────────────────
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = expf(row_in[i] * inv_t - max_val);
        row_out[i] = e;
        sum += e;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    // ── Phase 3: normalize → row_out holds the full softmax p_i, sum == 1 ───
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
    __syncthreads();

    // ── top_p >= 1.0 guard: nucleus disabled → identity truncation. With a
    // top_k OR min_p cutoff the truncation is NOT identity (each still limits the
    // candidate set), so only short-circuit when ALL three are disabled.
    if (top_p >= 1.0f && top_k <= 0 && min_p <= 0.0f) {
        if (threadIdx.x == 0) {
            tau_cut[row] = 0.0f;
            Z[row] = 1.0f;
        }
        return;
    }

    // ── Phase 4: p_max over the normalized row (bisection upper bound) ──────
    float pmax = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        pmax = fmaxf(pmax, row_out[i]);
    }
    sdata[threadIdx.x] = pmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    pmax = sdata[0];
    __syncthreads();

    // ── Phase 5': histogram nucleus (replaces the 20-iteration bisection) ───
    // The bisection re-read all `n` probs ~20× — the dominant cost when the
    // trunk forward is cheap (MoE A3B: 665µs/call, 13% of GPU time). Instead,
    // ONE pass bins each SIGNIFICANT prob's MASS into NBINS linear buckets over
    // [0, pmax]; thread 0 then scans buckets top-down (highest prob first) for
    // the crossing where cumulative kept-mass >= top_p. tau = the lower edge of
    // that bucket — the largest bucket-resolution threshold whose kept mass
    // still covers top_p, matching the bisection's intent at bucket precision.
    // Probs below pmax*1e-5 carry negligible mass AND sit far below any
    // realistic nucleus threshold, so they're skipped — which also kills the
    // atomic contention the 248K-long tail would otherwise put on the low bins.
    // Phase 6 recomputes the exact kept mass Z from this tau, so only the
    // threshold (not Z) is bucket-approximate.
    const int NBINS = 512;
    __shared__ float shist[512];     // per-bin prob MASS  (→ top_p threshold)
    __shared__ float shist_cnt[512]; // per-bin prob COUNT (→ top_k threshold)
    for (int b = threadIdx.x; b < NBINS; b += blockDim.x) {
        shist[b] = 0.0f;
        shist_cnt[b] = 0.0f;
    }
    __syncthreads();
    float floor_p = pmax * 1e-5f;
    float inv_pmax_bins = (pmax > 0.0f) ? ((float)NBINS / pmax) : 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float p = row_out[i];
        if (p < floor_p) continue;
        int bin = (int)(p * inv_pmax_bins);
        if (bin >= NBINS) bin = NBINS - 1;
        atomicAdd(&shist[bin], p);
        atomicAdd(&shist_cnt[bin], 1.0f);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Scan buckets top-down (highest prob first). tau_p = lower edge of the
        // bucket where cumulative MASS first reaches top_p (the nucleus cut).
        // tau_k = lower edge of the bucket where cumulative COUNT first reaches
        // top_k (the k-th largest prob, at bucket precision). top_k AND top_p
        // keep tokens above BOTH thresholds, so the effective cutoff is the
        // tighter (higher) of the two. The pmax*1e-5 floor skipped the long tail,
        // which sits below the top-k for any realistic k, so the count is exact.
        float cum_mass = 0.0f;
        float cum_cnt = 0.0f;
        float tau_p = 0.0f;
        float tau_k = 0.0f;
        bool found_p = (top_p >= 1.0f); // top_p disabled → no mass cut (tau_p=0)
        bool found_k = (top_k <= 0);    // top_k disabled → no count cut (tau_k=0)
        for (int b = NBINS - 1; b >= 0 && !(found_p && found_k); --b) {
            cum_mass += shist[b];
            cum_cnt += shist_cnt[b];
            if (!found_p && cum_mass >= top_p) {
                tau_p = ((float)b / (float)NBINS) * pmax;
                found_p = true;
            }
            if (!found_k && cum_cnt >= (float)top_k) {
                tau_k = ((float)b / (float)NBINS) * pmax;
                found_k = true;
            }
        }
        // min_p: keep tokens with p >= min_p * pmax (relative-to-peak floor).
        // Folded as the tighter (higher) of the three thresholds — no histogram
        // needed since pmax is already known. min_p<=0 → tau_minp=0 (no cut).
        float tau_minp = (min_p > 0.0f) ? (min_p * pmax) : 0.0f;
        sdata[0] = fmaxf(fmaxf(tau_p, tau_k), tau_minp);
    }
    __syncthreads();
    float tau = sdata[0];

    // ── Phase 6: exact Z = mass(tau) from the converged threshold ──────────
    float kept = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float p = row_out[i];
        if (p >= tau) kept += p;
    }
    sdata[threadIdx.x] = kept;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    kept = sdata[0];
    __syncthreads();

    if (threadIdx.x == 0) {
        tau_cut[row] = tau;
        Z[row] = kept;
    }
}
