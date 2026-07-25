// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batched categorical sampler — one block per draft position.
//
// Inputs:
//   probs[batch * vocab]  — already-softmaxed probabilities (out-of-place from
//                           softmax_temp_batched_f32 or _topp_batched_f32).
//                           The buffer is NOT modified.
//   tau_cut[batch]        — nucleus threshold per row (0.0 = no truncation).
//                           Comes from softmax_temp_topp_batched_f32.
//   z[batch]              — kept mass per row (1.0 when tau_cut==0).
//   vocab                 — vocabulary size.
//   seed                  — u32 base seed; each block uses (seed ^ blockIdx.x | 1).
//
// Outputs:
//   out_tokens[batch]     — sampled token id per row (i32).
//   out_probs[batch]      — effective probability at the sampled token (f32),
//                           AFTER top-p truncation (p_eff = p/Z if p>=tau, else 0).
//                           This is the p_d value the chain-accept kernel needs.
//   out_top_probs[batch]  — optional raw maximum softmax probability per row.
//                           Pass nullptr when the caller does not need it.
//
// At temp == 0: the caller should use argmax_f32_batched instead. This kernel
// does NOT special-case temp; it is only invoked when use_temp_sampling is true.
//
// RNG: LCG matching ddtree_swor_walk convention:
//   s = s * 1664525 + 1013904223  (Numerical Recipes)
//   u = (s >> 8) / 2^24
// Seeded per block as (seed ^ blockIdx.x) | 1 to avoid cross-row correlation.
//
// The CDF walk uses the effective probability:
//   p_eff[i] = (tau_cut > 0 && probs[i] < tau_cut) ? 0 : probs[i] / z
// where z = 1.0 when tau_cut == 0 (identity, no truncation).
// Total effective mass is 1.0 after truncation (Z is the kept mass, and we
// divide each kept prob by Z → they sum to 1).
//
// Block-parallel CDF: each thread accumulates a partial CDF over its strided
// slice; a block-reduce of the partial sums builds a prefix; then thread 0
// combines partials sequentially to locate the CDF crossing point.

#include <hip/hip_runtime.h>

#define BCAT_NTH 256

extern "C" __global__ void batched_categorical_sample_f32(
    const float* __restrict__ probs,      // [batch * vocab] f32
    const float* __restrict__ tau_cut,    // [batch] f32
    const float* __restrict__ z,          // [batch] f32
    int   vocab,
    unsigned int seed,
    int*  __restrict__ out_tokens,        // [batch] i32
    float* __restrict__ out_probs,        // [batch] f32
    float* __restrict__ out_top_probs     // optional [batch] f32
) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int nth  = blockDim.x;

    const float* row_p = probs + (size_t)row * vocab;
    const float  tau   = tau_cut[row];
    const float  inv_z = 1.0f / fmaxf(z[row], 1e-30f);
    // When tau == 0.0 truncation is disabled; inv_z == 1.0.

    // LCG RNG per block, seeded to avoid same-seed correlation across rows.
    __shared__ unsigned int s_rng;
    __shared__ float        s_red[BCAT_NTH];  // for block sum reduction
    __shared__ float        s_total_mass;
    __shared__ float        s_top_prob;
    __shared__ int          s_pick;
    __shared__ float        s_pick_prob;

    if (tid == 0) {
        s_rng  = (seed ^ (unsigned int)row) | 1u;
        s_pick = vocab - 1;  // fallback: last token
        s_pick_prob = 0.0f;
    }
    __syncthreads();

    // ── Step 1: draw u uniform ─────────────────────────────────────────────
    // Thread 0 advances RNG; result broadcast via shared mem.
    // (total_mass should be 1.0 after topp; keep the scan against u * total_mass
    // to stay robust if upstream leaves a slightly non-1 distribution.)
    float total = 0.0f;
    float top_prob = 0.0f;
    for (int i = tid; i < vocab; i += nth) {
        float p = row_p[i];
        float p_eff = (tau > 0.0f && p < tau) ? 0.0f : p * inv_z;
        total += p_eff;
        top_prob = fmaxf(top_prob, p);
    }
    s_red[tid] = total;
    __syncthreads();
    for (int s = nth / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        __syncthreads();
    }
    if (tid == 0) s_total_mass = s_red[0];
    __syncthreads();

    s_red[tid] = top_prob;
    __syncthreads();
    for (int s = nth / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] = fmaxf(s_red[tid], s_red[tid + s]);
        __syncthreads();
    }
    if (tid == 0) s_top_prob = s_red[0];
    __syncthreads();

    // Thread 0 draws u (scaled to total mass, which should be ~1.0).
    if (tid == 0) {
        s_rng = s_rng * 1664525u + 1013904223u;
        float u = (float)(s_rng >> 8) * (1.0f / 16777216.0f) * s_total_mass;
        // CDF walk: sequential over vocab, thread 0 only.
        // For large vocab this is a single-thread bottleneck, but correctness
        // requires sequential accumulation in index order.  In production the
        // caller invokes this kernel grid=[batch,1,1] block=[256,1,1] so the
        // per-block cost is O(vocab) sequential steps for thread 0 only; the
        // block-parallel portion above halved the constant by distributing the
        // mass-sum.  A parallel prefix-scan CDF would be faster but requires
        // O(vocab) LDS which is 600 KB for 152K vocab — infeasible.  The
        // sequential CDF walk is the same pattern used in ddtree_swor_walk.
        float acc = 0.0f;
        int pick = vocab - 1;
        float pick_p_eff = 0.0f;
        for (int i = 0; i < vocab; ++i) {
            float p = row_p[i];
            float p_eff = (tau > 0.0f && p < tau) ? 0.0f : p * inv_z;
            if (p_eff <= 0.0f) continue;
            acc += p_eff;
            if (u < acc) {
                pick = i;
                pick_p_eff = p_eff;
                break;
            }
        }
        // Fallback: u >= total_mass (floating point rounding), pick last nonzero.
        if (pick_p_eff == 0.0f) {
            for (int i = vocab - 1; i >= 0; --i) {
                float p = row_p[i];
                float p_eff = (tau > 0.0f && p < tau) ? 0.0f : p * inv_z;
                if (p_eff > 0.0f) { pick = i; pick_p_eff = p_eff; break; }
            }
        }
        s_pick      = pick;
        s_pick_prob = pick_p_eff;
    }
    __syncthreads();

    if (tid == 0) {
        out_tokens[row] = s_pick;
        out_probs[row]  = s_pick_prob;
        if (out_top_probs != nullptr) out_top_probs[row] = s_top_prob;
    }
}
