// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Per-row Gumbel-top-k SWOR sampler: draws k tokens WITHOUT replacement from
// softmax(logits/temp) per row of a [B × vocab] matrix via the Gumbel-top-k trick
// (top-k by `logit/temp + Gumbel(0,1)` is an exact size-k sample without
// replacement), returning each token's TRUE per-row log-prob
// `log_softmax(logits/temp)[token]`. Replaces the host-side Gumbel sampling that
// forced a [B × vocab] D2H of the draft logits — the device draft logits now stay
// resident and only B×k indices/log-probs come back.
//
// One block per row. Output order is by descending perturbed score = the SWOR
// draw order the verbatim verify expects.

#include <hip/hip_runtime.h>

#define GTK_MAXK 8

extern "C" __global__ void ddtree_gumbel_topk_batched_f32(
    const float* __restrict__ logits,  // [B × vocab]
    int* __restrict__ top_idx,         // [B × K] draw-ordered token ids
    float* __restrict__ top_logp,      // [B × K] true log_softmax(logit/temp)
    int vocab, int K, float temp, unsigned int seed)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const float inv_t = 1.0f / fmaxf(temp, 1e-4f);
    const float* row = logits + (size_t)b * vocab;

    // Per-thread: top-K by perturbed score, plus running max of the TRUE score.
    float loc_score[GTK_MAXK]; // logit/temp + Gumbel
    int loc_idx[GTK_MAXK];
#pragma unroll
    for (int j = 0; j < GTK_MAXK; j++) {
        loc_score[j] = -INFINITY;
        loc_idx[j] = 0;
    }
    float local_max = -INFINITY; // over true logit/temp (for log-sum-exp)

    for (int i = tid; i < vocab; i += nth) {
        const float s = row[i] * inv_t;
        if (s > local_max) local_max = s;
        // Gumbel(0,1) from a murmur3 fmix32 hash of (seed, b, i). A strong
        // integer hash (vs a 2-round LCG) is essential: weak per-element noise is
        // correlated across vocab indices, which keeps rank-0 marginally correct
        // but biases the lower-rank without-replacement draws → noisy acceptance.
        unsigned int h =
            ((unsigned int)i * 0x9E3779B1u) ^ ((unsigned int)b * 0x85EBCA77u) ^ seed;
        h ^= h >> 16;
        h *= 0x7FEB352Du;
        h ^= h >> 15;
        h *= 0x846CA68Bu;
        h ^= h >> 16;
        float u = (float)(h >> 8) * (1.0f / 16777216.0f);
        if (u <= 1e-7f) u = 1e-7f;
        if (u >= 1.0f) u = 0.9999999f;
        const float g = -__logf(-__logf(u));
        const float score = s + g;
        int min_j = 0;
        float min_s = loc_score[0];
#pragma unroll
        for (int j = 1; j < GTK_MAXK; j++) {
            if (j < K && loc_score[j] < min_s) {
                min_s = loc_score[j];
                min_j = j;
            }
        }
        if (score > min_s) {
            loc_score[min_j] = score;
            loc_idx[min_j] = i;
        }
    }

    // Global max of true score (for lse). Wave shuffle + LDS.
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    extern __shared__ float sdata[];
    float* warp_max = sdata;                 // [nth/32]
    float* cand = sdata + 32;                // [nth × MAXK × 2]
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < (nth >> 5)) ? warp_max[lane] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off));
        if (lane == 0) warp_max[0] = v;
    }
    __syncthreads();
    const float gmax = warp_max[0];

    // Drop each thread's top-K (perturbed score, idx) for a serial merge.
    for (int j = 0; j < K; j++) {
        cand[tid * GTK_MAXK * 2 + j * 2 + 0] = loc_score[j];
        *reinterpret_cast<int*>(&cand[tid * GTK_MAXK * 2 + j * 2 + 1]) = loc_idx[j];
    }
    __syncthreads();

    // Pass B: sum_exp of true scores for the log-prob denominator.
    double lsum = 0.0;
    for (int i = tid; i < vocab; i += nth) lsum += (double)__expf(row[i] * inv_t - gmax);
    float ls = (float)lsum;
    for (int off = 16; off > 0; off >>= 1) ls += __shfl_xor(ls, off);
    if (lane == 0) warp_max[warp_id] = ls;
    __syncthreads();
    float lse = 0.0f;
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (nth >> 5); w++) tot += warp_max[w];
        lse = gmax + __logf(tot);
        // Merge nth×K candidates → final top-K by perturbed score (desc).
        float fs[GTK_MAXK];
        int fi[GTK_MAXK];
#pragma unroll
        for (int j = 0; j < GTK_MAXK; j++) {
            fs[j] = -INFINITY;
            fi[j] = 0;
        }
        const int total = nth * K;
        for (int c = 0; c < total; c++) {
            const int t = c / K, kk = c % K;
            const float v = cand[t * GTK_MAXK * 2 + kk * 2 + 0];
            const int i = *reinterpret_cast<int*>(&cand[t * GTK_MAXK * 2 + kk * 2 + 1]);
            int ins = K;
#pragma unroll
            for (int j = GTK_MAXK - 1; j >= 0; j--) {
                if (j < K && v > fs[j]) ins = j;
            }
            if (ins < K) {
                for (int j = K - 1; j > ins; j--) {
                    fs[j] = fs[j - 1];
                    fi[j] = fi[j - 1];
                }
                fs[ins] = v;
                fi[ins] = i;
            }
        }
        for (int j = 0; j < K; j++) {
            top_idx[b * K + j] = fi[j];
            top_logp[b * K + j] = row[fi[j]] * inv_t - lse; // true log q
        }
    }
}
