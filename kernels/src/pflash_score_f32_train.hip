// SPDX-License-Identifier: Apache-2.0
// hipfire — PFlash importance-score head, fp32 TRAINING twin.
//
// Forward matches the production inference kernel `pflash_score_q8_kv` and the
// CPU `compute_scores_batched`: per block, score = cosine(block_mean_K,
// last_token_K) over the FULL kv_dim vector (not per-head averaged). The
// training twin adds the backward (gradient w.r.t. K) that inference never
// needs, so the drafter can be trained to reproduce a target's block ranking.
//
// One workgroup (256 threads) per block; threads stride over kv_dim and a
// shared-memory reduction folds the (dot, |mean|^2, |last|^2) fragments.
//
//   k:       [n_pos * kv_dim] fp32   (drafter's last-block post-rope K)
//   scores:  [n_blocks]       fp32
//   dscores: [n_blocks]       fp32   (upstream grad)
//   dk:      [n_pos * kv_dim] fp32   (MUST be zeroed before bwd; accumulated)

#include <hip/hip_runtime.h>

#define EPS 1e-12f

extern "C" __global__ void pflash_score_f32_fwd(
    const float* __restrict__ k,
    float* __restrict__ scores,
    int n_pos,
    int kv_dim,
    int block_size,
    int n_blocks,
    int last_pos
) {
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    int tid = threadIdx.x;

    int start = b * block_size;
    int end = start + block_size;
    if (end > n_pos) end = n_pos;
    int len = end - start;

    __shared__ float s_dot[256];
    __shared__ float s_nb[256];
    __shared__ float s_nl[256];

    if (len <= 0) {
        if (tid == 0) scores[b] = 0.0f;
        return;
    }
    float finv = 1.0f / (float)len;

    float dot = 0.0f, nb = 0.0f, nl = 0.0f;
    for (int d = tid; d < kv_dim; d += 256) {
        float last_v = k[last_pos * kv_dim + d];
        float m = 0.0f;
        for (int t = start; t < end; t++) m += k[t * kv_dim + d];
        m *= finv;
        dot += m * last_v;
        nb += m * m;
        nl += last_v * last_v;
    }
    s_dot[tid] = dot;
    s_nb[tid] = nb;
    s_nl[tid] = nl;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_dot[tid] += s_dot[tid + s];
            s_nb[tid] += s_nb[tid + s];
            s_nl[tid] += s_nl[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float Nb = sqrtf(s_nb[0]);
        float Nl = sqrtf(s_nl[0]);
        scores[b] = s_dot[0] / (Nb * Nl + EPS);
    }
}

extern "C" __global__ void pflash_score_f32_bwd(
    const float* __restrict__ k,
    const float* __restrict__ dscores,
    float* __restrict__ dk,
    int n_pos,
    int kv_dim,
    int block_size,
    int n_blocks,
    int last_pos
) {
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    int tid = threadIdx.x;

    int start = b * block_size;
    int end = start + block_size;
    if (end > n_pos) end = n_pos;
    int len = end - start;
    if (len <= 0) return;
    float finv = 1.0f / (float)len;

    __shared__ float s_dot[256];
    __shared__ float s_nb[256];
    __shared__ float s_nl[256];

    // pass 1: reduce dot, |mean|^2, |last|^2 (identical to fwd)
    float dot = 0.0f, nb = 0.0f, nl = 0.0f;
    for (int d = tid; d < kv_dim; d += 256) {
        float last_v = k[last_pos * kv_dim + d];
        float m = 0.0f;
        for (int t = start; t < end; t++) m += k[t * kv_dim + d];
        m *= finv;
        dot += m * last_v;
        nb += m * m;
        nl += last_v * last_v;
    }
    s_dot[tid] = dot;
    s_nb[tid] = nb;
    s_nl[tid] = nl;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_dot[tid] += s_dot[tid + s];
            s_nb[tid] += s_nb[tid + s];
            s_nl[tid] += s_nl[tid + s];
        }
        __syncthreads();
    }
    __syncthreads();
    float DOT = s_dot[0];
    float Nb = sqrtf(s_nb[0]) + EPS;
    float Nl = sqrtf(s_nl[0]) + EPS;
    float denom = Nb * Nl + EPS;
    float invden2 = 1.0f / (denom * denom);
    float ds = dscores[b];

    // pass 2: scatter grads. score = DOT/denom, denom = Nb*Nl+eps.
    //   d score/d mean_d = (last_d*denom - DOT*Nl*mean_d/Nb) / denom^2
    //   d score/d last_d = (mean_d*denom - DOT*Nb*last_d/Nl) / denom^2
    // mean_d = (1/len) Σ_{t∈block} k[t,d]  ⇒ d mean_d/d k[t,d] = 1/len.
    // last_d is shared by every block ⇒ accumulate into dk[last_pos] (atomic).
    for (int d = tid; d < kv_dim; d += 256) {
        float last_v = k[last_pos * kv_dim + d];
        float m = 0.0f;
        for (int t = start; t < end; t++) m += k[t * kv_dim + d];
        m *= finv;

        float dmean = ds * (last_v * denom - DOT * Nl * m / Nb) * invden2;
        float g_per_t = dmean * finv;
        for (int t = start; t < end; t++) {
            atomicAdd(&dk[t * kv_dim + d], g_per_t);
        }
        float dlast = ds * (m * denom - DOT * Nb * last_v / Nl) * invden2;
        atomicAdd(&dk[last_pos * kv_dim + d], dlast);
    }
}
