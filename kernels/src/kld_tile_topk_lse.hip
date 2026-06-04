// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Exact per-row/per-chunk top-256 + logsumexp stats for KLD reference
// generation. Each block processes one 2048-logit chunk from one row:
//
//   logits     [B, vocab_tile]
//   top_vals   [B, n_chunks, 256]
//   top_idx    [B, n_chunks, 256] global token ids
//   chunk_max  [B, n_chunks]
//   chunk_sum  [B, n_chunks] sum(exp(logit - chunk_max))
//
// The host merges chunk top-256 candidate sets exactly. Any global top-256
// token must be inside the top-256 of its own 2048-token chunk.

#include <hip/hip_runtime.h>
#include <math.h>

#define KLD_CHUNK 2048
#define KLD_TOPK 256
#define KLD_THREADS 256

__device__ __forceinline__ bool less_pair(float av, int ai, float bv, int bi) {
    return (av < bv) || (av == bv && ai > bi);
}

__launch_bounds__(KLD_THREADS, 2)
extern "C" __global__ void kld_tile_topk_lse_f32(
    const float* __restrict__ logits,
    float*       __restrict__ top_vals,
    int*         __restrict__ top_idx,
    float*       __restrict__ chunk_max,
    float*       __restrict__ chunk_sum,
    int B,
    int vocab_tile,
    int global_start,
    int n_chunks
) {
    const int row = blockIdx.x;
    const int chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= B || chunk >= n_chunks) return;

    __shared__ float vals[KLD_CHUNK];
    __shared__ int idxs[KLD_CHUNK];
    __shared__ float partial[KLD_THREADS];

    const int local_base = chunk * KLD_CHUNK;
    for (int i = tid; i < KLD_CHUNK; i += KLD_THREADS) {
        const int local_idx = local_base + i;
        if (local_idx < vocab_tile) {
            vals[i] = logits[(long long)row * vocab_tile + local_idx];
            idxs[i] = global_start + local_idx;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = 0x7fffffff;
        }
    }
    __syncthreads();

    // Ascending bitonic sort by (logit asc, index desc). Reading the tail in
    // reverse gives descending logits with lower token id first on ties.
    for (int k = 2; k <= KLD_CHUNK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < KLD_CHUNK; i += KLD_THREADS) {
                int ixj = i ^ j;
                if (ixj > i) {
                    float vi = vals[i];
                    float vx = vals[ixj];
                    int ii = idxs[i];
                    int ix = idxs[ixj];
                    bool ascending = ((i & k) == 0);
                    bool should_swap = ascending
                        ? !less_pair(vi, ii, vx, ix)
                        : less_pair(vi, ii, vx, ix);
                    if (should_swap) {
                        vals[i] = vx;
                        vals[ixj] = vi;
                        idxs[i] = ix;
                        idxs[ixj] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }

    const float max_v = vals[KLD_CHUNK - 1];
    float sum = 0.0f;
    for (int i = tid; i < KLD_CHUNK; i += KLD_THREADS) {
        if (isfinite(vals[i]) && isfinite(max_v)) {
            sum += expf(vals[i] - max_v);
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (int stride = KLD_THREADS >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    const long long stat_off = (long long)row * n_chunks + chunk;
    if (tid == 0) {
        chunk_max[stat_off] = max_v;
        chunk_sum[stat_off] = partial[0];
    }

    const long long out_base = ((long long)row * n_chunks + chunk) * KLD_TOPK;
    for (int r = tid; r < KLD_TOPK; r += KLD_THREADS) {
        int src = KLD_CHUNK - 1 - r;
        top_vals[out_base + r] = vals[src];
        top_idx[out_base + r] = idxs[src];
    }
}
