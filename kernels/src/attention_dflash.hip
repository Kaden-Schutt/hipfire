// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 alpineq
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Non-causal cross-attention kernel for DFlash draft forward.
//
// Each query position attends to ALL key/value positions (non-causal,
// bidirectional). Supports GQA where n_heads > n_kv_heads. Q/K/V are
// separate buffers — K can be longer than Q. Grid: [n_heads, B, 1],
// one block per (head, query-position).
//
// Layout (row-major, flat):
//   q    : [B  * n_heads    * head_dim]
//   k    : [L  * n_kv_heads * head_dim]
//   v    : [L  * n_kv_heads * head_dim]
//   out  : [B  * n_heads    * head_dim]
//
// rep = n_heads / n_kv_heads (must divide evenly).
//
// PR3 step 2 follow-up: this is a tiled online-softmax (FlashAttention-style)
// implementation. Replaces the prior single-pass `scores[L]` allocation that
// overflowed gfx1100/RDNA3's 64 KB LDS budget at L > ~16128. With tiling,
// LDS = (tile_size + block_size + head_dim) * 4 bytes — independent of L.
// Single-tile case (tile_size = L when L ≤ tile_size) is mathematically
// equivalent to the prior kernel up to FP non-associativity in the same
// reduce trees. Multi-tile case carries a running (max, sum, output)
// triplet across tiles, mathematically identical to single-pass softmax
// modulo FP order.

#include <hip/hip_runtime.h>

extern "C" __global__ void attention_dflash_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale,
    int tile_size
) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    int qi = blockIdx.y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (head >= n_heads || qi >= B) return;

    int rep = n_heads / n_kv_heads;
    int kv_head = head / rep;
    int q_stride = n_heads * head_dim;
    int kv_stride = n_kv_heads * head_dim;

    // LDS layout: [tile_scores[tile_size]] [ws[nthreads]] [out_run[head_dim]]
    float* tile_scores = sdata;                                // [tile_size]
    float* ws          = sdata + tile_size;                    // [nthreads]
    float* out_run     = sdata + tile_size + nthreads;         // [head_dim]

    const float* q_ptr = q + qi * q_stride + head * head_dim;

    // Initialize running output to 0.
    for (int d = tid; d < head_dim; d += nthreads) {
        out_run[d] = 0.0f;
    }
    __syncthreads();

    // Online-softmax running state. m_run starts at -INFINITY so the
    // first-tile bootstrap is exact: alpha = exp(-inf - finite) = 0
    // by IEEE-754, no reliance on FP underflow of (-1e30 - finite).
    float m_run = -INFINITY;  // running max of all scores seen so far
    float l_run = 0.0f;       // running sum of exp(scores - m_run)

    int n_tiles = (L + tile_size - 1) / tile_size;
    for (int t = 0; t < n_tiles; ++t) {
        int tile_start = t * tile_size;
        int tile_len = L - tile_start;
        if (tile_len > tile_size) tile_len = tile_size;

        // Phase A: compute scores for this tile, track tile-local max.
        //
        // Inner dot product is unrolled by 8 with 8 independent
        // accumulators so the FMA chain isn't serially dependent on
        // a single accumulator. RDNA3 FMA latency is ~4 cycles, so a
        // 128-element single-accumulator chain stalls for ~512 cycles
        // per j; 8-way ILP lets up to 8 FMAs be in flight at once,
        // bringing the per-j latency close to the 16 sequential adds
        // of the final tree reduce.
        //
        // Requires head_dim divisible by 8 for the fast path. Tail
        // loop handles the remainder; in practice the configurations
        // we care about (head_dim ∈ {64, 72, 128, 256, 512}) are all
        // multiples of 8.
        float local_max = -INFINITY;
        for (int j = tid; j < tile_len; j += nthreads) {
            int abs_j = tile_start + j;
            const float* k_ptr = k + abs_j * kv_stride + kv_head * head_dim;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
            const int head_dim_8 = head_dim & ~7;
            #pragma unroll 4
            for (int d = 0; d < head_dim_8; d += 8) {
                a0 += q_ptr[d    ] * k_ptr[d    ];
                a1 += q_ptr[d + 1] * k_ptr[d + 1];
                a2 += q_ptr[d + 2] * k_ptr[d + 2];
                a3 += q_ptr[d + 3] * k_ptr[d + 3];
                a4 += q_ptr[d + 4] * k_ptr[d + 4];
                a5 += q_ptr[d + 5] * k_ptr[d + 5];
                a6 += q_ptr[d + 6] * k_ptr[d + 6];
                a7 += q_ptr[d + 7] * k_ptr[d + 7];
            }
            // Tail for head_dim % 8 != 0 (no-op for hd ∈ {64,72,128,256,512}).
            for (int d = head_dim_8; d < head_dim; ++d) {
                a0 += q_ptr[d] * k_ptr[d];
            }
            // Tree reduce 8 accumulators → 1.
            float r01 = a0 + a1, r23 = a2 + a3, r45 = a4 + a5, r67 = a6 + a7;
            float dot = (r01 + r23) + (r45 + r67);
            float s = dot * scale;
            tile_scores[j] = s;
            if (s > local_max) local_max = s;
        }
        ws[tid] = local_max;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float a = ws[tid];
                float b = ws[tid + s];
                ws[tid] = a > b ? a : b;
            }
            __syncthreads();
        }
        float tile_max = ws[0];
        __syncthreads();

        // Phase B: rescale running state to the new global max.
        //   m_new   = max(m_run, tile_max)
        //   alpha   = exp(m_run - m_new)             (= 1 if m_run won, < 1 otherwise)
        //   l_new   = alpha * l_run + sum_j exp(score_j - m_new)
        //   out_new = alpha * out_run + sum_j exp(score_j - m_new) * V[j]
        // First tile (m_run = -inf): alpha = exp(-inf - finite) = 0, so the
        // running state correctly reduces to fresh-tile contribution.
        float m_new = m_run > tile_max ? m_run : tile_max;
        float alpha = expf(m_run - m_new);

        // Convert tile_scores in place to exp(score - m_new), summing.
        float local_sum = 0.0f;
        for (int j = tid; j < tile_len; j += nthreads) {
            float e = expf(tile_scores[j] - m_new);
            tile_scores[j] = e;
            local_sum += e;
        }
        ws[tid] = local_sum;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) ws[tid] += ws[tid + s];
            __syncthreads();
        }
        float tile_sum = ws[0];
        __syncthreads();

        // Update running denominator.
        l_run = alpha * l_run + tile_sum;

        // Phase C: update running output. Mathematically:
        //   out_run[d] = alpha * out_run[d] + sum_j tile_scores[j] * V[j, d]
        //
        // The naive layout — `for (d = tid; d < head_dim; d += nthreads)` with
        // a serial inner j-loop — only parallelises across head_dim. For
        // typical DFlash configs (small L, small head_dim) that's fine.
        // For large L (vision tower at L≈20k), each active thread serially
        // loops `tile_len` (up to ~14k) j-positions per tile, and half the
        // 256 threads idle when head_dim=128 — the V-accumulation becomes
        // the kernel's hot spot.
        //
        // Cooperative layout: when `nthreads > head_dim` we have spare
        // threads. Map `tid -> (j_chunk_id, d_lane)`:
        //   d_lane     = tid % head_dim
        //   j_chunk_id = tid / head_dim
        //   j_chunks   = nthreads / head_dim    (≥ 1)
        // Each (j_chunk_id, d_lane) thread accumulates its slice of j into
        // a per-thread `partial` register, all threads write to ws[], then
        // the `j_chunk_id == 0` lanes reduce across j_chunks for their d.
        //
        // FP-equivalence: same multiply-add set as the naive layout, just
        // reordered into j_chunk groups. Numerical drift vs the naive form
        // is bounded by FP non-associativity (~1e-8 on the parity test).
        //
        // When nthreads <= head_dim (small DFlash configs), j_chunks=1 and
        // the cooperative layout collapses exactly to the naive layout —
        // identical work, identical order.
        int j_chunks = nthreads / head_dim;
        if (j_chunks < 1) j_chunks = 1;
        int d_lane = tid % head_dim;
        int j_chunk_id = tid / head_dim;

        float partial = 0.0f;
        if (j_chunk_id < j_chunks && d_lane < head_dim) {
            int j_per_chunk = (tile_len + j_chunks - 1) / j_chunks;
            int j_start_local = j_chunk_id * j_per_chunk;
            int j_end_local = j_start_local + j_per_chunk;
            if (j_end_local > tile_len) j_end_local = tile_len;
            // 4-way ILP on the j-loop: same motivation as phase A —
            // break the FMA chain into 4 independent accumulators so
            // the v load + score multiply + accumulate can overlap.
            const int v_lane_off = kv_head * head_dim + d_lane;
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            int j = j_start_local;
            for (; j + 4 <= j_end_local; j += 4) {
                int base = tile_start + j;
                p0 += tile_scores[j    ] * v[(base    ) * kv_stride + v_lane_off];
                p1 += tile_scores[j + 1] * v[(base + 1) * kv_stride + v_lane_off];
                p2 += tile_scores[j + 2] * v[(base + 2) * kv_stride + v_lane_off];
                p3 += tile_scores[j + 3] * v[(base + 3) * kv_stride + v_lane_off];
            }
            // Tail (≤ 3 iterations).
            for (; j < j_end_local; ++j) {
                p0 += tile_scores[j] * v[(tile_start + j) * kv_stride + v_lane_off];
            }
            partial = (p0 + p1) + (p2 + p3);
        }
        // Threads with tid >= j_chunks * head_dim are idle (when
        // `nthreads % head_dim != 0`); their partial stays 0.
        ws[tid] = partial;
        __syncthreads();

        // The j_chunk_id == 0 lane (i.e. tid in [0, head_dim)) reduces
        // across j_chunks for its d and updates out_run[d].
        if (j_chunk_id == 0 && d_lane < head_dim) {
            float total = ws[d_lane];
            for (int jc = 1; jc < j_chunks; ++jc) {
                total += ws[jc * head_dim + d_lane];
            }
            out_run[d_lane] = alpha * out_run[d_lane] + total;
        }
        // Handle nthreads < head_dim: the cooperative layout above only
        // assigned d_lane = tid % head_dim values; tids that wrap past
        // head_dim never happen (j_chunks would be 1 and tid < nthreads
        // < head_dim, so d_lane = tid). But head_dim may still exceed
        // nthreads (e.g. head_dim=256 with nthreads=128). Cover the
        // remaining slots with the original per-d serial-j loop.
        for (int d = nthreads + tid; d < head_dim; d += nthreads) {
            float scaled_old = alpha * out_run[d];
            float contrib = 0.0f;
            for (int j = 0; j < tile_len; ++j) {
                int abs_j = tile_start + j;
                contrib += tile_scores[j] * v[abs_j * kv_stride + kv_head * head_dim + d];
            }
            out_run[d] = scaled_old + contrib;
        }
        __syncthreads();

        m_run = m_new;
    }

    // Finalize: out = out_run / l_run.
    float* out_ptr = out + qi * q_stride + head * head_dim;
    float inv_l = 1.0f / l_run;
    for (int d = tid; d < head_dim; d += nthreads) {
        out_ptr[d] = out_run[d] * inv_l;
    }
}
