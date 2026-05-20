// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Per-row top-K + log-sum-exp over a [B × vocab] logits matrix.
// Used by DDTree's `topk_from_logits` to avoid a 20 ms CPU sort per
// tree-verify cycle — this kernel does both the top-K selection AND
// the log-normalization on-device and returns only (B × K) tokens and
// log-probs to the host.
//
// Grid: [B, 1, 1]          — one block per row.
// Block: [256, 1, 1]       — 256 threads per block (8 waves, RDNA3).
// Shared memory (dynamic): (nthreads * K + nthreads) * 4 bytes =
//   (256 * MAX_K + 256) floats ≈ 9 KB at K=8.
//
// Inputs:
//   logits     : [B × vocab] f32  (row-major).
//   vocab      : row width.
//   K          : 1..MAX_K.
//   B          : number of rows (ignored by kernel — grid sets it).
//
// Outputs:
//   top_idx    : [B × K] i32 token indices (descending by log-prob).
//   top_logp   : [B × K] f32 log-probs (i.e. `logit - log_z`), matched
//                to `top_idx`.
//
// Algorithm (per block, single row):
//   Pass A: running max + per-thread top-K of (val, idx) over the full
//           vocab with stride nthreads. Each thread keeps K best.
//   Reduce: all 256 threads' top-K lists drop into LDS. Thread 0 does
//           one serial top-K merge (256×K candidates → K winners). K is
//           small (≤8) so this is ~2K comparisons total on one thread.
//   Pass B: sum_exp using the global max (from pass A reduce).
//   Final: thread 0 computes log_z = max + ln(sum_exp), writes
//          (top_idx, top_logp) with log_z subtracted.
//
// Note on numerics: max is reduced via warp shuffles + LDS. sum_exp is
// computed in f64 to avoid underflow on large-magnitude logits, then
// cast to f32 for log_z output.

#define MAX_K 8

extern "C" __launch_bounds__(256, 4)
__global__ void topk_logsumexp_batched_f32(
    const float* __restrict__ logits,    // [B × vocab]
    int*         __restrict__ top_idx,   // [B × K]
    float*       __restrict__ top_logp,  // [B × K]
    int vocab,
    int K
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const float* row = logits + (size_t)b * vocab;

    // Per-thread state: local top-K (val, idx) + running max.
    float loc_val[MAX_K];
    int   loc_idx[MAX_K];
    #pragma unroll
    for (int j = 0; j < MAX_K; j++) {
        loc_val[j] = -INFINITY;
        loc_idx[j] = 0;
    }
    float local_max = -INFINITY;

    // ── Pass A: running max + per-thread top-K ────────────────────────
    for (int i = tid; i < vocab; i += nth) {
        const float v = row[i];
        if (v > local_max) local_max = v;

        // Find the min among this thread's K and replace if v > min.
        int   min_j = 0;
        float min_v = loc_val[0];
        #pragma unroll
        for (int j = 1; j < MAX_K; j++) {
            if (j < K && loc_val[j] < min_v) {
                min_v = loc_val[j];
                min_j = j;
            }
        }
        if (v > min_v) {
            loc_val[min_j] = v;
            loc_idx[min_j] = i;
        }
    }

    // ── Reduce global max across waves ────────────────────────────────
    // Within-wave reduction via shuffles.
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    }
    // Cross-wave via LDS (nth / 32 entries).
    extern __shared__ float sdata[];
    float* warp_max = sdata;                   // [nth / 32] wave maxes
    float* sorted_candidates = sdata + 32;     // [nth × MAX_K × 2]
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    float global_max = -INFINITY;
    if (warp_id == 0) {
        float v = (lane < (nth >> 5)) ? warp_max[lane] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor(v, off));
        }
        if (lane == 0) warp_max[0] = v;
    }
    __syncthreads();
    global_max = warp_max[0];

    // ── Drop each thread's top-K into LDS for serial merge ────────────
    // Layout: sorted_candidates[t * MAX_K * 2 + k * 2 + 0] = val (f32)
    //         sorted_candidates[t * MAX_K * 2 + k * 2 + 1] = idx (int, reinterpret)
    for (int j = 0; j < K; j++) {
        sorted_candidates[tid * MAX_K * 2 + j * 2 + 0] = loc_val[j];
        *reinterpret_cast<int*>(&sorted_candidates[tid * MAX_K * 2 + j * 2 + 1]) = loc_idx[j];
    }
    __syncthreads();

    // Thread 0 merges 256 × K candidates into a final top-K.
    float final_val[MAX_K];
    int   final_idx[MAX_K];
    if (tid == 0) {
        #pragma unroll
        for (int j = 0; j < MAX_K; j++) { final_val[j] = -INFINITY; final_idx[j] = 0; }
        const int total = nth * K;
        for (int c = 0; c < total; c++) {
            const int t  = c / K;
            const int kk = c % K;
            const float v = sorted_candidates[t * MAX_K * 2 + kk * 2 + 0];
            const int   i = *reinterpret_cast<int*>(&sorted_candidates[t * MAX_K * 2 + kk * 2 + 1]);
            // Insert into sorted (desc) final_val[0..K). Find insertion slot.
            int ins = K;
            #pragma unroll
            for (int j = K - 1; j >= 0; j--) {
                if (v > final_val[j]) ins = j;
            }
            if (ins < K) {
                for (int j = K - 1; j > ins; j--) {
                    final_val[j] = final_val[j - 1];
                    final_idx[j] = final_idx[j - 1];
                }
                final_val[ins] = v;
                final_idx[ins] = i;
            }
        }
    }
    __syncthreads();

    // ── Pass B: sum_exp using the global max ──────────────────────────
    // Double-precision accumulator per thread, then wave + LDS reduction.
    double local_sum = 0.0;
    for (int i = tid; i < vocab; i += nth) {
        local_sum += (double)expf(row[i] - global_max);
    }
    // Fold f64 via two f32 shuffles (coarse) — reasonable for logits
    // magnitudes we see in practice; precision loss < 1 ULP in log_z.
    float ls = (float)local_sum;
    for (int off = 16; off > 0; off >>= 1) {
        ls += __shfl_xor(ls, off);
    }
    if (lane == 0) warp_max[warp_id] = ls;
    __syncthreads();
    float total_sum = 0.0f;
    if (warp_id == 0) {
        float v = (lane < (nth >> 5)) ? warp_max[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor(v, off);
        }
        if (lane == 0) warp_max[0] = v;
    }
    __syncthreads();
    total_sum = warp_max[0];

    // ── Write output (thread 0 only) ──────────────────────────────────
    if (tid == 0) {
        const float log_z = global_max + __logf(total_sum);
        for (int j = 0; j < K; j++) {
            top_idx[(size_t)b * K + j] = final_idx[j];
            top_logp[(size_t)b * K + j] = final_val[j] - log_z;
        }
    }
}
