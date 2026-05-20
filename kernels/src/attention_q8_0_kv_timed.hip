// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Phase-timed variant of attention_q8_0_kv. Functionally equivalent to
// the baseline kernel but records cycle counts for each of the three
// internal phases (QK^T, softmax, V-weighted-sum) into a per-head output
// buffer so we can see which phase dominates at long context.
//
// Grid: [n_heads]. Block: same as baseline (power of 2 ≤ 256).
// Shared: (seq_len + block_size + head_dim) * 4  — same as baseline.
// Timing output: cycle_counts[n_heads * 3], per-head (phase1, phase2, phase3).
//
// Cycle unit: wall_clock64() returns a wall-clock counter
// (not a per-CU cycle counter), so cross-head counts are comparable.
#include <hip/hip_runtime.h>

extern "C" __global__ void attention_q8_0_kv_timed(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    long long* __restrict__ cycle_counts  // [n_heads * 3]
) {
    int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];

    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* q_head = q + h * head_dim;
    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;

    float* scores = sdata;
    float* workspace = sdata + seq_len;

    // Preload Q head into shared memory (loaded once, used for all positions)
    float* q_shared = workspace + nthreads;  // after workspace
    for (int d = tid; d < head_dim; d += nthreads)
        q_shared[d] = q_head[d];
    __syncthreads();

    // ═══ Phase 1: QK^T + local_max ═══
    long long t0 = wall_clock64();

    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float dot = 0.0f;
        for (int bi = 0; bi < blocks_per_head; bi++) {
            const unsigned char* block = k_cache + (size_t)t * total_blocks_per_pos * 34
                                        + (kv_head_block_start + bi) * 34;
            float d = (float)*((const _Float16*)block);
            const float* qb = q_shared + bi * 32;
            for (int j = 0; j < 32; j++) {
                dot += qb[j] * (d * (float)((signed char)block[2 + j]));
            }
        }
        float s = dot * scale_attn;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }
    __syncthreads();

    long long t1 = wall_clock64();

    // ═══ Phase 2: softmax ═══
    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    float max_val = workspace[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    float sum_val = workspace[0];
    __syncthreads();

    for (int t = tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    long long t2 = wall_clock64();

    // ═══ Phase 3: V-weighted sum ═══
    float* out_head = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        int bi = d / 32;
        int bj = d % 32;
        for (int t = 0; t < seq_len; t++) {
            const unsigned char* block = v_cache + (size_t)t * total_blocks_per_pos * 34
                                        + (kv_head_block_start + bi) * 34;
            float vd = (float)*((const _Float16*)block) * (float)(signed char)block[2 + bj];
            val += scores[t] * vd;
        }
        out_head[d] = val;
    }

    long long t3 = wall_clock64();

    // Write phase cycles (thread 0 only)
    if (tid == 0) {
        cycle_counts[h * 3 + 0] = t1 - t0;  // phase 1: QK^T
        cycle_counts[h * 3 + 1] = t2 - t1;  // phase 2: softmax
        cycle_counts[h * 3 + 2] = t3 - t2;  // phase 3: V-weighted sum
    }
}
