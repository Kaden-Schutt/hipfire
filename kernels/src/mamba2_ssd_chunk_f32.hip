// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Mamba-2 SSD (selective state-space duality) — CHUNKED PREFILL scan, f32.
//
// This is the GPU port of the CPU oracle `hipfire_arch_nemotron::ssd::ssd_chunked`
// (the genuinely-parallel intra-chunk decomposition), and the CORRECTNESS FLOOR
// for the bf16-WMMA kernel that replaces the four inner matmuls (CB Gram, masked
// intra-chunk `M·X`, inter-chunk `C·h_in`, state carry `coef·B`). It is laid out
// so that exact reassociation matches `ssd_chunked` bit-for-bit, and `ssd_chunked`
// matches the sequential scan `ssd_sequence` within ~1e-4. See
// `mamba2_ssd_seq.hip` for the sequential floor and `docs/plans/
// 2026-06-28-mamba2-ssd-wmma-prefill.md` for the WMMA staging plan.
//
// Per chunk of L (≤ chunk_size) positions, per (head, head_dim channel p), with
// S_t = Σ_{r≤t} dt_r·A the inclusive cumulative log-decay (A = −exp(A_log[head])):
//
//   y_t[p] = exp(S_t)·(C_t · h_in[p])                       # inter-chunk (state)
//          + Σ_{s≤t} exp(S_t − S_s)·dt_s·(C_t · B_s)·x_s[p] # intra-chunk (L⊙G)
//          + D·x_t[p]                                       # skip
//   h_out[p][n] = exp(S_{L-1})·h_in[p][n]
//               + Σ_s exp(S_{L-1} − S_s)·dt_s·B_s[n]·x_s[p] # carry to next chunk
//
// Using exp(S_t − S_s) (a difference of cumulative SUMS) rather than a product of
// per-step dA keeps the decay matrix stable for long chunks. The state row is
// updated in place: outputs read it as h_in (still unmodified), then the carry
// rewrites it (`srow *= exp(S_end)` reuses the old h_in, then adds the B/x sum),
// so no per-thread h_in snapshot is needed. B/C use the interleave group
// head/(num_heads/n_groups), matching the decode/seq kernels.

#include <hip/hip_runtime.h>

// Max GPU chunk tile. Bounds the per-thread cumulative-decay scratch. The host
// caps chunk_size to this; a larger model chunk is processed as several tiles
// (numerically still equivalent to the sequential scan).
#define MAMBA2_CHUNK_MAX 64

__device__ __forceinline__ float softplus_stable_chunk(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

extern "C" __global__ void mamba2_ssd_chunk_f32(
    float* __restrict__ y,             // [seq * num_heads * head_dim]
    float* __restrict__ state,         // [num_heads * head_dim * state_size] (in/out)
    const float* __restrict__ x,       // [seq * num_heads * head_dim]
    const float* __restrict__ B,       // [seq * n_groups * state_size]
    const float* __restrict__ C,       // [seq * n_groups * state_size]
    const float* __restrict__ dt_raw,  // [seq * num_heads]
    const float* __restrict__ A_log,   // [num_heads]
    const float* __restrict__ D,       // [num_heads]
    const float* __restrict__ dt_bias, // [num_heads]
    int seq,
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    float dt_min,
    float dt_max,
    int chunk_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // (head, p)
    int total = num_heads * head_dim;
    if (idx >= total) return;

    int head = idx / head_dim;
    int p = idx - head * head_dim;
    int grp = head / (num_heads / n_groups);

    float a = -expf(A_log[head]);
    float Dh = D[head];
    float dtb = dt_bias[head];

    const int xd = num_heads * head_dim; // per-position x/y stride
    const int bd = n_groups * state_size; // per-position B/C stride
    float* srow = state + (size_t)idx * state_size;

    // Per-thread cumulative-decay scratch for the current chunk.
    float dt_loc[MAMBA2_CHUNK_MAX];
    float scum_loc[MAMBA2_CHUNK_MAX];

    for (int t0 = 0; t0 < seq; t0 += chunk_size) {
        int L = chunk_size;
        if (L > seq - t0) L = seq - t0;

        // dt[ti] and inclusive cumulative S_ti = Σ_{r≤ti} dt_r·A (per head).
        float run = 0.0f;
        for (int ti = 0; ti < L; ++ti) {
            float v = softplus_stable_chunk(dt_raw[(t0 + ti) * num_heads + head] + dtb);
            v = fminf(fmaxf(v, dt_min), dt_max);
            dt_loc[ti] = v;
            run += v * a;
            scum_loc[ti] = run;
        }
        float s_end = scum_loc[L - 1];

        // ── outputs (read srow as the still-unmodified h_in) ────────────────
        for (int ti = 0; ti < L; ++ti) {
            const float* Cti = C + (size_t)(t0 + ti) * bd + grp * state_size;
            // inter-chunk state term: exp(S_ti)·(C_ti · h_in)
            float ch = 0.0f;
            for (int n = 0; n < state_size; ++n) {
                ch += Cti[n] * srow[n];
            }
            float yv = expf(scum_loc[ti]) * ch;
            // intra-chunk: Σ_{s≤ti} exp(S_ti − S_s)·dt_s·(C_ti·B_s)·x_s[p]
            for (int s = 0; s <= ti; ++s) {
                float l_ts = expf(scum_loc[ti] - scum_loc[s]);
                const float* Bs = B + (size_t)(t0 + s) * bd + grp * state_size;
                float cb = 0.0f;
                for (int n = 0; n < state_size; ++n) {
                    cb += Cti[n] * Bs[n];
                }
                yv += l_ts * dt_loc[s] * cb * x[(size_t)(t0 + s) * xd + idx];
            }
            float xt = x[(size_t)(t0 + ti) * xd + idx];
            y[(size_t)(t0 + ti) * xd + idx] = yv + Dh * xt;
        }

        // ── carry state to next chunk, in place ─────────────────────────────
        float decay_end = expf(s_end);
        for (int n = 0; n < state_size; ++n) {
            srow[n] *= decay_end; // exp(S_end)·h_in[n]
        }
        for (int s = 0; s < L; ++s) {
            float coef = expf(s_end - scum_loc[s]) * dt_loc[s] * x[(size_t)(t0 + s) * xd + idx];
            const float* Bs = B + (size_t)(t0 + s) * bd + grp * state_size;
            for (int n = 0; n < state_size; ++n) {
                srow[n] += coef * Bs[n];
            }
        }
    }
}
