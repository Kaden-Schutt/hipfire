// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Chunked (parallel) FP32 gated-DeltaNet — the chunked sibling of
// gated_delta_net_f32_batch_seq.hip. Numerically EQUAL to the sequential
// recurrence (alpha-inside-delta convention), proven by the numpy oracle
// gdn_chunk_parity_3.py (gdn_chunked_f32) at 1.3e-15 max abs diff.
//
// One workgroup processes exactly ONE chunk of C<=CS tokens for ONE head.
// Cross-chunk dependency (chunk ci's S_in = chunk ci-1's S_out) is serialized
// by the host: the Rust wrapper launches one kernel per chunk in a loop,
// advancing the persistent FP32 state in place (same s_f32 buffer the
// sequential kernel uses). The win is the INTRA-chunk parallelism replacing
// the per-token serial inner loop of the sequential kernel.
//
// THREADING: parameterized over blockDim.x (BLK). The HD=128 row-parallel work
// (S_in@k, S_in@q, state carry, fwd-subst columns) is distributed `r = tid;
// r < HD; r += BLK`, so BLK=32 gives 4 rows/lane (1 wave), BLK=128 gives 1
// row/lane (4 waves → latency hiding), BLK=256 adds waves for the CxC sections.
// The inclusive cumsum's __shfl_up only needs lanes 0..C-1 (C<=32, wave 0), so
// it is correct at any BLK. Launched with BLK from HIPFIRE_GDN_BLK (default 128).
//
// Recurrence decomposition (matches gdn_chunked_f32 EXACTLY):
//   G_i      = sum_{m<=i} g_m                       (INCLUSIVE cumsum)
//   D[i,j]   = exp(G_i - G_j)                       (pairwise decay)
//   w_j      = exp(G_j) * (S_in @ k_j)              (inter-chunk pull)
//   A[j,l]   = exp(G_j - G_l) * (k_j . k_l), l<j    (STRICT-lower)
//   L        = diag(beta) * A                       (strict-lower)
//   rhs      = beta ⊙ (v - w)
//   (I + L) delta = rhs                             (UNIT-lower fwd-subst)
//   out_i    = exp(G_i)*(S_in @ q_i)
//              + sum_{j<=i} exp(G_i-G_j)*(q_i.k_j) * delta_j   (lower-INCL-diag)
//   S_out    = exp(G_{C-1})*S_in
//              + sum_j exp(G_{C-1}-G_j) * outer(delta_j, k_j)
//
// LDS budget (CS_MAX=32, HD=128): K and Q chunk tiles are NOT cached in LDS
// (re-read from global), staying ~45.7 KB < 64 KB/workgroup. Resident LDS:
//   5 * CS_MAX vectors (g/G/expG/decay/beta)        =   640 B
//   3 * CS_MAX*CS_MAX matrices (D / L / Mkq)         = 12288 B
//   2 * CS_MAX*HD intermediates (wrhs / delta)       = 32768 B
//   total ≈ 45.7 KB
//
// V1 is F32-only. Q8/EF requant-at-chunk-seams is stage 2 (a separate
// gated_delta_net_q8_chunked.hip mirroring the q8 EF block).

#define HD 128
#define CS_MAX 32

// ─── 1. inclusive prefix sum (Kogge-Stone __shfl_up over C<=32 lanes) ───────
// G_i = sum_{m<=i} g_m. INCLUSIVE (G_i includes g_i). Exclusive shifts every
// decay exponent by one gate and breaks parity. Only lanes 0..C-1 (wave 0,
// since C<=32) participate meaningfully; higher waves shuffle harmlessly and
// never write (the L<C guard).
__device__ static __forceinline__ void inclusive_cumsum(
    const float* __restrict__ g_lds, float* __restrict__ G_lds, int C) {
    const int Lx = threadIdx.x;
    float val = (Lx < C) ? g_lds[Lx] : 0.0f;
    // Kogge-Stone inclusive scan; lane L adds the value from lane L-d when L>=d.
    for (int d = 1; d < 32; d <<= 1) {
        float up = __shfl_up(val, d);
        if (Lx >= d) val += up;
    }
    if (Lx < C) G_lds[Lx] = val;
}

// ─── 2. pairwise decay mask D[i][j] = exp(G_i - G_j) ───────────────────────
// CxC; lanes stride the flattened (i*C+j) index by blockDim.x. expf (precise),
// NOT __expf. D is reused to mask BOTH the strict-lower A term and the
// lower-incl-diag output term (the masking differs; the matrix is shared).
__device__ static __forceinline__ void build_decay_mask(
    const float* __restrict__ G_lds, float* __restrict__ D_lds, int C) {
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    const int CC = C * C;
    for (int idx = tid; idx < CC; idx += blk) {
        int i = idx / C;
        int j = idx - i * C;
        D_lds[idx] = expf(G_lds[i] - G_lds[j]);
    }
}

// ─── 3. unit-lower forward substitution  (I + L) delta = rhs ───────────────
// L is strict-lower CxC (L = diag(beta)*A); rhs is C x HD. UNIT diagonal =>
//   delta_i = rhs_i - sum_{l<i} L[i][l]*delta_l        (NO division).
// Sequential in i (data dependency), parallel over the HD columns: lane owns
// strided columns d in {tid, tid+blk, ...}. __syncthreads() after each i so
// row l is visible before row i>l reads it.
__device__ static __forceinline__ void forward_subst_unit_lower(
    const float* __restrict__ L_lds, const float* __restrict__ rhs,
    float* __restrict__ delta, int C) {
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    for (int i = 0; i < C; i++) {
        for (int d = tid; d < HD; d += blk) {
            float acc = rhs[i * HD + d];
            for (int l = 0; l < i; l++) {
                acc -= L_lds[i * C + l] * delta[l * HD + d];
            }
            delta[i * HD + d] = acc;
        }
        __syncthreads();  // row i visible before row i'>i reads it
    }
}

extern "C" __launch_bounds__(256, 1)
__global__ void gated_delta_net_f32_chunked(
    const float* __restrict__ q,
    const float* __restrict__ k,        // must be L2-normalized per (t,h) by caller
    const float* __restrict__ v,
    const float* __restrict__ gate,     // log-decay logit; alpha = expf(gate)
    const float* __restrict__ beta,     // write gate
    float* __restrict__ s_f32,          // [n_heads, HD, HD] state, in-place advance
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim,                       // unused (HD hardcoded 128); signature parity
    int chunk_size,                     // CS, runtime, <= CS_MAX
    int chunk_index                     // ci, host-driven serialization (c0 = ci*CS)
) {
    (void)head_dim;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int ci = chunk_index;           // chunk index (host-serialized loop)
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    const int stride = n_heads * HD;

    const int c0 = ci * chunk_size;
    if (c0 >= n_tokens) return;
    int c1 = c0 + chunk_size;
    if (c1 > n_tokens) c1 = n_tokens;
    const int C = c1 - c0;                // 1 <= C <= chunk_size <= CS_MAX

    // S_in (read-only this chunk) read directly from global; K/Q chunk tiles
    // re-read from global in the GEMM loops (LDS-budget constraint).
    __shared__ float g_lds[CS_MAX];
    __shared__ float G_lds[CS_MAX];
    __shared__ float expG[CS_MAX];
    __shared__ float decay_to_end[CS_MAX];
    __shared__ float beta_lds[CS_MAX];
    __shared__ float D[CS_MAX * CS_MAX];   // decay mask (shared by A and Mkq)
    __shared__ float L[CS_MAX * CS_MAX];   // KK -> A -> L  (strict-lower)
    __shared__ float Mkq[CS_MAX * CS_MAX]; // QK -> Mkq     (lower-incl-diag)
    __shared__ float wrhs[CS_MAX * HD];    // w then rhs in place
    __shared__ float delta[CS_MAX * HD];

    float* s_head = s_f32 + h * HD * HD;   // S_in[r,c] = s_head[r*HD + c]
    // Per-chunk global bases (row j of the chunk = base + j*stride).
    const float* k_base = k + c0 * stride + h * HD;
    const float* q_base = q + c0 * stride + h * HD;
    const float* v_base = v + c0 * stride + h * HD;

    // ── 0. load gate / beta scalars ────────────────────────────────────────
    if (tid < C) {
        g_lds[tid] = gate[(c0 + tid) * n_heads + h];
        beta_lds[tid] = beta[(c0 + tid) * n_heads + h];
    }
    __syncthreads();

    // ── 1. inclusive cumsum  G_i = sum_{m<=i} g_m ──────────────────────────
    inclusive_cumsum(g_lds, G_lds, C);
    __syncthreads();
    if (tid < C) expG[tid] = expf(G_lds[tid]);
    const float G_C = G_lds[C - 1];        // chunk-final cumsum (math's G_C)
    if (tid < C) decay_to_end[tid] = expf(G_C - G_lds[tid]);
    __syncthreads();

    // ── 2. pairwise decay D[i][j] = exp(G_i - G_j) ─────────────────────────
    build_decay_mask(G_lds, D, C);
    __syncthreads();

    // ── 3. inter-chunk pull  w_j = exp(G_j) * (S_in @ k_j) ─────────────────
    // Sk[j,r] = sum_c S_in[r,c]*k_j[c]; lane owns strided rows r.
    for (int j = 0; j < C; j++) {
        const float* k_j = k_base + j * stride;
        for (int r = tid; r < HD; r += blk) {
            const float* srow = s_head + r * HD;
            float acc = 0.0f;
            #pragma unroll 4
            for (int c = 0; c < HD; c++) acc += srow[c] * k_j[c];
            wrhs[j * HD + r] = expG[j] * acc;
        }
    }
    __syncthreads();

    // ── 4. strict-lower A[j][l] = exp(G_j-G_l)*(k_j.k_l), l<j; L = diag(beta)*A
    {
        const int CC = C * C;
        for (int idx = tid; idx < CC; idx += blk) {
            int j = idx / C;
            int l = idx - j * C;
            float Ljl = 0.0f;
            if (l < j) {  // STRICT lower (no diagonal)
                const float* k_j = k_base + j * stride;
                const float* k_l = k_base + l * stride;
                float kk = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < HD; d++) kk += k_j[d] * k_l[d];
                Ljl = beta_lds[j] * D[j * C + l] * kk;  // beta_j * A[j,l]
            }
            L[idx] = Ljl;
        }
    }
    __syncthreads();

    // ── 5. rhs = beta ⊙ (v - w)  (overwrite w in wrhs in place) ────────────
    for (int j = 0; j < C; j++) {
        const float* v_j = v_base + j * stride;
        float bj = beta_lds[j];
        for (int d = tid; d < HD; d += blk) {
            wrhs[j * HD + d] = bj * (v_j[d] - wrhs[j * HD + d]);
        }
    }
    __syncthreads();

    // ── 6. solve (I + L) delta = rhs  (unit diag, no division) ─────────────
    forward_subst_unit_lower(L, wrhs, delta, C);
    __syncthreads();

    // ── 7. build Mkq = tril(D .* (q_i.k_j), 0)  (lower-INCL-diag, j<=i) ─────
    {
        const int CC = C * C;
        for (int idx = tid; idx < CC; idx += blk) {
            int i = idx / C;
            int j = idx - i * C;
            float m = 0.0f;
            if (j <= i) {  // lower WITH diagonal
                const float* q_i = q_base + i * stride;
                const float* k_j = k_base + j * stride;
                float qk = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < HD; d++) qk += q_i[d] * k_j[d];
                m = D[i * C + j] * qk;
            }
            Mkq[idx] = m;
        }
    }
    __syncthreads();

    // ── 7c. out_i = exp(G_i)*(S_in @ q_i) + sum_{j<=i} Mkq[i,j]*delta_j ─────
    for (int i = 0; i < C; i++) {
        const float* q_i = q_base + i * stride;
        for (int r = tid; r < HD; r += blk) {
            const float* srow = s_head + r * HD;
            float sq = 0.0f;
            #pragma unroll 4
            for (int c = 0; c < HD; c++) sq += srow[c] * q_i[c];
            float acc = expG[i] * sq;
            for (int j = 0; j <= i; j++) acc += Mkq[i * C + j] * delta[j * HD + r];
            output[(c0 + i) * stride + h * HD + r] = acc;
        }
    }
    __syncthreads();

    // ── 8. state carry  S_out = exp(G_C)*S_in + sum_j decay[j]*outer(delta_j,k_j)
    // Each lane owns its own (r,c) cells; reads S_in[r,c] then overwrites it.
    // The outer-product term uses delta + k (not S), so no cross-lane S race.
    const float expGC = expf(G_C);
    for (int r = tid; r < HD; r += blk) {
        float* srow = s_head + r * HD;
        for (int c = 0; c < HD; c++) {
            float acc = expGC * srow[c];
            for (int j = 0; j < C; j++) {
                const float* k_j = k_base + j * stride;
                acc += decay_to_end[j] * delta[j * HD + r] * k_j[c];
            }
            srow[c] = acc;
        }
    }
}
