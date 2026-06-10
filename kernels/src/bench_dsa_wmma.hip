// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Microbench kernels for the DSA-attention WMMA head-batching study.
// Question being answered: does a head-batched f16-WMMA flash-attention
// match the f32 per-head reference closely enough (precision) and run
// faster (the shared-K/V reuse + WMMA matmul lever)?
//
// Two kernels, both computing dense joint-softmax attention with K/V SHARED
// across heads (the DSA structure — DeepSeek MLA shares compressed KV):
//   O[b,h,d] = softmax_p( scale * sum_d' Q[b,h,d'] K[b,p,d'] ) · V[b,p,d]
//
//   dsa_attn_f32_baseline : one block per (head, batch), block=[D]. Mirrors
//       the production per-head GEMV pattern (the thing we'd replace).
//   dsa_attn_wmma_hb      : one warp per (16-head group, batch). The score
//       and output become 16×N / 16×D WMMA GEMMs (f16), reading the shared
//       K/V once per group instead of once per head.
//
// Shapes: D (head_dim) and N (n_total keys) must be multiples of 16.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

// ───────────────────────── f32 baseline (the current pattern) ─────────────
// Grid: [n_heads, batch]   Block: [D] (one thread per head_dim element).
extern "C" __global__ __launch_bounds__(512, 1) void dsa_attn_f32_baseline(
    const float* __restrict__ Q,   // [B, H, D]
    const float* __restrict__ K,   // [B, N, D]   (shared across heads)
    const float* __restrict__ V,   // [B, N, D]   (shared across heads)
    float*       __restrict__ O,   // [B, H, D]
    int H, int D, int N, int B
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int t = threadIdx.x;
    if (h >= H || b >= B) return;

    const float inv_scale = rsqrtf((float)D);
    extern __shared__ float lds[];
    float* q_lds = lds;          // [D]
    float* s_lds = lds + D;      // [N]

    const size_t q_base = (size_t)(b * H + h) * D;
    const size_t kv_base = (size_t)b * N * D;

    if (t < D) q_lds[t] = Q[q_base + t];
    __syncthreads();

    // scores
    for (int p = t; p < N; p += blockDim.x) {
        float s = 0.0f;
        const float* k_row = K + kv_base + (size_t)p * D;
        for (int i = 0; i < D; ++i) s += q_lds[i] * k_row[i];
        s_lds[p] = s * inv_scale;
    }
    __syncthreads();

    // parallel block reduction (production v3-style), 512 threads / 16 warps
    __shared__ float mx, sm, red[16];
    float lmax = -1e30f;
    for (int p = t; p < N; p += blockDim.x) lmax = fmaxf(lmax, s_lds[p]);
    for (int off = 16; off > 0; off >>= 1) lmax = fmaxf(lmax, __shfl_down(lmax, off));
    if ((t & 31) == 0) red[t >> 5] = lmax;
    __syncthreads();
    if (t == 0) {
        float m = -1e30f;
        const int nw = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nw; ++w) m = fmaxf(m, red[w]);
        mx = m;
    }
    __syncthreads();
    for (int p = t; p < N; p += blockDim.x) s_lds[p] = expf(s_lds[p] - mx);
    __syncthreads();
    float lsum = 0.0f;
    for (int p = t; p < N; p += blockDim.x) lsum += s_lds[p];
    for (int off = 16; off > 0; off >>= 1) lsum += __shfl_down(lsum, off);
    if ((t & 31) == 0) red[t >> 5] = lsum;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        const int nw = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nw; ++w) s += red[w];
        sm = s;
    }
    __syncthreads();
    const float inv_sum = 1.0f / sm;

    // output: thread d accumulates over keys
    if (t < D) {
        const int d = t;
        float acc = 0.0f;
        for (int p = 0; p < N; ++p)
            acc += s_lds[p] * inv_sum * V[kv_base + (size_t)p * D + d];
        O[q_base + d] = acc;
    }
}

// ───────────────────────── head-batched WMMA ──────────────────────────────
// Grid: [H/16, batch]   Block: [32] (1 warp).  Processes 16 heads × 1 batch.
//   Score GEMM:  S[16,N] = Q[16,D] @ K[N,D]^T   (acc over D)
//   Output GEMM: O[16,D] = P[16,N] @ Vt[D,N]^T  (acc over N; Vt = V transposed)
// LDS: S[16*N] f32 + P[16*N] f16.
#define HG 16
extern "C" __global__ __launch_bounds__(32, 8) void dsa_attn_wmma_hb(
    const _Float16* __restrict__ Q,   // [B, H, D]   f16
    const _Float16* __restrict__ K,   // [B, N, D]   f16 (shared)
    const _Float16* __restrict__ Vt,  // [B, D, N]   f16 (V transposed, shared)
    float*          __restrict__ O,   // [B, H, D]   f32
    int H, int D, int N, int B
) {
    const int hg = blockIdx.x;        // head-group (×16)
    const int b  = blockIdx.y;
    const int lane = threadIdx.x;     // 0..31
    const int l16  = lane & 15;
    if (hg * HG >= H || b >= B) return;

    const float inv_scale = rsqrtf((float)D);

    extern __shared__ char lds_raw[];
    float* s_lds = (float*)lds_raw;          // [16 * N] exp-scores
    __shared__ float inv_sum_lds[HG];        // per-head 1/sum

    const size_t q_base  = (size_t)(b * H + hg * HG) * D; // Q[b, hg*16 + m, d]
    const size_t k_base  = (size_t)b * N * D;             // K[b, p, d]
    const size_t vt_base = (size_t)b * D * N;             // Vt[b, d, p]

    // ── Score GEMM: S[16,N] = Q @ K^T ──
    for (int nt = 0; nt < N / 16; ++nt) {
        float8_t acc = {0,0,0,0,0,0,0,0};
        for (int kt = 0; kt < D / 16; ++kt) {
            half16_t a, bb;
            const _Float16* qrow = Q + q_base + (size_t)l16 * D + kt * 16;
            const _Float16* krow = K + k_base + (size_t)(nt * 16 + l16) * D + kt * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) { a[i] = qrow[i]; bb[i] = krow[i]; }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc);
        }
        // acc[j] = S[head 2j+(lane>>4)][key nt*16 + l16]
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int head = 2 * j + (lane >> 4);
            const int key  = nt * 16 + l16;
            s_lds[head * N + key] = acc[j] * inv_scale;
        }
    }
    // (single warp — no cross-lane sync needed for LDS visibility within a wave,
    //  but the WMMA output above is wave-wide; a barrier keeps it simple/safe)
    __builtin_amdgcn_s_barrier();

    // ── Softmax per head row (lanes 0..15 each own one row, fixed order).
    //    Leaves exp-scores in s_lds and 1/sum in inv_sum_lds; the output GEMM
    //    normalizes inline (avoids a separate f16 P buffer → half the LDS). ──
    if (lane < 16) {
        const int head = lane;
        float* row = s_lds + head * N;
        float m = -1e30f;
        for (int p = 0; p < N; ++p) m = fmaxf(m, row[p]);
        float sum = 0.0f;
        for (int p = 0; p < N; ++p) { float e = expf(row[p] - m); row[p] = e; sum += e; }
        inv_sum_lds[head] = 1.0f / sum;
    }
    __builtin_amdgcn_s_barrier();

    // ── Output GEMM: O[16,D] = P @ Vt^T  (P = exp-scores · 1/sum, inline) ──
    for (int dt = 0; dt < D / 16; ++dt) {
        float8_t acc = {0,0,0,0,0,0,0,0};
        const float invs = inv_sum_lds[l16];
        for (int kt = 0; kt < N / 16; ++kt) {
            half16_t a, bb;
            const float*    prow = s_lds + (size_t)l16 * N + kt * 16;
            const _Float16* vrow = Vt + vt_base + (size_t)(dt * 16 + l16) * N + kt * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) { a[i] = (_Float16)(prow[i] * invs); bb[i] = vrow[i]; }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc);
        }
        // acc[j] = O[head 2j+(lane>>4)][d dt*16 + l16]
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int head = 2 * j + (lane >> 4);
            const int d    = dt * 16 + l16;
            O[(size_t)(b * H + hg * HG + head) * D + d] = acc[j];
        }
    }
}
