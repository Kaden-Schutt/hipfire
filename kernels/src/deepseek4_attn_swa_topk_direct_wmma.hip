// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Head-batched f16-WMMA port of deepseek4_attn_swa_topk_direct_batched_f32.
// gfx11/RDNA3.5 (gfx1151). Validated approach: see bench_dsa_wmma.{hip,rs}.
//
// Computes the SAME joint-softmax DSA attention (SWA window + direct top-K
// from the compressed KV cache + attention sink), but batches HG=16 heads of
// one query position per block so the per-head score/output GEMVs become
// 16×N WMMA GEMMs that read the SHARED K/V once per head-group.
//
//   S[h,p] = inv_scale · Σ_d Q[h,d]·K[p,d]            (score GEMM, k-dim=d)
//   P      = softmax_p( S , sink )                     (two-pass, in LDS)
//   O[h,d] = Σ_p P[h,p]·V[p,d]                         (output GEMM, k-dim=p)
//
// K=V tied (MLA latent): SWA keys come from swa_kv[b, d, key] (d-major,
// key-contiguous); top-K keys from kv_cache[idx, d] (row-major) via topk_idx.
// The sink contributes expf(sink-max) to the denominator only (no V).
//
// Determinism: the softmax max/sum are per-row serial fixed-order (lanes
// 0..15 each own one head row) → run-to-run reproducible. WMMA accumulates
// deterministically. Only the f16 precision of the matmuls differs from f32.
//
// Grid: [H/16, batch]   Block: [NWARPS·32] (multi-warp: the NWARPS warps split
// the score n-tiles and output d-tiles, sharing the LDS score buffer so the
// shared K/V is read once per head-group). Dynamic LDS: q_lds[16·D] f16 +
// s_lds[16·n_pad] f32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define HG 16

extern "C" __global__ __launch_bounds__(256, 1)
void deepseek4_attn_swa_topk_direct_wmma(
    const float* __restrict__ q,                 // [B, H, D]
    const float* __restrict__ swa_kv,            // [B, D, swa_window]  (K=V)
    const float* __restrict__ kv_cache,          // [n_compressed, D]   (K=V)
    const int*   __restrict__ topk_idx,          // [B, topk_window]
    const float* __restrict__ attn_sink,         // [H]
    const int*   __restrict__ n_valid_swa_arr,   // [B]
    const int*   __restrict__ n_active_topk_arr, // [B]
    float*       __restrict__ attn_out,          // [B, H, D]
    int H, int D, int swa_window, int topk_window, int n_compressed, int B
) {
    const int hg      = blockIdx.x;
    const int b       = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int n_warps = blockDim.x >> 5;
    const int lane    = tid & 31;     // within-warp 0..31
    const int l16     = lane & 15;
    if (hg * HG >= H || b >= B) return;

    const int n_valid  = n_valid_swa_arr[b];
    const int n_active = n_active_topk_arr[b];
    const int n_total  = n_valid + n_active;
    const int n_pad    = ((n_total + 15) / 16) * 16;
    const float inv_scale = rsqrtf((float)D);

    extern __shared__ char lds_raw[];
    _Float16* q_lds = (_Float16*)lds_raw;          // [16 * D]
    float*    s_lds = (float*)(q_lds + HG * D);    // [16 * n_pad]
    __shared__ float inv_sum_lds[HG];

    const size_t q_base   = (size_t)(b * H + hg * HG) * D;
    const size_t swa_base = (size_t)b * D * swa_window;
    const int*   topk_row = topk_idx + (size_t)b * topk_window;

    // Stage Q[16 heads × D] → f16 LDS.
    for (int idx = tid; idx < HG * D; idx += blockDim.x)
        q_lds[idx] = (_Float16)q[q_base + idx];
    __syncthreads();

    // ── Score GEMM: S[16, n_pad] = Q @ K^T  (k-dim = d; lane = key).
    //    The NWARPS warps split the n-tiles. ──
    for (int nt = warp_id; nt < n_pad / 16; nt += n_warps) {
        const int key = nt * 16 + l16;
        int ktype = 2;          // 0=SWA, 1=topK, 2=pad/invalid
        int tk_idx = -1;
        if (key < n_valid) {
            ktype = 0;
        } else if (key < n_total) {
            tk_idx = topk_row[key - n_valid];
            ktype = (tk_idx >= 0 && tk_idx < n_compressed) ? 1 : 2;
        }
        float8_t acc = {0,0,0,0,0,0,0,0};
        for (int kt = 0; kt < D / 16; ++kt) {
            half16_t a, bb;
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = q_lds[l16 * D + kt * 16 + i];
            if (ktype == 0) {            // SWA: swa_kv[b, kt*16+i, key] (stride window)
                #pragma unroll
                for (int i = 0; i < 16; ++i)
                    bb[i] = (_Float16)swa_kv[swa_base + (size_t)(kt * 16 + i) * swa_window + key];
            } else if (ktype == 1) {     // topK: kv_cache[idx, kt*16+i] (contiguous)
                #pragma unroll
                for (int i = 0; i < 16; ++i)
                    bb[i] = (_Float16)kv_cache[(size_t)tk_idx * D + kt * 16 + i];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; ++i) bb[i] = (_Float16)0.0f;
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc);
        }
        // acc[j] = S[head 2j+(lane>>4)][key l16]
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int head = 2 * j + (lane >> 4);
            s_lds[head * n_pad + key] = (ktype != 2) ? acc[j] * inv_scale : -1e30f;
        }
    }
    __syncthreads();

    // ── Softmax per head row (warp 0 lanes 0..15, serial fixed-order, sink) ──
    if (tid < 16) {
        const int head = tid;
        const float sink = attn_sink[hg * HG + head];
        float* row = s_lds + head * n_pad;
        float m = -1e30f;
        for (int p = 0; p < n_total; ++p) m = fmaxf(m, row[p]);
        m = fmaxf(m, sink);
        float sum = 0.0f;
        for (int p = 0; p < n_total; ++p) { float e = expf(row[p] - m); row[p] = e; sum += e; }
        for (int p = n_total; p < n_pad; ++p) row[p] = 0.0f;   // pad → 0 prob
        sum += expf(sink - m);
        inv_sum_lds[head] = 1.0f / sum;
    }
    __syncthreads();

    // ── Output GEMM: O[16, D] = P @ V  (k-dim = key; lane = d).
    //    The NWARPS warps split the d-tiles. ──
    for (int dt = warp_id; dt < D / 16; dt += n_warps) {
        const float invs = inv_sum_lds[l16];
        float8_t acc = {0,0,0,0,0,0,0,0};
        for (int kt = 0; kt < n_pad / 16; ++kt) {
            half16_t a, bb;
            #pragma unroll
            for (int i = 0; i < 16; ++i)
                a[i] = (_Float16)(s_lds[l16 * n_pad + kt * 16 + i] * invs);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int key = kt * 16 + i;
                float vv = 0.0f;
                if (key < n_valid) {
                    vv = swa_kv[swa_base + (size_t)(dt * 16 + l16) * swa_window + key];
                } else if (key < n_total) {
                    const int idx = topk_row[key - n_valid];
                    if (idx >= 0 && idx < n_compressed)
                        vv = kv_cache[(size_t)idx * D + dt * 16 + l16];
                }
                bb[i] = (_Float16)vv;
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc);
        }
        // acc[j] = O[head 2j+(lane>>4)][d dt*16+l16]
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int head = 2 * j + (lane >> 4);
            const int d = dt * 16 + l16;
            attn_out[(size_t)(b * H + hg * HG + head) * D + d] = acc[j];
        }
    }
}
