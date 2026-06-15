// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Head-batched f16-WMMA port of deepseek4_attn_swa_topk_batched_f32 (the
// "gathered" DSA variant — top-K K/V already staged into topk_kv[B,D,topk_win]
// rather than read from the compressed cache via indices). gfx11/RDNA3.5.
//
// Sibling of deepseek4_attn_swa_topk_direct_wmma.hip; identical math and WMMA
// layout, but the top-K keys come from a d-major staged buffer (key-contiguous)
// instead of kv_cache[idx]. Both SWA and top-K are K=V tied. See that file and
// bench_dsa_wmma.{hip,rs} for the derivation/validation.
//
// Grid: [H/16, batch]   Block: [NWARPS·32].  LDS: q_lds[16·D] f16 + s_lds[16·n_pad] f32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define HG 16

extern "C" __global__ __launch_bounds__(256, 1)
void deepseek4_attn_swa_topk_batched_wmma(
    const float* __restrict__ q,                 // [B, H, D]
    const float* __restrict__ swa_kv,            // [B, D, swa_window]  (K=V)
    const float* __restrict__ topk_kv,           // [B, D, topk_window] (K=V)
    const float* __restrict__ attn_sink,         // [H]
    const int*   __restrict__ n_valid_swa_arr,   // [B]
    const int*   __restrict__ n_active_topk_arr, // [B]
    float*       __restrict__ attn_out,          // [B, H, D]
    int H, int D, int swa_window, int topk_window, int B
) {
    const int hg      = blockIdx.x;
    const int b       = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int n_warps = blockDim.x >> 5;
    const int lane    = tid & 31;
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

    const size_t q_base    = (size_t)(b * H + hg * HG) * D;
    const size_t swa_base  = (size_t)b * D * swa_window;
    const size_t topk_base = (size_t)b * D * topk_window;

    for (int idx = tid; idx < HG * D; idx += blockDim.x)
        q_lds[idx] = (_Float16)q[q_base + idx];
    __syncthreads();

    // ── Score GEMM: S[16, n_pad] = Q @ K^T  (k-dim = d; lane = key) ──
    for (int nt = warp_id; nt < n_pad / 16; nt += n_warps) {
        const int key = nt * 16 + l16;
        int ktype = 2;                  // 0=SWA, 1=topK, 2=pad
        int tcol = 0;
        if (key < n_valid) ktype = 0;
        else if (key < n_total) { ktype = 1; tcol = key - n_valid; }
        float8_t acc = {0,0,0,0,0,0,0,0};
        for (int kt = 0; kt < D / 16; ++kt) {
            half16_t a, bb;
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = q_lds[l16 * D + kt * 16 + i];
            if (ktype == 0) {            // SWA: swa_kv[b, kt*16+i, key]
                #pragma unroll
                for (int i = 0; i < 16; ++i)
                    bb[i] = (_Float16)swa_kv[swa_base + (size_t)(kt * 16 + i) * swa_window + key];
            } else if (ktype == 1) {     // topK: topk_kv[b, kt*16+i, tcol]
                #pragma unroll
                for (int i = 0; i < 16; ++i)
                    bb[i] = (_Float16)topk_kv[topk_base + (size_t)(kt * 16 + i) * topk_window + tcol];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; ++i) bb[i] = (_Float16)0.0f;
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc);
        }
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
        for (int p = n_total; p < n_pad; ++p) row[p] = 0.0f;
        sum += expf(sink - m);
        inv_sum_lds[head] = 1.0f / sum;
    }
    __syncthreads();

    // ── Output GEMM: O[16, D] = P @ V  (k-dim = key; lane = d) ──
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
                if (key < n_valid)
                    vv = swa_kv[swa_base + (size_t)(dt * 16 + l16) * swa_window + key];
                else if (key < n_total)
                    vv = topk_kv[topk_base + (size_t)(dt * 16 + l16) * topk_window + (key - n_valid)];
                bb[i] = (_Float16)vv;
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc);
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int head = 2 * j + (lane >> 4);
            const int d = dt * 16 + l16;
            attn_out[(size_t)(b * H + hg * HG + head) * D + d] = acc[j];
        }
    }
}
