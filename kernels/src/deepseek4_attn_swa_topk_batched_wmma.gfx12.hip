// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx12-native head-batched WMMA port of
// deepseek4_attn_swa_topk_batched_f32. The gathered DSA K/V is shared by all
// query heads, so one block processes sixteen heads and reuses each K/V tile.
//
// RDNA4 WMMA differs materially from gfx11: each lane supplies half8 operands,
// the K tile is split by k_grp=(lane>>4), and accumulator element j maps to
// C[8*k_grp+j][lane&15]. This symbol is compiled/routed only for gfx1201.
//
// H need not be divisible by 16. The final group zero-fills inactive Q rows and
// masks its softmax/output writes; this is required by TP3's 24/24/16 split.
//
// Grid: [ceil(H/16), batch]   Block: [8*32].
// LDS: q_lds[16*D] f16 + s_lds[16*n_pad] f32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#define HG 16

extern "C" __global__ __launch_bounds__(256, 1)
void deepseek4_attn_swa_topk_batched_wmma_gfx12(
    const float* __restrict__ q,                 // [B, H, D]
    const float* __restrict__ swa_kv,            // [B, D, swa_window]  (K=V)
    const float* __restrict__ topk_kv,           // [B, D, topk_window] (K=V)
    const float* __restrict__ attn_sink,         // [H]
    const int*   __restrict__ n_valid_swa_arr,   // [B]
    const int*   __restrict__ n_active_topk_arr, // [B]
    float*       __restrict__ attn_out,          // [B, H, D]
    int H, int D, int swa_window, int topk_window, int B
) {
    const int hg       = blockIdx.x;
    const int b        = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int n_warps  = blockDim.x >> 5;
    const int lane     = tid & 31;
    const int lane_row = lane & 15;
    const int k_grp    = lane >> 4;
    const int head0    = hg * HG;
    if (head0 >= H || b >= B) return;

    const int active_heads = min(H - head0, HG);
    const int n_valid  = n_valid_swa_arr[b];
    const int n_active = n_active_topk_arr[b];
    const int n_total  = n_valid + n_active;
    const int n_pad    = ((n_total + 15) / 16) * 16;
    const float inv_scale = rsqrtf((float)D);

    extern __shared__ char lds_raw[];
    _Float16* q_lds = (_Float16*)lds_raw;       // [16, D]
    float* s_lds = (float*)(q_lds + HG * D);    // [16, n_pad]
    __shared__ float inv_sum_lds[HG];

    const size_t q_base    = (size_t)(b * H + head0) * D;
    const size_t swa_base  = (size_t)b * D * swa_window;
    const size_t topk_base = (size_t)b * D * topk_window;

    // Stage Q once for the head group. Do not read beyond the local TP rank's
    // H rows in the masked final group.
    for (int idx = tid; idx < HG * D; idx += blockDim.x) {
        const int local_head = idx / D;
        const int d = idx - local_head * D;
        q_lds[idx] = local_head < active_heads
            ? (_Float16)q[q_base + (size_t)local_head * D + d]
            : (_Float16)0.0f;
    }
    __syncthreads();

    // S[16,n_pad] = Q[16,D] @ K[D,n_pad]. Eight warps split N tiles.
    for (int nt = warp_id; nt < n_pad / 16; nt += n_warps) {
        const int key = nt * 16 + lane_row;
        int ktype = 2; // 0=SWA, 1=topK, 2=pad
        int tcol = 0;
        if (key < n_valid) {
            ktype = 0;
        } else if (key < n_total) {
            ktype = 1;
            tcol = key - n_valid;
        }

        float8_t acc = {0,0,0,0,0,0,0,0};
        for (int kt = 0; kt < D; kt += 16) {
            half8_t a;
            half8_t bb;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int kd = kt + k_grp * 8 + i;
                a[i] = q_lds[lane_row * D + kd];
                if (ktype == 0) {
                    bb[i] = (_Float16)swa_kv[
                        swa_base + (size_t)kd * swa_window + key];
                } else if (ktype == 1) {
                    bb[i] = (_Float16)topk_kv[
                        topk_base + (size_t)kd * topk_window + tcol];
                } else {
                    bb[i] = (_Float16)0.0f;
                }
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc);
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int head = 8 * k_grp + j;
            s_lds[head * n_pad + key] =
                (ktype != 2 && head < active_heads) ? acc[j] * inv_scale : -1e30f;
        }
    }
    __syncthreads();

    // Fixed-order softmax per active head. The attention sink contributes to
    // the denominator only, matching the F32 production kernel.
    if (tid < HG) {
        const int head = tid;
        if (head < active_heads) {
            const float sink = attn_sink[head0 + head];
            float* row = s_lds + head * n_pad;
            float m = -1e30f;
            for (int p = 0; p < n_total; ++p) m = fmaxf(m, row[p]);
            m = fmaxf(m, sink);
            float sum = 0.0f;
            for (int p = 0; p < n_total; ++p) {
                const float e = expf(row[p] - m);
                row[p] = e;
                sum += e;
            }
            for (int p = n_total; p < n_pad; ++p) row[p] = 0.0f;
            sum += expf(sink - m);
            inv_sum_lds[head] = 1.0f / sum;
        } else {
            inv_sum_lds[head] = 0.0f;
        }
    }
    __syncthreads();

    // O[16,D] = P[16,n_pad] @ V[n_pad,D]. Eight warps split D tiles.
    for (int dt = warp_id; dt < D / 16; dt += n_warps) {
        float8_t acc = {0,0,0,0,0,0,0,0};
        for (int kt = 0; kt < n_pad; kt += 16) {
            half8_t a;
            half8_t bb;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int key = kt + k_grp * 8 + i;
                a[i] = (_Float16)(
                    s_lds[lane_row * n_pad + key] * inv_sum_lds[lane_row]);
                float vv = 0.0f;
                if (key < n_valid) {
                    vv = swa_kv[
                        swa_base + (size_t)(dt * 16 + lane_row) * swa_window + key];
                } else if (key < n_total) {
                    vv = topk_kv[
                        topk_base + (size_t)(dt * 16 + lane_row) * topk_window
                        + (key - n_valid)];
                }
                bb[i] = (_Float16)vv;
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc);
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int head = 8 * k_grp + j;
            if (head < active_heads) {
                const int d = dt * 16 + lane_row;
                attn_out[(size_t)(b * H + head0 + head) * D + d] = acc[j];
            }
        }
    }
}
