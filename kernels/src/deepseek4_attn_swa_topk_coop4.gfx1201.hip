// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1201 cooperative DS4 compressed-attention screen.
//
// Four 128-thread workgroups cooperate on each head.  This keeps the launch
// count at one while exposing 4x as many resident blocks as the 512-thread
// reference.  The score, softmax, and value arithmetic orders intentionally
// reproduce deepseek4_attn_swa_topk_f32_buf.

#include <hip/hip_runtime.h>
#include <math.h>

#define COOP4_SEGMENTS 4
#define COOP4_THREADS 128
#define COOP4_MAX_TOTAL 1024
#define COOP4_VIRTUAL_THREADS 512
#define COOP4_SCRATCH_STRIDE 1540

extern "C" __global__ __launch_bounds__(COOP4_THREADS, 2)
void deepseek4_attn_swa_topk_coop4_gfx1201(
    const float* __restrict__ q,
    const float* __restrict__ swa_k,
    const float* __restrict__ swa_v,
    const float* __restrict__ topk_k,
    const float* __restrict__ topk_v,
    const float* __restrict__ attn_sink,
    float*       __restrict__ attn_out,
    const int*   __restrict__ n_valid_swa_buf,
    const int*   __restrict__ n_active_topk_buf,
    float*       __restrict__ scratch,
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window
) {
    const int h = blockIdx.x;
    const int segment = blockIdx.y;
    const int t = threadIdx.x;
    if (h >= n_heads || segment >= COOP4_SEGMENTS) return;

    const int n_valid_swa = n_valid_swa_buf[0];
    const int n_active_topk = n_active_topk_buf[0];
    const int n_total = n_valid_swa + n_active_topk;
    const float inv_scale = rsqrtf((float)head_dim);

    float* head_scratch = scratch + (long long)h * COOP4_SCRATCH_STRIDE;
    float* scores = head_scratch;
    float* virtual_sums = head_scratch + COOP4_MAX_TOTAL;
    unsigned int* state =
        (unsigned int*)(head_scratch + COOP4_MAX_TOTAL + COOP4_VIRTUAL_THREADS);

    __shared__ float q_lds[512];
    for (int d = t; d < head_dim; d += COOP4_THREADS) {
        q_lds[d] = q[(long long)h * head_dim + d];
    }
    __syncthreads();

    // The reference assigns virtual thread v to positions v and v+512.
    // Mapping virtual threads across four workgroups preserves each score's
    // serial head-dimension accumulation exactly.
    const int virtual_t = segment * COOP4_THREADS + t;
    for (int p = virtual_t; p < n_total; p += COOP4_VIRTUAL_THREADS) {
        const float* k_col;
        int stride;
        int col;
        if (p < n_valid_swa) {
            k_col = swa_k;
            stride = swa_window;
            col = p;
        } else {
            k_col = topk_k;
            stride = topk_window;
            col = p - n_valid_swa;
        }
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sum += q_lds[d] * k_col[(long long)d * stride + col];
        }
        scores[p] = sum * inv_scale;
    }
    __threadfence();
    __syncthreads();
    if (t == 0) {
        const unsigned int prior = atomicAdd(state + 0, 1u);
        if (prior == COOP4_SEGMENTS - 1) {
            __threadfence();
            atomicExch(state + 1, 1u);
        }
    }
    __syncthreads();
    if (t == 0) {
        while (atomicAdd(state + 1, 0u) < 1u) { }
    }
    __syncthreads();

    if (segment == 0) {
        __shared__ float max_score_lds;
        __shared__ float inv_sum_lds;

        if (t == 0) {
            float max_score = attn_sink[h];
            for (int p = 0; p < n_total; ++p) {
                max_score = fmaxf(max_score, scores[p]);
            }
            max_score_lds = max_score;
        }
        __syncthreads();

        for (int p = t; p < n_total; p += COOP4_THREADS) {
            scores[p] = expf(scores[p] - max_score_lds);
        }
        __syncthreads();

        // Recreate the reference's 512 virtual thread sums.  Thread zero then
        // performs the identical lane-ordered 32-term warp sums followed by
        // the identical warp-ordered 16-term accumulation.
        for (int v = t; v < COOP4_VIRTUAL_THREADS; v += COOP4_THREADS) {
            float thread_sum = 0.0f;
            for (int p = v; p < n_total; p += COOP4_VIRTUAL_THREADS) {
                thread_sum += scores[p];
            }
            virtual_sums[v] = thread_sum;
        }
        __syncthreads();
        if (t == 0) {
            float total = expf(attn_sink[h] - max_score_lds);
            for (int warp = 0; warp < 16; ++warp) {
                float warp_sum = 0.0f;
                for (int lane = 0; lane < 32; ++lane) {
                    warp_sum += virtual_sums[warp * 32 + lane];
                }
                total += warp_sum;
            }
            inv_sum_lds = 1.0f / total;
        }
        __syncthreads();
        for (int p = t; p < n_total; p += COOP4_THREADS) {
            scores[p] *= inv_sum_lds;
        }
        __threadfence();
        __syncthreads();
        if (t == 0) {
            atomicExch(state + 1, 2u);
        }
    } else {
        if (t == 0) {
            while (atomicAdd(state + 1, 0u) < 2u) { }
        }
        __syncthreads();
    }

    const int d = segment * COOP4_THREADS + t;
    if (d < head_dim) {
        float acc = 0.0f;
        for (int p = 0; p < n_valid_swa; ++p) {
            acc += scores[p] * swa_v[(long long)d * swa_window + p];
        }
        for (int p = 0; p < n_active_topk; ++p) {
            acc += scores[n_valid_swa + p] * topk_v[(long long)d * topk_window + p];
        }
        attn_out[(long long)h * head_dim + d] = acc;
    }

    __threadfence();
    __syncthreads();
    if (t == 0) {
        const unsigned int prior = atomicAdd(state + 2, 1u);
        if (prior == COOP4_SEGMENTS - 1) {
            __threadfence();
            atomicExch(state + 0, 0u);
            atomicExch(state + 1, 0u);
            atomicExch(state + 2, 0u);
        }
    }
}
