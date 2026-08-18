// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// HIP-graphs-safe twin of deepseek4_attn_swa_topk_f32. Reads `n_valid_swa`
// and `n_active_topk` from device buffers instead of i32 kernargs, so
// the captured launch picks up updated positions on each graph replay
// without re-capture.
//
// Body math is byte-identical to deepseek4_attn_swa_topk_f32 — only the
// kernarg layout changes.

#include <hip/hip_runtime.h>
#include <math.h>

#define MAX_TOTAL 1024
#define HEAD_DIM_LDS 512
#define MAX_WARPS 16

extern "C" __global__ __launch_bounds__(512, 1) void deepseek4_attn_swa_topk_f32_buf(
    const float* __restrict__ q,
    const float* __restrict__ swa_k,
    const float* __restrict__ swa_v,
    const float* __restrict__ topk_k,
    const float* __restrict__ topk_v,
    const float* __restrict__ attn_sink,
    float*       __restrict__ attn_out,
    const int*   __restrict__ n_valid_swa_buf,    // [1]
    const int*   __restrict__ n_active_topk_buf,  // [1]
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window
) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    if (h >= n_heads) return;

    const int n_valid_swa   = n_valid_swa_buf[0];
    const int n_active_topk = n_active_topk_buf[0];
    const int n_total = n_valid_swa + n_active_topk;
    const float inv_scale = rsqrtf((float)head_dim);

    __shared__ float q_lds[HEAD_DIM_LDS];
    __shared__ float scores_lds[MAX_TOTAL];
    __shared__ float warp_scratch[MAX_WARPS];

    if (t < head_dim) {
        q_lds[t] = q[h * head_dim + t];
    }
    __syncthreads();

    for (int p = t; p < n_total; p += blockDim.x) {
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
        float s = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            s += q_lds[i] * k_col[i * stride + col];
        }
        scores_lds[p] = s * inv_scale;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int p = t; p < n_total; p += blockDim.x) {
        local_max = fmaxf(local_max, scores_lds[p]);
    }
    if (t == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, off));
    }
    const int warp_id = t >> 5;
    const int lane_id = t & 31;
    if (lane_id == 0) {
        warp_scratch[warp_id] = local_max;
    }
    __syncthreads();

    float max_score;
    if (warp_id == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        float v = (lane_id < n_warps) ? warp_scratch[lane_id] : -1e30f;
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_down(v, off));
        }
        if (lane_id == 0) {
            warp_scratch[0] = v;
        }
    }
    __syncthreads();
    max_score = warp_scratch[0];

    // Phase 4a: parallel exp into scores_lds.
    //
    // Phase 4 history mirrors deepseek4_attn_swa_topk_batched:
    //   v1: __shfl_down reduction — non-deterministic FP order (commit 734e371).
    //   v2: thread-0 serial sum — deterministic but serial ~n_total adds.
    //   v3 (THIS): per-thread slice → __shfl broadcast warp sum (literal src) →
    //   thread-0 cross-warp sum. ~47 deterministic adds vs v2's ~n_total.
    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] = expf(scores_lds[p] - max_score);
    }
    __syncthreads();

    // Phase 4b: per-thread partial sum in fixed stride order.
    float thread_sum = 0.0f;
    for (int p = t; p < n_total; p += blockDim.x) {
        thread_sum += scores_lds[p];
    }

    // Phase 4c: warp-wide deterministic sum via __shfl broadcasts in fixed
    // lane order 0..31. Top-of-kernel `if (h >= n_heads) return;` is
    // block-uniform, so all wave lanes participate here.
    const int warp_id4 = t >> 5;
    const int lane_id4 = t & 31;
    float ws = 0.0f;
    #pragma unroll
    for (int src = 0; src < 32; ++src) {
        ws += __shfl(thread_sum, src);
    }
    if (lane_id4 == 0) {
        warp_scratch[warp_id4] = ws;
    }
    __syncthreads();

    // Phase 4d: thread 0 sums warp partials in fixed warp order + attn_sink.
    float sum_exp;
    if (t == 0) {
        float s = expf(attn_sink[h] - max_score);
        const int n_warps_p4 = (blockDim.x + 31) >> 5;
        for (int w = 0; w < n_warps_p4; ++w) {
            s += warp_scratch[w];
        }
        warp_scratch[0] = s;
    }
    __syncthreads();
    sum_exp = warp_scratch[0];

    const float inv_sum = 1.0f / sum_exp;
    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] *= inv_sum;
    }
    __syncthreads();

    if (t >= head_dim) return;
    const int d = t;
    float acc = 0.0f;
    for (int p = 0; p < n_valid_swa; ++p) {
        acc += scores_lds[p] * swa_v[d * swa_window + p];
    }
    for (int p = 0; p < n_active_topk; ++p) {
        acc += scores_lds[n_valid_swa + p] * topk_v[d * topk_window + p];
    }
    attn_out[h * head_dim + d] = acc;
}

// gfx1151 within-lane ILP variant. It retains the baseline 512-thread mapping
// and deterministic softmax reduction, but breaks the two long serial dot
// products into four independent accumulator chains.
extern "C" __global__ __launch_bounds__(512, 1)
void deepseek4_attn_swa_topk_ilp4_f32_buf(
    const float* __restrict__ q,
    const float* __restrict__ swa_k,
    const float* __restrict__ swa_v,
    const float* __restrict__ topk_k,
    const float* __restrict__ topk_v,
    const float* __restrict__ attn_sink,
    float*       __restrict__ attn_out,
    const int*   __restrict__ n_valid_swa_buf,
    const int*   __restrict__ n_active_topk_buf,
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window
) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    if (h >= n_heads) return;

    const int n_valid_swa = n_valid_swa_buf[0];
    const int n_active_topk = n_active_topk_buf[0];
    const int n_total = n_valid_swa + n_active_topk;
    const float inv_scale = rsqrtf((float)head_dim);

    __shared__ float q_lds[HEAD_DIM_LDS];
    __shared__ float scores_lds[MAX_TOTAL];
    __shared__ float warp_scratch[MAX_WARPS];

    if (t < head_dim) {
        q_lds[t] = q[h * head_dim + t];
    }
    __syncthreads();

    for (int p = t; p < n_total; p += blockDim.x) {
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
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int i = 0;
        for (; i + 3 < head_dim; i += 4) {
            sum0 += q_lds[i]     * k_col[i * stride + col];
            sum1 += q_lds[i + 1] * k_col[(i + 1) * stride + col];
            sum2 += q_lds[i + 2] * k_col[(i + 2) * stride + col];
            sum3 += q_lds[i + 3] * k_col[(i + 3) * stride + col];
        }
        for (; i < head_dim; ++i) {
            sum0 += q_lds[i] * k_col[i * stride + col];
        }
        scores_lds[p] = ((sum0 + sum1) + (sum2 + sum3)) * inv_scale;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int p = t; p < n_total; p += blockDim.x) {
        local_max = fmaxf(local_max, scores_lds[p]);
    }
    if (t == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, off));
    }
    const int warp_id = t >> 5;
    const int lane_id = t & 31;
    if (lane_id == 0) {
        warp_scratch[warp_id] = local_max;
    }
    __syncthreads();
    if (warp_id == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        float value = lane_id < n_warps ? warp_scratch[lane_id] : -1e30f;
        for (int off = 16; off > 0; off >>= 1) {
            value = fmaxf(value, __shfl_down(value, off));
        }
        if (lane_id == 0) {
            warp_scratch[0] = value;
        }
    }
    __syncthreads();
    const float max_score = warp_scratch[0];

    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] = expf(scores_lds[p] - max_score);
    }
    __syncthreads();
    float thread_sum = 0.0f;
    for (int p = t; p < n_total; p += blockDim.x) {
        thread_sum += scores_lds[p];
    }
    float warp_sum = 0.0f;
    #pragma unroll
    for (int source = 0; source < 32; ++source) {
        warp_sum += __shfl(thread_sum, source);
    }
    if (lane_id == 0) {
        warp_scratch[warp_id] = warp_sum;
    }
    __syncthreads();
    if (t == 0) {
        float total = expf(attn_sink[h] - max_score);
        const int n_warps = (blockDim.x + 31) >> 5;
        for (int w = 0; w < n_warps; ++w) {
            total += warp_scratch[w];
        }
        warp_scratch[0] = total;
    }
    __syncthreads();
    const float inv_sum = 1.0f / warp_scratch[0];
    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] *= inv_sum;
    }
    __syncthreads();

    if (t >= head_dim) return;
    const int d = t;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int p = 0;
    for (; p + 3 < n_valid_swa; p += 4) {
        const float* values = swa_v + d * swa_window + p;
        acc0 += scores_lds[p]     * values[0];
        acc1 += scores_lds[p + 1] * values[1];
        acc2 += scores_lds[p + 2] * values[2];
        acc3 += scores_lds[p + 3] * values[3];
    }
    for (; p < n_valid_swa; ++p) {
        acc0 += scores_lds[p] * swa_v[d * swa_window + p];
    }
    p = 0;
    for (; p + 3 < n_active_topk; p += 4) {
        const float* values = topk_v + d * topk_window + p;
        const int score = n_valid_swa + p;
        acc0 += scores_lds[score]     * values[0];
        acc1 += scores_lds[score + 1] * values[1];
        acc2 += scores_lds[score + 2] * values[2];
        acc3 += scores_lds[score + 3] * values[3];
    }
    for (; p < n_active_topk; ++p) {
        acc0 += scores_lds[n_valid_swa + p] * topk_v[d * topk_window + p];
    }
    attn_out[h * head_dim + d] = (acc0 + acc1) + (acc2 + acc3);
}

// gfx1151 short-context score-grid variant.  The ILP4 kernel assigns one
// thread to each position and leaves most of its 512-thread block idle when
// n_total is small.  Here every wave processes a different head-dimension
// slice while lanes remain position-contiguous, then wave 0 reduces the 16
// partials per score.  The admitted ILP4 value phase remains unchanged.
extern "C" __global__ __launch_bounds__(512, 1)
void deepseek4_attn_swa_topk_scoregrid_f32_buf(
    const float* __restrict__ q,
    const float* __restrict__ swa_k,
    const float* __restrict__ swa_v,
    const float* __restrict__ topk_k,
    const float* __restrict__ topk_v,
    const float* __restrict__ attn_sink,
    float*       __restrict__ attn_out,
    const int*   __restrict__ n_valid_swa_buf,
    const int*   __restrict__ n_active_topk_buf,
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window
) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    if (h >= n_heads) return;

    const int warp_id = t >> 5;
    const int lane_id = t & 31;
    const int n_valid_swa = n_valid_swa_buf[0];
    const int n_active_topk = n_active_topk_buf[0];
    const int n_total = n_valid_swa + n_active_topk;
    const float inv_scale = rsqrtf((float)head_dim);

    __shared__ float q_lds[HEAD_DIM_LDS];
    __shared__ float scores_lds[MAX_TOTAL];
    __shared__ float warp_scratch[MAX_WARPS];
    __shared__ float score_partials[MAX_WARPS * 32];

    if (t < head_dim) {
        q_lds[t] = q[h * head_dim + t];
    }
    __syncthreads();

    if (n_total <= 64) {
        for (int p_base = 0; p_base < n_total; p_base += 32) {
            const int p = p_base + lane_id;
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            if (p < n_total) {
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
                int d = warp_id;
                for (; d + 48 < head_dim; d += 64) {
                    sum0 += q_lds[d]      * k_col[d * stride + col];
                    sum1 += q_lds[d + 16] * k_col[(d + 16) * stride + col];
                    sum2 += q_lds[d + 32] * k_col[(d + 32) * stride + col];
                    sum3 += q_lds[d + 48] * k_col[(d + 48) * stride + col];
                }
                for (; d < head_dim; d += 16) {
                    sum0 += q_lds[d] * k_col[d * stride + col];
                }
            }
            score_partials[warp_id * 32 + lane_id] =
                (sum0 + sum1) + (sum2 + sum3);
            __syncthreads();
            if (warp_id == 0 && p < n_total) {
                float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
                #pragma unroll
                for (int w = 0; w < MAX_WARPS; w += 4) {
                    r0 += score_partials[(w + 0) * 32 + lane_id];
                    r1 += score_partials[(w + 1) * 32 + lane_id];
                    r2 += score_partials[(w + 2) * 32 + lane_id];
                    r3 += score_partials[(w + 3) * 32 + lane_id];
                }
                scores_lds[p] = ((r0 + r1) + (r2 + r3)) * inv_scale;
            }
            __syncthreads();
        }
    } else {
        for (int p = t; p < n_total; p += blockDim.x) {
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
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            int i = 0;
            for (; i + 3 < head_dim; i += 4) {
                sum0 += q_lds[i]     * k_col[i * stride + col];
                sum1 += q_lds[i + 1] * k_col[(i + 1) * stride + col];
                sum2 += q_lds[i + 2] * k_col[(i + 2) * stride + col];
                sum3 += q_lds[i + 3] * k_col[(i + 3) * stride + col];
            }
            for (; i < head_dim; ++i) {
                sum0 += q_lds[i] * k_col[i * stride + col];
            }
            scores_lds[p] = ((sum0 + sum1) + (sum2 + sum3)) * inv_scale;
        }
        __syncthreads();
    }

    float local_max = -1e30f;
    for (int p = t; p < n_total; p += blockDim.x) {
        local_max = fmaxf(local_max, scores_lds[p]);
    }
    if (t == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, off));
    }
    if (lane_id == 0) {
        warp_scratch[warp_id] = local_max;
    }
    __syncthreads();
    if (warp_id == 0) {
        float value = lane_id < MAX_WARPS ? warp_scratch[lane_id] : -1e30f;
        for (int off = 16; off > 0; off >>= 1) {
            value = fmaxf(value, __shfl_down(value, off));
        }
        if (lane_id == 0) {
            warp_scratch[0] = value;
        }
    }
    __syncthreads();
    const float max_score = warp_scratch[0];

    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] = expf(scores_lds[p] - max_score);
    }
    __syncthreads();
    float thread_sum = 0.0f;
    for (int p = t; p < n_total; p += blockDim.x) {
        thread_sum += scores_lds[p];
    }
    float warp_sum = 0.0f;
    #pragma unroll
    for (int source = 0; source < 32; ++source) {
        warp_sum += __shfl(thread_sum, source);
    }
    if (lane_id == 0) {
        warp_scratch[warp_id] = warp_sum;
    }
    __syncthreads();
    if (t == 0) {
        float total = expf(attn_sink[h] - max_score);
        for (int w = 0; w < MAX_WARPS; ++w) {
            total += warp_scratch[w];
        }
        warp_scratch[0] = total;
    }
    __syncthreads();
    const float inv_sum = 1.0f / warp_scratch[0];
    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] *= inv_sum;
    }
    __syncthreads();

    if (t >= head_dim) return;
    const int d = t;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int p = 0;
    for (; p + 3 < n_valid_swa; p += 4) {
        const float* values = swa_v + d * swa_window + p;
        acc0 += scores_lds[p]     * values[0];
        acc1 += scores_lds[p + 1] * values[1];
        acc2 += scores_lds[p + 2] * values[2];
        acc3 += scores_lds[p + 3] * values[3];
    }
    for (; p < n_valid_swa; ++p) {
        acc0 += scores_lds[p] * swa_v[d * swa_window + p];
    }
    p = 0;
    for (; p + 3 < n_active_topk; p += 4) {
        const float* values = topk_v + d * topk_window + p;
        const int score = n_valid_swa + p;
        acc0 += scores_lds[score]     * values[0];
        acc1 += scores_lds[score + 1] * values[1];
        acc2 += scores_lds[score + 2] * values[2];
        acc3 += scores_lds[score + 3] * values[3];
    }
    for (; p < n_active_topk; ++p) {
        acc0 += scores_lds[n_valid_swa + p] * topk_v[d * topk_window + p];
    }
    attn_out[h * head_dim + d] = (acc0 + acc1) + (acc2 + acc3);
}

// gfx1151 decode mapping. The baseline assigns one thread to each score and
// makes it carry the full 512-element dot-product dependency chain. Its value
// phase assigns one output dimension per lane, which makes adjacent lanes read
// rows separated by swa_window/topk_window. This variant assigns one wave to a
// score (and later one wave to an output dimension), so the matrix reads are
// contiguous across lanes and both dot products are wave-reduced.
extern "C" __global__ __launch_bounds__(256, 2)
void deepseek4_attn_swa_topk_warp_f32_buf(
    const float* __restrict__ q,
    const float* __restrict__ swa_k,
    const float* __restrict__ swa_v,
    const float* __restrict__ topk_k,
    const float* __restrict__ topk_v,
    const float* __restrict__ attn_sink,
    float*       __restrict__ attn_out,
    const int*   __restrict__ n_valid_swa_buf,
    const int*   __restrict__ n_active_topk_buf,
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window
) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    if (h >= n_heads) return;

    const int lane = t & 31;
    const int warp = t >> 5;
    const int n_warps = blockDim.x >> 5;
    const int n_valid_swa = n_valid_swa_buf[0];
    const int n_active_topk = n_active_topk_buf[0];
    const int n_total = n_valid_swa + n_active_topk;
    const float inv_scale = rsqrtf((float)head_dim);

    __shared__ float q_lds[HEAD_DIM_LDS];
    __shared__ float scores_lds[MAX_TOTAL];
    __shared__ float warp_scratch[MAX_WARPS];

    for (int d = t; d < head_dim; d += blockDim.x) {
        q_lds[d] = q[h * head_dim + d];
    }
    __syncthreads();

    for (int p = warp; p < n_total; p += n_warps) {
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
        float partial = 0.0f;
        for (int i = lane; i < head_dim; i += 32) {
            partial += q_lds[i] * k_col[i * stride + col];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down(partial, offset);
        }
        if (lane == 0) {
            scores_lds[p] = partial * inv_scale;
        }
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int p = t; p < n_total; p += blockDim.x) {
        local_max = fmaxf(local_max, scores_lds[p]);
    }
    if (t == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }
    if (lane == 0) {
        warp_scratch[warp] = local_max;
    }
    __syncthreads();
    if (warp == 0) {
        float value = lane < n_warps ? warp_scratch[lane] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_down(value, offset));
        }
        if (lane == 0) {
            warp_scratch[0] = value;
        }
    }
    __syncthreads();
    const float max_score = warp_scratch[0];

    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] = expf(scores_lds[p] - max_score);
    }
    __syncthreads();

    float thread_sum = 0.0f;
    for (int p = t; p < n_total; p += blockDim.x) {
        thread_sum += scores_lds[p];
    }
    float warp_sum = 0.0f;
    #pragma unroll
    for (int source = 0; source < 32; ++source) {
        warp_sum += __shfl(thread_sum, source);
    }
    if (lane == 0) {
        warp_scratch[warp] = warp_sum;
    }
    __syncthreads();
    if (t == 0) {
        float total = expf(attn_sink[h] - max_score);
        for (int w = 0; w < n_warps; ++w) {
            total += warp_scratch[w];
        }
        warp_scratch[0] = total;
    }
    __syncthreads();
    const float inv_sum = 1.0f / warp_scratch[0];
    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] *= inv_sum;
    }
    __syncthreads();

    for (int d = warp; d < head_dim; d += n_warps) {
        float partial = 0.0f;
        for (int p = lane; p < n_valid_swa; p += 32) {
            partial += scores_lds[p] * swa_v[d * swa_window + p];
        }
        for (int p = lane; p < n_active_topk; p += 32) {
            partial += scores_lds[n_valid_swa + p]
                     * topk_v[d * topk_window + p];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down(partial, offset);
        }
        if (lane == 0) {
            attn_out[h * head_dim + d] = partial;
        }
    }
}
