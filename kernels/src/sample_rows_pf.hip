// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Product-semantics sampler for independent continuous-batch lanes.
//
// One workgroup owns one logit row. Lanes therefore sample concurrently and
// retain independent RNG streams. The penalty phase matches sample_top_p_pf:
// repeat, presence, and frequency penalties are applied over each lane's
// chronological repeat window before the top-k/top-p/min-p draw.
//
// out is lane-major [token, advanced_rng]. Launch one 64-thread block per row
// with 64 * 64 * (sizeof(float) + sizeof(int)) = 32 KiB dynamic LDS.
extern "C" __global__ void sample_rows_pf_f32(
    float* __restrict__ logits,
    const unsigned int* __restrict__ repeat_tokens,
    const int* __restrict__ repeat_lengths,
    const unsigned int* __restrict__ rng_states,
    unsigned int* __restrict__ out,
    int batch_size,
    int vocab_size,
    int repeat_stride,
    float temperature,
    float top_p,
    float repeat_penalty,
    float presence_penalty,
    float frequency_penalty,
    int top_k_req,
    float min_p
) {
    constexpr int TOP_K = 64;
    const int lane = blockIdx.x;
    if (lane >= batch_size) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    float* row = logits + (size_t)lane * vocab_size;
    const unsigned int* history =
        repeat_tokens + (size_t)lane * repeat_stride;
    int repeat_window = repeat_lengths[lane];
    if (repeat_window < 0) repeat_window = 0;
    if (repeat_window > repeat_stride) repeat_window = repeat_stride;

    if (repeat_window > 0 &&
        (repeat_penalty > 1.0f ||
         presence_penalty > 0.0f ||
         frequency_penalty > 0.0f)) {
        const float window_len = (float)repeat_window;
        for (int i = tid; i < repeat_window; i += nthreads) {
            const unsigned int tok = history[i];
            if (vocab_size <= 0 || tok >= (unsigned int)vocab_size) continue;
            bool has_later = false;
            int count = 1;
            for (int j = 0; j < repeat_window; j++) {
                if (j == i) continue;
                if (history[j] == tok) {
                    if (j > i) {
                        has_later = true;
                        break;
                    }
                    count++;
                }
            }
            if (has_later) continue;
            if (repeat_penalty > 1.0f) {
                const float recency = ((float)i + 1.0f) / window_len;
                float effective =
                    powf(repeat_penalty, (float)count * recency);
                if (effective > 1.5f) effective = 1.5f;
                const float value = row[tok];
                row[tok] = value > 0.0f
                    ? value / effective
                    : value * effective;
            }
            if (presence_penalty > 0.0f || frequency_penalty > 0.0f) {
                row[tok] -=
                    frequency_penalty * (float)count + presence_penalty;
            }
        }
    }
    __syncthreads();

    extern __shared__ char smem[];
    float* topk_val = (float*)smem;
    int* topk_idx = (int*)(topk_val + nthreads * TOP_K);

    float my_vals[TOP_K];
    int my_idx[TOP_K];
#pragma unroll
    for (int k = 0; k < TOP_K; k++) {
        my_vals[k] = -1e30f;
        my_idx[k] = 0;
    }
    int min_pos = 0;
    float min_val = -1e30f;
    for (int i = tid; i < vocab_size; i += nthreads) {
        const float value = row[i];
        if (value > min_val) {
            my_vals[min_pos] = value;
            my_idx[min_pos] = i;
            min_val = my_vals[0];
            min_pos = 0;
#pragma unroll
            for (int k = 1; k < TOP_K; k++) {
                if (my_vals[k] < min_val) {
                    min_val = my_vals[k];
                    min_pos = k;
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < TOP_K; k++) {
        topk_val[tid * TOP_K + k] = my_vals[k];
        topk_idx[tid * TOP_K + k] = my_idx[k];
    }
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float* a_val = topk_val + tid * TOP_K;
            int* a_idx = topk_idx + tid * TOP_K;
            float* b_val = topk_val + (tid + stride) * TOP_K;
            int* b_idx = topk_idx + (tid + stride) * TOP_K;
#pragma unroll
            for (int j = 0; j < TOP_K; j++) {
                const float b = b_val[j];
                const int bi = b_idx[j];
                float a_min = a_val[0];
                int a_min_pos = 0;
#pragma unroll
                for (int k = 1; k < TOP_K; k++) {
                    if (a_val[k] < a_min) {
                        a_min = a_val[k];
                        a_min_pos = k;
                    }
                }
                if (b > a_min) {
                    a_val[a_min_pos] = b;
                    a_idx[a_min_pos] = bi;
                }
            }
        }
        __syncthreads();
    }

    if (tid != 0) return;

    float vals[TOP_K];
    int idx[TOP_K];
#pragma unroll
    for (int k = 0; k < TOP_K; k++) {
        vals[k] = topk_val[k];
        idx[k] = topk_idx[k];
    }

    float max_val = vals[0];
    int max_pos = 0;
    for (int k = 1; k < TOP_K; k++) {
        if (vals[k] > max_val) {
            max_val = vals[k];
            max_pos = k;
        }
    }
    unsigned int* lane_out = out + (size_t)lane * 2;
    const unsigned int rng_in = rng_states[lane];
    if (max_val <= -1e29f) {
        lane_out[0] = 0;
        lane_out[1] = rng_in;
        return;
    }
    if (temperature <= 1e-6f) {
        lane_out[0] = (unsigned int)idx[max_pos];
        lane_out[1] = rng_in;
        return;
    }

    const float inv_temp = 1.0f / temperature;
    float probs[TOP_K];
#pragma unroll
    for (int k = 0; k < TOP_K; k++) {
        probs[k] = vals[k] <= -1e29f
            ? 0.0f
            : expf((vals[k] - max_val) * inv_temp);
    }

    int order[TOP_K];
#pragma unroll
    for (int k = 0; k < TOP_K; k++) order[k] = k;
    for (int i = 1; i < TOP_K; i++) {
        const int current = order[i];
        const float current_prob = probs[current];
        int j = i;
        while (j > 0 && probs[order[j - 1]] < current_prob) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = current;
    }

    int cap = TOP_K;
    if (top_k_req > 0 && top_k_req < TOP_K) cap = top_k_req;
    if (min_p > 0.0f) {
        const float floor_p = min_p * probs[order[0]];
        for (int i = 1; i < cap; i++) {
            if (probs[order[i]] < floor_p) {
                cap = i;
                break;
            }
        }
    }
    if (cap < 1) cap = 1;

    float sum = 0.0f;
    for (int i = 0; i < cap; i++) sum += probs[order[i]];
    if (sum <= 0.0f) {
        lane_out[0] = (unsigned int)idx[order[0]];
        lane_out[1] = rng_in;
        return;
    }

    const float p_thresh = top_p * sum;
    float trunc_sum = 0.0f;
    int trunc_k = cap;
    for (int i = 0; i < cap; i++) {
        trunc_sum += probs[order[i]];
        if (trunc_sum >= p_thresh) {
            trunc_k = i + 1;
            break;
        }
    }

    unsigned int rng = rng_in;
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    const float draw =
        (float)rng / 4294967295.0f * trunc_sum;
    float cumulative = 0.0f;
    int picked = trunc_k - 1;
    for (int i = 0; i < trunc_k; i++) {
        cumulative += probs[order[i]];
        if (cumulative >= draw) {
            picked = i;
            break;
        }
    }
    lane_out[0] = (unsigned int)idx[order[picked]];
    lane_out[1] = rng;
}
