// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// HIP-graphs-safe twin of `hash_router_normalize_f32`. Reads
// `token_id` from a device buffer instead of a kernarg so the
// captured graph re-reads it on every replay. Host writes the
// per-step `token_id` into a stable `Box<[i32; 1]>` source whose
// captured memcpy node propagates the value to `token_id_buf` at
// graph_launch time.
//
// Identical semantics otherwise (see `hash_router_normalize_f32.hip`).

#include <hip/hip_runtime.h>

#define HASH_ROUTER_BUF_K_MAX 16

extern "C" __global__ void hash_router_normalize_f32_buf(
    const unsigned int* __restrict__ tid2eid,      // [vocab, k]
    const float*        __restrict__ scores,       // [n_exp]
    const int*          __restrict__ token_id_buf, // [1]
    int*                __restrict__ topk_idx,     // [k]
    float*              __restrict__ topk_w,       // [k]
    int n_exp, int k, float route_scale
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const int token_id = token_id_buf[0];
    int ids[HASH_ROUTER_BUF_K_MAX];
    float ws[HASH_ROUTER_BUF_K_MAX];
    float sum = 0.0f;

    const size_t row_off = (size_t)token_id * (size_t)k;
    for (int i = 0; i < k; ++i) {
        int eid = (int)tid2eid[row_off + i];
        if (eid >= n_exp) eid = n_exp - 1;
        ids[i] = eid;
        float s = scores[eid];
        ws[i] = s;
        sum += s;
    }

    if (sum > 0.0f) {
        float inv = route_scale / sum;
        for (int i = 0; i < k; ++i) {
            topk_idx[i] = ids[i];
            topk_w[i] = ws[i] * inv;
        }
    } else {
        for (int i = 0; i < k; ++i) {
            topk_idx[i] = ids[i];
            topk_w[i] = 0.0f;
        }
    }
}
