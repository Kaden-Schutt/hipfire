// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.
//
// Benchmark-only stable LSE merge of two online-softmax attention partials.
//
// Each record is F32 [m, l, O_unnormalized[HD]] (stride 2+HD). Prefix and
// suffix cover disjoint KV ranges produced by
// attention_q8_0_flash_prefill_wmma_partial_gfx1201. Merge:
//   m  = max(mp, ms)
//   wp = (lp > 0) ? exp(mp - m) : 0
//   ws = (ls > 0) ? exp(ms - m) : 0
//   den = wp*lp + ws*ls
//   O[d] = (wp*Np[d] + ws*Ns[d]) / den   (0 if den == 0)
//
// Geometry: one wave32 per record, eight records per block (blockDim=256),
// zero LDS. Grid x = ceil(B*H / 8).
// Fixed Muse Glimmer verifier shape: B=16, H=32, HD=128.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(256) void attention_partition_merge_f32(
    const float* __restrict__ prefix,
    const float* __restrict__ suffix,
    float* __restrict__ out,
    int batch,
    int n_heads,
    int head_dim
) {
    const int tid  = (int)threadIdx.x;
    const int wave = tid >> 5;          // 0..7
    const int lane = tid & 31;
    const int rec  = (int)blockIdx.x * 8 + wave;
    const int n_rec = batch * n_heads;
    if (rec >= n_rec)
        return;

    const int p_stride = 2 + head_dim;
    const float* p = prefix + (long long)rec * p_stride;
    const float* s = suffix + (long long)rec * p_stride;
    float* o = out + (long long)rec * head_dim;

    const float mp = p[0];
    const float lp = p[1];
    const float ms = s[0];
    const float ls = s[1];
    const float m  = fmaxf(mp, ms);
    const float wp = (lp > 0.0f) ? __expf(mp - m) : 0.0f;
    const float ws = (ls > 0.0f) ? __expf(ms - m) : 0.0f;
    const float den = wp * lp + ws * ls;
    const float inv = (den > 0.0f) ? (1.0f / den) : 0.0f;

    for (int d = lane; d < head_dim; d += 32) {
        const float np = p[2 + d];
        const float ns = s[2 + d];
        o[d] = (wp * np + ws * ns) * inv;
    }
}
