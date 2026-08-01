// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 mHC α-scaling post-step.
//
// hc_compute_control writes c = (X · W_fn · rsqrt) + base [24].
// Upstream DeepSeek V4's hc_split_sinkhorn applies:
//   pre[h]      = sigmoid(mixes[h]      * hc_scale[0] + hc_base[h])  + eps   // h ∈ [0, hc=4)
//   post[h]     = 2*sigmoid(mixes[h+hc] * hc_scale[1] + hc_base[h+hc])        // h ∈ [0, hc)
//   comb[j,k]   = mixes[2*hc + j*hc + k] * hc_scale[2] + hc_base[2*hc + j*hc + k]
//
// Segments in the 24-element mixes vector (upstream layout):
//   [0..hc=4]:           PRE  (uses hc_scale[0])
//   [hc..2*hc=4..8]:     POST (uses hc_scale[1])
//   [2*hc..mix_hc=8..24]: COMB (uses hc_scale[2])
//
// This kernel post-rescales c so each segment has the right α:
//   c_new[i] = α[seg(i)] · (c[i] - base[i]) + base[i]
//            = α[seg(i)] · c[i] + (1 - α[seg(i)]) · base[i]
//   Algebra check: result = α · (X·W·rsqrt) + base, matching upstream's
//   `mixes * hc_scale + hc_base`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void hc_apply_alpha(
    float*        __restrict__ c,        // [24] fp32 in/out
    const __half* __restrict__ alpha,    // [3]  fp16 (DeepSeek V4: stored as F16 in HFQ)
    const __half* __restrict__ base      // [24] fp16
) {
    if (blockIdx.x != 0) return;
    const int i = threadIdx.x;
    if (i >= 24) return;

    // Upstream layout: [pre(4), post(4), comb(16)] at offsets [0, 4, 8].
    int seg;
    if      (i < 4) seg = 0;  // pre  → α[0]
    else if (i < 8) seg = 1;  // post → α[1]
    else            seg = 2;  // comb → α[2]

    float a = __half2float(alpha[seg]);
    float b = __half2float(base[i]);
    float ci = c[i];
    c[i] = a * ci + (1.0f - a) * b;
}
