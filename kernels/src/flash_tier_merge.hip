// SPDX-License-Identifier: Apache-2.0
// hipfire — deferred-hierarchical KV compression, Phase 2b: HOT+COLD tier MERGE.
//
// Folds two independently-computed flash-attention tiers into one. Each tier T
// contributes its NORMALIZED output out_T = acc_T / l_T, its running max m_T, and
// its denom l_T (the standard flash partials). The online-softmax combine is:
//
//   m   = max(m_a, m_b)
//   w_a = l_a · exp(m_a - m)         w_b = l_b · exp(m_b - m)
//   out = (w_a · out_a + w_b · out_b) / (w_a + w_b)
//   l   = w_a + w_b                  (combined denom, rebased to the new m)
//
// because w_T · out_T = exp(m_T - m) · acc_T, so the numerator is exactly the
// rebased Σ acc and the denominator the rebased Σ l. Tier "a" is the hot KVarN
// flash (recent window, causal); tier "b" is `attention_cold_slots` (compacted
// older tokens, all-visible). The merge is mask-agnostic: each tier already
// applied its own mask when producing its partials. Emitting (m,l) keeps the
// result chainable, e.g. for a future third NPU/CPU-offloaded tier.
//
// ZERO LDS — one wave (32 lanes) per q-head, 8 head_dim elements per lane.
// Parity oracle: a full-softmax attention over the concatenation of both tiers'
// keys (equivalently `attention_cold_slots` over all slots, split then merged).

#include <hip/hip_runtime.h>

#define CHD 256   // head_dim
#define CPL 8     // CHD / 32 elements per lane

__launch_bounds__(32)
extern "C" __global__ void flash_tier_merge(
    const float* __restrict__ OA, const float* __restrict__ MA, const float* __restrict__ LA, // hot
    const float* __restrict__ OB, const float* __restrict__ MB, const float* __restrict__ LB, // cold
    float* __restrict__ OUT,   // [n_heads * 256] merged, normalized
    float* __restrict__ M_OUT, // [n_heads] combined max (or null)
    float* __restrict__ L_OUT, // [n_heads] combined denom (or null)
    int n_heads
) {
    const int hq = blockIdx.x;
    if (hq >= n_heads) return;
    const int lane = threadIdx.x;

    const float ma = MA[hq], mb = MB[hq];
    const float la = LA[hq], lb = LB[hq];
    const float m  = fmaxf(ma, mb);
    const float wa = la * __expf(ma - m);
    const float wb = lb * __expf(mb - m);
    const float wsum = wa + wb;
    const float inv  = (wsum > 0.0f) ? (1.0f / wsum) : 0.0f;

    const float* oa = OA + (long long)hq * CHD + lane * CPL;
    const float* ob = OB + (long long)hq * CHD + lane * CPL;
    float* o = OUT + (long long)hq * CHD + lane * CPL;
    #pragma unroll
    for (int i = 0; i < CPL; ++i) o[i] = (wa * oa[i] + wb * ob[i]) * inv;

    if (lane == 0) {
        if (M_OUT) M_OUT[hq] = m;
        if (L_OUT) L_OUT[hq] = wsum;
    }
}
