// SPDX-License-Identifier: Apache-2.0
// hipfire — deferred-hierarchical KV compression, Phase 2b: COLD-SLOT attention.
//
// Decode-time attention of ONE query over the compacted COLD tier's merged slots
// (all visible — the cold tier holds only tokens older than the hot window, so no
// causal masking within it). GQA. The slots are few (compaction reduces the key
// count), so this is cheap. ZERO LDS — one wave (32 lanes) per q-head, online
// softmax in registers, __shfl_xor butterfly all-reduce — the gfx1103-safe pattern
// (no shared memory → no LDS page-fault/MES-hang hazard). head_dim fixed at 256
// (qwen3.5): 8 elements per lane.
//
// Parity oracle: hipfire_kvquant::ColdTier::two_tier_attend with n_hot = 0.

#include <hip/hip_runtime.h>

#define CHD 256       // head_dim
#define CPL 8         // CHD / 32 elements per lane

__launch_bounds__(32)
extern "C" __global__ void attention_cold_slots(
    const float* __restrict__ Q,   // [n_heads * 256]
    const float* __restrict__ K,   // [n_kv_heads * n_slots * 256]
    const float* __restrict__ V,   // [n_kv_heads * n_slots * 256]
    float*       __restrict__ OUT, // [n_heads * 256]
    int n_heads, int n_kv_heads, int n_slots, float scale
) {
    const int hq = blockIdx.x;
    if (hq >= n_heads) return;
    const int lane = threadIdx.x;          // 0..31
    const int kv = hq / (n_heads / n_kv_heads);

    // Load this lane's 8 query elements.
    float q[CPL];
    const float* qh = Q + (long long)hq * CHD + lane * CPL;
    #pragma unroll
    for (int i = 0; i < CPL; ++i) q[i] = qh[i];

    float m = -1e30f, l = 0.0f;
    float acc[CPL];
    #pragma unroll
    for (int i = 0; i < CPL; ++i) acc[i] = 0.0f;

    const long long kv_base = (long long)kv * n_slots * CHD + lane * CPL;
    for (int s = 0; s < n_slots; ++s) {
        const float* k = K + kv_base + (long long)s * CHD;
        float d = 0.0f;
        #pragma unroll
        for (int i = 0; i < CPL; ++i) d += q[i] * k[i];
        // butterfly all-reduce across the 32-lane wave → every lane holds the sum.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) d += __shfl_xor(d, off);
        d *= scale;

        const float m_new = fmaxf(m, d);
        const float corr = __expf(m - m_new);   // <= 1
        const float p = __expf(d - m_new);
        l = l * corr + p;
        const float* v = V + kv_base + (long long)s * CHD;
        #pragma unroll
        for (int i = 0; i < CPL; ++i) acc[i] = acc[i] * corr + p * v[i];
        m = m_new;
    }

    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    float* oh = OUT + (long long)hq * CHD + lane * CPL;
    #pragma unroll
    for (int i = 0; i < CPL; ++i) oh[i] = acc[i] * inv;
}
