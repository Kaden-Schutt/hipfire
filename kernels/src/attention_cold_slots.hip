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
#include <hip/hip_fp16.h>

#define CHD 256       // head_dim
#define CPL 8         // CHD / 32 elements per lane

__launch_bounds__(32)
extern "C" __global__ void attention_cold_slots(
    const float* __restrict__ Q,   // [n_heads * 256]
    const float* __restrict__ K,   // [n_kv_heads * n_slots * 256]
    const float* __restrict__ V,   // [n_kv_heads * n_slots * 256]
    float*       __restrict__ OUT, // [n_heads * 256] (normalized)
    float*       __restrict__ M_OUT, // [n_heads] flash max (or null) — for tier merge
    float*       __restrict__ L_OUT, // [n_heads] flash denom (or null)
    int n_heads, int n_kv_heads, int n_slots, float scale,
    // K/V memory layout per kv-head. 0 = SLOT-major: [n_kv_heads × slot_stride × 256],
    // element (s,d) at (kv*slot_stride+s)*256 + d (contiguous head vec per slot).
    // 1 = CHANNEL-major __half: [n_kv_heads × 256 × slot_stride], element (s,d) at
    // kv*256*slot_stride + d*slot_stride + s — the kvarn_dequant_tile output layout
    // (which emits f16), so the cold tier reads straight off its dequantized 4-bit
    // records with no transpose and no f32 widening. K/V are reinterpreted as
    // const __half* in this mode (Q stays f32). slot-major (0) keeps K/V f32.
    int kv_layout,
    // Per-kv-head slot row stride. Attend over the first `n_slots` slots but index
    // with this stride (= the padded tile width n_slots_padded), so an odd valid
    // count can read an even-padded KVarN tile without the pad slot polluting the
    // softmax. Pass 0 to default to n_slots (dense, no padding).
    int slot_stride,
    // Optional per-slot attention-mass accumulator [n_slots] (or null). When set,
    // a second pass adds this query's normalized attention weight to MASS_OUT[s]
    // (atomic, summed over q-heads). The hot tier accumulates this across decode
    // steps → a real CASK-style per-token importance for cold compaction. Reading
    // it costs one extra K pass; pass null on the cold read (no accumulation).
    float* __restrict__ MASS_OUT
) {
    const int sstride = (slot_stride > 0) ? slot_stride : n_slots;
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

    const long long slot_base = (long long)kv * sstride * CHD + lane * CPL; // slot-major
    const long long ch_base   = (long long)kv * CHD * sstride + lane * CPL * (long long)sstride; // channel-major
    for (int s = 0; s < n_slots; ++s) {
        float kk[CPL], vv[CPL];
        if (kv_layout == 0) {
            const float* k = K + slot_base + (long long)s * CHD;
            const float* v = V + slot_base + (long long)s * CHD;
            #pragma unroll
            for (int i = 0; i < CPL; ++i) { kk[i] = k[i]; vv[i] = v[i]; }
        } else {
            // channel-major f16: dim (lane*CPL+i) of slot s is at ch_base + i*sstride + s.
            const __half* k = (const __half*)K + ch_base + s;
            const __half* v = (const __half*)V + ch_base + s;
            #pragma unroll
            for (int i = 0; i < CPL; ++i) {
                kk[i] = __half2float(k[(long long)i * sstride]);
                vv[i] = __half2float(v[(long long)i * sstride]);
            }
        }
        float d = 0.0f;
        #pragma unroll
        for (int i = 0; i < CPL; ++i) d += q[i] * kk[i];
        // butterfly all-reduce across the 32-lane wave → every lane holds the sum.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) d += __shfl_xor(d, off);
        d *= scale;

        const float m_new = fmaxf(m, d);
        const float corr = __expf(m - m_new);   // <= 1
        const float p = __expf(d - m_new);
        l = l * corr + p;
        #pragma unroll
        for (int i = 0; i < CPL; ++i) acc[i] = acc[i] * corr + p * vv[i];
        m = m_new;
    }

    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    float* oh = OUT + (long long)hq * CHD + lane * CPL;
    #pragma unroll
    for (int i = 0; i < CPL; ++i) oh[i] = acc[i] * inv;

    // Flash partials (wave-uniform: every lane holds the same m,l) — lane 0 emits.
    // Lets a caller merge this tier with the hot KVarN flash via online softmax:
    //   m = max(m_hot, m_cold); w_t = l_t·exp(m_t - m); out = Σ w_t·out_t / Σ w_t.
    if (lane == 0) {
        if (M_OUT) M_OUT[hq] = m;
        if (L_OUT) L_OUT[hq] = l;
    }

    // Optional attention-mass accumulation: second pass recomputes each slot's
    // logit and adds its normalized weight exp(d_s - m)/l to MASS_OUT[s] (atomic,
    // so all q-heads sum into the same per-token mass). Only the hot tier passes
    // MASS_OUT; slot-major f32 only (the hot ring is f32).
    if (MASS_OUT) {
        for (int s = 0; s < n_slots; ++s) {
            const float* k = K + slot_base + (long long)s * CHD;
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < CPL; ++i) d += q[i] * k[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor(d, off);
            d *= scale;
            if (lane == 0) {
                atomicAdd(&MASS_OUT[s], __expf(d - m) * inv);
            }
        }
    }
}
