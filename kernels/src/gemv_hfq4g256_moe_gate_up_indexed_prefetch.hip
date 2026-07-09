// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// ── RESEARCH ARTIFACT — CERTIFIED DEAD (do not enable in production) ──────────
// S_PREFETCH_DATA variants of the indexed MoE gate_up decode GEMV. gfx12/RDNA4
// only, opt-in via HIPFIRE_GFX1201_MOE_PREFETCH ∈ {1,2,3,4}; default OFF. Byte-
// identical math to gemv_hfq4g256_moe_gate_up_indexed.hip (prefetch is a pure
// hint), so token-id parity holds. Kept as a reproducible same-daemon A/B lever
// behind the falsification below.
//
// VERDICT (2026-07-09, ab_certify_v2p.sh vs f33ae7e9, gfx1201/R9700, a3b mq4r,
// kv q8, adaptive-sampled): DEAD. gate_up next-quad len0 = +0.16%, f=0.611
// (noise). Sibling probes on the same card: qkvza next-quad len0 = DEAD −0.26%
// (f=0.625); moe_down next-quad len0 = DEAD +0.46% (f=0.635). All token-id
// exact, all var_coh=OK.
//
// WHY (harness profile_feedback, profile_standard PMC): the a3b batch-1 decode
// GEMVs are NOT latency-exposed — they run at 48–72% achieved occupancy, so
// thread-level parallelism (many concurrent waves) ALREADY hides the weight-load
// latency; the hypothesised "83–94% exposed DEP_WAIT" does not hold in-model.
// s_prefetch_data only ADDS VGPR pressure (gate_up 72→96 VGPR), which DROPS
// occupancy (gate_up 71.6%→51.0%) and thus HURTS the very TLP that was hiding
// the latency. Weights are also 61–76% L2-resident, so little DRAM latency
// remains to overlap. Net: zero-to-negative. See the agent-memory note
// project_sprefetch_gfx1201_decode_gemv_falsified_2026_07_09.
//
// Encoding (verified via clang -S, gfx1201):
//   s_prefetch_data s[base:base+1], <byte-offset>, null, <LEN>
//   base = wave-uniform SGPR pair (gate_ptr/up_ptr are uniform → legal);
//   LEN = immediate cache-line count (0..31; one RDNA4 line = 128 B).
//
// Variants (HIPFIRE_GFX1201_MOE_PREFETCH selector):
//   1 (_pf1): NEXT quad (g+4), gate+up, LEN=0  — the certified DEAD idiom.
//   2 (_pf2): NEXT quad (g+4), gate+up, LEN=5  — prefetch whole quad (~5 lines).
//   3 (_pf3): 2 quads ahead (g+8), gate+up, LEN=5 — deeper.
//   4 (_pf0): NEXT quad (g+4), gate+up, LEN=2  — light.

#define MOE_GATE_UP_PREFETCH_BODY(FN, PF_DIST_GROUPS, PF_LEN)                 \
extern "C" __global__ void FN(                                                \
    const unsigned long long* __restrict__ expert_ptrs,                       \
    const int* __restrict__ topk_indices,                                     \
    const float* __restrict__ x,                                              \
    float* __restrict__ y_gate,                                               \
    float* __restrict__ y_up,                                                 \
    int M, int K                                                              \
) {                                                                           \
    const int mi = M >> 1;                                                     \
    const int row = blockIdx.x;                                               \
    if (row >= mi) return;                                                     \
    const int krank = blockIdx.y;                                             \
    const int tid = threadIdx.x;                                              \
    const int expert_id = topk_indices[krank];                                \
    const char* __restrict__ A =                                              \
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);                \
    const int groups_per_row = K / 256;                                       \
    const char* gate_ptr = A + (long long)row * groups_per_row * 136;         \
    const char* up_ptr = A + (long long)(row + mi) * groups_per_row * 136;    \
    const int boff = tid * 4;                                                  \
    float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;             \
    float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;             \
    const int quads = groups_per_row >> 2;                                    \
    const int tail = groups_per_row & 3;                                      \
    for (int q = 0; q < quads; q++) {                                         \
        const int g = q << 2;                                                  \
        const int base0 = g * 256 + tid * 8;                                  \
        { const int gpf = g + (PF_DIST_GROUPS);                               \
          if (gpf < groups_per_row) {                                         \
            __builtin_amdgcn_s_prefetch_data(                                 \
                (const void*)(gate_ptr + (long long)gpf * 136), (PF_LEN));    \
            __builtin_amdgcn_s_prefetch_data(                                 \
                (const void*)(up_ptr + (long long)gpf * 136), (PF_LEN));      \
          } }                                                                 \
        { LOAD_X8(base0);       DOG_X8(gate_ptr + g * 136, gacc0);            \
                                DOG_X8(up_ptr + g * 136, uacc0); }            \
        { LOAD_X8(base0 + 256); DOG_X8(gate_ptr + (g + 1) * 136, gacc1);      \
                                DOG_X8(up_ptr + (g + 1) * 136, uacc1); }      \
        { LOAD_X8(base0 + 512); DOG_X8(gate_ptr + (g + 2) * 136, gacc2);      \
                                DOG_X8(up_ptr + (g + 2) * 136, uacc2); }      \
        { LOAD_X8(base0 + 768); DOG_X8(gate_ptr + (g + 3) * 136, gacc3);      \
                                DOG_X8(up_ptr + (g + 3) * 136, uacc3); }      \
    }                                                                         \
    if (tail >= 1) { const int g = quads << 2; int base = g * 256 + tid * 8;  \
        LOAD_X8(base); DOG_X8(gate_ptr + g * 136, gacc0);                     \
        DOG_X8(up_ptr + g * 136, uacc0); }                                    \
    if (tail >= 2) { const int g = (quads << 2) + 1; int base = g*256+tid*8;  \
        LOAD_X8(base); DOG_X8(gate_ptr + g * 136, gacc1);                     \
        DOG_X8(up_ptr + g * 136, uacc1); }                                    \
    if (tail >= 3) { const int g = (quads << 2) + 2; int base = g*256+tid*8;  \
        LOAD_X8(base); DOG_X8(gate_ptr + g * 136, gacc2);                     \
        DOG_X8(up_ptr + g * 136, uacc2); }                                    \
    float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);                           \
    float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);                           \
    for (int offset = 16; offset > 0; offset >>= 1) gacc += __shfl_down(gacc, offset); \
    for (int offset = 16; offset > 0; offset >>= 1) uacc += __shfl_down(uacc, offset); \
    if (tid == 0) { y_gate[krank * mi + row] = gacc; y_up[krank * mi + row] = uacc; } \
}

#define LOAD_X8(b) \
    const float x0 = x[(b) + 0]; \
    const float x1 = x[(b) + 1]; \
    const float x2 = x[(b) + 2]; \
    const float x3 = x[(b) + 3]; \
    const float x4 = x[(b) + 4]; \
    const float x5 = x[(b) + 5]; \
    const float x6 = x[(b) + 6]; \
    const float x7 = x[(b) + 7]

#define DOG_X8(gp, a) do { \
    float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp)); \
    float zp = __builtin_bit_cast(float, *(const unsigned int*)((gp) + 4)); \
    unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
    (a) += (sc * (float)((pk) & 0xFu)         + zp) * x0 \
         + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x1 \
         + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x2 \
         + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x3 \
         + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x4 \
         + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x5 \
         + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x6 \
         + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x7; \
} while (0)

MOE_GATE_UP_PREFETCH_BODY(gemv_hfq4g256_moe_gate_up_k8_indexed_pf1, 4, 0)
MOE_GATE_UP_PREFETCH_BODY(gemv_hfq4g256_moe_gate_up_k8_indexed_pf2, 4, 5)
MOE_GATE_UP_PREFETCH_BODY(gemv_hfq4g256_moe_gate_up_k8_indexed_pf3, 8, 5)
MOE_GATE_UP_PREFETCH_BODY(gemv_hfq4g256_moe_gate_up_k8_indexed_pf0, 4, 2)

#undef DOG_X8
#undef LOAD_X8
#undef MOE_GATE_UP_PREFETCH_BODY
