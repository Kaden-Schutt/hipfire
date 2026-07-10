// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Index-aware fused MoE gate_up GEMV (Phase 2b/2c) — reads expert IDs from
// a device-side topk_indices buffer and the per-expert weight base from a
// device-side expert pointer table. No CPU kernargs depend on the routing
// result, so the whole kernel replays cleanly under hipGraph capture.
//
// Expert ptrs layout: `expert_ptrs` is a raw [n_exp] array of `const char*`
// weight-base pointers (packed as `unsigned long long` slots, 8B each).
// Built once at load time from `ExpertWeights.gate_up.buf.as_ptr()`. The
// kernel reads `expert_ptrs[topk_indices[krank]]` as its weight base.
//
// Output layout matches the non-indexed variant: split into y_gate[K_TOP×MI]
// and y_up[K_TOP×MI], feeding the existing batched silu_mul_rotate.
//
// Baseline on loop/gfx1201_w2: R20c2 hoist_x32 + R29c2 cluster_gu.
// BOD: 11.9% wall, L2~64% (lowest among big GEMVs), mem_busy~55%,
// occ~52%, vgpr=96 — TLP hides latency, NOT MLP-limited.
//
// R35c2_gup_w_preload: after 32-wide x hoist, fully separate the weight
// MEMORY phase from the DOG VALU phase for each group-quad:
//   1) stream-load all 4 gate sc/zp/pk (sequential along gate_ptr)
//   2) stream-load all 4 up sc/zp/pk (sequential along up_ptr)
//   3) then 8 DOG FMAs G0..G3, U0..U3 (same acc slots, same math)
// Baseline DOG_X8 interleaves load+FMA per group; dual_wload (INCONCLUSIVE)
// only dual-issued gate+up per single group. Hypothesis: pure sequential
// weight traffic (no VALU between loads) improves L1/L2 request batching
// on the cold expert path (L2~64%), then a pure FMA burst dual-issues
// against already-resident sc/zp/pk. Value-identical.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x,                         // [K]
    float* __restrict__ y_gate,                          // [K_TOP × MI]
    float* __restrict__ y_up,                            // [K_TOP × MI]
    int M, int K
) {
    const int mi = M >> 1;
    const int row = blockIdx.x;
    if (row >= mi) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    // Read expert_id from device and index into the pointer table.
    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const char* gate_ptr = A + (long long)row * groups_per_row * 136;
    const char* up_ptr = A + (long long)(row + mi) * groups_per_row * 136;
    const int boff = tid * 4;

    float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
    float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    // R21c2_gateup_notail: target A3B mq4r decode has K/256 divisible by 4,
    // so the HFQ4 group tail is cold code for this certify path.
    const int tail = 0;

    #define DOG_X8(sc, zp, pk, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        (a) += ((sc) * (float)((pk) & 0xFu)         + (zp)) * (x0) \
             + ((sc) * (float)(((pk) >> 4) & 0xFu)  + (zp)) * (x1) \
             + ((sc) * (float)(((pk) >> 8) & 0xFu)  + (zp)) * (x2) \
             + ((sc) * (float)(((pk) >> 12) & 0xFu) + (zp)) * (x3) \
             + ((sc) * (float)(((pk) >> 16) & 0xFu) + (zp)) * (x4) \
             + ((sc) * (float)(((pk) >> 20) & 0xFu) + (zp)) * (x5) \
             + ((sc) * (float)(((pk) >> 24) & 0xFu) + (zp)) * (x6) \
             + ((sc) * (float)(((pk) >> 28) & 0xFu) + (zp)) * (x7); \
    } while (0)

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

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        // Hoist full 32-wide x for this group-quad (4 groups × 8) via float4 pairs
        // before any weight load/DOG — same schedule as residual R15c2 / qkvza R17c2.
        const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
        const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);
        const float4 xv2a = *reinterpret_cast<const float4*>(x + base0 + 512);
        const float4 xv2b = *reinterpret_cast<const float4*>(x + base0 + 516);
        const float4 xv3a = *reinterpret_cast<const float4*>(x + base0 + 768);
        const float4 xv3b = *reinterpret_cast<const float4*>(x + base0 + 772);

        // MEMORY phase: sequential gate weights, then sequential up weights.
        const char* gg0 = gate_ptr + g * 136;
        const char* gg1 = gg0 + 136;
        const char* gg2 = gg1 + 136;
        const char* gg3 = gg2 + 136;
        float gsc0 = __builtin_bit_cast(float, *(const unsigned int*)(gg0));
        float gzp0 = __builtin_bit_cast(float, *(const unsigned int*)(gg0 + 4));
        float gsc1 = __builtin_bit_cast(float, *(const unsigned int*)(gg1));
        float gzp1 = __builtin_bit_cast(float, *(const unsigned int*)(gg1 + 4));
        float gsc2 = __builtin_bit_cast(float, *(const unsigned int*)(gg2));
        float gzp2 = __builtin_bit_cast(float, *(const unsigned int*)(gg2 + 4));
        float gsc3 = __builtin_bit_cast(float, *(const unsigned int*)(gg3));
        float gzp3 = __builtin_bit_cast(float, *(const unsigned int*)(gg3 + 4));
        unsigned int gpk0 = *(const unsigned int*)(gg0 + 8 + boff);
        unsigned int gpk1 = *(const unsigned int*)(gg1 + 8 + boff);
        unsigned int gpk2 = *(const unsigned int*)(gg2 + 8 + boff);
        unsigned int gpk3 = *(const unsigned int*)(gg3 + 8 + boff);

        const char* uu0 = up_ptr + g * 136;
        const char* uu1 = uu0 + 136;
        const char* uu2 = uu1 + 136;
        const char* uu3 = uu2 + 136;
        float usc0 = __builtin_bit_cast(float, *(const unsigned int*)(uu0));
        float uzp0 = __builtin_bit_cast(float, *(const unsigned int*)(uu0 + 4));
        float usc1 = __builtin_bit_cast(float, *(const unsigned int*)(uu1));
        float uzp1 = __builtin_bit_cast(float, *(const unsigned int*)(uu1 + 4));
        float usc2 = __builtin_bit_cast(float, *(const unsigned int*)(uu2));
        float uzp2 = __builtin_bit_cast(float, *(const unsigned int*)(uu2 + 4));
        float usc3 = __builtin_bit_cast(float, *(const unsigned int*)(uu3));
        float uzp3 = __builtin_bit_cast(float, *(const unsigned int*)(uu3 + 4));
        unsigned int upk0 = *(const unsigned int*)(uu0 + 8 + boff);
        unsigned int upk1 = *(const unsigned int*)(uu1 + 8 + boff);
        unsigned int upk2 = *(const unsigned int*)(uu2 + 8 + boff);
        unsigned int upk3 = *(const unsigned int*)(uu3 + 8 + boff);

        // VALU phase: cluster G0..G3 then U0..U3 (same destinations as baseline).
        DOG_X8(gsc0, gzp0, gpk0, gacc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(gsc1, gzp1, gpk1, gacc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(gsc2, gzp2, gpk2, gacc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(gsc3, gzp3, gpk3, gacc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);

        DOG_X8(usc0, uzp0, upk0, uacc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(usc1, uzp1, upk1, uacc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(usc2, uzp2, upk2, uacc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(usc3, uzp3, upk3, uacc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
    }

    if (tail >= 1) {
        const int g = quads << 2;
        int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float gsc = __builtin_bit_cast(float, *(const unsigned int*)(gate_ptr + g * 136));
        float gzp = __builtin_bit_cast(float, *(const unsigned int*)(gate_ptr + g * 136 + 4));
        unsigned int gpk = *(const unsigned int*)(gate_ptr + g * 136 + 8 + boff);
        float usc = __builtin_bit_cast(float, *(const unsigned int*)(up_ptr + g * 136));
        float uzp = __builtin_bit_cast(float, *(const unsigned int*)(up_ptr + g * 136 + 4));
        unsigned int upk = *(const unsigned int*)(up_ptr + g * 136 + 8 + boff);
        DOG_X8(gsc, gzp, gpk, gacc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(usc, uzp, upk, uacc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float gsc = __builtin_bit_cast(float, *(const unsigned int*)(gate_ptr + g * 136));
        float gzp = __builtin_bit_cast(float, *(const unsigned int*)(gate_ptr + g * 136 + 4));
        unsigned int gpk = *(const unsigned int*)(gate_ptr + g * 136 + 8 + boff);
        float usc = __builtin_bit_cast(float, *(const unsigned int*)(up_ptr + g * 136));
        float uzp = __builtin_bit_cast(float, *(const unsigned int*)(up_ptr + g * 136 + 4));
        unsigned int upk = *(const unsigned int*)(up_ptr + g * 136 + 8 + boff);
        DOG_X8(gsc, gzp, gpk, gacc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(usc, uzp, upk, uacc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float gsc = __builtin_bit_cast(float, *(const unsigned int*)(gate_ptr + g * 136));
        float gzp = __builtin_bit_cast(float, *(const unsigned int*)(gate_ptr + g * 136 + 4));
        unsigned int gpk = *(const unsigned int*)(gate_ptr + g * 136 + 8 + boff);
        float usc = __builtin_bit_cast(float, *(const unsigned int*)(up_ptr + g * 136));
        float uzp = __builtin_bit_cast(float, *(const unsigned int*)(up_ptr + g * 136 + 4));
        unsigned int upk = *(const unsigned int*)(up_ptr + g * 136 + 8 + boff);
        DOG_X8(gsc, gzp, gpk, gacc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(usc, uzp, upk, uacc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    #undef DOG_X8

    float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
    float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
    // Dual-issue gate+up shfl-reduce (residual style); per-acc add order unchanged.
    for (int offset = 16; offset > 0; offset >>= 1) {
        gacc += __shfl_down(gacc, offset);
        uacc += __shfl_down(uacc, offset);
    }

    if (tid == 0) {
        y_gate[krank * mi + row] = gacc;
        y_up[krank * mi + row] = uacc;
    }
}
