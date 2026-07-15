// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_MOE_GATE_UP_WG_WAVES
#define HIPFIRE_MOE_GATE_UP_WG_WAVES 1
#endif

#ifndef HIPFIRE_MOE_GATE_UP_KERNEL
#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed
#endif

#if HIPFIRE_MOE_GATE_UP_WG_WAVES == 2
#define HIPFIRE_MOE_GATE_UP_BLOCK 64
#define HIPFIRE_MOE_GATE_UP_MIN_BLOCKS 8
#else
#define HIPFIRE_MOE_GATE_UP_BLOCK 32
#define HIPFIRE_MOE_GATE_UP_MIN_BLOCKS 16
#endif

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

__launch_bounds__(HIPFIRE_MOE_GATE_UP_BLOCK, HIPFIRE_MOE_GATE_UP_MIN_BLOCKS)
extern "C" __global__ void HIPFIRE_MOE_GATE_UP_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x,                         // [K]
    float* __restrict__ y_gate,                          // [K_TOP × MI]
    float* __restrict__ y_up,                            // [K_TOP × MI]
    int M, int K
) {
    const int mi = M >> 1;
#if HIPFIRE_MOE_GATE_UP_WG_WAVES == 2
    const int wg_wave = threadIdx.x >> 5;
    const int tid = threadIdx.x & 31;
    const int row = blockIdx.x * 2 + wg_wave;
#else
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
#endif
    if (row >= mi) return;
    const int krank = blockIdx.y;

    // Read expert_id from device and index into the pointer table.
    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const char* gate_ptr = A + (long long)row * groups_per_row * 136;
    const char* up_ptr = A + (long long)(row + mi) * groups_per_row * 136;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    // One expert-wide SRD. Keep row/group displacement in scalar `soffset`
    // and only the lane's packed-weight byte displacement in `voffset`.
    const unsigned int row_stride = (unsigned int)groups_per_row * 136u;
    const unsigned int gate_row_offset = (unsigned int)row * row_stride;
    const unsigned int up_row_offset = (unsigned int)(row + mi) * row_stride;
    const auto expert_rsrc = HIPFIRE_WEIGHT_RSRC(A);
#endif

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

    #define LOAD_UNIFORM_U32(ptr, vector_offset, scalar_offset) \
        HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(expert_rsrc, ptr, vector_offset, scalar_offset)
    #define LOAD_PACKED_U32(ptr, vector_offset, scalar_offset) \
        HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(expert_rsrc, ptr, boff + vector_offset, scalar_offset)

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
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
        const unsigned int gate_base = gate_row_offset + (unsigned int)g * 136u;
#endif
        float gsc0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg0, 0, gate_base));
        float gzp0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg0 + 4, 4, gate_base));
        float gsc1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg1, 136, gate_base));
        float gzp1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg1 + 4, 140, gate_base));
        float gsc2 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg2, 272, gate_base));
        float gzp2 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg2 + 4, 276, gate_base));
        float gsc3 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg3, 408, gate_base));
        float gzp3 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg3 + 4, 412, gate_base));
        unsigned int gpk0 = LOAD_PACKED_U32(gg0 + 8 + boff, 8, gate_base);
        unsigned int gpk1 = LOAD_PACKED_U32(gg1 + 8 + boff, 144, gate_base);
        unsigned int gpk2 = LOAD_PACKED_U32(gg2 + 8 + boff, 280, gate_base);
        unsigned int gpk3 = LOAD_PACKED_U32(gg3 + 8 + boff, 416, gate_base);

#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
        // Release the gate header registers before loading the up stream. The
        // global-load control retains the original all-weights-ahead schedule.
        DOG_X8(gsc0, gzp0, gpk0, gacc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(gsc1, gzp1, gpk1, gacc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(gsc2, gzp2, gpk2, gacc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(gsc3, gzp3, gpk3, gacc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
#endif

        const char* uu0 = up_ptr + g * 136;
        const char* uu1 = uu0 + 136;
        const char* uu2 = uu1 + 136;
        const char* uu3 = uu2 + 136;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
        const unsigned int up_base = up_row_offset + (unsigned int)g * 136u;
#endif
        float usc0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu0, 0, up_base));
        float uzp0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu0 + 4, 4, up_base));
        float usc1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu1, 136, up_base));
        float uzp1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu1 + 4, 140, up_base));
        float usc2 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu2, 272, up_base));
        float uzp2 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu2 + 4, 276, up_base));
        float usc3 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu3, 408, up_base));
        float uzp3 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu3 + 4, 412, up_base));
        unsigned int upk0 = LOAD_PACKED_U32(uu0 + 8 + boff, 8, up_base);
        unsigned int upk1 = LOAD_PACKED_U32(uu1 + 8 + boff, 144, up_base);
        unsigned int upk2 = LOAD_PACKED_U32(uu2 + 8 + boff, 280, up_base);
        unsigned int upk3 = LOAD_PACKED_U32(uu3 + 8 + boff, 416, up_base);

        // VALU phase: cluster G0..G3 then U0..U3 (same destinations as baseline).
#if !HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
        DOG_X8(gsc0, gzp0, gpk0, gacc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(gsc1, gzp1, gpk1, gacc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(gsc2, gzp2, gpk2, gacc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(gsc3, gzp3, gpk3, gacc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
#endif

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
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
        const unsigned int goff = gate_row_offset + (unsigned int)g * 136u;
        const unsigned int uoff = up_row_offset + (unsigned int)g * 136u;
#endif
        float gsc = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gate_ptr + g * 136, 0, goff));
        float gzp = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gate_ptr + g * 136 + 4, 4, goff));
        unsigned int gpk = LOAD_PACKED_U32(gate_ptr + g * 136 + 8 + boff, 8, goff);
        float usc = __builtin_bit_cast(float, LOAD_UNIFORM_U32(up_ptr + g * 136, 0, uoff));
        float uzp = __builtin_bit_cast(float, LOAD_UNIFORM_U32(up_ptr + g * 136 + 4, 4, uoff));
        unsigned int upk = LOAD_PACKED_U32(up_ptr + g * 136 + 8 + boff, 8, uoff);
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
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
        const unsigned int goff = gate_row_offset + (unsigned int)g * 136u;
        const unsigned int uoff = up_row_offset + (unsigned int)g * 136u;
#endif
        float gsc = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gate_ptr + g * 136, 0, goff));
        float gzp = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gate_ptr + g * 136 + 4, 4, goff));
        unsigned int gpk = LOAD_PACKED_U32(gate_ptr + g * 136 + 8 + boff, 8, goff);
        float usc = __builtin_bit_cast(float, LOAD_UNIFORM_U32(up_ptr + g * 136, 0, uoff));
        float uzp = __builtin_bit_cast(float, LOAD_UNIFORM_U32(up_ptr + g * 136 + 4, 4, uoff));
        unsigned int upk = LOAD_PACKED_U32(up_ptr + g * 136 + 8 + boff, 8, uoff);
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
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
        const unsigned int goff = gate_row_offset + (unsigned int)g * 136u;
        const unsigned int uoff = up_row_offset + (unsigned int)g * 136u;
#endif
        float gsc = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gate_ptr + g * 136, 0, goff));
        float gzp = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gate_ptr + g * 136 + 4, 4, goff));
        unsigned int gpk = LOAD_PACKED_U32(gate_ptr + g * 136 + 8 + boff, 8, goff);
        float usc = __builtin_bit_cast(float, LOAD_UNIFORM_U32(up_ptr + g * 136, 0, uoff));
        float uzp = __builtin_bit_cast(float, LOAD_UNIFORM_U32(up_ptr + g * 136 + 4, 4, uoff));
        unsigned int upk = LOAD_PACKED_U32(up_ptr + g * 136 + 8 + boff, 8, uoff);
        DOG_X8(gsc, gzp, gpk, gacc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(usc, uzp, upk, uacc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    #undef LOAD_PACKED_U32
    #undef LOAD_UNIFORM_U32
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

#undef HIPFIRE_MOE_GATE_UP_BLOCK
#undef HIPFIRE_MOE_GATE_UP_MIN_BLOCKS
