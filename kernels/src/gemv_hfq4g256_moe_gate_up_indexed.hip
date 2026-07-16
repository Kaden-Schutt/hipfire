// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#if defined(HIPFIRE_RDNA3_MOE_GATE_UP_X_BUFFER)
using hipfire_gate_i32x4 = int __attribute__((ext_vector_type(4)));
using hipfire_gate_f32x4 = float __attribute__((ext_vector_type(4)));
#endif
#if defined(HIPFIRE_RDNA3_MOE_GATE_UP_ROUTE_BUFFER) || \
    defined(HIPFIRE_GFX1151_MOE_GATE_UP_HYBRID_BUFFER)
using hipfire_gate_u32x2 = unsigned int __attribute__((ext_vector_type(2)));
#endif

#ifndef HIPFIRE_MOE_GATE_UP_WG_WAVES
#define HIPFIRE_MOE_GATE_UP_WG_WAVES 1
#endif

#ifndef HIPFIRE_MOE_GATE_UP_KERNEL
#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed
#endif

#ifndef HIPFIRE_MOE_GATE_UP_RANK_INTERLEAVE
#define HIPFIRE_MOE_GATE_UP_RANK_INTERLEAVE 0
#endif

#ifndef HIPFIRE_MOE_GATE_UP_LOW_VGPR
#define HIPFIRE_MOE_GATE_UP_LOW_VGPR 0
#endif

#ifndef HIPFIRE_MOE_GATE_UP_PAIR_VGPR
#define HIPFIRE_MOE_GATE_UP_PAIR_VGPR 0
#endif

#if HIPFIRE_MOE_GATE_UP_WG_WAVES == 2
#define HIPFIRE_MOE_GATE_UP_BLOCK 64
#ifndef HIPFIRE_MOE_GATE_UP_MIN_BLOCKS
#define HIPFIRE_MOE_GATE_UP_MIN_BLOCKS 8
#endif
#else
#define HIPFIRE_MOE_GATE_UP_BLOCK 32
#ifndef HIPFIRE_MOE_GATE_UP_MIN_BLOCKS
#define HIPFIRE_MOE_GATE_UP_MIN_BLOCKS 16
#endif
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
#if HIPFIRE_MOE_GATE_UP_RANK_INTERLEAVE
    // gfx1100 K=8 scheduling experiment: make expert rank the fastest-moving
    // workgroup coordinate. Eight consecutive workgroups consume the same
    // 8-KiB activation while issuing independent expert-weight streams. The
    // per-row arithmetic and output address are unchanged.
    const int tid = threadIdx.x;
    const int linear = blockIdx.x;
    const int krank = linear & 7;
    const int row = linear >> 3;
#elif HIPFIRE_MOE_GATE_UP_WG_WAVES == 2
    const int wg_wave = threadIdx.x >> 5;
    const int tid = threadIdx.x & 31;
    const int row = blockIdx.x * 2 + wg_wave;
#else
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
#endif
    if (row >= mi) return;
#if !HIPFIRE_MOE_GATE_UP_RANK_INTERLEAVE
    const int krank = blockIdx.y;
#endif

    // Read expert_id from device and index into the pointer table. The
    // gfx1151 route-buffer candidate deliberately keeps both dependent reads
    // on VMEM, then scalarizes their uniform result without touching the
    // scalar data cache. This makes the route dependency eligible for the
    // narrower same-agent Redline acquire.
#if defined(HIPFIRE_RDNA3_MOE_GATE_UP_ROUTE_BUFFER)
    const auto topk_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<int*>(topk_indices), 0, 0xffffffffu, 0x31004000u);
    const unsigned int expert_id_v = __builtin_amdgcn_raw_buffer_load_b32(
        topk_rsrc, (unsigned int)krank * 4u, 0, 0);
    const int expert_id = __builtin_amdgcn_readfirstlane(expert_id_v);
    const auto expert_ptrs_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<unsigned long long*>(expert_ptrs),
        0, 0xffffffffu, 0x31004000u);
    const hipfire_gate_u32x2 expert_ptr_v = __builtin_amdgcn_raw_buffer_load_b64(
        expert_ptrs_rsrc, (unsigned int)expert_id * 8u, 0, 0);
    const unsigned int expert_ptr_lo =
        (unsigned int)__builtin_amdgcn_readfirstlane(expert_ptr_v.x);
    const unsigned int expert_ptr_hi =
        (unsigned int)__builtin_amdgcn_readfirstlane(expert_ptr_v.y);
    const unsigned long long expert_ptr =
        (unsigned long long)expert_ptr_lo | ((unsigned long long)expert_ptr_hi << 32);
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_ptr);
#else
    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
#endif

#if defined(HIPFIRE_RDNA3_MOE_GATE_UP_K2048)
    // The gfx1100 A3B decode route is fixed at K=2048. Host dispatch enforces
    // that contract so LLVM can delete the runtime divide and constant-fold
    // both row strides and the two-quad loop bound.
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* gate_ptr = A + (long long)row * groups_per_row * 136;
    const char* up_ptr = A + (long long)(row + mi) * groups_per_row * 136;
    const int boff = tid * 4;
    const unsigned int row_stride = (unsigned int)groups_per_row * 136u;
    const unsigned int gate_row_offset = (unsigned int)row * row_stride;
    const unsigned int up_row_offset = (unsigned int)(row + mi) * row_stride;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    // One expert-wide SRD. Keep row/group displacement in scalar `soffset`
    // and only the lane's packed-weight byte displacement in `voffset`.
    const auto expert_rsrc = HIPFIRE_WEIGHT_RSRC(A);
#endif
#if defined(HIPFIRE_RDNA3_MOE_GATE_UP_X_BUFFER)
    const auto x_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<float*>(x), 0, 0xffffffffu, 0x31004000u);
    #define LOAD_X4(index) __builtin_bit_cast(hipfire_gate_f32x4, \
        __builtin_amdgcn_raw_buffer_load_b128( \
            x_rsrc, (unsigned int)(index) * 4u, 0, 0))
    #define DECLARE_PAIR_X(base) \
        const hipfire_gate_f32x4 x0a = LOAD_X4(base); \
        const hipfire_gate_f32x4 x0b = LOAD_X4((base) + 4); \
        const hipfire_gate_f32x4 x1a = LOAD_X4((base) + 256); \
        const hipfire_gate_f32x4 x1b = LOAD_X4((base) + 260); \
        asm volatile("" : : "v"(x0a), "v"(x0b), "v"(x1a), "v"(x1b) : "memory")
#else
    #define DECLARE_PAIR_X(base) \
        const float4 x0a = *reinterpret_cast<const float4*>(x + (base)); \
        const float4 x0b = *reinterpret_cast<const float4*>(x + (base) + 4); \
        const float4 x1a = *reinterpret_cast<const float4*>(x + (base) + 256); \
        const float4 x1b = *reinterpret_cast<const float4*>(x + (base) + 260)
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

#if HIPFIRE_MOE_GATE_UP_LOW_VGPR
    // gfx1100 occupancy experiment: retain the four independent accumulator
    // chains, but keep only one group's eight activation values and two weight
    // headers live at a time. The default schedule hoists four groups (32 x
    // values) to maximize load clustering, which is profitable on gfx12 but
    // leaves this kernel near the gfx1100 VGPR occupancy boundary.
    #define PROCESS_LOW_VGPR(group_index, chain) do { \
        const int group = (group_index); \
        const int base = group * 256 + tid * 8; \
        const float4 xa = *reinterpret_cast<const float4*>(x + base); \
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4); \
        const char* gg = gate_ptr + group * 136; \
        const char* uu = up_ptr + group * 136; \
        const unsigned int group_offset = (unsigned int)group * 136u; \
        float gsc = __builtin_bit_cast(float, \
            LOAD_UNIFORM_U32(gg, 0, gate_row_offset + group_offset)); \
        float gzp = __builtin_bit_cast(float, \
            LOAD_UNIFORM_U32(gg + 4, 4, gate_row_offset + group_offset)); \
        unsigned int gpk = LOAD_PACKED_U32( \
            gg + 8 + boff, 8, gate_row_offset + group_offset); \
        DOG_X8(gsc, gzp, gpk, gacc##chain, \
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w); \
        float usc = __builtin_bit_cast(float, \
            LOAD_UNIFORM_U32(uu, 0, up_row_offset + group_offset)); \
        float uzp = __builtin_bit_cast(float, \
            LOAD_UNIFORM_U32(uu + 4, 4, up_row_offset + group_offset)); \
        unsigned int upk = LOAD_PACKED_U32( \
            uu + 8 + boff, 8, up_row_offset + group_offset); \
        DOG_X8(usc, uzp, upk, uacc##chain, \
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w); \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        PROCESS_LOW_VGPR(g, 0);
        PROCESS_LOW_VGPR(g + 1, 1);
        PROCESS_LOW_VGPR(g + 2, 2);
        PROCESS_LOW_VGPR(g + 3, 3);
    }
    #undef PROCESS_LOW_VGPR
#elif HIPFIRE_MOE_GATE_UP_PAIR_VGPR
    // gfx1100 middle-occupancy schedule: keep two groups' activation values
    // live at once. The default four-group schedule emits 96 VGPR and the
    // one-group schedule gives up too much memory-level parallelism. Pairing
    // targets the intervening occupancy band while preserving each group's
    // accumulator and arithmetic order exactly.
    #define PROCESS_PAIR(group_index, chain0, chain1) do { \
        const int group = (group_index); \
        const int base = group * 256 + tid * 8; \
        DECLARE_PAIR_X(base); \
        const char* gg0 = gate_ptr + group * 136; \
        const char* gg1 = gg0 + 136; \
        const unsigned int gate_base = \
            HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS \
                ? gate_row_offset + (unsigned int)group * 136u \
                : 0u; \
        float gsc0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg0, 0, gate_base)); \
        float gzp0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg0 + 4, 4, gate_base)); \
        float gsc1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg1, 136, gate_base)); \
        float gzp1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg1 + 4, 140, gate_base)); \
        unsigned int gpk0 = LOAD_PACKED_U32(gg0 + 8 + boff, 8, gate_base); \
        unsigned int gpk1 = LOAD_PACKED_U32(gg1 + 8 + boff, 144, gate_base); \
        DOG_X8(gsc0, gzp0, gpk0, gacc##chain0, \
               x0a.x, x0a.y, x0a.z, x0a.w, x0b.x, x0b.y, x0b.z, x0b.w); \
        DOG_X8(gsc1, gzp1, gpk1, gacc##chain1, \
               x1a.x, x1a.y, x1a.z, x1a.w, x1b.x, x1b.y, x1b.z, x1b.w); \
        const char* uu0 = up_ptr + group * 136; \
        const char* uu1 = uu0 + 136; \
        const unsigned int up_base = \
            HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS \
                ? up_row_offset + (unsigned int)group * 136u \
                : 0u; \
        float usc0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu0, 0, up_base)); \
        float uzp0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu0 + 4, 4, up_base)); \
        float usc1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu1, 136, up_base)); \
        float uzp1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu1 + 4, 140, up_base)); \
        unsigned int upk0 = LOAD_PACKED_U32(uu0 + 8 + boff, 8, up_base); \
        unsigned int upk1 = LOAD_PACKED_U32(uu1 + 8 + boff, 144, up_base); \
        DOG_X8(usc0, uzp0, upk0, uacc##chain0, \
               x0a.x, x0a.y, x0a.z, x0a.w, x0b.x, x0b.y, x0b.z, x0b.w); \
        DOG_X8(usc1, uzp1, upk1, uacc##chain1, \
               x1a.x, x1a.y, x1a.z, x1a.w, x1b.x, x1b.y, x1b.z, x1b.w); \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        PROCESS_PAIR(g, 0, 1);
        PROCESS_PAIR(g + 2, 2, 3);
    }
    #undef PROCESS_PAIR
#else
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        // Hoist full 32-wide x for this group-quad (4 groups × 8) via float4 pairs
        // before any weight load/DOG — same schedule as residual R15c2 / qkvza R17c2.
#if defined(HIPFIRE_RDNA3_MOE_GATE_UP_X_BUFFER)
        const hipfire_gate_f32x4 xv0a = LOAD_X4(base0);
        const hipfire_gate_f32x4 xv0b = LOAD_X4(base0 + 4);
        const hipfire_gate_f32x4 xv1a = LOAD_X4(base0 + 256);
        const hipfire_gate_f32x4 xv1b = LOAD_X4(base0 + 260);
        const hipfire_gate_f32x4 xv2a = LOAD_X4(base0 + 512);
        const hipfire_gate_f32x4 xv2b = LOAD_X4(base0 + 516);
        const hipfire_gate_f32x4 xv3a = LOAD_X4(base0 + 768);
        const hipfire_gate_f32x4 xv3b = LOAD_X4(base0 + 772);
        // LLVM otherwise sinks each B128 activation request beside its first
        // dependent DOG, collapsing the intended VMEM clause into a chain of
        // load/wait/VALU islands. Keep all eight vectors live at one compiler
        // barrier so gfx1151 can issue the hot activation stream before the
        // independent expert-weight requests.
        asm volatile("" : :
            "v"(xv0a), "v"(xv0b), "v"(xv1a), "v"(xv1b),
            "v"(xv2a), "v"(xv2b), "v"(xv3a), "v"(xv3b) : "memory");
#else
        const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
        const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);
        const float4 xv2a = *reinterpret_cast<const float4*>(x + base0 + 512);
        const float4 xv2b = *reinterpret_cast<const float4*>(x + base0 + 516);
        const float4 xv3a = *reinterpret_cast<const float4*>(x + base0 + 768);
        const float4 xv3b = *reinterpret_cast<const float4*>(x + base0 + 772);
#endif

        // MEMORY phase: sequential gate weights, then sequential up weights.
        const char* gg0 = gate_ptr + g * 136;
        const char* gg1 = gg0 + 136;
        const char* gg2 = gg1 + 136;
        const char* gg3 = gg2 + 136;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
        const unsigned int gate_base = gate_row_offset + (unsigned int)g * 136u;
#endif
#if defined(HIPFIRE_GFX1151_MOE_GATE_UP_HYBRID_BUFFER)
        // Scale and zero-point share one aligned, wave-uniform 8-byte header.
        // The constant address space makes that uniformity explicit to LLVM,
        // lowering these to scalar-memory loads. Only the lane-divergent
        // packed word below uses raw-buffer VMEM.
        const hipfire_gate_u32x2 gh0 =
            *(const hipfire_gate_u32x2 __attribute__((address_space(4)))*)gg0;
        const hipfire_gate_u32x2 gh1 =
            *(const hipfire_gate_u32x2 __attribute__((address_space(4)))*)gg1;
        const hipfire_gate_u32x2 gh2 =
            *(const hipfire_gate_u32x2 __attribute__((address_space(4)))*)gg2;
        const hipfire_gate_u32x2 gh3 =
            *(const hipfire_gate_u32x2 __attribute__((address_space(4)))*)gg3;
        float gsc0 = __builtin_bit_cast(float, gh0.x);
        float gzp0 = __builtin_bit_cast(float, gh0.y);
        float gsc1 = __builtin_bit_cast(float, gh1.x);
        float gzp1 = __builtin_bit_cast(float, gh1.y);
        float gsc2 = __builtin_bit_cast(float, gh2.x);
        float gzp2 = __builtin_bit_cast(float, gh2.y);
        float gsc3 = __builtin_bit_cast(float, gh3.x);
        float gzp3 = __builtin_bit_cast(float, gh3.y);
#else
        float gsc0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg0, 0, gate_base));
        float gzp0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg0 + 4, 4, gate_base));
        float gsc1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg1, 136, gate_base));
        float gzp1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg1 + 4, 140, gate_base));
        float gsc2 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg2, 272, gate_base));
        float gzp2 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg2 + 4, 276, gate_base));
        float gsc3 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg3, 408, gate_base));
        float gzp3 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(gg3 + 4, 412, gate_base));
#endif
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
#if defined(HIPFIRE_GFX1151_MOE_GATE_UP_HYBRID_BUFFER)
        const hipfire_gate_u32x2 uh0 =
            *(const hipfire_gate_u32x2 __attribute__((address_space(4)))*)uu0;
        const hipfire_gate_u32x2 uh1 =
            *(const hipfire_gate_u32x2 __attribute__((address_space(4)))*)uu1;
        const hipfire_gate_u32x2 uh2 =
            *(const hipfire_gate_u32x2 __attribute__((address_space(4)))*)uu2;
        const hipfire_gate_u32x2 uh3 =
            *(const hipfire_gate_u32x2 __attribute__((address_space(4)))*)uu3;
        float usc0 = __builtin_bit_cast(float, uh0.x);
        float uzp0 = __builtin_bit_cast(float, uh0.y);
        float usc1 = __builtin_bit_cast(float, uh1.x);
        float uzp1 = __builtin_bit_cast(float, uh1.y);
        float usc2 = __builtin_bit_cast(float, uh2.x);
        float uzp2 = __builtin_bit_cast(float, uh2.y);
        float usc3 = __builtin_bit_cast(float, uh3.x);
        float uzp3 = __builtin_bit_cast(float, uh3.y);
#else
        float usc0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu0, 0, up_base));
        float uzp0 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu0 + 4, 4, up_base));
        float usc1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu1, 136, up_base));
        float uzp1 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu1 + 4, 140, up_base));
        float usc2 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu2, 272, up_base));
        float uzp2 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu2 + 4, 276, up_base));
        float usc3 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu3, 408, up_base));
        float uzp3 = __builtin_bit_cast(float, LOAD_UNIFORM_U32(uu3 + 4, 412, up_base));
#endif
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
#endif

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
    #undef DECLARE_PAIR_X
#if defined(HIPFIRE_RDNA3_MOE_GATE_UP_X_BUFFER)
    #undef LOAD_X4
#endif

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
