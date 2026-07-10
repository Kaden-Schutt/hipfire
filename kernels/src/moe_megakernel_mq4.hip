// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Fused MoE routed-core MEGAKERNEL v2 (mq4 / uniform HFQ4-G256), based on the
// loop/gfx1201 WINNING kernels (53aab4775: baseline_v4 + residual lb32 +
// moe_down spref4rows). Collapses the four decode routed-core dispatches
//   1. gemv_hfq4g256_moe_gate_up_k8_indexed            (x[D] -> y_gate,y_up)
//   2. fused_silu_mul_mq_rotate                        (-> rot_batch)
//   3. gemv_hfq4g256_moe_down_k8_indexed_batched_expanded  (-> expert_outputs)
//   4. moe_down_combine_k8_batched                     (-> x_residual +=)
// into ONE cooperative-launch kernel (cg::this_grid().sync() between phases).
//
// UNLIKE v1 (which forced graph-OFF on the false premise that cooperative
// launch can't be hipGraph-captured — that is a gfx1100 blob-path limitation
// only), this kernel is meant to run UNDER hipGraph capture on gfx1201: a
// standalone test confirmed cooperative + capture + replay is correct on
// gfx1201/ROCm7.2. So the +7% graph gain is KEPT; only the fusion cost
// (grid.sync + co-residency) remains, vs 4 graph nodes (3 inter-node replay
// gaps) for the split path.
//
// BIT-EXACT: each phase's arithmetic + scheduling is copied VERBATIM from its
// WINNING standalone kernel (hoist_x32 + weight-preload + notail for gate_up;
// NR=4 + s_prefetch + x-hoist for down-expanded; unchanged silu; unchanged
// combine). Only the block->work mapping changes (grid-stride onto the fixed
// co-resident grid). No early return anywhere (would hang grid.sync).
//
// Shapes (decode, N=1): D = model hidden (gate_up in / down out); F = mi (ffn
// width per expert, = down contraction). Specialized to the a3b mq4r shape:
// D/256 divisible by 4 (gate_up notail); F/256 == 2 (down 2-group).
//   expert_ptrs_gu/dn : [n_exp] device weight-base ptr tables
//   topk_indices      : [K_TOP]   topk_weights : [K_TOP]
//   x                 : [D]       gate_up input (post norm+rotate)
//   signs1,signs2     : [256]     per-256-group FWHT signs
//   y_gate,y_up       : [K_TOP x F] scratch   rot_batch : [K_TOP x F] scratch
//   expert_outputs    : [K_TOP x D] scratch (phase-3 out / phase-4 in)
//   x_residual        : [D] += combined routed output
// MQ4 group = 136 bytes (4 scale + 4 zp + 128 packed 4-bit), 256 elems/group.

__launch_bounds__(32, 16)
extern "C" __global__ void moe_megakernel_mq4_hfq4g256(
    const unsigned long long* __restrict__ expert_ptrs_gu, // [n_exp]
    const unsigned long long* __restrict__ expert_ptrs_dn, // [n_exp]
    const int*   __restrict__ topk_indices,                // [K_TOP]
    const float* __restrict__ topk_weights,                // [K_TOP]
    const float* __restrict__ x,                           // [D]
    const float* __restrict__ signs1,                      // [256]
    const float* __restrict__ signs2,                      // [256]
    float* __restrict__ y_gate,                            // [K_TOP x F]
    float* __restrict__ y_up,                              // [K_TOP x F]
    float* __restrict__ rot_batch,                         // [K_TOP x F]
    float* __restrict__ expert_outputs,                    // [K_TOP x D]
    float* __restrict__ x_residual,                        // [D] += result
    int D, int F, int K_TOP
) {
    cg::grid_group grid = cg::this_grid();
    const int tid  = threadIdx.x;
    const int pb   = blockIdx.x;
    const int nblk = gridDim.x;

    // ================= PHASE 1 — gate_up (WINNING: hoist_x32 + w-preload + notail)
    {
        const int mi = F;
        const int groups_per_row = D / 256;
        const int total = mi * K_TOP;
        const int boff = tid * 4;
        const int quads = groups_per_row >> 2;   // notail: D/256 divisible by 4

        #define MEGA_DOG_X8(sc, zp, pk, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
            (a) += ((sc) * (float)((pk) & 0xFu)         + (zp)) * (x0) \
                 + ((sc) * (float)(((pk) >> 4) & 0xFu)  + (zp)) * (x1) \
                 + ((sc) * (float)(((pk) >> 8) & 0xFu)  + (zp)) * (x2) \
                 + ((sc) * (float)(((pk) >> 12) & 0xFu) + (zp)) * (x3) \
                 + ((sc) * (float)(((pk) >> 16) & 0xFu) + (zp)) * (x4) \
                 + ((sc) * (float)(((pk) >> 20) & 0xFu) + (zp)) * (x5) \
                 + ((sc) * (float)(((pk) >> 24) & 0xFu) + (zp)) * (x6) \
                 + ((sc) * (float)(((pk) >> 28) & 0xFu) + (zp)) * (x7); \
        } while (0)

        for (int lb = pb; lb < total; lb += nblk) {
            const int krank = lb / mi;
            const int row   = lb - krank * mi;
            const int expert_id = topk_indices[krank];
            const char* __restrict__ A =
                reinterpret_cast<const char*>(expert_ptrs_gu[expert_id]);
            const char* gate_ptr = A + (long long)row * groups_per_row * 136;
            const char* up_ptr   = A + (long long)(row + mi) * groups_per_row * 136;

            float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
            float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;

            for (int q = 0; q < quads; q++) {
                const int g = q << 2;
                const int base0 = g * 256 + tid * 8;
                // hoist full 32-wide x for this group-quad (float4 pairs)
                const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
                const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
                const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
                const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);
                const float4 xv2a = *reinterpret_cast<const float4*>(x + base0 + 512);
                const float4 xv2b = *reinterpret_cast<const float4*>(x + base0 + 516);
                const float4 xv3a = *reinterpret_cast<const float4*>(x + base0 + 768);
                const float4 xv3b = *reinterpret_cast<const float4*>(x + base0 + 772);
                // MEMORY phase: all 4 gate weights, then all 4 up weights.
                const char* gg0 = gate_ptr + g * 136; const char* gg1 = gg0 + 136;
                const char* gg2 = gg1 + 136;          const char* gg3 = gg2 + 136;
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
                const char* uu0 = up_ptr + g * 136; const char* uu1 = uu0 + 136;
                const char* uu2 = uu1 + 136;         const char* uu3 = uu2 + 136;
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
                // VALU phase: cluster G0..G3 then U0..U3.
                MEGA_DOG_X8(gsc0, gzp0, gpk0, gacc0, xv0a.x,xv0a.y,xv0a.z,xv0a.w,xv0b.x,xv0b.y,xv0b.z,xv0b.w);
                MEGA_DOG_X8(gsc1, gzp1, gpk1, gacc1, xv1a.x,xv1a.y,xv1a.z,xv1a.w,xv1b.x,xv1b.y,xv1b.z,xv1b.w);
                MEGA_DOG_X8(gsc2, gzp2, gpk2, gacc2, xv2a.x,xv2a.y,xv2a.z,xv2a.w,xv2b.x,xv2b.y,xv2b.z,xv2b.w);
                MEGA_DOG_X8(gsc3, gzp3, gpk3, gacc3, xv3a.x,xv3a.y,xv3a.z,xv3a.w,xv3b.x,xv3b.y,xv3b.z,xv3b.w);
                MEGA_DOG_X8(usc0, uzp0, upk0, uacc0, xv0a.x,xv0a.y,xv0a.z,xv0a.w,xv0b.x,xv0b.y,xv0b.z,xv0b.w);
                MEGA_DOG_X8(usc1, uzp1, upk1, uacc1, xv1a.x,xv1a.y,xv1a.z,xv1a.w,xv1b.x,xv1b.y,xv1b.z,xv1b.w);
                MEGA_DOG_X8(usc2, uzp2, upk2, uacc2, xv2a.x,xv2a.y,xv2a.z,xv2a.w,xv2b.x,xv2b.y,xv2b.z,xv2b.w);
                MEGA_DOG_X8(usc3, uzp3, upk3, uacc3, xv3a.x,xv3a.y,xv3a.z,xv3a.w,xv3b.x,xv3b.y,xv3b.z,xv3b.w);
            }
            float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
            float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
            for (int offset = 16; offset > 0; offset >>= 1) gacc += __shfl_down(gacc, offset);
            for (int offset = 16; offset > 0; offset >>= 1) uacc += __shfl_down(uacc, offset);
            if (tid == 0) { y_gate[krank * mi + row] = gacc; y_up[krank * mi + row] = uacc; }
        }
        #undef MEGA_DOG_X8
    }
    grid.sync();

    // ================= PHASE 2 — SwiGLU + FWHT rotate (unchanged winning silu)
    {
        const int groups_F = F / 256;
        const int total = groups_F * K_TOP;
        const int d0 = tid * 8;
        for (int lb = pb; lb < total; lb += nblk) {
            const int krank = lb / groups_F;
            const int group = lb - krank * groups_F;
            const long long batch_off = (long long)krank * F;
            const float* gate_b = y_gate + batch_off;
            const float* up_b   = y_up   + batch_off;
                  float* rot_b  = rot_batch + batch_off;
            const int base = group * 256 + d0;
            #define MEGA_SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
            float g0 = gate_b[base];     float u0 = up_b[base];
            float g1 = gate_b[base + 1]; float u1 = up_b[base + 1];
            float g2 = gate_b[base + 2]; float u2 = up_b[base + 2];
            float g3 = gate_b[base + 3]; float u3 = up_b[base + 3];
            float g4 = gate_b[base + 4]; float u4 = up_b[base + 4];
            float g5 = gate_b[base + 5]; float u5 = up_b[base + 5];
            float g6 = gate_b[base + 6]; float u6 = up_b[base + 6];
            float g7 = gate_b[base + 7]; float u7 = up_b[base + 7];
            float v0 = MEGA_SILU_MUL(g0, u0) * signs1[d0];
            float v1 = MEGA_SILU_MUL(g1, u1) * signs1[d0 + 1];
            float v2 = MEGA_SILU_MUL(g2, u2) * signs1[d0 + 2];
            float v3 = MEGA_SILU_MUL(g3, u3) * signs1[d0 + 3];
            float v4 = MEGA_SILU_MUL(g4, u4) * signs1[d0 + 4];
            float v5 = MEGA_SILU_MUL(g5, u5) * signs1[d0 + 5];
            float v6 = MEGA_SILU_MUL(g6, u6) * signs1[d0 + 6];
            float v7 = MEGA_SILU_MUL(g7, u7) * signs1[d0 + 7];
            #undef MEGA_SILU_MUL
            float t;
            t = v0; v0 = v0 + v1; v1 = t - v1;  t = v2; v2 = v2 + v3; v3 = t - v3;
            t = v4; v4 = v4 + v5; v5 = t - v5;  t = v6; v6 = v6 + v7; v7 = t - v7;
            t = v0; v0 = v0 + v2; v2 = t - v2;  t = v1; v1 = v1 + v3; v3 = t - v3;
            t = v4; v4 = v4 + v6; v6 = t - v6;  t = v5; v5 = v5 + v7; v7 = t - v7;
            t = v0; v0 = v0 + v4; v4 = t - v4;  t = v1; v1 = v1 + v5; v5 = t - v5;
            t = v2; v2 = v2 + v6; v6 = t - v6;  t = v3; v3 = v3 + v7; v7 = t - v7;
            #define MEGA_HBFLY(v, pat, str) do { \
                float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
                if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
            } while(0)
            #define MEGA_HBFLY8(pat, str) \
                MEGA_HBFLY(v0,pat,str); MEGA_HBFLY(v1,pat,str); MEGA_HBFLY(v2,pat,str); MEGA_HBFLY(v3,pat,str); \
                MEGA_HBFLY(v4,pat,str); MEGA_HBFLY(v5,pat,str); MEGA_HBFLY(v6,pat,str); MEGA_HBFLY(v7,pat,str)
            MEGA_HBFLY8(0x041F, 1); MEGA_HBFLY8(0x081F, 2); MEGA_HBFLY8(0x101F, 4);
            MEGA_HBFLY8(0x201F, 8); MEGA_HBFLY8(0x401F, 16);
            #undef MEGA_HBFLY8
            #undef MEGA_HBFLY
            const float s = 0.0625f;
            rot_b[base]     = v0 * s * signs2[d0];
            rot_b[base + 1] = v1 * s * signs2[d0 + 1];
            rot_b[base + 2] = v2 * s * signs2[d0 + 2];
            rot_b[base + 3] = v3 * s * signs2[d0 + 3];
            rot_b[base + 4] = v4 * s * signs2[d0 + 4];
            rot_b[base + 5] = v5 * s * signs2[d0 + 5];
            rot_b[base + 6] = v6 * s * signs2[d0 + 6];
            rot_b[base + 7] = v7 * s * signs2[d0 + 7];
        }
    }
    grid.sync();

    // ================= PHASE 3 — down GEMV -> expanded (WINNING: NR=4 + s_prefetch + hoist)
    // Specialized to F/256 == 2 groups (a3b mq4r). Logical: (D/4) x K_TOP.
    {
        const int groups_per_row = F / 256;                 // == 2
        const long long row_stride = (long long)groups_per_row * 136;
        const int boff = tid * 4;
        const int tiles = (D + 3) / 4;                      // row-tiles of 4
        const int total = tiles * K_TOP;

        #define MEGA_DOG_D(gp, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp)); \
            float zp = __builtin_bit_cast(float, *(const unsigned int*)((gp) + 4)); \
            unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
            (a) += (sc * (float)((pk) & 0xFu)         + zp) * (x0) \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (x1) \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (x2) \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (x3) \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * (x4) \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (x5) \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (x6) \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (x7); \
        } while (0)

        for (int lb = pb; lb < total; lb += nblk) {
            const int krank = lb / tiles;
            const int tile  = lb - krank * tiles;
            const int row0 = tile * 4;
            const int row1 = row0 + 1, row2 = row0 + 2, row3 = row0 + 3;
            const bool have_row1 = row1 < D, have_row2 = row2 < D, have_row3 = row3 < D;
            const int expert_id = topk_indices[krank];
            const char* __restrict__ A =
                reinterpret_cast<const char*>(expert_ptrs_dn[expert_id]);
            const float* __restrict__ xr = rot_batch + (long long)krank * F;
            const char* row_ptr0 = A + (long long)row0 * row_stride;
            const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
            const char* row_ptr2 = A + (long long)(have_row2 ? row2 : row0) * row_stride;
            const char* row_ptr3 = A + (long long)(have_row3 ? row3 : row0) * row_stride;
#if defined(__gfx1200__) || defined(__gfx1201__)
            __builtin_amdgcn_s_prefetch_data((const void*)row_ptr0, 0);
            __builtin_amdgcn_s_prefetch_data((const void*)row_ptr1, 0);
            __builtin_amdgcn_s_prefetch_data((const void*)row_ptr2, 0);
            __builtin_amdgcn_s_prefetch_data((const void*)row_ptr3, 0);
#endif
            float acc0 = 0.0f, acc1 = 0.0f, bcc0 = 0.0f, bcc1 = 0.0f;
            float ccc0 = 0.0f, ccc1 = 0.0f, dcc0 = 0.0f, dcc1 = 0.0f;
            const int base0 = tid * 8;
            const float4 xv0a = *reinterpret_cast<const float4*>(xr + base0);
            const float4 xv0b = *reinterpret_cast<const float4*>(xr + base0 + 4);
            const float4 xv1a = *reinterpret_cast<const float4*>(xr + base0 + 256);
            const float4 xv1b = *reinterpret_cast<const float4*>(xr + base0 + 260);
            MEGA_DOG_D(row_ptr0 + 0 * 136, acc0, xv0a.x,xv0a.y,xv0a.z,xv0a.w,xv0b.x,xv0b.y,xv0b.z,xv0b.w);
            MEGA_DOG_D(row_ptr1 + 0 * 136, bcc0, xv0a.x,xv0a.y,xv0a.z,xv0a.w,xv0b.x,xv0b.y,xv0b.z,xv0b.w);
            MEGA_DOG_D(row_ptr2 + 0 * 136, ccc0, xv0a.x,xv0a.y,xv0a.z,xv0a.w,xv0b.x,xv0b.y,xv0b.z,xv0b.w);
            MEGA_DOG_D(row_ptr3 + 0 * 136, dcc0, xv0a.x,xv0a.y,xv0a.z,xv0a.w,xv0b.x,xv0b.y,xv0b.z,xv0b.w);
            MEGA_DOG_D(row_ptr0 + 1 * 136, acc1, xv1a.x,xv1a.y,xv1a.z,xv1a.w,xv1b.x,xv1b.y,xv1b.z,xv1b.w);
            MEGA_DOG_D(row_ptr1 + 1 * 136, bcc1, xv1a.x,xv1a.y,xv1a.z,xv1a.w,xv1b.x,xv1b.y,xv1b.z,xv1b.w);
            MEGA_DOG_D(row_ptr2 + 1 * 136, ccc1, xv1a.x,xv1a.y,xv1a.z,xv1a.w,xv1b.x,xv1b.y,xv1b.z,xv1b.w);
            MEGA_DOG_D(row_ptr3 + 1 * 136, dcc1, xv1a.x,xv1a.y,xv1a.z,xv1a.w,xv1b.x,xv1b.y,xv1b.z,xv1b.w);
            float acc = acc0 + acc1, bcc = bcc0 + bcc1, ccc = ccc0 + ccc1, dcc = dcc0 + dcc1;
            for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
            for (int offset = 16; offset > 0; offset >>= 1) bcc += __shfl_down(bcc, offset);
            for (int offset = 16; offset > 0; offset >>= 1) ccc += __shfl_down(ccc, offset);
            for (int offset = 16; offset > 0; offset >>= 1) dcc += __shfl_down(dcc, offset);
            if (tid == 0) {
                const long long out_base = (long long)krank * D;   // [krank][row], N=1
                expert_outputs[out_base + row0] = acc;
                if (have_row1) expert_outputs[out_base + row1] = bcc;
                if (have_row2) expert_outputs[out_base + row2] = ccc;
                if (have_row3) expert_outputs[out_base + row3] = dcc;
            }
        }
        #undef MEGA_DOG_D
    }
    grid.sync();

    // ================= PHASE 4 — combine K_TOP -> residual (unchanged combine math)
    // Each thread owns one output column; block covers 32 columns. Logical: ceil(D/32).
    {
        const int col_tiles = (D + 31) / 32;
        for (int lb = pb; lb < col_tiles; lb += nblk) {
            const int col = lb * 32 + tid;
            if (col < D) {
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (k < K_TOP) {
                        float w = topk_weights[k];
                        float v = expert_outputs[(long long)k * D + col];
                        acc += w * v;
                    }
                }
                x_residual[col] += acc;
            }
        }
    }
}
