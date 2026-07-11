// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Fused MoE routed-core MEGAKERNEL (mq4 / uniform HFQ4-G256).
//
// Collapses the three decode routed-core dispatches
//   1. gemv_hfq4g256_moe_gate_up_k8_indexed   (x[D] -> y_gate,y_up [K_TOP x F])
//   2. fused_silu_mul_mq_rotate               (-> rot_batch [K_TOP x F])
//   3. gemv_hfq4g256_moe_down_k8_indexed_fused_acc (-> x_residual[D] +=)
// into ONE cooperative-launch kernel. All blocks are co-resident (occupancy-
// bounded grid from the launcher), so cg::this_grid().sync() is a device-wide
// barrier between phases. The intermediates (y_gate/y_up/rot_batch) are the
// SAME global scratch buffers the separate kernels use, so they survive the
// barrier untouched.
//
// WHY (hypothesis): a3b batch-1 decode looked ~87% GPU-idle — ~122 kernel
// dispatches/token, each seemingly paying a ~45us GPU-side inter-kernel gap.
// Fusing 3 dispatches/layer into 1 (x40 layers) should remove ~80 of those gaps.
// Spike-validated 3.6-4.9x on the pure dispatch-overhead micro on gfx1100/gfx1201.
//
// MEASURED RESULT — FALSIFIED for decode (2026-07-10, gfx1201 R9700, a3b mq4r,
// warm, registry sampling, q8 KV, max256; serve_harness A/B, all turns stable):
//   A baseline graph-ON  = 153.1 tok/s   (target to beat)
//   C baseline graph-OFF = 143.0 tok/s
//   B THIS megakernel    = 132.4 tok/s   -> -7.4% vs C, -13.5% vs A
// Output is BYTE-EXACT to the split path (parity PASS), but perf REGRESSES. Two
// independent causes, both structural to the cooperative approach:
//   1. The fusion itself loses -7.4% (B vs C, both graph-off): grid.sync barriers
//      (2/layer x 40) + the co-residency cap (all blocks must fit at once, so each
//      phase runs on FEWER blocks than the split kernels' full grids and grid-
//      strides serially) cost MORE than the dispatch gap they remove. The rocprof
//      "idle" is intra-kernel memory-latency stall (batch-1 under-saturates BW),
//      NOT a recoverable inter-kernel gap.
//   2. Cooperative launch cannot be hipGraph-captured, forfeiting the +7.1% graph
//      gain (A vs C) — unrecoverable.
// => Dispatch fusion is a DEAD END for latency-bound batch-1 decode; the lever is
//    batch-K (MTP/spec-decode). Kept opt-in (default OFF, byte-exact) as a
//    documented negative result + reusable cooperative-launch reference. DO NOT
//    enable HIPFIRE_MOE_MEGAKERNEL for decode perf.
//
// BIT-EXACTNESS: each phase's per-output-element math is copied VERBATIM from
// its standalone kernel (same accumulation order, same 4-acc pairwise +
// __shfl_down reduce, same sequential over-expert accumulate). The megakernel
// changes only WHICH physical block computes WHICH logical work item (a grid-
// stride remap onto the fixed co-resident grid) — never the arithmetic — so the
// output is token-id identical to HIPFIRE_MOE_DOWN_FUSED=1 separate dispatch.
//
// GRID-SYNC INVARIANT: there is NO early `return` anywhere. Every block runs all
// three phase loops (a block with no work for a phase simply does zero grid-
// stride iterations) so every block reaches every grid.sync(). An early return
// would hang the barrier.
//
// Shapes (decode, N=1):
//   expert_ptrs_gu : [n_exp]         gate_up per-expert weight base (ull ptr)
//   expert_ptrs_dn : [n_exp]         down    per-expert weight base (ull ptr)
//   topk_indices   : [K_TOP]         routed expert ids
//   topk_weights   : [K_TOP]         routed expert gate weights
//   x              : [D]             gate_up input activation (post norm+rotate)
//   signs1,signs2  : [256]           per-256-group FWHT sign vectors
//   y_gate,y_up    : [K_TOP x F]     scratch (phase-1 out / phase-2 in)
//   rot_batch      : [K_TOP x F]     scratch (phase-2 out / phase-3 in)
//   x_residual     : [D]             += combined routed output
// MQ4 group = 136 bytes (4 scale + 4 zp + 128 packed 4-bit), 256 elems/group.

// Phase-3 row-tile width. Must match the divisor the launcher uses for the
// phase-3 logical block count, and matches gemv_hfq4g256_moe_down_k8_indexed_
// fused_acc's MOE_DOWN_FUSED_NUM_ROWS (N=2 is the reuse-vs-occupancy optimum on
// gfx1201 a3b mq4r decode).
#ifndef MOE_MEGA_NUM_ROWS
#define MOE_MEGA_NUM_ROWS 2
#endif

__launch_bounds__(32, 16)
extern "C" __global__ void moe_megakernel_mq4_hfq4g256(
    const unsigned long long* __restrict__ expert_ptrs_gu, // [n_exp]
    const unsigned long long* __restrict__ expert_ptrs_dn, // [n_exp]
    const int*   __restrict__ topk_indices,                // [K_TOP]
    const float* __restrict__ topk_weights,                // [K_TOP]
    const float* __restrict__ x,                           // [D]
    const float* __restrict__ signs1,                      // [256]
    const float* __restrict__ signs2,                      // [256]
    float* __restrict__ y_gate,                            // [K_TOP x F] scratch
    float* __restrict__ y_up,                              // [K_TOP x F] scratch
    float* __restrict__ rot_batch,                         // [K_TOP x F] scratch
    float* __restrict__ x_residual,                        // [D] += result
    int D, int F, int K_TOP
) {
    cg::grid_group grid = cg::this_grid();
    const int tid  = threadIdx.x;
    const int pb   = blockIdx.x;   // physical block id
    const int nblk = gridDim.x;    // co-resident physical block count

    // ==================================================================
    // PHASE 1 — gate_up GEMV.  (verbatim from gemv_hfq4g256_moe_gate_up_k8_indexed)
    // Logical work: mi rows x K_TOP experts, mi = F.  input dim = D.
    // ==================================================================
    {
        const int mi = F;
        const int groups_per_row = D / 256;
        const int total = mi * K_TOP;
        const int boff = tid * 4;
        const int quads = groups_per_row >> 2;
        const int tail  = groups_per_row & 3;

        #define MEGA_LOAD_X8(b) \
            const float x0 = x[(b) + 0]; \
            const float x1 = x[(b) + 1]; \
            const float x2 = x[(b) + 2]; \
            const float x3 = x[(b) + 3]; \
            const float x4 = x[(b) + 4]; \
            const float x5 = x[(b) + 5]; \
            const float x6 = x[(b) + 6]; \
            const float x7 = x[(b) + 7]

        #define MEGA_DOG_X8(gp, a) do { \
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
                { MEGA_LOAD_X8(base0);
                  MEGA_DOG_X8(gate_ptr + g * 136, gacc0);
                  MEGA_DOG_X8(up_ptr   + g * 136, uacc0); }
                { MEGA_LOAD_X8(base0 + 256);
                  MEGA_DOG_X8(gate_ptr + (g + 1) * 136, gacc1);
                  MEGA_DOG_X8(up_ptr   + (g + 1) * 136, uacc1); }
                { MEGA_LOAD_X8(base0 + 512);
                  MEGA_DOG_X8(gate_ptr + (g + 2) * 136, gacc2);
                  MEGA_DOG_X8(up_ptr   + (g + 2) * 136, uacc2); }
                { MEGA_LOAD_X8(base0 + 768);
                  MEGA_DOG_X8(gate_ptr + (g + 3) * 136, gacc3);
                  MEGA_DOG_X8(up_ptr   + (g + 3) * 136, uacc3); }
            }
            if (tail >= 1) {
                const int g = quads << 2; int base = g * 256 + tid * 8;
                MEGA_LOAD_X8(base);
                MEGA_DOG_X8(gate_ptr + g * 136, gacc0);
                MEGA_DOG_X8(up_ptr   + g * 136, uacc0);
            }
            if (tail >= 2) {
                const int g = (quads << 2) + 1; int base = g * 256 + tid * 8;
                MEGA_LOAD_X8(base);
                MEGA_DOG_X8(gate_ptr + g * 136, gacc1);
                MEGA_DOG_X8(up_ptr   + g * 136, uacc1);
            }
            if (tail >= 3) {
                const int g = (quads << 2) + 2; int base = g * 256 + tid * 8;
                MEGA_LOAD_X8(base);
                MEGA_DOG_X8(gate_ptr + g * 136, gacc2);
                MEGA_DOG_X8(up_ptr   + g * 136, uacc2);
            }

            float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
            float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
            for (int offset = 16; offset > 0; offset >>= 1)
                gacc += __shfl_down(gacc, offset);
            for (int offset = 16; offset > 0; offset >>= 1)
                uacc += __shfl_down(uacc, offset);
            if (tid == 0) {
                y_gate[krank * mi + row] = gacc;
                y_up[krank * mi + row]   = uacc;
            }
        }
        #undef MEGA_DOG_X8
        #undef MEGA_LOAD_X8
    }

    grid.sync();

    // ==================================================================
    // PHASE 2 — fused SwiGLU + FWHT rotate.  (verbatim from fused_silu_mul_mq_rotate)
    // Logical work: (F/256) groups x K_TOP experts. batch dim = krank.
    // ==================================================================
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
            t = v0; v0 = v0 + v1; v1 = t - v1;
            t = v2; v2 = v2 + v3; v3 = t - v3;
            t = v4; v4 = v4 + v5; v5 = t - v5;
            t = v6; v6 = v6 + v7; v7 = t - v7;

            t = v0; v0 = v0 + v2; v2 = t - v2;
            t = v1; v1 = v1 + v3; v3 = t - v3;
            t = v4; v4 = v4 + v6; v6 = t - v6;
            t = v5; v5 = v5 + v7; v7 = t - v7;

            t = v0; v0 = v0 + v4; v4 = t - v4;
            t = v1; v1 = v1 + v5; v5 = t - v5;
            t = v2; v2 = v2 + v6; v6 = t - v6;
            t = v3; v3 = v3 + v7; v7 = t - v7;

            #define MEGA_HBFLY(v, pat, str) do { \
                float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
                if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
            } while(0)
            #define MEGA_HBFLY8(pat, str) \
                MEGA_HBFLY(v0,pat,str); MEGA_HBFLY(v1,pat,str); MEGA_HBFLY(v2,pat,str); MEGA_HBFLY(v3,pat,str); \
                MEGA_HBFLY(v4,pat,str); MEGA_HBFLY(v5,pat,str); MEGA_HBFLY(v6,pat,str); MEGA_HBFLY(v7,pat,str)
            MEGA_HBFLY8(0x041F, 1);
            MEGA_HBFLY8(0x081F, 2);
            MEGA_HBFLY8(0x101F, 4);
            MEGA_HBFLY8(0x201F, 8);
            MEGA_HBFLY8(0x401F, 16);
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

    // ==================================================================
    // PHASE 3 — down GEMV + K_TOP weighted-accumulate + residual add.
    // (verbatim from gemv_hfq4g256_moe_down_k8_indexed_fused_acc, N=1)
    // Logical work: ceil(D/NR) row-tiles. down input dim = F, output dim = D.
    // ==================================================================
    {
        constexpr int NR = MOE_MEGA_NUM_ROWS;
        const int total = (D + NR - 1) / NR;
        const int K = F;                        // down input dim
        const int groups_per_row = K / 256;
        const int boff  = tid * 4;
        const int quads = groups_per_row >> 2;
        const int tail  = groups_per_row & 3;

        #define MEGA_DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * xr[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * xr[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * xr[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * xr[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * xr[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * xr[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * xr[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * xr[(b) + 7]

        for (int lb = pb; lb < total; lb += nblk) {
            const int row_base = lb * NR;
            const int nrow = min(NR, D - row_base);

            float weighted_acc[NR];
            #pragma unroll
            for (int r = 0; r < NR; r++) weighted_acc[r] = 0.0f;

            for (int krank = 0; krank < K_TOP; krank++) {
                const int expert_id = topk_indices[krank];
                const char* __restrict__ A =
                    reinterpret_cast<const char*>(expert_ptrs_dn[expert_id]);
                const float* __restrict__ xr = rot_batch + (long long)krank * K;

                float acc0[NR], acc1[NR], acc2[NR], acc3[NR];
                #pragma unroll
                for (int r = 0; r < NR; r++) { acc0[r]=0.0f; acc1[r]=0.0f; acc2[r]=0.0f; acc3[r]=0.0f; }

                for (int q = 0; q < quads; q++) {
                    const int g = q << 2;
                    const int base = g * 256 + tid * 8;
                    #pragma unroll
                    for (int r = 0; r < NR; r++) {
                        const int rr = min(row_base + r, D - 1);
                        const char* row_ptr = A + (long long)rr * groups_per_row * 136;
                        const char* gp0 = row_ptr + g * 136;
                        const char* gp1 = gp0 + 136;
                        const char* gp2 = gp1 + 136;
                        const char* gp3 = gp2 + 136;
                        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
                        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
                        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
                        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
                        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
                        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
                        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
                        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));
                        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
                        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
                        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
                        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);
                        MEGA_DOG(pk0, sc0, zp0, base,       acc0[r]);
                        MEGA_DOG(pk1, sc1, zp1, base + 256, acc1[r]);
                        MEGA_DOG(pk2, sc2, zp2, base + 512, acc2[r]);
                        MEGA_DOG(pk3, sc3, zp3, base + 768, acc3[r]);
                    }
                }

                #pragma unroll
                for (int r = 0; r < NR; r++) {
                    const int rr = min(row_base + r, D - 1);
                    const char* row_ptr = A + (long long)rr * groups_per_row * 136;
                    if (tail >= 1) {
                        const int g = quads << 2; const char* gp = row_ptr + g * 136;
                        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
                        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
                        int base = g * 256 + tid * 8; MEGA_DOG(pk, sc, zp, base, acc0[r]);
                    }
                    if (tail >= 2) {
                        const int g = (quads << 2) + 1; const char* gp = row_ptr + g * 136;
                        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
                        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
                        int base = g * 256 + tid * 8; MEGA_DOG(pk, sc, zp, base, acc1[r]);
                    }
                    if (tail >= 3) {
                        const int g = (quads << 2) + 2; const char* gp = row_ptr + g * 136;
                        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
                        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
                        int base = g * 256 + tid * 8; MEGA_DOG(pk, sc, zp, base, acc2[r]);
                    }
                }

                const float w = topk_weights[krank];
                #pragma unroll
                for (int r = 0; r < NR; r++) {
                    float acc = (acc0[r] + acc1[r]) + (acc2[r] + acc3[r]);
                    for (int offset = 16; offset > 0; offset >>= 1)
                        acc += __shfl_down(acc, offset);
                    if (tid == 0) weighted_acc[r] += w * acc;
                }
            }
            #undef MEGA_DOG

            if (tid == 0) {
                #pragma unroll
                for (int r = 0; r < NR; r++) {
                    if (r < nrow)
                        x_residual[row_base + r] += weighted_acc[r];
                }
            }
        }
    }
}
