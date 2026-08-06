// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Nine-path fused MoE gate_up (routed k=8, decode T=1).
//
// Replaces gemv_hfq4g256_moe_gate_up_k8_indexed, which launches one 32-thread
// block per (row, krank) — 4096 blocks each re-staging the 8KB activation
// into LDS. Here one 256-thread CTA stages x ONCE and all 8 routed-expert
// warps share it: warp w owns expert topk_indices[w], grid.x tiles the
// intermediate rows (RPB=8 rows per CTA → 64 CTAs at mi=512).
//
// BYTE-EXACT with gemv_hfq4g256_moe_gate_up_k8_indexed: same per-row
// 4-accumulator quad interleave, same tail distribution (tail group g goes
// to acc[g%4]), same pairwise combine ((a0+a1)+(a2+a3)), same shuffle-down
// tree, same x values (float4 LDS stage). Only the x residency and the
// grid/warp organization differ.
//
// Requirements: K <= 2048 (LDS stage; larger K reads x from global with
// identical values), mi % RPB == 0, blockDim == 256 (8 warps × 32).
// Wave-width safe: all shuffles are width-32 groups within the wave.

__launch_bounds__(256, 4)
extern "C" __global__ void gemv_hfq4g256_moe_ninepath_d3(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [8]
    const float* __restrict__ x,                         // [K]
    float* __restrict__ y_gate,                          // [8 × mi]
    float* __restrict__ y_up,                            // [8 × mi]
    int mi, int K
) {
    constexpr int RPB = 8;
    __shared__ float xs[2048];
    const int tid = threadIdx.x;
    const bool use_lds = (K > 0 && K <= 2048);
    if (use_lds) {
        float4* __restrict__ xs4 = reinterpret_cast<float4*>(xs);
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
        const int n4 = K >> 2;
        for (int i = tid; i < n4; i += 256) xs4[i] = x4[i];
        __syncthreads();
    }
    const float* __restrict__ xsrc = use_lds ? xs : x;

    const int warp = tid >> 5;   // 0..7 = krank
    const int lane = tid & 31;
    const int expert_id = topk_indices[warp];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const int groups_per_row = K / 256;
    const int r0 = blockIdx.x * RPB;
    const int boff = lane * 4;

    #define LOAD_X8_INTO(b, pfx) \
        const float4 pfx##v0 = *reinterpret_cast<const float4*>(xsrc + (b)); \
        const float4 pfx##v1 = *reinterpret_cast<const float4*>(xs + (b) + 4); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w

    #define DOG_X8(gp, a, pfx) do { \
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp)); \
        float zp = __builtin_bit_cast(float, *(const unsigned int*)((gp) + 4)); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * pfx##0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * pfx##1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * pfx##2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * pfx##3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * pfx##4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * pfx##5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * pfx##6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * pfx##7; \
    } while (0)

    #define DOG_X8_TAIL(gp, a, pfx) DOG_X8(gp, a, pfx)

    for (int r = 0; r < RPB; r++) {
        const int row = r0 + r;
        const char* gate_ptr = A + (long long)row * groups_per_row * 136;
        const char* up_ptr = A + (long long)(row + mi) * groups_per_row * 136;

        float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
        float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;
        const int quads = groups_per_row >> 2;
        const int tail = groups_per_row & 3;

        for (int q = 0; q < quads; q++) {
            const int g = q << 2;
            const int base0 = g * 256 + lane * 8;

            LOAD_X8_INTO(base0, xa);
            LOAD_X8_INTO(base0 + 256, xb);
            LOAD_X8_INTO(base0 + 512, xc);
            LOAD_X8_INTO(base0 + 768, xd);

            DOG_X8(gate_ptr + g * 136, gacc0, xa);
            DOG_X8(up_ptr + g * 136, uacc0, xa);
            DOG_X8(gate_ptr + (g + 1) * 136, gacc1, xb);
            DOG_X8(up_ptr + (g + 1) * 136, uacc1, xb);
            DOG_X8(gate_ptr + (g + 2) * 136, gacc2, xc);
            DOG_X8(up_ptr + (g + 2) * 136, uacc2, xc);
            DOG_X8(gate_ptr + (g + 3) * 136, gacc3, xd);
            DOG_X8(up_ptr + (g + 3) * 136, uacc3, xd);
        }

        if (tail >= 1) {
            const int g = quads << 2;
            LOAD_X8_INTO(g * 256 + lane * 8, xt0);
            DOG_X8_TAIL(gate_ptr + g * 136, gacc0, xt0);
            DOG_X8_TAIL(up_ptr + g * 136, uacc0, xt0);
        }
        if (tail >= 2) {
            const int g = (quads << 2) + 1;
            LOAD_X8_INTO(g * 256 + lane * 8, xt1);
            DOG_X8_TAIL(gate_ptr + g * 136, gacc1, xt1);
            DOG_X8_TAIL(up_ptr + g * 136, uacc1, xt1);
        }
        if (tail >= 3) {
            const int g = (quads << 2) + 2;
            LOAD_X8_INTO(g * 256 + lane * 8, xt2);
            DOG_X8_TAIL(gate_ptr + g * 136, gacc2, xt2);
            DOG_X8_TAIL(up_ptr + g * 136, uacc2, xt2);
        }

        float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
        float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
        for (int offset = 16; offset > 0; offset >>= 1)
            gacc += __shfl_down(gacc, offset);
        for (int offset = 16; offset > 0; offset >>= 1)
            uacc += __shfl_down(uacc, offset);

        if (lane == 0) {
            y_gate[warp * mi + row] = gacc;
            y_up[warp * mi + row] = uacc;
        }
    }

    #undef DOG_X8_TAIL
    #undef DOG_X8
    #undef LOAD_X8_INTO
}
