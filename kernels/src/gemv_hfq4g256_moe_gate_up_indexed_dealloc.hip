// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// ── VGPR-dealloc variant of gemv_hfq4g256_moe_gate_up_k8_indexed ────────────
//
// Byte-identical arithmetic to `gemv_hfq4g256_moe_gate_up_indexed.hip`
// (verbatim inner loop + reduce + stores). The ONLY difference is a single
// scalar message at the very tail — after ALL VGPR use, including the
// global-store data/address reads (consumed at issue) — that releases this
// wave's VGPR allocation early:
//
//   s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
//
// Freeing the VGPR pool before `s_endpgm` lets the SIMD launch the next wave
// sooner → occupancy on exactly this VGPR-heavy (88/96) latency-bound MoE
// gate_up decode kernel. Pure scheduling optimization: no math changes,
// identical decode output vs the base kernel (value-preserving).
//
// Selected only when HIPFIRE_VGPR_DEALLOC=1 (default OFF); the default path
// keeps calling the unmodified `gemv_hfq4g256_moe_gate_up_k8_indexed`. The
// dealloc message opcode only exists on gfx11+/gfx12; on older RDNA archs the
// macro expands to nothing so this variant stays byte-identical to the base.

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__) || defined(__gfx1153__) || defined(__gfx1200__) || \
    defined(__gfx1201__)
#define HF_DEALLOC_VGPRS() __asm__ volatile("s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)")
#else
#define HF_DEALLOC_VGPRS() ((void)0)
#endif

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_gate_up_k8_indexed_dealloc(
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
    const int tail = groups_per_row & 3;

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

        {
            LOAD_X8(base0);
            DOG_X8(gate_ptr + g * 136, gacc0);
            DOG_X8(up_ptr + g * 136, uacc0);
        }
        {
            LOAD_X8(base0 + 256);
            DOG_X8(gate_ptr + (g + 1) * 136, gacc1);
            DOG_X8(up_ptr + (g + 1) * 136, uacc1);
        }
        {
            LOAD_X8(base0 + 512);
            DOG_X8(gate_ptr + (g + 2) * 136, gacc2);
            DOG_X8(up_ptr + (g + 2) * 136, uacc2);
        }
        {
            LOAD_X8(base0 + 768);
            DOG_X8(gate_ptr + (g + 3) * 136, gacc3);
            DOG_X8(up_ptr + (g + 3) * 136, uacc3);
        }
    }

    if (tail >= 1) {
        const int g = quads << 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(gate_ptr + g * 136, gacc0);
        DOG_X8(up_ptr + g * 136, uacc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(gate_ptr + g * 136, gacc1);
        DOG_X8(up_ptr + g * 136, uacc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(gate_ptr + g * 136, gacc2);
        DOG_X8(up_ptr + g * 136, uacc2);
    }
    #undef DOG_X8
    #undef LOAD_X8

    float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
    float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        gacc += __shfl_down(gacc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        uacc += __shfl_down(uacc, offset);

    if (tid == 0) {
        y_gate[krank * mi + row] = gacc;
        y_up[krank * mi + row] = uacc;
    }

    // Free the VGPR pool early — must be after every VGPR read above.
    HF_DEALLOC_VGPRS();
}
