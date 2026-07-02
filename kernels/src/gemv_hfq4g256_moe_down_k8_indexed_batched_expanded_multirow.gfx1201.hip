// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1201 (RDNA4) HFQ4-G256 multi-row variant of the atomic-free indexed,
// batched, expanded MoE-down GEMV: one warp computes R adjacent output
// ROWS for the SAME (token, krank) — i.e. the same routed expert and the
// same rotated activation vector `x`. That is exactly the case the base
// `gemv_hfq4g256_multirow*` mechanism was built for: hoist the x-vector
// load once per group-quad and reuse it across R rows, growing the
// per-block contiguous A (weight) traffic from 272 B (A3B moe_down's
// K=512 tail-only shape) to R×272 B.
//
// ───────────────────────────────────────────────────────────────────────────
// STATUS: opt-in via HIPFIRE_GEMV_ROWS=2 (bounded R=2 spike — see
// docs/gfx1201-native-surface.md "moe_down re-tile check + Phase-B final
// decision"). Default (HIPFIRE_GEMV_ROWS unset / =1) keeps the single-row
// `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` kernel — NO default
// behavior change. Only R=2 is instantiated here; R=4/R=8 are left for a
// follow-on if R=2 clears the VGPR/occupancy + on-device bench bar.
//
// Numerics: byte-identical accumulator order and tail handling to
// `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hip` — same 4-way
// group-quad interleaving, same tail-into-acc[r][g%4] discipline (the
// bug class from commit 5302926: dumping all tail groups into acc[r][0]
// silently breaks MQ4 output when groups_per_row isn't a multiple of 4).
// The expert gather (`topk_indices[bid*K_TOP+krank]` → `expert_ptrs[...]`)
// and the per-(token,krank) expanded output layout
// (`expert_outputs[bid][krank][row]`) are untouched — only the row/grid
// tiling changes.
// ───────────────────────────────────────────────────────────────────────────
//
// Grid: (ceil(M/R), K_TOP, batch). Block = 32 threads. Each block reads
// ONE expert pointer + ONE x row (shared across its R output rows) and
// writes R output cells, each to its own row of `expert_outputs` — no
// atomics, same as the base kernel.

template<int R>
__device__ __forceinline__ void gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_multirow_body(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M]
    int M, int K, int K_TOP
) {
    const int row_base = blockIdx.x * R;
    if (row_base >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    // Same (token, krank) → same routed expert + same x row for all R
    // output rows in this block. Gathered/hoisted exactly once per block.
    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const int boff = tid * 4;

    // Row pointers within the shared expert's weight matrix. For partial
    // final workgroups (row_base + r >= M) clamp to row_base so reads stay
    // in-bounds; the write is masked separately by the row_valid check.
    const char* row_ptrs[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        int row = row_base + r;
        int safe_row = (row < M) ? row : row_base;
        row_ptrs[r] = A + (long long)safe_row * row_stride;
    }

    float acc[R][4];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
        acc[r][2] = 0.0f;
        acc[r][3] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // Main loop: 4 groups per iteration.
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base = g * 256 + tid * 8;

        // Hoist x loads — identical across all R rows (same token/krank).
        const float x00 = x[base + 0],   x01 = x[base + 1],
                    x02 = x[base + 2],   x03 = x[base + 3],
                    x04 = x[base + 4],   x05 = x[base + 5],
                    x06 = x[base + 6],   x07 = x[base + 7];
        const float x10 = x[base + 256 + 0], x11 = x[base + 256 + 1],
                    x12 = x[base + 256 + 2], x13 = x[base + 256 + 3],
                    x14 = x[base + 256 + 4], x15 = x[base + 256 + 5],
                    x16 = x[base + 256 + 6], x17 = x[base + 256 + 7];
        const float x20 = x[base + 512 + 0], x21 = x[base + 512 + 1],
                    x22 = x[base + 512 + 2], x23 = x[base + 512 + 3],
                    x24 = x[base + 512 + 4], x25 = x[base + 512 + 5],
                    x26 = x[base + 512 + 6], x27 = x[base + 512 + 7];
        const float x30 = x[base + 768 + 0], x31 = x[base + 768 + 1],
                    x32 = x[base + 768 + 2], x33 = x[base + 768 + 3],
                    x34 = x[base + 768 + 4], x35 = x[base + 768 + 5],
                    x36 = x[base + 768 + 6], x37 = x[base + 768 + 7];

        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp0 = row_ptrs[r] + g * 136;
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

            acc[r][0] += (sc0 * (float)((pk0 >>  0) & 0xFu) + zp0) * x00
                       + (sc0 * (float)((pk0 >>  4) & 0xFu) + zp0) * x01
                       + (sc0 * (float)((pk0 >>  8) & 0xFu) + zp0) * x02
                       + (sc0 * (float)((pk0 >> 12) & 0xFu) + zp0) * x03
                       + (sc0 * (float)((pk0 >> 16) & 0xFu) + zp0) * x04
                       + (sc0 * (float)((pk0 >> 20) & 0xFu) + zp0) * x05
                       + (sc0 * (float)((pk0 >> 24) & 0xFu) + zp0) * x06
                       + (sc0 * (float)((pk0 >> 28) & 0xFu) + zp0) * x07;
            acc[r][1] += (sc1 * (float)((pk1 >>  0) & 0xFu) + zp1) * x10
                       + (sc1 * (float)((pk1 >>  4) & 0xFu) + zp1) * x11
                       + (sc1 * (float)((pk1 >>  8) & 0xFu) + zp1) * x12
                       + (sc1 * (float)((pk1 >> 12) & 0xFu) + zp1) * x13
                       + (sc1 * (float)((pk1 >> 16) & 0xFu) + zp1) * x14
                       + (sc1 * (float)((pk1 >> 20) & 0xFu) + zp1) * x15
                       + (sc1 * (float)((pk1 >> 24) & 0xFu) + zp1) * x16
                       + (sc1 * (float)((pk1 >> 28) & 0xFu) + zp1) * x17;
            acc[r][2] += (sc2 * (float)((pk2 >>  0) & 0xFu) + zp2) * x20
                       + (sc2 * (float)((pk2 >>  4) & 0xFu) + zp2) * x21
                       + (sc2 * (float)((pk2 >>  8) & 0xFu) + zp2) * x22
                       + (sc2 * (float)((pk2 >> 12) & 0xFu) + zp2) * x23
                       + (sc2 * (float)((pk2 >> 16) & 0xFu) + zp2) * x24
                       + (sc2 * (float)((pk2 >> 20) & 0xFu) + zp2) * x25
                       + (sc2 * (float)((pk2 >> 24) & 0xFu) + zp2) * x26
                       + (sc2 * (float)((pk2 >> 28) & 0xFu) + zp2) * x27;
            acc[r][3] += (sc3 * (float)((pk3 >>  0) & 0xFu) + zp3) * x30
                       + (sc3 * (float)((pk3 >>  4) & 0xFu) + zp3) * x31
                       + (sc3 * (float)((pk3 >>  8) & 0xFu) + zp3) * x32
                       + (sc3 * (float)((pk3 >> 12) & 0xFu) + zp3) * x33
                       + (sc3 * (float)((pk3 >> 16) & 0xFu) + zp3) * x34
                       + (sc3 * (float)((pk3 >> 20) & 0xFu) + zp3) * x35
                       + (sc3 * (float)((pk3 >> 24) & 0xFu) + zp3) * x36
                       + (sc3 * (float)((pk3 >> 28) & 0xFu) + zp3) * x37;
        }
    }

    // Tail: per-row, accumulate into acc[r][gi] to preserve the 4-way
    // interleaving (same reason as the single-row kernel — 5302926).
    #define TAIL_LOAD_X(b) \
        const float txv0 = x[(b) + 0], txv1 = x[(b) + 1], \
                    txv2 = x[(b) + 2], txv3 = x[(b) + 3], \
                    txv4 = x[(b) + 4], txv5 = x[(b) + 5], \
                    txv6 = x[(b) + 6], txv7 = x[(b) + 7]
    #define TAIL_DOG(pk, sc, zp, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * txv0  \
             + (sc * (float)(((pk) >>  4) & 0xFu) + zp) * txv1 \
             + (sc * (float)(((pk) >>  8) & 0xFu) + zp) * txv2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * txv3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * txv4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * txv5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * txv6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * txv7

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            TAIL_DOG(pk, sc, zp, acc[r][0]);
        }
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            TAIL_DOG(pk, sc, zp, acc[r][1]);
        }
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            TAIL_DOG(pk, sc, zp, acc[r][2]);
        }
    }
    #undef TAIL_LOAD_X
    #undef TAIL_DOG

    // Per-row warp reduction and direct (non-atomic) write-back into the
    // expanded [N × K_TOP × M] output — same layout as the single-row
    // kernel, expert_outputs[bid][krank][row].
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float s = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        int row = row_base + r;
        if (tid == 0 && row < M) {
            const long long out_offset =
                ((long long)bid * K_TOP + krank) * (long long)M + row;
            expert_outputs[out_offset] = s;
        }
    }
}

// R=2 only (bounded spike scope). Distinct module name (this file compiles
// into its own module, see kernels.rs) AND a `_gfx1201` symbol suffix so a
// future arch-general port can't silently shadow this entry point under a
// shared kernel-cache module name (see commit 5302926 / the multirow
// kernel-cache-collision lesson).
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_multirow_r2_gfx1201(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float*       __restrict__ expert_outputs,
    int M, int K, int K_TOP
) {
    gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_multirow_body<2>(
        expert_ptrs, topk_indices, rot_batch, expert_outputs, M, K, K_TOP);
}
