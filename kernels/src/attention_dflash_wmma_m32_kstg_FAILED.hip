// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// ═════════════════════════════════════════════════════════════════════
// LABELED-FAILED EXPERIMENT — DO NOT WIRE THROUGH THE DISPATCH PATH.
// Kept in-tree as a reference point for the rocprof + source-review
// follow-on. See `docs/plans/dots-ocr-devlog.md` §5 "Speedup roadmap"
// for the diagnosis trail.
//
// Result on smoke image (gfx1151, ROCm 7.13, N=B=L=19520, hd=128):
//   * Without K-staging (production M=32 kernel):       2881 ms / block
//   * With K-staging, UNPADDED K_lds[16*hd]:            5141 ms / block
//   * With K-staging, PADDED   K_lds[16*(hd+1)] (below): 4721 ms / block
//
// The padded variant below is the better of two failure modes, still
// a clear regression vs the strided-global-load baseline.
//
// What went wrong (best current hypothesis, pending rocprof):
//   - Per-block LDS went 43 → 50 KB, but occupancy was *already*
//     1 block / CU at 43 KB so no parallelism was lost.
//   - Bank conflicts in phase A's WMMA b_reg loads were a real
//     contributor (16-way at unpadded stride; +1 pad fixed those).
//     But the fix only recovered ~8% — still a regression.
//   - Likely real cause: the strided global K-loads in the baseline
//     hit L2 better than expected (RDNA3 may coalesce / prefetch in
//     ways we don't have insight into), so the "uncoalesced K
//     traffic" diagnosis from the M=32 commit message was
//     incomplete. The K-staging adds a __syncthreads + cooperative-
//     load loop + bank-conflict-free LDS reads, but the original
//     path was apparently faster on net.
//
// Reach for rocprof BEFORE attempting another hypothesis-driven
// kernel rewrite.
// ═════════════════════════════════════════════════════════════════════

// FlashAttention-style non-causal attention with WMMA (RDNA3 wave32),
// M=32 variant of `attention_dflash_wmma_f32` + cooperative K-staging
// into LDS (failed experiment — see header above).
//
// Why: at vision-encoder shapes (B=L=N_patches≈20k, head_dim=128, 12
// heads) the M=16 variant is ~25× over the gfx1151 FMA floor — bound
// by global-memory K traffic, not compute. Each K-tile of 16 keys is
// read once per query-tile block, so total K reads scale as
// `(B/M) * (L/16) * n_heads`. Doubling M halves the per-K-tile
// query-tile count → halves global K reads → ~2× speedup at large N.
//
// LDS budget at head_dim=128:
//   M=16:  Q[16×128] + V[16×128] + O[16×128] + S[16×16] + 48f = 25.8 KB
//   M=32:  Q[32×128] + K[16×128] + V[16×128] + O[32×128] + S[32×16] + 96f = 50.0 KB
//   M=64:  would exceed 64 KB cap.
// M=32 is the maximum that fits the 64 KB RDNA3 wave32 SLM cap with the
// full Q_lds + K_lds + O_lds + V_lds layout we use.
//
// K_lds is staged via coalesced cooperative loads at the top of each
// K-tile iteration: per-wave, 8 rows × 4 cacheline-chunks, where all
// 32 lanes within a wave read 32 consecutive floats (one cacheline)
// per chunk. Phase A then reads K from LDS instead of global. The
// uncoalesced pattern from the original M=16 kernel (each lane reads
// from its own K row at kv_stride apart) was the actual bottleneck —
// effective K bandwidth was ~0.7 GB/s vs gfx1151's 115 GB/s peak.
//
// Block shape: 2 waves (64 threads). wave 0 owns query rows 0..15;
// wave 1 owns rows 16..31. Each wave does its own WMMA on its own Q
// rows and its own S rows; V_lds is shared. K is read per-wave from
// global; the L2 cache coalesces the two waves' identical K loads, so
// effective DRAM K traffic is one fetch per K-tile per block.
//
// Algorithm: identical to `attention_dflash_wmma_f32`, duplicated
// per-wave on the M-axis. See that kernel's header for the
// FlashAttention-2 per-K-tile loop description.
//
// Grid:  [n_heads, ceil(B / 32), 1]
// Block: [64]
// LDS:   (2*32*head_dim + 16*head_dim + 32*16 + 3*32) * 4 bytes
//        ≈ 43 KB for head_dim=128.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

extern "C" __launch_bounds__(64, 2)
__global__ void attention_dflash_wmma_m32_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale
) {
    const int head = blockIdx.x;
    const int qt = blockIdx.y;
    const int q_start = qt * 32;
    const int tid = threadIdx.x;

    if (head >= n_heads || q_start >= B) return;

    const int rep = n_heads / n_kv_heads;
    const int kv_head = head / rep;
    const int q_stride = n_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;

    // Wave / lane split. Two waves per block: wave 0 (tid 0..31) owns
    // M-rows 0..15, wave 1 (tid 32..63) owns M-rows 16..31. Within each
    // wave the lane convention matches the M=16 kernel exactly.
    const int wave = tid >> 5;             // 0 or 1
    const int lane = tid & 31;             // 0..31 within wave
    const int half = lane & 15;            // 0..15, lane-within-half
    const int half_id = lane >> 4;         // 0 (lanes 0..15) or 1 (lanes 16..31)
    const int my_row_base = wave * 16;     // wave 0 → 0, wave 1 → 16

    // ─── LDS layout ──────────────────────────────────────────────────
    //   Q_lds  [32 * head_dim]       — query tile, persists across K-tiles
    //   K_lds  [16 * (head_dim + 1)] — K tile; the +1 column pad breaks
    //                                  LDS bank conflicts in phase A's
    //                                  WMMA reads. Without the pad,
    //                                  row-stride=head_dim=128 floats =
    //                                  16 banks, so all 16 active lanes
    //                                  hit bank 0 simultaneously (16-way
    //                                  conflict, 16× slowdown on K_lds
    //                                  reads). +1 → row-stride=129,
    //                                  gcd(129,32)=1 → conflict-free.
    //   V_lds  [16 * head_dim] — V tile in [k, d] layout, shared by both waves
    //   O_lds  [32 * head_dim] — running output, persists
    //   S_lds  [32 * 16]       — scores / softmax weights, per K-tile
    //   m_lds  [32]            — per-row running max
    //   l_lds  [32]            — per-row running denominator
    //   alpha_lds [32]         — per-row exp(m_old - m_new) per tile
    const int k_stride = head_dim + 1;
    extern __shared__ float sdata[];
    float* Q_lds = sdata;
    float* K_lds = Q_lds + 32 * head_dim;
    float* V_lds = K_lds + 16 * k_stride;
    float* O_lds = V_lds + 16 * head_dim;
    float* S_lds = O_lds + 32 * head_dim;
    float* m_lds = S_lds + 32 * 16;
    float* l_lds = m_lds + 32;
    float* alpha_lds = l_lds + 32;

    // ─── Init Q_lds, O_lds, m_lds, l_lds ─────────────────────────────
    // Each wave loads its own 16 query rows. Within a wave, two lanes
    // per row (32 / 16 = 2): lanes 0..15 load first half of head_dim,
    // lanes 16..31 load second half. Identical layout to the M=16 kernel
    // but offset by `my_row_base` rows into Q_lds / O_lds.
    const int cols_per_half = head_dim / 2;
    const int my_col_base = half_id * cols_per_half;
    for (int row = 0; row < 16; ++row) {
        int gq = q_start + my_row_base + row;
        const float* q_row = (gq < B)
            ? (q + gq * q_stride + head * head_dim)
            : nullptr;
        #pragma unroll
        for (int d = 0; d < cols_per_half; ++d) {
            int idx = (my_row_base + row) * head_dim + my_col_base + d;
            Q_lds[idx] = q_row ? q_row[my_col_base + d] : 0.0f;
            O_lds[idx] = 0.0f;
        }
    }
    // Lanes 0..15 of wave 0 init m/l rows 0..15; lanes 0..15 of wave 1
    // init rows 16..31. Lanes 16..31 of both waves are idle for this
    // tiny init.
    if (lane < 16) {
        m_lds[my_row_base + lane] = -INFINITY;
        l_lds[my_row_base + lane] = 0.0f;
    }
    __syncthreads();

    // ─── Loop over K-tiles ────────────────────────────────────────────
    for (int kt_start = 0; kt_start < L; kt_start += 16) {
        int kt_size = L - kt_start;
        if (kt_size > 16) kt_size = 16;

        // ┌─── Stage K-tile into K_lds (coalesced cooperative load) ────┐
        // Per-wave layout: each wave handles 8 K rows. wave 0 → rows
        // 0..7, wave 1 → rows 8..15. Within a wave, the 32 lanes process
        // one row at a time over 4 cacheline-chunks (head_dim=128 / 32
        // = 4). All 32 lanes in a wave read 32 consecutive floats per
        // chunk — one full cacheline transaction per global fetch.
        //
        // Generalised: cacheline_chunks_per_row = head_dim / 32. Loops
        // unrolled because head_dim is a compile-time constant for the
        // shapes we care about.
        const int chunks_per_row = head_dim / 32;
        const int rows_per_wave = 8;
        for (int row_off = 0; row_off < rows_per_wave; ++row_off) {
            const int kr = wave * rows_per_wave + row_off;
            const int gk = kt_start + kr;
            const bool kr_valid = gk < L;
            const float* k_load_src = kr_valid
                ? (k + gk * kv_stride + kv_head * head_dim)
                : nullptr;
            #pragma unroll
            for (int chunk = 0; chunk < 8; ++chunk) {           // upper bound; runtime check
                if (chunk >= chunks_per_row) break;
                const int col = chunk * 32 + lane;
                K_lds[kr * k_stride + col] = kr_valid ? k_load_src[col] : 0.0f;
            }
        }
        // Note: no __syncthreads here — phase A reads each wave's own
        // rows of K_lds (wave 0 reads rows 0..15? actually NO — WMMA's
        // `b_reg` for both waves reads K_lds[half * head_dim + d0..],
        // i.e. rows 0..15. wave 1 needs rows 8..15 (written by itself)
        // AND rows 0..7 (written by wave 0). So we DO need a sync.
        __syncthreads();

        // ┌─── Phase A: S_tile = Q_tile @ K_tile^T  via WMMA ───────────┐
        //   Each wave does its own WMMA on its own 16 Q rows × shared K_lds.
        //   A := Q row `half` from Q_lds (offset by my_row_base)
        //   B := K row `half` from K_lds (filled above)
        //   D := S[16, 16] accumulated over head_dim/16 chunks
        // └─────────────────────────────────────────────────────────────┘
        float8_t s_acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int d0 = 0; d0 < head_dim; d0 += 16) {
            half16_t a_reg;
            const float* q_row = Q_lds + (my_row_base + half) * head_dim + d0;
            #pragma unroll
            for (int j = 0; j < 16; ++j) a_reg[j] = (_Float16)q_row[j];

            half16_t b_reg;
            const float* k_row_lds = K_lds + half * k_stride + d0;
            #pragma unroll
            for (int j = 0; j < 16; ++j) b_reg[j] = (_Float16)k_row_lds[j];
            s_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, s_acc);
        }

        // Write S to LDS (apply scale + mask invalid K columns).
        //   s_acc[j] at lane t → S[my_row_base + 2*j + (t>>4), t & 15]
        const int s_col = half;
        const bool col_valid = (kt_start + s_col) < L;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int s_row = my_row_base + 2 * j + half_id;
            float s_val = s_acc[j] * scale;
            if (!col_valid) s_val = -INFINITY;
            S_lds[s_row * 16 + s_col] = s_val;
        }

        // Stage V into LDS in [k, d] layout. Only wave 0 does this work
        // (V_lds is shared; both waves read it in phase C). Lanes 0..15
        // of wave 0 cover the first half of head_dim across all 16 K
        // rows; lanes 16..31 of wave 0 cover the second half.
        if (wave == 0) {
            const int v_writer_row = half;
            const float* my_v_row = (kt_start + v_writer_row) < L
                ? (v + (kt_start + v_writer_row) * kv_stride + kv_head * head_dim)
                : nullptr;
            for (int d = my_col_base; d < my_col_base + cols_per_half; ++d) {
                float vv = (my_v_row && v_writer_row < kt_size)
                    ? my_v_row[d]
                    : 0.0f;
                V_lds[v_writer_row * head_dim + d] = vv;
            }
        }
        __syncthreads();

        // ┌─── Phase B: per-row online softmax update ──────────────────┐
        // Each wave handles 16 rows of its own. Lanes 0..15 of wave 0
        // process rows 0..15; lanes 0..15 of wave 1 process rows 16..31.
        if (lane < 16) {
            int row = my_row_base + lane;
            float* s_row = S_lds + row * 16;
            float tm = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 16; ++j) tm = fmaxf(tm, s_row[j]);

            float m_old = m_lds[row];
            float m_new = fmaxf(m_old, tm);
            float alpha = expf(m_old - m_new);
            alpha_lds[row] = alpha;

            float ts = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                float e = expf(s_row[j] - m_new);
                s_row[j] = e;
                ts += e;
            }
            l_lds[row] = alpha * l_lds[row] + ts;
            m_lds[row] = m_new;
        }
        __syncthreads();

        // ┌─── Phase C: O_run = alpha * O_run + softmax(S) @ V  via WMMA ─┐
        //   Each wave does its own WMMA on its own 16 S rows × shared V.
        //   A := softmax(S) [16q × 16k] (constant across d-chunks)
        //   B[j, k] := V[k, d_chunk + j]  ←  lane j holds column j of V
        //              read as V_lds[k * head_dim + d_chunk + j]
        // └───────────────────────────────────────────────────────────────┘
        // Scale own 16 O rows by alpha. Each wave handles its own rows.
        for (int i = lane; i < 16 * head_dim; i += 32) {
            int row = my_row_base + (i / head_dim);
            int col = i % head_dim;
            O_lds[row * head_dim + col] *= alpha_lds[row];
        }
        __syncthreads();

        // A row of softmax(S) for this wave is `(my_row_base + half)`'s
        // row of S_lds.
        half16_t a_reg_sm;
        {
            const float* sm_row = S_lds + (my_row_base + half) * 16;
            #pragma unroll
            for (int j = 0; j < 16; ++j) a_reg_sm[j] = (_Float16)sm_row[j];
        }

        for (int d0 = 0; d0 < head_dim; d0 += 16) {
            // B row `half` is V's column (d0 + half), across 16 K positions.
            half16_t b_reg;
            int my_d = d0 + half;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                b_reg[j] = (_Float16)V_lds[j * head_dim + my_d];
            }

            float8_t o_acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            o_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg_sm, b_reg, o_acc);

            // o_acc[j] at lane t → O[my_row_base + 2*j + (t>>4), d0 + (t & 15)]
            const int o_col = d0 + half;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = my_row_base + 2 * j + half_id;
                O_lds[o_row * head_dim + o_col] += o_acc[j];
            }
        }
        __syncthreads();
    }

    // ─── Finalize: out = O_run / l_run ────────────────────────────────
    // Each wave writes its own 16 query rows.
    for (int i = lane; i < 16 * head_dim; i += 32) {
        int row = my_row_base + (i / head_dim);
        int col = i % head_dim;
        int gq = q_start + row;
        if (gq < B) {
            float l = l_lds[row];
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            out[gq * q_stride + head * head_dim + col] = O_lds[row * head_dim + col] * inv_l;
        }
    }
}
