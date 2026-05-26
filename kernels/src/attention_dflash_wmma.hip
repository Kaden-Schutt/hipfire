// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// FlashAttention-style non-causal attention with WMMA (RDNA3 wave32).
//
// Companion to `attention_dflash_f32` for the **large-B / large-L** case
// where the scalar kernel's serial dot products become the dominant
// cost. The DFlash speculative-decoding path keeps using the scalar
// kernel — at B=8..16 we don't have a useful "B-tile of 16 queries"
// to feed WMMA, and the scalar variant is already memory-bound at
// short L. The dots.ocr vision encoder has B=L=N_patches ≈ 20k —
// plenty of B-axis for WMMA tiling.
//
// Computes: out[q, d] = (softmax(Q[q] · K[:]^T / √d) · V[:])[d]
// for all q in [0, B), all d in [0, head_dim).
//
// Layout (same as attention_dflash_f32):
//   Q [B,  n_heads,    head_dim]
//   K [L,  n_kv_heads, head_dim]
//   V [L,  n_kv_heads, head_dim]
//   out [B, n_heads,   head_dim]
//
// GQA via rep = n_heads / n_kv_heads. Constraint: head_dim ≤ 256 and
// head_dim a multiple of 16 (LDS budget + WMMA tile size).
//
// Grid:  [n_heads, ceil(B / 16), 1]
//        — one block per (head, 16-query-tile)
// Block: [32]  (one wave32 warp; all WMMA ops are wave-cooperative)
// LDS:   (3 * 16 * head_dim + 16*16 + 48) * 4 bytes
//        — ≈ 26 KB for head_dim=128, fits under the 64 KB RDNA3 budget.
//
// Per-block algorithm (FlashAttention-2 style):
//   1. Cooperatively load 16-row Q tile to LDS, init m_run/l_run/O_run.
//   2. For each K-tile of 16 keys (k_start = 0, 16, 32, …, ceil(L/16)):
//      a. S_tile = Q_tile @ K_tile^T              via WMMA  (head_dim/16 ops)
//      b. Scale S_tile, mask invalid k-positions to −inf.
//      c. Per-row online softmax (m_new, alpha, exp-in-place, l_new).
//      d. Load V_tile [16 k × head_dim] to LDS.
//      e. Alpha-scale O_run.
//      f. O_run += S_tile @ V_tile                via WMMA  (head_dim/16 ops)
//   3. out[q, d] = O_run[q, d] / l_run[q]
//
// V_tile is staged through LDS so the second WMMA can load V's
// **columns** (B[j, k] := V[k, j] per the WMMA contract D = A @ B^T,
// solving O[i, d] = Σ_k softmax_S[i, k] · V[k, d]). Loading V rows
// directly into the WMMA B operand (as a naive copy of the QK^T
// load pattern would) computes Σ_k softmax_S[i, k] · V[d, k] —
// wrong — so the explicit V LDS stage is load-bearing.
//
// WMMA register layout (RDNA3 wave32, F16 inputs, F32 accum):
//   Input  a/b: `half16_t` per lane.  Lane t in 0..15 holds row t of the
//               16×16 input tile (16 F16 elements). Lanes 16..31 mirror.
//   Output acc: `float8_t` per lane.  acc[j] at lane t maps to
//               D[2*j + (t>>4), t & 15] — even rows on lanes 0..15,
//               odd rows on lanes 16..31. See `gemm_qkv_hfq4g256_wmma.hip`.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

extern "C" __launch_bounds__(32, 4)
__global__ void attention_dflash_wmma_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads, int head_dim,
    float scale
) {
    const int head = blockIdx.x;
    const int qt = blockIdx.y;
    const int q_start = qt * 16;
    const int tid = threadIdx.x;

    if (head >= n_heads || q_start >= B) return;

    const int rep = n_heads / n_kv_heads;
    const int kv_head = head / rep;
    const int q_stride = n_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;

    const int lane = tid & 31;
    const int half = lane & 15;          // 0..15, lane-within-half
    const int half_id = lane >> 4;       // 0 (lanes 0..15) or 1 (lanes 16..31)

    // ─── LDS layout ──────────────────────────────────────────────────
    //   Q_lds  [16 * head_dim] — query tile, persists across K-tiles
    //   V_lds  [16 * head_dim] — V tile in [k, d] layout, per K-tile
    //   O_lds  [16 * head_dim] — running output, persists
    //   S_lds  [16 * 16]       — scores / softmax weights, per K-tile
    //   m_lds  [16]            — per-row running max
    //   l_lds  [16]            — per-row running denominator
    //   alpha_lds [16]         — per-row exp(m_old - m_new) per tile
    extern __shared__ float sdata[];
    float* Q_lds = sdata;
    float* V_lds = Q_lds + 16 * head_dim;
    float* O_lds = V_lds + 16 * head_dim;
    float* S_lds = O_lds + 16 * head_dim;
    float* m_lds = S_lds + 16 * 16;
    float* l_lds = m_lds + 16;
    float* alpha_lds = l_lds + 16;

    // ─── Init Q_lds, O_lds, m_lds, l_lds ─────────────────────────────
    // Two threads per row (32 threads / 16 rows = 2). Lanes 0..15 load
    // the first half of head_dim; lanes 16..31 load the second half.
    const int cols_per_half = head_dim / 2;
    const int my_col_base = half_id * cols_per_half;
    for (int row = 0; row < 16; ++row) {
        int gq = q_start + row;
        const float* q_row = (gq < B)
            ? (q + gq * q_stride + head * head_dim)
            : nullptr;
        #pragma unroll
        for (int d = 0; d < cols_per_half; ++d) {
            int idx = row * head_dim + my_col_base + d;
            Q_lds[idx] = q_row ? q_row[my_col_base + d] : 0.0f;
            O_lds[idx] = 0.0f;
        }
    }
    if (tid < 16) {
        m_lds[tid] = -INFINITY;
        l_lds[tid] = 0.0f;
    }
    __syncthreads();

    // ─── Loop over K-tiles ────────────────────────────────────────────
    for (int kt_start = 0; kt_start < L; kt_start += 16) {
        int kt_size = L - kt_start;
        if (kt_size > 16) kt_size = 16;

        const int my_k = kt_start + half;
        const bool k_valid = my_k < L;
        const float* my_k_row = k_valid
            ? (k + my_k * kv_stride + kv_head * head_dim)
            : nullptr;
        const float* my_v_row = k_valid
            ? (v + my_k * kv_stride + kv_head * head_dim)
            : nullptr;

        // ┌─── Phase A: S_tile = Q_tile @ K_tile^T  via WMMA ───────────┐
        //   A := Q row `half` (from Q_lds)
        //   B := K row `half` (from global, zero if my_k >= L)
        //   D := S[16, 16] accumulated over head_dim/16 chunks
        // └─────────────────────────────────────────────────────────────┘
        float8_t s_acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int d0 = 0; d0 < head_dim; d0 += 16) {
            half16_t a_reg;
            const float* q_row = Q_lds + half * head_dim + d0;
            #pragma unroll
            for (int j = 0; j < 16; ++j) a_reg[j] = (_Float16)q_row[j];

            half16_t b_reg;
            if (k_valid) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) b_reg[j] = (_Float16)my_k_row[d0 + j];
            } else {
                #pragma unroll
                for (int j = 0; j < 16; ++j) b_reg[j] = (_Float16)0.0f;
            }
            s_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, s_acc);
        }

        // Write S to LDS (apply scale + mask invalid K columns).
        //   s_acc[j] at lane t → S[2*j + (t>>4), t & 15]
        const int s_col = half;
        const bool col_valid = (kt_start + s_col) < L;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int s_row = 2 * j + half_id;
            float s_val = s_acc[j] * scale;
            if (!col_valid) s_val = -INFINITY;
            S_lds[s_row * 16 + s_col] = s_val;
        }

        // Stage V tile into LDS in natural [k, d] layout. We do this
        // here in parallel with the softmax computation below (the
        // softmax only touches lanes 0..15; lanes 16..31 cover V loads
        // for the back half). All 32 lanes write 8 floats each.
        const int v_writer_row = half;     // K-tile row
        for (int d = my_col_base; d < my_col_base + cols_per_half; ++d) {
            float vv = (k_valid && v_writer_row < kt_size)
                ? my_v_row[d]
                : 0.0f;
            V_lds[v_writer_row * head_dim + d] = vv;
        }
        __syncthreads();

        // ┌─── Phase B: per-row online softmax update ──────────────────┐
        if (tid < 16) {
            int row = tid;
            float* s_row = S_lds + row * 16;
            float tm = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 16; ++j) tm = fmaxf(tm, s_row[j]);

            float m_old = m_lds[row];
            float m_new = fmaxf(m_old, tm);
            float alpha = expf(m_old - m_new);   // first tile: m_old=-inf → alpha=0
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
        //   A := softmax(S) [16q × 16k] (constant across d-chunks)
        //   B[j, k] := V[k, d_chunk + j]  ←  lane j holds column j of V
        //              read as V_lds[k * head_dim + d_chunk + j]
        // └───────────────────────────────────────────────────────────────┘
        for (int i = tid; i < 16 * head_dim; i += 32) {
            int row = i / head_dim;
            O_lds[i] *= alpha_lds[row];
        }
        __syncthreads();

        // A row of softmax(S) is `half`'s row of S_lds.
        half16_t a_reg_sm;
        {
            const float* sm_row = S_lds + half * 16;
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

            // o_acc[j] at lane t → O[2*j + (t>>4), d0 + (t & 15)]
            const int o_col = d0 + half;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = 2 * j + half_id;
                O_lds[o_row * head_dim + o_col] += o_acc[j];
            }
        }
        __syncthreads();
    }

    // ─── Finalize: out = O_run / l_run ────────────────────────────────
    for (int i = tid; i < 16 * head_dim; i += 32) {
        int row = i / head_dim;
        int col = i % head_dim;
        int gq = q_start + row;
        if (gq < B) {
            float l = l_lds[row];
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            out[gq * q_stride + head * head_dim + col] = O_lds[i] * inv_l;
        }
    }
}
