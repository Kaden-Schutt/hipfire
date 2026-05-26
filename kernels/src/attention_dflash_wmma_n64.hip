// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// FlashAttention-style non-causal attention with WMMA (RDNA3 wave32),
// M=32 query tile, **N=64 K-tile**, Q kept register-resident across all
// K-tiles within a query block.
//
// Why: rocprof on the vision-encoder shape (B=L=19520, hd=128, n_heads=12)
// showed the M=32 baseline running at GPU_BUSY=100% with L2_HIT=0.8%.
// We are DRAM-bandwidth-bound, but the per-K-tile fixed cost (sync,
// alpha-scale of O, softmax, LDS Q reload) is what dominates because
// we visit each K-tile fixed cost 1220 times at N=16 versus 305 at N=64.
//
// Three changes over M=32 baseline:
//   1. N (K-tile width) 16 → 64. Outer loop trips: 1220 → 305.
//      `s_acc[4]` (one float8 per K-chunk) lives in registers across
//      the 4 inner K-chunks of a tile; no LDS staging of K.
//   2. Q register-resident: Q_frags[d_chunks] half16 per lane is loaded
//      once at kernel entry from global, reused across all K-tiles.
//      Eliminates 1220 × d_chunks LDS reads of Q per block.
//   3. Fused alpha-scale + SV epilogue: phase C reads O_lds, alpha-scales
//      in place, then adds o_acc — saves one full O_lds traversal and
//      one __syncthreads per K-tile.
//
// LDS budget at head_dim=128 (no Q_lds: Q is in registers):
//   V_lds[64*128] + O_lds[32*128] + S_lds[32*64] + m/l/alpha[32*3]
//   = (8192 + 4096 + 2048 + 96) * 4 B = 57.7 KB ≤ 64 KB cap.
//
// VGPR budget per lane (rough):
//   Q_frags[8] half16 = 64 VGPRs (head_dim=128)
//   s_acc[4]   float8 = 32 VGPRs
//   o_acc      float8 =  8 VGPRs scratch
//   a_reg/b_reg                = 16 VGPRs scratch
//   ≈ 130 VGPRs/lane, under the gfx1151 wave32 per-lane budget.
//
// Block shape: 2 waves × 32 = 64 threads. Wave 0 owns query rows
// 0..15; wave 1 owns rows 16..31. V_lds is shared across both waves.
//
// Grid:  [n_heads, ceil(B / 32), 1]
// Block: [64]

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

extern "C" __launch_bounds__(64, 2)
__global__ void attention_dflash_wmma_n64_f32(
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

    // Wave / lane split (matches M=32 baseline).
    const int wave = tid >> 5;             // 0 or 1
    const int lane = tid & 31;             // 0..31 within wave
    const int half = lane & 15;            // 0..15
    const int half_id = lane >> 4;         // 0 (lanes 0..15) or 1 (lanes 16..31)
    const int my_row_base = wave * 16;     // 0 or 16

    // d_chunks is hard-coded to 8 (head_dim=128) so the Q_frags array and
    // the dc loop become compile-time-bounded — without this, the compiler
    // puts Q_frags into private/scratch memory (544 B/lane) because the
    // loop index is runtime-variable, defeating the "Q in registers"
    // optimization. Dispatch enforces head_dim==128 for this kernel.
    constexpr int d_chunks = 8;
    if (head_dim != 128) return;

    // ─── LDS layout ──────────────────────────────────────────────────
    //   V_lds  [64 * head_dim] — 64 K-rows shared across both waves
    //   O_lds  [32 * head_dim] — running output, persists
    //   S_lds  [32 * 64]       — scores / softmax weights, per K-tile
    //   m_lds  [32]            — per-row running max
    //   l_lds  [32]            — per-row running denominator
    //   alpha_lds [32]         — per-row exp(m_old - m_new) per tile
    extern __shared__ float sdata[];
    float* V_lds = sdata;
    float* O_lds = V_lds + 64 * head_dim;
    float* S_lds = O_lds + 32 * head_dim;
    float* m_lds = S_lds + 32 * 64;
    float* l_lds = m_lds + 32;
    float* alpha_lds = l_lds + 32;

    // ─── Load Q into per-lane registers ──────────────────────────────
    // WMMA frag_a layout: in a wave, lane t (t=0..31) holds row (t mod 16)
    // of A. Lanes 0..15 and 16..31 within a wave both hold row `half`
    // (the WMMA hardware mirrors). For each d-chunk dc, lane holds
    // Q[my_row_base + half, dc*16 .. dc*16+15] cast to f16.
    //
    // At head_dim=128 that's 8 half16_t per lane = 64 VGPRs of Q.
    half16_t Q_frags[8];
    {
        int gq = q_start + my_row_base + half;
        const float* q_row = (gq < B)
            ? (q + gq * q_stride + head * head_dim)
            : nullptr;
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            int d0 = dc * 16;
            if (q_row) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) Q_frags[dc][j] = (_Float16)q_row[d0 + j];
            } else {
                #pragma unroll
                for (int j = 0; j < 16; ++j) Q_frags[dc][j] = (_Float16)0.0f;
            }
        }
    }

    // ─── Init O_lds, m_lds, l_lds ────────────────────────────────────
    const int cols_per_half = head_dim / 2;
    const int my_col_base = half_id * cols_per_half;
    for (int row = 0; row < 16; ++row) {
        #pragma unroll
        for (int d = 0; d < cols_per_half; ++d) {
            int idx = (my_row_base + row) * head_dim + my_col_base + d;
            O_lds[idx] = 0.0f;
        }
    }
    if (lane < 16) {
        m_lds[my_row_base + lane] = -INFINITY;
        l_lds[my_row_base + lane] = 0.0f;
    }
    __syncthreads();

    // ─── Loop over K-tiles of width 64 ───────────────────────────────
    for (int kt_start = 0; kt_start < L; kt_start += 64) {

        // ┌─── Phase A: S[16, 64] = Q[16, D] @ K[64, D]^T  via WMMA ────┐
        //   Each wave does its own 16 Q rows × 64 K cols → S_chunk[4].
        //   K is streamed inline from global (no LDS stage).
        //   Outer loop is over K-chunk (kc); inner is over d-chunk (dc)
        //   so that within a K-row visit, all d-chunks fire back-to-back
        //   (improves K-row L1/L2 locality).
        // └─────────────────────────────────────────────────────────────┘
        float8_t s_acc[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            s_acc[c] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }

        #pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
            int my_k = kt_start + kc * 16 + half;
            bool k_valid = my_k < L;
            const float* my_k_row = k_valid
                ? (k + my_k * kv_stride + kv_head * head_dim)
                : nullptr;

            #pragma unroll
            for (int dc = 0; dc < d_chunks; ++dc) {
                int d0 = dc * 16;
                half16_t a_reg = Q_frags[dc];

                half16_t b_reg;
                if (k_valid) {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) b_reg[j] = (_Float16)my_k_row[d0 + j];
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) b_reg[j] = (_Float16)0.0f;
                }
                s_acc[kc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, s_acc[kc]);
            }
        }

        // Write S to LDS, apply scale + mask invalid K columns.
        // s_acc[c][j] at lane t → S[my_row_base + 2*j + (t>>4), c*16 + (t & 15)]
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int s_col = c * 16 + half;
            bool col_valid = (kt_start + s_col) < L;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int s_row = my_row_base + 2 * j + half_id;
                float s_val = s_acc[c][j] * scale;
                if (!col_valid) s_val = -INFINITY;
                S_lds[s_row * 64 + s_col] = s_val;
            }
        }

        // Stage V into LDS in [k, d] layout. 64 K-rows × head_dim cols.
        // Each wave handles 32 contiguous K-rows. Within a wave, all 32
        // lanes load adjacent dwords across one row at a time
        // (coalesced; head_dim must be a multiple of 32).
        for (int kr_local = 0; kr_local < 32; ++kr_local) {
            int kr = wave * 32 + kr_local;
            int gk = kt_start + kr;
            bool valid = gk < L;
            const float* v_row = valid
                ? (v + gk * kv_stride + kv_head * head_dim)
                : nullptr;
            for (int d_base = 0; d_base < head_dim; d_base += 32) {
                int d = d_base + lane;
                float vv = v_row ? v_row[d] : 0.0f;
                V_lds[kr * head_dim + d] = vv;
            }
        }
        __syncthreads();

        // ┌─── Phase B: per-row online softmax update ─────────────────┐
        //   Each wave handles 16 of its own rows. Lanes 0..15 active;
        //   lanes 16..31 idle for this phase.
        // └─────────────────────────────────────────────────────────────┘
        if (lane < 16) {
            int row = my_row_base + lane;
            float* s_row = S_lds + row * 64;
            float tm = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 64; ++j) tm = fmaxf(tm, s_row[j]);

            float m_old = m_lds[row];
            float m_new = fmaxf(m_old, tm);
            float alpha = expf(m_old - m_new);
            alpha_lds[row] = alpha;

            float ts = 0.0f;
            #pragma unroll
            for (int j = 0; j < 64; ++j) {
                float e = expf(s_row[j] - m_new);
                s_row[j] = e;
                ts += e;
            }
            l_lds[row] = alpha * l_lds[row] + ts;
            m_lds[row] = m_new;
        }
        __syncthreads();

        // ┌─── Phase C: O = alpha * O + softmax(S) @ V via WMMA ───────┐
        //   Fused: alpha-scaling happens inline with the SV add per
        //   (row, col) write. Saves one __syncthreads + one full O_lds
        //   traversal vs the M=16/M=32 baseline.
        //
        //   For each d-chunk dc, accumulate o_acc over the 4 K-chunks
        //   c=0..3, then commit `O[row,col] = alpha*O[row,col] + o_acc`.
        //
        //   A row = softmax(S)[my_row_base + half, c*16 .. c*16+15]
        //   B row at lane half = V[c*16 + 0..15, d0 + half]
        // └─────────────────────────────────────────────────────────────┘
        #pragma unroll
        for (int dc = 0; dc < d_chunks; ++dc) {
            int d0 = dc * 16;
            float8_t o_acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int s_col_base = c * 16;

                half16_t a_reg_sm;
                {
                    const float* sm_row = S_lds + (my_row_base + half) * 64 + s_col_base;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) a_reg_sm[j] = (_Float16)sm_row[j];
                }

                half16_t b_reg;
                int my_d = d0 + half;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    b_reg[j] = (_Float16)V_lds[(s_col_base + j) * head_dim + my_d];
                }

                o_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg_sm, b_reg, o_acc);
            }

            // Commit: O[row, col] = alpha*O[row, col] + o_acc[j].
            int o_col = d0 + half;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int o_row = my_row_base + 2 * j + half_id;
                float alpha = alpha_lds[o_row];
                int o_idx = o_row * head_dim + o_col;
                O_lds[o_idx] = alpha * O_lds[o_idx] + o_acc[j];
            }
        }
        __syncthreads();
    }

    // ─── Finalize: out = O_run / l_run ───────────────────────────────
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
