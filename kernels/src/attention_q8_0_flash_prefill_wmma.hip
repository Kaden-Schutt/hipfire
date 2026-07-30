// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Query-tiled Q8_0 flash prefill attention on RDNA3 WMMA (wave32).
//
// Stage C of the flash-prefill work. Same algorithm as
// attention_q8_0_flash_prefill (query tiling + online softmax, LDS
// independent of context) but the QK^T and P·V inner products run on the
// matrix cores instead of scalar VALU FMAs.
//
// Why this is not a port of the attention_dflash_wmma_* family: every one of
// those kernels hard-guards head_dim == 128, and the single head_dim-generic
// variant (attention_dflash_wmma_m32) stages Q/V/O in LDS sized by head_dim,
// which needs ~84 KB at head_dim=256 — over the 64 KB limit. This kernel fits
// by (a) storing everything in LDS as f16 rather than f32, and (b) never
// staging K in LDS at all: each B fragment is dequantised straight out of the
// Q8_0 cache in global memory, exactly as the v6 f16kv exemplar reads its f16
// K rows.
//
// The Q8 dequant cost is amortised by the matrix layout itself: one WMMA B
// fragment (16 keys × 16 dims) is reused across all 16 query rows of the A
// fragment, so each cache byte is converted once per 16 queries instead of
// once per query as in the scalar kernel.
//
// Grid:  [ceil(batch_size / M_TILE), n_heads]
// Block: 32 threads (one wave32).
//
// LDS (head_dim=256): Q 16×256 f16 = 8192 | V^T 256×18 f16 = 9216
//                     | S 16×18 f16 = 576 | m,l,alpha 16×3 f32 = 192
//                     ≈ 18.0 KB → 3 workgroups/CU.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define M_TILE 16   // query rows per workgroup (one WMMA A fragment)
#define N_TILE 16   // keys per K/V tile (one WMMA B fragment)
#define S_STRIDE 18 // N_TILE + 2 padding, breaks LDS bank conflicts
#define V_STRIDE 18 // N_TILE + 2 padding
#define MAX_D_CHUNKS 16 // head_dim / 16, bounded so O_frags is a fixed array
#ifndef SPLIT_Q
// 1 = carry Q as a double-single (hi+lo) pair through two WMMA passes.
// Rounding Q to f16 for the A fragment was MEASURED to be the dominant error
// term: f16 Q alone through an otherwise-f32 kernel gives rel_l2 5.36e-4 of a
// 5.48e-4 total at CTX=12288 (82-98% across shapes). Everything else -- K, V,
// P, the accumulator -- contributes ~1.1e-4 combined.
#define SPLIT_Q 0
#endif

extern "C" __global__ __launch_bounds__(32) void attention_q8_0_flash_prefill_wmma(
    const float* __restrict__ q,               // [batch_size × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache, // [pos][n_kv_heads × bph][34]
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,                   // [batch_size × n_heads × head_dim]
    const int* __restrict__ positions,         // [batch_size]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn
) {
    const int q_tile = blockIdx.x;
    const int h      = blockIdx.y;
    if (h >= n_heads) return;

    const int q_base = q_tile * M_TILE;
    if (q_base >= batch_size) return;

    const int lane    = threadIdx.x & 31;
    const int half    = lane & 15;   // 0..15 — row within A / col within C
    const int half_id = lane >> 4;   // 0..1  — C fragment row parity

    const int bph          = head_dim / 32;      // Q8_0 blocks per head
    const int total_blocks = n_kv_heads * bph;
    const int row_stride   = total_blocks * 34;  // bytes per cache position
    const int kv_group     = n_heads / n_kv_heads;
    const int kv_h         = h / kv_group;
    const int kv_blk       = kv_h * bph;
    const int q_stride     = n_heads * head_dim;
    const int d_chunks     = head_dim / 16;

    extern __shared__ char smem_raw[];
    _Float16* Q_lds   = (_Float16*)smem_raw;                        // [M_TILE][head_dim]
#if SPLIT_Q
    _Float16* Q_lo_lds = Q_lds + M_TILE * head_dim;                 // [M_TILE][head_dim]
    _Float16* V_lds_T = Q_lo_lds + M_TILE * head_dim;               // [head_dim][V_STRIDE]
#else
    _Float16* V_lds_T = Q_lds + M_TILE * head_dim;                  // [head_dim][V_STRIDE]
#endif
    _Float16* S_lds   = V_lds_T + head_dim * V_STRIDE;              // [M_TILE][S_STRIDE]
    float*    m_lds   = (float*)(S_lds + M_TILE * S_STRIDE);        // [M_TILE]
    float*    l_lds   = m_lds + M_TILE;                             // [M_TILE]
    float*    a_lds   = l_lds + M_TILE;                             // [M_TILE]

    const int rows = min(M_TILE, batch_size - q_base);

    // ── Stage Q tile (f16) ──────────────────────────────────────────────
    for (int i = lane; i < M_TILE * head_dim; i += 32) {
        const int r = i / head_dim;
        const int d = i % head_dim;
        const float qv = (r < rows)
            ? q[(long long)(q_base + r) * q_stride + h * head_dim + d]
            : 0.0f;
        const _Float16 qhi = (_Float16)qv;
        Q_lds[i] = qhi;
#if SPLIT_Q
        Q_lo_lds[i] = (_Float16)(qv - (float)qhi);
#endif
    }
    if (lane < M_TILE) {
        m_lds[lane] = -INFINITY;
        l_lds[lane] = 0.0f;
        a_lds[lane] = 1.0f;
    }
    __syncthreads();

    // Output accumulator: C fragment j → row (2*j + half_id), col `half` = dim.
    float8_t O_frags[MAX_D_CHUNKS];
    for (int dc = 0; dc < d_chunks; ++dc)
        O_frags[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // Longest causal window in this tile bounds the key sweep.
    int seq_len = 0;
    for (int r = 0; r < rows; ++r) seq_len = max(seq_len, positions[q_base + r] + 1);

    for (int kt = 0; kt < seq_len; kt += N_TILE) {
        const int t_len = min(N_TILE, seq_len - kt);

        // ── Phase A: S = Q · K^T on the matrix cores ────────────────────
        // This lane owns key (kt + half); it walks head_dim in 16-dim chunks.
        // K is read straight from global, NOT staged in LDS. Staging it the
        // same coalesced way V is staged was measured and REJECTED: it costs
        // 8 KB, dropping occupancy 3 -> 2 WG/CU, and made the kernel 2.26x
        // slower at CTX=8192 (9.45 -> 21.38 ms). Occupancy dominates the
        // coalescing gain here, matching the Stage A tile-size finding.
        float8_t s_acc = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        {
            const int my_k = kt + half;
            const bool k_ok = (half < t_len);
            const unsigned char* kb =
                k_ok ? (k_cache + (size_t)my_k * row_stride + kv_blk * 34) : nullptr;

            for (int dc = 0; dc < d_chunks; ++dc) {
                const int d0 = dc * 16;
                half16_t a_reg;
                for (int j = 0; j < 16; ++j) a_reg[j] = Q_lds[half * head_dim + d0 + j];
#if SPLIT_Q
                half16_t a_lo;
                for (int j = 0; j < 16; ++j) a_lo[j] = Q_lo_lds[half * head_dim + d0 + j];
#endif
                half16_t b_reg;
                if (k_ok) {
                    // 16 consecutive dims sit inside one 32-dim Q8_0 block.
                    const unsigned char* blk = kb + (d0 >> 5) * 34;
                    const float ks = (float)*((const _Float16*)blk);
                    const int off = 2 + (d0 & 31);
                    for (int j = 0; j < 16; ++j)
                        b_reg[j] = (_Float16)(ks * (float)((signed char)blk[off + j]));
                } else {
                    for (int j = 0; j < 16; ++j) b_reg[j] = (_Float16)0.0f;
                }
                s_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, s_acc);
#if SPLIT_Q
                s_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b_reg, s_acc);
#endif
            }
        }

        // Publish S with the causal mask applied per (row, key).
        for (int j = 0; j < 8; ++j) {
            const int r  = 2 * j + half_id;   // query row within tile
            const int c  = half;              // key within tile
            float sv = s_acc[j] * scale_attn;
            const bool valid =
                (r < rows) && (c < t_len) && ((kt + c) <= positions[q_base + r]);
            // -INF (not f16 min) so a fully-masked row-tile yields m_new = -INF,
            // which the softmax guard turns into zero weight rather than exp(0)=1.
            S_lds[r * S_STRIDE + c] = valid ? (_Float16)sv : (_Float16)(-INFINITY);
        }
        __syncthreads();

        // ── Phase B: online softmax over this tile, one row per lane ────
        if (lane < rows) {
            _Float16* srow = S_lds + lane * S_STRIDE;
            float tm = -INFINITY;
            for (int c = 0; c < N_TILE; ++c) tm = fmaxf(tm, (float)srow[c]);
            const float m_old = m_lds[lane];
            const float m_new = fmaxf(m_old, tm);
            const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
            float sum = 0.0f;
            for (int c = 0; c < N_TILE; ++c) {
                const float e = (m_new == -INFINITY) ? 0.0f : __expf((float)srow[c] - m_new);
                srow[c] = (_Float16)e;
                sum += e;
            }
            m_lds[lane] = m_new;
            l_lds[lane] = l_lds[lane] * alpha + sum;
            a_lds[lane] = alpha;
        } else if (lane < M_TILE) {
            for (int c = 0; c < N_TILE; ++c) S_lds[lane * S_STRIDE + c] = (_Float16)0.0f;
            a_lds[lane] = 1.0f;
        }
        __syncthreads();

        // ── Phase C: stage V^T (dequantised) then O += P · V ────────────
        // Lane L owns the contiguous dim run [L*DPL, (L+1)*DPL). Two things
        // fall out: the 32 lanes together read one key's 272 contiguous cache
        // bytes (coalesced) instead of each lane touching a different cache
        // position row_stride apart, and a lane's whole run lives inside ONE
        // 32-dim Q8_0 block, so the fp16 scale is loaded once per key rather
        // than once per element.
        const int dpl = head_dim / 32;              // dims per lane
        const int d_lo = lane * dpl;
        const int v_blk = d_lo >> 5;                // block index (run-uniform)
        const int v_off = 2 + (d_lo & 31);          // byte offset within block
        for (int kr = 0; kr < N_TILE; ++kr) {
            if (kr < t_len) {
                const unsigned char* vb = v_cache + (size_t)(kt + kr) * row_stride
                                        + kv_blk * 34 + v_blk * 34;
                const float vs = (float)*((const _Float16*)vb);
                for (int e = 0; e < dpl; ++e)
                    V_lds_T[(d_lo + e) * V_STRIDE + kr] =
                        (_Float16)(vs * (float)((signed char)vb[v_off + e]));
            } else {
                for (int e = 0; e < dpl; ++e)
                    V_lds_T[(d_lo + e) * V_STRIDE + kr] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        // Rescale the running accumulator before adding this tile.
        for (int dc = 0; dc < d_chunks; ++dc)
            for (int j = 0; j < 8; ++j)
                O_frags[dc][j] *= a_lds[2 * j + half_id];

        // A = P (row `half`, 16 keys); B = V^T (dim, 16 keys).
        half16_t p_reg;
        for (int j = 0; j < 16; ++j) p_reg[j] = S_lds[half * S_STRIDE + j];
        for (int dc = 0; dc < d_chunks; ++dc) {
            const int d = dc * 16 + half;
            half16_t v_reg;
            for (int j = 0; j < 16; ++j) v_reg[j] = V_lds_T[d * V_STRIDE + j];
            O_frags[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p_reg, v_reg, O_frags[dc]);
        }
        __syncthreads();
    }

    // ── Write O = acc / l ───────────────────────────────────────────────
    for (int dc = 0; dc < d_chunks; ++dc) {
        const int d = dc * 16 + half;
        for (int j = 0; j < 8; ++j) {
            const int r = 2 * j + half_id;
            if (r >= rows) continue;
            const float denom = l_lds[r];
            const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
            out[(long long)(q_base + r) * q_stride + h * head_dim + d] = O_frags[dc][j] * inv;
        }
    }
}
