// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Query-tiled Q8_0 flash prefill attention.
//
// Grid:  [ceil(batch_size / BR), n_heads] in legacy (single-slot) mode, or
//        [n_tiles, n_heads] in multi-slot mode, where n_tiles = tile_slot.len()
//        from crates/rdna-compute/src/kv_slots.rs::build_tiles.
// Block: NTHREADS (256) threads.
//
// One workgroup owns BR query rows for one head and streams the K/V cache in
// BC-key tiles staged in LDS in native Q8_0 block form. K/V is therefore read
// once per BR queries rather than once per query, which is the whole point:
// the legacy attention_q8_0_kv_batched launches one workgroup per individual
// query and re-streams the entire prefix for each one.
//
// Because the LDS footprint depends only on BR/BC and never on context length,
// this kernel serves every sequence length — there is no LDS-capacity crossover
// and no occupancy decay as the context grows.
//
// TASK 1 SCOPE: exactly one K/V tile. The tile loop and online softmax arrive
// in Task 2.
//
// LDS layout (all float regions first so the byte regions stay 4-aligned and
// each 34-byte block start stays 2-aligned for the fp16 scale load):
//   s[BR*BC] | m[BR] | l[BR] | corr[BR] | q[BR*head_dim] | kt[BC*bph*34] | vt[...]

#include <hip/hip_runtime.h>
#include <assert.h>
#include "kv_slot_desc.h"

#ifndef BR
#define BR 16
#endif
#ifndef BC
#define BC 32
#endif
#ifndef NTHREADS
#define NTHREADS 256
#endif

extern "C" __global__ __launch_bounds__(NTHREADS) void attention_q8_0_flash_prefill(
    const float* __restrict__ q,               // [batch_size × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache, // [max_seq × n_kv_heads × bph × 34]
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,                   // [batch_size × n_heads × head_dim]
    const int* __restrict__ positions,         // [batch_size]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn,
    const KvSlotDesc* __restrict__ slot_descs, // [n_slots] or nullptr = legacy
    const int* __restrict__ tile_slot,         // [n_tiles] or nullptr = legacy
    const int* __restrict__ tile_row0,         // [n_tiles] or nullptr = legacy
    const int* __restrict__ tile_qbase         // [n_tiles] or nullptr = legacy
) {
    const int tile = blockIdx.x;
    const int h    = blockIdx.y;
    const int tid  = threadIdx.x;
    if (h >= n_heads) return;

    const int slot = (tile_slot != nullptr) ? tile_slot[tile] : 0;
    // Slot-relative row: the tile's first row *within its slot*. positions[]
    // keeps its existing meaning and layout (one absolute causal position per
    // FLAT row, indexed the same way q/out are — see kv_offset_for_k/v's
    // doc comment and docs/specs/2026-08-07-batched-attention-slots.md §6.2),
    // so causal masking below reads positions[qbase + r], not row0. row0 is
    // still received (matching the Task 3 tile-builder ABI) and is available
    // to any future per-slot-relative use; today it is not needed for
    // arithmetic in this kernel.
    const int row0 = (tile_row0 != nullptr) ? tile_row0[tile] : (tile * BR);
    (void)row0;
    // Global flat row, for indexing q/out/positions. In legacy mode these
    // coincide with row0. Conflating qbase and row0 leaves slot 0 correct
    // and every later slot reading the wrong query — see kv_slots.rs.
    const int qbase = (tile_qbase != nullptr) ? tile_qbase[tile] : (tile * BR);
    if (qbase >= batch_size) return;

    // `desc` supplies only the K/V slab BASE ADDRESS (k_base/v_base), never a
    // loop/causal bound — positions[] stays authoritative for that, exactly
    // as attention_flash_q8_0_tile_batched.hip and
    // attention_flash_asym3_tile_batched.hip do (566ce7d1). In legacy mode
    // k_base = v_base = 0, so kv_offset_for_k/v degenerate to the original
    // `pos * per_pos_bytes` addressing and output is bitwise identical.
    const KvSlotDesc desc = (slot_descs != nullptr) ? slot_descs[slot] : kv_slot_legacy(0, 0);

    const int bph          = head_dim / 32;          // Q8_0 blocks per head
    const int total_blocks = n_kv_heads * bph;
    const int per_pos_bytes = total_blocks * 34;     // bytes per cache position (K == V stride here: both Q8_0)
    const int kv_group     = n_heads / n_kv_heads;
    const int kv_h         = h / kv_group;
    const int kv_blk       = kv_h * bph;
    const int q_stride     = n_heads * head_dim;

    extern __shared__ float smem[];
    float* s_lds    = smem;                       // [BR*BC]
    float* m_lds    = s_lds + BR * BC;            // [BR]
    float* l_lds    = m_lds + BR;                 // [BR]
    float* corr_lds = l_lds + BR;                 // [BR]
    float* q_lds    = corr_lds + BR;              // [BR*head_dim]
    unsigned char* kt = (unsigned char*)(q_lds + BR * head_dim);
    unsigned char* vt = kt + BC * bph * 34;

    // Rows present in this tile. In legacy (single-slot) mode a tile can
    // only be ragged if it is the batch's last tile, so `batch_size - qbase`
    // bounds it correctly (original behavior, preserved when tile_qbase ==
    // nullptr). In multi-slot mode a tile can ALSO be ragged because its
    // OWNING SLOT ends before a full BR rows — build_tiles (Task 3) never
    // lets a tile span a slot boundary — and slots are laid out contiguously
    // in the flat row space q/out use, so the boundary where THIS tile's
    // valid rows end is exactly the NEXT tile's qbase (or batch_size, if
    // this is the last tile in the whole launch). Getting this wrong would
    // let a ragged tile read past its own slot into the next slot's Q rows,
    // attend them against the WRONG slot's KV (`desc` above is fixed for
    // this whole tile), and race the next slot's own correct tile for the
    // same `out` row.
    int rows_end = batch_size;
    if (tile_qbase != nullptr && (tile + 1) < (int)gridDim.x) {
        rows_end = tile_qbase[tile + 1];
    }
    const int rows = min(BR, rows_end - qbase);

    // ── Stage Q tile ────────────────────────────────────────────────────
    for (int i = tid; i < rows * head_dim; i += NTHREADS) {
        const int r = i / head_dim;
        const int d = i % head_dim;
        q_lds[r * head_dim + d] = q[(long long)(qbase + r) * q_stride + h * head_dim + d];
    }
    // Per-row running state.
    if (tid < BR) { m_lds[tid] = -1e30f; l_lds[tid] = 0.0f; corr_lds[tid] = 1.0f; }
    __syncthreads();

    // Output accumulator: thread owns row r = tid / lanes_per_row, dims
    // [d0, d0+dpt) where d0 = (tid % lanes_per_row) * dpt.
    const int lanes_per_row = NTHREADS / BR;
    const int acc_r  = tid / lanes_per_row;
    const int acc_c  = tid % lanes_per_row;
    const int dpt    = head_dim / lanes_per_row;
    const int d0     = acc_c * dpt;
    float acc[32];                                   // host asserts dpt <= 32
    for (int i = 0; i < dpt; i++) acc[i] = 0.0f;

    // Longest causal window in this tile decides how many keys to visit.
    // positions[] is authoritative — never desc.seq_len (see kv_offset_for_k
    // doc comment above).
    int seq_len = 0;
    for (int r = 0; r < rows; r++) seq_len = max(seq_len, positions[qbase + r] + 1);

    // Sanity invariant, not a correctness mechanism: a row may never attend
    // past its own slot's logical KV length. Descriptor construction must
    // keep desc.seq_len in sync with (max positions[row]+1 over that slot's
    // rows); this simply catches a violation early in debug builds.
    // NOTE: a debug `assert(seq_len <= desc.seq_len)` used to sit here. It was
    // removed: crates/rdna-compute/src/compiler.rs never passes -DNDEBUG, so the
    // assert shipped in RELEASE and cost 64 bytes/lane of scratch on every one of
    // these kernels. The paired `__builtin_assume` was worse than useless — it
    // would only take effect under -DNDEBUG, where it becomes a UB licence on a
    // fact the kernel never otherwise uses, buying no optimisation. The invariant
    // is documented on KvSlotDesc and checked host-side by the builders instead.

    // ── K/V tile loop with online softmax ───────────────────────────────
    for (int t0 = 0; t0 < seq_len; t0 += BC) {
        const int t_len = min(BC, seq_len - t0);

        __syncthreads();                   // protect kt/vt from the previous iteration
        for (int i = tid; i < t_len * bph * 34; i += NTHREADS) {
            const int j  = i / (bph * 34);
            const int bo = i % (bph * 34);
            const int t  = t0 + j;
            // ONE shared slab offset for both caches, deliberately.
            //
            // Using kv_offset_for_v here as well would keep a SECOND runtime-
            // unknown 64-bit base (desc.v_base) live across this staging loop,
            // which stages K and V in a single pass. Measured on gfx1151 with the
            // production flags, that costs 23 VGPRs and drops occupancy from 16 to
            // 12 waves/SIMD — a 25% loss, on the NULL-descriptor legacy path, in
            // the chunked-prefill kernel on the agent critical path. It regresses
            // identically on gfx1201, the deployment target.
            //
            // Safe because this kernel is Q8_0-only, where K and V share a stride
            // (see per_pos_bytes above), so a slot occupies the same offset in
            // both arenas. That is an ABI CONSTRAINT on the descriptor for this
            // kernel: v_base MUST equal k_base. asym3 must never adopt this — its
            // K and V strides genuinely differ.
            const unsigned long long slab = kv_offset_for_k(desc, t, per_pos_bytes);
            kt[i] = k_cache[slab + kv_blk * 34 + bo];
            vt[i] = v_cache[slab + kv_blk * 34 + bo];
        }
        __syncthreads();

        // S = Q · K^T with per-row causal mask.
        for (int e = tid; e < BR * BC; e += NTHREADS) {
            const int r = e / BC;
            const int j = e % BC;
            float sv = -INFINITY;
            if (r < rows && j < t_len && (t0 + j) <= positions[qbase + r]) {
                float dot = 0.0f;
                const unsigned char* kb = kt + (size_t)j * bph * 34;
                for (int bi = 0; bi < bph; bi++) {
                    const float ks = (float)*((const _Float16*)(kb + bi * 34));
                    const float* qb = q_lds + r * head_dim + bi * 32;
                    for (int c = 0; c < 32; c++)
                        dot += qb[c] * (ks * (float)((signed char)kb[bi * 34 + 2 + c]));
                }
                sv = dot * scale_attn;
            }
            s_lds[e] = sv;
        }
        __syncthreads();

        // Online softmax: new running max, rescale factor, running sum.
        if (tid < rows) {
            float mx = m_lds[tid];
            for (int j = 0; j < BC; j++) mx = fmaxf(mx, s_lds[tid * BC + j]);
            const float corr = (m_lds[tid] <= -1e30f) ? 1.0f : __expf(m_lds[tid] - mx);
            float sum = 0.0f;
            for (int j = 0; j < BC; j++) {
                const float sv = s_lds[tid * BC + j];
                const float e  = (sv == -INFINITY) ? 0.0f : __expf(sv - mx);
                s_lds[tid * BC + j] = e;
                sum += e;
            }
            m_lds[tid]    = mx;
            l_lds[tid]    = l_lds[tid] * corr + sum;
            corr_lds[tid] = corr;
        }
        __syncthreads();

        // O = O * corr + P · V
        if (acc_r < rows) {
            const float corr = corr_lds[acc_r];
            if (corr != 1.0f) for (int i = 0; i < dpt; i++) acc[i] *= corr;
            for (int j = 0; j < t_len; j++) {
                const float p = s_lds[acc_r * BC + j];
                if (p == 0.0f) continue;
                const unsigned char* vb = vt + (size_t)j * bph * 34;
                for (int i = 0; i < dpt; i++) {
                    const int d  = d0 + i;
                    const int bi = d >> 5;
                    const int bj = d & 31;
                    const float vs = (float)*((const _Float16*)(vb + bi * 34));
                    acc[i] += p * (vs * (float)((signed char)vb[bi * 34 + 2 + bj]));
                }
            }
        }
    }

    // ── Write output ────────────────────────────────────────────────────
    if (acc_r < rows) {
        float* o = out + (long long)(qbase + acc_r) * q_stride + h * head_dim;
        const float denom = l_lds[acc_r];
        const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        for (int i = 0; i < dpt; i++) o[d0 + i] = acc[i] * inv;
    }
}
