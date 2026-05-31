# WMMA Flash Attention for prefill (issue #237, item 2)

**Branch:** `feat/wmma-fa-prefill` (off master)
**Targets, in landing order:**
1. gfx1100/1101/1102 (RDNA3 wave32 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`).
2. **gfx1151/1150/1152** (RDNA3.5 Strix Halo APU) — same builtin, but the perf win on this arch is **conditional on Phase 1.0 measurement passing the bandwidth-bound gate** (see §Phase 1.0). If gfx1151 scalar FA is already at VRAM ceiling, this arch is dropped from Phase 1.
3. gfx1200/1201 (RDNA4) — `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` sibling. Pulled into Phase 1.1 during PR triage with separate gfx12 sources.

**Date:** 2026-05-17 (rev 2 — incorporated findings from three adversarial reviews; review files dropped per project memory rule on review-as-scaffolding).

**Issue reference:** [#237 item 2 — WMMA flash attention for prefill](https://github.com/Kaden-Schutt/hipfire/issues/237). lemon-mlx-engine measured **+33-39% prefill on Strix Halo** with BF16 KV. **That number is a ceiling, not a target for hipfire** — hipfire's asym4 baseline has 8× lower K-bandwidth than BF16, so the WMMA lift available to us is the ALU-only portion of lemon-mlx's win, minus the per-nibble dequant overhead. Realistic target: +15-25% prefill on the ALU-bound path; possibly 0% on bandwidth-bound parts.

## Goal

Close the prefill ALU gap on the asymmetric KV path on **RDNA3 dGPU arches first** (where scalar FA is ALU-bound), gate the iGPU path on a measurement. Decode (batch_size=1) stays scalar — WMMA M=16 wastes 15 rows.

## What changed from rev 1

Three findings from external review forced a structural rewrite:

1. **gfx1151 default KV mode is asym2, not asym4** (verified at `cli/index.ts:754`). The rev-1 Phase 1 asym4-only kernel would never fire on the canonical bench host's default config. Rev 2 ships asym4 + asym2 in Phase 1.
2. **gfx1151 prefill is empirically bandwidth-bound** (`benchmarks/results/devlog_20260508_lloyd_wmma_phase_c.md:149-151`: 3.2× slower than gfx1100, matching the 250/960 GB/s ratio). Rev 2 inserts a Phase 1.0 spike (rocprof + ALU-only stub) that GATES the entire effort on the bench host before any production code lands.
3. **The rev-1 algorithm sketch had three real defects**: the C/D-layout to A-layout repack was a bogus cast (needs LDS round-trip), the per-row softmax state was scalar (each lane owns 8 rows), and the kernel claimed BLOCK_N=64 but the partials/reducer contract is fixed at TILE_SIZE=128 — the inner loop has to walk both. Rev 2 spells out the full loop nest.

## Non-goals (do not let scope creep)

- **No decode-path WMMA.** Decode stays scalar wave32.
- **No new quant format.** asym{2,4} stay byte-for-byte identical.
- **No FA-3 / online softmax revisit.** Same FA-2 recurrence as the scalar kernel.
- **No tree-attention support in Phase 1** (tree_bias != null → fall through to scalar).
- **No fwht{2,3,4} variants in Phase 1** (the FWHT-shfl inverse on K complicates the WMMA B-fragment build).
- **No BLOCK_N=128 or 4-waves-per-workgroup.** Both rejected by review — first runs against the bandwidth-bound finding; second is speculative without precedent in this codebase.

## Why this is the right next kernel (vs the other open items in #237)

| Item | Effort | Expected lift on RDNA3 dGPU prefill | Expected lift on gfx1151 prefill | Risk |
|---|---|---|---|---|
| Tiled QMV (item 1) | M | +10-20% **decode** GEMV — not prefill | same | Low |
| **WMMA FA (this plan)** | **M-H** | **+15-25% prefill (after dequant tax)** | **0-15%, conditional on bandwidth measurement** | Medium-High |
| CU-aware tile sizing (item 3) | L | <5% on its own | <5% | Low |
| iGPU unified mem (item 4) | M | 0% (PR #239) | 0% in PR #239's reproduction | Owner gated, requires reproducible config-attached A/B |
| iGPU sync elim (item 5) | L | 0% | 0% | Same gate as #4 |

Item 2 is still the best next lever for **gfx1100** dGPU (where scalar FA is genuinely ALU-limited). On **gfx1151**, the lift is conditional and may be zero — Phase 1.0 measures this before any production kernel writes.

## Existing surface this builds on

### Reference WMMA kernel (the CORRECT template)

**Use `kernels/src/gemm_gate_up_hfq4g256_wmma.hip:43,93-97` as the structural template** — NOT `gemm_f16_wmma.hip`. The latter writes `a_reg[i*16+j]` for i,j ∈ [0,16), which is OOB for the 16-element `half16_t`; production WMMA kernels use the per-lane pattern:

```c
const int my_row = row_start + (tid & 15);   // lane (tid&15) owns row my_row
half16_t a_reg;
// Fill a_reg[0..15] with my_row's 16 K-values (e.g., dequantized from 4-bit indices):
DQ(0, pk0, 0); DQ(1, pk0, 4); ... DQ(15, pk1, 28);
// WMMA layout contract: lanes 0..15 hold rows 0..15; lanes 16..31 hold the same rows redundantly.
acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
```

The wave32 WMMA contract: each lane provides `half16_t` (16 fp16 values), and the lane-to-row mapping is fixed by the hardware (lane `tid & 15` → row, `tid >> 4` → redundancy bit).

Output accumulator `float8_t acc` — 8 fp32 partials per lane, mapping `acc[j] → (row, col) = (2*j + tid>>4, tid & 15)`. **This is C/D layout, not A layout — you cannot cast acc to half16_t for use as a WMMA A-input on the next call.** See §LDS round-trip below.

### Reference scalar FA kernel (semantics to preserve)

[`kernels/src/attention_flash_asym4_tile_batched.hip`](../../kernels/src/attention_flash_asym4_tile_batched.hip) (161 lines, scalar wave32). Key invariants:

- **TILE_SIZE = 128** (hard-coded at `dispatch.rs:14714`). The WMMA kernel must preserve this — one partial-write per 128 KV positions, period. The reducer (`attention_flash_asym_reduce_batched.hip:37`) iterates `max_tiles = ceil(seq / 128)` and would silently break if the WMMA kernel wrote 2× partials.
- Inline Givens forward on Q (no pre-pass kernel) — see §Inline Givens.
- Phase A (QK dot), B (tile max), C (exp + sum), D (V accumulate) → write partials → consume in reducer.

### Dispatch entry points

| Function | File:line | Role |
|---|---|---|
| `attention_flash_asym4_batched_masked` | `dispatch.rs:14937` | Batched (PFlash + DFlash) — main prefill entry |
| `attention_flash_asym2_batched` | `dispatch.rs:15006` | gfx1151 default-KV-mode entry |
| `attention_flash_asym4` | `dispatch.rs:15516` | Single-token (decode) — stays scalar |
| `launch_asym_flash_batched` | `dispatch.rs:14701` | Shared helper |

Rev-2 inserts a parallel branch inside `launch_asym_flash_batched` for the WMMA path:

```rust
let use_wmma = is_wmma_fa_enabled()                                  // env override + arch gate
    && has_wmma_f16(&self.arch)
    && batch_size >= wmma_fa_min_batch()
    && matches!(head_dim, 64 | 128 | 256)
    && tree_bias.is_none();
```

Same `partials` layout, same `attention_flash_asym_reduce_batched` reduce.

### Arch gate predicates

- `has_wmma_f16("gfx11*")` at `dispatch.rs:156` — reuse verbatim.
- `has_wmma_f16_gfx12("gfx12*")` at `dispatch.rs:163` — Phase 3.
- **New:** `is_wmma_fa_enabled()` — OnceLock-cached env check (`HIPFIRE_WMMA_FA ∈ {0, 1, auto}`), default off in Phase 1, default on after Phase 1.3 (separate PR).

### profiler.rs gfx1151 arm (BLOCKING for any analysis)

`crates/rdna-compute/src/profiler.rs:37-74` has no gfx1151 arm — gfx1151 falls into the "unknown" catch-all with `vgprs_per_simd: 1024, max_waves_per_simd: 20, l2_cache_mb: 4.0`. **Wrong.** Add before any rocprof analysis (10-line fix, Phase 1.0 task):

```rust
"gfx1150" | "gfx1151" | "gfx1152" => ArchSpec {
    generation: "RDNA3.5", simds_per_cu: 2, max_waves_per_simd: 16,
    vgprs_per_simd: 1536, lds_per_cu: 65536,
    l2_cache_mb: 2.0, infinity_cache_mb: 0.0,    // verify L2 size before merging
    default_bus_width: 256,
},
```

(L2 size for Strix Halo iGPU: published spec is 2 MB GPU L2; some sources cite 4 MB. Verify empirically via `rocprofv2 --list-counters` before merging.)

## Kernel shape — full loop nest

### Tile parameterization

| Dimension | Size | Meaning |
|---|---:|---|
| `TILE_SIZE` | **128** | KV positions per outer tile (matches existing partials contract) |
| `BLOCK_M` | 16 | Q rows per workgroup (one WMMA M-tile). Phase 1.0 A/Bs vs {32, 64} on synthetic spike. |
| `BLOCK_N` | 64 | KV positions per inner block (4× WMMA_N=16 inside one TILE_SIZE) |
| `WMMA_N` | 16 | WMMA tile N-dim |
| `BLOCK_K` | 16 | head_dim chunk per WMMA tile |
| Block size | 32 threads (wave32) | Mandated by `_w32` WMMA builtin |
| Resident waves | **2 minimum** (`__launch_bounds__(32, 2)`) | A *minimum* — actual occupancy with ~100 VGPRs/wave on RDNA3.5 is ~8-15 waves resident. The bound is a compiler floor, not a ceiling. |

### Grid

- `gridDim.x = n_heads`
- `gridDim.y = ceil(batch_size / BLOCK_M)` — Q-row tiles
- `gridDim.z = max_tiles = ceil(seq / TILE_SIZE)` — outer kv tiles, **same as scalar kernel**

### Full inner loop nest

```c
// Per-workgroup at (h, m_tile, kv_tile):
//   m_tile covers Q rows [m_tile*16 .. m_tile*16+15]
//   kv_tile covers KV positions [kv_tile*128 .. kv_tile*128+127] (TILE_SIZE)

const int my_row = m_tile * BLOCK_M + (tid & 15);
const int redundant_bit = (tid >> 4);   // 0 or 1, lanes 16..31 are redundant

// Per-lane state (VGPRs):
float8_t acc_o = {0};            // Output accumulator, C-layout, live across whole kv_tile
float m_state[8], l_state[8];    // Per-row softmax state — each lane owns 8 rows via C-layout
for (int j = 0; j < 8; j++) { m_state[j] = -1e30f; l_state[j] = 0.0f; }

// LDS layout (per workgroup):
//   q_pos[BLOCK_M=16] (i32, 64 B)       — per-row position, broadcast for causal mask
//   p_tile[16][16] (fp16, 512 B)        — softmax P fragment for the LDS round-trip
//   reuse same 512 B for V loading if needed
// Total LDS: 576 B per workgroup. Trivial.

// One-time: load Q rotated inline (no pre-pass kernel), broadcast q_pos via LDS.
half16_t q_reg;   // each lane holds my_row's 16 Q values, rotated
load_and_givens_rotate_inline(q_reg, Q_global, my_row, cos_theta, sin_theta);
if (tid < BLOCK_M) lds.q_pos[tid] = positions[m_tile * BLOCK_M + tid];
__syncthreads();

// ─── Outer: walk inside this kv_tile in 64-KV chunks (2× per TILE_SIZE=128) ───
for (int kv_block_base = 0; kv_block_base < TILE_SIZE; kv_block_base += BLOCK_N) {
    const int kv_block_start = kv_tile * TILE_SIZE + kv_block_base;
    if (kv_block_start >= seq_len) break;

    // ─── Middle: 4× WMMA_N=16 sub-blocks within BLOCK_N=64 ───
    for (int kv_sub = 0; kv_sub < BLOCK_N / WMMA_N; kv_sub++) {
        const int kv_sub_start = kv_block_start + kv_sub * WMMA_N;
        float8_t acc_qk = {0};  // QK accumulator for this 16x16 fragment

        // ─── Inner: walk head_dim in 16-wide K-tiles ───
        for (int k0 = 0; k0 < head_dim; k0 += BLOCK_K) {
            // a_reg: my_row's 16 Q values at columns k0..k0+15 (from q_reg slice)
            half16_t a_reg = slice_q(q_reg, k0);

            // b_reg: my_row-NUMBERED-LANE owns KV position (kv_sub_start + (tid&15)).
            // Lane (tid & 15) is responsible for filling b_reg[0..15] with that KV
            // position's 16 K-values at head_dim slots k0..k0+15, dequantized inline:
            half16_t b_reg = dequant_k_asym4(k_cache, kv_sub_start + (tid & 15),
                                              k0, kv_h, head_dim);
            // (dequant_k_asym4: read 8 bytes of nibbles, lookup TURBO_C4, multiply by cnorm)

            acc_qk = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc_qk);
        }
        // acc_qk now holds 16x16 fragment of QK for this sub-block, scaled.

        // Per-row causal mask + scale.
        // Each lane owns 8 cells at (row = 2*j + (tid>>4), col = tid & 15).
        // Need per-cell mask: kv_pos > q_pos[row] → -inf.
        const int my_col_pos = kv_sub_start + (tid & 15);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int my_cell_row = 2 * j + redundant_bit;   // 0..15 (the global row in M-tile)
            int q_pos_j = lds.q_pos[my_cell_row];
            float val = acc_qk[j] * scale_attn;
            acc_qk[j] = (my_col_pos > q_pos_j) ? -FLT_MAX : val;
        }

        // FA-2 online softmax update — PER SUB-BLOCK (precise; we accept the 4× cost
        // because per-block accumulation requires holding 64 columns of acc_qk state,
        // which doesn't fit in C-layout without a 4x VGPR expansion).
        //
        // Per-row max: reduce 16 columns within the half-wave.
        // For each row, the 16 columns sit at lanes where (tid & 15) ∈ [0, 16) and
        // (tid >> 4) == redundant_bit_for_that_row. Reduce via __shfl_xor(v, off) for
        // off ∈ {1, 2, 4, 8} — NEVER 16, that crosses the half-wave.
        float m_new[8], l_new[8], alpha[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float row_max = acc_qk[j];
            for (int off = 8; off > 0; off >>= 1)
                row_max = fmaxf(row_max, __shfl_xor(row_max, off));
            m_new[j] = fmaxf(m_state[j], row_max);
            alpha[j] = expf(m_state[j] - m_new[j]);
            float p = expf(acc_qk[j] - m_new[j]);
            float row_sum = p;
            for (int off = 8; off > 0; off >>= 1)
                row_sum += __shfl_xor(row_sum, off);
            l_new[j] = alpha[j] * l_state[j] + row_sum;
            acc_qk[j] = p;   // overwrite with post-softmax probability (still fp32)
        }

        // ═══ LDS round-trip: C-layout (acc_qk) → A-layout (p_reg) ═══
        // Each lane writes its 8 cells into LDS at (row, col) = (2*j + redundant_bit, tid & 15).
        // Then __syncthreads, then each lane reads p_reg[0..15] = lds.p_tile[my_row][0..15].
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int r = 2 * j + redundant_bit;
            lds.p_tile[r][tid & 15] = (_Float16) acc_qk[j];
        }
        __syncthreads();
        half16_t p_reg = *((half16_t*)&lds.p_tile[my_row][0]);

        // V dequant: lane (tid & 15) owns KV position (kv_sub_start + (tid&15)),
        // fills v_reg[0..15] with that position's 16 V values (Q8_0 dequant inline).
        // V is in normal space — no rotation needed.
        half16_t v_reg = dequant_v_q8_0(v_cache, kv_sub_start + (tid & 15),
                                          /*v_dim_start*/ 0, kv_h, head_dim);

        // P @ V WMMA — note: WMMA C input is zeroed; we manually add alpha * acc_o below.
        float8_t pv = {0};
        pv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p_reg, v_reg, pv);

        // Rescale prior O by alpha, add new P @ V slice.
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc_o[j] = alpha[j] * acc_o[j] + pv[j];
            m_state[j] = m_new[j];
            l_state[j] = l_new[j];
        }

        // NOTE: for head_dim > 16, the V WMMA only covers 16 V-dims per call.
        // To cover all head_dim values, this whole softmax-and-P-V block needs
        // to repeat across head_dim chunks — see "head_dim sub-loop" below.
        __syncthreads();   // before the LDS round-trip is reused next iteration
    }
}

// ─── End: write per-row partials for this kv_tile ───
// acc_o is C-layout: cell (row, col) = (2*j + redundant_bit, tid & 15) holds
// that row's column-`col` output. Write to partials at
//   partials[(m_tile*BLOCK_M + row) × n_heads × max_tiles × stride
//            + h × max_tiles × stride + kv_tile × stride].
// Per the existing layout, stride = 2 + head_dim.
write_partials(acc_o, m_state, l_state, m_tile, kv_tile, h);
```

### Open question on V WMMA dim coverage (resolve in Phase 1.1)

The pseudocode above shows the P @ V WMMA covering 16 V-dims at a time. For head_dim = 128, the output `acc_o` needs to span 128 dims — but `float8_t` only has 8 cells per lane in C-layout, and those cells map to `(row, col)` where col is the V-dim slot. With WMMA_N = 16 cols, one WMMA call covers cols 0..15 (V-dim slots 0..15).

To cover head_dim=128, we need **8 separate `float8_t acc_o` accumulators**, one per V-dim chunk, OR we need to loop the entire softmax-then-P@V block 8× per sub-block (with the QK already computed once). The latter is correct but increases the inner-loop cost; the former increases VGPR pressure.

**Decision deferred to Phase 1.0 spike measurement.** The spike kernel implements both and A/Bs them on RDNA3 dGPU + gfx1151.

## Inline Givens — register-only, no pre-pass kernel

**Inline the Givens rotation at Q-load time** rather than splitting into a separate kernel. The Givens block at `kernels/src/givens_common.h:12` is 6 fmadds per (a, b, c, d) quadruple. For BLOCK_M=16 Q rows × head_dim/4 quadruples = 4 fmadds × 8 quadruples = 32 fmadds per lane on the Q-load path. Trivially small compared to the head_dim/16 × BLOCK_N/16 × ~16 WMMA calls per workgroup.

This avoids:
- 4 MB scratch buffer (`pbs.fa_q_rot`)
- Extra kernel launch (~5-15 µs)
- L2 pollution (Q evicted by Q_rot write)
- hipGraph capture surface complication

## Phase 1.0 — Spike (COMPLETE, see results below)

**This was a GO/NO-GO gate for the entire effort on gfx1151.** All four
tasks ran on the bench host (Radeon 8060S, ROCm 7.12, kernel-cache
fresh). Results inline.

### Task 1 — profiler.rs gfx1151 arm (DONE)

Committed `80ed5d8a`. RDNA3.5 arm added with empirically-verified L2=2 MB
(rocminfo: Cache Info `L2: 2048 KB`). Unblocked downstream rocprof.

### Task 2 — rocprof scalar FA on gfx1151 (PARTIAL — kernel-trace done, PMC counters timed out)

`rocprofv3 --kernel-trace` on a 2048-token prefill of Qwen 3.5 9B mq3
with asym2 KV (gfx1151's default) gave per-kernel timings cleanly:
`attention_flash_asym2_tile_batched` averaged **2.048 ms/call** (504
calls, min 158 µs, max 2.836 ms — wide spread reflects per-position
seq_len growth).

`rocprofv3 --pmc FETCH_SIZE WRITE_SIZE MemUnitBusy L2CacheHit ...`
exceeded the 10-minute timeout budget twice (multi-pass counter replay
on a 9B model load is slow). **PMC measurement deferred** — not needed
once Task 3 produced an unambiguous signal.

**Arithmetic-intensity sanity check** (sufficient on its own):
unique-bytes traffic per call ≈ 34 MB → ≥ 48 GB/s sustained at the
measured 0.697 ms scalar fp16 spike, **vs gfx1151's ~150-200 GB/s
ceiling**. Not bandwidth-locked. (The asym2 production kernel is even
LESS bandwidth-pressured per byte than the fp16 spike, since asym2
packing is 7× smaller per K element.)

### Task 3 — ALU-only spike kernel (DONE — **5.91× WMMA win on gfx1151**)

Spike at `experiments/wmma_fa_spike/`, harness at
`crates/hipfire-runtime/examples/wmma_fa_spike.rs`. Both kernels
operate on pre-dequantized fp16 K/V — no asym dequant, no Givens —
so the A/B isolates ALU only.

Synthetic random Q/K/V, batch=32, seq=2048, n_heads=28, n_kv_heads=4,
head_dim=128, 2 warmup + 5 measure:

|                     | scalar fp16 | WMMA fp16  | speedup |
|---------------------|------------:|-----------:|--------:|
| median time         | 0.694 ms    | 0.117 ms   | **5.91×** |
| min time            | 0.672 ms    | 0.109 ms   | —       |

Numerical compare (partials buffer, fp32):
- max |Δ| = 0.0017
- max rel-diff = 11% on cells with |val| > 0.01 (928k cells)
- |Δ| histogram: 35% of cells > 1e-4, **0.05%** > 1e-3, **zero** > 5e-3

Kernel metadata (from `gfx-kernel-metadata` skill on gfx1151 hsaco):

|                   | VGPR | SGPR | LDS (B) | Spills | Max waves/SIMD |
|-------------------|-----:|-----:|--------:|-------:|---------------:|
| `fa_scalar_fp16`  | 28   | 37   | 0 (dyn 512 via launch) | 0 | 16 (unconstrained) |
| `fa_wmma_fp16`    | **239** | 47 | 5248 (static) | 0 | 6 (VGPR-bound) |

239 VGPRs > plan's 128 target → 38% of max occupancy on gfx1151. The
5.91× still cleared by holding 8 acc_o fragments live to cover hd=128.
Phase 1.1 has headroom to drop ~56 VGPRs by looping over V-dim slices
instead of keeping all 8 fragments live.

**Pass:** clears the +25% gfx1100 floor and the ≥0% gfx1151 floor by a
wide margin. Caveats: synthetic uniform inputs (not real attention,
which has more peaked softmax); no asym dequant overhead (production
WMMA-asym2 will erode the margin — realistic production target ~3-4×);
single shape (batch=32 only), no sweep.

### Task 4 — fp16 P-narrow precision spike (DONE — **FAIL, ΔNLL/tok 6× over gate**)

Patched `attention_flash_asym2_tile.hip` (single-token FA) with a
one-line fp16 round-trip on P:

```c
float e = expf(scores[i] - tile_max);
e = (float)(_Float16) e;   // simulate WMMA P → fp16 narrow
```

Force-cleaned `rdna-compute` + kernel cache between runs. Perplexity
on `wikitext2-1024s-2048ctx.txt`, Qwen 3.5 9B mq3-lloyd, --kv-mode
asym2, --ctx 2048 --warmup 8.

|                  | NLL/tok           | PPL     |
|------------------|------------------:|--------:|
| baseline (fp32 P) | 2.9053274253      | 18.2712 |
| fp16 P-narrow    | 2.9240367511      | 18.6163 |
| **Δ**            | **+0.0187**       | +0.345  |

**ΔNLL/tok = 0.0187 is 6.2× the 0.003 gate.** PPL drift of 1.9%
relative is in the same envelope as switching from q8 KV to asym2 KV —
i.e. comparable to a quant-tier downgrade. **Fail.**

Caveat: spike applies fp16 narrow to P only. Production WMMA also
narrows V for the P @ V multiply (V comes in fp16 from asym2 dequant
anyway, so this is partly accounted for, but the WMMA inner product
truncates BEFORE the fp32 accumulate where scalar accumulates from
fp32 inputs). True production WMMA NLL drift is likely **≥ 0.0187**,
possibly higher.

### Phase 1.0 verdict

- **Perf path: GREEN.** WMMA-FA is a real, large ALU win on gfx1151
  despite the iGPU's bandwidth profile. The scalar-FA-is-ALU-bound
  hypothesis is correct on this hardware.
- **Precision path: RED.** The full-WMMA-FA design (with fp16 P-narrow
  in the LDS round-trip) fails the NLL gate by 6×. Cannot ship as-is.

### Phase 1.0 → Phase 1.1 redesign

Path forward: **drop the P @ V WMMA. Keep the QK^T WMMA.** The plan's
rev 1 Risk 1 fallback ("scalar P@V") is now the production design:

1. Compute QK^T via WMMA (the big ALU win).
2. Apply scale + causal mask + per-row online softmax → P held in fp32.
3. **Multiply P × V in scalar wave32** (per-tile loop, fp32 accumulate)
   — exactly what the current scalar FA does for Phase D.

This sacrifices the second WMMA call but preserves the precision the
gate requires. Perf hypothesis for the revised Phase 1.1: somewhere
between scalar (0.694 ms) and full-WMMA spike (0.117 ms). The QK^T
WMMA alone was ~75% of the speedup contribution; the P @ V WMMA was
~25%. Estimate: ~0.20-0.30 ms in the spike, → ~2.5-3.5× over scalar fp16
on gfx1151 (still well above the +25% gate).

The Phase 1.1 kernel-shape section below needs an update to reflect
this: remove the LDS P round-trip, replace the P @ V WMMA with a
scalar tile-loop that mirrors `attention_flash_asym2_tile.hip` Phase D.

### Failure modes (now historical; preserved for review trail)

- **Bandwidth-bound on gfx1151**: did not occur. Scalar FA was ALU-bound,
  not bandwidth-bound, on the bench host. Hypothesis from
  `devlog_20260508_lloyd_wmma_phase_c.md` was about whole-pipeline
  prefill scaling; the FA kernel specifically is not the bandwidth
  bottleneck within prefill.
- **Stub doesn't beat scalar by +25% on gfx1100**: not measured (no
  gfx1100 available to this session); user testing on gfx1100 is a
  separate validation step.
- **fp16 P-narrow drift > 0.005**: OCCURRED at 0.0187. Plan revised
  to scalar P@V (above).

## Phase 1.1 — Production kernel (5-7 days)

Reflects the Phase 1.0 redesign: WMMA for QK^T only; scalar P @ V to
preserve fp32 precision on the softmax output.

- **New kernel:** `kernels/src/attention_flash_asym4_wmma_tile_batched.hip` (~250-300 lines).
- **Asym4 only, hd=128 only, no tree-bias** (tree path falls through to scalar via the gate).
- **Inline Givens** (no pre-pass kernel; no `fa_q_rot` scratch).
- **TILE_SIZE = 128 preserved** — one partial-write per kv_tile, with 2× BLOCK_N=64 chunks per tile.
- **QK^T via WMMA, P @ V via scalar.** The Phase 1.0 precision spike
  showed fp16 P-narrow drift = 0.0187 NLL/tok (6× the gate). Solution:
  keep acc_qk in fp32 through softmax, accumulate scalar `w * V`
  per-cell with fp32 precision — exactly the existing Phase D in
  `attention_flash_asym2_tile_batched.hip`. The LDS P-tile and P @ V
  WMMA from rev-2 are dropped.
- **Per-row softmax state** as `float m_state[8], l_state[8]` per lane; row-max via `__shfl_xor` with off ∈ {1, 2, 4, 8}.
- **Per-row causal mask** via LDS-broadcast of `positions[BLOCK_M]`.
- **Scalar Phase D**: lane (tid & 15) accumulates V-dim values for
  rows {2j + half}, mirroring the production scalar V-accumulate. No
  LDS for V — the scalar pattern is one row per lane, fp16 V dequant
  inline. Drops the LDS V staging from rev-2 as well.
- **KernargBlob path mirrored** (per `dispatch.rs:14776` `launch_maybe_blob` pattern) for hipGraph capture.

**Expected perf** (extrapolated from the spike): scalar Phase D adds
~50-70% of the spike's ALU work back, since Phase D is the
V-multiply-and-accumulate loop. Spike full-WMMA was 0.117 ms; scalar
fp16 was 0.694 ms; estimated revised Phase 1.1 = **0.20-0.30 ms**, →
2.5-3.5× speedup over scalar fp16 on gfx1151. Production WMMA-asym2
will further erode the margin by the dequant tax, landing at ~1.5-2×
production-realistic.

**Dispatch wiring:**
- New helper `launch_asym_flash_batched_wmma` in `dispatch.rs`.
- `is_wmma_fa_enabled()` and `wmma_fa_min_batch()` env-cached gates (mirror `should_use_mmq` at `dispatch.rs:291`).
- Auto-route condition: `has_wmma_f16(arch) && batch_size >= 16 && head_dim == 128 && tree_bias.is_none() && is_wmma_fa_enabled()`.
- **Default off** behind `HIPFIRE_WMMA_FA=1` for Phase 1.

**Acceptance gates:**

1. `cargo check -p rdna-compute -p hipfire-arch-qwen35` clean.
2. **Kernel hsaco metadata:** zero spills, VGPR ≤ 128 (target 8-15 waves resident; `__launch_bounds__(32, 2)` is the minimum floor, not the ceiling). Check via `gfx-kernel-metadata` skill on the compiled `.hsaco`.
3. **Numerical drift:** `ΔNLL/tok < 0.005` vs scalar FA on Qwen 3.5 9B asym4, 2048 tokens, wikitext2 slice with `prompt_normalize=true`. With the QK-only WMMA + scalar P @ V design, the only precision loss is fp32 fmadd reorder inside the WMMA accumulator — same envelope as issue #188. Phase 1.0 baseline was 2.9053; gate fails above 2.9103.
4. **Coherence gate:** `./scripts/coherence-gate.sh` passes for Qwen 3.5 9B asym4 with `HIPFIRE_WMMA_FA=1`. No attractors, no token loops, no special-token leaks. *(Mandatory — see `feedback_v2_sgpr_lut_falsified_2026_05_10`, `project_gfx11_dot2_trickle_down_falsified_2026_05_11`, `project_fp8_wmma_hfp4g32_2026_05_10`: every prior kernel win that passed synthetic bench failed coherence on the first try.)*
5. **DFlash coherence gate:** `./scripts/coherence-gate-dflash.sh` passes for Qwen 3.5 27B-3.5 LRU. *(The DFlash path doesn't use the new kernel because `tree_bias != null` falls through to scalar — this gate verifies no regression in the scalar path from our dispatch changes.)*
6. **PFlash bench:** if a PFlash bench exists for the asym4 path (verify in `crates/hipfire-runtime/examples/`), run it; expect no τ regression on drafter acceptance.
7. **Perf gate** (gfx1100 dGPU primary, gfx1151 if Phase 1.0 passed):
   - Fresh-process A/B via `scripts/probe_commits.sh`, interleaved (not batched before/after).
   - Prompt: lru_cache PEP-8 strict + a ≥ 2048-token prose prompt. Prompt md5 recorded per row.
   - Sweep `n ∈ {16, 32, 64, 128, 256}`.
   - 5 cells per condition, median + range reported.
   - **Floor: median ≥ +15% prefill at n ≥ 128 on gfx1100.** Floor for gfx1151 set by Phase 1.0 ALU-only ceiling minus 1.5× the dequant tax.
   - Document BIOS/EC/DPM state for gfx1151 runs (per `tests/speed-baselines/gfx1151.txt:33-44` — same binary+prompt swings 245→151 across BIOS configs).

## Phase 1.1.5 — sub_batch alignment fix (~1.5-3 hours) — **BLOCKING for Phase 1.2 and Phase 1.3**

> **Status (2026-05-18):** promoted from "recommended" to **required** after
> a second measurement on gfx1100 reproduced the dilution pattern. Two
> independent fresh-process A/Bs — one on the bench iGPU (gfx1151), one
> on a dGPU (gfx1100) — both show the WMMA win pinned below the CLAUDE.md
> Δ≥5% ship gate while a large fraction of the prefill silently runs the
> scalar fallback. **All downstream phases (1.2 asym2, 1.3 default flip,
> Phase 2) are blocked on Phase 1.1.5 landing and producing a clean
> long-prefill measurement**, because that measurement — not the current
> short-prefill numbers — is the signal that says whether further work is
> justified.

Phase 1.1 surfaced a real-deployment limitation: the WMMA auto-route
gate requires `batch_size % WMMA_BLOCK_M == 0 && sub_batch % WMMA_BLOCK_M
== 0`. On Qwen 3.5 9B with the default `flash_partials` sizing, `sub_batch`
gets squeezed by `partials_capacity / per_pos_bytes` once `max_tiles`
grows past ~4 (i.e. prefill > ~512 tokens). For prefill=1024 we measured
`sub_batch ∈ {24, 36, 72}` — none 16-aligned, so the kernel falls back to
scalar.

This means the WMMA path **only fires on the short-prefill regime**
(~256-512 tokens for 9B). Two independent fresh-process A/Bs at
n_ctx=2048 (where most chunks are running scalar fallback) bear this
out:

| GPU | model | Δ pipeline | paired t | source |
|---|---|---:|---:|---|
| gfx1151 | 9B mq3   | +1.81% | +9.46  | `devlog_20260517_wmma_fa_phase11_fresh_ab.md` |
| gfx1151 | 0.8B mq4 | +4.06% | +34.87 | `devlog_20260517_wmma_fa_phase11_fresh_ab.md` |
| gfx1100 | 4B mq4   | +2.04% | +18.55 | `devlog_20260518_wmma_fa_phase11_gfx1100.md` |
| gfx1100 | 0.8B mq4 | +0.95% | +12.85 | `devlog_20260518_wmma_fa_phase11_gfx1100.md` |

All four measurements are statistically clean (every paired round
positive, no inversions) but none clear +5%. **The gfx1100 0.8B number
(+0.95%) is particularly damning**: gfx1100 was the plan's primary
target arch under the "ALU-bound on dGPU" hypothesis, and a clean t=12.85
sub-1% lift on a small model says either (a) the FA fraction is even
smaller than estimated once kernel-launch overhead is accounted for, or
(b) silent scalar fallback is masking the win even at small prefills,
or both. **Phase 1.1.5 is the cheapest experiment that disambiguates
this** — once chunk_size unblocks the WMMA route across all sub_batches,
the new long-prefill A/B is the signal we need.

Long prefill is where FA is the largest fraction of total prefill
compute time (15-25% vs ~7.5% at short prefill on 9B), so the real win
is likely larger but masked. **Without Phase 1.1.5 we cannot tell the
difference between "kernel is correct but FA is too small a fraction of
prefill to matter" and "kernel works fine but is dilution-hidden by
scalar fallback."** Those two diagnoses imply very different downstream
plans — one says ship default-off and move on, the other says push
further into the asym2 / hd=256 / RDNA4 stack.

Two approaches exist; do **Approach B first**, then Approach A only if
B's long-prefill bench shows enough lift to justify the memory cost.

### Approach A — Resize `flash_partials` buffer

Make the partials capacity scale so `sub_batch` always equals
`min(batch_size, PREFILL_MAX_BATCH)` regardless of `max_tiles`. Touch
`crates/hipfire-arch-qwen35/src/qwen35.rs:Qwen35Scratch::new`
(and the `PrefillBatchScratch` allocator) to size `flash_partials` at
`n_heads × ceil(max_ctx_len / TILE_SIZE) × (2 + head_dim) ×
PREFILL_MAX_BATCH` floats.

**Effort: ~2-3 hours**
- 30 min — trace `flash_partials` allocation sites + understand current sizing
- 30 min — fix allocation to use the worst-case `max_tiles` × max_batch
- 30 min — graph-capture / address-stability sanity check (the new alloc
  must remain pinned across replay)
- 30 min — VRAM footprint regression check on tighter cards (gfx1102 12 GB,
  gfx1032 8 GB)
- 30 min — bench at long prefill (n_ctx=2048, 4096) with WMMA enabled

**Risk: Medium.** Memory grows roughly 5× for hd=256 + max_ctx=8K (from
the current size to the worst-case). Could regress tighter-VRAM
deployments. The buffer is shared across DFlash, PFlash, regular prefill
paths — all must agree on the new size.

**Doesn't fully solve the problem on its own** — `batch_size` itself is
caller-determined; non-16-aligned `batch_size` (e.g. PFlash with B=14)
still misses the gate. Approach B is needed for full coverage.

### Approach B — `chunk_size` kernel arg + in-kernel OOB masking ← **RECOMMENDED FIRST**

Add an i32 `chunk_size` to the kernel signature. Inside the kernel, mask
the Q load + `positions[]` read + partials write for lanes where
`m_start + lane_row >= chunk_size`. Drop the `batch_size %
WMMA_BLOCK_M == 0` and `sub_batch % WMMA_BLOCK_M == 0` gate clauses;
keep only `batch_size >= wmma_fa_min_batch()`.

**Effort: ~1.5-2 hours**
- 30 min — modify `kernels/src/attention_flash_asym4_wmma_tile_batched.hip`
  to add `chunk_size` arg and OOB-guard the trailing m_tile's lanes
- 15 min — modify `launch_asym_flash_batched` to push `chunk` into the
  KernargBlob (and the `Vec<*mut c_void>` mirror)
- 15 min — drop the alignment clauses from the wmma_ok gate
- 15 min — build + smoke test (`dump_logits_qwen35` at varied prefills,
  verify argmax still matches scalar across the firing window)
- 30 min — fresh-process A/B at long prefills (2048, 4096) via
  `benchmarks/results/wmma-fa-probe.sh`

**Risk: Low.** Kernel logic is unchanged except for additional guards.
Backward compatible — when `chunk_size == batch_size` (the typical
case), behavior is bit-identical to the current kernel. No memory
changes. No partials-buffer co-ordination needed.

**Pays off the data we don't have yet.** The +1.81% on 9B was measured
under the gate-failure regime where most chunks ran scalar. Approach B
unblocks a true long-prefill measurement, which is where FA is the
largest fraction of total compute and the WMMA win should be most
visible. If long-prefill shows ≥ +5% pipeline, this kernel earns its
complexity and Phase 1.3 (default flip) becomes defensible.

### Recommendation

**Approach B (1.5-2 hours) blocks Phase 1.2.** Without it, the current
0.95–4.06% pipeline numbers across two GPUs are misleading — they're
averaging WMMA chunks with silent scalar fallbacks. After Approach B,
the new long-prefill A/B (n_ctx ≥ 4096) gives the *real* WMMA win on
each hardware target. That number — not any of the current Phase 1.1
measurements — is what determines whether Phase 1.2 (asym2), Phase 1.3
(default flip), and Phase 2 (hd=256 + asym3) are worth pursuing.

**Decision tree post-Phase-1.1.5:**

- **Long-prefill ≥ +5% on at least one target arch** → Phase 1.2 and
  Phase 1.3 are defensible. Proceed.
- **Long-prefill +2-5% on both targets** → kernel is correct but
  pipeline lift is structurally capped. Keep default-off, ship as
  research scaffolding, drop Phase 1.2 / 1.3 / Phase 2 for now.
- **Long-prefill < +2% on both targets** → kernel cost (LDS round-trip,
  asym dequant overhead) is eating the WMMA win. Park the whole effort,
  pivot to a different lever from issue #237.

Approach A becomes a follow-up *only if* Approach B's number is high
enough to justify the memory cost. If Approach B shows ~+5-10% on long
prefill, then Approach A's incremental coverage (the few-percent of
calls where `batch_size` itself is non-16-aligned) is worth the ~2-3
hours. If Approach B still shows ~+2%, Approach A wouldn't move the
needle and isn't worth doing.

## Phase 1.2 — Asym2 + final gates (2-3 days) — **blocked on Phase 1.1.5**

**Do not start until Phase 1.1.5 lands and its long-prefill A/B clears
the decision tree above.** Without that signal there is no evidence the
asym2 kernel would clear any ship gate — the asym2 kernel inherits the
same dilution problem the asym4 kernel currently has.

- **Asym2 source file** `kernels/src/attention_flash_asym2_wmma_tile_batched.hip` (~300 lines). 2-bit packing, `TURBO_C2` LUT. Same structural template as asym4; the dequant inner block changes.
- **Re-run all Phase 1.1 acceptance gates** on asym2 model paths (Qwen 3.5 9B asym2 on gfx1151 specifically — this is the gfx1151 default-config path).
- Coverage matrix: {gfx1100, gfx1151} × {asym4 explicit, asym2 default} × {9B, 27B} × {short, long-context}.

## Phase 1.3 — Default flip (separate PR, after independent reproduction) — **blocked on Phase 1.1.5 + Phase 1.2**

Mirror the `prompt_normalize` default-flip pattern (opt-in 2026-04-25 →
default-flip 2026-04-26, separate commit 9a2c667). Two prerequisites,
both required:

1. **Phase 1.1.5 long-prefill A/B clears ≥ +5% on at least one target
   arch.** This replaces the prior "≥ +15%" bar, which the Phase 1.1
   numbers across two GPUs make clear is structurally unreachable
   without a much larger kernel-side change than what's planned. The
   +5% bar matches the CLAUDE.md ship gate and is the lowest defensible
   bar for moving a kernel default-on.
2. **Phase 1.2 has landed** with reproducible numbers across two
   independent bench runs (different sessions, fresh processes) on
   both asym4 and asym2 paths.

## Phase 2 — head_dim=256 + asym3 (~3-4 days)

- Add hd=256 path: head_dim loops 2× the inner WMMA tile (k0 = 0..255 step 16). VGPR pressure goes up; verify ≤ 128 still.
- Asym3 source file: 3-bit packing, unaligned 3-byte reads, `TURBO_C3` LUT. Trickier dequant.
- Re-run gates per quant format.

## Phase 3 — Tree mask + RDNA4 (~3-4 days)

- Tree-bias path: per-row bias add inside the softmax block (matches `attention_flash_asym4_tile_batched.hip:102-106`). Removes the `tree_bias.is_none()` clause from the auto-route gate.
- `.gfx12.hip` sibling kernels: rename `_w32` builtins to `_w32_gfx12`. Mechanical for fp16; verify the f32 accumulator behavior is unchanged.

## Risk register

| # | Risk | Probability | Mitigation |
|---|---|---|---|
| 1 | gfx1151 bandwidth-bound — WMMA wins nothing on iGPU | **High** (devlog evidence) | Phase 1.0 measurement is the gate. If true, drop gfx1151 from Phase 1. |
| 2 | Synth-win → prod-falsify (project pattern) | **High** (4+ prior cases) | Coherence + DFlash + PFlash gates all mandatory before any claim. Default-off until 1.3. |
| 3 | fp16 P-narrow shifts NLL > 0.005 | **Medium** | Phase 1.0 precision spike measures before coding. Fallback: scalar P@V (no WMMA) or fp32-P (no narrow). |
| 4 | LDS round-trip overhead eats the WMMA win | **Medium** | Spike kernel measures. Each round-trip = 16×16 fp16 stores + 16×16 loads + 1 barrier per sub-block, 4× per BLOCK_N=64. |
| 5 | Asym dequant overhead is 50-80% of ALU time | **Medium** | Spike kernel A/Bs stub (no dequant) vs full kernel. If full kernel < 50% of stub, the LUT approach needs an alternate (SGPR LUT, LDS LUT, etc.). Note: `feedback_v2_sgpr_lut_falsified` says SGPR LUT failed prior coherence — don't repeat without revalidating. |
| 6 | VGPR overflow forces 1-wave occupancy | **Low-Medium** | `gfx-kernel-metadata` skill on hsaco at every kernel revision. Hard ceiling: zero spills, VGPR ≤ 128. |
| 7 | gfx1151 BIOS variance produces unreproducible A/B | **High** | Document BIOS/EC/DPM state in every bench row. Median-of-5 minimum; range reported. |
| 8 | TILE_SIZE=128 partials contract gets violated silently | **Low (now)** | The full loop nest in §Kernel shape is explicit. Reducer untouched. Reviewer check: confirm one partial-write per kv_tile. |

## Validation methodology

- **Bench host:** gfx1151 (Strix Halo, Radeon 8060S, this machine) — `project_bench_host_gfx1151.md`.
- **Secondary host:** any gfx1100 dGPU available (the actual prefill ALU-bound target).
- **Prompts:** `benchmarks/prompts/lru_cache_pep8_strict.txt` (short) + a ≥ 2048-token prose prompt (to be added). Prompt md5 in every bench row.
- **Process model:** fresh `hipfire run` per measurement (via `~/.hipfire/bin/daemon`, refreshed per `feedback_hipfire_run_uses_prod_daemon`). Kernel cache cleared between kernel source edits per `feedback_rdna_compute_kernel_staleness`.
- **ROCm:** 7.12 at `/opt/rocm-7.12` per `project_rocm_path_7_12`.
- **A/B mode:** interleaved per PR #239 owner gate.
- **BIOS/EC/DPM state:** documented in every gfx1151 bench row (per `tests/speed-baselines/gfx1151.txt:33-44`).
- **GPU lock:** `source gpu-lock.sh && gpu_acquire "wmma-fa-bench"`.

## Files touched

### New (Phase 1)

| File | Purpose |
|---|---|
| `experiments/wmma_fa_spike/spike.hip` | Phase 1.0 ALU-only stub kernel |
| `experiments/wmma_fa_spike/precision_spike.rs` | Phase 1.0 fp16 P-narrow drift measurement |
| `kernels/src/attention_flash_asym4_wmma_tile_batched.hip` | Phase 1.1 production kernel |
| `kernels/src/attention_flash_asym2_wmma_tile_batched.hip` | Phase 1.2 production kernel |

### New (Phase 2)

| File | Purpose |
|---|---|
| `kernels/src/attention_flash_asym3_wmma_tile_batched.hip` | Phase 2 |

### New (Phase 3)

| File | Purpose |
|---|---|
| `kernels/src/attention_flash_asym{2,3,4}_wmma_tile_batched.gfx12.hip` | RDNA4 siblings |
| `kernels/src/attention_flash_asym{2,3,4}_wmma_tile_batched_tree.hip` | Tree-mask path (or merged via `#ifdef TREE_MODE`) |

### Modified

| File | Change | Phase |
|---|---|---|
| `crates/rdna-compute/src/profiler.rs` | Add gfx1151/1150/1152 arm to `arch_spec()` | 1.0 |
| `crates/rdna-compute/src/kernels.rs` | Register `ATTENTION_FLASH_ASYM{2,4}_WMMA_TILE_BATCHED_SRC` | 1.1, 1.2 |
| `crates/rdna-compute/src/dispatch.rs` | Add `launch_asym_flash_batched_wmma`, `is_wmma_fa_enabled()`, `wmma_fa_min_batch()`. Auto-route in `launch_asym_flash_batched` | 1.1 |
| `crates/hipfire-arch-qwen35/src/qwen35.rs` | **No change** — inline Givens means no new scratch buffer | — |

### Deliberately untouched

- All scalar `attention_flash_*_tile.hip` kernels (decode keeps them; tree-mode keeps them in Phase 1).
- `attention_flash_asym_reduce_batched.hip` (partials layout preserved).
- `attention_flash_q8_0_reduce.hip` (reused for normal-space V output, unchanged).

## Open questions (resolved during Phase 1.0)

1. **V WMMA dim coverage strategy** — 8 accumulators (more VGPR) vs 8× loop (more LDS traffic). Decided by spike A/B.
2. **BLOCK_M ∈ {16, 32, 64}** — decided by spike A/B per arch.
3. **L2 size for gfx1151** — verify 2 MB vs 4 MB empirically via rocprofv2 before merging the profiler arm.
4. **fp16 P-narrow vs scalar P@V fallback** — decided by Phase 1.0 precision spike result.

## Related work

- `docs/plans/mq3-lloyd-wmma-prefill.md` — WMMA prefill GEMM template (kernel-shape vocabulary).
- `docs/methodology/perf-benchmarking.md` — fresh-process probe methodology.
- `benchmarks/results/devlog_20260508_lloyd_wmma_phase_c.md` — gfx1151 prefill scaling evidence informing the bandwidth-bound risk register.
- `tests/speed-baselines/gfx1151.txt` — gfx1151 BIOS-variance documentation.
- Issue #237 comments + PR #239 retrospective — original lemon-mlx analysis.
