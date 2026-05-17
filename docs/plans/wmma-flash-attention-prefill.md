# WMMA Flash Attention for prefill (issue #237, item 2)

**Branch:** `feat/wmma-fa-prefill` (off master)
**Targets, in landing order:**
1. gfx1100/1101/1102 (RDNA3 wave32 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`).
2. **gfx1151/1150/1152** (RDNA3.5 Strix Halo APU) — same builtin, but the perf win on this arch is **conditional on Phase 1.0 measurement passing the bandwidth-bound gate** (see §Phase 1.0). If gfx1151 scalar FA is already at VRAM ceiling, this arch is dropped from Phase 1.
3. gfx1200/1201 (RDNA4) — `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` sibling. Phase 3.

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

## Phase 1.0 — Spike (REQUIRED, 1-2 days, before any production code)

**This is a GO/NO-GO gate for the entire effort on gfx1151.** Three measurements + one code fix:

### Tasks

1. **profiler.rs gfx1151 arm** (10-line fix, A11). Unblocks downstream rocprof analysis.

2. **rocprof scalar FA on gfx1151** (1-2 hours). Measure on Qwen 3.5 9B asym2 (gfx1151 default), prompt ≥ 2048 tokens:
   - VRAM → L2 read throughput vs LPDDR5x ceiling (~200 GB/s effective on this host)
   - VALUUtil on the scalar FA tile kernel
   - L2 hit rate

   **Pass condition:** VRAM throughput < 80% of ceiling AND VALUUtil > 40%. If those hold, scalar FA is ALU-bound and WMMA can win. If VRAM > 90% of ceiling and VALUUtil < 30%, FA is bandwidth-locked and WMMA cannot win on gfx1151 — drop gfx1151 from Phase 1, target dGPU only.

3. **ALU-only stub kernel** (~150 lines, in `experiments/wmma_fa_spike/`). Takes already-dequantized fp16 Q, K, V (no asym dequant). Runs WMMA-FA with the LDS round-trip spelled out. A/B vs scalar FA on synthetic fp16 inputs, BLOCK_M ∈ {16, 32, 64}, BLOCK_N ∈ {32, 64} on gfx1100 + gfx1151.

   **Pass condition:** ≥ +25% on gfx1100; ≥ 0% on gfx1151 (the inline-dequant kernel will be slower than the stub by some delta, so the stub needs headroom).

4. **fp16 P-narrow precision spike** (small Python or Rust harness, half a day). Run scalar FA with an artificial fp16 narrow inserted at the P-step. Measure NLL drift on Qwen 3.5 9B asym4 + asym2 over a 256-token completion on `benchmarks/prompts/lru_cache_pep8_strict.txt`.

   **Pass condition:** ΔNLL/tok < 0.003 (leaves 0.002 headroom below the 0.005 Phase 1.2 gate for the V-WMMA fp16-input loss).

### Failure modes

- **Bandwidth-bound on gfx1151** (likely per `devlog_20260508_lloyd_wmma_phase_c.md:149-151`): drop gfx1151 from Phase 1. Target RDNA3 dGPU only. Document in plan rev 3 and continue.
- **Stub doesn't beat scalar by +25% on gfx1100**: kill the whole effort. The inline-dequant production kernel will be slower than the stub; if the stub can't win on the best case, no production kernel will.
- **fp16 P-narrow drift > 0.005**: investigate alternatives — keep P in fp32 (skip the P-WMMA, do scalar P@V), or use a mixed-precision WMMA variant if available. May force Phase 1 to fp32-only P@V (~30% slower than the WMMA P@V but still benefits from the QK WMMA).

## Phase 1.1 — Production kernel (5-7 days)

Only enter if Phase 1.0 gates all pass.

- **New kernel:** `kernels/src/attention_flash_asym4_wmma_tile_batched.hip` (~300-400 lines).
- **Asym4 only, hd=128 only, no tree-bias** (tree path falls through to scalar via the gate).
- **Inline Givens** (no pre-pass kernel; no `fa_q_rot` scratch).
- **TILE_SIZE = 128 preserved** — one partial-write per kv_tile, with 2× BLOCK_N=64 chunks per tile.
- **LDS round-trip for P → A-fragment** (~512 B per workgroup).
- **Per-row softmax state** as `float m_state[8], l_state[8]` per lane; row-max via `__shfl_xor` with off ∈ {1, 2, 4, 8}.
- **Per-row causal mask** via LDS-broadcast of `positions[BLOCK_M]`.
- **KernargBlob path mirrored** (per `dispatch.rs:14776` `launch_maybe_blob` pattern) for hipGraph capture.

**Dispatch wiring:**
- New helper `launch_asym_flash_batched_wmma` in `dispatch.rs`.
- `is_wmma_fa_enabled()` and `wmma_fa_min_batch()` env-cached gates (mirror `should_use_mmq` at `dispatch.rs:291`).
- Auto-route condition: `has_wmma_f16(arch) && batch_size >= 16 && head_dim == 128 && tree_bias.is_none() && is_wmma_fa_enabled()`.
- **Default off** behind `HIPFIRE_WMMA_FA=1` for Phase 1.

**Acceptance gates:**

1. `cargo check -p rdna-compute -p hipfire-arch-qwen35` clean.
2. **Kernel hsaco metadata:** zero spills, VGPR ≤ 128 (target 8-15 waves resident; `__launch_bounds__(32, 2)` is the minimum floor, not the ceiling). Check via `gfx-kernel-metadata` skill on the compiled `.hsaco`.
3. **Numerical drift:** `ΔNLL/tok < 0.005` vs scalar FA on Qwen 3.5 9B asym4, 256 tokens, lru_cache PEP-8 strict prompt with `prompt_normalize=true`. Two precision-loss points budget combined: fp16 P-narrow (Phase 1.0 measured) + fp16 P×V WMMA inputs.
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

## Phase 1.2 — Asym2 + final gates (2-3 days)

- **Asym2 source file** `kernels/src/attention_flash_asym2_wmma_tile_batched.hip` (~300 lines). 2-bit packing, `TURBO_C2` LUT. Same structural template as asym4; the dequant inner block changes.
- **Re-run all Phase 1.1 acceptance gates** on asym2 model paths (Qwen 3.5 9B asym2 on gfx1151 specifically — this is the gfx1151 default-config path).
- Coverage matrix: {gfx1100, gfx1151} × {asym4 explicit, asym2 default} × {9B, 27B} × {short, long-context}.

## Phase 1.3 — Default flip (separate PR, after independent reproduction)

Mirror the `prompt_normalize` default-flip pattern (opt-in 2026-04-25 → default-flip 2026-04-26, separate commit 9a2c667). Only after Phase 1.2 lands and shows reproducible ≥ +15% across two independent bench runs (different sessions, fresh processes).

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
