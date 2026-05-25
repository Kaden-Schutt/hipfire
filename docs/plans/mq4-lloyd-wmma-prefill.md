# MQ4-Lloyd WMMA prefill kernels (Phase 5b — sibling of #116 Phase 5)

**Branch:** `feat/mq4-lloyd-wmma-prefill`
**Targets:** gfx1100/1101/1102/1150/1151 (RDNA3+3.5 wave32 WMMA) and gfx1200/1201 (RDNA4) sibling kernels.
**Date:** 2026-05-07.
**Depends on:** PR #182 (MQ4-Lloyd Phase 1 + 2 production kernels) — the runtime needs `DType::MQ4G256Lloyd` + the GEMV / fused-decode kernels for B=1 fallback. If #182 hasn't merged when implementation starts, base this branch on it.

## Goal

Close the prefill-perf gap on Lloyd-MQ4. Without WMMA prefill, the format will fall out of the batched-prefill path (the `is_batchable_la` allowlist gate at `crates/hipfire-arch-qwen35/src/qwen35.rs:3725`) and route per-token through `forward_scratch`, the same regression PR #181 found for Lloyd-MQ3 (108 tok/s vs 493 tok/s, a 5× gap).

This plan is the MQ4-Lloyd sibling of `docs/plans/mq3-lloyd-wmma-prefill.md`. The MQ3 plan landed Phase A in commit `869236d` (`feat(mq3-lloyd-wmma): Phase A residual kernel + parity bench (fp16-LDS wins)`) and that work resolves several shared design questions — MQ4 inherits those answers and proceeds without re-deciding them.

## Acceptance criteria

1. `cargo check -p rdna-compute -p hipfire-arch-qwen35 -p hipfire-runtime` clean.
2. Coherence-gate passes on a long-prompt (≥ `MIN_BATCH` tokens; verify value at `qwen35.rs:3520`) row for `qwen3.5-{4b,9b}.mq4-lloyd`.
3. **`ΔNLL/tok < 0.01`** (≤ 1 % PPL drift) vs the per-token decode path on `qwen3.5-9b.mq4-lloyd`. NOT byte-stable: WMMA's hardware accumulation order differs from GEMV's K-loop order; the resulting fp32-reorder drift is the same envelope as MQ3-Lloyd Phase A's. Note: this is a **second** drift surface added on top of the MQ4-Lloyd production kernels' single-acc byte-equality vs slow generic — the prefill path's WMMA accumulation is hardware-defined and not byte-equal to the K4 single-acc GEMV path. The 1 % gate is the same envelope MQ3 Phase A measured (max-abs 5.83e-5 at K=12288 on the 9B-down-proj scale).
4. Cross-process prefill perf gate per Phase C decision rules below (data-driven, not a fixed percentage).
5. The Phase B2 reviewer checklist (5 items, mirrored from the MQ3 plan, MQ4 substitutions) all pass.

## What's reusable

**MQ3-Lloyd Phase A (just landed) is the primary structural template** — the per-row codebook-in-LDS pattern, sync discipline, and parity-test harness all transfer. The remaining design template is the **HFQ4 WMMA family** for the 4-bit nibble-pair K-tile decode shape:

| File | Lines | Role |
|---|---:|---|
| `kernels/src/gemm_qkvza_hfq4g256_wmma.hip` | (read at HEAD) | LA preamble (qkv + z + beta + alpha, 4-way fused) |
| `kernels/src/gemm_qkv_hfq4g256_wmma.hip` | (read at HEAD) | FA preamble (qkv, 3-way fused) |
| `kernels/src/gemm_gate_up_hfq4g256_wmma.hip` | (read at HEAD) | FFN gate + up (2-way fused) |
| `kernels/src/gemm_hfq4g256_residual_wmma.hip` | (read at HEAD) | basic residual (`y += A·X`) |

Plus their `.gfx12.hip` siblings for RDNA4. Note: `kernels/src/gemm_hfq4g256_residual_wmma2.hip`, `_k2.hip`, `_k4.hip`, `_ksplit.hip` exist as alternate residual variants — start from the canonical `gemm_hfq4g256_residual_wmma.hip` and reference the others only if Phase C identifies a perf gap that one of the alternates closes.

The per-row codebook structure inherited from `kernels/src/gemm_mq3g256_lloyd_residual_wmma.hip` (the MQ3 Phase A kernel):

- `__launch_bounds__(32, 2)`, wave32 WMMA, K2-unroll inside the K-tile loop.
- 16-row × 16-batch tile per workgroup.
- Cooperative-load mapping `load_tile_row = tid >> 1`, `load_lo = (tid & 1) * <half-codebook>` is the load discipline; sync exactly twice per group (post-load + pre-next-load), zero in K-tile inner loop.
- WMMA accumulation: single `float8_t acc` per lane, accumulated in-place across K-tiles.
- Output convention: `acc[j] = C[2*j + (tid>>4)][tid & 15]` from `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`.

## What's new for MQ4 (vs MQ3-Lloyd Phase A)

Per group (160 B/group instead of MQ3's 112):

```
[0..32)   : 16 × fp16 codebook entries (sorted ascending, fp16 in storage)
[32..160) : 128 bytes packed 4-bit indices (low nibble = idx[2i], high nibble = idx[2i+1])
```

### Codebook size: 16 entries instead of 8

Per-row LDS budget doubles relative to MQ3:

- **16 rows × 16 entries = 256 codebook entries per group**
- At fp16 storage: **512 B per group** (vs MQ3's 256 B)
- At fp32 storage: 1024 B per group (vs MQ3's 512 B)

Both still trivial on 64 KB LDS. **Storage type is settled — adopt fp16 directly per MQ3 Phase A's empirical finding** (commit `869236d`, `devlog_20260507_lloyd_wmma_phase_a.md`: fp16-LDS wins 7.15 % aggregate over fp32 with bit-identical numerical output across all 8 shapes). MQ3 Phase A skipped the fp32 sibling for production but kept it as bench-A/B documentation; MQ4 can either:

- **(default)** Skip the fp32 sibling entirely. The MQ3 finding (bit-identical f16/f32 outputs because the f32→f16 narrow at decode is exact for fp16-stored codebooks) applies identically to MQ4 since the codebook storage format is the same.
- **(optional)** Implement an fp32 sibling for parity-test reviewer cross-check. Adds a few hundred LOC and a one-time bench. Skip unless a reviewer asks.

### Cooperative load: 32 threads × 8 fp16 each (vs MQ3's × 4)

256 entries / 32 threads = 8 fp16 per thread (4 bytes per LDS write × 4 = 16 B per thread × 32 threads = 512 B total). Pattern adapted from MQ3 Phase A:

```c
// MQ3 Phase A:
// const int load_tile_row = tid >> 1;        // 0..15
// const int load_lo       = (tid & 1) * 4;    // 0 or 4
// for (int i = 0; i < 4; i++) cb_lds[load_tile_row * 8 + load_lo + i] = cb_h[load_lo + i];

// MQ4 adaptation:
const int load_tile_row = tid >> 1;        // 0..15
const int load_lo       = (tid & 1) * 8;    // 0 or 8
for (int i = 0; i < 8; i++) cb_lds[load_tile_row * 16 + load_lo + i] = cb_h[load_lo + i];
```

Each thread issues 8 scalar fp16 reads instead of 4. Alignment is guaranteed because the 32 B codebook header sits at offset 0 of the 160 B group, and `160 % 16 == 0` — group bases are 16 B-aligned (better than MQ3's 8 B). Vectorized loads (e.g. `half8_t`) become possible later as a Phase C optimization if needed; the MQ3 plan flagged the same opportunity at lower priority.

**Same sync discipline as MQ3 Phase A** — exactly two `__syncthreads()` per group (one after cooperative load, one before next iteration's load), zero inside the K-tile loop.

### Index decode: 4-bit nibble pair instead of 3-bit cross-byte

MQ4's bit packing is **simpler** than MQ3's. 16 weights per K-tile → **8 bytes** (vs MQ3's 6 bytes). No cross-byte reassembly:

```c
// MQ3 Phase A (3-bit cross-byte, 6 bytes for 16 weights):
// unsigned int q0  = a0 & 7u;
// unsigned int q1  = (a0 >> 3) & 7u;
// unsigned int q2  = ((a0 >> 6) | (a1 << 2)) & 7u;     // ← cross-byte
// ...

// MQ4 (4-bit nibble pair, 8 bytes for 16 weights):
unsigned int q0  =  dpa[0]       & 0xFu;
unsigned int q1  = (dpa[0] >> 4) & 0xFu;
unsigned int q2  =  dpa[1]       & 0xFu;
unsigned int q3  = (dpa[1] >> 4) & 0xFu;
//  ...  through q14, q15  =  (dpa[7] >> 4) & 0xFu
```

K-tile pointer increment: `dp + kt * 8` (vs MQ3's `* 6`). 16 K-tiles per group at K=256: 8 B × 16 = 128 B index section ✓.

`cb_base = my_tile_row * 16` (vs MQ3's `* 8`). Decode: `a_reg[i] = cb_lds[cb_base + q]`, identical structure to MQ3 with the 16-entry codebook width baked into `cb_base`.

### FWHT rotation

Already verified: `quantize_mq4g256_lloyd` (`crates/hipfire-quantize/src/main.rs`) calls `cpu_fwht_256(&mut group, signs1, signs2)` before Lloyd-Max k-means. MQ4-Lloyd weights are FWHT-rotated identically to MQ3. **Reuse the existing `rotate_x_mq` dispatch wrapper** with the Lloyd kernel; no new rotation logic.

## Four new kernel shapes (× 2 archs = 8 files)

| File | Source template |
|---|---|
| `kernels/src/gemm_qkvza_mq4g256_lloyd_wmma.hip` | `gemm_qkvza_hfq4g256_wmma.hip` + per-row codebook from MQ3 Phase A |
| `kernels/src/gemm_qkvza_mq4g256_lloyd_wmma.gfx12.hip` | `.gfx12.hip` sibling |
| `kernels/src/gemm_qkv_mq4g256_lloyd_wmma.hip` | `gemm_qkv_hfq4g256_wmma.hip` + per-row codebook |
| `kernels/src/gemm_qkv_mq4g256_lloyd_wmma.gfx12.hip` | sibling |
| `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma.hip` | `gemm_gate_up_hfq4g256_wmma.hip` + per-row codebook |
| `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma.gfx12.hip` | sibling |
| `kernels/src/gemm_mq4g256_lloyd_residual_wmma.hip` | `gemm_hfq4g256_residual_wmma.hip` + MQ3-Phase-A per-row codebook layout |
| `kernels/src/gemm_mq4g256_lloyd_residual_wmma.gfx12.hip` | sibling |

Symbol naming convention: match HFQ4 WMMA's existing convention exactly (read `gemm_qkvza_hfq4g256_wmma.hip:<symbol-line>` and the gfx12 sibling for ground truth before writing — do NOT invent a new convention). Gemini's review of the MQ3 plan called out this trap: parameter ordering in fused kernel signatures is non-trivial.

The two archs are **separate kernel files** (no `#if`-spaghetti unification) — same rationale as MQ3 Phase A: gfx12 differs in WMMA builtin, lane decomposition, C-output mapping, K-unroll factor, and chunk-per-lane count.

## Dispatch wiring — the all-together discipline

The followup doc `docs/plans/mq-lloyd-batched-prefill-followup.md` describes 4 corruption-prevention conditions that MUST land together in the same commit (Phase B2 below). Same trap as MQ3-Lloyd. The 4 conditions, MQ4-substituted:

1. **`is_batchable_la` widening** — add `MQ4G256Lloyd` arm at `crates/hipfire-arch-qwen35/src/qwen35.rs:3725`, gated on the same arch list as MQ3-Lloyd's just-merged arm (gfx1100/1101/1102/1150/1151/1200/1201).
2. **Every `MQ4G256Lloyd` matcher arm** in `qwen35.rs` covers the prefill path. `MQ4G256Lloyd` arms already exist for the decode path from PR #182; the Phase B2 audit ensures the prefill matcher sites pick up the new WMMA dispatchers. Re-grep at the branch tip:
   ```
   grep -nE 'DType::MQ4G256Lloyd\b' crates/hipfire-arch-qwen35/src/qwen35.rs
   ```
3. **Every matcher** that gates a GEMM dispatch routes to a corresponding `gemm_*_mq4g256_lloyd_wmma` arm in `crates/rdna-compute/src/dispatch.rs`.
4. **Every dispatch arm** has a kernel file + `kernels.rs` constant + arch selector covering both gfx11 and gfx12 paths.

The coherence-gate row is a **separate** soft requirement (regression-prevention, not corruption-prevention) — track as a companion commit in the same PR but not part of the all-together package.

## Phasing

### Phase A — MVP single residual kernel + parity test (0.5–1 day, much shorter than MQ3 Phase A)

- Implement `gemm_mq4g256_lloyd_residual_wmma.hip` only (gfx11 first; gfx12 sibling is Phase B1).
- **Skip the fp32-LDS sibling** — MQ3 Phase A established fp16 wins; the same applies to MQ4.
- Add a parity-test example `crates/rdna-compute/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs` comparing kernel output vs CPU reference at the canonical 8 shapes (M ∈ {64, 256, 1024}, K ∈ {1024, 4096, 12288}, N ∈ {16, 64, 256}). Mirror MQ3 Phase A's harness — same shape grid, same fp32-acc-from-fp16 reference, same disc-distinguishable-input pattern for catching arg-reordering at parity time.
- Tolerance: **start with MQ3's 1.75e-4 (3× of 5.83e-5)** and tighten if MQ4 observes consistently smaller errors. The 16-entry codebook may produce slightly tighter reconstruction noise per element than MQ3's 8-entry, but the WMMA accumulation noise at K=12288 likely dominates; expect tolerance to land in the same envelope.
- Phase A acceptance:
  - Parity test PASS at empirically-set tolerance.
  - **No VGPR spills** in disassembly. Use the `gfx-kernel-metadata` skill (`docs/skills/gfx-kernel-metadata`, generalized in commit `9140c9d`) — verify `.private_segment_fixed_size: 0`. MQ3 Phase A landed at 82 VGPR / 18 SGPR / 256 B LDS; MQ4 with 512 B LDS and the larger codebook decode may push a few VGPRs higher. Hard gate is "no spills"; the soft target is "remain within `__launch_bounds__(32, 2)`'s wave-occupancy budget" (verify the binder isn't switched from launch_bounds to VGPRs).
  - **K-tile inner loop has zero `__syncthreads()`** — verify by source grep AND by reading the AMDGCN disassembly for `s_barrier` instructions in the inner loop.
  - Per the MQ3 Phase A pattern, commit a Phase A devlog at `benchmarks/results/devlog_2026XXXX_mq4_lloyd_wmma_phase_a.md` capturing parity table, disassembly metadata, and (briefly) noting that fp32-LDS was skipped per inheritance from `devlog_20260507_lloyd_wmma_phase_a.md`.

### Phase B1 — kernel family + arch selectors (kernels-only commit, 1–1.5 days)

- 7 additional kernel files (3 fused gfx11 + 4 gfx12 siblings, since Phase A landed only the residual gfx11).
- `kernels.rs` constants for all 8 files + 4 `*_for_arch` selectors (one per kernel shape) covering gfx1100/1101/1102/1150/1151 (rdna3) and gfx1200/1201 (rdna4); other archs fall through to `forward_scratch` (correct, slower).
- Parity tests for each new fused kernel — different fan-in/fan-out from the residual kernel. **Use mocked-but-distinguishable inputs** (e.g. weight-tensor-i has all zeros except row i has all-ones, X has lane-i = 1.0 elsewhere 0) to catch arg-reordering bugs at parity-test time, not at integration time.
- **No `dispatch.rs` arms beyond what the parity tests need, no matcher updates, no `is_batchable_la` change.** Kernels exist as dead code at the inference-dispatch level (constants are loaded but never invoked through the qwen35.rs matchers). Zero corruption risk.
- Single commit, reviewable as pure kernel work.

### Phase B2 — corruption-prevention all-together commit (1 day)

- Add `dispatch.rs` arms for all 4 kernel shapes (signatures mirror the HFQ4 WMMA arms exactly; copy structure verbatim, change only the kernel constant + module name + group-stride constant).
- Define `pub const LLOYD_MQ4_GROUP_BYTES: usize = 160;` somewhere central. Every Lloyd-MQ4 dispatch arm references this constant; **no magic-number `* 160` in dispatch arms**. Mirrors the MQ3 plan's `LLOYD_MQ3_GROUP_BYTES = 112` discipline.
- Update `is_batchable_la` (`qwen35.rs:3725`) to include `MQ4G256Lloyd` on gfx11/gfx12 archs.
- Update every matcher arm in `qwen35.rs` that handles `MQ3G256Lloyd` (or the prefill path equivalents) to also handle `MQ4G256Lloyd`. Run the audit grep at the branch tip:
  ```
  grep -nE 'DType::MQ3G256Lloyd\b' crates/hipfire-arch-qwen35/src/qwen35.rs
  grep -nE 'DType::MQ4G256Lloyd\b' crates/hipfire-arch-qwen35/src/qwen35.rs
  ```
  Each MQ3-Lloyd hit on the prefill path needs a Lloyd-MQ4 companion arm; each existing MQ4-Lloyd hit (from PR #182) is decode-side and probably needs no further widening.
- Build clean, parity tests still PASS for individual kernels, coherence-gate green.
- Single commit (the all-together discipline). The diff will be 300–500 lines but contains the 4 corruption-prevention conditions in lockstep.

#### Phase B2 reviewer checklist

Mechanical, auditable yes/no items (mirrored from MQ3 plan, MQ4-substituted):

1. [ ] `is_batchable_la` returns true for `DType::MQ4G256Lloyd` on `gfx1100 | gfx1101 | gfx1102 | gfx1150 | gfx1151 | gfx1200 | gfx1201`, matching the existing MQ3G256Lloyd arm's arch list verbatim (and the MQ4G256 arm if/when that lands).
2. [ ] Every line returned by the `grep` invocations above at branch tip has a corresponding `DType::MQ4G256Lloyd` prefill arm or is explicitly documented as intentionally not extended.
3. [ ] Every matcher branch that calls into a GEMM dispatch routes to a `gemm_*_mq4g256_lloyd_wmma` arm in `dispatch.rs`.
4. [ ] Every Lloyd-MQ4 dispatch arm computes weight strides via `LLOYD_MQ4_GROUP_BYTES = 160`, NOT a hardcoded 136 or 160. Grep:
   ```
   grep -E '\* 1(36|60)' crates/rdna-compute/src/dispatch.rs
   ```
   should show no Lloyd-MQ4-related hits — every stride goes through the named constant.
5. [ ] K % 256 == 0 holds for every projection in Qwen3.5-{4b,9b} layer dimensions (already verified for MQ3-Lloyd Phase B2; same model dims, so should hold trivially — re-confirm in the Phase B2 commit message).

### Phase B3 (companion commit) — coherence-gate row

- Add a long-prompt row to `scripts/coherence-gate.sh` for `qwen3.5-{4b,9b}.mq4-lloyd` that exercises the batched-prefill path (token count ≥ `MIN_BATCH`, single forward pass).
- Reuse the existing 4B / 9B MQ4-Lloyd model files from PR #182 (already on disk; no new model dependency).
- Commit the prompt as a separate file (per CLAUDE.md prompt-md5 rule), referenced by md5 in the gate script.
- Soft requirement: gate passes (fluent + on-topic + no attractor loops). Not part of the Phase B2 corruption-prevention package.

### Phase C — perf validation + ship gate decision rules

**Bench tool prerequisite:** identify whether MQ3-Lloyd Phase C built a prefill bench. If yes, reuse it directly (substitute MQ4-Lloyd model file). If MQ3 Phase C built a new `bench_qwen35_prefill.rs`, MQ4 inherits at zero cost. If MQ3 Phase C used `perplexity` per-window tok/s as a proxy, MQ4 does the same.

**Bench config:** cross-process A/B via `probe_commits.sh` or equivalent (per CLAUDE.md, within-session A/B has ±10–15 % drift — unusable for sub-10 % perf decisions). Compare 9B Lloyd-MQ4 prefill against:

- Pre-Phase-B baseline (per-token `forward_scratch` path with PR #182 single-acc kernels).
- 9B uniform MQ4 prefill (the structural ceiling; HFQ4 WMMA family).
- 9B Lloyd-MQ3 prefill (sibling format reference, post-MQ3 Phase 5 ship).

**Decision rules:**

| Lloyd-MQ4 prefill / MQ4 non-Lloyd prefill | Action |
|---|---|
| ≥ 60 % | **Ship.** Even at 60 %, Lloyd-MQ4 prefill is multi-× better than the per-token fallback. |
| 30 % to 60 % | **Investigate.** Likely culprits: VGPR pressure with the larger 512 B LDS, K-tile schedule on the 8-byte-per-tile decode (MQ3 was 6 bytes), reconvergent-sync timing on the longer cooperative load. Re-bench after each variant. |
| < 30 % | **Stop.** Request maintainer input — there's a structural issue in the WMMA + 512 B LDS combination that needs deeper investigation than this Phase 5b scope allows. |

### Phase C result on gfx1100 (2026-05-08, branch HEAD `1934aae`)

Cross-process A/B (3 fresh invocations × 2 models, `--prefill 256
--prefill-runs 3` with in-process median, mean across invocations)
on Qwen3.5-9B at gfx1100 (7900 XTX, ROCm 7.2):

| Comparison | Numerator | Denominator | Ratio |
|---|---|---|---|
| Lloyd-MQ4 prefill / uniform-MQ4 prefill | 1393.4 | 2299.5 | **60.6 %** |
| Lloyd-MQ4 prefill / pre-B per-token fallback | 1393.4 | ~120 | ~11.6 × |
| **Sibling reference: MQ3-Lloyd Phase C, same hardware** | 1516.6 | 1719.6 | 88.2 % |

**Decision: SHIP per rule.** 60.6 % ≥ 60 % clears the hard ship gate.
The 80 % soft target estimate is missed by ~19 pp; the gap to the MQ3
sibling's 88.2 % on the same hardware is the dominant signal. Per the
investigate-bucket text above, the candidate root causes are 2× LDS
footprint, longer 8-byte-per-tile decode K-schedule, and longer
cooperative-load sync. None of these are correctness issues — they
are perf-optimization headroom for follow-up work and **out of scope
for this PR**, which targeted correctness + a working ship gate.

Decode regression check via `probe_commits.sh` (master `85678ed`
canonical `--prefill 16 --gen 30`): uniform MQ4 117.1 → 117.3 tok/s
(+0.2 % noise) on master vs HEAD — no decode regression on the
non-Lloyd path. Lloyd MQ4 decode 97.3 tok/s on HEAD; master returns
BENCH_FAIL because Lloyd format runtime gating is removed during
Phase 5b (matches the MQ3-Lloyd Phase 5 pattern).

Full Phase C devlog with raw numbers and per-comparison breakdown:
`benchmarks/results/devlog_20260508_mq4_lloyd_wmma_phase_c.md`.

**gfx1151 + gfx12 ship-gate ratios are not yet measured.** Phase B2
established a pre/post-B2 12.3× speedup on gfx1151 (different
comparison: same-machine self-ratio, not Lloyd-vs-uniform). The
Lloyd-vs-uniform-MQ4 ratio on gfx1151 should track gfx1100's within
arch-class noise since both numerator and denominator share the
LPDDR5x bandwidth ceiling, but this is unverified. RDNA4 (gfx12)
remains help-wanted per PR #197.

## Out of scope (deferred)

- **MQ2-Lloyd WMMA prefill** — research-only format; MQ2 is permanently behind `--allow-mq2-lloyd` per PR #115.
- **MoE-LA / MoE-FA matchers (issue #179)** — orthogonal silent-correctness gap on plain MQ-Lloyd; track in B2's matcher audit. If MQ3-Lloyd Phase 5 closed it, MQ4 inherits the fix; if not, defer to a separate PR.
- **Vectorized cooperative load (`half8_t` LDS reads)** — MQ3 Phase A noted this as a future optimization; revisit if Phase C identifies a perf gap that this could close. With 512 B/group LDS on MQ4, vectorized loads have more headroom than MQ3's 256 B but the same memory-pressure-vs-issue-pressure trade-off.

## Risks and watch-items

- **VGPR pressure with 512 B LDS.** MQ3 Phase A landed at 82 VGPRs / 256 B LDS; 0 spills. MQ4 has 2× LDS budget and a slightly more complex decode (256-entry codebook lookup vs MQ3's 128). VGPR pressure could push past the `__launch_bounds__(32, 2)` budget. **Mitigation:** Phase A acceptance includes the disassembly check; if VGPRs become the binder before launch_bounds, retune `__launch_bounds__` (e.g. `(32, 1)`) and re-bench.
- **gfx12 sibling kernels untestable on local hardware.** Same as MQ3-Lloyd — bench host has gfx1100; secondary box has gfx1151. **gfx1200/1201 (RDNA4) is unavailable.** Phase B1's gfx12 kernels will compile-test only; ship as code-complete-but-runtime-unvalidated with an explicit note in the kernel header. Apply the same gating decision MQ3-Lloyd's Phase B1 used.
- **Codebook header alignment.** 16 fp16 entries × 2 B = 32 B header at offset 0 of each group. Group bases at ≥ 16 B alignment (better than MQ3's 8 B). Vectorized loads (`half8_t`) become possible later.
- **Parameter-ordering bug class in dispatch.rs.** HFQ4 fused kernel signatures are non-trivial (qkvza takes 4 weight pointers + 4 output pointers + 4 m-counts = 13 args). Copy-paste port that swaps any two pointers reads the wrong weight tensor → silent corruption. **Mitigation:** Phase B1's parity tests use mocked-distinguishable inputs that catch arg-reordering bugs at parity-test time.
- **Cross-path drift envelope vs the WMMA prefill of HFQ4.** MQ4-Lloyd shipped single-acc decode kernels in PR #182 that are byte-equal to slow generic at 10dp. The WMMA prefill path adds a *second* drift surface (hardware accumulation in WMMA differs from the K4 single-acc GEMV). Both drift surfaces are independently bounded. Acceptance criterion 3 (≤ 1 % PPL) is the gate.

## Open questions

**None left** for Phase A — MQ3-Lloyd Phase A resolved fp16-vs-fp32 LDS, sync discipline, register profile, and parity-test design. The remaining decisions for MQ4 (e.g. whether VGPR pressure pushes past launch_bounds at 512 B LDS) are *measurement* questions, not design questions; they get answered by the Phase A bench.

For comparison, MQ3 Phase A had one open question (fp16 vs fp32 LDS); MQ4 inherits that resolution. Three other questions were closed in the MQ3 plan (multi-acc/single-acc for WMMA, gfx11/gfx12 unification, B=1 fallback site coverage) and apply identically to MQ4 — single-acc-style WMMA accumulation, separate gfx11/gfx12 kernel files, and B=1 already-shipped through PR #182's `weight_gemv_prerotated` arm.

## References

- `docs/plans/mq3-lloyd-wmma-prefill.md` — sibling plan; structural template.
- `kernels/src/gemm_mq3g256_lloyd_residual_wmma.hip` (commit `869236d`) — MQ3 Phase A residual kernel; **the** structural template for the per-row codebook + cooperative-load + sync-discipline pattern.
- `benchmarks/results/devlog_20260507_lloyd_wmma_phase_a.md` (commit `869236d`) — MQ3 Phase A devlog with parity table, disassembly metadata, fp16-vs-fp32 bench data, the decision to ship fp16-LDS.
- `kernels/src/gemm_hfq4g256_residual_wmma.hip` (master HEAD) — HFQ4 WMMA residual; the canonical 4-bit nibble-pair K-tile decode template.
- `kernels/src/{gemm_qkvza_hfq4g256_wmma,gemm_qkv_hfq4g256_wmma,gemm_gate_up_hfq4g256_wmma}.{,gfx12.}hip` — fused HFQ4 WMMA kernels for Phase B1's fan-in/fan-out shape.
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3725` — `is_batchable_la`, the gate to widen.
- `crates/hipfire-quantize/src/main.rs` — `quantize_mq4g256_lloyd` (PR #182), confirms FWHT rotation applied at quant time.
- PR #181 — Lloyd-MQ3 GEMV K4+LDS work; codebook-LDS pattern source.
- PR #182 — MQ4-Lloyd Phase 1+2 (production single-acc kernels); the runtime substrate this WMMA prefill builds on.
- PR #189 — gfx1151 enablement for MQ3-Lloyd; arch list to match.
- Issue #116 — original Lloyd-MQ3 ship-gate; this plan extends it to MQ4.
- Issue #182 — MQ4-Lloyd implementation tracking.
- Issue #188 — multi-acc vs single-acc decision (cross-path drift envelope; same fp32-reorder pattern recurs in WMMA prefill vs GEMV decode).
- `docs/plans/mq-lloyd-batched-prefill-followup.md` — the all-together discipline + matcher-trap walkthrough. Re-read before Phase B2.
- `docs/skills/gfx-kernel-metadata` (commit `9140c9d`) — generalized disassembly-extraction recipe for the no-spill check.
