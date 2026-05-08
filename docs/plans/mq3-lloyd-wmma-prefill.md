# MQ3-Lloyd WMMA prefill kernels (Phase 5 from issue #116)

**Branch:** `feat/mq3-lloyd-wmma-prefill`
**Targets:** gfx1100/1101/1102/1150/1151 (RDNA3+3.5 wave32 WMMA) and gfx1200/1201 (RDNA4) sibling kernels.
**Date:** 2026-05-07 (rev 2 — folded findings from three independent adversarial reviews).

## Goal

Close the prefill-perf gap on Lloyd-MQ3 by writing WMMA-accelerated
batched-prefill GEMM kernels. PR #181 measured **9B Lloyd-MQ3 prefill
= 108 tok/s vs 9B uniform MQ4 = 493 tok/s** — a 5× gap caused by
Lloyd-MQ3 falling out of the batched-prefill path
(`is_batchable_la` allowlist gate at
`crates/hipfire-arch-qwen35/src/qwen35.rs:3676`) and routing per-token
through `forward_scratch`. Closing this is the biggest real-world
Lloyd-MQ3 lever still on the table.

The structural blocker per issue #116 (out-of-scope item):

> Lloyd-MQ3 WMMA prefill — the existing uniform MQ3 WMMA family
> doesn't transfer directly because reconstruction is a switch, not
> affine. Tracked separately under "Phase 5 — Q5 WMMA prefill kernels
> for MQ3 / MQ2".

This plan is Phase 5.

## Acceptance criteria

1. `cargo check -p rdna-compute -p hipfire-arch-qwen35 -p hipfire-runtime` clean.
2. Coherence-gate passes on a long-prompt (≥ `MIN_BATCH` tokens; verify
   value at `qwen35.rs:3520`) row for `qwen3.5-{4b,9b}.mq3-lloyd`.
3. **`ΔNLL/tok < 0.01`** (≤ 1% PPL drift) vs the per-token decode path
   on `qwen3.5-9b.mq3-lloyd`. NOT byte-stable: WMMA's hardware
   accumulation order differs from GEMV's K-loop order; the resulting
   fp32-reorder drift is the same envelope as issue #188's gfx1100
   GEMV multi-acc finding.
4. Cross-process prefill perf gate per Phase C decision rules below
   (data-driven, not a fixed percentage).
5. The Phase B2 reviewer checklist (5 items, below) all pass.

## What's reusable

The HFQ3 WMMA family is the structural template. Four distinct kernel
shapes on master, each with a `.gfx12.hip` sibling for RDNA4. Note:
**residual *is* the basic shape** for HFQ3 WMMA (output is `y += A·X`,
used for `wo` / `w_down` / `lm_head`); there is no separate
non-residual basic GEMM.

| File | Lines | Role |
|---|---:|---|
| `kernels/src/gemm_qkvza_hfq3g256_wmma.hip` | 209 | LA preamble (qkv + z + beta + alpha, 4-way fused) |
| `kernels/src/gemm_qkv_hfq3g256_wmma.hip` | 157 | FA preamble (qkv, 3-way fused) |
| `kernels/src/gemm_gate_up_hfq3g256_wmma.hip` | 158 | FFN gate + up (2-way fused) |
| `kernels/src/gemm_hfq3g256_residual_wmma.hip` | 136 | basic residual (`y += A·X`) |

Common structure:
- `__launch_bounds__(32, 2)`, wave32 WMMA.
- 16-row × 16-batch tile per workgroup.
- K-tile = 16 weights = 6 bytes of packed 3-bit indices, 16 K-tiles per
  group at K=256.
- X loaded as `half16_t` per K-tile.
- Per K-tile per row: decode 16 indices to fp16 → fill `half16_t a_reg`
  → WMMA mul-accumulate into `float8_t acc`.
- Per-row group header (`sc_h` and `zp_h`) loaded into registers from
  `row_base + g * 104`. **Each thread owns its own row's group
  header**; 16 distinct headers per workgroup per group.

The decode is the only place HFQ3 and Lloyd-MQ3 diverge. HFQ3 today
(verified at `gemm_qkvza_hfq3g256_wmma.hip:143`):

```c
#define DQ(i, q) a_reg[i] = sc_h * (_Float16)(float)(q) + zp_h
```

Branchless affine; compiler vectorizes cleanly.

## What's new for Lloyd

Per group (112 B/group instead of 104):

```
[0..16)   : 8 × fp16 codebook entries (sorted ascending, fp16 in storage)
[16..112) : 96 bytes packed 3-bit indices (same packing as HFQ3)
```

Decode becomes a per-(row, group) codebook lookup:

```c
half16_t a_reg = { cb[q0], cb[q1], ..., cb[q15] };
```

If kept as inline switch, the compiler emits the same divergent-
execution decision tree that bit Lloyd-MQ3 GEMV before PR #181
(~166 dispatch instructions per group inner body). Same fix applies:
**stage codebooks in LDS**.

### LDS staging — per-row, not per-group

**Critical:** each row in the 16-row tile has its own per-group
codebook. The 16 threads representing distinct rows need access to
**16 distinct codebooks per group**, not one shared codebook.

Per-group LDS budget:

- **16 rows × 8 entries = 128 codebook entries per group**
- At fp16 storage: 256 B per group
- At fp32 storage: 512 B per group

Both are trivial on 64 KB LDS. **Storage type (fp16 vs fp32) is a
Phase A empirical question** — fp16 saves the
`v_cvt_f16_f32` per element when assigning to `half16_t a_reg`
(~16 conversions × 16 K-tiles × 16 rows ≈ 4K conversions/group);
fp32 has cleaner bank-conflict math. Decide after measuring both
variants on real Qwen3.5-9B Lloyd weights in Phase A. **Do not
pre-commit in this plan.**

Cooperative load (per group, once before the K-tile loop): 32 threads
× 4 entries each (fp16) or via a 2-phase 32×2 load (fp32). Lloyd-MQ4
already uses the 2-phase pattern for its 16-entry codebook
(`kernels/src/gemv_mq4g256_lloyd.gfx1100.hip:67-79`); adapt
accordingly. **Exactly one `__syncthreads()` per group**, after the
cooperative load. **Zero `__syncthreads()` inside the K-tile loop.**
This is a hard performance constraint per Gemini's review: per-K-tile
sync would crater perf.

### FWHT rotation

Verified: `quantize_mq3g256_lloyd` (`crates/hipfire-quantize/src/main.rs:642+`)
calls `cpu_fwht_256(&mut group, signs1, signs2)` before Lloyd-Max
k-means. Lloyd weights ARE FWHT-rotated, identical to MQ3. **Reuse
the existing `rotate_x_mq` dispatch wrapper** with the Lloyd kernel;
no new rotation logic.

## Four new kernel shapes (× 2 archs = 8 files)

| File | Source template |
|---|---|
| `kernels/src/gemm_qkvza_mq3g256_lloyd_wmma.hip` | `gemm_qkvza_hfq3g256_wmma.hip` |
| `kernels/src/gemm_qkvza_mq3g256_lloyd_wmma.gfx12.hip` | `.gfx12.hip` sibling |
| `kernels/src/gemm_qkv_mq3g256_lloyd_wmma.hip` | `gemm_qkv_hfq3g256_wmma.hip` |
| `kernels/src/gemm_qkv_mq3g256_lloyd_wmma.gfx12.hip` | sibling |
| `kernels/src/gemm_gate_up_mq3g256_lloyd_wmma.hip` | `gemm_gate_up_hfq3g256_wmma.hip` |
| `kernels/src/gemm_gate_up_mq3g256_lloyd_wmma.gfx12.hip` | sibling |
| `kernels/src/gemm_mq3g256_lloyd_residual_wmma.hip` | `gemm_hfq3g256_residual_wmma.hip` |
| `kernels/src/gemm_mq3g256_lloyd_residual_wmma.gfx12.hip` | sibling |

Symbol naming convention: match HFQ3 WMMA's existing convention exactly
(read `gemm_qkvza_hfq3g256_wmma.hip:45` and the gfx12 sibling for the
ground truth before writing — do NOT invent a new convention).

The two archs are **separate kernel files**, not unified. The HFQ3
family already does this for good reason: gfx12 differs in WMMA
builtin (`_w32_gfx12`), lane decomposition (`half8_t` K-split via
`tid >> 4` lane-groups vs gfx11's `half16_t` full-tile per lane),
C-output mapping, K-unroll factor, and chunk-per-lane count.
`#if`-spaghetti unification is not viable.

## Dispatch wiring — the all-together discipline

The followup doc `docs/plans/mq-lloyd-batched-prefill-followup.md`
describes 4 corruption-prevention conditions that MUST land together
in the same commit (Phase B2 below). The trap: any one of them
landing alone produces either silent dead-code or stride-mismatched
silent corruption. The 4 conditions:

1. **`is_batchable_la` widening** — add `MQ3G256Lloyd` arm at
   `crates/hipfire-arch-qwen35/src/qwen35.rs:3676`, gated on the
   same arch list as MQ3 (gfx1100/1101/1102/1150/1151/1200/1201).
2. **Every `MQ3G256` matcher arm** in `qwen35.rs` also has a
   `MQ3G256Lloyd` arm. **Re-grep at the branch tip before B2** —
   the followup doc's enumerated line numbers (4063, 4360, 4768,
   4919, etc.) are stale. Current MQ3G256 matcher concentration is
   around `qwen35.rs:3315-3348` in
   `forward_prefill_batch_single_chunk_captured`, but a fresh
   `grep -nE 'DType::MQ3G256\b' crates/hipfire-arch-qwen35/src/qwen35.rs`
   at HEAD is the ground truth.
3. **Every matcher** that gates a GEMM dispatch routes to a
   corresponding `gemm_*_mq3g256_lloyd_wmma` arm in
   `crates/rdna-compute/src/dispatch.rs`.
4. **Every dispatch arm** has a kernel file + `kernels.rs` constant +
   arch selector covering both gfx11 and gfx12 paths.

The coherence-gate row is a **separate** soft requirement (regression-
prevention, not corruption-prevention) — track as a companion commit
in the same PR but not part of the all-together package.

## Phasing

### Phase A — MVP single residual kernel + parity test (1–1.5 days)

- Implement `gemm_mq3g256_lloyd_residual_wmma.hip` only (gfx11 first;
  gfx12 sibling is Phase B1).
- Add a parity-test example
  `crates/rdna-compute/examples/test_gemm_mq3g256_lloyd_residual_wmma.rs`
  comparing kernel output vs CPU reference at random shapes
  (N ∈ {16, 64, 256}, K ∈ {1024, 4096, 12288}, M ∈ {64, 256, 1024}).
- Tolerance: **measured-and-set, not specified upfront.** Run the
  kernel on real Qwen3.5-9B Lloyd `down_proj` weights at K=12288;
  observe the max-abs error vs CPU reference; set tolerance to ~3×
  observed (typical fp32-acc-from-fp16 with K=12288 lands in the
  5e-4 to 1e-3 range; pin down empirically).
- **fp16 vs fp32 codebook in LDS:** implement BOTH variants behind a
  build flag or sibling files; benchmark on the same parity test
  shapes; pick the faster for B1's family.
- No dispatch.rs arms beyond what the parity test needs (single arm).
  No `is_batchable_la` change. No matcher updates. The kernel is
  invoked only by the parity test.
- Phase A acceptance:
  - Parity test PASS at empirically-set tolerance.
  - **No VGPR spills** in disassembly (verify via `roc-obj-extract`
    or equivalent on the gfx11 build).
  - **K-tile inner loop has zero `__syncthreads()`** (verify by grep
    on source AND by reading the AMDGCN disassembly for `s_barrier`
    instructions in the inner loop).
  - **fp16-vs-fp32 LDS choice resolved** with empirical bench numbers
    in the Phase A devlog (committed to
    `benchmarks/results/devlog_2026XXXX_lloyd_wmma_phase_a.md`).

### Phase B1 — kernel family + arch selectors (kernels-only commit, 1.5–2 days)

- 7 additional kernel files (3 fused gfx11 + 4 gfx12 siblings, since
  Phase A landed only the residual gfx11).
- `kernels.rs` constants for all 8 files + 4 `*_for_arch` selectors
  (one per kernel shape) covering gfx1100/1101/1102/1150/1151
  (rdna3) and gfx1200/1201 (rdna4); other archs fall through to
  `forward_scratch` (correct, slower).
- Parity tests for each new fused kernel — these have different
  fan-in/fan-out from the residual kernel. **Use mocked-but-
  distinguishable inputs** (e.g. weight-tensor-i has all zeros
  except row i has all-ones, X has lane-i = 1.0 elsewhere 0) to
  catch arg-reordering bugs at parity-test time, not at integration
  time.
- **No `dispatch.rs` arms, no matcher updates, no `is_batchable_la`
  change.** Kernels exist as dead code (constants are loaded but
  never invoked through the dispatch layer). Zero corruption risk.
- Single commit, reviewable as pure kernel work.

### Phase B2 — corruption-prevention all-together commit (1–2 days)

- Add `dispatch.rs` arms for all 4 kernel shapes (signatures mirror
  the HFQ3 WMMA arms exactly; copy structure verbatim, change only
  the kernel constant + module name + group-stride constant).
- Define `pub const LLOYD_MQ3_GROUP_BYTES: usize = 112;` somewhere
  central (e.g. `crates/rdna-compute/src/dispatch.rs` or `kernels.rs`).
  Every Lloyd dispatch arm references this constant; **no magic-number
  `* 112` in dispatch arms**.
- Update `is_batchable_la` (`qwen35.rs:3676`) to include
  `MQ3G256Lloyd` on gfx11/gfx12 archs.
- Update every matcher arm in `qwen35.rs` that handles `MQ3G256` to
  also handle `MQ3G256Lloyd`. Run the audit grep at the branch tip:
  ```
  grep -nE 'DType::MQ3G256\b' crates/hipfire-arch-qwen35/src/qwen35.rs
  ```
  Each hit gets a Lloyd companion arm or is documented as
  intentionally-divergent. Cross-reference with PR #190 if any new
  pp-related dispatch sites need touching.
- Build clean, parity tests still PASS for individual kernels,
  coherence-gate green.
- Single commit (the all-together discipline). The diff will be
  300-500 lines but contains the 4 corruption-prevention conditions
  in lockstep.

#### Phase B2 reviewer checklist

Mechanical, auditable yes/no items:

1. [ ] `is_batchable_la` returns true for `DType::MQ3G256Lloyd` on
   `gfx1100 | gfx1101 | gfx1102 | gfx1150 | gfx1151 | gfx1200 | gfx1201`,
   matching the existing MQ3G256 arm's arch list verbatim.
2. [ ] Every line returned by
   `grep -nE 'DType::MQ3G256\b' crates/hipfire-arch-qwen35/src/qwen35.rs`
   at branch tip has a corresponding `DType::MQ3G256Lloyd` arm or is
   explicitly documented as intentionally not extended.
3. [ ] Every matcher branch that calls into a GEMM dispatch routes to
   a `gemm_*_mq3g256_lloyd_wmma` arm in `dispatch.rs`.
4. [ ] Every Lloyd dispatch arm computes weight strides via
   `LLOYD_MQ3_GROUP_BYTES = 112`, NOT a hardcoded 104 or 112. Grep:
   `grep -E '* 1(04|12)' crates/rdna-compute/src/dispatch.rs` should
   show no Lloyd-related hits — every stride goes through the named
   constant.
5. [ ] K % 256 == 0 holds for every projection in
   Qwen3.5-{4b,9b} layer dimensions. Verify by reading the model
   config and listing wq/wk/wv/wqkv/wz/w_beta/w_alpha/w_gate/w_up/
   w_down/wo K-dimensions; all must divide 256.

### Phase B3 (companion commit) — coherence-gate row

- Add a long-prompt row to `scripts/coherence-gate.sh` for
  `qwen3.5-{4b,9b}.mq3-lloyd` that exercises the batched-prefill
  path (token count ≥ `MIN_BATCH`, single forward pass).
- Reuse the existing 4B / 9B Lloyd model files (already on disk;
  no new model dependency).
- Commit the prompt as a separate file (per CLAUDE.md prompt-md5
  rule), referenced by md5 in the gate script.
- Soft requirement: gate passes (fluent + on-topic + no attractor
  loops). Not part of the Phase B2 corruption-prevention package.

### Phase C — perf validation + ship gate decision rules

**Bench tool prerequisite:** identify or build a prefill-perf bench.
Candidates to evaluate first:
- `target/release/examples/bench_qwen35_mq4` — used by
  `probe_commits.sh` for decode; check whether `--prefill` flag
  exists and produces a `prefill_tok_s` metric.
- `target/release/examples/perplexity` — emits `pos= N scored= M ...
  (X tok/s)` per window which is effectively prefill tok/s; usable
  but coarse.
- New `bench_qwen35_prefill.rs` if neither suffices (~half day to
  build).

**Bench config:** cross-process A/B via `probe_commits.sh` or
equivalent (per CLAUDE.md, within-session A/B has ±10–15% drift —
unusable for sub-10% perf decisions). Compare 9B Lloyd-MQ3 prefill
against:

- Pre-Phase-B baseline (per-token `forward_scratch` path, ~108 tok/s
  per PR #181 future-work section).
- 9B uniform MQ3 prefill (the structural ceiling; same kernel shape
  modulo decode form).

**Decision rules:**

| Lloyd-MQ3 prefill / MQ3 non-Lloyd prefill | Action |
|---|---|
| ≥ 60% | **Ship.** Even at 60%, Lloyd-MQ3 prefill is ~3× better than the per-token fallback (108 tok/s). |
| 30% to 60% | **Investigate.** Likely culprits: LDS bank-conflict regime (try fp16 if fp32 was picked, or vice versa), reconvergent-sync timing on cooperative load, K-tile schedule. Re-bench after each variant. |
| < 30% | **Stop.** Request maintainer input — there's a structural issue in the WMMA + LDS combination that needs deeper investigation than this Phase 5 scope allows. |

Watch-item from Gemini's review: gfx12's lane-group K-split
(`tid >> 4` selecting K-chunk within row) is a cleaner LDS-sharing
pattern than gfx11's full-tile-per-lane mapping. If gfx11 underperforms
relative to gfx12 in Phase C, this is a candidate root cause.

**Phase C results (recorded 2026-05-08):**

- gfx1100 (7900 XTX, GDDR6): Lloyd / uniform = 88.2 % → ship.
- gfx1151 (Strix Halo APU, LPDDR5x): Lloyd / uniform = 96.7 % → ship.

Both gfx11-class hosts clear the gate comfortably. gfx1151's higher
ratio reflects a memory-bound regime narrowing the per-tile-overhead
gap — useful precedent for future per-row-codebook formats. See
`benchmarks/results/devlog_20260508_lloyd_wmma_phase_c.md` for the
full numbers + ratio tables. gfx12 (RDNA4) parity remains community-
CI work.

## Out of scope (deferred)

- **MQ2-Lloyd WMMA prefill** — research-only format; PR #115 keeps
  MQ2-Lloyd permanently behind `--allow-mq2-lloyd`. Revisit if/when
  GPTQ/RHT lifts MQ2 absolute quality.
- **MQ4-Lloyd WMMA prefill** — different question (16-entry codebook,
  K2-vs-K4 LDS layout per issue #182). Scope separately after this
  lands and the per-row-codebook pattern is validated under WMMA.
- **MoE-LA / MoE-FA matchers (issue #179)** — orthogonal silent-
  correctness gap on plain MQ3, not Lloyd-specific. Track in B2's
  matcher audit: if #179 lands before B2, inherit the fix; if not,
  defer to a separate PR (don't widen Phase 5 scope).
- **PR #190 multi-GPU PFlash interaction** — verify before B2 whether
  PR #190 added any new dispatch sites; add to B2 matcher audit. If
  the new pp arms route through `is_batchable_la`, they're covered;
  if they have a parallel gate, that's its own audit.

## Risks and watch-items

- **WMMA + LDS spill under register pressure.** PR #181's K4+LDS
  GEMV was 74 VGPRs / 0 spills. WMMA lanes use vector-aliased
  registers; per-row codebook indexing into 128-entry LDS may push
  VGPR pressure over the `__launch_bounds__(32, 2)` budget.
  **Mitigation:** Phase A acceptance includes a disassembly check
  for spills.
- **gfx12 sibling kernels untestable on local hardware.** Bench host
  has gfx1100; the secondary box has gfx1151. **gfx1200/1201 (RDNA4)
  is unavailable.** Phase B1's gfx12 kernels will compile-test only.
  Either (a) gate the gfx12 selector arms behind a build-time feature
  for now, (b) ship as code-complete-but-runtime-unvalidated with an
  explicit note in the kernel header, or (c) request community CI on
  RDNA4 hardware before merge. Decide at Phase B1 start.

  **Resolved (2026-05-08, post-review):** option (a)-flavoured runtime
  env gate. `is_batchable_la` only returns true for `MQ3G256Lloyd` on
  gfx1200/1201 when `HIPFIRE_LLOYD_GFX12=1`; default behaviour falls
  through to per-token `forward_scratch` (correct, ~14× slower; matches
  pre-Phase-B2 baseline). RDNA4 reviewers set the env var to exercise
  the gfx12 WMMA path. The captured-prefill entry point has a parallel
  refusal so the gate can't be bypassed.

  Two follow-ups gated behind external RDNA4 CI:
  - confirm gfx12 parity tests at max_abs ≤ Phase A envelope;
  - confirm gfx12 coherence-gate row produces fluent output;
  …then drop the env gate (or default-flip it) in a follow-up commit.
- **Codebook header alignment.** 8 fp16 entries × 2 B = 16 B header at
  offset 0 of each group. Verify that `quantize_mq3g256_lloyd` writes
  group bases at ≥ 16-B alignment. If yes, vectorized loads are
  possible later as an optimization. If no, cooperative load must
  stay scalar-per-fp16. **Phase A pre-check.**
- **Codebook staging vs HFQ3's claimed `LDS: 0`.** HFQ3 WMMA kernel
  headers report zero LDS use. Adding 256–512 B of LDS is trivial in
  absolute terms but verify it doesn't change the wave occupancy
  (`__launch_bounds__(32, 2)` allows 2 waves/SIMD; 256 B × 2 waves =
  512 B per workgroup pair, well under the 64 KB LDS budget).
- **Parameter-ordering bug class in dispatch.rs.** The HFQ3 fused
  kernels have non-trivial dispatch signatures (qkvza takes 4 weight
  pointers + 4 output pointers + 4 m-counts = 13 args). A copy-paste
  port that swaps any two pointers reads the wrong weight tensor →
  silent corruption. **Mitigation:** Phase B1's parity tests use
  mocked-distinguishable inputs that catch arg-reordering bugs at
  parity-test time.

## Open questions

Reduced to one (down from four after cross-validation closed three):

- **fp16 vs fp32 codebook in LDS** (Phase A measurement). fp16 saves
  conversion cost when assigning to `half16_t a_reg`; fp32 has cleaner
  bank-conflict math. The faster variant on real Qwen3.5-9B weights
  ships in B1.

Closed:
- ~Multi-acc/single-acc for WMMA~ — confirmed: HFQ3 WMMA uses single
  `float8_t acc` accumulated in-place per K-tile; no merge step. The
  cross-path drift vs GEMV decode (covered by acceptance criterion 3)
  is hardware-defined and addressed at the criterion level, not the
  kernel-structure level.
- ~gfx11/gfx12 unification~ — confirmed: not viable. Separate files
  (matches HFQ3 convention).
- ~B=1 fallback site coverage~ — confirmed: `is_batchable_la` only
  gates the batched path; B=1 decode goes through
  `weight_gemv_prerotated` which already has the Lloyd GEMV arm
  shipped in PR #181. No B=1 changes needed.

## References

- Issue #116 — original Lloyd-MQ3 ship-gate issue with the Phase 5
  out-of-scope item this plan addresses.
- Issue #182 — MQ4-Lloyd implementation (codebook-in-LDS rationale,
  K2/K4 layout discussion).
- Issue #188 — multi-acc vs single-acc decision (cross-path drift
  envelope; same fp32-reorder pattern recurs in WMMA prefill vs
  GEMV decode).
- PR #181 — Lloyd-MQ3 GEMV K4+LDS work; codebook-LDS pattern source.
- PR #189 — gfx1151 enablement for MQ3-Lloyd GEMV (arch list to
  match in Phase B2).
- PR #190 — multi-GPU PFlash; verify dispatch-site impact during B2
  matcher audit.
- `docs/plans/mq-lloyd-batched-prefill-followup.md` — the all-
  together discipline + matcher-trap walkthrough. Re-read before
  Phase B2.
- `docs/plans/PR-115-lload-max-codebooks-mq3.md` — original Lloyd-MQ3
  GEMV plan (rev 2).
- `kernels/src/gemm_qkvza_hfq3g256_wmma.hip` — primary structural
  template.
- `kernels/src/gemv_mq3g256_lloyd.gfx1100.hip` — codebook-LDS
  reference for the decode replacement.
- `kernels/src/gemv_mq4g256_lloyd.gfx1100.hip:67-79` — 2-phase
  cooperative-load template for 16-entry codebook (adapt for our
  per-row layout).
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3676` —
  `is_batchable_la`, the gate to widen.
- `crates/hipfire-quantize/src/main.rs:642+` — `quantize_mq3g256_lloyd`,
  confirms FWHT rotation applied at quant time.
