# MQ4-Lloyd batch-tile fanout (Phase D — large-batch GEMM)

**Branch:** `feat/mq4-lloyd-batch-fanout` (off `feat/issue-182-mq4-lloyd`).
**Targets:** gfx1100/1101/1102/1150/1151 (RDNA3+3.5 wave32 WMMA) initially. gfx12 (RDNA4) deferred to D-B if D-A wins.
**Date:** 2026-05-09.
**Depends on:** PR #197 (MQ4-Lloyd Phase 5b WMMA prefill landed). This work modifies the kernel family that PR introduced.

## Goal

Close the prefill-perf gap that opens at batch ≥ 128 between MQ4-Lloyd
and uniform MQ4 on bandwidth-bound hardware (gfx1151 LPDDR5x; less
acutely on gfx1100 GDDR6, where L2 absorbs more of the duplicate
fetch).

Diagnosed in `benchmarks/results/devlog_20260509_mq4_lloyd_gfx1151_bench.md`:
HFQ4 dispatches a 128×128-tile LDS-staged `mmq` kernel for batch ≥ 128
that holds 25 GiB/s flat. MQ4-Lloyd has no equivalent — it stays on a
16×16-tile WMMA kernel and decays from 26 → 10 GiB/s on `gate_up` as
batch grows from 64 → 1024. Direct i8-mmq port is not viable for
Lloyd (i8 WMMA wants affine-reconstructible weights; Lloyd's 16-entry
codebook stores arbitrary fp16 values).

This plan ships an fp16-WMMA kernel that **amortizes weight decode
across multiple batch tiles per workgroup** — the same arithmetic-
intensity-per-byte-fetched fix mmq applied to HFQ4, in a form
compatible with Lloyd's per-row codebook decode.

## Acceptance criteria

1. `cargo check -p rdna-compute -p hipfire-arch-qwen35 -p hipfire-runtime` clean.
2. Parity at the same envelope as Phase A (max-abs ≤ ~1.75e-4 across the canonical 8-shape grid; suggested by Phase A devlog).
3. **gfx1151 9B Lloyd-MQ4 prefill / uniform-MQ4 prefill ≥ 60 %.** Current floor is 41.5 % (devlog 2026-05-09). Hitting 60 % means matching gfx1100's 9B ship gate on bandwidth-bound hardware — closing the cross-arch ratio gap.
4. **gfx1100 9B Lloyd-MQ4 prefill / uniform-MQ4 prefill: no regression vs current 60.6 %.** D's tile shape is wider; if anything, gfx1100 should also gain (less L2 churn from duplicate fetches).
5. No decode regression on the per-token forward path (Lloyd's GEMV decode is untouched by D; verify with `probe_commits.sh` against PR #197 HEAD).
6. Coherence-gate row green for `qwen3.5-9b.mq4-lloyd` on both gfx1100 and gfx1151.

Acceptance criterion 3 is the load-bearing one. Criteria 1, 2, 5, 6 are guard rails.

## What's reusable

The Phase 5b WMMA family (PR #197) is the structural template — D modifies the per-WG output shape but inherits the codebook-in-LDS, sync discipline, and parity-test harness:

| File | Role for D |
|---|---|
| `kernels/src/gemm_mq4g256_lloyd_residual_wmma.hip` | Direct parent of D-A's `_mb` variant. K-tile decode logic copies verbatim; only the WMMA dispatch loop and grid shape change. |
| `kernels/src/gemm_qkvza_mq4g256_lloyd_wmma.hip` | D-B fused sibling. |
| `kernels/src/gemm_qkv_mq4g256_lloyd_wmma.hip` | D-B fused sibling. |
| `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma.hip` | D-B fused sibling — biggest absolute time on the profile, highest-leverage win. |
| `crates/rdna-compute/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs` | Phase A parity harness — gets a `_mb` row added per shape. |
| `kernels/src/gemm_hfq4g256_residual_mmq.hip` | **Reference for the LDS-staging discipline only**, not for the i8 WMMA primitive. The `tile_x` LDS-staging pattern (lines 232-282) and per-WG output-write loop (lines 388-411) are the structural templates. |

## Approach

### Per-workgroup output shape

Current Phase A: 16 rows × 16 batch = 256 outputs per WG.
D-A: 16 rows × 64 batch = **1024 outputs per WG** (4 batch sub-tiles).

Grid changes from `(M/16, batch/16)` → `(M/16, batch/64)`. 4× fewer
workgroup-columns in the batch dim → 4× fewer redundant weight reads
per group across the grid.

Block dim stays `[32, 1, 1]` (one wave32, like Phase A). The 4 batch
sub-tiles share the same `A_reg` decode per K-tile via 4 sequential
WMMA dispatches, streaming B-tiles one at a time.

### Weight reuse pattern

Per K-tile inner loop, current Phase A:

```
decode A_reg (16 nibbles × 16 rows from global → cb_lds lookup)
wmma(A_reg, B_a, acc)
```

D-A:

```
decode A_reg                                  # same as Phase A
b0 = load X[batch_start_0 : +16, kt]
wmma(A_reg, b0, acc[0])
b1 = load X[batch_start_1 : +16, kt]          # reuses A_reg
wmma(A_reg, b1, acc[1])
b2 = load X[batch_start_2 : +16, kt]
wmma(A_reg, b2, acc[2])
b3 = load X[batch_start_3 : +16, kt]
wmma(A_reg, b3, acc[3])
```

4× the WMMA ops per A decode → 4× the per-byte arithmetic intensity
for weights. X reads grow 4× per WG but X duplicate factor across
the grid drops by the same 4× (fewer row-tile WGs re-reading the
same X span), so net X bandwidth is roughly unchanged.

### Why 4 batch tiles, not 8 or 2?

- **2 batch tiles** (16×32 output): only 2× weight reuse. Diminishing returns vs implementation cost; doesn't close the gap to HFQ4 mmq.
- **4 batch tiles** (16×64): 4× weight reuse. Predicted GiB/s on `gate_up` lifts from 11.4 → ~30 (within HFQ4 mmq's 25-26 GiB/s envelope, modulo the ~25 % within-WMMA format-tax overhead). VGPR budget headroom comfortable.
- **8 batch tiles** (16×128): 8× weight reuse but 8× the `acc` arrays (64 VGPRs for acc alone) plus the wider-batch codepath needs more X tiles in flight. Pushes against the `__launch_bounds__(32, 2)` envelope. **D-B candidate** if D-A wins and bench shows headroom.

D-A ships at 4. D-B (fused siblings) inherits 4. A possible D-C
revisit to 8 only if Phase D bench shows compute-not-memory headroom.

### Register pressure prediction

Phase A residual: 68 VGPR with `__launch_bounds__(32, 2)`, 0 spills (per devlog 2026-05-07 disassembly metadata).

D-A delta:
- 4× `float8_t acc` → 32 VGPR (vs Phase A's 8)
- B-streaming (one B-reg in flight) → +0 VGPR
- A_reg unchanged → 8 VGPR
- Per-output-write index arithmetic → ~+4 VGPR

Predicted total: ~95-100 VGPR. **Tight against the 2-occupancy slot at `__launch_bounds__(32, 2)`** — may force a drop to `(32, 1)`. Acceptance gate is the bench, not occupancy: a 4× weight-reuse win that costs 1 occupancy slot is still a clear net win on bandwidth-bound hardware.

If VGPR pressure spills, fall back to streaming acc accumulation (write one batch-tile's acc to global before starting the next) — costs an extra global write per group but eliminates 24 VGPR. Decision deferred to Phase D-A bench.

### Output-write convention

Phase A: `acc[j] = C[2*j + (tid>>4)][tid & 15]` for j in [0..8) — 8 outputs per lane, 32 lanes covering 16 rows × 16 batch.

D-A: same lane mapping per batch sub-tile, repeated 4× along the batch axis. Per lane: 32 outputs (8 per batch sub-tile × 4 sub-tiles), each acc[j] writes to `(out_col + sub_tile * 16) * M + out_row`. Bounds checks per sub-tile preserve Phase A's safe-row/safe-batch padding pattern.

## Phases

### D-A: residual kernel + parity + bench (the load-bearing phase)

1. New file `kernels/src/gemm_mq4g256_lloyd_residual_wmma_mb4.hip`. Derived from `_wmma.hip` with the multi-batch-tile dispatch above. Single variant — no fp32-LDS sibling, fp16-LDS settled in Phase A.
2. Add `GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_MB4_SRC` constant + `gemm_mq4g256_lloyd_residual_wmma_mb4_for_arch()` selector in `crates/rdna-compute/src/kernels.rs`. Same arch matrix as Phase 5b (gfx1100/1101/1102/1150/1151 currently; gfx12 deferred).
3. New `gemm_mq4g256_lloyd_residual_wmma_mb4(...)` method on `Gpu` in `dispatch.rs`. Same args as `_wmma`, different grid dims.
4. Extend `crates/rdna-compute/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs` with a `_mb4` row per shape. Verify parity at `≤ 1.75e-4` envelope inherited from Phase A.
5. Disassembly check via `gfx-kernel-metadata` skill: VGPR count, spill-segment-size, occupancy. Land result into D-A devlog.
6. Single-kernel bench (no dispatch wiring yet): time `_wmma` vs `_wmma_mb4` at K=4096, M=4096, batch ∈ {16, 64, 256, 1024}. Confirm batch=256 closes ≥ 50 % of the gap to HFQ4 mmq's 26 GiB/s.

**Phase D-A ship gate (decides whether D-B is worth doing):**

- If `_mb4` reaches ≥ 22 GiB/s on the residual at batch ≥ 128 on gfx1151: **proceed to D-B**.
- If 18-22 GiB/s: **investigate** B-streaming vs B-parallel-load, async LDS prefetch, K-unroll variants.
- If < 18 GiB/s: **stop and reconsider**. The multi-batch-tile theory was wrong or the implementation is leaving more on the table than the ship gate tolerates. Re-read the gfx1151 devlog and look for a missed signal.

### D-B: fused siblings (qkvza, qkv, gate_up)

Once D-A confirms the win, port the same multi-batch-tile pattern to the three fused kernels. Each is a tighter rewrite than D-A because the fused kernels already have multi-projection fan-out internally — the modification is along the batch axis only.

Files: `gemm_qkvza_mq4g256_lloyd_wmma_mb4.hip`, `gemm_qkv_mq4g256_lloyd_wmma_mb4.hip`, `gemm_gate_up_mq4g256_lloyd_wmma_mb4.hip`. Each gets an `_mb4_for_arch()` selector and a dispatch method.

Parity tests extend `test_gemm_fused_mq4g256_lloyd_wmma.rs` similarly.

D-B ship gate: the four-kernel set (residual + 3 fused) collectively reaches the criterion-3 60% Lloyd / uniform ratio on gfx1151 at the canonical bench config (`bench_qwen35_mq4 --prefill 256 --prefill-runs 3`).

### D-C: dispatch wiring + ship-gate bench

1. Wire the `_mb4` kernels into `is_batchable_la` and the LA/FA matchers in `qwen35.rs`. **Threshold-gated dispatch**: route to `_mb4` when `batch_size ≥ 64`, fall through to the existing `_wmma` family otherwise. Pattern follows HFQ4's WMMA-vs-mmq path selection.
2. The captured-path corruption-prevention guards already cover MQ4-Lloyd in dense and refuse it in MoE. Extend the LA/FA matchers to dispatch `_mb4` over `_wmma` only when the batch-size-gate fires; the guards remain unchanged. Adding `_mb4` does not widen the corruption surface — the stride is unchanged at 160 B/group.
3. Cross-process A/B bench identical to PR #197 Phase C: 3 invocations × 2 models (uniform + Lloyd) × 2 sizes (4B + 9B) on gfx1151. Add a gfx1100 row for non-regression confirmation.
4. Coherence-gate green on both gfx1100 and gfx1151 with `qwen3.5-9b.mq4-lloyd`.

## Out of scope (deferred)

- **gfx12 (RDNA4) `_mb4` siblings.** Defer to a follow-up. PR #197 already lacks gfx12 parity; D doesn't widen that gap. The gfx11 path is the gating one for the user's gfx1151 host.
- **8-batch-tile (`_mb8`) variant.** Only worth doing if D-A bench shows compute-bound headroom. Predicted ROI < D-A's.
- **MQ3-Lloyd `_mb` siblings.** MQ3-Lloyd at 88 % of uniform on gfx1100 isn't bottlenecked the same way. Run the gfx1151 cross-process A/B on MQ3 first to see if there's even a gap to close.
- **Codebook deduplication across rows.** A separate optimization axis: if rows in a tile share codebook layouts, LDS pressure could shrink. Not orthogonal to D, but a different direction.
- **i8 codebook quantization → mmq path.** Quantizing the 16-entry fp16 codebook to i8 + scale would unlock the mmq i8-WMMA primitive but would add a second quality-degradation surface on top of Lloyd's K-means. Not worth pursuing without a quality eval first.

## Risks and watch-items

- **VGPR pressure pushing past `__launch_bounds__(32, 2)`.** Mitigation in approach section. **Watch:** the disassembly metadata in D-A's devlog. Fall-back is `(32, 1)` + bench.
- **D-A wins on the residual but D-B hits diminishing returns on `gate_up`.** `gate_up` is the largest absolute time on the profile (44 % of prefill on Lloyd at prefill=256). If D-B `gate_up_mb4` doesn't lift commensurately, the criterion-3 ratio gate may not clear even with a great D-A. **Mitigation:** D-A's bench at K=4096 / batch=256 is a good proxy for `gate_up`'s shape (gate_up is ~K=4096 → 4×K projection); a D-A win there strongly predicts a D-B win.
- **gfx1100 regression risk from the path change.** D-C threshold-gates `_mb4` at batch ≥ 64 — small-batch decode and small-prefill paths are unchanged. Cross-process A/B in D-C catches any regression. If gfx1100 regresses but gfx1151 wins, consider per-arch threshold (gfx1100 might prefer `_mb` only at batch ≥ 256 if its L2 absorbs the smaller-tile duplicate fetches better).
- **Cross-path drift envelope.** `_mb4` accumulation order differs from `_wmma` by the inter-batch-tile interleaving. Same fp32-reorder envelope as Phase A's WMMA-vs-GEMV drift (PR #197 acceptance criterion 3, ≤ 1 % PPL). Bench confirms.
- **L1 capacity pressure from 4× X tiles.** Each WG now reads 4 batch rows × 16 cols × 2 B = 128 B per K-tile from X. Over 16 K-tiles per group × K/256 groups, this scales linearly. RDNA3 L1 is 32 KB per CU; well under capacity, but worth a profile-level confirmation post-D-A.

## Open questions

1. **B-streaming vs B-parallel-load.** Approach section assumes streaming (load b0, wmma, load b1, wmma, ...). Parallel-load (load b0..b3 first, then 4 wmmas) trades 24 VGPR for ILP. Parallel may schedule better but pushes VGPR pressure. Decide by D-A bench: if streaming under-runs LDS bandwidth, switch to parallel.
2. **Whether `_mb4` should also stage weights in LDS.** Current Phase A reads weights inline from global per K-tile. With 4× weight reuse already provided by multi-batch-tile, LDS staging adds another factor only if the kernel is still global-bound on weights post-D-A. Defer to D-A bench. If LDS-staged weights become the next lever, D-B's `gate_up_mb4` is the natural place to introduce the pattern.
3. **Whether the threshold gate at batch ≥ 64 is correct on gfx1100.** On gfx1100, the existing `_wmma` kernel already runs near 30 GiB/s at small batches; `_mb4` may not lift gfx1100 small-batch path the same way. Threshold may need to be arch-specific (gfx1151: ≥ 64; gfx1100: ≥ 256).

## References

- `benchmarks/results/devlog_20260509_mq4_lloyd_gfx1151_bench.md` — diagnostic data; the per-kernel profile + batch sweep + reframed lever menu.
- `docs/plans/mq4-lloyd-wmma-prefill.md` — Phase 5b plan (PR #197); D's parent plan.
- `kernels/src/gemm_mq4g256_lloyd_residual_wmma.hip` — Phase A parent for D-A.
- `kernels/src/gemm_hfq4g256_residual_mmq.hip` — LDS-staging structural reference (fp16-WMMA port of the discipline; not the i8 WMMA primitive).
- `crates/rdna-compute/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs` — parity test harness; D-A extends with `_mb4` row per shape.
- `crates/rdna-compute/examples/bench_qwen35_mq4` (via `target/release/examples/bench_qwen35_mq4`) — D-A and D-C use this for cross-process A/B.
- `docs/skills/gfx-kernel-metadata` — disassembly recipe for the VGPR/spill check.
- PR #197 — MQ4-Lloyd Phase 5b; D's predecessor.
- Issue #182 — MQ4-Lloyd implementation tracking; D rolls under this issue.
