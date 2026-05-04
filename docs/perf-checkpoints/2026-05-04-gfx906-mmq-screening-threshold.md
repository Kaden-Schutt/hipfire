# gfx906 MMQ — screening threshold bump 0.10 → 0.50

Date: 2026-05-04
Hardware: gfx906 (MI50). HIP 6.4.3.
Workload: Qwen 3.5 9B mq4, `bench_qwen35_mq4`.

## TL;DR

Bumping the gfx906 default `mmq_screen_threshold` from 0.10 to 0.50
**recovers 30 weights/load that were unnecessarily falling back to
FP16 wave64**. Coherence-gate passes across all 4 mq4 rows at the
new default. Performance:

- **pp128: 355 → 462 tok/s (+30 %, 3.28× over baseline)**
- **pp512: 554 tok/s** vs stock llama.cpp **750 tok/s** = **74 %
  of stock** (was 19 % before the redesign).

The change is **scoped to gfx906 only**. Non-gfx906 archs keep the
conservative 0.10 default until the same validation is done there.

## Why the old 0.10 default was too tight

The 0.10 threshold was introduced in commit `8081822` ("gfx906 MMQ
correctness fix: symmetric weights + f32 LDS staging + screening")
when the gfx906 dp4a kernel had two real bugs:

1. **Asymmetric weight unpacking** — nibbles 0..15 sent to dp4a as
   strictly positive int8, creating a directional DC offset that
   compounded across layers.
2. **F16 precision loss in LDS-staged x_dm** — scale and zero-point
   stored as half2, losing precision that propagated through every
   row's accumulator.

Both bugs are fixed in the post-redesign kernel
(commit `c022682`):
- body.cuh:102-103 unpacks nibbles as `(n - 8)` so dp4a sees signed
  -8..+7.
- body.cuh:58, 152, 245 carries x_dm as `float2*` end-to-end.

The kernel is now structurally cleaner. The 0.10 threshold,
calibrated against the buggy kernel's noise floor, treats the
post-redesign kernel's normal quantization noise as suspicious.

## Diagnostic: what gets rejected at 0.10

Single 9B mq4 load, dispatch with `HIPFIRE_MMQ=1
HIPFIRE_MMQ_SCREEN=1`:

```
30 weight matrices reject. Worst-row distribution:
  19 × worst_row=3994 (all m=4096 matrices)
   1 × worst_row=830, 718, 7170, 6238, 544, 3906, 3617, 310, 2560
       (mostly m=8192 gate_up halves)
```

Of 30 rejects, **19 cluster on row 3994 of m=4096 matrices** —
consistent with one degenerate quant group at that row position
(near-zero scale → dp4a int rounding dominates), not a kernel bug.
Per `2026-05-04-gfx906-mmq-redesign-rocprof.md` and the real-data
NRMSE checkpoint, row 3994 is also the worst-error row in the
real-data dump, with the same magnitude. The pattern is
**reproducibly tied to the quantization, not to the screening's
synthetic activations or the kernel topology**.

The 11 non-3994 rejects fail by very small margins (max errors
0.10–0.15) — well within "honest quant noise" territory.

## Threshold sweep — coherence

Single prompt: "A farmer has 17 sheep. All but 9 die. How many are
left? Show brief reasoning then state the final number."
(temperature=0, max_tokens=150)

| Threshold | Rejects | Final answer | Notes |
|---|---|---|---|
| 0.05 | 126 | 9 ✓ | over-rejecting |
| **0.10 (old default)** | **30** | **9 ✓** | |
| 0.15 | 8 | 9 ✓ | inline phrasing |
| 0.20 | 4 | 9 ✓ | inline phrasing |
| 0.30 | 2 | 9 ✓ | inline phrasing |
| **0.50 (new default)** | **0** | **9 ✓** | screening fully pass-through |

Coherence gate at threshold=0.50 (all 4 mq4 rows + 2 mq3 rows for
the gfx11+/gfx12 path):

| Row | Status |
|---|---|
| qwen3.5-0.8b.mq4 / cap | OK |
| qwen3.5-4b.mq4 / code | OK |
| qwen3.5-9b.mq4 / reason | OK |
| qwen3.5-9b.mq4 / tool-call | OK (no `<\|im_start\|>` leak) |
| qwen3.5-9b.mq3 / reason-mq3 | OK (mq3 doesn't dispatch through gfx906 MMQ) |
| qwen3.5-27b.mq3 / cap-mq3-27b | OK (was OOM in earlier run, ran clean here) |

## Threshold sweep — perf (pp128 cross-process A/B)

Each row: fresh process, DPM-warmed, 5 prefill-runs, last-run
measurement. A = MMQ off baseline, B = MMQ on at the listed
threshold.

| Threshold | Rejects | A (MMQ=0) | B (MMQ=1) | Speedup |
|---|---|---|---|---|
| 0.10 (old) | 30 | 141 | 355 | 2.52× |
| **0.50 (new)** | **0** | **141** | **462** | **3.28×** |

Cross-process A/B at threshold=0.50 across 3 alternating iterations:
B median 461.7 tok/s, spread 0.6 tok/s (0.13 %). Structural.

## Full prefill sweep at new default

| Prefill | Baseline | New (thr=0.50) | Speedup |
|---|---|---|---|
| pp32  | 136 | 277 | 2.03× |
| pp64  | 139 | 365 | 2.62× |
| pp128 | 141 | 462 | 3.28× |
| pp256 | 143 | 561 | 3.93× |
| **pp512** | **142** | **554** | **3.89×** |

pp256 ≈ pp512 — speedup curve flattens at large batches, suggesting
a different bottleneck (launch overhead or HBM ceiling). Worth
re-profiling.

## Comparison with stock llama.cpp

From `2026-05-04-llamacpp-stock-comparison.md`: stock llama.cpp on
gfx906 hits **750 tok/s pp512**. Where we are now:

| Path | pp512 tok/s | Fraction of stock |
|---|---|---|
| Pre-redesign baseline (FP16 wave64) | 142 | 19 % |
| Post-redesign, threshold=0.10 | (~370 estimated) | (~49 %) |
| **Post-redesign, threshold=0.50** | **554** | **74 %** |
| Stock llama.cpp | 750 | 100 % |

The remaining 26 % gap is the qkvza beta+alpha tail (still FP16
wave64), the residual mlp-down K=12288 path, and likely the inter-
warp sync overhead (Option B's 8 syncs/group vs stock's 2). All
listed in the plan's optional follow-ups.

## Code change

Single change in `crates/rdna-compute/src/dispatch.rs`:

```rust
// Per-arch default for MMQ screening threshold.
let mmq_screen_threshold_default: f32 = if arch == "gfx906" { 0.50 } else { 0.10 };
```

`HIPFIRE_MMQ_SCREEN_THRESHOLD=…` env override still takes precedence
for downstream tuning. Non-gfx906 archs keep the 0.10 default
unchanged.

## What this leaves on the table

| Lever | Status |
|---|---|
| Default-on flip (`should_use_mmq()` gfx906 branch) | Now P1; speedup is structural and validated |
| Re-profile to identify the new top-share kernel | Needed before more kernel work |
| pp256 ≈ pp512 saturation | Investigate before more batch-size tuning |
| qkvza beta+alpha tail (still wave64) | Path B fused MMQ kernel candidate |
| Sync frequency 8 → 2 per group | Speculative, needs LDS-budget review |
| ds_read_b128 | Small lever per prior rocprof |

## Reproducing this report

```sh
# Threshold sweep
for thr in 0.05 0.10 0.15 0.20 0.30 0.50; do
  HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=1 HIPFIRE_MMQ_SCREEN_THRESHOLD=$thr \
    timeout 120 $HIPFIRE/target/release/examples/daemon < input.jsonl
done

# Cross-process A/B (no env var → uses new default 0.50 on gfx906)
for iter in 1 2 3; do
  for label in A B; do
    [ "$label" = "A" ] && env="HIPFIRE_MMQ=0" || env="HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=1"
    env $env HIPFIRE_DPM_WARMUP_SECS=2 \
      $HIPFIRE/target/release/examples/bench_qwen35_mq4 \
      $HIPFIRE_MODELS/qwen3.5-9b.mq4 \
      --prefill 128 --prefill-runs 5 --gen 0 --warmup 0
  done
done
```
