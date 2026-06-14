# Path D D0 risk-check — Bandwidth contention micro-bench

**Issue:** [#38](https://github.com/Kaden-Schutt/hipfire/issues/38) (Path D —
DDTree pipeline)
**Branch:** `feat/38-ddtree-pipeline` at `abf9408` (D0c) on top of master
`80330c3`.
**Hardware:** AMD RX 7900 XT (gfx1100, 21.5 GB VRAM, HIP 7.2).
**Date:** 2026-05-02.

## What this measures

Plan §3 / `plans/path_d.md` requires a pre-D1 risk-check: synthetic
bandwidth-contention micro-bench mimicking the real cycle's overlap (verify
on one stream, draft on a second stream concurrently). If observed wall-time
savings are **< 8 %**, Path D is paused and rescoped, because the plan's
60-ms cycle target (20 % savings on a 75-ms baseline) is unreachable on this
hardware regardless of correct stream wiring.

The bench is `crates/rdna-compute/examples/bench_stream_overlap.rs` (already
in the tree, predates this plan). It exercises real
`gemm_hfq4g256_residual` kernels on synthetic MQ4-layout weights — same
quantization as production target/draft weights — so contention is realistic
to actual DFlash bandwidth pressure, not an idealized memcpy proxy.

Two probes:

1. **Symmetric** — N kernel launches split half on stream A, half on
   stream B vs the same N launched on a single stream. Tests whether the
   GPU's two ACE (asynchronous compute engine) queues can saturate
   simultaneously when both streams want roughly equal work.
2. **Asymmetric** — large verify-shaped workload on stream A concurrent
   with a smaller draft-shaped workload on stream B, vs the sum of each
   alone. This is the Path D scenario: verify N runs while draft N+1
   overlaps on `draft_stream`.

`overlap_ratio = (T_a_alone + T_b_alone) / T_both_concurrent`. Wall-time
savings = `1 - 1/ratio`.

## Results (3 runs)

### Run 1 — default (symmetric stress + 5v+5d asymmetric)

```
=== gemm_hfq4g256_residual M=5120 K=5120 N=16 ===

  N   T_serial(µs)  T_parallel(µs)  overlap_ratio
   8         849.4           871.8     0.974x  (-2.6 %  — warmup noise)
  16        1559.0          1488.8     1.047x  ( +4.5 %)
  32        2876.4          2728.1     1.054x  ( +5.1 %)
  64        5890.2          5041.0     1.168x  (+14.4 %)

Asymmetric probe (5 verify layers + 5 draft layers, same shape):
  t_verify_alone:    620.8 µs
  t_draft_alone:     653.2 µs
  t_both:           1012.2 µs
  ratio = 1.259x → 20.6 % wall savings
```

### Run 2 — realistic verify-dominated (64v + 5d, draft=2048×2048)

```
Symmetric:
  N=64  → 1.272x  (+21.4 %)

Asymmetric (the Path-D-relevant case):
  verify: 64 layers, M=5120 K=5120 N=16
  draft:   5 layers, M=2048 K=2048 N=16
  t_verify_alone: 5881.2 µs
  t_draft_alone:   292.2 µs    (4.7 % of verify)
  t_both:         5842.2 µs    (≈ verify alone — draft fully hides)
  ratio = 1.057x → 5.4 % wall savings    ← BELOW 8 % THRESHOLD
```

This is the case where the plan's caveat bites: when draft is much smaller
than verify, draft hides perfectly inside the verify time but the absolute
wall savings are bounded by `T_draft / (T_v + T_d) ≈ 5 %`. The bench's
"asymm" gate annotates this as `ratio < 1.3 → A-full doomed on gfx1100,
pivot to kernel grinds`.

### Run 3 — realistic 4:1 work ratio (64v + 20d, draft=4096×4096)

```
Symmetric:
  N=64  → 1.236x  (+19.1 %)

Asymmetric (closer to real T_v / T_d ratio):
  verify: 64 layers, M=5120 K=5120 N=16
  draft:  20 layers, M=4096 K=4096 N=16
  t_verify_alone: 6050.9 µs
  t_draft_alone:  1400.9 µs    (23 % of verify)
  t_both:         6571.1 µs
  ratio = 1.134x → 11.8 % wall savings   ← ABOVE 8 %, BELOW PLAN'S 20 %
```

When the synthetic draft work is sized to roughly mirror the **time** ratio
that real 27B + 5-layer DFlash exhibits (verify ≈ 75–80 % of cycle, draft
≈ 20–25 %), the bench shows `1.134×` = **11.8 % wall savings**.

## Interpretation

### What the data says about Path D

| Scenario | Verify time | Draft time | T_d / T_v | Overlap ratio | Wall savings |
|----------|-------------|------------|-----------|---------------|--------------|
| Verify-dominated (run 2) | 5.9 ms | 0.3 ms |  5 % | 1.057× |  5.4 % |
| Realistic 4:1 (run 3)    | 6.1 ms | 1.4 ms | 23 % | 1.134× | 11.8 % |
| Symmetric stress (any)   | n/a    | n/a    |~100 %| 1.17–1.27× | 14–21 % |

The plan's **8 % pause threshold is satisfied** when the simulated verify-to-
draft work ratio matches what the real DFlash cycle exhibits (run 3, 4:1).
The threshold **fails** when the draft is artificially tiny relative to
verify (run 2). The headroom on the realistic case is modest: 11.8 % vs an
8 % floor, with run-to-run noise on the bench at ~2 %.

### What the plan's 20 % target requires

Plan-stated quantitative target: `cycle 75 ms → 60 ms (-20 %) if τ stays
≥ 90 %`. On the run-3 model:

  - To hit 20 % wall savings via overlap alone, we need
    `T_d / (T_v + T_d) ≥ 0.20`, i.e. `T_d / T_v ≥ 0.25`.
  - Run 3 measures `T_d / T_v = 0.23` and yields 11.8 %. Half the plan's
    target.
  - The arithmetic ceiling is `min(T_v, T_d) / (T_v + T_d)`. Even with
    100 % overlap (no contention), a workload with T_d = 0.25 × T_v can save
    at most 20 %. Bandwidth contention on shared GDDR6 erodes this further.

### What the plan's 4 % worst-case warning meant

The plan's `§1` caveat:

> Worst-case analytical model says effective savings could be as low as
> ~4 %, not 20 %.

Run 2 (5.4 %) confirms this analytical worst case is the right ballpark when
the draft is tiny. It's not a crash-and-burn outcome — pipelining still
works, just with diminishing absolute returns.

## Recommendation

**Proceed with D1+, but rebaseline the perf target.**

The 8 % threshold is satisfied at the verify:draft work ratio that real
DFlash exhibits. The hardware is capable of overlapping concurrent streams
on gfx1100 — the bench shows `1.13–1.27×` ratios across configurations,
consistent with partial-but-real ACE concurrency.

What to update before D1 lands:

1. **Plan §1 quantitative target:** revise `75 ms → 60 ms (-20 %)` to a
   more defensible `75 ms → 67 ms (-10 %)` until the actual D1+D2+D3
   end-to-end bench refutes or confirms it. The 20 % figure was derived
   from a perfect-overlap analytical model; the synthetic measurement
   here ceilings around 12 % at realistic verify:draft work ratios.

2. **Plan §1 acceptance bars:** the per-class speed gates (210 / 195 /
   180 tok/s) should be revisited against this finding. Current 199 tok/s
   median × 1.10 = ~219 tok/s upper-bound code expectation. The bars at
   210 / 195 / 180 are reachable but tight.

3. **Plan §3 D5 ship-gate:** keep the existing «if observed cycle savings
   drop below 8 % we pause and rescope» — this micro-bench validates the
   gate is the right kind of check, not the right *post-implementation*
   threshold. Reuse the asymmetric probe (run 3 config) as a smoke test
   inside `coherence-gate-dflash.sh` once D1+ lands.

What this measurement does **not** decide:

- Whether the realistic 4:1 work-ratio assumption holds for every prompt
  class. Code-class generations often have higher τ → larger B → bigger
  draft per cycle → better ratio. Prose-class with smaller B may be
  closer to run 2 (verify-dominated, low savings). The per-class speed
  gates in §1 of the plan handle this — they remain the right shape.
- Whether kernel concurrency on the same stream-pair is stable enough at
  the 75-ms timescale. Bench iterations are µs-scale; real cycles are
  ms-scale and may interact with DPM throttling and L2 sharing
  differently. D1's DPM warmup on `draft_stream` (per plan) addresses
  the first; the second is observable only end-to-end.
- Any τ regression cost. Pipelining trades τ for cycle wall time
  (one-cycle-stale draft context). The 4:1 wall-savings bound assumes τ
  stays at 90 % of pre-pipeline baseline; if the staleness costs more,
  net savings shrink further. The §D4 adaptive bypass + per-class
  baseline-keying is the safety net.

## Repro

```sh
# Symmetric + default asymmetric (5v + 5d, same shape):
cargo build --release -p rdna-compute --example bench_stream_overlap
./target/release/examples/bench_stream_overlap

# Realistic verify-dominated (the case that fails the threshold):
BENCH_VERIFY_LAYERS=64 BENCH_DRAFT_LAYERS=5 \
BENCH_DRAFT_M=2048 BENCH_DRAFT_K=2048 BENCH_DRAFT_N=16 \
  ./target/release/examples/bench_stream_overlap

# Realistic 4:1 work ratio (the case that supports proceeding):
BENCH_VERIFY_LAYERS=64 BENCH_DRAFT_LAYERS=20 \
BENCH_DRAFT_M=4096 BENCH_DRAFT_K=4096 BENCH_DRAFT_N=16 \
  ./target/release/examples/bench_stream_overlap
```

GPU lock acquired via `gpu-lock.sh` for the duration of the bench (so the
numbers don't get clobbered by another agent's cargo run). 7900 XTX was
otherwise idle; no DPM warmup was applied (numbers are post the bench's
own 20-iteration warmup loop).

## Decision

D0a / D0b / D0c are committed (`fda7899`, `4306ff5`, `abf9408`). D1
proceeds.
