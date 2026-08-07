# 2026-08-07 — Batching-ceiling probe (Task 0 / SP1 Task 1)

Empirical check of the spec §8 bandwidth-roofline estimates (~3.3× aggregate
speedup for the 27B dense model, ~1.8× for the 35B-A3B MoE) before any batched
kernel code is written. Method, tool, and results below; verdict at the end.

## What this measures — and what it does not

`probe_batching_ceiling.rs` calls `attention_flash_q8_0_batched_masked`
`layers` times at batch 1, across four context lengths, and fits
`t(ctx) = a + b·ctx` by least squares. This is **the attention/KV term only**.
It is not a decode step of a real model: no DeltaNet layers, no dense
projections, no MoE router/expert gather, no embedding lookup. SP1 does not
run a full model forward, so the "weights" side of the roofline argument is
**inferred from the spec's byte accounting, not measured here**. Anything
below labelled a "predicted speedup" is an extrapolation of the attention
kernel's own timing model, not an end-to-end number — see the verdict section
for why that distinction is load-bearing to the conclusion.

## Method

- Whole-step wall time only: one `Instant::now()` before `layers` un-synced
  kernel launches, one `device_synchronize()` after, per timed iteration.
  **No per-operation `device_synchronize`** — that would fabricate GPU
  speedups and corrupt the slope fit (see brief and inline kernel comments).
- 3 warmup iterations (untimed, synced once after), 9 timed iterations,
  median reported (not mean) to resist outliers.
- K/V cache filled with a fixed synthetic Q8_0 pattern (scale=1.0, small code
  ramp) — not numerically meaningful, this is a timing probe, not a
  correctness test.
- Context lengths: 4096, 16384, 32768, 65536 (`CTXS` default).

## Commands run

```bash
# 35B-A3B shape (defaults: NH=16 NKV=2 HD=256 LAYERS=10)
cargo build --release -p rdna-compute --features deltanet --example probe_batching_ceiling
cargo run   --release -p rdna-compute --features deltanet --example probe_batching_ceiling

# 27B dense shape
NH=24 NKV=4 HD=256 LAYERS=16 \
  cargo run --release -p rdna-compute --features deltanet --example probe_batching_ceiling
```

Hardware: gfx1151 (Strix Halo, Radeon 8060S iGPU, dev box). The deployment
target is gfx1201 (R9700), which we do not have — absolute numbers below are
dev-hardware only; only the *shape* of the fit (sign and relative magnitude of
`b`) is expected to transfer.

Each shape was run twice to check stability before writing this note.

## Results — 35B-A3B shape (NH=16 NKV=2 HD=256 LAYERS=10)

| ctx | median_ms (run 1) | median_ms (run 2) |
|---|---|---|
| 4,096 | 0.914 | 0.924 |
| 16,384 | 2.888 | 2.969 |
| 32,768 | 6.797 | 6.834 |
| 65,536 | 14.146 | 14.344 |

Fit (run 1, used as canonical below): `t(ctx) = -0.3218 ms + 0.219151 ms/1K ctx`
Fit (run 2, reproducibility check): `t(ctx) = -0.3223 ms + 0.221924 ms/1K ctx`

- `a` (context-independent term in this probe): **-0.32 ms** — see "Reading
  the negative `a`" below; this is not the model's real weights term.
- `b` (KV/attention term, does not amortise across slots): **~0.220 ms per 1K
  ctx**, stable across reruns to <2%.

## Results — 27B dense shape (NH=24 NKV=4 HD=256 LAYERS=16)

| ctx | median_ms (run 1) | median_ms (run 2) |
|---|---|---|
| 4,096 | 1.661 | 1.708 |
| 16,384 | 7.774 | 7.719 |
| 32,768 | 14.897 | 15.193 |
| 65,536 | 31.222 | 32.247 |

Fit (run 1, canonical): `t(ctx) = -0.3379 ms + 0.479062 ms/1K ctx`
Fit (run 2, reproducibility check): `t(ctx) = -0.5289 ms + 0.496553 ms/1K ctx`

- `a`: **-0.34 ms** (run 1) / -0.53 ms (run 2) — same near-zero-negative
  pattern as the 35B shape, larger absolute jitter because this shape's `layers`
  (16) and per-layer bytes are both bigger, so the fixed 4-point fit has more
  leverage from the same relative timer noise.
- `b`: **~0.479-0.497 ms per 1K ctx**, i.e. **~2.2× steeper** than the 35B
  shape's `b`. Spec §8 gives the 27B moving 32 KB/token against the 35B's
  10 KB/token — a 3.2× byte ratio. The measured slope ratio (~2.2×) is in the
  same direction (noticeably steeper, as the brief predicted) but shallower
  than the raw byte ratio, most likely because the 27B shape's larger
  `n_heads`/`n_kv_heads` (24/4 vs 16/2) also means more parallel work per tile,
  so the kernel is not purely bandwidth-bound at these shapes on gfx1151.

Both shapes: `b` positive, growing with context, reproducible across reruns.
This is the pass condition in the brief — a near-zero or negative `b` would
have meant the kernel wasn't reading the full context; that did not happen
here.

## Reading the negative `a`

The brief's stated failure condition is `b` near zero or negative; that did
not occur, so the probe is not BLOCKED. But `a` came out small and negative in
every run (both shapes, both reruns), which is worth explaining rather than
silently reporting.

This probe runs **only** the attention kernel — no weights are loaded, no
dense/DeltaNet/MoE compute happens. So the "context-independent" intercept
here is not a stand-in for the model's real weights term; it is just whatever
fixed per-call overhead (kernel dispatch, argument marshalling) the attention
launch itself carries, extrapolated back to `ctx=0` by a 4-point least-squares
fit. That overhead is small (sub-millisecond) and, with only 4 unevenly-scaled
sample points and a relation that is not perfectly affine (the 4096-ctx point
runs slightly more expensive per-token than the trend, likely fixed
tile/launch overhead dominating at small ctx), the fit can — and reproducibly
does — land marginally on the negative side of zero. This is fit noise around
a true value near zero, not evidence of negative-time execution.

The practical consequence: **this probe's own N=2/4/8 "speedup" formula
(below) is not a valid proxy for end-to-end batching benefit**, because the
whole benefit in that formula comes from amortising `a` across slots, and this
`a` is not the real weights term (see "Verdict" below).

## Predicted speedups from this probe's own fit (attention-term-only — NOT end-to-end)

Using `seq = N·(a + b·ctx)`, `batched = a + N·b·ctx`, `speedup = seq/batched`,
with each shape's own canonical (run 1) fit:

### 35B-A3B (a=-0.3218, b=0.000219151 ms/ctx)

| N | ctx | seq (ms) | batched (ms) | speedup |
|---|---|---|---|---|
| 2 | 4,096 | 1.152 | 1.473 | 0.78× |
| 2 | 16,384 | 6.537 | 6.859 | 0.95× |
| 2 | 32,768 | 13.719 | 14.040 | 0.98× |
| 2 | 65,536 | 28.081 | 28.403 | 0.99× |
| 4 | 4,096 | 2.303 | 3.269 | 0.70× |
| 4 | 16,384 | 13.075 | 14.040 | 0.93× |
| 4 | 32,768 | 27.437 | 28.403 | 0.97× |
| 4 | 65,536 | 56.162 | 57.127 | 0.98× |
| 8 | 4,096 | 4.606 | 6.859 | 0.67× |
| 8 | 16,384 | 26.150 | 28.403 | 0.92× |
| 8 | 32,768 | 54.874 | 57.127 | 0.96× |
| 8 | 65,536 | 112.323 | 114.576 | 0.98× |

### 27B dense (a=-0.3379, b=0.000479062 ms/ctx)

| N | ctx | seq (ms) | batched (ms) | speedup |
|---|---|---|---|---|
| 2 | 4,096 | 3.249 | 3.587 | 0.91× |
| 2 | 16,384 | 15.022 | 15.360 | 0.98× |
| 2 | 32,768 | 30.720 | 31.058 | 0.99× |
| 2 | 65,536 | 62.116 | 62.454 | 0.99× |
| 4 | 4,096 | 6.497 | 7.511 | 0.87× |
| 4 | 16,384 | 30.044 | 31.058 | 0.97× |
| 4 | 32,768 | 61.440 | 62.454 | 0.98× |
| 4 | 65,536 | 124.232 | 125.245 | 0.99× |
| 8 | 4,096 | 12.995 | 15.360 | 0.85× |
| 8 | 16,384 | 60.088 | 62.454 | 0.96× |
| 8 | 32,768 | 122.880 | 125.245 | 0.98× |
| 8 | 65,536 | 248.463 | 250.829 | 0.99× |

Every one of these is **below 1.0×** — the formula says batching this
attention-only proxy is never a win. Do not read that as "batching hurts";
read it as the formula having nothing to amortise. `a·(1−N)` is the entire
numerator of the gap between `seq` and `batched`, and `a` here is ≈0 (slightly
negative). With no real weights term in the model, there is no amortisable
cost for batching to recover — this table demonstrates the mechanism (KV never
amortises) far more than it demonstrates a number.

## Verdict: does spec §8's ~3.3× (27B) / ~1.8× (35B) survive?

**Not confirmed, not refuted — correctly out of reach of this probe, as
designed.** Spec §8's aggregate speedup comes overwhelmingly from amortising
the *weights* term (~15 GB/step for the 27B; ~1.2 GB dense + ~2.2 GB experts
for the 35B) across N slots, with the KV term as the drag that erodes it at
long context. This probe deliberately does not load any weights (SP1 does not
run a full model forward — that is explicitly future work), so it cannot
measure the numerator that drives those ratios. Nothing here contradicts
§8's ~3.3×/~1.8×.

What this probe **does** establish, which §8 could only assume before now:

1. **The structural premise holds.** `b > 0` and growing with `ctx` in both
   shapes, reproducibly. The KV/attention term is real, it scales with
   context as expected, and the kernel is demonstrably reading the full
   context (the failure mode the brief warned about — `b` near zero or
   negative — did not occur).
2. **The 27B really does move more KV traffic per token than the 35B-A3B**,
   directionally consistent with spec's 32 KB vs 10 KB claim: measured `b` is
   ~2.2× steeper for the 27B shape, versus a 3.2× byte-ratio prediction. The
   shortfall from 3.2× to 2.2× is itself useful: it says the naive
   bytes-per-token arithmetic in §8 mildly *overstates* how much worse the
   27B's KV term is relative to the 35B's, at least on gfx1151 at these head
   counts — plausibly because the 27B's larger head count (24 vs 16) gives the
   kernel more parallel work per KV tile, so it is not purely bandwidth-bound
   here. That is a data point for tuning (§10's `TILE_SIZE`/BR/BC sweep), not
   a refutation of the aggregate ratio.
3. **The real weights term (`a`) still needs to be measured or at minimum
   estimated from a real forward pass**, not just inferred from bandwidth
   arithmetic, before the aggregate 3.3×/1.8× figures can be trusted. This
   probe's near-zero fitted `a` should not be confused with that term — it is
   attention-launch overhead, not model weights.

**Recommendation:** the 8 downstream SP1 tasks are not invalidated by
anything found here — the KV-term half of the roofline argument survives
contact with measurement. But before treating 3.3×/1.8× as load-bearing
product numbers (e.g. for the "3-4 agents on one R9700" framing), a follow-up
probe that actually loads real weights and runs a genuine decode step (even
single-layer-representative, not full model) is needed to pin down `a`. That
is a natural Task 0.5, not blocking for SP1's kernel work, which only needs
the KV term's structural behaviour (confirmed here) to proceed.

## Reproducibility

Both shapes were run twice; `b` was stable to <2% across reruns for the 35B
shape and ~4% for the 27B shape (larger jitter attributable to more `layers`
× larger per-layer bytes amplifying the same relative timer noise across a
4-point fit). `a` stayed small and negative in all four runs. See the raw
tables above for both runs' `median_ms` values.
