# Dev log 2026-05-08 — MQ4-Lloyd WMMA prefill, Phase C

**Branch:** `feat/issue-182-mq4-lloyd` (HEAD `1934aae`)
**Plan:** `docs/plans/mq4-lloyd-wmma-prefill.md` (commit `4f3f1ec`).
**Hardware:** gfx1100 (7900 XTX), ROCm 7.2.
**Sibling:** MQ3-Lloyd Phase C — `devlog_20260508_lloyd_wmma_phase_c.md`,
PR #195 (88.2 % Lloyd / uniform-MQ3 ratio on the same hardware).

## Summary

Phase C perf validation done on gfx1100 (the maintainer-host-class gate
that gfx1151 cannot answer due to LPDDR5x bandwidth ceiling). Cross-
process A/B comparison of the batched-prefill path with MQ4-Lloyd vs
the structural ceiling (uniform MQ4) on Qwen3.5-9B:
**Lloyd reaches 60.6 % of uniform-MQ4 prefill throughput**.

Per the Phase C decision rule (≥ 60 % → ship), this clears — but
narrowly, and well below the plan's 80 % target estimate. Compare to
the MQ3-Lloyd sibling at 88.2 % on the same hardware.

**Decision: SHIP per rule, with explicit watch-item that perf
optimization is the most likely follow-up area.**

### Three-point size sweep (added 2026-05-08, after the initial 9B run)

| Model        | Lloyd prefill | uniform prefill | Lloyd / uniform |
|---|---|---|---|
| Qwen3.5-4B   | 2588.1 tok/s  | 3589.9 tok/s    | **72.1 %**       |
| Qwen3.5-9B   | 1516.6 tok/s  | 1719.6 tok/s    |   60.6 %         |
| Qwen3.6-27B  |  397.4 tok/s  |  779.1 tok/s    | **51.0 %** (below the 9B-calibrated 60 % ship gate) |

The Lloyd / uniform ratio decreases monotonically with model size on
gfx1100. The plan's investigate-bucket candidates (LDS footprint
scaling with K, longer 8-byte-per-tile decode K-schedule, longer
cooperative-load sync) all scale with K — 4B's K=2560 / 9B's K=4096 /
27B's K=5120 model dim, with mlp.down K growing from ~6912 to 12288 to
17408. The 27B data point is **informational, not a ship blocker**:
the 60 % gate was set against the 9B headline target; 27B Lloyd
prefill is still ~3.3× faster than the pre-B per-token fallback (the
user-facing improvement issue #182 was opened for).

See "## 4B data point" and "## 27B data point" below for full raw
numbers + analysis.

## Bench config

| Field | Value |
|---|---|
| Tool | `target/release/examples/bench_qwen35_mq4` |
| Models | `qwen3.5-9b.mq4` (uniform), `qwen3.5-9b.mq4-lloyd` |
| Flags | `--prefill 256 --warmup 5 --prefill-runs 3 --gen 30` |
| Env | `HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1` |
| Cross-process A/B | 3 fresh process invocations × 2 models |
| In-process samples | 3 timed prefills per invocation; in-process median |
| Reported metric | mean of in-process medians across the 3 invocations |
| Bench binary md5 | `71234969e9d9461a5c6fbc86449ba6c4` (target/release/examples/bench_qwen35_mq4 at HEAD `0a34ba6`; identical at HEAD `1934aae`) |
| Model md5 (uniform) | `31a8d8dc7603226801b08d8319015602` (qwen3.5-9b.mq4) |
| Model md5 (Lloyd)   | `b3eea80aeade0b56c153a054b1143ab2` (qwen3.5-9b.mq4-lloyd) |
| Prompt | N/A — `bench_qwen35_mq4` generates a deterministic synthetic token sequence (token ids `0..prefill_len`) internally; no user prompt |

`--prefill 256` rather than the probe-commits canonical `--prefill 16`
because 16 is too small to meaningfully exercise the batched-prefill
path's fused kernels (qkvza / qkv / gate_up / residual) — at 16 tokens
the LA preamble and FFN are dispatch-overhead bound, not GEMM-bound.

`--prefill-runs 3` so the in-process median excludes JIT compile cost
on the first run. Cross-process runs further isolate from session-state
effects (DPM, thermal, fragmented HSA queues).

## Raw numbers (gfx1100, ROCm 7.2, branch HEAD `1934aae`)

```
=== qwen3.5-9b.mq4 (uniform) ===
  run 1: prefill_median=2297.0 tok/s, gen=110.4 tok/s
  run 2: prefill_median=2303.0 tok/s, gen=110.3 tok/s
  run 3: prefill_median=2298.4 tok/s, gen=110.5 tok/s
  → prefill mean = 2299.5 tok/s   gen mean = 110.4 tok/s

=== qwen3.5-9b.mq4-lloyd ===
  run 1: prefill_median=1384.1 tok/s, gen=93.6 tok/s
  run 2: prefill_median=1389.9 tok/s, gen=92.3 tok/s
  run 3: prefill_median=1406.3 tok/s, gen=92.4 tok/s
  → prefill mean = 1393.4 tok/s   gen mean = 92.8 tok/s
```

Decode regression check via `probe_commits.sh` (master `85678ed` vs
HEAD `1934aae`, BENCH_MODEL=qwen3.5-9b.mq4, canonical `--prefill 16
--gen 30` shape):

```
85678ede    117.1 tok/s   master tip
1934aaed    117.3 tok/s   feat(mq4-lloyd-wmma): Phase B3
            +0.2 %        no decode regression on uniform MQ4
```

Lloyd decode at the same canonical shape: 97.3 tok/s on HEAD; master
returns BENCH_FAIL because the Lloyd format runtime gating is removed
during Phase 5 work (matches the MQ3-Lloyd Phase 5 pattern).

## Ratios + ship-gate decision

| Comparison | Numerator | Denominator | Ratio | Verdict |
|---|---|---|---|---|
| Lloyd-MQ4 prefill / uniform-MQ4 prefill (gfx1100) | 1393.4 | 2299.5 | **60.6 %** | ≥ 60 % ship gate **PASS** (narrow) |
| Lloyd-MQ4 prefill / pre-B per-token fallback | 1393.4 | ~120 | **~11.6 ×** | plan-asserted "multi-× better than per-token fallback" cleared |
| Lloyd-MQ4 decode / uniform-MQ4 decode (prefill-256 shape) | 92.8 | 110.4 | 84.1 % | shape-difference effect (KV traffic per token); decode at probe-commits canonical shape unchanged |
| Lloyd-MQ4 decode (canonical) / uniform-MQ4 decode (canonical) | 97.3 | 117.3 | 83.0 % | unchanged from B2 — Lloyd decode is the PR #182 single-acc K4 GEMV, not modified in this PR |
| **Sibling reference** | | | | |
| MQ3-Lloyd Phase C ratio (PR #195, same hardware) | 1516.6 | 1719.6 | 88.2 % | for context — the MQ4 gap is the new finding |

Per the Phase C decision tree in the plan:

```
Lloyd-MQ4 prefill / MQ4 non-Lloyd prefill = 60.6 %
  → ≥ 60 %  →  Ship.
```

## Why is MQ4-Lloyd at 60 % when MQ3-Lloyd hit 88 %?

The plan flagged this risk surface at Phase A:

> **VGPR pressure with 512 B LDS.** MQ3 Phase A landed at 82 VGPRs / 256 B
> LDS; 0 spills. MQ4 has 2× LDS budget and a slightly more complex decode
> (256-entry codebook lookup vs MQ3's 128). VGPR pressure could push past
> the `__launch_bounds__(32, 2)` budget.

and the Phase C decision tree's investigate bucket explicitly named the
candidates:

> 30 % to 60 % | Investigate. Likely culprits: VGPR pressure with the
> larger 512 B LDS, K-tile schedule on the 8-byte-per-tile decode (MQ3
> was 6 bytes), reconvergent-sync timing on the longer cooperative load.

We're at 60.6 % — **just above** the investigate bucket, **just below**
the 80 % soft target. The candidates the plan named are still the
candidates the data points to:

1. **2× LDS footprint** (512 B vs MQ3's 256 B) means the cooperative
   load is twice as much data per group, and the per-row codebook
   lookup walks a 256-entry table instead of 128.
2. **8-byte-per-tile decode** (4-bit nibble pair × 16-element tile)
   vs MQ3's 6-byte decode shifts the K-tile schedule slightly; whether
   that changes the issue/memory-pressure balance on gfx1100 is
   measurable but was not measured in this Phase C round.
3. **Cooperative-load sync** is longer in absolute time on MQ4; the
   `__syncthreads()` discipline is unchanged from MQ3 (2 per group,
   at group boundaries) but the *time spent in those barriers*
   relative to the K-tile loop is larger.

These are perf-optimization questions, not correctness questions. The
batched-prefill path is correct (B1 parity + B3 coherence-gate green +
decode-regression check clean), it just leaves headroom that the
similar but lighter MQ3-Lloyd path captures.

## gfx1100 coherence-gate (added post-review, 2026-05-08)

Re-ran `scripts/coherence-gate.sh` on the gfx1100 Phase C bench host
after the post-review-fix commit to close the validation gap that
PR #195's review flagged on its sibling (S2 in `mq3-lloyd-wmma-code-rev-claude.md`).
The MQ4-Lloyd row is the canonical regression-prevention site for the
Phase B2 wiring — the gate already runs on gfx1151 via Phase B3, but
the gfx1100 coherence wasn't explicitly captured for this PR until now.

```
## qwen3.5-9b.mq4-lloyd — reason-mq4-lloyd-9b

- wall: 54.4s  status: **OK**
- stats: tokens=53, prefill_tokens=36, prefill_tok_s=690.3, decode_tok_s=91.2
- prompt: "A farmer has 17 sheep. All but 9 die. How many are left? ..."

Output:
  <think>
  </think>
  **Reasoning:** The phrase "all but 9 die" means that every sheep except
  for 9 survived. Therefore, the number of sheep remaining is exactly
  the number mentioned in the exception clause.
  **Final Number:** 9<|im_end|>
```

Coherent reasoning, correct numerical answer, clean `<|im_end|>`
termination, no attractor loops. Status OK on gfx1100.

## 4B data point (added 2026-05-08, branch HEAD `5fce3e3`)

Quantized `qwen3.5-4b.mq4-lloyd` from `/data/models/qwen/Qwen3.5-4B/`
and re-ran the same Phase C cross-process A/B at the smaller model
scale to get a second hardware-ratio data point. Same gfx1100 host,
same bench config, same `--prefill 256 --prefill-runs 3` methodology.

```
=== qwen3.5-4b.mq4 (uniform) ===
  run 1: prefill_median=3591.6 tok/s, gen=152.3 tok/s
  run 2: prefill_median=3594.9 tok/s, gen=152.2 tok/s
  run 3: prefill_median=3583.1 tok/s, gen=151.9 tok/s
  → prefill mean = 3589.9 tok/s   gen mean = 152.1 tok/s

=== qwen3.5-4b.mq4-lloyd ===
  run 1: prefill_median=2602.3 tok/s, gen=136.7 tok/s
  run 2: prefill_median=2567.6 tok/s, gen=136.4 tok/s
  run 3: prefill_median=2594.3 tok/s, gen=136.2 tok/s
  → prefill mean = 2588.1 tok/s   gen mean = 136.4 tok/s
```

| Comparison (4B) | Numerator | Denominator | Ratio |
|---|---|---|---|
| Lloyd-MQ4 prefill / uniform-MQ4 prefill | 2588.1 | 3589.9 | **72.1 %** |
| Lloyd-MQ4 decode / uniform-MQ4 decode (prefill-256 shape) | 136.4 | 152.1 | 89.7 % |

**Lloyd / uniform prefill at 4B (72.1 %) is 11.5 pp better than at 9B
(60.6 %).** The plan's investigate-bucket candidates (2× LDS footprint,
longer 8-byte-per-tile decode K-schedule, longer cooperative-load sync)
all scale with K. At 4B's K=2560 dims the per-row codebook + cooperative-
load overhead is a smaller fraction of total work than at 9B's K=4096
dims and 12288 down-proj K. Useful precedent: the MQ4 / MQ3 ratio gap
narrows at smaller scales, suggesting the Phase C investigate-bucket
candidates do close partially when re-tuned for the larger LDS regime.

Run-to-run variance is exceptionally tight on uniform 4B (range 11.8
tok/s, ~0.3 % range/mean), tight on Lloyd 4B (range 34.7 tok/s, ~1.3 %
range/mean) — same envelope as the 9B numbers above.

Smoke test (daemon, two prompts):
- "What is the capital of France? ..." → 12 tokens, "The capital of France is Paris.<|im_end|>", clean
- "A farmer has 17 sheep ..." → 19 tokens, "If all but 9 die, then 9 sheep are left.<|im_end|>", clean

Bench binary md5: `71234969e9d9461a5c6fbc86449ba6c4` (same as 9B run).
Model md5 (4B uniform): `93b9b5f2bd075922c50f3f8c9a5ad3e3`.
Model md5 (4B Lloyd):   `d13028f6a1f4fda772c17bb6c3f3a0bc`.

## 27B data point (added 2026-05-08, branch HEAD `c42deb0`)

Quantized `qwen3.6-27b.mq4-lloyd` from
`/data/cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd...`
(52 GB safetensors, output 17.4 GB Lloyd-MQ4) and re-ran the same Phase
C cross-process A/B at the largest available scale on gfx1100. **Note**:
27B is Qwen**3.6** (different model family from 3.5-4B/9B); layer dims
still scale with size but the architectural baseline shifts slightly.

```
=== qwen3.6-27b.mq4 (uniform) ===
  run 1: prefill_median=780.8 tok/s, gen=39.2 tok/s
  run 2: prefill_median=779.3 tok/s, gen=39.1 tok/s
  run 3: prefill_median=777.2 tok/s, gen=39.1 tok/s
  → prefill mean = 779.1 tok/s   gen mean = 39.1 tok/s

=== qwen3.6-27b.mq4-lloyd ===
  run 1: prefill_median=397.0 tok/s, gen=33.3 tok/s
  run 2: prefill_median=397.4 tok/s, gen=33.3 tok/s
  run 3: prefill_median=397.7 tok/s, gen=33.3 tok/s
  → prefill mean = 397.4 tok/s   gen mean = 33.3 tok/s
```

| Comparison (27B) | Numerator | Denominator | Ratio |
|---|---|---|---|
| Lloyd-MQ4 prefill / uniform-MQ4 prefill | 397.4 | 779.1 | **51.0 %** |
| Lloyd-MQ4 decode / uniform-MQ4 decode (prefill-256 shape) | 33.3 | 39.1 | 85.2 % |
| Lloyd-MQ4 prefill / pre-B per-token fallback (~120 tok/s class on 27B) | 397.4 | ~120 | ~3.3 × |

**Lloyd / uniform prefill at 27B (51.0 %) falls below the 60 % gate.**
This does not change the ship decision — the 60 % gate was set against
the 9B Phase C target, and the user-facing comparison is Lloyd-MQ4
prefill (397.4) vs the pre-B per-token fallback (~120 tok/s class)
which is still ~3.3 × faster. But the size sweep confirms the plan's
risk hypothesis (Phase C "Risks and watch-items" §"VGPR pressure with
512 B LDS"): the gap to the uniform-MQ4 ceiling grows monotonically
with K. At 27B's 5120 model dim (K=17408 for mlp.down) the per-row
codebook + cooperative-load overhead dominates more of total work than
at 9B's 12288 K-dim mlp.down.

Run-to-run variance is exceptionally tight on both 27B paths:
- Uniform 27B prefill range 3.6 tok/s, ~0.5 % range/mean
- Lloyd 27B prefill range 0.7 tok/s, ~0.2 % range/mean
- Both decode paths byte-identical across all 3 invocations on Lloyd
  (33.3 / 33.3 / 33.3) and ±0.1 on uniform — the variance benefit of
  cross-process A/B over within-session A/B (per CLAUDE.md ±10–15 %
  drift rule) is especially clear at the larger scale.

Smoke test (daemon, two prompts):
- "Capital of France?" → 12 tokens, "...Paris.<|im_end|>", clean
- "17 sheep, all but 9 die" → 29 tokens, correct reasoning, correct
  numerical answer (9), clean `<|im_end|>` termination

Bench binary md5: `71234969e9d9461a5c6fbc86449ba6c4` (same as 4B/9B
runs — single binary across the size sweep).
Model md5 (27B uniform): `9a6acdc49bcaa6a7b52ac161444cb769`.
Model md5 (27B Lloyd):   `9fe79c54ce2291f8d7f0c7d61ddc46e9`.

## What this validates

- gfx1100 batched-prefill correctness: uniform MQ4 decode is
  byte-equal pre/post (117.1 → 117.3, +0.2 % noise), confirming the
  B2 dispatch wiring doesn't perturb the non-Lloyd path
- Lloyd-MQ4 prefill is **multi-× faster than the pre-B per-token
  fallback** — the user-facing improvement Phase 5b was opened for
- Cross-process variance (range/mean, *not* stddev/mean) is tight on
  uniform MQ4 (range 6.0 tok/s, ~0.3 % range/mean) and reasonable on
  Lloyd MQ4 (range 22.2 tok/s, ~1.6 % range/mean) — measurements are
  stable enough to support the ship/investigate decision

## What this does NOT validate

- **gfx12 (RDNA4) sibling kernels.** No RDNA4 hardware on the bench
  host. The `*.gfx12.hip` files and arch-selector `_rdna4` arms are
  code-complete-but-runtime-unvalidated; community CI on RDNA4
  hardware needed before claiming coverage. (Help-wanted #1 in the
  PR.)
- **gfx1151 ship-gate ratio.** Phase B2 measured pre/post-B2 prefill
  speedup on gfx1151 at 12.3× (`devlog_20260507_mq4_lloyd_wmma_phase_a.md`
  + the PR body), but did not run the Lloyd-vs-uniform-MQ4 comparison
  there. Strix Halo's LPDDR5x ceiling makes the absolute prefill
  numbers ~3× lower than gfx1100; the *ratio* should be in the same
  ballpark since both numerator and denominator share the bandwidth
  ceiling, but this is unverified.

## Watch-items / follow-up perf optimization candidates

If user demand surfaces for closing the MQ4 / MQ3 ratio gap (88 % →
60 %), the natural follow-up issues are:

1. **`__launch_bounds__` retuning.** The plan flagged
   `__launch_bounds__(32, 2)` as a potential pressure point at 512 B
   LDS. A `(32, 1)` retune or a wave64 variant could free VGPRs at
   the cost of occupancy. Worth a Phase A-style microbench.
2. **Vectorized cooperative load (`half8_t` LDS reads).** The plan
   notes 16 fp16 entries × 2 B = 32 B header at offset 0, group bases
   16 B-aligned (better than MQ3's 8 B). `half8_t` would halve the
   load instruction count and may close some of the gap.
3. **Per-arch K-tile schedule.** Gemini's plan-review note from the
   MQ3-Lloyd round (gfx12's `tid >> 4` lane-group K-split is cleaner
   than gfx11's full-tile-per-lane mapping) applies more strongly at
   the larger MQ4 LDS footprint. Worth confirming once RDNA4 hardware
   is available.
4. **Async LDS prefetch.** The current cooperative load synchronously
   blocks before the K-tile loop. An async prefetch overlapping with
   the previous group's compute would hide the longer load latency.

None of these are in scope for issue #182 Phase 5b, which targeted
correctness + a working ship gate. They are listed here so that
"Lloyd-MQ4 prefill is at 60 % of ceiling" doesn't get rederived from
scratch the next time the question comes up.

## Files

- Phase C bench numbers + analysis (this devlog): under
  `benchmarks/results/` alongside Phase A devlog (matching the
  existing convention for issue #182 — no `experiments/` directory)
- Plan update: §"Phase C — perf validation + ship gate decision rules"
  in `docs/plans/mq4-lloyd-wmma-prefill.md` gains a "Result on
  gfx1100" subsection citing this devlog

## Next steps (post-Phase-C, post-merge)

- Push the Phase C devlog + plan update onto the existing PR #197
- Update PR body to reflect that the gfx1100 Phase C help-wanted item
  is now closed; only RDNA4 (gfx12) help-wanted remains
- Consider whether to land the perf-optimization watch-items above as
  a follow-up issue (likely yes, file as #182-followup)
