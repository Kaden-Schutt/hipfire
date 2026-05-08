# Dev log 2026-05-08 — MQ3-Lloyd WMMA prefill, Phase C

**Branch:** `feat/mq3-lloyd-wmma-prefill` (HEAD `053f78d`)
**Plan:** `docs/plans/mq3-lloyd-wmma-prefill.md` (rev 2, commit a654099)
**Hardware:** gfx1100 (7900 XTX), ROCm 7.2.

## Summary

Phase C perf validation done. Cross-process A/B comparison of the
batched-prefill path with MQ3-Lloyd vs the structural ceiling (uniform
MQ3) on Qwen3.5-9B: **Lloyd reaches 88.2% of uniform-MQ3 prefill
throughput**. Per the Phase C ship-gate decision rule (≥ 60% → ship),
this clears comfortably and is also above Gemini's 80% review estimate.

Decode is unchanged at the canonical probe-commits shape (verified at
B2 amend: master 122.2 vs branch 122.3); at the longer prefill-256
shape used here, Lloyd gen 114.3 vs uniform 121.5 (94%) reflects the
extra per-token codebook indexing the Lloyd decode path must do —
consistent with prior B2 numbers and not a Lloyd-specific regression.

**Decision: SHIP.**

## Bench config

| Field | Value |
|---|---|
| Tool | `target/release/examples/bench_qwen35_mq4` |
| Models | `qwen3.5-9b.mq3` (uniform), `qwen3.5-9b.mq3-lloyd` |
| Flags | `--prefill 256 --warmup 5 --prefill-runs 3 --gen 30` |
| Env | `HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1` |
| Cross-process A/B | 3 fresh process invocations × 2 models |
| In-process samples | 3 timed prefills per invocation; in-process median |
| Reported metric | mean of in-process medians across the 3 invocations |

`--prefill 256` rather than the probe-commits canonical `--prefill 16`
because 16 is too small to meaningfully exercise the batched-prefill
path's fused kernels (qkvza / qkv / gate_up / residual) — at 16 tokens
the LA preamble and FFN are dispatch-overhead bound, not GEMM-bound.
At 256 tokens the WMMA kernels are doing real work, which is what we
want to measure for ship-gate purposes.

`--prefill-runs 3` so the in-process median excludes JIT compile cost
on the first run. Cross-process runs further isolate from
session-state effects (DPM, thermal, fragmented HSA queues).

## Raw numbers (gfx1100, ROCm 7.2, branch HEAD 053f78d)

```
=== qwen3.5-9b.mq3 (uniform) ===
  run 1: prefill_median=1745.8 tok/s, gen=121.6 tok/s
  run 2: prefill_median=1673.2 tok/s, gen=121.3 tok/s
  run 3: prefill_median=1739.8 tok/s, gen=121.7 tok/s
  → prefill mean = 1719.6 tok/s   gen mean = 121.5 tok/s

=== qwen3.5-9b.mq3-lloyd ===
  run 1: prefill_median=1527.6 tok/s, gen=114.5 tok/s
  run 2: prefill_median=1514.1 tok/s, gen=114.3 tok/s
  run 3: prefill_median=1508.2 tok/s, gen=114.0 tok/s
  → prefill mean = 1516.6 tok/s   gen mean = 114.3 tok/s
```

## Ratios + ship-gate decision

| Comparison | Numerator | Denominator | Ratio | Verdict |
|---|---|---|---|---|
| Lloyd prefill / uniform-MQ3 prefill | 1516.6 | 1719.6 | **88.2 %** | ≥ 60 % ship gate **PASS** |
| Lloyd prefill / pre-B per-token fallback | 1516.6 | ~108 | **14.0 ×** | plan-asserted "even at 60 %, ~3× better" comfortably exceeded |
| Lloyd decode / uniform-MQ3 decode (prefill-256 shape) | 114.3 | 121.5 | 94.1 % | shape-difference effect, not regression (decode at probe-commits shape unchanged: 122.3 vs 122.2 master, see B2 amend) |

Per the Phase C decision tree in the plan:

```
Lloyd-MQ3 prefill / MQ3 non-Lloyd prefill = 88.2 %
  → ≥ 60 %  →  Ship.
```

## Stddev observation (informational, not gating)

Run-to-run variance is **tighter** for Lloyd than uniform:

- Lloyd prefill range: 1508.2 – 1527.6 tok/s (Δ = 19.4 tok/s, ~1.3 %)
- Uniform prefill range: 1673.2 – 1745.8 tok/s (Δ = 72.6 tok/s, ~4.3 %)

This is the opposite of what we'd expect if Lloyd had occupancy /
LDS-bank-conflict instability under thermal flux, and is consistent
with the cooperative-load + per-row-codebook layout being more
deterministic than uniform-MQ3's K-tile schedule (where the K-stride
through 3-bit indices interacts with DPM steps differently across
runs). Worth noting but not actionable.

## What this validates

- The Phase B1 fused kernel family (qkv, qkvza, gate_up, residual,
  plus their gfx12 siblings) and the Phase B2 dispatch wiring produce
  an end-to-end batched-prefill path whose throughput is competitive
  with the uniform-MQ3 ceiling, not a fraction of it.
- The 7.15 % fp16-LDS-vs-fp32-LDS margin established in Phase A
  carries through to the full inference path — i.e. the Phase A
  bench was a true predictor, not a microbench artifact.
- The 14× speedup over the per-token `forward_scratch` fallback (108
  tok/s, per PR #181 future-work section) means Phase 5 closes the
  gap that issue #116 was opened to address. Whatever fraction of the
  uniform ceiling we reach, the absolute throughput delta vs the
  fallback is the user-facing improvement.

## What this does NOT validate

- **gfx12 (RDNA4) sibling kernels.** No RDNA4 hardware on the bench
  host. The gfx12 selector arms in `crates/rdna-compute/src/kernels.rs`
  and the kernel sources `*.gfx12.hip` are code-complete-but-runtime-
  unvalidated as flagged in B1. Community CI on RDNA4 hardware needed
  before we can claim coverage.
- **gfx1151.** Smoke-tested at the Phase B2 stage but not in this Phase
  C bench round. The kernel arch matrix dispatches gfx1151 onto the
  RDNA3 (`_rdna3` suffix) kernels, same as gfx1100, so behaviour
  should match within hardware-config noise.
- **PFlash / spec-decode interaction.** Phase C measured non-spec
  prefill only. The DDTree / PFlash paths route through different
  matchers and were not exercised here. This is the obvious next
  area to bench once #116 lands.

## Watch-items carried forward

From Gemini's plan review — recorded here so they don't get lost
when the branch lands and the plan doc is archived:

1. **gfx12 lane-group K-split (`tid >> 4`) is cleaner than gfx11's
   full-tile-per-lane mapping.** If gfx11 underperforms relative to
   gfx12 once both are bench-able, the gfx11 kernel is the candidate
   root cause to investigate. Not actionable now (no RDNA4 hardware)
   but record-the-suspicion-now is cheaper than rederiving it later.

2. **Decode shape sensitivity.** The 94 % Lloyd-vs-uniform decode
   ratio at prefill-256 is not present at prefill-16 (where decode is
   122 / 122). The difference is per-token codebook indexing in the
   Lloyd decode kernel becoming visible only when KV-cache traffic
   per token is large. Not a Phase 5 problem (decode kernel was
   shipped in PR #181, not modified here) but worth flagging for the
   issue #182 MQ4-Lloyd follow-up where the same per-row-codebook
   pattern is being considered.

## Files

- `experiments` directory does not exist on this branch — Phase C
  result lives under `benchmarks/results/` alongside Phases A and B
  devlogs (matching the existing convention).

## Next steps (post-Phase-C, post-merge)

- Push the branch, open a PR for issue #116 Phase 5 covering Phase A,
  B1, B2, B3, C in one chunk.
- Issue #182 (MQ4-Lloyd WMMA prefill) inherits the per-row-codebook
  pattern validated here. Different K2-vs-K4 LDS layout, 16-entry
  codebook → re-derive Phase A occupancy budget; the rest of the
  framework (matcher arms, dispatch arms, coherence-gate row, Phase
  C bench config) translates 1:1.
- Community CI on RDNA4 hardware to validate the gfx12 sibling
  kernels — track as a separate issue, not blocking this PR.
