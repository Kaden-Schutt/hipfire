# Dev log 2026-05-08 — MQ3-Lloyd WMMA prefill, Phase C

**Branch:** `feat/mq3-lloyd-wmma-prefill` (HEAD `053f78d` for the
gfx1100 round; gfx1151 follow-up round added 2026-05-08).
**Plan:** `docs/plans/mq3-lloyd-wmma-prefill.md` (rev 2, commit a654099)
**Hardware:** gfx1100 (7900 XTX, GDDR6), ROCm 7.2 — primary round.
gfx1151 (Strix Halo APU, LPDDR5x), ROCm 7.12 — follow-up round below.

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

## gfx1151 round (Strix Halo APU, RDNA3.5, ROCm 7.12, 2026-05-08)

Same bench config as the gfx1100 round above (`--prefill 256 --warmup 5
--prefill-runs 3 --gen 30`, asym3 KV, GRAPH=1, 3 fresh process
invocations × 2 models). The gfx1151 selector arm dispatches to the
same `_rdna3` kernel as gfx1100, so this round confirms behaviour
matches across the gfx11 family.

```
=== qwen3.5-9b.mq3 (uniform) ===
  inv 1: prefill_median=514.8 tok/s, gen=53.8 tok/s
  inv 2: prefill_median=520.2 tok/s, gen=53.8 tok/s
  inv 3: prefill_median=520.6 tok/s, gen=53.9 tok/s
  → prefill mean = 518.5 tok/s   gen mean = 53.8 tok/s

=== qwen3.5-9b.mq3-lloyd ===
  inv 1: prefill_median=495.3 tok/s, gen=49.1 tok/s
  inv 2: prefill_median=503.3 tok/s, gen=48.9 tok/s
  inv 3: prefill_median=505.5 tok/s, gen=49.1 tok/s
  → prefill mean = 501.4 tok/s   gen mean = 49.0 tok/s
```

Decode-regression check at the canonical probe-commits shape
(`--prefill 16 --warmup 3 --gen 30`, 3 fresh invocations × 2 models):

```
qwen3.5-9b.mq3      : gen 55.7 / 56.0 / 55.7 → mean 55.8 tok/s
qwen3.5-9b.mq3-lloyd: gen 50.5 / 50.6 / 50.7 → mean 50.6 tok/s
```

| Comparison | Numerator | Denominator | Ratio | Verdict |
|---|---|---|---|---|
| Lloyd prefill / uniform-MQ3 prefill | 501.4 | 518.5 | **96.7 %** | ≥ 60 % ship gate **PASS** |
| Lloyd decode / uniform-MQ3 decode (prefill-256 shape) | 49.0 | 53.8 | 91.1 % | constant Lloyd codebook-indexing overhead, not regression |
| Lloyd decode / uniform-MQ3 decode (probe-commits shape) | 50.6 | 55.8 | 90.7 % | same constant overhead — gfx1151 doesn't show the gfx1100 shape-dependent behaviour |

**Per the Phase C decision tree: gfx1151 ships too.**

Two notable deltas from the gfx1100 round:

1. **Prefill ratio is *higher* on gfx1151 (96.7 %) than gfx1100 (88.2 %).**
   Strix Halo's shared LPDDR5x (~250 GB/s peak, shared with CPU) makes
   the workload more memory-bound; the per-tile cooperative codebook
   load that costs Lloyd ~12 % vs uniform on gfx1100's GDDR6 narrows
   to ~3 % when the bottleneck shifts toward off-chip bandwidth.
   Useful precedent for future per-row-codebook formats: arch-class
   matters less when the host is bandwidth-bound.

2. **Decode ratio is *flat* across prefill shapes on gfx1151
   (91.1 % at prefill-256, 90.7 % at prefill-16).** gfx1100 saw the
   ratio collapse to 100.1 % at the small-prefill shape (122.3 vs
   122.2). gfx1151's shared-memory regime apparently makes Lloyd
   codebook indexing a constant-cost overhead, not one that's
   absorbed into KV-cache traffic at long contexts. Documented for
   future Lloyd-decode-perf work but not actionable in this PR.

Absolute throughput is ~3.2× lower than gfx1100 on prefill and
~2.3× lower on decode — consistent with the LPDDR5x-vs-GDDR6
bandwidth ratio (~250 / 960 = 0.26).

## gfx1151 coherence (post-review, 2026-05-08)

`./scripts/coherence-gate.sh` run on gfx1151 against the 4 Lloyd rows
(cap-4b, reason-9b, long-prefill-4b, long-prefill-9b). All produce
fluent output and terminate cleanly with `<|im_end|>`. Raw rows from
`/tmp/coherence-gfx1151-pr195.md`:

```
qwen3.5-4b.mq3-lloyd / cap-mq3-lloyd-4b
  prefill 21 tok @ 457.4 tok/s  decode 72.2 tok/s  → "Paris is the capital of France."

qwen3.5-9b.mq3-lloyd / reason-mq3-lloyd-9b
  prefill 36 tok @ 363.3 tok/s  decode 50.6 tok/s  → "Final Number: 8"
  (sheep-riddle answer is wrong on this model class — the same prompt
   on gfx1100 produces the same off-by-one; not a kernel issue.)

qwen3.5-4b.mq3-lloyd / long-prefill-mq3-lloyd-4b
  prefill 190 tok @ 964.3 tok/s  decode 70.4 tok/s  → 220-token LRU cache
  walkthrough, fluent, on-topic, no attractor loops.

qwen3.5-9b.mq3-lloyd / long-prefill-mq3-lloyd-9b
  prefill 190 tok @ 496.2 tok/s  decode 49.0 tok/s  → 126-token LRU cache
  explanation, fluent, terminates cleanly.
```

The `HARD_ERROR exit=139` entries in the gate report are the
pre-existing daemon-shutdown segfault (affects every model row, MQ4
through MQ6 — not a Lloyd or kernel issue). Generation completes
successfully and emits the `done` event before the segfault on
cleanup; the `tokens=N` field in the gate report confirms full
output for each row.

The 964.3 / 496.2 tok/s long-prefill numbers confirm the new batched
WMMA path is engaged on gfx1151 (pre-B2 per-token `forward_scratch`
fallback would show ~30-50 tok/s at this prompt length).

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
  unvalidated as flagged in B1. **As of the post-review hardening
  commit, Lloyd-MQ3 on gfx12 ships behind an opt-in env gate
  (`HIPFIRE_LLOYD_GFX12=1`)**: with the gate unset, gfx1200/1201
  fall through to the per-token `forward_scratch` fallback (correct,
  ~14× slower) instead of dispatching the unvalidated WMMA kernels.
  RDNA4 reviewers should set `HIPFIRE_LLOYD_GFX12=1` when running the
  parity tests / coherence-gate to exercise the gfx12 path. Once
  external CI confirms gfx12 parity, the gate can be dropped or
  default-flipped in a follow-up commit.
- **PFlash / spec-decode interaction.** Phase C measured non-spec
  prefill only. The DDTree / PFlash paths route through different
  matchers and were not exercised here. This is the obvious next
  area to bench once #116 lands.

- **Decode-path numerical drift.** This PR's 4 WMMA prefill kernels
  (`gemm_*_mq3g256_lloyd_wmma.hip` × gfx11/gfx12) are single-acc by
  design — verified at `gemm_mq3g256_lloyd_residual_wmma.hip:72`,
  `gemm_qkvza_…:97`, `gemm_qkv_…:71`, `gemm_gate_up_…:65` (each uses
  one `float8_t acc` per lane). So **prefill** is drift-free.
  However, the **decode** path GEMV kernels (`gemv_mq3g256_lloyd*.hip`
  + the fused gemv siblings) are unchanged on this branch and still
  carry the universal multi-accumulator drift documented at
  `benchmarks/results/devlog_20260507_mq3_lloyd_gfx1151.md`: a 0.9 %
  PPL drift on Qwen3.5-9B caused by the K4
  `(acc0+acc1)+(acc2+acc3)` reduction order, reproducible on
  gfx1100/1101/1102/1151. The Phase C decode ratios reported here
  (94 % at gfx1100, 91 % at gfx1151) are throughput-only; they do
  not measure PPL drift, and the master decode path carries the
  same drift envelope independent of this PR. Closing it requires a
  single-accumulator port of the GEMV family (mirroring this PR's
  WMMA kernels and the production MQ4-Lloyd kernels) — separate
  follow-up, tracked outside this PR's scope.

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
