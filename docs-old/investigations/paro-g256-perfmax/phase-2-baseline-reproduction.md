# Phase 2 — Baseline reproduction on hiptrx gfx1201

> Branch `feat/paro-g256-perfmax` HEAD f833925f, hiptrx host
> (R9700, gfx1201, ROCm 7.2). Re-running the May 14 baseline command from
> `docs/investigations/2026-05-14-paroquant-hiptrx-baseline/README.md` to
> establish ground-truth before any Phase 3 fusion work.

## Headline

PARO4G128T 0.8B decode is **161.9 tok/s** (was 186.6 on May 14, SHA 26ebcfc3).
**-13.2% regression vs May 14**, but kernel timings are **byte-identical** to baseline.
The regression is entirely in the dispatch/host-side overhead (~0.8ms/token extra),
NOT in any kernel. Forward perf work proceeds against the new ground truth.

## Setup

```
host:     hiptrx (R9700 / gfx1201 / RDNA4)
ROCm:     7.2
HFQ:      ~/.hipfire/models/qwen3.5-0.8b.paro4g128-engine.hfq (1004630144 B, md5 fcb07cce)
import:   --layout engine --copy-floats f16 (PARO4G128T, qtype 29)
GPUs:     4× R9700, daemons killed before bench (GPU 60 MB held)
prompt:   internal bench (--prefill 32 tokens)
gen:      --gen 64
warmup:   --warmup 2 + HIPFIRE_DPM_WARMUP_SECS=3
env:      HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE=q8
```

## Fresh-process median × 3 (no profiling overhead)

| run | gen tok/s | prefill tok/s | avg ms/tok | BW GiB/s |
|---:|---:|---:|---:|---:|
| 1 | 161.9 | 172.8 | 6.09 | 151.5 |
| 2 | 161.9 | 172.6 | 6.08 | 151.5 |
| 3 | 161.9 | 172.5 | 6.09 | 151.5 |
| **median** | **161.9** | **172.6** | **6.09** | **151.5** |

Spread is ±0.1% per-call — extremely deterministic. Daemons killed, GPU idle, no
contention.

## Regression vs May 14 baseline

| metric | May 14 (26ebcfc3) | now (f833925f) | Δ |
|---|---:|---:|---:|
| gen tok/s | 186.6 | 161.9 | **-13.2%** |
| prefill tok/s | 193.4 | 172.6 | -10.8% |
| avg ms/tok | 5.27 | 6.09 | +15.6% |
| BW GiB/s | 174.6 | 151.5 | -13.2% |

Triggers Δ ≥ 5% investigation. Kernel profile breakdown (HIPFIRE_PROFILE_DECODE=1):

| kernel | May 14 ms | now ms | Δ |
|---|---:|---:|---:|
| paro4g128t_rotate (8064×) | 79.8 | 80.0 | +0.3% |
| gemv_paro4g128t_prerotated (6528×) | 77.0 | 77.0 | 0% |
| gemv_paro4g128t_prerotated_residual (3072×) | 44.7 | 44.7 | 0% |
| rmsnorm_f32 (3136×) | 31.1 | 30.7 | -1.3% |
| paro4g128t_swiglu_rotate (1536×) | 15.3 | 15.3 | 0% |

**Per-kernel timings are identical (±1.3%).** Sum of all profiled kernel time = 322.2ms
over 64 tokens = 5.03ms/token of kernel work. Wall time without profile = 6.09ms/token.
So dispatch/sync overhead = 1.06ms/token.

May 14 wall = 5.36ms/token (186.6 → 5.36ms). May 14 dispatch overhead = 0.33ms/token.

**Regression source: dispatch overhead grew from 0.33ms → 1.06ms = +0.73ms per token.**
Across 30784 launches / 64 tokens = 481 launches/token. Extra overhead = ~1.5µs per
kernel launch. Roughly consistent with the cost of one extra Rust-side
indirection / capture-aware abstraction layer.

## Suspect commits (regression source, deferred for now)

Between SHA 26ebcfc3 (baseline) and f833925f (HEAD) the branch merged:

- PR #318 `feat/paroquant-graph-capture` — added graph-capture support. Even though
  the gfx12 graph path is disabled (`HIPFIRE_GRAPH=0` default; gfx12 has a separate
  NaN bug per baseline doc), the dispatch path may now be running through extra
  capture-aware indirection per launch.
- PR #316 `feat/paroquant-native` — GemmaRMSNorm `(1+w)` bake on PARO load path
  (offline, no inference cost), MoE loader for A3B, conditional lm_head.
- PR #317 `fix/moe-hipgraph-atomicadd` — atomic-free MoE down (gated on MoE path,
  shouldn't affect dense 0.8B).
- Master merges (a09af869) — kernel and dispatch changes from trunk in the May 14–22 window.

**Investigation deferred** until after Phase 3 levers ship. Rationale: Lever 1
(rotate fusion) has measured upper bound +16.8% prefill — if it stacks cleanly on
this baseline, decode will land at ≥189 tok/s (above May 14 baseline). If Lever 1
underdelivers, *then* the regression hunt becomes priority.

## test_inference correctness — 9/9 PASS

```
forward() produces finite logits                  OK (7116ms) max=9.1978
forward_scratch() matches forward()               OK max_diff=0.000000
10-token sequence completes (no hang)             OK (143 tok/s)
</think> encodes to detectable token(s)           OK [248069]
ChatML special tokens encode as single tokens     OK im_start=248045 im_end=248046
givens4 KV cache allocates                        OK 24 layers, hd=256, asym3
givens4 forward completes                         OK (752ms, 2 tokens)
decode speed > 10 tok/s                           OK (151.5 tok/s)
VRAM: KV cache free + drain                       OK leak=0.00MB
```

## Phase 3 entry point

Baseline frozen at **161.9 tok/s decode / 172.6 tok/s prefill**. All Lever 1+2
deltas will be measured against this number with the same methodology (3 fresh
processes, median, deterministic prompt). Headroom budget:

| target | from baseline | path |
|---|---:|---|
| Recover to May 14 baseline (186.6) | +15.3% | Lever 1 alone (measured upper bound +16.8% prefill) |
| Atlas engineering ceiling (~218) | +34.7% | Lever 1 + Lever 2 + rmsnorm-rotate fusion |
| MQ4 0.8B parity (403) | +149% | structural — needs Path B (PARO4G256_MQ) |
| Dense 0.8B "≥90% MQ4" gate (≥363) | +124% | structural |
