<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# Cross-architecture bandwidth-utilization campaign (gfx1100 / gfx1201 / gfx1151)

| Field | Value |
|---|---|
| State | **planned** (see `docs/INDEX.md` truth states; no executable authority) |
| Date | 2026-07-19 |
| Author | Kaden Schutt (drafted with agent assistance) |
| Validation route | [`docs/VALIDATION.md`](../VALIDATION.md); promotion only via [`docs/REDLINE.md`](../REDLINE.md) §7 |
| Methodology owner | [`docs/methodology/perf-benchmarking.md`](../methodology/perf-benchmarking.md) |
| Publication surface | `redline-pub/` (standalone Redline crate repo; radiowave harness) |
| Loop machinery | `autoresearch/` (`ar/`, `config/loop_*.toml`, PR merge-gate) |

## 1. Motivation

Batch-1 autoregressive decode is memory-dominated, but achieved fraction of
peak memory bandwidth **falls as peak bandwidth rises**. This was measured on
NVIDIA (L4 ≈ 81% of analytic floor, H100 ≈ 27%; arXiv:2605.30571) and is
independently visible in hipfire's own cross-SKU results for the same model,
quant format, KV mode, and harness class:

| GPU | Arch | Peak DRAM BW (spec) | TG128 tok/s (baseline) | Implied traffic* | Utilization est.* |
|---|---|---:|---:|---:|---:|
| Radeon 7900 XTX | gfx1100 | 960 GB/s | ~250 | ~425 GB/s | **~44%** |
| Radeon AI PRO R9700 | gfx1201 | 640 GB/s | ~204 (203.93 median, 2026-07-13 checkpoint) | ~347 GB/s | **~54%** |
| Strix Halo (Ryzen AI Max) | gfx1151 | ~256 GB/s | ~115 | ~196 GB/s | **~76%** |

\* First-order estimate only: assumes ~1.7 GB effective weight traffic per
token for A3B MQ4R (~3B active params × ~0.55 B/param) and ignores KV and
activation traffic. **Phase 0 replaces this estimate with measured DRAM-byte
counters.** The tok/s baselines are campaign entry points to be re-measured
under §5 protocol before any publication use.

Two inversions define the campaign:

1. **gfx1201 converts bytes to tokens better than gfx1100** (tok/s ratio 0.82
   vs bandwidth ratio 0.67). The July Redline tape-shaping work is paying off;
   the shaping has not been ported to gfx1100.
2. **The slowest GPU is closest to its own roofline.** gfx1151 at ~76%
   utilization already sits at the level the NVIDIA study measured for its
   best SKU. The large dGPUs leave 25–55% of peak bandwidth unharvested.

**Targets.**

| Tier | gfx1100 | gfx1201 | gfx1151 |
|---|---|---|---|
| Primary: U ≥ 0.76 (Strix-Halo parity) | ≥ ~430 tok/s | ≥ ~280 tok/s | hold ≥ 0.76 (regression floor) |
| Stretch: U ≥ 0.85 (clearly beat NVIDIA's best documented batch-1 SKU, L4 81%) | ≥ ~480 tok/s | ≥ ~320 tok/s | ≥ ~135 tok/s |

Utilization, not tok/s, is the campaign metric: it normalizes across SKUs and
makes the result comparable to arXiv:2605.30571 (fraction-of-floor) and to
BaseRT's native-dispatch argument on Apple Silicon (arXiv:2607.00501).

**Publication intent.** A systems write-up: "Memory-bound but not
bandwidth-limited on RDNA" — the cross-arch utilization law plus retained-PM4
tape shaping as the countermeasure, with the three-SKU matrix, the July
110→204 campaign as case study, and the certification gates that rejected
invalid 200+ results. NVIDIA has the phenomenon documented with CUDA Graphs
as partial mitigation (1.259× on H100); hipfire has the stronger
countermeasure (launch elimination + tape reshaping) with cross-SKU evidence.
Assembled in `redline-pub/`; see §9 for its ABI-closure dependencies.

## 2. Definitions and measurement model

**Analytic memory floor** (per token): `B_floor = W_active + KV(ctx) + A`
where `W_active` is active-parameter bytes for the exact artifact (from the
sidecar/quantizer manifest, not param-count arithmetic), `KV(ctx)` is KV-read
bytes for the context length and KV mode, `A` is activation/scratch traffic
estimated from the tape's buffer census.

**Measured traffic** `B_meas`: DRAM bytes moved per token from hardware
counters (see §4 P0). `U_floor = B_floor / (BW_peak × t_token)`;
`U_meas = B_meas / (BW_peak × t_token)`. Report both; the NVIDIA comparison
uses `U_floor`.

**Dispatch residue**: `t_token − Σ kernel GPU µs` over one decode step,
decomposed into launch overhead, wait/fence stalls, and host-side gaps. The
retained tape gives exact dispatch counts (gfx1201 campaign: 833 → 733); the
profiled replay path gives GPU-side spans. At 204 tok/s and 733 dispatches
the mean dispatch slot is ~6.7 µs on gfx1201; quantifying how much of that
slot is kernel vs residue is the central Phase-0 measurement.

**Fixture identity (frozen for the campaign).**

- Model: `qwen3.6-35b-a3b.mq4r`, model md5 recorded per run.
- KV mode: `q8`. Sampler: per registry defaults; greedy for parity arms.
- Prompt: `benchmarks/prompts/` committed file, md5
  `d97ec9d3f761ec68093631be27d32441` (the gfx1201 loop prompt).
- Workload: TG128 at ctx ∈ {128, 512, 2048, 8192, 20000}; PP128 companion.
- Clocks: automatic (manual pinning measured as harmful on gfx1201; re-verify
  per arch, record policy in every checkpoint).

## 3. Hypotheses (ranked, falsifiable)

- **H1 — Dispatch residue dominates the dGPU gap.** Kernel bodies on
  gfx1100/gfx1201 are short; launch+wait residue per dispatch does not shrink
  with bandwidth. gfx1151's long kernels amortize it. Falsification: if
  measured residue < 10% of token time on gfx1100 while U stays < 60%, H1 is
  dead and H2/H4 take priority.
- **H2 — Occupancy/latency-hiding cliffs on short GEMV rows.** VGPR/SGPR
  pressure limits concurrent waves exactly where kernels are shortest
  (consistent with the redline-pub TILE=4 loss: occupancy dominated the
  shared-x win). Falsification: occupancy counters per kernel family vs
  kernel duration curve.
- **H3 — Tape shaping is not ported.** gfx12-specific encodings (GCR trim,
  acquire policy, register retention) have no gfx11 equivalents yet; gfx11
  needs its own cache-op audit. Falsification: port and measure; if residue
  unchanged, shaping was not the binding term on gfx11.
- **H4 — Memory topology.** 7900 XTX chiplet (GCD/MCD) partitioning and
  Infinity Cache policy penalize MoE expert streaming patterns; R9700 and
  Strix Halo have different cache hierarchies and (gfx1151) no Infinity
  Cache. Falsification: PMC hit-rate counters + expert-access stride study.
- **H5 — Clock/power residency.** DVFS residency interacts with short
  kernels; automatic clocks won on gfx1201 for utilization reasons that may
  not transfer. Falsification: fixed-clock A/B per arch under the noise
  protocol.

## 4. Phases

**P0 — Instrumentation (gate for everything else).**

1. DRAM-byte counters per arch via rocprofv3 (TCC/EA read/write request
   counters; document the counter→bytes mapping per arch in the checkpoint).
2. Counter validation: run a calibration kernel of known byte footprint;
   accept counters only if within 5% of analytic bytes. Without this, all U
   numbers are the §1 estimates and stay out of publication.
3. Per-dispatch latency histogram from the retained tape + profiled replay
   (GPU µs spans; redline core already exposes profiled replay in-workspace).
4. rocprof teardown-hang workaround (observed in the LFM2 trace): bounded
   traces, kill-after-CSV; instrumented runs are attribution-only, never
   baselines.
5. Dispatch census tooling: tape introspection per arch (count, unique
   kernels, waits, bytes per dispatch class).

**P1 — Campaign baselines.** Re-measure all three SKUs under §5 protocol
(fresh-process ABBA, ≥10 warmup rows, model/prompt md5 recorded). Publish as
dated `docs/perf-checkpoints/` entries; these are the **measured** numbers
every later claim diffs against.

**P2 — gfx1201 residue campaign (continuation).** Current edge: 733
dispatches, 203.93 tok/s. Lever queue from July checkpoints: wait census
follow-ups, context-bucketed tapes for the ctx≥8k utilization decay
(184 → 158 tok/s at 8k/20k — KV-traffic-dominated regime, candidate levers:
asym KV modes, KV-side HyperQuant-style bias correction), acquire-policy
unification, launch-register dirty-gating refinement.

**P3 — gfx1100 tape-port campaign.** Port the gfx12 shaping wins behind
gfx11-isolated kernels (existing isolation rule): cache-op encoding audit
(GCR is gfx12-specific), wave32/wave64 policy per kernel family, SRD
buffer-load policy, dispatch-elimination fusions (router wave64 fusion,
conv+norm fusions) re-validated on gfx11, Infinity-Cache-aware expert-layout
experiment (H4). Autoresearch loop drives the kernel composition search; see
§6.

**P4 — gfx1151 hold + documentation.** Regression-floor runs per release
candidate; document *why* UMA saturates (H-topology write-up: kernel length,
no PCIe, LPDDR latency-hiding) — this analysis is a paper section, not an
optimization target. Stretch: >VRAM MoE preparation pairs with the
weight_pager eviction-policy change (SpecMD Least-Stale; separate spec).

**P5 — Publication assembly.** redline-pub ABI closure per its
`docs/ABI-COVERAGE.md` checklist (arch-selecting capi builder — items 1–2;
`Pm4Ib.replay_profiled()` — item 3; per-arch smoke example — item 4), then
the arXiv draft with the P1 matrix, P2/P3 campaign arcs, and the
certification-gate methodology.

## 5. Measurement protocol (binding)

All campaign numbers obey the methodology owner plus campaign-specific rules:

1. Byte-identical prompts from `benchmarks/prompts/`; prompt md5 in every
   record. No canonical benches under `/tmp/`.
2. Fresh-process ABBA; reversed ordering for small wins; ≥10 warmup rows
   before retained-replay measurement.
3. Fifteen-position HIP-vs-PM4 shadow parity for any replay-attributed
   number; logits/KV/recurrent-state validation per REDLINE.md §7.
4. Eight-turn sampled serving battery with recall checks for any
   product-route claim.
5. gfx isolation: a kernel change for one arch must not alter another arch's
   binary behavior; pre-commit + PR merge-gate (Tier-3 GPU gate) enforce.
6. Automatic clocks unless a fixed-clock arm is the explicit hypothesis;
   record policy per row.
7. Rejection discipline: candidates failing shadow parity, sidecar
   freshness, or harness identity are recorded as rejected with cause (the
   July campaign's two invalid 200+ results are the pattern to preserve).
8. Every checkpoint is **measured**-class: date, fixture, binary and model
   identity, harness version. Promotion to a product default requires the
   full REDLINE.md §7 ladder and, ultimately, an `admissions.yml` row.

## 6. autoresearch integration

**Loop configs.** Add `autoresearch/config/loop_gfx1100.toml` and
`loop_gfx1151.toml` mirroring `loop_gfx1201.toml`: same model SKU, KV mode,
TG128, prompt md5; `baseline_ref = loop/gfxNNNN`; `cand_wall = 3.0`;
`k_exhaust = 5`; per-arch `[[workers]]` bound to the right card; watcher
leashes (`call_budget`, `wall_ttl_s`) unchanged. `prompt_gfx1151.md` exists;
the gfx1100 prompt seed gets the roofline map from P0 (top residue kernels,
occupancy cliffs, cache-policy notes) instead of the gfx1201 headroom list.

**Lever funnel (order is load-bearing).**

1. **autoresearch kernel loop** proposes compositions against the per-arch
   baseline ref; BOD candidate gate at 3% wall.
2. **redline-pub radiowave harness** isolates the lever: recipe-mode
   certified-vs-candidates A/B, correctness oracles (checked words), ≥3 reps,
   verdict table (WIN/LOSS/HOLD) with hypothesis labels — the
   `2026-07-17-q4-x4` SUMMARY is the format template.
3. **Product A/B** in hipfire under §5 protocol; only product-level wins
   enter checkpoints. Microbench wins do not transfer by default
   (TILE=4 precedent).
4. **Certification** per REDLINE.md §7 for any route promotion;
   `admissions.yml` rows are the campaign's terminal artifact.

**Agent routing.** gfx1201 loop currently runs one Sol@medium codex worker.
For the 3-arch campaign: one worker per arch loop, disjoint cards, disjoint
baseline refs; PR merge-gate classifies/routes/decides per
`config/pr_gate.toml`. Kernel-iteration cost on AMD hosts can optionally use
Kerncap-style extraction (arXiv:2605.03208) for reproducer generation; the
Redline recorder already captures equivalent dispatch state for HIP, so this
is a convenience, not a dependency.

## 7. Acceptance

| Gate | Criterion |
|---|---|
| P0 exit | Counters validated ≤5% vs analytic on calibration kernel; per-dispatch residue decomposition published for all three SKUs |
| P2 exit | gfx1201 U_floor ≥ 0.76 at ctx ≤ 2048 without correctness or serving regressions; ctx≥8k decay explained by measured KV traffic |
| P3 exit | gfx1100 U_floor ≥ 0.76 primary, ≥ 0.85 stretch, same gates |
| P4 exit | gfx1151 U_floor ≥ 0.76 held across the campaign window; topology analysis written |
| P5 exit | redline-pub ABI checklist closed; per-arch smoke green; paper draft assembled with only measured-class numbers |
| Campaign exit | First `admissions.yml` rows earned per REDLINE.md §7 for the certified per-arch routes |

Missing the stretch tier is not a failure state: the primary tier already
exceeds the best NVIDIA batch-1 utilization in the literature if U_floor
clears 0.81 on either dGPU.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| PMC counter semantics differ per arch; byte counts wrong | P0 calibration gate; counter mapping documented per arch; fall back to analytic-floor-only reporting |
| rocprof overhead/hangs distort attribution | Instrumented runs never baselines; bounded traces; kill-after-CSV |
| Microbench win fails to transfer to product | Funnel order §6; product A/B required; TILE=4 precedent cited in every radiowave summary |
| gfx isolation violation leaks gfx12 kernels into gfx11 measurements | Pre-commit hotspot gates + PR merge-gate; per-arch binary diffing |
| Long-context decay conflated with decode residue | Separate ctx-bucketed tapes; KV-traffic accounting in B_floor |
| Harness drift invalidates cross-week comparisons | Frozen fixture identity §2; md5s in every checkpoint; speed-gate floors untouched by this campaign |
| Utilization estimates (§1) quoted as measurements | All estimated cells marked until P1 re-baseline; publication uses measured-class only |

## 9. References

- arXiv:2605.30571 — batch-1 decode utilization falls as peak BW rises; CUDA
  Graphs 1.259× on H100 (the phenomenon + partial NVIDIA countermeasure).
- arXiv:2607.00501 — BaseRT: native-dispatch runtime beats abstraction-layer
  runtimes on unified memory (thesis parallel on Apple Silicon).
- arXiv:2606.23406 — HyperQuant: KV bias-correction under quantization
  (candidate lever for the ctx≥8k KV-traffic regime).
- arXiv:2602.03921 — SpecMD: MoE expert access violates LRU/LFU locality
  (input to the P4 weight_pager policy work; separate spec).
- arXiv:2605.03208 — Kerncap: AMD kernel extraction/isolation (optional
  reproducer tooling for the kernel loop).
- `docs/perf-checkpoints/2026-07-13-redline-mq4r-110-to-204.md` — the gfx1201
  campaign arc this spec extends cross-arch.
- `redline-pub/docs/ABI-COVERAGE.md` — ABI-closure checklist (P5 dependency).
- `autoresearch/config/loop_gfx1201.toml` — loop-config template.
