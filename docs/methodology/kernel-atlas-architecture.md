<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->

# Kernel Atlas architecture

How measurement emission, analysis, and experiment contracts fit together.
Usage recipes live in [`kernel-atlas.md`](kernel-atlas.md). This page is
layer ownership and status only.

| Field | Value |
|---|---|
| Inventory date | 2026-07-19 |
| Audited source ref | `692a726dde53508cb53de1a74c720e75a7c9f33e` |
| On-disk schema | `hipfire.kernel_atlas.v0` |
| Python CLI | [`scripts/kernel_atlas.py`](../../scripts/kernel_atlas.py) |
| Rust crate | [`crates/hipfire-atlas/`](../../crates/hipfire-atlas/) |
| Agent skill | [`.agents/skills/hipfire-kernel-atlas/`](../../.agents/skills/hipfire-kernel-atlas/) |
| Quality corpus (sibling, planned join) | Astrea — see [`astrea-atlas-pareto-workflow.md`](astrea-atlas-pareto-workflow.md), [`astrea-model-policy.md`](astrea-model-policy.md). Not a current VALIDATION/INDEX owner. |

Shared experiment identity when joining quality and runtime rows:

```text
git_sha + model_hash + workload + quant_variant + runtime_variant
```

Astrea answers “is this quant worth running?” Atlas answers “what did it
cost or buy on named hardware?” Neither file admits product defaults;
admissions stay in [`docs/admissions.yml`](../admissions.yml) (fail closed).

---

## Rust Atlas status

`crates/hipfire-atlas` is **present and transitional**. Its own
`main.rs` / `lib.rs` mark the long-term user and agent surface as
**Python-first**. Do not expand the Rust CLI as the primary Atlas UX.

Keep in the Rust crate (useful concepts, already shipped as **independent
transitional feature subsets** — not contract-compatible bridges to Python):

| Piece | Role |
|---|---|
| `schema::AtlasRow` + `ATLAS_SCHEMA` | Typed JSONL row; `append_to_jsonl` |
| `parse` | Legacy stdout → typed rows (`parse-bench` / `parse-dflash`) for binaries not yet on `--emit-atlas` |
| `render` / `suggest` / `task` / `eval` | Transitional subsets only: task JSON is not interoperable with Python; Rust `eval` lacks baseline/repeat/stability/ledger semantics; Rust `render` is not the Python ISA fit view |
| `profile_report` | Attach internal profile / rocprof cross-check structs |

Intended emission path for metrics a Rust binary already owns:
`--emit-atlas <path>` (example:
`crates/hipfire-runtime/examples/bench_qwen35_mq4.rs`). Prefer emit over
stdout scrape for new work. Wiring is **shipped / ref-pinned** with
**partial** binary coverage — do not assume every example emits.

Analysis at scale, ISA inspection, dispatch provenance scans, collect
orchestration, `graph-ab`, history-aware `suggest`, and agent workflows:
**`scripts/kernel_atlas.py`**.

---

## Layers

### Layer 1 — Measurement emission

Owners: bench / inference / demo binaries (and Python collectors that
drive them).

| Mechanism | Status |
|---|---|
| `--emit-atlas <path.jsonl>` in-process | **shipped / ref-pinned** — partial binary coverage; wire remaining examples deliberately; do not assume every binary emits |
| `kernel_atlas.py collect-ar` / `collect-dflash` | **shipped / ref-pinned** — scrape + enrich from live runs; success → `status: ok` observation rows; failure → `status: error` records |
| `hipfire-atlas parse-bench` / `parse-dflash` | **shipped / ref-pinned** — offline stdout → typed JSONL row migration |
| `kernel_atlas.py parse {bench,dflash}` | **shipped / ref-pinned** — metrics-only JSON extract (prints metrics; does **not** write Atlas rows) |

Row contract (minimum for new **successful** `status: ok` **observations** —
collector success proves collection, not INDEX **measured** eligibility):

| Field | Notes |
|---|---|
| `schema` | `hipfire.kernel_atlas.v0` |
| `status` | `ok` for successful observations; collector failures use `error` and are non-observations |
| `phase` | On `ok`: `prefill` \| `decode_ar` \| `decode_dflash`. Error records use provisional `ar` / `dflash` only |
| `workload_kind` | Binary or workload class |
| Identity | `git_sha`, host, `arch`, ROCm/hipcc when known |
| Model | path; hash/bytes when `--hash-models` or emit provides them |
| Quant / runtime labels | format + graph/KV/flash/config tuple as applicable |

**Measured vs exploratory:** apply INDEX **measured** only when the row also
carries the complete fixture, binary/model identity, and date manifest required
by [`docs/INDEX.md`](../INDEX.md) and [`perf-benchmarking.md`](perf-benchmarking.md)
(model hashing is optional on the collector — absence keeps the row
**exploratory**). `status: ok` alone is never durable measured-state eligibility.

Run hygiene (keep small wins honest):

- `pass_index` / discard-first-pass policy when multi-pass
- warmup / gen / prefill token counts
- `dpm_warmup_secs` when used
- `binary_md5`, dirty/diff md5
- `prompt_md5` when a prompt file is involved

Optional artifacts on the same row:

- `profile_kernels` (+ op attribution, rocprof coverage tags when present)
- `isa` manifest path or inline objects
- `dispatch` provenance path or inline entries
- `runtime_route` from `build_route_manifest` (kv_mode, graph_enabled,
  graph_blob_count, inferred attention_impl, warnings)

Route fields are **observed or inferred evidence**. Missing fields stay
unknown; do not invent flash/graph activity. Graph capture that changes
attention implementation must be visible here before perf is trusted.

### Layer 2 — Python Atlas CLI and analyzer

Owner: [`scripts/kernel_atlas.py`](../../scripts/kernel_atlas.py).

Responsibilities:

- Drive AR/DFlash matrices and write private JSONL under
  `.codeinsight+research/kernel-atlas/`
- Attach ISA manifests (HSACO/code-object inspect) and dispatch/source
  scans
- Join profile hot names to ISA symbols for fit view
- Render agent-readable ASCII fit tables
- Emit `suggest` queues and `task` / `task-pytorch` bundles
- Run `eval` and `graph-ab`; append local ledgers
- Annotate rocprof coverage / BLINDSPOT helpers (see
  [`rocprof-coverage.md`](rocprof-coverage.md))

ISA capability tables inside the script are **heuristic**. Unknown
`arch` → no native-matrix claim. Full occupancy, clock, and cache
models are **out of scope** (skill documents interpretation limits).

### Layer 3 — Advisor

Consumes the corpus to propose **bounded** experiments (launch-bound
retunes, K-unroll, graph thresholds, flash cutovers, KV policy,
baseline/candidate deltas).

| Piece | Status |
|---|---|
| `suggest` ranker + history demotion | **shipped / ref-pinned** in Python (`pass` and `unstable` with speedup < 1.0 demote; unstable demotion ≠ rejection evidence) |
| `task` / `eval` contracts | **shipped / ref-pinned** (Python primary; Rust subset not compatible) |
| Autonomous mutation / closed-loop ship | **planned** — blocked until task/eval are boring and VALIDATION routes stay human-owned |

Advisor output is an experiment queue, never a certification.

---

## Required eval modes (contracts)

### Graph A/B

Implemented as `python3 scripts/kernel_atlas.py graph-ab` (and manual
equivalent). Intent:

```text
HIPFIRE_GRAPH=0  pass 1 (record, not headline)
HIPFIRE_GRAPH=0  pass 2 (JIT-controlled)
HIPFIRE_GRAPH=1  pass 1
HIPFIRE_GRAPH=1  pass 2
```

Report must surface graph-off/on tok/s, lift, latency deltas, prefill
delta, **route changes**, and correctness status when a correctness
command is supplied. That command must be a **current** VALIDATION
route — not a retired coherence-gate battery presented as acceptance.

### Baseline / candidate compare

`eval --refresh-baseline` / `--baseline baseline.json` makes “lost vs
prior median” mechanical for a single task. Cross-quant Pareto with
Astrea quality rows is the composition goal; incomplete joins stay
labeled incomplete (`perf_only` when correctness evidence is missing).

### Correctness join

A performance row without correctness evidence is **incomplete** for
any ship/promote claim.

| Join target | Disposition |
|---|---|
| Path-specific parity / state oracle | **Current** when it exists for the surface |
| `test_kernels` / channel | **Current** for kernel numeric claims |
| Maintained serve / LFM / Redline harnesses | **Current** only for the semantics those harnesses own ([`VALIDATION.md`](../VALIDATION.md)) |
| Astrea KLD/PPL/MSE rows | **planned / blocked** as a validation route — quality evidence only until INDEX/VALIDATION name Astrea as owner; compose manually via shared ids |
| `scripts/coherence-gate-*.sh` | **Historical reproduction only** — not acceptance, not default `--correctness-command` |

Mark or treat rows as `perf_only` when correctness is absent. Fail
closed rather than implying a green gate.

---

## Relationship to Astrea

Astrea is the intended quality-evidence sibling (KLD vs reference, PPL,
attribution, MSE, calibration method, promotion maps, bpw, oracle traces
when available). It is **not** registered as a current owner in
[`docs/INDEX.md`](../INDEX.md) or [`docs/VALIDATION.md`](../VALIDATION.md);
under fail-closed rules the ownership/join stays **planned / blocked**
until those owners name it. Treat Astrea outputs as quality evidence only
— not a current validation route.

Handoff manifest fields (minimum **planned** intent for a candidate id;
not a shipped Atlas schema field):

`candidate_id`, `model_path`, `model_hash`, `source_model`,
`quant_format`, `calibration_methods`, `promotion_map`, `bpw`,
`size_bytes`, `kld_mean`, `ppl`, `mse_summary`, `reference_id`,
`quality_artifacts`.

Neither `scripts/kernel_atlas.py` nor `crates/hipfire-atlas` currently
defines or accepts `candidate_id`. Runtime/quality join under a shared id
is a **manual / planned** handoff contract — compose externally until
first-class auto-join ships. Final human decision surface is a
quality/performance Pareto table — still not an `admissions.yml` write.

Workflow prose: [`astrea-atlas-pareto-workflow.md`](astrea-atlas-pareto-workflow.md).
Policy: [`astrea-model-policy.md`](astrea-model-policy.md).

---

## Migration status (source-derived)

| Step | Change | Status |
|---|---|---|
| 1 | Treat `crates/hipfire-atlas` as transitional; do not grow as long-term UX | **shipped / ref-pinned** direction (crate docs state this; crate remains) |
| 2 | Python CLI as primary analysis/collect surface | **shipped / ref-pinned** (`scripts/kernel_atlas.py`) |
| 3 | Python parity for read-scale workflows + `render-fit` / `suggest` / `task` / `eval` | **shipped / ref-pinned** (Rust holds independent transitional subsets — not contract-compatible bridges) |
| 4 | Route manifests + `graph-ab` | **shipped / ref-pinned** in Python (`build_route_manifest`, `graph-ab`) |
| 5 | Baseline/candidate eval + correctness command hooks | **shipped / ref-pinned**; correctness **join policy** fail-closed per VALIDATION |
| 6 | First-class Astrea candidate manifest auto-join in Atlas CLI | **planned** / incomplete — compose manually via shared ids until owned; no `candidate_id` field in Atlas today |
| 7 | Remove Rust Atlas crate after full Python parity + emit coverage | **planned** — not done; crate remains |

Do not document step 7 as complete. Do not claim every bench emits
Atlas natively.

---

## Explicit non-goals

| Item | Disposition |
|---|---|
| Atlas as universal GPU gate | **Rejected** |
| Coherence-gate batteries as Atlas default correctness | **Rejected** (retired acceptance) |
| Fit view as occupancy or roofline certification | **Rejected** |
| Capability claims for hardware-unwitnessed arches | **Blocked** |
| Admissions from Atlas ledgers | **Rejected** |
| Expanding Rust Atlas as the agent-facing product | **Rejected** direction |

---

## Related owners

| Concern | Owner |
|---|---|
| How to run Atlas day-to-day | [`kernel-atlas.md`](kernel-atlas.md) |
| Claim → validation route | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Arch-port evidence tiers | [`arch-port-validation.md`](arch-port-validation.md) |
| Perf protocol | [`perf-benchmarking.md`](perf-benchmarking.md) |
| rocprof vs internal timers | [`rocprof-coverage.md`](rocprof-coverage.md) |
| Executable skill | `.agents/skills/hipfire-kernel-atlas/` |
