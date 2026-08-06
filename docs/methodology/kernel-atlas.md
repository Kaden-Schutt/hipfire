<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->

# Kernel Atlas

Phase-aware **measurement corpus** for hipfire kernels and runtimes.
Atlas turns bench / DFlash output (and optional ISA + dispatch
provenance) into JSONL rows, fit views, suggestion queues, and local
task/eval ledgers.

| Field | Value |
|---|---|
| Inventory date | 2026-07-19 |
| Audited source ref | `692a726dde53508cb53de1a74c720e75a7c9f33e` |
| Primary CLI | [`scripts/kernel_atlas.py`](../../scripts/kernel_atlas.py) |
| Schema constant | `hipfire.kernel_atlas.v0` ([`crates/hipfire-atlas/src/schema.rs`](../../crates/hipfire-atlas/src/schema.rs)) |
| Agent skill | [`.agents/skills/hipfire-kernel-atlas/`](../../.agents/skills/hipfire-kernel-atlas/) |
| Fit wrapper | [`.agents/skills/hipfire-kernel-atlas/render-fit.sh`](../../.agents/skills/hipfire-kernel-atlas/render-fit.sh) → `python3 scripts/kernel_atlas.py render-fit` |
| Architecture / layers | [`kernel-atlas-architecture.md`](kernel-atlas-architecture.md) |
| Validation routes | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Perf protocol | [`perf-benchmarking.md`](perf-benchmarking.md) |
| rocprof coverage | [`rocprof-coverage.md`](rocprof-coverage.md) |

**Atlas is not a gate.** Collector `status: ok` means a **successful
observation** was recorded. That alone is **not** INDEX **measured** evidence.
Apply **measured** only when the row carries the complete fixture, binary/model
identity, and date manifest required by [`docs/INDEX.md`](../INDEX.md) and
[`perf-benchmarking.md`](perf-benchmarking.md); otherwise treat the row as
**exploratory**. Neither class admits routes, replaces Tier C/P/S in
[`arch-port-validation.md`](arch-port-validation.md), or certifies Redline.
Promotion still follows VALIDATION + (when applicable)
[`REDLINE.md`](../REDLINE.md) and fail-closed
[`admissions.yml`](../admissions.yml) (schema v2; exact earned rows only). Collector `status: error` records
are **not** observations (see Phases).

---

## Phases

Successful observation rows (`status: ok`) use exactly one of:

| Phase | Source surface |
|---|---|
| `prefill` | AR prompt processing (e.g. `bench_qwen35_mq4 --prefill N`) |
| `decode_ar` | Target-only AR generation from the same bench class |
| `decode_dflash` | Speculative decode (`dflash_spec_demo`) |

This keeps prefill visible for non-DFlash users while preserving
acceptance/tau-style DFlash metrics when the demo prints them.

**Error records are not observations.** On collector failure,
`collect-ar` / `collect-dflash` still append JSONL with `status: error`
and provisional `phase: ar` or `phase: dflash` (plus `stderr_tail`). Those
rows fail closed: do not treat them as phase-contract observations, fit-view
inputs, or evidence for a perf claim.

---

## Private corpus location

Keep raw runs out of git:

```bash
mkdir -p .codeinsight+research/kernel-atlas/runs
mkdir -p .codeinsight+research/kernel-atlas/tasks
```

That tree is gitignored. Commit harness/docs changes or carefully
redacted public samples only — not private model paths or full corpora.

---

## CLI surface (`scripts/kernel_atlas.py`)

| Subcommand | Role |
|---|---|
| `collect-ar` | Run AR bench; emit `prefill` + `decode_ar` JSONL rows on success (`status: ok`); append `status: error` records on failure |
| `collect-dflash` | Run DFlash demo; emit `decode_dflash` rows on success; append `status: error` records on failure |
| `parse {bench,dflash}` | Extract a **metrics-only** JSON object from saved stdout (print to stdout; does **not** write Atlas rows) |
| `render-fit` | ASCII ISA/quant fit view for one row |
| `suggest` | Ranked experiment queue (not predicted wins) |
| `task` | Bounded `task.json` + `TASK.md` from a row |
| `task-pytorch` | Same task contract from a PyTorch op/shape (no hipfire row) |
| `eval` | Run task benchmark/correctness contract; ledger + result |
| `graph-ab` | `HIPFIRE_GRAPH=0/1` A/B from a row’s eval contract |

Rust companion (transitional): `crates/hipfire-atlas` binary
`hipfire-atlas` — corpus read/count/head, legacy **row** parse
(`parse-bench` / `parse-dflash` → typed JSONL), and independent
transitional subsets of render/suggest/task/eval. Rust and Python task
JSON shapes are **not** interoperable; Rust `eval` lacks Python
baseline/repeat/stability/ledger semantics; Rust `render` is **not** the
Python ISA fit view. Prefer Python for collection orchestration and large
analysis. See architecture doc.

Offline **row** migration from saved stdout: use Rust
`hipfire-atlas parse-bench` / `parse-dflash` (writes typed rows). Python
`parse` only dumps metrics JSON.

In-process emit (when the binary is wired): some bench binaries take
`--emit-atlas <path.jsonl>` (e.g.
`crates/hipfire-runtime/examples/bench_qwen35_mq4.rs`). Coverage is
partial — do not assume every example emits.

---

## AR collection

```bash
python3 scripts/kernel_atlas.py collect-ar \
  --model ~/.hipfire/models/qwen3.5-27b.mq4 \
  --workload qwen3.5-27b \
  --model-size 27b \
  --quant mq4 \
  --prefill 32 \
  --prefill 128 \
  --gen 50 \
  --kv-mode asym3 \
  --graph \
  --output .codeinsight+research/kernel-atlas/runs/$(date -u +%Y%m%dT%H%M%SZ)-ar.jsonl
```

Useful flags:

| Flag | Effect |
|---|---|
| `--env KEY=VALUE` | Variant knobs (e.g. `HIPFIRE_GEMV_ROWS=4`) |
| `--profile-prefill` / `--profile-decode` | Capture per-kernel profile tables into `artifacts.profile_kernels` with first-pass op tags |
| `--hash-models` | Model file md5 (opt-in; large models are slow) |
| `--dpm-warmup-secs` | Stationary GPU warmup before measure |
| `--arch` / env `HIPFIRE_TARGET_ARCH` / `HIPFIRE_BASELINE_ARCH` | Pin reported arch when detection is ambiguous |

Provenance on rows includes benchmark binary md5, git dirty state, and
diff md5 — required before comparing dirty-worktree numbers to committed
baselines.

---

## DFlash collection

```bash
python3 scripts/kernel_atlas.py collect-dflash \
  --target ~/.hipfire/models/qwen3.5-27b.mq4 \
  --draft ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
  --prompt-file benchmarks/prompts/merge_sort_thinking_off.txt \
  --workload qwen3.5-27b-dflash-merge-sort \
  --max-tokens 256 \
  --ctx 2048 \
  --kv-mode q8 \
  --output .codeinsight+research/kernel-atlas/runs/$(date -u +%Y%m%dT%H%M%SZ)-dflash.jsonl
```

DFlash rows record prompt md5 plus metrics the demo prints
(`decode_tok_s`, tau, TTFT, emitted/accepted tokens, cycles, …).
`collect-dflash` **rejects** every `--kv-mode` except `q8`, `fwht2`,
`fwht3`, or `fwht4` before launch (default `q8`). `asym*` is not accepted
by this collector — do not paste AR defaults into DFlash collection.

Tok/s alone is **not** DFlash correctness. For correctness claims, use a
path named in VALIDATION (path oracle and/or maintained harnesses) —
**not** retired `scripts/coherence-gate-*.sh` as acceptance. Those
scripts remain on disk for **historical reproduction only**.

---

## ISA manifests

Opt-in: inspect compiled HSACO/code objects under `.hipfire_kernels/`
(or a single file) via `llvm-readobj` / `llvm-objdump` (and
`clang-offload-bundler` when bundled).

Inline or externalize:

```bash
python3 scripts/kernel_atlas.py collect-ar \
  --model ~/.hipfire/models/qwen3.5-0.8b.mq4 \
  --workload qwen3.5-0.8b \
  --model-size 0.8b \
  --prefill 32 \
  --gen 5 \
  --isa-dir .hipfire_kernels \
  --isa-filter 'gemm_hfq4g256_residual' \
  --isa-limit 1 \
  --isa-output .codeinsight+research/kernel-atlas/runs/isa-gfx1201.json \
  --output .codeinsight+research/kernel-atlas/runs/atlas-with-isa.jsonl
```

Manifest fields (when tools succeed): object path + md5, offload bundle
target, `amdhsa.target`, per-kernel VGPR/SGPR/LDS/private/spills,
workgroup and wavefront size, instruction/opcode/category counts,
kernel symbols.

**Occupancy and full counter models are out of scope here.** Fit view
is a heuristic ISA/resource readout — details and interpretation rules
live in `.agents/skills/hipfire-kernel-atlas/SKILL.md`. Unknown arches
render observed ISA without inventing matrix-unit capability
(`arch_capability` fail-closed notes in `kernel_atlas.py`).

---

## Dispatch provenance

```bash
python3 scripts/kernel_atlas.py collect-ar \
  ... \
  --profile-prefill --profile-decode \
  --dispatch-provenance \
  --dispatch-output .codeinsight+research/kernel-atlas/runs/dispatch-gfx1201.json \
  --output .codeinsight+research/kernel-atlas/runs/atlas-gfx1201.jsonl
```

Per profiled kernel name, the manifest records source candidates under
`kernels/src/` (md5s), dispatch/source refs under `crates/`, `cli/`,
`kernels/src/`, inferred env controls, and op attribution.

This is **evidence to inspect**, not proof of a unique runtime branch.
When `arch` is known, ranking prefers arch-tagged sources
(e.g. `*.gfx1201.hip`) over generic siblings; if no arch-specific file
exists, Atlas falls back to the exact generic kernel source rather than
stale docs.

---

## ISA fit view

```bash
.agents/skills/hipfire-kernel-atlas/render-fit.sh \
  --row .codeinsight+research/kernel-atlas/runs/atlas-gfx1201.jsonl \
  --row-index 0 \
  --isa .codeinsight+research/kernel-atlas/runs/isa-gfx1201.json \
  --dispatch .codeinsight+research/kernel-atlas/runs/dispatch-gfx1201.json
```

If the row already references `artifacts.isa.manifest_path` /
`artifacts.dispatch.manifest_path`, those flags are optional. With
`artifacts.profile_kernels`, the view joins hot names to ISA symbols
and prints unmatched hot names when the filter missed runtime kernels.

Do not call the heuristic a roofline or full occupancy model.

---

## Suggest → task → eval

**suggest** — ranked levers (type, hot kernel, risk, impact guess,
allowed files, rationale, eval contract). Auto-loads history under
`.codeinsight+research/kernel-atlas/tasks/`. Current demotion:
`history_rejects_entry()` treats both `pass` and `unstable` results with
`speedup < 1.0` as rejected and demotes the matching suggestion.
**Do not** interpret an `unstable` demotion as correctness or lever
rejection evidence — it is only the current ranker filter, not a stable-only
history gate. `--history PATH` only for extra trees. Suggestions are
**not** perf claims.

```bash
python3 scripts/kernel_atlas.py suggest \
  --row .codeinsight+research/kernel-atlas/runs/atlas-gfx1201.jsonl \
  --row-index 1 \
  --isa .codeinsight+research/kernel-atlas/runs/isa-gfx1201.json \
  --dispatch .codeinsight+research/kernel-atlas/runs/dispatch-gfx1201.json \
  --format markdown
```

**task** — writes `task.json` + `TASK.md`. Pass every
`--allowed-file`. Optional `--correctness-command` must name a
**current** VALIDATION route (path oracle, `test_kernels`, maintained
harness). Do not paste retired coherence-gate paths as the default
acceptance command.

Profiling env on collect rows (`HIPFIRE_PROFILE`,
`HIPFIRE_PROFILE_DECODE`, …) is stripped from generated eval env and
kept as `baseline.row_env`. When stripped, `eval.requires_fresh_baseline`
is true — run `eval --refresh-baseline` before claiming speedup, or
status stays `needs_baseline`.

**eval**

```bash
python3 scripts/kernel_atlas.py eval \
  --task .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/task.json \
  --runs 5 --warmup-runs 1 --refresh-baseline \
  --output-dir .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/eval-baseline

python3 scripts/kernel_atlas.py eval \
  --task .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/task.json \
  --baseline .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/eval-baseline/baseline.json \
  --runs 5 --warmup-runs 1 \
  --output-dir .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/eval-001
```

Writes `result.json` + `ledger.jsonl` (and `baseline.json` on refresh).
If `(max − min) / median` exceeds `--max-rel-spread` (default `0.20`),
status is `unstable`. Treat `unstable` as a **measurement failure**, not a
win/loss. Note: suggest history still demotes `unstable` entries with
`speedup < 1.0` (see suggest above) — that demotion is ranker behavior,
not evidence that the lever was rejected.

**graph-ab** — first-class graph off/on compare from a row (pass hygiene
and route deltas). Correctness command, when supplied, follows the same
VALIDATION rules as `task`.

---

## PyTorch shape tasks

```bash
python3 scripts/kernel_atlas.py task-pytorch \
  --name llama-rmsnorm-shape \
  --op rmsnorm \
  --input-shape 1,2048,4096 \
  --dtype float16 \
  --eval-command 'python3 /path/to/your_rmsnorm_bench.py' \
  --allowed-file kernels/src/rmsnorm.hip \
  --output-dir .codeinsight+research/kernel-atlas/tasks/llama-rmsnorm-shape
```

Does **not** extract kernels from PyTorch. It only reuses the
task/eval/ledger contract for non-Qwen producer shapes. Supply a real
`--eval-command` and any candidate HIP path you own (example allowed file
above is the in-tree `kernels/src/rmsnorm.hip` reference — not a generated
candidate). Eval command is source of truth until a real extractor exists.

---

## Route manifests on rows

Collectors attach `artifacts.runtime_route` via `build_route_manifest`
(kv mode, graph enabled/blob count, inferred attention impl, warnings).
Example: graph-on + Q8 flash routes can warn that logits/coherence must
be verified before trusting perf — still not a substitute for a parity
oracle.

For wall-time vs internal profile blindspots, use
[`rocprof-coverage.md`](rocprof-coverage.md)
(`scripts/rocprof-wrap.sh`, `scripts/coverage-audit.py`).

---

## Multi-host hygiene

On shared or multi-GPU hosts, pin visibility (`ROCR_VISIBLE_DEVICES`)
and/or `HIPFIRE_TARGET_ARCH`. Do not compare rows across hosts unless
prompt md5, binary md5, git diff md5, model identity, and variant env
match. Follow [`perf-benchmarking.md`](perf-benchmarking.md) for
fresh-process / DPM / noise rules when a row will back a perf **claim**.

---

## Validation rule (claims that used Atlas)

Before a kernel or dispatch change is justified by Atlas rows:

1. Re-measure under the perf protocol (identity hashes, stationary GPU).
2. Run the **VALIDATION** route for the claim class (channel, path
   oracle, speed-gate when a baseline exists, serve harness only for
   semantics).
3. Never treat retired `coherence-gate-*.sh` pass/fail as current
   acceptance.
4. Never treat fit-view “left on table” text as a measured speedup.

---

## Related

| Doc / path | Role |
|---|---|
| [`kernel-atlas-architecture.md`](kernel-atlas-architecture.md) | Layers, Rust vs Python, Astrea handoff, migration status |
| [`arch-port-validation.md`](arch-port-validation.md) | GPU/model port evidence tiers |
| [`bench-suite.md`](bench-suite.md) | Bench layout |
| `.agents/skills/hipfire-kernel-atlas/` | Agent workflow + interpretation rules |
| `.agents/skills/hipfire-kernel-tuning/` | Tuning levers after Atlas points at a hot kernel |
