---
name: hipfire-eval-harness
description: Use when running or interpreting the unified Hipfire eval harness, choosing hipfire eval tiers/batteries/suites, validating quantized model candidates, collecting admission evidence, or deciding whether fast/medium/long/extensive eval results are sufficient for a quantization claim.
---

# Hipfire Eval Harness

Use this skill when the user asks to evaluate a Hipfire model, compare quantized
candidates, admit/reject a quant, run `hipfire eval`/`hipfire-eval`, inspect eval
reports, or decide which tier/battery/suite is appropriate.

## Core Rule

Treat `hipfire eval` as the unified evidence ledger for quantization work, not
as just another benchmark script. Do not claim a candidate is better, promoted,
or ship-ready unless the report has the relevant quality, performance, runtime,
and provenance evidence for that claim.

## Output Locations

Default report roots:

```bash
~/.hipfire/eval-results/runs/   # normal hipfire-eval runs
~/.hipfire/eval-results/smoke/  # harness smoke scripts
~/.hipfire/eval-results/cache/  # reusable per-battery result cache
```

Each run directory should contain:

```text
manifest.json
results.jsonl
summary.md
artifacts/
```

Use `--out <dir>` only when the user explicitly wants a custom location or a
test needs a controlled path. Prefer the default cache root over writing under
the source tree.

## Result Cache

The harness writes aggregate reports per run, but expensive model-backed rows
are cached under `~/.hipfire/eval-results/cache/`. Cache keys include the
battery, model/draft/baseline/reference hashes, prompt fixture hashes for
static batteries, Hipfire version, git commit/branch/dirty state, binary hash,
ROCm version, GPU arch, executor, KV mode, max tokens, DFlash/profile modes,
and relevant dataset or evidence inputs.

Use cache controls deliberately:

- `--force`: ignore cache hits for this run, but write replacement entries.
- `--regenerate`: delete matching cache entries before running and write new ones.
- `--no-cache`: disable cache reads and writes.
- `--result-cache <dir>`: use a custom cache root for tests or isolation.

Rows reused from cache carry `metrics.cache_hit=true`, plus `cache_key` and
`cache_path`. Treat cached evidence as valid only when its key metadata matches
the claim being made; use `--regenerate` after changing prompts, runtime
behavior, or evidence collection semantics.

## Benchmark Repeats

Default evals use one scored pass per battery. Use repeat controls when the
claim depends on performance stability:

- `--runs <N>`: collect N scored samples for each selected battery.
- `--warmup-runs <N>`: run and discard N warmup passes before scored samples.
- `--benchmark`: shorthand for `--runs 5` unless `--runs` is explicitly set.

Repeated runs keep raw JSONL rows with `metrics.benchmark_sample=true`,
`run_index`, and `run_count`. The harness also emits `::aggregate` rows with
median/mean/stddev/min/max/p10/p90/cv_pct metrics. Use aggregate medians for
regression checks; inspect raw rows when stddev or cv_pct is high.

For performance admission or regression evidence, prefer:

```bash
hipfire eval --model candidate.hfq --battery speed --benchmark --force
```

Use `--force` or `--regenerate` for benchmark reruns so stale cached evidence
does not hide runtime changes.

## Passive Profiling

`--profile off` is the default. `--profile passive` collects profiling evidence
after scored rows are collected. Profiling is evidence-only and must not affect
pass/fail scoring.

When `rocprofv3` is available, passive profiling runs a representative speed
anchor under rocprof and stores raw files under the run's `artifacts/rocprof/`.
The normalized records are ingested into `artifacts/profiling.json`. If rocprof
or an examples executor is unavailable, the harness records an explicit skip
instead of failing the eval.

## Tier Selection

- `fast`: CI-suitable small-model smoke and admission canaries. Target: under 60s.
- `medium`: bounded model-eval subset for local validation. Target: under 5 min.
- `long`: broader quality and long-context coverage. Target: under 20 min.
- `extensive`: full native/evidence suite with no fixed wall-clock target.

Use the smallest tier that can prove the decision:

- For PR/no-GPU integrity: `--executor mock` plus no-GPU smoke.
- For a local candidate sanity check: `--tier fast` on a small model.
- For quant admission: at least candidate-vs-baseline quality and speed evidence.
- For DFlash claims: AR anchor must pass before DFlash rows are accepted.
- For MoE claims: include or ingest MoE router histogram evidence.

## Common Commands

Build the runner and examples:

```bash
cargo build -p hipfire-runtime --bin hipfire-eval --example run --example dflash_spec_demo
```

No-GPU harness integrity:

```bash
HIPFIRE_EVAL_BIN=target/debug/hipfire-eval bash scripts/smoke/eval-harness-nogpu-smoke.sh
```

Fast local eval:

```bash
hipfire eval --model ~/.hipfire/models/qwen3.5-0.8b.mq4 --tier fast
```

Regenerate cached rows for a model:

```bash
hipfire eval --model ~/.hipfire/models/qwen3.5-0.8b.mq4 --tier fast --regenerate
```

Candidate-vs-baseline admission:

```bash
hipfire eval \
  --model candidate.hfq \
  --baseline baseline.hfq \
  --tier medium \
  --battery quality,speed,barrage \
  --quality-json result-data.json \
  --performance-json perf.json \
  --evidence-dir runtime-evidence \
  --fail-on-admission
```

Opt-in Hugging Face dataset fetch:

```bash
hipfire eval --model candidate.hfq --battery barrage --suite gpqa --fetch-datasets
```

Offline/cached dataset run:

```bash
hipfire eval --model candidate.hfq --battery barrage --suite gpqa --offline
```

Do not pass `--fetch-datasets` and `--offline` together.

## Report Audit Checklist

Before using a report as evidence, inspect:

- `manifest.json`: runner version, Hipfire version, git commit/branch/dirty,
  binary hash, arch, ROCm, model/draft/baseline/reference hashes, datasets.
- `results.jsonl`: one row per subcase with status, prompt hash where applicable,
  model hash, timing, KV mode, and metrics.
- `summary.md`: human-readable Models, Datasets, Comparisons, Admission,
  Evidence Artifacts, and Rows sections.
- `artifacts/admission.json`: verdict and findings.
- `artifacts/comparisons.json`: candidate deltas against baseline/reference.
- `artifacts/quality.json`: KLD/PPL/barrage accuracy evidence.
- `artifacts/performance.json`: tok/s, TTFT, prefill/decode metrics.
- `artifacts/dflash_trace.json`: AR and DFlash trace records.
- `artifacts/launch_counts.json`: kernel/graph/memcpy launch evidence.
- `artifacts/moe_router_histogram.json`: expert-hit/router evidence for MoE.
- `artifacts/profiling.json`: passive profiling evidence when requested.

Treat skipped evidence as explicit information, not as a pass. If required
evidence is skipped or missing, the admission should not be promoted.

## Validation Gates

Run the smallest gate that matches the change:

```bash
cargo test -p hipfire-runtime eval_harness --lib
bun test cli/eval.test.ts
bash -n scripts/smoke/eval-harness-nogpu-smoke.sh scripts/smoke/eval-harness-gpu-smoke.sh scripts/smoke/eval-harness-model-eval-smoke.sh
```

For full no-GPU coverage:

```bash
./scripts/no-gpu-ci.sh
```

For real GPU validation:

```bash
HIPFIRE_EVAL_BIN=target/debug/hipfire-eval \
HIPFIRE_EVAL_SMOKE_MODEL=~/.hipfire/models/qwen3.5-0.8b.mq4 \
scripts/smoke/eval-harness-gpu-smoke.sh
```

For candidate-vs-baseline smoke:

```bash
HIPFIRE_EVAL_BIN=target/debug/hipfire-eval \
HIPFIRE_EVAL_MODEL_SMOKE_CANDIDATE=~/.hipfire/models/qwen3.5-0.8b.mq4 \
HIPFIRE_EVAL_MODEL_SMOKE_BASELINE=~/.hipfire/models/qwen3.5-0.8b.mq6 \
scripts/smoke/eval-harness-model-eval-smoke.sh
```

For paired DFlash validation, provide a real draft:

```bash
HIPFIRE_EVAL_BIN=target/debug/hipfire-eval \
HIPFIRE_EVAL_SMOKE_GROUP=dflash \
HIPFIRE_EVAL_SMOKE_MODEL=~/.hipfire/models/qwen3.5-27b.mq4 \
HIPFIRE_EVAL_SMOKE_DRAFT=~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
scripts/smoke/eval-harness-gpu-smoke.sh
```

## Claim Discipline

- Quality claim: cite exact report path, model hashes, prompt/dataset hashes,
  quality artifact, and comparison/admission artifact.
- Performance claim: include binary hash, git commit, GPU arch, ROCm, KV mode,
  prompt hash, and performance artifact.
- DFlash claim: include AR anchor, DFlash anchor, draft hash, and DFlash trace.
- MoE claim: include router histogram evidence and explain expert-hit coverage.
- Profiling claim: profiling is evidence only; it must not affect pass/fail.

If evidence was generated with mock executor or generated smoke fixtures, call it
an integrity check only, not a quality/performance result.
