# Kernel Atlas - Python-first architecture

Kernel Atlas is hipfire's measurement corpus for kernel and runtime
performance. Astrea is the matching corpus for quantization quality.
The two tools should compose around a shared experiment key:

```text
git_sha + model_hash + workload + quant_variant + runtime_variant
```

Astrea answers "is this quant worth running?" Atlas answers "what did
it cost or buy on the target hardware?"

## Rust Atlas status

`origin/master` currently contains `crates/hipfire-atlas`. Treat that
crate and CLI as transitional and revert-bound. Do not expand the Rust
Atlas port into the long-term user or agent surface.

The useful concepts from the Rust crate should survive the revert:

- stable JSONL rows
- task/eval bundles
- schema validation
- render/suggest commands
- in-process bench emission when a Rust binary already owns the metric

The long-term Atlas tool should be Python-first because the dominant
workflow is analysis: loading many rows, grouping by quant/runtime/arch,
rendering Pareto tables, and rapidly changing ranking heuristics. Rust
is still appropriate inside hipfire binaries when they emit metrics they
already know, but the Atlas CLI, analyzer, agent workflow, and notebook
surface should live in Python.

## Layers

### Layer 1 - Measurement emission

Bench and inference binaries may emit Atlas JSONL directly through
`--emit-atlas <path>`. The binary that owns a metric should write that
metric rather than relying on a stdout parser. This is especially
important for route information that is not visible from a summary line.

Required row fields for new measurements:

- `schema`: `hipfire.kernel_atlas.v0`
- `phase`: `prefill`, `decode_ar`, or `decode_dflash`
- `workload_kind`: binary or workload class
- `git_sha`, `hostname`, `arch`, `rocm_version`, `hipcc_version`
- `model_path`, `model_hash`, `model_bytes`
- `quant_variant`: flat MQ4, KMD2-lite, KMD2-full, etc.
- `runtime_variant`: graph/KV/flash/config tuple

The row should also include enough run hygiene to keep small wins honest:

- `pass_index`
- `discard_first_pass`
- `warmup_tokens`
- `gen_tokens`
- `prefill_tokens`
- `dpm_warmup_secs`
- `binary_md5`
- `prompt_md5` when a prompt is involved

### Layer 2 - Python Atlas CLI and analyzer

The Python Atlas surface owns orchestration and analysis:

- run bench matrices
- discard pass 1 by default for JIT control
- join candidate rows to baseline rows
- render flat text tables for agents and humans
- emit constrained tuning tasks
- record eval results, diffs, correctness status, and lineage

The graph/KV investigation added one hard requirement: Atlas must record
the actual dispatch route, not just the requested environment.

Route manifest fields:

- `kv_mode`: `q8`, `asym2`, `asym3`, `asym4`, etc.
- `graph_enabled`
- `graph_blob_count`
- `attention_impl`: `q8_nonflash`, `q8_flash`, `asym2_flash`,
  `asym3_flash`, `asym4_flash`, etc.
- `flash_requested`: `never`, `auto`, or `always`
- `flash_active`
- `kernel_names`
- `grid`, `block`, and `shared_mem` for hot kernels when available
- `capture_safe`: true/false/unknown

This would have surfaced the Q8 graph failure directly: capture was
forcing Q8 flash attention at short context, and forced Q8 flash
reproduced NaN logits even with graph disabled. Atlas should flag any
case where graph capture changes the attention route.

### Layer 3 - Advisor

The advisor consumes the corpus and suggests bounded experiments:

- launch-bound retunes
- K-unroll changes
- graph capture enablement
- flash/non-flash dispatch thresholds
- KV policy choices
- candidate vs baseline regressions

Build this after the corpus has enough rows. Keep the first version as a
Python ranker plus task emitter. Do not jump to autonomous mutation until
`atlas task` and `atlas eval` contracts are boring and reliable.

## Required Atlas eval modes

### Graph A/B

Atlas should provide a first-class graph comparison that runs:

```text
HIPFIRE_GRAPH=0 pass 1
HIPFIRE_GRAPH=0 pass 2
HIPFIRE_GRAPH=1 pass 1
HIPFIRE_GRAPH=1 pass 2
```

Pass 1 is recorded but not used for the headline. Pass 2 is the
JIT-controlled row. The report must show:

- graph-off tok/s
- graph-on tok/s
- graph lift
- p50/p99 delta
- prefill delta
- route changes between graph off and graph on
- correctness status for graph-on

### Baseline/candidate compare

Atlas should make "perf lost vs flat MQ4" automatic. A candidate report
must accept a baseline row and print:

- candidate tok/s delta
- candidate p50/p99 delta
- candidate prefill delta
- candidate model-size delta
- quality row link from Astrea
- correctness/coherence status

For example, the KMD2 investigation should be one Atlas comparison:

```text
baseline = flat MQ4 + q8conv1d
candidate = full KMD2 + q8conv1d
runtime matrix = q8/asym3 x graph off/on
```

### Correctness join

A performance row without a correctness row is incomplete. Atlas should
join to:

- `coherence_probe`
- `coherence-gate-dflash.sh` for DFlash claims
- KLD/PPL/MSE rows from Astrea when the experiment changes quantization

The report should mark rows as `perf_only` when correctness is missing.

## Relationship to Astrea

Astrea owns quality evidence:

- KLD vs BF16 reference
- PPL
- per-tensor and per-layer attribution
- MSE and reconstruction stats
- calibration method inputs: none, imatrix, AWQ, GPTQ, stacked methods
- promotion maps and bpw
- PyTorch oracle traces when available

Astrea hands candidates to Atlas only after producing a candidate
manifest. Minimum manifest fields:

- `candidate_id`
- `model_path`
- `model_hash`
- `source_model`
- `quant_format`
- `calibration_methods`
- `promotion_map`
- `bpw`
- `size_bytes`
- `kld_mean`
- `ppl`
- `mse_summary`
- `reference_id`
- `quality_artifacts`

Atlas appends runtime rows under the same `candidate_id`. The final
decision surface is a quality/performance Pareto table, not separate
quality and speed reports.

## Migration plan

| Step | Change | Status |
|---|---|---|
| 1 | Stop expanding `crates/hipfire-atlas`; mark it transitional/revert-bound | active |
| 2 | Recreate the Atlas CLI surface in Python | TODO |
| 3 | Port useful Rust CLI commands to Python: `read`, `head`, `render-fit`, `suggest`, `task`, `eval` | TODO |
| 4 | Add route manifests and graph A/B to Python Atlas | TODO |
| 5 | Add baseline/candidate comparison and correctness join | TODO |
| 6 | Add Astrea candidate manifests and Atlas handoff rows | TODO |
| 7 | Remove the Rust Atlas crate once Python has parity | TODO |

## Open follow-ups from gfx1100 KMD2 work

- Add route-manifest capture so graph cannot silently change attention
  implementation.
- Add graph A/B as a one-command Atlas eval.
- Add pass-2/JIT-controlled headline reporting.
- Add KMD2-vs-flat comparison as a built-in report.
- Keep `asym3 + graph` and `q8 + graph` as the first runtime variants
  to track for Qwen3.5 0.8B KMD2 on gfx1100.
