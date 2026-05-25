# Astrea and Atlas Pareto Workflow

Astrea and Atlas are separate tools with one shared job: make quant
quality and runtime performance comparable enough that a human or agent
can choose the next experiment without guessing.

Astrea produces candidate quality rows. Atlas produces runtime
performance rows. A candidate is only decision-ready when both sides
join under the same `candidate_id`.

## Astrea responsibilities

Astrea owns quant-quality evidence and calibration lineage.

Required candidate manifest:

```json
{
  "schema": "hipfire.astrea.candidate.v0",
  "candidate_id": "qwen35-0.8b-mq4-kmd2-full-q8conv1d",
  "model_path": "/home/kaden/.hipfire/models/...",
  "model_hash": "sha256-or-md5",
  "source_model": "Qwen3.5-0.8B",
  "quant_format": "MQ4",
  "calibration_methods": ["kmd2"],
  "promotion_map": "kmd2-full",
  "bpw": 4.5,
  "size_bytes": 0,
  "reference_id": "bf16-ref-id",
  "quality": {
    "kld_mean": null,
    "ppl": null,
    "mse_summary": null
  },
  "artifacts": {
    "imatrix": null,
    "awq": null,
    "gptq": null,
    "policy_map": null
  }
}
```

Astrea should support multiple calibration strategies for the same
source model:

- uncalibrated baseline
- imatrix
- AWQ
- GPTQ
- stacked calibration when mathematically valid
- typed promotion maps such as KMD2

Astrea should not decide default status from KLD alone. It should mark a
candidate as `quality_ready` when the quality data is complete, then
hand it to Atlas for runtime measurement.

## Atlas responsibilities

Atlas owns runtime evidence for every Astrea candidate that is worth
measuring.

Required runtime matrix for decode candidates:

- `HIPFIRE_GRAPH=0`
- `HIPFIRE_GRAPH=1`
- `HIPFIRE_KV_MODE=q8`
- `HIPFIRE_KV_MODE=asym3`

`asym2` and `asym4` are optional unless a candidate specifically targets
those policies. The gfx1100 KMD2 sweep showed both were clearly slower
than `asym3` for this workload.

Required runtime row fields:

- `candidate_id`
- `baseline_candidate_id`
- `arch`
- `hostname`
- `git_sha`
- `binary_md5`
- `model_hash`
- `kv_mode`
- `graph_enabled`
- `graph_blob_count`
- `attention_impl`
- `flash_active`
- `pass_index`
- `discard_first_pass`
- `warmup_tokens`
- `gen_tokens`
- `prefill_tokens`
- `gen_tok_s`
- `prefill_tok_s`
- `avg_ms`
- `p50_ms`
- `p99_ms`
- `bw_gib_s`
- `correctness_status`

## JIT control

Atlas headline numbers should never come from a first run. The default
decode eval is:

```text
pass 1: record, discard from headline
pass 2: record, use for headline
```

The report should say this explicitly. If pass 1 and pass 2 diverge by
more than a configured tolerance, Atlas should keep both and mark the
result `unstable`.

## Route control

Requested environment is not enough. Atlas must record the actual route:

- Q8 short-context default should normally be `attention_q8_0_kv`
  (`q8_nonflash`).
- `asym3` should route through `attention_flash_asym3`.
- Graph capture must not silently switch the attention implementation.

The gfx1100 graph investigation found a correctness bug because capture
was forcing Q8 flash attention at short context. A route manifest would
have shown the invalid graph-off/graph-on route change immediately.

## Decision report

The combined Astrea/Atlas report should print one table per baseline:

```text
candidate        KLD     PPL    bpw   size   runtime       tok/s   delta
flat-mq4         ...     ...    ...   ...    q8+graph      ...     baseline
kmd2-full        ...     ...    ...   ...    q8+graph      ...     -8.0%
kmd2-full        ...     ...    ...   ...    asym3+graph   ...     -5.2%
```

Rows with no correctness result are allowed in exploratory mode but must
be marked `perf_only`. Rows with no Astrea quality result must be marked
`perf_unjoined`.

## Agent workflow

An agent should be able to run this loop:

1. Ask Astrea for candidate manifests.
2. Select candidates marked `quality_ready`.
3. Ask Atlas to run the runtime matrix with JIT control.
4. Ask Atlas to join candidate rows to the flat baseline.
5. Emit a bounded tuning task only when the joined table shows a real
   opportunity.

This keeps mutation grounded in measured quality and measured
performance instead of one-off benchmark output.
