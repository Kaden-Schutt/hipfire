# Astrea and Atlas Pareto Workflow

| Field | Value |
|---|---|
| Member state | **planned / blocked** (see [`docs/INDEX.md`](../INDEX.md) methodology collection exception) |
| Inventory date | 2026-07-19 |
| Audited source ref | `692a726dde53508cb53de1a74c720e75a7c9f33e` |

**This page is proposed intent, not a current owner.** Astrea is **not**
registered as a current owner in [`docs/INDEX.md`](../INDEX.md) or
[`docs/VALIDATION.md`](../VALIDATION.md). Under fail-closed rules, do not treat
the contracts below as executable authority, required manifests, or a
decision-ready product workflow until those owners name Astrea and first-class
`candidate_id` support ships in Atlas (see
[`kernel-atlas-architecture.md`](kernel-atlas-architecture.md) — neither
`scripts/kernel_atlas.py` nor `crates/hipfire-atlas` defines or accepts
`candidate_id` today).

## Proposed shared job (planned)

Astrea and Atlas are intended as separate tools with one shared job: make quant
quality and runtime performance comparable enough that a human or agent can
choose the next experiment without guessing.

- **Planned:** Astrea produces candidate quality rows.
- **Shipped / ref-pinned (Atlas side only):** Atlas produces runtime performance
  observation rows under [`kernel-atlas.md`](kernel-atlas.md).
- **Planned / blocked join:** A candidate would be decision-ready only when both
  sides join under the same `candidate_id`. That join is **manual / planned**
  until schema and ownership land. Incomplete joins stay labeled incomplete
  (`perf_only` / `perf_unjoined`).

Neither side admits product defaults; admissions stay in
[`docs/admissions.yml`](../admissions.yml) (fail closed).

## Proposed Astrea responsibilities (planned)

**Blocked as current ownership.** The following describes intended scope once
INDEX/VALIDATION name an owner.

Proposed candidate manifest shape (not a shipped Atlas or Astrea schema field
set):

```json
{
  "schema": "hipfire.astrea.candidate.v0",
  "candidate_id": "qwen35-0.8b-mq4-kmd2-full-q8conv1d",
  "model_path": "/path/to/model",
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

Intended calibration strategies for the same source model (when implemented):

- uncalibrated baseline
- imatrix
- AWQ
- GPTQ
- stacked calibration when mathematically valid
- typed promotion maps such as KMD2

Default status must not be decided from KLD alone. A future `quality_ready`
mark would mean quality data is complete enough to hand to Atlas — still not
admission.

## Proposed Atlas responsibilities for joined candidates (planned)

Atlas already owns runtime observation collection for workloads it can run.
Extending that to **every Astrea candidate** under a required `candidate_id`
matrix is **planned**.

Proposed runtime matrix axes for decode candidates (when the join exists):

- `HIPFIRE_GRAPH=0`
- `HIPFIRE_GRAPH=1`
- `HIPFIRE_KV_MODE=q8`
- `HIPFIRE_KV_MODE=asym3`

`asym2` and `asym4` would remain optional unless a candidate specifically
targets those policies. Historical gfx1100 KMD2 notes are not current floors.

Proposed runtime row fields for a joined candidate (beyond ordinary Atlas
observation fields) include `candidate_id`, `baseline_candidate_id`, and the
usual identity/route/metric set (`arch`, `hostname`, `git_sha`, `binary_md5`,
`model_hash`, `kv_mode`, `graph_enabled`, route fields, pass hygiene, tok/s and
latency metrics, `correctness_status`). Until `candidate_id` exists in schema,
compose externally and label the join incomplete.

## JIT control (Atlas practice when measuring)

Atlas headline numbers should never come from a first run. The default decode
eval pattern remains:

```text
pass 1: record, discard from headline
pass 2: record, use for headline
```

If pass 1 and pass 2 diverge by more than a configured tolerance, keep both and
mark the result `unstable`. This is measurement hygiene, not Astrea ownership.

## Route control

Requested environment is not enough. Atlas must record the actual route:

- Q8 short-context default should normally be `attention_q8_0_kv`
  (`q8_nonflash`).
- `asym3` should route through `attention_flash_asym3`.
- Graph capture must not silently switch the attention implementation.

Historical gfx1100 graph notes (capture forcing Q8 flash at short context) are
disposition-only motivation for route manifests — not a live certification.

## Proposed decision report (planned)

A future combined Astrea/Atlas report might print one table per baseline:

```text
candidate        KLD     PPL    bpw   size   runtime       tok/s   delta
flat-mq4         ...     ...    ...   ...    q8+graph      ...     baseline
kmd2-full        ...     ...    ...   ...    q8+graph      ...     -8.0%
kmd2-full        ...     ...    ...   ...    asym3+graph   ...     -5.2%
```

Rows with no correctness result are allowed in exploratory mode but must be
marked `perf_only`. Rows with no Astrea quality result must be marked
`perf_unjoined`. No such table is a current validation route or admission path.

## Proposed agent loop (planned / blocked)

An agent **should not** treat the following as a required current workflow:

1. Ask Astrea for candidate manifests.
2. Select candidates marked `quality_ready`.
3. Ask Atlas to run the runtime matrix with JIT control.
4. Ask Atlas to join candidate rows to the flat baseline.
5. Emit a bounded tuning task only when the joined table shows a real
   opportunity.

Until ownership and `candidate_id` support ship, keep quality evidence and
Atlas observations separate; join manually only with explicit incomplete labels.
Mutation stays grounded in whatever INDEX/VALIDATION currently own — not this
page.

## Related

| Concern | Owner / disposition |
|---|---|
| Current Atlas usage | [`kernel-atlas.md`](kernel-atlas.md) (**shipped / ref-pinned**) |
| Atlas layers / Astrea join status | [`kernel-atlas-architecture.md`](kernel-atlas-architecture.md) |
| Astrea policy boundary (also planned) | [`astrea-model-policy.md`](astrea-model-policy.md) |
| Validation routes | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Admissions | [`docs/admissions.yml`](../admissions.yml) |
