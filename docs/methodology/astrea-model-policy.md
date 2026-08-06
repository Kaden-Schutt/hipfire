# Astrea Model Policy

| Field | Value |
|---|---|
| Member state | **planned / blocked** (see [`docs/INDEX.md`](../INDEX.md) methodology collection exception) |
| Inventory date | 2026-07-19 |
| Audited source ref | `692a726dde53508cb53de1a74c720e75a7c9f33e` |

**This page is proposed intent, not a current owner.** Astrea is intended to own
the evidence and policy layer for model shaping — which weight transforms,
quant-calibration stages, tensor promotions, and KV cache policies are worth
testing — then hand runtime-sensitive candidates to Atlas for AR and DFlash
measurement. That ownership is **not** registered in
[`docs/INDEX.md`](../INDEX.md) or [`docs/VALIDATION.md`](../VALIDATION.md).
Fail closed: do not treat this file as executable product policy or a
validation route.

## Quant quality operations (no dedicated runbook owner)

There is **no** checked-in `quant-quality-tooling.md` (or other INDEX-named
operational runbook) for a current Qwen3.5 MQ4 quality workflow. Do not cite a
missing path as the operational owner.

Until INDEX/VALIDATION name an owner and a real runbook path:

- Treat quality probes, oracle scripts, and candidate hygiene as **planned** or
  ad-hoc engineering notes.
- Prefer shipped quant authority in [`docs/QUANTIZATION.md`](../QUANTIZATION.md)
  and [`docs/QUANTIZE.md`](../QUANTIZE.md).
- Prefer Atlas observation practice in [`kernel-atlas.md`](kernel-atlas.md).
- Compose any Astrea↔Atlas handoff manually per
  [`kernel-atlas-architecture.md`](kernel-atlas-architecture.md) and
  [`astrea-atlas-pareto-workflow.md`](astrea-atlas-pareto-workflow.md) (both
  mark `candidate_id` join as planned/blocked).

This file remains a **higher-level policy boundary sketch** only.

## Proposed scope (planned)

Intended Astrea scope when implemented and owned:

- Weight calibration: AWQ, imatrix-scale, GPTQ probes, k-map/promotion, MSE,
  percentile, minmax, FWHT/QuaRot-style transform lanes, and ParoQuant-style
  transform planning.
- Dynamic tensor policy: rank tensors by quality sensitivity per added byte and
  emit mixed-format recipes under a size budget.
- MoE ingress: separate router, expert, and shared dense tensors before
  optimizing a MoE model family.
- KV policy: compare current `asym3` against `q8`, TriAttention/CASK,
  TurboQuant-like, and RotorQuant-like candidates using an explicit policy
  artifact.
- Package planning: describe a future single-file HFQ package containing
  weights, transform metadata, KV policy, and embedded TriAttention/CASK
  centers.

## Deliberate boundary (still accurate as intent)

Astrea does not currently rewrite the model package format, mutate runtime
loaders, or prove a KV policy works at decode time. Proposed `kv-profile` and
`bundle-plan` surfaces would produce contracts for follow-up implementation and
measurement — not admissions.

Runtime/package work stays deferred until policy artifacts identify a candidate
worth carrying:

- HFQ package header and section table for `transform.paro`, `kv.policy`,
  `triattn.centers`, and evidence metadata.
- Loader-side validation and rejection of unsupported sections.
- Daemon and CLI preference for embedded TriAttention/CASK data over external
  sidecar paths.
- Kernels or decode paths for any non-existing KV policy, especially
  TurboQuant-like and RotorQuant-like candidates.
- Atlas joins for AR and DFlash perf, memory, and correctness rows under a
  shared id (**planned** — no `candidate_id` in Atlas schema today).

## Deferred PyTorch oracle lane (planned)

Astrea may eventually expose a first-class `oracle` command that runs the
hipfire hidden-state dumper plus a PyTorch/HF reference forward, records engine
fingerprints, prompt md5s, token ids, layerwise hidden drift, final-norm drift,
and logits drift, then classifies failures such as boundary mismatches, early
layer cliffs, smooth quant drift, or logits recovery.

This remains deliberately deferred. Standalone PyTorch oracle scripts may be
used directly as proof-of-concept debuggers for engine correctness and
quant-format bring-up. Promote a stable artifact shape into Astrea only after
that loop has found and fixed real mismatches **and** ownership is named.

## Proposed loop (planned / blocked)

Do not run this as a required current product workflow:

1. Use `astrea inspect` and `astrea fingerprint` to capture the model and engine.
2. Use `astrea policy --domain weights --domain kv` to rank weight and KV work.
3. Use `astrea kv-profile` to materialize the KV candidate set.
4. Use `astrea bundle-plan` to describe how the candidate would live inside the
   model artifact once loader support exists.
5. Use Astrea eval/metrics for KLD, PPL, MSE, and recovered above-floor KLD.
6. Use Atlas to validate AR and DFlash perf before any promotion claim.

ParoQuant should be treated as a high-priority **transform experiment lane**,
not a reason to invent runtime contracts. The first implementation target is
evidence: show whether Paro-style transforms improve MQ/HFQ/HFP/MFP quality
enough to justify new producer-consumer runtime contracts — still subject to
VALIDATION and fail-closed [`admissions.yml`](../admissions.yml) (schema v2; no inferred rows).

## Related

| Concern | Disposition |
|---|---|
| Pareto join sketch | [`astrea-atlas-pareto-workflow.md`](astrea-atlas-pareto-workflow.md) (**planned / blocked**) |
| Atlas architecture / join status | [`kernel-atlas-architecture.md`](kernel-atlas-architecture.md) |
| Quant formats | [`docs/QUANTIZATION.md`](../QUANTIZATION.md) |
| Validation routes | [`docs/VALIDATION.md`](../VALIDATION.md) |
