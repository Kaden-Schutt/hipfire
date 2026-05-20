# Astrea Model Policy

Astrea owns the evidence and policy layer for model shaping. It should decide
which weight transforms, quant-calibration stages, tensor promotions, and KV
cache policies are worth testing, then hand runtime-sensitive candidates to
Atlas for AR and DFlash measurement.

## Quant Quality Runbook

For the current Qwen3.5 MQ4 quality workflow, command examples, PyTorch oracle
probes, accepted/rejected candidate evidence, and artifact hygiene rules, see
[`quant-quality-tooling.md`](quant-quality-tooling.md). That document is the
operational runbook; this file remains the higher-level policy boundary for
Astrea.

## Current Scope

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

## Deliberate Boundary

Astrea does not currently rewrite the model package format, mutate runtime
loaders, or prove a KV policy works at decode time. `kv-profile` and
`bundle-plan` produce contracts for follow-up implementation and measurement.

Runtime/package work is deferred until the policy artifacts identify a
candidate worth carrying:

- HFQ package header and section table for `transform.paro`, `kv.policy`,
  `triattn.centers`, and evidence metadata.
- Loader-side validation and rejection of unsupported sections.
- Daemon and CLI preference for embedded TriAttention/CASK data over external
  sidecar paths.
- Kernels or decode paths for any non-existing KV policy, especially
  TurboQuant-like and RotorQuant-like candidates.
- Atlas joins for AR and DFlash perf, memory, and correctness rows.


## Deferred PyTorch Oracle Lane

Astrea should eventually expose a first-class `oracle` command that runs the
hipfire hidden-state dumper plus a PyTorch/HF reference forward, records engine
fingerprints, prompt md5s, token ids, layerwise hidden drift, final-norm drift,
and logits drift, then classifies failures such as boundary mismatches, early
layer cliffs, smooth quant drift, or logits recovery.

This is deliberately deferred for now. The current priority is to use the
standalone PyTorch oracle scripts directly as a proof-of-concept debugger for
hipfire engine correctness and quant-format bring-up. Once the standalone loop
has found and fixed real mismatches, promote the stable artifact shape into
Astrea.

## Recommended Loop

1. Use `astrea inspect` and `astrea fingerprint` to capture the model and engine.
2. Use `astrea policy --domain weights --domain kv` to rank weight and KV work.
3. Use `astrea kv-profile` to materialize the KV candidate set.
4. Use `astrea bundle-plan` to describe how the candidate would live inside the
   model artifact once loader support exists.
5. Use Astrea eval/metrics for KLD, PPL, MSE, and recovered above-floor KLD.
6. Use Atlas to validate AR and DFlash perf before any promotion claim.

ParoQuant should be treated as a high-priority transform lane, not a reason to
throw away the current Astrea path. The first implementation target is evidence:
show whether Paro-style transforms improve MQ/HFQ/HFP/MFP quality enough to
justify new producer-consumer runtime contracts.
