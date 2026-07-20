---
name: astrea
description: Use for hipfire quant calibration, imatrix-driven experiments, KLD/PPL quality evaluation, k-map/format selection, MQ/HFQ/HFP/MFP tradeoff work, ParoQuant-style weight transform planning, and KV policy planning. Use when deciding whether a calibrated model candidate should be promoted, rejected, packaged, or sent through Atlas for AR/DFlash perf validation.
---

# Astrea

Agent-native calibration harness for hipfire quant candidates. The executable
is `python3 scripts/astrea.py`. It emits JSON plan/policy/metrics artifacts;
you supply experiment judgment and promotion discipline.

Mutable format math, operator quantize flags, KV defaults, and validation
routes live in canonical owners — link them; do not copy inventories here.

## Reach for this when

- Calibrating or comparing weight formats (MQ / HFQ / HFP / MFP / Q8 / F16 / Paro)
- Joining GGUF imatrix coverage to HFQ tensor names
- Building mixed-format promotion policies under a size budget
- Planning KV-cache policy profiles for Atlas join
- Importing/oracling ParoQuant (`PARO4G128`) checkpoints
- Deciding promote / reject / iterate from measured KLD/PPL + runtime evidence

## Canonical owners (read these; do not fork)

| Concern | Owner |
|---|---|
| Quant design / formats / KV math | [`docs/QUANTIZATION.md`](../../../docs/QUANTIZATION.md) |
| `hipfire quantize` operator surface | [`docs/QUANTIZE.md`](../../../docs/QUANTIZE.md) |
| Validation route selection | [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) |
| Docs ownership / truth states | [`docs/INDEX.md`](../../../docs/INDEX.md) |
| Perf measurement protocol | [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md) |
| Kernel Atlas methodology | [`docs/methodology/kernel-atlas.md`](../../../docs/methodology/kernel-atlas.md) |
| Atlas agent workflow | [`.agents/skills/hipfire-kernel-atlas/`](../hipfire-kernel-atlas/SKILL.md) |
| Admissions registry | [`docs/admissions.yml`](../../../docs/admissions.yml) (empty = fail closed) |

Executable truth for CLI flags: `python3 scripts/astrea.py <cmd> --help`.

## Core rules

1. **No quality claim without measured KLD/PPL** (or an explicit blocked eval
   path). Dry-run Astrea JSON is a plan, not calibrated weights.
2. **No ship-ready claim without runtime compatibility + perf evidence** on
   the formats the candidate actually uses. Prefer Atlas AR and, when the
   path is DFlash-relevant, DFlash rows — via the Atlas skill/methodology.
3. **`bundle-plan` is a package contract artifact, not a model writer.** Do
   not claim Astrea packaged a loadable model unless a real writer path ran
   (`promote`, `calibrate --write-candidate`, or `paro-import`).
4. **Do not invent gates.** Pick routes from `docs/VALIDATION.md`. There is
   no universal GPU/coherence replacement gate. Retired
   `scripts/coherence-gate-*.sh` batteries are historical reproduction only.
5. **Do not compare KLD/PPL across different engine fingerprints or RoPE
   conventions** without marking the comparison historical.
6. **Non-finite logits fail the candidate.** Reject `KLD=0` rows unless the
   logit path is confirmed finite.
7. **Admissions stay machine-recorded only** in `docs/admissions.yml`. A green
   eval or bench does not create an admission.

## CLI surface

Run from the hipfire repo root. Prefer `--pretty` for humans; `--out PATH`
writes JSON and silences stdout.

```bash
python3 scripts/astrea.py inspect --model MODEL [--imatrix IMATRIX] [--format FORMAT]
python3 scripts/astrea.py imatrix-join --model MODEL --imatrix IMATRIX [--max-tensors N]
python3 scripts/astrea.py fingerprint [--engine-root REPO]
python3 scripts/astrea.py plan --model MODEL --format FORMAT --method METHOD \
  [--recipe-stage STAGE:METHOD] [--imatrix IMATRIX] [--source-dir BF16_DIR] \
  [--eval-command CMD] [--atlas-command CMD]
python3 scripts/astrea.py calibrate --plan PLAN.json [--source-dir BF16_DIR] \
  [--write-candidate] [--max-tensors N] [--tensor-filter NAME] [--workers N] [--dry-run]
python3 scripts/astrea.py eval --plan PLAN.json [--run]
python3 scripts/astrea.py metrics --quality-json result-data.json \
  --candidate-variant NAME [--baseline-variant NAME] [--floor-variant NAME] \
  [--arch ARCH] [--scoring-mode MODE] [--engine-root REPO]
python3 scripts/astrea.py policy --model MODEL --base-format FMT --promotion-format FMT \
  (--sensitivity-json SCORES.json | --imatrix IMATRIX) --max-extra-bytes N \
  [--method METHOD] [--objective OBJ] [--domain weights|kv] [--model-family FAMILY]
python3 scripts/astrea.py promote --policy POLICY.json --source-dir BF16_DIR --output CANDIDATE.hfq \
  [--max-tensors N] [--tensor-filter NAME]
python3 scripts/astrea.py kv-profile --model MODEL [--mode MODE] [--triattn PATH] \
  [--model-family FAMILY] [--engine-root REPO]
python3 scripts/astrea.py bundle-plan --model MODEL --output MODEL.hfq \
  [--include weights|paro|kv-policy|triattn|evidence] [--triattn PATH] [--policy-id ID]
python3 scripts/astrea.py paro-probe --model MODEL [--local-only] [--max-modules N]
python3 scripts/astrea.py paro-import --model MODEL --output OUT.hfq \
  [--layout native|engine] [--copy-floats f16|q8] [--local-only]
python3 scripts/astrea.py paro-oracle --source PARO_SAFE_DIR --hfq MODEL.hfq \
  [--module MODULE] [--samples N] [--atol TOL]
python3 scripts/astrea.py report ARTIFACT.json ...
```

Supported format/method/KV-mode tokens are defined in `scripts/astrea.py`
(`SUPPORTED_FORMATS`, `SUPPORTED_METHODS`, `SUPPORTED_KV_MODES`, …). Query the
script or `--help` rather than trusting a skill-local table.

## Workflow

1. **Scope** — model path, target format(s), BF16/higher-precision reference,
   eval corpus, size budget, and whether the change is weights, KV, or both.
2. **`inspect` / `imatrix-join`** — fingerprint inputs; confirm imatrix↔HFQ
   coverage before planning mutation.
3. **`fingerprint`** — capture engine git/source hashes and RoPE path
   (`halfsplit` / `interleaved_legacy` / `unknown`). Pass `--engine-root` when
   the evaluated binary is not this checkout.
4. **`plan`** — bounded experiment artifact. Methods stack as recipe stages
   (`--method` repeated; optional `--recipe-stage stage:method`). Attach
   `--eval-command` and `--atlas-command` so later steps are reproducible.
5. **`calibrate`** — without `--write-candidate`, reports join readiness only.
   With `--write-candidate`, may rewrite same-size tensor ranges for supported
   recipes (e.g. MFP4 imatrix-scale, MQ4 AWQ-style clipping, MQ4 LS when
   methods match). Use `--max-tensors` / `--tensor-filter` for smokes;
   `--workers N` for large rewrites. Treat unimplemented mutation paths as
   blocked, not as silent success.
6. **`eval` then `metrics`** — run KLD/PPL against a fixed reference; ingest
   `kld_reduce.py` `result-data.json` (or equivalent). Prefer a Q8 / accepted
   high-precision floor row so above-floor KLD and recovered damage % are
   meaningful. Attach exact command, reference, dataset/chunk count, and
   artifact identity to every quality claim.
7. **`policy` / `promote`** — optional mixed-format promotion under
   `--max-extra-bytes`. Objectives include `dynamic-tensor-policy`,
   `moe-probe`, `model-ingress`, `kv-policy`. Policy is a selector, not quality
   evidence — re-run `metrics` after every written candidate. Today `promote`
   writes selected `q8` promotions as runtime-compatible `Q8F16` records and
   rebuilds the HFQ index/data payload.
8. **`kv-profile`** — when KV behavior changes or a model should carry an
   embedded KV policy. Include a live baseline mode and each candidate mode
   under investigation. Output is a policy/evidence shape for Atlas join, not
   proof that kernels/loader implement the mode.
9. **Paro lane** — `paro-probe` → `paro-import` (writes loadable `PARO4G128`)
   → `paro-oracle` against the PyTorch source before quality or perf claims.
   See `docs/QUANTIZATION.md` Paro payload contract.
10. **`bundle-plan`** — future single-file package shape (weights, paro, kv-policy,
    triattn, evidence). Deferred loader/daemon work stays deferred until source
    implements it.
11. **Atlas perf** — if quality improves and the candidate touches runtime
    formats on AR and/or DFlash paths, collect phase-aware rows per
    `.agents/skills/hipfire-kernel-atlas/` and
    `docs/methodology/kernel-atlas.md`. Follow
    `docs/methodology/perf-benchmarking.md` for fresh-process / noise rules.
12. **`report`** — summarize artifacts; recommend promote, reject, or iterate.
    Promotion into product defaults still requires an admissions row when that
    registry defines one — currently fail closed on empty records.

## Experiment guidance (non-inventory)

- Start high-signal paths (e.g. `mfp4` + `imatrix-scale`, or MQ4 activation-aware
  recipes) and **compare stacks empirically** — do not treat a single past run
  as a permanent win.
- ParoQuant remains a first-class transform lane: probe/import/oracle are
  executable; fused runtime inverse + matvec still need producer-consumer proof
  before ship language.
- MoE policies must separate router, expert, and shared dense tensors; expert
  promotion needs hit-distribution + quality deltas, not name lists alone.
- KV: product defaults and mode math live in `docs/QUANTIZATION.md` /
  `docs/CONFIG.md` / `docs/env-vars.md`. Astrea `kv-profile` modes are policy
  candidates; research modes stay research until kernels, loader metadata, and
  measured AR/DFlash quality exist.
- Preserve HFP/MFP (and any other) producer-consumer contracts: block-size and
  runtime requirements move only with quantizer + loader + kernels + docs
  together.

## Guardrails

- Attach engine fingerprint + eval identity to every quality artifact.
- Atlas / speed-gate / harness green ≠ admission and ≠ universal correctness.
- Path-specific numerical/state parity uses arch-owned oracles when they exist;
  serve harnesses cover user-facing semantics only (`docs/VALIDATION.md`).
- Historical benchmark tables and campaign checkpoints stay historical.
- No C/A attestation from this skill.

## Related skills

- [hipfire-kernel-atlas](../hipfire-kernel-atlas/SKILL.md) — AR/DFlash rows, ISA fit, task/eval loop
- [hipfire-kernel-tuning](../hipfire-kernel-tuning/SKILL.md) — kernel levers after a candidate is quality-clean
- [hipfire-tester](../hipfire-tester/SKILL.md) — hardware bring-up / smoke matrix
