---
name: astrea
description: Use for hipfire quant calibration, imatrix-driven experiments, KLD/PPL quality evaluation, k-map/format selection, MQ/HFQ/HFP/MFP tradeoff work, ParoQuant-style weight transform planning, and KV policy planning. Use when deciding whether a calibrated model candidate should be promoted, rejected, packaged, or sent through Atlas for AR/DFlash perf validation.
---

# Astrea

Astrea is hipfire's agent-native model calibration harness. It is a Python CLI
for humans and a workflow contract for agents. The CLI emits plain JSON
artifacts for weight calibration, ParoQuant-style transform planning, KV-cache
policy planning, and future single-file model packaging; the agent supplies
judgment, guardrails, and the next experiment.

## Core Rule

Do not claim a quant candidate is better without measured quality evidence.
Do not claim it is ship-ready without runtime compatibility and perf evidence.
Do not claim Astrea wrote a packaged model unless the loader/package work exists;
`bundle-plan` is a contract artifact, not a model writer.

## CLI

Run from the hipfire repo root:

```bash
python3 scripts/astrea.py inspect --model MODEL [--imatrix IMATRIX] [--format FORMAT] [--pretty] [--out PATH]
python3 scripts/astrea.py imatrix-join --model MODEL --imatrix IMATRIX [--max-tensors N] [--pretty] [--out PATH]
python3 scripts/astrea.py fingerprint [--engine-root REPO] [--pretty] [--out PATH]
python3 scripts/astrea.py plan --model MODEL --format FORMAT --method METHOD [--recipe-stage STAGE:METHOD] [--imatrix IMATRIX] [--source-dir BF16_DIR] [--eval-command CMD] [--atlas-command CMD] [--pretty] [--out PATH]
python3 scripts/astrea.py calibrate --plan PLAN.json [--source-dir BF16_DIR] [--write-candidate] [--max-tensors N] [--tensor-filter NAME] [--workers N] --pretty [--out PATH]
python3 scripts/astrea.py eval --plan PLAN.json [--run] --pretty [--out PATH]
python3 scripts/astrea.py metrics --quality-json result-data.json --candidate-variant NAME [--baseline-variant NAME] [--floor-variant NAME] [--arch ARCH] [--scoring-mode MODE] [--engine-root REPO] --pretty [--out PATH]
python3 scripts/astrea.py policy --model MODEL --base-format FORMAT --promotion-format FORMAT (--sensitivity-json SCORES.json | --imatrix IMATRIX) --max-extra-bytes N [--method METHOD] [--objective dynamic-tensor-policy|moe-probe|model-ingress|kv-policy] [--domain weights|kv] [--model-family FAMILY] --pretty [--out PATH]
python3 scripts/astrea.py promote --policy POLICY.json --source-dir BF16_DIR --output CANDIDATE.hfq [--max-tensors N] [--tensor-filter NAME] --pretty [--out PATH]
python3 scripts/astrea.py kv-profile --model MODEL [--mode q8|asym3|triattn|cask|turbo3|rotor] [--triattn PATH] [--model-family FAMILY] [--engine-root REPO] --pretty [--out PATH]
python3 scripts/astrea.py bundle-plan --model MODEL --output MODEL.hfq [--include weights|paro|kv-policy|triattn|evidence] [--triattn PATH] [--policy-id ID] --pretty [--out PATH]
python3 scripts/astrea.py report ARTIFACT.json ... --pretty [--out PATH]
```

Prefer `--pretty` for human review and compact JSON for ledgers. Use `--out`
for reproducible run directories; it writes JSON to the path and leaves stdout
empty.

## Workflow

1. Identify the target model, desired format, reference model, eval dataset,
   and budget.
2. Run `inspect` to fingerprint the model and imatrix inputs.
   Use `imatrix-join` when you need a focused report of GGUF imatrix tensor
   coverage against HFQ tensor names before planning a calibration run.
3. Run `fingerprint` to capture the engine path. This records git state,
   relevant source hashes, `HIPFIRE_ROPE_INTERLEAVED_LEGACY`, and whether the
   default Qwen3.5 FA RoPE path is `halfsplit`, `interleaved_legacy`, or
   `unknown`.
4. Run `plan` to create a bounded experiment artifact.
   Calibration methods are stackable recipe stages. Use repeated `--method`
   flags for the candidate stack, and optional repeated `--recipe-stage`
   flags when you need an explicit stage order such as
   `scale_search:imatrix-scale`, `activation_aware:awq`, `rounding:gptq`,
   `promotion:kmap`, or `transform:quarot`.
5. Run `calibrate`. Without `--write-candidate`, Astrea joins GGUF imatrix
   logical tensors to HFQ tensor names and reports whether the candidate is
   ready for a weight-mutation pass. With `--write-candidate`, Astrea can write
   an MFP4 imatrix-scale candidate or an MQ4 AWQ-style activation-weighted
   clipping candidate by copying the base HFQ and patching selected same-size
   tensor byte ranges. Use `--max-tensors` or `--tensor-filter` for smoke
   passes before a full rewrite. Use `--workers N` for process-parallel tensor
   rewrites on large models; start with 4 workers unless memory headroom is
   known.
6. Run `eval` with KLD/PPL commands against a BF16 or accepted higher-precision
   reference.
7. Run `metrics` on the `kld_reduce.py` `result-data.json` artifact. Prefer
   a Q8 or accepted high-precision floor row when available, so Astrea can
   report above-floor KLD and recovered quantization damage percentage. Always
   pass `--engine-root` when the evaluated engine is not the checkout running
   Astrea.
8. Run `policy` when you want an Unsloth-like dynamic quant policy. It ranks
   tensors by sensitivity per added byte and emits a mixed-format promotion
   recipe under a size budget. Use `--objective moe-probe` for MoE models and
   `--objective model-ingress` when bringing up a new model family; these add
   router/expert and alias-map probe work items to the artifact. Add repeated
   `--domain` flags when the policy spans both weight transforms and KV-cache
   policy. Use `--method paroquant` to add the Paro weight-transform lane; this
   is a planned transform section until the quantizer/runtime have a compatible
   implementation. For rotated MQ/MFP bases promoted to Q8/F16, Astrea
   automatically bundles runtime anchor projections (`q`, `qkv`, `gate`) with
   dependent projections so mixed-format candidates do not read stale normalized
   activation buffers.
9. Run `promote` when a policy selects tensors for mixed-format promotion.
   Today this writes selected `q8` promotions as runtime-compatible `Q8F16`
   tensor records and rebuilds the HFQ index/data payload. Use `--max-tensors`
   for smoke candidates before writing a full policy. Legacy policies are also
   expanded with required runtime anchors at write time. Re-run `metrics` after
   every promotion candidate because the policy byte model is only a selector,
   not quality evidence.
10. Run `kv-profile` when a candidate changes KV-cache behavior or when a model
   should carry an embedded KV policy. Include at least the current baseline
   (`asym3`) and the candidates being investigated (`triattn`/`cask`,
   `turbo3`, `rotor`, or related modes). The output is the policy/evidence
   shape Atlas should join against AR and DFlash perf rows.
11. Run `bundle-plan` when the candidate needs a future single-file model
   package. The target is an HFQ package-style container with weights,
   transform metadata, KV policy, and TriAttention/CASK centers embedded inside
   the model artifact. External sidecars are not the target. Loader, daemon,
   CLI, and kernel support remain deferred runtime work until implemented.
12. If quality improves, run Atlas AR and DFlash perf collection before any
   promotion claim.
13. Use `report` to summarize evidence and recommend promote, reject, or
   iterate.

## Format Guidance

- Start with `mfp4 + imatrix-scale` when reproducing the known high-signal
  calibration path.
- For `mq4`, Astrea can now write a same-format AWQ-style activation-weighted
  clipping candidate. The first 9B run improved PPL but slightly worsened KLD,
  so treat this recipe as an empirical lane to iterate, not a validated win.
  Compare stackable recipes such as AWQ, imatrix-scale, GPTQ, k-map/promotion,
  and transform stages empirically.
- Treat ParoQuant as the highest-priority transform lane to prototype next
  after the existing imatrix/AWQ/GPTQ/k-map evidence loop is reliable. It needs
  a producer-consumer contract, not just an Astrea plan.
- Treat `asym3` as the current KV baseline. TriAttention/CASK are packageable
  persistent sidecar data once embedded in the model package. TurboQuant-like
  and RotorQuant-like KV policies are research candidates until kernels,
  loader metadata, and AR/DFlash quality gates exist.
- For MoE, separate router tensors, expert tensors, and shared dense tensors in
  the policy artifact. Expert promotion should be justified by expert-hit
  distribution plus quality deltas, not static tensor names alone.
- Keep `mq3`, `mq4`, `mq6`, `hfq4`, `hfq6`, `hfp4`, and `mfp4` eligible for
  experiments, but tie every recommendation to quality and perf artifacts.

## Guardrails

- Preserve producer-consumer contracts. HFP/MFP candidates must remain
  compatible with the current fast-path block-size/runtime requirements unless
  quantizer, loader, docs, and kernels are moved together.
- Attach exact eval commands, reference model, dataset/chunk count, and output
  artifact size to quality claims.
- Do not compare KLD/PPL rows across different engine fingerprints or RoPE
  conventions without explicitly marking the comparison as historical.
- Attach Atlas rows for AR and DFlash when the candidate affects runtime
  formats used by both paths.
- Treat dry-run Astrea artifacts as plans only, not calibrated weights.
- Treat `kv-profile` and `bundle-plan` artifacts as policy contracts only. They
  should drive loader/kernel/Atlas work, but they do not prove runtime support.
- If an eval run emits non-finite logits, fail the candidate. Do not accept
  `KLD=0` rows from older evaluators unless the logit path is confirmed finite.
