# Quant Quality Tooling

This document describes the current hipfire quant-quality tooling introduced while investigating the Qwen3.5-9B MQ4 quality gap. The tools are deliberately split into two layers:

- **Astrea**: a JSON-first CLI and agent skill for planning, recording, and summarizing model calibration experiments.
- **PyTorch oracle probes**: targeted Qwen3.5 scripts that compare hipfire HFQ/MQ execution against a Hugging Face BF16 reference, then rank the tensors that contribute most to quality loss.

The tools are meant to make quant work empirical. A candidate is only better when it improves measured quality under the same engine fingerprint, reference, prompt/chunk set, and runtime mode. A candidate is only shippable when it also passes runtime throughput and decode smoke checks.

## Why This Exists

The initial Qwen3.5-9B MQ4 baseline had a large quality gap versus the BF16 reference:

| Artifact | 20-chunk KLD | PPL | Eval tok/s | Notes |
|---|---:|---:|---:|---|
| baseline MQ4 | `0.330882` | `9.3404` | `346` | starting point |
| prior uniform LS/clip POC | `0.308592` | `9.3769` | n/a | KLD improved but PPL regressed; rejected |
| best current policy candidate | `0.206809` | `9.2878` | `328` | tensor-selective mixed MQ4/MQ6 policy, still short of Q4-class KLD |
| full-MQ6 + q8 KV control | `0.067687` | `9.3978` | `233` | proves the eval can see sub-`0.09` KLD, but fails PPL/speed |

The conclusion so far is precise: the simple tensor-selective loop moved KLD substantially, but it is flattening around `0.205-0.21`. That plateau is not a theoretical model floor; it is a limitation of the current policy/search method.

## Tool Inventory

### Astrea CLI

`./scripts/astrea.py` is the workflow spine. It emits JSON artifacts for inspection, engine fingerprints, calibration plans, metrics, dynamic tensor policies, model promotion recipes, KV-cache policy planning, bundle planning, and compact reports.

Top-level commands:

```bash
python3 scripts/astrea.py inspect --model MODEL --pretty
python3 scripts/astrea.py fingerprint --engine-root . --pretty
python3 scripts/astrea.py plan --model MODEL --format mq4 --method awq --pretty --out plan.json
python3 scripts/astrea.py calibrate --plan plan.json --source-dir BF16_DIR --pretty --out calibrate.json
python3 scripts/astrea.py metrics --quality-json result-data.json --candidate-variant CAND --baseline-variant BASE --pretty
python3 scripts/astrea.py policy --model MODEL --base-format mq4 --promotion-format mq6 --sensitivity-json scores.json --max-extra-bytes N --pretty --out policy.json
python3 scripts/astrea.py promote --policy policy.json --source-dir BF16_DIR --output candidate.hfq --pretty --out promote.json
python3 scripts/astrea.py kv-profile --model MODEL --mode asym3 --mode q8 --pretty --out kv.json
python3 scripts/astrea.py bundle-plan --model MODEL --output MODEL.hfq --include weights --include evidence --pretty --out bundle.json
python3 scripts/astrea.py report artifact1.json artifact2.json --pretty
```

Astrea is intentionally conservative. `plan`, `kv-profile`, and `bundle-plan` are contracts and evidence shapes, not proof that a candidate is good. `promote` writes selected model candidates, but every promoted candidate still needs independent quality and perf measurement.

### Agent Skill

`.agents/skills/astrea/SKILL.md` packages the workflow for agents. It defines the guardrails agents should follow when planning or evaluating quant calibration work:

- record engine fingerprints before comparing quality rows;
- treat KLD/PPL/MSE and above-floor KLD as first-class metrics;
- keep MoE router/expert/shared tensors separate;
- treat KV policies as measurable candidates, not assumptions;
- send runtime-sensitive candidates through Atlas for AR/DFlash perf before promotion claims.

### Hipfire Runtime Examples

Two small Qwen3.5 examples provide the bridge between hipfire execution and the PyTorch oracle scripts.

`dump_qwen35_hidden` dumps post-layer hidden states, final logits, and final RMSNorm output:

```bash
cargo build --release --features deltanet \
  --example dump_qwen35_hidden \
  --example probe_qwen35_l0_ops \
  --example eval_hipfire \
  -p hipfire-runtime

./target/release/examples/dump_qwen35_hidden \
  --model ~/.hipfire/models/qwen3.5-9b.mq4 \
  --out-prefix experiments/quant_fix_20260513/oracle/baseline-hidden \
  --prompt 'The quick brown fox jumps over the lazy dog.' \
  --layers all \
  --kv-mode f32
```

`probe_qwen35_l0_ops` dumps the layer-0 MQ4 path at a narrower granularity: embedding, RMSNorm, MQ rotation, split projection outputs, and fused projection outputs. Use it when checking whether a quality cliff is caused by the engine or by the quantized weights.

```bash
./target/release/examples/probe_qwen35_l0_ops \
  --model ~/.hipfire/models/qwen3.5-9b.mq4 \
  --out-prefix experiments/quant_fix_20260513/oracle_runtime/baseline-l0
```

`eval_hipfire` now fails fast on non-finite candidate logits and non-finite log partition values. This prevents bogus `KLD=0` or NaN-contaminated rows from entering the ledger.

### PyTorch Oracle Scripts

The Qwen3.5 scripts are standalone and intentionally not hidden behind Astrea yet. That keeps iteration fast while the method is still evolving.

`qwen35_pytorch_oracle.py` compares a hipfire hidden dump against a BF16 Hugging Face forward pass:

```bash
/home/kaden/.venvs/hipfire-rocm-torch-amd721/bin/python3 scripts/qwen35_pytorch_oracle.py \
  --hipfire-meta experiments/quant_fix_20260513/oracle/candidate-hidden.meta.json \
  --hf-model /path/to/Qwen3.5-9B/snapshot \
  --device cuda \
  --dtype bf16 \
  --out experiments/quant_fix_20260513/oracle/candidate-oracle.json
```

It reports layerwise hidden drift, final norm drift, and final logits drift. During this investigation it showed smooth accumulated quant drift, not a single RoPE-style runtime mismatch.

`qwen35_torch_hfq_attribution.py` replays Hugging Face layers with one HFQ-dequantized tensor swapped in at a time:

```bash
/home/kaden/.venvs/hipfire-rocm-torch-amd721/bin/python3 scripts/qwen35_torch_hfq_attribution.py \
  --hf-model /path/to/Qwen3.5-9B/snapshot \
  --hfq-model ~/.hipfire/models/qwen3.5-9b.mq4 \
  --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --chunk 0 \
  --seq-len 256 \
  --layers all \
  --tensor-filter conv1d \
  --tensor-filter in_proj_qkv \
  --device cuda \
  --dtype bf16 \
  --out experiments/quant_fix_20260513/oracle/attribution.json
```

Use this for local tensor ranking only. A high local rank is a hypothesis, not an acceptance signal. Several high-ranked tensors improved local error but regressed 20-chunk KLD or throughput.

`qwen35_projection_error.py` uses a hipfire hidden dump as the activation source and compares projection outputs for tensors whose inputs are available at layer boundaries. This is faster than full layer replay for qkv-style probes.

`qwen35_hfq_projection_probe.py`, `qwen35_layer0_dequant_oracle.py`, `qwen35_la_layer_dequant_oracle.py`, and `qwen35_fa_layer_dequant_oracle.py` are narrow engine-vs-HFQ checks. They are useful when the hypothesis is a runtime math mismatch rather than a quant policy problem.

## Evaluation Contract

Use the same fixed eval contract when comparing candidates:

```bash
HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE=asym3 \
./target/release/examples/eval_hipfire \
  --model CANDIDATE.hfq \
  --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --output experiments/quant_fix_20260513/eval/CANDIDATE.prefill-c20.kldseq \
  --scoring-mode prefill \
  --max-chunks 20
```

The current iteration gate was:

- accept only if KLD strictly decreases versus the current accepted candidate;
- require PPL `<= 9.3404`;
- require eval throughput `>= 328 tok/s`;
- run hidden/logits oracle and decode smoke before calling an accepted candidate deliverable quality.

For decode smoke, disable verify-graph capture when measuring the candidate itself:

```bash
HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE=asym3 HIPFIRE_VERIFY_GRAPH=0 \
./target/release/examples/dflash_spec_demo \
  --target CANDIDATE.hfq \
  --draft ~/.hipfire/models/qwen35-9b-dflash-mq4.hfq \
  --prompt "$(cat benchmarks/prompts/lru_cache_pep8_strict.txt)" \
  --max 120 \
  --ctx 2048 \
  --kv-mode asym3 \
  --no-adaptive-b \
  --no-chatml
```

## What The First Run Found

The first direct PyTorch loop ruled out an engine mismatch on the tested path:

- same-HFQ PyTorch replay matched hipfire at roughly `5e-7` to `1.34e-6` last-token relative RMSE across layers 1-31;
- layer-0 projection probe matched around `8e-8` to `9e-8` relative RMSE;
- Qwen3.5 MQ4 quality loss accumulated smoothly rather than appearing as a single hard layer cliff.

The useful candidate pattern was not uniform tuning. It was tensor-selective and multi-metric:

1. promote KLD-positive `linear_attn.conv1d` tensors to MQ6;
2. repair PPL regressions with targeted `linear_attn.in_proj_qkv` MQ4 refits;
3. reject candidates that improve local RMSE but fail KLD, PPL, or speed.

The current best accepted candidate is recorded in `experiments/quant_fix_20260513/policy-map.current-best.json`. It improved KLD from `0.330882` to `0.206809`, while keeping PPL and speed inside the gate. It is not Q4-class yet.

## What Did Not Work Yet

- Uniform MQ4 LS/clip improved KLD but regressed PPL.
- Simple original-domain imatrix/AWQ was not enough for MQ4 after FWHT.
- Local tensor attribution did not reliably predict full 20-chunk KLD.
- Diagonal activation-weighted rotated-domain refit over an 8x512 calibration slice did not beat the best plain LS path.
- Full-MQ6 is an important quality control but fails the target PPL/speed envelope.

## Next Work

The next credible quality lever is a stronger objective, not more single-tensor local ranking:

- GPTQ-style or Hessian-aware block refit against calibration activations;
- rotated-domain objectives that optimize downstream projection error instead of raw weight error;
- larger and mixed calibration sets, at least 512 sequences x 2048 tokens for final claims;
- explicit KV-policy evaluation once weights are accurate enough that asym3/q8 differences dominate;
- Atlas joins for AR and DFlash throughput before any promoted candidate is called shippable.

## Artifact Hygiene

Do not commit raw hidden dumps, logits dumps, model candidates, or large covariance files. Keep those under local `experiments/` directories. Commit only compact ledgers, policy maps, summaries, and methodology docs that let another developer reproduce or audit the result.
