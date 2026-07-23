# Qwen3.6 35B-A3B FastMTP data run

This directory prepares the 400K-example FastMTP run for the deployed
`qwen3.6-35b-a3b.mq4r` trunk.

The prompt assembler streams 440K candidates from the seven sources in
`corpus.toml`, discards every source answer, globally exact/MinHash
deduplicates prompts, applies the cached official Qwen3.6 chat template, and
writes pretokenized Hipfire jobs sorted by exact prompt length.

## Prepare on hiptrx

```bash
cd ~/hipfire-fastmtp35
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r scripts/mtp_train/fastmtp35/requirements.txt

ROOT=~/.hipfire/datasets/fastmtp-qwen36-a3b-v1
.venv/bin/python scripts/mtp_train/fastmtp35/assemble_prompts.py \
  --output "$ROOT"
```

Use a small deterministic smoke before the full assembly:

```bash
.venv/bin/python scripts/mtp_train/fastmtp35/assemble_prompts.py \
  --scale 0.001 \
  --output /tmp/fastmtp35-assembly-smoke
```

`manifest.json` pins source counts, tokenizer, Git head, job hashes, sampling
profiles, and rejection counts. `LICENSES.md` is the prompt-source license
ledger.

## Generate exact-trunk responses

Do not start while another GPU campaign owns any hiptrx card. The runner uses
one lock per physical GPU and the certified fixed-slot Redline PM4 profile.

```bash
scripts/mtp_train/fastmtp35/run_hiptrx_distill.sh \
  ~/.hipfire/datasets/fastmtp-qwen36-a3b-v1
```

The runner produces four completion shards for each of nine jobs:
three prompt/output-length buckets crossed with product-sampled, FastMTP
sampled, and greedy profiles.

## Clean and split

```bash
.venv/bin/python scripts/mtp_train/fastmtp35/finalize_distill.py \
  --root ~/.hipfire/datasets/fastmtp-qwen36-a3b-v1 \
  --output ~/.hipfire/datasets/fastmtp-qwen36-a3b-v1/clean \
  --target 400000
```

The finalizer rejects truncated, very short, repetitive, and duplicate
responses and emits deterministic 98/1/1 train/validation/test splits.
`training.json` pins the initial K=3 FastMTP optimization recipe.
