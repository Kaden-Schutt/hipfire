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
SMOKE=~/.hipfire/datasets/.fastmtp-qwen36-a3b-v1-smoke
rm -rf "$SMOKE"
.venv/bin/python scripts/mtp_train/fastmtp35/assemble_prompts.py \
  --scale 0.001 \
  --output "$SMOKE"
rm -rf "$SMOKE"
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

The finalizer rejects very short, repetitive, and duplicate trajectories and
emits deterministic 98/1/1 train/validation/test splits. Length-capped trunk
trajectories remain valid MTP supervision; the trainer masks their final
recursive targets rather than requiring an EOS-complete SFT answer.
`training.json` pins the initial K=3 FastMTP optimization recipe.

After GPU work releases the dataset volume, produce a read-only durable audit
of the original teacher corpus:

```bash
cargo run --release -p hipfire-mtp-data --bin audit_fastmtp_distill -- \
  ~/.hipfire/datasets/fastmtp-qwen36-a3b-v1 \
  --expected-rows 440000 \
  --output ~/.hipfire/datasets/fastmtp-qwen36-a3b-v1/distill-audit.json
```

The auditor streams every prompt and completion shard without rewriting valid
data. It verifies manifest/job hashes, the exact four-way GPU shard set,
partition ownership and unique indices, prompt IDs, sampling contracts, JSON
integrity, row and completion-token totals, and records per-job/per-GPU hashes
and finish-reason counts.

## Stage 2: exact deployed-trunk features on four R9700s

Stage 2 is deliberately separate from optimization. The full MQ4R trunk
occupies about 18 GB; a BF16 MoE MTP head plus gradients and Adam state also
fits a 32 GB R9700, but the two do not safely coexist. The Rust producer loads
the exact deployed trunk with the registry's Q8 KV/state contract, runs
teacher-forced prompt plus completion prefill, and stores only assistant-side
post-final-norm hidden rows.

Each portable `HFMTPF01` record contains:

- an independent MTP attention window and its absolute starting position;
- `N` committed assistant token ids and `N` BF16 hidden rows;
- three trailing token ids for the recursive K=3 targets;
- source ordinal and model/dataset/commit provenance.

Serving-aligned training pairs `h[t]` with the shifted token `x[t+1]` and
predicts `x[t+2]`. Schema 1 therefore exposes `N-1` usable rows from each
already-produced record; its final hidden row is ignored, preserving the
100M-row artifact without regeneration.

Two 128-row windows cover the beginning and tail of each trajectory. The
default 100M-row budget is about 410 GB at hidden size 2048. Output shards are
atomic, XXH3-checksummed (native-speed in both Rust and Python), independently
resumable per GPU, and bounded to roughly 1 GB.

After Stage 1 finalization has completed and released all GPUs:

```bash
scripts/mtp_train/fastmtp35/run_hiptrx_features.sh \
  ~/.hipfire/datasets/fastmtp-qwen36-a3b-v1
```

Do not launch this concurrently with the Stage 1 teacher service. The runner
uses one physical-GPU lock per R9700 and writes partition state beneath
`features/{train,validation}`. Each GPU is supervised independently: a failed
ROCr process resumes from its last atomic shard without stopping healthy
partitions. `FEATURE_RETRY_LIMIT` (default `8`) and
`FEATURE_RETRY_BACKOFF_SECS` (default `10`) bound that recovery loop. A
progress watchdog also terminates ROCr children that fault without exiting;
`FEATURE_STALL_TIMEOUT_SECS` (default `240`), `FEATURE_STALL_POLL_SECS`
(default `15`), and `FEATURE_TERM_GRACE_SECS` (default `10`) control it.

### Build a deployment-derived 16K vocabulary

Do not overwrite the stock sidecar's vocabulary map. Build a distinct map from
the exact runtime-shifted training targets retained in the Stage 2 features:

```bash
cargo run --release -p hipfire-mtp-data --bin build_mtp_vocab_map -- \
  --features ~/.hipfire/datasets/fastmtp-qwen36-a3b-v1/features/train \
  --base-map ~/.hipfire/datasets/fastmtp-qwen36-a3b-v1/features/vocab-map.json \
  --output ~/.hipfire/datasets/fastmtp-qwen36-a3b-v1/features/vocab-map-v3-deployment16k.json
```

The scanner reads token ids and structurally validates every record while
seeking over the roughly 400 GiB hidden-state payload. It records the exact
trunk, source-manifest, producer, target coverage, weighting, and overlap with
the stock map in the output. Full hidden-payload checksums remain the Stage 2
feature audit's responsibility.

## Stage 3: head-only K=3 training on four R9700s

Hiptrx currently has the CPU-only Torch wheel. After Stage 1 has released the
GPUs, bootstrap the Ubuntu ROCm Torch package and verify all four cards report
`gfx1201`:

```bash
scripts/mtp_train/fastmtp35/bootstrap_hiptrx_train_env.sh
```

The bootstrap refuses to run while the Stage 1 service is active. It creates a
`.venv-rocm` with the system ROCm Torch package, then installs only the
non-Torch Python dependencies; it never replaces ROCm Torch with a CPU PyPI
wheel.

The runner extracts the exact 16K vocab map embedded in the deployed `.mtp`,
loads only the official BF16 `mtp.*` warm start plus the frozen input
embedding/compressed LM rows, and uses DDP over the four R9700s:

```bash
scripts/mtp_train/fastmtp35/run_hiptrx_train.sh \
  ~/.hipfire/datasets/fastmtp-qwen36-a3b-v1
```

The launcher resumes from the newest matched
`step-*.{safetensors,optimizer.pt}` checkpoint pair in the output directory.
Set `HIPFIRE_FASTMTP_AUTO_RESUME=0` for an intentional fresh start, or pass
both `--resume-weights` and `--resume-optimizer` to select an explicit pair.

The v3 distribution-alignment pilot uses the deployment-derived map and blends
realized-token CE with the exact trunk distribution available from the next
retained hidden row:

```bash
VOCAB_MAP=~/.hipfire/datasets/fastmtp-qwen36-a3b-v1/features/vocab-map-v3-deployment16k.json \
OUTPUT=~/.hipfire/training/fastmtp-qwen36-a3b-v3-dist \
HIPFIRE_FASTMTP_AUTO_RESUME=0 \
scripts/mtp_train/fastmtp35/run_hiptrx_train.sh \
  ~/.hipfire/datasets/fastmtp-qwen36-a3b-v1 \
  --max-steps 1000 \
  --micro-batch-size 64 \
  --global-batch-size 512 \
  --soft-target-weight 0.5 \
  --soft-target-topk 256
```

Soft targets retain the teacher's top-K probabilities individually and one
aggregate probability bucket for the remaining 16K support. This preserves the
teacher's tail mass instead of renormalizing the top-K tokens, while avoiding a
second full-vocabulary projection and large persistent probability tensors.
Pilot checkpoints must still be selected by sampled product tau rather than
offline loss alone. The distribution graph requires a 64-sequence microbatch
with two-way gradient accumulation on 32 GB R9700s; MI300X can use the larger
microbatch directly.

The hiptrx default uses the verified 32 GB R9700 capacity point: 128
sequences per GPU, an effective global batch of 512, and one DDP all-reduce
per optimizer step. The peak learning rate is `5e-5`, matching the published
FastMTP recipe, with validation every 250 steps and resumable checkpoints
every 500. Training recursively reuses the shared head while teacher-forcing
the shifted ground-truth token at every depth; self-rollout is reserved for
diagnostic evaluation and product certification. Losses are K=3 CE with
normalized weights `[0.5102, 0.3061, 0.1837]`. Full-vocab targets outside the
deployed 16K draft vocabulary are explicitly excluded and reported as
coverage; they are not silently aliased. DDP caps all ranks to the smallest
feature partition, preventing an uneven final shard from hanging collectives.

## Stage 4: package a deployable sidecar

The trainer writes `final.safetensors` with canonical `mtp.*` names.
`mtp_extract --mtp-override` shadows the official MTP tensors with those
trained tensors while continuing to source config, embeddings, and the 16K
compressed LM rows from the official checkpoint:

```bash
scripts/mtp_train/fastmtp35/package_head.sh \
  ~/.hipfire/training/fastmtp-qwen36-a3b-v1 \
  ~/.hipfire/training/fastmtp-qwen36-a3b-v1/qwen3.6-35b-a3b.fastmtp.mtp
```

The packer defaults to the runtime-compatible `mixed` contract: shared MTP
matrices remain Q8 while the routed-MoE experts and compressed draft LM head
use MQ4G256. Full-Q8 routed experts are rejected because the serving runtime
cannot execute that representation. Set `MTP_QUANT=mq4` only for a comparison
artifact. The packer emits SHA256 and key-value provenance, including the
training checkpoint, manifest, vocabulary map, HF snapshot, quantization
contract, and producer commit, without modifying the stock sidecar. It fails
closed unless `steps == planned_steps == stop_step`, the four-rank Q8 feature
contract matches, and the vocabulary-map hash is exact. Pilot packaging
requires the explicit research-only
`HIPFIRE_FASTMTP_ALLOW_PARTIAL=1` override. After a complete run,
`HIPFIRE_FASTMTP_CHECKPOINT=/path/to/step-N.safetensors` can package a retained
validation checkpoint for product A/B without weakening the full-run gate.

## Stage 5: sampled product and Redline certification

Certification uses isolated same-filesystem hard-link fixtures, so the stock
model directory is never overwritten and no trunk bytes are copied. Hard links
are required because serve canonicalizes model paths before sibling-sidecar
discovery; symlink fixtures would silently resolve back to the global stock
`.mtp`. The script verifies the sidecar path after every MTP arm. It runs the
same eight-turn sampled multi-turn session at the registry defaults with
`thinking=med`, `max_tokens=4096`, and Q8 KV for plain AR, stock MTP, and
trained MTP. All three arms use the same recorded seed (42 by default) so the
comparison is repeatable without changing the registry distribution. It then
runs the retained-PM4 trunk shadow/parity diagnostic for 15 consecutive
positions.

```bash
scripts/mtp_train/fastmtp35/certify_head.sh \
  ~/.hipfire/training/fastmtp-qwen36-a3b-v1/qwen3.6-35b-a3b.fastmtp.mtp
```

Promotion requires coherent outputs with no new empty/runaway/attractor
failures, trained-MTP throughput above AR and stock MTP, higher useful tau,
strict MTP prefix reuse on all seven continuation turns, complete scored
session recall, and a passing Redline shadow report. Certification rebuilds
the CLI and daemon from the clean current commit and records hashes for the
trunk, both sidecars, session, and source revision. Offline CE or top-1
agreement alone is not a promotion result.

## Reuse for DeepSeek and MiniMax

`hipfire-mtp-data` is intentionally architecture-neutral. A later MI300X
pipeline reuses the shard reader, recursive trainer shape, checkpoint
provenance, and sampled certification contract. Only the architecture-specific
feature producer and head module/export mapping change.
