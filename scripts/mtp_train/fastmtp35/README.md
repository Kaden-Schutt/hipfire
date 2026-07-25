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

The hiptrx default uses the verified 32 GB R9700 capacity point: 128
sequences per GPU, an effective global batch of 512, and one DDP all-reduce
per optimizer step. The `2e-4` learning rate is a conservative 4x scale from
the original batch-64 run, with validation every 250 steps and resumable
checkpoints every 500. Losses are recursive K=3 CE with normalized
weights `[0.5102, 0.3061, 0.1837]`. Full-vocab targets outside the deployed
16K draft vocabulary are explicitly excluded and reported as coverage; they
are not silently aliased. DDP caps all ranks to the smallest feature
partition, preventing an uneven final shard from hanging collectives.

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

The packer emits the candidate and SHA256 provenance without modifying the
stock sidecar.

## Stage 5: sampled product and Redline certification

Certification uses isolated same-filesystem hard-link fixtures, so the stock
model directory is never overwritten and no trunk bytes are copied. Hard links
are required because serve canonicalizes model paths before sibling-sidecar
discovery; symlink fixtures would silently resolve back to the global stock
`.mtp`. The script verifies the sidecar path after every MTP arm. It runs the
same eight-turn sampled multi-turn session at the registry defaults with
`thinking=med`, `max_tokens=4096`, and Q8 KV for plain AR, stock MTP, and
trained MTP. It then runs the retained-PM4 trunk shadow/parity diagnostic for
15 consecutive positions.

```bash
scripts/mtp_train/fastmtp35/certify_head.sh \
  ~/.hipfire/training/fastmtp-qwen36-a3b-v1/qwen3.6-35b-a3b.fastmtp.mtp
```

Promotion requires coherent outputs with no new empty/runaway/attractor
failures, trained-MTP throughput above AR and stock MTP, higher useful tau,
and a passing Redline shadow report. Offline CE or top-1 agreement alone is
not a promotion result.

## Reuse for DeepSeek and MiniMax

`hipfire-mtp-data` is intentionally architecture-neutral. A later MI300X
pipeline reuses the shard reader, recursive trainer shape, checkpoint
provenance, and sampled certification contract. Only the architecture-specific
feature producer and head module/export mapping change.
