# Repro Recipe — v3 winner + iterative pipeline

All commands are executed in the **hiptrx awq-kmap-bench worktree**:
```bash
cd /home/kaden/hipfire/.worktrees/awq-kmap-bench
```

The worktree must be at origin/master HEAD post-PR-#273 (`a99b4643` or later) PLUS the iterative pipeline overlay. Branch `iterative-awq-gptq` at `f286bade` is the current tip.

## Step 0 — environment + prerequisites

Verify these files exist on hiptrx:

```bash
# BF16 source model
ls /home/kaden/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/

# Calibration imatrix (unsloth's published one)
ls /home/kaden/.hipfire/imatrix/unsloth/Qwen3.5-9B-GGUF/imatrix_unsloth.gguf_file

# Precomputed Hessian (c64 chunks at ctx=256)
ls ~/hipfire/.worktrees/paroquant/.codeinsight+research/astrea/mq4-gptq-9b-poc/20260515T-start/hessian-linear-c64-ctx256/stats-merged.npz

# GPTQ mask (67 linear tensors)
ls ~/hipfire/.worktrees/paroquant/.codeinsight+research/astrea/mq4-gptq-9b-poc/20260515T-start/mask.json

# BF16 reference dump for KLD eval
ls ~/hipfire/.worktrees/HIPa/benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin

# Calibration text corpus (for iterative re-collection)
ls benchmarks/calib/calib-1m.txt

# hipfire-quantize binary (Rust) and eval_hipfire example (Rust)
ls ./target/release/hipfire-quantize
ls ./target/release/examples/eval_hipfire
```

Rebuild if needed:
```bash
cargo build --release -p hipfire-quantize -p hipfire-runtime --features deltanet --example eval_hipfire
```

Python env:
```bash
PYTHON=/home/kaden/miniforge3/envs/hipfire-rocm/bin/python
```

## Step 1 — produce the F1 AWQ base (5 min, CPU + multi-thread)

```bash
./target/release/hipfire-quantize \
  --input /home/kaden/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/ \
  --output /home/kaden/.hipfire/models/qwen3.5-9b.mq4-awq-pr266-repro \
  --format mq4 --awq --awq-alpha 0.5 \
  --imatrix /home/kaden/.hipfire/imatrix/unsloth/Qwen3.5-9B-GGUF/imatrix_unsloth.gguf_file
```

Verify the output:
```bash
ls -lh /home/kaden/.hipfire/models/qwen3.5-9b.mq4-awq-pr266-repro
# Expected: ~5.0 GB (5314.8 MB written)

# AWQ sidecar count should be 184 (F1 input-side scope):
python3 -c "
import os
p='/home/kaden/.hipfire/models/qwen3.5-9b.mq4-awq-pr266-repro'
sz=os.path.getsize(p)
print(open(p,'rb').read(min(50_000_000,sz)).count(b'.awq_scale.weight'))
"
```

## Step 2 — AWQ-aware GPTQ pass (~5 min on GPU 0)

```bash
$PYTHON scripts/mq4_masked_calib.py quantize \
  --base /home/kaden/.hipfire/models/qwen3.5-9b.mq4-awq-pr266-repro \
  --source-dir /home/kaden/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --mask ~/hipfire/.worktrees/paroquant/.codeinsight+research/astrea/mq4-gptq-9b-poc/20260515T-start/mask.json \
  --stats-npz ~/hipfire/.worktrees/paroquant/.codeinsight+research/astrea/mq4-gptq-9b-poc/20260515T-start/hessian-linear-c64-ctx256/stats-merged.npz \
  --output /home/kaden/.hipfire/models/qwen3.5-9b.mq4-awq-pr266-gptq-v3 \
  --out /tmp/candidate-awq-gptq-v3.json \
  --method gptq --gptq-damp 0.01 --gptq-refit-iters 2 \
  --gpu 0 \
  --awq-aware-hessian /home/kaden/.hipfire/models/qwen3.5-9b.mq4-awq-pr266-repro \
  --tensor-filter "lm_head,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj,mlp.,self_attn." \
  --skip-unsupported --progress-every 30
```

Verify output:
```bash
ls -lh /home/kaden/.hipfire/models/qwen3.5-9b.mq4-awq-pr266-gptq-v3
# Expected: ~5.0 GB (same disk as flat MQ4)

# 184 sidecars preserved through GPTQ pass:
python3 -c "
import os
p='/home/kaden/.hipfire/models/qwen3.5-9b.mq4-awq-pr266-gptq-v3'
sz=os.path.getsize(p)
print(open(p,'rb').read(min(50_000_000,sz)).count(b'.awq_scale.weight'))
"
```

## Step 3 — bench (~24 min on 1× R9700 at c512)

```bash
mkdir -p .codeinsight+research/repro-v3/per-seq
ROCR_VISIBLE_DEVICES=0 ./target/release/examples/eval_hipfire \
  --model /home/kaden/.hipfire/models/qwen3.5-9b.mq4-awq-pr266-gptq-v3 \
  --ref ~/hipfire/.worktrees/HIPa/benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --output .codeinsight+research/repro-v3/per-seq/v3-repro__gfx1201__prefill.kldseq \
  --kv-mode q8 --scoring-mode prefill --max-chunks 512
```

Expected output line:
```
eval_hipfire: slice-mean KLD = 0.125730  mean NLL = 2.231066  PPL = 9.3098
```

Aggregate with the reducer:
```bash
$PYTHON benchmarks/quality-baselines/harness/kld_reduce.py \
  --result-dir .codeinsight+research/repro-v3/per-seq/ \
  --out-md /tmp/repro-summary.md \
  --out-json /tmp/repro-data.json
cat /tmp/repro-summary.md
```

Expected:
```
| Variant | Arch | Mode | n_chunks | Mean KLD ± 95% CI | p99 KLD | PPL |
| v3-repro | gfx1201 | prefill | 512 | 0.1257 (CI 0.1196–0.1322) | 13.714 | 9.310 |
```

## Step 4 — iterative AWQ+GPTQ (optional, ~2-3h on hiptrx 4× GPU)

```bash
$PYTHON scripts/mq4_masked_calib.py iterate \
  --hf-model /home/kaden/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --calib-text benchmarks/calib/calib-1m.txt \
  --imatrix-mask ~/hipfire/.worktrees/paroquant/.codeinsight+research/astrea/mq4-gptq-9b-poc/20260515T-start/mask.json \
  --base-output-dir .codeinsight+research/iterative-awq-gptq-9b/run-001 \
  --awq-alpha 0.5 --damping 0.5 --epsilon 0.01 --max-rounds 4 \
  --bench-each-round \
  --bench-ref ~/hipfire/.worktrees/HIPa/benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --gpu 0 \
  --collect-devices "0,1,2,3" \
  --ctx 256 --chunks 64 \
  --eval-bin ./target/release/examples/eval_hipfire \
  --kv-mode q8 --scoring-mode prefill --bench-max-chunks 512 \
  --gptq-damp 0.01 --gptq-refit-iters 2
```

Per-round artifacts emitted:
- `round_k/imatrix.npz` — collected Hessian
- `round_k/awq_scales.npz` — raw + damped scales
- `round_k/model.hfq` — quantized model
- `round_k/scales_delta.json` — relative L2 vs previous round
- `round_k/bench.kldseq` (if `--bench-each-round`) — per-chunk KLD
- `round_k/summary.md` — summary (KLD, PPL, scale-delta, time)

Best round = lowest KLD in `round_*/summary.md`. Final model = `round_<best>/model.hfq`.

## Step 5 — sanity check (`test_inference` 9/9 pass + coherence probe)

After producing v3 or any iterative round model:
```bash
./target/release/examples/test_inference /home/kaden/.hipfire/models/qwen3.5-9b.mq4-awq-pr266-gptq-v3
# Expected: 9/9 PASS (finite logits, forward parity, ChatML, asym3 KV alloc, decode, VRAM drain)
```

## Pinned commits (branch + SHA)

| Branch | HEAD | Contains |
|---|---|---|
| `awq-aware-hessian` | `85f6d055` | GPU GPTQ port, AWQ-aware Hessian transform, source pre-scale fix, AutoAWQ formula (opt-in) |
| `iterative-awq-gptq` | `f286bade` | All of awq-aware-hessian + iterate subcommand + parity tests |

## Negative experiments (don't repeat these)

| Attempt | KLD | Why it failed |
|---|---:|---|
| Naive GPTQ on AWQ base (no Hessian transform) | 1.7634 | GPTQ targets raw W, runtime applies x/s → off by factor s |
| AWQ-aware Hessian only (no source pre-scale) | 1.7531 | Hessian correct but source still W; same factor-s mismatch |
| F2 + AWQ-aware GPTQ at α=0.5 | 0.1386 | 64 extra output-side sidecars made things worse |
| F2 + AWQ-aware GPTQ at α=0.55 | 0.1396 | PR-author's PPL sweet spot; KLD-PPL inversion costs KLD |
| AutoAWQ formula at α=0.5 | 1.8257 | Weight-magnitude term widens dynamic range → MQ4 G=256 saturates |
| α=0.3 (less AWQ) | 0.1514 | Below the U-curve minimum |
| α=0.7 (more AWQ) | 0.1663 | Above the U-curve minimum |
| α=1.0 (max activation weighting) | 0.2032 | Far above the U-curve minimum |
| no-AWQ + GPTQ only | 0.2686 | Confirms AWQ contributes −53% lift |
