# MI300X Rental Runbook — DigitalOcean

One-page reference for spinning up a 1× MI300x droplet on DigitalOcean and
running the v3 model sweep + sub-0.10 KLD attempt. Estimated total cost:
**$25-40** for ~10-13 hours of dedicated compute.

## Why this runbook exists

Running the v3 recipe across the full Qwen3.5/3.6 model lineup on hiptrx
requires queuing against a shared workstation (4× R9700, 16GB each, TP-
required for BF16 27B). On 1× MI300x (192GB HBM3) every model fits solo,
the iterate pipeline runs 4× faster, and we get production-side validation
on a CDNA3 target. Total clock: ~10h compressed from hiptrx's ~30h
contested. Plus MI300x is a real deploy target for hipfire (CHANGELOG: 10
hot HFQ4 kernels wave64-ported, matches 7900 XTX baseline).

## Droplet config

| Setting | Value |
|---|---|
| Region | NYC2 / SFO3 / TOR1 (any with MI300x stock) |
| Size | 1× MI300x (192 GB HBM3) — `gpu-1m-192gb` (or DO's current sku name) |
| Image | **PyTorch 2.6.0 + ROCm 7.0.0** (NOT the ROCm 7.2 image — saves 30 min of PyTorch install) |
| Storage | Default + a **block volume ≥ 500 GB** attached at `/workspace` |
| SSH key | Your key |

**Cost:** ~$2.50-3/hr typical for 1× MI300x. Storage is ~$0.10/GB-mo, so
500GB ≈ $50/mo prorated to $1.60 for 12h.

## One-time prep (already done locally)

These are tracked in `iterative-awq-gptq` + `worktree-awq-raw-sumsq-converter`
branches on origin:
- `scripts/convert_gguf_imatrix_to_npz.py` — GGUF imatrix → raw-sumsq npz
- `scripts/mq4_masked_calib.py` — `--awq-raw-sumsq-npz` flag, F1-default scope
- `scripts/mi300x_bootstrap.sh` — this runbook's automation
- `scripts/mi300x_smoke_gfx942.sh` — pre-flight verification
- `scripts/mi300x_v3_matrix.sh` — model sweep
- `scripts/mi300x_sub_0_10_attempt.sh` — final lever

## Step-by-step

### 1. Spin up the droplet (DO console)

DO Console → Droplets → New Droplet → choose MI300x sku + PyTorch+ROCm 7.0
template + attach block storage volume named `hipfire-work` → SSH key →
Create. Wait ~2 min for it to come up.

### 2. SSH in and mount the volume

```bash
ssh root@<droplet-ip>

# DO block volumes attach but don't mount by default. Check + mount:
lsblk
mkdir -p /workspace
mount /dev/disk/by-id/scsi-0DO_Volume_hipfire-work /workspace
df -h /workspace
```

### 3. Set HF token and clone the bootstrap script

You need a HuggingFace token with read access (since some gated repos may
require it). Get one at https://huggingface.co/settings/tokens.

```bash
export HF_TOKEN="hf_..."  # your token

# Fetch just the bootstrap script (avoids needing a full clone yet)
curl -sL https://raw.githubusercontent.com/Kaden-Schutt/hipfire/worktree-awq-raw-sumsq-converter/scripts/mi300x_bootstrap.sh \
    -o /tmp/mi300x_bootstrap.sh
chmod +x /tmp/mi300x_bootstrap.sh
```

### 4. Run bootstrap (~25 min)

```bash
bash /tmp/mi300x_bootstrap.sh 2>&1 | tee /workspace/bootstrap.log
```

This phase will:
- Install rust + apt deps
- Clone hipfire @ `worktree-awq-raw-sumsq-converter`
- Patch `compile-kernels.sh` to skip RDNA-only WMMA on gfx942
- Compile HIP kernels for gfx942 (~5 min)
- `cargo build --release` (~10 min)
- Install Python deps
- Download 5 BF16 models (~200 GB) — slowest step, ~15 min on a decent net
- Download 5 unsloth imatrix files (~225 MB)

Idempotent — if it dies partway, just re-run; phases skip if their outputs
already exist.

### 5. Smoke test (~5 min)

```bash
bash /workspace/hipfire/scripts/mi300x_smoke_gfx942.sh
```

Verifies:
- `eval_hipfire --print-arch` shows gfx942
- Dequant byte-correctness
- AR decode produces finite logits + on-topic continuation
- `test_inference` suite passes
- `coherence_probe --self-check` works

**If this fails, stop and report. Don't proceed to the matrix — it'd just
burn paid time on a broken stack.**

### 6. v3 matrix sweep (~7-8 h)

```bash
bash /workspace/hipfire/scripts/mi300x_v3_matrix.sh
```

Runs v3 recipe + KLD + coherence + tok/s on all 5 trunk models:
- Qwen3.5-0.8B, 9B, 27B
- Qwen3.6-27B, 35B-A3B

Per-model output: `$WORK/results/v3-matrix/<ts>/<slug>/`
- `stage1_awq.log` — AWQ pre-scale quantize
- `stage2_gptq.log` — AWQ-aware GPTQ
- `kld.json` — KLD + PPL @ c512 q8 prefill
- `coherence.json` — hard/soft fails
- `bench.json` — decode/prefill tok/s
- `summary.json` — rolled-up blob

Top-level: `$WORK/results/v3-matrix/<ts>/table.md` — comparable result table.

**Special: A3B router.** v3 includes the MoE router in F1 AWQ scope by
default. Per memory `project_mfp4_v3_1_moe_falsified_2026_05_11` this can
corrupt tool-call schema. The script excludes router for A3B by default; set
`A3B_INCLUDE_ROUTER=1` env to also try the inclusive variant.

### 7. Sub-0.10 KLD attempt (~40 min)

```bash
bash /workspace/hipfire/scripts/mi300x_sub_0_10_attempt.sh
```

Uses the `--awq-raw-sumsq-npz` lever (commit 59d6be49) to bypass the
rotated-Hessian-diagonal bug. 4 KM-damped iterate rounds against unsloth's
raw per-channel sumsq. Final output: `$WORK/results/sub-0-10/<ts>/verdict.md`.

**Realistic expectation:** rounds land in 0.10-0.15 range. Sub-0.10 is
plausible if the rotated-Hessian-diagonal was indeed the binding constraint,
not guaranteed if the calibration corpus is also a factor. The investigation
documents this as a "plausible-but-unproven" lever.

### 8. Extract results + tear down

```bash
cd /workspace
ts=$(ls -td results/v3-matrix/* | head -1 | xargs basename)
tar czf /tmp/hipfire-mi300x-${ts}.tar.gz \
    results/v3-matrix/${ts}/ \
    results/sub-0-10/ \
    bootstrap.log

# Download to local
# (run on local machine)
scp root@<droplet-ip>:/tmp/hipfire-mi300x-${ts}.tar.gz ./
```

Then power off the droplet. **Keep the block volume** if you might iterate;
delete it if done (block storage cost continues while attached).

## Budgets and stop conditions

| Phase | Wall time | Stop if... |
|---|---|---|
| Bootstrap | ~25 min | rocm-smi doesn't show MI300x; cargo build fails after WMMA patch |
| Smoke | ~5 min | test_inference hard-fails on a no-AWQ model |
| v3 matrix | ~7-8 h | All 3 small models fail in same way (kernel issue, not model issue) |
| Sub-0.10 | ~40 min | Iterate round 0 KLD > 0.30 (data-side broken) |

Hard ceiling: $100. If we're past that and the matrix hasn't completed, the
plan needs revision — power off and report.

## Comparison points (what to look for in the rolled-up table)

| Model | Expected v3 KLD | Comparison anchor |
|---|---|---|
| Qwen3.5-0.8B | ~0.10-0.13 | KMD2-q8conv1d on hiptrx was 0.0804 (different recipe though) |
| Qwen3.5-9B | 0.1257 | hiptrx anchor — should match within 1% |
| Qwen3.5-27B | unknown | first measurement on this branch |
| Qwen3.6-27B | unknown | first measurement |
| Qwen3.6-35B-A3B | unknown | first measurement; router-AWQ risk |

**Caveat:** these KLDs are produced by `eval_hipfire` against `kldref.bin`
generated on the MI300x (via PyTorch BF16 forward). Cross-machine comparison
with hiptrx numbers is only valid if the same BF16 source revision was used —
which is what `MODEL_REVISIONS` in bootstrap pins.

## When you're done

Two new branches to PR (or to leave as research overlays):
- `iterative-awq-gptq` — F1 default + iterate scaffolding + investigation docs
- `worktree-awq-raw-sumsq-converter` — raw-sumsq converter + iterate flag + this runbook
