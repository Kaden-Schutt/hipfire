# ParoQuant G256 Milestone

This branch is a reconciliation staging branch built on the ParoQuant graph
capture stack from PR #318. It adds the local Hipfire ParoQuant investigation
artifacts and a CPU-only G256 probe so we can decide whether MQ4-class storage
is worth integrating before opening a larger PR.

## Current Runtime Baseline

PR #318 already has the active runtime path:

- load ParoQuant safetensors directly
- repack AWQ INT4 weights to `HFQ4G128`
- attach `ParoRotation` sidecars
- run `DType::ParoQ4G128`
- support graph-capturable MoE routing for the A3B Paro checkpoints

This branch also carries the older local qtype-28/qtype-29 HFQ probe path
forward for reconciliation:

- `scripts/paroquant_import.py` / `scripts/paroquant_oracle.py`
- `DType::PARO4G128` and `DType::PARO4G128T`
- `kernels/src/gemv_paro4g128.hip`
- `crates/rdna-compute/examples/test_gemv_paro4g128.rs`
- qtype-28/qtype-29 load and dispatch hooks in the runtime/engine

That path is still investigation/probe wiring, not a replacement for PR #318's
newer safetensors `ParoQ4G128` route. Keep both visible until the G256 gate
decides whether to invest in a production `PARO4G256_MQ` runtime.

## Format Decision Gate

The old local investigation found that native ParoQuant quality is materially
better than MQ4 on Qwen3.5-0.8B, but the current G128 storage and per-linear
rotation path was much slower than MQ4. The proposed path to MQ4-class speed is:

1. `PARO4G256`: true ParoQuant calibration/export at group size 256.
2. `PARO4G256_MQ`: Paro rotation side metadata plus Hipfire row-major G256 W4
   body so existing HFQ4/MQ4 kernel families can be reused downstream.

Do not treat a regrouped G128 checkpoint as proof of true `PARO4G256` quality.
If upstream ParoQuant cannot emit a real G256 checkpoint, label the G256 quality
result `UNVERIFIABLE`.

## CPU-Only Probe

The probe below does not use the GPU. It dequantizes the cached G128 Paro
weights, simulates G256 storage choices, and reports format-loss against the
source Paro output oracle.

```bash
python3 scripts/paroquant_g256_probe.py \
  --model /home/kaden/.cache/huggingface/hub/models--z-lab--Qwen3.5-0.8B-PARO/snapshots/1ed64e6caaf66c63f98422cffba2e8691d867699 \
  --local-only \
  --max-modules 6 \
  --samples 4 \
  --pretty
```

The first 6-module run on the local probe branch showed:

| Variant | Avg output NRMSE vs source Paro | Worst output NRMSE | Avg payload ratio vs source |
|---|---:|---:|---:|
| `PARO4G256` AWQ regroup | 0.0859 | 0.1095 | 0.9817x |
| `PARO4G256_MQ` row-major G256 body + Paro side metadata | 0.0951 | 0.1114 | 1.0220x |

This is useful negative-pressure evidence, not a final quality verdict.

## Contributor Ask

Before we reconcile this into a main PR:

1. Confirm whether upstream ParoQuant can calibrate/export true
   `group_size=256` checkpoints.
2. If yes, generate a Qwen3.5-0.8B `PARO4G256` checkpoint and run the same
   PPL/KLD cohort used in the local investigation.
3. If no, mark `PARO4G256` quality `UNVERIFIABLE` and evaluate either:
   - keep PR #318 native `ParoQ4G128` + graph capture as the near-term path, or
   - design `PARO4G128_MQ` kernels that preserve G128 quality at a larger model
     size.
4. Only after the G256 quality gate passes should we implement
   `PARO4G256_MQ` runtime wiring.

## GPU Safety

Do not launch GPU benches while another agent is using the GPU. Check first:

```bash
rocm-smi --showuse --showmemuse --showpids
```

If any KFD process is present, keep work CPU-only.
