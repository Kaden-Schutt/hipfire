<!-- SPDX-License-Identifier: Apache-2.0 -->
# DeepSeek-V4 independent reference oracle

Run the DeepSeek V4 Flash reference `model.py` **verbatim** under PyTorch as an
independent execution oracle for the Rust/GPU parent port.

## Why this exists

Every previous oracle shared a reading of `model.py` with the parent port. When
`Block.hc_post` contracted the wrong axis of the sinkhorn `comb` matrix, GPU and
`*_ref` agreed to ~1e-7 while the model was badly broken (PPL 163.89 vs 14.70).
This harness imports `model.py` unmodified so no human re-reading sits between
the file and the numbers.

## What it covers

| Step | What | Pass criterion |
|------|------|----------------|
| 1 | Floors: Linear fp8 path, **f32-diagnostic** and **bf16-fidelity** | State both numbers; do not invent a tolerance |
| 2a | `hc_post` fixed contraction vs `parent_hc_post_ref` (pure f32) | `max_abs` near 0 |
| 2b | Same with **deliberately transposed** `comb` | Must fail loudly (`max_abs ≫ fixed`) |
| 3 | Layer-0 stage bisect: hc_pre, attn_norm, attn internals, hc_post, ffn, out | First stage leaving its floor |
| 4 | Residual L2 after layers 0..L on the reference | Curve vs parent post-fix 1.168 geo |

## What it does **not** cover

- Full 43-layer PPL / production serving parity.
- Bit-exact match to tilelang GPU kernels (shim is naive f32 dequant+matmul).
- MTP / DSpark paths (`n_mtp_layers=0` in the harness).
- Tensor-parallel (`world_size=1` only).
- Direct import of Rust `parent/*` — `parent_hc_post_ref.py` is a pure-Python
  transcription of the **fixed** contraction for the deliberate-bug check only.
- Layers with `compress_ratio>0` in the stage-3 bisect (L0 is ratio=0 / pure SWA).
  Trajectory (step 4) does run compressed layers end-to-end via `model.py`.
- Cross-check against a live GPU parent dump is optional (`--parent-stages`);
  without it, step 3 reports reference-only L2s and internal self-checks.

## Judgement calls in `kernel_shim.py` (read before trusting a number)

1. **`fp8_gemm` / `fp4_gemm`**: dequantize fully to f32, then one matmul. The
   tilelang kernels accumulate unscaled products per 128/32 block then apply
   scales (two-accumulator form). Floor step 1 quantifies that difference.
2. **`fp4_act_quant` non-inplace packing**: nearest-magnitude E2M1, not a full
   RNE-ties-to-even bit mirror. **Inplace** path (indexer / compressor) only
   needs dequantized BF16 and is unaffected by packing.
3. **`sparse_attn`**: standard softmax over gathered KVs + sink logit, not the
   Flash-style online running max/sum. Algebraically equivalent in f32 for a
   fixed top-k set; ordering of ties may differ. Sink enters the denominator
   only (matches `kernel.py:345-348`).
4. **`hc_split_sinkhorn`**: matches kernel.py control flow exactly
   (row-softmax+eps → col-norm → `(iters-1)`×(row,col), **ends on col**;
   `post = 2*sigmoid`). Implemented with `torch.softmax` / sum, not the tiled
   kernel’s parallel reductions.
5. **`wo_a` load**: checkpoint stores FP8; `convert.py` dequants to BF16 because
   `Attention.forward` einsums it. We do the same on load.
6. **`route_scale`**: taken from `config.json` (`1.5`), never env `2.2`.
7. **Device**: default **CPU**. Do not pass `--device cuda` while another agent
   holds parent captures on the GPU.
8. **L0 attention bisect**: at 128 tokens, `window_size=128` means SWA covers the
   whole sequence (causal). `compress_ratio=0` so no compressor/indexer path.
   (`index_topk=512` is irrelevant on L0.)
9. **RoPE on L0**: `original_seq_len=0`, `rope_theta=10000` (YaRN disabled).
   Compressed layers use `compress_rope_theta=160000` + YaRN — only exercised in
   step-4 trajectory, not the L0 stage table.
10. **Floors are per-domain**: pure-f32 stages (hc_post, softmax, rope) floor near
    0; fp8 Linear stages floor at the step-1 number. Never apply the Linear floor
    as a blanket tolerance on an f32 stage.

## Reproduce

### 1. Environment (mi300x)

```bash
# ROCm wheel already at /mnt/scratch/torch_oracle_rocm (still run --device cpu
# unless the GPU is free AND you coordinated over hub).
export PYTHONPATH=/mnt/scratch/torch_oracle_rocm

python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.13.0+rocm7.2 True   (use --device cpu anyway while captures run)
```

Fresh install if needed:

```bash
python3 -m pip install --target=/mnt/scratch/torch_oracle_site   torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --target=/mnt/scratch/torch_oracle_site safetensors numpy

# or ROCm 7.2:
python3 -m pip install --target=/mnt/scratch/torch_oracle_rocm   torch --index-url https://download.pytorch.org/whl/rocm7.2
python3 -m pip install --target=/mnt/scratch/torch_oracle_rocm safetensors numpy
```

### 2. Layout (local worktree is source of truth)

```
crates/hipfire-arch-deepseek4/reference_oracle/
  README.md                 # this file
  kernel_shim.py            # eager kernel.py replacements
  weight_loader.py          # HF safetensors → model.py modules
  parent_hc_post_ref.py     # fixed hc_post formula + transpose switch
  run_oracle.py             # gates 1–4
  fast_hadamard_transform/  # shim for model.py rotate_activation import
  model.py -> …/ds4-parent-ref/inference/model.py   # symlink, NOT a copy
  config.json -> …/inference/config.json
```

`model.py` is imported **unmodified**. The only substitution is
`sys.modules["kernel"] = kernel_shim` before import. No lines of `model.py`
are edited.

Sync to the box:

```bash
rsync -av --delete \
  crates/hipfire-arch-deepseek4/reference_oracle/ \
  mi300x:/root/hipfire-work/ds4-parent-gate/crates/hipfire-arch-deepseek4/reference_oracle/
# also keep the scratch harness in sync if you use it:
rsync -av crates/hipfire-arch-deepseek4/reference_oracle/ \
  mi300x:/mnt/scratch/torch_oracle_harness/
```

### 3. Run

```bash
cd /root/hipfire-work/ds4-parent-gate/crates/hipfire-arch-deepseek4/reference_oracle
# or: cd /mnt/scratch/torch_oracle_harness
export PYTHONPATH=/mnt/scratch/torch_oracle_rocm:$PWD

# Full gates 1–4, layers 0..6, 128 tokens, CPU
nohup setsid python3 run_oracle.py \
  --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
  --tokens /mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens_128.bin \
  --device cpu --layers 7 --seq 128 \
  --parent-stages /tmp/parent_stages_only.txt \
  --out /tmp/torch_oracle_summary.json \
  > /tmp/torch_oracle_all.log 2>&1 < /dev/null &

# Floor only (fast)
python3 run_oracle.py --step 1 --device cpu

# hc_post teeth only
python3 run_oracle.py --step 2 --device cpu --layers 1

# Layer-0 stage bisect
python3 run_oracle.py --step 3 --device cpu --layers 1
```

Runtime (CPU, 128 tok): step1 ~seconds; step2+3 layer-0 with MoE ~minutes;
step4 ~7s/layer once weights are resident. Peak RAM dominated by expert FP4
weights (~1.5–2 GB/layer with experts).

### 4. Token fixtures (sha256)

| file | sha256 |
|------|--------|
| tokens_128.bin | `84f8c3f04e7876c4f37d59652217e13c42969f034e2508ee60a87871cd10ac20` |
| tokens_256.bin | (nested prefix of 512/1024) |
| tokens_512.bin | (nested prefix of 1024) |
| tokens.bin (1024) | (full) |

## Constants (all from config.json / tensors — never env)

- `route_scale = 1.5`
- `post = 2 * sigmoid(...)` (not production `post_scale=1.5`)
- `hc_sinkhorn_iters = 20`, `hc_mult = 4`, `swiglu_limit = 10.0`
- `score_func = sqrtsoftplus`, `n_activated_experts = 6`
- L0: `compress_ratio=0`, `window_size=128`, plain `rope_theta=10000`

## Landed findings (2026-08-02)

### RoPE frequency tables — CLEARED

See `ROPE_TABLE_DIFF.md`. Parent `precompute_rope_freqs` matches `model.py
precompute_freqs_cis` to max relative error ~7e-8 on all three configs
(ratio-0 plain 10k, ratio>0 YaRN 160k, plain-160k counterfactual). Max phase
error at pos 1000 is ~2e-5 rad — not a defect. Indexer shares the YaRN table
(code correct; stale "plain" comment in indexer.rs header is wrong).

### Full 43-layer residual trajectory — FLAT across position

See `RESIDUAL_POS_TRAJ.md` and `artifacts/residual_pos_traj.json.compact.json`.

```bash
# on mi300x with GPU free (~5 min, ~4.7 GiB VRAM peak):
export PYTHONPATH=/mnt/scratch/torch_oracle_rocm:/mnt/scratch/torch_oracle_harness
python3 -u residual_pos_traj.py --seq 1024 --layers all --out /tmp/residual_pos_traj.json
```

Streams one `Block` at a time (imports `model.py` verbatim). Dense per-row HC
L2 after every layer. **mean late/early excluding pos0 over 43 layers = 1.026**.
Reference residual does **not** degrade with position; the parent top-1
collapse after ~448–512 is not residual-growth-with-position in the reference.
L38 pos0/median = 269× (aggregate residual-growth story already retired).

### Index-space / cardinality — CLEARED at seq≤1024

See `INDEX_SPACE_CONVENTIONS.md`. `index_topk` applies only to compressed half;
SWA window exempt. At 1024 tokens top-k is a no-op (n_comp ≤ 256 < 512).

### Next localiser

Compressed-KV content / joint softmax value path / SWA staging ≡ abs-gather
equivalence — per-row attention outputs vs parent dumps, not residual L2.
