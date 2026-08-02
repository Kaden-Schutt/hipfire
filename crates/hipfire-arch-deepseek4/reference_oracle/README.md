<!-- SPDX-License-Identifier: Apache-2.0 -->
# DeepSeek-V4 independent reference oracle

Run the DeepSeek V4 Flash reference `model.py` **verbatim** under PyTorch and
compare it against the Rust/GPU parent port layer by layer.

## Why this exists

Every previous oracle shared a reading of `model.py` with the parent port. When
`Block.hc_post` contracted the wrong axis of the sinkhorn `comb` matrix, GPU and
`*_ref` agreed to ~1e-7 while the model was badly broken (PPL 163.89 vs 14.70).
This harness imports `model.py` unmodified so no human re-reading sits between
the file and the numbers.

## What it covers

| Step | What | Pass criterion |
|------|------|----------------|
| 1 | Floor: `Linear` fp8 path vs explicit dequant+matmul on the same weight | State the number; do not invent a tolerance from it |
| 2a | Layer-0 `Block.forward` on `tokens_128.bin` | Report residual L2 + (optional) parent dump diff |
| 2b | `hc_post` fixed contraction vs `parent_hc_post_ref` | `max_abs` near 0 (exact f32 algebra) |
| 2c | Same with **deliberately transposed** `comb` | Must fail loudly (`max_abs ≫ floor`) |
| 3 | Residual L2 after layers 0..L on the reference | Curve vs pre-fix parent 494…1188 |

## What it does **not** cover

- Full 43-layer PPL / production serving parity.
- Bit-exact match to tilelang GPU kernels (shim is naive f32 dequant+matmul).
- MTP / DSpark paths (`n_mtp_layers=0` in the harness).
- Tensor-parallel (`world_size=1` only).
- The Rust `parent/*` modules are not imported; `parent_hc_post_ref.py` is a
  pure-Python transcription of the **fixed** contraction formula for the
  deliberate-bug check only.

## Judgement calls in `kernel_shim.py` (read before trusting a number)

1. **`fp8_gemm` / `fp4_gemm`**: dequantize fully to f32, then one matmul. The
   tilelang kernels accumulate unscaled products per 128/32 block then apply
   scales (two-accumulator form). Floor step 1 quantifies that difference.
2. **`fp4_act_quant` non-inplace packing**: nearest-magnitude E2M1, not a full
   RNE-ties-to-even bit mirror. **Inplace** path (used by indexer / compressor)
   only needs dequantized BF16 and is unaffected by packing.
3. **`sparse_attn`**: standard softmax over gathered KVs + sink logit, not the
   Flash-style online running max/sum. Algebraically equivalent in f32 for a
   fixed top-k set; ordering of ties may differ.
4. **`hc_split_sinkhorn`**: matches kernel.py control flow exactly
   (row-softmax+eps → col-norm → `(iters-1)`×(row,col), **ends on col**;
   `post = 2*sigmoid`). Implemented with `torch.softmax` / sum, not the tiled
   kernel’s parallel reductions.
5. **`wo_a` load**: checkpoint stores FP8; `convert.py` dequants to BF16 because
   `Attention.forward` einsums it. We do the same on load.
6. **`route_scale`**: taken from `config.json` (`1.5`), never env `2.2`.
7. **Device**: default **CPU**. ROCm torch may be installed; do not pass
   `--device cuda` while another agent holds the parent on GPU.

## Reproduce

### 1. Environment (mi300x)

```bash
# Isolated site-packages (no system pip clobber). CPU wheel is enough;
# ROCm wheel is optional and currently at /mnt/scratch/torch_oracle_rocm.
export PYTHONPATH=/mnt/scratch/torch_oracle_site:/mnt/scratch/torch_oracle_venv/lib/python3.12/site-packages
# or ROCm build (still run --device cpu unless GPU is free):
# export PYTHONPATH=/mnt/scratch/torch_oracle_rocm

python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.13.0+cpu False   OR   2.13.0+rocm7.2 True
```

If you need a clean install:

```bash
# CPU (small, preferred while parent holds VRAM)
python3 -m pip install --target=/mnt/scratch/torch_oracle_site \
  torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --target=/mnt/scratch/torch_oracle_site safetensors numpy

# ROCm 7.2 (optional; ships its own runtime libs)
python3 -m pip install --target=/mnt/scratch/torch_oracle_rocm \
  torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
python3 -m pip install --target=/mnt/scratch/torch_oracle_rocm safetensors numpy
```

### 2. Layout

```
crates/hipfire-arch-deepseek4/reference_oracle/
  README.md                 # this file
  kernel_shim.py            # eager kernel.py replacements
  weight_loader.py          # HF safetensors → model.py modules
  parent_hc_post_ref.py     # fixed hc_post formula + transpose switch
  run_oracle.py             # gates 1–3
  fast_hadamard_transform/  # shim for model.py rotate_activation import
  model.py -> …/ds4-parent-ref/inference/model.py   # symlink, NOT a copy
  config.json -> …/inference/config.json
```

`model.py` is imported **unmodified**. The only substitution is
`sys.modules["kernel"] = kernel_shim` before import. No lines of `model.py`
are edited.

### 3. Run

```bash
cd crates/hipfire-arch-deepseek4/reference_oracle
export PYTHONPATH=/mnt/scratch/torch_oracle_site:/mnt/scratch/torch_oracle_venv/lib/python3.12/site-packages:$PWD

# Full gates 1–3, layers 0..6, 128 tokens, CPU
python3 run_oracle.py \
  --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
  --tokens /mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens_128.bin \
  --device cpu --layers 7 --seq 128 --out /tmp/torch_oracle_summary.json

# Floor only (fast)
python3 run_oracle.py --step 1 --device cpu

# hc_post + layer 0 only
python3 run_oracle.py --step 2 --device cpu --layers 1
```

### 4. Token fixtures (sha256)

| file | sha256 |
|------|--------|
| tokens_128.bin | `84f8c3f04e7876c4f37d59652217e13c42969f034e2508ee60a87871cd10ac20` |
| tokens_256.bin | `0b747dfb…` |
| tokens_512.bin | `f02a2a61…` |
| tokens.bin (1024) | `48b0f834…` |

## Pre-fix parent residual L2 (1024-tok, for comparison)

```
494.18, 474.71, 483.46, 482.98, 486.40, 777.97, 1188.70
```

Reference trajectory is printed next to these by step 3.

## Constants (all from config.json / tensors — never env)

- `route_scale = 1.5`
- `post = 2 * sigmoid(...)` (not production `post_scale=1.5`)
- `hc_sinkhorn_iters = 20`, `hc_mult = 4`, `swiglu_limit = 10.0`
- `score_func = sqrtsoftplus`, `n_activated_experts = 6`
