# Sub-0.10 KLD MQ4 via AWQ+GPTQ — Investigation 2026-05-18

**Goal**: produce a 9B Qwen3.5 MQ4G256 model with c512 q8-KV prefill KLD **below 0.10**, preserving flat-MQ4 wire format (5.0 GB on disk) and inference performance (no K-map promotions, no Q8 lifts on input/MLP weights, default Q8 conv1d only).

**Status**: in progress. Best result so far is `awq-aware-gptq-v3` at **KLD 0.1257 / PPL 9.31 / 5.0 GB**. Closing the remaining 0.0257 KLD gap to <0.10 via iterative AWQ+GPTQ rounds.

## Quick links

- [methodology.md](methodology.md) — full investigation arc, math, what worked / what didn't and why
- [results.md](results.md) — running KLD table across all variants on hiptrx gfx1201 c512 q8 prefill
- [repro-recipe.md](repro-recipe.md) — exact commands to regenerate the v3 winner + iterative pipeline
- [branch-state.md](branch-state.md) — code locations: which branch holds which commit
- [alpha-sweep.md](alpha-sweep.md) — 5-point α sensitivity curve, raw data + interpretation

## Current best (apples-to-apples, c512 q8 prefill, gfx1201, BF16 ref `qwen3.5-9b-bf16.kldref.bin`)

| Variant | KLD ± 95% CI | p99 | PPL | Size | Disk vs flat-mq4 |
|---|---:|---:|---:|---:|---:|
| **awq-aware-gptq-v3** | **0.1257 ± 0.006** | 13.7 | 9.31 | 5.0 GB | **+0 bytes** |
| cand151-gptq-all-compatible | 0.1565 ± 0.008 | 14.2 | 9.20 | 5.56 GB | +560 MB |
| kmd2-q8conv1d | 0.1605 ± 0.008 | 15.2 | 9.17 | 6.43 GB | +1.43 GB |
| flat-mq4 (baseline) | 0.3215 ± 0.012 | 18.7 | 8.72 | 5.0 GB | 0 |
| Q8F16 (engine floor) | 0.0186 | 1.8 | 9.26 | 9.53 GB | +4.53 GB |

v3 is **−61% KLD vs flat-mq4 at +0 bytes on disk**. Beats prior best `cand151-gptq-all-compatible` by **−20% KLD at 90% of cand151's size**.

## Hardware + environment

- **Bench machine**: hiptrx (4× AMD Radeon AI PRO R9700, gfx1201, ROCm via `hipfire-rocm` conda env)
- **Python env path**: `/home/kaden/miniforge3/envs/hipfire-rocm/bin/python`
- **Reference**: `qwen3.5-9b-bf16.kldref.bin` (top-K=256, n_ctx=2048, 1175 chunks) at `~/hipfire/.worktrees/HIPa/benchmarks/quality-baselines/refs/`
- **Engine binary state**: `awq-kmap-bench` worktree at origin/master HEAD post-PR-#273 (`a99b4643` or later) plus iterative pipeline overlay
- **Imatrix**: unsloth-published Qwen3.5-9B imatrix at `~/.hipfire/imatrix/unsloth/Qwen3.5-9B-GGUF/imatrix_unsloth.gguf_file`
- **Calibration corpus**: `benchmarks/calib/calib-1m.txt`
- **Calibration Hessian** (reused across all GPTQ experiments): `~/hipfire/.worktrees/paroquant/.codeinsight+research/astrea/mq4-gptq-9b-poc/20260515T-start/hessian-linear-c64-ctx256/stats-merged.npz` (814 MB, c64 chunks at ctx=256)

## Reproducibility commitments

1. **All KLD numbers** in this investigation are c512 q8-KV prefill on gfx1201 (R9700) against the BF16 reference dump above. No mixing of c256/c512, no mixing of KV modes, no mixing of arches.
2. **Hessian re-collection**: NOT necessary — the existing c64 Hessian is used for all paper-formula AWQ+GPTQ experiments. Iterative rounds collect their own per-round Hessian using the partial-quantized model.
3. **Imatrix re-collection**: NOT necessary for one-shot pipelines (uses unsloth's published imatrix). Iterative re-collects internally.
4. **Engine binary**: must match origin/master HEAD post-PR-#273. Pre-PR-#266 binaries cannot load AWQ sidecars; pre-PR-#273 binaries cannot dispatch the F2 output-side AWQ kernels (immaterial for v3 since v3 uses F1-scope only).
