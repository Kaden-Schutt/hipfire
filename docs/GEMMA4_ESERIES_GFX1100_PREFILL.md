# Gemma 4 E-Series gfx1100 Prefill Batch

## Result

On one W7900/gfx1100, increasing `HIPFIRE_GEMMA4_PREFILL_BATCH` from 8 to 64 substantially improves Q8 prefill while leaving decode throughput unchanged.

| Model | Batch 8 | Batch 16 | Batch 32 | Batch 64 | B64/B8 |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 418.265 tok/s | 689.175 tok/s | 994.240 tok/s | 1,171.270 tok/s | 2.80x |
| Gemma 4 E4B Q8 | 289.890 tok/s | 483.580 tok/s | 685.805 tok/s | 861.195 tok/s | 2.97x |

The sweep used ten fixed GSM8K 8-shot prompts per configuration, greedy decoding, a 16-token output cap, and a fresh daemon for every batch size. Output text hashes matched across all four batch sizes for all ten prompts on both models.

A second gate ran the first 30 GSM8K prompts with the normal 4,096-token output cap at batch 64. Relative to the existing batch-8 full run, both models had 30/30 byte-identical outputs and unchanged correctness decisions:

| Model | Median prefill, B64 | Median decode, B64 | Accuracy | Runtime errors |
|---|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 1,143.610 tok/s | 100.670 tok/s | 80.0% | 0 |
| Gemma 4 E4B Q8 | 856.420 tok/s | 68.140 tok/s | 90.0% | 0 |

## Dispatch Boundary

The Q8 projection path already selects the wave32 `gemm_q8_0_wmma` kernel on gfx1100. The main loss at batch 8 is under-filled 16-row WMMA tiles plus repeated per-chunk fixed overhead. This change therefore selects batch 64 by default only on exact `gfx1100`. `HIPFIRE_GEMMA4_PREFILL_BATCH` remains authoritative, and all unvalidated architectures retain the existing batch-1 default.

Reproduce with `scripts/bench-gemma4-gfx11-prefill-batch.sh`.

## Q8 Projection Fusion Probe

The existing gfx1100 fused Q8 WMMA kernels can stage one shared activation for QKV or gate/up projections. The Gemma path admits them only behind `HIPFIRE_GEMMA4_Q8_FUSED_PREFILL=1`; shared-KV layers remain Q-only, and the flag is ignored outside exact gfx1100.

Three repeats of the same ten-prompt batch-64 workload produced the following medians:

| Model | Fusion | Prefill tok/s | TTFT ms | Decode tok/s |
|---|---|---:|---:|---:|
| Gemma 4 E2B Q8 | off | 1,173.600 | 650.165 | 103.900 |
| Gemma 4 E2B Q8 | on | 1,182.945 | 645.754 | 103.230 |
| Gemma 4 E4B Q8 | off | 864.170 | 883.111 | 68.970 |
| Gemma 4 E4B Q8 | on | 869.055 | 875.057 | 68.235 |

Prefill improved by 0.80% on E2B and 0.57% on E4B; all 60 paired outputs were byte-identical. The gain is positive but too small to promote the route to a default. The flag remains an opt-in characterization path while the larger 64x64 four-wave WMMA dispatcher is evaluated.

Reproduce the paired runs with:

```bash
FUSED_Q8_PREFILL=0 BATCHES=64 LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
FUSED_Q8_PREFILL=1 BATCHES=64 LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
```

## Larger Q8 WMMA tile probes

The existing 64x64 four-wave Q8 WMMA kernel was tested first and rejected for Gemma 4: E2B prefill fell from 1,178.06 to 1,078.88 tok/s (-8.42%), while E4B fell from 869.87 to 805.65 tok/s (-7.38%). All 60 paired outputs were byte-identical, so this is a shape-specific performance boundary rather than a correctness failure.

The single-wave 16x64 kernel was also rejected. E2B fell from 1,178.06 to 1,154.79 tok/s (-1.98%), while E4B moved from 869.87 to 870.38 tok/s (+0.06%, effectively neutral). Its 60 paired outputs were also byte-identical. Neither route is retained in production dispatch; the existing single-wave 16x16 Q8 WMMA remains the gfx1100 default for these Gemma projection shapes.

Raw paired artifacts are preserved under `target/validation/gemma4-gfx11-prefill-4w/{off-r3,on-r3}` and `target/validation/gemma4-gfx11-prefill-x64/on-r3`.

## Batched PLE projection probe

E-series prefill originally projected each row into the packed per-layer-input buffer with a separate GEMV, re-streaming the same Q8 model-projection matrix once per row. `HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL=1` replaces those row-wise calls with one batched projection on exact gfx1100 while preserving the embedding, normalization, and fallback paths.

Paired three-repeat runs show a consistent gain that grows with the prefill batch size:

| Model | Batch | Flag off | Flag on | Prefill delta | TTFT delta |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 8 | 414.510 tok/s | 416.195 tok/s | +0.41% | -0.44% |
| Gemma 4 E4B Q8 | 8 | 283.055 tok/s | 284.610 tok/s | +0.55% | -0.53% |
| Gemma 4 E2B Q8 | 64 | 1,178.060 tok/s | 1,202.930 tok/s | +2.11% | -2.14% |
| Gemma 4 E4B Q8 | 64 | 869.870 tok/s | 885.415 tok/s | +1.79% | -1.68% |

All 120 paired generated outputs were byte-identical. The direct sequential-versus-batched correctness harness also passed at `B=1,2,8,64` for both models: every last-token and post-batch KV-check argmax matched, and the minimum observed logit cosine was `0.9999020`. Batch 1 deliberately retains the original GEMV route. The optimization remains opt-in because the batch-8 gain is small and the WMMA route is not bit-identical to the F32 GEMV reference.

Reproduce the timing runs with:

```bash
PLE_BATCHED_PREFILL=0 BATCHES="8 64" LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
PLE_BATCHED_PREFILL=1 BATCHES="8 64" LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
```

Reproduce the direct logit/KV gate with either E-series Q8 artifact:

```bash
HIP_VISIBLE_DEVICES=1 HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL=1 \
  target/release/examples/verify_batch_gemma4 \
  --model /path/to/gemma4-e2b-or-e4b-q8.hfq --bs 1,2,8,64
```

Raw timing artifacts are preserved under `target/validation/gemma4-gfx11-ple-batched/{b8-off-r3,b8-on-r3,on-r3}`; the matching batch-64 baseline is `target/validation/gemma4-gfx11-prefill-4w/off-r3`.

## Batched PLE branch projections

The larger E-series-specific gap was inside every layer's PLE branch. Prefill issued one `input_gate` GEMV and one output `projection` GEMV per row, so batch 64 re-read both Q8 matrices and launched both kernels 64 times per layer. `HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL=1` routes both projections through the existing batched Q8 dispatcher on exact gfx1100 when `B > 1`; all other architectures, dtypes, and batch 1 keep the row-wise path.

Three-repeat paired measurements show that eliminating these per-row launches is the dominant gfx1100 E-series prefill optimization:

| Model | Batch | Flag off | Flag on | Prefill delta | TTFT delta |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 8 | 414.510 tok/s | 443.485 tok/s | +6.99% | -6.56% |
| Gemma 4 E4B Q8 | 8 | 283.055 tok/s | 296.890 tok/s | +4.89% | -4.63% |
| Gemma 4 E2B Q8 | 64 | 1,172.740 tok/s | 1,725.170 tok/s | +47.11% | -32.46% |
| Gemma 4 E4B Q8 | 64 | 864.560 tok/s | 1,212.715 tok/s | +40.27% | -28.99% |

The 16-token paired gate produced 60/60 byte-identical outputs at both batch sizes. Direct sequential-versus-batched validation passed at `B=1,2,8,64` for E2B and E4B: every last-token and post-batch KV-check argmax matched, with minimum observed logit cosine `0.9998514`.

A separate 30-question GSM8K gate used a 4,096-token output cap. The generated wording diverged on 5/30 prompts for each model, as expected from a non-bit-identical WMMA path over long greedy trajectories, but every extracted prediction and correctness decision matched the row-wise baseline. Accuracy remained 80% for E2B and 90% for E4B. Median prefill improved from 1,143.610 to 1,702.915 tok/s (+48.91%) on E2B and from 856.420 to 1,194.965 tok/s (+39.53%) on E4B; decode throughput was unchanged within normal run variance.

The smaller model-projection probe and this branch probe are intentionally not composed. Enabling both produced one short-output trajectory divergence in the characterization set; when both flags are present, the higher-value branch route takes precedence and the model projection retains its row-wise GEMV.

Reproduce the paired branch measurements with:

```bash
PLE_BRANCH_BATCHED_PREFILL=0 BATCHES="8 64" LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
PLE_BRANCH_BATCHED_PREFILL=1 BATCHES="8 64" LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
```

Raw artifacts are under `target/validation/gemma4-gfx11-ple-branch-batched/{branch-off-r3,branch-on-r3,b8-branch-on-r3,branch-on-full30}`.
