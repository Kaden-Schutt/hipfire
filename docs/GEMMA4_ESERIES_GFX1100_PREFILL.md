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
