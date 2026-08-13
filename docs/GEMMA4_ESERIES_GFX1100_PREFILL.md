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
