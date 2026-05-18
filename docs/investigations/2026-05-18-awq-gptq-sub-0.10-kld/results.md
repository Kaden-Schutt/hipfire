# Results — Running KLD Table

All numbers are c512 q8-KV prefill on hiptrx (gfx1201) against `qwen3.5-9b-bf16.kldref.bin`.

## Final table (sorted by KLD)

| Rank | Variant | KLD ± 95% CI | p99 KLD | PPL | NLL mean | Size | Notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 🥇 | **awq-aware-gptq-v3** | **0.1257 ± 0.006** | 13.7 | 9.310 | 2.2313 | 5.00 GB | F1 AWQ + AWQ-aware GPTQ at α=0.5 |
| 2 | awq-aware-gptq-f2 | 0.1386 ± 0.007 | 14.4 | 9.501 | 2.2514 | 5.00 GB | F2 (input+output-side AWQ) at α=0.5 |
| 3 | awq-aware-gptq-f2-a055 | 0.1396 ± 0.007 | 14.5 | 9.323 | 2.2325 | 5.00 GB | F2 at PR-author's PPL sweet spot |
| 4 | f1-a030-gptq | 0.1514 ± 0.008 | 14.3 | 9.340 | 2.2343 | 5.00 GB | F1 α=0.3 — below U-curve min |
| 5 | cand151-gptq-all-compatible | 0.1565 ± 0.008 | 14.2 | 9.197 | 2.2189 | 5.56 GB | prior best, GPTQ-only mixed base |
| 6 | kmd2-q8conv1d | 0.1605 ± 0.008 | 15.2 | 9.173 | 2.2163 | 6.43 GB | K-map MQ4/MQ6 + Q8 conv1d, no AWQ no GPTQ |
| 7 | f1-a070-gptq | 0.1663 ± 0.008 | 15.3 | 8.965 | 2.1933 | 5.00 GB | F1 α=0.7 — above U-curve min |
| 8 | pr266-repro-q8 (AWQ alone) | 0.1867 ± 0.009 | 16.3 | 9.526 | 2.2541 | 5.00 GB | F1 AWQ no-GPTQ |
| 9 | f1-a100-gptq | 0.2032 ± 0.011 | 16.0 | 8.885 | 2.1844 | 5.00 GB | F1 α=1.0 — far above U-curve min |
| 10 | gptq-only-noawq | 0.2686 ± 0.011 | 17.9 | 8.905 | 2.1866 | 5.00 GB | flat-mq4 + GPTQ, no AWQ |
| 11 | flat-mq4 (baseline) | 0.3215 ± 0.012 | 18.7 | 8.715 | 2.1651 | 5.00 GB | no AWQ no GPTQ |
| (floor) | Q8F16 | 0.0186 ± 0.001 | 1.8 | 9.260 | 2.2259 | 9.53 GB | engine ceiling |

## Failed experiments (recorded so we don't repeat)

| Variant | KLD | PPL | Why broken |
|---|---:|---:|---|
| awq-gptq-stack (path 1) | 1.7634 | 49.86 | naive raw-x Hessian, source weights not pre-scaled |
| awq-aware-gptq-stack (path 2 v2) | 1.7531 | 49.14 | Hessian transformed but source still W (off by factor s) |
| autoawq-gptq-a050 | 1.8257 | 44.08 | weight-magnitude term widens dynamic range past MQ4 G=256 |

## Alpha sensitivity curve (paper formula + AWQ-aware GPTQ, c512 q8 prefill)

| α | KLD | PPL | p99 | Δ from v3 |
|---:|---:|---:|---:|---:|
| no AWQ | 0.2686 | 8.91 | 17.9 | +114% |
| 0.30 | 0.1514 | 9.34 | 14.3 | +20% |
| **0.50** | **0.1257** | **9.31** | **13.7** | **minimum** |
| 0.70 | 0.1663 | 8.97 | 15.3 | +32% |
| 1.00 | 0.2032 | 8.89 | 16.0 | +62% |

Convex U-shape on KLD. PPL monotonically decreases toward higher α (KLD-PPL inversion). v3 (α=0.5) is the global optimum.

## Per-tensor metric data (from GPTQ logs)

GPTQ writes per-tensor reconstruction metric (`metric=` field per tensor). v3's metrics show variance across tensors — some are 10× lower error than others. The per-tensor variance is the signal that motivates iterative AWQ+GPTQ.

(Will populate detailed per-tensor table from `round_*/candidate.json` artifacts when iterative completes.)

## Iterative AWQ+GPTQ rounds (in progress)

| Round | KLD c512 | PPL | scale-delta vs prev | Wall time | Notes |
|---:|---:|---:|---:|---:|---|
| 0 (= v3) | 0.1257 | 9.31 | — | 5 min | one-shot baseline |
| 1 | ⏳ | ⏳ | ⏳ | ⏳ | imatrix re-collected on Q⁽⁰⁾ |
| 2 | ⏳ | ⏳ | ⏳ | ⏳ | |
| 3 | ⏳ | ⏳ | ⏳ | ⏳ | |
| 4 | ⏳ | ⏳ | ⏳ | ⏳ | last round (max-rounds=4) |

Damping β=0.5, ε=0.01 stopping criterion. Will populate as monitor fires.
