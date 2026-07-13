# FWHT3 attention GQA reuse: cache already captures it; LDS staging regresses

## Verdict

Reverted. Co-scheduling the eight query heads that share each KV head does not
materially improve the gfx1201 FWHT3 attention tile, and explicitly staging the
shared K/V records in LDS is slower. The current one-wave-per-query-head kernel
already obtains the useful reuse through cache without paying cross-wave
barriers or an LDS occupancy tax.

## Shapes

The A3B full-attention geometry is 16 query heads, 2 KV heads, head_dim 256: an
8:1 GQA reuse opportunity. Two families were tested while preserving every
query head's existing FWHT, dot-product, softmax, and V-accumulation order:

1. co-schedule 2/4/8 wave32 query heads per workgroup and rely on L2/L1 reuse;
2. cooperatively stage each packed KV-head tile once in LDS, with K-only and
   K+V variants.

Five occupancy/co-scheduling shapes and five staging shapes were compiled as
separate kernels. Other architectures and the default kernel were untouched.

## tg128 screen

R9700/gfx1201, automatic clocks, Qwen 3.6 35B-A3B MQ4R, FWHT3 KV, retained
PM4, ten warmups, 100 measured tokens, five rows:

| Shape | PM4 tok/s |
|---|---:|
| control | 194.989 |
| QH2, min-blocks 8 | 195.049 |
| QH2, min-blocks 16 | 191.864 |
| QH4, min-blocks 4 | 193.685 |
| QH4, min-blocks 8 | 191.897 |
| QH8, min-blocks 2 | 194.718 |

The best co-scheduled arm is +0.03%, below reproducibility.

Staging screen:

| Shape | PM4 tok/s | Delta vs 193.915 control |
|---|---:|---:|
| QH2, K staged | 193.383 | -0.27% |
| QH2, K+V staged | 182.776 | -5.74% |
| QH4, K staged | 193.904 | -0.01% |
| QH4, K+V staged | 191.633 | -1.18% |
| QH8, K+V staged | 191.542 | -1.22% |

## 8K crossover

The least-bad arms were remeasured at 8192-token context with max_seq 32768:

| Shape | PM4 tok/s | Delta vs control |
|---|---:|---:|
| control | 181.611 | — |
| QH2, cache-only | 181.962 | +0.19% |
| QH4, K staged | 176.117 | -3.03% |
| QH4, K+V staged | 141.145 | -22.28% |

The cache-only result remains noise even where attention is a larger fraction
of token time. Explicit staging becomes worse as the larger 8K working set
exposes its barrier/LDS/occupancy cost.

## Interpretation

The same packed K/V records are indeed requested by eight query heads, but
their naturally adjacent workgroups already hit the cache hierarchy. Merely
placing the waves in one workgroup does not improve that. Copying records into
LDS replaces cached VMEM reads with extra copy instructions, two workgroup
barriers, and 13-39 KiB of LDS per workgroup; the occupancy loss dominates.

Do not pursue more GQA staging shapes for this kernel. The next structural AR
lever is removing DeltaNet's materialized Q/K repeat-interleave, which deletes
a launch and buffer round trip instead of relocating an already-cached read.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-w64-raw/.redline-work/fwht3-gqa-screen/
/home/kaden/.redline-work/hipfire-w64-raw/.redline-work/fwht3-gqa-stage-screen/
/home/kaden/.redline-work/hipfire-w64-raw/.redline-work/fwht3-gqa-8k/
```
