# GFX1201 HFQ4 weight buffer-load expansion

**Date:** 2026-07-11

**Host:** `hiptrx`, Radeon AI PRO R9700 (`gfx1201`)

**Model:** `qwen3.6-35b-a3b.mq4r`

**Clock policy:** automatic; no clock or performance-level override

## Result

The device-scope buffer-RT weight path now also covers:

- `gemv_hfq4g256_residual`;
- the active `gemv_hfq4g256_multirow_r2` entry point used by the lm head.

The final no-environment configuration improved retained-PM4 decode at 8K by
0.647% over the preceding `redline` default:

| Configuration | Process medians | Mean |
| --- | --- | ---: |
| preceding default | 174.799, 174.824 tok/s | 174.812 tok/s |
| residual + lm-head buffer RT | 175.980, 175.905 tok/s | 175.943 tok/s |

At Q8 context 128, the same final default measured 186.387 tok/s versus
186.084 tok/s for the preceding default (+0.163%). The 8K lift therefore does
not trade away short-context throughput.

## Individual and interaction matrix

Every individual arm used a fresh daemon, a separate JIT cache, three warmups,
and ten measured rows of 100 positions at FWHT3 context 8192. Individual arms
were repeated in reverse order.

| Arm | Process medians | Mean | Versus baseline |
| --- | --- | ---: | ---: |
| baseline | 174.799, 174.824 | 174.812 | — |
| sigmoid-scaled residual | 174.339, 174.083 | 174.211 | -0.344% |
| lm-head multirow R2 | 174.999, 174.813 | 174.906 | +0.054% |
| ordinary residual | 175.550, 175.202 | 175.376 | +0.323% |
| all three | 175.809, 175.976 | 175.893 | +0.619% |

Pairwise passes localized the interaction:

| Pair | Median | Versus baseline |
| --- | ---: | ---: |
| residual + lm-head | **175.980** | **+0.668%** |
| residual + sigmoid | 175.436 | +0.357% |
| lm-head + sigmoid | 174.050 | -0.436% |

The sigmoid-scaled conversion was removed. It was negative alone and added no
value to either useful pair. The lm-head conversion is neutral alone but
repeatably improves the residual pair, so both surviving kernels ship together.

## Resource gate

Both kernels use one matrix-wide SRD and remain zero-scratch. This did not prove
that descriptor count caused the earlier MoE gate/up spill: its first
expert-wide one-SRD arm also spilled. Follow-up work localized the pressure to
address and loaded-value lifetimes, then cleared it with scalar row/group
offsets and separate gate/up load-consume stages.

| Kernel | Global resources | Buffer-RT resources | Scratch |
| --- | --- | --- | ---: |
| lm-head multirow R2 | 94 VGPR / 36 SGPR | 94 VGPR / 19 SGPR | 0 B |
| ordinary residual | 94 VGPR / 36 SGPR | 94 VGPR / 19 SGPR | 0 B |
| sigmoid-scaled residual (rejected) | 67 VGPR / 29 SGPR | 69 VGPR / 21 SGPR | 0 B |

The lm-head matrix is roughly 270 MB, below the 32-bit raw-buffer offset limit.
The same one-SRD layout therefore covers every row without descriptor rotation.

## Correctness and isolation

- Final no-environment default: 15 consecutive positions exact for logits, KV,
  and recurrent state against direct HIP and exact captured-kernarg execution.
- Capture remains 833 launches / 27 kernels with the same sequence hash.
- Both new gfx1100 `.text` sections are byte-identical to pre-change HEAD.
- All promoted gfx1201 kernels remain zero-scratch and are accepted by retained
  PM4 replay.
- Local `rdna-compute` feature-flag tests pass.

## Cache interpretation

This result does not establish that the added paths increase cache-hit
residency. Named GL2 counters are not trustworthy on this box, the prior HT and
NT hint arms lost to ordinary RT, and the 270 MB lm-head matrix is not generally
cache-resident. The evidence supports a kernel-specific buffer-addressing and
generated-schedule benefit, including an interaction between the residual and
lm-head forms. It should not be generalized as a blanket cache-hint rule.

## Artifacts

Raw reports are retained on `hiptrx` under:

```text
/home/kaden/.redline-work/hipfire-cache-candidates/.redline-work/cache-candidates/
```

Notable files are `product-8k-{baseline,sigmoid,residual,multirow,all}.json`,
their `-reverse` counterparts, `product-8k-{residual_multirow,
residual_sigmoid,multirow_sigmoid}.json`, `product-8k-final-default.json`,
`product-tg128-{baseline,final}.json`, and `final-default-shadow15.json`.
