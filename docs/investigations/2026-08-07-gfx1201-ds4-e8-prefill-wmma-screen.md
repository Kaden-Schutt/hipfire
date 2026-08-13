# DeepSeek4 gfx1201 E8 prefill WMMA screen

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Candidate: `601fb8a7c`  
Device: gfx1201 R9700 card 0, ROCm 7.14

## Verdict

Accepted as the numeric kernel asset for the TP3 batched-prefill integration.
It is not yet a production route. The current TP3 daemon prefill executes the
single-token `forward_ep` loop, so product timing before the multi-rank batched
executor exists would not exercise this kernel.

The port is isolated from the accepted gfx1151 implementation:

- separate `gemm_mfp4g32_e8_soa_wmma.gfx12.hip` source;
- exact-gfx1201 Rust launch methods;
- no selector or default behavior change;
- caller-owned F16 staging, avoiding stale pointer-keyed conversion reuse.

ROCm 7.14 confirms half8 A/B operands, K split across the two lane groups,
and accumulator row mapping `8 * (tid >> 4) + j`. Direct source compilation
passed for both gfx1200 and gfx1201. The repository-wide compiler completed
938/1,094 inventory jobs; its 156 unrelated pre-existing source failures are
not evidence against this isolated source.

## Numeric channel

At B=128, all B1/B2/B4 arms agreed with the established row-wise gfx1201 E8
kernel within relative RMSE 0.000089–0.000184. Every output was finite.

| Shape | B1 rel-RMSE | B2 rel-RMSE | B4 rel-RMSE |
|---|---:|---:|---:|
| 512×2048 channel | 0.0001488 | 0.0001488 | 0.0001488 |
| 1024×4096 wq_a/wo_a | 0.0001220 | 0.0001220 | 0.0001220 |
| 32768×1024 wq_b | 0.0001835 | 0.0001835 | 0.0001835 |
| 4096×8192 wo_b | 0.0000888 | 0.0000888 | 0.0000888 |
| 2048×4096 shared up | 0.0001206 | 0.0001206 | 0.0001206 |
| 4096×2048 shared down | 0.0001521 | 0.0001521 | 0.0001521 |

## B=128 shape screen

Times include the existing row-wise launch sequence or one already-staged
F16 WMMA launch. F32→F16 staging is intentionally owned by the caller so a
single conversion can be shared by projections of the same activation.

| Shape | Row-wise µs | B1 µs | B2 µs | B4 µs | Fastest | Row/fastest |
|---|---:|---:|---:|---:|---:|---:|
| 512×2048 channel | 535.64 | 40.31 | 48.25 | 65.86 | B1 | 13.29× |
| 1024×4096 wq_a/wo_a | 873.58 | 86.59 | 94.41 | 128.27 | B1 | 10.09× |
| 32768×1024 wq_b | 3550.12 | 2720.17 | 1338.45 | 707.92 | B4 | 5.01× |
| 4096×8192 wo_b | 3543.53 | 613.59 | 462.15 | 346.88 | B4 | 10.22× |
| 2048×4096 shared up | 1839.62 | 123.13 | 97.20 | 115.13 | B2 | 18.93× |
| 4096×2048 shared down | 1501.05 | 126.62 | 84.73 | 85.95 | B2 | 17.72× |

The B=65 tail screen also passed and showed 2.60×–9.04× improvement over
the current row-wise loop. This verifies non-multiple-of-64 output guards.

## Next gate

Wire a genuine TP3 batched prefill executor with one scratch arena per rank,
local attention/shared-expert shapes, and fixed-rank-order peer reductions.
Select B1/B2/B4 by measured shape. Then run the model-level prefill parity
route and report prefill plus decode throughput from `serve_harness.py`.

## Evidence

- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-prefill-wmma/b65-run.log`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-prefill-wmma/b128-run.log`

Skipped: no production selector, model load, product timing, long-context,
TP4, weight/format/sampling/top-k/KV/speculation/Redline change, or adjacent
architecture runtime route.
