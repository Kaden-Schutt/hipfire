# QKVZA terminal-postop fusion: exact but neutral

## Verdict

Do not ship the gfx1201 QKVZA terminal-postop fusion. Folding the DeltaNet
beta sigmoid and alpha gate into the final 32 row workgroups of
`fused_qkvza_hfq4g256` is bit-exact and shrinks the retained-PM4 tape from 763
to 733 dispatches, but it does not beat the compact-head GDN champion after
order-balanced certification.

The standalone `fused_sigmoid_alpha_gate_f32` launch is already too cheap for
this fusion to matter. Adding the post-op path to the matrix-wide projection
raises its scalar-register footprint from 22 to 26 SGPR while leaving the 80
VGPR footprint unchanged. The saved dispatch and added projection-tail control
cost approximately cancel.

## Five-shape screen

R9700/gfx1201, automatic clocks, Qwen 3.6 35B-A3B MQ4R, Q8 KV, retained PM4,
ten warmups, 100 measured tokens, five rows per shape:

| Minimum blocks/CU | Retained PM4 |
|---:|---:|
| 2 | 195.094 tok/s |
| 4 | **197.312 tok/s** |
| 8 | 194.352 tok/s |
| 12 | 195.070 tok/s |
| 16 | 196.824 tok/s |

The B2/B4/B8 variants emitted identical `.text`; B12/B16 emitted a second
identical pair. B4 was therefore only the representative of the faster
five-row sample, not evidence that launch bounds alone changed occupancy.

## Thirty-row certification

| Arm | Median tg128 |
|---|---:|
| unfused control | 196.736 tok/s |
| fused B4 | 197.498 tok/s |
| unfused control, repeated | **197.818 tok/s** |

The candidate is +0.11% against the mean of the two controls and slower than
the second control. This is measurement noise, not a product win.

## Correctness and serve harness

- PM4 versus HIP shadow: exact over 15 consecutive positions.
- Materialized control versus fused candidate: all 40 post-layer hidden states
  byte-identical at position 128.
- Tape: 763 to 733 dispatches; replay contract probe admits all 24 kernels.
- Fixed-seed eight-turn Q8 session: identical 23.3k-context token trajectory,
  8/8 normal stops, recall 3/3 on both recall turns, no runaway/empty/attractor.
- Average decode was 164.8 tok/s control versus 164.4 tok/s fused. Per-turn
  prefill was effectively identical; no prefill claim applies to this
  decode-only experiment.

This closes terminal-postop fusion as a dispatch-count-only lever. Reaching
210 requires removing meaningful intermediate traffic or work, not another
30 already-amortized scalar launches.

Artifacts:

```text
hiptrx:/home/kaden/.redline-work/hipfire-w64-raw/.redline-work/qkvza-postop/
```
