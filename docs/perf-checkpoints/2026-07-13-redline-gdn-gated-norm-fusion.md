# gfx1201 compact GDN + gated-norm fusion: exact 703-launch negative

## Verdict

Do not ship the last-arriver compact-GDN + gated-normalization fusion. It is
bit-exact and removes all 30 standalone `gated_norm_f32` launches, shrinking
the retained-PM4 tape from 733 launches / 23 kernels to 703 / 22. The best
long-run candidate nevertheless regresses retained-PM4 tg128 by 0.95% against
the mean of two bracketing controls. The implementation was reverted; the
201 tok/s conv+Q/K-normalization champion remains unchanged.

## Implementation tested

The existing recurrence geometry has 32 independent four-row workgroups per
value head. No one workgroup owns the 128 raw outputs required by gated RMSNorm.
The experiment therefore used a deadlock-free last-arriver protocol:

1. Every row-tile workgroup published its four recurrence outputs.
2. Lane 0 incremented a persistent per-head arrival counter.
3. The workgroup observing arrival 32 performed the original gated-norm lane
   accumulation, XOR reduction, and SILU multiply, then reset the counter.

This avoids an unsafe grid-wide spin barrier: the normalizing workgroup is
known to be the last arrival, so all producers have completed. Two publication
mechanisms were tested. The first used `__threadfence` plus legacy atomics. The
second replaced the full fence with ROCm agent-scope release/acquire atomics.
Both were replay-exact for 15 consecutive positions.

## Five-shape screen

R9700/gfx1201, automatic clocks, Qwen 3.6 35B-A3B MQ4R, Q8 KV, retained PM4,
ten warmups, 100 measured tokens, five rows per shape:

| Minimum blocks/CU | HIP | Retained PM4 |
|---:|---:|---:|
| 2 | 173.972 | **201.208 tok/s** |
| 4 | 173.784 | 198.935 tok/s |
| 8 | 171.285 | 199.742 tok/s |
| 12 | 173.855 | 200.259 tok/s |
| 16 | 173.403 | **201.803 tok/s** |

B16 led the short PM4 screen and advanced to the order-balanced certification.

## Thirty-row certification

| Arm | Median tg128 |
|---|---:|
| 733-launch control | 200.323 tok/s |
| 703-launch fused B16 | **198.755 tok/s** |
| 733-launch control, repeated | 200.999 tok/s |

The candidate is -0.95% against the 200.661 tok/s mean of the controls. The
lighter agent-scope publication recovered some of the fence cost but did not
reverse the result: B16 measured 200.544 tok/s and B2 200.047 tok/s over five
rows, with B16 again passing the full 15-position replay shadow.

## Why fewer launches lose

The dispatch saving is only 30 small normalizer launches per token, but the
last-arriver protocol adds one cross-workgroup atomic RMW per recurrence tile:
32 value heads x 32 row tiles x 30 DeltaNet layers = 30,720 atomics per token.
Agent-scope release/acquire avoids unrelated L2 writeback, but it cannot remove
the atomic serialization or the final workgroup's normalization tail. The
extra work stretches the dominant recurrence kernel more than removing the
small consumer launch saves.

A profitable one-dispatch version therefore needs a recurrence decomposition
where one cooperative unit naturally owns all 128 outputs for a head without
30,720 cross-workgroup arrivals. The existing four-row geometry cannot provide
that cheaply. Retained PM4 already keeps the separate producer/consumer
boundary lean, so the correct product choice is to retain the standalone norm.

No eight-turn serve run was promoted because the candidate failed the 30-row
product-performance gate. Correctness was established before timing, and all
experimental source changes were reverted after the negative result.

Artifacts:

```text
hiptrx:/home/kaden/.redline-work/hipfire-w64-raw/.redline-work/gdn-norm-fusion/
```
