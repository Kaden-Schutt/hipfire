# QKVZA buffer-RT versus global loads: buffer policy wins by 1.02%

## Verdict

Keep buffer-RT weight loads in `fused_qkvza_hfq4g256` on gfx1201. Restoring
the kernel-oracle branch's global-load addressing regresses retained-PM4 tg128
by 1.02% in an order-balanced A/B.

## Why this A/B mattered

The refreshed clean AR timestamp profile assigns 18.71% of serialized kernel
time to `fused_qkvza_hfq4g256`, making it the largest live kernel family. The
kernel-oracle winner used global loads; the Redline branch later opted QKVZA
into the gfx1201 buffer-SRD policy alongside residual and LM-head kernels.
QKVZA's policy had not been isolated after that conversion.

The candidate changed only QKVZA's scale, zero-point, and packed-weight loads
back to `global_load`. Every other eligible kernel retained buffer-RT.

## Gates

- `hiptrx`, gfx1201, automatic clocks.
- Qwen 3.6 35B A3B MQ4R, Q8 KV, MTP off.
- Fifteen consecutive PM4 positions: exact logits/KV/recurrent/blob parity.
- Stable 833-launch, 26-kernel tape with the champion sequence hash.
- Context 128, 100 measured tokens, ten warmups, 30 rows in both A/B orders.
- Both code objects use zero scratch and zero spills.

| Addressing | VGPR | SGPR |
|---|---:|---:|
| buffer-RT control | 80 | 22 |
| QKVZA-global | 81 | 16 |

## Product A/B

| Order | Buffer-RT | Global | Global delta |
|---|---:|---:|---:|
| buffer then global | 193.584 tok/s | 191.985 tok/s | -0.83% |
| global then buffer | 193.384 tok/s | 191.022 tok/s | -1.22% |
| order-balanced mean | **193.484 tok/s** | 191.504 tok/s | **-1.02%** |

The direction is stable in both orders. The six-SGPR descriptor saving from
global addressing does not compensate for the buffer memory path on this hot
projection. The temporary QKVZA-only override was reverted.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/
  qkvza-global-shadow15.json
  qkvza-global-control-30.json
  qkvza-global-candidate-30.json
  qkvza-global-reverse-candidate-30.json
  qkvza-global-reverse-control-30.json
```
