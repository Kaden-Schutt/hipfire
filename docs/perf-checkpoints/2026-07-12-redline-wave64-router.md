# gfx1201 wave64 fused router: retained-PM4 win

## Result

The gfx1201 autoregressive path now fuses router softmax and top-8 selection
into one zero-LDS wave64 kernel. It replaces 40 `softmax_f32` plus 40
`moe_topk_renorm_k8` launches with 40 fused launches, shrinking the retained
PM4 tape from 833 to 793 dispatches.

The kernel is compiled in a separate translation unit with
`-mwavefrontsize64`; ROCm 7.2 per-function wave-size attributes are not safe for
the lane and shuffle builtins used here. `__launch_bounds__(64, 4)` won the
five-shape occupancy sweep.

## Performance

R9700/gfx1201, automatic clocks, Qwen 3.6 35B-A3B MQ4R, Q8 KV, retained PM4,
ten warmups, 100 measured tokens, 30 rows per arm:

| Order | Control | Wave64 router | Delta |
|---|---:|---:|---:|
| candidate then control | 193.17 tok/s | 194.93 tok/s | +0.91% |
| control then candidate | 193.05 tok/s | **195.53 tok/s** | **+1.28%** |

The five occupancy targets (minimum blocks 2/4/8/12/16) measured 195.08,
196.64, 196.25, 194.61, and 195.69 tok/s in the short gate. Four was selected
before the 30-row certification.

## Correctness and coherence

- PM4 versus HIP shadow: exact over 15 positions.
- Two 40-layer drift passes: 0/640 selected-expert index mismatches.
- Routing weights: max absolute drift 2.88e-4, max relative drift 1.08e-3,
  RMS drift 3.15e-5.
- Sampled eight-turn `serve_harness.py`, medium thinking, 4096 output-token
  cap: 8/8 normal stops, zero runaway/empty/attractor warnings, recall 3/3 on
  both recall turns. Average decode was 164.0 tok/s; final turn was 143 tok/s
  at 23.3k context. The matched control followed a shorter sampled trajectory
  (166.0 average, 148 at 20.1k), so those session rates are coherence evidence,
  not an equal-context speed claim.

## Isolation and fallback

The new path is selected only on gfx1201. Set
`HIPFIRE_GFX1201_ROUTER_W64=0` to restore the two-kernel router. Other
architectures retain their existing kernel selection and machine code.
