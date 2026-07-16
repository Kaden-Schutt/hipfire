# MQ4 MoE down+combine fusion: exact 40-dispatch reduction, no product win

## Verdict

Do not enable `gemv_hfq4g256_moe_down_k8_indexed_fused_acc` for A3B AR. The
prototype is bit-exact and shrinks the retained PM4 tape from 833 to 793
dispatches, but its tg128 direction reverses with A/B order and it loses the
sampled eight-turn product gate. The opt-in selector was reverted; the dormant
prototype remains available for future kernel-shape work.

## Candidate

The fused kernel replaces each layer's expanded MQ4 down GEMV plus
`moe_down_combine_k8_batched` with one launch. Each wave owns a two-row tile,
loops the eight routed experts internally, accumulates the routing-weighted
results in registers, and writes directly into the residual. This eliminates
the `[K_TOP x hidden]` expanded-buffer write/read and one launch per MoE layer.

The retained-PM4 recorder was temporarily taught the fused 52-byte kernarg ABI
and its five pointer effects so the experiment was not confounded by an unknown
kernel forcing conservative waits.

## Gates

- Host: `hiptrx`, gfx1201, automatic clocks.
- Model: Qwen 3.6 35B A3B MQ4R, Q8 KV, MTP off.
- Fifteen consecutive PM4 shadow positions against the HIP oracle.
- tg128: 100 measured tokens, ten warmups, 30 rows, both A/B orders.
- Product: fixed-seed registry sampling, medium thinking, eight-turn session
  through context 20,069.
- Prefill: the same eight-turn serve harness's per-turn prefill measurement.
- No kernel source changed during the selector experiment; gfx1100 kernel
  `.text` remained byte-identical.

## Structural result

| Arm | Launches | Unique kernels | Sequence hash | Shadow15 |
|---|---:|---:|---|---|
| expanded down + combine | 833 | 27 | `8ba4c8d66f32d116` | established control |
| fused down+combine | **793** | **25** | `909d48c9f5d66832` | bit-exact |

The fusion removes exactly one dispatch in each of 40 MoE layers. It does not
remove two launches per layer: the live path is expanded-down plus combine, so
the measured reduction is 833 to 793.

## tg128 product A/B

| Order | Control | Fused | Delta |
|---|---:|---:|---:|
| control then fused | 192.152 tok/s | 194.006 tok/s | +0.96% |
| fused then control | **194.523 tok/s** | 192.363 tok/s | -1.11% |

The sign follows run order. This is not a reproducible improvement and neither
fused run clears the 195 tok/s champion bar.

## Eight-turn serve and prefill

| Arm | Average decode | Decode at context 20,069 | Average prefill |
|---|---:|---:|---:|
| control | **166.300 tok/s** | **148.2 tok/s** | **99.938 ms** |
| fused | 165.738 tok/s | 147.7 tok/s | 100.050 ms |

All eight generated responses are byte-identical between arms, including token
counts, thinking counts, finish reasons, and final answer text. Both have zero
runaway, empty, or attractor cases and recall 2/3 on the two recall turns.
Prefill is neutral within 0.12 ms; the fusion therefore provides no hidden TTFT
benefit.

## Interpretation

The saved dispatch and expanded-buffer traffic are offset by the fused kernel's
loss of expert-axis parallelism: the control dispatches expert rank in the grid,
while the fused row block executes all eight expert dots serially. A future
down+combine attempt must preserve multiple experts in flight (for example a
cooperative partial reduction) rather than merely moving the K_TOP loop inside
one wave.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/fusion/down-combine/
  candidate-shadow15.json
  tg128-control.json
  tg128-candidate.json
  tg128-reverse-candidate.json
  tg128-reverse-control.json
  serve-control.json
  serve-candidate.json
```
