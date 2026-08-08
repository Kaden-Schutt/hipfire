# Raw wave64 MoE gate/up substitution: conservative shapes neutral, LPG64 incoherent

## Verdict

Do not enable the raw gfx1201 wave64 gate/up kernel for A3B AR. This experiment
kept the established downstream topology unchanged:

```text
wave64 gate/up -> fused_silu_mul_mq_rotate -> down GEMV
```

It therefore closes the missing control left by the earlier
gate/up+SwiGLU experiment. Four numerically conservative spatial-K shapes did
not beat the 195.53 tok/s single-stream retained-PM4 champion. A fifth, more
aggressive LPG64 shape exceeded 210 tok/s but failed the model coherence gate
decisively and is not a usable optimization.

## Five-shape screen

R9700/gfx1201, automatic clocks, Qwen 3.6 35B-A3B MQ4R, Q8 KV, retained PM4,
ten warmups, 100 measured tokens, five rows per shape:

| Shape | Rows/workgroup | Retained PM4 | Result vs 195.53 champion |
|---|---:|---:|---:|
| LPG8 | 2 | 190.998 tok/s | -2.32% |
| LPG8 | 1 | 195.228 tok/s | -0.15% |
| LPG16 | 2 | 194.125 tok/s | -0.72% |
| LPG32 | 2 | 192.499 tok/s | -1.55% |
| LPG64 | 2 | 218.691 tok/s | rejected: incoherent |

An initial 224.033 tok/s LPG64 row was invalid because the generic packed-word
loop handled zero words at four elements/lane. It was discarded immediately.
The reported 218.691 result is the corrected half-word implementation: adjacent
lanes consume the low and high four nibbles of one packed word.

## Correctness and drift

The corrected LPG64 retained-PM4 tape was exact against its own HIP execution
for 15 consecutive positions (`793` launches, `25` kernels). This proves the
mixed-wave replay and PM4 dispatch were correct; it does not establish
equivalence to the wave32 model oracle.

A deterministic graph-off hidden-state A/B against two mutually byte-identical
wave32 control runs showed immediate and accumulating model drift:

- layer 0: max absolute `2.158e-2`, RMS `5.781e-3`, cosine `0.9461`;
- layer 39: max absolute `2.494`, RMS `2.431e-1`, cosine `0.8341`;
- two independent control runs were byte-identical at every layer.

The sampled eight-turn FWHT3 serve harness then failed visibly:

- turn 1 reached the 4096-token cap with an attractor;
- six turns had empty visible answers;
- recall was `1/3` then `0/3`;
- summary: `runaway=1 empty=6 attractor=1`.

The displayed 203.7 tok/s session average is meaningless as a product result
because most turns terminated after only a few malformed tokens.

## Interpretation

The old LPG8 synthetic kernel win remains real in isolation, but it does not
transfer through the unchanged A3B product boundary. Increasing spatial-K
width to LPG64 creates enough throughput to clear 210, but its changed
reduction behavior moves the model outside the accepted coherence envelope.
This is a numerical-algorithm trade, not a replay or wave-size dispatch bug.

Raw wave64 gate/up is therefore closed. Keep the wave32 gate/up kernel and move
the optimization effort to FWHT3 attention GQA reuse, where traffic can be
removed without perturbing the MoE reduction tree.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-w64-raw/.redline-work/w64-screen/
```
