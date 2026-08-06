# Wave64 gate+up+SwiGLU fusion: ratchet-safe, slower, no tape reduction

## Verdict

Reverted. Combining the gfx1201 F2 spatial-wave64 gate/up kernel with terminal
SwiGLU removes one intermediate activation buffer, but retained-PM4 tg128 falls
2.41% and the sampled eight-turn session falls 2.40% with one attractor. The
candidate does not reduce dispatch count because hipfire already fuses SwiGLU
and FWHT rotation in the control path.

## Current-topology correction

The control is already:

```text
gate+up GEMV -> fused_silu_mul_mq_rotate -> down GEMV
```

A row-owned gate/up workgroup cannot also perform the 256-element FWHT: the
transform couples 256 output rows produced by many independent workgroups, and
HIP has no grid-wide barrier inside this ordinary kernel. The executable merged
experiment was therefore:

```text
wave64 LPG=8 gate+up+SwiGLU -> mq_rotate_x -> down GEMV
```

It replaces the two gate/up output buffers with one SwiGLU buffer, saving one
full write and read, but remains two launches. A real one-launch version needs a
256-row workgroup/cooperative barrier redesign or must fuse through the down
projection; neither is a local F2+#58 edit.

## Candidate and integration gates

- Separate gfx1201 source compiled with `-mwavefrontsize64`.
- `LPG=8`, eight HFQ4 groups spatially in flight, two adjacent rows per wave.
- Terminal expression exactly matches the existing
  `gate / (1 + exp(-gate)) * up` formula.
- Retained-PM4 kernarg ownership and pointer effects were fully described.
- Mixed-wave PM4 used the kernel descriptor as the source of truth:
  `CS_W32_EN=0` only for the wave64 node and one for surrounding wave32 nodes.
- Corrected capture: 833 launches, 26 kernels, sequence
  `08d239bead5221b2`; the first 793-launch capture was rejected because a
  delegated batched FWHT call was invisible to Redline's recorder.
- Corrected PM4-vs-HIP shadow: exact for 15 consecutive positions.

## Numerical ratchet

The representative A3B-shape standalone A/B (`M=1536`, `MI=768`, `K=2048`,
eight experts) applies terminal SwiGLU to both the wave32 and F2 outputs:

| Comparison | Max absolute | Max relative | Gate |
|---|---:|---:|---|
| gate/up outputs | 4.768e-7 | 8.659e-4 | pass |
| terminal SwiGLU | 6.730e-7 | 8.120e-4 | pass |

The requested absolute-or-relative ratchet (`abs <= 1e-3` or
`rel <= 2e-3`) passes. The same standalone kernel reaches 438.1 GB/s versus
298.7 GB/s wave32 (1.4666x), reproducing F2's synthetic win. This makes the
product loss a context-transfer failure, not a local kernel-throughput or
numerical-tolerance failure.

## 30-row tg128 result

Automatic clocks, ten warmups, 30 rows, 100 measured tokens:

| Arm | Ordinary HIP | Retained PM4 | PM4 min | PM4 max |
|---|---:|---:|---:|---:|
| wave32 control | 162.951 | **193.536** | 191.141 | 193.786 |
| wave64 gate+up+SwiGLU | 162.094 | 188.878 | 186.834 | 189.193 |

The candidate is 2.41% slower in retained PM4 and 0.53% slower through HIP.

## Eight-turn serve and prefill

| Arm | Average decode | Final decode | First-seven prefill average | Attractors |
|---|---:|---:|---:|---:|
| control | **165.663 tok/s** | **147.7 @ 20,069** | **94.03 ms** | 0 |
| candidate | 161.688 tok/s | 146.3 @ 19,035 | 96.21 ms | 1 |

The non-bit-exact candidate causes sampled trajectories to diverge despite the
fixed seed. Turn seven trips the attractor detector. Turn eight then misses the
same prompt-cache shape and pays 9,046 ms prefill versus 140.1 ms control; this
is a downstream consequence of changed session history, not a direct 64x
prefill-kernel regression. Even excluding that turn, candidate prefill is 2.3%
slower.

## Mechanism

The standalone repeated-kernel benefit still does not transfer to the real
gate/up boundary. The saved intermediate buffer is smaller than the established
model-context penalty of the spatial-wave64 work decomposition. Because the
current control already performs activation+rotation in one launch, there is no
dispatch-amortization prize to compensate.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/fusion/gate-up-swiglu-w64/
```

The extended standalone ratchet remains under:

```text
/home/kaden/ClaudeCode/autorocm/redline/.redline-work/wave64-decode/
```
