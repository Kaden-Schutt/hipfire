# DeepSeek V4 gfx1201 TP3 grouped-E8 + low-LDS norm checkpoint

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Candidate: `0d2a50aca`  
Parent promoted line: `607edb7ea` (`50.860761` tok/s median)

## Verdict

Promoted as a measured bundle. The exact-gfx1201 DeepSeek V4 MQ2R route now:

1. collapses the eight O-LoRA E8 GEMVs into one 2-D launch using a grouped
   kernel derived from the accepted gfx1201 row body; and
2. selects the low-LDS wave-first reduction for DeepSeek's fused RMSNorm +
   FWHT + plain-output kernel.

Both defaults require exact `gfx1201` and MQ2R. The generic RMSNorm method
used by Qwen and MiniMax is unchanged, and no gfx1151, gfx1100, gfx942, weight,
sampling, expert-count, or KV default is widened.

## Fixture

- Model: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`
- Model SHA-256: `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`
- Prompt: `benchmarks/prompts/ds4-gfx942-ar-2048.txt`
- Prompt MD5: `25e22faef15a20ae53501f1956e62b79`
- Effective context: 2,052 tokens
- Generation: 512 tokens, batch 1, greedy, thinking off, speculation off
- Experts per token: checkpoint default 6
- Requested KV: Q8; current DS4 implementation remains F32 contiguous
- Route: TP3 on three gfx1201 R9700s through `scripts/serve_harness.py`

## Channel evidence

The production O-LoRA shape is `G=8, M=1024, K=4096`. Four independent
synthetic cases compared eight accepted gfx1201 single GEMVs with the grouped
kernel:

- 32,768/32,768 output words raw-bit identical
- eight launches: 55.933 us
- grouped launch: 25.028 us
- isolated speedup: 2.235x

Its standalone product screen measured 51.360710 tok/s, +0.983% over the
parent median. It removed 215 graph blobs (8,059 to 7,844), but remained below
the 2% promotion threshold and was retained only as a component of the bundle.

Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-grouped-olora/`

The low-LDS fused norm channel at `K=4096` measured:

- incumbent: 9.753 us
- low-LDS candidate: 5.292 us
- isolated speedup: 1.843x
- plain output: 29,381/32,768 words raw-bit identical
- rotated output: 29,541/32,768 words raw-bit identical

The low-LDS kernel is mathematically equivalent but changes the FP32 reduction
order, so it is not an internal raw-bit transform. Promotion therefore rests
on the required full-route decoded-output gate below, not on a claim of
bit-identical intermediate state.

Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-rmsnorm-rotate-nox/`

## Product checkpoint

| Fresh process | Decode tok/s | Prefill tok/s | Graph blobs |
|---|---:|---:|---:|
| 1 | 51.909692 | 55.614667 | 7,844 |
| 2 | 51.991029 | 55.629737 | 7,844 |
| 3 | 51.970852 | 55.607348 | 7,844 |

- Decode median: **51.970852 tok/s**
- Prior promoted median: **50.860761 tok/s**
- Delta: **+1.110091 tok/s / +2.1826%**
- Decode range spread: **0.1565%**
- Prefill median: 55.614667 tok/s, diagnostic only; TP prefill remains
  token-serial
- Route identity: three ranks, 86 peer barriers, 7,844 kernarg blobs

All three decoded outputs were byte-identical to the promoted baseline:
`b6255240b6ccd34d621f152cc9898c71c340fa9393680ddb3d3efde2172b9c41`.
Every run generated 512 tokens, ended by length, contained 395 answer words,
and reported zero empty or attractor failures.

Candidate binary SHA-256:

- `hipfire`: `471fc2c7c8b24f8ec5329f63b5238625885f0d6df3b0d00d8ce747cd270efe3e`
- `daemon`: `116a2d09eee745382f4a630e8c99e2c1caed78b138f0271f56c2c4250fabb304`

Product evidence:

- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-grouped-nox/product-screen/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-grouped-nox/product-run2/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-grouped-nox/product-run3/`

## Scope and next gate

No weight, format, top-k, sampling, expert-count, KV, speculation, retained
PM4, or long-context change was included. Redline PM4 parity is not applicable
to the three-rank HipGraph route; the kernel channel and three fresh
`serve_harness.py` product processes cover the changed route.

The decode campaign remains active at 51.97 tok/s. Re-profile this route and
pursue only another occurrence-weighted candidate projecting at least 2%
toward 60 tok/s. Production prefill remains the next phase after the decode
gate; its 55.6 tok/s diagnostic is still the known token-serial implementation.
