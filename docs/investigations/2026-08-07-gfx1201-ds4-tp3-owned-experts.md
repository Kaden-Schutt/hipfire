# DeepSeek V4 Flash 0731 MQ2R — gfx1201 TP3 owned-expert checkpoint

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Implementation commit: `252192fc6`

## Result

Exact gfx1201 TP3 now avoids dequantizing and multiplying routed experts that
the local rank does not own. The fixed launch grids and arithmetic of owned
blocks are unchanged. Gate/up detects the sharded loader's shared non-owned
dummy pointer; down uses the corresponding gate/up pointer table because the
existing loader deliberately aliases non-owned down pointers to an owned
allocation after zeroing their activations.

The change is admitted only by the exact gfx1201 MQ2R backend. Portable,
gfx1151, gfx942, Qwen, MiniMax, and single-rank routes keep their prior kernel
and dispatch behavior.

## Product checkpoint

Fixture:

- Model: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`
- Artifact SHA-256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`
- Prompt: `benchmarks/prompts/ds4-gfx942-ar-2048.txt`
- Prompt MD5: `25e22faef15a20ae53501f1956e62b79`
- Effective context: 2,052 tokens
- Generated: 512 tokens
- Batch 1, greedy, thinking off, speculation off
- Checkpoint-default top-k 6
- Q8 requested; DS4's current cache implementation remains F32 contiguous
- Three gfx1201 R9700 devices, TP degree 3

| Route | Fresh-process decode samples (tok/s) | Median | Delta |
|---|---:|---:|---:|
| Pre-candidate TP3 | 41.4250 / 41.4534 / 41.4030 | 41.4250 | -- |
| + owned-expert gfx1201 kernels | 47.1821 / 47.3987 / 47.4222 | 47.3987 | +14.42% |

The candidate range spread was 0.506%. Median prefill was 50.3996 tok/s from
50.3318 / 50.4035 / 50.3996, a 0.142% range spread. Prefill is reported as a
diagnostic here; this checkpoint was admitted for decode.

Every full decoded output was byte-identical to the preserved TP3 baseline:
`b6255240b6ccd34d621f152cc9898c71c340fa9393680ddb3d3efde2172b9c41`.
Each product log reports three exact-gfx1201 MQ2R backends and the TP3 graph
route with 86 barriers and 8,833 kernarg blobs. No fallback, panic, illegal
memory access, or HIP error was observed.

Measured binaries:

- `hipfire`:
  `78c8dd892526ddde8fd305fae57b78d2bbc9c65da01da7ba42cd4b2276d682de`
- `daemon`:
  `df050166c5b84354d4056e36228a702a971b243e903e165a1441ae8733ca493a`

Raw evidence:

- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-tp3-expert-skip-valid-screen/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-tp3-expert-skip-run2/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-tp3-expert-skip-run3/`

## Kernel channel

The real DS4 decode shapes were checked on one gfx1201 with exactly two of six
selected experts owned by the rank:

- Gate: 12,288 raw-bit comparisons, zero mismatches.
- Up: 12,288 raw-bit comparisons, zero mismatches.
- Down: 24,576 raw-bit comparisons, zero mismatches.
- Gate/up kernel: 0.053853 -> 0.022772 ms, 2.365x.
- Down kernel: 0.052677 -> 0.021246 ms, 2.479x.

The candidate and incumbent compile to the same owned-path resources: gate/up
74 VGPR, 19 SGPR, 64 bytes LDS; down 93 VGPR, 38 SGPR, no LDS; wave32, no
scratch or spill in either arm.

Durable channel output:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-tp3-expert-skip/channel.txt`

## Excluded attempts

The first harness invocation did not build the CLI target and exited before
model load. A second preflight explicitly set experts-per-token to six; TP
correctly rejected the redundant override before inference because it must use
the checkpoint default. Both attempts are excluded from performance evidence.

No weight, format, sampling, top-k, KV, speculation, PM4/Redline, or adjacent
architecture behavior changed. No TP4 or long-context claim is made by this
checkpoint.
