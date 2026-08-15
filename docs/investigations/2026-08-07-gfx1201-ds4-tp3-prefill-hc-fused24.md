# gfx1201 DeepSeek V4 TP3 exact HC-control prefill fusion

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Product commit: `b48249d8fe6fef11066fb0044d7084082d876b5b`  
Candidate kernel commit: `d4c74f9cd`

## Result

The established Hyper-Connection control projection launched 24 workgroups per
token. Every workgroup independently loaded the same 16,384-element stream row
and recomputed the same RMS reduction for one of the 24 control outputs. The
promoted exact-gfx1201 kernel assigns one workgroup to a token, shares that X
load and RMS calculation, and evaluates all 24 control rows together.

The kernel preserves the established scalar F32 accumulation, wave shuffle,
eight-wave LDS reduction, `rsqrtf`, multiply, and base-add order. It is not the
faster WMMA experiment described below.

On the canonical TP3 fixture, prefill improved from **438.7834 tok/s** to
**482.7392 tok/s**, a **10.0172%** gain. Decode remained flat at **53.1912
tok/s**. Relative to the first repaired B128 prefill route at 206.8064 tok/s,
the accumulated production prefill improvement is **133.42%**. The 512-token
assistant output was byte-identical to the golden output.

## Fixture

- Model: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`
- Prompt: `benchmarks/prompts/ds4-gfx942-ar-2048.txt`
- Prompt MD5: `25e22faef15a20ae53501f1956e62b79`
- Effective prompt tokens: 2,052
- Generated tokens: 512
- Devices: three Radeon AI PRO R9700 (`gfx1201`), TP3
- Decode: batch-1 AR, greedy, top-k 6 checkpoint default
- Speculation: off
- Thinking: off
- KV request: Q8; current DS4 contiguous cache remains F32
- Prefill chunk: 1,024 tokens

## Micro gate

The exact production shape is B=1,024, M=24, K=16,384. Timings are one gfx1201
with the same HIP 7.14 runtime used for product validation:

| Kernel | Time (us) | Speedup | Raw-bit mismatches |
|---|---:|---:|---:|
| Established 24-workgroup scalar | 1,189.593 | 1.00x | reference |
| Fused-24 exact scalar | 180.001 | 6.609x | 0 / 24,576 |

The 6.609x micro reduction applied to the 7.47% HC share of the promoted-route
profile projected about 6.3% end-to-end. Product composition delivered 10.02%.

## Rejected WMMA experiment

An earlier one-wave WMMA implementation measured 290.147 us at the same shape
and produced **479.3291 prefill tok/s**, but changed the greedy output SHA from
the golden `b625...9c41` to `b3ba...`. It was coherent but failed the mandatory
byte-identical gate and was not promoted. Its product default was replaced by
the arithmetic-identical fused-24 route; the WMMA implementation remains only
as a measured research/micro asset.

## Correctness and scope

- Assistant-content SHA-256:
  `b6255240b6ccd34d621f152cc9898c71c340fa9393680ddb3d3efde2172b9c41`
- Generated tokens: 512
- Empty responses: 0
- Attractor failures: 0
- Daemon SHA-256:
  `f3a761b29b4efb3a9e1f4be7a4f2357e1315e66e44da74f2775cd2282852968e`
- Architecture gate: exact `gfx1201`, B=1,024 DS4 batched HC prefill only
- Default: enabled in code
- Diagnostic rollback: `HIPFIRE_DEEPSEEK4_GFX1201_HC_FUSED24=0`

No Qwen, gfx1100, gfx1151, weight, format, sampling, expert-count, KV,
speculation, decode, or retained-PM4 route changed. Tail batches and every
non-gfx1201 architecture retain the established scalar kernel.

This was one fresh-process product sample after a decisive raw-bit micro gate;
repeated product samples, long-context, and TP4 coverage were not run at this
checkpoint.

## Evidence

- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-hc-control-fused24-micro/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-hc-control-fused24-product/`
- Rejected WMMA product:
  `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-hc-control-wmma-product/`
