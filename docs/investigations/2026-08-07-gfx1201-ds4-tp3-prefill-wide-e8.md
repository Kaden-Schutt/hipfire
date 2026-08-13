# gfx1201 DeepSeek V4 TP3 wide-E8 prefill promotion

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Product commit: `984f199b1c8cf053bc6975f822a57f6bbda67d84`  
Candidate kernel commit: `693557e06`

## Result

The exact-gfx1201 MFP4G32E8SOA prefill route now retains more query tiles per
wave at the measured 1,024-token production chunk. This reuses each decoded E8
weight fragment across B8 or B16 WMMA accumulators on the wide projections,
while the smaller TP shards use their measured B2/B4 schedules.

On the canonical TP3 fixture, prefill improved from **422.3075 tok/s** to
**438.7834 tok/s**, a **3.9014%** gain. Decode remained flat at **53.1632
tok/s**. The 512-token assistant output was byte-identical to the golden output.

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

## Channel screen

The B1/B2/B4/B8/B16 screen used all fourteen full and rank-local dense shapes
at B=1,024. Every schedule produced finite output and the same relative-error
envelope against the established row-wise E8 kernel. Relevant TP3 timings:

| Shape | Previous | Selected | Previous us | Selected us | Reduction |
|---|---:|---:|---:|---:|---:|
| Wq-b, 12,288 x 1,024 | B4 | B8 | 948.25 | 705.59 | 25.59% |
| Wq-b, 8,192 x 1,024 | B4 | B8 | 665.69 | 431.69 | 35.15% |
| Wo-b, 4,096 x 3,072 | B2 | B8 | 1,324.26 | 816.96 | 38.31% |
| Wo-b, 4,096 x 2,048 | B2 | B8 | 926.12 | 617.80 | 33.29% |
| Shared up, 768 x 4,096 | B1 | B4 | 571.52 | 479.23 | 16.15% |
| Shared up, 512 x 4,096 | B1 | B2 | 308.09 | 295.23 | 4.17% |
| Shared down, 4,096 x 768 | B2 | B8 | 334.39 | 182.72 | 45.36% |
| Shared down, 4,096 x 512 | B2 | B8 | 240.69 | 137.29 | 42.96% |

B16 remains selected only for full-width rows of at least 16,384. TP3's
12,288-row shard uses B8 because B16's additional register pressure bought
only another 2.1% in isolation there. Batches other than 1,024 retain the prior
selector until separately measured.

## Correctness and scope

- Assistant-content SHA-256:
  `b6255240b6ccd34d621f152cc9898c71c340fa9393680ddb3d3efde2172b9c41`
- Generated tokens: 512
- Empty responses: 0
- Attractor failures: 0
- Daemon SHA-256:
  `b83a9fbbd48af7d1ab3b239dbc149a24ab2958d552fb3d25b5d3a35bfda5795b`
- Architecture gate: exact `gfx1201` DS4 MFP4G32E8SOA route only
- Default: enabled in code
- Diagnostic rollback: `HIPFIRE_DEEPSEEK4_GFX1201_E8_WIDE=0`

No Qwen, gfx1100, gfx1151, weight, format, sampling, expert-count, KV,
speculation, or retained-PM4 path changed. This was one fresh-process product
sample after a decisive channel screen; repeated acceptance and long-context
coverage were not run at this checkpoint.

## Evidence

- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-wide-prefill-micro/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-wide-prefill-product/`
