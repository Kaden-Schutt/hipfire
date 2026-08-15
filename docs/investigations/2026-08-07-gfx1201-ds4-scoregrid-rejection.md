# DeepSeek V4 gfx1201 TP3 score-grid attention screen

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Candidate: `ecd433f87`  
Promoted parent: `0d2a50aca` (`51.970852` tok/s median)

## Verdict

Rejected after one product screen. The exact-gfx1201 MQ2R route was allowed to
select the score-grid/ILP implementation already used by the certified
gfx1151 route. The production-shape channel was positive, but the canonical
product improvement was only `+0.8026%`, below the campaign's 2% gate. No
additional product repetitions were run.

The product route was restored immediately. The focused channel harness is
retained for future attention-kernel work; it does not change any shipping
dispatch by itself.

## Production-shape channel

Fixture: F32 K=V, head dimension 512, SWA 128, compressed top-K 512. These are
the two local head partitions used by TP3.

| Local heads | Generic | Score-grid | Speedup | Raw-bit equal | Max abs |
|---:|---:|---:|---:|---:|---:|
| 24 | 67.649 us | 49.300 us | 1.3722x | 697 / 12,288 | 1.0245e-8 |
| 16 | 55.843 us | 47.251 us | 1.1818x | 448 / 8,192 | 1.0245e-8 |

The implementation changes FP32 association and is not an intermediate
raw-bit transform. No non-finite output was observed. The product output gate
below remained byte-identical.

Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-scoregrid-attention/result.txt`

## Product screen

- Model: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`
- Prompt: `benchmarks/prompts/ds4-gfx942-ar-2048.txt`
- Prompt MD5: `25e22faef15a20ae53501f1956e62b79`
- Effective context: 2,052 tokens
- Generation: 512 tokens, batch 1, greedy, thinking off, speculation off
- Experts per token: checkpoint default 6
- TP3 devices: three gfx1201 R9700s
- Candidate decode: `52.387979` tok/s
- Promoted parent median: `51.970852` tok/s
- Delta: `+0.417126` tok/s / `+0.8026%`
- Candidate prefill: `55.992846` tok/s (diagnostic; prefill is still serial)
- Route: three ranks, 86 barriers, 7,844 kernarg blobs

The candidate generated 512 tokens and 395 answer words. Its decoded output
SHA-256 was identical to the promoted route:
`b6255240b6ccd34d621f152cc9898c71c340fa9393680ddb3d3efde2172b9c41`.

Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-scoregrid-attention/product-screen/`

## Scope

No weight, format, top-k, sampling, expert-count, KV, speculation, long-context,
gfx1151, gfx1100, gfx942, or Qwen behavior was changed. One product process
was sufficient to reject the candidate because it missed the 2% gate; no
promotion or absolute performance claim is made from that single sample.
