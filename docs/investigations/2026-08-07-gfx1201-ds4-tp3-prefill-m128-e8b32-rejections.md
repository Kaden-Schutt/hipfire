# gfx1201 DeepSeek V4 TP3 M128 and grouped-E8 B32 screens

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Screening commit: `757fc2ede`  
Accepted baseline: `b48249d8fe6fef11066fb0044d7084082d876b5b`

## Result

Neither candidate was promoted.

The gfx1201 grouped-MQ2 M128 kernel improved the canonical TP3 prefill rate
from **482.7392 tok/s** to **491.2210 tok/s** (**+1.7570%**), but changed the
512-token greedy assistant SHA from the golden `b625...9c41` to
`1b6b...970b`. It therefore failed the mandatory byte-identical output gate.

The grouped E8 B32 kernel was byte-identical to the golden output, but the
isolated product run measured **469.0213 tok/s** (**-2.8417%**). Its
product-shape micro improved 1.437x, so this is another case where isolated
query reuse did not translate through the full TP3 route. It failed the
product-performance gate.

Enabling both candidates together measured **492.1977 tok/s** (**+1.9593%**)
and inherited the M128 output mismatch. The combined result was below the
2% promotion threshold even before correctness was considered.

The pre-promotion product switches were removed. The two kernels and their
microbench coverage remain as rejected research assets; neither is reachable
from the default DeepSeek or Qwen routes.

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

## Micro and product results

| Candidate | Product-shape micro | Product prefill | Delta | Output SHA |
|---|---:|---:|---:|---|
| Accepted fused-HC baseline | - | 482.7392 tok/s | reference | `b625...9c41` |
| MQ2 M128 | gate/up 1.019x; down 1.081x; raw exact | 491.2210 tok/s | +1.7570% | `1b6b...970b` (reject) |
| Grouped E8 B32 | 1.437x; raw exact | 469.0213 tok/s | -2.8417% | `b625...9c41` |
| M128 + E8 B32 | projected about 3.4% | 492.1977 tok/s | +1.9593% | `1b6b...970b` (reject) |

The M128 synthetic raw-bit check did not cover enough product data to prove
greedy equivalence. Its arithmetic/layout change is not safe to promote. The
E8 B32 screen demonstrates that a large isolated kernel delta can still be
masked by the full TP3 schedule; no product claim is made from its micro.

## Scope and evidence

- Candidate daemon SHA-256:
  `d87999f2ff0dc0454a92270c9623039c8d54e3be2f9eadfb6e1837ec0bba262e`
- No weight, format, sampling, expert-count, KV, decode, speculation, Redline,
  gfx1100, gfx1151, or Qwen route changed.
- These were one fresh-process diagnostic product sample per arm. Repetition
  was skipped because M128 failed correctness and B32 failed directionally.

Evidence:

- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-mq2-m128-micro/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-grouped-b32-micro/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-prefill-m128-e8b32-bundle/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-prefill-m128-product-isolate/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-prefill-e8b32-product-isolate/`

## Next lever

The established grouped MQ2-Lloyd family remains the largest prefill tier.
The next screen is a native gfx1201 port of the gfx1151 Lloyd-to-int8 MMQ
algorithm, not another row/query tiling variant. It can change the throughput
class of the family by using gfx1201 integer WMMA while preserving the frozen
MQ2R weights.
