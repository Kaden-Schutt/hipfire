# gfx1201 DeepSeek V4 TP3 prefill: native gathered DSA WMMA

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Product commit: `fa08752c7`

## Verdict

Promoted. DeepSeek V4 Flash 0731 MQ2R prefill now uses an exact-gfx1201
wave32 WMMA kernel for gathered DSA attention. The production TP3 head split
is 24/24/16; the new symbol launches `ceil(H/16)` head groups and masks the
final eight rows on the 24-head ranks, so all three ranks use the same native
route.

The gfx11 symbol and selector remain unchanged. The gfx1201 kernel uses the
RDNA4 half8 operand split, `_w32_gfx12` intrinsic, and contiguous accumulator
mapping. The established developer disable remains available, but the good
route is the product default and requires no environment variable.

## Mechanism screen

Production layout: `D=512`, SWA window 128, top-K window 512, K=V tied. The
maximum dynamic LDS footprint is about 56 KiB per 16-head group and does not
grow with prefill batch size.

| TP3 local shape | F32 | gfx1201 WMMA | Speedup | Max absolute delta |
|---|---:|---:|---:|---:|
| B=128, H=24 | 9,604.714 us | 1,367.807 us | 7.022x | 8.356199e-5 |
| B=128, H=16 | 5,607.935 us | 788.950 us | 7.108x | 8.849427e-5 |

Both arms produced finite output. H=24 explicitly validates the masked tail;
H=16 validates the full-tile third rank.

Micro evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-dsa-wmma-micro/`

## Product result

Fixture:

- model: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`
- three gfx1201 R9700s, TP3, devices `0,1,2`
- prompt: `benchmarks/prompts/ds4-gfx942-ar-2048.txt`
- prompt MD5: `25e22faef15a20ae53501f1956e62b79`
- effective context: 2,052 tokens
- generated: 512 tokens
- batch 1, greedy, checkpoint-default top-k 6
- thinking off, speculation off
- prefill chunk B=1024; scratch remains 2.84 GiB/rank

| Route | Prefill | AR decode |
|---|---:|---:|
| B=1024, gathered F32 DSA | 287.7785 tok/s | 53.0931 tok/s |
| B=1024, native gfx1201 DSA WMMA | **360.6392 tok/s** | **53.1964 tok/s** |

Prefill improves by **25.32%**. Decode moved by only +0.19%, so no decode
gain is claimed.

Correctness:

- assistant SHA-256:
  `b6255240b6ccd34d621f152cc9898c71c340fa9393680ddb3d3efde2172b9c41`
- byte-identical to the accepted F32 route
- generated tokens: 512
- answer words: 395
- empty responses: 0
- attractor failures: 0
- finish: requested length

Daemon SHA-256:
`233c5efda16ad411c023bc5f186df19180bed5214de480d817a71c1b17e6c219`

Product evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-dsa-wmma-product/`

## Scope and next gate

The gathered path is promoted. The direct compressed-cache WMMA path remains
disabled on gfx1201 and was not changed because it is not the profiled product
route. The next gate is a fresh profile of this promoted B=1024 route and an
occurrence-weighted attack on the new dominant prefill stage.

Skipped: repeated product samples, direct-DSA port, B=2048, long-context, TP4,
weights, format, quality, sampling, top-k, expert count, KV representation,
speculation, retained PM4, gfx1100/gfx1151 runtime, and Qwen runtime. The
separate exact-architecture symbol, two-rank-shape screen, +25.32% product
delta, and byte-identical 512-token output clear this checkpoint.
