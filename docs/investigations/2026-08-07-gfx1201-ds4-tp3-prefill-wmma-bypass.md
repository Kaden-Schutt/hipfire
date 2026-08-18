# gfx1201 DeepSeek V4 TP3 prefill: pathological portable-WMMA bypass

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Promotion head: `9902ffeecdfef3f165683656ba599b33c70e41b1`  
Behavioral fix: `c8d4cb149`

## Verdict

Accepted. DeepSeek V4 Flash 0731 MQ2R TP3 prefill on three gfx1201 R9700s
was not limited to roughly 8 tok/s. TP rank 2 alone selected the portable DSA
WMMA attention route because its local head count is 16; ranks 0 and 1 have 24
heads and selected the established F32 route. On gfx1201 the portable route
took about 351 ms per rank-2 call, versus about 0.8 ms per call on ranks 0 and
1. Every peer reduction therefore waited on rank 2.

The exact-gfx1201 selector now bypasses that portable WMMA route in both the
direct and gathered DSA branches. No arithmetic, weights, format, sampling,
KV policy, expert count, or adjacent-architecture route changed.

## Root-cause evidence

An opt-in stage timer was introduced only for diagnosis and then reverted
before promotion. On the committed 180-token coherence prompt, 246 attention
calls decomposed as follows:

| Stage | Aggregate time |
|---|---:|
| compressor | 234.923 ms |
| indexer | 141.445 ms |
| attend | 28,917.673 ms |
| project/ring | 111.957 ms |

The attend time split by rank was decisive:

| Rank | Local heads | Calls | Attend total | Mean/call |
|---|---:|---:|---:|---:|
| 0 | 24 | 82 | 64.212 ms | 0.783 ms |
| 1 | 24 | 82 | 64.712 ms | 0.789 ms |
| 2 | 16 | 82 | 28,788.749 ms | 351.082 ms |

Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-tp3-prefill-attn-timing/`

The final selector is at
`crates/hipfire-arch-deepseek4/src/forward.rs:12064` and `:12126`.
The diagnostic timers were removed by `93e09cf70` and `9902ffeec`.

## Product result

Canonical fixture:

- model: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`
- model route: DeepSeek V4 Flash 0731 MQ2R
- devices: `0,1,2`, TP3, three gfx1201 R9700s
- prompt: `benchmarks/prompts/ds4-gfx942-ar-2048.txt`
- prompt MD5: `25e22faef15a20ae53501f1956e62b79`
- effective context: 2,052 tokens
- generated: 512 tokens
- batch 1, greedy, top-k 6 checkpoint default
- thinking off, speculation off
- Q8 requested; current DS4 contiguous cache implementation

| Route | Prefill | Decode |
|---|---:|---:|
| broken grouped-O-LoRA checkpoint | 7.9545 tok/s | diagnostic only |
| fixed, clean promotion build | **206.8064 tok/s** | **53.1396 tok/s** |

The prefill improvement is **25.998x**. The clean run used no diagnostic timing
environment and the daemon SHA-256 was
`edb87b5497a0e84d706df579514b942757d99223efe9efd707c522e0526c0e42`.

Correctness:

- assistant bytes SHA-256:
  `b6255240b6ccd34d621f152cc9898c71c340fa9393680ddb3d3efde2172b9c41`
- matches the previously accepted TP3 AR output exactly
- empty responses: 0
- attractor failures: 0
- finish: the requested 512-token length

Promotion evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-tp3-prefill-wmma-bypass-clean-acceptance/`

## Scope and next gate

This result removes a catastrophic rank-asymmetric route defect; it does not
establish a TP3 prefill ceiling. At about 207 tok/s, three R9700s have only
recovered roughly the established single-Strix-Halo-class prefill rate. The
next gate is a clean 2K production profile and occurrence-weighted optimization
of the new dominant prefill stage. Decode remains below the 60 tok/s TP3 target
and is not claimed improved by this prefill fix.

Skipped: repeated product samples, long-context, TP4, weights, format, quality,
sampling, top-k, KV, speculation, Redline/PM4, gfx1100 runtime, and Qwen runtime.
The 26x route correction and byte-identical 512-token output made a second
screening sample unnecessary before banking the exact-architecture fix.
