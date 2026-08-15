# DeepSeek V4 gfx1201 TP3 mixed E8 projection checkpoint

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Candidate: `425d3f1a9`  
Parent promoted line: `0d2a50aca` (`51.970852` tok/s median)

## Verdict

Promoted as a measured decode win. The exact-gfx1201 DeepSeek V4 MQ2R
backend now co-schedules independent MFP4G32E8SOA projections that share the
same rotated K=4096 activation:

- Q-LoRA A, joint KV, the main compressor pair, and ratio-4 indexer weights
  run in one mixed-row launch.
- After the main compressor consumes its scratch, the ratio-4 indexer
  compressor pair runs in a second mixed-row launch.
- The incumbent norm, APE, ring-state, cache-commit, RoPE, indexer score, and
  top-K arithmetic remains unchanged.

Admission requires the model-owned `Mq2rBackend::Gfx1201` and exact
`gpu.arch == "gfx1201"`. Portable, gfx1151, gfx942, Qwen, heterogeneous
gfx1100, and MTP paths retain their prior calls.

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

The mixed-job kernel preserved each accepted E8 row body and changed only
launch composition. Device-event micro-screens used production row sets and
large replica working sets:

| Compression ratio | Jobs | Raw-bit comparisons | Serial | Packed | Speedup |
|---:|---:|---:|---:|---:|---:|
| 4 | 7 | 4,160/4,160 | 46.856824 us | 21.094294 us | 2.2213x |
| 128 | 4 | 2,560/2,560 | 26.557223 us | 14.283074 us | 1.8593x |

The occurrence-weighted channel projection was approximately 0.799 ms per
rank/token. That projection was used only as the product-bench admission gate;
the promotion verdict below comes from the full route.

Micro evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-mixed-jobs/`

## Product checkpoint

| Fresh process | Decode tok/s | Prefill tok/s | Graph blobs |
|---|---:|---:|---:|
| 1 | 53.376417 | 57.406769 | 7,349 |
| 2 | 53.457052 | 57.354549 | 7,349 |
| 3 | 53.293764 | 57.177072 | 7,349 |

- Decode median: **53.376417 tok/s**
- Prior promoted median: **51.970852 tok/s**
- Delta: **+1.405564 tok/s / +2.7045%**
- Decode range spread: **0.3059%**
- Prefill median: 57.354549 tok/s, +3.1285% diagnostic; TP prefill remains
  token-serial
- Route identity: three ranks, 86 peer barriers, 7,349 kernarg blobs
- Graph reduction: 7,844 to 7,349 blobs, removing 495 nodes

All three decoded outputs were byte-identical to the promoted route:
`b6255240b6ccd34d621f152cc9898c71c340fa9393680ddb3d3efde2172b9c41`.
Every run generated 512 tokens, ended by length, contained 395 answer words,
and reported zero empty or attractor failures.

Candidate binary SHA-256:

- `hipfire`: `f91206fdab2ad2db68ce9ef468fbcad9f9ec9bde2b7743c230daecd7e93dd2d4`
- `daemon`: `b0dda1768e68c25dc26800ed9c170c22f642186b3f4a981c17553bc4b237b35f`

Product evidence:

- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-mixed-product/product-screen/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-mixed-product/product-run2/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-mixed-product/product-run3/`

## Scope and next gate

No weight, format, top-k, sampling, expert-count, KV, speculation, retained
PM4, or long-context change was included. Redline PM4 is not applicable to
this three-rank HipGraph route.

The decode campaign remains active at 53.38 tok/s. Re-profile the promoted
route and pursue only another occurrence-weighted candidate projecting at
least 2% toward 60 tok/s. Production batched prefill follows the decode gate.
