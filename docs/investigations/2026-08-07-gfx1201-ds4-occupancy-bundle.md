# DeepSeek V4 gfx1201 TP3 occupancy bundle checkpoint

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Candidate source checkpoint: `6712a3d8d`
Parent promoted line: `150d3585c` (`53.376417` tok/s median)

## Verdict

**Corrected after rebuilding the actual product daemon.** The three-process
**54.903757 tok/s** checkpoint is a real **+2.8615% / +1.527340 tok/s** win,
but it contains the T1024 HC-control lever only. The daemon used for those runs
was stale with respect to the head-strided RoPE source and did not contain its
kernel or environment gate. Three fresh processes produced byte-identical
decoded output, and a separate five-genre baseline/candidate battery also
matched every response byte-for-byte. T1024 remains pending explicit operator
approval for a persistent default flip because it changes the FP32 reduction
tree.

The branch remains safe by default at this checkpoint: T1024 is selected only
with this developer opt-in while certification is in progress:

- `HIPFIRE_HC_CTRL_T1024=1`

The head-strided RoPE remains opt-in and is not part of the product checkpoint.
Portable, gfx1151, gfx1100, gfx942, Qwen, MiniMax, and non-MQ2R paths remain on
their prior kernels.

## Mechanisms and channel gates

### Head-strided indexer-Q RoPE

The incumbent H64/D128/R64 kernel launches one 32-lane wave and loops over all
64 query heads. The candidate preserves its arithmetic loop but stripes head
iterations across 32 independent waves.

| Geometry | Incumbent | Candidate | Speedup | Saved per rank/token |
|---:|---:|---:|---:|---:|
| 8 head waves | 14.244437 us | 4.298514 us | 3.3138x | 0.208864 ms |
| 16 head waves | 14.001454 us | 3.596750 us | 3.8928x | 0.218499 ms |
| **32 head waves** | **14.227992 us** | **3.241646 us** | **4.3891x** | **0.230713 ms** |

The selected 32-wave geometry passed **98,304/98,304 raw-bit comparisons** at
positions 0, 1, 2,052, and 131,071. An earlier per-head specialization was
rejected before routing because LLVM constant folding introduced 1-ULP drift.

After rebuilding the real product daemon, the corrected direct-HIP profile
reduced this symbol from about 1.211 to 0.139 ms per position across all three
ranks and reduced mean traced GPU time from 14.610 to 13.911 ms/rank/position.
The canonical product screen nevertheless measured **54.928617 tok/s**, only
**+0.045%** over the T1024-only median. The route is therefore rejected for
product despite its device-time win.

Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-indexer-rope-heads/`
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-occupancy-profile-v2/`
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-occupancy-bundle-v2/`

### Wide HC control/finalize

The existing T1024 source widens each of 24 control blocks from 256 to 1,024
threads. On gfx1201 it measured:

- 8.758969 -> 7.050363 us/call, 1.2423x
- 1.708605 us saved/call
- 0.146940 ms projected per rank/token across 86 calls
- 24/24 raw-equal outputs on the production-shape synthetic channel

This source is not generally bit-exact because the LDS reduction tree changes
from 8 to 32 partials; model-level generation equality is therefore the
promotion gate, not the synthetic ULP result.

Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-hc-control-t1024/`

## Canonical product fixture

- Model: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`
- Model SHA-256: `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`
- Prompt: `benchmarks/prompts/ds4-gfx942-ar-2048.txt`
- Prompt MD5: `25e22faef15a20ae53501f1956e62b79`
- Effective context: 2,052 tokens
- Generation: 512 tokens, batch 1, greedy, thinking off, speculation off
- Experts per token: checkpoint default 6
- Requested KV: Q8; current DS4 path remains F32 contiguous
- Route: TP3 on three gfx1201 R9700s through `scripts/serve_harness.py`

| Fresh process | Decode tok/s | Prefill tok/s | Output SHA-256 |
|---:|---:|---:|---|
| 1 | 54.913088 | 57.654797 | `b6255240...b9c41` |
| 2 | 54.557241 | 58.591721 | `b6255240...b9c41` |
| 3 | 54.903757 | 59.000910 | `b6255240...b9c41` |

- Decode median: **54.903757 tok/s**
- Prior median: **53.376417 tok/s**
- Delta: **+1.527340 tok/s / +2.8615%**
- Decode range spread: **0.6481%**
- Prefill median: 58.591721 tok/s, diagnostic only
- Graph identity: 3 ranks, 86 barriers, 7,349 kernarg blobs
- Every run: 512 generated tokens, 395 answer words, finish by length, zero
  empty responses, zero attractor failures

Product evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-occupancy-bundle/`

## Varied-prompt correctness battery

A matched baseline/candidate `serve_harness.py --mode battery` run covered
code, reasoning, factual, prose, and instruction prompts at greedy sampling.
All five candidate responses matched their baseline response byte-for-byte,
including generated-token count and finish reason. Both length-capped rows and
all three natural-stop rows matched; neither arm produced empty or attractor
failures.

Candidate decode was 60.27-60.96 tok/s across the five short rows versus
58.20-58.76 for baseline. Those numbers are correctness diagnostics, not the
canonical performance claim.

## Binary identity and remaining gate

- `hipfire` SHA-256:
  `682f3853cfb2d19bdf4f31be688d945bbcb212bda86893c339272f3c8b869f0b`
- `daemon` SHA-256:
  `26d33688b9cee2ea6e91e3efb0fed1578b7282960a6393564bc170ad05b84bd1`

The recorded binary hashes above identify the original evidence bundle, but
the product attribution is corrected here: only T1024 was active. The remaining
checkpoint action is operator approval to make T1024 the exact-gfx1201 MQ2R
default. The head-strided RoPE is product-rejected. No weight, format, top-k,
sampling, expert-count, KV, speculation, Redline/PM4, TP4, or long-context
change is part of this checkpoint.
