# 2026-08-15 — Matched vs mismatched DFlash draft: τ +28%, decay unchanged

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.

Companion to
[`2026-08-15-dflash-3.6-27b-awq-four-arm-matrix.md`](2026-08-15-dflash-3.6-27b-awq-four-arm-matrix.md).
Same fixture, same host, same sampling; the variable is the target/draft pair.

## Why this run exists

`dflash_spec.rs:142` cites a 6-turn chain holding **τ 4.4–6.1 out to ctx
20695**. Every arm measured on the dense 3.6 pair topped out near 2.4 mean /
3.37 peak. `AGENTS.md` records that the 3.6 drafts were trained on 3.5 traces,
so a matched 3.5 pair is the direct test of whether draft/target distribution
mismatch accounts for the gap.

## Fixture

| | |
|---|---|
| host | `hiptrx`, 4× R9700 (gfx1201), ROCm 7.14 |
| branch | `arch/saddle` @ `2acd33999` |
| **arm A** | trunk `qwen3-27b-3.6.mq4-awq.remote-mi300x` (sha256 `86a5f80f…`, **pinned**) + draft `qwen36-27b-dflash-mq4.hfq` (md5 `204c4c4c…`, **pinned**) |
| **arm B** | trunk `qwen3.5-27b.mq4` (registry `qwen3.5:27b` canonical) + draft `qwen35-27b-dflash-mq4.hfq` (md5 `7b6df2a4…`, **matches AGENTS.md**) |
| common | `--mode session`, `--sampling greedy`, `--thinking-effort medium`, `--max-seq 32768`, `--max-tokens 4096`, fixture md5 `c0d470288b…` |

**Arm B required `HIPFIRE_DFLASH_CTX_CAP=0` and is therefore NOT a deployable
configuration** — see "The 3.5 draft is not an escape hatch" below.

## Result

| turn | 3.6 τ | 3.5 τ | 3.6 tok/s | 3.5 tok/s |
|---|---:|---:|---:|---:|
| t1 | 1.99 | 2.89 | 48.0 | 60.3 |
| t2 | 3.01 | 3.23 | 44.5 | 48.3 |
| t3 | 2.10 | 2.94 | 35.8 | 39.7 |
| t4 | 2.64 | **4.14** | 30.1 | 41.1 |
| t5 | 2.70 | 2.70 | 33.8 | 26.3 |
| t6 | 2.50 | 1.77 | 10.5 | 25.5 |
| t7 | 3.37 | **4.38** | 27.7 | 22.8 |
| t8 | 1.98 | 2.90 | 18.6 | 22.7 |
| **mean τ** | **2.44** | **3.12** | | |
| **decode decay** | | | **−61%** | **−62%** |

## Two findings

**1 — Draft mismatch explains the τ gap.** Mean τ rises 28% (2.44 → 3.12) and
peak reaches **4.38**, landing at the floor of the documented 4.4–6.1 range.
Together with that figure having been taken on 7900 XTX rather than R9700, the
`dflash_spec.rs:142` claim is no longer anomalous. It was a matched pair on
different silicon. Nothing is wrong with the comment; the 3.6 pair simply
carries a distribution penalty, exactly as `AGENTS.md` warns.

**2 — Acceptance and cost are independent. This is the important one.**
τ improves 28% and peaks near double the 3.6 pair's worst turns, and decode
still decays **−62% versus −61%** — indistinguishable. Across two draft pairs,
two samplings, and prefix cache on/off, *nothing that moves acceptance moves
the decay*.

That closes the question the four-arm matrix opened. The decay is not a
speculation-quality phenomenon. It is drafter cost per cycle rising with $L$,
and the only lever that touches it is reducing that cost.

## The 3.5 draft is not an escape hatch

Arm B's τ is bought with VRAM. The 3.5 draft declares no sliding window, so it
runs the **Legacy** path: all five layers hold full KV over the whole context.

| ctx | 3.5 legacy (5 full) | 3.6 windowed (4×W2048 + 1 full) |
|---:|---:|---:|
| 8,192 | 0.78 GB | 0.31 GB |
| 32,768 | **3.12 GB** | 0.78 GB |
| 131,072 | **12.50 GB** | 2.66 GB |

That is why `HIPFIRE_DFLASH_CTX_CAP` defaults to 8192, and arm B only completed
because the cap was removed. Windowing bounds four of five layers; the fifth is
still $O(L)$ and dominates both curves — 2.58 GB of the windowed 2.66 GB at
128k comes from that one layer.

## Defect found while running this

With the 3.5 draft at default settings, turns past the 8192 cap produced a hard
`[context_length retryable=false]` error that killed 6 of 8 turns.
`AGENTS.md` documents over-cap behaviour as *"over-cap requests fall back to AR
(identical output, slower)"*. **That graceful degradation does not happen on
the Legacy path** — the request fails instead. Any user pairing a windowless
draft with a long session hits this.

## Reproduce

```bash
# arm B (non-deployable; uncapped Legacy draft KV)
HIPFIRE_DFLASH_CTX_CAP=0 python3 scripts/serve_harness.py \
  --model ~/.hipfire/models/qwen3.5-27b.mq4 \
  --draft ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
  --mode session --thinking-effort medium --sampling greedy \
  --max-seq 32768 --max-tokens 4096 --dflash on
```
