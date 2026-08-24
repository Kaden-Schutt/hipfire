# 2026-08-15 — DFlash on dense Qwen3.6-27B loses to AR beyond ~3k context

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.
See [`README.md`](README.md) in this directory before citing it anywhere.

## Disposition

On this pair, at this configuration, **DFlash is a net throughput loss on 7 of
8 turns.** It is not a marginal loss: by ctx 11.6k it is running at 40% of AR.
The default `dflash_mode=off` is doing real work here.

## Fixture

| | |
|---|---|
| host | `hiptrx`, 4× Radeon AI PRO R9700 (gfx1201), ROCm 7.14 |
| branch / commit | `arch/saddle` @ `2acd33999` |
| target | `qwen3.6-27b.mq4` (14,979,312,640 B) |
| draft | `qwen36-27b-dflash-mq4.hfq` (919,401,472 B) |
| harness | `scripts/serve_harness.py --mode session` (user-facing serve path) |
| fixture | `benchmarks/prompts/session_coding.json`, md5 `c0d470288bde3f1e54e4bba04da8f8a2`, 8 turns |
| config | `--sampling registry` (default), `--thinking-effort medium`, `--max-seq 32768`, `--max-tokens 4096`, kv=q8 |
| arms | `--dflash on` vs `--dflash off`, same fixture, same seed policy |

Context accrues through generated output rather than a long prompt, so each
turn measures decode at a progressively larger KV.

## Result

| turn | ctx | AR tok/s | DFlash tok/s | Δ | τ |
|---|---:|---:|---:|---:|---:|
| t1 | 49 | 36.5 | 33.8 | −7.4% | 1.80 |
| t2 | 2,706 | 36.1 | **39.0** | **+8.0%** | 2.35 |
| t3 | 4,779 | 35.5 | 28.6 | −19.4% | 1.99 |
| t4 | 7,349 | 35.0 | 24.6 | −29.7% | 2.28 |
| t5 | 9,396 | 34.7 | 22.4 | −35.4% | 1.62 |
| t6 | 11,623 | 34.4 | 13.8 | −59.9% | 2.09 |
| t7 | 12,174 | 34.0 | 22.5 | −33.8% | **2.83** |
| t8 | 13,544 | 33.6 | 19.7 | −41.4% | 2.06 |

- **AR is flat in context**: 36.5 → 33.6 across 15.6k, −8%.
- **DFlash falls 49.5%** from its t2 peak to t8.
- **τ does not decay**: mean 2.13, range 1.62–2.83, correlation with ctx
  `r = +0.278` — if anything faintly *positive*.
- Throughput correlation with ctx: AR ≈ flat, DFlash `r = −0.874`.

## The diagnostic point

**t7 settles it.** It carries the highest acceptance in the entire run
(τ = 2.83) and is still **33.8% slower than AR**. Acceptance is not the
failing variable.

Since τ is flat and target decode is flat, the entire degradation is
**drafter-side cost per cycle growing with context**. DFlash needs

$$\tau > \frac{c_{\text{draft}}(L) + c_{\text{verify}}}{c_{\text{target}}}$$

and with τ pinned near 2.1 while $c_{\text{draft}}(L)$ grows, the break-even
threshold walks up through the achieved τ somewhere between ctx 2.7k and 4.8k.

That has a concrete implication for where effort goes: **improving draft
*quality* cannot fix this.** Only reducing draft *cost* can. The drafter is an
NInfer 1-full + (n−1)-SWA stack (`dflash_spec.rs:126`); the SWA layers are
bounded at the trained `sliding_window` (2048 for this artifact), so the
growing term is the single full-attention layer sweeping all of $L$.

## Confounder that must be settled first

`AGENTS.md` records that the 3.6 drafts were trained on 3.5 traces and shows
the resulting target-distribution mismatch on A3B (τ = 1.22 on hard code).
`dflash_spec.rs:142` cites a **6-turn chain holding τ 4.4–6.1 out to ctx
20695** — roughly 2× the acceptance measured here. If that figure came from the
**3.5** pair, then this pair carries a large acceptance penalty before context
is a factor, and the cheapest available win is a distribution-matched draft,
not a cost reduction.

**Re-running this exact fixture against `qwen3.5-27b.mq4` +
`qwen35-27b-dflash-mq4.hfq` is the next measurement**, and it is a
prerequisite for interpreting the numbers above as a statement about DFlash
rather than about this particular draft.

## Incidental

Both arms failed the harness retrieval gate (`EXIT=1`, missing `dedupe`), and
both scored recall 2/3 on t7. Reading t7's DFlash text: asked which function
was requested *first*, it answered "the streaming BLAKE3 hasher" — that was
turn [1]; turn [0] asked for an architecture sketch. The off-by-one is present
in **both** arms, so it is a target-model behaviour, not a speculation
artifact.

## Reproduce

```bash
python3 scripts/serve_harness.py \
  --model  ~/.hipfire/models/qwen3.6-27b.mq4 \
  --draft  ~/.hipfire/models/qwen36-27b-dflash-mq4.hfq \
  --mode session --thinking-effort medium \
  --max-seq 32768 --max-tokens 4096 \
  --dflash on            # and again with: --dflash off
```
