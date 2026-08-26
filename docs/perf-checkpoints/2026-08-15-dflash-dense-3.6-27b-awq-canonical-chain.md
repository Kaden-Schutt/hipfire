# 2026-08-15 — DFlash vs AR on the *pinned* dense Qwen3.6-27B AWQ trunk

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.

Supersedes
[`2026-08-15-dflash-dense-3.6-27b-chain-decay.md`](2026-08-15-dflash-dense-3.6-27b-chain-decay.md)
and its
[amendment](2026-08-15-dflash-dense-3.6-27b-chain-decay-AMENDMENT.md), which
used a trunk that failed the pin. Both are left unmodified. **This record is the
one to cite** — it is the first of the three run on a digest-verified trunk.

## Fixture — verified, not assumed

| | |
|---|---|
| host | `hiptrx`, 4× Radeon AI PRO R9700 (gfx1201), ROCm 7.14 |
| branch | `arch/saddle` @ `2acd33999` |
| trunk | `qwen3-27b-3.6.mq4-awq.remote-mi300x` |
| trunk size | `14984158208` — **matches** the AGENTS.md pin |
| trunk SHA-256 | `86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc` — **matches** the pin |
| draft | `qwen36-27b-dflash-mq4.hfq`, md5 `204c4c4ceab30cb9ebc118fa9d59a446` — **matches** AGENTS.md |
| harness | `scripts/serve_harness.py --mode session` |
| fixture | `benchmarks/prompts/session_coding.json`, md5 `c0d470288bde3f1e54e4bba04da8f8a2` |
| config | registry sampling (`temp 1.0 / top_p 0.95 / top_k 20 / min_p 0.0`), `--thinking-effort medium`, `--max-seq 32768`, `--max-tokens 4096`, kv=q8 |

**Trunk identity is a trap on this host.** Three local artifacts share the
pinned size `14984158208`; one of them, `qwen3-27b-3.5.mq4-awq`, is a
*different model* (`ea615949…`). Selecting by size — or by filename — picks
wrong. Only the digest distinguishes them.

## Result

| turn | ctx | AR tok/s | DFlash tok/s | Δ | τ |
|---|---:|---:|---:|---:|---:|
| t1 | 49 | 35.7 | **40.7** | **+14.0%** | 1.76 |
| t2 | 2,513 | 35.3 | **37.8** | **+7.1%** | 2.19 |
| t3 | 5,144 | 34.8 | 26.6 | −23.6% | 1.62 |
| t4 | 7,618 | 34.3 | 31.4 | −8.5% | 2.22 |
| t5 | 10,559 | 33.9 | 22.2 | −34.5% | 1.58 |
| t6 | 12,827 | 33.7 | 11.9 | −64.7% | 2.55 |
| t7 | 13,279 | 33.5 | 28.6 | −14.6% | **3.09** |
| t8 | 15,639 | 33.1 | 16.5 | −50.2% | 1.65 |

- **AR is almost perfectly flat**: 35.7 → 33.1, −7.3%, `r(ctx) = −0.997`.
- **DFlash falls 59.5%**, `r(ctx) = −0.845`.
- **τ does not decay**: mean 2.08, range 1.58–3.09, `r(ctx) = **+0.298**`.
- DFlash wins **2 of 8 turns**, both below ctx 2.6k. Crossover is between
  ctx 2,513 and 5,144.

## The diagnostic point

**t7 is decisive.** It carries the highest acceptance of the run (τ = 3.09) and
is still **14.6% slower than AR**. t6 pairs τ = 2.55 with the worst throughput
in the run (−64.7%).

Acceptance is not the failing variable. Target decode is flat. Therefore the
entire degradation is **drafter cost per cycle rising with context**:

$$\tau > \frac{c_{\text{draft}}(L) + c_{\text{verify}}}{c_{\text{target}}}$$

With τ pinned near 2.1 and $c_{\text{draft}}(L)$ growing, the break-even
threshold walks up through the achieved τ between ctx 2.5k and 5.1k.

**Implication for where effort goes:** improving draft *quality* cannot fix
this — τ is already sometimes 3.09 and still losing. Only reducing draft *cost*
can. The drafter is an NInfer 1-full + (n−1)-SWA stack
(`dflash_spec.rs:126`); its SWA layers are bounded at the trained
`sliding_window` (2048 for this artifact), so the term growing with $L$ is the
single full-attention layer.

## Against the registry claim

`registry/models.json` → `qwen3.6:27b` advertises
**"44 tok/s AR / 185 tok/s w/ draft on code"**. Measured here on the pinned
trunk: **35.7 AR / 40.7 peak DFlash**. The AR figure is 24% optimistic; the
draft figure is **4.5× optimistic**.

The fixture is a multi-turn *coding session* with `reasoning_effort: medium`,
not the single-shot code completion the desc likely refers to, and this is one
card class. That plausibly explains part of the gap but not a 4.5× one. **The
desc should be re-derived or scoped** — as written it will mislead anyone
sizing hardware from the registry.

## Quality

Both arms scored recall **3/3** on t7 and t8, and the AR arm exited clean
(`EXIT=0`). On the *non*-pinned trunk, t7 scored 2/3 in both arms and named the
wrong function; on the pinned trunk both arms correctly identify
`dedupe_files`. That is one more reason the earlier records are not comparable.

## Reproduce

```bash
sha256sum ~/.hipfire/models/qwen3-27b-3.6.mq4-awq.remote-mi300x
# require 86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc

python3 scripts/serve_harness.py \
  --model ~/.hipfire/models/qwen3-27b-3.6.mq4-awq.remote-mi300x \
  --draft ~/.hipfire/models/qwen36-27b-dflash-mq4.hfq \
  --mode session --thinking-effort medium \
  --max-seq 32768 --max-tokens 4096 \
  --dflash on            # and again with: --dflash off
```
