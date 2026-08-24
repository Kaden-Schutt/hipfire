# Qwen3.8 27B — v6 chat-template calibration corpus, scored 2×2 against prose and conversation references (gfx1201)

**Lifecycle:** `historical`. Fixture-bound measured evidence from the
chat-template calibration-corpus campaign. It is **not** a current default, an
automatic baseline, a product claim, or an admission decision.

**Disposition:** At a fixed MQ4G256 wire format, swapping the calibration corpus
from **v5** (prose-weighted) to **v6** (rendered chat-template conversations)
produces a result that *depends on which reference you score against*, and the
two references disagree about which arm wins:

- On **WikiText-2** (prose, the OOD tripwire) the arms are effectively tied on
  KLD — v6 ahead by 0.0011 absolute, 1.3 % relative — and **v5 wins PPL**
  (6.4898 vs 6.6727).
- On the **v6 held-out selector** (100 % rendered conversation, 0 % prose) v6
  wins **both** metrics decisively: KLD 1.023882 vs 1.064046 (0.040 absolute,
  3.8 % relative) and PPL 10.9569 vs 13.232 (2.27 absolute, 17 % relative).

**Ranking arms on WikiText-2 PPL alone would have selected v5, which is the
wrong arm for a chat model.** Even on KLD — where the two references agree on
the *direction* — WikiText-2 understates the margin by roughly 30× in absolute
delta and 3× in relative delta.

The load-bearing consequence is retroactive and is the reason this record
exists: **every prior arm ranking in this campaign that was scored on a
prose reference was measured on the wrong instrument for a chat model.** Those
rankings are not thereby *wrong*, but they are no longer *sufficient*, and any
ranking whose margin was under a few percent on prose should be considered
unresolved until re-scored on a deployment-distribution reference.

**Source:** branch `quant/quality`, producer binary at commit
`814d8fd95859a4843504fae2c81e9c073db2038d`.

---

## Fixture identity

| field | value |
|---|---|
| producer host | MI300X, `gfx942`, CDNA3, 235 GB RAM, 205 GB VRAM — **PRODUCER ONLY** |
| scoring host | hiptrx, 4× `gfx1201` Radeon AI PRO R9700, 122 GB RAM, 34.2 GB VRAM |
| producer binary | `hipfire-quantize` @ `814d8fd9`, md5 `200a7f0e80624a09fcfb5da63901d5f9`, sha256 `cb42f5344e8a7c7448492a3254c649dfd4b2a554b99f05a9368e38246d3cf4bf` |
| scoring binary | `eval_hipfire`, md5 `89083d01a9d42e418b17996b673f71cf` (worktree `d64e58fb`) |
| parent checkpoint | `/scratch/parents/qwen3.8-27b`, upstream unquantized, 18/18 shards, 52 GB |
| format | MQ4G256, 136 B/group, 15,662,615,552 B written |

### Calibration artifacts

| arm | path | bytes | md5 |
|---|---|---|---|
| v5 | `qwen3.8-27b.calib.hfq` | 31,481,106,432 | `c94a6560e4b82ac1e7a2236561c2216d` |
| v6 | `qwen3.8-27b.calibv6-814d8fd.hfq` | 31,481,139,200 | `9e2a7ab00e0b982e6793f4d2b0b86379` |

Both carry 496/496 tensors at 262,144 tokens/tensor uniform, 992 keys
(496 Hessian diagonals + 496 imatrix vectors).

### Quantized artifacts

| arm | bytes | md5 | hash match after transfer |
|---|---|---|---|
| v5 | 15,662,615,552 | `547f890e47e28761073e0edd27177ff2` | yes |
| v6 | 15,662,615,552 | `56fb12b91d2c71ada2d8c7f4f6a5a9f0` | yes |

### References

| ref | bytes | md5 | oracle NLL | oracle PPL |
|---|---|---|---|---|
| WikiText-2 | 50,675,552 | `8a21364051d844b97c122e2c895f56d8` | 1.830742 | 6.2385 |
| v6 selector | 50,675,552 | `4e5618b6a3485c3afcf326cb26f81fc8` | 2.555774 | 12.8813 |

Both: `n_ctx` 2048, top-k 256, 24 chunks, 24,552 scored tokens.

---

## The 2×2

| calibration | ref | KLD | NLL | PPL |
|---|---|---|---|---|
| v5 | WikiText-2 | 0.087921 | 1.870233 | 6.4898 |
| v6 | WikiText-2 | **0.086790** | 1.898022 | 6.6727 |
| v5 | v6 selector | 1.064046 | 2.582638 | 13.232 |
| v6 | v6 selector | **1.023882** | **2.393972** | **10.9569** |

Margins for v6 over v5:

| ref | ΔKLD abs | ΔKLD rel | ΔPPL abs | ΔPPL rel |
|---|---|---|---|---|
| WikiText-2 | 0.001131 | 1.3 % | −0.1829 (v5 wins) | −2.8 % |
| v6 selector | 0.040164 | 3.8 % | 2.2751 | 17.0 % |

---

## Recipe held constant across both arms

Only the calibration artifact differs. Everything else is pinned:

- `--hessian <calib.hfq>` (alias `--calib`) **plus explicit `--awq`**.
  `--hessian` alone does **not** imply AWQ — its only consumer is LDLQ, which
  was never completed — whereas `--imatrix` alone *does* imply AWQ at 0.55. HFQM
  input therefore requires `--awq` to be passed explicitly or AWQ silently does
  not run.
- AWQ alpha 0.55 (the shipped default), geo-mean normalized to 1.
- `HIPFIRE_Q8_CLASSES=lm_head,embed` **and** `--q8-router`. Both are required:
  without narrowing, `is_q8_tensor` drags attention to Q8. `--no-q8-router` is
  inert on a dense model.
- No `--ldlq`, no `--hessian-dir`, no `--allow-mq*`, no `--kmap-dense`, no
  `--no-kmap`, no `--tier-ratio` change.
- Scoring: 24 chunks, top-k 256, `n_ctx` 2048, prefill scoring mode, kv-mode q8,
  `HIPFIRE_NORMALIZE_PROMPT=0`, `HIPFIRE_GRAPH=0`, deterministic greedy prefill
  against the fixed BF16 teacher reference.

---

## Validity controls

**The arms are not byte-identical.** `cmp` diverges at byte 2,716,205,761
(201 vs 215) and the md5s differ. This is checked because a prior run of this
same experiment produced *byte-identical* artifacts (md5 `129909ad0fed`) and
identical KLD 0.066668 — a silent no-op caused by `--imatrix` not implying AWQ
in that configuration. A calibration A/B whose outputs hash equal has measured
nothing, and that failure is not visible in the metrics.

**No quantized artifact was scored on the producer.** MI300X ran only
`calib_sweep` and `hipfire-quantize`. All four KLD/PPL figures and the decoded
sample were measured exclusively on hiptrx `gfx1201`.

**Coherence was eyeballed, not inferred from the metrics.** A 1278-token greedy
sample from the v6 arm on the prompt *"Write a short story about a robot
learning to paint"* is coherent, on-prompt, and closes properly. Most frequent
token id 13 appears 103/1278 times (8 %) — no single-token attractor. This
matters because the uncalibrated gfx942 baseline in
`2026-08-16-qwen38-27b-mq4-awq-alpha-kld.md` emits a two-token attractor on
every prompt while passing statistical gates.

**Sanity band.** WT2 KLD 0.086–0.087 sits slightly above the prior gfx1201 band
of 0.036–0.067 but inside the alpha-sweep range 0.063–0.083 and the documented
0.063–0.136. PPL 6.48–6.67 is near the 6.2385 oracle and the prior 6.51–6.57.
Degenerate output on this stack reads KLD ≈ 12.7 / PPL ≈ 1.8e6 against a
uniform `ln(248320) = 12.42`, so neither arm is degenerate.

**Append-only.** New output names throughout; no existing `.calib.hfq`, `.bin`,
or `.mq4` was overwritten.

---

## Caveats that bound what this record supports

1. **KLD here is top-k 256, not full-vocabulary.** Absolute values are **not**
   comparable to `llama.cpp --kl-divergence` tables or to any full-vocab figure.
   Relative ranking *within* this instrument is valid; cross-table absolute
   comparison is not. The oracle NLL/PPL were computed from full-vocab teacher
   logits before top-k serialization, so those two anchors are full-vocab while
   the KLD column is truncated.

2. **The two PPL columns are different instruments and must never be compared
   as model-quality figures.** The v6 selector's 12.88 oracle PPL versus
   WikiText-2's 6.24 does not mean the model is "worse" on conversation; the
   selector is 100 % rendered conversation bytes — function names, JSON
   arguments, tool results, 237 `im_start` markers — and is intentionally harder
   and out-of-distribution relative to the pretraining corpus. It is a *ranking*
   instrument on the deployment distribution, nothing more.

3. **Two references, one model family, one format.** This is a single 2×2 on
   Qwen3.8-27B at MQ4G256. It establishes that reference choice can invert an
   arm ranking; it does not quantify how often that happens, and it does not
   generalize to other formats or model families without measurement.

4. **This is not an admission decision.** Current product claims live in
   `docs/BENCHMARKS.md`; validation routes live in `docs/VALIDATION.md`.

---

## What this changes going forward

Any calibration or quantization arm comparison for a **chat/instruct** target
should carry a deployment-distribution reference alongside the prose tripwire.
Prose remains valuable *as a tripwire* — it catches catastrophic regressions
that a narrow in-distribution reference can mask — but it is not sufficient as
the selector, because it demonstrably ranks arms in the wrong order at margins
this campaign cares about.

---

## Amendment, same day — the disposition above overstates the KLD result

The commit that introduced this record is titled *"prose references rank
calibration arms in the wrong order."* That is **true only of PPL**, and the
title should be read as amended to *"prose understates calibration margins ~3×
and inverts the PPL ranking."*

On **KLD**, both references select the same arm:

| ref | v5 KLD | v6 KLD | winner |
|---|---|---|---|
| WikiText-2 | 0.087921 | 0.086790 | v6 |
| v6 selector | 1.064046 | 1.023882 | v6 |

This campaign ranked arms on **KLD**, not PPL. So the direction of prior
prose-scored rankings was most likely correct; what was wrong was the
**resolution** — prose compresses the margin by roughly 3× relative and 30×
absolute.

That narrows the rework this record implies. It is **not** "re-score
everything." It is:

- **Arm pairs previously declared a tie on prose are unresolved.** A tie at
  prose resolution can be a ~4 % separation on the deployment distribution.
  This is the actionable set.
- **Clear prose wins probably survive**, since direction agreed here.
- **Any ranking that turned on PPL rather than KLD is suspect outright**, because
  PPL inverted.

The original disposition's phrase "measured on the wrong instrument" remains
accurate as to *sensitivity*, but should not be read as "measured the wrong
winner." One 2×2 on one model at one format cannot support the stronger claim.

---

## Amendment 2, same day — the 2×2 ran at the worst AWQ alpha, and its sanity check was invalid

Two defects in how the WikiText-2 KLD of this 2×2 was justified. Neither changes
the measured numbers, both bound what they support.

**1. Both arms ran at AWQ `alpha=0.55`, the worst end of the documented sweep.**
`2026-08-16-qwen38-27b-mq4-awq-alpha-kld.md` measured `alpha=0.05` at
`KLD 0.063102 / NLL 1.875801 / PPL 6.5260` and found `alpha=0.05` beats the
shipped `alpha=0.55` default by **24.3 %** on WikiText-2. This 2×2 held
`alpha=0.55` fixed across both arms, so it is **internally valid** as a corpus
comparison — but it was conducted roughly 24 % away from the achievable KLD at
this format.

The consequence: **the corpus × alpha interaction is unmeasured.** The v6 corpus
wins at `alpha=0.55`; whether it still wins at `alpha=0.05` is not established by
this record. A corpus that helps most when smoothing is aggressive could help
less, or more, when smoothing is nearly off. Re-running this 2×2 at
`alpha=0.05` is the experiment that would close it.

**2. The originally-recorded sanity check was cross-architecture and
arithmetically wrong.** The producing agent justified WT2 KLD 0.086–0.087 as
"slightly above prior gfx1201 band 0.036–0.067 but within alpha-sweep range
0.063–0.083." Both halves fail:

- `0.087921 > 0.083`, so the value is **outside** the range cited to excuse it.
- The alpha sweep it compared against was measured on **gfx942**; this 2×2 was
  measured on **gfx1201**. Absolute KLD is not comparable across
  architectures, so that band was never a valid reference for these numbers in
  the first place.

The value is not anomalous once alpha is accounted for — `alpha=0.55` *should*
produce the sweep's worst KLD — but it was not anomalous *for the reason given*,
and the reasoning as recorded should not be reused.

**Host verification (unaffected, confirmed independently).** All four figures
were re-read from the scoring logs on hiptrx after this record was first
committed:

```
GPU dev 0: gfx1201 (34.2 GB VRAM, HIP 7.14)
eval_hipfire: arch=gfx1201  detected arch_id=5 (HFQ header)
eval_hipfire: scored 24552 tokens in 147.6s (166 tok/s)
v5 × wt2     KLD 0.087921  NLL 1.870233  PPL  6.4898
v5 × v6sel   KLD 1.064046  NLL 2.582638  PPL 13.2320
v6 × wt2     KLD 0.086790  NLL 1.898022  PPL  6.6727
v6 × v6sel   KLD 1.023882  NLL 2.393972  PPL 10.9569
```

Scoring ran on hiptrx `gfx1201` as claimed, not on the gfx942 producer, and the
recorded values match the logs exactly.
