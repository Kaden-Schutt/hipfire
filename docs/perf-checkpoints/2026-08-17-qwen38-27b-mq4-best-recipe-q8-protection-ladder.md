# Qwen3.8 27B MQ4 — best known recipe, and the Q8-protection ladder that produced it (gfx1201)

**Lifecycle:** `historical`. Fixture-bound measured evidence. It is **not** a
current default, an automatic baseline, a product claim, or an admission
decision.

**Disposition:** The best measured Qwen3.8-27B MQ4 arm is
**consumed off-the-shelf GGUF imatrix + AWQ `alpha=0.55` + Q8 protection on
`lm_head, embed, ssm_out, ssm_in`**, at **slice-mean KLD 0.030479** on
WikiText-2 and **18.614 GB**.

Two findings matter more than the winner itself:

1. **At byte-identical size, consumed off-the-shelf imatrix beats our native
   HFQM calibration by 1.98×** — KLD 0.043776 vs 0.086790, both at
   15,662,615,552 bytes with the same Q8 scope.
2. **Native calibration is worse than no calibration at all.** Uncalibrated
   scores 0.066668 while native-calibrated scores 0.086790 — 30% worse — *despite*
   the native arm additionally carrying Q8 head protection that the uncalibrated
   arm lacks. Native AWQ is not merely weaker than off-the-shelf; on this model
   it is actively harmful.

**Why this record exists:** these numbers previously lived only in
`/home/kaden/attn_results.txt` on hiptrx. They were produced during this
campaign, were not committed anywhere under `docs/`, and were consequently lost
track of mid-session. The ladder is the campaign's most decision-relevant result
and had no durable home.

---

## Fixture identity

| field | value |
|---|---|
| scoring host | hiptrx, `gfx1201` Radeon AI PRO R9700 |
| reference | `qwen3.8-27b.ref_wt2.bin`, md5 `8a21364051d844b97c122e2c895f56d8`, oracle PPL 6.2385 |
| protocol | 24 chunks, top-k 256, `n_ctx` 2048, prefill scoring, kv-mode q8, `HIPFIRE_NORMALIZE_PROMPT=0`, `HIPFIRE_GRAPH=0` |
| quantize binary | md5 prefix `c644fc7f9272` (prior arms `838dd87567b6`) |
| off-the-shelf imatrix | `Qwen3.8-27B-imatrix.gguf`, 13,642,688 B |
| run window | `2026-08-17T02:54:23Z` → `03:04:28Z` |
| driver | `run_attn.sh` on hiptrx |

---

## The ladder

All rows WikiText-2, gfx1201, 24 chunks, same reference.

| arm | KLD | size (B) | size | decode |
|---|---|---|---|---|
| uncalibrated | 0.066668 | — | 14.980 GB | — |
| barto imatrix, AWQ a55 | 0.052140 | 14,987,185,152 | 14.987 GB | — |
| barto a55 + Q8 head | **0.043776** | 15,662,615,552 | 15.663 GB | — |
| *native HFQM calib a55 + Q8 head* | *0.086790* | *15,662,615,552* | *15.663 GB* | — |
| barto a55 + Q8 head + `ssm_out` | 0.036746 | 16,464,182,272 | 16.464 GB | **33.10 tok/s** |
| barto a55 + Q8 head + `ssm_out` + `attn_full` | 0.033862 | 17,354,779,648 | 17.355 GB | — |
| **barto a55 + Q8 head + `ssm_out` + `ssm_in`** | **0.030479** | **18,613,828,608** | **18.614 GB** | **unmeasured** |

Arm md5 prefixes: `ctl2` (`+ssm_out`) `00f95cb58fbd`; `attn_full`
`57c035451ef7`; `ssm_in` `950a1216f874`.

### The size-matched pair — the decisive comparison

| | imatrix source | KLD | size (B) |
|---|---|---|---|
| `q38.ctl.mq4` | consumed off-the-shelf GGUF | **0.043776** | 15,662,615,552 |
| `qwen3.8-27b.mq4-v6-814d8fd.mq4` | native HFQM calibration | **0.086790** | 15,662,615,552 |

Identical byte size, identical Q8 scope, identical AWQ alpha (0.55), identical
format (MQ4G256). Only the imatrix source differs. **1.98× in favour of
off-the-shelf.**

### Marginal value of each Q8 protection class

| step | ΔKLD | Δsize | KLD per GB spent |
|---|---|---|---|
| uncalibrated → barto a55 | −0.014528 | +0.007 GB | effectively free |
| + Q8 head | −0.008364 | +0.676 GB | 0.0124 / GB |
| + `ssm_out` | −0.007030 | +0.802 GB | 0.0088 / GB |
| + `ssm_in` | −0.006267 | +2.150 GB | 0.0029 / GB |
| + `attn_full` (instead of `ssm_in`) | −0.002884 | +0.891 GB | 0.0032 / GB |

The imatrix swap is the only step that is nearly free. Every Q8 class after that
buys quality with size at a steadily worsening rate, and `ssm_in` is the most
expensive lever on the ladder — 2.15 GB for 0.0063 KLD.

---

## Winning recipe, exactly

```bash
IM=/home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf

HIPFIRE_Q8_CLASSES="lm_head,embed,ssm_out,ssm_in" hipfire-quantize \
  --input  <parent qwen3.8-27b> \
  --output q38.ssmin.mq4 \
  --format mq4 \
  --q8-router \
  --imatrix "$IM" \
  --awq-alpha 0.55
```

Note `--imatrix` implies AWQ; `--awq-alpha 0.55` is the shipped default and is
stated explicitly here so the arm is reproducible without relying on the default.
No `--hessian`, no `--ldlq`, no native `.calib.hfq` is involved in the winning
arm.

### dtype census per arm

```
q38.ctl2.mq4      attn(full)      {F16: 96, MQ4G256: 64}
                  linear_attn(dn) {F16: 336, Q8F16: 96, MQ4G256: 192}
                  lm_head         {Q8F16: 1}

q38.attnfull.mq4  attn(full)      {F16: 32, Q8F16: 64}
                  linear_attn(dn) {F16: 336, Q8F16: 96, MQ4G256: 192}
                  lm_head         {Q8F16: 1}

q38.ssmin.mq4     attn(full)      {F16: 96, MQ4G256: 64}
                  linear_attn(dn) {F16: 144, Q8F16: 288}
                  lm_head         {Q8F16: 1}
```

The winning arm leaves full attention at MQ4G256 and instead promotes the
DeltaNet/linear-attention input projections to Q8F16 — 288 Q8F16 tensors against
`ctl2`'s 96.

---

## Reproducibility control

`ctl2` was re-quantized and **reproduces the earlier `head_ssm` arm
byte-for-byte**, which is what licenses comparing its KLD 0.036746 against the
rest of the ladder. Byte-level reproduction is checked because a prior
calibration A/B in this campaign silently emitted byte-identical artifacts and
identical KLD (0.066668) — a no-op that the metrics do not reveal.

---

## Open gaps — do not present this record as complete

1. **The best arm's decode speed is UNMEASURED.** The only decode figure on the
   ladder, **33.10 tok/s, belongs to the 16.464 GB `+ssm_out` arm**, not to the
   18.614 GB winner. Decode on this path is bandwidth-bound, so the winner is
   necessarily *slower*: naive byte scaling gives
   `33.10 × 16.464 / 18.614 ≈ 29.3 tok/s`. **[INFERENCE]** — that number is
   arithmetic, not a measurement, and must not be quoted as one.

   This matters because the ladder's top is also its heaviest and slowest rung.
   The best-quality arm costs +2.15 GB and roughly −11% decode against the arm
   one step down, for 0.0063 KLD. Whether that trade is worth taking is a
   product decision that cannot be made until the winner's decode is measured.

2. **The off-the-shelf imatrix GGUF is not hashed here.** Only its size
   (13,642,688 B) is recorded. Hash it before treating this recipe as pinned.

3. **Deployment-distribution scoring is missing for the ladder.** Every KLD above
   is WikiText-2, i.e. prose. A same-day finding
   (`2026-08-16-qwen38-27b-v6-chat-template-calibration-2x2.md`, plus its two
   amendments) established that prose compresses arm margins ~3× relative for a
   chat model. The two most closely spaced rungs here — `attn_full` at 0.033862
   versus `ssm_in` at 0.030479, only 0.0034 apart — are exactly the kind of pair
   that ranking is unresolved for until re-scored on the v6 conversation
   selector.

4. **`ssm_out`/`ssm_in` naming.** These are the DeltaNet / linear-attention
   projections on Qwen3.8, not a Mamba-style SSM. The class names come from the
   Q8-protection class table, not from the architecture.

---

## Amendment — decode measured; open gap 1 closed

Measured on hiptrx `gfx1201` (34.2 GB VRAM, HIP 7.14, arch `qwen3_5`) via
`hipfire bench --runs 5`, `max_tokens` 128, batch 1, bench binary md5 prefix
`8cca7887e33c`. Each arm's on-disk size was asserted against the ladder before
benching (all `SIZE_OK`).

| arm | Q8 classes | size | KLD | decode tok/s | prefill tok/s | ttft ms |
|---|---|---|---|---|---|---|
| `ctl2` | head + `ssm_out` | 16.464 GB | 0.036746 | **33.20** | 396.3 | 60.6 |
| `attnfull` | head + `ssm_out` + `attn_full` | 17.355 GB | 0.033862 | **31.70** | 383.0 | 62.7 |
| `ssmin` | head + `ssm_out` + `ssm_in` | 18.614 GB | 0.030479 | **29.70** | 355.7 | 67.5 |

Per-run decode samples: `ctl2` [33.2 ×5]; `attnfull` [31.8, 31.8, 31.7, 31.7,
31.7]; `ssmin` [29.7 ×5].

**`ctl2` independently reproduces the previously recorded 33.10 tok/s at 33.20**,
which cross-validates the earlier figure and the harness.

### The [INFERENCE] estimate was sound

The original record predicted the winner at `33.10 × 16.464 / 18.614 ≈ 29.3`
tok/s by naive byte scaling. **Measured: 29.70**, i.e. +1.1% against a prediction
derived purely from size. Decode on this path is bandwidth-bound to within about
a percent, so on this model family Q8-class size can be used to predict decode
cost before paying to build an arm.

### The real trade at the top of the ladder

| step from `ctl2` | KLD | decode | size |
|---|---|---|---|
| → `attnfull` | **+7.8% better** | −4.5% | +5.4% |
| → `ssmin` (best KLD) | **+17.1% better** | −10.5% | +13.1% |

Quality gained per percent of decode surrendered: `attnfull` 1.73, `ssmin` 1.63.
`attnfull` is marginally the more efficient exchange; `ssmin` buys more absolute
quality and is the quality-first pick. The choice is a genuine product tradeoff,
not a dominance relation — **the best-KLD arm costs 10.5% of decode throughput
and 2.15 GB against the arm two rungs down.**

### Caveat on the spread

Every arm reports `stdev = 0.0` with samples identical to the reported median.
This campaign's standing rule is that a suspiciously tight spread is a warning,
not reassurance. Here it is benign and explained: `hipfire bench` rounds
`decode_tok_s` to 0.1, and AR decode at batch 1 on a bandwidth-bound path is
deterministic, so five runs land in the same 0.1 bucket. The `attnfull` samples
(31.8, 31.8, 31.7, 31.7, 31.7) show the quantization boundary being crossed,
confirming the values are measured rather than cached. This reasoning does **not**
transfer to spec-decode `tau`, where a tight spread remains a red flag.

Open gaps 2–4 from the original record (imatrix GGUF unhashed; prose-only
scoring for the ladder; `ssm_out`/`ssm_in` naming) remain open.

---

## Amendment 2 — the full 9-arm matrix, decode measured for every arm

The first decode amendment covered only 3 of the ladder's arms. This is all of
them. hiptrx `gfx1201`, `hipfire bench --runs 5`, `max_tokens` 128, batch 1,
bench binary md5 prefix `8cca7887e33c`; sizes asserted before benching. KLD/PPL
are per-arm from each arm's own `sc_<tag>.log`, not inferred from co-occurrence.

| arm | Q8 scope / imatrix | size GB | KLD | PPL | decode tok/s | prefill | ttft ms |
|---|---|---|---|---|---|---|---|
| `a05` | none, barto a05 | 14.987 | 0.059824 | 6.5483 | **36.00** | 413.4 | 58.0 |
| `a55` | none, barto a55 | 14.987 | 0.052140 | 6.4665 | **35.90** | 410.5 | 58.5 |
| `q8head` | head, **no imatrix** | 15.656 | 0.058256 | 6.5103 | **35.50** | 405.5 | 59.2 |
| `ctl` | head, barto a55 | 15.663 | 0.043776 | 6.4088 | **34.60** | 401.3 | 59.8 |
| `native_v6` | head+embed, **native HFQM** | 15.663 | 0.086790 | 6.6727 | **34.70** | 400.3 | 60.0 |
| `ssm_only` | ssm only | 15.789 | 0.045249 | 6.4368 | **34.40** | 399.9 | 60.0 |
| `ctl2` | head+`ssm_out` | 16.464 | 0.036746 | 6.3773 | **33.20** | 396.3 | 60.6 |
| `attnfull` | head+`ssm_out`+`attn_full` | 17.355 | 0.033862 | **6.3285** | **31.70** | 383.0 | 62.7 |
| `ssmin` | head+`ssm_out`+`ssm_in` | 18.614 | **0.030479** | 6.3682 | **29.70** | 355.7 | 67.5 |

Decode is monotonic in size across all nine arms, confirming the path is
bandwidth-bound. Full span: **36.00 → 29.70 tok/s (−17.5%) buys 1.96× better
KLD**.

### 1. Same bytes, same speed, half the divergence

| | size | decode | KLD |
|---|---|---|---|
| `ctl` (off-the-shelf imatrix) | 15,662,615,552 | 34.60 | **0.043776** |
| `native_v6` (native HFQM calib) | 15,662,615,552 | 34.70 | 0.086790 |

Byte-identical size, decode within **0.29%**, KLD **1.98× better** for
off-the-shelf. The imatrix source is a pure quality lever with no size or speed
consequence whatsoever — which is the cleanest possible form of this result.

### 2. The imatrix is strictly better than buying Q8 head protection

| | size | decode | KLD |
|---|---|---|---|
| `a55` (imatrix, no Q8 head) | 14.987 GB | 35.90 | 0.052140 |
| `q8head` (Q8 head, no imatrix) | 15.656 GB | 35.50 | 0.058256 |

The imatrix delivers **10.5% better KLD while being 0.669 GB smaller and 0.40
tok/s faster**. Consuming an off-the-shelf imatrix is not a tradeoff at all; it
dominates a Q8-protection class that costs 0.67 GB.

### 3. `head` protection dominates `ssm only`

`ctl` beats `ssm_only` on KLD (0.043776 vs 0.045249), size (15.663 vs 15.789 GB)
**and** decode (34.60 vs 34.40). If only one class is affordable, protect the
head.

### Pareto frontier (minimise KLD, maximise decode)

`a05` → `a55` → `ctl` → `ctl2` → `attnfull` → `ssmin`. Six of nine arms are
non-dominated; the ladder is a genuine frontier and the choice of rung is a
product decision.

### Dominated arms

| arm | dominated by |
|---|---|
| `ssm_only` | `ctl` |
| `q8head` | `a55` |
| **`native_v6`** | **`a55`, `q8head`, `a05`** |

The native-calibration arm is dominated by **three** arms, including `a05` —
which is simultaneously **smaller (14.987 vs 15.663 GB), faster (36.00 vs 34.70
tok/s) and better (KLD 0.059824 vs 0.086790)**. Native calibration is beaten on
all three axes by the *worst* off-the-shelf arm measured. That is the strongest
form of the earlier "worse than no calibration" finding and it closes the
question.

### Metric disagreement at the top of the ladder

`ssmin` has the best KLD (0.030479) but `attnfull` has the best PPL (**6.3285**
vs 6.3682) *and* the best NLL (1.845059 vs 1.851313). **The two metrics disagree
about which arm is best**, and they disagree between adjacent rungs whose KLD gap
is only 0.0034.

Both were scored on WikiText-2, i.e. prose. Per
`2026-08-16-qwen38-27b-v6-chat-template-calibration-2x2.md` and its amendments,
prose compresses chat-model arm margins ~3× relative and is the metric pair that
inverted there too. **The `attnfull`-vs-`ssmin` ranking is therefore unresolved**
and needs the v6 conversation selector before either is called "best". Open gap 3
of this record is the blocker for the top of its own ladder.
