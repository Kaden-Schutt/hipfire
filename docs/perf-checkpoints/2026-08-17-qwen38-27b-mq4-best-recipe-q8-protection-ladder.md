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
