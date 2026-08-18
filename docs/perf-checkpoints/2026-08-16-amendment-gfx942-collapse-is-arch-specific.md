# Amendment — the qwen3.8 MQ4 "collapse" is a gfx942 execution bug, not a quantization defect

**Lifecycle:** `historical`. Amendment record. The originals are **unchanged**:

- [`2026-08-16-qwen38-27b-mq4-awq-alpha-kld.md`](2026-08-16-qwen38-27b-mq4-awq-alpha-kld.md)
- [`2026-08-16-qwen38-27b-mq4-lever-exhaustion-and-gfx942-blockers.md`](2026-08-16-qwen38-27b-mq4-lever-exhaustion-and-gfx942-blockers.md)
- [`2026-08-16-amendment-qwen38-mq4r-variant-label.md`](2026-08-16-amendment-qwen38-mq4r-variant-label.md)

**What this corrects:** the alpha record's § "Uncalibrated MQ4 collapse" presents a
degenerate uncalibrated baseline (KLD 12.92, PPL 2.6e6, two-token attractor) as a
property of the artifact, flagging the gfx1100-vs-gfx942 discrepancy as open. It
is now closed, and the answer inverts the finding: **the artifact is healthy; the
gfx942 execution path is broken.**

The consequence is larger than a footnote. Every improvement figure in the
originals that is expressed *relative to the uncalibrated baseline* measures a
CDNA3 bug rather than calibration quality, and must not be cited.

---

## The measurement

Byte-identical files, same references, same 24-chunk / 24,552-token method. The
`.mq4r` scored here is sha256 `61072980798ac1d3325020a63171d1a9cf99103eaa5bb1675a37845ea7d7762e`,
verified identical on both hosts, and is the shipped registry `qwen3.8:27b-fast`.

| artifact | ref | gfx942 (MI300X) KLD / PPL | gfx1201 (RDNA4, 4x R9700) KLD / PPL |
|---|---|---|---|
| `qwen3.8-27b.mq4` (lm_head Q8F16) | WT2 | 12.756767 / 2,232,384 | **0.058256 / 6.5103** |
| `qwen3.8-27b.mq4` | AG | 12.724525 / 1,861,388 | — |
| `qwen3.8-27b.mq4r` (lm_head MQ4G256) | WT2 | 12.921389 / 2,632,098 | **0.066668 / 6.5690** |
| *oracle (BF16 teacher)* | WT2 | — | *6.2385* |

`ln(248320) = 12.42`, so the gfx942 NLLs (14.4-14.8) are at or past uniform over
the vocabulary. On RDNA4 the same bytes land within 5 % of the oracle perplexity.

## Two hypotheses this kills

1. **"4-bit lm_head causes the collapse."** `.mq4` and `.mq4r` differ by exactly
   one tensor class — `lm_head` Q8F16 vs MQ4G256, a 675,430,400 B delta, every
   other class byte-identical. **Both collapse on gfx942** (12.757 vs 12.921).
   The lm_head is not the trigger.
2. **"Uncalibrated MQ4 is unusable on qwen3.8."** It is fine on RDNA4. The
   quantization is sound.

## What this invalidates in the originals

- The alpha record's baseline row (KLD 12.92 / PPL 2.6e6) is a gfx942 artifact.
  It remains true *as measured on that host* but must not be read as the quality
  of the uncalibrated model.
- Any "AWQ improves KLD by N%" framed against that baseline is measuring the
  arch bug. On gfx942 the AWQ arms scored **healthy** (0.063-0.136) while the
  shipped uncalibrated models scored 12.9 — i.e. **AWQ masks the gfx942 bug**.
  That is itself a clue to the root cause, but it means the gfx942 delta is not
  a calibration-quality measurement.
- The same caution applies to the glimmer figures reported in session
  (base 1.235188 -> awq 0.047959 on gfx942, a "96 % reduction"). Glimmer degrades
  rather than collapses on gfx942, so its baseline is less obviously wrong, but
  it is still an unvalidated host for that comparison.

## What survives

- The **alpha ranking** is internally consistent: all 12 arms scored sanely and
  monotonically on gfx942, and `alpha=0.05` beat `alpha=0.55` on *both*
  references. Whether that ranking reproduces on RDNA4 is being re-measured; it
  is not assumed here.
- The **calibration artifacts** are unaffected. They were produced from a BF16
  teacher whose own oracle perplexity is plausible (qwen 6.2385, glimmer 6.3783)
  and verified tensor-by-tensor from artifact bytes.
- The **token-count null result** (N/K 15.1 -> 22.6 changes KLD < 1 %) is an
  arm-vs-arm comparison on one host and does not depend on the baseline.

## New result, measured on the working arch

The `.mq4` vs `.mq4r` tradeoff, gfx1201, WT2:

| variant | lm_head | KLD | PPL | bytes |
|---|---|---|---|---|
| `.mq4` | Q8F16 | 0.058256 | 6.5103 | 15,655,791,616 |
| `.mq4r` | MQ4G256 | 0.066668 | 6.5690 | 14,980,361,216 |

**+14.4 % KLD to save 675,430,400 B (4.3 %).** That is the number the
`:fast` variant decision should be made on, and it is only measurable on an arch
where both variants run correctly.

## Open — the gfx942 root cause

Not diagnosed. What is known:
- Affects both qwen3.8 quantized variants identically; glimmer only degrades
  (KLD 1.235 gfx942), so it is not a blanket MQ4G256 failure.
- Independent of lm_head precision.
- **AWQ-scaled arms do not exhibit it**, which points at dequant range/scaling in
  the gfx942 MQ4 path rather than at layout or addressing.
- Prior evidence that the same bytes generate coherently on gfx1100/gfx1201 is in
  [`2026-08-14-qwen38-27b-qwen36-parity.md`](2026-08-14-qwen38-27b-qwen36-parity.md).

## Method note — why this was not caught sooner

Every number in the originals was measured on the MI300X because that is where
the calibration ran. The MI300X is demonstrably the *worst* host for these
formats: no dense Lloyd kernels at all, a broken gemma4 forward (lossless teacher
at PPL 1392), and now this. Quantization quality should be measured on an arch
with correct kernel coverage for the format under test; the calibration host and
the evaluation host do not have to be the same machine, and the KLD references
are 50.7 MB and fully portable.
