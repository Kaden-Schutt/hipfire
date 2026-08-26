# The header encoding *is* the scalar 4-bit design space — and per-16 with cheap scales wins

- **Date:** 2026-08-17
- **Lifecycle:** `historical`
- **Fixture:** 278,528 real post-FWHT 256-blocks, Qwen3.8-27B bf16 parent, layers 0 / 20 / 40,
  engine sign seeds 42/1042. Tail threshold = 99th pct \|w\| = 2.869166e-02.
- **Harness:** [`tools/quant-design/sweep_header_encoding_ladder.hip`](../../tools/quant-design/sweep_header_encoding_ladder.hip)
- **Disposition:** design study, codec-side, all variants **data-free**. No KLD. Rank on
  tail-1%, which is the only metric with predictive support here.

## Why this is the whole space

At 4 bits the payload is locked at 128 B per 256 weights. Non-uniform level placement
(codebooks) is retired — it wins MSE and loses the tail, monotonically. Asymmetric min/max
fitting is required. That leaves exactly one design question: **how cheaply can you encode
per-sub-block `(scale, zero)` pairs?** The format *is* its header encoding.

## The ladder

| variant | B | bpw | overall MSE | tail-1% MSE |
|---|---|---|---|---|
| v1 asym 1×256, f32 hdr *(today)* | 136 | 4.2500 | 1.4415e-06 | 9.4635e-07 |
| **v1.5** asym 1×256, fp16 hdr | **132** | 4.1250 | 1.4415e-06 | 9.4643e-07 |
| **v2** asym 2×128, fp16 hdr | **136** | 4.2500 | 1.2089e-06 | 5.6621e-07 |
| asym 4×64, fp16 hdr | 144 | 4.5000 | 9.7617e-07 | 3.2450e-07 |
| **hier 8×32, 6-bit s+z** *(= Q4_K structure)* | **144** | 4.5000 | **7.4647e-07** | **2.1632e-07** |
| hier 8×32, 8-bit s+z | 148 | 4.6250 | 7.4363e-07 | 1.8300e-07 |
| hier 16×16, 4-bit s+z | 148 | 4.6250 | 6.0015e-07 | 6.8051e-07 |
| **hier 16×16, 6-bit s+z** | **156** | 4.8750 | **5.2008e-07** | **1.3329e-07** |
| asym 8×32, fp16 hdr | 160 | 5.0000 | 7.4343e-07 | 1.8091e-07 |

Two headline results:

1. **Hierarchical per-32 at 144 B matches exact-fp16 per-32 at 160 B** — 16 bytes cheaper for
   essentially equal quality. At equal 144 B it strictly dominates fp16 per-64 by 23.5% MSE and
   33% tail. Above 136 B, the header *encoding* is the binding choice, not the granularity.
2. **Per-16 with cheap scales beats per-32 with exact scales.** `hier 16×16 6b` at **156 B**
   beats `asym 8×32 fp16` at 160 B by **30% MSE and 26% tail using 4 fewer bytes**. The lesson
   that has held all campaign: **spend bytes on granularity, not on precision.**

Q4_K is per-32. Going per-16 at 4 bits is where the FWHT rotation actually pays — Gaussianised
blocks make fine granularity productive instead of noisy.

## Two hypotheses tested and falsified

### "Rotation homogenises scales, so fewer than Q4_K's 6 bits should suffice" — NO

| sub-scale bits | B | overall MSE | tail-1% MSE |
|---|---|---|---|
| 3 | 138 | 9.2062e-07 | 1.9911e-06 |
| 4 | 140 | 7.8869e-07 | 7.2476e-07 |
| 5 | 142 | 7.5593e-07 | 3.2888e-07 |
| 6 | 144 | 7.4647e-07 | 2.1632e-07 |
| 8 | 148 | 7.4363e-07 | 1.8300e-07 |

**MSE does saturate at 6 bits** (6→8 buys 0.4%), consistent with rotation homogenising the
scales — so the hypothesis was half right. But the **tail keeps improving strongly**: dropping
to 4 bits costs **3.35×** on tail. Q4_K's 6 bits is well-calibrated, on the axis that matters.

### "Round the sub-scale so the range only grows, and the extremes stop clipping" — NO

Ceil the step, floor the zero, so the reconstructed interval always covers `[lo, hi]`:

| bits | Q4_K round-nearest | range-growing (ceil) |
|---|---|---|
| 4 | 7.8869e-07 / 7.2476e-07 | 8.6637e-07 / 1.4585e-06 |
| 5 | 7.5593e-07 / 3.2888e-07 | 8.0501e-07 / 8.3692e-07 |
| 6 | 7.4647e-07 / 2.1632e-07 | 7.6702e-07 / 3.4432e-07 |

Worse on **both** axes, ~1.6× on tail at 6 bits. The premise was wrong: ceiling guarantees no
clipping but **does not make the max exact** — it lands strictly inside a now-wider interval,
so the grid coarsens everywhere and loses more than the clip cost.

**Structural conclusion:** exact extremes require an *exact* scale, since only then does
`15·step` equal `hi − lo` precisely. **Extreme-exactness is fundamentally incompatible with
quantized sub-scales**, rounded in any direction. Hierarchical headers buy bytes and must give
up exactness; the trade is favourable anyway because granularity outweighs it.

## Method note — three harness bugs in one file today

This sweep file produced three wrong-but-plausible results before this run: raw-vs-normalised
units (inflating MSE ~1e5×), a max-space codebook applied to RMS-space data (clipping
everything at 1.0), and an outer guard admitting only 2 of 4 codec modes so new arms silently
reported a fall-through path. **Two of the three reached committed conclusions before being
caught**, including a retracted "12× zero-point" claim.

Any arm added here must carry a self-check that it executed its intended path. The tells that
caught the last one were: two different bit-widths returning bit-identical numbers, and a new
arm exactly equalling an unrelated one.

## Provenance

Hierarchical sub-scale quantization — per-sub-block scale and min quantized against a
per-super-block fp16 `d`/`dmin` — is **GGML's k-quant design (Q4_K)**, not original here. What
is different in these measurements is that it sits on FWHT-rotated weights and is evaluated at
per-16 granularity. Per this repository's attribution norms
([`AGENTS.md`](../../AGENTS.md), [`PRIOR-ART.md`](../../PRIOR-ART.md)), any format adopting this
structure should name and cite it rather than absorb it silently — hence the proposed `mq4_k`
naming.
