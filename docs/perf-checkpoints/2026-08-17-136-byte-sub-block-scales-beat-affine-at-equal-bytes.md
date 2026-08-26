# At affine's exact byte count, two sub-block scales beat it by 25% — and outlier refinement does not

- **Date:** 2026-08-17
- **Lifecycle:** `historical`
- **Fixture:** 278,528 real post-FWHT 256-blocks — first 4096 rows of
  `layers.20.mlp.down_proj` from the `Qwen3.8-27B` bf16 parent, engine FWHT
  (sign seeds 42 / 1042). Tail threshold = 99th percentile of |w| = 2.869166e-02.
- **Harness:** [`tools/quant-design/sweep_mq4_header_allocation.hip`](../../tools/quant-design/sweep_mq4_header_allocation.hip),
  gfx1201, 14 configurations in 2.9 s.
- **Disposition:** design study. Codec-side only. **No KLD claim** — every metric
  here is a proxy, and on this exact format family a 22.69% codec-MSE win already
  became an 11.28% KLD loss once.

## The question

MQ4-GL (qt=40, 130 B) lost KLD because its fixed outermost level clips the
dominant coefficient of 82.57% of blocks. qt=43 fixes that with max-normalisation
plus a 6-bit level-shape selector at 132 B. The open question: if we spend
affine's full **136 B**, what is the best possible allocation?

The payload is locked — 256 weights × 4 bits = 128 B, and 5 bits would be 160 B,
far over budget. So the entire lever is the 8-byte header, and it can buy exactly
three things: range (already exact via max-normalisation), level shape (already
64 profiles), and one of two ways to spend what is left.

## Result

| variant | hdr B | tot B | bpw | overall MSE | tail-1% MSE |
|---|---|---|---|---|---|
| affine qt=1 (f32 scale + f32 zero) | 8 | 136 | 4.2500 | 1.4415e-06 | 9.4635e-07 |
| qt=43: fp16 max + 6b sel, p=0 | 4 | 132 | 4.1250 | 1.0338e-06 | 4.8843e-06 |
| qt=43: fp16 max + 6b sel, p=2 | 4 | 132 | 4.1250 | 1.2831e-06 | 1.6103e-06 |
| A: p=2 + top-2 indexed refine @12b | 8 | 136 | 4.2500 | 1.2749e-06 | 9.8502e-07 |
| **B: 2×128 sub-block max+sel, p=2** | **6** | **136** | **4.2500** | **1.0805e-06** | **9.9410e-07** |
| B: 4×64 sub-block max+sel, p=2 | 12 | 144 | 4.5000 | 8.6563e-07 | 5.5926e-07 |

**Sub-block splitting dominates outlier refinement.** At identical 136 B and
identical tail (9.94e-07 vs affine's 9.46e-07 — parity), two 128-wide sub-scales
deliver **25.0% better overall MSE than affine**, and beat the refinement variant
by **15.2%**. It needs only a **6-byte header** — 2 × (fp16 max + 6-bit selector)
— leaving 2 bytes spare at 136 B.

Obvious in retrospect: refinement repairs 2–4 coefficients, while a second scale
improves range accuracy for all 256. Refinement is a point fix; granularity is a
distributed one.

At 144 B, 4×64 sub-blocks beat affine on **both** axes at once — 40.0% better
overall, 40.9% better tail — at 5.9% more bytes.

## Two negative results worth keeping

**Index-free refinement does not work.** Max-normalisation makes level 15 land
exactly on the block max, so the largest coefficient needs no refinement and the
tail error lives at levels 13–14. Those outer-two levels hold **13.47
coefficients per block** (p95 23, max 43), so slots filled in scan order miss the
actual largest ones. Measured: index-free reaches only 2.22× affine's tail versus
0.79× for indexed top-4. Indices are load-bearing, and they cost 8 bits each.

**Widening the outermost level is worse than not.** Pinning level 15 at 4.0 cuts
clipped blocks from 82.57% to 1.29% on paper, and measures worse on all three
proxies (overall 1.399e-06, tail 1.969e-05, max-coef 14.921%) — including the one
it targeted. Without adding levels it opens a coarse gap across 2.7–4.0, exactly
where block maxima live (max|z| p50 2.999). That variant is deleted, not shelved.

## Consequence for the format ladder

| bytes | bpw | allocation | overall MSE vs affine | tail vs affine |
|---|---|---|---|---|
| 132 | 4.1250 | fp16 max + 6b sel (qt=43, p=2) | 11.0% better | 1.70× worse |
| **136** | **4.2500** | **2 × (fp16 max + 6b sel)** | **25.0% better** | **parity** |
| 144 | 4.5000 | 4 × (fp16 max + 6b sel) | 40.0% better | 40.9% better |

One kernel consequence: with two sub-blocks the selector is **no longer
wave-uniform**. On a 32-thread workgroup lanes 0–15 read half 0 and lanes 16–31
half 1, so it becomes two scalar loads plus a `v_cndmask` rather than the single
SGPR fetch qt=43 enjoys. Still cheap, but the clean wave-uniform property is lost.
The 256-wide FWHT is unaffected — sub-blocking changes quantisation granularity
only, not the transform.

**Gate before any of this ships:** qt=43's KLD against both references. It is the
calibration point for whether tail-1% MSE predicts KLD at all. If qt=43 at p=2
(11.0% better overall, 1.70× worse tail) lands near affine, the proxy is sound and
the 136 B sub-block variant should beat affine outright. If qt=43 still loses
badly, the proxy is wrong again and this whole ladder is unfounded.
