# The 6-bit selector is not a cheap MQ4-Lloyd — at 136 B it dominates MQ4-Lloyd's 160 B on both axes

- **Date:** 2026-08-17
- **Lifecycle:** `historical`
- **Fixture:** 278,528 real post-FWHT 256-blocks, `layers.20.mlp.down_proj` of the
  `Qwen3.8-27B` bf16 parent, engine FWHT (seeds 42 / 1042). Tail threshold =
  99th pct of |w| = 2.869166e-02.
- **Harness:** [`tools/quant-design/sweep_selector_vs_per_block_lloyd.hip`](../../tools/quant-design/sweep_selector_vs_per_block_lloyd.hip)
- **Disposition:** design study, codec-side only. **No KLD claim.**

## The challenge

Reasonable objection to the qt=43 selector: it is MQ4-Lloyd (qt=30) re-derived.
MQ4-Lloyd already stores a per-block 16-entry fp16 codebook — 32 B/group, 160 B
total, 5.0 bpw — and `4a018afdf` already ruled that a per-block codebook "costs 8×
a per-channel scale for 1.8%".

## Result

| codebook | norm | B | bpw | overall MSE | tail-1% MSE |
|---|---|---|---|---|---|
| single global (GL, qt=40 shape) | RMS | 130 | 4.0625 | 1.1441e-06 | 1.3853e-05 |
| per-block Lloyd 16×fp16 (MQ4-Lloyd, qt=30) | RMS | 160 | 5.0000 | 8.9010e-07 | 5.7709e-06 |
| single global | MAX | 130 | 4.0625 | 1.3439e-06 | 1.1287e-05 |
| 64-profile selector, 6b | MAX | 132 | 4.1250 | 1.0338e-06 | 4.8843e-06 |
| per-block Lloyd 16×fp16 | MAX | 160 | 5.0000 | 9.1342e-07 | 5.3254e-06 |
| **64-profile sel, 2×128 sub-block** | **MAX** | **136** | **4.2500** | **8.9482e-07** | **2.5662e-06** |
| per-block Lloyd, 2×128 sub-block | MAX | 192 | 6.0000 | 7.8531e-07 | 3.3831e-06 |

**At 136 B the selector beats MQ4-Lloyd's 160 B on both axes** — 2.0% better
overall MSE and **2.08× better tail** — using **15% fewer bytes**.

Even at 132 B the 6-bit selector captures most of what a full 32-byte per-block
codebook delivers: 13.2% behind on overall MSE while **8.3% ahead on the tail**,
for 28 fewer bytes.

## Why the cheap version wins the tail

Structural, and it is the same mechanism as the original GL clipping failure,
inverted. Every profile in the family **pins its outermost level at ±1.0**, which
under max-normalisation is exactly the block max — so the dominant coefficient is
representable by construction. Free per-block Lloyd fits **conditional means**,
which drift inward from the extremes, so it systematically under-reaches the block
max. MQ4-Lloyd spends 32 B buying flexibility that it then uses to get the tail
wrong.

MQ4-Lloyd asks "what 16 levels best fit this block?" and pays 32 B for an
unconstrained answer. The selector asks "which of 64 pre-fitted shapes, all
anchored at the true max, fits this block?" and pays 12 bits. **The constraint is
the feature**, not a compromise.

## A hypothesis of mine that this falsified

I predicted per-block codebooks would be near-worthless under RMS normalisation —
the FWHT plus CLT make every block the same Gaussian shape, so there should be
nothing to adapt to, which would have explained `4a018afdf`'s 1.8%. **Wrong.**
Under RMS, per-block Lloyd buys 22.2% overall MSE and 2.4× tail over the single
global codebook. Per-block adaptation is worth real money in both spaces.

So the gap versus `4a018afdf`'s 1.8% is NOT explained by normalisation. That study
measured NRMSE at **2-bit** on **A3B** tensors; this one measures MSE at **4-bit**
on a dense Qwen3.8 tensor. Bit width, metric, and fixture all differ, and this
sweep does not isolate which is responsible. Recorded as an open discrepancy rather
than resolved.

## Method note — a units bug this harness first reproduced

The initial run reported the RMS arms ~14× worse than the MAX arms (1.85e-05 vs
1.34e-06). That was not a result: `GL_CB4S64` lives in **max-normalised** space with
outermost level ±1.0, and applying it to RMS-normalised data (which reaches ±3–5)
clips every block at 1.0. The fix is a space-matched codebook — `RMS_CB4` = GL_CB4,
the Lloyd-Max optimum for a unit Gaussian — for the RMS arms, and seeding per-block
Lloyd from whichever codebook matches its space. **A codebook is only meaningful
paired with the normalisation it was fitted in**; the harness now carries both and
selects by `norm_mode`.
