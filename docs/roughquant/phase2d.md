# RoughQuant — Phase 2c/2d: foldable variants (permutation + channel-consistent)

**VERDICT: confirms NOT deployable on Qwen3.5-0.8B.** Following the "think in
channels" steer, two free-folding variants were tested to escape the de-risk-B
trap (dense per-weight rotation wins but doesn't fold). Both help vs no-protection
QTIP-3, the write-side lever is real, but **every foldable variant is strictly
dominated by mq4.** The decorrelation that beats mq4 lives only in the unfoldable
per-weight rotation.

## Why these variants

De-risk B showed: per-weight dense PCA rotation wins (28.28 < mq4 29.08) but
doesn't fold; the one foldable shared rotation loses (30.68). Permutations and
per-channel bit-allocation, unlike rotations, **fold for free** (no runtime
transform — they reindex or just store some channels at higher precision). So
they're the right place to look for a *deployable* win.

- **2c — roughquant3 (permutation, read-side):** reorder each weight's input
  columns by diag(H) so salient channels lead, protect them, QTIP the bulk,
  un-permute. Read-side only.
- **2d — roughquant4 (channel-consistent, read+write):** the steer's variant.
  Rank residual channels by aggregated energy once; keep the top set exact in
  the **columns** of every residual reader (k=1024) AND the **rows** of every
  residual writer (o_proj/out_proj/down_proj, m=1024) — so a high-energy channel
  is exact where it's written and where it's read. Non-residual inputs
  (o_proj/down_proj internal activations) use per-weight diag(H) column
  protection. No rotation, no permutation.

## Results (Qwen3.5-0.8B, wikitext2 slice, Q8 embed, vs mq4 29.08)

| variant | foldable | protect | avg-bits | PPL |
|---|---|---|---|---|
| per-weight rotation (de-risk A) | ✗ | 3% | ~3.5 | **28.28** |
| shared rotation (de-risk B) | ✓ | 3% | ~3.5 | 30.68 |
| **2c** permutation read-only | ✓ | 3% | ~3.3 | 30.56 |
| **2c** permutation read-only | ✓ | 6% | ~3.5 | 30.62 |
| **2d** channel-consistent r+w | ✓ | 3% | ~3.4 | 30.68 |
| **2d** channel-consistent r+w | ✓ | 6% | ~3.9 | 29.67 |
| **2d** channel-consistent r+w | ✓ | 12% | ~4.67 | 29.41 |
| mq4 | ✓ | — | 4.25 | 29.08 |

## Reading

1. **The write-side lever is real (steer validated).** channel-consistent r+w at
   6% (29.67) beats read-only permutation at 6% (30.62) by ~1 PPL. Keeping
   high-energy residual channels exact *where they are written* (writer rows),
   not just where read, measurably helps — energy flowing down high-res channels
   end-to-end is the right intuition.
2. **But no foldable variant beats mq4.** The best (2d f0.12) is 29.41 at ~4.67
   avg-bits — *more* bits than mq4 (4.25) AND worse PPL (29.08). mq4 strictly
   dominates the entire foldable frontier. At iso-bits (~4.25) the foldable
   scheme interpolates to ~29.5 > 29.08.
3. **Decorrelation is the missing ingredient and it doesn't fold.** All foldable
   schemes (share/permute/channel-protect) cluster at ~29.4–30.7 because none
   *mixes* channels. Only the per-weight dense rotation decorrelates, reaching
   28.28 — and that's exactly the part that can't fold (de-risk B). Protection
   and reordering recover some of the 3-bit→4-bit quality gap but not enough.

## Final conclusion (unchanged, now comprehensive)

RoughQuant is **not deployable on Qwen3.5-0.8B**. The design space is now swept:
energy-concentration helps quality, the channel-flow / write-side framing is
correct and measurable, but every *foldable* (zero-runtime-cost) realization is
dominated by existing mq4, and the only realization that beats mq4 needs
unfoldable per-weight rotations. No Phase 3.

**Only remaining avenue (speculative):** cross-model on a 7B/9B — bigger models
may (a) make 2-bit bulk viable and (b) admit a better single shared transform.
Needs a fresh Hessian (~1–3 h collect). Not auto-run; awaits an explicit reason
to expect big models to differ qualitatively.

## Artifacts

- Code: `main.rs` — `qtip_simquant_protected`, `qtip_simquant_masked`,
  `permute_cols`/`unpermute_cols`, and the `roughquant3-sim` / `roughquant4-sim`
  post-passes. Env: `HIPFIRE_RQ3_*`, `HIPFIRE_RQ4_*`.
- All `.hfq` transient (quantize→PPL→delete). Fixtures as Phase 0.
