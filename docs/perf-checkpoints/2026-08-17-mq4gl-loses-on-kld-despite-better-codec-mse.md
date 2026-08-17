# MQ4-G256-GL — 22.69% better codec MSE, 0.57 GB smaller, and 11.28% WORSE KLD (gfx1201)

**Lifecycle:** `historical`. Fixture-bound measured evidence. It is **not** a
current default, an automatic baseline, a product claim, or an admission
decision.

**Disposition:** The controlled comparison between MQ4-G256-GL (qt=40, tensor-global
Lloyd codebook, 4.0625 bpw) and uniform-affine MQ4-G256 (4.25 bpw) has finally been
taken, and **GL loses the metric of record**:

| arm | GB | bpw | WT2 KLD | NLL | PPL |
|---|---|---|---|---|---|
| `mq4` uniform affine | 15.66 | 4.2500 | **0.043776** | 1.857666 | 6.4088 |
| `mq4gl` GL_CB4 | **15.09** | **4.0625** | 0.048713 | **1.844915** | **6.3276** |

Only `--format` differs — same parent, same off-the-shelf GGUF imatrix, same
`--awq-alpha 0.55`, same `HIPFIRE_Q8_CLASSES=lm_head,embed`, same scoring protocol.

- bytes: **−3.66%** (0.57 GB smaller) — **win**
- **KLD: +11.28% — LOSS**
- NLL: −0.69% — win
- PPL: −1.27% — win

A **22.69%** codec-MSE advantage produced an **11.28% KLD regression**. Codec MSE did
not convert. This is the **fourth** non-conversion in this campaign: the byte win did
not convert to time, `v_perm_b32` did not convert to time, the polynomial codebook
did not convert to quality, and now codec MSE does not convert to KLD.

**Source:** branch `quant/quality`, artifact `q38.mq4gl.mq4` 15.09 GB, scored on
hiptrx `gfx1201` with `eval_hipfire` at 24,552 tokens, 1553.8 s, 16 tok/s.

---

## The format could not be loaded until today

Before this measurement existed, qt=40 was unloadable. `dequant_weight_raw` fell
through to the `RAW_CODECS` passthrough table, found no row for 40, and errored:

```
load weights: HipError { message: "unsupported quant_type 40 for dequant_weight_raw" }
```

Its 2-bit and 3-bit siblings (qt 38 `MQ2G256GL`, qt 39 `MQ3G256GL`) were both
registered. The encoder worked, the GEMV dispatch worked, and
`DType::{row_stride, requires_k_mod_256}` both handled `MQ4G256GL`. **One missing
one-line registry row** made every MQ4-GL artifact ever produced unloadable.

Fixed in `cf061b7ed`. The consequence for the campaign is worth stating plainly:
**six kernel-optimization attempts, a layout/padding study, and a register-codebook
study were all spent making a format fast that could not be loaded, and none of them
touched the reason it was worse.**

---

## The likely mechanism: GL optimises the wrong objective

`GL_CB4` is Lloyd-fitted to minimise **unweighted** MSE on a unit Gaussian. KLD is
not unweighted — it is dominated by the weights that move logits.

The direction of the evidence is the tell:

| metric | what it weights | GL result |
|---|---|---|
| codec MSE (rotated domain) | all errors equally | **win, 22.69%** |
| NLL / PPL | absolute LM quality | **win, ~1%** |
| **KLD** | **match to the teacher's distribution** | **LOSS, 11.28%** |

GL wins every unweighted metric and loses the distribution-matching one. That is the
signature of a codebook that reconstructs the **bulk** well and the **tails** worse.
A Gaussian-optimal codebook concentrates levels where the density is — near zero —
while uniform affine spreads them evenly across the range and thereby protects
large-magnitude weights better. Large-magnitude weights are exactly the ones that
move logits.

This is the standard MSE-versus-task-loss gap, and it is the reason
importance-weighted methods (imatrix, AWQ) exist at all.

### The implied fix, which no prior GL work touched

Refit `GL_CB4` to minimise **activation-weighted** error using the imatrix that the
quantize path already consumes, instead of unweighted MSE on an idealised Gaussian.
The codebook is 16 `f32` values; refitting is cheap and needs no kernel, no layout
change, and no extra bytes. Until that is tried, GL's thesis is untested rather than
disproven — what is disproven is the *unweighted-MSE* version of it.

---

## What is NOT the problem

- **Not speed.** At matched row counts on gfx1201, `gemv_mq4g256gl` is 21.72 µs
  against HFQ4's 19.12 µs at R=2 — **+5.6%**, affordable. The famous "57% deficit"
  was a row-count mismatch artifact (GL at R=1 vs HFQ4 at R=2, because
  `gemv_rows_default()` returns 2 on RDNA4).
- **Not bytes.** GL delivers the predicted 0.57 GB saving on a 27B body, matching
  the 0.1875 bpw scale-header calculation exactly.
- **Not the scale fit.** The per-block scale was RMS, which is not MSE-optimal; that
  was fixed the same day (least-squares, `58748db1a`) and is worth 2.73% MSE on
  GL_CB4 — widening the codec-MSE margin over affine from 20.52% to 22.69%. **The
  artifact scored here predates that fix**, so a rebuild would carry a slightly
  better codec MSE. It would not plausibly close an 11.28% KLD gap, since the extra
  2.17 points of MSE margin is the same *unweighted* quantity that already failed to
  convert.
- **Not layout.** Row-level AoS moved the bandwidth ratio 0.843 → 0.840 (nil), and a
  4-byte-aligned 132 B group reached 0.866 for +1.5% bytes — real but below a ship
  bar. See the row-count-mismatch record's amendment.

---

## Deployment blocker, separate from quality

The scoring run measured **16 tok/s** against ~163 tok/s for arms with a fast path.
qt=40 has a decode GEMV (`K::GemvMq4G256GlPrerotated` →
`gemv_mq4g256gl_multirow`, honouring `gemv_rows_default()`) but **no batched
prefill/GEMM path**, so `--scoring-mode prefill` falls back to per-token. Even had
GL won on quality, it would not be shippable for prefill-heavy workloads without a
WMMA prefill family.

---

## Caveats

1. **One reference so far.** This is WikiText-2 only. The v6 conversation-selector
   run was still in flight when this record was written. Given prose has already
   inverted a PPL ranking in this campaign, the selector result matters — but an
   11.28% KLD loss is large enough that a reversal would be surprising.
2. **PPL is not usable as a fidelity metric on this fixture.** GL wins PPL and loses
   KLD; separately, several arms score PPL *below* the oracle (12.1735 vs 12.8813 on
   the selector), which is impossible for a faithful measure. Rank on KLD, treat PPL
   as a degeneracy tripwire only.
3. **The four sibling new formats remain unscored.** `mq1`, `mq2gl`, `mq3gl` and
   `mq3.5gl` were all built (artifacts and byte counts recorded) but have no decode
   kernels, so only encoder-side codec MSE is available for them. Given that codec
   MSE just failed to predict KLD for `mq4gl`, their MSE figures should **not** be
   read as quality predictions.
