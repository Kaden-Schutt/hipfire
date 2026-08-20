# hipfire quant ladder — the product tier table

This file is the SOURCE OF TRUTH for what a hipfire quant **product name**
promises. It is the product-facing sibling of [`qt-register.txt`](qt-register.txt),
which owns wire-format numbers.

Three namespaces, deliberately separate. Conflating them is what produced the
current zoo (`mq4`, `mq4r`, `mq4p`, `mq2lloyd`, `mq4v2`, `mq4c`, …), where a
codec generation, a calibration recipe, and a layer ladder all competed for the
same token.

| namespace | example | owner | changes |
|---|---|---|---|
| product name | `mq4-xt`, `mq4`, `mq4 pro` | this file | rarely |
| format name | `MQ4G256V2`, `MQ4CG256`, `MQ5G256` | `qt-register.txt` | per codec |
| wire id | `qt=44`, `qt=45`, `qt=31` | `qt-register.txt` | **never** |

`MQ` (Magnum, i.e. FWHT-rotated — see [`../../PRIOR-ART.md`](../../PRIOR-ART.md))
stays the family name in **both** the product and format columns. What leaves
the product name is everything that is not a tier: codec generations (`v2`, `C`,
`planar`, `pad`) and calibration recipes (`Lloyd`, `GL`, `sel`, AWQ alpha) are
how you buy a rung more cheaply, not which rung it is.

The product number is a **bpw class**, not a codec. `mq5` means "measured 5.x
bpw," whichever codec produced it. `MQ5G256` (qt=31) is a format that may or may
not be used to build an `mq5`-class product. Today it builds none.

---

## 1 · The honesty invariant

> **`mq<N>` means measured model bpw ∈ [N, N+1). No exceptions.**

**Model bpw, not codec bpw.** These differ and only one of them is the product's
promise:

- *Codec bpw* — what the encoding costs per weight it covers. MQ4G256 is
  136 B per 256 weights = **4.25 bpw**.
- *Model bpw* — total artifact bytes × 8 ÷ parameter count. Lands at
  **4.46–4.90** for the same codec, because embeddings and lm_head sit above
  base width and the structural F16 tensors add a sliver.

The product quotes model bpw, because that is what the user downloads and holds
in VRAM. Codec bpw is what you would get if the ladder were flat, which is what
the `xtx` rung approximates.

This is the one rule that must be machine-checked rather than promised, because
it is the rule the rest of the industry breaks. Enforcement is the same
both-directions shape `scripts/check-quant-registry.py` already applies to the
qt register: reject a publish whose product name's leading digit disagrees with
the artifact's measured model bpw.

Measured 2026-08-20 on Qwen3.8-27B (26,895,998,464 parameters), showing the
failure mode this rule exists to prevent — all three are named `.mq4`:

| artifact | model bpw | |
|---|---:|---|
| `q38.attnfull.mq4` | 5.162 | over |
| `q38.ssmin.mq4` | 5.537 | over |
| `q38.v2-attn.mq4` | 5.801 | over |

Legitimate internal experiments. None may ship under a 4-bit name.

---

## 2 · Rungs

A rung names which **tensor roles** are held above the base width. Roles are
architecture-neutral; each arch maps its own tensors into them. This is what
makes a label portable — today `mq2r` on DeepSeek-V4 means lm_head *at* Q8
while `mq4r` on Qwen3.8 means lm_head *not* at Q8, because the suffix fused a
| product | roles above base width | Qwen3.8-27B bpw | current label |
|---|---|---:|---|
| `mq<N>-xt` | embeddings | 4.456 | `.mq4r` |
| `mq<N>` | embeddings + lm_head | 4.657 | `.mq4` |
| `mq<N> pro` | + recurrent/SSM state | 4.897 | `ctl2`, `head_ssm` |

**Three rungs per bit width. No more, no fourth.** Every rung carries a claim,
which is why there is no "truly uniform, no guarantee" tier — a rung that
promises nothing is not a product, and the base-width-everywhere point is what
the codec bpw already describes.

`xt` denotes the **faster** variant, matching its GPU meaning: it drops the
lm_head to base width, which also makes its GEMV tape-eligible and so carries
the retained-Redline route. That is exactly what `r` denotes today. `pro` takes
the precision direction; it deliberately does not use `xtx`, which would promise
speed while delivering the opposite.

The structural F16 tensors — norms, `A_log`, `dt_bias`, AWQ scale vectors — are
never quantized at any rung. On Qwen3.8-27B they total 679,424 elements, about
0.0025% of parameters, and are not a ladder axis.

### Roles are per-architecture categories

A dense model's roles are embeddings, lm_head, and recurrent/SSM state. A MoE's
ladder is dominated by an **expert** role the dense model does not have — on
Qwen3.6-35B-A3B the experts are 93% of parameters — so its three rungs order
experts before embeddings. The rung names stay portable because they name a
position on the ladder, not a tensor list.

### The product grid

| | `-xt` | base | ` pro` |
|---|---|---|---|
| `mq2` | ✓ | ✓ | ✓ |
| `mq3` | ✓ | ✓ | ✓ |
| `mq4` | ✓ | ✓ | ✓ |
| `mq5` | ✓ | ✓ | ✓ |
| `mq6` | ✓ | ✓ | ✓ |
| `q8` | — | ✓ | — |

Fifteen products plus the ceiling, and every cell has a defined meaning for any
architecture. `q8` carries no rungs: lifting above Q8 means F16, which is a
different product, and uniform Q8 already *is* `q8`.

**Rungs lift into each other.** What a rung lifts *to* is drawn from the same
published set, one or two classes up: `mq2 pro` lifts to MQ3/MQ4, `mq4 pro` to
MQ6, `mq6 pro` to Q8. The ladder terminates at Q8 rather than at an unrelated
format, so no codec sits outside the ladder and the § 1 budget governs each step.

### Why there is no rung above `pro`

A product that lifts attention costs 5.162 bpw on the dense 27B, so it is an
`mq5`-class product. There is no `XL`/`XXL`.

The counter-argument is real and worth recording: an explicit oversize suffix
would make *lineage* visible — a 3-bit base grown fat is not the same animal as
a native 4-bit quant, even at equal bytes — and reclassifying it hides that.

It loses anyway, on two grounds:

1. **An oversize suffix breaks the only checkable promise.** `mq4 pro xl` at
   5.16 bpw means the leading digit no longer predicts size. That is precisely
   the llama.cpp failure mode, with a warning label attached — and the people
   most harmed by the zoo are the ones who do not decode suffixes. A class
   defined by measured bpw cannot drift: growth reclassifies.
2. **Lineage does not need to live in the name.** At equal class the KLD column
   picks the winner, and a lifted-3-bit only ships at all if it *beats* a native
   4-bit at equal bytes — otherwise you would ship the native one. § 3 makes
   base codec a required column so the two animals stay distinguishable.

---

## 3 · Required columns

Every published product row carries, per model:

| column | why |
|---|---|
| model bpw | the § 1 promise; machine-checked |
| KLD vs pinned f16 teacher | the axis hipfire can publish and ggml structurally cannot |
| base codec + qt | lineage — distinguishes a native N-bit from a lifted (N−1)-bit at equal class |
| rung | which roles are held above base width |

Rows are filled from the routes in [`../VALIDATION.md`](../VALIDATION.md). A rung
without a measured KLD on a given model is not publishable for that model. Do
not infer a KLD from a sibling model or a sibling codec.


### Worked example — Qwen3.6-35B-A3B as billed today

Nine product labels in `registry/v1.json`, 34,660,610,688 parameters. Measured:

| billed | bytes | model bpw | class | recodified |
|---|---:|---:|---:|---|
| `mq2` | 11.61 GB | 2.680 | 2 | `mq2` |
| `mq2r` | 12.33 GB | 2.845 | 2 | **needs adjudication** (see below) |
| `mq3p` | 17.24 GB | 3.980 | 3 | `mq3 pro` |
| `mq4r` | 18.70 GB | 4.316 | 4 | `mq4-xt` |
| `mq4p` | 19.76 GB | 4.561 | 4 | `mq4` |
| `mfp4` | 20.18 GB | 4.659 | 4 | `mq4`, codec MFP4G32 |
| `mq5` | 23.69 GB | 5.469 | 5 | `mq5` |
| `mq6` | 27.72 GB | 6.399 | 6 | `mq6` |

Every label already lands inside its own bpw class — the A3B ladder is honest on
§ 1 today. The defect is legibility: nine labels where `p`, `r`, and `mfp` are
opaque, and where `mq4p` and `mfp4` are the same rung differing only by codec.

**The `mq2r` adjudication.** Its ladder is five precisions deep:

| qt | format | elems | share |
|---|---|---:|---:|
| 19 | MQ2G256Lloyd | 21,474,836,480 | 62.0% |
| 20 | MQ3G256Lloyd | 10,737,418,240 | 31.0% |
| 13 | MQ4G256 | 1,938,636,800 | 5.6% |
| 3 | Q8F16 | 509,542,400 | 1.5% |
| 1 | F16 | 176,768 | ~0% |

A third of the experts are lifted to 3-bit Lloyd, and the artifact is 0.165 bpw
*heavier* than `mq2`. By content and by bytes that is a `pro`. The `r` suffix was
specified to mean the lighter, Redline-oriented variant. Artifact and spec
disagree about which direction the label points, and the file cannot tell us
which one drifted.

That is the case for this table. With the rung declared at build time and bpw
measured at publish, the disagreement is a build failure on the day it appears
rather than something reconstructed from expert-tensor counts months later.

### Hybrid lifts — what the roles are lifted *to*

A rung says which roles sit above base width. The precision they are lifted *to*
is an encoder decision and may legitimately differ per architecture. Q8F16 costs
about 8.5 bpw per weight against MQ4G256's 4.25, measured as the 0.201 bpw step
between `qwen3.8-27b.mq4` and `.mq4r`, which differ only in the lm_head.

Lifting to MQ6 or MQ5 instead of Q8 therefore buys a rung back, and the size of
that rebate depends entirely on how much of the model the lifted roles are:

| model | embed + lm_head share | Q8 → MQ6 rebate |
|---|---:|---:|
| Qwen3.8-27B dense | 2.54B / 26.9B = 9.5% | ~0.19 bpw |
| Qwen3.6-35B-A3B | 0.51B / 34.7B = 1.5% | ~0.04 bpw |

On the dense model that is nearly a whole rung — the `pro` state-lift costs
0.24 bpw — so an MQ6 lift can fund `pro` where only the base rung fit before. On
the MoE it is noise, because the experts dominate the budget. Same product name
either way, which is the reason codec is kept out of it.

---

## 4 · Orientation for people arriving from llama.cpp

Approximate neighbours by size. hipfire quotes measured model bpw, so these are
comparable on bytes; compare quality on the published KLD, not on the tier name.

| hipfire | model bpw | nearest ggml |
|---|---:|---|
| `mq4-xtx` | ~4.26 | IQ4_XS |
| `mq4-xt` | 4.456 | Q4_K_S |
| `mq4` | 4.657 | between Q4_K_S and Q4_K_M |
| `mq4 pro` | 4.897 | Q4_K_M |

---

## 5 · Migration

- **File extensions do not move.** `mq4r_redline_default` keys on the `.mq4r`
  extension, and the golden fixtures are SHA-256 pinned. Renaming artifacts
  would break sealed evidence. The product name lives in the registry card.
- **`qt` numbers never move.** Existing files keep loading forever.
- **Withdraw codec generations from user-facing names** (`mq4v2`, `mq4c`). They
  are days old and cost nothing to retire; the qt rows stay permanently.
- **New quantizer output** is named by rung, and the encoder is free to improve
  underneath without renaming the product.
