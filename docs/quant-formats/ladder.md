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
rung with a route and the fusion inverts per architecture.

| product | roles above base width | Qwen3.8-27B bpw | current label |
|---|---|---:|---|
| `mq<N>-xtx` | none — truly uniform | ~4.26 (projected) | none today |
| `mq<N>-xt` | embeddings | 4.456 | `.mq4r` |
| `mq<N>` | embeddings + lm_head | 4.657 | `.mq4` |
| `mq<N> pro` | + recurrent/SSM state | 4.897 | `ctl2`, `head_ssm` |

`xt` and `xtx` both point the **same** direction — faster, smaller, fewer
guarantees — which is how AMD uses them. `xt` drops the lm_head to base width,
which also makes its GEMV tape-eligible and so carries the retained-Redline
route; that is exactly what `r` denotes today. `xtx` additionally drops the
embedding table, leaving every weight matrix at base width. **`xtx` carries no
quality floor**: it publishes measured KLD like every rung, but asserts no bound.

`pro` takes the precision direction. It deliberately does not reuse `xtx`, which
would promise speed while delivering the opposite.

The structural F16 tensors — norms, `A_log`, `dt_bias`, AWQ scale vectors — are
never quantized at any rung. On Qwen3.8-27B they total 679,424 elements, about
0.0025% of parameters, and are not a ladder axis.

### Why there is no rung above `pro`

A product that lifts attention costs 5.162 bpw on this model, so it is an
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
