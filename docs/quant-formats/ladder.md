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
| product name | `RX4-XT`, `RX4`, `RX4 PRO` | this file | rarely |
| format name | `MQ4G256V2`, `MQ4CG256` | `qt-register.txt` | per codec |
| wire id | `qt=44`, `qt=45` | `qt-register.txt` | **never** |

`MQ` (Magnum, i.e. FWHT-rotated) stays the format-family name forever. It is no
longer what a user types. Codec generations (`v2`, `C`, `planar`, `pad`) and
calibration recipes (`Lloyd`, `GL`, `sel`, AWQ alpha) are format/encoder detail
and MUST NOT appear in a product name — they are how you buy a rung more
cheaply, not which rung it is.

---

## 1 · The honesty invariant

> **`RX<N>` means measured bpw ∈ [N, N+1). No exceptions.**

Measured means computed from the artifact — total file bytes × 8 ÷ parameter
count — not from the nominal width of the dominant codec. A product that would
land at 5.0 bpw or above is not a 4-bit product with an asterisk; it is `RX5`.

This is the one rule that must be machine-checked rather than promised, because
it is the rule the rest of the industry breaks. Enforcement is the same
both-directions shape `scripts/check-quant-registry.py` already applies to the
qt register: reject a publish whose product name's leading digit disagrees with
the artifact's measured bpw.

Measured 2026-08-20 on Qwen3.8-27B (26,895,998,464 parameters), showing the
failure mode this rule exists to prevent — all three of these are named `.mq4`:

| artifact | bpw | |
|---|---:|---|
| `q38.attnfull.mq4` | 5.162 | over |
| `q38.ssmin.mq4` | 5.537 | over |
| `q38.v2-attn.mq4` | 5.801 | over |

They are legitimate internal experiments. None may ship under a 4-bit name.

---

## 2 · Rungs

A rung names which **tensor roles** are held above the base width. Roles are
architecture-neutral; each arch maps its own tensors into them. This is what
makes a label portable — today `mq2r` on DeepSeek-V4 means lm_head *at* Q8
while `mq4r` on Qwen3.8 means lm_head *not* at Q8, because the suffix fused a
rung with a route and the fusion inverts per architecture.

| product | roles above base width | Qwen3.8-27B bpw | current label |
|---|---|---:|---|
| `RX<N>-XT` | none | 4.456 | `.mq4r` |
| `RX<N>` | lm_head | 4.657 | `.mq4` |
| `RX<N> PRO` | lm_head + recurrent/SSM state | 4.897 | `ctl2`, `head_ssm` |

`XT` denotes the **faster** variant, matching its GPU meaning: fewer bits, less
memory traffic, and — because the lm_head GEMV becomes tape-eligible — the
retained-Redline route. That is exactly what `r` already means today, so the
suffix stops carrying a hidden quality claim. `PRO` takes the precision
direction, borrowing AMD's own workstation idiom rather than `XTX`, which would
promise speed while delivering the opposite.

Rung availability is bounded by § 1. On a model where lifting attention would
cross N+1, that rung does not exist at N bits.

### Evidence for the rung boundaries

Same parameter count, so bpw *is* the ladder. Measured on Qwen3.8-27B:

- `qwen3.8-27b.mq4r` — `lm_head.weight` at `qt=13` (MQ4G256), 4.456 bpw
- `qwen3.8-27b.mq4` — `lm_head.weight` at `qt=3` (Q8F16), 4.657 bpw
- `q38.mq4v2.mq4` — `lm_head.weight` at `qt=3`, 4.659 bpw

The lm_head is 248320 × 5120 = 1,271,398,400 weights; moving it between Q8 and
4-bit is the entire 0.20 bpw step between `RX4-XT` and `RX4`.

### Why codec is not a rung

At a fixed rung the codec moves bpw by roughly ±0.11 and the calibration recipe
by nothing at all:

| artifact | bpw | note |
|---|---:|---|
| `q38.ctl.mq4` | 4.659 | rung `RX4`, v1 codec |
| `q38.ctl.mq4c` | 4.546 | same rung, MQ4C codec |
| `q38.a05.mq4` | 4.458 | AWQ alpha 0.05 |
| `q38.a55.mq4` | 4.458 | AWQ alpha 0.55 — identical bytes |

A better codec buys the same promise for fewer bytes. A better calibration buys
quality for free. Neither changes what the product is.

---

## 3 · Required columns

Every published product row carries measured bpw **and** measured KLD against
the pinned f16 teacher on a committed fixture. KLD is the column hipfire can
publish and the ggml ladder structurally cannot; it is the reason this table is
worth maintaining rather than a letter soup.

Rows are filled from the routes in [`../VALIDATION.md`](../VALIDATION.md).
A rung without a measured KLD on a given model is not publishable for that
model. Do not infer a KLD from a sibling model or a sibling codec.

---

## 4 · Migration

- **File extensions do not move.** `mq4r_redline_default` keys on the `.mq4r`
  extension, and the golden fixtures are SHA-256 pinned. Renaming artifacts
  would break sealed evidence. The product name lives in the registry card.
- **`qt` numbers never move.** Existing files keep loading forever.
- **Withdraw codec generations from user-facing names** (`mq4v2`, `mq4c`).
  They are days old and cost nothing to retire; the qt rows stay permanently.
- **New quantizer output** is named by rung, and the encoder is free to improve
  underneath without renaming the product.
