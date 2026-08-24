# Amendment — the 2026-08-16 qwen3.8 MQ4 records measure `.mq4r`, not Q8-router-pinned `.mq4`

**Lifecycle:** `historical`. Amendment record. The two originals are **unchanged**
and remain the authoritative measurements:

- [`2026-08-16-qwen38-27b-mq4-awq-alpha-kld.md`](2026-08-16-qwen38-27b-mq4-awq-alpha-kld.md)
- [`2026-08-16-qwen38-27b-mq4-lever-exhaustion-and-gfx942-blockers.md`](2026-08-16-qwen38-27b-mq4-lever-exhaustion-and-gfx942-blockers.md)

**What this amends:** both records label the format `MQ4G256` and state the
group size, which is correct, but neither states **which tier layout** was
produced. They should be read as measuring the **`.mq4r` variant**. No measured
number changes; this is a labelling correction so the numbers are cited against
the right artifact shape.

---

## Measured tier layout

Read from the arm log of `qwen3.8-27b.awq_a0p05.mq4`
(`--format mq4 --hessian <calib> --awq-alpha 0.05`, no `--no-q8-router`):

| tensor class | emitted format |
|---|---|
| `model.language_model.embed_tokens.weight` [248320, 5120] | **`Q8_F16`** |
| `lm_head` / output | `MQ4G256` |
| attention `q_proj` / `k_proj` / `v_proj` / `o_proj` | `MQ4G256` (+ `AWQ` scale sidecar) |
| MLP `gate_proj` / `up_proj` / `down_proj` | `MQ4G256` (+ `AWQ` scale sidecar) |
| DeltaNet `in_proj_qkv` / `_z` / `_a` / `_b`, `out_proj` | `MQ4G256` (+ `AWQ` scale sidecar) |
| norms | `F16` |

Whole-file histogram: **497 `MQ4G256` · 496 `AWQ` · 305 `F16`**, plus the single
`Q8_F16` embedding. Output size 14,972.8 MB.

So only the **embedding** is Q8-pinned. Attention and lm_head — the tiers that
`--no-q8-router` exists to un-pin — are already `MQ4G256` for this dense model.
That is the `.mq4r` shape.

## Byte-rate direction (correcting a statement made in session, not in the records)

`.mq4r` reads **fewer** bytes per token than a Q8-router-pinned build, not more.
MQ4 is 0.53125 B/weight against Q8's 1.0625 B/weight, so pinning the fixed tier
at Q8 **doubles** the dominant term. Per `hipfire-quantize --help` on
`--no-q8-router`: the fixed tier is ~66 % of per-token decode bytes on a3b, and
`.mq2` reads *45 % more* bytes/token than `.mq4r` despite being 7 GB smaller on
disk. Neither original record makes a byte-rate claim, so neither is wrong on
this point; the correction is recorded here to keep the session's reasoning
trail accurate.

## Why the distinction matters for citation

The two variants are different artifacts with different decode-byte profiles and
different quality characteristics. A KLD figure measured on `.mq4r` must not be
quoted against a Q8-router-pinned `.mq4`, in either direction. All figures in
the two originals — including `alpha=0.05` at WT2 KLD 0.063102 / AG 0.224205,
and the degenerate uncalibrated baseline — are `.mq4r`.

## Cross-reference

The uncalibrated baseline arm in the originals is md5
`129909ad0fed21dcf72b5b9225e85604`, byte-identical to the `qwen3.8-27b.mq4`
recorded in
[`2026-08-14-qwen38-27b-qwen36-parity.md`](2026-08-14-qwen38-27b-qwen36-parity.md).
That shipped artifact is therefore also this `.mq4r` shape, which is what makes
the gfx942 two-token-attractor observation in the alpha record apply to a
shipped file rather than to a synthetic one.
