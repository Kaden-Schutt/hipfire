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
not be used to build an `mq5`-class product. Today the measured `mq5` cells use
`MQ5G256V2` (qt=48).

---

## 1 · The honesty invariant

> **`mq<N>` means measured model bpw ∈ [N, N+1). No exceptions.**

**Model bpw, not codec bpw.** These differ and only one of them is the product's
promise:

- *Codec bpw* — what the encoding costs per weight it covers. MQ4G256V2 is
  136 B per 256 weights = **4.25 bpw**.
- *Model bpw* — total artifact bytes × 8 ÷ parameter count. Lands at
  **4.46–4.90** for the same codec under the three rungs below, because
  embeddings, lm_head, and (on `pro`) recurrent/SSM lifts sit above base width
  and the structural F16 tensors add a sliver.

The product quotes model bpw, because that is what the user downloads and holds
in VRAM. Codec bpw is what you would get if every weight used the base format
with no role lifts.

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
while historical `mq4r` on Qwen3.8 meant lm_head *not* at Q8, because the
suffix fused a Redline route claim with a role ladder. The product ladder
below separates those concerns: three rungs, role lifts only.

| product | roles above base width | Qwen3.8-27B MQ4V2 bpw | legacy label (do not mint new) |
|---|---|---:|---|
| `mq<N>-xt` | embeddings (+ always-on conv Q8 on Qwen3.8) | 4.456 | `.mq4r` |
| `mq<N>` | embeddings + lm_head | 4.659 | `.mq4` |
| `mq<N> pro` | + recurrent/SSM state (`ssm_out`) | 4.897 | `ctl2`, `head_ssm` |

**Three rungs per bit width. No more, no fourth.** Every rung carries a claim,
which is why there is no "truly uniform, no guarantee" tier — a rung that
promises nothing is not a product, and the base-width-everywhere point is what
the codec bpw already describes. There is no `xtx` (or any other fourth speed
suffix): speed claims live in measured AR/DFlash columns, not in the product
name.

`xt` denotes the **faster** variant, matching its GPU meaning: it drops the
lm_head to base width. Historical `.mq4r` paired that recipe with a retained-
Redline admission, which is what `r` denoted. **The new V2 XT cells are not
automatically Redline-admitted**; transport eligibility remains a separate,
fixture-bound certification decision. `pro` takes the precision direction.

The structural F16 tensors — norms, `A_log`, `dt_bias`, AWQ scale vectors — are
never quantized at any rung. On Qwen3.8-27B they total 679,424 elements, about
0.0025% of parameters, and are not a ladder axis. On Qwen3.8, linear-attn
`conv1d` is held at Q8F16 on **every** rung (not a ladder axis); it is listed
in the role recipes so the census is exact.

### Qwen3.8-27B role recipes (this campaign)

Architecture-neutral product tiers map to these encoder lifts on Qwen3.8-27B
(quantize log `product tier:` lines; recipe census is deterministic):

| rung | lifted classes (encoder) | default precision of lifts | notes |
|---|---|---|---|
| `xt` | `embed` | Q8F16 | `lm_head` and `ssm_out` stay at base codec; `conv1d` always Q8F16 |
| base | `embed`, `lm_head` | Q8F16 | `ssm_out` stays at base codec; `conv1d` always Q8F16 |
| `pro` | `embed`, `lm_head`, `ssm_out` | Q8F16 | `conv1d` always Q8F16 |

**Fixed-tier overrides used on the measured ladder** (manifest
`fixed_tier`; keeps the product inside its bpw class when a full Q8 lift
would overshoot or when low-bit heads need a mid ladder step):

| cell | `fixed_tier` | effect |
|---|---|---|
| `mq2 pro` | `lm_head:mq6v2,ssm_out:mq6v2` | head + `ssm_out` → **MQ6G256V2** (not Q8) |
| `mq3 pro` | `ssm_out:mq6v2` | `ssm_out` → **MQ6G256V2**; head stays Q8F16 |
| all other cells | `null` | default table above |

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
published set, one or two classes up: `mq2 pro` lifts head/`ssm_out` to MQ6,
`mq3 pro` lifts `ssm_out` to MQ6, `mq4`/`mq5`/`mq6 pro` lift `ssm_out` to Q8.
The ladder terminates at Q8 rather than at an unrelated format, so no codec
sits outside the ladder and the § 1 budget governs each step.

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

### Measured grid — Qwen3.8-27B MQ V2 family (2026-08-20)

Authoritative evidence: hiptrx `ssh://hiptrx/home/kaden/qcal/ladder-v2/`
(`summary.md` / `summary.json` / `summary.csv`), generated
`2026-08-20T14:13:16Z`, commit `bfe3cfead5856c5ee85e972b609712b0badbd1b1`,
params **26,895,998,464**. Host: 4×gfx1201, ROCm/HIP 7.14. KLD protocol: WT2 +
v6sel, 24 chunks, prefill scoring, kv-mode q8. Prompt fixture
`benchmarks/prompts/merge_sort_thinking_off.txt` md5
`253c7ac50857fe6d0e10fb0d2c5e35c0` (sha256
`d671894964cb957643fcb961151f3d1b407cb5c206766eaed60e9c593e6ed9d0`). Binaries
(sha256): `hipfire-quantize`
`8e123d06d481d91c43e0a70ba383d0ee6e37b1be0bf3390ef1b4a95707d59661`,
`hipfire` `c005c4dd7fba4794fb107348f1d1a12d6a05d7e8e89a11e6dd41135adf275fdd`,
`eval_hipfire` `53f9b948355756a1f6ed7e27d84d485fb6f4815d7e9643e660be7678d06e7de2`,
`dflash_convert` `ebbe90cfcdd28365e507d751fa45477c918317b5af0b88a40a97b01ae7f2bc31`.

**Status is evidence status, not registry publication.** No cell below is
declared shipping or registry-published from this table alone. AR acceptance is
`null` (`unavailable_in_native_generate_v1`) for every cell — never derived.
DFlash τ is recorded in the remote summary; acceptance remains unavailable.

| product | rung | model bpw | class | KLD WT2 | KLD v6sel | base codec | qt | AR dec/pre tok/s | status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| `mq2-xt` | xt | 2.551 | 2 | 13.239513 | 14.202591 | MQ2G256V2 | 50 | 49.0 / 545.6 | **measured / rejected** |
| `mq2` | base | 2.848 | 2 | 12.441964 | 13.428047 | MQ2G256V2 | 50 | 43.8 / 523.8 | **measured / rejected** |
| `mq2 pro` | pro | 2.966 | 2 | 12.466792 | 13.447430 | MQ2G256V2 | 50 | 42.7 / 517.0 | **measured / rejected** |
| `mq3-xt` | xt | 3.503 | 3 | 0.248348 | 1.411523 | MQ3G256V2 | 49 | 40.7 / 467.8 | measured candidate |
| `mq3` | base | 3.753 | 3 | 0.153658 | 1.032401 | MQ3G256V2 | 49 | 37.6 / 459.4 | measured candidate |
| `mq3 pro` | pro | 3.922 | 3 | 0.130314 | 0.924635 | MQ3G256V2 | 49 | 36.7 / 452.5 | measured candidate |
| `mq4-xt` | xt | 4.456 | 4 | 0.057449 | 0.771410 | MQ4G256V2 | 44 | 35.3 / 490.7 | measured candidate |
| `mq4` | base | 4.659 | 4 | 0.039033 | 0.544517 | MQ4G256V2 | 44 | 33.2 / 479.0 | measured candidate |
| `mq4 pro` | pro | 4.897 | 4 | 0.032495 | 0.484145 | MQ4G256V2 | 44 | 31.8 / 473.8 | measured candidate |
| `mq5-xt` | xt | 5.408 | 5 | 0.015028 | 0.440984 | MQ5G256V2 | 48 | 29.4 / 346.3 | measured candidate |
| `mq5` | base | 5.564 | 5 | 0.010255 | 0.278077 | MQ5G256V2 | 48 | 28.2 / 341.8 | measured candidate |
| `mq5 pro` | pro | 5.746 | 5 | 0.009006 | 0.237993 | MQ5G256V2 | 48 | 27.4 / 346.1 | measured candidate |
| `mq6-xt` | xt | 6.361 | 6 | 0.004389 | 0.220915 | MQ6G256V2 | 47 | 26.0 / 333.9 | measured candidate |
| `mq6` | base | 6.469 | 6 | 0.002771 | 0.152813 | MQ6G256V2 | 47 | 25.2 / 330.5 | measured candidate |
| `mq6 pro` | pro | 6.596 | 6 | 0.002208 | 0.136504 | MQ6G256V2 | 47 | 24.8 / 331.2 | measured candidate |

Every row satisfies § 1: measured bpw ∈ [N, N+1) for product `mqN`.

**`mq2` rejection (product admission, not measurement failure).** All three
`mq2` cells were built, KLD-scored, AR-benched, and serve-probed. WT2/v6sel KLD
is catastrophic (~12–14 nats). Decoded serve text is coherent enough to read
but degraded/repetitive (`serve-mq2.json`: no attractor/empty flags; content
quality is not shipping-grade). **Do not call MQ2V2 a shipping product.** The
rows stay on the grid so the class is defined and the failure is dated.

**`mq3`–`mq6` candidates.** Serve batteries on the base artifact of each class
(`serve-mq{3,4,5,6}.json`): coherent completions, no attractor/empty. KLD is in
a usable range and improves monotonically with class. Still **candidates**, not
registry cards — publication requires the separate registry/validation path.

Artifact digests (sha256 of each cell's `.hfq`; full rows also carry md5/bytes
in `summary.json`):

| cell | artifact | sha256 | bytes | md5 |
|---|---|---|---:|---|
| mq2-xt | `…/qwen3.8-27b.mq2v2.xt.hfq` | `b86de92dd07e3ac53e239459ef4f4881b798ba1296ce15a0974f59007a8e894e` | 8574872576 | `12d3cafef619b552f78e9ae557b899ed` |
| mq2-base | `…/qwen3.8-27b.mq2v2.base.hfq` | `547bc0ae3dfefaef583c4f8756511f362389e93765f2136fab64691b778ac186` | 9574976512 | `8693a1305eec5f77eed8db19bd49a94b` |
| mq2-pro | `…/qwen3.8-27b.mq2v2.pro.hfq` | `6cc00eddd39bbb06824905276bfa0fd34edeed40b3dd3020b7b7a0412b0fad1b` | 9971776512 | `4823a452391cb3cdc58616b327a74ed2` |
| mq3-xt | `…/qwen3.8-27b.mq3v2.xt.hfq` | `3e04fc8db80bda557b965ec60ac876cf2500fced7f340624f3fcbeae134af5c5` | 11777616896 | `80bb9198e6a565fc006b2ae1b7c89eca` |
| mq3-base | `…/qwen3.8-27b.mq3v2.base.hfq` | `09c3544690aceca29e1822d79adab6ffcc8fd9e4b58359fe8dfb185ef49811c9` | 12618796032 | `ad20b3d3a9a7254e7a9c596fc97b411a` |
| mq3-pro | `…/qwen3.8-27b.mq3v2.pro.hfq` | `394c50966bf4f68172df8eb34cd7ded8f9d0576c9ef24ea6ba639a88c184f795` | 13184433152 | `b171dd618eecf1fe6aed2c1dc5eef4dc` |
| mq4-xt | `…/qwen3.8-27b.mq4v2.xt.hfq` | `9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7` | 14980361216 | `e45d15bfe0c9a87132697101d17cbed6` |
| mq4-base | `…/qwen3.8-27b.mq4v2.base.hfq` | `5bb556a6cc84035234995c017c9791aa3951ad1eae4cf8c8172b0eaef399e507` | 15662615552 | `d1292b4d5bd6046693604201a6ca8074` |
| mq4-pro | `…/qwen3.8-27b.mq4v2.pro.hfq` | `e6f2ac87042b9e314c323f00bc499a6304cd5624021ae501ce90b12a3a7ea3fa` | 16464182272 | `279c563786499c6651de6a3a57b42b02` |
| mq5-xt | `…/qwen3.8-27b.mq5v2.xt.hfq` | `f4760c159f80d5d3ca237593a191dd553790cba0f89de59827f430b23109b39b` | 18183105536 | `e7c17382812f33df90f5138eb7cf9973` |
| mq5-base | `…/qwen3.8-27b.mq5v2.base.hfq` | `c018a0d7510bffbb3788844d5d8f72694464244e16f128b7e34804045324cf25` | 18706435072 | `cff5d051ce8a2a44a5eba6b9b43d595d` |
| mq5-pro | `…/qwen3.8-27b.mq5v2.pro.hfq` | `7a46204f5ce16b260ebb028359a945c8c02f5abe87c040df063db782a99ee7cd` | 19319258112 | `50119af7b6974320e574774b663d2f6d` |
| mq6-xt | `…/qwen3.8-27b.mq6v2.xt.hfq` | `9d472ddc5b4e11a1986bfc83c54c0dac979a8e1d7186613dd7c7436f69ec8b2f` | 21385849856 | `de6eb059a10577b80f0a162e5c89249e` |
| mq6-base | `…/qwen3.8-27b.mq6v2.base.hfq` | `b798ea1166fc03a568f6daf8090b20b7d6314af7429951c03d86540385568db8` | 21750254592 | `ae18a5d9e7926474064586387126aa5c` |
| mq6-pro | `…/qwen3.8-27b.mq6v2.pro.hfq` | `58ac3ee645ede2bad2f6f833db1d4abbfdc1850047fdadef036d35a023bb9401` | 22174333952 | `aac76dcc771b424d605e813b0b4aa1c1` |

Paths are under `/home/kaden/qcal/ladder-v2/artifacts/` on hiptrx.

#### DFlash same-bit vs mq4 control (decode tok/s median)

Lifecycle: fresh-process DFlash bench, 3 reps × 5 samples, choose higher
same-bit vs mq4v2 control median (tie → smaller draft). Draft control sha256
`d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc`. Acceptance
null on every arm.

| cell | same-bit dec | ctrl (mq4) dec | chosen draft | τ same | τ ctrl |
|---|---:|---:|---|---:|---:|
| mq2-xt | **60.1** | 39.5 | mq2v2 | 1.82 | 0.87 |
| mq2-base | 46.0 | **63.3** | mq4v2 | 1.28 | 2.15 |
| mq2-pro | 62.4 | **88.2** | mq4v2 | 2.17 | 3.54 |
| mq3-xt | 237.4 | **264.8** | mq4v2 | 11.70 | 13.11 |
| mq3-base | 225.5 | **250.8** | mq4v2 | 11.70 | 13.11 |
| mq3-pro | 221.2 | **248.1** | mq4v2 | 11.70 | 13.11 |
| mq4-xt | 251.6 | — | mq4v2 | 11.70 | — |
| mq4-base | 263.3 | — | mq4v2 | 13.11 | — |
| mq4-pro | 258.2 | — | mq4v2 | 13.11 | — |
| mq5-xt | 221.3 | **225.2** | mq4v2 | 13.11 | 13.11 |
| mq5-base | 214.5 | **217.4** | mq4v2 | 13.11 | 13.11 |
| mq5-pro | 213.2 | **217.0** | mq4v2 | 13.11 | 13.11 |
| mq6-xt | 207.8 | **211.5** | mq4v2 | 13.11 | 13.11 |
| mq6-base | 202.8 | **206.6** | mq4v2 | 13.11 | 13.11 |
| mq6-pro | 202.1 | **205.9** | mq4v2 | 13.11 | 13.11 |

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
about 8.5 bpw per weight against MQ4G256V2's 4.25, measured as the 0.203 bpw step
between `mq4` (4.659) and `mq4-xt` (4.456), which differ primarily in the lm_head.

Lifting to MQ6 or MQ5 instead of Q8 therefore buys a rung back, and the size of
that rebate depends entirely on how much of the model the lifted roles are:

| model | embed + lm_head share | Q8 → MQ6 rebate |
|---|---:|---:|
| Qwen3.8-27B dense | 2.54B / 26.9B = 9.5% | ~0.19 bpw |
| Qwen3.6-35B-A3B | 0.51B / 34.7B = 1.5% | ~0.04 bpw |

On the dense model that is nearly a whole rung — the measured `mq4` → `mq4 pro`
step is 0.238 bpw — so an MQ6 lift can fund `pro` where only the base rung fit
before (see `mq2 pro` / `mq3 pro` fixed-tier overrides). On the MoE it is noise,
because the experts dominate the budget. Same product name either way, which is
the reason codec is kept out of it.

---

## 4 · Orientation for people arriving from llama.cpp

Approximate neighbours by size. hipfire quotes measured model bpw, so these are
comparable on bytes; compare quality on the published KLD, not on the tier name.
Three rungs only:

| hipfire | model bpw (Qwen3.8 MQ4V2) | nearest ggml |
|---|---:|---|
| `mq4-xt` | 4.456 | Q4_K_S |
| `mq4` | 4.659 | between Q4_K_S and Q4_K_M |
| `mq4 pro` | 4.897 | Q4_K_M |

Codec bpw (~4.25 for MQ4G256V2) is not a product rung; it is the encoding cost
before role lifts.

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
