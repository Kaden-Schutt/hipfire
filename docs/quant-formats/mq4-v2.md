# MQ4 v2.0 / HFQ4-G256 v2 — spec

**Status:** qt=44 `MQ4G256V2` and qt=45 `MQ4CG256` (padded-compat layout) are **wired for dense gfx12** (gfx1200/gfx1201). Scalar fused decode sources are cross-RDNA and compile on gfx1010; batched prefill and lm_head GEMM remain **gfx12-only**, with unsupported batched prefill falling back to per-token decode rather than a gfx12 WMMA dispatch. Codec, KLD, exact kernel parity, and model generation have been measured on the supported surfaces; see §§ 5, 9, and 10.

**One-line summary:** keep HFQ4-G256's 136 B/group and its byte-identical 128 B nibble
payload; re-spend the 8 header bytes from `f32 scale + f32 zero` per **256** weights to
`fp16 scale + fp16 zero` per **128** weights. Measured **−16.2% codec MSE** and **−40.2%
tail-1% MSE** at **zero size change**, data-free, throughput-neutral on 4 of 5 architectures.

---

## 1 · Why

Affine's header is over-provisioned for FWHT-rotated weights — but **only in precision, not in
the zero-point.** Measured on 278,528 real post-FWHT 256-blocks from the Qwen3.8-27B bf16
parent (layers 0 / 20 / 40, engine sign seeds 42/1042):

- **f32 header precision is unused.** Storing scale and zero as fp16 instead of f32 changes
  overall MSE by **0.00%** (1.4415e-06 either way) and tail-1% MSE by **0.008%**
  (9.4643e-07 vs 9.4635e-07). This frees 4 of the 8 header bytes.
- **The zero-point is worth keeping, but narrowly — roughly one doubling of granularity.**
  At matched 136 B, asymmetric per-128 beats symmetric per-64 by **1.4% MSE and 12.9%
  tail** (1.2089e-06 / 5.6621e-07 vs 1.2253e-06 / 6.3948e-07). Since a zero costs the same
  bytes as a scale, spending those bytes on the zero rather than on halving the group is the
  better trade — but only just. Symmetric variants are viable, not disastrous:
  sym 2×128 at 132 B gives 1.4554e-06 / 1.0203e-06, about 1% MSE and 7.8% tail behind v1.

  *Two earlier claims in this document were wrong and are retracted.* The first asserted the
  zero-point was "nearly worthless after rotation", inferred from mean-absolute asymmetry of
  0.0757 without measuring it. The second asserted dropping it cost **12× on the tail** — that
  came from a harness whose outer guard admitted only two of four codec modes, so the symmetric
  arms never executed and silently reported a fall-through path. The numbers above are from the
  fixed harness. Asymmetric fitting does make **both** block extremes exactly representable
  where symmetric makes one, which is why it wins; the effect is just an order of magnitude
  smaller than claimed.

So the 8 header bytes are 2× over-precise, and the zero-point earns its keep by a modest
margin. Halving the precision and spending the saving on **granularity** is what v2 does.

## 1b · v1.5 / MQ4C pad — same size as v1, fp16 header at +0

Applying only the fp16-header half of the change, at per-256 granularity, gives
**MQ4C (qt=45) pad: 136 B/group**, byte-for-byte v1's geometry — the **shipping**
form of v1.5:

| offset | size | field |
|---|---|---|
| `[0..4)` | 1 dword | packed fp16 header: low 16 bits fp16 `scale`, high 16 bits fp16 `zero` (per-256, `w = q·scale + zero`) |
| `[4..8)` | 4 B | zero padding (must be `0x00000000`) |
| `[8..136)` | 128 B | 4-bit nibbles, **byte-identical offset and stride to qt=1/6/13** |

Total `m·gpr·136` bytes per weight tensor, the same size as v1. The 4 pad bytes are the
deliberate price of keeping the payload at **+8**: a 132 B stride (`[0..2)` fp16 scale,
`[2..4)` fp16 zero, `[4..132)` nibbles, or planar `128 B payload plane + 4 B header plane`)
left the payload 4-byte aligned half the time and cost 7–11% prefill on
`global_load_b128`; that **132 B compact/planar layout is retired** — measured identical
quality to v1 (MSE 1.4415e-06, tail 9.4643e-07, 0.008% header-rounding cost, 2.43% smaller)
but not shipping. The pad and the retired compact share the same fp16 header precision, so
quality is identical; the shipping choice is alignment, not bytes. Header loads still halve
(1 scalar packed dword vs 2 scalar f32) with 2 `cvt_f32_f16` per group; unlike v2 there is
no `cndmask`, which is why the header-only change avoids v2's R=4 behaviour.

**v1.5 pad and v2 are independent decisions.** v1.5 pad is header-precision at fixed geometry;
v2 is granularity (per-128) at fixed size. They cannot be combined, because per-128
asymmetric needs both header halves' 8 B.

## 1c · Above 136 B, hierarchical sub-scales beat raw fp16 ones

Raw fp16 sub-headers cost 4 B per sub-block, so per-32 granularity costs 32 B of header
(160 B total). Q4_K's approach — quantise each sub-block's scale and zero to **6 bits** against
a per-256 fp16 super-scale `d` and super-zero `dmin` — costs `nsub × 12 bits + 4 B`, i.e. 16 B
for per-32. Measured, data-free, all asymmetric:

| variant | B | bpw | overall MSE | tail-1% MSE |
|---|---|---|---|---|
| v1 asym 1×256 f32 hdr | 136 | 4.2500 | 1.4415e-06 | 9.4635e-07 |
| **v1.5 pad** asym 1×256 fp16 hdr (pad `[0..4)` hdr, `[4..8)` zero, `[8..136)` nibbles — shipping) | **136** | 4.2500 | 1.4415e-06 | 9.4643e-07 |
| *v1.5 compact (retired)* asym 1×256 fp16 hdr (`[0..2)` scale, `[2..4)` zero, `[4..132)` nibbles — 2.43% smaller, alignment cost) | *132* | 4.1250 | 1.4415e-06 | 9.4643e-07 |
| **v2** asym 2×128 fp16 hdr | **136** | 4.2500 | 1.2089e-06 | 5.6621e-07 |
| asym 4×64 fp16 hdr | 144 | 4.5000 | 9.7617e-07 | 3.2450e-07 |
| **hier 8×32, 6-bit s+z** | **144** | 4.5000 | **7.4647e-07** | **2.1632e-07** |
| hier 8×32, 8-bit s+z | 148 | 4.6250 | 7.4363e-07 | 1.8300e-07 |
| asym 8×32 fp16 hdr | 160 | 5.0000 | 7.4343e-07 | 1.8091e-07 |
| hier 16×16, 6-bit s+z | 156 | 4.8750 | 5.2008e-07 | 1.3329e-07 |

Two conclusions:

1. **Hierarchical per-32 at 144 B matches raw-fp16 per-32 at 160 B** (7.4647e-07 / 2.1632e-07
   vs 7.4343e-07 / 1.8091e-07) — **16 bytes cheaper for essentially equal quality**. 6 bits is
   enough; going to 8 bits buys 0.4% MSE for 4 more bytes.
2. **At equal 144 B, hierarchical per-32 strictly dominates raw-fp16 per-64** by 23.5% MSE and
   33% tail. So above 136 B the header encoding, not the granularity, is the binding choice.

**This does not change v1.5 pad or v2**, which are the best options at **136 B** — v1.5 pad for size-neutral header precision and v2 for quality at the same size. The retired 132 B compact was best at 132 B before alignment retired it; hierarchical encoding needs ≥3 sub-blocks before its 4 B super-header amortises.
It does mean that **if bytes are available, 144 B hierarchical is the next format to build**,
not 4×64 fp16, and it is worth 38% MSE / 62% tail over v2 for 5.9% more bytes.

This is deliberately **not a codebook**. Level placement stays uniform over `[min, max]`,
identical in rule to qt=1/6/13. See § 8 for why the historical codebook line (provisional GL id, qt=43 SEL) is retired.

---

## 1d · Name and qt allocation

`MQ` already denotes FWHT-rotated (qt=6 `HFQ4G256` is unrotated; qt=13 `MQ4G256` is rotated),
so no variant needs to restate the rotation.

| `--format` | qt | layout | status |
|---|---|---|---|
| `mq4` *(alias)*, `mq4v2` *(canonical)* | **44** | v2 — 136 B, per-128 asym, fp16 scale+zero at +0/`+4`, nibbles at +8 | **implemented / wired** (dense gfx12; gfx1010 scalar decode TU compile, batched prefill falls back) |
| `mq4c` | **45** | **v1.5 pad — 136 B, per-256 asym, fp16 scale+zero at +0, zero pad +4..8, 128 B nibbles at +8** (padded / v1-compatible geometry; **not** compact 132 B) | **implemented / wired** (same arch split as qt=44) |
| `mq4_k` | 46 | hierarchical — per-32 @144 B or per-16 @156 B, 6-bit s+z | specified, § 1c |
| `mq4v1`, `mq4g256`, `magnum` | 13 | unchanged — 136 B, per-256 asym, f32 scale+zero at +0/+4, nibbles at +8 | shipping |
| — | 43 | MQ4-G256-SEL, **retired** | burned, never reuse |

Three naming decisions, each deliberate:

- **`--format mq4` becomes an alias for qt=44, and qt=13 is NOT redefined.** Artifacts
  self-describe by qt, so no file is ever ambiguous; redefining qt=13 would make the scored
  baseline (`WT2 0.043776`) ambiguous by date, which is the same silent-reinterpretation hazard
  that produced the qt=43 / GLW mess. qt=13 stays reachable as `mq4v1`.
- **Not `mq4r`.** `.mq4r` is a live file extension in `crates/hipfire-cli/src/main.rs:63` that
  selects the Redline replay backend, and denotes "MQ4 gate + graded experts"
  (`qwen35.rs:6011`). `mq4c` names the **padded-compat** v1.5 form (136 B), not the retired 132 B compact.
- **`mq4_k` names its provenance.** Hierarchical sub-scale quantization is GGML's k-quant
  design; what is ours is the FWHT rotation beneath it and the per-16 granularity. Per
  [`AGENTS.md`](../../AGENTS.md) and [`PRIOR-ART.md`](../../PRIOR-ART.md) that should be cited
  in the name rather than absorbed silently.

## 1e · How v1 and v2 coexist

The landed implementation uses **dedicated translation units and symbols** for qt=44 and
qt=45 rather than compiling the v1 source behind a format macro. That keeps each header
decoder explicit: v1 reads two f32 fields, qt=44 selects one of two packed fp16 pairs, and
qt=45 reads one packed fp16 pair plus the required zero pad. Rust dispatch selects the
matching symbol from the tensor's `DType`; no format is allowed to consume another format's
bytes.

The important shared contract is the 136 B stride and payload offset, not shared source text.
All three formats keep their 128 nibble bytes at `[8..136)`. This gives two load-bearing
validation properties:

1. **v1 remains isolated.** Adding qt=44/45 does not add a conditional header decoder to the
   v1 translation units, so the legacy kernel path is unchanged.
2. **Bit-identity is directly testable.** Setting `scale_half0 == scale_half1` and
   `zero_half0 == zero_half1` makes qt=44 numerically identical to v1 on the same nibbles.
   `mq4v2_parity` and `mq4v2_gemm_parity` exercise that construction; `mq4c_parity` uses a
   discriminating padded fixture to catch a wrong +4 payload offset or 132 B stride.

## 2 · Byte layout

136 B per 256-weight group. Little-endian throughout.

| offset | size | field |
|---|---|---|
| `[0..2)` | fp16 | `scale` for **half 0** (weights 0–127) |
| `[2..4)` | fp16 | `zero` for **half 0** |
| `[4..6)` | fp16 | `scale` for **half 1** (weights 128–255) |
| `[6..8)` | fp16 | `zero` for **half 1** |
| `[8..136)` | 128 B | 4-bit nibbles, **byte-identical to qt=1/6/13** |

The payload is unchanged in layout *and* offset: lane `t` reads the u32 at `8 + 4*t`,
covering weights `8t .. 8t+7`, exactly as today.

### Why the halves land on lane boundaries

The existing kernels index `base = g*256 + tid*8` with `boff = tid*4`, so each lane's eight
weights are **contiguous**. On wave32 that puts weights 0–127 entirely in lanes 0–15 and
weights 128–255 entirely in lanes 16–31. **No lane straddles a half.** The header is
therefore uniform per half-wave, and both header dwords are read from **lane-invariant
addresses**, so they remain scalar loads exactly as in v1.

Group stride, alignment (8-byte), and `K % 256 == 0` are all unchanged from qt=1.

---

## 3 · Encoder

For each 256-weight group, after the format's normal rotation (FWHT-256 for the MQ line,
none for the HFQ4 line):

```
for h in {0, 1}:
    lo = min(w[128h .. 128h+127])
    hi = max(w[128h .. 128h+127])
    step_f32 = (hi - lo) / 15
    scale[h] = f16(step_f32)          # stored
    zero[h]  = f16(lo)                # stored
    # REQUIRED: quantize against the ROUND-TRIPPED fp16 values, not the f32 ones,
    # so encoder and decoder agree bit-exactly.
    st = f32(scale[h]); z = f32(zero[h])
    for i in half h:
        q[i] = clamp(rint((w[i] - z) / st), 0, 15)
```

Degenerate case: `hi == lo` (constant half) ⇒ `step_f32 == 0`. Emit `scale[h] = 0`,
`zero[h] = f16(lo)`, `q[i] = 0`; the decoder reproduces `lo` exactly.

**Round-tripping the header through fp16 before quantizing is mandatory.** Skipping it makes
the encoder's reconstruction differ from the kernel's, which shows up as a quality regression
that looks like a format defect. The measured cost of fp16 headers (0.008%) is only valid
with this discipline.

---

## 4 · Decoder

`w = q * scale[h] + zero[h]` where `h = (weight_index >= 128)`. In the kernels this is a
5-line substitution per unrolled group, replacing:

```c
float sc = __builtin_bit_cast(float, LOAD_WEIGHT_HEADER(gp,     goff));
float zp = __builtin_bit_cast(float, LOAD_WEIGHT_HEADER(gp + 4, goff + 4u));
```

with:

```c
const unsigned int hA = LOAD_WEIGHT_HEADER(gp,     goff);        // address UNCHANGED
const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);   // address UNCHANGED
const unsigned int hs = (tid < 16) ? hA : hB;                    // half-wave select
float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
```

Requires `#include <hip/hip_fp16.h>`. Both offsets must stay as written — changing them to a
lane-dependent address (`gp + (tid<16 ? 0 : 4)`) converts two scalar loads into a vector load
and loses the whole point.

### The half select is layout-derived, NOT a lane test

**Do not assume `(tid < 16)`.** The correct predicate is: *is this weight's byte offset within
the 128 B payload below 64?* Half 0 is weights 0–127 = payload bytes 0–63. Derive it from the
kernel's own nibble address expression, never from the lane id.

Measured across the 11 ported kernels, three distinct forms occur:

| predicate | sites | where | derivation |
|---|---|---|---|
| `tid < 16` | 34 | decode: `fused_qkv/qkvza/gate_up`, `gemv_hfq4g256_residual` | nibbles at `gp + 8 + tid*4`, so lane `t` owns bytes `4t..4t+3`; `4t < 64` ⇔ `t < 16` |
| `kt < 8` | 8 | WMMA GEMM main + BT bodies | nibbles at `gp + 8 + kt*8 + k_grp*4` with `k_grp = tid>>4 ∈ {0,1}`; `kt*8 + 4 < 64` ⇔ `kt < 8` for both `k_grp` |
| `quarter_in_group < 2` | 2 | WMMA `ldsstage` bodies | nibbles at `gp + 8 + quarter*32 + {0,8,16,24} + k_grp*4`; max offset at `quarter=1` is 60 < 64 |

In the WMMA main path the loop steps `kt += 4` over tiles `kt..kt+3`, so all four tiles in one
body share a half (kt=0,4 → half 0; kt=8,12 → half 1) and one select per body suffices.

A wrong predicate here **compiles, runs, and silently applies the wrong scale to half of every
tensor**. It is the single highest-risk detail in the port. For any kernel not in the 11, redo
this derivation from its own addressing rather than copying a form from this table. `<HALF>`
being "16 on wave32, 32 on wave64" is only true for the `tid`-indexed decode form.

**Net cost:** one `v_cndmask` plus two `v_cvt_f32_f16` per group per row.

For wave64 kernels the split is lanes 0–31 / 32–63 by the same contiguity argument, but this
has **not** been verified — see § 9.

---

## 5 · Measured quality

278,528 real post-FWHT 256-blocks; tail threshold = 99th percentile of |w| = 2.869166e-02.
Reproduced independently by two harnesses agreeing within **0.11%**
(`tools/quant-design/` GPU sweep; `crates/hipfire-quantize/examples/mq_composable_bench.rs`).

| variant | B | bpw | overall MSE | tail-1% MSE | max-coef err |
|---|---|---|---|---|---|
| qt=1/6/13 affine 1×256 f32 hdr | 136 | 4.2500 | 1.4423e-06 | 9.4561e-07 | 0.000% |
| **v2 affine 2×128 fp16 hdr** | **136** | **4.2500** | **1.2085e-06** (−16.2%) | **5.6684e-07** (−40.1%) | **0.000%** |
| affine 1×256 fp16 hdr — pad (shipping qt=45, `[0..4)` fp16 pair, `[4..8)` zero pad, `[8..136)` nibbles) | 136 | 4.2500 | 1.4415e-06 | 9.4643e-07 | 0.000% |
| *affine 1×256 fp16 hdr — compact (retired, `[0..2)`/`[2..4)`/`[4..132)` — 2.43% smaller)* | *132* | 4.1250 | 1.4415e-06 | 9.4643e-07 | 0.000% |
| affine 4×64 fp16 hdr | 144 | 4.5000 | 9.7617e-07 | 3.2450e-07 | 0.000% |
| affine 8×32 fp16 hdr | 160 | 5.0000 | 7.4343e-07 | 1.8091e-07 | 0.000% |
| GL codebook *(historical provisional id; not current qt=40)* | 130 | 4.0625 | 1.1441e-06 | 2.0147e-05 | 0.1076 |
| SEL codebook qt=43 | 132 | 4.1250 | 1.0338e-06 | 4.8843e-06 | 0.000% |

`max_rel = 0.000%` for every affine row is structural: min/max fitting makes each block's
extremes exactly representable. That is the property the codebook formats lose.

**Metric discipline.** Codec MSE does not stand in for model KLD; the two are recorded
separately. On the matched Qwen3.8 `ctl` fixture, qt=44 measured WT2 KLD **0.039033** versus
qt=13's **0.043776** (−10.83%), and v6-selector KLD **0.544517** versus **0.587566**
(−7.33%) at byte-identical size. The protected `ctl2` arm measured **0.032495**, and the
`attn` arm **0.025437**. Those fixture-bound results are recorded in
`docs/perf-checkpoints/2026-08-18-mq4-v2-qt44-kld-beats-v1-at-equal-bytes.md`; they do not
make overall MSE a KLD proxy. The earlier affine-vs-GL inversion remains the counterexample:
overall MSE, relative MSE, imatrix-diagonal-weighted MSE, and full activation-weighted
`Tr(E A Eᵀ)` all ranked GL ahead while KLD ranked affine ahead; tail-restricted MSE and
max-coefficient error were the useful diagnostics.

The ladder does not stop at v2: 4×64 and 8×32 are better on both metrics if bytes are
available. v2 is the best point **at 136 B**.

---

## 6 · Measured throughput

hipx + k9lin, interleaved sample-by-sample, device hipEvent timing, ≥32 warmups, 100 iters,
3 runs. Ratio = v2 / v1; > 1 is slower. Each arch's **shipping default R** is the only row
that matters, per `ArchCaps::gemv_rows_default()` =
`if is_wave64_native || is_rdna2 || is_rdna3_dgpu { 1 } else { 2 }`.

| arch | GPU | default R | ratio @4096×5120 | ratio @4096×17408 | VGPR v1→v2 | spills | verdict |
|---|---|---|---|---|---|---|---|
| gfx1201 | RX 9070 | 2 | 0.9645 | — | 94 → 94 | 0 | **free** |
| gfx1100 | RX 7900 XTX | **1** | 0.97–0.99 | 0.97–0.99 | 72 → 76 | 0 | **free** |
| gfx1151 | Radeon 8060S | 2 | 0.9846 | 0.9978 | 94 → 94 | 0 | **free** |
| gfx1010 | RX 5700 XT | 2 | 1.0020 | 0.9984 | 36 → 36 | 0 | **free** |
| **gfx1030** | **RX 6950 XT** | **1** | **1.1386** | **1.1758** | 61 → 61 | 0 | **regression** |

R=1 exercises the single-row `gemv_hfq4g256.hip`; R≥2 exercises
`gemv_hfq4g256_multirow.hip`. **Both had to be ported** — gfx1100 and gfx1030 never execute
the multirow path.

### gfx1030: why, and what to do

Confirmed over 6 invocations (5120-shape 1.1386 / 1.1287 / 1.1226 / 1.1594; 17408-shape
1.1758 / 1.1810 / 1.1709 / 1.1775). VGPR is unchanged at 61 with zero spills, so the cost is
instruction latency, not occupancy.

**Mechanism:** gfx1030 runs the 5120 shape at 17.37 µs for 11.18 MB = **643 GB/s**, against
the RX 6950 XT's **576 GB/s** VRAM peak. Exceeding VRAM peak means it is not reading VRAM —
the weight matrix fits inside the **128 MB Infinity Cache** (the 17408 shape, ~38 MB, does
too). On RDNA2 this kernel is therefore **compute-bound rather than bandwidth-bound**, and
v2's three extra VALU ops per group are fully exposed. Consistent with the regression
*growing* on the larger shape (13.9% → 17.6%) instead of amortising.

This inverts the natural intuition: **bandwidth saturation is what makes v2 free.** The
vulnerable target is the one with enough cache to have stopped being memory-bound.

Two options, both idiomatic given the five existing `gemv_hfq4g256.gfx1030.v*.hip` variants:

1. **Arch-gate the format** — gfx1030 keeps qt=1. Contained; costs a dtype branch at load.
2. **Raise gfx1030 to R=2**, where v2 measures 0.999. One line in `ArchCaps`, but it moves a
   tuned default for *every* 4-bit format on RDNA2 and needs its own validation.

Recommendation: (1). Not decided.

Also observed, not blocking because they are not shipping configs: gfx1201 R=4 is 1.0679
(VGPR 93 → 117); gfx1010 R=4/R=8 regress 45% / 21–24% and **spill** at R=4.

---

## 7 · Migration

**Legacy artifacts cannot be upgraded in place, and attempting it makes them worse.**

A qt=1 artifact holds only nibbles `q ∈ [0,15]` and a per-256 `(s, z)`; dequantised values
lie on a lattice of spacing `s`. v2 needs per-**128** min/max of the *original* weights,
which first quantization destroyed. Re-quantising `ŵ` onto a finer per-128 grid recovers
nothing and **adds** error: if a half spans 8 of 16 codes the new step is `s' = (8/15)s`, and
existing points at `min + k·s` map to `m = 1.875k` on the new grid — integral only at `k = 0`
and `k = 8`. Every intermediate point falls between new levels. Round-tripping through a
finer-but-misaligned grid is strictly lossy.

**Therefore: coexist.** This is unusually cheap because stride and payload are identical, so
v1 and v2 are **one kernel template with two instantiations**, selected by dtype through the
existing `KernelKey` mechanism. No second memory layout, no second GEMM family, no conversion
pass, no re-download. Legacy artifacts keep working indefinitely; new quantizations emit v2
when the parent is available.

### Optional, independent: a 136 B pad repack of legacy artifacts (same-size header rewrite)

`affine 1×256 fp16 hdr` (pad) measures identically to qt=1 (§ 5), so a loader can repack
legacy 136 B (f32 header at +0/+4, nibbles at +8) → **136 B pad** (`[0..4)` fp16 header,
`[4..8)` zero padding, `[8..136)` nibbles verbatim) as a **pure header rewrite**: rewrite 8
header bytes as one packed fp16 dword, zero 4 pad bytes, keep all 128 nibble bytes, stride
unchanged. No re-download, no re-fit (0 nibble flips in 95M groups, ≤0.26-step drift, 0.007%
MSE / 0.09% RMS), no size change — the earlier 132 B compact (`136→132`, ~440 MB on 15 GB)
is retired for the same alignment reason as § 1b. This is safe precisely because it only
discards unused f32 header precision — there is no grid realignment, so none of the lossy
round-trip above applies. It does add a second header encoding to maintain; file it as a
separate decision, not part of v2.

---

## 8 · Why this is not a codebook

A **global-Lloyd (GL) codebook prototype** once used a **provisional identifier that now conflicts with the canonical registry**. That prototype and qt=43 (SEL, 64-profile selector) are **historically retired** as MQ4 quality directions. **Current qt=40 is `TQ2G128` and qt=41 is `BQ1G128`; this PR does not alter them.** The tradeoff that retired GL/SEL on the MQ4 line is monotone and structural:

| format | level placement | overall MSE | tail-1% |
|---|---|---|---|
| affine | **uniform** | 1.4415e-06 *(worst)* | **9.4635e-07** *(best)* |
| SEL qt=43 *(retired)* | 64 Lloyd-ish profiles, max-norm | **1.0338e-06** *(best)* | 4.8843e-06 (5.2×) |
| GL *(historical prototype)* | single Lloyd, RMS-norm | 1.1441e-06 | 2.0147e-05 (21×) |

A codebook's entire value is non-uniform placement, but MSE-optimal placement means
**conditional means**, which drift inward from the extremes — exactly what damages the tail.
A codebook with uniform spacing *is* affine. At 16 levels on a post-FWHT Gaussian there is no
room: the thing you would buy a codebook for is the thing that costs you the metric that
predicts KLD.

qt=43 demonstrated that fixing **clipping** is not sufficient — it achieved `max_rel =
0.000%`, identical to affine, and still lost the tail by 5.2×, because clipping and level
*shape* are independent failures.

Independently fatal for both historical MQ4 codebooks: **neither had a GEMM**, so neither had a prefill path at any
quality. qt=43 additionally measured **9.16× slower** than qt=1 at R=2 and spilled at R≥4.

**Where a codebook may still win: ≤3 bits.** mq3lloyd beat mq3 by 29.7% KLD, though with 6%
more bytes so it is not byte-matched. At low bit counts bulk error dominates and the tail is
relatively less of the damage. "Codebooks are for the sub-4-bit tier" is defensible;
"codebooks improve mq4" is not.

---

## 9 · Landed state and remaining scope

### Landed / validated

| area | state |
|---|---|
| Codec layout (qt=44 v2 136 B per-128; qt=45 pad 136 B per-256) | specified and implemented |
| Loader / `RAW_CODECS` / `DType` / quantizer `--format` wiring | done |
| FWHT-256 rotation plan + prerotated dispatch | done |
| Dense decode fused QKV / QKVZA / gate_up + residual GEMV | cross-RDNA sources; **compile on gfx1010**; wired |
| Batched prefill WMMA GEMM + batched lm_head GEMM | **gfx12-only** (gfx1200/gfx1201); all six explicit `GemmMq4*` registry keys are gfx12-only |
| Unsupported batched prefill on non-gfx12 | **per-token decode fallback** (does not dispatch a gfx12 WMMA kernel) |
| FusedQkv / FusedGateUp decode registrations | remain **cross-arch** (not narrowed to gfx12) |
| gfx1201 exact parity examples | `mq4v2_parity`, `mq4v2_gemm_parity`, `mq4c_parity` |
| Qwen3.8 fixture-bound KLD | qt=44 `ctl` WT2 0.039033 / v6 0.544517; `ctl2` WT2 0.032495; `attn` WT2 0.025437 (§ 5) |
| gfx1010 | scalar fused decode TUs compile; batched prefill falls back as above |
| qt=40 `TQ2G128` / qt=41 `BQ1G128` | **untouched** by this work |
| qt=43 SEL | historically retired; id burned, never reuse |

### Not claimed / out of scope

- **wave64** half-split (§ 4) — not verified; wave64 remains unsupported for these formats.
- **MoE** paths (`gemv_hfq4g256_moe_*`, `gemm_*_moe_grouped_*`) — out of scope.
- **gfx1030 default-R decision** (§ 6) — still open if/when gfx1030 ships these dtypes; not a dense-gfx12 blocker.
- Research-only surfaces: `muse_*`, `.gfx11*` / `.gfx942` / dp4a / cpol / `ldscoop` / `ldsx` / `.v1`–`.v5` / `XBATCH` single-row path.

### Port surface that landed for dense gfx12

Traced through the dispatch tables (`forward_slots.rs`, `families/fused_qkv.rs`,
`families/gemm.rs`, `families/gemv.rs`, `rdna-compute/src/gemm.rs`, `gemv.rs`), a dense
Qwen3.5/3.8-27b (arch_id=5) on gfx1201 touches **11 translation units, ~13 header site
pairs** — not the ~700 sites across ~127 files that the whole HFQ4-G256 family spans.

**Prefill** — gfx12 WMMA fused GEMMs plus residual. `_bt` siblings are separate sources
selected when `HIPFIRE_GATE_UP_BT` is on and the batch triggers BT:

| file family | serves |
|---|---|
| `gemm_qkvza_{mq4g256v2,mq4cg256}_wmma.gfx12.hip` + `_gfx12_bt` siblings | linear-attn `in_proj_qkv/z/a/b` |
| `gemm_qkv_{mq4g256v2,mq4cg256}_wmma.gfx12.hip` | full-attn `q/k/v_proj` |
| `gemm_gate_up_{mq4g256v2,mq4cg256}_wmma.gfx12.hip` + `_gfx12_bt` siblings | `mlp.gate/up_proj` |
| `gemm_{mq4g256v2,mq4cg256}_residual_wmma.gfx12.hip` + `_gfx12_bt` siblings | `o_proj` / `down_proj` |

**Decode** — fused projections plus residual GEMV (cross-RDNA scalar sources):

| file family | serves |
|---|---|
| `fused_qkvza_{mq4g256v2,mq4cg256}.hip` | linear-attn projections |
| `fused_qkv_{mq4g256v2,mq4cg256}.hip` | full-attn projections |
| `fused_gate_up_{mq4g256v2,mq4cg256}.hip` | MLP gate+up |
| `gemv_{mq4g256v2,mq4cg256}_residual.hip` | `o_proj` / `down_proj` |

**Correction to earlier measurements in this campaign.** The first microbench numbers in § 6
used `gemv_hfq4g256_multirow`, which is a header-cost proxy rather than the dense projection
path. After fused qt44/45 routing landed, model-scale decode on the actual dense path measured
34.6 tok/s for qt13, 34.5 for qt44 (−0.29%), and 34.8 for qt45 (+0.58%) under the same
graph-disabled Q8-KV fixture. The proxy remains useful for the cross-architecture header-cost
analysis, but production claims use the fused-path measurement.

### On factoring the header decode

The kernels do support shared headers — `turbo_common.h` is included by 51 files — but the
mechanism is textual: the Rust side strips the `#include` and prepends the header source
(`attention.rs:3925`, `.replace("#include \"turbo_common.h\"", "")`). So a shared
`hfq4_load_hdr` helper needs per-launcher plumbing on top of the kernel edits.

At 11 files with 13 site pairs, **inline the substitution per kernel** was the landed approach;
bit-identity tests (equal half headers ⇒ v2 matches v1 on the same nibbles) make the port
safe. Factoring remains the right move for a broader quantization refactor covering all ~127
files, where the launcher plumbing amortises.

### Design harnesses (historical / still useful)

| artifact | what |
|---|---|
| `/home/kaden/v2k.hip` | early v2 port of `gemv_hfq4g256_multirow.hip`, entries `gemv_mq4v2_multirow_r{2,4,8}`; 0 leftover f32 header reads, 15 half-wave selects |
| `/home/kaden/v2_single.hip` | early v2 port of `gemv_hfq4g256.hip` (single-row, R=1); 7 header pairs; **`HIPFIRE_HFQ4G256_XBATCH_KERNEL` excluded, not ported** |
| `tools/quant-design/` | GPU codec sweep harness (14 configs in 2.9 s) |
| `crates/hipfire-quantize/examples/mq_composable_bench.rs` | quality metrics per variant, self-checked to 0.11% |
| `crates/rdna-compute/examples/bench_gemv_paired_throughput.rs` | paired device-side GEMV throughput |
| `mq4v2_parity`, `mq4v2_gemm_parity`, `mq4c_parity` | gfx1201 exact parity examples (landed) |

---

## 10 · Evidence

- `docs/perf-checkpoints/2026-08-17-mq4-v2-affine-2x128-fp16-header.md` — v2 codec + 5-arch
  throughput, and the gfx1030 Infinity-Cache analysis
- `docs/perf-checkpoints/2026-08-17-what-predicts-kld-weight-mse-predicts-ppl-not-kld.md` —
  why overall MSE is a PPL proxy and not a KLD proxy; HIGGS linearity theorem
  ([arXiv:2411.17525](https://arxiv.org/abs/2411.17525)); the falsified cross-term prediction
- `docs/perf-checkpoints/2026-08-17-136-byte-sub-block-scales-beat-affine-at-equal-bytes.md` —
  sub-block granularity beats outlier refinement
- `docs/perf-checkpoints/2026-08-17-selector-dominates-mq4-lloyd-at-fewer-bytes.md` —
  matched-granularity Lloyd comparison, and its amendment
- `docs/perf-checkpoints/2026-08-17-gl-clips-the-largest-coefficient-in-83pct-of-blocks.md` —
  the clipping diagnosis that started this line
