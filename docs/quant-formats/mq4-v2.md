# MQ4 v2.0 / HFQ4-G256 v2 — spec

**Status:** specified, partially prototyped, **not wired**. Quality and GEMV throughput are
measured; KLD is not. Do not ship without the open items in § 9.

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

## 1b · v1.5 — a strictly free intermediate, worth taking independently

Applying only the fp16-header half of the change, at per-256 granularity, gives a **132 B**
format with **identical quality to v1** (MSE 1.4415e-06, tail 9.4643e-07) that is **smaller
and faster**:

| | header loads | added VALU | B | R=2 ratio | R=4 ratio |
|---|---|---|---|---|---|
| v1 | 2 scalar (f32 scale, f32 zero) | 0 (`bit_cast` is free) | 136 | 1.0000 | 1.0000 |
| **v1.5** | **1 scalar** (packed fp16 pair) | 2 `cvt_f32_f16` | **132** | **0.9847** | **0.9773** |
| v2 | 2 scalar | 1 `cndmask` + 2 `cvt` | 136 | 0.9766 | 1.0128 |

v1.5 both **halves the header load count** and cuts **2.9% of weight traffic**, so on a
bandwidth-bound kernel it wins outright at quality parity. It also has no `cndmask`, which is
why it avoids v2's R=4 regression. Layout: `[0..2)` fp16 scale, `[2..4)` fp16 zero,
`[4..132)` nibbles — note the payload offset moves 8 → 4 and the stride 136 → 132, so unlike
v2 the payload is **not** byte-identical to v1.

**v1.5 and v2 are independent decisions.** v1.5 is size + speed at fixed quality; v2 is
quality at fixed size. They cannot be combined, because per-128 asymmetric needs the full 8 B.

## 1c · Above 136 B, hierarchical sub-scales beat raw fp16 ones

Raw fp16 sub-headers cost 4 B per sub-block, so per-32 granularity costs 32 B of header
(160 B total). Q4_K's approach — quantise each sub-block's scale and zero to **6 bits** against
a per-256 fp16 super-scale `d` and super-zero `dmin` — costs `nsub × 12 bits + 4 B`, i.e. 16 B
for per-32. Measured, data-free, all asymmetric:

| variant | B | bpw | overall MSE | tail-1% MSE |
|---|---|---|---|---|
| v1 asym 1×256 f32 hdr | 136 | 4.2500 | 1.4415e-06 | 9.4635e-07 |
| **v1.5** asym 1×256 fp16 hdr | **132** | 4.1250 | 1.4415e-06 | 9.4643e-07 |
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

**This does not change v1.5 or v2**, which are the best options at 132 B and 136 B
respectively — hierarchical encoding needs ≥3 sub-blocks before its 4 B super-header amortises.
It does mean that **if bytes are available, 144 B hierarchical is the next format to build**,
not 4×64 fp16, and it is worth 38% MSE / 62% tail over v2 for 5.9% more bytes.

This is deliberately **not a codebook**. Level placement stays uniform over `[min, max]`,
identical in rule to qt=1/6/13. See § 8 for why the codebook line (qt=40, qt=43) is retired.

---

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
| affine 1×256 fp16 hdr (§ 7) | 132 | 4.1250 | 1.4415e-06 | 9.4643e-07 | 0.000% |
| affine 4×64 fp16 hdr | 144 | 4.5000 | 9.7617e-07 | 3.2450e-07 | 0.000% |
| affine 8×32 fp16 hdr | 160 | 5.0000 | 7.4343e-07 | 1.8091e-07 | 0.000% |
| GL codebook qt=40 | 130 | 4.0625 | 1.1441e-06 | 2.0147e-05 | 0.1076 |
| SEL codebook qt=43 | 132 | 4.1250 | 1.0338e-06 | 4.8843e-06 | 0.000% |

`max_rel = 0.000%` for every affine row is structural: min/max fitting makes each block's
extremes exactly representable. That is the property the codebook formats lose.

**Metric discipline.** Rank on **tail-1% MSE**, not overall MSE. Overall MSE is a confirmed
PPL proxy and a confirmed **non-predictor of KLD** — see
`docs/perf-checkpoints/2026-08-17-what-predicts-kld-weight-mse-predicts-ppl-not-kld.md`. In
the only byte-comparable KLD comparison available (affine 0.043776 vs GL 0.048713), overall
MSE, relative MSE (HIGGS `t²`), imatrix-diagonal-weighted MSE, and full activation-weighted
`Tr(E A Eᵀ)` **all rank it backwards**; only tail-restricted MSE and max-coef error get it
right. v2 wins on the metric that has predictive support.

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

### Optional, independent: a 132 B repack of legacy artifacts

`affine 1×256 fp16 hdr` measures identically to qt=1 (§ 5), so a loader could repack legacy
136 B → **132 B** on the way into VRAM: **≈440 MB saved on a 15 GB model** for **0.008%**
quality, with no re-download. This is safe precisely because it only discards unused f32
header precision — there is no grid realignment, so none of the lossy round-trip above
applies. It does add a third layout to maintain; file it as a separate decision, not part of
v2.

---

## 8 · Why this is not a codebook

qt=40 (GL, tensor-global Lloyd) and qt=43 (SEL, 64-profile selector) are **retired**. The
tradeoff is monotone and structural:

| format | level placement | overall MSE | tail-1% |
|---|---|---|---|
| affine | **uniform** | 1.4415e-06 *(worst)* | **9.4635e-07** *(best)* |
| SEL qt=43 | 64 Lloyd-ish profiles, max-norm | **1.0338e-06** *(best)* | 4.8843e-06 (5.2×) |
| GL qt=40 | single Lloyd, RMS-norm | 1.1441e-06 | 2.0147e-05 (21×) |

A codebook's entire value is non-uniform placement, but MSE-optimal placement means
**conditional means**, which drift inward from the extremes — exactly what damages the tail.
A codebook with uniform spacing *is* affine. At 16 levels on a post-FWHT Gaussian there is no
room: the thing you would buy a codebook for is the thing that costs you the metric that
predicts KLD.

qt=43 demonstrated that fixing **clipping** is not sufficient — it achieved `max_rel =
0.000%`, identical to affine, and still lost the tail by 5.2×, because clipping and level
*shape* are independent failures.

Independently fatal for both: **neither has a GEMM**, so neither has a prefill path at any
quality. qt=43 additionally measures **9.16× slower** than qt=1 at R=2 and spills at R≥4.

**Where a codebook may still win: ≤3 bits.** mq3lloyd beat mq3 by 29.7% KLD, though with 6%
more bytes so it is not byte-matched. At low bit counts bulk error dominates and the tail is
relatively less of the damage. "Codebooks are for the sub-4-bit tier" is defensible;
"codebooks improve mq4" is not.

---

## 9 · Implementation checklist and open items

### Blocking before ship

- [ ] **KLD on both references.** Everything in § 5 is codec-side. Unlike qt=43, v2 *can* be
      scored at full speed because affine already has the WMMA GEMM family. Baselines to beat:
      qt=1 WT2 **0.043776** / v6-selector **0.587566**.
- [ ] **GEMM port.** Required for prefill at all.
- [ ] **gfx1030 decision** (§ 6).

### Wiring points

A format's identity is currently spread across ~8 match statements in 5 crates. Omitting any
one is **not a compile error** — this is how `qt=6 → HFQ4G256` and `RotationPlan::FwhtG128`
were once silently deleted with a green build. Every item here must be checked by execution,
not inspection:

- [ ] `QuantType` enum + `from_u8` — `crates/hipfire-quantize/src/main.rs`. **Next free qt is
      44**; 43 is burned by the retired SEL format and must not be reused.
- [ ] Encoder branch and `--format` alias (both the K-map chain and the non-K-map chain —
      there are at least two, and missing one silently routes tensors to the legacy
      `Q4F16G64` fallback)
- [ ] `DType` variant + `size` / `row_stride` / `requires_k_mod_256` —
      `crates/rdna-compute/src/dispatch.rs`, and the `lib.rs` re-export
- [ ] **`RAW_CODECS`** — `crates/hipfire-runtime/src/weight_backend.rs`. Omitting this row is
      what made every qt=40 artifact unloadable with `unsupported quant_type 40`. Verify by
      loading a real artifact, not by reading the table.
- [ ] `dtype_from_quant_type` in every arch crate
- [ ] `dtype_rotation_plan` → `FwhtG256` for the MQ line
- [ ] `dtype_post_rotation_variant` → `Prerotated`
- [ ] arch predicate → `HasWave32` (and a wave64 variant only after § 4's wave64 split is
      verified)
- [ ] `tables/gemv_table.rs` `register_prerotated`
- [ ] `KernelKey` variant + `for_gemv_prerotated`
- [ ] `families/gemv.rs` dispatch arm
- [ ] `kernels.rs` `include_str!` per new kernel file
- [ ] `embed_classify` only if the format can ever back an embedding table

### Port surface — the ACTUAL minimal set for gfx1201 dense

Traced through the dispatch tables (`forward_slots.rs`, `families/fused_qkv.rs`,
`families/gemm.rs`, `families/gemv.rs`, `rdna-compute/src/gemm.rs`, `gemv.rs`), a dense
Qwen3.5/3.8-27b (arch_id=5) on gfx1201 touches **11 translation units, ~13 header site
pairs** — not the ~700 sites across ~127 files that the whole HFQ4-G256 family spans.

**Prefill** — gfx12 WMMA fused GEMMs plus residual. `_bt` siblings are separate sources
selected when `HIPFIRE_GATE_UP_BT` is on and the batch triggers BT, so they are required, not
optional:

| file | serves | header pairs |
|---|---|---|
| `gemm_qkvza_hfq4g256_wmma.gfx12.hip` | linear-attn `in_proj_qkv/z/a/b` | 1 |
| `gemm_qkvza_hfq4g256_wmma_gfx12_bt.hip` | same, BT path | 1 |
| `gemm_qkv_hfq4g256_wmma.gfx12.hip` | full-attn `q/k/v_proj` | 1 |
| `gemm_gate_up_hfq4g256_wmma.gfx12.hip` | `mlp.gate/up_proj` (+ `ldsstage` symbol, same source) | 2 |
| `gemm_gate_up_hfq4g256_wmma_gfx12_bt.hip` | same, BT path | 1 |
| `gemm_hfq4g256_residual_wmma.gfx12.hip` | `o_proj` / `down_proj` (+ `ldsstage`) | 2 |
| `gemm_hfq4g256_residual_wmma_gfx12_bt.hip` | same, BT path | 1 |

**Decode** — fused projections plus residual GEMV:

| file | serves |
|---|---|
| `fused_qkvza_hfq4g256.hip` | linear-attn projections |
| `fused_qkv_hfq4g256.hip` | full-attn projections |
| `fused_gate_up_hfq4g256.hip` | MLP gate+up |
| `gemv_hfq4g256_residual.hip` | `o_proj` / `down_proj`, **rows forced to 1 on non-RDNA3**, so R=1 on gfx1201 |

**Correction to earlier measurements in this campaign.** All the v1 / v1.5 / v2 throughput
numbers in § 6 were taken on `gemv_hfq4g256_multirow`, which the scout established is **NOT on
the dense MQ4 projection path** — fused kernels cover qkv/qkvza/gate_up and the residual GEMV
covers o_proj/down_proj. Those numbers remain valid as a *proxy* for the header change (same
payload access, same header pattern, same lane mapping) but they are **not** measurements of
the shipping decode path. Re-measure on `fused_*` and `gemv_hfq4g256_residual` before quoting
throughput for a shipped v2.

**Explicitly out of scope**, and porting them speculatively costs real work for nothing: all
MoE variants (`gemv_hfq4g256_moe_*`, `gemm_*_moe_grouped_*`), all `muse_*` research variants,
`.gfx11*` / `.gfx942` / `.gfx1030` / wave64 / dp4a / cpol variants, `ldscoop`, `ldsx`,
`.v1`-`.v5`, and the `XBATCH` single-row path.

### On factoring the header decode

The kernels do support shared headers — `turbo_common.h` is included by 51 files — but the
mechanism is textual: the Rust side strips the `#include` and prepends the header source
(`attention.rs:3925`, `.replace("#include \"turbo_common.h\"", "")`). So a shared
`hfq4_load_hdr` helper needs per-launcher plumbing on top of the kernel edits.

At 11 files with 13 site pairs, **inline the substitution per kernel**; the bit-identity test
in § 9 is what makes the port safe, not the factoring. Factoring is the right move for a
broader quantization refactor covering all ~127 files, where the launcher plumbing amortises.

### Prototypes and harnesses that exist

| artifact | what |
|---|---|
| `/home/kaden/v2k.hip` | v2 port of `gemv_hfq4g256_multirow.hip`, entries `gemv_mq4v2_multirow_r{2,4,8}`; 0 leftover f32 header reads, 15 half-wave selects |
| `/home/kaden/v2_single.hip` | v2 port of `gemv_hfq4g256.hip` (single-row, R=1); 7 header pairs; **`HIPFIRE_HFQ4G256_XBATCH_KERNEL` excluded, not ported** |
| `tools/quant-design/` | GPU codec sweep harness (14 configs in 2.9 s) |
| `crates/hipfire-quantize/examples/mq_composable_bench.rs` | quality metrics per variant, self-checked to 0.11% |
| `crates/rdna-compute/examples/bench_gemv_paired_throughput.rs` | paired device-side GEMV throughput |

### Also unverified

- [ ] wave64 half-split (§ 4)
- [ ] the `XBATCH` single-row path
- [ ] MoE paths (`gemv_hfq4g256_moe_*`, 14 header sites each)
- [ ] behaviour when `K % 256 == 0` but a tensor has fewer than 256 columns in a tail group

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
