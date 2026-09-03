# Escha-W2 port — design

**Status:** Phase 1 implemented and gated; see §10 for measured results and §10.5 for the limitations that remain open.
**Date:** 2026-09-02
**Branch:** `nw_escha_w2` (worktree `~/repos/hipfire-escha`, off `origin/master` @ `8cd15a62b`)
**Targets:** `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` (first), `EschaLabs/Qwen3.8-27B-Escha-W2` (second)

## 1. What Escha-W2 is

Escha Labs publishes a 2-bit *quantization format*, not a model family. Two
checkpoints are in scope:

| release | HF repo | on disk | `quant_method` | hipfire `arch_id` |
|---|---|---|---|---|
| 35B-A3B | `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` | 12.3 GB | `eschamoe` | 6 |
| 27B | `EschaLabs/Qwen3.8-27B-Escha-W2` | 10.15 GB | `escha` | 5 |

**Essentially no architecture work.** `Qwen3.8-27B`'s `config.json` differs from
`Qwen3.6-27B`'s only in the `quantization_config` block — every architectural
field is identical. Both models are hybrid GatedDeltaNet + full-attention
(`full_attention_interval = 4`), already served by `hipfire-arch-qwen35` as
`arch_id` 5 (dense) and 6 (MoE). The registry already ships `qwen3.6:27b` and
`qwen3.6:35b-a3b`.

Base identity was checked both ways: the 35B's config differs from
`Qwen/Qwen3.6-35B-A3B` only in `transformers_version`, and the 27B's differs
from `Qwen/Qwen3.6-27B` only in the `quantization_config` block. Both bases are
VL-shaped composites (nested `text_config` plus a `vision_config` carrying no
weights), which is exactly the case `Qwen35Config::is_vl_text` already covers —
the converter takes that branch rather than treating them as plain text configs.

The one exception: the 27B's escha linears carry fp16 biases the base
architecture does not have, so arch-5 gains bias slots (§1.3, §9). Otherwise
this is a codec + loader project.

### 1.1 Format

The codec is open and bit-exactly specified: `EschaLabs/escha-mlx` is Apache-2.0
and ships `escha_mlx/ref.py`, a NumPy reference that is the format contract,
plus golden vectors under `tests/data/`. Nothing needs reverse-engineering.
Their `THIRD_PARTY_LICENSES` vendors exllamav3, and the codec is QTIP lineage.

**Weight stream.** 16x16 tiles, K bits/weight, packed `int16[in/16, out/16, 16K]`
(MoE exports carry a leading `[E]` axis; dense exports do not). Verified against
the shipped safetensors headers:

- `gate_up_proj.escha_code` `[256, 128, 64, 32]`, `in=2048 out=1024 K=2` -> 2.00 bpw
- `down_proj.escha_code` `[256, 32, 128, 48]`, `in=512 out=2048 K=3` -> 3.00 bpw

**Codebook.** Not a table — a 3-op integer hash. For a 16-bit state `s`:

```
r     = ((s * 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60      # 32-bit
value = f16_lo(r) + f16_hi(r)                              # fp16 RNE add
```

**This is a trellis, not a per-weight codebook.** Each weight's value is a full
16-bit state indexing 65536 fp16 values; consecutive states are overlapping
windows of the bitstream sliding by K bits. The K bpw is amortized across the
overlap. This is why no lossless repack into MQ2 exists (see §2).

**Rotation.** Unnormalized 128-point Walsh-Hadamard (Sylvester / natural order)
on contiguous 128-channel blocks, applied on *both* sides, with
`RS = 1/sqrt(128) = 0.088388347648`:

```
xh  = f16( H128(x_f32 * rin_f32) * RS )
mid = xh_f32 @ W_f32
y   = f16( H128(mid) * RS * rout_f32 )
```

Escha folds its sign flips into `rin`/`rout` rather than carrying a separate
sign stage. `s_in`/`s_out` are the end-to-end fine-tune scales; MoE exports ship
them all-ones, dense exports ship real values. Both collapse into `rin`/`rout`
by `fold_scales`, which keeps the product in f32 and rounds once.

**Verified 2026-09-02.** The above was checked, not inferred: `ref.py`'s
`reconstruct_fast` run against the committed goldens reproduces
`expected_gu_e0_k2.f16` and `expected_down_e0_k3.f16` **bit-exactly** at the
stated tile grid and packing, for both K. A single expert projection uses
**10,746 distinct fp16 values** — direct evidence for the trellis claim and for
§2. The goldens are also not synthetic: `packed_gu_e0_k2.i16` is byte-identical
to the shipped layer-0 / expert-0 `gate_up_proj.escha_code`, so G0 and G2 gate
against real model data.

**Coverage differs between the two models**, and the dense one is the larger
surface:

- **35B (`eschamoe`)** — escha covers routed experts only, as a *single fused*
  `gate_up_proj` (K=2, `[256,128,64,32]`) plus `down_proj` (K=3,
  `[256,32,128,48]`). Everything else is int8 W8A16: attention projections,
  linear-attn `in_proj_qkv`/`in_proj_z`/`out_proj`, shared expert, embeddings,
  `lm_head`. No bias tensors anywhere.
- **27B (`escha`)** — escha covers *every* projection: both attention families
  and all three MLP legs, with `gate_proj`/`up_proj` as **separate tensors
  carrying different K**:

  | tensor | layers | shape | in x out | K |
  |---|---|---|---|---|
  | `linear_attn.in_proj_qkv` | 48 | `[320, 640, 32]` | 5120 x 10240 | 2 |
  | `linear_attn.in_proj_z` | 48 | `[320, 384, 32]` | 5120 x 6144 | 2 |
  | `linear_attn.out_proj` | 48 | `[384, 320, 32]` | 6144 x 5120 | 2 |
  | `self_attn.q_proj` | 16 | — | 5120 x 6144 | 2 |
  | `self_attn.k_proj` | 16 | — | 5120 x 1024 | 2 |
  | `self_attn.v_proj` | 16 | — | 5120 x 1024 | 2 |
  | `self_attn.o_proj` | 16 | — | 6144 x 5120 | 2 |
  | `mlp.gate_proj` | 64 | `[320, 1088, 32]` | 5120 x 17408 | 2 |
  | `mlp.up_proj` | 64 | `[320, 1088, 48]` | 5120 x 17408 | 3 |
  | `mlp.down_proj` | 64 | `[1088, 320, 48]` | 17408 x 5120 | 3 |

  That is 10 distinct projections over 64 layers (48 linear-attention, 16 full
  attention) — 400 escha tensors. **The full-attention layers are escha-coded
  too**, which the linear-attention-only layer 0 does not reveal; sample a
  layer 3 as well as a layer 0 when validating the converter. Consequence for
  dispatch: the 27B needs the 3-way `FusedQkvQ8_0` path as well as the 4-way
  `FusedQkvzaQ8_0` one.

  Only `in_proj_a`/`in_proj_b` and the norms sit outside — but see §1.3 on what
  `ignore` does and does not mean.

### 1.2 `rout` carries a per-expert channel prune mask

`rin` is a clean sign x scale vector (no zeros, tight magnitude spread). **`rout`
on `gate_up_proj` is not.** Measured on layer 0:

| expert | `gate_up.rout` zeros | non-zero magnitude range |
|---|---|---|
| 0 | 560 / 1024 (54.7%) | 1.81 – 3.26 |
| 1 | 594 / 1024 (58.0%) | 1.99 – 3.21 |
| 2 | 142 / 1024 (13.9%) | 0.991 – 1.62 |
| 7 | 378 / 1024 (36.9%) | 0.979 – 1.02 |

The zeros are exact, the rate varies per expert, and the masks are **not shared**
between experts (pairwise agreement ~0.47, i.e. chance). `down_proj.rout` by
contrast has **zero** zeros on every expert sampled.

The structure is exact, not incidental. `rout` is applied last
(`y = f16(H128(mid) * RS * rout)`), so a zero hard-zeroes that output channel
for every input. For expert 0, the 560 zeros split as **280 in the gate half and
the same 280 channels in the up half** — `gate-only = 0`, `up-only = 0`.
Confirmed end-to-end: running `expert_linear` on random inputs yields 560/1024
output channels identically zero, and after SwiGLU **280 of the 512 intermediate
channels are dead for all inputs**.

So Escha's fine-tune leaves behind **structured, per-expert width pruning**,
carried in `rout` rather than in a mask tensor. Consequences:

- The "signs folded into `rin`/`rout`" description is correct for `rin` and for
  `down_proj.rout`, and incomplete for `gate_up.rout`, which is
  sign x scale x prune-mask.
- It is exploitable — see §4.5. It is also a correctness trap: a kernel that
  "optimizes away" the zero multiply without preserving exact-zero output would
  change results.
- The pruned `gate_up` columns are still stored in the code stream at 2 bits
  each and are never used. Physically dropping them would shrink the model but
  would break the verbatim/`memcmp` property, so it is not done in this port.

### 1.3 Four metadata traps

**`escha_config` has two lengths — and is optional (§1.4).** When present, MoE
exports ship `[9]`
= `[16, K, 2, 1, E, in, out, in_p, out_p]`; dense exports ship `[6]`
= `[16, K, 2, 1, in, out]`. Fields 0 (tile = 16), 1 (K) and the trailing dims
are identified. **Fields 2 and 3 (values `2` and `1`) are not identified** — most
likely the QTIP vector dim V and a version/flag. They are asserted equal to
their observed values, never interpreted; if a future release changes them,
conversion fails loudly.

**Trust `K`, not `bits`.** `quantization_config.layer_meta` disagrees with
itself across the two releases: the 35B records `down_proj` as
`{"bits": 3.0, "K": 3}` while the 27B records it as `{"bits": 2.0, "K": 3}`.
`bits` tracks the marketing rate on one and the true rate on the other. `K` is
consistent in both, and matches `escha_config[1]` and the code-tensor shapes.
The converter keys off `K` and uses `bits` only as a cross-check it is allowed
to fail.

**`ignore` means "not escha-coded", not "not quantized".** Both models list
`embed_tokens` and `lm_head` in `quantization_config.ignore`, and both ship them
as `weight_int8` + `weight_scale` anyway (`int8_embedding: true`). A converter
that reads `ignore` as "keep at source precision" will go looking for f16
tensors that do not exist. Classify on the tensor suffix actually present, not
on the ignore list. The 27B's list is shorter still (`in_proj_a`, `in_proj_b`,
`lm_head`) because its norms simply never match.

**The 27B carries biases the base model does not have.** Every escha linear in
the dense export ships an F16 `bias` (`in_proj_qkv.bias [10240]`,
`in_proj_z.bias`, `out_proj.bias`, `mlp.{gate,up,down}_proj.bias`). Base
Qwen3.8-27B has `attention_bias: false` and no MLP bias — these are the additive
fp16 output correction Escha's end-to-end fine-tune leaves behind, applied after
the output transform per `ref.py::dense_linear`. **`hipfire-arch-qwen35`'s
arch-5 path has no bias on these projections today and must gain one.** The 35B
MoE path needs no such change.

MTP packaging also differs: the 35B ships inline `mtp.*` tensors in the main
shards; the 27B ships a separate `mtp/` subdirectory with its own `config.json`
and `model.safetensors`.

### 1.4 Leaf contract: three required, four optional

Escha's own tests state the loader contract, and it is stricter and looser than
the spec first assumed. The namespace is
`CODED_LEAVES = (escha_code, escha_rin, escha_rout, escha_s_in, escha_s_out,
escha_config)`, plus `bias`.

**Required — `escha_code`, `escha_rin`, `escha_rout`.** A coded linear missing
any transform vector must fail loudly. Their test is named
`rejects_incomplete_linear` and its docstring is explicit that this must "fail
loudly at load, not decode into noise". Our converter and loader adopt the same
rule.

**Optional — `escha_s_in`, `escha_s_out`, `escha_config`, `bias`.** An export
produced without the end-to-end fine-tune stage ships none of them and must
still load and run (`test_dense_checkpoint_without_optional_leaves`). Note their
assertion is `bias is None`, not a zero bias — absence is a distinct state from
zero, even though the two are numerically identical once applied.

**Unknown `escha_*` leaves are a format mismatch and must be rejected**, not
silently ignored and not allowed to fail deep inside a parameter-name error.
Their named example is `escha_rotation_theta` — evidence the format anticipates
a theta-parameterized (Givens-style) rotation variant that today's checkpoints
do not use. hipfire already has `RotationPlan::Givens` for ParoQuant, so such a
variant would not be alien, but it is **out of scope here**: if a future release
ships it, conversion must stop rather than decode the codes under the wrong
rotation.

**Consequence for the converter.** Because `escha_config` is optional, it cannot
be the source of truth for `K`. `K` is always derivable from the code tensor's
own shape — the last dimension is `16K` — so that is the primary source, with
`escha_config[1]` and `layer_meta` used as cross-checks *when present*. This is
strictly more robust than §1.3's `bits`/`K` disagreement work-around: the shape
cannot disagree with itself.

## 2. Why there is no lossless repack into an existing MQ codec

MQ2 assigns each weight one of 4 levels within a linearly grouped run sharing
one scale. Escha assigns each weight any of 65536 fp16 values, at 2.0 amortized
bits, with no block scale. The alphabets are not nested and no MQ group scale
recovers the mapping — a lossless transcode into MQ2 is not merely lossy, it is
impossible. MQ8 would be near but still not lossless (256 levels vs arbitrary
fp16). Only fp16 storage is exactly lossless, at 8x the bytes.

This is measured, not argued from the spec: decoding the shipped layer-0 /
expert-0 `gate_up_proj` yields **10,746 distinct fp16 values** across the
2048 x 1024 matrix, spanning [-3.949, 3.949]. MQ2 can express 4 per group.

Separately, folding the Hadamards into an effective weight
`W_eff = diag(rin) . H128 . W . H128 . diag(rout) . RS^2` is exactly computable
(the Hadamards are block-diagonal), but it skips the intermediate f16 rounding
of `xh`, so it is a different numerical contract from the deployed one. Escha
flag the same deviation for their own Q8 repack.

**Therefore:** the repack is lossless into hipfire's *container* (codes stored
byte-for-byte, `memcmp` as post-condition), and the kernels are mandatory.
"Lossless repack" and "no new kernels" cannot both hold.

## 3. Decisions

| decision | value |
|---|---|
| quant types | `ESCHA2T16 = 42`, `ESCHA3T16 = 43` |
| registry quant label | `escha` |
| registry model ids | `qwen3.6:35b-a3b-escha`, `qwen3.8:27b-escha` |
| rotation plan | new `RotationPlan::EschaH128` |
| Phase 1 GPU path | decode tiles to `Q8_0` resident at load; H128 kept at runtime |
| Phase 2 GPU path | fused decode+GEMV; rotations unchanged |
| model order | 35B-A3B first, 27B second |

**Why ids 42/43.** The authoritative registry is
`crates/hipfire-quantize/src/hfq.rs` — both the `#[repr(u8)] enum QuantType`
and its `from_u8`, whose doc comment requires the two be kept in sync when a
variant is added. Everything in the crate imports `crate::hfq::QuantType`.

On `origin/master` the taken ids are **0–22, 24, 28–41, 44, 45, 47–51**, with
23/25/26/27 documented do-not-reuse reservations. **42 and 43 are the lowest
free pair**, and 46 and 52+ are also free.

Note what is *already merged* here, because it contradicts what the branch
survey suggests: 38/39 are `MQ2G256GL`/`MQ3G256GL` and **40/41 are
`TQ2G128`/`BQ1G128`** — the Bonsai ternary/binary types. Anyone re-deriving
this must read `hfq.rs` on master, not a `QuantType` enum in another checkout:
the `loop/gfx1151` branch carries a stale partial copy that stops at
`MFP2G32E8 = 37` and would suggest 38–41 are free. It also means the Neutrino
plan's reservation of 40/41 for `FV5G256`/`FV5B256` is stale and needs
renumbering independently of this port.

**Why two types, not one.** `decode8_k2` reads a 16-word tile at a fixed 16-bit
stride; `decode8_k3` walks 24 words at a computed bit offset with a modular
wrap. They are structurally different, and hipfire dispatches kernels off
`(QuantType, RotationPlan)` — one type with K in `group_size` would force a
runtime K branch through every existing match arm.

**Why `T16` and not `G256`.** The block *is* 256 weights, so `ESCHA2G256` would
look consistent with `MQ2G256`. It would also mislead: everywhere else in
hipfire `G256` means a linear run of 256 contiguous weights along a row sharing
one scale, whereas escha's 256 is a 16x16 two-dimensional tile with no block
scale. `T16` reads as "16x16 tile" and cannot be mistaken for linear grouping.

## 4. Components

### 4.1 `escha-ref` — CPU reference

A Rust port of `ref.py`: `cba_decode`, `decode_tile`, `reconstruct`, `h128`,
`input_transform`, `output_transform`, `expert_linear`, `fold_scales`, `swiglu`,
`w8a16`. Pure functions, no GPU and no hipfire dependencies.

One practical note from running it: `ref.moe_block` calls `expert_linear`
without a pre-decoded weight, so a literal port re-decodes the same expert once
per (token, slot) — 128 full tile decodes for an 8-token fixture. Memoize
`reconstruct` per (expert, projection) or the reference is unusably slow at
G4/G5 scale.

**This is the numerical oracle for every other component.** The Phase 1
Q8-resident build is explicitly *not* the oracle — it carries its own
quantization error. Two different artifacts, two different roles.

### 4.2 Converter — `hipfire-quantize`

Arch-detect on `quant_method in {escha, eschamoe}`, emitting `.hfq`. Every
tensor in the safetensors index is classified; there is no default-skip branch.

| source | destination |
|---|---|
| `*.escha_code` | `ESCHA2T16`/`ESCHA3T16` verbatim, K from `shape[-1] / 16` (§1.4) |
| `*.escha_{rin,rout,s_in,s_out}` | folded to one f32 pair per projection |
| `*.weight_int8` + `*.weight_scale` | `Q8_0`, row scale replicated (§4.2.1) |
| `*.bias` (27B only) | F16, new arch-5 bias slots |
| norms, `A_log`, `conv1d`, `dt_bias`, `in_proj_a/b`, `mlp.gate`, `shared_expert_gate` | F16/F32, as today |
| `mtp.*` (35B inline) / `mtp/` dir (27B) | existing `.mq4-mtp` trailer |

`escha_config`, when present, is read at both lengths (`[9]` MoE, `[6]` dense)
and cross-checked against `quantization_config.layer_meta` rather than trusted.
`K` comes from the code tensor's shape; `escha_config[1]`, `layer_meta.K` and
`layer_meta.bits` are cross-checks, and `bits` is one that is allowed to fail
(§1.3, §1.4).

#### 4.2.1 The int8 repack is lossless only if done one specific way

Escha's int8 is **per-output-row**: `w8 [O, K] int8` with `scale [O]` f16, and
`ref.py::w8a16` dequantizes as `f16(w8 * scale)`. hipfire's `Q8_0` is
**per-32-element block** (34 bytes per 32: 32 int8 plus one f16 scale).

Do not recompute per-block scales from the dequantized values — that is a second
quantization and adds avoidable error. Instead **replicate the row scale into
every block of that row** and pass the int8 bytes through unchanged. The
reconstruction is then bit-identical to Escha's, at a cost of 2 bytes per 32
elements (6.25% overhead) for scales that are all equal within a row.

Note also that a single logical tensor can straddle safetensors shards (the 27B's
`mlp.up_proj` has its `escha_code` in shard 2 while its metadata sits in shard 1),
so the converter resolves tensors through the index, never per-file.

### 4.3 Dispatch

`DType::Escha2T16` / `Escha3T16` and `RotationPlan::EschaH128`. Both types need
the guard that they can never fall through to `GemvVariant::Plain` — the rule
`coverage_tests.rs:521` already enforces for `MQ4G128`, where falling through
would double-rotate. Here it would *un*-rotate, which is worse: the output is
coherent-looking text rather than a crash.

### 4.4 Phase 1 kernels

- `escha_decode_tiles.hip` — one-shot expansion to `Q8_0` resident at load.
- `escha_h128_in.hip` / `escha_h128_out.hip` — the two activation transforms.
  The butterfly is reused from `gemv_mq4g128.hip`, which already pins the exact
  `0.0883883476f`; the sign-seed stage (43, 1043) is dropped, since escha folds
  its signs into `rin`/`rout`.

Everything downstream is untouched: `GemvQ8_0`, `FusedGateUpQ8_0` and
`FusedQkvzaQ8_0` already exist, and the last is exactly the shape the 27B's
`in_proj_qkv` / `in_proj_z` need.

**Why Q8 is a good intermediate.** The decoded weights live in the rotated
domain, where incoherence processing has already made them near-Gaussian and
outlier-free. Per-row Q8 is close to the best case for that distribution —
which is also why Escha themselves ship a Q8 repack path.

### 4.5 Phase 2 kernels

Fused decode+GEMV: hash decode inline, lane bit-extraction, both H128
transforms. MoE indexed variants for K=2 `gate_up` and K=3 `down`; dense
variants for the 27B including the QKVZA shapes. Rotations unchanged from
Phase 1, so the only new thing under test is the fusion.

The dense gate+up is the one place fusion does not come free: `gate_proj` is
K=2 and `up_proj` is K=3 (§1.1), so a fused dense gate+up kernel must either
run mixed-K or stay split. Decide by measurement, not up front.

**Exploiting the prune mask (MoE only).** Per §1.2 a large, per-expert fraction
of `gate_up`'s output channels is identically zero — 55% on layer-0 expert 0,
14–58% across the experts sampled. What that does and does not buy:

- **Not skippable:** the `gate_up` GEMV itself. `rout` is applied *after* the
  output H128, and the Hadamard mixes all 128 channels of a block, so producing
  the surviving channels still requires the full `mid` vector. Dropping GEMV
  columns for pruned outputs would corrupt their block-mates.
- **Skippable:** the final `rout` multiply and SwiGLU for pruned channels, and —
  the real prize — the corresponding **input rows of `down_proj`**. Those
  intermediate activations are exactly zero, so for layer-0 expert 0 the K=3
  `down_proj` GEMV can skip 280 of its 512 input rows. `down_proj` is the more
  expensive of the two (K=3, 3.0 bpw), so this lands on the dominant half of the
  expert.

The mask is static per expert, so it can be precomputed once at load into a
compacted row index rather than tested per token. Treat this as a Phase 2
optimization gated on measurement, not a Phase 1 requirement — and note the
correctness trap in §1.2: pruned outputs must stay *exactly* zero, not
approximately.

### 4.6 Loader + registry

`qwen3.6:35b-a3b-escha` (arch 6), `qwen3.8:27b-escha` (arch 5), quant label
`escha`.

## 5. Data flow — decode step

Per escha linear:

```
x -> *rin_eff -> H128 blockwise -> *RS -> round f16
  -> Q8_0 GEMV -> mid f32
  -> H128 -> *RS -> *rout_eff -> f16 -> (+bias, dense only)
```

For the MoE, the router, top-k and shared expert are *intended* to be untouched
existing arch-6 code; only the two expert projections change. Three rounding
points are load-bearing and easy to lose:

- SwiGLU runs on the **f16-rounded merged** `gate_up` output, gate first half.
- The expert combine multiplies by `f16(score)`, not the f32 score.
- **Router logits are rounded to f16 before top-k**: `ref.py` computes
  `f16(x @ gate_w.T)` and only then widens to f32 to select. Selecting on
  unrounded f32 logits is a different function.

**f16 logit ties are real, and they are the one place "untouched arch-6 router"
needs checking rather than assuming.** Rounding to f16 before top-k manufactures
exact ties that f32 would not produce, and they are not rare: in an 8-token
fixture, one token has two experts on identical logits (§7 G4b). Consequences:

- A tie *inside* the selected k is harmless — the combine is a sum over slots,
  so slot order does not change the output.
- A tie *at the k / k+1 boundary* changes the selected set and therefore the
  output. It is implementation-defined which expert wins, so hipfire and
  `escha-ref` may legitimately disagree on such a token.
- Therefore: if hipfire's arch-6 router does not round logits to f16 before
  top-k, it will diverge from Escha more often than tie-breaking alone explains.
  Verify this before assuming the router is reusable as-is.
- And when reading G5: a handful of divergent positions traceable to boundary
  ties is expected, not a codec bug. Attribute before investigating.

**Fusion and orientation.** `qwen35/weights.rs:69` documents
`experts[X].gate_up: [2*moe_intermediate, hidden]`, i.e. hipfire already stores
the MoE `gate_up` **fused and out-major** — matching Escha's single fused
`gate_up_proj`. But escha's tile grid is **in-major** (`[in/16, out/16, 16K]`),
so the decode kernel transposes on the way out. The dense 27B is the mirror
problem: `gate_proj` and `up_proj` are separate tensors with **different K**
(2 and 3), so they can only reach a fused gate+up kernel after Phase 1 has
normalized both to `Q8_0`. A Phase 2 fused gate+up for the dense model would
have to be mixed-K, or stay split.

## 6. Constraints

**No codebook LUT.** 65536 x f16 = 128 KB; gfx1151 has 64 KB LDS
(`profiler.rs` records `lds_per_cu: 65536`). Decode inline
instead — multiply, and, xor, two f16 unpacks, one f16 add. Five ops and no
memory traffic, which is what makes Phase 2 plausible at all.

**`lane_positions` is the trap.** It is a non-obvious permutation, and a wrong
one still yields a full-rank, plausible-looking weight matrix. Gate it directly
on golden vectors, never on end-to-end coherence.

**Every `ESCHA2T16` site needs an `ESCHA3T16` twin.** Same shape as the Neutrino
`FV5G256`/`FV5B256` rule. `weight_backend.rs::dequant_f32` is specifically where
Bonsai lost hours to a missing arm that surfaced only at e2e as a `token_embd`
panic.

## 7. Gates

- **G0** — `escha-ref` bit-exact against committed goldens: `packed_gu_e0_k2` ->
  `expected_gu_e0_k2` and `packed_down_e0_k3` -> `expected_down_e0_k3` exact;
  `w8a16` fixture. The MoE-block fixture is **not** a bit-exact gate — see
  below.
  **Already demonstrated in NumPy** (§1.1), so G0 is a port-fidelity gate on the
  Rust translation, not an open question about the format. The goldens are real
  shipped tensors, so passing G0 means decoding the actual model correctly.
- **G1** — converter `memcmp` round-trip on code streams; zero unclassified
  tensors in the index; and the §1.4 leaf contract enforced in both directions —
  a checkpoint with a transform vector removed must be rejected, and one with
  `s_in`/`s_out`/`config`/`bias` absent must convert cleanly.
- **G2** — GPU decode vs `escha-ref::reconstruct`, exact fp16 on every tile of a
  sampled expert set, **both K**.
- **G3** — H128 kernels vs `escha-ref::h128` directly. A round-trip check
  (`H128 . H128 = 128 I`) is **not** sufficient: a wrong butterfly order is also
  self-inverse, so it passes while being wrong.
- **G4** — single-expert `expert_linear` on GPU vs reference, then the MoE block
  against `moeblk_out`. Two things about this fixture, both measured:

  **It is a tolerance gate, not a bit-exact one.** Running `ref.moe_block`
  against it with the real shipped layer-0 weights gives `max|diff| = 1.22e-4`,
  `mean|diff| = 2.1e-6`, with 4752 of 16384 values differing at ULP level
  against outputs whose mean magnitude is 0.0185. The golden was evidently
  produced by the Metal path, not by `ref.py`, so "within fp16 rounding" is too
  vague to gate on. Use **`max|diff| <= 2e-4` and `mean|diff| <= 1e-5`**, and do
  not assert equality. The codec goldens in G0 *are* bit-exact; do not
  generalize this tolerance to them.

  **It does not gate the router.** The fixture ships `moeblk_ids.i64` and
  `moeblk_scores.f32` and injects them, bypassing selection entirely. Router
  correctness needs its own check — G4b below.

- **G4b** — router, gated separately since G4 cannot see it. Reproducing
  selection from `mlp.gate.weight` on the fixture's `x` gives the **identical
  top-8 set for all 8 tokens** and scores agreeing to 3e-8, confirming the
  contract. Assert the set and the scores, **not the order**: token 3 orders two
  experts differently because experts 65 and 43 have *exactly equal* f16 logits
  (both 1.80078). See §5 on why that is benign here and when it is not.
- **G5** — e2e coherence, then KLD on a fixed wikitext slice. Not on the model's
  own output: for ds4 that scored 8x better on the median and was optimistic.

  **The reference is `escha-ref` on CPU, not any Escha runtime.** None of
  Escha's three runtimes can run on this box: `escha-mlx` is Metal / Apple
  Silicon, the `escha` wheel is CUDA (sm_80–sm_120), and ZML requires an NVIDIA
  driver. There is no cross-checking against their engine on gfx1151, and a gate
  that cannot be executed proves nothing. This is not a downgrade: `ref.py`
  declares itself "the semantic contract for every Metal kernel in this package"
  and is itself gated on the goldens, so agreeing with `escha-ref` *is*
  agreeing with their runtime — and it is exact rather than cross-machine.
  Cost is CPU time; budget a few hundred positions, not thousands.

  Run a second KLD against the **bf16 parent** (`Qwen/Qwen3.6-35B-A3B`) as well.
  That one answers a different and independently useful question — whether
  Escha's 2-bit delivers what they claim — and it is the number to compare
  against the existing `qwen3.6:35b-a3b-mq2` measurement.
- **G6** (Phase 2) — fused GEMV compared against the Phase 1 path at Q8
  precision, then KLD against `escha-ref`.

## 8. Error handling

Refuse rather than guess:

- unknown `quant_method`, or `format_version != "2.0"` — hard error
- `escha_config` disagreeing with `layer_meta` — hard error. This is what
  catches a shape assumption silently breaking on a future Escha release.
- any tensor in the safetensors index left unclassified — hard error, not skip
- a coded linear missing `escha_code`, `escha_rin` or `escha_rout` — hard error
  ("incomplete escha linear"), never a partial decode (§1.4)
- an `escha_*` leaf outside the known six — hard error naming the leaf. This is
  the guard that stops a future `escha_rotation_theta` export from being decoded
  under the wrong rotation (§1.4).
- a missing `ESCHA3T16` dispatch arm must never resolve to `Plain`

Conversely, absence of `escha_s_in`, `escha_s_out`, `escha_config` or `bias` is
**not** an error — those are optional by contract and the fold/decode paths must
run without them.

## 9. Risks and expectations

**Phase 1 perf will be bad, and that is the expected outcome.** Worked through
at `Q8_0`'s 34 bytes per 32 elements:

| | 35B experts | 35B non-expert | 35B total |
|---|---|---|---|
| escha native | 9.4 GB | 2.3 GB | ~11.7 GB (published 12.3) |
| Phase 1 `Q8_0` | 34.2 GB | 2.5 GB | **~36.7 GB** |

MoE reads 8 of 256 experts per token, so per-token expert traffic goes from
**0.29 GB to 1.07 GB — 3.6x**. It will lose to `qwen3.6:35b-a3b-mq2` on tok/s.
The Phase 1 deliverable is correctness plus a servable artifact; speed is
Phase 2's job. 36.7 GB is comfortable on 128 GB but is not free — check it
against the standing heap baseline before assuming headroom.

The same arithmetic applied to the 27B gives ~10 GB native against a published
10.15 GB, and ~28 GB at `Q8_0`. That both reconstructions land within a few
percent of the published sizes is independent confirmation that the format model
in §1.1 is right — a wrong bpw or a missed tensor class would not close.

**The 27B is not benchmarked before Phase 2.** At ~28 GB dense, re-read every
token, a decode-at-load build is not worth timing. It still gets built and
gated for correctness through G4 — that is how the dense code path, the
different-K gate/up split and the new bias slots get exercised — but no
tok/s number is reported for it until fused kernels exist.

**The 27B needs a change outside the codec.** Its escha linears carry fp16
biases the base architecture does not have (§1.3), so `hipfire-arch-qwen35`'s
arch-5 path gains bias slots on `in_proj_qkv`, `in_proj_z`, `out_proj` and the
three MLP projections. This is the only work in the port that touches an
existing architecture, and it must not perturb the existing `qwen3.6:27b` SKUs
that share that path — those have no bias tensors and must keep taking the
no-bias branch.

**Kernel cache hygiene.** Mixed-toolchain blobs on gfx1151 produce attractor
garbage that will look exactly like a codec bug. Single-toolchain cache, hipcc
on PATH. See `hipfire_kernel_rebuild_gfx1151`.

**Golden coverage is thin** — one expert of one layer per K, plus one small MoE
block. Broader expert sampling is checked against `escha-ref`, not against
goldens.

**Format drift.** `format_version` is pinned and asserted; a future Escha
release that changes tile size or codebook constants must fail loudly at
conversion, not decode into garbage.

## 10. Phase 1 results (measured)

Artifact: `/data/hipfire-models/escha-35b.hfq`, sha256
`bd186b37037ee6c1cb58ce6a5c053b785e9ba30b7f1de04fda7ef4f4f06e3010`, 12.34 GB,
arch_id 6, registered as `qwen3.6:35b-a3b-escha`. Host gfx1151, ROCm 7.2.2.

### 10.1 Quality (G5)

Reproduce with `scripts/escha-kld.sh`.

| | value |
|---|---|
| corpus | `benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`, n_ctx 384, 6 chunks |
| scored positions | 1146, teacher-forced |
| **mean KLD vs weight-exact escha** | **0.00276 nats** (95% CI 0.0019–0.0039) |
| p99 KLD | 0.054 nats |
| reference PPL / NLL | 7.6651 / 2.036678 |
| candidate PPL / NLL | 7.6585 / 2.035821 |
| negative control (exact vs itself) | **0.000000** nats as printed (actual 2.1341e-10, p99 6.03e-09) |
| KLD vs the bf16 parent | **not run** — see §10.2 |

Near-zero, as §7 predicted, and it is the `Q8_0` intermediate rather than the
codec: the reference arm is the same forward with the experts stored
weight-exactly (`HIPFIRE_ESCHA_EXPERT_STORE=f16`, bit-identical to
`escha_ref::reconstruct`), so the only thing that differs between the arms is
the 8-bit re-quantisation. The p99 sits ~20x the mean, consistent with the
boundary-tie behaviour in §5 concentrating the error on a handful of positions
rather than spreading it; that shape is expected and is not a codec bug.

Three things about the method are load-bearing and were each got wrong once
before being fixed:

- **Teacher forcing is structural, not a flag.** `build_kld_ref_native` writes
  the token stream into the HFKLDR file and `eval_hipfire` reads the tokens
  from that file, so both arms are scored on one identical committed stream by
  construction. Nothing is scored on a model's own greedy output.
- **`--kv-mode f32` matters more than it looks.** The reference builder uses an
  unquantised F32 KV cache; `eval_hipfire` defaults to `asym3`. Leaving that
  default in place gives **0.018357** nats on the identical reference — 6.5x
  the real figure, and almost all of it is KV quantisation, not the codec.
- **The negative control is the thing that makes the number attributable.**
  Scoring the exact arm against its own reference prints 0.000000. The
  underlying float is not bit-zero — it is 2.1341004702939135e-10 (p99
  6.029504362788427e-09) — and re-running it against the same reference
  reproduces both figures BIT-FOR-BIT. That reproducibility is the point: a
  nondeterminism floor would not repeat. The residue is the fixed difference
  between two programs computing the same forward (`build_kld_ref_native`
  writes the reference, `eval_hipfire` scores it), and it is ~1.3e7 times
  below the 0.0027576 the production arm reports. `scripts/escha-kld.sh` now
  asserts both halves of that — the control rounds to 0.000000 at the printed
  precision, and it is at least 1e4x below the headline number — rather than
  leaving "must print exactly 0.000000" to a human reading stdout.

The reference could not be `escha_ref` driving a CPU forward: `escha_ref` is a
block-level oracle (codec, H128, `expert_linear`, `swiglu`) and this repo has
no CPU transformer, so a CPU reference would have meant hand-writing a
40-layer hybrid DeltaNet MoE forward and then trusting it more than the thing
under test. Storing escha's own decoded fp16 weights and reusing the gated GPU
forward keeps `escha_ref` as the authority for every step it actually defines
(G2 and G3 gate the decode and the H128 pair bit-exact against it).

### 10.2 The bf16 parent comparison was not run

`Qwen/Qwen3.6-35B-A3B` is not on this box in any form. `/data/hipfire-models`
has no safetensors copy, and the only cached artifact of the parent is
`unsloth/Qwen3.6-35B-A3B-GGUF: Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` — a 4-bit
quant, which is a peer rather than a reference. Fetching the parent is ~70 GB.
Skipped deliberately and left open rather than substituted with the Q4 GGUF,
which would have produced a number that reads like the answer and is not.

Consequence: **the "does Escha's 2-bit deliver" question is still open**, and
there is no comparable figure against `qwen3.6:35b-a3b-mq2` on quality. §10.1
prices the port, not the codec.

### 10.3 Speed and memory

All figures re-measured on the branch head (gfx1151, ROCm 7.2.2) with
`escha_prefill_bench` and `scripts/escha-gtt-probe.sh`. An earlier revision of
this section reported the state before `6547c78b6` (contiguous expert buffers)
and `06ab4db8e` (reachable batched prefill) and understated prefill by 4x; the
numbers below supersede it.

| | escha (Phase 1) | `qwen3.6:35b-a3b-mq4r` |
|---|---|---|
| decode | **40.0–40.2 tok/s** | 71.8 tok/s (MTP engaged, tau 1.52) |
| prefill, n=512 | **161.9 tok/s** | — |
| prefill, n=2048 | **165.0 tok/s** | — |
| prefill, short prompt | — | 289.6 tok/s |
| file on disk | 12.34 GB | 18.7 GB |
| resident (GTT) | **37.58 GB** | — |

Batched prefill IS admissible and IS taken: `escha_routed_prefill_indexed`
runs the routed half over `n_tokens * k` slots in one launch per (layer,
projection, side), which the bench confirms as 160 H128 launches for a
512-token prompt (a per-token fallback would be 81 920). Prefill is therefore
~4.1x decode, not equal to it.

The remaining prefill gap to `mq4r` is ~1.8x, and §10.5 says what closing it
would cost. The decode row is still not like-for-like — `mq4r` has its MTP
sidecar attached and escha has no speculator wired (and, per the MTP refusal
added on this branch, cannot have one until the H128 pair is taught to that
forward).

`qwen3.6:35b-a3b-mq2` remains the honest speed comparison and is still not on
this box, so the table uses `mq4r` from the same family; mq2 is 11.6 GB
against mq4r's 18.7 GB, reads strictly fewer expert bytes per token and takes
the same indexed GPU-top-K path, so it is at least as fast as the column
shown.

**Resident memory is 37.58 GB**, measured as an amdgpu GTT delta over a
3.36 GB idle baseline (40.94 GB peak). 34.2 GB of that is the Q8_0 routed
experts and ~3.3 GB is everything else.

It was **67.9 GB** until `6547c78b6`, and the cause was allocation
granularity, not the codec or the carrier. While each of the 20 480 per-expert
projections was its own device allocation, the HIP allocator's 2 MiB granule
rounded up every one of them:

| | logical | rounded to 2 MiB granules |
|---|---|---|
| gate_up `[1024, 2048]` Q8_0 | 2.125 MiB | 4 MiB |
| down `[2048, 512]` Q8_0 | 1.0625 MiB | 2 MiB |
| x 10240 experts | 31.9 GiB (34.2 GB) | 60 GiB (64.4 GB) |

64.4 GB of granules plus ~3.4 GB of everything else is 67.8 GB against 67.9 GB
observed — the arithmetic closed to 0.1%, which is why it was attributed to
granularity rather than to a leak, and why packing the experts into one buffer
per (layer, projection) recovered the predicted ~30 GB exactly. The logical
34.2 GB now IS the resident cost.

One consequence survives the fix: the weight-exact **F16** expert store is no
longer free. It was free at 67.9 GB, because F16's 4 MiB / 2 MiB projections
were exactly what the rounded Q8_0 already occupied. Packed, F16 is a genuine
2x on the expert half (~68 GB resident). That still fits on this box and is
what the §10.1 reference arm uses, but it is now a real cost rather than an
accounting artifact, and F32 (129 GB) still does not fit.

### 10.4 Coherence

Verbatim, greedy (`temperature: 0.0`), via `scripts/_coherence_runner.py`:

    prompt: /no_think What is the capital of France? Answer in one word.
    reasoning channel: (a short numbered plan, ending) 4. Final Output
      Generation: - Paris - Matches all constraints.
    answer channel: Paris

    prompt: /no_think Write three sentences describing a harbour at dawn.
    Pale gold light spills across the still water, painting the masts of
    sleeping boats in soft amber hues. A gentle mist clings to the wooden
    docks while the distant cry of a gull breaks the quiet morning air.
    Slowly, the first fishing trawler rumbles to life, its engine echoing
    through the quiet cove as the sun finally crests the horizon.

Attractor detector clean (unique-token ratio 0.617, max token frequency 0.055
over the first 128 tokens; no 3-gram flag). First run was already coherent —
no kernel-cache garbage, single-toolchain cache with hipcc on PATH.

One thing to know rather than to fix: the model ignores `/no_think` as a mode
switch and reasons anyway, so a short `max_tokens` truncates mid-think and the
daemon reports `open think span at end of generation (validation)`. That is a
budget artifact, not a decode fault — the same prompt with headroom closes the
span and emits a normal answer.

### 10.5 Limitations that remain open

Two, both measured, both required reading before this is treated as finished.

**(a) The routed half does not amortise in prefill.** Batched prefill amortises
the *launch* cost of the H128 transforms and the *activation* traffic, and that
is what took prefill from decode speed to ~163 tok/s. It does not amortise the
expert **weight** traffic: each token's top-8 experts are read for that token,
so a batch of `n` tokens still moves ~`n` x 34.2 GB / (n_layers x tokens)
worth of expert bytes — 34.2 GB of expert traffic per full pass over the batch,
the same total as the per-token route. That is why escha stays ~1.8x behind
`mq4r`'s 289.6 tok/s at prefill despite the batched body.

Closing it needs a Q8_0 grouped GEMM over **sorted expert groups**: sort the
`n * k` slots by expert, then run one GEMM per expert over all the tokens that
chose it, so each expert's weights are read once per batch instead of once per
token. That is the same Path-2 scatter/grouped structure the MQ dtypes already
use, and it is a real change rather than a port: grouping changes the order in
which a token's `k` expert contributions accumulate, so the result would no
longer be bit-identical to the per-token route. G4's
`batched prefill route vs per-token indexed route: 0 differing floats` — an
equality claim, not a tolerance — would have to be restated as a bound, and
the argument for the new bound would have to be made from scratch. Do not land
that work while leaving G4 asserting equality; the gate would go red and the
temptation would be to loosen it.

That grouped arm is also what would spring the prefill hole this branch closed
with `check_moe_prefill_supported`: it is the first generic Q8_0 routed prefill
arm, and without the guard an escha layer with `HIPFIRE_ESCHA_INDEXED=0` would
silently take it and drop both Hadamard transforms.

**(b) Prefill and decode select different experts, at 2.96% per expert slot.**
The design-time estimate in §5 was ~0.42% of routing decisions straddling an
f16 boundary. The measured figure is **8x that**, and — more importantly — the
dominant term is not the one the estimate was about.

`escha_prefill_batch_gate` records the `topk_indices` both routes actually
indexed with. Measured at n=64:

| | rate |
|---|---|
| expert-SLOT divergence, whole stack | **2.96–3.13%** |
| (token, layer) SET divergence, whole stack | 24.1% |
| (token, layer) SET divergence, layer 0 | 0.00% |

Layer 0 is 0% because both routes feed it the same embedding, and it is the
only layer that cannot have compounded. The whole-stack rate is higher because
a flip at layer L changes that token's hidden state for every later layer; the
per-layer series rises from 0% to a ~25-35% plateau, which is compounding, not
a defect.

The dominant contribution is an **f16 activation downcast**, not accumulation
reordering. The batched dense half converts its F32 activations to F16 for the
WMMA GEMMs where the per-token route runs F32 GEMVs; that downcast is a much
coarser perturbation of the router input than the last-bit differences of a
reordered f32 sum, and it is what pushes decisions across the f16 rounding
boundary that `router_logits_round_f16_rne` then quantises them to. Attributing
this to "accumulation reordering" — as the 0.42% estimate did — is why the
estimate was 8x low.

Consequences: escha's prefill and decode are NOT interchangeable for anything
that assumes identical routing (KV-cache reuse across a prefill/decode boundary
is fine; a cache of routed-expert outputs would not be). And the final-token
logit delta between the routes is `max 4.393e-1 / mean 7.160e-2` — two orders
of magnitude above what reordering alone gives, dominated by this divergence.
The argmax is stable, which is what G6 asserts, but the logits are not close.

Neither limitation is a blocker for Phase 1, whose claim is correctness plus a
servable artifact. Both are stated here because §10.3 reads like a success and
a reader who stops there would carry away two wrong beliefs.

### 10.6 Correctness gates (measured)

The six gates, their commands, and the results on the branch head. All are
re-runnable; G1-G4b need no model beyond the checkpoint and the committed
fixtures. Registered in `docs/VALIDATION.md` and `scripts/gates.sh`.

| Gate | What it asserts | Result |
|---|---|---|
| **G1** | every `escha_code` tensor in the `.hfq` is byte-identical to the source safetensors | **80/80 byte-identical** (40 layers x 2 projections; count asserted against `model.safetensors.index.json`, not hardcoded) |
| **G2** | GPU tile decode == `escha_ref::reconstruct` exactly, in fp16 | **bit-exact**, 0 mismatched, at both golden shapes and at production shapes of 89 128 960 elements (K=2 and K=3) |
| **G3** | the H128 pair == `escha_ref`, all launch forms | **bit-exact**, 0 mismatched: `h128_in`, `h128_out`, batched broadcast / per-slot / grouped input mappings, `out_batched`, and swiglu |
| **G4b** | arch-6 router selects the same experts as escha's reference routing | **0/8 tokens with a differing top-8 set** |
| **G4** | the whole MoE block against escha's `moeblk_out.f16` golden | F32 (weight-exact) arm **max 1.828e-4 / mean 9.673e-6**; Q8_0 (production) arm **max 2.633e-4 / mean 3.027e-5**, against a golden mean magnitude of 1.85e-2. Indexed route vs host route: **0 differing floats**. Batched prefill route vs per-token indexed route: **0 differing floats**. Q8_0 re-quantisation cost, arm2 - arm1: max 1.687e-4 / mean 2.864e-5 |
| **G5** | KLD against the weight-exact escha arm on a fixed teacher-forced corpus | **0.0027576 nats** (95% CI 0.0019491-0.0038610), p99 0.054, **PPL 7.6585**; negative control **2.13e-10 (prints 0.000000)** and bit-reproducible across runs — see §10.1 |
| **G6** | batched prefill vs the per-token route, whole model | **argmax stable** (135744 on both arms); max\|delta\| 4.393e-1, mean\|delta\| 7.160e-2, 0 non-finite logits. Both deltas are now asserted, not printed — see §10.5(b) for why they are this large |

The two `0 differing floats` rows in G4 are EQUALITY claims, not tolerances,
and are only true because every kernel in the escha routed pipeline is purely
slot-parallel: slot `s` performs the same FLOPs in the same order whether the
launch carried 8 slots or 2048. §10.5(a) explains what would break them.

G4's F32 arm is the codec+transform accuracy; its Q8_0 arm is that plus the
8-bit expert store. The gap between them (max 1.687e-4) is the price of the
Q8_0 carrier at block level, and G5's 0.0027576 nats is the same price
end-to-end over 40 layers.

## 11. Out of scope

- Reproducing Escha's quantization or recovery fine-tune. We consume their
  weights; we do not build a quantizer.
- The vision tower. Declared in both configs, no weights shipped, in the
  quantization `ignore` list. Text-only.
- Transcoding escha into MQ formats. Established impossible-to-do-losslessly in
  §2; a lossy MQ transcode would discard exactly the quality that motivates the
  port.
- MTP-driven speculative decode. The head is carried through conversion so it is
  available, but wiring it is separate work.

## 12. References

- `EschaLabs/escha-mlx` (Apache-2.0) — `escha_mlx/ref.py` is the format
  contract; `tests/data/codec/` holds the golden vectors.
- `EschaLabs/Qwen3.6-35B-A3B-Escha-W2`, `EschaLabs/Qwen3.8-27B-Escha-W2`.
- `docs/architecture-ids.md` — arch 5 and 6 are `hipfire-arch-qwen35`.
- `kernels/src/gemv_mq4g128.hip` — existing 128-point FWHT butterfly.
