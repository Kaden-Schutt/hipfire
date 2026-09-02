# Escha-W2 port — design

**Status:** design approved, plan pending
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

**Why ids 42/43.** On `origin/master` the enum tops out at `MFP2G32E8 = 37`.
Ids 23/25/26/27 are documented do-not-reuse reservations. Ids 38/39 are claimed
by the unmerged `feat/batched-attn-impl` (`MQ2G256GL`/`MQ3G256GL`) and 40/41 by
the unmerged `nw_neutrino_fv5` (`FV5G256`/`FV5B256`). 42/43 is the next clear
pair, taken the same way Neutrino skipped past 38/39.

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

For the MoE, the router, top-k and shared expert are untouched existing arch-6
code; only the two expert projections change. Two rounding points are
load-bearing and easy to lose:

- SwiGLU runs on the **f16-rounded merged** `gate_up` output, gate first half.
- The expert combine multiplies by `f16(score)`, not the f32 score.

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
  `w8a16` fixture; `moeblk_x` -> `moeblk_out` within fp16 rounding.
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
- **G4** — single-expert `expert_linear` on GPU vs reference, then the full MoE
  block against `moeblk_out`.
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

## 10. Out of scope

- Reproducing Escha's quantization or recovery fine-tune. We consume their
  weights; we do not build a quantizer.
- The vision tower. Declared in both configs, no weights shipped, in the
  quantization `ignore` list. Text-only.
- Transcoding escha into MQ formats. Established impossible-to-do-losslessly in
  §2; a lossy MQ transcode would discard exactly the quality that motivates the
  port.
- MTP-driven speculative decode. The head is carried through conversion so it is
  available, but wiring it is separate work.

## 11. References

- `EschaLabs/escha-mlx` (Apache-2.0) — `escha_mlx/ref.py` is the format
  contract; `tests/data/codec/` holds the golden vectors.
- `EschaLabs/Qwen3.6-35B-A3B-Escha-W2`, `EschaLabs/Qwen3.8-27B-Escha-W2`.
- `docs/architecture-ids.md` — arch 5 and 6 are `hipfire-arch-qwen35`.
- `kernels/src/gemv_mq4g128.hip` — existing 128-point FWHT butterfly.
