# Gemma 4 Forward-as-Pipeline Migration Plan

**Date:** 2026-06-09  
**Status:** Planning (rev 2 — incorporates adversarial review findings)  
**Depends on:** `feat/dispatch-unification-gemma4` tip (`d42771da`), upstream `integration/dispatch-unification` merged  
**Blocks:** None — this is a progressive migration behind a feature flag  
**Reviews folded in:** `gemma4_forward_as_pipeline_plan_rev_claude.mde` (B1–B4, M1–M3, N1–N9, P1–P4) and `gemma4_forward_as_pipeline_plan_rev_gemini.md` (REV-01…REV-06). The most load-bearing corrections: Full layers have **no `v_proj`** and use **separate** Q/K projections (not a fused QKV + standalone V); the post-attention norm must run **exactly once**; the MoE residual op is **identical** to the dense path; and graph-capture is **not** a reason to gate off the lowered path.

## Goal

Migrate Gemma 4's decode forward from `execute_steps` (per-token resolution +
fusion matching) to the upstream **forward-as-pipeline** (#397 Ship 6)
lowered-super-op substrate. The result: a pre-resolved `LayerProgram` per
layer, executed via `run_layer_program` + a `ForwardBindings` impl. No perf
regression, byte-identical output, default-off until validated.

## Background

### How the upstream pattern works

Each migrated arch follows the same 3-step pattern:

1. **Load time:** `lower_variant(layer_type)` returns a `LayerProgram` — a
   `Vec<SuperOp>` where each `SuperOp` carries a `SuperOpKind` (Proj / Attend /
   Moe / Norm / ResidualGemv / Conv / Recurrent / Escape) plus an
   `OpBinding` with an arch-local opcode in `weights[0]`. No `GpuTensor`
   borrows — pure POD.

2. **Decode time:** For each layer, construct a `FooBindings<'a>` struct that
   borrows the live layer weights, scratch, KV cache, config, and position.
   Call `run_layer_program(gpu, ctx, &program, &mut bindings)`.

3. **Feature gate:** `HIPFIRE_FORWARD_LOWERED` env var (default off initially,
   flipped to default-on after byte-parity validation). When off, the existing
   hand-path `execute_steps` runs unchanged.

### Reference implementations

Counts below re-verified against source (rev 2 — the rev 1 numbers were wrong):

- **qwen35** (`qwen35.rs:12555` `lower_variant`): 4 variants (DeltaNet / FullAttn ×
  dense/MoE), **10** opcode constants in `q35_op`, `ForwardBindings` impl overrides
  **10** methods (the 8 core + `run_moe_ep` + `ep_add_into_residual`). This is the
  canonical reference; default-ON since 2026-06-07. `Qwen35Bindings` borrows 12 fields.
- **LFM2** (`lfm2.../forward.rs:551` variants): 4 variants (Conv/Attn × dense/MoE),
  **2** explicit opcodes (`DENSE_GATE_UP`, `DENSE_DOWN`); Conv/Attend reuse code 0.
- **MiniMax** (`forward.rs:723`): single uniform shape — a 2-op program
  `[Attend, Moe]`; reuses the qwen35 pattern with EP.
- **DeepSeek V4** (`forward.rs:2078`): also a 2-op `[Attend, Moe]` program. **Does NOT
  use Escape** — `run_escape` returns `Err`; the MLA compressor / indexer / sparse-SWA
  are bundled *inside the `Attend` handler*. (Do not model Gemma's softcap-Escape on a
  DeepSeek precedent that doesn't exist; the softcap-as-Escape stands on its own merits
  — `EscapeKind::GemmaLogitSoftcap` is real.)

The opcode is carried in `OpBinding.weights[0]` as `WeightSlot(code)` (the fleet
convention; see `qwen35.rs:12545` and its doc-comment at `:12497`). `SuperOpKind` routes
to the `ForwardBindings` method; the opcode disambiguates *which* op of that kind.

### Already migrated arches (default ON)

qwen35, MiniMax, LFM2, DeepSeek V4. Gemma 4 is the only served arch not yet
migrated.

## Gemma 4 layer structure

Gemma 4 has two layer types (sliding + full), with an optional MoE branch on
every layer (26B-A4B variant). Each layer has sandwich norms and per-head
q/k/v normalization.

### Sliding layer (25/30 on 26B, all on 12B)

```
residual = x
x = input_layernorm(x)
q = q_proj(x)                    ← Proj
k = k_proj(x)                    ← Proj
v = v_proj(x)                    ← Proj (separate, not fused into QKV)
q = q_norm(q)                    ← fused into Attend prep (learned weight)
k = k_norm(k)                    ← fused into Attend prep (learned weight)
v = v_norm(v)                    ← fused into Attend prep; NO learned weight —
                                   uses scratch.v_norm_ones_full (no-scale RMSNorm,
                                   divide only). True on BOTH sliding and full.
q *= sqrt(head_dim)              ← fused into Attend prep
rope(q, k)                       ← fused into Attend prep
kv_write + flash_attn(q,k,v)     ← Attend (window=1024, q8 ring-buffer)
attn_out = o_proj(attn_out)      ← Proj
attn_out = post_attn_norm(attn_out)
x = residual + attn_out
residual = x
x = pre_ffn_norm(x)
gate = gate_proj(x)              ← Proj
up   = up_proj(x)                ← Proj
hidden = gelu_tanh(gate) * up
ffn_out = down_proj(hidden)      ← Proj (or ResidualGemv if fused)
[MoE branch if present]          ← Moe
post_ffn_norm(ffn_out)           ← Norm
x = residual + ffn_out
x *= layer_scalar                ← (inline, not a super-op)
```

### Full layer (5/30 on 26B, none on 12B)

Same structure but:
- `head_dim = 512` (vs 256 for sliding)
- `n_kv = 1` (vs 8 for sliding)
- No sliding window (full causal attention)
- `k_eq_v = true` (V is a copy of K, weightless V-RMSNorm prelude)
- `partial_rotary_factor = 0.5` (only first 256 of 512 dims rotate)
- Uses `rope_partial_halved_f32` instead of `rope_f32`

### MoE branch (26B-A4B only, on every layer)

The dense FFN and the MoE branch are **two branches off the same post-attention
residual** (parallel in *dataflow*), but they execute **sequentially**: the dense FFN
finishes and writes `ffn_out`, then `apply_moe_branch` consumes it as `cur_mlp` and
combines. Do **not** read "parallel" as "concurrent streams" — it's one ordered CPU
dispatch sequence. The whole thing is encapsulated in the existing `apply_moe_branch`
(`gemma4.rs:1250`), which `run_moe(MOE_BRANCH)` delegates to:
```
cur_mlp = post_ffn_norm_1(ffn_out)          ← consumes the dense FFN output
pre2 = pre_ffn_norm_2(residual)
router_logits = router(pre2)
topk = softmax_topk(router_logits, k=8)
expert_gate_up[topk] = gate_up_proj[expert](pre2)
hidden = gelu_tanh(gate) * up
moe_out = down_proj[expert](hidden)
moe_out = post_ffn_norm_2(moe_out)
combined = cur_mlp + moe_out                ← gemma4.rs:1501
tmp = post_feedforward_layernorm(combined)  ← gemma4.rs:1505  (the outer norm, folded in)
```
On exit `tmp` already holds the normalized combined output, so the layer-level residual
is just `x = residual + tmp; x *= layer_scalar` — **byte-identical to the dense
`ResidualGemv(POST_FFN)`**. No MoE-specific residual branch is needed (this corrects the
rev-1 "adjust RESID_POST_FFN" item). On MoE layers the standalone `Norm(POST_FFN)` is
**replaced** by `Moe(MOE_BRANCH)`, not emitted alongside it.

### Logit softcap (output stage, not per-layer)

```
logits = lm_head(norm(x))          ← Proj
logits = tanh(logits / cap) * cap  ← Escape(GemmaLogitSoftcap)
```

## Design

### Variant enum

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gemma4Variant {
    SlidingDense,    // 12B + 26B sliding layers without MoE
    SlidingMoe,      // 26B sliding layers with MoE
    FullDense,       // 26B full-attention layers without MoE
    FullMoe,         // 26B full-attention layers with MoE
}
```

### Opcodes

**Projection opcodes are split sliding vs full** because the two paths differ in both
fusion and the presence of `v_proj` (see "Gemma 4 layer structure"). Sliding fuses
q+k through `execute_steps` and has a standalone `v_proj`; full issues **separate**
`weight_gemv` for q and k and has **no `v_proj`** (V is a copy of pre-norm K).

```rust
mod g4_op {
    // Proj (projection clusters) — variant-specific
    pub const PROJ_QK_SLIDING: u32 = 0; // sliding: q_proj + k_proj fused (execute_steps)
    pub const PROJ_V_SLIDING:  u32 = 1; // sliding: v_proj standalone (weight_gemv)
    pub const PROJ_Q_FULL:     u32 = 2; // full: q_proj standalone (weight_gemv)
    pub const PROJ_K_FULL:     u32 = 3; // full: k_proj standalone (weight_gemv); V←K copied in Attend(FULL)
    pub const PROJ_O:          u32 = 4; // o_proj (execute_steps on sliding, weight_gemv on full)
    pub const PROJ_GATE_UP:    u32 = 5; // gate_proj + up_proj (fused)
    pub const PROJ_DOWN:       u32 = 6; // down_proj (folds gelu_tanh + mul — see below)
    // NOTE: no PROJ_LM_HEAD — the output stage (final norm + lm_head + softcap) runs
    // OUTSIDE run_layer_program, same as qwen35. Do not place it in any LayerProgram.

    // Attend
    pub const ATTEND_SLIDING: u32 = 0;  // window=1024, head_dim=256, full rope, n_kv=8
    pub const ATTEND_FULL: u32 = 1;     // window=0, head_dim=512, partial rope, n_kv=1,
                                        // k_eq_v: first action copies V←K BEFORE k_norm

    // Norm
    pub const NORM_INPUT: u32 = 0;      // input_layernorm
    pub const NORM_POST_ATTN: u32 = 1;  // post_attention_layernorm (owns the rmsnorm)
    pub const NORM_PRE_FFN: u32 = 2;    // pre_feedforward_layernorm
    pub const NORM_POST_FFN: u32 = 3;   // post_feedforward_layernorm (DENSE layers only)
    // NOTE: no NORM_FINAL — final norm is part of the output stage (run separately).

    // ResidualGemv
    pub const RESID_POST_ATTN: u32 = 0; // residual plumbing only (NO norm — see B4/Step 4)
    pub const RESID_POST_FFN:  u32 = 1; // residual + scale(layer_scalar); IDENTICAL dense vs MoE

    // Moe
    pub const MOE_BRANCH: u32 = 0;      // sandwiched MoE branch (encapsulates both sandwich
                                        // norms + cur_mlp/cur_moe combine + outer norm)

    // Escape
    // Uses EscapeKind::GemmaLogitSoftcap (already defined in superop.rs)
}
```

**Variant-aware dispatch (important):** `SuperOpKind::Proj` + opcode is not always
enough — `o_proj` is `execute_steps` on sliding but `weight_gemv` on full. Each
`run_proj` / `run_attend` arm therefore branches on `(opcode, layer_variant)`, reading
the `LayerWeights::Sliding | Full` discriminant from `self.layer`. The split opcodes
above make most of this explicit; `PROJ_O` is the one that still inspects the variant.

### Layer programs

`Proj(DOWN)` folds `gelu_tanh + mul` (the activation) — there is no standalone
activation super-op. `Norm(POST_ATTN)` owns the post-attention rmsnorm; the following
`ResidualGemv(POST_ATTN)` does **only** residual plumbing (no second norm — see B4).

```
SlidingDense:
  Norm(INPUT)               — rmsnorm → tmp
  Proj(QK_SLIDING)          — q_proj + k_proj via execute_steps
  Proj(V_SLIDING)           — v_proj via weight_gemv
  Attend(SLIDING)           — q/k/v_norm(no-scale v) + scale_q + rope + kv_write + flash_attn → tmp
  Proj(O)                   — o_proj via execute_steps → tmp
  Norm(POST_ATTN)           — rmsnorm tmp → tmp   (the ONLY post-attn norm)
  ResidualGemv(POST_ATTN)   — x←residual; x+=tmp; residual←x   (NO norm here)
  Norm(PRE_FFN)             — rmsnorm x → tmp
  Proj(GATE_UP)             — gate + up via execute_steps
  Proj(DOWN)                — gelu_tanh(gate)*up, then down_proj → ffn_out (activation folded in)
  Norm(POST_FFN)            — rmsnorm ffn_out → tmp   (DENSE only)
  ResidualGemv(POST_FFN)    — x←residual; x+=tmp; x*=layer_scalar

SlidingMoe: (identical to SlidingDense through PROJ_DOWN, then:)
  [... SlidingDense through Proj(DOWN) ...]
  Moe(MOE_BRANCH)           — encapsulates post_ffn_norm_1, router+topk, expert gate_up/down,
                              post_ffn_norm_2, cur_mlp+cur_moe combine, AND the outer post_ffn
                              norm → leaves tmp = post_ffn_norm(cur_mlp+cur_moe).
                              (Replaces the standalone Norm(POST_FFN) — do NOT emit both.)
  ResidualGemv(POST_FFN)    — x←residual; x+=tmp; x*=layer_scalar   (IDENTICAL to dense path)

FullDense:
  Norm(INPUT)
  Proj(Q_FULL)              — q_proj via weight_gemv
  Proj(K_FULL)              — k_proj via weight_gemv      (NO Proj(V): full layers have no v_proj)
  Attend(FULL)              — copy V←K (pre-k_norm!) FIRST, then q/k/v_norm + partial rope
                              + kv_write + flash_attn → tmp
  Proj(O)                   — o_proj via weight_gemv
  Norm(POST_ATTN)
  ResidualGemv(POST_ATTN)
  Norm(PRE_FFN)
  Proj(GATE_UP)
  Proj(DOWN)                — gelu_tanh + mul folded in
  Norm(POST_FFN)            — DENSE only
  ResidualGemv(POST_FFN)

FullMoe:
  [... FullDense through Proj(DOWN) ...]
  Moe(MOE_BRANCH)           — replaces standalone Norm(POST_FFN), as in SlidingMoe
  ResidualGemv(POST_FFN)
```

**Note on `gelu_tanh + mul`:** This isn't a norm — it's an activation. The
qwen35 pattern folds this into `ResidualGemv(RESID_DOWN_SWIGLU)` via
`weight_gemv_swiglu_residual`. For Gemma 4, the activation is `gelu_tanh`
(not `silu`), so we can't reuse that helper directly. Options:

- (A) Add a standalone `Act` super-op (but `SuperOpKind` doesn't have one —
  it uses `OpFlavor::Act(GeluTanhMul)` on the gate_up Proj/residual).
- (B) Fold it into the Proj(DOWN) as a combined op that does gelu_tanh + mul +
  down_proj in one handler. This is what the hand path does (separate kernel
  launches).
- (C) Use `Norm` as a "misc elementwise" opcode, which is what qwen35 does for
  some of its non-norm steps.

**Recommendation: (B)** — fold gelu_tanh + mul into the Proj(DOWN) handler.
The handler calls `gpu.gelu_tanh_f32` + `gpu.mul_f32` + the down_proj GEMV,
mirroring the hand path exactly (`gemma4.rs:2149-2155`). No new SuperOpKind needed. Make
sure the activation is applied **only** here, not also as a standalone step — the rev-1
`Norm(NONE)` activation row has been removed from the programs to avoid double-applying.

For a later self-describing pass, note `ActFlavor::GeluTanhMul` **already exists**
(`superop.rs:70`): the DOWN `OpBinding` can carry `OpFlavor::Act(GeluTanhMul)` instead of
an implicit fold. Not required for v1 (review M3).

### Bindings struct

```rust
struct Gemma4Bindings<'a> {
    layer: &'a LayerWeights,          // enum { Sliding(SlidingLayerWeights), Full(FullLayerWeights) }
    config: &'a Gemma4Config,
    scratch: &'a Gemma4Scratch,
    kv_sliding: &'a mut KvCache,
    kv_full: &'a mut KvCache,
    pos: usize,
    sliding_kv_idx: usize,
    full_kv_idx: usize,
}
```

### ForwardBindings impl

Each method matches on the opcode and dispatches to existing helper functions:

| Method | Opcodes | Delegates to |
|--------|---------|-------------|
| `run_proj` | QK_SLIDING, V_SLIDING, Q_FULL, K_FULL, O, GATE_UP, DOWN | `execute_steps` (QK_SLIDING, GATE_UP, sliding O), `weight_gemv` (V_SLIDING, Q_FULL, K_FULL, full O, DOWN). DOWN folds `gelu_tanh + mul` before the GEMV. |
| `run_attend` | SLIDING, FULL | `kv_cache_write_*` + `attention_flash_*_window` sequence factored from `sliding_layer_decode_impl` / `full_layer_decode_impl`. **FULL first copies `V←K` (pre-k_norm), then applies q/k/v_norm.** |
| `run_norm` | INPUT, POST_ATTN, PRE_FFN, POST_FFN | `gpu.rmsnorm_f32` (POST_FFN dense layers only) |
| `run_residual_gemv` | POST_ATTN, POST_FFN | `memcpy_dtod` + `add_inplace_f32` (+ `scale_f32` for POST_FFN). **No norm inside.** POST_FFN is identical for dense and MoE. |
| `run_moe` | MOE_BRANCH | `apply_moe_branch` (existing) — leaves `tmp = post_ffn_norm(cur_mlp+cur_moe)` |
| `run_escape` | GemmaLogitSoftcap | `gpu.logit_softcap_f32` (output stage only) |
| `run_recurrent` | — | `Err(Unsupported)` |
| `run_conv` | — | `Err(Unsupported)` |
| `run_moe_ep`, `ep_add_into_residual` | — | default `Err` (EP out of scope) |

The bindings struct must borrow the full `&Gemma4Scratch` (it needs
`v_norm_ones_full` for the no-scale v-norm on both layer types, plus all `moe_*`
buffers), not a hand-picked subset.

### Output stage

The final norm + lm_head + softcap is NOT part of any layer's program. It
runs after the layer loop (same as qwen35). For the lowered path, this stays
as direct `execute_steps` calls outside `run_layer_program`, or becomes a
separate "output program" executed once:

```rust
// After the per-layer loop:
let ctx = DispatchCtx::new(gpu);
gpu.rmsnorm_f32(&scratch.x, &weights.final_norm, &scratch.tmp, config.norm_eps)?;
let wr = weights.lm_head.dispatch_ref();
execute_steps(gpu, &ctx, &[Step::Gemv { w: &wr, input: GemvInput::Raw(&scratch.tmp), out: &scratch.logits }])?;
if config.final_logit_softcapping > 0.0 {
    gpu.logit_softcap_f32(&scratch.logits, config.vocab_size, config.final_logit_softcapping)?;
}
```

## Implementation steps

### Step 1 — Scaffold (low risk, no behavior change)

Add to `crates/hipfire-arch-gemma4/src/gemma4.rs`:

1. `Gemma4Variant` enum and `g4_op` module with opcode constants.
2. `lower_variant(v: Gemma4Variant) -> LayerProgram` (pure, unit-testable).
3. `Gemma4Bindings<'a>` struct.
4. `impl ForwardBindings for Gemma4Bindings` — all methods return
   `Err(Unsupported)` initially.
5. `forward_lowered_enabled()` OnceLock gate (default OFF).
6. Gate in **`forward_scratch_inner`** (gemma4's per-layer loop;
   `gemma4.rs:1807` — *not* `forward_scratch_layers`, which is qwen35's name):
   if `forward_lowered_enabled()`, route to `forward_scratch_inner_lowered`.
   Do **not** add a graph-capture condition — qwen35's gate is
   `forward_lowered_enabled() && hidden_rb.is_none()`, where `hidden_rb` is the
   **spec-decode** ring buffer, not graph capture (review N7 / Gemini REV-04).
   Gemma 4 has no equivalent spec-decode capture path, so there is nothing extra to
   gate on. (Graph capture is hardwired off in gemma4 today anyway —
   `!false /* graph-capture-not-wired */`.)

**Validation:** `cargo test` passes, existing behavior unchanged (gate off).

### Step 2 — Wire up Norm + Proj (decode sanity)

Implement `run_norm` and `run_proj`:

- `run_norm`: match opcode → `gpu.rmsnorm_f32` with the correct weight tensor.
- `run_proj`: match opcode → `execute_steps` or `weight_gemv` with the
  correct weight tensor and scratch buffers.

Temporarily leave `run_attend` and `run_moe` as `Err(Unsupported)` so only
dense 12B (no MoE, no full layers) can partially run.

**Validation:** 12B model produces partial output (will error at Attend).
Verify the norm + proj stages produce correct intermediate values via
diagnostic dumps.

### Step 3 — Wire up Attend

Implement `run_attend` for both SLIDING and FULL:

Factor out the attention-prep + kv_write + flash-attn sequence from
`sliding_layer_decode_impl` and `full_layer_decode_impl` into shared helpers
that the bindings can call. The key differences:

| | Sliding | Full |
|---|---------|------|
| head_dim | 256 | 512 |
| n_kv | 8 | 1 |
| rope | full `rope_f32` | `rope_partial_halved_f32` |
| v_norm | `v_norm_ones_full` | k_eq_v (copy K, weightless RMSNorm) |
| window | 1024 | 0 (full causal) |
| cache | kv_sliding (q8 ring-buffer) | kv_full (asym3) |

**Validation:** 12B model produces byte-identical output to hand path
(`HIPFIRE_FORWARD_LOWERED=1`, compare logits). **But 12B is all-Sliding — it does NOT
exercise `Attend(FULL)`.** Add a 26B-A4B run with `HIPFIRE_MOE_BYPASS=1` (dense-only)
at long context (>1024 tokens, so the sliding window engages) to validate the Full
path — especially the `V←K` pre-k_norm copy ordering — *here*, the moment Attend lands,
rather than discovering a Full-path mismatch tangled with MoE three steps later. The
pre-k_norm V capture is the #1 parity risk: copying V *after* k_norm silently corrupts
V and only shows up at long context.

### Step 4 — Wire up ResidualGemv

Implement `run_residual_gemv`. **Neither arm applies a norm** — the post-attn norm is
owned by `Norm(POST_ATTN)` and the post-FFN norm by `Norm(POST_FFN)` (dense) or
`Moe(MOE_BRANCH)` (MoE). Running the norm here too would double-normalize (rev-1 bug;
review B4 / Gemini REV-02).

- `RESID_POST_ATTN`: `x ← residual`; `x += tmp`; `residual ← x` (the re-save seeds the
  FFN/MoE residual stream). Input `tmp` already holds `post_attn_norm(o_proj(...))`.
- `RESID_POST_FFN`: `x ← residual`; `x += tmp`; `x *= layer_scalar`. **Identical for
  dense and MoE** — `tmp` is already the normalized FFN-or-(MoE-combined) output.

This is the trickiest part because Gemma 4's sandwich-norm residual pattern has more
save/restore points than qwen35's simpler residual. The sequence must exactly match the
hand path (`gemma4.rs:2115-2131` post-attn, `:2176-2184` post-FFN).

**Validation:** 12B model byte-identical at all positions.

### Step 5 — Wire up Moe (26B-A4B)

Implement `run_moe`:

- `MOE_BRANCH`: delegate to `apply_moe_branch` (existing helper). It internally does
  both sandwich norms, router+topk, expert gate_up/down, the `cur_mlp + cur_moe`
  combine, and the outer post-FFN norm, leaving `tmp` ready for the residual add.
- Ensure the `SlidingMoe`/`FullMoe` programs emit `Moe(MOE_BRANCH)` **instead of** the
  standalone `Norm(POST_FFN)` (not in addition to it).
- **No change to `RESID_POST_FFN`** — it is byte-identical to the dense path (review
  M2 / Gemini REV-03). The rev-1 "adjust for MoE" item is dropped.

**Validation:** 26B-A4B model byte-identical with `HIPFIRE_FORWARD_LOWERED=1`. Because
26B is where the Full layers AND the MoE branch both first appear, this is the first
run that exercises `Attend(FULL)`, the `V←K` ordering, and `run_moe` — keep diagnostic
dumps on for the 5 Full layers and at least one MoE layer.

### Step 6 — Wire up Escape (logit softcap)

Implement `run_escape` for `EscapeKind::GemmaLogitSoftcap`:

```rust
EscapeKind::GemmaLogitSoftcap => {
    gpu.logit_softcap_f32(&scratch.logits, config.vocab_size, config.final_logit_softcapping)?;
    Ok(())
}
```

**Validation:** Output-stage softcap matches hand path.

### Step 7 — Byte-parity gate

Run both paths side-by-side across:
- 12B at short context (256 tokens) and long context (1200 tokens)
- 26B-A4B at short and long context
- Multiple temperatures (0.0 greedy, 0.3, 0.7)

**Design invariant — bit-exact, not "epsilon" (review P1).** The lowered path must
issue the *same kernels in the same order* as the hand path, so logits should be
**bit-identical**, not merely close. qwen35 hit true byte-identical parity this way.
"Within epsilon" is not an acceptable standard: under greedy decode a sub-epsilon logit
difference can flip an argmax → different token → cascading divergence
(cf. `feedback_attention_precision`: ~5% attn error → attractor within ~10 tokens).

For each config, verify:
1. Logits are **bit-identical** (mechanism gate)
2. Generated token sequences match exactly (symptom gate)
3. No panics, no NaN

**Gate criterion:** 5 consecutive runs with identical prompts produce identical token
sequences across both paths, AND the change passes the **mandatory**
`./scripts/coherence-gate.sh` (required by CLAUDE.md for any dispatch/forward-pass
change; the pre-commit hook runs it when relevant files are staged — review P3). The
bespoke logit-diff harness here is a *supplement* to the coherence gate, not a
replacement for it.

### Step 8 — Flip default ON

After byte-parity validation:
- Change `forward_lowered_enabled()` to default ON (same as qwen35:
  `std::env::var("HIPFIRE_FORWARD_LOWERED").ok().as_deref() != Some("0")`)
- Escape hatch: `HIPFIRE_FORWARD_LOWERED=0` to force legacy path

### Step 9 — Remove hand-path duplication (optional, follow-up)

Once the lowered path is default-ON and fleet-validated:
- Remove the hand-path arms from `forward_scratch_inner`
- Keep `HIPFIRE_FORWARD_LOWERED=0` escape hatch for one release cycle
- Remove the hand path entirely in the next release

## Non-goals (deferred)

- **EP (expert parallelism):** Gemma 4 26B-A4B MoE EP is not in scope. The
  `run_moe_ep` and `ep_add_into_residual` defaults (Err) are correct.
- **Prefill migration:** The prefill path (`forward_prefill_batch_v1/v2`)
  stays on `execute_steps`. Prefill ingests many tokens at once → it is
  **throughput-dominated** (compute-bound); decode is the latency-dominated,
  bandwidth-bound one. Deferral is still correct, but for the right reason: the
  per-resolve overhead amortizes over many tokens, so it's negligible per token.
  (Rev-1 had this terminology backwards — review N8 / Gemini REV-05.)
- **AttnFlavor population in OpBinding:** The current design uses opcodes
  to encode layer-variant context (sliding vs full, dense vs MoE). A future
  step can populate `OpFlavor::Attn(AttnFlavor { window, qk_norm, ... })`
  to make the attention ops fully self-describing, but this is cosmetic.
- **WeightSlot / ScratchSlot binding:** The current design stores opcodes
  and resolves tensors inside the handler. A future step can populate
  `OpBinding.weights` and `OpBinding.scratch` with slot indices at lower
  time, allowing the executor to bind tensors mechanically. This is the
  "Step 3+" from the superop.rs TODO.

## File changes

| File | Change |
|------|--------|
| `crates/hipfire-arch-gemma4/src/gemma4.rs` | Add variant enum, opcodes, bindings struct, ForwardBindings impl, lowered gate, lowered forward function |
| `crates/hipfire-arch-gemma4/Cargo.toml` | **No change** — `hipfire-dispatch` is already a dependency (gemma4.rs already uses `execute_steps`, `Step`, `KvTierPlan`, `DispatchCtx`; review N6) |
| `crates/hipfire-dispatch/src/pipeline/superop.rs` | No changes needed — `EscapeKind::GemmaLogitSoftcap` already defined |

## Estimated effort

| Step | Lines | Complexity | Risk |
|------|-------|------------|------|
| 1 — Scaffold | ~100 | Low | None |
| 2 — Norm + Proj | ~150 | Low | Low |
| 3 — Attend | ~200 | Medium | Medium (two very different attention paths) |
| 4 — ResidualGemv | ~100 | Medium | Medium (sandwich-norm residual pattern) |
| 5 — MoE | ~50 | Low | Low (delegates to existing helper) |
| 6 — Escape | ~10 | Low | None |
| 7 — Byte-parity gate | ~0 (testing only) | Low | None |
| 8 — Flip default | ~5 | Low | None |

**Total:** ~600 lines of new code. Steps 1-2 and 5-6 are genuinely boilerplate; **Steps
3-4 are the real engineering** and where parity will break first — the two divergent
attention paths (the `V←K`-before-k_norm ordering on Full), and Gemma's multi-save
sandwich-norm residual. Do not treat the whole migration as "boilerplate dispatch
matching" (review P4); budget the risk on 3-4.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Sandwich-norm residual pattern differs from qwen35 (more save/restore points) | Factor out helpers carefully; norm ownership lives in `Norm(*)` ops, residual ops are plumbing-only; bit-exact parity gate catches any mismatch |
| Full-layer attention (hd=512, no v_proj, k_eq_v, partial rope, n_kv=1) is unique to Gemma 4 | Separate `PROJ_Q_FULL`/`PROJ_K_FULL` opcodes + `V←K` copy inside `run_attend(FULL)`; validate the Full path at Step 3 via 26B + `HIPFIRE_MOE_BYPASS=1` (12B has no Full layers); diagnostic dumps at every stage |
| MoE branch combines with dense FFN | The `Moe` super-op fires after the dense FFN in the program and consumes its output; `run_moe` encapsulates the combine + outer norm, so the trailing `ResidualGemv(POST_FFN)` is identical dense vs MoE. Matches hand-path ordering exactly. |
| `V←K` copy ordering on Full layers | Highest parity risk: the copy MUST precede k_norm. `run_attend(FULL)`'s first action is the copy; validate at long context (review B2). |
| Post-attention double-norm | `Norm(POST_ATTN)` owns the rmsnorm; `ResidualGemv(POST_ATTN)` does residual-only. Never emit/run the norm twice (review B4). |
| Graph capture interaction | **Not gated.** qwen35's `hidden_rb` gate is spec-decode, not graph capture, and gemma4 has no spec-decode capture path + graph capture is hardwired off today. If gemma4 graph capture is later wired, it needs its own parity check (review N7). |
| Prefill still uses hand path | Not a risk — prefill and decode are separate functions |

## Naming convention

Follow the existing fleet pattern, but with gemma4's actual function names (the entry
is `forward_scratch`, the per-layer loop is `forward_scratch_inner` —
`forward_scratch_layers` is qwen35's name and does not exist here; review N9 /
Gemini REV-06):
- Function: `forward_scratch_inner_lowered`
- Gate: `forward_lowered_enabled()` (OnceLock, default OFF initially)
- Bindings: `Gemma4Bindings<'a>`
- Lower: `lower_variant(variant_of(layer))`
- Opcodes: `g4_op::*`
