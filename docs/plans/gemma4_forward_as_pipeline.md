# Gemma 4 Forward-as-Pipeline Migration Plan

**Date:** 2026-06-09  
**Status:** Planning  
**Depends on:** `feat/dispatch-unification-gemma4` tip (`d42771da`), upstream `integration/dispatch-unification` merged  
**Blocks:** None — this is a progressive migration behind a feature flag

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

- **qwen35** (`qwen35.rs:12490`): 4 variants (DeltaNet / FullAttn × dense/MoE),
  8 opcodes, `ForwardBindings` with 5 methods.
- **LFM2** (`forward.rs:570`): 4 variants (Conv/Attn × dense/MoE), ~5 opcodes.
- **MiniMax** (`forward.rs`): reuses qwen35 pattern with MoE extension.
- **DeepSeek V4** (`forward.rs`): 6 Escape kinds for compressor/indexer/SWA.

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
q = q_norm(q)                    ← fused into Attend prep
k = k_norm(k)                    ← fused into Attend prep
v = v_norm(v)                    ← fused into Attend prep
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

Runs in **parallel** with the dense FFN:
```
cur_mlp = post_ffn_norm_1(ffn_out)
pre2 = pre_ffn_norm_2(residual)
router_logits = router(pre2)
topk = softmax_topk(router_logits, k=8)
expert_gate_up[topk] = gate_up_proj[expert](pre2)  ← Moe
hidden = gelu_tanh(gate) * up
moe_out = down_proj[expert](hidden)                ← Moe
moe_out = post_ffn_norm_2(moe_out)
x = cur_mlp + moe_out + residual
x *= layer_scalar
```

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

```rust
mod g4_op {
    // Proj (projection clusters)
    pub const PROJ_QKV: u32 = 0;        // q_proj + k_proj (fused); v_proj separate
    pub const PROJ_V: u32 = 1;           // v_proj (standalone — not same quant as q/k)
    pub const PROJ_O: u32 = 2;           // o_proj
    pub const PROJ_GATE_UP: u32 = 3;    // gate_proj + up_proj (fused)
    pub const PROJ_DOWN: u32 = 4;       // down_proj
    pub const PROJ_LM_HEAD: u32 = 5;    // final lm_head (output stage)

    // Attend
    pub const ATTEND_SLIDING: u32 = 0;  // window=1024, head_dim=256, full rope
    pub const ATTEND_FULL: u32 = 1;     // window=0, head_dim=512, partial rope, k_eq_v

    // Norm
    pub const NORM_INPUT: u32 = 0;      // input_layernorm
    pub const NORM_POST_ATTN: u32 = 1;  // post_attention_layernorm
    pub const NORM_PRE_FFN: u32 = 2;    // pre_feedforward_layernorm
    pub const NORM_POST_FFN: u32 = 3;   // post_feedforward_layernorm (dense)
    pub const NORM_FINAL: u32 = 4;      // final norm (output stage)

    // ResidualGemv
    pub const RESID_POST_ATTN: u32 = 0; // residual + o_proj(norm(attn_out))

    // Moe
    pub const MOE_BRANCH: u32 = 0;      // parallel MoE FFN

    // Escape
    // Uses EscapeKind::GemmaLogitSoftcap (already defined in superop.rs)
}
```

### Layer programs

```
SlidingDense:
  Norm(INPUT)        — rmsnorm
  Proj(QKV)          — q_proj + k_proj via execute_steps
  Proj(V)            — v_proj via weight_gemv
  Attend(SLIDING)    — q/k/v_norm + scale_q + rope + kv_write + flash_attn
  Proj(O)            — o_proj via execute_steps
  Norm(POST_ATTN)    — rmsnorm
  ResidualGemv(POST_ATTN) — memcpy residual + add
  Norm(PRE_FFN)      — rmsnorm
  Proj(GATE_UP)      — gate + up via execute_steps
  Norm(NONE)         — gelu_tanh + mul (not really a norm — see below)
  Proj(DOWN)         — down_proj via execute_steps
  Norm(POST_FFN)     — rmsnorm
  ResidualGemv(POST_FFN) — memcpy residual + add + scale(layer_scalar)

SlidingMoe: (same as SlidingDense through DOWN, then:)
  [... same as SlidingDense through PROJ_DOWN ...]
  Moe(MOE_BRANCH)    — router + topk + expert gate_up/down + sandwich norms
  ResidualGemv(POST_FFN) — memcpy residual + add + scale(layer_scalar)

FullDense:
  Norm(INPUT)
  Proj(QKV)
  Proj(V)
  Attend(FULL)       — q/k/v_norm + partial rope + k_eq_v prelude + kv_write + flash_attn
  Proj(O)
  Norm(POST_ATTN)
  ResidualGemv(POST_ATTN)
  Norm(PRE_FFN)
  Proj(GATE_UP)
  Norm(NONE)         — gelu_tanh + mul
  Proj(DOWN)
  Norm(POST_FFN)
  ResidualGemv(POST_FFN)

FullMoe:
  [... same as FullDense through PROJ_DOWN ...]
  Moe(MOE_BRANCH)
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
mirroring the hand path exactly. No new SuperOpKind needed.

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
| `run_proj` | QKV, V, O, GATE_UP, DOWN, LM_HEAD | `execute_steps` (QKV, O, GATE_UP), `weight_gemv` (V, DOWN), `execute_steps` (LM_HEAD) |
| `run_attend` | SLIDING, FULL | Existing `kv_cache_write_*` + `attention_flash_*_window` sequence (factored out of `sliding_layer_decode_impl` / `full_layer_decode_impl`) |
| `run_norm` | INPUT, POST_ATTN, PRE_FFN, POST_FFN, FINAL | `gpu.rmsnorm_f32` |
| `run_residual_gemv` | POST_ATTN, POST_FFN | `memcpy_dtod` + `add_inplace_f32` + `scale_f32` |
| `run_moe` | MOE_BRANCH | `apply_moe_branch` (existing) |
| `run_escape` | GemmaLogitSoftcap | `gpu.logit_softcap_f32` |
| `run_recurrent` | — | `Err(Unsupported)` |
| `run_conv` | — | `Err(Unsupported)` |

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
6. Gate in `forward_scratch_layers`: if `forward_lowered_enabled()` and
   not in graph-capture mode, route to `forward_scratch_layers_lowered`.

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

**Validation:** 12B model produces byte-identical output to hand path.
Run with `HIPFIRE_FORWARD_LOWERED=1` and compare logits against hand path.

### Step 4 — Wire up ResidualGemv

Implement `run_residual_gemv`:

- `RESID_POST_ATTN`: memcpy residual + add_inplace + post_attn_norm + memcpy
  residual again (for FFN residual stream).
- `RESID_POST_FFN`: memcpy residual + add_inplace + scale(layer_scalar).

This is the trickiest part because Gemma 4's sandwich-norm residual pattern
has more steps than qwen35's simpler residual. The residual save/restore
sequence must exactly match the hand path.

**Validation:** 12B model byte-identical at all positions.

### Step 5 — Wire up Moe (26B-A4B)

Implement `run_moe`:

- `MOE_BRANCH`: delegate to `apply_moe_branch` (existing helper).
- Adjust `RESID_POST_FFN` to handle the MoE variant (which has a different
  residual combination: `x = cur_mlp + moe_out + residual`).

**Validation:** 26B-A4B model byte-identical with `HIPFIRE_FORWARD_LOWERED=1`.

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

For each, verify:
1. Logits are byte-identical (or within floating-point epsilon)
2. Generated token sequences match exactly
3. No panics, no NaN

**Gate criterion:** 5 consecutive runs with identical prompts produce
identical token sequences across both paths.

### Step 8 — Flip default ON

After byte-parity validation:
- Change `forward_lowered_enabled()` to default ON (same as qwen35)
- Escape hatch: `HIPFIRE_FORWARD_LOWERED=0` to force legacy path

### Step 9 — Remove hand-path duplication (optional, follow-up)

Once the lowered path is default-ON and fleet-validated:
- Remove the hand-path arms from `forward_scratch_layers`
- Keep `HIPFIRE_FORWARD_LOWERED=0` escape hatch for one release cycle
- Remove the hand path entirely in the next release

## Non-goals (deferred)

- **EP (expert parallelism):** Gemma 4 26B-A4B MoE EP is not in scope. The
  `run_moe_ep` and `ep_add_into_residual` defaults (Err) are correct.
- **Prefill migration:** The prefill path (`forward_prefill_batch_v1/v2`)
  stays on `execute_steps`. Prefill is latency-dominated, not throughput-
  dominated, so the per-resolve overhead matters less.
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
| `crates/hipfire-arch-gemma4/Cargo.toml` | Add `hipfire-dispatch` dependency if not already present |
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

**Total:** ~600 lines of new code, mostly boilerplate dispatch matching.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Sandwich-norm residual pattern differs from qwen35 | Factor out helpers carefully; byte-parity gate catches any mismatch |
| Full-layer attention (hd=512, k_eq_v, partial rope) is unique to Gemma 4 | Test both variants separately; diagnostic dumps at every stage |
| MoE branch runs in parallel with dense FFN | The `Moe` super-op fires after the dense FFN in the program, which matches the hand-path ordering exactly |
| Graph capture interaction | Gate off for graph-capture mode (same as qwen35: `hidden_rb.is_none()` equivalent) |
| Prefill still uses hand path | Not a risk — prefill and decode are separate functions |

## Naming convention

Follow the existing fleet pattern:
- Function: `forward_scratch_layers_lowered`
- Gate: `forward_lowered_enabled()` (OnceLock, default OFF initially)
- Bindings: `Gemma4Bindings<'a>`
- Lower: `lower_variant(variant_of(layer))`
- Opcodes: `g4_op::*`
