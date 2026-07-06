# Cross-arch `WeightBackend` Adoption (qwen2 + llama) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate qwen2 and llama single-tensor weight loading onto the existing `WeightBackend` trait, keeping each arch's own layer-walk, and collapse llama's two loaders into one shared walk.

**Architecture:** Add one method (`bias()`) to `WeightBackend`. qwen2's `load_layer` and a new `llama::load_layer<B>` build an `HfqBackend`/`ParoBackend` and call `b.proj/norm/raw_f32/bias`. Name resolution reuses the already-built `flat_name_candidates` (prefix-less `hfq_proj_name`/`hfq_plain_name` + `model.` prepend) and `norm_bias = 0.0`. Each arch passes its OWN existing reader as `read_proj` so the dequant path stays byte-identical; only name resolution moves to the candidates fn.

**Tech Stack:** Rust; `rdna-compute` (`Gpu`/`GpuTensor`) → `hipfire-runtime` (`WeightBackend`/`HfqBackend`/`ParoBackend`, `dequant_*`) → `hipfire-arch-qwen2` / `hipfire-runtime::hfq` (llama).

**Design doc:** `docs/plans/2026-06-11-cross-arch-weightbackend-adoption-design.md`.

**Key facts verified during planning:**
- `HfqBackend::read_proj` field type: `fn(&HfqFile, &Gpu, &str, usize, usize, fn(&str)->Vec<String>) -> HipResult<WeightTensor>` (`weight_backend.rs:983`). qwen35 sets it to its own `load_weight_tensor`. qwen2/llama will set it to their OWN widened readers.
- `HfqBackend::norm` already does `dequant_norm(qt, data, shape, self.norm_bias)`; with `norm_bias=0.0` this is **byte-identical** to llama's `load_f16_tensor` for F16/F32 (both: `f16_to_f32`/`from_le_bytes` → `upload_f32`; +0.0 is a no-op). Verified by inspection (`weight_backend.rs:447`, `hfq.rs:552`).
- `dequant_norm`, `dequant_f32`, `dequant_weight_raw` are `pub` in `weight_backend.rs`.
- Error constructor in scope: `HipError::new(0, "msg")` (used at `hfq.rs:37`).
- llama `LayerWeights` (`llama.rs:658`) is dense-only: `attn_norm, wq, wk, wv, wo, q_norm: Option, k_norm: Option, ffn_norm, w_gate, w_up, w_down`. No biases, no MoE.
- qwen2 `Qwen2LayerWeights` fields: `attn_norm, wq, wq_bias, wk, wk_bias, wv, wv_bias, wo, ffn_norm, w_gate, w_up, w_down`.

---

## Task 0: Prerequisite — green build + source models + baseline capture

**This task is verification setup, not code. It MUST complete before Tasks 3/5/6 can be gated.**

- [ ] **Step 1: Confirm the tree compiles (gap #1)**

Run: `cargo check -p hipfire-arch-qwen35 -p hipfire-runtime -p hipfire-loader`
Expected: no `error[...]`. If the `shallow_clone` E0599 at `qwen35.rs:1387` is still present, STOP — gap #1 must be fixed first; this plan's byte-identical gate cannot run on a red tree.

- [ ] **Step 2: Source three models into `~/.hipfire/models/`**

Need: one HFQ qwen2 (`arch_id 7`), one HFQ llama / Qwen3-dense (`arch_id < 5`), one PARO/safetensors llama directory. Record absolute paths and `md5` of each in a scratch note (`/home/bjoern/hipfire-cross-arch-baselines/paths.txt` — not `/tmp`).

- [ ] **Step 3: Capture pre-refactor baselines**

Build the dump tool once: `cargo build --release --example greedy_dump -p hipfire-runtime`
For EACH model, with a fixed prompt and `--temperature 0.0`, capture the first 32 greedy token ids to a file under `/home/bjoern/hipfire-cross-arch-baselines/` (e.g. `qwen2-before.txt`, `llama-hfq-before.txt`, `llama-paro-before.txt`). Use the GPU lock per CLAUDE.md (`source scripts/gpu-lock.sh && gpu_acquire "cross-arch" ... gpu_release`).
Expected: 32 token ids per model. These are the byte-identical reference.

---

## Task 1: Add `bias()` to `WeightBackend`

**Files:**
- Modify: `crates/hipfire-runtime/src/weight_backend.rs`

- [ ] **Step 1: Add the trait method**

In `weight_backend.rs:968` extend the trait:

```rust
pub trait WeightBackend {
    fn set_layer(&mut self, layer: usize);
    fn proj(&mut self, rel: &str, m: usize, k: usize) -> HipResult<WeightTensor>;
    fn norm(&mut self, rel: &str, shape: &[usize]) -> HipResult<GpuTensor>;
    fn raw_f32(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor>;
    /// Load a bias vector (f32). Only qwen2 attention biases use this today.
    fn bias(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor>;
}
```

- [ ] **Step 2: Implement `HfqBackend::bias`**

In the `impl<'a> WeightBackend for HfqBackend<'a>` block (after `raw_f32`, ~`weight_backend.rs:999`):

```rust
    fn bias(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor> {
        let name = hfq_plain_name(self.layer, rel);
        let (info, data) = read_first(self.hfq, &name, self.candidates)
            .unwrap_or_else(|| panic!("tensor not found: {name}"));
        let t = dequant_f32(self.gpu, info.quant_type, &data, n)?;
        assert_eq!(t.numel(), n,
            "bias {name} has {} elements, expected {n}", t.numel());
        Ok(t)
    }
```

This reproduces qwen2's `load_bias_f32` (resolve → `dequant_f32` → length assert), with candidate-based name resolution.

- [ ] **Step 3: Implement `ParoBackend::bias`**

In the `impl<'a> WeightBackend for ParoBackend<'a>` block (after `raw_f32`, ~`weight_backend.rs:1046`):

```rust
    fn bias(&mut self, _rel: &str, _n: usize) -> HipResult<GpuTensor> {
        Err(hip_bridge::HipError::new(0, "ParoBackend: attention biases unsupported"))
    }
```

If `hip_bridge::HipError` is not already imported at the top of the file, add `use hip_bridge::HipError;` (or use the fully-qualified path as shown).

- [ ] **Step 4: Build**

Run: `cargo check -p hipfire-runtime`
Expected: compiles. (qwen35's `load_layer` does not implement `bias` because it uses `HfqBackend`/`ParoBackend` from this crate — the impls above satisfy the trait for all callers.)

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-runtime/src/weight_backend.rs
git commit -m "feat(weight_backend): add bias() to WeightBackend (HfqBackend impl, ParoBackend errors)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: qwen2 — adopt `HfqBackend` in `load_layer`

**Files:**
- Modify: `crates/hipfire-arch-qwen2/src/qwen2.rs`

- [ ] **Step 1: Widen qwen2's `load_weight_tensor` to be candidates-aware**

Replace `load_weight_tensor` (`qwen2.rs:430-440`) with:

```rust
fn load_weight_tensor(
    hfq: &HfqFile,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
    candidates: fn(&str) -> Vec<String>,
) -> HipResult<WeightTensor> {
    for cand in candidates(name) {
        if let Some((info, data)) = hfq.tensor_data_vec(&cand) {
            return dequant_weight_raw(gpu, info.quant_type, &data, m, k);
        }
    }
    panic!("qwen2: tensor not found: {name}");
}
```

This keeps qwen2's exact dequant path (`tensor_data_vec` + `dequant_weight_raw`); only adds candidate resolution. Add imports at the top of `qwen2.rs` if missing: `use hipfire_runtime::weight_backend::{HfqBackend, flat_name_candidates};` (`dequant_weight_raw`, `dequant_norm`, `dequant_f32` are already imported by the existing helpers).

- [ ] **Step 2: Rewrite `load_layer` to drive `HfqBackend`**

Replace the body of `load_layer` (`qwen2.rs:351-383`) with:

```rust
fn load_layer(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &Qwen2Config,
    i: usize,
) -> HipResult<Qwen2LayerWeights> {
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    let mut b = HfqBackend {
        hfq, gpu, norm_bias: 0.0,
        candidates: flat_name_candidates,
        read_proj: load_weight_tensor,
        layer: i,
    };

    Ok(Qwen2LayerWeights {
        attn_norm: b.norm("input_layernorm.weight", &[cfg.hidden_size])?,
        wq:      b.proj("self_attn.q_proj", q_dim, cfg.hidden_size)?,
        wq_bias: b.bias("self_attn.q_proj.bias", q_dim)?,
        wk:      b.proj("self_attn.k_proj", kv_dim, cfg.hidden_size)?,
        wk_bias: b.bias("self_attn.k_proj.bias", kv_dim)?,
        wv:      b.proj("self_attn.v_proj", kv_dim, cfg.hidden_size)?,
        wv_bias: b.bias("self_attn.v_proj.bias", kv_dim)?,
        wo:      b.proj("self_attn.o_proj", cfg.hidden_size, q_dim)?,
        ffn_norm: b.norm("post_attention_layernorm.weight", &[cfg.hidden_size])?,
        w_gate:  b.proj("mlp.gate_proj", cfg.intermediate_size, cfg.hidden_size)?,
        w_up:    b.proj("mlp.up_proj", cfg.intermediate_size, cfg.hidden_size)?,
        w_down:  b.proj("mlp.down_proj", cfg.hidden_size, cfg.intermediate_size)?,
    })
}
```

Name equivalence (byte-identical): `b.proj("self_attn.q_proj", …)` → `hfq_proj_name(i,"self_attn.q_proj")` = `layers.{i}.self_attn.q_proj.weight` → `flat_name_candidates` → `model.layers.{i}.self_attn.q_proj.weight` (== today's `{p}.self_attn.q_proj.weight`, `p="model.layers.{i}"`). `b.norm`/`b.bias` use `hfq_plain_name` → `layers.{i}.{rel}` → same `model.` prefix.

The caller (`qwen2.rs:280`, `load_layer(hfq, gpu, cfg, i)`) is unchanged.

- [ ] **Step 3: Delete the now-orphaned helpers**

Delete `load_norm_weight_raw` (`qwen2.rs:398-403`) and `load_bias_f32` (`qwen2.rs:413-422`) — both are replaced by `HfqBackend::norm`/`bias`. Keep the widened `load_weight_tensor` (it is now `read_proj`).

- [ ] **Step 4: Build**

Run: `cargo check -p hipfire-arch-qwen2`
Expected: compiles; no `dead_code` warning for `load_norm_weight_raw`/`load_bias_f32` (deleted). If either is still referenced elsewhere, grep `rg 'load_norm_weight_raw|load_bias_f32' crates/hipfire-arch-qwen2` and migrate those call sites first.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-arch-qwen2/src/qwen2.rs
git commit -m "refactor(qwen2): load_layer drives HfqBackend; drop bespoke norm/bias helpers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: qwen2 byte-identical gate

**Files:** none (verification). Requires Task 0 baselines + green build.

- [ ] **Step 1: Capture post-refactor tokens**

Rebuild and re-run `greedy_dump` on the qwen2 HFQ model with the SAME prompt + `--temperature 0.0` used in Task 0, to `qwen2-after.txt`. GPU lock per CLAUDE.md.

- [ ] **Step 2: Diff**

Run: `diff /home/bjoern/hipfire-cross-arch-baselines/qwen2-before.txt /home/bjoern/hipfire-cross-arch-baselines/qwen2-after.txt`
Expected: **no diff** (pure refactor; loader output must be identical).
If diff is non-empty: a name-resolution or dtype mismatch exists — bisect by comparing one tensor's device-buffer hash before/after; do NOT proceed.

- [ ] **Step 3: Coherence gate**

Run: `./scripts/coherence-gate.sh` (it must include the qwen2 model, or run the daemon on it manually and eyeball fluency).
Expected: fluent, on-topic, no verbatim loop.

- [ ] **Step 4: Commit the gate record**

```bash
git commit --allow-empty -m "test(qwen2): byte-identical + coherence green after HfqBackend adoption

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: llama — add `load_layer<B>` + widen reader

**Files:**
- Modify: `crates/hipfire-runtime/src/hfq.rs`

- [ ] **Step 1: Widen llama's `load_weight_tensor` to be candidates-aware**

Replace `load_weight_tensor` (`hfq.rs:624`, the llama one with the inline quant-type match) signature and name resolution, KEEPING its exact match body:

```rust
fn load_weight_tensor(
    hfq: &HfqFile,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
    candidates: fn(&str) -> Vec<String>,
) -> HipResult<WeightTensor> {
    let st_name = candidates(name)
        .into_iter()
        .find(|c| hfq.find_tensor_info(c).is_some())
        .unwrap_or_else(|| panic!("tensor not found: {name}"));
    let (info, data) = hfq.tensor_data(&st_name)
        .unwrap_or_else(|| panic!("tensor not found: {st_name}"));
    // ... existing `let mut wt = match info.quant_type { 0 => …, 3 => …, … }` body UNCHANGED …
}
```

(Only the lookup at the top changes from a single `tensor_data(st_name)` to candidate resolution; the `match info.quant_type` block and the AWQ-sidecar tail stay verbatim. `find_tensor_info` is the existing presence probe used at `hfq.rs:22`.)

- [ ] **Step 2: Fix the `lm_head.weight` call site**

`load_weights_hfq` calls `load_weight_tensor(hfq, gpu, "lm_head.weight", …)` (`hfq.rs:~817`). Add the resolver argument: `load_weight_tensor(hfq, gpu, "lm_head.weight", config.vocab_size, config.dim, flat_name_candidates)`. (`lm_head.weight` is not under `model.` so `flat_name_candidates` yields `model.lm_head.weight` then bare — preserving today's exact name `lm_head.weight`? NOTE: today's call passes the literal `"lm_head.weight"` to `tensor_data`. `flat_name_candidates("lm_head.weight")` → `["model.lm_head.weight", "lm_head.weight"]`, trying `model.lm_head.weight` FIRST. If the on-disk name is bare `lm_head.weight`, this still resolves (2nd candidate) but only if `model.lm_head.weight` is absent. Confirm via the byte-identical gate; if a model has BOTH, pin this call to a single-name resolver `|n| vec![n.to_string()]` to preserve exact behavior.)

- [ ] **Step 3: Add `llama::load_layer<B>`**

Add to `hfq.rs` (imports: `use hipfire_runtime::weight_backend::WeightBackend;` is internal — already in crate; `use crate::llama::LayerWeights;`):

```rust
/// Single llama per-layer walk over a `WeightBackend`. Dense-only (no MoE,
/// no DeltaNet). `q_out_dim`/`kv_dim` are passed in so the caller reuses the
/// exact dims it already computes.
pub(crate) fn load_layer<B: WeightBackend>(
    b: &mut B,
    config: &crate::llama::LlamaConfig,
    q_out_dim: usize,
    kv_dim: usize,
    i: usize,
) -> HipResult<LayerWeights> {
    b.set_layer(i);
    Ok(LayerWeights {
        attn_norm: b.norm("input_layernorm.weight", &[config.dim])?,
        wq: b.proj("self_attn.q_proj", q_out_dim, config.dim)?,
        wk: b.proj("self_attn.k_proj", kv_dim, config.dim)?,
        wv: b.proj("self_attn.v_proj", kv_dim, config.dim)?,
        wo: b.proj("self_attn.o_proj", config.dim, q_out_dim)?,
        q_norm: if config.has_qk_norm {
            Some(b.norm("self_attn.q_norm.weight", &[config.head_dim])?)
        } else { None },
        k_norm: if config.has_qk_norm {
            Some(b.norm("self_attn.k_norm.weight", &[config.head_dim])?)
        } else { None },
        ffn_norm: b.norm("post_attention_layernorm.weight", &[config.dim])?,
        w_gate: b.proj("mlp.gate_proj", config.hidden_dim, config.dim)?,
        w_up: b.proj("mlp.up_proj", config.hidden_dim, config.dim)?,
        w_down: b.proj("mlp.down_proj", config.dim, config.hidden_dim)?,
    })
}
```

(Confirm `LlamaConfig` field names `dim`, `hidden_dim`, `head_dim`, `has_qk_norm` match those used in `load_weights_hfq`'s loop — they are taken from there.)

- [ ] **Step 4: Build**

Run: `cargo check -p hipfire-runtime`
Expected: compiles (no callers yet — `load_layer` is `pub(crate)`, may warn `dead_code`; that clears in Task 5).

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-runtime/src/hfq.rs
git commit -m "feat(llama): candidates-aware reader + load_layer<B> walk (not yet wired)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: llama HFQ — route `load_weights_hfq` through `load_layer`

**Files:**
- Modify: `crates/hipfire-runtime/src/hfq.rs`

- [ ] **Step 1: Replace the per-layer loop body**

In `load_weights_hfq` (`hfq.rs:748`), the embed / `output_norm` / `lm_head` loading BEFORE the loop and the `LlamaWeights` assembly AFTER stay unchanged. Replace the `for i in 0..config.n_layers { … let layer = LayerWeights { … }; layers.push(layer); }` block (`hfq.rs:~85`) with:

```rust
    let q_out_dim = config.n_heads * config.head_dim; // KEEP whatever expr exists today
    let kv_dim = config.n_kv_heads * config.head_dim; // KEEP whatever expr exists today
    let mut layers = Vec::with_capacity(config.n_layers);
    {
        let mut b = HfqBackend {
            hfq, gpu, norm_bias: 0.0,
            candidates: flat_name_candidates,
            read_proj: load_weight_tensor,
            layer: 0,
        };
        for i in 0..config.n_layers {
            layers.push(load_layer(&mut b, config, q_out_dim, kv_dim, i)?);
        }
    } // `b` dropped here, releasing the &mut Gpu borrow
```

IMPORTANT: reuse the EXACT `q_out_dim` and `kv_dim` expressions already present in `load_weights_hfq` before the loop today (do not re-derive — copy the existing let-bindings). Add `use hipfire_runtime::weight_backend::{HfqBackend, flat_name_candidates};` — but this is the same crate, so use `use crate::weight_backend::{HfqBackend, flat_name_candidates};`.

- [ ] **Step 2: Build**

Run: `cargo check -p hipfire-runtime`
Expected: compiles; `load_layer` `dead_code` warning gone. The bespoke per-layer `load_f16_tensor`/`load_weight_tensor` norm calls inside the old loop are removed (the walk now handles per-layer norms via `HfqBackend::norm`). `load_f16_tensor` is STILL used for `output_norm`/embed — keep it.

- [ ] **Step 3: Byte-identical gate (llama HFQ)**

Re-run `greedy_dump` on the llama HFQ model (same prompt, temp 0.0) → `llama-hfq-after.txt`.
Run: `diff …/llama-hfq-before.txt …/llama-hfq-after.txt`
Expected: **no diff**. Then `./scripts/coherence-gate.sh` (llama HFQ) green.

- [ ] **Step 4: Commit**

```bash
git add crates/hipfire-runtime/src/hfq.rs
git commit -m "refactor(llama): load_weights_hfq drives load_layer over HfqBackend

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: llama PARO — route safetensors path through `load_layer`

**Files:**
- Modify: `crates/hipfire-runtime/src/hfq.rs`

> Implementation choice (within the approved design): `load_weights_paroquant_llama` KEEPS its `pub fn` signature (so `LlamaCarrier`'s Dir arm at `carriers.rs:255` is untouched). The DUPLICATION the design targets — its separate per-layer walk — is eliminated by funnelling through the shared `load_layer`. This is strictly less risky than deleting the fn + rewiring the carrier.

- [ ] **Step 1: Replace the PARO per-layer loop body**

In `load_weights_paroquant_llama` (`hfq.rs:1117`), keep the embed / `output_norm` / `lm_head` PARO loading (uses `paro_load_llama_norm_raw`, `load_fp16_weight_tensor_from_source`) and the final assembly. Replace its `for i in 0..config.n_layers { … }` per-layer body (`hfq.rs:~1149`) with:

```rust
    let q_out_dim = config.n_heads * config.head_dim; // KEEP existing expr
    let kv_dim = config.n_kv_heads * config.head_dim; // KEEP existing expr
    let mut layers = Vec::with_capacity(config.n_layers);
    {
        let mut b = crate::weight_backend::ParoBackend {
            source, gpu,
            mp: "model",          // KEEP whatever model-prefix the current code passes to paro_proj_name
            layer: 0,
            norm_bias: 0.0,
        };
        for i in 0..config.n_layers {
            layers.push(load_layer(&mut b, config, q_out_dim, kv_dim, i)?);
        }
    }
```

NOTE on `mp`: `ParoBackend::proj` builds `paro_proj_name(self.mp, layer, rel)` = `{mp}.layers.{layer}.{rel}` then the augmentor appends `.qweight`/`.weight`. Confirm `mp` reproduces the prefix the current `load_weights_paroquant_llama` uses (the design notes llama is flat `model.layers.…`, so `mp = "model"`). Verify against the current `format!`/`paro_load_*` names in this fn before building.

`ParoBackend::norm` uses `paro_load_norm(…, norm_bias=0.0)` — confirm this matches the current `paro_load_llama_norm_raw` for llama's norm dtype (both should be raw F16/F32 → upload; the gate proves it). `q_norm`/`k_norm` are loaded conditionally inside `load_layer` (Task 4) via `b.norm`, matching the current `if config.has_qk_norm` blocks.

- [ ] **Step 2: Delete now-orphaned PARO per-layer helpers**

If `paro_load_llama_norm_raw` (and any per-layer-only PARO weight helper) is now referenced ONLY by the embed/output code (not the deleted loop), keep it. Run `rg 'paro_load_llama_norm_raw' crates/hipfire-runtime` — delete it only if it has zero remaining callers. Do the same for any per-layer PARO weight loader the old loop used.

- [ ] **Step 3: Build**

Run: `cargo check -p hipfire-runtime -p hipfire-loader`
Expected: compiles; no new `dead_code` (or remove whatever the old loop solely used).

- [ ] **Step 4: Byte-identical gate (llama PARO)**

Re-run `greedy_dump` on the PARO/safetensors llama dir (same prompt, temp 0.0) → `llama-paro-after.txt`.
Run: `diff …/llama-paro-before.txt …/llama-paro-after.txt`
Expected: **no diff**. Then `./scripts/coherence-gate.sh` (PARO llama) green — fluent, no attractor.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-runtime/src/hfq.rs
git commit -m "refactor(llama): paroquant loader funnels through shared load_layer (ParoBackend)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 7: Final sweep + close the gap

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`, `crates/hipfire-runtime/src/hfq.rs` (remove the two `@todo(unified-loading)` markers if their conditions are now met)

- [ ] **Step 1: Remove the llama `@todo(unified-loading)` marker**

`hfq.rs:1110` carries `// @todo(unified-loading): llama PARO still has its own loader; … Migrate llama onto the same generic walk`. llama now runs through `load_layer`; delete this comment block.

- [ ] **Step 2: Full workspace build + warning scan**

Run: `cargo check --workspace 2>&1 | rg '^warning|^error' | rg -i 'qwen2|llama|weight_backend|hfq' || echo CLEAN`
Expected: no new errors; no `dead_code` for anything this plan touched.

- [ ] **Step 3: Final coherence gate (all three)**

Run: `./scripts/coherence-gate.sh`
Expected: green across qwen2 + llama HFQ + PARO llama (plus the existing qwen35 models). Eyeball each report per CLAUDE.md.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs(loading): close llama @todo(unified-loading) — llama on shared load_layer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Update the review memory**

Mark gap #2 done in `unified-loading-review-todos.md` (memory), with the commit range. The qwen2 `@todo`-free state and the deleted llama duplication are the evidence.

---

## Self-review (done at plan-write time)

- **Spec coverage:** §1 trait change → Task 1. §2 qwen2 → Tasks 2-3. §3 llama collapse → Tasks 4-6. §4 verification → Task 0 + the per-arch gates in Tasks 3/5/6. §6 risks: Risk-1 (norm) resolved by inspection + gated; Risk-2 (name prefix) handled by `flat_name_candidates` + gated; Risk-3 (`bias` on ParoBackend) compile-isolated (only qwen2 calls `bias`, qwen2 has no ParoBackend).
- **Placeholder scan:** the two `// KEEP existing expr` notes on `q_out_dim`/`kv_dim` are deliberate instructions to copy existing exact code (not invent), because re-deriving the dim formula risks a silent mismatch — the implementer lifts the present let-bindings verbatim. The `mp` prefix in Task 6 is flagged for confirmation against current PARO names. These are verification instructions, not unfilled blanks.
- **Type consistency:** `load_layer<B>` signature `(b, config, q_out_dim, kv_dim, i)` is identical in Tasks 4/5/6. `read_proj` fn-ptr signature matches the widened readers in Tasks 2 (qwen2) and 4 (llama). `bias(rel, n)` matches between trait (Task 1) and call sites (Task 2).

## Deferred (unchanged by this plan)

- Generic arch-agnostic `load_layer` (rejected — reverted once as 611e10ce/f2a5895b).
- qwen2 safetensors/PARO + multi-GPU (no path exists).
- Embedding/output assembly unification (stays arch-local per the qwen35 precedent).
- The multi-GPU `@todo(unified-loading)` at `qwen35.rs:1566` stays (out of scope).
