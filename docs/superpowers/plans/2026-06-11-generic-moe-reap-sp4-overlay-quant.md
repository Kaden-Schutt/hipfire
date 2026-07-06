# Generic MoE REAP — SP4 (part a): Selective Re-Quant Overlay Builder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `hipfire reap quant` to the (CPU-only) quantizer: re-quantize ONLY the tensors named by a reap plan's `quant_overrides` — read from the original fp16/bf16 safetensors — into a small `overlay.hfq`, so a quant config can be iterated without re-quantizing the whole model.

**Architecture:** Thread a `hipfire_reap::plan::ReapPlan` into `hipfire-quantize`. Two new pure-ish units: `quantize_to_format(fmt, &[f32], shape) -> HfqTensor` (an encoder-dispatch over the existing self-calibrating `quantize_*` fns) and `reap_override_for(name, &plan) -> Option<&str>` (arch-aware role+layer+expert → target tier). A new `--reap-overlay <plan-dir> --reap-out <overlay.hfq>` mode filters the existing tensor loop to overridden tensors only and writes a subset `.hfq` via the existing `write_hfq()`. The full quantize loop is untouched in default mode.

**Tech Stack:** Rust, `hipfire-quantize` (CPU; reads safetensors, writes HFQ), `hipfire-reap` (ReapPlan), `hipfire-runtime` (HfqFile reader for tests). No GPU.

**Spec:** `docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md` §3 (`reap quant`), §1 (`reap_plan.json` schema).

> **CPU-verifiable.** Unlike SP1's loader (GPU-gated), this whole plan is CPU-runnable: byte-equivalence of `quantize_to_format` vs the underlying `quantize_*` fns, name resolution unit tests, and a real overlay-build smoke comparing overlay bytes to a full-quant of the same tensor. Run everything.

**Scope note:** This is the **overlay builder** (the iterate loop). `reap bake` (whole-model quantize with overrides + expert pruning/renumber → standalone `.hfq`) is **SP4b**, a separate plan — it overlaps SP1's prune logic and is lower priority. Serving an overlay that mixes tiers *within a layer* additionally needs **SP2** (GPU mixed dispatch); per-layer-uniform overlays serve through existing dispatch once **SP3** (load-time splice) lands. This plan only builds correct overlay bytes; consuming them is SP3.

---

## Pre-read (grounding the implementer)

The quantizer is `crates/hipfire-quantize/src/main.rs` (~6k lines, CPU). Key facts (verify, don't trust):
- It ALWAYS reads original safetensors (fp16/bf16) via `SafetensorsFile` (`main.rs:88`, `tensor_data()` `:117`, `to_f32()` `:235`). FP8 ds4 weights pair `.weight`+`.scale` (`tensor_to_f32_with_optional_fp8_scale`, called ~`:5683`).
- Per-tensor encoders are free fns taking f32: `quantize_q8f16(&[f32])` (`:700`), `quantize_mq4g256(&[f32],&signs1,&signs2)` (`:819`), `quantize_mq6g256` (`:862`), `quantize_hfq4g256(&[f32])` (`:957`), `quantize_hfq3g256` (`:2710`), `quantize_mq2g256_lloyd(&[f32],&s1,&s2)` (`:2398`). FWHT signs are deterministic: `gen_fwht_signs(seed, 256)` (`:5685` uses seeds 42 / 1042). Lloyd codebooks are LOCAL per 256-group (self-calibrating; no imatrix needed for the plain variants).
- The quantize loop is `for (name, file_idx) in &all_tensors {` (`:5628`); each branch computes f32, calls a `quantize_*`, pushes `HfqTensor { name, quant_type, shape, group_size, data, spilled_len }`, `continue`s. Final `write_hfq(...)` (`:3398`) writes the whole `hfq_tensors: Vec<HfqTensor>` and accepts an ARBITRARY subset (no full-model assumption).
- `QuantType` enum + the qt byte codes are in the writer/`hfq.rs` (e.g. Q8F16=3, HFQ4G256=6, MQ4G256=13, MQ2G256Lloyd=19). `HfqTensor` struct is defined in `main.rs` (grep `struct HfqTensor`).
- `ReapPlan` (from `hipfire-reap`) exposes `quant_overrides: Vec<QuantOverride>` where `QuantOverride { layer: usize, role: Role, experts: Vec<u32>, tier: String }`, `Role ∈ {RoutedExperts, SharedExpert, Attention, Router, LmHead, Embed}`. Load via `ReapPlan::load_any(dir, num_layers, orig_experts)` — but the quantizer may not know `num_layers`/`orig_experts` up front; see Task 3 for loading without the model's counts (use a lenient loader).

---

### Task 1: Add `hipfire-reap` dep + a lenient plan loader for the quantizer

The quantizer reads the plan BEFORE it knows the model's layer/expert counts (those come from the safetensors). Add a `ReapPlan::load_unchecked(dir)` that parses without cross-validating against model counts (validation of indices vs the actual model happens when tensors are matched).

**Files:**
- Modify: `crates/hipfire-quantize/Cargo.toml`
- Modify: `crates/hipfire-reap/src/plan.rs` (add `load_unchecked`)
- Test: inline in `plan.rs`

- [ ] **Step 1: Add the dep**

In `crates/hipfire-quantize/Cargo.toml` `[dependencies]`: `hipfire-reap = { path = "../hipfire-reap" }`. Verify it builds: `cargo build -p hipfire-quantize` (may be slow; that's fine).

- [ ] **Step 2: Write the failing test for `load_unchecked`**

In `crates/hipfire-reap/src/plan.rs` `#[cfg(test)]`:
```rust
#[test]
fn load_unchecked_parses_without_model_counts() {
    let d = write_plan(
        r#"{"original_experts":256,"num_layers":43,
            "quant_overrides":[{"layer":20,"role":"routed_experts","experts":[7,12],"tier":"mq3lloyd"},
                               {"layer":41,"role":"attention","tier":"q8"}]}"#,
    );
    let p = ReapPlan::load_unchecked(d.path().to_str().unwrap()).unwrap();
    assert_eq!(p.original_experts, 256);
    assert_eq!(p.num_layers, 43);
    assert_eq!(p.quant_overrides.len(), 2);
    assert_eq!(p.quant_overrides[0].tier, "mq3lloyd");
    assert_eq!(p.quant_overrides[0].experts, vec![7, 12]);
}
```
(`write_plan` already exists in this module's tests.)

- [ ] **Step 3: Run it to confirm it fails**

Run: `cargo test -p hipfire-reap load_unchecked` → FAIL (`load_unchecked` not found).

- [ ] **Step 4: Implement `load_unchecked`**

In `crates/hipfire-reap/src/plan.rs`, add a method that reads `reap_plan.json`, parses `original_experts`/`num_layers`/`keep`/`quant_overrides` using the SAME field-parsing as `load`, but takes the model counts FROM the json (not from caller args) and skips the `!= expected` cross-checks. Factor the shared field-parsing so `load` and `load_unchecked` don't duplicate (extract a private `fn parse_value(v: &serde_json::Value, dir: &str) -> Result<Self, String>` that both call; `load` then additionally asserts `original_experts == orig_experts_expected` and `num_layers == num_layers_expected`). Keep the per-override validations (non-integer experts → Err, experts-on-non-routed-role → Err). Expert-index-vs-original bounds: keep the existing check against the plan's own `original_experts`.

- [ ] **Step 5: Run tests**

Run: `cargo test -p hipfire-reap` → all prior tests + `load_unchecked_parses_without_model_counts` pass.

- [ ] **Step 6: Commit**

```bash
git add crates/hipfire-reap/src/plan.rs crates/hipfire-quantize/Cargo.toml
git commit -m "feat(reap): ReapPlan::load_unchecked for the quantizer; quantize dep"
```

---

### Task 2: `quantize_to_format` — reusable per-tensor encoder dispatch

A single fn mapping a tier name → the right existing `quantize_*` call, returning an `HfqTensor`. This is the DRY core both overlay and (later) bake reuse.

**Files:**
- Create: `crates/hipfire-quantize/src/reap_overlay.rs`
- Modify: `crates/hipfire-quantize/src/main.rs` (add `mod reap_overlay;`, make the needed `quantize_*` fns + `HfqTensor`/`QuantType` reachable — `pub(crate)` if not already)
- Test: inline `#[cfg(test)]` in `reap_overlay.rs`

- [ ] **Step 1: Write the failing byte-equivalence test**

`crates/hipfire-quantize/src/reap_overlay.rs`:
```rust
//! SP4: selective re-quant overlay builder. See
//! docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md §3.
use crate::{HfqTensor, QuantType, quantize_q8f16, quantize_hfq4g256, quantize_hfq6g256,
            quantize_mq4g256, quantize_mq6g256, gen_fwht_signs};
// NOTE: adjust the import list / paths to the ACTUAL symbol names + visibility in main.rs.
// If a `quantize_hfq6g256` free fn doesn't exist, find the real HFQ6 encoder and use it.

/// Quantize one tensor's f32 data to the named tier, returning the HFQ tensor.
/// Covers the self-calibrating tiers usable in an overlay without an imatrix.
/// `shape` is the row-major tensor shape (e.g. `[rows, cols]`).
pub fn quantize_to_format(
    name: &str,
    fmt: &str,
    f32_data: &[f32],
    shape: &[usize],
) -> Result<HfqTensor, String> {
    let shape_u32: Vec<u32> = shape.iter().map(|&s| s as u32).collect();
    let (qt, gs, data) = match fmt {
        "q8" | "q8f16" => (QuantType::Q8F16, 0u32, quantize_q8f16(f32_data)),
        "hfq4" | "hfq4g256" => (QuantType::HFQ4G256, 256, quantize_hfq4g256(f32_data)),
        "hfq6" | "hfq6g256" => (QuantType::HFQ6G256, 256, quantize_hfq6g256(f32_data)),
        "mq4" | "mq4g256" => {
            let (s1, s2) = (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
            (QuantType::MQ4G256, 256, quantize_mq4g256(f32_data, &s1, &s2))
        }
        "mq6" | "mq6g256" => {
            let (s1, s2) = (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
            (QuantType::MQ6G256, 256, quantize_mq6g256(f32_data, &s1, &s2))
        }
        other => return Err(format!("reap: unsupported overlay tier '{other}' for {name}")),
    };
    Ok(HfqTensor { name: name.to_string(), quant_type: qt, shape: shape_u32,
                   group_size: gs, data, spilled_len: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_underlying_encoder_byte_for_byte() {
        // 512 f32 (two 256-groups) of varied values.
        let f32: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.013).sin()).collect();
        let direct = quantize_hfq4g256(&f32);
        let t = quantize_to_format("x", "hfq4g256", &f32, &[2, 256]).unwrap();
        assert_eq!(t.data, direct, "overlay encode must equal direct encode");
        assert_eq!(t.shape, vec![2u32, 256]);
        assert_eq!(t.quant_type, QuantType::HFQ4G256);
    }
    #[test]
    fn mq4_matches_with_canonical_signs() {
        let f32: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
        let (s1, s2) = (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
        let direct = quantize_mq4g256(&f32, &s1, &s2);
        let t = quantize_to_format("x", "mq4", &f32, &[1, 256]).unwrap();
        assert_eq!(t.data, direct);
    }
    #[test]
    fn rejects_unknown_tier() {
        let err = quantize_to_format("x", "bogus", &[0.0; 256], &[1, 256]).unwrap_err();
        assert!(err.contains("unsupported overlay tier 'bogus'"), "got: {err}");
    }
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test -p hipfire-quantize reap_overlay::` → FAIL (module/symbols missing).

- [ ] **Step 3: Wire visibility + module**

Add `mod reap_overlay;` to `main.rs`. The `quantize_*` fns + `gen_fwht_signs` + `HfqTensor` + `QuantType` are currently private to `main.rs`; mark each one this file imports as `pub(crate)`. Find the REAL names: grep `fn quantize_` and `enum QuantType` / `struct HfqTensor`. If `quantize_hfq6g256` doesn't exist as a free fn (HFQ6 may be produced via a different code path), either add a thin free fn wrapping the real HFQ6 encoder or map `"hfq6"` to whatever the real entry point is — match the byte output the full quantizer would produce for `--format hfq6`. Add the Lloyd tiers (`mq2lloyd`/`mq3lloyd`/`mq4lloyd`) by mapping to the real `quantize_mq*g256_lloyd` fns (signs seeds 42/1042 as the loop uses); if an mq3-lloyd encoder needs args you can't supply context-free, add it to the match with the correct call and extend the tests.

- [ ] **Step 4: Run tests**

Run: `cargo test -p hipfire-quantize reap_overlay::` → 3 tests pass (byte-equivalence proves the overlay encode is identical to the full quantizer's encode).

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-quantize/src/reap_overlay.rs crates/hipfire-quantize/src/main.rs
git commit -m "feat(reap): quantize_to_format encoder-dispatch for overlay (byte-equiv tested)"
```

---

### Task 3: `reap_override_for` — arch-aware tensor → target-tier resolver

Given a tensor name + the plan, return the override tier (or None). Encapsulates the per-arch role+layer+expert → tensor-name matching.

**Files:**
- Modify: `crates/hipfire-quantize/src/reap_overlay.rs`
- Test: inline

- [ ] **Step 1: Write the failing tests**

Append to `reap_overlay.rs`:
```rust
use hipfire_reap::plan::{ReapPlan, Role};

/// Detected arch family for tensor-name matching (the quantizer already knows
/// the arch_id; pass the matching variant in).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReapArch { Deepseek4, Qwen35, Lfm2Moe, Minimax }

/// Resolve a tensor name to its override tier under `plan`, or None.
/// Matches by (layer, role, [expert]) using the arch's tensor naming.
pub fn reap_override_for<'a>(name: &str, arch: ReapArch, plan: &'a ReapPlan) -> Option<&'a str> {
    for ov in &plan.quant_overrides {
        if tensor_matches(name, arch, ov) {
            return Some(ov.tier.as_str());
        }
    }
    None
}

fn tensor_matches(name: &str, arch: ReapArch, ov: &hipfire_reap::plan::QuantOverride) -> bool {
    // Layer gate: the name must reference `ov.layer`. All four arches embed the
    // layer index as `.layers.{L}.` or `layers.{L}.`.
    let layer_tok_a = format!("layers.{}.", ov.layer);
    if !name.contains(&layer_tok_a) { return false; }
    match ov.role {
        Role::RoutedExperts => {
            let (seg, w_ok): (&str, fn(&str) -> bool) = match arch {
                ReapArch::Deepseek4 => (".ffn.experts.", |n| n.ends_with(".w1.weight") || n.ends_with(".w2.weight") || n.ends_with(".w3.weight")),
                ReapArch::Qwen35    => (".mlp.experts.", |n| n.ends_with(".gate_up_proj.weight") || n.ends_with(".down_proj.weight")),
                ReapArch::Lfm2Moe   => (".feed_forward.experts.", |n| n.ends_with(".w1.weight") || n.ends_with(".w2.weight") || n.ends_with(".w3.weight")),
                ReapArch::Minimax   => (".block_sparse_moe.experts.", |n| n.ends_with(".w1.weight") || n.ends_with(".w2.weight") || n.ends_with(".w3.weight")),
            };
            if !name.contains(seg) || !w_ok(name) { return false; }
            if ov.experts.is_empty() { return true; } // whole role at this layer
            // expert index: the token right after `seg`
            let after = &name[name.find(seg).unwrap() + seg.len()..];
            let eidx: u32 = match after.split('.').next().and_then(|s| s.parse().ok()) {
                Some(e) => e, None => return false,
            };
            ov.experts.contains(&eidx)
        }
        Role::Attention => name.contains(".self_attn.") || name.contains(".attn.") || name.contains(".attention."),
        Role::Router => name.contains(".gate.weight") || name.contains(".router") || name.contains(".gate.tid2eid"),
        Role::SharedExpert => name.contains(".shared_expert") || name.contains(".shared_experts"),
        Role::LmHead => name.contains("lm_head") || name.contains("output.weight"),
        Role::Embed => name.contains("embed_tokens") || name.contains("tok_embeddings"),
    }
}

#[cfg(test)]
mod resolve_tests {
    use super::*;
    fn plan_with(json: &str) -> ReapPlan {
        // write to a tempdir & load_unchecked
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("reap_plan.json"), json).unwrap();
        ReapPlan::load_unchecked(d.path().to_str().unwrap()).unwrap()
    }
    #[test]
    fn ds4_specific_experts() {
        let p = plan_with(r#"{"original_experts":256,"num_layers":43,
            "quant_overrides":[{"layer":20,"role":"routed_experts","experts":[7],"tier":"mq3lloyd"}]}"#);
        assert_eq!(reap_override_for("layers.20.ffn.experts.7.w1.weight", ReapArch::Deepseek4, &p), Some("mq3lloyd"));
        assert_eq!(reap_override_for("layers.20.ffn.experts.8.w1.weight", ReapArch::Deepseek4, &p), None); // wrong expert
        assert_eq!(reap_override_for("layers.21.ffn.experts.7.w1.weight", ReapArch::Deepseek4, &p), None); // wrong layer
    }
    #[test]
    fn qwen35_whole_role() {
        let p = plan_with(r#"{"original_experts":128,"num_layers":48,
            "quant_overrides":[{"layer":5,"role":"routed_experts","tier":"hfq6"}]}"#);
        assert_eq!(reap_override_for("model.layers.5.mlp.experts.99.gate_up_proj.weight", ReapArch::Qwen35, &p), Some("hfq6"));
        assert_eq!(reap_override_for("model.layers.5.mlp.experts.99.down_proj.weight", ReapArch::Qwen35, &p), Some("hfq6"));
        assert_eq!(reap_override_for("model.layers.5.self_attn.q_proj.weight", ReapArch::Qwen35, &p), None);
    }
    #[test]
    fn attention_role() {
        let p = plan_with(r#"{"original_experts":256,"num_layers":43,
            "quant_overrides":[{"layer":41,"role":"attention","tier":"q8"}]}"#);
        assert_eq!(reap_override_for("model.layers.41.self_attn.q_proj.weight", ReapArch::Qwen35, &p), Some("q8"));
        assert_eq!(reap_override_for("model.layers.40.self_attn.q_proj.weight", ReapArch::Qwen35, &p), None);
    }
}
```
Add `tempfile` to `crates/hipfire-quantize` `[dev-dependencies]` if not present.

- [ ] **Step 2: Run to confirm fail, then it should pass once compiled**

Run: `cargo test -p hipfire-quantize reap_overlay` → the `resolve_tests` pass. If `Role`/`QuantOverride` field names differ, fix imports to the real `hipfire-reap` API.

- [ ] **Step 3: Verify the ds4 expert-index parse against real names**

The ds4 expert token-after-`seg` parse assumes `...experts.{E}.w1.weight`. Confirm against a real ds4 tensor name (grep the quantizer's ds4 branch ~`:5848`). Adjust `split('.').next()` if the real layout differs (e.g. an extra segment).

- [ ] **Step 4: Commit**

```bash
git add crates/hipfire-quantize/src/reap_overlay.rs crates/hipfire-quantize/Cargo.toml
git commit -m "feat(reap): reap_override_for arch-aware tensor->tier resolver"
```

---

### Task 4: `--reap-overlay` CLI mode — emit the subset overlay.hfq

Wire it into the quantizer: a new mode that, instead of the full quantize, loops the model tensors, quantizes ONLY the overridden ones via `quantize_to_format`, and writes `overlay.hfq`.

**Files:**
- Modify: `crates/hipfire-quantize/src/main.rs` (arg parse + the mode branch)
- Modify: `crates/hipfire-quantize/src/reap_overlay.rs` (a `build_overlay` driver if cleaner)
- Test: an integration-style test (see Step 4) + manual smoke

- [ ] **Step 1: Add the CLI args**

In the arg parsing (near `--format` at `main.rs:4715`), add `--reap-overlay <plan-dir>` and `--reap-out <overlay.hfq path>` and `--reap-arch <deepseek4|qwen35|lfm2moe|minimax>` (or detect from arch_id already resolved). When `--reap-overlay` is set, the quantizer enters overlay mode (skip the normal whole-model write).

- [ ] **Step 2: Implement the overlay loop**

Add a function (in `reap_overlay.rs`) the main flow calls when overlay mode is active:
```rust
pub fn build_overlay(
    arch: ReapArch,
    plan: &ReapPlan,
    // an iterator yielding (name, shape, f32_data) for the model's tensors —
    // the caller (main.rs) provides this from the already-open SafetensorsFile(s),
    // reusing tensor_to_f32_with_optional_fp8_scale for each tensor it decides to emit.
    tensors: impl Iterator<Item = (String, Vec<usize>, Vec<f32>)>,
) -> Result<Vec<HfqTensor>, String> {
    let mut out = Vec::new();
    for (name, shape, f32) in tensors {
        if let Some(fmt) = reap_override_for(&name, arch, plan) {
            out.push(quantize_to_format(&name, fmt, &f32, &shape)?);
        }
    }
    Ok(out)
}
```
In `main.rs`, the overlay branch iterates `all_tensors`, and for each tensor checks `reap_override_for(name, arch, &plan).is_some()` BEFORE doing the expensive f32 decode — only decode+quantize the matched ones (the whole point: skip the rest). Reuse `tensor_to_f32_with_optional_fp8_scale`. Collect into `hfq_tensors`, then call the existing `write_hfq(&reap_out_path, &hfq_tensors, ...)` (match `write_hfq`'s real signature — it also writes metadata; for an overlay, pass minimal/empty metadata or copy the base's arch_id, whichever `write_hfq` requires — read its signature).

- [ ] **Step 3: Guard: error if no overrides matched**

If `hfq_tensors` is empty after the loop, return an error: `"reap overlay: no tensors matched the plan's quant_overrides (check arch/layer/expert names)"`. A silent empty overlay is a footgun.

- [ ] **Step 4: Integration test (CPU) — overlay bytes == full-quant bytes for the same tensor**

Add a test under `crates/hipfire-quantize/tests/reap_overlay_integ.rs` that:
1. Builds a tiny synthetic safetensors file in a tempdir with ~3 named tensors matching a known arch (e.g. `layers.0.ffn.experts.0.w1.weight` shape `[4,256]` fp16, plus a non-matching `layers.0.self_attn.q_proj.weight`).
2. Writes a `reap_plan.json` overriding `layer 0 routed_experts expert 0 → hfq4g256`.
3. Runs `build_overlay` over the tensors and asserts: (a) exactly the 1 matched tensor is emitted, (b) its bytes equal `quantize_hfq4g256(f32_of_that_tensor)` directly, (c) reading the written `overlay.hfq` back via `hipfire_runtime::hfq::HfqFile` yields that tensor by name with the right quant_type.
If constructing a synthetic safetensors is heavy, test `build_overlay` directly with a hand-built tensor iterator (skip the file round-trip for the loop logic; still round-trip `write_hfq`→`HfqFile` for the container).

- [ ] **Step 5: Run + manual smoke**

Run: `cargo test -p hipfire-quantize reap_overlay` (unit + integ). Then a real smoke if a model is handy (CPU): `cargo run --release -p hipfire-quantize -- --reap-overlay <plan-dir> --reap-arch deepseek4 --reap-out /tmp/overlay.hfq <model-dir>` and confirm it writes only the overridden tensors (inspect with `HfqFile::tensor_names()`).

- [ ] **Step 6: Commit**

```bash
git add crates/hipfire-quantize/src/main.rs crates/hipfire-quantize/src/reap_overlay.rs crates/hipfire-quantize/tests/reap_overlay_integ.rs
git commit -m "feat(reap): --reap-overlay mode emits subset overlay.hfq (CPU, byte-verified)"
```

---

### Task 5: Docs

**Files:**
- Modify: `scripts/reap/README.md`

- [ ] **Step 1: Document the overlay workflow**

Add a "Selective re-quant (overlay)" section: the `--reap-overlay` invocation, that it reads the ORIGINAL safetensors (so up-quant recovers precision), the supported tiers (`q8`, `hfq4/6`, `mq4/6`, lloyd variants), that the overlay is consumed at load by SP3 (`HIPFIRE_REAP_PLAN` with an `overlay.hfq` in the dir), and that per-expert-within-a-layer overlays need SP2 to serve (per-layer-uniform serve via existing dispatch).

- [ ] **Step 2: Commit**

```bash
git add scripts/reap/README.md
git commit -m "docs(reap): selective re-quant overlay workflow"
```

---

## Self-Review

**Spec coverage (§3 `reap quant`):**
- Re-quantize only targeted tensors from original safetensors → Task 4 (`build_overlay` decodes/quantizes only matched). ✓
- Write small `overlay.hfq` keyed by original tensor names → Task 4 (`write_hfq` subset). ✓
- Reuse existing per-format encoders → Task 2 (`quantize_to_format` dispatches to `quantize_*`). ✓
- Per-expert + whole-role targeting → Task 3 (`reap_override_for`, `experts` empty = whole role). ✓
- `reap bake` → **explicitly deferred to SP4b** (noted up front). ✓ (gap is intentional, documented)

**Placeholder scan:** Tasks 2/3 say "find the REAL symbol names / adjust to actual API" — these are genuine grep-and-match steps against a known file (`main.rs`), with concrete fallbacks given, not hand-waving. The encoder set in `quantize_to_format` is concrete; lloyd-tier wiring is the one spot needing the implementer to locate the real `quantize_mq3g256_lloyd` entry (flagged with a fallback). Acceptable.

**Type consistency:** `quantize_to_format(name, fmt, &[f32], &[usize]) -> Result<HfqTensor>`, `reap_override_for(name, ReapArch, &ReapPlan) -> Option<&str>`, `ReapArch` enum, `build_overlay(...) -> Vec<HfqTensor>`, `ReapPlan::load_unchecked` — names consistent across Tasks 1–4. `HfqTensor` field names (`name/quant_type/shape/group_size/data/spilled_len`) taken from the real `:5691` push — implementer must confirm.

**Known follow-ups:** SP4b (bake + prune/renumber); imatrix-weighted lloyd tiers in overlays (pass `--imatrix` through); SP3 makes overlays loadable; SP2 makes intra-layer-mixed overlays serveable.
