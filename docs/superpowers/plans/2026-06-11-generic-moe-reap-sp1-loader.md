# Generic MoE REAP — Sub-project 1: Generic Keep-Map Loader — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the DeepSeek-V4-specific REAP keep-map loader into a model-agnostic `hipfire-reap` crate and wire it into all 4 MoE arches present on the base (`deepseek4`, `qwen35`, `lfm2moe`, `minimax`), preserving the byte-identical default-off / keep-all-identity guarantee.

**Architecture:** A new `hipfire-reap` crate owns plan parsing (`ReapPlan`), an overlay-then-base tensor resolver (`TensorSource`), an exact byte row-gather (`gather_rows`), and a per-(layer,role) `ExpertPlan` the arch loaders consume at their existing expert-enumeration seam. Arch-specific oddities (deepseek4 `tid2eid`/MTP) stay in the arch crate behind a `ReapArchHook`. Quant-override/overlay *application* and mixed dispatch are SP2–SP4; this plan ships pruning + the overlay-aware resolver plumbing only (tier table is parsed and threaded, but every tier resolves to the base dtype until SP2).

**Tech Stack:** Rust, `serde_json`, the in-repo `hipfire-runtime` (`HfqFile`, `Gpu`, `DType`, `GpuTensor`). Tests via `cargo test -p hipfire-reap` and arch-level identity NLL checks via existing `examples/deepseek4_perplexity.rs`.

**Spec:** `docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md` (§1, §4 SP1 gates).

> ⛔ **GPU EMBARGO (2026-06-11):** the GPU is in use by separate cohere work. Do **NOT** run any
> step that loads a model on the GPU: no `examples/daemon`, no perplexity/NLL/PPL/KLD runs, no
> `cargo run`, and no `cargo test` on the arch crates (they may init the GPU). Allowed: editing
> code, `cargo build -p <crate>` (compile only — kernels JIT at runtime, not build time), and
> `cargo test -p hipfire-reap` (pure-CPU unit tests). Steps marked **[GPU — DEFERRED]** must be
> left unchecked with a note; implement the code they verify, but do not execute them.

**Scope note:** `cohere2moe` is in-flight on `nw_cohere2moe_support` and not on this base; it gets the identical Task-9-style wiring once merged. SP2 (mixed dispatch), SP3 (overlay application), SP4 (bake) get their own plans authored after this lands.

---

### Task 1: Scaffold the `hipfire-reap` crate

**Files:**
- Create: `crates/hipfire-reap/Cargo.toml`
- Create: `crates/hipfire-reap/src/lib.rs`
- Modify: `Cargo.toml` (workspace `members`)

- [ ] **Step 1: Create the crate manifest**

`crates/hipfire-reap/Cargo.toml`:
```toml
[package]
name = "hipfire-reap"
version = "0.1.0"
edition = "2021"

[dependencies]
hipfire-runtime = { path = "../hipfire-runtime" }
rdna_compute = { path = "../../rdna_compute" }
serde_json = "1"

[dev-dependencies]
tempfile = "3"
```
(Verify the exact `rdna_compute` relative path and `serde_json`/`tempfile` versions against a sibling crate's `Cargo.toml`, e.g. `crates/hipfire-arch-deepseek4/Cargo.toml`; match whatever the workspace already pins.)

- [ ] **Step 2: Create a placeholder lib so the crate compiles**

`crates/hipfire-reap/src/lib.rs`:
```rust
//! Model-agnostic REAP: selective expert pruning + (SP2+) selective re-quant
//! overlay for MoE models. See docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md
```

- [ ] **Step 3: Register the crate in the workspace**

In the root `Cargo.toml` `[workspace] members = [...]` list, add `"crates/hipfire-reap"` (alphabetically near the other `crates/hipfire-*` entries). Confirm the list style (some workspaces glob `crates/*`; if so, no edit needed — check first).

- [ ] **Step 4: Verify it builds**

Run: `cargo build -p hipfire-reap`
Expected: compiles clean (an unused-crate warning is fine).

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-reap/Cargo.toml crates/hipfire-reap/src/lib.rs Cargo.toml
git commit -m "feat(reap): scaffold hipfire-reap crate"
```

---

### Task 2: `ReapPlan` — parse & validate `reap_plan.json`

This generalizes `ReapKeepMap::load` (currently `crates/hipfire-arch-deepseek4/src/deepseek4.rs:256-328`) — same validation, but model-agnostic and extended with the optional `quant_overrides` manifest. Pruning is optional (`keep` may be absent ⇒ keep-all).

**Files:**
- Create: `crates/hipfire-reap/src/plan.rs`
- Modify: `crates/hipfire-reap/src/lib.rs` (add `pub mod plan;`)
- Test: inline `#[cfg(test)]` in `plan.rs`

- [ ] **Step 1: Write the failing tests**

Add `pub mod plan;` to `lib.rs`, then create `crates/hipfire-reap/src/plan.rs`:
```rust
use std::path::{Path, PathBuf};

/// One selective-requant edit (applied in SP2+; parsed & validated now).
#[derive(Debug, Clone, PartialEq)]
pub struct QuantOverride {
    pub layer: usize,
    pub role: Role,
    /// Only meaningful for `Role::RoutedExperts`; empty ⇒ whole role at this layer.
    pub experts: Vec<u32>,
    pub tier: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    RoutedExperts,
    SharedExpert,
    Attention,
    Router,
    LmHead,
    Embed,
}

impl Role {
    pub fn parse(s: &str) -> Result<Role, String> {
        Ok(match s {
            "routed_experts" => Role::RoutedExperts,
            "shared_expert" => Role::SharedExpert,
            "attention" => Role::Attention,
            "router" => Role::Router,
            "lm_head" => Role::LmHead,
            "embed" => Role::Embed,
            other => return Err(format!("reap: unknown role '{other}'")),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ReapPlan {
    pub model_arch: Option<String>,
    pub num_layers: usize,
    pub original_experts: usize,
    /// `keep[l][slot]` = original expert index in compact slot `slot`.
    /// `None` ⇒ no pruning (keep all `original_experts`).
    pub keep: Option<Vec<Vec<u32>>>,
    pub quant_overrides: Vec<QuantOverride>,
    pub dir: PathBuf,
}

impl ReapPlan {
    pub fn kept_per_layer(&self) -> usize {
        match &self.keep {
            Some(k) => k.first().map(|r| r.len()).unwrap_or(0),
            None => self.original_experts,
        }
    }

    /// Load `<dir>/reap_plan.json`, validating against the model's layer/expert
    /// counts (passed BEFORE any n_routed_experts override).
    pub fn load(
        dir: &str,
        num_layers_expected: usize,
        orig_experts_expected: usize,
    ) -> Result<Self, String> {
        let path = Path::new(dir).join("reap_plan.json");
        let txt = std::fs::read_to_string(&path)
            .map_err(|e| format!("reap: read {path:?}: {e}"))?;
        let v: serde_json::Value =
            serde_json::from_str(&txt).map_err(|e| format!("reap: parse {path:?}: {e}"))?;

        let original_experts = v["original_experts"]
            .as_u64()
            .unwrap_or(orig_experts_expected as u64) as usize;
        if original_experts != orig_experts_expected {
            return Err(format!(
                "reap: original_experts {original_experts} != model n_routed_experts {orig_experts_expected}"
            ));
        }
        let num_layers = v["num_layers"].as_u64().unwrap_or(num_layers_expected as u64) as usize;
        if num_layers != num_layers_expected {
            return Err(format!(
                "reap: num_layers {num_layers} != model num_hidden_layers {num_layers_expected}"
            ));
        }

        let keep = match v["keep"]["per_layer"].as_array() {
            None => None,
            Some(arr) => {
                if arr.len() != num_layers_expected {
                    return Err(format!(
                        "reap: keep.per_layer has {} layers, model has {num_layers_expected}",
                        arr.len()
                    ));
                }
                let kept = arr.first().and_then(|r| r.as_array()).map(|r| r.len()).unwrap_or(0);
                let mut out = Vec::with_capacity(arr.len());
                for (l, row) in arr.iter().enumerate() {
                    let r = row
                        .as_array()
                        .ok_or_else(|| format!("reap: keep layer {l} not an array"))?;
                    if r.len() != kept {
                        return Err(format!(
                            "reap: keep layer {l} has {} entries, expected {kept}",
                            r.len()
                        ));
                    }
                    let mut v32 = Vec::with_capacity(kept);
                    for x in r {
                        let idx = x
                            .as_u64()
                            .ok_or_else(|| format!("reap: keep layer {l} non-integer index"))?
                            as u32;
                        if idx as usize >= original_experts {
                            return Err(format!(
                                "reap: keep layer {l} index {idx} >= original_experts {original_experts}"
                            ));
                        }
                        v32.push(idx);
                    }
                    out.push(v32);
                }
                Some(out)
            }
        };

        let mut quant_overrides = Vec::new();
        if let Some(arr) = v["quant_overrides"].as_array() {
            for (i, o) in arr.iter().enumerate() {
                let layer = o["layer"]
                    .as_u64()
                    .ok_or_else(|| format!("reap: quant_override[{i}] missing layer"))?
                    as usize;
                if layer >= num_layers_expected {
                    return Err(format!(
                        "reap: quant_override[{i}] layer {layer} >= num_layers {num_layers_expected}"
                    ));
                }
                let role = Role::parse(
                    o["role"].as_str().ok_or_else(|| format!("reap: quant_override[{i}] missing role"))?,
                )?;
                let experts: Vec<u32> = o["experts"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect())
                    .unwrap_or_default();
                if !experts.is_empty() && role != Role::RoutedExperts {
                    return Err(format!(
                        "reap: quant_override[{i}] lists experts but role is not routed_experts"
                    ));
                }
                let tier = o["tier"]
                    .as_str()
                    .ok_or_else(|| format!("reap: quant_override[{i}] missing tier"))?
                    .to_string();
                quant_overrides.push(QuantOverride { layer, role, experts, tier });
            }
        }

        Ok(ReapPlan {
            model_arch: v["model_arch"].as_str().map(|s| s.to_string()),
            num_layers: num_layers_expected,
            original_experts,
            keep,
            quant_overrides,
            dir: PathBuf::from(dir),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_plan(json: &str) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        let mut f = std::fs::File::create(d.path().join("reap_plan.json")).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        d
    }

    #[test]
    fn keep_all_when_keep_absent() {
        let d = write_plan(r#"{"original_experts":8,"num_layers":2}"#);
        let p = ReapPlan::load(d.path().to_str().unwrap(), 2, 8).unwrap();
        assert!(p.keep.is_none());
        assert_eq!(p.kept_per_layer(), 8);
    }

    #[test]
    fn parses_keep_and_overrides() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":2,
                "keep":{"per_layer":[[0,2,3],[1,2,3]]},
                "quant_overrides":[{"layer":1,"role":"routed_experts","experts":[2],"tier":"mq3lloyd"}]}"#,
        );
        let p = ReapPlan::load(d.path().to_str().unwrap(), 2, 4).unwrap();
        assert_eq!(p.kept_per_layer(), 3);
        assert_eq!(p.keep.as_ref().unwrap()[0], vec![0, 2, 3]);
        assert_eq!(p.quant_overrides.len(), 1);
        assert_eq!(p.quant_overrides[0].tier, "mq3lloyd");
    }

    #[test]
    fn rejects_out_of_range_index() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":1,"keep":{"per_layer":[[0,9]]}}"#,
        );
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("index 9 >= original_experts 4"), "got: {err}");
    }

    #[test]
    fn rejects_experts_on_non_routed_role() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":1,
                "quant_overrides":[{"layer":0,"role":"attention","experts":[1],"tier":"q8"}]}"#,
        );
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("not routed_experts"), "got: {err}");
    }

    #[test]
    fn rejects_layer_count_mismatch() {
        let d = write_plan(r#"{"original_experts":4,"num_layers":3,"keep":{"per_layer":[[0,1]]}}"#);
        let err = ReapPlan::load(d.path().to_str().unwrap(), 3, 4).unwrap_err();
        assert!(err.contains("keep.per_layer has 1 layers"), "got: {err}");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail (compile-first)**

Run: `cargo test -p hipfire-reap plan::`
Expected: passes once written — but first confirm `tempfile` is a dev-dep (Task 1). If it doesn't compile, fix the manifest, not the test.

- [ ] **Step 3: (impl already inline above) Run tests to verify they pass**

Run: `cargo test -p hipfire-reap plan::`
Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/hipfire-reap/src/plan.rs crates/hipfire-reap/src/lib.rs
git commit -m "feat(reap): ReapPlan json parse + validation"
```

---

### Task 3: `gather_rows` — exact byte row-gather

Generalizes `upload_quant_or_f16_keep`'s gather math (`crates/hipfire-arch-deepseek4/src/arch.rs:384-424`) into a pure, GPU-free function returning gathered bytes + new shape. Exact for row-independent quant (F16/Q8/MQ*-G256); rows that don't divide the byte length are a hard error.

**Files:**
- Create: `crates/hipfire-reap/src/gather.rs`
- Modify: `crates/hipfire-reap/src/lib.rs` (`pub mod gather;`)

- [ ] **Step 1: Write the failing tests**

`crates/hipfire-reap/src/gather.rs`:
```rust
/// Gather kept rows from a row-major tensor's raw bytes. `shape[0]` is the
/// row count (e.g. experts); every row must be `bytes.len()/shape[0]` bytes.
/// Returns `(new_shape, gathered_bytes)`. Exact for row-independent quant.
pub fn gather_rows(shape: &[usize], bytes: &[u8], keep: &[u32]) -> Result<(Vec<usize>, Vec<u8>), String> {
    let orig_rows = *shape.first().unwrap_or(&0);
    if orig_rows == 0 || bytes.len() % orig_rows != 0 {
        return Err(format!(
            "reap: row-gather: {orig_rows} rows don't divide {} bytes",
            bytes.len()
        ));
    }
    let rowstride = bytes.len() / orig_rows;
    let mut out = Vec::with_capacity(rowstride * keep.len());
    for &oe in keep {
        let oe = oe as usize;
        if oe >= orig_rows {
            return Err(format!("reap: row-gather keep idx {oe} >= rows {orig_rows}"));
        }
        out.extend_from_slice(&bytes[oe * rowstride..(oe + 1) * rowstride]);
    }
    let mut new_shape = shape.to_vec();
    new_shape[0] = keep.len();
    Ok((new_shape, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gathers_subset_in_order() {
        // 4 rows × 3 bytes
        let bytes: Vec<u8> = vec![0,0,0, 1,1,1, 2,2,2, 3,3,3];
        let (shape, out) = gather_rows(&[4, 3], &bytes, &[2, 0, 3]).unwrap();
        assert_eq!(shape, vec![3, 3]);
        assert_eq!(out, vec![2,2,2, 0,0,0, 3,3,3]);
    }

    #[test]
    fn identity_keep_is_byte_identical() {
        let bytes: Vec<u8> = (0..24).collect();
        let (_, out) = gather_rows(&[4, 6], &bytes, &[0, 1, 2, 3]).unwrap();
        assert_eq!(out, bytes);
    }

    #[test]
    fn errors_on_indivisible_rows() {
        let err = gather_rows(&[3, 2], &[0, 1, 2, 3, 4], &[0]).unwrap_err();
        assert!(err.contains("don't divide"), "got: {err}");
    }

    #[test]
    fn errors_on_out_of_range_keep() {
        let err = gather_rows(&[2, 2], &[0, 1, 2, 3], &[5]).unwrap_err();
        assert!(err.contains("keep idx 5 >= rows 2"), "got: {err}");
    }
}
```
Add `pub mod gather;` to `lib.rs`.

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p hipfire-reap gather::`
Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/hipfire-reap/src/gather.rs crates/hipfire-reap/src/lib.rs
git commit -m "feat(reap): exact byte row-gather primitive"
```

---

### Task 4: `TensorSource` + `ExpertPlan` — the loader-facing API

`TensorSource` wraps a base `&HfqFile` (overlay handle is `None` in SP1; the field exists so SP3 slots in without changing call sites). `ExpertPlan` is what an arch asks for per (layer, role): the optional keep slice and a tier resolver (always base dtype in SP1).

**Files:**
- Create: `crates/hipfire-reap/src/source.rs`
- Modify: `crates/hipfire-reap/src/lib.rs` (`pub mod source;`, re-exports)

- [ ] **Step 1: Write the source module + a unit test for the keep accessor**

`crates/hipfire-reap/src/source.rs`:
```rust
use crate::plan::ReapPlan;
use hipfire_runtime::hfq::HfqFile; // adjust path to wherever HfqFile is exported

/// Overlay-then-base tensor resolver. SP1: overlay is always None (base only);
/// SP3 adds the overlay HfqFile and prefers it when it holds `name`.
pub struct TensorSource<'a> {
    pub base: &'a HfqFile,
    pub overlay: Option<&'a HfqFile>,
}

impl<'a> TensorSource<'a> {
    pub fn new(base: &'a HfqFile) -> Self {
        TensorSource { base, overlay: None }
    }

    /// Resolve a tensor by name: overlay first (SP3), else base.
    pub fn tensor(&self, name: &str) -> Option<(hipfire_runtime::hfq::TensorInfo, Vec<u8>)> {
        if let Some(ov) = self.overlay {
            if let Some(hit) = ov.tensor_data_pread(name) {
                return Some(hit);
            }
        }
        self.base.tensor_data_pread(name)
    }
}

/// Per-(layer, role) plan slice the arch loader consumes at its expert loop.
pub struct ExpertPlan<'a> {
    /// `keep[slot]` = original expert index for compact slot. `None` ⇒ identity.
    keep: Option<&'a [u32]>,
}

impl<'a> ExpertPlan<'a> {
    pub fn keep(&self) -> Option<&'a [u32]> {
        self.keep
    }
    /// Original expert index for a compact slot (identity when no keep map).
    pub fn src(&self, slot: usize) -> usize {
        self.keep.map(|k| k[slot] as usize).unwrap_or(slot)
    }
    /// Number of compact expert slots for this layer.
    pub fn n_slots(&self, full: usize) -> usize {
        self.keep.map(|k| k.len()).unwrap_or(full)
    }
}

impl ReapPlan {
    /// Build the per-layer expert plan (routed experts). `None`-keep ⇒ identity.
    pub fn expert_plan(&self, layer: usize) -> ExpertPlan<'_> {
        ExpertPlan {
            keep: self.keep.as_ref().map(|k| k[layer].as_slice()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ExpertPlan::src/n_slots are pure; test them without an HfqFile.
    #[test]
    fn identity_src_when_no_keep() {
        let ep = ExpertPlan { keep: None };
        assert_eq!(ep.src(5), 5);
        assert_eq!(ep.n_slots(8), 8);
    }
    #[test]
    fn remaps_src_with_keep() {
        let k = vec![3u32, 1, 0];
        let ep = ExpertPlan { keep: Some(&k) };
        assert_eq!(ep.src(0), 3);
        assert_eq!(ep.n_slots(8), 3);
    }
}
```
**Before running:** confirm the real module paths for `HfqFile`, `TensorInfo`, and `tensor_data_pread`'s return type by grepping the deepseek4 loader (it calls `hfq.tensor_data_pread(&name)` returning `Option<(info, bytes)>`). Fix the `use` and the `tensor()` signature to match the actual types (the return is likely `(&TensorInfo, Vec<u8>)` or owned — match it; the test only exercises the pure `ExpertPlan` methods, so it compiles regardless of the exact HfqFile signature once the `use` resolves).

Add `pub mod source;` and `pub use source::{ExpertPlan, TensorSource};` to `lib.rs`.

- [ ] **Step 2: Run tests**

Run: `cargo test -p hipfire-reap source::`
Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/hipfire-reap/src/source.rs crates/hipfire-reap/src/lib.rs
git commit -m "feat(reap): TensorSource + ExpertPlan loader API"
```

---

### Task 5: `ReapArchHook` trait for arch-specific extras

Lets the arch crate inject its own sidecar handling (deepseek4 `tid2eid` remap, MTP-skip) without `hipfire-reap` knowing about it.

**Files:**
- Create: `crates/hipfire-reap/src/hook.rs`
- Modify: `crates/hipfire-reap/src/lib.rs`

- [ ] **Step 1: Define the trait**

`crates/hipfire-reap/src/hook.rs`:
```rust
use crate::plan::ReapPlan;

/// Arch-specific REAP extras. Default impls are no-ops so arches that need
/// nothing (qwen35, lfm2moe, minimax) don't implement anything.
pub trait ReapArchHook {
    /// Path to an arch sidecar file inside the plan dir, if the arch uses one.
    fn sidecar_path(&self, plan: &ReapPlan, name: &str) -> std::path::PathBuf {
        plan.dir.join(name)
    }
    /// Whether this layer's auxiliary head (e.g. ds4 MTP) is skipped under reap.
    fn skip_aux_head(&self, _plan: &ReapPlan, _layer: usize) -> bool {
        false
    }
}
```
Add `pub mod hook;` and `pub use hook::ReapArchHook;` to `lib.rs`.

- [ ] **Step 2: Verify build**

Run: `cargo build -p hipfire-reap`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/hipfire-reap/src/hook.rs crates/hipfire-reap/src/lib.rs
git commit -m "feat(reap): ReapArchHook trait for arch-specific extras"
```

---

### Task 6: Re-point deepseek4 onto `hipfire-reap` (the lift)

Replace the in-crate `ReapKeepMap` + `HIPFIRE_DEEPSEEK4_REAP_KEEPMAP` path with `ReapPlan`/`HIPFIRE_REAP_PLAN`, keeping the old env var as a back-compat alias that loads a `keep`-only plan. The existing `upload_quant_or_f16_keep`/`upload_global_f16_as_f32_keep` GPU helpers stay (they wrap `gather_rows`'s math + an upload); we only swap their byte-gather core to call `hipfire_reap::gather::gather_rows` so there is one source of truth.

**Files:**
- Modify: `crates/hipfire-arch-deepseek4/Cargo.toml` (add `hipfire-reap` dep)
- Modify: `crates/hipfire-arch-deepseek4/src/deepseek4.rs:100-236` (env hook, field type) and `:242-328` (delete `ReapKeepMap`)
- Modify: `crates/hipfire-arch-deepseek4/src/arch.rs:384-424` (gather core → `gather_rows`)

- [ ] **Step 1: Add the dependency**

In `crates/hipfire-arch-deepseek4/Cargo.toml` `[dependencies]`, add:
```toml
hipfire-reap = { path = "../hipfire-reap" }
```

- [ ] **Step 2: Swap the env hook (deepseek4.rs ~229-236)**

Replace the block that reads `HIPFIRE_DEEPSEEK4_REAP_KEEPMAP` and builds `ReapKeepMap` with one that prefers `HIPFIRE_REAP_PLAN` and falls back to the old var:
```rust
// REAP plan: emulate a pruned expert pool by partial-load. New generic env
// HIPFIRE_REAP_PLAN=<dir> (reap_plan.json); legacy HIPFIRE_DEEPSEEK4_REAP_KEEPMAP
// (keep_by_layer.json) still honored as a keep-only alias.
let reap_dir = std::env::var("HIPFIRE_REAP_PLAN")
    .ok()
    .or_else(|| std::env::var("HIPFIRE_DEEPSEEK4_REAP_KEEPMAP").ok());
if let Some(dir) = reap_dir {
    let plan = hipfire_reap::plan::ReapPlan::load_any(
        &dir,
        config.num_hidden_layers,
        config.n_routed_experts,
    )?;
    eprintln!(
        "deepseek4: REAP plan ACTIVE — keeping {} of {} routed experts/layer; dir {dir}",
        plan.kept_per_layer(),
        config.n_routed_experts
    );
    config.n_routed_experts = plan.kept_per_layer();
    config.reap_keep = Some(std::sync::Arc::new(plan));
}
```
Change the `reap_keep` field type (deepseek4.rs ~109) from `Option<Arc<ReapKeepMap>>` to `Option<Arc<hipfire_reap::plan::ReapPlan>>`. Update all readers of `.reap_keep` (the arch.rs upload sites and the `tid2eid_path` caller) to use `plan.keep` / the `ExpertPlan` accessors and the `ReapArchHook` for `tid2eid_path` (move `tid2eid_path` onto a small ds4 `ReapArchHook` impl).

- [ ] **Step 3: Add `ReapPlan::load_any` (keep-only-alias loader)**

In `crates/hipfire-reap/src/plan.rs`, add a helper that accepts either `reap_plan.json` or a legacy `keep_by_layer.json` in the dir:
```rust
impl ReapPlan {
    /// Load `reap_plan.json` if present, else a legacy `keep_by_layer.json`
    /// (keep-only; no overrides). Lets old ds4 sidecars keep working.
    pub fn load_any(
        dir: &str,
        num_layers_expected: usize,
        orig_experts_expected: usize,
    ) -> Result<Self, String> {
        if Path::new(dir).join("reap_plan.json").exists() {
            return Self::load(dir, num_layers_expected, orig_experts_expected);
        }
        Self::load_legacy_keepmap(dir, num_layers_expected, orig_experts_expected)
    }
}
```
Add `load_legacy_keepmap` that reads `keep_by_layer.json` (the old schema: `kept_per_layer`, `original_experts`, `keep`) and produces a `ReapPlan` with `keep: Some(...)`, `quant_overrides: vec![]`. Reuse the existing validation logic. Add a unit test mirroring an existing legacy sidecar shape.

- [ ] **Step 4: Swap the gather core in arch.rs (~404-412)**

Inside `upload_quant_or_f16_keep`, replace the manual `rowstride`/loop with:
```rust
let shape_usize: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
let (new_shape, sub) = hipfire_reap::gather::gather_rows(&shape_usize, &bytes, keep)?;
```
then upload `sub`/`new_shape` as before, preserving the `info.quant_type` → `t.dtype` mapping. Do the analogous swap is NOT needed for `upload_global_f16_as_f32_keep` (it decodes f16→f32 element-wise; leave it, or refactor to call `gather_rows` then decode — optional, keep behavior identical).

- [ ] **Step 5a: Build + CPU tests (run now)**

Run: `cargo build -p hipfire-arch-deepseek4 && cargo test -p hipfire-reap`
Expected: compiles; `hipfire-reap` unit tests (incl. the new `load_legacy_keepmap` test) pass.

- [ ] **Step 5b: [GPU — DEFERRED] ds4 identity NLL gate**

The identity NLL gate (existing keep-all-256 sidecar): confirm a keep-all `reap_plan.json` reproduces the no-plan baseline NLL to 10 decimals (use `scripts/reap/build_keepall_sidecar.py` regenerated to emit `reap_plan.json`, or point `HIPFIRE_DEEPSEEK4_REAP_KEEPMAP` at the legacy sidecar to exercise `load_legacy_keepmap`).
**Do NOT run under the GPU embargo** — leave unchecked; note "deferred: GPU in use". The lift must be behavior-preserving; this is the gate to run once the GPU frees.

- [ ] **Step 6: Commit**

```bash
git add crates/hipfire-arch-deepseek4 crates/hipfire-reap
git commit -m "refactor(ds4): lift REAP onto hipfire-reap; HIPFIRE_REAP_PLAN + legacy alias"
```

---

### Task 7: Wire `qwen35` (buffer arch)

Buffer arch: experts are individual `ExpertWeights { gate_up, down }` (`crates/hipfire-arch-qwen35/src/qwen35.rs:4331-4346`). Pruning = iterate compact slots and load only kept experts by remapped name; router logits get a `gather_rows` on the per-expert rows.

**Files:**
- Modify: `crates/hipfire-arch-qwen35/Cargo.toml` (add `hipfire-reap`)
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs` (MoE loader ~4320-4380, config/env hook, router load)

- [ ] **Step 1: Add dependency + thread the plan**

Add `hipfire-reap = { path = "../hipfire-reap" }`. Locate where qwen35 reads its config/builds weights; add the same `HIPFIRE_REAP_PLAN` env read as ds4 (Task 6 Step 2), storing `Option<Arc<ReapPlan>>` on the config/weights struct, and override the routed-expert count (`n_routed_experts`/equivalent) to `plan.kept_per_layer()` when active.

- [ ] **Step 2: Write the failing identity test**

Add an arch-level test (or reuse the qwen35 perplexity/eval example) asserting a keep-all plan reproduces baseline logits for a small qwen35-MoE fixture. If no tiny fixture exists, gate this as a manual NLL check documented in the task and add a `#[test]` that at least exercises the loader path with a keep-all plan on a toy-sized config (skip if no fixture; note it).

- [ ] **Step 3: Implement the loop change**

Replace the expert loop (`for x in 0..n_exp`) with compact-slot iteration:
```rust
let plan = config.reap_keep.as_ref();
let ep = plan.map(|p| p.expert_plan(layer_idx));
let n_slots = ep.as_ref().map(|e| e.n_slots(n_exp)).unwrap_or(n_exp);
for slot in 0..n_slots {
    let x = ep.as_ref().map(|e| e.src(slot)).unwrap_or(slot);
    let gate_up = load_weight_tensor(hfq, gpu, &format!("{p}.mlp.experts.{x}.gate_up_proj.weight"), 2 * mi, config.dim)?;
    let down = load_weight_tensor(hfq, gpu, &format!("{p}.mlp.experts.{x}.down_proj.weight"), config.dim, mi)?;
    experts.push(ExpertWeights { gate_up, down });
}
```
For the **router** (`{p}.mlp.gate.weight` or equivalent, `[n_exp, dim]`): when `ep` has a keep, read its bytes, `gather_rows(&[n_exp, dim_stride], &bytes, keep)`, and upload the gathered subset so the router emits logits only for kept experts. (Find the exact router tensor name in the qwen35 loader and match its existing upload.)

- [ ] **Step 4: Build (run now) + [GPU — DEFERRED] identity check**

Run: `cargo build -p hipfire-arch-qwen35` (compile only).
The keep-all NLL check is **[GPU — DEFERRED]** under the embargo — leave unchecked, note "deferred: GPU in use".

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-arch-qwen35
git commit -m "feat(qwen35): generic REAP keep-map loader wiring"
```

---

### Task 8: Wire `lfm2moe` (buffer arch, fused-at-load gate_up)

Same buffer-arch pattern as qwen35, but gate_up is fused from `w1`/`w3` at load (`crates/hipfire-arch-lfm2moe/src/lfm2moe.rs:345-454`) and per-layer AWQ scales hang off `experts[0]`.

**Files:**
- Modify: `crates/hipfire-arch-lfm2moe/Cargo.toml` (add `hipfire-reap`)
- Modify: `crates/hipfire-arch-lfm2moe/src/lfm2moe.rs:345-454` + config/env hook

- [ ] **Step 1: Add dependency + env hook + count override** (as Task 7 Step 1).

- [ ] **Step 2: Convert the expert loop to compact slots**

In the `MoeFfnWeights::load` expert loop, replace `e` with `slot`/`src` exactly as Task 7 Step 3, applying `src` to the `{prefix}.feed_forward.experts.{src}.{w1,w3,w2}.weight` names. Keep the w1‖w3 fuse unchanged (it operates per expert). **AWQ caveat:** the shared per-layer AWQ scale read from `experts[0]` must read from `experts[src(0)]`'s scale tensor — confirm the AWQ scale is layer-shared (same for all experts) or per-expert; if per-expert, gather it under keep. Document which, in the commit.

- [ ] **Step 3: Router gather** as Task 7 Step 3 (find lfm2moe's router/gate tensor).

- [ ] **Step 4: Build (run now) + [GPU — DEFERRED] identity check**

Run: `cargo build -p hipfire-arch-lfm2moe` (compile only).
Keep-all NLL check is **[GPU — DEFERRED]** — leave unchecked, note "deferred: GPU in use".

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-arch-lfm2moe
git commit -m "feat(lfm2moe): generic REAP keep-map loader wiring"
```

---

### Task 9: Wire `minimax` (blob arch, like deepseek4)

Blob arch: all experts packed into single `gu_combined`/`dn_combined` blobs with a stride pointer table (`crates/hipfire-arch-minimax/src/minimax.rs:280-517`). Pruning = pack only kept experts' bytes via `src(slot)`, mirroring ds4's `upload_layer_routed_experts` change.

**Files:**
- Modify: `crates/hipfire-arch-minimax/Cargo.toml` (add `hipfire-reap`)
- Modify: `crates/hipfire-arch-minimax/src/minimax.rs:280-517` + config/env hook

- [ ] **Step 1: Add dependency + env hook + count override** (as Task 7 Step 1). MiniMax already has EP-shard logic; assert reap and EP-shard are mutually exclusive (mirror ds4 arch.rs's guard: error if both a keep plan and a shard config are present).

- [ ] **Step 2: Pack only kept experts**

In the blob-build loops (the `for e in 0..n_exp` that `extend_from_slice` w1/w3/w2 bytes), iterate compact slots and read `experts.{src(slot)}`:
```rust
let ep = config.reap_keep.as_ref().map(|p| p.expert_plan(layer_idx));
let n_slots = ep.as_ref().map(|e| e.n_slots(n_exp)).unwrap_or(n_exp);
for slot in 0..n_slots {
    let e = ep.as_ref().map(|x| x.src(slot)).unwrap_or(slot);
    // ... read experts.{e}.w1/w3/w2, extend into gu_combined/dn_combined ...
}
```
The stride pointer table is then built over `n_slots` compact entries (unchanged math). **EP-shard:** since reap and EP-shard are mutually exclusive (Step 1), the non-owned-zero path is untouched when reap is active.

- [ ] **Step 3: Router gather** (minimax router/gate `[n_exp, dim]`) as Task 7 Step 3.

- [ ] **Step 4: Build (run now) + [GPU — DEFERRED] identity check**

Run: `cargo build -p hipfire-arch-minimax` (compile only).
Keep-all NLL check is **[GPU — DEFERRED]** — leave unchecked, note "deferred: GPU in use".

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-arch-minimax
git commit -m "feat(minimax): generic REAP keep-map loader wiring"
```

---

### Task 10: Cross-arch regression gate + end-to-end smoke

**Files:**
- Create: `crates/hipfire-reap/tests/identity_gate.md` (documented manual gate runner) or extend `scripts/reap/`
- Modify: `scripts/reap/build_keepall_sidecar.py` → emit `reap_plan.json` (generic)

- [ ] **Step 1: Generalize the keep-all sidecar builder**

Update `scripts/reap/build_keepall_sidecar.py` to write a generic `reap_plan.json` with `keep.per_layer = [[0..N-1]] * num_layers` for a given (num_layers, n_experts), arch-agnostic.

- [ ] **Step 2: [GPU — DEFERRED] Run the identity gate on every wired arch**

For each of ds4, qwen35, lfm2moe, minimax with an available MoE checkpoint: load with `HIPFIRE_REAP_PLAN=<keepall dir>` and confirm logits/NLL match the no-plan baseline to 10 decimals. Record results in `scripts/reap/README.md`.
**Do NOT run under the GPU embargo** — leave unchecked, note "deferred: GPU in use". Expected (when run): all 4 reproduce baseline (exact no-op under keep-all).

- [ ] **Step 3: [GPU — DEFERRED] ds4 end-to-end smoke**

Reproduce the known K144 result through the new generic path: full-256 PPL ≈ 7.56 vs pruned-144 ≈ 17.73 on wikitext2 ctx=1024 (existing `scripts/reap/run_ppl_kld.sh`, regenerated keep-map as `reap_plan.json`).
**Do NOT run under the GPU embargo** — leave unchecked, note "deferred: GPU in use". Expected (when run): matches pre-lift numbers — no regression.

- [ ] **Step 4: Commit the (Python/script) changes**

Step 1 (CPU, the `build_keepall_sidecar.py` generalization) is runnable now; Steps 2–3 are deferred. Commit what's done:
```bash
git add scripts/reap crates/hipfire-reap/tests
git commit -m "test(reap): generalize keep-all sidecar builder; cross-arch identity gate (GPU-deferred)"
```

---

## Self-Review

**Spec coverage (§1, §4-SP1):**
- `hipfire-reap` crate + `ReapPlan`/`TensorSource`/`gather_rows`/`ExpertPlan`/`ReapArchHook` → Tasks 1–5. ✓
- Overlay-then-base resolver present but overlay=None until SP3 → `TensorSource.overlay` field, Task 4. ✓
- Blob-arch per-tier sub-blob packing → **deferred to SP2** (this plan packs kept experts into the *existing single* blob; multi-tier sub-blobs need the dtype table from SP2). Noted in Architecture. ✓ (pruning-only is correct for SP1.)
- All 4 base arches wired → Tasks 6–9; cohere2moe on-merge noted. ✓
- Byte-identical / keep-all identity gate → Tasks 6 Step 5, 10. ✓
- `gather_rows` exactness + `ReapPlan` validation unit tests → Tasks 2, 3. ✓
- Legacy env alias → Task 6 Step 3. ✓

**Placeholder scan:** Tasks 7–9 Step 2 reference "find the exact router tensor name" / "if no fixture exists" — these are genuine per-arch lookups the implementer must do at the seam; each gives the file:line and the transform pattern with concrete code, not hand-waving. Acceptable (the unknown is a tensor *name* in a known file, not missing logic).

**Type consistency:** `ReapPlan` (fields `keep: Option<Vec<Vec<u32>>>`, `quant_overrides`, `dir`), `ExpertPlan::{src,n_slots,keep}`, `gather_rows(shape,bytes,keep)->(Vec<usize>,Vec<u8>)`, `TensorSource::{new,tensor,overlay}`, `ReapArchHook` — names used consistently across Tasks 4–10. `config.reap_keep: Option<Arc<ReapPlan>>` used identically in Tasks 6–9. ✓

**Known follow-ups (SP2+):** mixed-tier sub-blob packing, `per_expert_quant` dtype table, overlay application (`TensorSource.overlay = Some(...)`), bake. Each gets its own plan.
