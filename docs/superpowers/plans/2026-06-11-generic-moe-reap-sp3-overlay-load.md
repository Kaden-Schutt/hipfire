# Generic MoE REAP — SP3: Load-Time Overlay Splice — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a `<HIPFIRE_REAP_PLAN>/overlay.hfq` (built by SP4a) take effect at load: when present, `HfqFile` tensor reads resolve overlay-then-base, so re-quantized tensors splice over the base model transparently — no arch changes.

**Architecture:** Add an optional `overlay: Option<Box<HfqFile>>` to `HfqFile` (`hipfire-runtime`). Its read methods (`find_tensor_info`/`tensor_data`/`tensor_data_pread`/`tensor_data_vec`/`tensor_names`) check the overlay first, fall back to base. `HfqFile::open` auto-attaches `<dir>/overlay.hfq` when `HIPFIRE_REAP_PLAN=<dir>` is set and the overlay's `arch_id` matches the base — so every consumer (config + weight loading, all 4 arches, the keep-map gather) gets overlay resolution with ZERO call-site changes, and an overridden tensor's new `quant_type` flows naturally to dispatch.

**Tech Stack:** Rust, `hipfire-runtime` (`HfqFile`), `hipfire-quantize` (`write_hfq` — for integration tests only). No GPU.

**Spec:** `docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md` §1 (loader resolution rule), §3.

> **Mostly CPU-verifiable.** The overlay-first resolution + env-attach are CPU-unit/integration-testable here. What's GPU-DEFERRED: serving a model whose overlay mixes tiers *within one layer* (needs SP2). A **per-layer-uniform** overlay (a whole layer's experts at a new tier) flows its dtype to `MoeDtypes` and serves through existing dispatch — verify that end-to-end once the GPU frees.

**Relationship to SP1's `TensorSource`:** SP1 added a `TensorSource { base, overlay }` wrapper (overlay always None) as an SP3 seam. This plan implements the overlay at the **`HfqFile` level instead** (lower-touch — no per-arch read rewiring). `TensorSource` becomes vestigial; Task 4 removes it to avoid two overlay mechanisms.

---

### Task 1: `HfqFile` overlay field + overlay-first reads

**Files:**
- Modify: `crates/hipfire-runtime/src/hfq.rs` (struct `:42`, `open_at_offset` `:82`, read methods `:306`–`~485`)
- Test: inline `#[cfg(test)]` in `hfq.rs` (with a minimal in-test HFQ writer)

- [ ] **Step 1: Add the field**

In `pub struct HfqFile` (`:42`), add: `overlay: Option<Box<HfqFile>>,`. In `open_at_offset` (`:82`), initialize `overlay: None` in the constructed `Self { ... }`. Add an accessor + attach:
```rust
impl HfqFile {
    /// Attach an overlay whose tensors shadow this file's by name. Used by the
    /// REAP load-time splice (SP3). Errors if arch_id differs (wrong model).
    pub fn attach_overlay(&mut self, overlay: HfqFile) -> Result<(), String> {
        if overlay.arch_id != self.arch_id {
            return Err(format!(
                "reap overlay: arch_id {} != base arch_id {}", overlay.arch_id, self.arch_id));
        }
        self.overlay = Some(Box::new(overlay));
        Ok(())
    }
    pub fn has_overlay(&self) -> bool { self.overlay.is_some() }
}
```

- [ ] **Step 2: Write the failing resolution tests (with an in-test HFQ writer)**

Add to `hfq.rs` `#[cfg(test)]`. First a minimal writer (the format is: 32B header `HFQM`/version/arch_id/n_tensors/metadata_offset/data_offset, then metadata JSON, then index `[u32 n, per-tensor: u16 name_len, name, u8 quant_type, u8 n_dims, n_dims×u32 shape, u32 group_size, u64 data_size]`, then 4096-aligned data). Write a helper `fn write_min_hfq(path, arch_id, &[(name, quant_type, shape, data)])` that emits a valid container (mirror the real `write_hfq` layout — read `hipfire-quantize`'s `write_hfq` at `main.rs:3398` for the exact byte order; data region 4096-aligned). Then:
```rust
#[test]
fn overlay_tensor_shadows_base() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.hfq");
    let ov = dir.path().join("overlay.hfq");
    // base has tensorA (qt=3) + tensorB (qt=3); overlay re-quantizes tensorA to qt=8.
    write_min_hfq(&base, 9, &[("A", 3, &[2,4], &vec![1u8;  2*4]), ("B", 3, &[2,4], &vec![2u8; 2*4])]);
    write_min_hfq(&ov,   9, &[("A", 8, &[2,4], &vec![9u8;  2*4])]);
    let mut f = HfqFile::open(&base).unwrap();
    f.attach_overlay(HfqFile::open(&ov).unwrap()).unwrap();
    // A resolves to overlay (qt 8, bytes 9); B falls through to base (qt 3, bytes 2).
    let (ia, da) = f.tensor_data_vec("A").unwrap();
    assert_eq!(ia.quant_type, 8); assert!(da.iter().all(|&b| b == 9));
    let (ib, db) = f.tensor_data_vec("B").unwrap();
    assert_eq!(ib.quant_type, 3); assert!(db.iter().all(|&b| b == 2));
    assert_eq!(f.find_tensor_info("A").unwrap().quant_type, 8);
    // tensor_names is the union (base ∪ overlay), no dup.
    let mut names = f.tensor_names(); names.sort();
    assert_eq!(names, vec!["A".to_string(), "B".to_string()]);
}
#[test]
fn overlay_arch_mismatch_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("b.hfq"); let ov = dir.path().join("o.hfq");
    write_min_hfq(&base, 9, &[("A", 3, &[1,4], &vec![0u8;4])]);
    write_min_hfq(&ov,   6, &[("A", 3, &[1,4], &vec![0u8;4])]);
    let mut f = HfqFile::open(&base).unwrap();
    let err = f.attach_overlay(HfqFile::open(&ov).unwrap()).unwrap_err();
    assert!(err.contains("arch_id 6 != base arch_id 9"), "got: {err}");
}
```
Add `tempfile` to `crates/hipfire-runtime` `[dev-dependencies]` if absent.

- [ ] **Step 3: Run to confirm fail**

Run: `cargo test -p hipfire-runtime overlay_` → FAIL (overlay not consulted yet).

- [ ] **Step 4: Implement overlay-first resolution**

In each read method, add an overlay check at the TOP that returns the overlay's result when the overlay has the name:
- `find_tensor_info` (`:306`): `if let Some(ov) = &self.overlay { if let Some(i) = ov.find_tensor_info(name) { return Some(i); } }`
- `tensor_data` (`:311`): same guard, return `ov.tensor_data(name)` if `ov.find_tensor_info(name).is_some()`.
- BOTH `tensor_data_pread` (`:330` unix, `:359` non-unix): same guard → `ov.tensor_data_pread(name)`.
- `tensor_data_vec` (`:368`): same guard → `ov.tensor_data_vec(name)`.
- `tensor_names` (`~:481`): return the UNION of base names + overlay names, deduped (base name list with any overlay-only names appended). Keep deterministic order (base order, then overlay-only sorted) so the test's sort is stable.
Guard each with `self.overlay.is_some()` so the no-overlay path is byte-identical to today (zero overhead).

- [ ] **Step 5: Run tests**

Run: `cargo test -p hipfire-runtime overlay_` → both pass. Then `cargo test -p hipfire-runtime` → no regressions.

- [ ] **Step 6: Commit**

```bash
git add crates/hipfire-runtime/src/hfq.rs crates/hipfire-runtime/Cargo.toml
git commit -m "feat(reap): HfqFile overlay — overlay-then-base tensor resolution"
```

---

### Task 2: Auto-attach the overlay from `HIPFIRE_REAP_PLAN` in `open()`

**Files:**
- Modify: `crates/hipfire-runtime/src/hfq.rs` (`open` `:68`)
- Test: inline

- [ ] **Step 1: Write the failing env-attach test**

```rust
#[test]
fn open_auto_attaches_overlay_from_env() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.hfq");
    write_min_hfq(&base, 9, &[("A", 3, &[1,4], &vec![1u8;4])]);
    // overlay.hfq lives in the plan dir
    let plan = tempfile::tempdir().unwrap();
    write_min_hfq(&plan.path().join("overlay.hfq"), 9, &[("A", 8, &[1,4], &vec![7u8;4])]);
    // Env-scoped: set, open, assert, unset. (Serialize env tests via a mutex if needed.)
    std::env::set_var("HIPFIRE_REAP_PLAN", plan.path());
    let f = HfqFile::open(&base).unwrap();
    std::env::remove_var("HIPFIRE_REAP_PLAN");
    assert!(f.has_overlay());
    assert_eq!(f.find_tensor_info("A").unwrap().quant_type, 8); // overlay won
}
```
(Env mutation in tests is process-global; if `hipfire-runtime` has other env-reading tests, guard with a shared `static MUTEX` or run this test alone. Note it in the test.)

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p hipfire-runtime open_auto_attaches` → FAIL (no auto-attach).

- [ ] **Step 3: Implement auto-attach in `open`**

`open` currently just calls `open_at_offset(path, 0)`. Change it to, after the base opens, check the env and attach (only for the canonical `open`, NOT `open_at_offset`, so embedded-container opens are unaffected):
```rust
pub fn open(path: &Path) -> std::io::Result<Self> {
    let mut f = Self::open_at_offset(path, 0)?;
    if let Ok(dir) = std::env::var("HIPFIRE_REAP_PLAN") {
        let ov_path = std::path::Path::new(&dir).join("overlay.hfq");
        if ov_path.exists() {
            match Self::open_at_offset(&ov_path, 0).map_err(|e| e.to_string())
                .and_then(|ov| { let n = ov.tensors.len(); f.attach_overlay(ov).map(|_| n) }) {
                Ok(n) => eprintln!("reap: overlay ACTIVE — {n} tensor(s) from {ov_path:?} shadow the base"),
                Err(e) => eprintln!("reap: WARNING overlay at {ov_path:?} not attached: {e}"),
            }
        }
    }
    Ok(f)
}
```
(Use `open_at_offset` for the overlay so it does NOT recursively env-attach. `attach_overlay` enforces the arch_id match — a mismatch logs a warning and proceeds base-only, which is the safe default for unrelated model opens.)

- [ ] **Step 4: Run tests**

Run: `cargo test -p hipfire-runtime open_auto_attaches` → pass. `cargo test -p hipfire-runtime` → green.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-runtime/src/hfq.rs
git commit -m "feat(reap): HfqFile::open auto-attaches HIPFIRE_REAP_PLAN/overlay.hfq (arch-guarded)"
```

---

### Task 3: End-to-end integration test (real overlay via SP4)

**Files:**
- Create: `crates/hipfire-quantize/src/reap_overlay.rs` test addition OR `crates/hipfire-quantize` in-module test (binary crate — no `tests/` lib access; mirror SP4 Task 4's in-module `integ` location)

- [ ] **Step 1: Write the test**

Using the real `write_hfq` (in `hipfire-quantize`) and `HfqFile`:
1. Write a `base.hfq` with two tensors (`layers.0.ffn.experts.0.w1.weight` qt Q8F16, and `layers.0.self_attn.q_proj.weight` qt Q8F16) to a tempdir.
2. Build an `overlay.hfq` re-quantizing the expert tensor to HFQ4G256 via `build_overlay` (or `quantize_to_format` directly) + `write_hfq`, placed in a `plan-dir`.
3. Set `HIPFIRE_REAP_PLAN=<plan-dir>`, `HfqFile::open(base)`, assert: the expert tensor resolves to the overlay (quant_type HFQ4G256, overlay bytes) and the attention tensor falls through to base (Q8F16). Unset env.
This proves the SP4-overlay → SP3-load round trip on real containers.

- [ ] **Step 2: Run**

Run: `cargo test -p hipfire-quantize reap_overlay` → all pass incl. the new round-trip.

- [ ] **Step 3: Commit**

```bash
git add crates/hipfire-quantize/src/reap_overlay.rs
git commit -m "test(reap): SP4-overlay -> SP3-load end-to-end (real containers, CPU)"
```

---

### Task 4: Remove the now-redundant `TensorSource`; docs

`TensorSource` (SP1, `crates/hipfire-reap/src/source.rs`) was a placeholder for SP3; the overlay now lives in `HfqFile`. Keeping both is two mechanisms for one job.

**Files:**
- Modify: `crates/hipfire-reap/src/source.rs`, `lib.rs`
- Modify: `scripts/reap/README.md`

- [ ] **Step 1: Remove `TensorSource`, keep `ExpertPlan`**

`source.rs` holds both `TensorSource` and `ExpertPlan` (+ `ReapPlan::expert_plan`). `ExpertPlan` IS used by the arches (SP1); keep it. Delete only `TensorSource` (and its `tensor()`/`overlay` field) and its `pub use` in `lib.rs`. Grep to confirm `TensorSource` has no other users (it should have none — it was inert). Run `cargo test -p hipfire-reap` → still green (ExpertPlan tests unaffected).

- [ ] **Step 2: Document SP3 in README**

In the "Selective re-quant (overlay)" section (added by SP4), update the "Consuming the overlay" note: SP3 is now DONE — `HIPFIRE_REAP_PLAN=<dir>` with an `overlay.hfq` present makes `HfqFile` resolve overlay-then-base automatically (no arch changes); a per-layer-uniform overlay serves via existing dispatch, intra-layer-mixed still needs SP2.

- [ ] **Step 3: Commit**

```bash
git add crates/hipfire-reap/src/source.rs crates/hipfire-reap/src/lib.rs scripts/reap/README.md
git commit -m "refactor(reap): drop vestigial TensorSource (overlay now in HfqFile); SP3 docs"
```

---

## Self-Review

**Spec coverage (§1 resolution rule, §3 overlay consumption):**
- Overlay-then-base tensor resolution → Task 1 (all 5 read methods). ✓
- Activated by `HIPFIRE_REAP_PLAN`/`overlay.hfq` → Task 2 (env auto-attach, arch-guarded). ✓
- Zero arch changes (transparent) → achieved by putting it in `HfqFile` (all arches read through it). ✓
- Overridden tensor's new quant_type reaches dispatch → Task 1 (`find_tensor_info`/`tensor_data*` return overlay `HfqTensorInfo` with its quant_type). ✓
- End-to-end with a real SP4 overlay → Task 3. ✓
- Intra-layer-mixed serving → **GPU-deferred to SP2** (noted). ✓

**Placeholder scan:** Task 1 Step 2 asks the implementer to mirror `write_hfq`'s byte layout for a minimal test writer — concrete (the format is fully specified + the real writer is at a given file:line), not hand-waving. No TBDs.

**Type consistency:** `HfqFile.overlay: Option<Box<HfqFile>>`, `attach_overlay(&mut self, HfqFile) -> Result<(),String>`, `has_overlay()`, read methods unchanged signatures (overlay checked internally). `HfqTensorInfo.quant_type` (u8) used consistently. `write_min_hfq` test helper consistent across Task 1/2 tests.

**Known follow-ups:** SP2 (intra-layer-mixed dispatch) makes per-expert overlays serveable; the per-layer-uniform overlay serve path is GPU-owed to verify; SP4b bake.
