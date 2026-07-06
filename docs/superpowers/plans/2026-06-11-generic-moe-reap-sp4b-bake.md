# Generic MoE REAP — SP4b: Bake to Standalone .hfq — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `hipfire reap bake`: a full-model quantize from the original safetensors that applies a reap plan's `quant_overrides` per-tensor AND (optionally) prunes/renumbers experts per the plan's keep-map, producing a standalone `.hfq` that serves through the normal load path with NO env var — the "freeze once happy" step.

**Architecture:** Reuse the existing whole-model quantize loop in `hipfire-quantize`. Two additive hooks: (1) an **override hook** at the top of the per-tensor loop body — if `reap_override_for(name)` returns a tier, quantize via `quantize_to_format` (SP4a) and skip the arch's default branch; otherwise fall through to the normal quantize. (2) a **prune hook** — for routed-expert tensors, skip experts not in `keep[L]` and rename kept experts to compact slots; gather router/per-expert-bias rows to the kept set. The expert count in the output metadata is reduced to the kept count. Unlike SP4a's overlay (subset only), bake emits the WHOLE model.

**Tech Stack:** Rust, `hipfire-quantize` (CPU), `hipfire-reap` (`ReapPlan`, `reap_override_for`, `quantize_to_format`, `gather_rows`). No GPU.

**Spec:** `docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md` §3 (`reap bake`).

> **CPU-verifiable.** Anchor test: a bake with NO overrides + NO keep (keep-all) must be **byte-identical** to the normal `--format <base>` quantize of the same model (bake is a pure superset hook). Override and prune are verified by targeted tests. Serving a baked model whose experts mix tiers within a layer still needs SP2 to run on GPU; a uniform-tier or per-layer-tier baked model serves through existing dispatch.

**Scope notes:**
- **ds4 hash-layer `tid2eid` remap under prune is DEFERRED** (a follow-up): pruning a deepseek4 model's hash layers (0–2) requires rewriting `tid2eid` to the compact expert space (the logic exists in `scripts/reap/build_reap_keepmap.py`). This plan prunes score-layer experts + non-ds4 arches correctly; a ds4 bake with `keep` errors out on hash layers with a clear "tid2eid remap not yet supported in bake — use load-time keep-map" message rather than producing a wrong model. Override-bake (no prune) works for ds4.
- Imatrix-weighted lloyd override tiers: `quantize_to_format` covers the self-calibrating tiers; an override to an imatrix-calibrated tier is out of scope (overlay/bake both use the self-calibrating encoders).

---

### Task 1: `--reap-bake` mode + per-tensor override application (no prune yet)

**Files:**
- Modify: `crates/hipfire-quantize/src/main.rs` (arg parse near `--reap-overlay`; override hook in the loop at `:5628`)
- Modify: `crates/hipfire-quantize/src/reap_overlay.rs` (rename module concept is fine; add a `bake`-mode flag or reuse helpers)
- Test: in-module test in `reap_overlay.rs`

- [ ] **Step 1: Add the CLI arg**

Near the `--reap-overlay` parsing, add `--reap-bake <plan-dir>` (mutually exclusive with `--reap-overlay`; error if both). When set, the normal quantize runs to completion BUT with the override hook active and the output written to the normal `--format` output path (or a `--reap-out` path if you reuse that arg). Arch detection reuses `ReapArch::from_arch_id`/`from_flag` (SP4a).

- [ ] **Step 2: Write the failing tests**

In `reap_overlay.rs` tests, add a unit test for the override-decision helper (pure): the bake override hook should, for a tensor matching the plan, return the override tier, and for a non-matching tensor return None (so it flows to normal quantize). This is exactly `reap_override_for` (already tested), so the NEW test is the bake-loop behavior. Since the loop is in `main.rs` (binary, hard to unit-test directly), add an in-module integration test that drives a small `bake_tensors` helper:
```rust
/// Apply bake to a stream of (name, shape, f32) tensors: overridden tensors are
/// quantized to their tier; non-overridden tensors are quantized to `base_fmt`.
/// (Pruning is Task 2 — this version emits every tensor.)
pub fn bake_tensors(
    arch: ReapArch, plan: &ReapPlan, base_fmt: &str,
    tensors: impl Iterator<Item = (String, Vec<usize>, Vec<f32>)>,
) -> Result<Vec<HfqTensor>, String> {
    let mut out = Vec::new();
    for (name, shape, f32) in tensors {
        let fmt = reap_override_for(&name, arch, plan).unwrap_or(base_fmt);
        out.push(quantize_to_format(&name, fmt, &f32, &shape)?);
    }
    Ok(out)
}
#[cfg(test)]
mod bake_tests {
    use super::*;
    #[test]
    fn no_override_bakes_all_at_base_fmt() {
        let p = /* plan_with empty quant_overrides */;
        let tensors = vec![("layers.0.self_attn.q_proj.weight".into(), vec![1,256], vec![0.3f32;256])];
        let out = bake_tensors(ReapArch::Qwen35, &p, "q8", tensors.into_iter()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].quant_type, QuantType::Q8F16);
        // == direct q8 of the same data
        assert_eq!(out[0].data, quantize_to_format("x","q8",&vec![0.3f32;256],&[1,256]).unwrap().data);
    }
    #[test]
    fn override_wins_over_base_fmt() {
        let p = /* plan_with override: layer 0 attention -> hfq6 */;
        let tensors = vec![("layers.0.self_attn.q_proj.weight".into(), vec![2,256], vec![0.1f32;512])];
        let out = bake_tensors(ReapArch::Qwen35, &p, "q8", tensors.into_iter()).unwrap();
        assert_eq!(out[0].quant_type, QuantType::HFQ6G256); // override, not base q8
    }
}
```
(Use the real `plan_with` helper / `ReapPlan::load_unchecked` from a tempdir as the SP4a tests do. Match `QuantType` variant names.)

- [ ] **Step 3: Run to confirm fail → implement `bake_tensors` → pass**

Run: `cargo test -p hipfire-quantize reap_overlay` → the bake tests pass.

- [ ] **Step 4: Wire the override hook into the real main.rs loop**

At the TOP of the `for (name, file_idx) in &all_tensors` loop body (`:5628`), after the skip filters but BEFORE the arch-specific branches, add (only when bake mode is active):
```rust
if let Some(plan) = &reap_bake_plan {
    if let Some(fmt) = reap_overlay::reap_override_for(name, reap_arch, plan) {
        let (meta, raw) = st_files[*file_idx].tensor_data(name).unwrap();
        let f32 = tensor_to_f32_with_optional_fp8_scale(name, raw, meta, &fp8_scale_for, &st_files);
        let shape: Vec<usize> = meta.shape.iter().map(|&s| s as usize).collect();
        hfq_tensors.push(reap_overlay::quantize_to_format(name, fmt, &f32, &shape)?);
        st_files[*file_idx].drop_tensor_pages(name);
        continue; // skip the normal arch branch for this overridden tensor
    }
}
```
Non-overridden tensors fall through to the unchanged normal quantize. The final `write_hfq` writes the whole model to the output path. **Default mode (no `--reap-bake`) is untouched** (the hook is behind `if let Some(plan) = &reap_bake_plan`).

- [ ] **Step 5: Anchor test — no-override bake == normal quantize**

Add an in-module integration test: write a tiny synthetic safetensors (or drive the loop's tensor source) for ~2 tensors; run the bake path with an EMPTY-overrides plan and `base_fmt = q8`; assert the resulting `hfq_tensors` bytes equal the normal `--format q8` quantize of the same tensors. (If driving the real main loop in a test is impractical, assert via `bake_tensors` with empty overrides == direct `quantize_to_format(base_fmt)` for each — which the `no_override_bakes_all_at_base_fmt` test already covers at the helper level; note the main-loop hook is exercised by the manual smoke in Step 6.)

- [ ] **Step 6: Manual smoke (CPU, if a model is handy)**

`cargo run --release -p hipfire-quantize -- --reap-bake <plan-dir> --reap-arch qwen35 --format q8 --reap-out /tmp/baked.hfq <model-dir>`; open `/tmp/baked.hfq` with `HfqFile`, confirm the overridden tensors carry the override quant_type and the rest are q8. Note in report whether a model was available.

- [ ] **Step 7: Commit**

```bash
git add crates/hipfire-quantize/src/main.rs crates/hipfire-quantize/src/reap_overlay.rs
git commit -m "feat(reap): --reap-bake applies quant_overrides over a full quantize (CPU)"
```

---

### Task 2: Expert pruning + renumber in bake

**Files:**
- Modify: `crates/hipfire-quantize/src/reap_overlay.rs` (a name-rewrite + keep helper)
- Modify: `crates/hipfire-quantize/src/main.rs` (prune hook in the loop; reduce expert count in output metadata)
- Test: in-module

- [ ] **Step 1: Write the failing tests for the prune/rename helper**

```rust
/// For a routed-expert tensor under an active keep, decide: drop it, or emit it
/// renamed to its compact slot. Returns None to drop, Some(new_name) to keep.
/// `keep_l` is keep[layer] (original expert indices in compact-slot order).
pub fn bake_expert_rename(name: &str, arch: ReapArch, layer: usize, keep_l: &[u32]) -> Option<String> {
    // parse the expert index E from `name` (arch seg, as reap_override_for does);
    // if E in keep_l at position `slot`, return name with E replaced by `slot`;
    // else None (drop).
}
#[cfg(test)]
mod prune_tests {
    use super::*;
    #[test]
    fn renames_kept_expert_to_compact_slot() {
        // keep[0] = [0,2,3]; original expert 2 -> compact slot 1
        let n = "layers.0.ffn.experts.2.w1.weight";
        assert_eq!(bake_expert_rename(n, ReapArch::Deepseek4, 0, &[0,2,3]),
                   Some("layers.0.ffn.experts.1.w1.weight".to_string()));
    }
    #[test]
    fn drops_pruned_expert() {
        let n = "layers.0.ffn.experts.5.w1.weight";
        assert_eq!(bake_expert_rename(n, ReapArch::Deepseek4, 0, &[0,2,3]), None);
    }
}
```

- [ ] **Step 2: Implement `bake_expert_rename`**

Reuse the arch expert-segment + index parse from `reap_override_for`/`tensor_matches` (factor the "extract expert index from name for arch" into a shared `fn expert_index_of(name, arch) -> Option<u32>` so both use it — DRY). Compute `slot = keep_l.iter().position(|&e| e == E)`; if Some, rebuild the name replacing the `E` token with `slot`; if None, return None.

- [ ] **Step 3: Wire the prune hook into the loop**

In the bake branch of the main loop, BEFORE the override hook, for routed-expert tensors when `plan.keep` is Some: determine the tensor's layer (parse `layers.{L}.`) and call `bake_expert_rename`. If None → `continue` (drop; `drop_tensor_pages`). If `Some(new_name)` → use `new_name` as the output `HfqTensor.name` (the override/normal quantize still reads the ORIGINAL tensor bytes, but writes under the compact name). **ds4 hash-layer guard:** if `arch == Deepseek4` and the layer is a hash layer (0,1,2) and `plan.keep` is Some, return a hard error ("tid2eid remap not supported in bake — see scope note") rather than emitting a wrong hash layer.

- [ ] **Step 4: Router + per-expert-bias gather under prune**

For the router/gate weight (`[n_exp, dim]`) and any per-expert bias (`[n_exp]`) at a layer with an active keep: gather the kept rows via `hipfire_reap::gather::gather_rows(&shape, &raw_or_f32_bytes, keep_l)` before quantizing, so the baked router emits logits only for kept experts (mirror the loader's gather, but at quantize time). Quantize the gathered tensor normally. Add a test that the gathered router has `keep_l.len()` rows.

- [ ] **Step 5: Reduce expert count in output metadata**

The output `.hfq` metadata JSON carries the model config (incl. `n_routed_experts`/`num_experts`). Under an active keep, patch that field to `kept_per_layer` in the metadata before `write_hfq`, so the baked model loads with the compact expert count (no env var, no keep-map needed at load). Find where `metadata_json` is built and apply the patch (string/JSON edit of the experts field). Add a test asserting the patched metadata has the reduced count.

- [ ] **Step 6: Run tests**

Run: `cargo test -p hipfire-quantize reap_overlay` → all prune tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/hipfire-quantize/src/main.rs crates/hipfire-quantize/src/reap_overlay.rs
git commit -m "feat(reap): bake prunes+renumbers kept experts, gathers router/bias, patches expert count"
```

---

### Task 3: Docs

**Files:**
- Modify: `scripts/reap/README.md`

- [ ] **Step 1: Document bake**

Add a "Bake (freeze) — SP4b" subsection under the overlay section: `hipfire-quantize --reap-bake <plan-dir> --format <base> --reap-out final.hfq <safetensors-dir>` produces a standalone `.hfq` that serves with NO env var — `quant_overrides` applied per-tensor, kept experts pruned+renumbered, expert count patched into metadata. Note the ds4-hash-`tid2eid` prune limitation (use load-time keep-map for pruned ds4 hash layers) and that intra-layer-mixed-tier baked models need SP2 to serve.

- [ ] **Step 2: Commit**

```bash
git add scripts/reap/README.md
git commit -m "docs(reap): bake-to-standalone workflow (SP4b)"
```

---

## Self-Review

**Spec coverage (§3 `reap bake`):**
- Full-model quantize with per-tensor overrides → Task 1 (override hook). ✓
- Pruned experts dropped + renumbered to compact slots → Task 2 (`bake_expert_rename` + loop hook). ✓
- Router/bias gathered to kept set → Task 2 Step 4. ✓
- Standalone servable .hfq (expert count in metadata) → Task 2 Step 5. ✓
- Arch sidecars folded in → **ds4 tid2eid prune DEFERRED** (hard-error guard, scope note). ✓ (documented gap)
- Anchor: no-override+keep-all bake == normal quantize → Task 1 Step 5 (+ helper test). ✓

**Placeholder scan:** Task 1 Step 2 / Task 2 Step 1 test bodies use `/* plan_with ... */` placeholders for plan construction — the implementer fills these with the real `ReapPlan::load_unchecked(tempdir)` pattern already established in SP4a's tests (concrete, same module). Acceptable (references an existing, working helper). No TBDs in logic.

**Type consistency:** `bake_tensors(ReapArch, &ReapPlan, base_fmt, iterator) -> Result<Vec<HfqTensor>>`, `bake_expert_rename(name, ReapArch, layer, &[u32]) -> Option<String>`, `expert_index_of(name, ReapArch) -> Option<u32>` (shared with `reap_override_for`), reuse `quantize_to_format`/`reap_override_for`/`gather_rows`. Consistent with SP4a symbols.

**Known follow-ups:** ds4 hash-layer `tid2eid` remap in bake; imatrix-calibrated override tiers; GPU serve-verify of baked uniform-tier models; SP2 for intra-layer-mixed serve.
