# Device-Mesh Clippy Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove diagnostics introduced by the Frozen MoE residency checkpoint while preserving Qwen35 TP refusal, ownership semantics, and existing tests.

**Architecture:** Work in four isolated batches ordered by risk. Preserve behavior first, remove provably orphaned migration code second, apply local mechanical simplifications third, and use narrow `#[expect]` annotations only for intentional ownership/API shapes.

**Tech Stack:** Rust 2021, Cargo, Clippy 1.97, rustfmt, crate unit tests, and residency boundary scripts.

---

## File Map

- `crates/hipfire-arch-qwen35/src/arch.rs`: Qwen35 TP admission policy.
- `crates/hipfire-runtime/src/{llama.rs,weight_store.rs}`: generic ownership seams.
- `crates/hipfire-loader/src/{carriers.rs,lib.rs}`: loader transactions and carrier errors.
- `crates/hipfire-arch-qwen35/src/{carrier.rs,store.rs}`: Frozen publication and projections.
- `crates/hipfire-arch-qwen35/src/qwen35.rs`: assembly, arithmetic guards, and bindings.

### Task 1: Preserve Qwen35 TP Refusal Without a Never-Loop

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/arch.rs:55-78`
- Test: `crates/hipfire-arch-qwen35/src/arch.rs:594-615`

- [ ] **Step 1: Reproduce the deny-level lint**

Run `cargo clippy -p hipfire-arch-qwen35 --lib --no-deps --message-format=short`.
Expected: exit 101 with `clippy::never_loop` at `arch.rs:68`.

- [ ] **Step 2: Replace the loop with explicit first-layer refusal**

```rust
if let Some(layer_type) = cfg.layer_types.first() {
    let reason = match layer_type {
        LayerType::LinearAttention => "DeltaNet wqkv",
        LayerType::FullAttention => "packed QGate wq",
    };
    return Err(format!(
        "qwen35: Tp={tp} is unsupported for {reason} (layer 0)"
    ));
}
Ok(())
```

- [ ] **Step 3: Verify exact refusal behavior**

Run:

```bash
cargo test -p hipfire-arch-qwen35 tp_preflight_ --lib
cargo clippy -p hipfire-arch-qwen35 --lib --no-deps --message-format=short
```

Expected: all three TP preflight tests pass and `never_loop` is absent.

- [ ] **Step 4: Commit**

```bash
git add crates/hipfire-arch-qwen35/src/arch.rs
git commit -m "fix(qwen35): preserve TP refusal without never-loop" \
  -m "Assisted-by: OpenCode:openai/gpt-5.6-sol"
```

### Task 2: Remove Runtime and Loader Migration Residue

**Files:**
- Modify: `crates/hipfire-runtime/src/llama.rs`
- Modify: `crates/hipfire-runtime/src/weight_store.rs`
- Modify: `crates/hipfire-loader/src/carriers.rs`
- Modify: `crates/hipfire-loader/src/lib.rs`

- [ ] **Step 1: Record the 16 checkpoint diagnostics**

```bash
cargo clippy -p hipfire-runtime -p hipfire-loader --all-targets --no-deps \
  --message-format=short -- -A clippy::erasing_op
```

Confirm eight findings in `weight_store.rs`, six in loader `lib.rs`, one dead
`free_checked` in `llama.rs`, and one large carrier enum in `carriers.rs`.

- [ ] **Step 2: Prove dead private methods have no consumers**

Run `rg -n "free_checked|into_parts|extract_origin_map" crates`.
Remove `free_checked`, `into_parts`, and `extract_origin_map` only when the
search shows no production or test consumer; remove type assertions that exist
only to name a deleted private method.

- [ ] **Step 3: Apply local cleanup**

Use direct results instead of identity maps:

```rust
// Before
result.map(|value| value)
// After
result
```

Remove `mut` from checkpoint bindings reported at runtime
`weight_store.rs:704,6415,6791` and loader `lib.rs:8108`. Remove unused locals
`entries` and `enqueuer` when their initializers are side-effect free. Rename
an intentionally discarded error binding to `_e` only when the branch still
needs pattern structure.

- [ ] **Step 4: Document intentional representation sizes narrowly**

```rust
#[expect(
    clippy::large_enum_variant,
    reason = "variants retain concrete rollback owners until publication"
)]
```

For explicit staged-owner tuple signatures:

```rust
#[expect(
    clippy::type_complexity,
    reason = "the tuple preserves each staged owner for exact rollback"
)]
```

- [ ] **Step 5: Verify and commit**

```bash
cargo test -p hipfire-runtime weight_store --lib
cargo test -p hipfire-loader --lib
rustfmt --edition 2021 --check --config skip_children=true \
  crates/hipfire-runtime/src/llama.rs crates/hipfire-runtime/src/weight_store.rs \
  crates/hipfire-loader/src/carriers.rs crates/hipfire-loader/src/lib.rs
git add crates/hipfire-runtime/src/llama.rs crates/hipfire-runtime/src/weight_store.rs \
  crates/hipfire-loader/src/carriers.rs crates/hipfire-loader/src/lib.rs
git commit -m "refactor(runtime): clean frozen owner migration residue" \
  -m "Assisted-by: OpenCode:openai/gpt-5.6-sol"
```

Expected: focused tests and formatting pass before the commit.

### Task 3: Clean Qwen35 Owner and Store Surfaces

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/carrier.rs`
- Modify: `crates/hipfire-arch-qwen35/src/store.rs`

- [ ] **Step 1: Classify private dead symbols**

Reference-search `set_mtp_head`, `is_legacy`, metadata helpers near
`store.rs:3013`, transaction count helpers near `store.rs:3311`,
`is_moe_entry`, `expert_tags`, and `Res`. Delete symbols with no consumer.
Retain a compile-time seam only with:

```rust
#[expect(dead_code, reason = "compile-time ownership seam exercised by type tests")]
```

- [ ] **Step 2: Remove unused imports, locals, and mutability**

Remove checkpoint-only imports around `store.rs:5867-5870`, side-effect-free
unused locals at `store.rs:3967,3969,4121,7986,7993,8804,9295`, and `mut` from
carrier bindings at `carrier.rs:609,636`.

- [ ] **Step 3: Apply local iterator simplifications**

```rust
// Before
slice.iter().copied().collect::<Vec<_>>()
// After
slice.to_vec()

// Before
values.len() == 0
// After
values.is_empty()
```

Replace redundant closures only when the function item has the identical
ownership and error signature.

- [ ] **Step 4: Handle intentional API-shape diagnostics**

Add item-level `#[expect]` for remaining `large_enum_variant`,
`result_large_err`, `type_complexity`, and `too_many_arguments` findings only
after dead-code cleanup. Each reason must name retained failed-free ownership,
one-shot assembly inputs, or another concrete invariant.

- [ ] **Step 5: Verify and commit**

```bash
cargo test -p hipfire-arch-qwen35 store:: --lib
cargo test -p hipfire-arch-qwen35 carrier:: --lib
bash scripts/check_moe_residency_boundary.sh
git add crates/hipfire-arch-qwen35/src/carrier.rs \
  crates/hipfire-arch-qwen35/src/store.rs
git commit -m "refactor(qwen35): clean frozen owner surfaces" \
  -m "Assisted-by: OpenCode:openai/gpt-5.6-sol"
```

Expected: focused tests and the boundary script pass.

### Task 4: Clean Qwen35 Forward and Arithmetic Diagnostics

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs`

- [ ] **Step 1: Remove proven orphaned migration functions**

Reference-check and remove uncalled private checkpoint functions:
`repack_awq_to_hfq4g128`, `load_paroquant_weight`,
`load_fp16_weight_from_source`, `paro_repack_moe_projection`,
`paro_load_moe_shared_sidecars`, `alias_paro_rotation`, `paro_load_moe_ffn`,
`load_any_as_f32`, `load_raw_f32`, `moe_ffn_decode`,
`ffn_gate_side_mq4_for_moe`, `moe_ffn_decode_with_scratch_inner`, and
`Qwen35Scratch::build_error`. Retain a real type-test seam only with
`#[expect(dead_code, reason = "compile-time ownership seam exercised by type tests")]`.

- [ ] **Step 2: Rewrite 12 manually checked divisions**

At the checkpoint sites around lines 4635-4972, preserve integer types,
fallbacks, and equality targets while replacing:

```rust
if divisor != 0 && dividend / divisor == expected {
    // existing body
}
```

with:

```rust
if dividend.checked_div(divisor) == Some(expected) {
    // existing body
}
```

- [ ] **Step 3: Apply mechanical helpers in small batches**

Use `.is_multiple_of()`, `.div_ceil()`, range `.contains()`, `is_some_and`, and
iterator `.flatten()` only at checkpoint-introduced sites. Apply compiler
suggestions for needless borrows only when the callee signature is unchanged.
Remove side-effect-free unused bindings and unit-valued test bindings.

- [ ] **Step 4: Document intentional ABI and owner shapes**

After mechanical cleanup, add item-level expectations for remaining
`too_many_arguments`, `large_enum_variant`, and `type_complexity` findings.
Reasons must identify kernel ABI mirroring, complete scratch construction, or
owner-preserving error transport.

- [ ] **Step 5: Verify and commit**

```bash
cargo test -p hipfire-arch-qwen35 --lib
rustfmt --edition 2021 --check --config skip_children=true \
  crates/hipfire-arch-qwen35/src/qwen35.rs
bash scripts/check_moe_residency_boundary.sh
git add crates/hipfire-arch-qwen35/src/qwen35.rs
git commit -m "refactor(qwen35): resolve introduced clippy diagnostics" \
  -m "Assisted-by: OpenCode:openai/gpt-5.6-sol"
```

Expected: Qwen35 tests, formatting, and the boundary script pass.

### Task 5: Final Changed-Line Gate

**Files:** Verify only; modify no production files.

- [ ] **Step 1: Run scoped formatting**

```bash
rustfmt --edition 2021 --check --config skip_children=true \
  crates/hipfire-arch-qwen35/src/arch.rs crates/hipfire-arch-qwen35/src/carrier.rs \
  crates/hipfire-arch-qwen35/src/qwen35.rs crates/hipfire-arch-qwen35/src/store.rs \
  crates/hipfire-runtime/src/llama.rs crates/hipfire-runtime/src/weight_store.rs \
  crates/hipfire-loader/src/carriers.rs crates/hipfire-loader/src/lib.rs
```

- [ ] **Step 2: Run affected clippy targets**

```bash
cargo clippy -p hipfire-arch-qwen35 -p hipfire-runtime -p hipfire-loader \
  --all-targets --no-deps --message-format=json -- -A clippy::erasing_op
```

Expected: exit zero. Review primary spans and confirm no diagnostic lands on
lines changed after checkpoint `60b7f62a`; catalogued pre-existing warnings may
remain elsewhere.

- [ ] **Step 3: Run behavioral and boundary verification**

```bash
cargo test
bash scripts/check_moe_residency_boundary.sh
bash scripts/check_moe_residency_boundary.sh --self-test
git diff --check
```

Expected: full tests pass; both boundary commands exit zero; diff check emits no
output. STEP-002 remains open because this plan adds no missing GPU evidence.
