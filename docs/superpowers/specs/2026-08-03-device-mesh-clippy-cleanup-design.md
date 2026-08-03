# Device-Mesh Clippy Cleanup Design

## Context

Commit `60b7f62a` checkpoints the approved Frozen MoE residency work with an
explicit clippy waiver. The current branch has diagnostics both in newly added
code and in unrelated pre-existing regions. This cleanup must remove defects
introduced by the device-mesh work without turning into repository-wide lint
churn.

## Scope

The cleanup covers diagnostics whose primary spans are on lines introduced by
the checkpoint, plus the branch-level `qwen35_tp_preflight` correctness bug in
`crates/hipfire-arch-qwen35/src/arch.rs`.

The current cleanup files are:

- `crates/hipfire-arch-qwen35/src/arch.rs`
- `crates/hipfire-arch-qwen35/src/carrier.rs`
- `crates/hipfire-arch-qwen35/src/qwen35.rs`
- `crates/hipfire-arch-qwen35/src/store.rs`
- `crates/hipfire-runtime/src/llama.rs`
- `crates/hipfire-runtime/src/weight_store.rs`
- `crates/hipfire-loader/src/carriers.rs`
- `crates/hipfire-loader/src/lib.rs`

Diagnostics in untouched regions and the unrelated deny-level
`clippy::erasing_op` in `crates/hipfire-runtime/src/ddtree.rs` are excluded.

## Approach

### Correctness First

Fix `qwen35_tp_preflight` without changing admission policy. Qwen35 has no TP
manifest, so both current layer variants remain refused for `tp > 1`; replace
the never-iterating loop with an explicit first-layer lookup and retain the
existing exact-error tests for both variants. This fix is committed separately
from lint cleanup.

Review each checked-operation diagnostic in `qwen35.rs` for its actual
division, overflow, and fallback semantics. Rewrites must preserve behavior;
the lint suggestion is not itself proof that a replacement is correct.

### Migration Cleanup

Classify new dead methods, fields, imports, and locals using reference searches.
Remove an item only when it is private, unreferenced, and not an intentional
compile-time seam. If an intentional seam is retained, document that intent
with the narrowest justified lint expectation.

### Mechanical Cleanup

Apply local, behavior-preserving simplifications for needless borrows and
dereferences, redundant closures, identity mappings, standard integer helpers,
and unused mutability. Avoid changes outside checkpoint-introduced lines unless
a directly adjacent edit is required for valid code.

### API-Shape Diagnostics

Do not redesign stable signatures, enums, or error ownership solely to satisfy
`too_many_arguments`, `large_enum_variant`, `result_large_err`, or
`type_complexity`. Prefer a narrowly scoped `#[expect(clippy::...)]` with a
reason when the shape is intentional. Refactor only when the existing shape is
itself incorrect or obscures ownership.

## Verification

For each cleanup batch:

1. Run focused tests for the affected crate and behavior.
2. Run scoped `rustfmt --check` on modified Rust files.
3. Capture clippy JSON and fail on diagnostics whose primary spans overlap the
   cleanup diff. Suppress only explicitly catalogued unrelated deny-level
   findings needed to complete analysis.
4. Run `cargo test` and `git diff --check` before committing.

The cleanup does not close STEP-002, STEP-002R, or any missing GPU evidence.
