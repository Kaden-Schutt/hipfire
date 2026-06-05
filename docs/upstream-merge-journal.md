# Upstream Merge Journal — feature/dispatch-unification

**Date:** 2026-06-05  
**Upstream remote:** https://github.com/Kaden-Schutt/hipfire.git  
**Upstream/master HEAD:** `02634f4c` (same as our local master — fully in sync)  
**Our branch HEAD:** `715f966c` refactor(dispatch): route ParoQ4G128 through per-op gemv path (Phase 2c)  
**Commits behind upstream/master:** 0  
**Commits ahead of upstream/master:** 79  

**Key finding:** upstream/master is identical to our local master. The actionable upstream work
lives on `upstream/integration/dispatch-migration` — 7 bug-fix commits targeting the same
dispatch-unification work as PR #393.

---

## Source branch

`upstream/integration/dispatch-migration` (ahead of our branch by 7 commits, behind by 1)

Our branch has **1 commit** not in upstream/integration:

| SHA | Message |
|-----|---------|
| `715f966c` | refactor(dispatch): route ParoQ4G128 through per-op gemv path (Phase 2c) |

---

## Incoming upstream commits (7 total)

| SHA | Message |
|-----|---------|
| `7b35e700` | fix(qwen35): rmsnorm_rotate_dispatch must use MQ8 INT8 path, not FWHT |
| `6e250899` | fix(dispatch): route rotation-free dtypes through plain GEMV in for_gemv_prerotated |
| `b79d9fc6` | fix(runtime): weight_gemv_prerotated double-rotates pre-rotated xr (#393) |
| `3c8a7f4d` | fix(dispatch): restore MoE CPU-top-K generic fallback dropped by #393 |
| `a4e226dd` | fix(gemm,llama): admit gfx12/RDNA4 in 3 WMMA arch gates (dead-gate sweep) |
| `5a9083a3` | fix(dispatch): wire Q8_0 + ParoQ4G128 through residual & prerotated GEMV |
| `6d2500a2` | test(dispatch): (op×dtype×arch) coverage guardrail — catches missing-arm panics at CI |

Also available: `upstream/docs/dispatch-phase0-decisions` (`5b8432cb` docs(dispatch): #397 Phase 0 contract decision spike) — informational doc only, no code.

---

## Dry-run merge result

`git merge --no-commit --no-ff upstream/integration/dispatch-migration`

**Result: Automatic merge — no text conflicts.**  
Git auto-merged `crates/hipfire-arch-qwen35/src/qwen35.rs` and `crates/hipfire-dispatch/src/types.rs`.
All other files are additive (new file or non-overlapping edits).

---

## File-level conflict analysis

| File | Classification | Notes |
|------|---------------|-------|
| `crates/hipfire-dispatch/src/coverage_tests.rs` | UPSTREAM-APPLIES-CLEAN | New file — GPU-free guardrail test suite. Purely additive. |
| `crates/hipfire-dispatch/src/lib.rs` | UPSTREAM-APPLIES-CLEAN | Adds `mod coverage_tests`. No overlap with our changes. |
| `crates/hipfire-dispatch/src/pipeline/steps.rs` | NEEDS-ADAPTATION | Upstream GemvResidual fallback calls `gemv.run(Plain)` which skips Givens rotation for ParoQ4G128. See action item #1. |
| `crates/hipfire-dispatch/src/types.rs` | MERGE-TRIVIAL | Auto-merged. Upstream adds Q8_0 + ParoQ4G128 to `for_gemv_prerotated` and rotation-free dtype fallback. Our commit deleted 3 Fused Paro KernelKey variants. Regions are disjoint. |
| `crates/hipfire-dispatch/src/families/moe.rs` | UPSTREAM-APPLIES-CLEAN | Upstream adds `routed_experts`+`gate_up_buf` to `MoeParams` struct. Our branch didn't change this struct. |
| `crates/hipfire-dispatch/src/pipeline/mod.rs` | UPSTREAM-APPLIES-CLEAN | Upstream adds `run_moe_decode_cpu_fallback`. No overlap with our additions to this file. |
| `crates/hipfire-dispatch/src/families/gemv.rs` | NEEDS-ADAPTATION | Our `dispatch_residual` for ParoQ4G128 (gemv+add_inplace) becomes dead code once steps.rs uses the upstream `else` branch. See action item #1. |
| `crates/hipfire-arch-qwen35/src/qwen35.rs` | MERGE-TRIVIAL | Auto-merged. Upstream adds: MQ8 INT8 early-return in `rmsnorm_rotate_dispatch`, MoE CPU fallback calller + `routed_experts`/`gate_up_buf` population. Our commit: QKV Givens path branching. Regions disjoint. Verify post-merge (see action item #2). |
| `crates/hipfire-runtime/src/llama.rs` | UPSTREAM-APPLIES-CLEAN | Upstream fixes double-rotation (`gemv.run_auto` → `gemv.run(Prerotated)` in `weight_gemv_prerotated`) + gfx12 Q8 prefill gate (`has_wmma_w32()` → `has_wmma()`). Our branch also modified llama.rs (earlier); regions are disjoint, auto-merge correct. |
| `crates/rdna-compute/src/gemm.rs` | UPSTREAM-APPLIES-CLEAN | Upstream widens 2 WMMA gates for HFQ4G256 and HFQ6G256 batched lmhead on gfx12. Our branch didn't touch these sites. |

---

## Required adaptations (action items)

### Action item #1 — `steps.rs` GemvResidual + ParoQ4G128 Givens rotation (BLOCKING)

**File:** `crates/hipfire-dispatch/src/pipeline/steps.rs`

**Problem:** After the merge, the `GemvResidual { input: Raw(x) }` else-branch (for dtypes without a fused residual kernel) calls:
```rust
gemv.run(ctx, gpu, &GemvParams { w, x, y: &tmp, variant: GemvVariant::Plain, ... })
```
`GemvFamily::run(..., Plain)` calls `launch()` directly — it does NOT apply the Givens rotation. For ParoQ4G128, `gemv_hfq4g128` receives un-rotated `fa_attn_out` → wrong output.

The upstream comment ("plain GEMV applies the dtype's own rotation") is only true for Q8_0 (no rotation). It is WRONG for ParoQ4G128.

**Fix:** Replace `gemv.run(...)` in the else branch with `gemv.run_auto()`:

```rust
// Before (upstream else branch — wrong for ParoQ4G128):
gemv.run(ctx, gpu, &GemvParams {
    w, x, y: &tmp, variant: GemvVariant::Plain,
    residual: None, gate: None, up: None,
})?;

// After — run_auto applies Givens rotation internally via run_input(RotInput::Raw):
gemv.run_auto(ctx, gpu, w, x, &tmp)?;
```

**Consequence:** The `ParoQ4G128` arm in `dispatch_residual` (`crates/hipfire-dispatch/src/families/gemv.rs:439-449`) becomes unreachable dead code. Remove it:
```rust
// DELETE this arm from dispatch_residual():
ParoQ4G128 => {
    let tmp = gpu.alloc_tensor(&[m], DType::F32)...
    ...
}
```

**Verification:** After applying, run:
```bash
cargo test -p hipfire-dispatch coverage_tests
```
Then test PARO o_proj decode coherence with an A3B model.

---

### Action item #2 — Verify `MoeParams.routed_experts` population (VERIFY, not blocking)

**File:** `crates/hipfire-arch-qwen35/src/qwen35.rs`

**What to check:** After the merge, the `moe_params` construction (around line ~4601) should include two new fields from upstream `3c8a7f4d`:
```rust
routed_experts: &ffn.experts.iter().map(|e| (e.gate_up.dispatch_ref(), e.down.dispatch_ref())).collect::<Vec<_>>(),
gate_up_buf: s.moe_gate_up_buf.as_ref().expect("MoE scratch"),
```

If the auto-merge placed these correctly, no action needed. If the fields are missing or misplaced (e.g. `routed_experts` uses wrong types), the CPU-top-K fallback will panic at runtime on non-k8 or non-indexable MoE layers.

**Verification:**
```bash
cargo build -p hipfire-arch-qwen35
cargo test -p hipfire-dispatch coverage_tests -- non_k8_and_q8_routed_moe_has_a_dispatch_plan
```

---

## Other upstream branches of interest

These branches were fetched but not analyzed in detail — worth checking before the next rebase:

| Branch | Description |
|--------|-------------|
| `upstream/test/dispatch-coverage-guardrail` | 2 commits: subset of integration/dispatch-migration (commits 6d2500a2 + 5a9083a3) |
| `upstream/docs/dispatch-phase0-decisions` | 1 doc commit: #397 Phase 0 contract decision spike (informational) |
| `upstream/feature/dispatch-unification` | Our branch minus our latest commit — tracking reference |

---

## Merge strategy recommendation

**Preferred: cherry-pick the 7 commits in order, apply action items inline.**

```bash
# Apply in commit order (oldest first):
git cherry-pick 6d2500a2  # test: coverage guardrail
git cherry-pick 5a9083a3  # fix: Q8_0 + Paro residual/prerotated  ← THEN APPLY ACTION ITEM #1
git cherry-pick a4e226dd  # fix: gfx12 WMMA gates
git cherry-pick 3c8a7f4d  # fix: MoE CPU fallback  ← THEN VERIFY ACTION ITEM #2
git cherry-pick b79d9fc6  # fix: double-rotation in weight_gemv_prerotated
git cherry-pick 6e250899  # fix: rotation-free dtypes in for_gemv_prerotated
git cherry-pick 7b35e700  # fix: MQ8 INT8 path in rmsnorm_rotate_dispatch
```

After each cherry-pick: `cargo build -p hipfire-dispatch hipfire-arch-qwen35 hipfire-runtime`.  
After all 7: `cargo test -p hipfire-dispatch coverage_tests` (must pass 5/5).  
After action items applied: run `./scripts/coherence-gate.sh` (required for dispatch changes).
