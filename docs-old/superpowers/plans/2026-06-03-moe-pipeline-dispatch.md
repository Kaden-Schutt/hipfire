# MoE Pipeline Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the qwen35 MoE decode path as a composable pipeline driven by a single typed eligibility-resolution struct, so the dispatch layer (not the model) owns expert-GEMM kernel selection — closing PR #393 item #2 and unblocking #9.

**Architecture:** Phase 1 only — the **decode** path (`moe_ffn_decode_impl`). First extract the fused-eligibility lattice (qwen35.rs:4598-4671) into a pure, unit-tested `MoeResolution` in the dispatch crate (the load-bearing piece per adversarial review finding #1). Then migrate `PipelineParams` to an enum, add MoE `PipelineOp` variants + a generic `MoeParams`, and relocate the existing `gpu.*` call sequence into an `execute_pipeline` MoE arm keyed by `MoeResolution`. Every GPU-touching step is gated on **byte-identical output** vs the pre-refactor commit.

**Tech Stack:** Rust workspace; crates `hipfire-dispatch` (GPU-independent dispatch logic + `rdna_compute::Gpu` calls) and `hipfire-arch-qwen35` (the model). Tests: `cargo test -p hipfire-dispatch` (pure, no GPU); byte-parity via `./scripts/coherence-gate.sh` + a fresh-process A/B probe.

**Out of scope (Phase 2, separate plan):** the grouped-expert HFQ4G256 kernel (#6), `MoeFamily::run()` wrapper (#7), the batched-prefill path, deepseek4 routing (needs a k=6 kernel-family port — review finding #2).

---

## File Structure

- `crates/hipfire-dispatch/src/families/moe.rs` — add `MoeDtypes`, `MoeResolution` (pure resolve), grow `MoeParams`; later the executor entry.
- `crates/hipfire-dispatch/src/types.rs` — add MoE `PipelineOp` variants.
- `crates/hipfire-dispatch/src/pipeline/mod.rs` — rename `PipelineParams` struct → `LinearParams`; add `PipelineParams` enum; add the MoE executor arm.
- `crates/hipfire-dispatch/src/tests.rs` — unit tests for `MoeResolution` (truth table) and the op-list `can_satisfy`.
- `crates/hipfire-dispatch/src/families/gemv.rs:274-278` — wrap construction in `PipelineParams::Linear(..)`.
- `crates/hipfire-arch-qwen35/src/qwen35.rs:4563-5143` — `moe_ffn_decode_impl`: source booleans from `MoeResolution` (Task 2), then marshal into `MoeParams` + call the executor (Task 5).

---

## Task 1: Pure eligibility lattice `MoeResolution`

The fused-vs-fallback choices in `moe_ffn_decode_impl` are a coupled lattice (review #1): e.g. `use_gpu_topk` depends on the *routed-expert* dtype, not on top-k alone. Extract them into one pure, GPU-free struct so the coupling lives in one tested place.

**Files:**
- Modify: `crates/hipfire-dispatch/src/families/moe.rs`
- Test: `crates/hipfire-dispatch/src/tests.rs`

- [ ] **Step 1: Write the failing test** (append to `crates/hipfire-dispatch/src/tests.rs`)

```rust
// ── MoeResolution eligibility lattice (mirrors qwen35.rs:4598-4671) ──
use crate::families::moe::{MoeDtypes, MoeResolution};

fn dtypes_all_mq4() -> MoeDtypes {
    MoeDtypes {
        router: DType::MQ4G256,
        shared_gate: DType::MQ4G256,
        shared_expert_gate: DType::MQ4G256,
        shared_expert_up: DType::MQ4G256,
        experts_all_gate_up_mq4: true,
        routed_gate_up: DType::MQ4G256,
        routed_down: DType::MQ4G256,
        has_paro_shared: false,
    }
}

#[test]
fn moe_res_all_mq4_k8_uses_gpu_topk_and_xrot() {
    let r = MoeResolution::resolve(&dtypes_all_mq4(), 8);
    assert!(r.gate_side_mq4);
    assert!(r.routed_indexable_mq4);
    assert!(r.use_gpu_topk);
    assert!(r.needs_x_rot_local);
}

#[test]
fn moe_res_q8_router_still_gpu_topk() {
    // The non-obvious coupling: a Q8 router disqualifies the 4-way fused
    // gate-side GEMV (gate_side_mq4=false) but the routed experts are still
    // MQ4, so the device-side top-K + indexed path stays on (use_gpu_topk=true).
    let mut d = dtypes_all_mq4();
    d.router = DType::Q8_0;
    d.experts_all_gate_up_mq4 = true; // experts unchanged
    let r = MoeResolution::resolve(&d, 8);
    assert!(!r.gate_side_mq4);
    assert!(r.routed_indexable_mq4);
    assert!(r.use_gpu_topk);
    assert!(r.needs_x_rot_local); // routed_gate_up_mq4 alone fires x_rot
}

#[test]
fn moe_res_k6_disables_gpu_topk_even_when_indexable() {
    // deepseek-shaped: indexable routed dtype but k != 8 => no GPU fast path
    // (the k=8 fused kernel family doesn't cover it). Review finding #2.
    let r = MoeResolution::resolve(&dtypes_all_mq4(), 6);
    assert!(r.routed_indexable_mq4);
    assert!(!r.use_gpu_topk);
}

#[test]
fn moe_res_mq6_routed_indexable() {
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::MQ6G256;
    d.routed_down = DType::MQ6G256;
    let r = MoeResolution::resolve(&d, 8);
    assert!(r.routed_indexable_mq6);
    assert!(!r.routed_indexable_mq4);
    assert!(r.use_gpu_topk);
}

#[test]
fn moe_res_paro_needs_sidecar() {
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::ParoQ4G128;
    d.routed_down = DType::ParoQ4G128;
    d.has_paro_shared = false;
    assert!(!MoeResolution::resolve(&d, 8).routed_indexable_paro);
    d.has_paro_shared = true;
    let r = MoeResolution::resolve(&d, 8);
    assert!(r.routed_indexable_paro);
    assert!(r.use_gpu_topk);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p hipfire-dispatch moe_res_ 2>&1 | tail -20`
Expected: FAIL to compile — `MoeDtypes` / `MoeResolution` not found.

- [ ] **Step 3: Write minimal implementation** (add to `crates/hipfire-dispatch/src/families/moe.rs`, after the `use` block)

```rust
use rdna_compute::DType;

/// Per-layer dtype snapshot the MoE eligibility lattice reads. Built by the
/// model from its weight structs; kept dtype-only so this stays GPU-free and
/// the dispatch crate needs no dependency on any arch crate.
///
/// `experts_all_gate_up_mq4` mirrors the `ffn.experts.iter().all(..)` clause
/// the original `gate_side_mq4` check used (qwen35.rs:4598-4605); the routed
/// fields use experts[0] as representative (the loader builds all experts in a
/// layer with matching dtype, so [0] == all — same invariant the original
/// routed_* checks relied on).
pub struct MoeDtypes {
    pub router: DType,
    pub shared_gate: DType,          // ffn.shared_expert_gate
    pub shared_expert_gate: DType,   // ffn.shared_expert.gate
    pub shared_expert_up: DType,     // ffn.shared_expert.up
    pub experts_all_gate_up_mq4: bool,
    pub routed_gate_up: DType,       // ffn.experts[0].gate_up
    pub routed_down: DType,          // ffn.experts[0].down
    pub has_paro_shared: bool,       // ffn.paro_shared.is_some()
}

/// Resolved fused-vs-fallback eligibility for one MoE decode layer. This IS the
/// routing-config logic, relocated from `moe_ffn_decode_impl` into one typed,
/// testable place (review finding #1). Pure function of `MoeDtypes` + k.
#[derive(Clone, Copy, Debug)]
pub struct MoeResolution {
    pub gate_side_mq4: bool,
    pub routed_indexable_mq4: bool,
    pub routed_indexable_mq6: bool,
    pub routed_indexable_paro: bool,
    pub use_gpu_topk: bool,
    pub needs_x_rot_local: bool,
}

impl MoeResolution {
    pub fn resolve(d: &MoeDtypes, k: usize) -> Self {
        use DType::*;
        let gate_side_mq4 = d.router == MQ4G256
            && d.shared_gate == MQ4G256
            && d.shared_expert_gate == MQ4G256
            && d.shared_expert_up == MQ4G256
            && d.experts_all_gate_up_mq4;

        let routed_gate_up_mq4 = d.routed_gate_up == MQ4G256;
        let routed_gate_up_mq6 = d.routed_gate_up == MQ6G256;
        let routed_gate_up_paro = d.routed_gate_up == ParoQ4G128 && d.has_paro_shared;

        let routed_indexable_mq4 = (d.routed_down == MQ4G256) && routed_gate_up_mq4;
        let routed_indexable_mq6 = (d.routed_down == MQ6G256) && routed_gate_up_mq6;
        let routed_indexable_paro =
            (d.routed_down == ParoQ4G128 && d.has_paro_shared) && routed_gate_up_paro;

        let routed_dtype_indexable =
            routed_indexable_mq4 || routed_indexable_mq6 || routed_indexable_paro;

        let use_gpu_topk = k == 8 && routed_dtype_indexable;
        let needs_x_rot_local = gate_side_mq4
            || routed_gate_up_mq4
            || routed_gate_up_mq6
            || routed_gate_up_paro;

        Self {
            gate_side_mq4,
            routed_indexable_mq4,
            routed_indexable_mq6,
            routed_indexable_paro,
            use_gpu_topk,
            needs_x_rot_local,
        }
    }

    pub fn routed_indexable(&self) -> bool {
        self.routed_indexable_mq4 || self.routed_indexable_mq6 || self.routed_indexable_paro
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p hipfire-dispatch moe_res_ 2>&1 | tail -20`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-dispatch/src/families/moe.rs crates/hipfire-dispatch/src/tests.rs
git commit -m "feat(dispatch): pure MoeResolution eligibility lattice (PR #393 #2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Source qwen35 decode booleans from `MoeResolution` (byte-parity)

No op changes — only replace the inline boolean computation (qwen35.rs:4598-4671) with one `MoeResolution::resolve(..)` call and reference its fields. Isolates "did centralizing the lattice change anything?" from the later executor relocation, giving a clean bisect point.

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs` (`moe_ffn_decode_impl`, 4563-5142)

- [ ] **Step 1: Build the descriptor + resolution** — at the top of `moe_ffn_decode_impl`, immediately after the `let n_exp = config.num_experts;` line (≈4576), insert:

```rust
    let moe_dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: ffn.router.gpu_dtype,
        shared_gate: ffn.shared_expert_gate.gpu_dtype,
        shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
        shared_expert_up: ffn.shared_expert.up.gpu_dtype,
        experts_all_gate_up_mq4: ffn
            .experts
            .iter()
            .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256),
        routed_gate_up: ffn
            .experts
            .first()
            .map(|e| e.gate_up.gpu_dtype)
            .unwrap_or(DType::F32),
        routed_down: ffn
            .experts
            .first()
            .map(|e| e.down.gpu_dtype)
            .unwrap_or(DType::F32),
        has_paro_shared: ffn.paro_shared.is_some(),
    };
    let moe_res = hipfire_dispatch::families::moe::MoeResolution::resolve(&moe_dtypes, k);
```

- [ ] **Step 2: Replace the inline boolean block** — delete the original computations of `gate_side_mq4`, `routed_mq4`, `routed_gate_up_mq4`, `routed_mq6`, `routed_gate_up_mq6`, `routed_paro`, `routed_gate_up_paro`, `routed_dtype_indexable_mq4`, `routed_dtype_indexable_mq6`, `routed_dtype_indexable_paro`, `routed_dtype_indexable`, `use_gpu_topk`, `needs_x_rot_local` (qwen35.rs:4598-4673) and replace with the bindings below. Keep `routed_mq4`/`routed_gate_up_paro` etc. as locals so the rest of the body (which references them by name, e.g. 4675, 4868, 5031) compiles unchanged:

```rust
    let gate_side_mq4 = moe_res.gate_side_mq4;
    let routed_gate_up_mq4 = ffn.experts.first().map(|e| e.gate_up.gpu_dtype == DType::MQ4G256).unwrap_or(false);
    let routed_mq4 = ffn.experts.first().map(|e| e.down.gpu_dtype == DType::MQ4G256).unwrap_or(false);
    let routed_gate_up_mq6 = ffn.experts.first().map(|e| e.gate_up.gpu_dtype == DType::MQ6G256).unwrap_or(false);
    let routed_gate_up_paro = ffn.experts.first().map(|e| e.gate_up.gpu_dtype == DType::ParoQ4G128).unwrap_or(false) && ffn.paro_shared.is_some();
    let routed_dtype_indexable_mq4 = moe_res.routed_indexable_mq4;
    let routed_dtype_indexable_mq6 = moe_res.routed_indexable_mq6;
    let routed_dtype_indexable_paro = moe_res.routed_indexable_paro;
    let use_gpu_topk = moe_res.use_gpu_topk;
    let needs_x_rot_local = moe_res.needs_x_rot_local;
```

(Note: `routed_mq6`, `routed_paro`, `routed_dtype_indexable` were only used to *derive* the above; if the post-4673 body references any of them directly, keep a local `let` for it too. Verify with `cargo build` in Step 3.)

- [ ] **Step 3: Build** to verify no dangling references

Run: `cargo build -p hipfire-arch-qwen35 2>&1 | grep -E "error|warning: unused variable: \`(routed|gate|use_gpu)" | head`
Expected: no `error` lines. If an `unused`/`undefined` boolean appears, add/remove the matching local `let` until clean.

- [ ] **Step 4: Byte-parity gate** (requires GPU + an A3B MoE model)

```bash
# Capture deterministic output before/after this commit on the same prompt.
git stash list  # ensure clean; this task's edits are staged-not-committed
PROMPT=benchmarks/prompts/lru_cache_pep8_strict.txt
md5sum "$PROMPT"
./scripts/coherence-gate.sh 2>&1 | tail -5
```
Expected: coherence-gate reports **no hard errors**. Manually confirm the A3B-MQ4 cell's text is byte-identical to a run from `HEAD` before this task (e.g. diff two `coherence_probe --temperature 0.0` transcripts on the same prompt md5). Any divergence = a boolean was mis-sourced; bisect against Step 2.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-arch-qwen35/src/qwen35.rs
git commit -m "refactor(qwen35): source MoE decode eligibility from MoeResolution

Byte-identical; lattice now lives in dispatch::families::moe.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Migrate `PipelineParams` to an enum

Make room for `MoeParams` in the pipeline without disturbing GEMV behavior. Pure refactor; verified by the existing dispatch unit tests staying green.

**Files:**
- Modify: `crates/hipfire-dispatch/src/pipeline/mod.rs`
- Modify: `crates/hipfire-dispatch/src/families/gemv.rs:274-278`

- [ ] **Step 1: Rename the struct + add the enum** — in `crates/hipfire-dispatch/src/pipeline/mod.rs`, rename `pub struct PipelineParams` to `pub struct LinearParams` (lines 20-26), then add directly below it:

```rust
pub enum PipelineParams<'a> {
    Linear(LinearParams<'a>),
    // Moe(MoeParams<'a>) added in Task 4
}
```

- [ ] **Step 2: Thread the enum through the two consumers** — `execute_pipeline` (line 28) and `dispatch_fused` (line 100) take `params: &PipelineParams`. Inside each, bind the linear payload at the top so the existing field accesses keep working:

In `execute_pipeline`, after the signature, add:
```rust
    let params = match params {
        PipelineParams::Linear(p) => p,
    };
```
In `dispatch_fused`, after the signature, add the same `let params = match params { PipelineParams::Linear(p) => p };`. (Both bodies then reference `params.x`, `params.buf`, etc. unchanged.)

- [ ] **Step 3: Update the one external caller** — `crates/hipfire-dispatch/src/families/gemv.rs:274-278`, change:

```rust
                    let pipe_params = PipelineParams {
                        x: params.x, y: params.y, buf: params.w.buf,
                        m: params.w.m, k: params.w.k,
                    };
                    return dispatch_fused(gpu, KernelKey::GemvMfp4G32Fused, &pipe_params);
```
to:
```rust
                    let pipe_params = PipelineParams::Linear(LinearParams {
                        x: params.x, y: params.y, buf: params.w.buf,
                        m: params.w.m, k: params.w.k,
                    });
                    return dispatch_fused(gpu, KernelKey::GemvMfp4G32Fused, &pipe_params);
```
(Add `LinearParams` to the `use crate::pipeline::{...}` import at gemv.rs:14.)

- [ ] **Step 4: Run tests + build**

Run: `cargo test -p hipfire-dispatch 2>&1 | tail -5 && cargo build -p hipfire-dispatch 2>&1 | grep -c error`
Expected: all existing tests PASS; `0` errors. `can_satisfy` / GEMV tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-dispatch/src/pipeline/mod.rs crates/hipfire-dispatch/src/families/gemv.rs
git commit -m "refactor(dispatch): PipelineParams -> enum { Linear(LinearParams) }

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Add MoE `PipelineOp` variants + grow `MoeParams`

Define the vocabulary and the generic parameter carrier the executor arm will consume. Compile-only + one `can_satisfy` unit test for the decode op-list prefix.

**Files:**
- Modify: `crates/hipfire-dispatch/src/types.rs` (PipelineOp enum, lines 6-17)
- Modify: `crates/hipfire-dispatch/src/families/moe.rs` (MoeParams)
- Modify: `crates/hipfire-dispatch/src/pipeline/mod.rs` (enum arm placeholder)
- Test: `crates/hipfire-dispatch/src/tests.rs`

- [ ] **Step 1: Write the failing test** (append to `crates/hipfire-dispatch/src/tests.rs`)

```rust
#[test]
fn moe_decode_oplist_prefix_matches_gate_side() {
    // The 4-way fused gate-side projection is capturable as a length-1 prefix.
    let oplist = [
        PipelineOp::MoeGateSideProj, PipelineOp::Softmax, PipelineOp::TopKRenorm,
        PipelineOp::SharedExpertDown, PipelineOp::IndexedGateUp,
        PipelineOp::SiluMulRotate, PipelineOp::IndexedDownExpanded, PipelineOp::MoeCombine,
    ];
    let fused = Pipeline::new(&[PipelineOp::MoeGateSideProj]);
    assert!(fused.can_satisfy(&oplist));
    let too_long = Pipeline::new(&[PipelineOp::MoeGateSideProj, PipelineOp::TopKRenorm]);
    assert!(!too_long.can_satisfy(&oplist)); // second op mismatches Softmax
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p hipfire-dispatch moe_decode_oplist 2>&1 | tail -10`
Expected: FAIL to compile — `PipelineOp::MoeGateSideProj` etc. not found.

- [ ] **Step 3a: Add the op variants** — in `crates/hipfire-dispatch/src/types.rs`, extend the `PipelineOp` enum (after `GivensRotate,` at line 16):

```rust
    // MoE decode ops (Phase 1). TopKRenorm / MoeCombine fused impls are
    // k=8-only today (review finding #2); the variant name is k-agnostic so a
    // future k=6 kernel family can reuse it.
    MoeGateSideProj,
    Softmax,
    TopKRenorm,
    SharedExpertDown,
    IndexedGateUp,
    IndexedDownExpanded,
    MoeCombine,
```
(`SiluMulRotate` already exists at line 13 — reuse it.)

- [ ] **Step 3b: Grow `MoeParams`** — in `crates/hipfire-dispatch/src/families/moe.rs`, replace the existing `pub struct MoeParams` (lines 28-33) with the generic carrier the executor needs. All fields are `rdna_compute` types or scalars (no qwen35 dep):

```rust
/// Everything the MoE decode executor arm reads, marshaled by the model from
/// its weight/config/scratch structs. Generic (GpuTensor + scalars + GivensRef)
/// so the dispatch crate needs no arch dependency.
pub struct MoeParams<'a> {
    pub res: super::moe::MoeResolution,
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,         // moe_intermediate_size
    pub smi: usize,        // shared_expert_intermediate_size
    pub k: usize,          // num_experts_per_tok
    pub n_exp: usize,      // num_experts
    pub norm_topk_prob: bool,
    pub x_rot_prerotated: bool,
    // activations / residual
    pub x_norm: &'a GpuTensor,
    pub x_residual: &'a GpuTensor,
    // gate-side weights (buffers + their m/k)
    pub router: WeightRef<'a>,
    pub shared_expert_gate: WeightRef<'a>,
    pub shared_gate_w: WeightRef<'a>,   // shared_expert.gate
    pub shared_up_w: WeightRef<'a>,     // shared_expert.up
    pub shared_down_w: WeightRef<'a>,   // shared_expert.down
    // routed expert pointer tables + representative dims (experts[0])
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    pub routed_gate_up_k: usize,
    pub routed_down_m: usize,
    pub routed_down_k: usize,
    // paro sidecars for routed experts (None unless routed_indexable_paro)
    pub routed_gate_up_paro: Option<GivensRef<'a>>,
    pub routed_down_paro: Option<GivensRef<'a>>,
    // scratch (mirrors MoeScratchRef)
    pub router_logits: &'a GpuTensor,
    pub scalar_buf: &'a GpuTensor,
    pub x_rot_local: &'a GpuTensor,
    pub gate_buf: &'a GpuTensor,
    pub up_buf: &'a GpuTensor,
    pub ffn_hidden: &'a GpuTensor,
    pub ffn_out: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub down_expanded: &'a GpuTensor,
}
```
Add `use crate::families::gemv::{GivensRef, WeightRef};` to the imports if not present. Delete the old `variant`/`weights`/`x`/`y` fields and the now-stale `MoeVariant` resolve usage in `MoeFamily::resolve` only if it stops compiling — otherwise leave `MoeFamily` untouched this task.

- [ ] **Step 3c: Extend the enum** — in `crates/hipfire-dispatch/src/pipeline/mod.rs`, add the `Moe` arm to `PipelineParams`:

```rust
pub enum PipelineParams<'a> {
    Linear(LinearParams<'a>),
    Moe(crate::families::moe::MoeParams<'a>),
}
```
In `execute_pipeline`/`dispatch_fused`, change the top-of-body bind to keep the linear paths working and reject Moe for now:
```rust
    let params = match params {
        PipelineParams::Linear(p) => p,
        PipelineParams::Moe(_) => return Err(DispatchError::UnsupportedVariant {
            family: "pipeline", variant: "moe-not-wired", arch: "", quant: "",
        }),
    };
```

- [ ] **Step 4: Run test + build**

Run: `cargo test -p hipfire-dispatch moe_decode_oplist 2>&1 | tail -5 && cargo build -p hipfire-dispatch 2>&1 | grep -c error`
Expected: test PASS; `0` errors.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-dispatch/src/types.rs crates/hipfire-dispatch/src/families/moe.rs crates/hipfire-dispatch/src/pipeline/mod.rs crates/hipfire-dispatch/src/tests.rs
git commit -m "feat(dispatch): MoE PipelineOp vocabulary + generic MoeParams

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Relocate the decode body into the executor MoE arm (byte-parity)

Move the existing `gpu.*` call sequence (qwen35.rs:4722-5142) verbatim into a new `run_moe_decode` in the dispatch crate, driven by `MoeParams.res`. **Do not change any `gpu.*` call, argument, buffer, or order** — this is a relocation, not a rewrite. The spec is the source of the op→kernel mapping; the existing code is the source of the exact arguments.

**Files:**
- Modify: `crates/hipfire-dispatch/src/pipeline/mod.rs` (add `run_moe_decode`; wire the `Moe` arm)
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs` (`moe_ffn_decode_impl` body → marshal + call)

- [ ] **Step 1: Add `run_moe_decode`** in `crates/hipfire-dispatch/src/pipeline/mod.rs`. Signature:

```rust
pub fn run_moe_decode(gpu: &mut Gpu, p: &crate::families::moe::MoeParams)
    -> Result<(), DispatchError>
{
    use crate::families::moe::MoeResolution;
    let res: MoeResolution = p.res;
    macro_rules! hip { ($e:expr) => { $e.map_err(|e| DispatchError::Hip(e.to_string())) }; }
    // ── Op: MoeGateSideProj  (qwen35.rs:4674-4763) ──
    // ── Op: Softmax + TopKRenorm  (qwen35.rs:4765-4827) ──
    // ── Op: SharedExpertDown  (qwen35.rs:4829-4865) ──
    // ── Op: IndexedGateUp / SiluMulRotate / IndexedDownExpanded / MoeCombine
    //       (qwen35.rs:4867-5142) ──
    // ... relocated body ...
    Ok(())
}
```
Port the body **block-by-block** from qwen35.rs:4674-5142, substituting:
- `gate_side_mq4` → `res.gate_side_mq4`, `use_gpu_topk` → `res.use_gpu_topk`,
  `routed_dtype_indexable_mq4` → `res.routed_indexable_mq4`, etc.
- `ffn.router.buf` → `p.router.buf`, `ffn.shared_expert.down` → `p.shared_down_w`,
  `ffn.expert_gate_up_ptrs` → `p.expert_gate_up_ptrs`, scratch `s.foo` → `p.foo`.
- `config.dim` → `p.hidden`, `mi`/`smi`/`k`/`n_exp` → `p.mi`/`p.smi`/`p.k`/`p.n_exp`,
  `config.norm_topk_prob` → `p.norm_topk_prob`, `x_rot_prerotated` → `p.x_rot_prerotated`.
- The paro branches read `ffn.experts[0].gate_up.paro` → `p.routed_gate_up_paro`
  (`Option<GivensRef>`), `...down.paro` → `p.routed_down_paro`.
- Helper fns called inside (`weight_gemv`, `rotate_x_mq_for`, `fused_silu_mul_rotate_mq_for`,
  `fused_silu_mul_rotate_mq_batched_for`, `slice_f32_view`, `rotate_x_paro_for`) live in
  `hipfire-runtime::llama` / qwen35; they take `gpu` + `WeightRef`-equivalent buffers.
  Where a helper needs a qwen35 `Linear`, replace with the underlying `gpu.*` call it wraps
  (read the helper body; e.g. `weight_gemv` dispatches on dtype to a `gpu.gemv_*`). Keep the
  selected kernel + args identical. **If a helper cannot be cleanly reduced to `gpu.*` +
  generic args, stop and flag it** — that helper must move to the dispatch crate or its
  inputs added to `MoeParams`, and the plan author should be consulted before guessing.

- [ ] **Step 2: Wire the `Moe` arm** — in `execute_pipeline`, before the `let params = match` linear bind, handle Moe by delegating:
```rust
    if let PipelineParams::Moe(p) = params {
        return run_moe_decode(gpu, p);
    }
```
(Leave the existing linear `match` bind after it for the `Linear` case.)

- [ ] **Step 3: Marshal + call from qwen35** — replace `moe_ffn_decode_impl`'s body from line 4722 to the function's end (`}` at 5143) with construction of `MoeParams` from `ffn`/`config`/`s` (+ the `moe_res`/`moe_dtypes` from Task 2) and a single call:

```rust
    let params = hipfire_dispatch::families::moe::MoeParams {
        res: moe_res,
        hidden, mi, smi, k, n_exp,
        norm_topk_prob: config.norm_topk_prob,
        x_rot_prerotated,
        x_norm, x_residual,
        router: weight_ref(&ffn.router),
        shared_expert_gate: weight_ref(&ffn.shared_expert_gate),
        shared_gate_w: weight_ref(&ffn.shared_expert.gate),
        shared_up_w: weight_ref(&ffn.shared_expert.up),
        shared_down_w: weight_ref(&ffn.shared_expert.down),
        expert_gate_up_ptrs: &ffn.expert_gate_up_ptrs,
        expert_down_ptrs: &ffn.expert_down_ptrs,
        routed_gate_up_k: ffn.experts[0].gate_up.k,
        routed_down_m: ffn.experts[0].down.m,
        routed_down_k: ffn.experts[0].down.k,
        routed_gate_up_paro: ffn.experts[0].gate_up.paro.as_ref().map(givens_ref),
        routed_down_paro: ffn.experts[0].down.paro.as_ref().map(givens_ref),
        router_logits: s.router_logits, scalar_buf: s.scalar_buf,
        x_rot_local: s.x_rot_local, gate_buf: s.gate_buf, up_buf: s.up_buf,
        ffn_hidden: s.ffn_hidden, ffn_out: s.ffn_out,
        gate_batch: s.gate_batch, up_batch: s.up_batch, rot_batch: s.rot_batch,
        topk_indices: s.topk_indices, topk_weights: s.topk_weights,
        down_expanded: s.down_expanded,
    };
    hipfire_dispatch::pipeline::run_moe_decode(gpu, &params)
        .map_err(|e| HipError::from(e.to_string()))?;
    Ok(())
```
Phase 1 calls `run_moe_decode` **directly** — the MoE arm needs no `DispatchCtx`/registry/op-list because `MoeResolution` already did all gating (there is no arch-predicate or fused-search left to run). The `execute_pipeline` Moe arm (Step 2) exists only as the generic-entry forwarder for future callers; the qwen35 hot path skips it to avoid threading an unused ctx/registry down into `moe_ffn_decode_impl`.

Add two small local helpers in qwen35 mapping a `Linear` → dispatch `WeightRef` and a paro sidecar → `GivensRef` (mirror the existing `WeightRef` construction already used by this crate's `GemvFamily::run_auto` call sites — grep `WeightRef {` in qwen35 for the exact field mapping; for `GivensRef` map `paro.pairs`/`paro.theta`/`paro.channel_scales`/`paro.krot as usize` → `pairs`/`theta`/`scales`/`krot`).

- [ ] **Step 4: Build, then byte-parity gate**

Run: `cargo build 2>&1 | grep -c error` → expected `0`.
Then the gate (GPU + A3B model):
```bash
md5sum benchmarks/prompts/lru_cache_pep8_strict.txt
./scripts/coherence-gate.sh 2>&1 | tail -5
```
Expected: no hard errors AND the A3B-MQ4 / MQ6 / Paro transcripts are **byte-identical** to a `HEAD~1` run on the same prompt md5 (greedy, temp 0.0). Compare all three routed dtypes — they exercise the mq4 / mq6 / paro branches of the relocated body. Any divergence → a relocation substitution is wrong; bisect by reverting `run_moe_decode` block-by-block.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-dispatch/src/pipeline/mod.rs crates/hipfire-arch-qwen35/src/qwen35.rs
git commit -m "refactor(dispatch): relocate MoE decode body into execute_pipeline arm

Byte-identical; qwen35 now marshals MoeParams and calls the executor.
Closes PR #393 item #2.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes on verification honesty

Tasks 1, 3, 4 are true TDD (pure logic, GPU-free unit tests). Tasks 2 and 5 are GPU integration refactors whose correctness criterion is **byte-identical output** vs the prior commit (the spec's stated success criterion and the CLAUDE.md coherence rule) — there is no meaningful "failing unit test first" for a faithful relocation, so the byte-parity A/B + coherence gate IS the test. Run each on a **fresh process** with a **byte-identical prompt** (record md5) per the CLAUDE.md bench rule. Exercise all three routed dtypes (MQ4, MQ6, ParoQ4G128) since they take different branches.

## Known risk to watch (Task 5)

The relocation assumes every `gpu.*` call in qwen35.rs:4674-5142 reduces to a `Gpu` method + generic-buffer args. The qwen35 helper fns (`weight_gemv`, `rotate_x_mq_for`, `fused_silu_mul_rotate_mq*_for`, `rotate_x_paro_for`, `slice_f32_view`) are the friction points — some may read a qwen35 `Linear`'s AWQ sidecar or dtype to pick a kernel. If reducing one to generic args is not mechanical, **do not guess**: either thread its missing input into `MoeParams` or move the helper into the dispatch crate, and re-confirm byte-parity. This is the single most likely place Phase 1 stalls.
