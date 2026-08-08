# Executor Merge — one `run_layer_program(mesh, …)` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the single-GPU `run_layer_program` and the EP `run_layer_program_ep` into exactly ONE mesh-driven executor, delivering plan §1 of the device-mesh design ("exactly ONE executor") and retiring the "two mesh-aware executors" compromise (retracted 2026-07-06).

**Architecture:** ONE `run_layer_program(mesh, gpus, bindings: &mut [B], partials: &[GpuTensor], program)` in `hipfire-dispatch`. Single-GPU is a 1×1 mesh + 1-element `bindings` slice + empty `partials`; EP is a 1×N Ep mesh + N-element slices. The MoE op branches on `mesh.has_axis(Ep)`: no Ep axis → the byte-identical single-GPU `run_moe` path via `dispatch_super_op`; Ep axis → the zero / `run_moe_ep` / all-reduce / `ep_add_into_residual` path. `DispatchCtx` is built inside per rank; `residual_dim` is derived from `partials[0].numel()`. No EP-specific params leak onto the single-GPU hot path — the "union params struct" the N1 review rejected is avoided.

**Tech Stack:** Rust, `hipfire-dispatch` (executor + `ForwardBindings`/`LayerProgram`/`dispatch_super_op`), `hipfire-hardware` (`Gpus`, `DeviceMesh`, `DimKind`, collectives), the 5 lowered arch crates (qwen35, qwen2, deepseek4, minimax, lfm2moe), the daemon example binary.

## Global Constraints

- **NEVER `cargo fmt`** and **never `scripts/fmt-changed.sh`**. Format only files you edited, per-file: `rustfmt --edition 2021 --config skip_children=true <file>`. NEVER rustfmt the fmt-debt files: `daemon.rs`, `qwen35.rs`, `deepseek4/minimax forward.rs`.
- **Byte-exact is the gate.** Every step that touches an executor or hot path must keep `HIPFIRE_FORWARD_LOWERED=0` (bespoke) vs `=1` (lowered) **committed-token md5 identical** for single-GPU, and `ep_decode_parity` per-step argmax parity for EP. A soft output change is a FAIL here, not a pass.
- **Single-thread HIP.** The executor stays single-threaded (device-switch + FIFO stream per rank). Never spawn an OS thread per rank.
- **The 1×1 arm must not touch stream state.** It must never call `ensure_rank_streams` (which flips `active_stream None→Some` and switches the hot-path memset sync→async). Only the ≥1×N EP arm touches streams.
- **Build under the dev shell:** `nix develop -c cargo …` (the bare sandbox linker returns `ld 127`).
- Conventional commits; end messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Run `cargo clippy -p <changed-crate>` before each commit; per-file rustfmt the files you edited.
- Commit after each task (frequent commits).

---

## File Structure

- `crates/hipfire-dispatch/src/pipeline/superop.rs` — **the merged executor.** `run_layer_program` gains the new `(mesh, gpus, bindings: &mut [B], partials, program)` signature and absorbs `ep.rs`'s MoE branch. Add two default trait methods to `ForwardBindings` only if a residual-dim accessor is preferred over deriving from `partials` (see Task 1 decision — default is to derive, adding no trait methods).
- `crates/hipfire-dispatch/src/ep.rs` — **shrinks to a shim** in Step 1 (`run_layer_program_ep` delegates to the merged core), then is **deleted** in Step 2 once EP call sites move to the merged entry. `ensure_rank_streams` moves to `superop.rs` (still EP-only, called by the EP drivers, not by the executor).
- `crates/hipfire-arch-{qwen35,qwen2,deepseek4,minimax,lfm2moe}/src/{forward.rs,qwen35.rs,qwen2.rs}` — **single-GPU call-site migration** (Step 2): the lowered driver switches from `run_layer_program(gpu, &ctx, &program, &mut bind)` to the merged entry, and its function signature threads `&mut Gpus` instead of `&mut Gpu`.
- `crates/hipfire-runtime/examples/daemon.rs` — **`Gpus::single` ownership** (Step 2): single-GPU models hold their device as `Gpus::single(gpu)` from load, so the forward path has a `&mut Gpus` to hand the executor. (Do NOT rustfmt this file.)
- No new files. No new gate script (the merge is covered by the existing `HIPFIRE_FORWARD_LOWERED` oracle + `ep_decode_parity` + `coherence-gate.sh` + `store-pp-gate.sh`).

---

## Reference: current shapes (verified 2026-07-06)

Merged-signature target:
```rust
pub fn run_layer_program<B: ForwardBindings>(
    mesh: &DeviceMesh,
    gpus: &mut Gpus,
    bindings: &mut [B],
    partials: &[GpuTensor],       // empty on non-EP meshes; residual_dim = partials[0].numel()
    program: &LayerProgram,
) -> Result<(), DispatchError>
```

Current single-GPU executor (`superop.rs:363`):
```rust
pub fn run_layer_program<B: ForwardBindings>(
    gpu: &mut Gpu, ctx: &DispatchCtx, program: &LayerProgram, bindings: &mut B,
) -> Result<(), DispatchError> {
    for op in program { dispatch_super_op(gpu, ctx, op, bindings)?; }
    Ok(())
}
```

Current EP executor (`ep.rs:71`): `run_layer_program_ep(mesh, gpus, bindings: &mut [B], partials: &[GpuTensor], program, residual_dim)` — loops ops; `Moe` → per-rank memset partial, `run_moe_ep`, `all_reduce_sum_f32[_peer](group_along(Ep,coord), refs, residual_dim)`, `ep_add_into_residual`; else → per-rank `dispatch_super_op`.

Single-GPU call sites (Step 2 targets):
- `crates/hipfire-arch-lfm2moe/src/forward.rs:696`
- `crates/hipfire-arch-minimax/src/forward.rs:754`
- `crates/hipfire-arch-deepseek4/src/forward.rs:2153`
- `crates/hipfire-arch-qwen2/src/qwen2.rs:1380`
- `crates/hipfire-arch-qwen35/src/qwen35.rs:13061`

EP call sites (Step 2 targets — drop the shim, derive residual_dim):
- `crates/hipfire-arch-minimax/src/forward.rs:1189`
- `crates/hipfire-arch-deepseek4/src/forward.rs:2242`
- `crates/hipfire-arch-deepseek4/src/forward.rs:2888`

`ForwardBindings` already carries the EP hooks with `Err` defaults (`run_moe_ep`, `ep_add_into_residual`), so non-EP arches need no new impls.

---

## STEP 1 — Unify the executor body (dispatch-only, oracle-gated)

Contained entirely in `hipfire-dispatch`. Single-GPU and EP behavior byte-unchanged: the old single-GPU 4-arg entry becomes a thin wrapper, and `run_layer_program_ep` becomes a thin shim, both delegating to the new merged core. No arch call site changes yet.

### Task 1: Add the merged core `run_layer_program_mesh`, keep both old entries as delegators

**Files:**
- Modify: `crates/hipfire-dispatch/src/pipeline/superop.rs` (add merged core; rewrite `run_layer_program` at `:363` to delegate)
- Modify: `crates/hipfire-dispatch/src/ep.rs` (move `ensure_rank_streams` out to `superop.rs`; make `run_layer_program_ep` a shim)
- Test: `crates/hipfire-dispatch/src/pipeline/superop.rs` (`#[cfg(test)]` unit test for the no-GPU control-flow decisions)

**Interfaces:**
- Produces: `pub fn run_layer_program_mesh<B: ForwardBindings>(mesh: &DeviceMesh, gpus: &mut Gpus, bindings: &mut [B], partials: &[GpuTensor], program: &LayerProgram) -> Result<(), DispatchError>` — the merged core.
- Produces: `pub fn ensure_rank_streams(gpus: &mut Gpus) -> Result<(), DispatchError>` (relocated verbatim from `ep.rs`).
- Consumes: `DeviceMesh::has_axis`, `DimKind::Ep`, `group_along`, `coord_of` (hardware); `dispatch_super_op`, `ForwardBindings::{run_moe_ep, ep_add_into_residual}` (superop).

- [ ] **Step 1: Write the failing no-GPU test for the MoE-branch decision**

The merged core's only GPU-free, testable decision is "does op `Moe` take the EP path?" — a function of `mesh.has_axis(Ep)`. Extract that predicate so it is unit-testable.

Add to `superop.rs`:
```rust
/// True iff a `Moe` op on this mesh runs the EP reduce path (vs the replicated
/// single-GPU `run_moe`). Sole cross-device branch of the merged executor.
pub(crate) fn moe_is_ep(mesh: &DeviceMesh) -> bool {
    mesh.has_axis(hipfire_hardware::DimKind::Ep)
}
```
Add the test:
```rust
#[cfg(test)]
mod merge_tests {
    use super::*;
    use hipfire_hardware::{DeviceMesh, DimKind};
    #[test]
    fn moe_branch_is_ep_only_with_ep_axis() {
        assert!(!moe_is_ep(&DeviceMesh::single()));                       // single-GPU
        assert!(!moe_is_ep(&DeviceMesh::rect(&[(DimKind::Pp, 2)])));      // PP-only
        assert!(moe_is_ep(&DeviceMesh::rect(&[(DimKind::Ep, 4)])));       // EP
        assert!(moe_is_ep(&DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Ep, 2)])));
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `nix develop -c cargo test -p hipfire-dispatch --lib moe_branch_is_ep_only_with_ep_axis`
Expected: FAIL to compile — `moe_is_ep` not yet defined (add the fn in Step 3 alongside the core).

- [ ] **Step 3: Implement the merged core + relocate `ensure_rank_streams`; wire both old entries to delegate**

In `superop.rs`, add `ensure_rank_streams` (copy verbatim from `ep.rs:48-56`, keep `pub`), add `moe_is_ep` (Step 1), and add the merged core:
```rust
use hip_bridge::DeviceBuffer;
use hipfire_hardware::{DeviceMesh, DimKind, Gpus};

/// The ONE executor (device-mesh plan §1). Single-GPU = 1×1 mesh + 1-elem
/// `bindings` + empty `partials`; EP = 1×N Ep mesh + N-elem slices. `Moe`
/// branches on the Ep axis; all other ops run replicated per rank. `ctx` is
/// built per rank inside; `residual_dim` is derived from `partials[0]`.
pub fn run_layer_program_mesh<B: ForwardBindings>(
    mesh: &DeviceMesh,
    gpus: &mut Gpus,
    bindings: &mut [B],
    partials: &[GpuTensor],
    program: &LayerProgram,
) -> Result<(), DispatchError> {
    let n = bindings.len();
    debug_assert_eq!(gpus.devices.len(), n, "run_layer_program_mesh: gpus/bindings length mismatch");
    let ep = moe_is_ep(mesh);
    if ep {
        assert_eq!(partials.len(), n, "run_layer_program_mesh: EP needs one partial per rank");
    }
    for op in program {
        if ep && matches!(op.kind, SuperOpKind::Moe) {
            let residual_dim = partials[0].numel();
            // 1. zero each rank's partial on its stream
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                let stream = gpus.devices[r].active_stream.as_ref().ok_or_else(|| {
                    DispatchError::Hip(format!(
                        "run_layer_program_mesh: device {r} has no active_stream (call ensure_rank_streams)"))
                })?;
                gpus.devices[r].hip
                    .memset_async(&partials[r].buf, 0, residual_dim * 4, stream)
                    .map_err(hip_err)?;
            }
            // 2. owned-expert partial (+ shared on rank 0)
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                let ctx = DispatchCtx::new(&gpus.devices[r]);
                bindings[r].run_moe_ep(&mut gpus.devices[r], &ctx, &op.binding, &partials[r], r != 0)?;
            }
            // 3. all-reduce over the Ep group
            let refs: Vec<&DeviceBuffer> = partials.iter().map(|p| &p.buf).collect();
            let group = mesh.group_along(DimKind::Ep, &mesh.coord_of(0));
            debug_assert_eq!(group.len(), refs.len(),
                "run_layer_program_mesh: single Ep-group (1×N) only; composed/multi-group EP is Phase 5b");
            static PEER_DECODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let use_peer = *PEER_DECODE.get_or_init(||
                std::env::var("HIPFIRE_EP_PEER_ALLREDUCE_DECODE").as_deref() == Ok("1"));
            if use_peer {
                gpus.all_reduce_sum_f32_peer(&group, &refs, residual_dim).map_err(hip_err)?;
            } else {
                gpus.all_reduce_sum_f32(&group, &refs, residual_dim).map_err(hip_err)?;
            }
            // 4. add reduced partial into each rank's residual
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                bindings[r].ep_add_into_residual(&mut gpus.devices[r], &partials[r])?;
            }
        } else {
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                let ctx = DispatchCtx::new(&gpus.devices[r]);
                dispatch_super_op(&mut gpus.devices[r], &ctx, op, &mut bindings[r])?;
            }
        }
    }
    Ok(())
}

fn hip_err(e: hip_bridge::HipError) -> DispatchError { DispatchError::Hip(e.to_string()) }
```
Rewrite the existing single-GPU entry at `:363` to delegate (keeps every current single-GPU call site compiling **unchanged** through Step 1):
```rust
/// Single-GPU convenience wrapper (unchanged signature). Delegates to the
/// merged core with a 1×1 mesh via `Gpus::single`. NOTE: `Gpus::single`
/// consumes the `Gpu`, so this takes the `Gpu` by value and returns it —
/// used only by call sites not yet migrated to the merged entry (Step 2).
```
Because `Gpus::single` consumes the `Gpu`, a by-`&mut Gpu` delegator cannot build a `Gpus` without moving the borrow. Therefore **do NOT** rewrite the `&mut Gpu` entry to delegate; instead keep the old `run_layer_program(gpu, ctx, program, bindings)` body as-is (its 3-line loop) for now — it is retired in Step 2 when its callers get a `Gpus`. Only `ep.rs` delegates in Step 1:

In `ep.rs`, replace the body of `run_layer_program_ep` with a shim (drop the now-derived `residual_dim` param usage — assert it matches):
```rust
pub fn run_layer_program_ep<B: ForwardBindings>(
    mesh: &DeviceMesh, gpus: &mut Gpus, bindings: &mut [B],
    partials: &[GpuTensor], program: &LayerProgram, residual_dim: usize,
) -> Result<(), DispatchError> {
    debug_assert_eq!(partials[0].numel(), residual_dim, "ep shim: residual_dim mismatch");
    crate::pipeline::superop::run_layer_program_mesh(mesh, gpus, bindings, partials, program)
}
```
Re-export `ensure_rank_streams` from `ep.rs` for its existing callers: `pub use crate::pipeline::superop::ensure_rank_streams;` (remove the old definition from `ep.rs`).

- [ ] **Step 4: Run the unit test + full dispatch lib tests**

Run: `nix develop -c cargo test -p hipfire-dispatch --lib`
Expected: PASS (incl. `moe_branch_is_ep_only_with_ep_axis`); no other dispatch test regresses.

- [ ] **Step 5: EP byte-identity — `ep_decode_parity` anchor**

Run: `bash .agent-progress/run-ep-parity.sh` then `grep "PARITY exit" .agent-progress/ep-parity.log`
Expected: `PARITY exit: 0` (mesh-driven EP through the shim == production forward_scratch, 16 steps, qwen3.6-35b-a3b). This proves the EP path is byte-unchanged after folding it into the merged core.

- [ ] **Step 6: Commit**

```bash
rustfmt --edition 2021 --config skip_children=true crates/hipfire-dispatch/src/pipeline/superop.rs crates/hipfire-dispatch/src/ep.rs
nix develop -c cargo clippy -p hipfire-dispatch
git add crates/hipfire-dispatch/src/pipeline/superop.rs crates/hipfire-dispatch/src/ep.rs
git commit -m "feat(dispatch): merged run_layer_program_mesh core; EP shim delegates

$(printf 'Fold the EP MoE branch into ONE mesh-driven executor core; single Ep-group\nreduce derives residual_dim from partials[0]. run_layer_program_ep becomes a\nshim. Single-GPU 4-arg entry unchanged (retired in Step 2). ep_decode_parity\nanchor PASS (byte-identical EP).\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## STEP 2 — Migrate all call sites onto the merged entry; delete the shim + old signature

The substantive step, split so the wide **byte-neutral** daemon edits are isolated (coherence-gated) from the behavior-bearing per-arch ones (md5-gated). Two byte-neutral prep tasks first — **2.0** (`main()` owns `Gpus::single`) and **2.0b** (the shared `generate()` dispatcher carries `Gpus`) — neither changes any forward. Then **2.1–2.5** migrate one arch at a time (its forward chain + the executor-call swap), each `FORWARD_LOWERED=0/1` md5-gated, so a reviewer can reject one arch without blocking the others. The daemon god-functions (`main`, `generate`) are threaded once in 2.0/2.0b; per-arch tasks only flip *their* `generate_<arch>` call from `&mut gpus.devices[0]` to `&mut gpus`.

### Task 2.0: Byte-neutral `Gpus::single` ownership swap in `fn main()` (daemon)

**Concrete boundary (verified 2026-07-06):** both `let mut gpu = rdna_compute::Gpu::init()` sites (daemon.rs:1421 and 1452) are inside **`fn main()` (daemon.rs:1391–3132)**. That scope holds **~59** of the file's `gpu` references; the other ~268 are `gpu:` **parameters in other functions** and DO NOT change in this task. **This task edits ONLY `fn main()`.**

**Byte-neutral contract:** NO function signature changes anywhere. Every forward/helper called from `main()` keeps its `&mut Gpu` parameter and is fed `&mut gpus.devices[0]`. No `&mut Gpus` is threaded yet (that is per-arch, 2.1–2.5). Behavior is bit-identical — this task only moves *where the device is owned*, not how the forward runs.

**Files:**
- Modify: `crates/hipfire-runtime/examples/daemon.rs`, `fn main()` scope only (1391–3132). **Do NOT rustfmt this file.**

**Interfaces:**
- Consumes: `Gpus::single(gpu: Gpu) -> Gpus` (`hipfire_hardware` / `multi_gpu.rs:156`); `gpus.devices: Vec<Gpu>`.
- Produces: `main()` owns `gpus: Gpus` (one device); every other function is untouched.

- [ ] **Step 1: Confirm the scope**

Run: `nix develop -c bash -c "awk 'NR>=1391 && NR<3133' crates/hipfire-runtime/examples/daemon.rs | grep -c '\bgpu\b'"`
Expected: ~59. Read the two init sites (1421, 1452) and note every `main()`-local use of `gpu` (the forwards/helpers it passes `&mut gpu` to). Do not edit yet.

- [ ] **Step 2: Swap ownership, byte-neutral**

At both init sites, wrap the initialized `Gpu` in `Gpus::single` (it takes `Gpu` by value — keep the existing init error handling, then wrap the `Ok` value):
```rust
// was: let mut gpu = match rdna_compute::Gpu::init() { Ok(g) => g, Err(e) => ... };
let mut gpus = hipfire_hardware::Gpus::single(match rdna_compute::Gpu::init() {
    Ok(g) => g,
    Err(e) => /* unchanged error arm */,
});
```
Then, **within `fn main()` only**, rewrite each remaining use: `&mut gpu` → `&mut gpus.devices[0]`, `&gpu` → `&gpus.devices[0]`. Do NOT touch any `gpu:` parameter in other functions. Do NOT introduce any `&mut gpus` argument (no callee accepts one until 2.1).

- [ ] **Step 3: Build**

Run: `nix develop -c cargo build --release -p hipfire-runtime --example daemon`
Expected: compiles. (All callees still take `&mut Gpu`, now fed `&mut gpus.devices[0]`.)

- [ ] **Step 4: Coherence gate (the byte-neutral regression net)**

Run: `nix develop -c ./scripts/coherence-gate.sh`
Expected: clean across the matrix (fluent, no panic / zero-tokens / timeout). No `FORWARD_LOWERED` md5 A/B here — the forward is unchanged, so coherence is the correctness net for the ownership move.

- [ ] **Step 5: Commit** (do NOT rustfmt daemon.rs)

```bash
git add crates/hipfire-runtime/examples/daemon.rs
git commit -m "refactor(daemon): fn main() owns Gpus::single (byte-neutral executor-merge prep)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

**BLOCKED escalation:** if a `gpus.devices[0]` reborrow conflicts with another live borrow of `gpus` in `main()` (should not happen in 2.0 — nothing takes `&mut gpus` yet), STOP and report. Do not paper over it with `.clone()` or `unsafe`.

### Task 2.0b: Byte-neutral `Gpus` threading through the `generate()` dispatcher (daemon)

**Why a separate task (verified 2026-07-06):** every arch's `generate_<arch>` is routed through ONE dispatcher `fn generate(m, gpu: &mut Gpu, drafter_gpu, …)` (daemon.rs:8451, **2463 lines, 62 direct `gpu` refs**), called once from `main()` at daemon.rs:2594. That dispatcher is shared plumbing the *first* arch would otherwise have to migrate; pulling it out as its own **byte-neutral** task keeps the per-arch tasks small and isolates the one big daemon edit behind the coherence gate (same pattern as 2.0).

**Byte-neutral contract:** change ONLY `fn generate`'s own `gpu: &mut Gpu` → `gpus: &mut Gpus` and its 62 internal `gpu` uses → `gpus.devices[0]`; update `main()`'s call site (daemon.rs:2594) to pass `&mut gpus`. **Every `generate_<arch>` keeps its `&mut Gpu` parameter** and is fed `&mut gpus.devices[0]` (they migrate one at a time in 2.1–2.5). `drafter_gpu: Option<&mut Gpu>` is UNCHANGED (spec-decode drafter device, out of scope). No forward behavior change.

**Files:**
- Modify: `crates/hipfire-runtime/examples/daemon.rs` — `fn generate` (8451–~10914) + its `main()` call site (2594). **Do NOT rustfmt this file.**

**Interfaces:**
- Consumes: `gpus.devices: Vec<Gpu>` (main already owns `gpus: Gpus` from Task 2.0).
- Produces: `generate()` carries `gpus: &mut Gpus`; all `generate_<arch>` still take `&mut Gpu`, fed `gpus.devices[0]`.

- [ ] **Step 1:** `nix develop -c bash -c "awk 'NR>=8451 && NR<10914' crates/hipfire-runtime/examples/daemon.rs | grep -c '\bgpu\b'"` → ~62. Note the `generate_<arch>` call sites (each will receive `&mut gpus.devices[0]`) and the `drafter_gpu` uses (leave those alone).
- [ ] **Step 2:** `fn generate(..., gpu: &mut rdna_compute::Gpu, ...)` → `gpus: &mut hipfire_hardware::Gpus`; every internal `gpu` (NOT `drafter_gpu`, NOT `pflash_drafter_gpu`) → `gpus.devices[0]` (`&mut gpu` → `&mut gpus.devices[0]`, etc.); each `generate_<arch>(m, gpu, …)` → `generate_<arch>(m, &mut gpus.devices[0], …)`. Update `main()`:2594 `generate(m, &mut gpus.devices[0]/gpu, …)` → `generate(m, &mut gpus, …)`.
- [ ] **Step 3:** `nix develop -c cargo build --release -p hipfire-runtime --example daemon` → compiles.
- [ ] **Step 4:** `nix develop -c ./scripts/coherence-gate.sh` → clean (byte-neutral regression net; no md5 A/B — forward unchanged).
- [ ] **Step 5:** Commit (do NOT rustfmt daemon.rs): `refactor(daemon): generate() dispatcher carries Gpus (byte-neutral)`.

**BLOCKED escalation:** if threading `gpus` collides with the `drafter_gpu`/spec-decode borrows, STOP and report — do not touch `drafter_gpu`.

### Task 2.1–2.5: Per-arch single-GPU driver migration (repeat for each arch)

Do this **once per arch**, in this order (simplest dense first): **qwen2** (`qwen2.rs:1380`), **lfm2moe** (`forward.rs:696`), **minimax** (`forward.rs:754`), **deepseek4** (`forward.rs:2153`), **qwen35** (`qwen35.rs:13061`). Each is one commit with its own byte-exact gate.

**Prereq:** Tasks 2.0 + 2.0b done, so `main()` and `generate()` already carry `gpus: Gpus` and feed `&mut gpus.devices[0]` to this (still-unmigrated) arch. This task flips exactly this arch to `&mut gpus`.

**Files (per arch):**
- Modify (arch crate): the whole forward chain from the public entry to the lowered driver — each `gpu: &mut Gpu` → `gpus: &mut Gpus`, internal helper calls fed `&mut gpus.devices[0]`, and the `run_layer_program(gpu, &ctx, &program, &mut bind)` call → `run_layer_program_mesh(...)`. **qwen2 (Task 2.1):** `forward_step`(:717), `forward_step_with_embed`(:747), `forward_step_greedy`(:969), `forward_step_after_x`(:813, incl. its non-lowered hand loop), `forward_step_after_x_lowered`(:1369, the executor call at :1380) — all funnel through `forward_step_after_x`, so atomic. (qwen35/ds4/minimax/lfm2moe: their `*_lowered` driver + entry chain, mapped in that task's brief.)
- Modify (`daemon.rs`, do NOT rustfmt): `generate_<arch>` (e.g. `generate_qwen2`:12650) sig `gpu` → `gpus`, its internal `gpu` refs → `gpus.devices[0]`, and flip `generate()`'s `generate_<arch>(m, &mut gpus.devices[0], …)` → `(m, &mut gpus, …)`. Any OTHER daemon caller of this arch's forward entries migrates too — **qwen2 additionally:** `generate_vl_dots_ocr`(:13381, shares `forward_step_greedy`; sig `gpu`→`gpus`, ~15 refs, called from `main`:2502) and the `main` warm-pass `forward_step` at :2930.

**Threading rule:** each fn on the chain changes `gpu: &mut Gpu` → `gpus: &mut Gpus`; its OTHER `&mut Gpu` helper calls (attention/norms/kv) are fed `&mut gpus.devices[0]` (sequential reborrows — do NOT hold a `&mut gpus.devices[0]` across the `run_layer_program_mesh(mesh, gpus, …)` call); only the executor call gets the whole `gpus`. Must land atomically (signature + all callers) or the crate won't build. qwen2.rs is NOT a fmt-debt file (per-file rustfmt OK); daemon.rs and the qwen35/ds4/minimax `forward.rs` ARE (do not rustfmt).

**Interfaces:**
- Consumes: `run_layer_program_mesh` (Task 1), `DeviceMesh::single()`, `std::slice::from_mut`.

- [ ] **Step 1: Rewrite the call site**

Replace:
```rust
superop::run_layer_program(gpu, &ctx, &program, &mut bind)
```
with:
```rust
superop::run_layer_program_mesh(
    &hipfire_hardware::DeviceMesh::single(),
    gpus,
    std::slice::from_mut(&mut bind),
    &[],                       // no EP partials on 1×1
    &program,
)
```
Change the enclosing driver fn signature `gpu: &mut Gpu` → `gpus: &mut Gpus`; inside, any other single-device use becomes `&mut gpus.devices[0]`. Drop the now-unused local `ctx` (the merged core builds it). Do NOT rustfmt qwen35.rs / the fmt-debt forward.rs files — leave formatting untouched there.

- [ ] **Step 2: Build**

Run: `nix develop -c cargo build --release -p hipfire-arch-<arch> --example daemon` (or `-p hipfire-runtime --example daemon`)
Expected: compiles.

- [ ] **Step 3: Byte-exact gate — `HIPFIRE_FORWARD_LOWERED=0` vs `=1` md5 A/B**

The refactor changes the lowered (`=1`) path; it must still equal the bespoke (`=0`) path token-for-token. On a fixed prompt (record its md5), run the daemon twice and diff committed tokens:
```bash
# pick a small model for this arch; greedy, fixed prompt bytes
M=~/.hipfire/models/<arch-small>.mq4
P="benchmarks/prompts/lru_cache_pep8_strict.txt"
for L in 0 1; do
  HIPFIRE_FORWARD_LOWERED=$L nix develop -c \
    ./target/release/examples/daemon --model "$M" --prompt-file "$P" \
    --max-tokens 64 --temperature 0.0 --emit-token-ids > /tmp/fl_$L.jsonl
done
diff <(grep committed /tmp/fl_0.jsonl) <(grep committed /tmp/fl_1.jsonl) && echo "FORWARD_LOWERED A/B IDENTICAL"
```
Expected: identical committed-token stream (`FORWARD_LOWERED A/B IDENTICAL`). If the daemon CLI flags differ, use this arch's existing coherence-gate invocation with `HIPFIRE_FORWARD_LOWERED` set — the requirement is a byte-identical `=0`/`=1` committed-token comparison on identical prompt bytes.

- [ ] **Step 4: Coherence gate**

Run: `nix develop -c ./scripts/coherence-gate.sh`
Expected: this arch fluent, on-topic, no attractor/loop (soft-diff OK, panics/zero-tokens/timeouts FAIL).

- [ ] **Step 5: Commit**

```bash
rustfmt --edition 2021 --config skip_children=true <only the files you edited that are NOT fmt-debt>
nix develop -c cargo clippy -p hipfire-arch-<arch>
git add crates/hipfire-arch-<arch>/...
git commit -m "refactor(<arch>): single-GPU forward through merged run_layer_program_mesh

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 2.6: Migrate EP call sites, delete the shim + old single-GPU signature

**Files:**
- Modify: `crates/hipfire-arch-minimax/src/forward.rs:1189`, `crates/hipfire-arch-deepseek4/src/forward.rs:2242`, `:2888` (call `run_layer_program_mesh` directly, drop the `residual_dim` arg)
- Modify: `crates/hipfire-dispatch/src/ep.rs` (delete `run_layer_program_ep` shim; keep the `ensure_rank_streams` re-export or repoint callers to `superop::ensure_rank_streams`)
- Modify: `crates/hipfire-dispatch/src/pipeline/superop.rs` (delete the old `&mut Gpu` `run_layer_program` — now unused after Tasks 2.1–2.5)

**Interfaces:**
- Consumes: `run_layer_program_mesh` (Task 1).

- [ ] **Step 1: Rewrite the 3 EP call sites**

Replace each:
```rust
hipfire_runtime::ep::run_layer_program_ep(&mesh, gpus, binds.as_mut_slice(), partials, &program, hidden)
```
with:
```rust
hipfire_dispatch::pipeline::superop::run_layer_program_mesh(&mesh, gpus, binds.as_mut_slice(), partials, &program)
```
(`hidden` was `residual_dim`; the core derives it from `partials[0]`.) Keep the `mesh` these sites already build (`DeviceMesh::rect(&[(Ep, partials.len())])`).

- [ ] **Step 2: Delete the dead code**

Remove `run_layer_program_ep` from `ep.rs` and the old 4-arg `run_layer_program(gpu, ctx, program, bindings)` from `superop.rs`. Repoint any `ensure_rank_streams` import to `superop::ensure_rank_streams`. If `ep.rs` is now empty besides the re-export, collapse it (remove the module + its `mod ep;` and re-exports).

- [ ] **Step 3: Build the workspace**

Run: `nix develop -c cargo build --release --features deltanet --workspace`
Expected: 0 errors; no reference to the deleted functions remains (`grep -rn run_layer_program_ep crates/` returns nothing).

- [ ] **Step 4: EP byte-identity + coherence**

Run: `bash .agent-progress/run-ep-parity.sh && grep "PARITY exit" .agent-progress/ep-parity.log`
Expected: `PARITY exit: 0`. Then `nix develop -c ./scripts/coherence-gate.sh` clean, and `nix develop -c ./scripts/serve-multiturn-gate.sh` green for the EP arches (ds4/minimax).

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-arch-minimax/src/forward.rs crates/hipfire-arch-deepseek4/src/forward.rs crates/hipfire-dispatch/src/ep.rs crates/hipfire-dispatch/src/pipeline/superop.rs
git commit -m "refactor(dispatch): all paths on run_layer_program_mesh; delete dual executor

EP call sites call the merged core directly; run_layer_program_ep + the old
single-GPU run_layer_program are deleted. ONE executor (device-mesh plan §1).
ep_decode_parity + coherence + serve-multiturn green.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 2.7: Update the ledgers

**Files:**
- Modify: `.agent-progress/device-mesh-status.md`, `.agent-progress/device-mesh-phase0.md`, `docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md` (mark Phase-0b executor merge DONE; note the deleted dual executor).

- [ ] **Step 1: Record completion** — flip the "DECISION (RETRACTED)" note to "DONE: merged executor landed (commits …)"; update the plan's Phase-0b bullet from "REINSTATED" to "✅ merged".
- [ ] **Step 2: Commit** (`git add -f` the plan doc — `docs/superpowers` is gitignored):
```bash
git add .agent-progress/device-mesh-status.md .agent-progress/device-mesh-phase0.md
git add -f docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md
git commit -m "docs(device-mesh): executor merge DONE — one run_layer_program (plan §1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Verification (whole-plan exit criteria)

- **No-GPU / CI:** `nix develop -c cargo test -p hipfire-dispatch --lib` (incl. `moe_is_ep`); `nix develop -c cargo build --release --features deltanet --workspace` (0 errors).
- **Single-GPU byte-exact:** `HIPFIRE_FORWARD_LOWERED=0` vs `=1` committed-token md5 identical for each migrated arch (qwen2/lfm2moe/minimax/ds4/qwen35), on byte-identical prompts (record md5).
- **EP byte-exact:** `ep_decode_parity` `PARITY exit: 0` (qwen3.6-35b-a3b), before (shim) and after (direct) migration.
- **Coherence:** `./scripts/coherence-gate.sh` clean across the matrix; `./scripts/serve-multiturn-gate.sh` green for ds4/minimax.
- **Dead code gone:** `grep -rn "run_layer_program_ep\|fn run_layer_program\b" crates/` shows only `run_layer_program_mesh`.
- **Structural win:** the two-executor duplication is deleted; single-GPU, PP (via Phase 1c banded driver), and EP all route through `run_layer_program_mesh`.

## Out of scope (do not do here)
- TP all-reduce hints wired into the executor (Phase 5 — `CollectiveHint::AllReduce{Tp}` consumption); this plan keeps the executor's collective set = the current EP reduce.
- `BandXfer` injection inside the executor (PP sync stays in the outer driver, as today — plan §1's "outer driver between per-layer calls").
- `ModelParallel`/`ArchDispatch` daemon hoist (Phase 3).
- Composed/ragged/multi-group meshes (Phase 5b) — the merged core keeps the single Ep-group `debug_assert`.

## Self-review notes
- **Spec coverage:** plan §1 "exactly ONE executor" → Tasks 1 + 2.1–2.6 (merged core + all call sites + dual-executor deletion). "1×1 leaves stream state untouched" → merged core never calls `ensure_rank_streams`; the non-EP branch never touches `active_stream` (memset stays sync). "ctx built inside" → merged core. "residual_dim derived" → `partials[0].numel()`.
- **Borrow safety:** `partials` is a separate `&[GpuTensor]` slice (not pulled from `&mut bindings`), so the memset/all-reduce immutable borrow never conflicts with the `&mut bindings[r]` mutable borrow — the reason it stays an explicit param (`GpuTensor` is not `Clone`).
- **Byte-identity risk (single-GPU):** the merged core rebuilds `DispatchCtx` per op and calls `bind_thread` per op, where the old single-GPU entry built `ctx` once. **Confirm `DispatchCtx::new` is side-effect-free / non-allocating** (Task 2.1 Step 1 pre-check); if it caches or allocates, hoist ctx per-rank outside the op loop for `n==1`. `bind_thread` is idempotent (sets current device) → no byte effect. The `=0/=1` md5 gate is the backstop.
- **Type consistency:** `run_layer_program_mesh` signature is identical everywhere it's referenced (Tasks 1, 2.1–2.6). `residual_dim` never appears in the merged signature (derived). `ensure_rank_streams` has one definition (superop.rs) after Step 1.

## Execution log

- **Task 1 — COMPLETE** (commit `fd879524`, 2026-07-06). Merged `run_layer_program_mesh` core + `moe_is_ep` + relocated `ensure_rank_streams` in `superop.rs`; `run_layer_program_ep` reduced to a delegating shim. Gates: 145/145 dispatch lib tests + `moe_is_ep` unit test; `ep_decode_parity` **PARITY exit 0** (35B-A3B EP byte-identical through the shim). Task review: Approved (EP fold verified as a faithful verbatim move; non-EP arm confirmed stream-clean).
  - **Open finding (Minor, for the final whole-branch pass): re-harden the softened length guard.** `superop.rs:407` uses `debug_assert_eq!(gpus.devices.len(), bindings.len())`, where the old `run_layer_program_ep` had a **release-visible** `assert_eq!(bindings.len(), n)`. The EP `partials.len() == n` check is still a hard `assert_eq!`, but the gpus/bindings mismatch is now debug-only. Cheap length check on a hot-path invariant → promote back to `assert_eq!` before merge. (Recorded per user request 2026-07-06.)
  - Incidental brief-authorized `rustfmt` churn on pre-existing `superop.rs` items (logic-neutral diff noise); `.agent-progress/run-ep-parity.sh` got a benign `cargo` → `nix develop -c cargo` gate-infra fix.
