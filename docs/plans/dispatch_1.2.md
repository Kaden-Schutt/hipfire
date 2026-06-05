# Ship 1.2 — Paro fused-kernel recovery (qwen35 decode)

**Branch:** `feature/dispatch-unification`
**Tracking:** #397 (ship 1, item 1.2)
**Depends on:** Ship 1.1 (treated as landed) — provides `QKVZA4` op-kind, the
`qkvza_*`/`qkv_*` guards, the `launch_fused` QKVZA arm, `qkvza_via_execute_steps`,
and `qkv_via_execute_steps`.
**Phase 0 contracts:** PR #402 (`docs/plans/dispatch-phase0-decisions.md`) — 0.2
(scratch ownership) governs Commit 1; 0.4 (`HasWmmaW32 → HasWmma`) governs arch
predicates; 0.6 governs verification (`HIPFIRE_FORCE_UNFUSED`, RDNA4 non-optional,
byte-identical token streams).

**Scope change (2026-06-05):** Q4K and Q8_0 fused-table completion **moved to
Ship 2** (entry + llama/qwen2 model integration + GPU byte-parity land together,
avoiding a merged-but-GPU-unexecuted window). 1.2 is now **Paro-only**: recover
the ParoQ4G128 fused path that the dispatch refactor regressed to per-op.

> **Review provenance.** This revision folds three adversarial reviews:
> `findings/dispatch_1.2_plan_rev_gemini.md` (Antigravity),
> `findings/dispatch_1.2_plan_rev_glm5.md` (GLM-5), and
> `/tmp/dispatch_1.2_plan_rev_claude.md` (claude). Consensus findings drove the
> three biggest changes below (scratch home, Raw guards, force-unfused oracle).

---

## Goal

Every Paro projection in qwen35's single-token decode (`forward_scratch_layers`)
goes through `execute_steps` **and selects the fused Paro kernel** — restoring the
two-launch fused path master had, now routed through the pipeline. After 1.2, a
Paro decode forward issues `fused_qkvza_paro4g128t` (DeltaNet LA),
`fused_gate_up_paro4g128t` (FFN), and the `m3=0` synthesis of 3-way QKV (FullAttn),
not N per-op GEMVs.

---

## Grounded state — corrections to the roadmap (verified at `715f966c` + 1.1)

The #397 roadmap's 1.2 section has three factual errors; all three are confirmed by
the external reviews:

1. **Paro is regression recovery, not a new feature.** `master` qwen35 calls
   `fused_qkvza_paro4g128t` / `fused_gate_up_paro4g128t` across decode/prefill/multi
   (`master:qwen35.rs:12368, 12788, 13349, 13497, …`). On this branch nobody calls
   them — `715f966c` ("route ParoQ4G128 through per-op gemv path") replaced the
   fused path with per-op. **Perf baseline to recover = `master` (fused).**
2. **Producer is `RmsnormAutomatic(rotation=None)`, not `Givens`.** Paro fused
   kernels rotate **internally per-weight** (`paro4g128t_quad_rotate` /
   `paro4g128t_dual_rotate` then `*_pack4`); master feeds them plain rmsnorm
   (`&s.tmp`). `rotation=Givens` would FWHT the input the kernel re-rotates → garbage.
   Confirmed: `qkvza_via_execute_steps` Givens branch already sets `rotation:
   RotationPlan::None`.
3. **`fused_qkv_paro4g128t` (3-way) does not exist.** Only 2-way
   `fused_gate_up_paro4g128t` and 4-way `fused_qkvza_paro4g128t` exist. Master
   synthesizes 3-way FullAttn QKV via the **4-way kernel with the 4th width zeroed**
   (`master:qwen35.rs:13497`: `a3=wq.buf, m3=0`). 1.2 reproduces this *inside the
   dispatch crate* — see Commit 2, Option A.

### Three load-bearing facts the reviews surfaced

- **F-scratch (gemini, glm5, claude): the rotation pool must live in `gpu.scratch`,
  not `ResourceManager`.** `DispatchCtx` is constructed **per token** on the stack
  in `forward_scratch_layers` (`let ctx = DispatchCtx::new(gpu)`). A pool owned by
  `ResourceManager`/`DispatchCtx` would (a) re-allocate every decode step (hot-loop
  allocation) and (b) **leak VRAM** — `GpuTensor`/`DeviceBuffer` does not `Drop`-free;
  `ResourceManager::drop` cannot reach `&mut Gpu` to call `free_tensor`. `gpu.scratch`
  (`ScratchState`, `scratch.rs:25`) persists for the session, is reachable from
  `launch_fused` (which holds `&mut Gpu`), and is exactly where `mq_x_rot` already
  lives. `DeviceBuffer::alias()` lets us build owned `GpuTensor` descriptors over the
  scratch with **no Rust borrow held**, sidestepping the `RefCell`/lifetime trap
  entirely. **`ResourceManager` stays the empty `_priv: ()` stub.**

- **F-guard (glm5 F13): Paro guards must match `GemvInput::Raw`, not `Prerotated`.**
  `gemv_steps_uniform` (`steps.rs`) matches **only** `Step::Gemv { input:
  GemvInput::Prerotated(_), .. }` and returns `false` for `Raw`. The Paro branches of
  `qkvza_via_execute_steps`/`qkv_via_execute_steps` emit `GemvInput::Raw(x_rot)`
  (verified) because the per-weight Givens rotation happens inside the kernel/`run_auto`,
  not as a producer pre-rotation. **If a Paro guard reuses `gemv_steps_uniform`, the
  `FUSED_TABLE` row never matches → fused Paro never fires → silent per-op fallback →
  the entire 1.2 perf recovery silently no-ops.** New guards need a Raw-accepting
  uniformity check.

- **F-gateup (glm5 F1/F2, claude M2): Paro gate+up uses 2 rotation buffers, not 1.**
  `fused_gate_up_paro4g128t` takes 1 explicit `x_rot_gate` **and internally aliases
  `self.scratch.mq_x_rot` as `x_rot_up`** (`gemv.rs:537–542`, asserts `≥ k`). The
  caller must guarantee `mq_x_rot` is sized `≥ k` and distinct from the explicit gate
  buffer. `fused_qkvza_paro4g128t` (4-way) takes 4 explicit buffers and does **not**
  touch `mq_x_rot`.

---

## Inventory — Paro entries, the 6-layer stack

(1) `KernelKey` `types.rs` · (2) `FusedQkvFamily::run` arm `fused_qkv.rs` · (3) table
`fused_qkv_table.rs` · (4) `FUSED_TABLE` `FusedPattern` `steps.rs` · (5) Raw guard
`steps.rs` · (6) `launch_fused` arm. All ❌ = add.

| Entry | pattern | (1)–(6) | producer | kernel fn | rotation scratch |
|---|---|---|---|---|---|
| **FusedGateUpParo4G128T** | `GATE_UP2` | ❌ all | `Rmsnorm(None)` | ✅ `fused_gate_up_paro4g128t` | **1 explicit + `mq_x_rot` (internal, ≥k)** |
| **FusedQkvzaParo4G128T** | `QKVZA4` | ❌ all | `Rmsnorm(None)` | ✅ `fused_qkvza_paro4g128t` | **4 explicit** |
| **FusedQkvParo4G128T** (3-way, FullAttn) | `QKV3` | ❌ all | `Rmsnorm(None)` | ✅ via `fused_qkvza_paro4g128t` w/ `m3=0` | **4 explicit (x_rot3 aliased, unused)** |

`paro4g128t` arch predicate = `ArchPredicate::HasDp4a` (matches
`dtype_arch_predicate(ParoQ4G128)`, `types.rs:441`) — **not** `HasWmmaW32`. If 0.4's
`HasWmmaW32 → HasWmma` collapse lands first, this is unaffected (Paro is dp4a-gated).

---

## Plan — three commits

### Commit 1 · `gpu.scratch` Paro rotation buffers + `FusedQkvParams` scratch field

**Scope:** `rdna-compute` (scratch + helper) + dispatch crate (`FusedQkvParams`).
No behavior change yet. **`ResourceManager` is untouched (stays a stub).**

1. **`rdna-compute/src/scratch.rs`:** add `pub paro_fused_scratch:
   Option<Vec<GpuTensor>>` to `ScratchState` (lazy, `None` by default — keeps
   GPU-free `for_test()`/no-GPU construction working).
2. **`rdna-compute/src/gemv.rs`:** add `ensure_paro_fused_scratch(&mut self, k:
   usize) -> HipResult<()>` — allocates 4 `[k]` F32 buffers on first use; on
   subsequent calls, grows any buffer whose `size()/4 < k` (gemini's impl). 4 covers
   QKVZA (4 explicit) and the 3-way `m3=0` synthesis; gate+up uses `buffers[0]` as
   `x_rot_gate` and the kernel's internal `mq_x_rot` for `up`.
3. **`gemv.rs`:** also ensure `mq_x_rot` is sized `≥ k` (via the existing
   `ensure_mq_signs`) before any gate+up Paro launch (F-gateup), and `debug_assert!`
   the gate buffer pointer `!=` `mq_x_rot` pointer.
4. **`FusedQkvParams<'a>` (`families/fused_qkv.rs`):** add `rot_scratch: &'a
   [GpuTensor]` — **owned aliased descriptors, single indirection** (not `&'a [&'a
   GpuTensor]`; glm5 F7). Empty slice for non-Paro keys; existing arms ignore it.
5. **`launch_fused` reaches scratch via `&mut Gpu`** (already in signature): call
   `gpu.ensure_paro_fused_scratch(k)`, then build aliased `GpuTensor` descriptors
   over `gpu.scratch.paro_fused_scratch` (via `DeviceBuffer::alias()`) into a local
   `Vec`, pass as `rot_scratch`. No borrow held across the kernel launch → no
   `RefCell`, no lifetime conflict.

**Verify:** `cargo test -p hipfire-dispatch -p rdna-compute`; scratch helper unit
test (allocates 4, reuses, grows on larger k, buffers distinct & non-aliasing).

**Est:** ~90 lines (scratch field + helper + `FusedQkvParams` field + alias plumbing).

---

### Commit 2 · Paro fused entries — recover the regressed path

**Scope:** dispatch crate + (no new qwen35 call sites — the 1.1 helpers already emit
the right step slices for Paro).

1. **types.rs:** add `FusedGateUpParo4G128T`, `FusedQkvzaParo4G128T`,
   `FusedQkvParo4G128T`.
2. **fused_qkv.rs — three `run` arms** (each asserts `m%8==0 && k%128==0` and returns
   `UnsupportedVariant` rather than panicking if violated):
   - `FusedGateUpParo4G128T` → `gpu.fused_gate_up_paro4g128t(...)` with
     `params.rot_scratch[0]` as `x_rot_gate`.
   - `FusedQkvzaParo4G128T` → `gpu.fused_qkvza_paro4g128t(...)` with the 4
     `rot_scratch` buffers.
   - `FusedQkvParo4G128T` (**Option A, gemini §4 — encapsulated `m3=0` synthesis**):
     call `gpu.fused_qkvza_paro4g128t` with `a0/a1/a2 = wq/wk/wv`, `y0/y1/y2 = q/k/v`,
     `m0/m1/m2 = mq/mk/mv`, **`a3=wq` (alias), `y3=q` (alias), `m3=0`** (zero-width →
     no 4th GEMV/write), `x_rot3 = rot_scratch[0]` (alias, unused). This keeps the
     degenerate 4th slot inside the dispatch crate — **no dummy step leaks into
     `qwen35.rs`** — and matches master's speed baseline. Supersedes the previous
     "leave FullAttn QKV per-op (Option B)" plan, which was a preventable regression
     vs master (and rested on an unverified "minority of layers" claim — claude M3).
3. **fused_qkv_table.rs:** register all three with `ArchPredicate::HasDp4a`.
4. **steps.rs — Raw-accepting guards (F-guard, glm5 F13):** add a
   `gemv_steps_uniform_raw(steps, dtype, require_no_awq)` helper (mirrors
   `gemv_steps_uniform` but matches `GemvInput::Raw(_)`), and write
   `guard_gate_up_paro4g128t`, `guard_qkvza_paro4g128t`, `guard_qkv_paro4g128t`. Each:
   `force_unfused` early-return → `window_gemv_dtype == ParoQ4G128` →
   `gemv_steps_uniform_raw(..)` → `m%8==0 && k%128==0`. Add `GATE_UP2`/`QKVZA4`/`QKV3`
   `FusedPattern` rows for the three keys.
5. **steps.rs `launch_fused`:** add three arms. They `ensure_paro_fused_scratch(k)`,
   build aliased descriptors, and call `fused_qkv.run` with `rot_scratch` populated.
   Producer step is `RmsnormAutomatic(rotation=None)` → `activated` (plain rmsnorm) is
   the kernel `x` it rotates internally.

**No qwen35 changes needed**, but **verify** the 1.1 helpers route Paro into these
patterns: `qkvza_via_execute_steps` (4-way) and `gate_up_via_execute_steps` (2-way)
emit `[Rmsnorm(None), Gemv(Raw)×N]` for ParoQ4G128 (confirmed); `qkv_via_execute_steps`
(3-way FullAttn) does likewise via its Givens branch.

**Paro weight alignment invariant (glm5 F4):** the guards' `m%8/k%128` checks mask the
*fused* kernel's asserts, but the per-op fallback (`gemv_hfq4g128`) has its own
alignment contract. Before relying on the guard, **confirm every ParoQ4G128 weight in
qwen35 has `m % 8 == 0` and `k % 128 == 0`** (group-128 quant → almost certainly true).
State this as an invariant; the guard is an optimization gate, not the correctness
boundary.

**Verify:**
- **Primary oracle — byte-identical committed token IDs vs `master`** (same fused
  kernel, same accumulation order) on a qwen35-A3B-PARO model, gfx1100 **and gfx1201**
  (`HIPFIRE_EMIT_TOKEN_IDS=1`, temp 0.0, fixed prompt, md5 recorded).
- **`HIPFIRE_FORCE_UNFUSED` parity = coherence + cosine ≥ 0.9999, NOT byte-identical**
  (claude H1, glm5 F11). Per-op (per-weight Givens via `run_auto` → `gemv_hfq4g128`)
  and fused (`quad_rotate` + `pack4`, batched launches) are **different kernels with
  different FP reduction order** — byte equality is not a property they have. The
  master-vs-fused check above is the byte-exact oracle; force-unfused proves the per-op
  path stays *coherent*, not bit-equal.
- `probe_commits.sh master HEAD` A/B on gfx1100 + gfx1201: parity with master (±1–3%)
  and a measurable **gain vs the immediate parent** (per-op → fused). Δ ≥ 5%
  triggers the investigation rule.
- `coherence-gate.sh` + `coherence-gate-dflash.sh` (DeltaNet participates in spec).

**Est:** ~200 lines (3 keys + 3 family arms + 3 table rows + 3 guards +
`gemv_steps_uniform_raw` + 3 launch arms).

---

### Commit 3 · Verification + cleanup

- [ ] Byte-identical-vs-`master` token IDs on A3B-PARO, **gfx1100 + gfx1201**.
- [ ] Force-unfused: coherence pass + cosine ≥ 0.9999 (not byte-exact) on both archs.
- [ ] `probe_commits.sh master HEAD` ±1–3% (parity recovered) + gain vs parent.
- [ ] `coherence-gate.sh` + `coherence-gate-dflash.sh`.
- [ ] **Multi path still works (glm5 F10):** confirm `forward_scratch_layers_multi`
      (still per-op, deferred to Ship 5) passes the coherence gate — the shared 1.1
      helpers feed it too; verify no regression on the un-migrated path.
- [ ] **Arch-predicate / coverage golden:** `(op × dtype × arch)` golden incl. the
      RDNA4 row (Phase 0.4 gate) for the three Paro keys; `force_unfused` rejection
      tests on the guards.
- [ ] **Grep audit:** `fused_qkvza_paro4g128t` / `fused_gate_up_paro4g128t` reachable
      only via `FusedQkvFamily::run` — no direct arch decode call sites.
- [ ] Dev-log the Paro weight alignment invariant and the Ship-2 handoff for Q4K/Q8_0.

---

## Risks

1. **Silent perf no-op (F-guard).** If guards reject `Raw`, fused never fires and 1.2
   "passes" coherence while delivering zero perf recovery. The `probe_commits.sh`
   gain-vs-parent check is the backstop — a *missing* gain means the guard didn't fire.
   Add an assert/log in `launch_fused` (debug builds) that the Paro arm was actually
   reached at least once per forward.
2. **Producer rotation = None (not Givens).** Highest correctness risk. A future edit
   setting `rotation=Givens` for Paro double-rotates → garbage. The master-vs-fused
   byte check is the guard.
3. **gate+up hidden `mq_x_rot` aliasing (F-gateup).** `mq_x_rot` must be `≥ k` and
   distinct from the explicit gate buffer. Commit 1 step 3 enforces both; the
   `debug_assert` catches accidental aliasing.
4. **`m3=0` synthesis safety (gemini §4).** Reusing `wq`/`q`/`rot_scratch[0]` for the
   degenerate 4th slot is safe **only because `m3=0` guarantees no 4th-projection
   write**. Add a comment at the call and confirm the kernel skips writes when `m3=0`
   (master relied on this).
5. **Perf direction.** Recovering master parity is the bar; per-op → fused must be a
   gain, not a loss. Bisect any loss against master per CLAUDE.md; do not wave off as
   noise (Δ ≥ 5% rule).

---

## Out of scope (tracked elsewhere)

| Item | Ship |
|---|---|
| **Q4K fused entries** (`FusedQkvQ4K`/`FusedGateUpQ4K` interpreter wiring + goldens + GPU parity) | **Ship 2** (with llama integration) |
| **Q8_0 fused gate+up** (`FusedGateUpQ8_0` full stack + GPU parity) | **Ship 2** |
| `Step::Rmsnorm` variant (deferred, **not rejected** — needed for the Ship 6 forward-as-pipeline capstone; glm5 F9) | Ship 6 |
| Paro fused in `forward_prefill_chunk` + `forward_scratch_layers_multi` | Ship 5 |
| Dedicated `fused_qkv_paro4g128t` 3-way kernel (1.2 uses the `m3=0` synthesis) | not planned (synthesis suffices) |
| Phase 0.4 `HasWmmaW32 → HasWmma` collapse | Phase 0 cleanup |

---

## Dev log

| Date | Commit | What | Result |
|---|---|---|---|
| 2026-06-05 | — | Plan written; Q4K/Q8_0 → Ship 2; folded gemini + glm5 + claude reviews (scratch→`gpu.scratch`, Raw guards, 2-buffer gate+up, Option A 3-way, force-unfused→cosine) | — |
| 2026-06-05 | `6da3c7bb` | Commit 1: paro_fused_scratch (4×[k] F32) in ScratchState + ensure_paro_fused_scratch + FusedQkvParams.rot_scratch | Clean build, coherence pass, all dispatch tests green |
| 2026-06-05 | `284c119e` | Commit 2: 3 Paro KernelKeys + guards (Raw-accepting) + FUSED_TABLE entries + launch_fused arms + fused_qkv dispatch + table rows | Clean build, coherence pass, 12 new tests green (paro guards + coverage + arch). GPU verification deferred to coworker (D-11) |
