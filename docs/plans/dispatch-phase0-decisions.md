# #397 Phase 0 — Dispatch Contract Decision Spike

**Status:** Design gate. No production code in this document. Unblocks #397 Ship 1–6.
**Date:** 2026-06-04
**Author of spike:** dispatch-phase0 work-chunk C6
**Scope:** The 6 cross-cutting contracts the dispatch-unification crate
(`crates/hipfire-dispatch/`, PR #393, branch `feature/dispatch-unification`)
must settle *before* any model is migrated off the inline `match dtype` /
`if has_wmma_w32()` dispatch that lives in the arch crates today.

> **How to read the file:line citations.** Two code bases are referenced:
> - **master** (this worktree) — the current production dispatch, all inline in
>   `crates/hipfire-runtime/` + the arch crates. This is the correctness oracle.
> - **`feature/dispatch-unification`** (fetched as `FETCH_HEAD` from `fivetide`,
>   tip merge `9feb5644`, which folds in stopgap `953ea648`) — the new
>   `hipfire-dispatch` crate. Citations into it are written `dispatch:<path>`.
>
> All line numbers were re-grounded by grep/Read against these two trees on
> 2026-06-04. The dispatch crate is **NOT** on `origin/master`.

---

## Crate dependency graph (load-bearing for 0.1 and 0.2)

Grounded from the Cargo.toml files on `feature/dispatch-unification`:

```
hip-bridge        (FFI)
   ▲
rdna-compute      (kernels, ArchCaps, Gpu, GpuTensor, DType)
   ▲   ▲
   │   └────────────── hipfire-dispatch   (deps: hip-bridge, rdna-compute ONLY)
   │                        ▲      ▲
hipfire-runtime ────────────┘      │   (deps: rdna-compute, hipfire-dispatch)
   ▲                               │
hipfire-arch-{qwen35,deepseek4,llama,...} (deps: hipfire-runtime, rdna-compute, hipfire-dispatch)
```

- `dispatch:Cargo.toml` — `hipfire-dispatch` depends **only** on `hip-bridge`
  and `rdna-compute`. It does **not** depend on `hipfire-runtime` or any arch crate.
- `crates/hipfire-runtime/Cargo.toml:54` and the arch crates'
  `Cargo.toml` all depend **down** on `hipfire-dispatch`.

The arrows only point upward. `hipfire-dispatch` is a *leaf-ward* crate: it can
see `rdna-compute` types but **cannot** name a `KvCache`, `PrefillBatchScratch`,
or `Qwen35Weights` (those live in `hipfire-runtime` / the arch crates above it).
This single fact decides 0.2 and constrains 0.1.

---

## 0.1 Per-layer resolve-cache + `invalidate()` — CORRECTNESS-CRITICAL but PROSPECTIVE

### (a) Grounded current-state finding
- The mutation that triggers the hazard:
  `KvAdaptive::maybe_downshift` at
  `crates/hipfire-runtime/src/kv_adaptive.rs:178`. It walks `self.steps`
  and, for each crossed threshold, calls `kv.transcode_v_step(...)` /
  `kv.transcode_k_step(...)` and updates `self.cur_v` / `self.cur_k`
  **mid-sequence** (kv_adaptive.rs:184–185). The underlying
  `KvCache::v_mode` field is mutated inside `transcode_v_step`
  (`crates/hipfire-runtime/src/llama.rs:4772`, also :4889) and
  `transcode_k_step` (`llama.rs:4925`). It is called after **every**
  committed token write on all three paths — prefill, post-prefill, and
  decode (`daemon.rs:8027`, `:8097`, `:8462`).
- The kernel key that *must* track that mutation: the KV-write and
  flash-attend families are keyed **per quant tier**, e.g.
  `KvWriteAsym4 / KvWriteAsym3 / KvWriteAsym2 / KvWriteQ8_0 / KvWriteF32`
  and the parallel `AttnFlashAsym4 / …Asym3 / …Asym2 / AttnFlashQ8_0 / AttnF32`
  (`dispatch:src/types.rs:230–238` and `:219–228`; registered in
  `dispatch:src/tables/attention_table.rs:7–48`). The selected
  `KernelKey` is therefore a **function of the live `v_mode`/`k_mode`**, which
  `maybe_downshift` changes underfoot.
- **Today there is no hazard**, because `KernelRegistry::resolve()`
  (`dispatch:src/tables/mod.rs:31–58`) is re-run on every call — it does a
  `HashMap::get(&key)` + linear predicate scan and returns a borrow; nothing is
  memoised. A fresh `v_mode` simply produces a fresh `key` next call.

### (b) Correctness-critical vs architecture
**Correctness-critical.** If a cache returns a stale `KvWrite*`/`AttnFlash*`
variant after a downshift, the engine writes/reads the KV cache with the wrong
quant codec for a sequence position. That is a silent-wrong-output /
memory-format-mismatch class bug (sibling of the #30 and Hunt-2 H3 OOB classes).

### (c) Codeable vs human-decision
**Mixed, but the recommendation is codeable.** The hazard is *prospective*:
it only materialises **if Ship 2 introduces a per-layer `KernelKey` cache** at
init (the stated Ship-2 optimisation — resolve once, store the variant pointer,
skip the per-call HashMap lookup). The mechanism of the fix is codeable once the
author picks one of the two policies in (e). The *policy choice* is the
human-decision (open Q below).

### (d) RECOMMENDED decision
**Cache the arch/shape-stable families; exempt the KV-tier-dependent families
from caching entirely (do NOT add a runtime `invalidate()` hook into the
adaptive controller).** Rationale:
1. Cleanest dependency story. An `invalidate()` would require
   `KvAdaptive::maybe_downshift` (which lives in `hipfire-runtime`) to reach
   *down* into the dispatch cache and poke it on every downshift. That couples
   the adaptive controller to the cache's internals and adds a "did you remember
   to invalidate?" footgun on **every future** mid-sequence mode mutation — the
   exact recurring-bug shape memory already records for the GDN-tape eligibility
   gate (`feedback_gdn_tape_replay_eligibility`).
2. The exempt set is tiny and statically known: the `KvWrite*` and `AttnFlash*`
   keys are the only ones whose `KernelKey` is a function of a *mid-sequence
   mutable* quant tier. Everything else (GEMV/GEMM/MoE/rotation/fused-QKV) keys
   off the *weight* dtype, which is immutable for the model's lifetime → safe to
   cache at init.
3. The exempt families' `resolve()` is already the cheap path — a single
   `HashMap::get` + a 5-to-9-entry linear scan whose predicates are all
   `ArchPredicate::Always` (`attention_table.rs:8–15, 30–38`). Re-resolving them
   per token is negligible vs the attention kernel itself; the cache buys nothing
   there anyway.

So: Ship 2 caches `resolve()` results for the arch-stable families only, and the
KV-tier families call the live `resolve(key_from_current_mode, …)` each token,
exactly as today. No `invalidate()` surface is added.

### (e) OPEN QUESTION for the author
**Exempt `KvWrite*`/`AttnFlash*` from caching, OR add an `invalidate()` the
adaptive controller calls on downshift?** The recommendation is *exempt*. If the
author wants caching for those too (e.g. a measured per-token win on a
long-context decode), then `invalidate()` must be wired into **every** site that
mutates `v_mode`/`k_mode` — today that is `transcode_v_step` (llama.rs:4772,
:4889) and `transcode_k_step` (:4925), reached via the three `maybe_downshift`
call sites — and a guard test added that fails if a new `v_mode` writer is added
without an invalidate.

---

## 0.2 Scratch-ref ownership — confirm the dep-cycle risk

### (a) Grounded current-state finding
- `dispatch:src/resource/mod.rs:4–25` — `ResourceManager` is currently an
  **empty stub** (`struct ResourceManager { _priv: () }`, `new(_gpu)` ignores its
  argument). It is the placeholder where "scratch ownership moves down a crate"
  would land.
- The real scratch lives **above** dispatch, in the arch crates. The b4adca1
  leak-fix allocator/free pair `PrefillBatchScratch` is defined in three places:
  `crates/hipfire-arch-qwen35/src/qwen35.rs` (the b4adca1 site, with
  `free_gpu` at qwen35.rs:6121 freeing the grouped-WMMA MoE scratch fields
  `moe_y_gate_up_grouped` / `moe_y_down_grouped`),
  `crates/hipfire-arch-deepseek4/src/forward.rs`, and
  `crates/hipfire-runtime/src/llama.rs`.
- Those scratch structs are built out of `rdna_compute::GpuTensor` but are
  **named** in `hipfire-runtime` / arch crates — strictly *above* dispatch in the
  graph.

### (b) Correctness-critical vs architecture
**Architecture (with a correctness tail).** The leak fix itself is already
landed and correct on master; the question is purely *where the allocator
lives*. The correctness tail: if scratch ownership is split across the crate
boundary clumsily, a future double-free / use-after-free becomes possible — the
exact class b4adca1 just fixed.

### (c) Codeable vs human-decision
**Human-decision** (the location), then codeable. The compiler enforces the
constraint: a naïve "move `PrefillBatchScratch` into `ResourceManager`" **does
not compile** — `hipfire-dispatch` cannot name `PrefillBatchScratch` without
depending up on `hipfire-runtime`/arch, which inverts the arrow and creates the
cycle. **The dep-cycle risk is REAL and confirmed.**

### (d) RECOMMENDED decision
**Scratch lifetime stays arch-owned. Pass scratch into dispatch by `&mut`
reference at call time; do not move ownership down into `ResourceManager`.**
Rationale:
1. Moving `PrefillBatchScratch` down requires its definition to move down too
   (or to be re-exposed via `rdna-compute`), because dispatch can only name
   `rdna-compute` types. Either inverts the dependency arrow (cycle) or balloons
   `rdna-compute` into a god-crate holding arch-specific scratch layouts.
2. The arch crate is where scratch *sizing* logic already lives (it knows
   `n_experts`, `moe_y_*` widths, batch caps). Keeping the allocator next to the
   shape logic is what made the b4adca1 fix localisable in the first place.
3. `ResourceManager` can still own **arch-generic, dtype-generic** scratch that
   is expressible in pure `rdna-compute` types (e.g. a reusable rotation
   scratch buffer pool) — that is a legitimate future use of the stub. The line
   to hold is: *model/arch-specific* scratch (anything with `moe_`, expert
   counts, or arch weight layout in its shape) stays arch-owned.

### (e) OPEN QUESTION for the author
**Scratch lifetime: stays arch-owned (passed by `&mut`) vs moves down into
`hipfire-dispatch::ResourceManager`?** Recommendation: stays arch-owned. If the
author still wants it down a crate, the prerequisite is deciding *which* crate
owns the scratch *type* — and that crate must be at or below `rdna-compute` in
the graph, or the build breaks.

---

## 0.3 Paired write-then-attend — the #30-class drift risk

### (a) Grounded current-state finding
- The two halves are dispatched through **separate** keys and **separate**
  families: KV write via `KvWrite*` (`dispatch:src/types.rs:230–237`,
  registered `attention_table.rs:7–25`) and the attend via `AttnFlash*`
  (`types.rs:219–228`, registered `attention_table.rs:28–47`). Both populate the
  same `AttentionFamily` registry but resolve as independent keys.
- The shared derived state they must agree on: both must be selected from the
  **same** live KV quant tier (`v_mode`/`k_mode`, see 0.1) and the **same**
  derived FWHT/AWQ sub-plan. Today that agreement is implicit — the caller
  passes the same `v_mode` into the key-building helper for both halves, so they
  cannot diverge *as long as the caller derives the key once and reuses it*.
- The #30-class risk (silent-wrong-output from a per-row/derived-state fallback
  being dropped on one of two paired paths) is the precedent:
  `project_kernel_surface_audit_2026_06_03` H3 (`gemm_qkvza` MMQ tile-routing
  drops alpha = the #30 deleted-`per_row` fallback regression). The shape is:
  two code paths that *must* derive the same intermediate independently, and one
  silently re-derives it differently.

### (b) Correctness-critical vs architecture
**Correctness-critical.** A write/attend tier or FWHT-plan mismatch is a
silent-wrong-output bug, not a crash. It will pass a green build and only show as
quality drift / attractor under coherence-gate — the most expensive class to
catch.

### (c) Codeable vs human-decision
**Codeable.** The fix is a structural API choice: derive the paired key/plan
**once** and thread the *same derived value* into both the `KvWrite` and
`AttnFlash` resolve calls, rather than letting each call re-derive from raw
`v_mode`.

### (d) RECOMMENDED decision
**Introduce a single `KvTierPlan` (or reuse the existing derived-state struct)
that is computed once per attention step and carries: the resolved `KvWrite*`
key, the resolved `AttnFlash*` key, and the shared FWHT/AWQ sub-plan. Both
halves consume this one value.** A debug-build `debug_assert` (or a cheap release
check behind the existing force/unfused knob) verifies the write tier and attend
tier are the *same* tier before dispatch. This makes the #30-class divergence a
compile-time/structural impossibility rather than a discipline requirement.

### (e) OPEN QUESTION for the author
**Should the paired plan be a first-class type threaded through the attention
step API, or is a documented "derive-once, pass-twice" convention plus a
`debug_assert` sufficient?** Recommendation: first-class type — the audit history
(#30, H3) shows the convention-only approach is exactly what drifts.

---

## 0.4 gfx12 > gfx11 > dp4a ladder — collapse `HasWmmaW32Gfx12`?

### (a) Grounded current-state finding
- The 953ea648 stopgap (merged into the feature branch at 9feb5644, both
  confirmed present in `FETCH_HEAD` history) is live in
  `dispatch:src/tables/mod.rs:74–90`:
  ```
  Self::HasWmmaW32 => ctx.arch.has_wmma_w32() || ctx.arch.has_wmma_w32_gfx12(),
  ...
  Self::HasMmq     => ctx.arch.has_mmq() || ctx.arch.is_rdna4(),
  ```
  i.e. `HasWmmaW32` now also admits RDNA4, and `HasMmq` admits RDNA4 explicitly.
- `HasWmmaW32Gfx12` exists as a predicate variant (`dispatch:src/types.rs:266`)
  and is evaluated (`tables/mod.rs:82`), but has **0 kernel registrations** —
  grep across all six table files (`gemm/gemv/moe/rotation/attention/fused_qkv
  _table.rs`) returns zero. Its only other references are two assertions in
  `dispatch:src/tests.rs:147–148`. **Confirmed: it gates nothing.**
- The arch-caps primitives all exist on master:
  `has_wmma()` (`crates/rdna-compute/src/arch_caps.rs:329`),
  `has_wmma_w32()` (:332), `has_wmma_w32_gfx12()` (:335), `is_rdna4()` (:321),
  `has_mmq()` (:341), `has_dot2_f32_f16()` (:338). So any recommended collapse is
  buildable against master's `ArchCaps` with no new methods.

### (b) Correctness-critical vs architecture
**Architecture, with a correctness *origin*.** The stopgap itself **fixed** a
correctness regression: before 953ea648, every WMMA-family quant (MQ3, Lloyd,
fused QKV/gate-up, MoE grouped, GQA-fused attn) resolved to `MissingImpl` on
RDNA4 (the dead-gate class catalogued in `project_pr393_dispatch_gfx12_regression`
and the kernel-audit D-1/D-2 lm-head gate). The remaining question — whether to
keep a separate `HasWmmaW32Gfx12` or fold to one predicate — is architectural
(API surface / extensibility), not a live correctness bug, because the stopgap
already restored the RDNA4 paths.

### (c) Codeable vs human-decision
**Codeable**, pending one naming/extensibility call by the author.

### (d) RECOMMENDED decision
**Collapse to a single extensible `HasWmma` predicate backed by
`ArchCaps::has_wmma()`, expressed as a priority-ordered ladder, and DELETE
`HasWmmaW32Gfx12`.** Rationale:
1. `ArchCaps::has_wmma()` (arch_caps.rs:329) already means "this arch has *some*
   wave32 WMMA path (gfx11 or gfx12)". `HasWmmaW32 == has_wmma_w32() ||
   has_wmma_w32_gfx12()` is exactly `has_wmma()` for RDNA — the OR in the stopgap
   is re-deriving a fact the cap already exposes. Folding `HasWmmaW32 → HasWmma`
   removes the `||` and the "did you remember to add the gfx12 arm?" footgun that
   produced the original RDNA4 dead-gates.
2. `HasWmmaW32Gfx12` gates **nothing** (0 registrations). Keeping a live,
   evaluated predicate that no kernel uses is dead surface that invites a future
   author to register a kernel under it and accidentally exclude gfx11. Delete it.
3. The *ladder* (gfx12 > gfx11 > dp4a > scalar) is a **resolve-order** concern,
   not a per-predicate concern. The registry already resolves the **first**
   passing variant (`tables/mod.rs:43–53`). So the ladder is expressed by
   **registration order**: register a gfx12-tuned WMMA variant first (if/when one
   exists, gated `HasWmma` + a `ShapePredicate` or arch sub-check), then the
   shared WMMA variant (`HasWmma`), then the `HasDp4a` fallback, then the
   `Always` scalar. Priority = list position, which the existing resolver already
   honours.

**On the "reserve vs delete" axis:** delete now. If a genuinely gfx12-*only*
kernel ever ships (one that must NOT run on gfx11), reintroduce a predicate at
that point with a registration — a predicate should never exist before the
kernel it gates. Reserving it now is the anti-pattern that created the dead gate.

### (e) OPEN QUESTION for the author
**Collapse `HasWmmaW32` → a single extensible `HasWmma` + priority-ordered
ladder via registration order, and DELETE the 0-registration `HasWmmaW32Gfx12`
(vs. keep it reserved for a future gfx12-only kernel)?** Recommendation: collapse
+ delete. Reserve only when a gfx12-only kernel actually lands.

---

## 0.5 Family API doc — the "adding a new family" contract

### (a) Grounded current-state finding
The pattern is uniform across the six existing families. Using `GemvFamily`
(`dispatch:src/families/gemv.rs:149–164`) and `AttentionFamily`
(`dispatch:src/families/attention.rs:35–48`) as the templates, a family is:
1. A `KernelKey` arm (or several) in the flat enum (`dispatch:src/types.rs:113`+).
2. A `<family>_table::populate(&mut KernelRegistry)` fn that `register`s one
   `KernelVariant { key, arch_required, shape_gate, steps, has_awq }` per kernel
   (`dispatch:src/tables/attention_table.rs:5`).
3. A `<Family>` struct holding its own `KernelRegistry`, built once in
   `new()` via `populate()` + `validate()` (gemv.rs:155–160).
4. A `resolve(key, ctx, shape) -> Result<&KernelVariant>` method delegating to
   `self.registry.resolve` (gemv.rs:170–184), plus typed `run*` entry points.
5. A `pub mod` line in `dispatch:src/families/mod.rs` and the table in
   `dispatch:src/tables/mod.rs`.

### (b) Correctness-critical vs architecture
**Architecture / documentation.** Not a live bug; it is the contract that keeps
future families from re-introducing inline `match dtype`.

### (c) Codeable vs human-decision
**Codeable** (it is a doc deliverable). The contract below is the outline Ship 5
should formalise.

### (d) RECOMMENDED "adding a new family" contract (outline)
A new kernel family MUST:
1. **Add `KernelKey` arms** for every variant, in the flat enum
   (`types.rs`). One key per (quant tier × fusion variant) that the resolver must
   distinguish.
2. **Write `<family>_table::populate`** registering exactly one `KernelVariant`
   per key, each with: the **narrowest** correct `ArchPredicate` (prefer
   `HasWmma`/`HasDp4a`/`Always` over hand-rolled arch ORs — see 0.4), an optional
   `ShapePredicate` for runtime-dimension gating, the `PipelineOp` `steps`, and
   the `has_awq` flag. **Register in priority order** (most-specific/fastest
   first) — the resolver returns the first passing variant.
3. **Add a `<Family>` struct** that owns its `KernelRegistry`, builds it once in
   `new()` and calls `registry.validate()` (which hard-errors on any empty
   entry — `tables/mod.rs:60–67`).
4. **Expose `resolve()` + typed `run*()`** — models call the typed entry point,
   never the registry directly, and **never** `match` on `DType` (the crate's
   stated invariant, `dispatch:src/lib.rs:3–6`).
5. **Register the module** in `families/mod.rs` and `tables/mod.rs`.
6. **Add golden tests** in `hipfire-dispatch-tests` (the branch already carries
   `crates/hipfire-dispatch-tests/` with per-arch golden fixtures) asserting the
   resolved key for each (arch × dtype) the family supports, including the RDNA4
   row — the dead-gate class (0.4) is exactly what a per-arch golden catches.
7. **Forbid a predicate without a registration** (0.4 lesson): a new
   `ArchPredicate` variant may only land in the same change as the kernel it
   gates.

### (e) OPEN QUESTION for the author
None blocking. Confirm whether the family doc lives inline as a `//!` module doc
on `families/mod.rs` or as a standalone `docs/` page; recommendation: the
checklist as `//!` on `families/mod.rs` (next to the code it governs) plus a
short pointer from the architecture doc.

---

## 0.6 Verification contract — the per-migration gate

### (a) Grounded current-state finding
- The cross-process perf harness exists: `scripts/probe_commits.sh` (present in
  worktree) — the CLAUDE.md-mandated A/B tool that handles warmup + multi-run
  aggregation across two commits.
- The coherence gates exist: `scripts/coherence-gate.sh` and
  `scripts/coherence-gate-dflash.sh` (both present).
- **`HIPFIRE_FORCE_UNFUSED` does NOT yet exist** — grep across `crates/` and
  `scripts/` returns zero hits. It is a **prospective** byte-parity knob this
  contract proposes. Existing sibling force knobs that establish the pattern:
  `HIPFIRE_LLOYD_FORCE_BASELINE`, `HIPFIRE_DDTREE_FORCE_SLOW`,
  `HIPFIRE_BLOB_FORCE`. So the new knob must be *implemented* as part of Ship 1
  (or earlier), not assumed.

### (b) Correctness-critical vs architecture
**Correctness-critical (process).** The whole point of unifying dispatch is that
each migration is provably output-neutral. Without a byte-parity oracle, a
migration that silently changes which kernel runs is the #30 / H3 class again.

### (c) Codeable vs human-decision
**Codeable**, except the per-migration sign-off (human reads the coherence
report) which is inherently a human gate per the CLAUDE.md coherence protocol.

### (d) RECOMMENDED per-migration gate (the contract every Ship 1–6 must pass)
For **each** model/family migrated from inline dispatch to `hipfire-dispatch`:
1. **Byte-parity (the strongest, cheapest signal).** Implement
   `HIPFIRE_FORCE_UNFUSED` (or, more precisely for this work, a
   `HIPFIRE_DISPATCH_OLD=1` / `_NEW=1` selector) so the **same binary** can run
   the legacy inline path and the new dispatch path. Then assert
   **byte-identical** committed-token streams (use `HIPFIRE_EMIT_TOKEN_IDS=1`,
   the existing committed-event stream) on a fixed prompt set, temp 0.0. A
   pure dispatch refactor MUST be byte-identical — any diff is a real divergence
   to root-cause before landing.
2. **Cross-process perf A/B** on **both** GPU classes:
   `scripts/probe_commits.sh <pre> <post>` on gfx1100 (RDNA3) **and** gfx1201
   (RDNA4). Gate: within **±1–3 %** (the established warm within-session band,
   per CLAUDE.md). A ≥5 % delta activates the mandatory investigation rule —
   do not hand-wave as noise. RDNA4 is non-optional here precisely because 0.4
   is where the dead-gates lived.
3. **Coherence:** `./scripts/coherence-gate.sh` (full matrix) on the migrated
   model. If the migration touches any spec-decode path, also
   `./scripts/coherence-gate-dflash.sh`. Human reads the report and confirms
   fluent / on-topic / not looping per the coherence protocol.
4. **Per-arch golden** (0.5 item 6): the `hipfire-dispatch-tests` golden suite
   must assert the resolved `KernelKey` for the migrated family on RDNA3 **and**
   RDNA4 (and any CDNA target the family claims) — a GPU-free guard that catches
   a re-introduced dead-gate before it ever reaches a coherence run.

Prompt bytes for every perf cell follow the mandatory τ-sensitivity rule:
byte-identical, committed prompt file, md5 recorded alongside the result.

### (e) OPEN QUESTION for the author
**Is the byte-parity knob a `HIPFIRE_DISPATCH_OLD/NEW` selector kept alive for
the duration of the migration (deleted after Ship 6), or does each Ship delete
the old path immediately and rely on `probe_commits.sh` across the migration
commit instead?** Recommendation: keep a temporary selector through the
migration window — same-binary byte-parity is a far stronger oracle than
cross-commit comparison and removes "did something else change between commits?"
confounds. Delete the selector in Ship 6.

---

## Decisions needed from author

These three are the **human-decision** items that gate Ship 1–6. The other
contracts (0.3, 0.4 mechanism, 0.5, 0.6 mechanism) are codeable once these land.

1. **Resolve-cache exemption (0.1):** When Ship 2 adds the per-layer
   `KernelKey` cache — *exempt* the KV-tier-dependent families
   (`KvWrite*` / `AttnFlash*`) from caching (recommended), **or** add an
   `invalidate()` that `KvAdaptive::maybe_downshift` calls on every downshift?
   The recommendation avoids a cross-crate invalidate footgun; choosing
   `invalidate()` obligates wiring it into every `v_mode`/`k_mode` writer
   (`transcode_v_step` llama.rs:4772/4889, `transcode_k_step` :4925) plus a guard
   test.

2. **Scratch lifetime location (0.2):** Keep `PrefillBatchScratch` &
   model-specific scratch **arch-owned**, passed into dispatch by `&mut`
   (recommended), **or** move ownership down into
   `hipfire-dispatch::ResourceManager`? The latter is **blocked by a confirmed
   dependency cycle** unless the scratch *type* is first relocated to a crate at
   or below `rdna-compute`.

3. **`HasWmmaW32Gfx12` keep-vs-delete (0.4):** Collapse `HasWmmaW32` to a single
   extensible `HasWmma` predicate (backed by `ArchCaps::has_wmma()`) with a
   priority-ordered ladder via registration order, and **DELETE** the
   0-registration `HasWmmaW32Gfx12` (recommended), **or** keep `HasWmmaW32Gfx12`
   reserved for a future gfx12-only kernel? Recommendation: delete; reintroduce a
   gated predicate only in the same change as a real gfx12-only kernel.
