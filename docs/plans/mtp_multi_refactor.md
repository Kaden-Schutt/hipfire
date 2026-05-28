# Plan: PP+MTP serve dispatch via consolidated generate path (Stage 2b) — **v2**

Status: v2 plan, post-adversarial-review. Stage 2a foundation already
shipped in commit `54e18194`; this doc scopes Stage 2b — the
daemon-serve dispatch for PP+MTP.

**v2 changes vs v1** (see `mtp_multi_refactor_plan_rev_claude.md`
for the full audit):

1. **Scope correction:** v1 said "unify three functions"; reality is
   `generate` is itself a top-level dispatcher with an inline AR impl,
   plus three child functions (`generate_multi`, `generate_mtp`,
   `generate_dflash`). True LOC base is **1930**, not 1586.
2. **DFlash delegated, not inlined** (per user input 2026-05-28): PP-DFlash
   is a known perf loss; the unified function dispatches `Path::DFlashSingle`
   to today's `generate_dflash` and returns. ~491 LOC stays put.
3. **Borrow-checker prerequisite added.** Stage 2a already shipped an
   `unsafe { &mut *dev0_ptr }` hack to satisfy `peer_clone_tensor`'s
   `(&Gpu, &mut Gpu)` signature when src/dst are the same device. Step −1
   designs a clean `Gpus::disjoint_pair_mut` / `single_mut` API to
   eliminate that pattern before propagation.
4. **`forward_prefill_batch_multi` extension or tape-less fallback** decision
   raised to Step 0. Two options costed; user picks at execution.
5. **Surgery split into 5 per-path commits** (gemini's recommendation),
   not one all-or-nothing commit.
6. **`GenerateCtx` struct** added to tame the 23-param union.
7. **Honest validation budget.** Coherence-gate runs cost ~15 min each;
   v1's "validate at every step" was 5× under-budgeted.
8. **PFlash+MTP refused with explicit bypass event**, not silently enabled.
9. **Decision gate** moved from step 3 (impossible to fire) to step 4.
10. **Top-level `generate` dispatcher kept** for arch routing
    (qwen2/vl); only the qwen35 path is unified into `generate_qwen35`.

---

## Background

Stage 2a (`commit 54e18194`) built the foundation:
- `spec_step_mtp_compressed_serial_hetero` learned a same-device
  shortcut for the cycle-exit handoff (D2D memcpy when
  `target_gpu.device_id == drafter_gpu.device_id`).
- `peer_clone_tensor` / `seed_prev_hidden` got the same shortcut.
- `MtpState.drafter_state: Option<MtpHeteroDrafterState>` added.
- `load_model_pp` extended with MTP head load on `output_device`,
  token_embd mirror, and full state allocation.

Remaining: the daemon needs a serve dispatch that, when
`m.pp > 1 && m.mtp.is_some()`, runs the MTP spec loop with the trunk
verify on multi-gpu (PP path) and the chain on the drafter gpu (which
is `output_device` in our layout). Today the dispatch routes
`m.pp > 1` to `generate_multi`, which doesn't know about MTP — so
load succeeds but decode falls through to AR.

## Current state of the four functions

| function | LOC | role | trunk path | spec path |
| --- | --- | --- | --- | --- |
| `generate_dflash` | 491 | DFlash single-gpu | own | own |
| `generate_mtp` | 454 | MTP single-gpu | `forward_prefill_batch` | `spec_step_mtp_compressed_serial` |
| `generate_multi` | 485 | PP version, AR only | `forward_prefill_batch_multi` | n/a |
| `generate` | 991 | top-level dispatcher (qwen2/vl/dflash/mtp/multi) PLUS inline AR impl | own (AR) | n/a (AR) |

**Total: 2421 LOC across four functions.** PFlash logic lives at
daemon-main-loop scope (daemon.rs:564-574), not on `LoadedModel`.

`generate` itself dispatches to qwen2/vl/dflash/mtp/multi at the top and
runs an inline AR body if none of those match (~900 LOC). After this
refactor `generate` keeps its arch-id dispatch role; the qwen35-arch
inline body + the three qwen35 children consolidate into one
`generate_qwen35`.

## Refactor target

Single `generate_qwen35` function dispatched from `generate` for the
qwen35 (arch_id 5/6) case. Calling convention:

```rust
fn generate_qwen35(
    m: &mut LoadedModel,
    main_gpu: &mut Gpu,                 // unused when m.pp > 1, see P0-4
    ctx: &mut GenerateCtx,              // bundles stdout/id/prompt/sampling/etc.
) {
    // 1. Pick path from m state (post-load-validation, exhaustive over VALID combos):
    let path = match (m.pp > 1, m.mtp.is_some(), m.dflash.is_some()) {
        (false, false, false) => Path::Ar,
        (false, true,  false) => Path::MtpSingle,    // includes hetero (drafter_state.is_some())
        (false, false, true)  => Path::DFlashSingle,
        (true,  false, false) => Path::PpAr,
        (true,  true,  false) => Path::PpMtp,        // ← Stage 2b's new arm
        // (true, false, true) refused at load (DFlash+PP requires HIPFIRE_PP_DFLASH
        //                                       opt-in; treat as Ar for now).
        // (_, true, true) refused at load (MTP+DFlash mutually exclusive).
        _ => unreachable!("invalid path combo reached generate_qwen35; load handler missed a refusal"),
    };

    // 2. DFlash delegates to existing function (NOT inlined; PP-DFlash known
    //    perf loss per prior experiments). Returns after dispatch.
    if matches!(path, Path::DFlashSingle) {
        return generate_dflash(m, main_gpu, ctx, ...);  // wrapper that decomposes ctx
    }

    // 3. Common prelude: prompt encoding + ChatML/Jinja frame + capacity
    //    check + auto-reset. ~120 LOC.
    let frame = build_prompt_frame(m, ctx)?;

    // 4. Prefill (path-dispatched):
    //    Ar/MtpSingle → forward_prefill_batch on main_gpu
    //    PpAr/PpMtp   → forward_prefill_batch_multi on m.pp_gpus
    let prefill = prefill_for_path(m, main_gpu, &path, &frame)?;

    // 5. Decode loop (path-dispatched):
    //    Ar/PpAr  → per-token forward path with sampling
    //    MtpSingle → spec_step_mtp_compressed_serial OR _serial_hetero
    //                (existing branch on drafter_state.is_some())
    //    PpMtp    → spec_step_mtp_compressed_serial_hetero with
    //                target_gpu = drafter_gpu = output_device
    decode_for_path(m, main_gpu, &path, &prefill, ctx)?;

    // 6. Common postlude: emit `done` event, push to conversation_tokens,
    //    update seq_pos. ~50 LOC.
    emit_done(ctx, ...);
}
```

`GenerateCtx<'a>` bundles the ~15 daemon-locals that today's functions
take as separate params:

```rust
struct GenerateCtx<'a> {
    stdout: &'a mut std::io::Stdout,
    id: &'a str,
    prompt: &'a str,
    system_prompt: Option<&'a str>,
    sampling: SamplingCfg,           // temp, top_p, repeat_penalty, repeat_window
    max_tokens: usize,
    max_think_tokens: usize,
    assistant_prefix: AssistantPrefix,
    budget_alert: BudgetAlert,        // at_tok, text
    pflash_state: Option<&'a mut PflashState>,
    pflash_cfg: Option<&'a PflashConfig>,
    drafter_gpu: Option<&'a mut Gpu>,  // hetero MTP sibling for pp=1 path
    tools: Option<&'a [serde_json::Value]>,
    messages_history: Option<&'a [Message]>,
}
```

## Wins

1. **Path-decision matrix is explicit and exhaustive.** Today's
   refusal contracts (DFlash+CASK, DFlash+pp>1 in v1, MTP+DFlash, etc.)
   collapse to one match. Adding a 6th combo (e.g. PP+PFlash+MTP)
   becomes one match arm + one decode chunk.
2. **Stage 2b's MTP-under-PP dispatch is one new match arm at #5.**
3. **Param explosion contained** via `GenerateCtx`.
4. **`generate_dflash` stays intact** — no PP-DFlash interaction risk.

Honest LOC accounting (post-glm5 #9 correction):
- ~120 LOC genuinely deletable across prelude (chatml/jinja, capacity
  check, conversation reset)
- ~50 LOC genuinely deletable across postlude (done emission,
  conversation_tokens push, seq_pos update)
- ~50 LOC new for `GenerateCtx` struct and helpers
- ~80 LOC new for the dispatch-arm decode helpers

**Net change: ~50 LOC deletion + a much smaller function-count footprint.**
The argument for this refactor is maintenance, not LOC count. v1's
claim of "300 LOC removed" was overstated by ~2.5×.

## Risks (rev2)

1. **Touching ~1500 LOC of working serve code.** Each existing
   function has subtle behavior (multi-turn KV reuse, eviction
   triggers, PFlash decision branching). Mitigation: per-path
   migration commits (see step 5) keep each switch surgical.

2. **`forward_prefill_batch_multi` lacks `per_token_hidden_out` /
   `gdn_tape` / `tree_verify` params** that MTP needs (P0-3). Two
   options; user picks at execution:

   - **Option A: extend multi prefill (~200 LOC, full fidelity).**
     Add `per_token_hidden_out: Option<&GpuTensor>` (lives on
     output_device) and `gdn_tape: Option<&mut GdnTape>` (per-band
     capture + cross-device assembly). Full τ parity with single-gpu.

   - **Option B: tape-less fallback (~20 LOC, ~6% tok/s loss).** Pass
     `gdn_tape: None` through multi prefill; mtp_spec's existing
     tape-less branch (mtp_spec.rs:1884-1894) takes a "full-trunk
     replay of committed prefix" on every partial-accept cycle.
     Math: τ=3.25 → ~35% partial-accept rate × ~30 ms extra forward
     per partial-accept × 20 cycles = ~+6% wall under PP-MTP.

   Default: Option B for v1. Option A is correct long-term but costs
   another 1-2 days. Stage 2b ships with B; Option A becomes
   Stage 2c if perf matters more than ship-speed.

3. **Borrow-checker (P0-5):** Stage 2a's `unsafe { &mut *dev0_ptr }`
   in daemon.rs:2677 is a smell. Prerequisite Step −1 lands a
   `Gpus::disjoint_pair_mut(i, j) -> (&Gpu, &mut Gpu)` and
   `single_mut(i) -> &mut Gpu` helper pair that handles same-device
   internally without unsafe.

4. **PFlash + MTP under PP not in scope.** PFlash compresses prompt;
   MTP relies on KV offsets (gemini §2.3 — concrete mechanism). Add
   explicit `pflash_bypass` event when `m.pflash_state.is_some() &&
   m.mtp.is_some()`, matching today's DFlash+PFlash bypass pattern
   (daemon.rs:4681-4685). Validation deferred to Stage 2c.

5. **DFlash co-residence with MTP / PFlash + DFlash:** today's
   refusals stay. The unified function's match never sees these
   combos (filtered at load); `unreachable!()` guards them.

6. **VL / qwen2 paths:** unchanged. `generate` dispatcher routes
   them as today before reaching `generate_qwen35`.

## Sequence (rev2)

Each step is independently revertable. Step 5 is split into per-path
migrations.

### Step −1 (NEW): clean `Gpus` access pattern (~80 LOC)

Add to `multi_gpu.rs`:
```rust
impl Gpus {
    pub fn single_mut(&mut self, i: usize) -> &mut Gpu { ... }
    pub fn disjoint_pair_mut(&mut self, i: usize, j: usize)
        -> (&Gpu, &mut Gpu) { ... }  // handles i == j via split_at_mut + alias-safe wrapper
}
```

Migrate `daemon.rs:2677-2681` (the existing `unsafe { &mut *dev0_ptr }`)
to use `disjoint_pair_mut`. No behavior change; eliminates the
unsafe block. Validated via `cargo build` + Stage 2a regression
(load_model_pp + mtp succeeds).

### Step 0a: build a PP-AR coherence test (~60 LOC)

There is no PP-AR coherence test today. Build one
(`scripts/coherence-gate-pp.sh`) that exercises pp=2 + AR on the
same 9-model matrix as `coherence-gate.sh` but with the
`HIPFIRE_ALLOW_MIXED_ARCH=1 HIPFIRE_PP_LAYERS=48,16` env. Without
this we can't validate "PP-AR doesn't regress" at any step.

### Step 0b: pick prefill_multi extension option (Option A or B above)

User decision point. Default to B for v1 unless A's cost is acceptable.

If A: extend `forward_prefill_batch_multi` signature. ~200 LOC,
half a day. Validate: hetero MTP still works (single-gpu callers
unaffected); PP-AR coherence still passes.

If B: stub the new args as `None` in multi caller. ~20 LOC. Document
the τ cost.

### Step 1: read all four function bodies into a comparison table

~1 hour, no code change. Output: a comparison table by sub-section
(prelude, frame, prefill, decode-loop body, postlude) showing which
function does what. Stops at "OK, now I know exactly what to extract."

### Step 2: extract common prelude/postlude helpers (~250 LOC touched, ~120 LOC saved)

Pure-helper extractions, each as a separate commit:
- `build_prompt_frame(m, ctx) -> Frame` (chatml/jinja, ~80 LOC)
- `check_capacity(m, frame, max_tokens) -> Result<()>` (~20 LOC)
- `emit_done(ctx, stats)` (~30 LOC)
- `push_conversation_token(m, tok)` (~10 LOC)

Each commit changes ONE of the four existing functions to use the
new helper. Other three keep their inline version. **Validate after
each commit:** coherence-gate-dflash (5 min) for spec-decode-touching
changes; just `cargo build` for prelude-only changes.

### Step 3: introduce `GenerateCtx` (~50 LOC)

Add the struct and migrate each existing function's signature one at
a time. Pure mechanical change. Validate: `cargo build` per commit;
single coherence-gate run at end.

### Step 4: introduce `Path` enum + `pick_path` helper (~80 LOC)

Define the enum and the function. Each existing generate function
gets a `let path = pick_path(m); debug_assert!(...);` at its top that
asserts which path it expects. No dispatch change yet — this is
type-system documentation that catches load-handler bugs.

**Decision gate after step 4:** if the path-picking function reveals
existing load-handler bugs (e.g. a combo that shouldn't reach a
function actually can), fix the load handler FIRST. If the match
matrix is clean, proceed.

### Step 5: per-path migration (5 commits, ~600 LOC moved total)

Each commit creates one `Path::X` arm in `generate_qwen35` and
migrates one call site:

- **5a (Path::Ar):** extract `generate`'s inline AR body. New
  function `generate_qwen35` with just the Ar arm. `generate`
  routes `arch_id ∈ {5,6} && everything-else-None` to it.
  Validate: AR coherence-gate matches pre-refactor.
- **5b (Path::PpAr):** absorb `generate_multi`. Move its body into
  the PpAr arm. `generate` routes `m.pp > 1 && !mtp` to
  `generate_qwen35`. Old `generate_multi` deleted in same commit.
  Validate: PP-AR coherence-gate (built in step 0a).
- **5c (Path::MtpSingle):** absorb `generate_mtp`. Move into
  MtpSingle arm. Hetero (drafter_state.is_some()) handled inside
  the arm via existing branch. Old `generate_mtp` deleted.
  Validate: single-gpu MTP τ + tok/s unchanged; hetero MTP τ + tok/s
  unchanged.
- **5d (Path::PpMtp) — STAGE 2B PAYOFF:** new arm. Trunk verify uses
  `forward_prefill_batch_multi` (option A or B from step 0b). Spec
  loop calls `spec_step_mtp_compressed_serial_hetero` with
  `target_gpu = drafter_gpu = m.pp_gpus.devices[output_device]`.
  Same-device shortcut in spec function avoids cross-device peer
  copy. **Validate: coherent output; τ within 5% of single-gpu MTP;
  tok/s within 10%; VRAM matches audit projection.**
- **5e (cleanup):** delete `generate_dflash`-as-leaf and instead have
  `Path::DFlashSingle` arm wrap-and-return. ~5 LOC. Optional; keeps
  `generate_dflash` as a free function if cleaner.

If any of 5a-5d regresses, **revert that one commit** back to the
prior state (which is itself a working refactor step). The whole
series doesn't have to revert.

### Step 6: validation & docs

After all 5 migrations land:
- Full `coherence-gate-dflash.sh` run (5 min)
- Full `coherence-gate.sh` run (10 min) — non-spec-decode sanity
- New `coherence-gate-pp.sh` run (15 min) — PP-AR + PP-MTP
- Single-shot tok/s probe: AR baseline must be within 1% of
  pre-refactor on a small model (gemini's §4 concern, downgraded to
  sanity check)
- Devlog summarizing what changed, τ delta, tok/s delta, VRAM at
  pp=2 + mtp at 64k ctx

## Honest timeline

**Per glm5 #12: v1's "2-3 days" was optimistic.** Realistic budget:

| step | desc | est |
| --- | --- | --- |
| −1 | Gpus access cleanup | 0.5 day |
| 0a | PP-AR coherence test | 0.5 day |
| 0b | prefill_multi extension (A or B) | 0.5–1 day |
| 1 | read & compare | 1 hour |
| 2 | extract helpers (4 commits) | 1 day |
| 3 | GenerateCtx | 0.5 day |
| 4 | Path enum + decision gate | 0.5 day |
| 5a-5e | per-path migrations | 2 days |
| 6 | validation + docs | 0.5 day |
| **total** | | **5–6 days** focused |

Plus debugging buffer (1-2 days) for the inevitable issues a 1500-LOC
refactor surfaces.

**Realistic: 6-8 days end-to-end** for the unification path.

## Alternative: Option X (separate `generate_mtp_multi`, ~600 LOC)

External reviewers (glm5 #13) argued I undersold this. Honest
re-comparison:

| | Unification (rev2) | Option X (separate fn) |
|---|---|---|
| LOC change | +180 (new helpers/Ctx) -120 (dedup) +600 moved | +600 (new fn) +20 (prefill_multi) +80 (Gpus access cleanup) |
| Files touched | daemon.rs (heavy), multi_gpu.rs, mtp_spec.rs | daemon.rs (light), multi_gpu.rs, mtp_spec.rs |
| Risk to existing paths | High (touches AR/MtpSingle/PpAr inside the refactor) | Low (new function added beside existing) |
| Maintenance debt | Low after refactor | One more per-combo function to keep in sync |
| Time | 6-8 days | 2-3 days |
| Stage 2b payoff | At end of step 5d | At end of new function |

**Option X is the lower-risk Stage 2b deliverable.** The unification
pays back over time IF we keep adding combos. Today we have 5 combos;
adding a 6th becomes ~600 LOC under Option X vs ~80 LOC under
unification — but that's a future cost, not today's. Both options
require the same step −1 (Gpus cleanup) and step 0 (prefill_multi
decision).

**My recommendation:** ship Option X first as Stage 2b. Land the PP+MTP
combo working in production at ~3 days. Schedule the unification as a
follow-up sprint (Stage 2c) when there's time and the combo count
justifies it. This is the lower-risk path to user-visible value.

If user prefers the unification path, the v2 plan above is the
concrete roadmap.

## Open questions for user

1. **Option A or Option B for prefill_multi (step 0b)?** A: ~200 LOC,
   full τ parity. B: ~20 LOC, ~6% tok/s loss under PP-MTP from
   tape-less replay on partial-accepts.
2. **Option X (separate function, ship in 3 days) or unification (rev2
   plan, 6-8 days)?** Both deliver PP+MTP working in production.
   Unification trades 4-5 extra days of refactor work for a lower
   maintenance footprint going forward.
3. **PFlash+MTP under PP timeline.** Refused with bypass event in v1;
   if you need it sooner, it's its own work package.

## Decision gates

- **After step 4 (path enum + decision gate):** if the path-picking
  function reveals load-handler bugs, fix those FIRST before
  proceeding. The unification cannot mask load-handler invariants.
- **After step 5a (AR migration):** if AR coherence-gate regresses,
  abort the unification and revert to step 4 + ship Option X.
- **After step 5d (PpMtp):** if τ regresses >5% or tok/s >10% vs
  single-gpu MTP, debug before declaring done. The hetero work we
  shipped today is the right baseline.

## What the v1 plan got right (carried forward)

- Direction (path-dispatched function) is sound.
- "Ship Option X as fallback" is the right escape valve.
- Decision-gate idea is the right governance pattern.

## What v2 corrects vs v1

| v1 claim | v2 reality |
|---|---|
| 3 functions, 1586 LOC | 4 functions, 2421 LOC; DFlash stays as delegate |
| ~300 LOC deletable | ~120 LOC deletable; main payoff is maintenance not size |
| 2-3 days | 6-8 days (or 2-3 for Option X) |
| One surgery commit | 5 per-path migration commits |
| `(gpu, drafter_gpu)` signature works | Borrow checker forbids same-device aliasing; needs Gpus helper |
| 18-param function is fine | Need GenerateCtx struct |
| Validate at every step | Honest gate cost ~15 min; budget accordingly |
| PFlash+MTP "enable silently" | Refuse with bypass event; validate later |
| Eviction "disable at load" | Already refused for pp>1; for pp=1+MTP carry defensive code |
| Step 3 decision gate | Move to step 4 (step 3 can't actually fail) |
