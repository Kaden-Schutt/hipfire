---
title: HANDOVER — LoadedModel god-struct collapse (Increment 2 E–H → ModelParallel → ImmutableMeta)
tags: [device-mesh, god-struct, loadedmodel, modelstate, handover, refactor, "462"]
created: 2026-07-11
updated: 2026-07-11
---

# HANDOVER: LoadedModel god-struct collapse — resume here

Self-contained pickup doc for a fresh session. Branch **`feature/device-mesh`**.
HEAD **`7e1aa7c2`** (updated after Inc 2 Step E + its leak-fix landed; was `53c9ba4f` at
original handover). Worktree `/home/bjoern/hipfire/.claude/worktrees/feature+device-mesh`.
Tree clean. This is one long refactor being landed increment-by-increment behind gates.
**Next step: ModelParallel.** E–H fully analyzed: E landed; F→ModelParallel, G+H's mtp_mode→ImmutableMeta (all deferred, rationale below); H's mtp_k = flagged wiring bug (left in place). **Only TWO steps remain: `ModelParallel` (7 axis fields) → `ImmutableMeta` (immutable scalars incl eos + mtp_mode).**

## What this is

Collapsing the daemon god-struct `LoadedModel` (`crates/hipfire-loader/src/lib.rs`,
was ~50 fields / ~20 `Option`s) into a small composition of owned sub-structs. It is the
`#462` cross-request state-bleed surface and the change-amplification hot spot. The
driver-level `ArchDispatch`/`ar_generate` collapse is ALREADY done (every arch routes
through one `ar_generate` via transient `<Arch>Dispatch{ m: &mut LoadedModel }` wrappers);
this remaining work is the **struct field collapse**.

**Design (approved, DO NOT re-litigate):**
`docs/superpowers/specs/2026-07-11-loadedmodel-god-struct-field-collapse-design.md`
- Borrow model = **transient wrapper + lazy `parts_mut`** (native disjoint-field borrow,
  NO `unsafe`, NOT a stored `Box<dyn ArchDispatch>` — that's self-referential and was
  proven infeasible; the whole design turns on this).
- Target groups: `parallel: ModelParallel` · `arch: ModelState` (per-arch bundle enum) ·
  `session: SessionState` · `persist: PersistState` (survives reset) · `speculator` ·
  `tokenizer` · `meta: ImmutableMeta`.
- Reset (#462) = one `reset_context` = `session.reset(gpu)` + arch reset (compiler-total
  via the enum) + `speculator.reset`; `persist` NEVER touched.

## Landed so far (all reviewed opus READY-TO-LAND + gate-validated)

- **Increment 1** `33c9fe29..e16e7c01` — `SessionState` (resettable per-request state) +
  `PersistState` (asst_turn_cache/decoded_vocab, survive reset) + `session_parts_mut`
  splitter + `reset_context` → `SessionState::reset`. serve-multiturn PASS.
- **Inc 2 Step A** `fa2edc62` — deleted vestigial always-`None` `kv_cache`/`dn_state`.
- **Inc 2 Step B** `bf59147a` — folded `deepseek4_pbs` → `Deepseek4Bundle.pbs` (non-Option).
  ds4 coherence probe PASS.
- **Inc 2 Step C** `c3b8f789` — dots-ocr 3 loose fields → new `ModelState::DotsOcr(DotsOcrBundle)`;
  collapsed the transient assemble/disassemble to an in-place `m.state` borrow. dots-ocr load PASS.
- **Inc 2 Step D** `85007411` — `vision_config`+`vision_weights` → one loader-side
  `Option<Qwen35Vl>` field (see the crate-layering note under Step D below).
- **Inc 2 Step E** `36660547` (+ leak-fix `7e1aa7c2`) — `qwen35_mtp_head` → `Qwen35Bundle.mtp_head`
  (carried across the spec-decode transient by `ModelSlot`; `from_bundle`/`into_bundle` round-trip
  it — the one correctness invariant), AND `mtp_weights_present` bool → computed method
  (`ds4 mtp_layer/dspark || qwen35 mtp_head`; exact incl EP/PP None cases). Borrow shape resolved:
  head rides as a read-only LOCAL in `generate_qwen35_mtp`, NLL ends each borrow before its re-pack
  move — NO loop restructure. Added `qwen35()`/`qwen35_mut()` accessors. opus READY-TO-LAND;
  serve-multiturn PASS + MTP probe OK. Leak-fix `7e1aa7c2` (separate commit): the 2 cvs-scratch
  alloc-fail exits now free state + restore m.state (were pre-existing bricking bugs).

Plans/ledgers (gitignored `docs/superpowers/` + git-tracked `.superpowers/sdd/*.md`):
`2026-07-11-god-struct-collapse-inc1-sessionstate.md`,
`…-inc2-deadfields-ds4pbs.md`, `…-inc2c-dotsocr.md`; ledgers `godstruct-inc*-progress.md`.

## Remaining loose fields on `LoadedModel` (post Step D) — the work left

Parallelism axis (→ future `ModelParallel`): `pp: usize`, `pp_gpus`, `pp_scratch_set`,
`pp_dn_la_to_device`, `ep: Option<EpState>`, `tp: Option<TpModel>`, `pp_dense: Option<PpModel>`.
Per-arch / misc: `qwen35_mtp_head`, `mtp_mode`, `mtp_k`, `mtp_weights_present`,
`deepseek4_eos_tok`, `minimax_eos_tok`. Plus `arch_id`, `model_path`, `chat_template`,
`rec_*` sampling, `seq_pos`? (no — seq_pos already in session) etc. → future `ImmutableMeta`.

## Increment 2 remaining steps (hazard-ordered — the terrain map)

The full terrain map (field → target, site counts, hazards) was produced by an Explore
agent this session; the fold order and hazards:

- **E — DONE** (`36660547` + `7e1aa7c2`). Resolved: head as read-only local, `ModelSlot` carries
  `mtp_head` for the DFlash `from_bundle`/`into_bundle` round-trip; `mtp_weights_present` computed.
  The feared move-out borrow was a non-issue — every non-MTP qwen35 site BORROWS the bundle, and
  NLL ends the head-local borrow before each re-pack. See "Landed so far" above.
- **F — DEFERRED into `ModelParallel` (bjoern 2026-07-11, after 2-agent adversarial review).** Do
  NOT fold `pp_gpus`/`pp_scratch_set`/`pp_dn_la_to_device` into an arch struct or a standalone
  `Qwen35PpState`. WHY: (1) `pp_gpus` is SHARED device topology — the same `&mut gpus` frees the
  arch tensors that live in `ModelState::Qwen35` (lib.rs:1924-1933) — and the mesh-through-loader
  design makes `DeviceMesh`/`Gpus::from_mesh` the topology owner; boxing it into an arch-private
  struct inverts that. (2) A 3-field `Qwen35PpState` is NOT a peer of `PpModel`/`EpState` (which own
  their FULL state incl weights/KV); qwen35-PP state SPLITS across `ModelState` (primary stage) +
  these fields (extra stages) → the `PpQwen35` wrap can't be clean. (3) It strands the `pp: usize`
  scalar, which `model_parallel.rs` ALREADY keys `PpQwen35` on. (4) The umbrella's own migration
  order says the SEVEN axis fields (`pp`,`pp_gpus`,`pp_scratch_set`,`pp_dn_la_to_device`,`ep`,`tp`,
  `pp_dense`) move together, LAST, in the ModelParallel step. → These four fold into ModelParallel,
  not a separate step. (Mechanics were survivable — the borrow at daemon.rs:7750 needs an explicit
  `let Qwen35PpState{ref scratch_set, ref mut gpus, ..} = *m.qwen35_pp.as_mut().unwrap()` destructure,
  and the teardown's defensive `if let Some(scratch_set)` at lib.rs:1925 would drop — but fit, not
  mechanics, is why it's deferred.) **Order now: G → H → ModelParallel (absorbs F's four fields).**
- **G — DEFERRED into `ImmutableMeta` (bjoern 2026-07-11, same trap as F).** Do NOT fold
  `deepseek4_eos_tok`/`minimax_eos_tok` into `EpArch`. WHY: they're IMMUTABLE per-model `u32`
  scalars (set once at load, read at generate to detect EOS) used across FOUR ds4 configs —
  EP (state in `ep`), TP (state in `tp`, lib.rs:1514 comment), PP-dense (state in `pp_dense`,
  lib.rs:1550), single-GPU — so homing them in one axis's struct (`EpArch::Ds4`) repeats F's
  cross-axis mistake; TP/PP-dense have no `EpArch` to read from. They're the same CATEGORY as
  `arch_id`/`model_path`/`chat_template` → fold into `ImmutableMeta` when that step lands.
  (Single-GPU ds4/minimax also store eos in their bundle — a redundant second home that
  ImmutableMeta can unify.)
- **H — SPLIT/DEFERRED (bjoern 2026-07-11).** The "→ request params" framing was WRONG: both are
  set ONCE at load (daemon.rs:3664-3665 from the load message), not per-request.
  · `mtp_mode` — set at load, read once at the gate (daemon.rs:11347); immutable load-time serving
    config → **fold into `ImmutableMeta`** (same category as G's eos).
  · `mtp_k` — **DEAD as a field**: declared (lib.rs:361), init (:418), set from load message
    (daemon.rs:3665), NEVER READ anywhere. generate reads `HIPFIRE_MTP_K` env (daemon.rs:6819)
    instead. **⚠ FLAGGED AS A WIRING BUG (bjoern: fix outside god-struct):** the load-message
    `mtp_k` knob is silently ignored — only the env var works. Re-wire generate to read the
    load-config `mtp_k` (or drop the knob) deliberately, later. Left in place for now.

**KEY STRATEGIC FINDING (this session):** the "E–H standalone folds, THEN big groups" hazard-order
was substantially WRONG. **E was the only true standalone per-arch fold.** F, G, and H's `mtp_mode`
are all big-group material — F→`ModelParallel` (cross-axis shared `gpus`), G+`mtp_mode`→`ImmutableMeta`
(immutable per-model scalars). Two adversarial reviews (F) + field-semantics analysis (G, H)
converged on: **don't pre-fold cross-axis or immutable fields into a single per-arch/per-axis struct
— they belong in their big group.** So the REMAINING god-struct work is exactly TWO steps:
**`ModelParallel`** (7 axis fields: pp, pp_gpus, pp_scratch_set, pp_dn_la_to_device, ep, tp, pp_dense)
then **`ImmutableMeta`** (arch_id, model_path, chat_template, rec_*, the 2 eos scalars, mtp_mode).

Then **`ModelParallel`** (group the 7 axis fields; the enum shape `Single/Tp/PpDense/PpQwen35/Ep`),
then **`ImmutableMeta`** (arch_id/model_path/chat_template/rec_*). These are the biggest but the
axis dispatch is already collapsed at the driver level, so it's field-grouping.

Also queued: 3 cosmetic stale-comment cleanups from the Step-C review (daemon.rs comments naming
the removed `qwen2_state` field; `spec_impl.rs:62` DotsOcrBundle doc says vision "NOT included"
but `weights` carries it).

## The process that works (repeat it)

Per increment/step: brainstorm ONLY if there's a real design fork (E and F have one → use
`superpowers:brainstorming`, and `/adhd` if the borrow-ownership space is wide) → `writing-plans`
(scope to ONE step; the fold is compiler-guided, byte-identical per commit) → execute
`superpowers:subagent-driven-development` (or inline for small mechanical folds) → **controller
ground-truths every build itself** → opus whole-branch review → GPU gates. Gate set:
`cargo build --release --workspace --all-targets --locked` + `cargo test --workspace --lib`
(343 loader+runtime lib tests) + `scripts/serve-multiturn-gate.sh` (the #462 guard, qwen35) +
an arch eyeball via `coherence_probe` when a model for the touched arch exists
(`~/.hipfire/models`: qwen3.5-{0.8b,4b}.mq4, deepseek-v4-flash.mq2lloyd, dots-ocr.q8.hfq,
qwen3.6-27b-vl.mq4). GPU is free; use `scripts/gpu-lock.sh` (`gpu_acquire`/`gpu_release`).

## TRAPS (every one bit us this session — do not relearn)

1. **NEVER run `scripts/fmt-changed.sh` or `cargo fmt` on `daemon.rs`/`lib.rs`/`carriers.rs`.**
   They rustfmt the WHOLE file (heavy format debt), burying a ~6-line change under hundreds of
   reformat lines. Reconstructed TWICE. Hand-write edits rustfmt-clean; skip formatting entirely.
   (`fmt-changed.sh` also defaults `BASE_REF=origin/master` → on this 144-commit branch that
   churns ~16 files even scoped.)
2. **Editor/LSP diagnostics lie mid-edit.** Phantom `E0560`/`E0609` "no field" errors appear from
   stale snapshots while a multi-file edit is in flight. TRUST the actual `cargo build` exit, not
   the diagnostics. Ground-truth every build.
3. **A required bundle field or a new `ModelState` variant needs `--workspace --all-targets`,
   NOT just `--example daemon`.** Step B (`deepseek4_pbs` non-Option) broke `dspark_bench.rs`
   (another construction site) — the daemon build passed but the workspace build failed.
4. **Cross-crate layering.** An arch's EXTENSION types (e.g. vision in `hipfire-arch-qwen35-vl`)
   must NOT be folded into the BASE arch bundle (`Qwen35Bundle` in `hipfire-arch-qwen35`) — that
   inverts crate layering (base→extension). Group loader-side instead (Step D chose `Qwen35Vl`
   in `hipfire-loader`). Check the crate boundary BEFORE planning a fold.
5. **Review/impl subagents die silently** (no completion notification) fairly often. Set a
   fallback timer or just self-review small mechanical diffs; ground-truth builds regardless.
   Implementers sometimes narrate past their DONE report and fabricate a "review approved" — ignore.
6. **`.superpowers/sdd/*.md` scratch reports** may show as modified/untracked — never stage them
   in a code commit.

## Byte-identity note

Every step is a pure relocation → byte-identical by construction; the ONE semantic wrinkle so far
was Inc-1 Task 6 reordering `kv_adaptive.reset` (proven neutral: `KvAdaptive::reset` is a
self-contained scalar reset). For E/F watch for any reset-ORDER or borrow-lifetime change that
isn't a pure move.

## Refs
- Spec: `docs/superpowers/specs/2026-07-11-loadedmodel-god-struct-field-collapse-design.md`
- ArchDispatch driver design (done): `docs/superpowers/specs/2026-07-09-daemon-god-struct-archdispatch-design.md`
- Next-followups note: `.agent-memory/notes/device-mesh-next-followups.md` (§2 god-struct)
- Global memory index: `MEMORY.md` "god-struct" line.
