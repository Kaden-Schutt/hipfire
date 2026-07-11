---
title: HANDOVER — LoadedModel god-struct collapse (Increment 2 E–H → ModelParallel → ImmutableMeta)
tags: [device-mesh, god-struct, loadedmodel, modelstate, handover, refactor, "462"]
created: 2026-07-11
updated: 2026-07-11
---

# HANDOVER: LoadedModel god-struct collapse — resume here

Self-contained pickup doc for a fresh session. Branch **`feature/device-mesh`**,
HEAD **`53c9ba4f`** at handover. Worktree `/home/bjoern/hipfire/.claude/worktrees/feature+device-mesh`.
Tree clean. This is one long refactor being landed increment-by-increment behind gates.

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

- **E — `qwen35_mtp_head` (+ `mtp_weights_present`) → `Qwen35Bundle`.** MEDIUM. Hazard:
  `generate_qwen35_mtp` (daemon.rs) MOVES the qwen35 bundle out of `m.state`, then reads
  `m.qwen35_mtp_head` — with the head inside the bundle it must be borrowed BEFORE the
  move-out, or the fn restructured to borrow (not move) the bundle. Brainstorm the borrow
  shape first. `mtp_weights_present` can likely be computed on the fly (`b.mtp_head.is_some()`).
- **F — `pp_gpus`/`pp_scratch_set`/`pp_dn_la_to_device` → `Qwen35Bundle` (or a `Qwen35PpState`).**
  MEDIUM. Hazard: `reset_qwen35_recurrent` (daemon.rs ~5567) borrows `m.state` (for `b.dn_state`)
  AND `m.pp_gpus` as disjoint fields; moving pp_* under `&mut b` removes that disjointness →
  may need a split. Also EP teardown unwraps `m.pp_gpus` separately from `m.state.take()`.
- **G — `deepseek4_eos_tok`/`minimax_eos_tok` → `EpArch::{Ds4,Minimax}` fields.** MEDIUM.
  These are EP-path carriers (single-GPU ds4/minimax already store eos in the bundle). Hazard:
  ds4 TP/PP-dense paths REUSE `deepseek4_eos_tok` though their state isn't in `EpArch` — needs
  a home for those (bundle field or separate). All within `hipfire-loader` (no cross-crate).
- **H — `mtp_mode`/`mtp_k` → request params.** LOW but a design choice: these are per-request
  knobs (set from the request), arguably not model state — thread them as generate params
  instead of storing on `LoadedModel`.

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
