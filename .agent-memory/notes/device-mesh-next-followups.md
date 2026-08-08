---
title: NEXT device-mesh follow-ups — #4 EP manifest replication + god-struct field collapse
tags: [device-mesh, review, next, handover, weight-manifest, god-struct, archdispatch]
created: 2026-07-11
updated: 2026-07-11
---

> **Historical document.** This file preserves dated implementation and validation evidence. Current status and remaining work are tracked only in [device-mesh-refactor-tracker.md](../../.agent-progress/device-mesh-refactor-tracker.md).

# HANDOVER: tackle these two next (device-mesh review leftovers)

Context: the 2026-07-10 external review had 6 findings (see
`device-mesh-review-findings-2026-07-10.md`). #2 (TP/PP unload leak) + #3
(peer-access order) fixed and GPU-validated; #1 (dense PP per-stage residency)
DONE + validated (`eafd8663..460e3800`). **Remaining, in priority order:**

## 1. #4 — Manifest replication is wrong for EP  ✅ DONE 2026-07-11

**Fixed** in commits `4f55a274` (`DeviceMesh::stage_devices` = the owning stage's full
compute grid), `8c441c76` (`placement_devices` + `plan_manifest` state route every
non-Pin/Tied policy through `stage_devices`), `be5c4bdb` (`fulfill_manifest` derives the
shard count from `mesh.size_of(Tp|Ep)`, not `devices.len()` — so a TP-shard policy on a
Tp-less/EP mesh replicates instead of slicing/refusing). Both the `Replicate` (deepseek4)
and the `ColumnShard`/`RowShard`/`FusedQkv` (minimax) attention classes now land on every
EP rank. Composed Tp×Ep: placement is composed-correct (pure topology); fulfill
`debug_assert`s single-axis — composed slicing deferred to **Phase 5b** (unreachable today:
`resolve_mesh` builds single-axis only). Opus whole-branch review: READY TO MERGE, 3
doc-only Minors (non-blocking). Verify: build + `--lib` PASS; mesh 8/8, weight_manifest
10/10, weight_store 8/8. Spec: `docs/superpowers/specs/2026-07-11-ep-manifest-replication-design.md`.

<details><summary>original bug writeup (kept for context)</summary>


**Bug:** `crates/hipfire-runtime/src/weight_manifest.rs:68` —
`_ => mesh.group_along(DimKind::Tp, &coord)`. On an **Ep-only mesh** there is no
Tp axis, so `group_along(Tp)` returns a **singleton `[0]`**. A generic EP load
would therefore place *replicated* attention / router / norm weights (the
`Replicate` / `ColumnShard` / etc. arms that fall through to `_`) on **device 0
only** — every other EP rank would be missing them.

**Encoded in a test that currently asserts the bug** (`weight_manifest.rs:~520`):
```rust
let ep = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
assert_eq!(placement_devices(&e, &ep, 4), vec![0]);   // <-- a HeadSharded weight lands ONLY on dev 0
```

**Why it hasn't bitten yet (latent):** the LIVE EP path (deepseek4 / minimax)
uses hand-written `forward_ep` + imperative loaders, NOT this manifest. The
manifest/`fulfill_manifest` path is scaffolding not yet wired for EP serving. So
this is correct-it-before-you-wire-EP-through-the-manifest, not a prod fire.

**Fix direction (needs a decision, then small code):** on an Ep-only mesh,
replicated (non-sharded) weights should span the **Ep group** (all ranks), not a
Tp singleton. The clean shape is probably: replicate spans "the whole device set
of the owning stage" = Tp-group ∪ Ep-group as appropriate, or explicitly the Ep
group when there's no Tp axis. Decide the semantics for composed Tp×Ep meshes too
(replicated weight on a 2×2 Tp×Ep mesh = on all 4? on the Tp group of each Ep
rank?). Then flip the test to assert the correct placement. **Small diff, real
design question — brainstorm the replicate-on-EP semantics first.** Verify with
the existing `placement_where_by_mesh_and_policy` / `head_sharded_*` unit tests
(pure CPU, no GPU).

</details>

## 2. God-struct field collapse — `LoadedModel` ~20 `Option` fields  (the #462 surface)  ← IN PROGRESS

**Spec DONE + Increment 1 LANDED 2026-07-11** (commits `33c9fe29..e16e7c01`): `SessionState`
(resettable per-request fields: seq_pos/conversation_tokens/prefill+dflash_checkpoints/kv_adaptive)
+ `PersistState` (asst_turn_cache/decoded_vocab — SURVIVE reset) + `session_parts_mut` disjoint-borrow
splitter + `reset_context`→`SessionState::reset(gpu)`. Opus whole-branch review READY-TO-LAND;
serve-multiturn PASS (#462 guard); byte-preserving. **Design resolved the self-ref fork** (see below):
transient wrapper + lazy `parts_mut` (native disjoint-field borrow, no unsafe, NOT a stored trait
object), 4-way split, compiler-total reset. Spec: `docs/superpowers/specs/2026-07-11-loadedmodel-god-struct-field-collapse-design.md`;
Inc-1 plan+ledger: `docs/superpowers/plans/2026-07-11-god-struct-collapse-inc1-sessionstate.md` /
`.superpowers/sdd/godstruct-inc1-progress.md`. **Inc 2 Steps A+B DONE 2026-07-11** (`fa2edc62` drop dead always-None kv_cache/dn_state; `bf59147a`
fold deepseek4_pbs → `Deepseek4Bundle.pbs`; opus READY-TO-LAND; ds4 probe + qwen35 serve-multiturn PASS;
plan `docs/superpowers/plans/2026-07-11-god-struct-collapse-inc2-deadfields-ds4pbs.md`). **Inc 2 REMAINING
(Steps D-H, hazard-ordered, own plans):** **C DONE 2026-07-11** (`c3b8f789` dots-ocr → `ModelState::DotsOcr`,
transient bundle collapsed to in-place m.state borrow; opus READY-TO-LAND; dots-ocr load + qwen35 serve-multiturn PASS;
plan `docs/superpowers/plans/2026-07-11-god-struct-collapse-inc2c-dotsocr.md`); **D DONE 2026-07-11** (vision → ONE `Option<Qwen35Vl>` LOADER-SIDE field, NOT `Qwen35Bundle` — VisionConfig/Weights are in the separate `hipfire-arch-qwen35-vl` crate so base→ext layering inversion was avoided per bjoern; workspace + qwen35 serve-multiturn PASS); E `qwen35_mtp_head`+`mtp_weights_present` → `Qwen35Bundle`
(needs `generate_qwen35_mtp` move-out borrow restructure); F `pp_gpus`/`pp_scratch_set`/`pp_dn_la_to_device`
→ `Qwen35Bundle` (disjoint-borrow hazard: `reset_qwen35_recurrent` borrows `m.state` AND `m.pp_gpus`); G
`deepseek4_eos_tok`/`minimax_eos_tok` → `EpArch::{Ds4,Minimax}` fields; H `mtp_mode`/`mtp_k` → request params.
Then `ModelParallel` (7 axis fields), then `ImmutableMeta`. Terrain map in the Inc-2 exploration. LESSON:
adding a REQUIRED bundle field needs `--workspace --all-targets` (breaks all construction sites, e.g.
dspark_bench.rs, not just the daemon).
GOTCHA: never run `fmt-changed.sh`/`cargo fmt` on daemon.rs/lib.rs (whole-file reformat churn) — edits
are pure field-path renames, hand-write them rustfmt-clean. Original problem writeup below (still applies to Inc 2+).

**State:** the ArchDispatch / ModelParallel work flipped ALL arches onto the one
`ar_generate` driver (Axis A/B + minimax-EP + dense TP/PP folds — all done,
`.superpowers/sdd/progress.md`). But that was the **driver-level** collapse. The
`LoadedModel` **struct** still carries ~20 `Option<...>` fields (kv/dn state,
tp/pp_dense/ep bundles, pflash, eviction, checkpoints, decoded_vocab, …) — the
change-amplification / unknown-unknowns surface that the #462 class of bugs lives
on. This is the one unification "actually worth it, INDEPENDENT of the Step-IR
decomposition."

**KEY CONSTRAINT discovered during the archdispatch review (do NOT forget):** the
per-arch dispatch structs (`Qwen35Dispatch<'m>{ m: &'m mut LoadedModel }`, etc.,
daemon.rs) **borrow `&mut LoadedModel`**, so they **cannot** be stored inside
`LoadedModel` as the once-"approved" `LoadedModel { arch: Box<dyn ArchDispatch> }`
— that is self-referential and won't compile. The current design sidesteps it by
constructing the dispatch **transiently per-call**. So the god-struct collapse
must adopt a DIFFERENT ownership model — e.g. arch state owned in a single
`ModelState`/enum the dispatch borrows transiently, NOT a stored trait object.
**This is a genuine design fork → brainstorm first, don't jump to code.**

Refs: `docs/superpowers/specs/2026-07-09-daemon-god-struct-archdispatch-design.md`,
`docs/superpowers/specs/2026-07-10-axis-b-modelparallel-collapse-design.md`,
and the "Still NOT done" line in global MEMORY.md.

## Process reminder (worked well this round)

Both items are structural + latent (no GPU needed to START). The pattern that
worked for #1: adversarial pre-review of the plan (correctness / borrow /
teardown lenses) BEFORE implementing → subagent-driven per-task execution →
opus final whole-branch review. Reach for `superpowers:brainstorming` first on
both (each has a real design fork), then `writing-plans`, then
`subagent-driven-development`.
