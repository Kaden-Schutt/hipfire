---
title: Dense load_model_pp UNLOAD panics — informational `pp` scalar falls through into the qwen35-PP teardown (`pp_gpus.expect`)
date: 2026-07-07
tags: [device-mesh, pp, unload, panic, loader, pp_gpus, pp_dense, qwen35, 462-class, bug, dense-pp]
---

Surfaced by the 2026-07-07 device-mesh review (was a vague "lib.rs:1702 pp>1 unload panic" TODO in
[[device-mesh-pivot-execute-steps-spine]]). Root-caused from code here. **Live, not merely latent; not fixed.**

## Symptom
Unloading (or model-switching away from) a DENSE pipeline-parallel model — llama-family arch 0/1 loaded
via `load_model_pp` with `pp>1` — panics: `pp>1 must carry pp_gpus`.

## Site
`crates/hipfire-loader/src/lib.rs`, `fn unload_model` (:1623):
- panic at **:1708** `let mut gpus = m.pp_gpus.expect("pp>1 must carry pp_gpus");`
- (secondary, same arm) :1715 `m.pp_dn_la_to_device.expect("pp>1 must carry la_to_device")`
- NB the old TODO said ":1702" — line drift; :1702 is now `drain_pool()` inside the EP arm.

## Root cause (traced) — an informational field bleeds into a path keyed on it
1. `load_model_pp` sets `pp: mesh.size_of(DimKind::Pp)` (≥2) as an **informational** "requested degree"
   (:1355); the actual dense-PP state lives in `pp_dense: Some(PpModel)` and **`pp_gpus` stays `None`**
   (`pp_gpus` is the *qwen35*-PP field, set only by the qwen35 loader).
2. In `unload_model` the dense-PP arm `if let Some(pp) = m.pp_dense.take() { drop(pp); }` (:1633) frees
   the `PpModel` (which owns its own `Gpus`/scratch/KV) BUT does **not `return`** — unlike the EP arm,
   which `return`s at :1704.
3. Execution falls through to `if m.pp > 1 {` (:1707) — TRUE because of the informational scalar — the
   **qwen35-PP teardown** ("Only Qwen35 supports pp>1 today", :1717) — and hits
   `m.pp_gpus.expect(...)` (:1708) → panic (`pp_gpus` is `None` for dense PP).

**#462-class:** a field set for one purpose (informational degree) drives a code path that assumes a
different invariant (qwen35 `pp_gpus` present).

## Why the other axes are safe
- **TP:** the `m.tp.take()` arm (:1628) also doesn't `return`, but a TpModel leaves `pp` at its default
  (1), so `if m.pp > 1` is false. Safe.
- **qwen35-PP:** sets `pp_gpus` (+ `pp_dn_la_to_device`), so the `.expect()`s hold. Safe.
- Only the **dense-PP** case (`pp_dense=Some`, `pp` scalar ≥2, `pp_gpus=None`) collides.

## Interaction with recent work (corrects the old "pre-existing / qwen35-only" framing)
This is NOT a pre-existing qwen35-only issue. It's the composition of two device-mesh changes: the
informational `pp` scalar at :1355 (mesh-through-loader) + the dense-PP `pp_dense` arm (P-C). Together
they route a dense-PP unload into the qwen35-PP arm.

## Candidate fix (NOT applied)
Prefer: add `return;` after the `m.pp_dense.take()` drop at :1633 (mirror the EP arm at :1704) — a dense
`PpModel` drop fully tears down its own `Gpus`/scratch/KV, so nothing else is owed. Alternative: guard
`if m.pp > 1 && m.pp_gpus.is_some()` (equivalently `&& m.pp_dense.is_none()`). The `return` is the
cleanest and matches the EP/TP arm intent.

## Status
Traced from code, **not reproduced live**. Confirm: load a dense llama `pp:2` under
`HIPFIRE_EMULATE_GPUS=2`, then unload / switch models — the daemon should crash at :1708 today. Then
apply the `return` fix + add a dense-PP unload regression check (load→unload→reload). Link
[[device-mesh-pivot-execute-steps-spine]].
