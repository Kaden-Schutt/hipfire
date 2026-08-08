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

## Fix (applied 2026-07-07)
`return;` after the `m.pp_dense.take()` drop (mirror the EP arm) — a dense `PpModel` drop fully tears
down its own `Gpus`/scratch/KV, so nothing else is owed, and returning restores the invariant :1708's
`.expect()` assumes (by the time control reaches `if m.pp > 1`, `pp>1` again genuinely means qwen35-PP).
Chosen over the guard `if m.pp > 1 && m.pp_gpus.is_some()` because it *removes* the ambiguity rather than
checking around it (define-the-error-out-of-existence).

## Status — REPRODUCED + FIXED + VALIDATED (2026-07-07)
- Repro: `crates/hipfire-runtime/examples/pp_unload_reload.rs` drives the LOADER path (`load_model_pp` →
  `unload_model`, emulated Pp-2). **Pre-fix: panics `crates/hipfire-loader/src/lib.rs:1708:34: pp>1 must
  carry pp_gpus` (EXIT 101)** — observed on gfx1151, exactly as traced.
- **Post-fix: `PASS: dense-PP load->unload->reload->unload, no panic` (EXIT 0)** — the reload proves the
  first unload freed cleanly. `cargo build --release --workspace --all-targets --locked` green.
- `pp_unload_reload` is the committed regression check (build-time in no-GPU CI; run under
  `HIPFIRE_EMULATE_GPUS=2` on a GPU box — it panics pre-fix, passes post-fix). Link
  [[device-mesh-pivot-execute-steps-spine]].
