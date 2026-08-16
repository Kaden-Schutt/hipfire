# Phase 3 — the `LoadedModel` descent

**Status: scoped and measured, not started.** This is the only remaining path to two
Phase-2 deliverables that are otherwise structurally unreachable. Everything below was
measured against `arch/saddle` at `1f1bf9d1c`, not estimated.

## The two items this closes

| deliverable | state at `1f1bf9d1c` | why it is stuck |
|---|---|---|
| `carriers.rs` is registration only | 2,292 lines (was 2,656) | `Carrier::load` returns `LoadedModel`, a loader type |
| loader + daemon `ModelState::` refs 0 | daemon 3, loader 97 | Rust cannot construct an enum variant without naming it |

Both reduce to one fact: **`hipfire-loader` depends on every arch crate, so arch crates
cannot depend on the loader.** A `Carrier` impl living in an arch crate would have to name
`LoadedModel`, and that closes the cycle.

## The design that breaks it

Move `LoadedModel` and `Carrier` down into `hipfire-runtime`, which both sides already
depend on. Then arch crates implement `Carrier` in their own crate, and `carriers.rs`
becomes what the objective asks for: a list.

That requires `LoadedModel` to stop naming architecture types. It names six today:

```
pub state:             Option<ModelState>            // 11-variant closed enum
pub pp_scratch_set:    Option<Qwen35ScratchSet>
pub dots_ocr_config:   Option<dots_ocr::DotsOcrConfig>
pub dots_ocr_weights:  Option<dots_ocr::DotsOcrWeights>
pub vision_config:     Option<qwen35_vl::VisionConfig>
pub vision_weights:    Option<qwen35_vl::VisionWeights>
```

plus `qwen2_state` and `deepseek4_pbs` reachable from the same struct.

`state` becomes `Option<Box<dyn ArchModel>>`; `ArchModel` gains an `Any` supertrait with
`as_any_mut()`. The five side-car fields move **into their owning bundles**, where they
belonged all along — the VL pair exists only because `Qwen35Carrier` side-loads a vision
tower it cannot store anywhere else (`carriers.rs:497-543`).

## Measured blast radius

| surface | count | notes |
|---|---:|---|
| `hipfire-generate` `ModelState::` sites | **143** | 45 `if let Some(X(b))`, 33 match arms, 9 constructions |
| `hipfire-loader` `ModelState::` | 97 | adapter (22) + free dispatch (11) + construction |
| daemon `ModelState::` | 3 | redline snapshot, two VL-path resets |
| arch-typed `LoadedModel` fields to rehome | 6 | listed above |

`hipfire-generate`'s matches become downcasts. That is compatible with the objective's
protection of that crate — it keeps naming arch crates, which is the point of a composition
root; only the *form* changes from enum match to `as_any_mut().downcast_mut::<T>()`.

Cost is a downcast per request, not per token, provided the concrete reference is taken once
at the top of each `generate_*` body. Anything that downcasts inside the decode loop is a
defect, not a design.

## What this does NOT buy

**Roughly two registration sites.** The tax is already at **10 required out-of-crate code
sites**, which meets the objective. Do not undertake this for the number. Undertake it
because a closed 11-variant enum in another crate is a thing every new architecture must
edit, and deleting it is a genuine structural win.

## Why it was not done in the Phase-2 session

Not scope reduction — sequencing, on one specific ground: **the paths it touches are the
ones that session could not runtime-verify.**

The verification harness in use covered lfm2moe, gemma4, muse-glimmer and qwen35, all
text-only, on hiptrx. The refactor touches the VL side-load, dots-ocr, PP scratch, EP state
and unload. hiptrx carries **zero** VL/OCR fixtures; the six that exist are on the
single-GPU local box. A change to the VL side-load proven only to compile is exactly the
kind that ships silent breakage, and this repo's own rule is that decoded output — not a
green build — is the evidence.

## Verification this phase owes before it can land

1. **Text regression** — the existing four-architecture sweep on hiptrx, decoded output read.
2. **VL** — `dots-ocr` and an `ovisocr2` variant with a real image, locally, output read.
   No VL fixture exists on hiptrx; either copy one or accept the local single-GPU run.
3. **PP / EP** — the multi-GPU paths that own `pp_scratch_set` and `EpState`, on hiptrx's
   four R9700s. These have no coverage in the current harness and need one built.
4. **Unload** — a load/unload/reload loop per architecture. `free_gpu` moved once already in
   Phase 1; the Glimmer arm leaks ~1.3 GB over five cycles if only one side is freed
   (PR #566), so this needs a VRAM-delta check, not just an absence of crashes.

## Sequencing note

Do the side-car rehoming **first**, as its own landable change. Moving `vision_config` /
`vision_weights` into the Qwen35 bundle and the dots-ocr pair into its own is independently
valuable, independently verifiable, and shrinks the descent to the `state` field alone. A
single change that moves `LoadedModel`, deletes `ModelState`, rehomes six fields and rewrites
143 call sites is not reviewable.
