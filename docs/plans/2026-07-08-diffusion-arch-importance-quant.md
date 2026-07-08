# Diffusion as first-class registry arch (A2) + shared bit-allocation pipeline

Status: proposed
Owner: (tbd)
Decision: **Option A2** — diffusion families become real registry arches with per-family
`arch_id`s written into the container header; server/coexist/runtime detect + route by id (with a
legacy fallback for existing `0x3046_4944` containers). The original goal — arch-importance-driven
diffusion quant — then rides on that registry membership.
Related: `crates/hipfire-arch-api/src/{lib,ingest}.rs`, `crates/hipfire-diffusion/src/{config,lib,quant_encode}.rs`,
`crates/hipfire-quantize/src/main.rs`, `crates/hipfire-server/src/routes/{health,sdapi}.rs`,
`crates/hipfire-server/src/lib.rs`, `crates/hipfire-diffusion-coexist/src/lib.rs`.

## 1. Problem / motivation

Two original questions: (1) should diffusion weights get an arch-importance prior driving bitwidth
(like LLMs), and (2) should the diffusion and standard quantizers be merged? The chosen direction
makes diffusion a **first-class arch** in the shared registry, which delivers (1) as a consequence
and answers (2) as "converge the pipeline, keep the CLI namespace." A2 additionally unifies the
on-disk `arch_id` + runtime routing, not just the quant path.

### Ground truth (verified)

LLM quant: `Ingest::importance(name) -> u8` (`ingest.rs:88`) -> `target_bits` (`:126`, 2/4/8 bpw)
-> `allocate(imp, req, k, codecs)` (`:143`) picks the smallest codec meeting the floor; called via
`high_precision_via_ingest(arch_id, name, k)` (`main.rs:2526`) against a 2-codec `MENU`.
Calibration (imatrix/Hessian) is a **separate** lever (AWQ scaling, Lloyd centroids) at a fixed bit
budget. `weight = calib * arch` is NOT current behavior on either path.

Diffusion today: no `hipfire-arch-api` dep; one reserved container id `HFQ_ARCH_DIFFUSION =
0x3046_4944` (`lib.rs:24`) used purely as an *is-diffusion* discriminator that the server routes on
(`health.rs:185/310`, `sdapi.rs:4469/8336`, `server/lib.rs:532/697`) and coexist reads
(`diffusion-coexist/lib.rs:1320`). Family is detected by **topology**, not the header —
`TransformerDenoiserFamily{QwenImage, Krea2, Unknown}` + `TransformerDenoiserWeightTopology`
(`config.rs:87,104`). Quant is uniform per format with one Opus rule (conv rank-4 -> oq8, linear
rank-2 -> oq4/oq4++ + optional LDLQ) in `encode_opus_tensor()` (`quant_encode.rs:117`). Codecs,
`oq4_ldlq_pack`, and `HessianSidecar` are already shared from `hipfire-quantize`.

### Two constraints A2 must respect

- **ID widths.** On-disk `HfqPackage.arch_id` is `u32`; registry `ArchId` is `u16`. Diffusion must
  therefore use **small u16 ids** (e.g. Krea2=40, QwenImage=41) that fit both — the old u32
  `0x3046_4944` cannot be a registry key.
- **Routing can't infer modality from an id.** Small diffusion ids share the integer space with LLM
  ids, so "is this diffusion?" must come from a **registry modality marker**, not an id range or a
  magic constant.

## 2. Goals / non-goals

Goals:
- Krea2/QwenImage are real registry arches (id + family + `Ingest` role prior + diffusion modality).
- New containers write the per-family `arch_id`; server/coexist/loader route by id via a
  `is_diffusion_arch(id)` predicate backed by the registry marker.
- **Legacy `0x3046_4944` containers keep loading and routing** unchanged (compat is mandatory).
- Diffusion quant importance flows through the *shared* `allocate()` (the original payoff).
- Default quant behavior unchanged until an opt-in flag; validate before flipping.

Non-goals (this plan): per-channel/mixed-width bit allocation; a bulk re-stamp of existing artifacts
(a converter is offered, not required); the `calib x arch` unified policy is opt-in and last.

## 3. Design

### 3.1 Registry: modality + diffusion arch ids

- Add a **modality marker** to the arch registry so routing is data-driven. Preferred: a
  `Diffusion` capability trait in `hipfire-arch-api` (mirrors `BatchedPrefill`/`SpecDecodeChain`),
  surfaced in `Caps`, so `RegisteredArch` exposes `caps.diffusion.is_some()`. Add
  `ArchRegistry::is_diffusion(id) -> bool` and (for the writer/loader) a small
  `diffusion_family(id)`/`arch_id_for_family(family)` mapping.
- Reserve small ids for diffusion families (Krea2, QwenImage) in the arch-api id constants next to
  the LLM `ARCH_ID_*`. Keep `0x3046_4944` defined as the **legacy generic-diffusion** id, recognized
  by `is_diffusion_arch` but never written for new containers.

### 3.2 Per-family `-spec` crates

Create `hipfire-arch-krea2-spec` and `hipfire-arch-qwenimage-spec` (leaf: `hipfire-arch-api` only,
same shape as `hipfire-arch-qwen35-spec`). Each:
- `impl Arch` (id, family) + the `Diffusion` capability.
- `impl Ingest` — the DiT role classifier -> importance prior. Initial MMDiT taxonomy (refine
  against real tensor names in P2):
  - embedders/patch/time/text-in, modulation/AdaLN (`.modulation.`, `norm_out.linear`,
    `attn.to_gate`), final projection (`final_layer.linear`), norms -> **255** (protect);
  - attention qkv/out proj -> **~200**; block MLP (`img_mlp`/`txt_mlp`) -> **128** (compress).
- Force-link them from `hipfire-archs` (add to `force_link` + the `hipfire-arch-specs` bundle).

Topology->family detection (currently `config.rs`) is factored into a pure classifier shared by the
quantizer (to choose the id to stamp) and the loader-compat path (to recover family from a legacy
DIF0 container).

### 3.3 Writer / detection (the A2 core) — CORRECTED

**Correction (verified in code):** routing is **metadata-based, not id-based**. `is_diffusion_hfq`
-> `inspect_hfq` keys on `artifact_kind == "diffusion"` + schema/class_name; the load fork is
`sdapi.rs:406 inspect_hfq(&path).is_ok()`. There is **no `arch_id == HFQ_ARCH_DIFFUSION` routing
check** — every `HFQ_ARCH_DIFFUSION` use is a writer (coexist import + test helpers). So there is
nothing to migrate on the routing side, and legacy compat is automatic (metadata unchanged).

The real, smaller A2 change:
- **Writer** (coexist import `write_import_entries_to_hfq`): stamp
  `diffusion_arch_id_for_metadata(metadata_json)` — the per-family id from the transformer
  `class_name` (Krea2 -> 17, Qwen-Image -> 18), else the legacy id. `quantize_diffusion_hfq` already
  preserves `hfq.arch_id`, so quant carries the stamp through.
- **Detection (hardened)**: `is_diffusion_hfq` keeps metadata as the primary signal and adds a
  secondary one — a registered diffusion `arch_id` in the header (via `hipfire_archs::is_diffusion_arch`,
  index-only read). Covers a container whose metadata is stripped but whose header identifies the family.
- **Family-from-id**: `ArchRegistry::diffusion_family(id)` gives P2 a clean `header_id -> family ->
  Ingest` path.
- **Legacy `0x3046_4944`**: single-sourced as `ARCH_ID_DIFFUSION_LEGACY`; still recognized by
  `is_diffusion_arch`; untouched containers load/route exactly as before.

### 3.4 Quant importance through shared `allocate()` (the payoff)

With the family known (from id or topology), `encode_opus_tensor()` consults the registry
`Ingest` for that arch and calls the shared `allocate(imp, req, k, &DIFFUSION_MENU)` instead of the
flat conv/linear heuristic. Conv still resolves to oq8 via `requires()` (random-access / no valid
grouped codec) — the special case falls out of the shared mechanism. Gated behind
`--arch-importance` (env `HIPFIRE_DIFFUSION_ARCH_IMPORTANCE`); default = today's flat rule until
validated.

### 3.5 Optional unified `calib x arch -> bits` (last, opt-in, both paths)

`target_bits_calibrated(importance, sensitivity)` in the shared layer: reduce imatrix/Hessian to a
per-tensor scalar `s`, combine as `target_bits(imp) * g(s)` (bounded), select codec. Wired into both
`high_precision_via_ingest` (LLM) and the diffusion MENU path behind one flag. Per-tensor only.

## 4. Phasing

- **P0 — registry membership, nothing routes yet.** Add the `Diffusion` capability + modality
  helpers to `hipfire-arch-api`; reserve Krea2/QwenImage ids; create the two `-spec` crates with
  `Ingest` role maps; force-link. Unit-test registry resolution + role map. No writer/routing change.
- **P1 — writer + hardened detection (DONE).** Routing turned out to be metadata-based (see 3.3),
  so no routing migration exists. Delivered: `hipfire_archs::is_diffusion_arch`; coexist import
  stamps per-family ids via `diffusion_arch_id_for_metadata`; `is_diffusion_hfq` also accepts a
  diffusion header id; `HFQ_ARCH_DIFFUSION` single-sourced to `ARCH_ID_DIFFUSION_LEGACY`; legacy
  containers unchanged.
- **P2 — importance through `allocate()` (the original goal).** Route diffusion Opus through the
  shared `allocate()` behind `--arch-importance`; refine the role taxonomy against real tensor names;
  validate Krea2 size/quality; flip default per-family when green.
- **P3 — `calib x arch` unified policy.** Shared `target_bits_calibrated`, opt-in, validated on
  tiny-quant-gate (LLM) + Krea2 (diffusion). Later: more families, per-channel/mixed-width.

## 5. Compatibility / migration

- Legacy `0x3046_4944` containers: recognized by `is_diffusion_arch`, loaded via topology detection.
  No forced re-quant.
- Offer (not require) `hipfire diffusion restamp <in> <out>` to rewrite a legacy header to the
  detected per-family id (pure metadata edit; no tensor transform), reusing the compose/decompose
  header-write machinery.
- No on-disk tensor format change; only the header `arch_id` value differs.

## 6. Validation

- P0: `cargo test -p hipfire-arch-api -p hipfire-arch-krea2-spec -p hipfire-arch-qwenimage-spec -p hipfire-archs`
  (registry resolves ids; `is_diffusion_arch` true for legacy + new; role-map assertions).
- P1: diffusion serve smoke on BOTH a new per-family container and a legacy DIF0 container (server
  routes both to the diffusion pipeline); coexist import stamps the right id.
- P2: Krea2 encode at a fixed target — size + a denoise/coherence check vs the pre-change baseline
  (the Krea2 timestep-embedding path already has a test, ref `cc9ee6e5d`); no regression at <= size.
- P3: `./tests/tiny-quant-gate.sh` stays `kld_drift=0` with the flag OFF; intended trade with it ON.
- `./tests/no-gpu-ci.sh` before every handoff.

## 7. Risks / open questions

- **Routing migration is inference-path + on-disk-format surgery** (portability-sensitive). The
  legacy fallback + `is_diffusion_arch` predicate are the safety net; every `HFQ_ARCH_DIFFUSION`
  call-site must be migrated (grep enumerated: server health/sdapi/lib, coexist) — none missed.
- Id allocation: pick Krea2/QwenImage ids that don't collide with any LLM `ArchId` (registry panics
  on conflict — good, but choose deliberately).
- The `Diffusion` capability is a new cross-cutting marker; confirm `Caps`/`register_arch!` extend
  cleanly and that non-diffusion arches default to "not diffusion".
- Role-map accuracy needs real Krea2/QwenImage tensor names (P2, not guessed).
- `hipfire-diffusion -> hipfire-arch-api` (+ the family->id map) is a new edge; arch-api is a leaf,
  no cycle — confirm build graph.

## 8. Answers to the original two questions

1. **Yes**, diffusion gets arch importance — via first-class registry membership (3.1–3.2) routed
   through the shared `allocate()` (3.4). The `calib x arch` multiply is a distinct, opt-in unified
   policy (3.5), not today's behavior.
2. **Converge, don't merge the binary.** Shared codecs/LDLQ/Hessian already; A2 also unifies
   `arch_id`/routing so diffusion is a first-class arch end-to-end. The `hipfire diffusion` CLI
   namespace stays.
