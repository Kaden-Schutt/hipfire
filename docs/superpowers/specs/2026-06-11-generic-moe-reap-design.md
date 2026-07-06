# Generic MoE REAP — selective expert pruning + selective re-quant

**Date:** 2026-06-11
**Branch:** `nw_generic_moe_reap` (worktree `~/repos/hipfire-reap`, based off `nw_reap162b_keepmap` @ `43ed446d`)
**Author:** Nick Woolmer

## Problem

The existing REAP loader (`43ed446d`, branch `nw_reap162b_keepmap`) is **DeepSeek-V4-specific**.
It lets us emulate a REAP-pruned checkpoint (e.g. 0xSero 162B, 256→144 experts) by
partial-loading the kept experts from an existing full mq2-lloyd quant — no re-quant —
via an env-gated keep-map sidecar (`HIPFIRE_DEEPSEEK4_REAP_KEEPMAP`). It validated cleanly
(keep-all identity reproduces baseline NLL to 10 decimals) and produced the K144 finding
(full PPL 7.56 vs pruned 17.73).

We want two generalizations:

1. **Generic across all MoE archs.** Any user should be able to selectively reap any MoE
   model — not just ds4. Target archs present on the integration base are `deepseek4`,
   `qwen35`, `lfm2moe`, `minimax`. **`cohere2moe` is currently in-flight on branch
   `nw_cohere2moe_support` and not yet on master** — it gets the *identical* one-call
   `ExpertLoaderHook` wiring once it merges; we do not build on that moving branch here.
2. **Lightweight selective re-quant.** Selectively up- or down-quant specific layers/experts
   *without* re-quantizing the whole model, to iterate and fine-tune a quant config cheaply.

## Prior art (reuse, don't duplicate)

- **Existing ds4 REAP loader** (`43ed446d`): `ReapKeepMap` + `from_hfq` env hook in `deepseek4.rs`;
  per-expert byte row-gather `upload_quant_or_f16_keep` + hash `tid2eid` remap in `arch.rs`;
  `examples/deepseek4_perplexity.rs` + `scripts/reap/`. This is the foundation we lift.
- **Mixed-precision K-map** (`docs/superpowers/specs/2026-05-08-mixed-quant-kmap-design.md`,
  implemented in `hipfire-quantize/src/main.rs`: `enum QuantLevel`, `kmap_resolve_mode`,
  `Promote6`). This is the *build-time, heuristic* mixed-precision path (tensor role/layer/MoE →
  level → one new `.hfq`). Our `quant_overrides` is the *explicit, manual* counterpart, and
  `reap bake` reuses the quantizer's per-format encoders + K-map machinery rather than
  reimplementing quant math. **Distinction:** K-map promotes by *tensor role* (lm_head, router,
  edge layers) where tensors are already separately dispatched; our novel runtime work is mixing
  tiers *among routed experts within a single layer's expert pool*, which needs new dispatch.

## Non-goals

- New mixed-type MoE GEMV kernels (we bucket existing single-tier kernels instead).
- Prefill grouped-GEMM support for mixed-tier layers (decode/scoring is the target; see §2 scope honesty).
- Combining REAP with EP (expert-parallel) sharding — kept mutually exclusive, as today.

## Artifacts & contract

Two artifacts per experiment, plus an activation env var.

### A. `reap_plan.json` — metadata/contract (no weight bytes)

```jsonc
{
  "version": 1,
  "model_arch": "deepseek4",        // optional; validated against detected arch
  "original_experts": 256,          // base per-layer routed count; validated pre-override
  "num_layers": 43,

  "keep": {                         // OPTIONAL — omit ⇒ no pruning (keep all)
    "per_layer": [[0,1,5, /* … */ ], /* … */ ]   // kept ORIGINAL indices, compact-slot order
  },

  "quant_overrides": [              // OPTIONAL — manifest of re-quantized tensors
    { "layer": 20, "role": "routed_experts", "experts": [7,12], "tier": "mq3lloyd" },
    { "layer": 41, "role": "attention",                          "tier": "q8"       }
  ],

  "arch": { "deepseek4": { "tid2eid_layers": [0,1,2] } }  // arch-specific extras
}
```

- `role` ∈ `routed_experts | shared_expert | attention | router | lm_head | embed`.
- `experts` is valid only for `role: routed_experts` (absent ⇒ whole role at that layer).
- `tier` is a quant-format name the quantizer understands (`q8`, `f16`, `mq2lloyd`,
  `mq3lloyd`, `mq4lloyd`, `hfq4g256`, `hfq6g256`, …).

### B. `overlay.hfq` — small HFQ sidecar with only the re-quantized tensors

Keyed by the **same tensor names** as the base file (`{prefix}.ffn.experts.7.w1.weight`, …).
Built by `hipfire reap quant`. Cheap because only targeted tensors are re-quantized.

### Loader resolution rule (the heart of it)

When fetching tensor `N`: check `overlay.hfq` first, else base `.hfq`; then apply keep-gather.
Dispatch reads each expert's effective tier from the plan to bucket kernels. This reuses the
existing tensor-by-name HFQ lookup, so all 5 arches get overlay support at the same seam they
already load from.

### Activation

`HIPFIRE_REAP_PLAN=<dir>` (dir holds `reap_plan.json` + optional `overlay.hfq` + arch sidecars),
generalizing today's `HIPFIRE_DEEPSEEK4_REAP_KEEPMAP`. Default-off ⇒ untouched base load.
The old env var is kept as a thin alias (parses into a `keep`-only plan) for back-compat with
existing ds4 sidecars.

## Decomposition (4 sub-projects, dependency order)

1. **Generic reap keep-map loader** — `hipfire-reap` crate + `ExpertLoaderHook` seam across all
   5 arches. Independent; extends existing code.
2. **Mixed-quant dispatch** — `MoeDtypes` per-expert tier vectors + bucketed dispatch. Prereq for
   per-expert/mixed quant at runtime.
3. **Load-time quant overlay** — `overlay.hfq` spliced over base at load (fast iterate loop).
   Depends on 2.
4. **Offline bake** — quantizer writes a standalone `.hfq` from plan + overlay. Depends on 1+3
   plan format.

---

## §1 — `hipfire-reap` crate & loader seam (sub-project 1)

**New crate `hipfire-reap`** (deps: `hipfire-runtime` for `HfqFile`/`Gpu`/`DType`; depended on by
each arch crate). Owns all model-agnostic logic so arch crates stay thin.

- **`ReapPlan`** — parse + validate `reap_plan.json`. Generalizes the ds4 `ReapKeepMap`
  validation (layer count, `original_experts`, in-range indices, role/expert misuse). Resolves
  the optional `overlay.hfq` handle.
- **`TensorSource`** — overlay-then-base resolution: `fn tensor(name) -> (info, bytes)`. Replaces
  raw `hfq.tensor_data_pread(name)` in arch loaders.
- **`gather_rows(info, bytes, keep) -> (info', bytes')`** — generalized, arch-free version of ds4
  `upload_quant_or_f16_keep`. Exact byte gather for row-independent quant (F16/Q8/MQ*-G256);
  errors on row-coupled formats. Used for router gate/bias and any expert-pruned role.
- **`ExpertPlan { keep: Option<&[u32]>, tier_of_slot: impl Fn(usize) -> DType }`** per (layer,
  role) — what the arch loader asks `hipfire-reap` for.
- **`ReapArchHook`** — callback the plan invokes for arch-specific bits (ds4 `tid2eid` remap +
  MTP-skip). Only ds4 implements it; `hipfire-reap` never learns about it.

**The seam in each arch** is the existing expert-enumeration loop
(`for e in 0..n_exp { … pread(experts.{e}…) … }`). It becomes: iterate compact slots, resolve
`src = keep[slot]`, fetch via `TensorSource`, group the upload **by tier**:

- **Blob-packing arches (deepseek4 `arch.rs:163-333`, minimax `minimax.rs:280-517`):** build
  **one sub-blob per quant tier present in the layer** instead of one blob/layer. Each expert's
  pointer-table slot points into its tier's sub-blob (pointer table is already per-expert → no
  kernel change). Uniform-tier layer ⇒ one sub-blob ⇒ byte-identical to today.
- **Buffer arches (qwen35 `qwen35.rs:4331-4435`, cohere2moe `cohere2moe.rs:188-237`, lfm2moe
  `lfm2moe.rs:345-454`):** already one `WeightTensor` (+ `gpu_dtype`) per expert → mixed tiers
  naturally representable; only change is recording the per-expert tier for dispatch.

**Byte-identical guarantee:** no plan ⇒ original path unchanged. Keep-all + no overrides ⇒ gather
is identity and single-tier sub-blob equals today's blob. The identity-sidecar test becomes the
cross-arch regression gate.

**Files touched:** new `crates/hipfire-reap/`; `arch.rs`/`*.rs` expert loops in all 5 arch crates;
`deepseek4.rs` env hook → delegate to `hipfire-reap` + register `ReapArchHook`.

---

## §2 — Mixed-quant dispatch (sub-project 2)

Today `MoeDtypes` (`hipfire-dispatch/src/families/moe.rs:41-50`) carries one `routed_gate_up`/
`routed_down` `DType` sampled from `expert[0]`; `pipeline/mod.rs:207-435` branches monolithically.

- **Extend `MoeDtypes`** with `per_expert_gate_up: Option<Vec<DType>>` and
  `per_expert_down: Option<Vec<DType>>`. `None` ⇒ today's uniform path, untouched. `Some(v)` ⇒
  mixed path. Arch `MoeDtypes` builder fills these from the `ExpertPlan` tier table only when the
  layer is actually mixed.
- **`MoeResolution::resolve`** gains `mixed: bool`. Set ⇒ **bucketed path**: group the layer's
  top-k selected experts by tier; for each tier call the existing single-tier
  `gemv_*_moe_*_indexed` kernel over that tier's sub-blob + remapped compact indices, accumulating
  into the same output. 2-3 tiers ⇒ 2-3 launches/layer; uniform layers never enter this path.
- **Index remapping:** each bucket gets a compact index list (selected experts of that tier) +
  the tier sub-blob pointer table from §1 — the kernel sees the contiguous-stride layout it
  already expects. **No kernel signature changes.**
- **Both decode sites covered:** generic `run_moe_decode` (`pipeline/mod.rs:207`) *and* ds4
  `run_moe_decode_bias_aware` (`pipeline/mod.rs:604`) / `ffn_hash_routed`
  (`deepseek4/forward.rs:3505`). The memory flags these as two separate dispatch sites (the
  all-NaN gotcha) — the bucketing helper lives in `hipfire-dispatch`; both call it.

**Scope honesty:** this targets decode/scoring (PPL/KLD, serve-decode). Prefill grouped-GEMM
kernels stay single-tier per layer; a mixed layer falls back to per-tier grouped calls, or if a
tier lacks a prefill kernel (e.g. mq3) that layer is **decode-only** — matching the current mq3
"score-not-serve" limitation. Spec calls this out; no silent degradation.

---

## §3 — Quant tooling: overlays & bake (sub-projects 3 & 4)

**`hipfire reap quant`** (build an overlay): reads the base `.hfq`; for each `quant_overrides`
entry re-quantizes only the targeted tensors to `tier`, writing them into `overlay.hfq` under
their original names. Reuses existing per-format encoders + the K-map `kmap_resolve_mode` path in
`hipfire-quantize/src/main.rs` — no new quant math. Re-quanting "layer 20 experts 7,12" is
seconds, not full-model hours. This is the fast iterate loop.

- Per-expert `routed_experts` with `experts: [...]` ⇒ only those experts' `w1/w3/w2` (or
  `gate_up_proj/down_proj`) rows are re-quantized; the loader tier table marks those slots so §1
  packs them into the right tier sub-blob and §2 buckets them.
- Whole-role overrides (attention/router/lm_head/embed) replace the single tensor wholesale.

**`hipfire reap bake`** (freeze to standalone `.hfq`): consumes the same `reap_plan.json` +
`overlay.hfq` and writes an ordinary servable `.hfq`:
- pruned experts dropped, kept experts renumbered to compact slots (byte row-gather → disk);
- overlay tensors written in place of base counterparts;
- arch sidecars (ds4 `tid2eid`) folded into file metadata so the baked model needs no plan/env at
  load.

Result loads through the normal path, no env var. Overlay = iterate; bake = freeze once happy.
Both consume one plan, so what you tuned is exactly what you bake.

---

## §4 — Testing & validation

Per-sub-project gates, anchored on the **identity invariant** (keep-all + no overrides reproduces
baseline NLL to 10 decimals).

- **SP1 (generic loader):** identity-sidecar test → **cross-arch regression gate**: each of the 5
  MoE arches must reproduce its no-plan baseline logits (bit-for-bit, or NLL to 10 decimals) under
  a keep-all plan. Unit tests on `ReapPlan` validation (bad layer counts, out-of-range indices,
  role/expert misuse) and `gather_rows` exactness (gathered subset == direct per-row reads, for
  F16/Q8/MQ*-G256).
- **SP2 (mixed dispatch):** **bucketing-equivalence test** — a layer whose experts are all one
  tier, routed through the *mixed* path, must match the uniform path exactly (bucketing is a no-op
  decomposition before any real mixing). Run on both `run_moe_decode` and ds4 bias-aware/hash
  paths.
- **SP3 (overlay):** overlay re-quantizing a tensor **to the tier it already is** must reproduce
  base load (plumbing exact when re-quant is a no-op). Then a real down-quant (one layer mq2→mq3)
  shows the expected small, *localized* NLL shift via the generalized `scripts/reap` PPL/KLD
  harness.
- **SP4 (bake):** baked model must produce **identical logits to the same plan applied via overlay
  at load** (bake == freeze the overlay path). Baked file loads with no env var and serves through
  the normal path.

**End-to-end smoke:** reproduce the existing 162B K144 result (full 7.56 vs pruned 17.73 PPL)
through the new *generic* ds4 path — confirms no regression from the lift.

Each gate is a written assertion before the code (TDD-friendly).

## Open questions / risks

- **`gather_rows` on row-coupled quant:** must hard-error (not silently corrupt) for any format
  whose rows aren't self-contained. Enumerate which current formats are row-independent up front.
- **Tier sub-blob count blow-up:** pathological plans (many tiers/layer) multiply sub-blobs +
  launches. Cap/ warn beyond N tiers per layer.
- **K-map ↔ overrides precedence:** if a baked model is produced with both `--kmap` and a reap
  plan, define precedence (proposed: explicit `quant_overrides` win over heuristic `kmap_resolve`).
