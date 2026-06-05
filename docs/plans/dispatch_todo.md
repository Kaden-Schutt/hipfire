# Dispatch unification — deferred items (out of 1.1 scope)

Real findings from the Ship 1.1 plan review that are correct but outside
our work packet. Each item tracks back to the ship that should address it.

---

## From Ship 1.1 review

### D-1 · `w_down` Step variant for full pipeline coverage
**Source:** Gemini §2.1, Claude §2.1 implicit
**Ship:** Post-1.1 cleanup or Ship 1.2

`weight_gemv_swiglu_residual` is dispatch-internal (calls `GemvFamily::run`
with `GemvVariant::WithSwiGLUResidual`) but lives in
`hipfire-runtime/src/llama.rs:1168`. A `Step::GemvSwigluResidual` variant
would close the last model-code → dispatch-family dependency for `w_down`.
Low urgency — new quant formats don't require model code changes since the
function already selects the correct kernel via the dispatch crate.

### D-2 · Multi-GPU `forward_scratch_layers_multi` migration
**Source:** Gemini §4.1
**Ship:** Ship 5

`forward_scratch_layers_multi` uses `weight_gemv_prerotated` (not the
deleted helpers), so 1.1's deletions don't affect it. Migration to
`execute_steps` is Ship 5 scope alongside `forward_prefill_chunk`.

### D-3 · gfx1201 (RDNA4) hardware verification for QKVZA
**Source:** Claude §H4
**Ship:** Pre-merge gate for the dispatch-unification branch

Phase 0.6 makes RDNA4 verification non-optional. The QKVZA table entries
gate on `HasWmmaW32` which already ORs gfx12 (`tables/mod.rs:81`), so no
dead-gate risk. But actual `coherence-gate.sh` and `probe_commits.sh` on
gfx1201 hardware must run before the branch merges. The GPU-free
`(arch × dtype)` coverage golden is in 1.1 scope; hardware testing is
hardware-gated.

### D-4 · Same-binary `HIPFIRE_DISPATCH_OLD/_NEW` selector
**Source:** Claude §M2, Phase 0.6
**Ship:** Ship 1 or 2 (before mass migration)

Phase 0.6 calls for a temporary selector alive through the migration
window. For 1.1 (4 call sites), cross-commit comparison is adequate.
The selector becomes higher-value when Ships 2–5 migrate multiple model
archs simultaneously. Track as a pre-Ship-3 deliverable.

---

## From broader branch review (not specific to 1.1)

### D-5 · Phase 0.4: collapse `HasWmmaW32` → `HasWmma`, delete `HasWmmaW32Gfx12`
**Source:** Phase 0.4 decision, #397 roadmap
**Ship:** Pre-Ship-3 (before new kernels register under stale predicates)

`HasWmmaW32Gfx12` has zero kernel registrations. The decision is to
collapse to a single `HasWmma` predicate backed by `ArchCaps::has_wmma()`.
`HasWmmaW32` currently ORs both gfx11 and gfx12, so functionally correct
today. The rename is a hygiene fix that prevents future authors from
registering under a gfx12-only predicate that excludes gfx11.

### D-6 · `err_wrong_arity` mislabel in `dispatch_fused_qkv`
**Source:** Ship 1.1 checklist item 1.4
**Ship:** Ship 1.1 Commit 3 (cleanup) or Ship 1.4

The `err_wrong_arity` function in `families/fused_qkv.rs` labels arity
errors with the wrong `FusedQkvVariant` for QKVZA keys (reports `Qkv`
instead of `Qkvza`). Cosmetic but confusing during debugging.

### D-7 · `debug_assert!` on double-rotation probe in `weight_gemv_prerotated`
**Source:** Ship 1.1 checklist item 1.4
**Ship:** Ship 1.4 (F1: double-rotation probe)

MQ-family `run_auto` on already-FWHT-rotated input re-rotates. FWHT is
involutory so the result is effectively un-rotated — silent correctness
hazard. Add `debug_assert!` probe at `llama.rs:1055` (or equivalent).
If assert fires in any real run → correctness bug → route through
`GemvVariant::Prerotated`.

### D-8 · Dead code after dispatch 1.1 migration
**Source:** Ship 1.1 Commit 2 (forward_from_x_gpu collapse)
**Ship:** Post-1.1 cleanup or Ship 5

`forward_from_x_gpu` was collapsed to delegate to `forward_scratch_layers`,
which made `moe_ffn_decode` (the per-call-alloc variant) dead code — all
MoE paths now go through `moe_ffn_decode_with_scratch` or the `_prerotated`
variant. `moe_ffn_decode_impl` and the `_with_scratch` variants remain
alive. Deleting `moe_ffn_decode` is safe once Ship 5 confirms the prefill
path also uses the scratch variants.

Also safe to clean up in the same pass:
- `MoeScratchRef::gate_up_buf` field (compiler-flagged dead — read by
  dispatch crate but not by qwen35.rs directly)
- Pre-existing unused variables: `kv_layer_idx` in forward_scratch_layers,
  `load_norm_weight_raw`, `slice_f32_view`

### D-9 · Paro weight alignment invariant (Ship 1.2)
**Source:** Ship 1.2 commit 2 (Paro fused entries)
**Status:** Verified by code inspection; GPU byte-parity deferred to coworker

ParoQ4G128 group-128 quantization guarantees `k % 128 == 0` by construction
(128-element groups). `m % 8 == 0` holds for all qwen35 Paro weight matrices
because the model's hidden dims are multiples of 8. The fused kernel's
`m%8==0` / `k%128==0` asserts in `fused_qkv_table.rs` / `steps.rs` guards
mirror the kernel asserts exactly — the guards are optimization gates, not
the correctness boundary. The per-op fallback (`gemv_hfq4g128`) has its own
alignment contract that is always satisfied.

### D-10 · Ship 2 handoff: Q4K / Q8_0 fused entries
**Source:** Ship 1.2 scope change (Q4K/Q8_0 → Ship 2)
**Ship:** Ship 2

Q4K and Q8_0 fused-table entries (`FusedQkvQ4K`, `FusedGateUpQ4K`,
`FusedGateUpQ8_0`) were moved to Ship 2 because they need llama/qwen2 model
integration + GPU byte-parity landing together. The dispatch crate already
has `FusedQkvQ4K` and `FusedGateUpQ4K` kernel keys, table entries, and
dispatch arms — what's missing is the step-pattern wiring in `steps.rs`
(guards + FUSED_TABLE entries + launch_fused arms) and GPU-verified parity.

### D-11 · Paro GPU verification deferred
**Source:** Ship 1.2 commit 3
**Status:** Deferred to coworker with Paro model + gfx1100/gfx1201

The following Commit 3 verification items require a Paro model on GPU:
- Byte-identical-vs-master token IDs on A3B-PARO, gfx1100 + gfx1201
- Force-unfused: coherence pass + cosine ≥ 0.9999
- probe_commits.sh master HEAD: parity with master + gain vs parent
- Multi path (forward_scratch_layers_multi) still works
The infrastructure is in place and CPU-side tests pass; coworker will run
the GPU verification and report results.
