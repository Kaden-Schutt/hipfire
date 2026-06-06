# Ship 3.3 Dev Log — WMMA tile attention + vision/dflash + llama liveness (Phase C)

Branch: `integration/dispatch-unification`
Tracking: #397
GPU: gfx1151 (RDNA3.5 APU)

---

## Commit C0 · Dispatch infra: variant discriminator + shape AND + gfx12 predicate + 4-key schema + dead enum removal

### 2026-06-06 — C0 complete

**Sub-items:**

1. `types.rs`:
   - Removed dead `AttentionVariant` enum
   - Added `TileImpl` enum (12 variants + `None` default)
   - Added `KernelVariant.tile: TileImpl` field (with `Default` impl → `None`)
   - Added `ShapePredicate` variants: `BatchGe`, `HeadDimLe`, `HeadDimMultipleOf`, `HeadDimIn`, `IsTree`, `And`
   - Added `ShapeInfo.is_tree: bool` field (defaults to `false`)
   - Added `ArchPredicate::HasWmmaGfx12`
   - Added 4 full-attention `KernelKey`s: `AttnFullF16`, `AttnFullF32`, `AttnFullF16Causal`, `AttnFullF32Causal`
   - Updated `ShapeInfo` docs (m=seq_len for attention, batch_size=n_patches for vision)
2. `tables/mod.rs`: eval arms for all new `ShapePredicate` variants + `HasWmmaGfx12` arch check
3. All existing `KernelVariant` construction sites updated with `tile: TileImpl::None`
4. All existing `ShapeInfo` construction sites updated with `is_tree: false`

**Tests:** 115/115 hipfire-dispatch, 58/58 hipfire-dispatch-tests, 1/1 other.
All existing 3.1/3.2 keys resolve identically (`tile=None` path unchanged).

---

## Commit C1 · WMMA-FA acceleration of quantized prefill → registry variant

### Status: NOT STARTED

---

## Commit C2 · dots-ocr vision attention → `run_full_attention`

### Status: NOT STARTED

---

## Commit C3 · DFlash draft decoder attention → `run_full_attention`

### Status: NOT STARTED

---

## Commit C4 · llama legacy KV-mode liveness + registration

### Status: NOT STARTED

---

## Commit C5 · Verification sweep + env-gate retirement + cleanup

### Status: NOT STARTED
