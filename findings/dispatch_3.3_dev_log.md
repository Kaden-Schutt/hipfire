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

### 2026-06-06 — C1 complete

Registered WMMA-FA tile variants under `AttnFlashAsym4BatchedMasked` with priority
ordering: gfx12 → gfx11 → scalar (DO NOT REORDER).

**rdna-compute:**
- Added `attention_flash_asym4_wmma_tile_batched` (gfx11) and
  `attention_flash_asym4_wmma_tile_batched_gfx12` (gfx12) public methods
- Both call `launch_asym_flash_batched` with the WMMA kernel name directly
  (bypassing the inline env-gated ladder)

**attention_table.rs:**
- Registered `Asym4WmmaTileGfx12` with `HasWmmaGfx12` + `And(&[HeadDimIn(&[128,256]),
  BatchGe(16), IsTree(false)])`
- Registered `Asym4WmmaTile` with `HasWmma` + same shape gate
- Scalar variant registered after (fallback)

**dispatch_attend:**
- Now takes `tile: TileImpl` parameter
- Tile-first dispatch: WMMA tiles → dedicated arms, `None` → existing key-only arms
- `run_attention` threads `is_tree` into `ShapeInfo` and passes `variant.tile`

The inline WMMA ladder in `launch_asym_flash_batched` is preserved — it still fires
when called directly. The dispatch path bypasses it by specifying the WMMA kernel
name directly.

**Verification (gfx1151):**
- coherence-gate.sh: 5/5, no hard errors
- coherence-gate-dflash.sh: 4/4, no hard errors
- 115/115 dispatch tests, 58/58 dispatch-tests

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

### Status: DONE (826b143c)

---

## Post-C5 finding fixes

### F-2: Q8 batched kernel swap documentation

**Context:** Ship 3.2 unified Q8 batched prefill onto `AttnQ8_0KvBatchedMasked`,
which dispatches to the P-1 tiled kernel (two-pass tile + online-softmax
reduce). On master, Q8 batched used `attention_q8_0_kv_batched_masked`
(single-pass LDS-staged softmax) for max_ctx_len ≤ 15000, and only
switched to the tiled kernel past 15k.

The dispatch migration eliminated the ≤15k path entirely — ALL Q8 batched
prefill now goes through the P-1 tiled kernel. Different reduction order
means the output is not byte-identical to master's ≤15k path.

**Numerical verification:** NIAH 32k needle-in-haystack passed on gfx1151.
No regression in coherence-gate or dflash-gate across C2–C5.

**Action taken:** Added explicit NOTE(F2) comment at the registration in
`attention_table.rs` documenting the algorithm swap and its implications.

**Acceptance:** This is a deliberate numeric change gated on E2E task output,
not byte-parity. Future regression investigations should compare against
master's two-path Q8, not this single-path kernel.
