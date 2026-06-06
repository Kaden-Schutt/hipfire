# Dispatch unified architecture — open work items

Tracked findings from the tri-code review (GLM-5 / Gemini 2.5 Pro / Claude Opus 4.8),
plus architectural items surfaced during Ship 3.3 implementation. Ordered by severity.

Source reviews: `findings/dispatch_3.x_code_rev_{glm5,gemini,claude}.md`

---

## Hardware verification

### F-3 · HIGH — Cross-arch verification needed (gfx1100 + gfx1201)

All Ship 3.3 verification ran on gfx1151 only. Phase 0.6 requires gfx1100 (primary
deploy target) and gfx1201 (RDNA4).

**Action:** Run coverage + coherence battery on gfx1100 and gfx1201. No code change.
**Blocks:** Phase 0.6 sign-off.

---

## Integration work (follow-up ships)

### F-8 · MED — Multi-GPU path migration (`forward_scratch_layers_multi`)

38 direct `gpu.kv_cache_write_*` / `gpu.attention_*` calls in an inline match tree
in `forward_scratch_layers_multi`. No `KvTierPlan` coverage, no LDS-overflow fix,
divergent Q8 heuristic from the single-GPU path.

**Action:** Mirror the single-GPU dispatch migration (Ship 3.3 C4 pattern) into the
multi-GPU path. Separate ship — orthogonal to dispatch unification.
**Scope:** ~38 call sites in `crates/hipfire-arch-qwen35/src/qwen35.rs`.

### Multi-GPU / MoE attention dispatch

Ship 3.3 only covers single-GPU paths. The multi-GPU band-mode forward pass has its
own attention ladder that needs dispatch migration independently.

**Action:** Ship 4 / later. Depends on F-8 completion.

### qwen2 text decode/prefill attention

The qwen2 text-side trait impl delegates to `hipfire-arch-qwen2` which has its own
inline attention ladder. Ship 3.3 migrated qwen35 + dots-ocr + llama + dflash but
not qwen2 text.

**Action:** Ship 3.1b / 3.2 llama-family follow-up.

---

## Kernel work

### F-19 · LOW — Tile kernel OOB Q read when `head_dim < 256`

The WMMA-FA tile kernels read `head_dim` elements from Q unconditionally. If a model
has `head_dim < 256`, the read extends past the allocated tensor → potential OOB.

**Action:** Add a `head_dim` guard in the tile kernels (clamp or bounds-check). Requires
kernel changes + careful testing. Tracked for kernel cleanup pass.

### F-16 · LOW — Q8 batched write is 2 launches vs fused 1

All other quant tiers use a fused K+V write kernel. Q8 uses two separate launches
(`kv_cache_write_q8_0` called twice). Inherent to Q8 kernel API — no fused variant exists.

**Action:** Would need a new fused Q8 write kernel. Low priority — perf impact is
minimal (2 cheap launches vs 1). Documented as known asymmetry.

### WMMA-FA for fwht4 / asym3 / fwht3 batched-masked

Only asym4 has a WMMA tile today (via `Asym4WmmaTile`). The fwht4, asym3, and fwht3
batched-masked paths use scalar kernels.

**Action:** New kernels (future). The scalar paths are correct; WMMA would improve
prefill throughput.

### 2-bit tree-verify kernel (the 3.2 `UnsupportedTreeTier` gap)

`batched_keys` returns `Err(UnsupportedTreeTier)` for asym2 + tree-verify because no
`_batched_masked` variant exists. The F-4 guard forces per-token fallback.

**Action:** Future kernel work to add a 2-bit tree-verify masked variant.

---

## Cosmetic / design

### F-18 · LOW — `AttnQ8_0KvBatchedMasked` naming inconsistency

Inconsistent with other `_BatchedMasked` keys (e.g. `AttnFlashAsym4BatchedMasked`).
The `Q8_0Kv` infix breaks the `{tier}BatchedMasked` pattern.

**Action:** Cosmetic rename to `AttnFlashQ8_0BatchedMasked`. Use `pub use OldName = NewName`
alias for one release cycle to avoid breaking consumers. Low priority.

### F-14 · LOW — `TileImpl` in shared `types.rs`

30+ sites specify `tile: TileImpl::None`. Could use `#[default]` + struct-update
syntax (`..Default::default()`) or wrap in `Option<>`.

**Action:** Design cleanup. No functional impact. Consider when adding Ship 4 tile
variants (append-only enum discipline at Ship 3 ⊥ Ship 4 boundary).

### F-15 · LOW — `HeadDimIn(&'static [usize])` forces compile-time

`ShapePredicate::HeadDimIn` takes `&'static [usize]`, requiring compile-time known
head dims. Fine for init-time registration but limits dynamic model loading.

**Action:** API design — acceptable for now. Revisit if dynamic head_dim loading
becomes a requirement.

### F-28 — `attention_dflash_*` naming collision

GPU method names like `attention_dflash_f32` conflate the DFlash spec-decode project
with the generic tiled online-softmax algorithm family. A rename (e.g. `attention_tiled_f32`)
would resolve the ambiguity.

**Action:** Future cleanup. Noted as TODO in `attention.rs` header. Low priority —
no functional impact.

### Priority field in `KernelVariant`

Registration-order-is-priority works but is fragile. An explicit `priority: u32` field
would make the ordering invariant visible and catch accidental reorderings.

**Action:** Future improvement. Current system works — all tables have `PRIORITY ORDER`
comments and the completeness tests catch missing arms.

---

## Closed in Ship 3.3

| Finding | Status | Commit |
|---|---|---|
| F-1 WMMA grid shape | ✅ FIXED | Bug-fix round |
| F-2 Q8 kernel swap docs | ✅ FIXED | `53795fbe` |
| F-4 KV-tier guard | ✅ FIXED | `53795fbe` |
| F-5 Tile completeness test | ✅ FIXED | Bug-fix round |
| F-6 Reverse completeness | ✅ FIXED | Bug-fix round |
| F-7 Coverage gate | ✅ FIXED | Bug-fix round |
| F-9 DispatchCtx hoisting | ✅ FIXED | `53795fbe` |
| F-10 ShapeInfo.m | ✅ FIXED | Bug-fix round |
| F-11 UnsupportedTreeTier batch_size | ✅ FIXED | Bug-fix round |
| F-12 F32+batched comment | ✅ FIXED | Bug-fix round |
| F-13 Unused binding | ✅ FIXED | Bug-fix round |
| F-17 is_boundary comment | ✅ FIXED | Bug-fix round |
| F-20 Q8 heuristic | ✅ FIXED | Bug-fix round |
| F-21 Trailing newline | ✅ FIXED | Bug-fix round |
| F-22 kv_write tile-oblivious | ✅ VERIFIED | C5 sweep |
| F-23 WMMA draft rung warning | ✅ DOCUMENTED | C3 commit |
| F-24 Full-attention completeness | ✅ TESTED | C5 sweep |
| F-28 Naming collision TODO | ✅ NOTED | C5 sweep |

---

*Last updated: 2026-06-06 (post Ship 3.3 close, tracking #397).*
