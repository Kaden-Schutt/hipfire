# Dispatch unified architecture — open work items

Tracked findings from the tri-code review (GLM-5 / Gemini 2.5 Pro / Claude Opus 4.8),
plus architectural items surfaced during Ship 3.3 implementation. Ordered by severity.

Source reviews: `findings/dispatch_3.x_code_rev_{glm5,gemini,claude}.md`

---

## Hardware verification

### F-3 · MED — gfx1201 (RDNA4) verification still needed

gfx1100 verified 2026-06-06 by Kevin Read (unverbraucht). Results:

| Check | gfx1100 result |
|-------|----------------|
| `hipfire-dispatch-tests` | ✅ pass |
| `hipfire-dispatch` (internal) | ✅ pass |
| Coherence battery (short) | ✅ no hard errors; 11/17 ran (6 skipped: missing models) |
| Decode A/B (±5%) | ✅ −0.7% to −1.7% (neutral) |
| Prefill A/B (512 tok) | ⚠️ −5.8% to −12.4% (arch-agnostic, amortizes at long ctx) |
| Prefill A/B (16384 tok) | ⚠️ −3.8% to −5.6% (converges but does not reverse; gfx906 reversal is tiled-kernel-specific) |
| Correctness (temp=0) | ✅ benign argmax flips (FP accumulation order) |
| Binary md5s | bench_A=`0bdde243` bench_B=`99175bc` |

Pre-existing quality issues confirmed on master (not dispatch regressions):
`qwen3.5-9b.mq3` attractor loop in thinking, `qwen3.5-9b.mq4-lloyd` incoherent.

Full report: `/tmp/dispatch_validation_gfx1100.md`
Coherence report: `/tmp/coherence-20260606-211357.md`
Issue comment: [#397 (comment 4639841863)](https://github.com/Kaden-Schutt/hipfire/issues/397#issuecomment-4639841863)
Long-context follow-up: [#397 (comment 4639924237)](https://github.com/Kaden-Schutt/hipfire/issues/397#issuecomment-4639924237)

**Remaining:** gfx1201 (RDNA4) coverage + coherence + A/B. Need hardware.
**Blocks:** Phase 0.6 sign-off (gfx1201 only).

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

## Ship 4.1 — MoE family owns resolution (open verification items)

### SF-4.1.1 · HIGH — gfx1201 cross-arch verification

Ship 4.1 verified on gfx1151 only (gfx11-family). Phase 0.6 requires gfx1201 (RDNA4).

**Action:** Run coherence-gate.sh --full on gfx1201. Byte-identical token IDs vs gfx1151.
**Blocks:** Phase 0.6 sign-off. **Assignee: Kaden.**

### SF-4.1.2 · MED — k≠8 CPU-top-K fallback fixture

A3B is k=8 only — it exercises only the GPU indexed top-K path. The CPU-top-K
fallback (`run_moe_decode_cpu_fallback`) was re-plumbed (ctx threaded through
per-expert loop) but has no on-GPU fixture to validate it.

**Action:** Procure or build a k≠8 MoE model (e.g. k=4 or k=16 variant), or a
non-MQ4/MQ6/Paro routed-expert model, and run through coherence-gate.sh.
**Doc:** Document the residual until a fixture is available.
**Assignee: Kaden** (model procurement); Kevin can validate once fixture exists.

### SF-4.1.3 · LOW — A3B DFlash draft model missing locally

The pinned A3B MoE DFlash fixture (AGENTS.md §"Pinned A3B MoE DFlash fixtures")
requires `qwen36-35b-a3b-dflash-mq4.hfq` (md5 `8254bbe1`). This file is not present
locally; the coherence-gate.sh DFlash gate is skipped.

**Action:** Acquire the draft or document the DFlash coverage gap permanently.
A3B AR (`coherence-gate.sh --full`) covers the MoE decode path without DFlash.
**Assignee: Kaden** (has HF upload access).

### SF-4.1.4 · LOW — coherence-gate.sh rebuild trigger incomplete for dispatch crate

The coherence-gate.sh timestamp check only covered qwen35.rs, llama.rs, hfq.rs,
daemon.rs, dispatch.rs, and deepseek4 sources. Dispatch-crate files (moe.rs,
pipeline/mod.rs, steps.rs, gemv.rs, attention.rs, fused_qkv.rs) were not in the
trigger list — incremental builds could go stale when only dispatch code changed.

**Fixed in Ship 4.1:** Dispatch-crate files added to trigger list.
**Symptom:** A3B MoE model produced gibberish output after struct layout changes
until a `cargo clean` + rebuild was performed.

### SF-4.2.1 · HIGH — MQ6 grouped-WMMA kernel for gfx11

`gemm_hfq6g256_moe_grouped_wmma` exists for gfx12 only. On gfx11/gfx1151,
MQ6 MoE batched prefill falls back to Path 1 (indexed batched GEMV) via
`MoePrefillResolution` guard. The gfx11 variant of this kernel needs to be
implemented to unlock Path 2 for MQ6 A3B models on RDNA3/RDNA3.5.

**Action:** Port the gfx12 MQ6 grouped-WMMA kernel to gfx11 (WMMA wave32
builtin port — follow `gemm_hfq4g256_moe_grouped_wmma_k2` as template).
**Assignee: Kaden.**

### SF-4.2.2 · HIGH — gfx1201 verification for MQ6 grouped prefill

`qwen3.6-35b-a3b.mq4` has MQ6 FFN weights. The MQ6 grouped-WMMA path (Path 2)
was never exercised in batched prefill because `mq6_batched_admit_enabled_from_env`
defaults false on gfx11. On gfx12 it defaults true — but no gfx12 hardware has
validated this path end-to-end.

**Action:** Run `HIPFIRE_MOE_MQ6_ADMIT=1` + `coherence-gate.sh --full` on gfx1201
with the A3B model. Verify byte-parity vs gfx1151 Path 1 (indexed batched GEMV).
**Assignee: Kaden** (has gfx1201 hardware).
**Blocks:** Phase 0.6 sign-off for MQ6 MoE prefill.

### SF-4.1.5 · LOW — batch_size guard unit test (GPU-gated)

The `batch_size != 1` runtime guard in `run_moe_decode` is not unit-testable
without a GPU (MoeParams requires GpuTensor references). The guard fires at the
top before any GPU access, but constructing a minimal MoeParams requires unsafe
pointer fabrication.

**Action:** Add a unit test when a GPU-free mock DispatchCtx + scratch pattern
is available, or validate via a one-shot GPU test (pass batch_size=2, expect
UnsupportedVariant error).
**Mitigation:** Guard mirrors the identical bias-aware guard pattern; coherence-gate
ensures the guard never fires in real operation.

---

## #397 Ship 6(g) — DISPATCH_OLD/NEW selector removal (status)

The temporary `HIPFIRE_DISPATCH_OLD` / `HIPFIRE_DISPATCH_NEW` byte-parity
selector (planned for deletion in Ship 6 per `dispatch-phase0-decisions.md`)
is **removed** — grep across the repo finds zero live-code hits for
`DISPATCH_OLD` / `DISPATCH_NEW` / `HIPFIRE_DISPATCH_OLD` / `HIPFIRE_DISPATCH_NEW`
(only design-discussion mentions remain). Ship 6(g) satisfied.

---

*Last updated: 2026-06-06 (post Ship 1.4b + 4c + 6g, tracking #397).*
