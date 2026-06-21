# Ship 3.3 — WMMA tile attention variants + vision/dflash + llama liveness (Phase C)

**Branch:** `integration/dispatch-unification`
**Tracking:** #397 (Ship 3 · Attention + KV cache unification, Phase C)
**Owner:** Kevin / unverbraucht (Ship 3 lane)
**Depends on:** **Ships 3.1 + 3.2 landed** — `AttentionFamily` with `run_attention(plan, io)`,
`dispatch_kv_write`/`dispatch_attend` split, `KvTierPlan`, `Step::Attend`, the batched keys +
`ShapePredicate::BatchGt`/`BatchEq`, the `#382` no-LDS-cap Q8 kernel, the dispatch-arm
completeness test + `(op × dtype × arch × batch)` coverage gate.
**Phase 0 contracts in force:** 0.4 (the gfx12 escape clause — *"reintroduce a gfx12-only
predicate only in the same change as a real gfx12-only kernel"* — invoked here), 0.5 (family
API checklist), 0.6 (same-binary byte-parity, RDNA3 **and** RDNA4 probe).

**Adversarial reviews folded:** `findings/dispatch_3.3_plan_rev_glm5.md` (GLM-5, Gemini, ds4),
28 findings (5 Critical, 6 High, 9 Medium, 8 Info). Findings `[Fn]` mark where each
is addressed. This revision resolves all 28.

**⚠ C0 is the rebase-onto-3.2 commit.** Both 3.2 and 3.3 touch `dispatch_attention`/
`dispatch_attend`. C0 must land on the 3.2 tip, not on the pre-3.2 branch. [F27]

**Goal:** the WMMA-accelerated tile attention kernels join the dispatch surface, selected by
`(arch × shape)` at resolve time rather than env vars and inline `if`-ladders. Three deliverables
from the roadmap: **(1)** register the live WMMA tile variants with `shape_gate`; **(2)** migrate
the **qwen2 dots-ocr** vision attention pattern; **(3)** audit **llama** legacy KV-mode liveness
and register only the live keys. After 3.3, a faster attention kernel for a given shape/arch is a
*registry entry* (priority-ordered variant), not an inline `if HIPFIRE_WMMA_FA && head_dim==128…`.

---

## Grounded starting state (verified on the branch tip)

### The two WMMA attention families (distinct modalities)

**(a) WMMA-FA acceleration of the quantized causal prefill** — `attention_flash_asym4_wmma_
tile_batched[.gfx12]` (`kernels.rs:2507/2509`). A WMMA tiling of the **asym4 + Q8-V** batched-
masked attend (the 3.2 path). Selected today **inline + env-gated** inside rdna-compute
(`attention.rs:2557-2589`):

```rust
let wmma_fa_kernel = if arch.has_wmma_w32_gfx12() { Some("…_gfx12") }
                     else if arch.has_wmma_w32()  { Some("…") } else { None };
let wmma_ok = is_wmma_fa_enabled()        // HIPFIRE_WMMA_FA (attention.rs:17)
    && wmma_fa_kernel.is_some()
    && (head_dim == 128 || head_dim == 256)
    && tree_bias.is_none()                // ⚠ DISABLED for tree-verify (sparse mask)
    && v_mode_bits == V_MODE_Q8
    && tile_func_name == "attention_flash_asym4_tile_batched"
    && batch_size >= wmma_fa_min_batch()  // HIPFIRE_WMMA_FA_MIN_BATCH (default 16)
    && batch_size % WMMA_BLOCK_M == 0 && sub_batch % WMMA_BLOCK_M == 0;
```

Note the **inline gates not captured by the plan's original D5**: `head_dim == 128 || head_dim == 256`
(the plan only had `HeadDimEq(128)` — **this excludes qwen35-27b hd=256** [F2]), `v_mode_bits ==
V_MODE_Q8` [F4], `batch_size % WMMA_BLOCK_M == 0` (divisibility, not just minimum), and
`sub_batch % WMMA_BLOCK_M == 0`.

**(b) Non-causal / causal-masked F16-K/V and F32-K/V tile attention** — the
`attention_dflash_wmma_*` family (~17 kernels, `kernels.rs:2724-2844`). **Different modality:**
full or causal self-attention, **no KV cache**, **no write-then-attend**, with **two K/V dtype
sub-families** [F15]:

- **F16-K/V rungs** (v5, n128): caller allocates and casts F32→F16 before the call.
- **F32-K/V rungs** (m32, wmma_f32, scalar): caller passes F32 tensors directly.

Used by:
- **dots-ocr vision tower** — inline ladder `dots_ocr.rs:1121-1158`:
  ```
  v5(hd128,n≥64)       → F16 K/V  (HasWmmaGfx12 || HasWmma)
  n128_f16kv(hd128,n≥32) → F16 K/V  (HasWmma)
  m32(hd%16,h≤128,n≥32)  → F32 K/V  (HasWmma)
  wmma_f32(hd%16)       → F32 K/V  (HasWmma)
  dflash_f32(scalar)    → F32 K/V  (Always)
  ```
  Note: **`m32` requires `head_dim % 16 == 0`**, omitted from the original D5 [F16].
  **n128 uses F16 K/V** — the F16/F32 split is per-rung, not per-key [F15].
- **DFlash draft decoder** — `dflash.rs:1460` `attention_dflash_f32` (scalar; **F32 K/V**;
  non-causal cross-attention over target-layer extractions). [F15]

The **causal** variant: `v3_causal_f32` (+ `.gfx12`) and scalar `attention_causal_batched`
(NOT `attention_dflash_f32` — which is non-causal). [F14]

### ⚠ Most of the 17 tile kernels are dead benchmark variants

Production call sites use **only**: `v5_f32` (+ `.gfx12`), `n128_f16kv_f32`, `m32_f32`,
`wmma_f32`, `dflash_f32` (scalar), and the causal pair `v3_causal_f32` (+ `.gfx12`) plus the
causal scalar floor `attention_causal_batched`. The `v2/v3/v4/v6/v7/v7b/n64*` kernels are **A/B
experiments** (only v5 won); their only callers are `examples/` benches. **Register only the
live ~6 + 2 gfx12 + 2 causal variants.** [F6]

### Registry mechanics (the constraint that shapes the design)

- `KernelRegistry` is `HashMap<KernelKey, Vec<KernelVariant>>`; `resolve()` **loops variants in
  registration order, returns the first passing `(arch × shape)`** (`tables/mod.rs:41–54`).
  Registration order IS the priority order — `resolve()` picks the first match. [F18]
- **`KernelVariant` carries no implementation discriminator** (`types.rs:295`): `{ key,
  arch_required, shape_gate, steps, has_awq }`. `dispatch_attend` matches on **`key`** — so two
  variants under one key are indistinguishable to dispatch. **This is the gap 3.3 must close.**
- `ShapePredicate` = `BatchGt | HeadDimEq | MLt` (`types.rs:286`) — **single** `Option` per
  variant, **no AND-composition**, no `BatchGe`. Tile selection needs `head_dim==128 AND
  batch≥64`. **Use `ShapePredicate::And(&[ShapePredicate])` wrapper** rather than changing the
  `Option<ShapePredicate>` field type — avoids touching all 6 table files (~100+ registrations)
  [F6]. Only new attention entries that need AND-composition use `Some(And(&[...]))`.
- `ArchPredicate` = `Always | HasWmma | HasDp4a | HasSdot4 | HasMmq | HasCdna3LdsGemv | GemvDp4a`
  (`types.rs:272`) — **no gfx12 discriminator** (Phase 0.4 collapsed `HasWmmaW32Gfx12` → `HasWmma`).
  Real `.gfx12.hip` tile siblings exist → 3.3 reintroduces `HasWmmaGfx12`. [D3]

### llama KV-mode liveness (audit done)

llama decode attention (`arch.rs:197-290`) has 7 KV branches. Liveness via which
`KvCache::new_gpu_*` constructors are actually called:

| Mode | Kernel | Live? | KernelKey today |
|---|---|---|---|
| HFQ4 | `attention_hfq4_kv` | **LIVE** (⚠ constructors never called externally [F10]) | ❌ → add `AttnHfq4Kv` + `KvWriteHfq4` |
| Q4 | `attention_q4kv` | **LIVE** (⚠ constructors never called externally [F10]) | ❌ → add `AttnQ4Kv` + `KvWriteQ4` |
| Q8_0 | `attention_q8_0_kv` | LIVE | ✅ `AttnQ8_0Kv` (3.1) |
| F32 | `attention_f32` | LIVE | ✅ `AttnF32` |
| Asym2/3/4 | `attention_flash_asym*` | LIVE | ✅ (3.1) |
| **HFQ8** | `attention_hfq8_kv` | **DEAD** (`new_gpu_hfq8` never called) | **do NOT register** |
| **INT8-F16** | `attention_int8c_f16_kv` | **DEAD** (`new_gpu_int8c` never called) | **do NOT register** |

`KvTierInputs` currently carries `quant_asym{4,3,2}`, `quant_q8`, `quant_fwht` — **no
`quant_hfq4` or `quant_q4`**. C4 must add these so the `derive()` lattice can produce the new
keys. [F19]

---

## Design decisions (revised)

### D1 · Variant discriminator — dispatch by resolved variant, not by key

Add an implementation tag to `KernelVariant` so `resolve()` can return one of several
shape/arch-gated implementations under one key, and the family dispatches on it:

```rust
pub struct KernelVariant {
    pub key: KernelKey,
    pub arch_required: ArchPredicate,
    pub shape_gate: Option<ShapePredicate>,   // D2: And(&[...]) for compound gates
    pub steps: &'static [PipelineOp],
    pub has_awq: bool,
    pub tile: TileImpl,                         // NEW discriminator
}
pub enum TileImpl {
    None,   // existing key-based dispatch — 3.1/3.2 arms unchanged
    // WMMA-FA (quantized causal prefill)
    Asym4WmmaTile, Asym4WmmaTileGfx12,
    // Vision/dflash F16-K/V rungs
    DflashV5, DflashV5Gfx12, DflashN128,
    // Vision/dflash F32-K/V rungs
    DflashM32, DflashWmmaF32,
    // Causal (F16-K/V rungs)
    DflashV3Causal, DflashV3CausalGfx12,
    // Scalar floors — separate for causal vs non-causal [F14]
    DflashScalar,          // non-causal floor → gpu.attention_dflash_f32
    CausalScalar,           // causal floor → gpu.attention_causal_batched
}
```

- `TileImpl::None` ⇒ existing key-based dispatch (3.1/3.2 arms **byte-identical**).
- `run_attention` / `run_full_attention` resolve to a `&KernelVariant` and pass
  `variant.tile` to dispatch. `dispatch_attend` uses **tile-first dispatch** [F21]:
  ```rust
  match tile {
      TileImpl::None => match key { /* existing arms — untouched */ },
      TileImpl::Asym4WmmaTile => { /* WMMA dispatch */ },
      TileImpl::DflashV5 => { /* v5 dispatch */ },
      // ...
  }
  ```
  This separates tile variants from key variants cleanly. The completeness test is: for each
  tile variant, verify all reachable `(key, tile)` pairs have arms. For `TileImpl::None`, the
  existing key-only test covers it.
- `AttentionVariant { Decode, Prefill, FlashDecode, FlashPrefill }` in `types.rs` is dead.
  Remove it in C0 to avoid confusion with `TileImpl`. [F20]
- **Coupling note:** `TileImpl` is attention-specific. Other families pass `TileImpl::None`
  (use `#[default]` for convenience). Future families that need multi-variant dispatch should
  consider a generic type parameter or opaque index rather than extending this enum. [F17]

### D2 · `ShapePredicate::And` for AND-composition + new predicates

Tile selection is conjunctive (`head_dim==128 && batch≥64`). Rather than changing
`shape_gate: Option<ShapePredicate>` to `Option<&'static [ShapePredicate]>` (which would touch
~100 registrations across 6 table files [F6]), add `ShapePredicate::And(&'static [ShapePredicate])`
as a new variant. AND-composed gates use `Some(And(&[HeadDimEq(128), BatchGe(64)]))`. Existing
single-predicate registrations stay `Some(BatchGt(1))` — **no mechanical conversion needed**.
[F6]

New predicates (promoted from D5 parenthetical to C0 main text [F9]):

| Predicate | Semantics | Needed by |
|-----------|-----------|------------|
| `BatchGe(usize)` | `batch_size ≥ n` | WMMA-FA, v5, n128, m32 minimum batch |
| `HeadDimLe(usize)` | `head_dim ≤ n` | m32 upper bound |
| `HeadDimMultipleOf(usize)` | `head_dim % n == 0` | m32, wmma_f32 head-divisibility [F16] |
| `HeadDimIn(&[usize])` | `head_dim ∈ {…}` | WMMA-FA hd∈{128,256} [F2] |
| `IsTree(bool)` | `is_tree == bool` | WMMA-FA exclusion for tree-verify [F3] |

`HeadDimMultipleOf` is named explicitly rather than `HeadDimMod` for clarity [F9]. `HeadDimLe`
uses `≤` semantics, matching the inline code's `head_dim <= 128`.

### D3 · Reintroduce `HasWmmaGfx12` — *with* the kernels it gates (Phase 0.4 clause)

The v5/v3_causal/asym4 gfx12 siblings are **real gfx12-only kernels**, so Phase 0.4's escape
clause applies. Reintroduce `ArchPredicate::HasWmmaGfx12` (backed by
`ArchCaps::has_wmma_w32_gfx12()`) **in the same commit** that registers the gfx12 variants.
Register via **priority order** [F18]: gfx12 variant (`HasWmmaGfx12`) first, gfx11 (`HasWmma`)
second, scalar (`Always`) last — `resolve()` returns the first passing. **Document priority
order with `// PRIORITY ORDER: gfx12 → gfx11 → scalar — DO NOT REORDER` comments above each
ladder in `attention_table.rs`**.

The coverage gate gains a gfx12 row asserting the gfx12 variant resolves on `gfx1201` and the
gfx11 variant on `gfx1100` — catches the dead-gate class Phase 0.4 warned about.

The `DflashV5`/`DflashV5Gfx12` split is **redundant** (v5 internally dispatches gfx12 vs gfx11 in
the kernel function) [F5]. Both `ArchPredicate` and internal dispatch agree; accept the redundancy
for documentary clarity.

### D4 · Vision/full attention is a separate entry path (not `KvTierPlan`) — 4-key split

The `attention_dflash_wmma_*` modality has no KV cache, no write-then-attend, no quant tier — so
`run_attention(plan, io)` / `KvTierPlan` **do not apply**. Add a sibling entry on the family:

```rust
pub fn run_full_attention(&self, ctx, gpu, p: &FullAttnParams) -> Result<(), DispatchError>;
pub struct FullAttnParams<'a> {
    pub kind: KernelKey,  // AttnFullF16 | AttnFullF32 | AttnFullF16Causal | AttnFullF32Causal
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,  // dtype determined by key: F16 for AttnFullF16*, F32 for AttnFullF32*
    pub v: &'a GpuTensor,  // same
    pub out: &'a GpuTensor,
    pub batch: usize, pub seq_len: usize, pub n_heads: usize,
    pub n_kv_heads: usize, pub head_dim: usize,
    // No `causal: bool` [F8] — kind determines causal vs non-causal
    // No `k_f16`/`v_f16` [F15] — dtype is implicit in the key
    // No `v_mode_bits` [F25] — no KV cache
}
```

**Four keys** (not two) to handle the F16/F32 K/V dtype split [F15] and the causal/non-causal
scalar floor split [F14]:

| Key | K/V dtype | Causal? | Variants |
|-----|-----------|---------|----------|
| `AttnFullF16` | F16 | No | V5Gfx12, V5, N128, **Scalar(zero-cost cast floor)** |
| `AttnFullF32` | F32 | No | M32, WmmaF32, DflashScalar |
| `AttnFullF16Causal` | F16 | Yes | V3CausalGfx12, V3Causal, **CausalScalar(zero-cost cast floor)** |
| `AttnFullF32Causal` | F32 | Yes | CausalScalar |

**Why 4 keys, not 2 [F15]:** The dots-ocr ladder has F16-K/V rungs (v5, n128) and F32-K/V rungs
(m32, wmma_f32, scalar) under a single if-else chain. Under dispatch, the caller doesn't know
which variant will resolve, so it can't decide whether to F16-cast. The 4-key split makes dtype
implicit: `AttnFullF16` always receives F16 K/V, `AttnFullF32` always receives F32 K/V. The
dots-ocr caller tries `resolve(AttnFullF16, shape)` first (cast K and V to F16); if
`MissingImpl`, tries `resolve(AttnFullF32, shape)` (no cast). This mirrors the inline ladder's
if-else structure exactly.

**Why `CausalScalar` is not `DflashScalar` [F14]:** `attention_dflash_f32` is a **non-causal**
online-softmax kernel. The causal scalar floor must be `gpu.attention_causal_batched`. These are
different kernels and cannot share a TileImpl.

`run_full_attention` builds `ShapeInfo { batch: n_patches, head_dim, m: seq_len }` [F7]
(`batch` is `seq_len` / number of patches for vision, not a token batch — document on the field).
resolves under the key, dispatches `variant.tile`. The `ResourceManager` stays empty (Phase 0.2)
— no persistent arch-specific scratch in dispatch.

### D5 · Four logical keys for the tile family, priority-ordered variants

**`AttnFullF16`** (non-causal, F16 K/V) — the F16-K/V rungs of the dots-ocr ladder:

| Priority | TileImpl | Arch gate | Shape gate (`And(&[...])`) | Note |
|----------|----------|-----------|----------------------------|------|
| 1 | `DflashV5Gfx12` | `HasWmmaGfx12` | `[HeadDimEq(128), BatchGe(64)]` | F16 K/V |
| 2 | `DflashV5` | `HasWmma` | `[HeadDimEq(128), BatchGe(64)]` | F16 K/V |
| 3 | `DflashN128` | `HasWmma` | `[HeadDimEq(128), BatchGe(32)]` | F16 K/V |
| 4 | *no Scalar floor for F16* | | | F16-F32 boundary: fall to `AttnFullF32` |

**`AttnFullF32`** (non-causal, F32 K/V) — the F32-K/V rungs + scalar floor:

| Priority | TileImpl | Arch gate | Shape gate | Note |
|----------|----------|-----------|------------|------|
| 1 | `DflashM32` | `HasWmma` | `And(&[HeadDimMultipleOf(16), HeadDimLe(128), BatchGe(32)])` | F32 K/V [F16] |
| 2 | `DflashWmmaF32` | `HasWmma` | `And(&[HeadDimMultipleOf(16)])` | F32 K/V |
| 3 | `DflashScalar` | `Always` | `None` | → `gpu.attention_dflash_f32` |

**`AttnFullF16Causal`** (causal, F16 K/V):

| Priority | TileImpl | Arch gate | Shape gate | Note |
|----------|----------|-----------|------------|------|
| 1 | `DflashV3CausalGfx12` | `HasWmmaGfx12` | `[HeadDimEq(128)]` | F16 K/V |
| 2 | `DflashV3Causal` | `HasWmma` | `[HeadDimEq(128)]` | F16 K/V |
| 3 | *no CausalScalar for F16* | | | Fall to `AttnFullF32Causal` |

**`AttnFullF32Causal`** (causal, F32 K/V):

| Priority | TileImpl | Arch gate | Shape gate | Note |
|----------|----------|-----------|------------|------|
| 1 | `CausalScalar` | `Always` | `None` | → `gpu.attention_causal_batched` [F14] |

**WMMA-FA** under `AttnFlashAsym4BatchedMasked` (3.2 key) — additional variants:

| Priority | TileImpl | Arch gate | Shape gate | Note |
|----------|----------|-----------|------------|------|
| 1 | `Asym4WmmaTileGfx12` | `HasWmmaGfx12` | `And(&[HeadDimIn(&[128,256]), BatchGe(WMMA_BLOCK_M), IsTree(false)])` | [F2, F3, F4] |
| 2 | `Asym4WmmaTile` | `HasWmma` | `And(&[HeadDimIn(&[128,256]), BatchGe(WMMA_BLOCK_M), IsTree(false)])` | [F2, F3] |
| 3 | *(existing scalar batched)* | `Always` | `Some(BatchGt(1))` | from 3.2 |

Notes on the WMMA-FA shape gate:
- `HeadDimIn(&[128, 256])` covers both qwen35-9b (hd=128) and qwen35-27b (hd=256) [F2].
- `IsTree(false)` excludes tree-verify paths at the registry level, not via a hidden
  dispatch-arm fallback [F3].
- `v_mode_bits == V_MODE_Q8` is a precondition of the `AttnFlashAsym4BatchedMasked` key
  (the Q8-masked path is the only one with a WMMA tile). Document this as a key-level
  invariant. Lloyd-V modes use different keys (`AttnFlashAsym4FwhtBatchedMasked` etc.) [F4].
- `batch % WMMA_BLOCK_M == 0` and `sub_batch % WMMA_BLOCK_M == 0` also apply but are
  checked in the dispatch arm since `ShapeInfo` doesn't carry `sub_batch`.

---

## Plan

> Recommended split (independent vertical cuts): **3.3a = C0+C1** (infra + WMMA-FA accel);
> **3.3b = C2+C3** (vision/dflash full-attention modality); **3.3c = C4** (llama liveness —
> folds the deferred 3.1b llama attention). C0 is the shared prerequisite for all three.
> **C0 must rebase onto the 3.2 tip** before landing. [F27]

### Commit C0 · Dispatch infra: variant discriminator + shape AND + gfx12 predicate + 4-key schema + dead enum removal (GPU-free)

1. `types.rs`:
   - Add `TileImpl` enum with `#[default]` on `None` [F12, F17]. Document that it's
     attention-specific and other families use `None` [F17].
   - Add `ShapePredicate::And(&'static [ShapePredicate])`, `BatchGe(usize)`,
     `HeadDimLe(usize)`, `HeadDimMultipleOf(usize)`, `HeadDimIn(&'static [usize])`,
     `IsTree(bool)` [F2, F3, F6, F9, F16]. Implementation in `eval()`: `And` iterates
     all predicates and ANDs them; `HeadDimMultipleOf(n)` checks `head_dim % n == 0`;
     `HeadDimIn(&[...])` checks containment; `IsTree(b)` checks shape's `is_tree` field.
   - Add `ShapeInfo.is_tree: bool` field (defaults to `false`) [F3].
   - Add `ArchPredicate::HasWmmaGfx12` backed by `ArchCaps::has_wmma_w32_gfx12()` [D3].
   - Add `KernelKey` variants: `AttnFullF16`, `AttnFullF32`, `AttnFullF16Causal`,
     `AttnFullF32Causal` [F15].
   - Add `TileImpl` variants as per D1 (including `CausalScalar` [F14]).
   - Remove `AttentionVariant` enum (dead since 3.1) [F20].
   - Document that `ShapeInfo.m` means `seq_len` for attention families (GEMV-centric
     original description) [F26].
2. `tables/mod.rs`: `resolve()` ANDs the `And(...)` predicate slice and evaluates the
   new `ShapePredicate` variants. `resolve()` must test `is_tree` when `IsTree` predicates
   are present.
3. `families/attention.rs`:
   - `dispatch_kv_write` stays **tile-oblivious** — no `tile` parameter needed (WMMA-FA
     only accelerates attend, not KV write) [F22].
   - `dispatch_attend` uses **tile-first dispatch** [F21]: `match tile { None => match key { … },
     Asym4WmmaTile => …, … }`.
   - `run_attention` passes `resolved.tile` to `dispatch_attend`.
   - `run_full_attention` + `FullAttnParams` (D4): resolve under the key, dispatch on
     `variant.tile`. **No `causal: bool`** (key determines causality) [F8]. **No `k_f16`/
     `v_f16`** (dtype is implicit in the key) [F15].
   - Add `DISPATCHED_FULL_ATTENTION_KEYS` completeness test constant for the 4 full-attention
     keys [F24].
4. Tests (GPU-free):
   - Completeness test asserts every `(key, tile)` pair has an arm [F21].
   - Coverage gate gains gfx11-vs-gfx12 resolution rows (dead-gate guard) [D3].
   - Shape-AND resolve tests including `And(&[HeadDimEq(128), BatchGe(64)])`.
   - `IsTree(false)` correctly excludes WMMA variants when `is_tree=true` [F3].
   - `HeadDimIn(&[128, 256])` resolves on hd=256 (qwen35-27b) [F2].
   - `HeadDimMultipleOf(16)` rejects hd=120 [F16].
   - Priority-order test: assert `resolve(gfx1201, shape_hd128_batch64) == DflashV5Gfx12`
     AND `resolve(gfx1100, shape_hd128_batch64) == DflashV5` [F18].
   - `CausalScalar` resolves on non-WMMA archs (gfx906, gfx1030) [F14].
   - `DflashScalar` resolves on non-WMMA archs (gfx906, gfx1030) [D-12].

**Verify:** `cargo test -p hipfire-dispatch -p hipfire-dispatch-tests`; **all 3.1/3.2 keys
resolve byte-identically** (tile=None path unchanged, shape_gate: no existing registrations use
`And`, existing `Some(BatchGt(1))` stays unchanged [F6]) — diff the resolved key for every
existing (arch×dtype) row before/after. **C0 must land on the 3.2 tip** [F27].

### Commit C1 · WMMA-FA acceleration of quantized prefill → registry variant (3.3a)

**Two-phase approach** [F1]: C1a registers WMMA variants for visibility and testing; C1b retires
the inline ladder.

**C1a (registry visibility — can land immediately after C0):** Register
`Asym4WmmaTileGfx12`/`Asym4WmmaTile` as variants under `AttnFlashAsym4BatchedMasked` with the
shape gates from D5. The inline `attention.rs:2557-2589` env-ladder **stays** — dispatch falls
through to the existing code when `tile != None`. This provides the coverage gate test surface
without changing any runtime behavior.

**C1b (ladder retirement — requires rdna-compute API split):** Add
`gpu.attention_flash_asym4_wmma_tile_batched(…)` as a public entry point. The inline ladder
retires — `dispatch_attend(TileImpl::Asym4WmmaTile, ...)` calls the new entry point directly.
`HIPFIRE_WMMA_FA`/`_MIN_BATCH` are retired or documented as registry overrides (resolve the
ambiguity: confirm whether anything depends on forcing WMMA off; if so, keep as a debug
override that gates the variant's arch predicate [F8 from original risks]).

**WMMA ≠ scalar bit-exact** (FP reduction reorder). Oracle: **byte-parity vs master *with
`HIPFIRE_WMMA_FA=1`*** (same kernel both sides), not vs scalar. Plus `coherence-gate.sh` +
`coherence-gate-dflash.sh` for the numeric change where WMMA fires by default. Verification
phrasing: "E2E task output, not raw byte-parity" for C2/C3 [F13].

`probe_commits.sh` ±1–3% on gfx1100 **and gfx1201**.

### Commit C2 · dots-ocr vision attention → `run_full_attention` (3.3b)

1. `attention_table.rs`: register the 4 ladders (D5) with `// PRIORITY ORDER:
   gfx12 → gfx11 → scalar — DO NOT REORDER` comments [F18].
2. `hipfire-arch-dots-ocr`: add `hipfire-dispatch` dep; replace the inline ladder
   (`dots_ocr.rs:1121-1158`). The caller code becomes:
   ```rust
   // Try F16-K/V path first (v5, n128)
   if let Ok(variant) = family.resolve(AttnFullF16, ctx, Some(&shape)) {
       let k_f16 = gpu.alloc_tensor(&[n_patches, h], F16)?;
       let v_f16 = gpu.alloc_tensor(&[n_patches, h], F16)?;
       gpu.cast_f32_to_f16(&k_buf, &k_f16)?;
       gpu.cast_f32_to_f16(&v_buf, &v_f16)?;
       family.run_full_attention(ctx, gpu, &FullAttnParams { kind: AttnFullF16, k: &k_f16, v: &v_f16, … })?;
       gpu.free_tensor(k_f16); gpu.free_tensor(v_f16);
   } else {
       // Fall back to F32-K/V path (m32, wmma_f32, scalar)
       let variant = family.resolve(AttnFullF32, ctx, Some(&shape))?;
       family.run_full_attention(ctx, gpu, &FullAttnParams { kind: AttnFullF32, k: &k_buf, v: &v_buf, … })?;
   }
   ```
   [F15: the try-F16-then-F32 pattern mirrors the inline ladder's if-else structure.]
3. The `cast_f32_to_f16` stays caller-side for the F16 path. The F32 path has no cast.

**Verify:** **dots-ocr E2E OCR match** (the validated oracle: F1 1.000, 13/13 text match) on
gfx1100 **and gfx1201** (exercises the gfx12 v5 sibling). NOT byte-parity — WMMA numerics
differ from scalar [F13]; the OCR task output is the gate. Confirm each ladder rung resolves on
its intended (arch×shape) via coverage gate. `DflashScalar` resolves on gfx906/gfx1030 [D-12].
`CausalScalar` resolves on gfx906/gfx1030 [F14].

### Commit C3 · DFlash draft decoder attention → `run_full_attention` (3.3b)

Migrate `dflash.rs:1460` (`attention_dflash_f32`, non-causal **cross-attention** [F11] over
target-layer extractions) onto `run_full_attention` under **`AttnFullF32`** [F15 — draft decoder
uses F32 K/V]. It takes the `DflashScalar` floor on all archs (no WMMA rung for draft today).

**⚠ NOT a "free upgrade."** Adding a WMMA rung for draft attention later would change spec-decode
numerics → draft logits → acceptance patterns. Any future WMMA draft rung requires a dedicated
`feedback_attention_precision` sweep with acceptance-rate tracking [F23].

**Verify:** `coherence-gate-dflash.sh` (draft attention feeds τ/acceptance); spec-decode τ +
tok/s parity vs master; **decoded-text eyeball** (a draft-attention numeric change is the
attractor-risk class).

### Commit C4 · llama legacy KV-mode liveness + registration (3.3c — folds deferred 3.1b llama)

1. Register the **2 live** missing keys: `AttnHfq4Kv`+`KvWriteHfq4`, `AttnQ4Kv`+`KvWriteQ4`
   (+ dispatch arms → `gpu.attention_hfq4_kv`/`attention_q4kv`/`kv_cache_write_hfq4`/`_q4`).
   Wire `k_scales`/`v_scales` if the HFQ4 path needs them (clarify: "if needed" is
   insufficient — confirm by inspecting the kernel signatures [F19]).
2. **Do NOT register** `attention_hfq8_kv` / `attention_int8c_f16_kv` — `new_gpu_hfq8` /
   `new_gpu_int8c` are never called. Mark with `// DEAD: no live constructor (audit 3.3) —
   do not migrate`.
3. Migrate llama decode attention (`arch.rs:197-290`) onto `run_attention` + `KvTierPlan` for
   the **live** modes. `KvTierInputs::derive` gains `quant_hfq4: bool` and `quant_q4: bool`
   flags. Extend `tiers_match` for HFQ4/Q4 pairs. Update the single-quant-flag `debug_assert`
   to include both new flags. [F19]

The HFQ4/Q4 constructors are "potentially live" (never called outside their definitions) [F10] —
byte-parity verification requires fixtures that may not yet exist. Create minimal test fixtures
or document the gap.

**Verify:** byte-parity vs master on **HFQ4 and Q4 llama** fixtures (gfx1100 + gfx1201);
`coherence-gate.sh`; coverage gate has HFQ4/Q4 rows; `KvTierInputs::derive` produces the
correct write/attend key pairs for HFQ4 and Q4 tiers; `tiers_match` extended; **asserts no key
exists for the dead HFQ8/INT8 modes**.

### Commit C5 · Verification sweep + env-gate retirement + cleanup

- [ ] Coverage golden incl. RDNA4: every `(key, tile)` resolves on its intended arch/shape.
      gfx12 siblings resolve on gfx1201, gfx11 on gfx1100, scalar everywhere [F18].
- [ ] `CausalScalar` resolves on gfx906/gfx1030 (non-WMMA archs) [F14].
- [ ] `DflashScalar` resolves on gfx906/gfx1030 (non-WMMA archs) [D-12].
- [ ] `DISPATCHED_FULL_ATTENTION_KEYS` covers `AttnFullF16`, `AttnFullF32`,
      `AttnFullF16Causal`, `AttnFullF32Causal` [F24].
- [ ] **Grep audit:** zero inline `gpu.attention_dflash_wmma_*` / `attention_dflash_f32` calls
      in `dots_ocr.rs` + `dflash.rs`; zero `is_wmma_fa_enabled()` ladder in the rdna-compute
      batched dispatch — all through the registry.
- [ ] `AttentionVariant` enum removed [F20].
- [ ] **Priority-order comments** above each variant ladder [F18].
- [ ] `dispatch_kv_write` confirmed tile-oblivious (no `tile` parameter) [F22].
- [ ] Dead WMMA benchmark variants confirmed unregistered.
- [ ] Dead llama HFQ8/INT8 branches marked or removed.
- [ ] Dev-log every fixture (model + shape + arch + prompt/image md5 + binary md5).
- [ ] `attention_dflash_*` naming collision noted as future cleanup item [F28].

---

## Risks

1. **Variant-discriminator refactor touches the live 3.1/3.2 dispatch.** Mitigation:
   `TileImpl::None` preserves the exact existing key-arms (tile-first dispatch leaves the
   `None` branch unchanged); C0 diffs every existing (arch×dtype) resolution before/after
   for byte-identity before any tile variant is added.
2. **gfx12 predicate dead-gate** (Phase 0.4's exact warning). Mitigation: D3 lands
   `HasWmmaGfx12` *with* the gfx12 kernels + a coverage row asserting gfx12 resolves the gfx12
   variant and gfx11 the gfx11 variant.
3. **WMMA ≠ scalar bit-exact** (FP reorder). Mitigation: the oracle for C1 is byte-parity vs
   master *with the same WMMA env*, **not** vs scalar; coherence + dflash gates cover the
   numeric change where WMMA fires by default. C2/C3 gate on **task output** (OCR F1, DFlash
   coherence/τ, decoded-text eyeball), not bytes [F13].
4. **dots-ocr correctness regression** feeds OCR accuracy. Mitigation: C2's gate is the
   validated E2E (F1 1.000, 13/13), not a microbench.
5. **DFlash draft attention drift → spec-decode attractors.** Mitigation: C3 mandates
   `coherence-gate-dflash.sh` + decoded-text eyeball. **Adding WMMA rungs to draft attention
   later is NOT a free upgrade** — it requires a dedicated `feedback_attention_precision`
   sweep [F23].
6. **Registering dead kernels** (17 tile variants, llama HFQ8/INT8). Mitigation: liveness
   audit is explicit; coverage gate asserts the dead ones have no key.
7. **Shape-gate AND-composition uses `And(&[...])` wrapper** — no `shape_gate` type change.
   Mitigation: existing `Some(BatchGt(1))` registrations stay unchanged; only new attention
   entries use `Some(And(&[...]))`. The `eval()` AND-loop is tested in C0 [F6].
8. **`HIPFIRE_WMMA_FA` default semantics** (original risk #8). Resolve: C1a keeps the inline
   ladder; C1b retires it. If anything depends on forcing WMMA off, keep as a debug override
   that removes the WMMA variant from the registry (or forces `resolve` to skip it).
9. **Non-WMMA archs must keep their scalar floor** (D-12). `DflashScalar` and `CausalScalar`
   are `Always`-floored in their respective ladders [F14]. Coverage gate tests resolution on
   gfx906/gfx1030.
10. **4-key split means dots-ocr caller makes two resolve calls** (try F16, fall to F32).
    Mitigation: this mirrors the inline ladder's if-else exactly. The `MissingImpl` fallthrough
    is a normal registry result, not an error.
11. **`v_mode_bits == Q8` is not in ShapeInfo** (original risk + [F4]). The WMMA-FA variants
    are registered under `AttnFlashAsym4BatchedMasked` which is the Q8-V path. The
    asym4/fwht variants use different keys. Document as a key-level invariant.
12. **`dispatch_kv_write` does not gain tile dispatching** [F22]. WMMA-FA only accelerates
    attend; KV writes use existing 3.2 arms. Document explicitly.

---

## Out of scope (tracked elsewhere)

| Item | Where |
|---|---|
| WMMA-FA for **fwht4 / asym3 / fwht3** batched-masked (only asym4 has a WMMA tile today) | new kernels (future) |
| **2-bit tree-verify** kernel (the 3.2 `UnsupportedTreeTier` gap) | future kernel work |
| Dead WMMA benchmark variants (v2/v3/v4/v6/v7/v7b/n64*) | stay unregistered; revisit only if one wins a future bench |
| dots-ocr **2-D RoPE** / vision pre-attention ops | not attention dispatch (stay in dots-ocr) |
| qwen2 **text** decode/prefill attention | Ship 3.1b/3.2 llama-family follow-up |
| `pflash_score_*` scoring kernels | possible `ScoringFamily` (not attention) |
| Multi-GPU / MoE attention | Ship 4 / later |
| `attention_dflash_*` naming collision with DFlash spec-decode | future cleanup [F28] |
| Making every rung accept F16 K/V (m32/wmma_f32 kernel changes) | out of scope; 4-key split avoids the need |
| Priority field in `KernelVariant` (registration-order-is-priority works but is fragile) | future improvement [F18] |

---

## Dev log

| Date | Commit | What | Result |
|---|---|---|---|
| 2026-06-06 | — | Plan drafted. Grounded two WMMA modalities, llama liveness audit, registry mechanics. Decisions D1–D5. | — |
| 2026-06-06 | — | Adversarial review (GLM-5, Gemini, ds4): 28 findings (5 Critical, 6 High, 9 Medium, 8 Info). Key changes: F14 (causal scalar floor ≠ DflashScalar), F15 (4-key F16/F32 split), F16 (m32 needs hd%16==0), F2 (WMMA-FA covers hd=256), F3 (IsTree shape predicate for tree-verify), F6 (And() wrapper avoids type change), F14+F15+F16 elevate to Critical. D4 revised to 4-key schema. D2 revised to And() wrapper + HeadDimMultipleOf/HeadDimIn/IsTree. D5 revised with complete variant tables. C1 split into C1a (registry visibility) + C1b (ladder retirement). C4 gains KvTierInputs flag extension. | — |