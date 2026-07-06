# Quant/Loading Extraction — Follow-up Pass

**Goal:** Hoist four quant/loading concerns out of the per-arch crates
(`hipfire-arch-qwen35::qwen35`, `hipfire-arch-qwen2::qwen2`) into the shared
`hipfire-runtime::weight_backend` layer, and eliminate the tied-embedding 2×
VRAM double-allocation on the single-GPU path. The `EmbeddingFormat` enum stays
as-is (kernel-coverage-bound — see Component 5).

This is the follow-up to `2026-06-10-declarative-layer-driver.md` (landed): that
plan hoisted the per-layer dequant/norm/name-resolution primitives and built the
`load_layer` driver. This pass extracts the remaining *entry-point* loaders
(embeddings, tied lm_head, AWQ sidecar, bias) that the prior plan left as
"out of scope: the per-entry-point prologue (`token_embd` / `output_norm` /
`output`)".

**Tech Stack:** Rust stable, `hipfire-runtime` (`weight_backend`, `llama`,
`hfq`), `hipfire-arch-qwen35`, `hipfire-arch-qwen2`, `rdna-compute`
(`Gpu`, `GpuTensor`, `DType`, `DeviceBuffer`), `hip-bridge` (`HipResult`).

---

## Background: the cross-arch duplication (verified)

| Concern | Copies today | Arch-varying part |
|---|---|---|
| Embedding load: `quant_type → (EmbeddingFormat, upload_raw vs host-f32)` | qwen35 `:1358-1399` (handles qt 6/7/3/2/16/1), qwen2 `load_embed_tokens:301-325` (handles qt 6/7/3/1) | **none** — pure quant-format knowledge; coverage *diverges* only by accident |
| Tied lm_head re-upload: `quant_type → DType` match + re-upload of `embed_tokens.weight` bytes | qwen35 `:1442-1513` (qt 6/7/8/13/14/3/else), qwen2 `load_lm_head:353-392` (qt 6/7/3/1) | the qt→DType list (both subsets of `dequant_weight_raw`'s ~14) |
| AWQ sidecar load: `*.awq_scale.weight` → F16→F32 → `GpuTensor` | qwen35 `load_awq_scale_for:1228-1256` | none — self-contained |
| `load_bias_f32`: F16/F32 reader with quant_type branch | qwen2 `:461-472` | none — `dequant_f32` already covers qt 1/2 |

Facts that make the hoist safe:
- `dequant_weight_raw` (`weight_backend.rs:79`) already handles all ~14 quant
  types: `quant_type` + bytes + `(m, k)` → `WeightTensor`, no arch types.
- `dequant_f32` (`weight_backend.rs:356`) already host-decodes qt 1 (F16),
  2 (F32), 3 (Q8_0), 14 (MQ8G256), etc. — covers both the embedding `HostF32`
  path's qt 1/2/16 *and* `load_bias_f32`'s qt 1/2.
- `GpuTensor::sub_offset` (`dispatch.rs:125`) establishes the non-owning-view
  precedent: `DeviceBuffer::from_raw` produces a handle that is **not** freed on
  Drop; freeing is explicit via `gpu.free_tensor`. `shallow_clone` is the
  full-buffer form of the same pattern.
- `EmbeddingFormat` lives in `hipfire_runtime::llama`; `GpuTensor`/`DType`/
  `DeviceBuffer` in `rdna_compute`/`hip_bridge`.

**Hard topology constraint (drives Component 2):** the multi-GPU qwen35 path
(`free_gpu_multi:761`, `load_weights_multi`) places `token_embd` on device 0 but
`output` (lm_head) on `gpus.output_device`, which can be a **different physical
GPU**. A buffer cannot be shared across two GPUs, so the double-alloc fix is only
valid on the single-GPU, same-device tied path. Multi-GPU and any future
cross-device tied placement MUST keep re-uploading.

---

## Component 1 — Full embedding loader (`weight_backend.rs`)

Split into a GPU-free classifier (unit-testable) plus a thin GPU wrapper.

```rust
/// How an embedding table's on-disk bytes map to the device.
enum EmbedPlan {
    /// Upload bytes verbatim; the lookup kernel dequantizes on the fly.
    Raw(EmbeddingFormat),
    /// Host-decode to f32 (via `dequant_f32`) then upload as F32.
    HostF32,
}

/// Pure quant_type → plan. GPU-free, unit-testable.
/// qt 6 → Raw(HFQ4G256), 7 → Raw(HFQ4G128), 3 → Raw(Q8_0),
/// qt 1|2|16 → HostF32, else → panic with the supported-format list.
pub(crate) fn embed_classify(quant_type: u8) -> EmbedPlan;

/// Load an embedding table to the device. Unifies the qwen35 and qwen2
/// hand-written matches. Returns the device tensor + its on-GPU format.
pub fn load_embedding(
    gpu: &mut Gpu,
    quant_type: u8,
    data: &[u8],
    vocab: usize,
    dim: usize,
) -> HipResult<(GpuTensor, EmbeddingFormat)>;
```

`load_embedding` dispatches on `embed_classify`:
- `Raw(fmt)` → `gpu.upload_raw(data, &[data.len()])` → `(buf, fmt)`.
- `HostF32` → `dequant_f32(gpu, quant_type, data, vocab * dim)` → `(buf, F32)`.
  (`dequant_f32` already decodes qt 1's f16, qt 2's f32, qt 16's bf16-high-half.)

**Shape note (verified):** `dequant_f32` uploads with shape `[n]` (1D), whereas
the old qwen35 F32 path used `[vocab, dim]`. This is behaviorally identical — the
embedding-lookup kernels (`embedding_lookup` and the `_q8`/`_hfq4*` variants in
`rdna-compute/src/embedding.rs`) compute byte offsets from `token_id` + `dim`
against `table.buf` directly and **never read `table.shape`**. No downstream
code reads the embedding's 2D shape.

**Coverage unification + latent-bug fix:** qwen35 previously handled qt
{6,7,3,2,16,1} and qwen2 {6,7,3,1}; both now share the full set. qwen35's old
`else` branch decoded *any* unknown qt as F16 (`chunks_exact(2)`), which would
silently corrupt an MQ4 (qt 13) or MQ8 (qt 14) embedding table. `embed_classify`
**panics** on unknown qt instead — strictly safer; no shipping model exercises
the old fallback (embeddings are F16/F32/HFQ4/Q8 in every current checkpoint).

Call-site reduction (both arches):
```rust
let (token_embd, embd_format) = load_embedding(gpu, qt, &data, vocab, dim)?;
```
The qwen35 `eprintln!("    (… raw, {} MB)")` size logging is dropped (cosmetic;
the generic "loading token_embd…" line stays).

---

## Component 2 — Tied lm_head: dedup + double-alloc fix

### New primitive (`rdna-compute/src/dispatch.rs`)

```rust
impl GpuTensor {
    /// Full-buffer non-owning alias (the whole-tensor form of `sub_offset`).
    /// The returned tensor shares the source's device pointer; it is a VIEW —
    /// do NOT pass it to `free_tensor`. `DeviceBuffer::from_raw` has no
    /// Drop-time free, so the alias and source coexist safely until the OWNER
    /// is freed exactly once.
    pub fn shallow_clone(&self) -> GpuTensor;
}
```

### Tied-load logic, split by device topology

- **Single-GPU, same device (the fix):**
  `output.buf = token_embd.shallow_clone()`. No re-read of `embed_tokens.weight`,
  no second upload — **saves `vocab × dim × dtype_bytes` of VRAM**. The
  `WeightTensor` carries the embedding's dtype (via `embedding_format_dtype`),
  `m = vocab`, `k = dim`, `paro: None`, `awq_scale: None`. Set the weights
  struct's `lm_head_aliases_embd = true`.

- **Multi-GPU / cross-device (qwen35 `output_device != 0`):** a cross-GPU alias
  is impossible. Keep re-uploading, but collapse the hand-written qt→DType match
  to a single call:
  `dequant_weight_raw(gpu_out, qt, &tied_data, vocab, dim)`. The result owns its
  buffer; `lm_head_aliases_embd = false`.

### Format→dtype helper (`weight_backend.rs`)

```rust
/// EmbeddingFormat → the DType tag for a tied lm_head WeightTensor.
/// Replaces both arches' inline matches. (Q4K is not a valid tied format → panic.)
pub fn embedding_format_dtype(fmt: EmbeddingFormat) -> DType;
```

### Free-path guard (prevents double-free of the alias)

- **qwen2** already carries `tied_lm_head: bool` (`:216`). qwen2 has no multi-GPU
  path, so tied always aliases. `free_gpu` skips `free_tensor(self.output.buf)`
  when `tied_lm_head` is true; frees it otherwise.
- **qwen35**: add `lm_head_aliases_embd: bool` to `Qwen35Weights`.
  - `free_gpu` (`:684`): skip `self.output.free_all(gpu)` when
    `lm_head_aliases_embd` (the alias has no owned buf, and `paro`/`awq_scale`
    are `None` on a tied lm_head, so skipping `free_all` entirely is safe).
  - `free_gpu_multi` (`:761`): always `false` — the multi path never aliases, so
    `self.output.free_all(...)` runs unchanged.

The alias is only ever the *non-owning* side; `token_embd` remains the single
owner and is freed exactly once. Because `DeviceBuffer::from_raw` has no
Drop-free (verified via the `sub_offset` view contract), the alias dropping is a
no-op.

---

## Component 3 — AWQ sidecar (MEDIUM)

Move `load_awq_scale_for` (qwen35 `:1228-1256`) verbatim into `weight_backend.rs`
as a `pub fn`. It depends only on `HfqFile::tensor_data_pread`, `f16_to_f32`
(`hipfire_runtime::llama`), and `gpu.upload_raw` — all already in
`hipfire-runtime`. qwen35's `load_weight_tensor` imports the moved symbol.
qwen2 does not currently attach AWQ sidecars; it is left untouched (no scope
creep).

## Component 4 — `load_bias_f32` (MEDIUM)

Replace the body (qwen2 `:461-472`) with:
```rust
let (info, data) = hfq.tensor_data_vec(name).unwrap_or_else(|| panic!(...));
assert_eq!(data_elem_count, n, "...");   // config-mismatch guard, kept
dequant_f32(gpu, info.quant_type, &data, n)
```
The inline f16/f32 match is now `dequant_f32`'s job. The length assert is
retained as a cheap guard against config/tensor mismatch (`dequant_f32` does not
assert `n`).

## Component 5 — `EmbeddingFormat` enum (LOW) — out of scope

The enum (`llama.rs:641-647`) stays unchanged. Its 5 variants are **not**
redundant with `DType`: each gates selection of a dedicated embedding-lookup
kernel (`embedding_lookup_hfq4g256_batched`, `embedding_lookup_q8_batched`,
`embedding_lookup_q4k`, …). Formats with no lookup kernel (MQ3G256, HFP4G32,
MQ4G256Lloyd, …) deliberately fall back to host-dequant→F32. Collapsing the enum
into `DType` would require writing new per-format lookup kernels — a feature, not
a refactor — so it is explicitly deferred.

---

## File Map

**Modified:**
- `crates/hipfire-runtime/src/weight_backend.rs` — add `embed_classify`,
  `load_embedding`, `embedding_format_dtype`, `load_awq_scale_for` (moved);
  add GPU-free unit tests.
- `crates/rdna-compute/src/dispatch.rs` — add `GpuTensor::shallow_clone`.
- `crates/hipfire-arch-qwen35/src/qwen35.rs` — delete the embedding match, the
  tied-lm_head match, and `load_awq_scale_for`; route embeddings through
  `load_embedding`; route tied lm_head through `shallow_clone` (single-GPU) /
  `dequant_weight_raw` (multi-GPU); add `lm_head_aliases_embd` to
  `Qwen35Weights` and update both free paths.
- `crates/hipfire-arch-qwen2/src/qwen2.rs` — route `load_embed_tokens` through
  `load_embedding`; route tied lm_head through the shared helper +
  `shallow_clone`; replace `load_bias_f32` body with `dequant_f32`; skip the
  aliased buffer in `free_gpu`.

**No new files.**

---

## Verification

1. **GPU-free unit tests** (`weight_backend.rs`):
   - `embed_classify`: qt 6/7/3 → `Raw(<fmt>)`, qt 1/2/16 → `HostF32`,
     unknown qt → panics.
   - `embedding_format_dtype`: each of the 5 variants → expected `DType`; Q4K
     panics.
2. **Build** both arch crates (`cargo build -p hipfire-arch-qwen35 -p
   hipfire-arch-qwen2`).
3. **Tied-alias correctness + double-free** (the load-bearing test): load →
   unload (`free_gpu`) → **reload** a `tie_word_embeddings=true` model
   (Qwen2-1.5B-Instruct). A double-free or use-after-free surfaces as a HIP
   error or crash on the second load. Confirm VRAM does not leak across the
   cycle.
4. **Coherence:** `./scripts/coherence-gate.sh` on a qwen35 model (HFQ4G256) and
   a tied qwen2 model — embeddings feed the forward path, so corruption would
   show as incoherent output. Per CLAUDE.md the pre-commit hook runs this when
   loader files are staged.

## Risks

- **Double-free of the alias** — mitigated by (a) `shallow_clone` being
  non-owning (`from_raw`, no Drop-free, per the `sub_offset` contract) and
  (b) the free-path guard skipping the aliased buffer. Test #3 is the gate.
- **Cross-device alias** — structurally prevented: aliasing is gated on
  single-GPU/same-device; `free_gpu_multi` always re-uploads
  (`lm_head_aliases_embd = false`).
- **Stricter unknown-qt panic** — `embed_classify` panics where qwen35 used to
  silently mis-decode as F16. No shipping model hits this path; the panic is a
  net safety improvement, not a behavior regression for real inputs.
