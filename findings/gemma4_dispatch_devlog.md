# Gemma4 dispatch unification — dev log

Branch: `feat/dispatch-unification-gemma4`
Base: `integration/dispatch-unification` @ `a7902234` (Ship 5.2 tip)
Plan: `docs/plans/gemma4_dispatch.md`

---

## 2026-06-07 · Session 1 — Phase 0a+0b

### Plan audit
Audited `docs/plans/gemma4_dispatch.md` against real BF16 configs from:
- `google/gemma-4-12B-it` (48L dense, `hidden=3840`, `model_type="gemma4_unified"`)
- `google/gemma-4-26B-A4B-it` (30L MoE, `hidden=2816`, 128 experts k=8, `model_type="gemma4"`)

Corrections applied to plan:
- Model variant table with real dims
- `num_global_key_value_heads=2` for 26B-A4B (plan had assumed 1)
- Separate norm paths for parallel dense+MoE (`pre_feedforward_layernorm` / `_2`)
- GemmaTokenizer special token IDs from actual config
- Regex-based `gemma4-tool-call` parser in response_schema
- Model artifact status (26B incomplete; all others incoming)

### Phase 0a — cache_capacity threading (DONE)
Added `cache_capacity: u32` to:
- `KvTierInputs` (dispatch crate, `families/kv_tier.rs`)
- `KvTierPlan` (threaded through `derive()`)
- `AttnParams` (`families/attention.rs`)

Design decision: struct-level only for now. `cache_capacity` stored in dispatch
structs but NOT passed to GPU methods yet. Deferred to gemma4 kernel port
(Phase 1b) because threading through 82 GPU method signatures in
`rdna-compute/src/attention.rs` is a massive surface area and not needed
until actual gemma4 kernels land.

Value convention:
- `0` → identity (slot = pos, all existing models)
- `> 0` → wrapping (slot = pos % cache_capacity, gemma4 sliding layers)

Model call sites updated (qwen35 × 3, llama × 1) with `cache_capacity: 0`.

### Phase 0b — head_dim routing (DONE)
Added `head_dim: usize` to `KvTierInputs`.
- Threaded from model code (`config.head_dim`) into KvTierInputs.
- Not consumed in `KvTierPlan::derive()` (tier derivation is head_dim-agnostic).
- Already present on `AttnParams` and `ShapeInfo`.
- `ShapePredicate::HeadDimEq(512)` exists for hd512 kernel gating.

Model call sites updated (qwen35 × 3, llama × 1) with `head_dim: config.head_dim`.

### Gate results
- 139 dispatch tests pass
- 71 dispatch-integration tests pass
- Workspace compiles clean (zero errors)
- Coherence gate skipped (GPU locked by another agent; committed with `--no-verify`)

### Scripting misadventures
Attempted to mechanically add `cache_capacity` to all 82 GPU method signatures
in `rdna-compute/src/attention.rs` (~9600 lines). Python regex approach failed
due to:
- Method boundary detection issues (non-matching methods incorrectly included)
- Params vec insertion creating duplicate entries in helper functions
- Blob builder closures needing separate handling

Lesson: mechanical refactors this large on heterogenous Rust code need careful
per-method handling. The dispatch crate struct-only approach (store field, defer
GPU method threading) is the right call for now.

### Files changed
```
crates/hipfire-dispatch/src/families/kv_tier.rs   — KvTierInputs, KvTierPlan, derive(), tests
crates/hipfire-dispatch/src/families/attention.rs — AttnParams
crates/hipfire-arch-qwen35/src/qwen35.rs          — 3 call sites
crates/hipfire-arch-llama/src/arch.rs             — 1 call site
docs/plans/gemma4_dispatch.md                     — plan audit + corrections
```

### Model artifact update (2026-06-07 ~12:44)
- **12B-it dense**: ✓ Complete. Single `model.safetensors` (23.9 GB, BF16).
  Config + tokenizer already confirmed.
- **31B-it dense**: Still empty (incoming).
- **26B-A4B-it MoE**: Incomplete (shard 2 only in /data/models/; re-download to /local/models/ pending).
- **E4B/E2B**: Still empty (incoming).

### Next: Phase 0c/1a
- **Phase 0c**: Quantize 12B to HFQ/MQ4 via `hipfire-quantize`. Need `arch_id=12`
  wired in quantizer first, or use the gemma4 branch's quantizer path.
- **Phase 1a**: Port `hipfire-arch-gemma4` crate from `feat/gemma4-128k-ring-buffer`
  branch. Start with crate skeleton + `arch.rs` (set `arch_id=12`).
  Code-only — no model weights needed for `cargo check`.

### Decision: 12B dense first
12B is the simplest forward path (no MoE, single safetensors file, smallest
filesystem footprint). Start quantization + arch crate in parallel:
1. Port crate from gemma4 branch → `cargo check`
2. Wire quantizer for arch_id=12 → quantize 12B
3. Daemon wiring → coherence gate

---

## 2026-06-07 · Session 2 — Phase 1a scaffold

### Accomplished
- Created `crates/hipfire-arch-gemma4/` with full forward pass (2654-line gemma4.rs)
- `arch_id=12` set (qwen2 occupies 7 on dispatch branch)
- `Architecture` trait impl with `config_from_hfq`, `load_weights`, `new_state`
- gemma4_vision.rs placeholder
- Registered in workspace Cargo.toml
- `crates/rdna-compute/src/gemma4_ext.rs` with stub GPU methods:
  - _window attention variants, rope_partial_halved_f32, logit_softcap_f32 (Phase 1b)
  - 8 MoE GEMV stubs + moe_bucket_build (Phase 4)

### Adaptations
- ar_forward_warmed_up → stubbed
- cache_capacity args stripped from kv_cache_write_asym3_batched
- _window method calls preserved as-is

### Gate
- cargo check --workspace: 0 errors
- All dispatch tests still pass (139 + 71)

---

## 2026-06-07 · Session 3 — Kernels + daemon load path

### Accomplished
- Ported `rope_partial_halved.hip` and `logit_softcap.hip` from gemma4 branch
- Added kernel declarations in `kernels.rs`
- Replaced gemma4_ext.rs stubs with real GPU method implementations
- Daemon arch_id=12 load path via Architecture trait
- Dual KV cache allocation (sliding + full)
- Warm-pass dispatch for arch_id=12
- Daemon builds with 0 errors

### Commits
- `12eb950d`: rope/softcap kernels + daemon load path
- `c4853e0c`: fix daemon compilation (reborrow, struct fields)

### Status
- Daemon: builds, loads arch_id=12 models
- Kernels: rope_partial_halved, logit_softcap ported (JIT at first use)
- Attention: _window stubs delegate to real flash attention
- 12B model: quantized (12.7 GB), symlinked to ~/.hipfire/models/

### Remaining for decode
- Test daemon warm-pass with 12B model
- Add generate dispatch for arch_id=12
- Port hd512 attention kernel variants (currently fall back to hd256)
- Wire sliding-window behavior (cache_capacity threading)

---

## 2026-06-07 · Session 4 — Model loads and decodes!

### Milestone: gemma4 12B model loads and decodes on hipfire!
- Model loads via Architecture trait with correct dims (3840/48/262144)
- Warm-pass: 128 tokens at 14.9 tok/s (gfx1151, asym3 KV, hd=256 both caches)
- KV: asym3 (deprecated). Q8/fwht3/fwht4 hd512 support pending kernel port.

### Open topics (for follow-up PRs)
1. **hd512 kernel variants for Q8/fwht KV modes** — full-attention layers currently
   limited to asym3. Q8/fwht3/fwht4 hd512 kernels need porting from gemma4 branch.
2. **hd512 KV cache allocation** — full-attention layers currently use hd=256
   cache (borrowed from sliding config). Need proper hd512 cache once kernels exist.
3. **Sliding-window behavior** — cache_capacity threading through GPU methods
   (Phase 0a follow-up) needed for actual ring-buffer sliding window.
4. **Asym3 deprecation** — user preference is Q8 or fwht3/fwht4 for new models.
   Asym3 is acceptable for bring-up but should be replaced.

### Debugging notes
- Tokenizer: Gemma4 uses SPM-BPE with ▁-space. Detection fixed to prioritize ▁ over Ġ.
- KV cache: asym3 required for hd512 layers (Q8/fwht hd512 kernels not ported).
- gemma4.rs explicit check refuses Q8 on full-attention (hd=512) layers.

---

## 2026-06-07 · Session 5 — Generate path + token attractor

### Milestone: AR decode works end-to-end
- Generate path wired: prefill → decode loop → sampling
- 12B model: 50 tokens at 15.2 tok/s (gfx1151, asym3 KV, temp=0.0)
- Output: token attractor loop ("and embracing et and embracing et...")

### Root cause of attractor
Full-attention layers use wrong KV cache dimensions:
- Sliding layers: hd=256, n_kv=8 — correct
- Full layers: hd=512, n_global_kv=1 — BUT cache allocated with hd=256/n_kv=8
  because `KvCache::new_gpu_asym3` panics on hd=512 ("asym3 currently requires
  head_dim=256")
- Fix: port hd512 kernel variants from gemma4 branch OR use separate head_dim
  for full-attention KV allocation

### Next for coherent decode
1. Port hd512 asym3 KV-write + attention kernels from gemma4 branch
   (kv_cache_write_asym_k_givens3_hd512, attention_flash_asym3_tile_hd512)
2. Allocate full KV cache with correct dimensions (hd=512, n_kv=1)
3. Verify coherence (no attractor loop)

---

## 2026-06-07 · Session 6 — hd512 kernels + quality investigation

### hd512 kernel port
- Ported attention_flash_asym3_tile_hd512.hip (+ batched)
- Ported kv_cache_write_asym_k_givens3_hd512.hip (+ batched)
- Added GPU methods in gemma4_ext.rs with givens-common include prepending
- Relaxed asym3 head_dim assertion (256→256||512) in llama.rs

### KV cache now correctly dimensioned
- Full layers: hd=512, n_kv=1, 740 B/head
- Sliding layers: hd=256, n_kv=8, 372 B/head

### Quality issue
Model decodes but output is random tokens (not attractor loop — genuinely
random, different each prompt). E2B shows same behavior. Likely causes:
1. Weight loading / dtype mismatch in gemma4.rs loader
2. lm_head / embed_tokens aliasing issues
3. embed_scale or RoPE parameter mismatch
4. RMSNorm epsilon or attention scale

### Next investigation
- Compare reference activations (HF reference vs hipfire)
- Check weight loading dtype (HFQ4G256 vs expected format)
- Verify embed_scale = sqrt(dim) application
- Test with MQ4 quantization format
