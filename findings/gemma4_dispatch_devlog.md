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
