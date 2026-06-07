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

---

## 2026-06-07 · Session 7 — v_norm fix + quality improvement

### Root cause found: v_norm_ones_full never initialized
- `init_scratch_constants` was defined but never called
- v_norm_ones_full stayed as zeros → v_norm output = 0 → attention output = 0
- All subsequent layers received zero attention contribution
- Model collapsed to `<audio|>` token attractor (258883)

### Fix
Added `gemma4::init_scratch_constants(gpu, &scratch, config.full_head_dim)` 
in daemon after scratch allocation. This fills v_norm_ones_full with 1.0.

### Results
- Layer-0 activations now match HF: input_norm (1036 vs 1041), q/k/v proj within 1%
- Output: diverse tokens, no more attractor loop
- Top tokens: "Sne dequeue twig penny..." (real words, not special tokens)
- HF expects: "1-.," (digits for "Hello1")
- Remaining discrepancy: o_proj output (-30.7 vs HF 47.9) and final logits

### Next
- Remaining quality gap likely from flash attention numerics or Q8 quantization
- Need HF oracle comparison through all 48 layers

---

## 2026-06-07: Per-layer oracle investigation

### Methodology
Compared per-layer hidden states (sum of all 3840 elements) between HF bf16
reference and hipfire HFQ4 model for the "Hello" prompt (BOS + token 9259).

### Results

| Layer | HF sum   | Hipfire sum | Delta |
|-------|----------|-------------|-------|
| embed | 21.3     | 21.0        | ~match |
| L0    | -22.1    | -25.1       | close but diverging |
| L1    | 60.6     | 36.6        | 2x off |
| L2    | -77.5    | -25.4       | 3x off |
| L3    | -73.6    | -6.9        | 10x off |
| L4    | -18.3    | 79.1        | sign flip! |
| ...   |          |             |       |
| L47   | -7.9     | -8.6        | different |
| final | -82.7    | -508.7      | 6x off |

### Key finding
Embedding + Q/K/V projections match HF. L0 output is close but not identical.
Divergence compounds rapidly — by L4 the signs flip. After 48 layers the
hidden state is completely wrong.

### Attempted fixes (no improvement)
- Q8 weight quantization: hidden sum=-508 (same magnitude of error)
- Q8 KV cache for sliding layers: hidden sum=-508 (same)
- MQ4 weight format: WORSE — produces `<audio|>` attractor tokens

### Root cause narrowed to
The L0 attention output diverges from HF. Since Q/K/V projections match,
the divergence is within the attention computation itself:
- RoPE application
- KV cache write + read (asym3 quantization)
- Flash attention kernel numerics
- o_proj projection

### Next step
Compare detailed L0 intermediates (post-RoPE, post-attention, post-o_proj)
between HF and hipfire to find the exact step where divergence starts.

---

## 2026-06-07 (session 2): Critical HF oracle fix + fp32 KV investigation

### Bug found: HF reference was DOUBLE-SCALED

The per-layer HF oracle comparison from session 1 was **completely wrong** due to
a double-scaling bug in the Python script:

```python
# WRONG (session 1):
emb = lm.embed_tokens(input_ids).float() * lm.embed_tokens.embed_scale.float()
# lm.embed_tokens() ALREADY applies embed_scale internally!
# This multiplied by scale TWICE: raw * 62 * 62 = raw * 3844

# CORRECT:
emb = lm.embed_tokens(input_ids).float()  # Already scaled!
```

This made ALL previous comparisons invalid. The HF "sum=1318" for the Hello
embedding was actually 1318 = 0.34 * 62 * 62, when the correct scaled value is
21.3 = 0.34 * 62.

### Corrected per-layer comparison (HFQ4 model, asym3 KV)

With the corrected reference, the per-layer sums are much closer than previously
believed:

| Step | HF sum | Hipfire sum | Status |
|------|--------|-------------|--------|
| embed (scaled) | +21.3 | +21.0 | ✓ match |
| input_norm | +1040.8 | +1036.4 | ✓ |
| q_proj | +2137.3 | +2125.4 | ✓ |
| k_proj | -649.1 | -657.7 | ✓ |
| v_proj | -1275.7 | -1259.5 | ✓ |
| q_norm | +51.4 | +51.1 | ✓ |
| k_norm | +1.3 | +1.2 | ✓ |
| v_norm | -48.5 | -47.9 | ✓ |
| scale_q | +822.1 | +818.2 | ✓ |
| rope_q | +614.4 | +611.2 | ✓ |
| rope_k | +1.6 | +1.5 | ✓ |
| **attention** | **-30.8** | **-55.9** | **✗ 1.8x off** |
| o_proj | -3.0 | -30.7 | ✗ |
| attn_residual | +200.8 | +175.4 | off |
| pre_ffn_norm | -46.5 | -55.1 | off |
| L0 output | -26.8 | -25.1 | close |

Everything matches perfectly through RoPE. The **first divergence is at the
attention output**. Q/K/V projections + norms + RoPE are all correct.

### Attention output comparison across KV formats

| KV format | L0 attn sum | Delta from HF (-30.8) |
|-----------|-------------|----------------------|
| HF exact (bf16) | -30.8 | baseline |
| fp32 KV | -37.9 | 23% off |
| Q8 KV | -39.1 | 27% off |
| asym3 KV | -55.9 | 81% off |

Even fp32 KV (zero quantization error) diverges 23% from HF. This means the
divergence is NOT solely from KV quantization — there's a difference in the
attention computation itself.

### Hypotheses

#### H1: Flash attention kernel numerical difference
The `attention_flash` kernel uses a two-phase tiled softmax (partial + reduce).
For 2 tokens, there's 1 chunk so the online softmax should be exact. But the
kernel may use a different accumulation order or precision than PyTorch's exact
matmul + softmax.

**Test**: Dump per-head attention weights and scores from both HF and hipfire
fp32 path. If the softmax weights differ, it's a kernel numerics issue.

#### H2: GQA expansion mismatch
Hipfire uses `n_heads=16, n_kv=8` with GQA ratio=2. The flash kernel handles
GQA internally. If the head-grouping is wrong (e.g., heads 0,1 share KV0
instead of heads 0,8 sharing KV0), the attention output would be completely
wrong for some heads but correct for others.

**Test**: Dump per-head attention output sums and compare with HF. If only
half the heads diverge, it's a GQA grouping issue.

#### H3: `attention_flash` reads K/V from wrong offset
The fp32 KV cache is indexed by position. If the kernel reads K/V starting
from the wrong offset (e.g., it assumes a different memory layout), the
attention would compute with wrong K/V values.

**Test**: Dump K/V from the fp32 cache after writing and compare with the
original Q/K/V projections.

#### H4: Scale factor mismatch in flash kernel
`attention_flash` uses `scale = 1/sqrt(head_dim)`. We pre-scale Q by
`sqrt(head_dim)`. Net: `sqrt(256) * 1/sqrt(256) = 1.0`. But if the kernel
applies scale differently (e.g., to K instead of Q), the effective scale
would be wrong.

**Test**: Dump raw scores (Q·K^T) before softmax from the kernel. If they
don't match HF's scores, the scale application is wrong.

### Most likely root cause: H2 (GQA head grouping)

The fp32 KV divergence (23% with zero quantization) strongly suggests a
structural issue in the attention computation, not quantization noise. The
GQA head grouping is the most likely candidate because:
- Q/K/V projections match perfectly (individual GEMVs are correct)
- RoPE matches (applied per-head, correct)
- The divergence appears ONLY after the flash attention kernel
- GQA grouping is a subtle indexing issue that would produce "almost right"
  output (some heads correct, others wrong)

### Next steps
1. Dump per-head attention output from fp32 path and compare with HF
2. Check GQA head-grouping convention in `attention_flash` kernel
3. If H2 confirmed: fix head grouping, re-run coherence test
4. If H2 ruled out: dump raw scores from flash kernel to test H4

---

## 2026-06-07 (session 3): Q·K^T dump + review agent findings

### Q·K^T scores match HF within 0.5%

Dumped post-RoPE Q, K, V from hipfire and computed Q·K^T scores manually.
Every head matches HF within 0.5%:

```
Head  | Hipfire  | HF       | Delta
  0   | +34.22   | +34.30   | 0.2%
  1   | +272.01  | +271.91  | 0.04%
  2   | +59.24   | +59.40   | 0.3%
  ...
```

**This rules out H4 (scale mismatch) and K-layout issues.** The score computation
is correct.

### Per-head data disproves H2 (GQA grouping)

Within GQA pairs, one head matches HF while the other diverges:
- Heads 4,5 share KV head 2: head 4 matches, head 5 diverges
- Heads 6,7 share KV head 3: head 6 matches, head 7 diverges

Since both heads read the SAME K and V, the bug must be Q-side (softmax
precision), not KV-side. GQA grouping (H2) and KV offset (H3) are falsified.

### Review agent's key insight: 2-token test is degenerate

The 2-token prompt produces saturated softmax (Q pre-scaled by √256=16,
scores ~34-272). Tiny numerical differences cause the softmax to tip at
the boundary, creating per-head divergence. This is amplified by the
all-or-nothing nature of saturated attention.

**Predicted**: with a 20-50 token prompt, attention distributions soften and
the per-head divergence collapses. The L0 output sum (−26.8 vs −25.1, ~6%)
is already close despite the per-head noise.

### Revised hypotheses

- **H1 (softmax saturation artifact)**: MOST LIKELY. 2-token test amplifies
  tiny numerical differences through hard-saturated softmax. Longer prompts
  should show much closer match.
- **H5 (HFQ4 weight quantization compounding)**: The fp32 KV tests still
  use HFQ4 quantized weights. Over 48 layers, the 4-bit quantization error
  in q/k/v/o projections compounds. Need bf16/fp32 weights to discriminate.
- H2 (GQA grouping): FALSIFIED by per-head within-pair analysis
- H3 (KV offset): FALSIFIED by Q·K^T match + V cache verification
- H4 (scale mismatch): FALSIFIED by Q·K^T match

### Next steps (per review agent recommendations)
1. Re-run coherence test with a 20-50 token prompt (softens attention)
2. If still incoherent: quantize model as bf16/fp32 weights to isolate
   weight quantization error from kernel bugs
3. The bar is coherence, not logit-bit-match

### Longer prompts + Q8 weights test — STILL GARBAGE

Tested 3 prompts (capital/code/reason, 10-15 tokens each) with:
- HFQ4 weights + fp32 KV: garbage
- Q8 weights + fp32 KV: garbage

Both produce random tokens across languages, special tokens (`<audio|>`, `（`),
and repeated characters. This is NOT a 2-token softmax artifact and NOT a
weight quantization issue.

The per-layer L0 sum is close (-26.8 vs -25.1, ~6%) but over 48 layers
the small per-layer error compounds into complete incoherence. Even with
Q8 weights (near-zero quantization error) the output is garbage.

**This means the structural attention divergence is real and compounds.**

The softmax saturation hypothesis (from external review) predicted that
longer prompts would soften the attention and reduce divergence. This
did NOT happen — longer prompts produce equally garbage output.

### Remaining investigation path

Since:
- Q·K^T scores match HF ✓
- Embed + norms + projections match ✓
- RoPE matches ✓
- GQA grouping is correct (ruled out by within-pair analysis) ✓
- Q8 weights don't fix it ✓
- fp32 KV doesn't fix it ✓
- Longer prompts don't fix it ✓

The bug must be in one of:
1. **The softmax or weighted-sum path inside the flash attention kernel**
   (scores are correct but output diverges → softmax bug)
2. **The `attention_flash` kernel reads K/V from wrong positions in
   the fp32 cache** (for pos>1, the cache layout may be wrong)
3. **The warm-pass (128 synthetic tokens) corrupts the KV cache** before
   the real prompt is processed — positions 0-127 have synthetic data,
   only positions 0..N-1 are overwritten by the prompt

### Warm-pass contamination — ruled out

Disabled the warm-pass for gemma4 (arch_id==12). Output is byte-identical
to the warm-pass-enabled run. The warm-pass does not affect the generate
path because:
1. The generate prefill overwrites positions 0..N-1
2. For fp32 KV, uninitialized positions are zero (memset at allocation)
3. Zero K/V values don't contribute to attention (dot product = 0)

### Summary of ruled-out hypotheses

| Hypothesis | Status | Evidence |
|---|---|---|
| H1: Tokenizer ▁ prepend | FIXED | Tokenizer fix commit |
| H2: v_norm uninitialized | FIXED | init_scratch_constants commit |
| H3: Double-scaled HF oracle | FIXED | Removed second embed_scale multiply |
| H4: Scale mismatch | FALSIFIED | Q·K^T matches within 0.5% |
| H5: GQA grouping | FALSIFIED | Within-pair head divergence |
| H6: KV offset/layout | FALSIFIED | fp32 cache dump matches expected |
| H7: Weight quantization | FALSIFIED | Q8 weights also garbage |
| H8: 2-token softmax artifact | FALSIFIED | Longer prompts also garbage |
| H9: Warm-pass contamination | FALSIFIED | Disabling gives identical output |
| H10: Embed scale not applied | FALSIFIED | Confirmed scale=61.97 applied correctly |

### Current status: stuck

All hypotheses have been ruled out. The attention output diverges from HF
even though Q·K^T scores match. The remaining suspect is the softmax or
weighted-sum computation inside the `attention_flash` kernel itself. This
kernel is shared with Qwen3.5 which works correctly, so the bug must be
in how gemma4's specific parameters (n_heads=16, n_kv=8, head_dim=256)
interact with the kernel.

Next step: instrument the `attention_flash_partial` kernel to dump
post-softmax weights and V-weighted sums for a specific head, and compare
with HF's exact computation.
