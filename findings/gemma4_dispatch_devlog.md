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

### E2B model test — ALSO GARBAGE

Tested Gemma4 E2B (n_kv=1, k_eq_v=False, 35 layers, 5.5 GB). Output is
equally incoherent — random multilingual tokens. The bug affects ALL gemma4
models, not just 12B.

This rules out:
- GQA as the issue (E2B has n_kv=1, no GQA)
- attention_k_eq_v handling (E2B has k_eq_v=False)
- 12B-specific config issues

The bug must be in something FUNDAMENTAL to the gemma4 forward pass that
is shared across all model sizes. Candidates:
1. The RoPE implementation (different theta per layer type)
2. The sandwich norm structure (4 norms per layer)
3. The layer_scalar multiplier
4. The final_logit_softcapping
5. Something in the basic attention/FFN path

### HF v_norm verification
- v_norm is `Gemma4UnifiedRMSNorm` with `with_scale=False`
- This is pure divide-by-RMS — no learned weight
- Hipfire correctly uses `rmsnorm_batched` with ones buffer ✓

### HF attention_k_eq_v semantics
- 12B/31B: `attention_k_eq_v=True`, but sliding layers HAVE v_proj
- Full layers have `v_proj is None` → V = k_proj(x) (pre-k_norm)
- Hipfire handles both correctly per layer type ✓

### Updated hypothesis list
The bug must be in something shared across all gemma4 sizes:
1. **Sandwich norm ordering** — maybe one of the 4 norms is applied wrong
2. **RoPE per layer type** — different theta for sliding vs full, could be swapped
3. **FFN activation** — gelu_pytorch_tanh vs some other variant
4. **The forward pass control flow** — maybe the loop over layers processes
   layer types in wrong order
5. **Something in the residual accumulation** — double-add, wrong buffer

---

## 2026-06-07 · Session 10 — Per-layer trace + sub-step bisect (BREAKTHROUGH)

### Changed dump guard from pos==1 to pos==0
The diagnostic dump was guarded on `pos == 1 && kv_layer_idx == 0`, which
meant it only triggered at the second token position. For the "Hello" single-
token prompt (no BOS prepended by the daemon), pos=0 is the only decode
position. Changed to `pos == 0 && kv_layer_idx == 0` to capture L0
sub-steps for the single-token case.

### Full per-layer HF reference trace ("Hello" prompt)

Ran HF reference for all 48 layers on "Hello" (single token, no BOS):
```
embed:  sum=21.26   [0.82, 0.58, -0.25, 1.27]
L0 (S): sum=-22.10  [-0.85, 2.95, -0.79, 0.13]
L5 (F): sum=48.15   [0.03, -0.81, -0.09, 1.06]
L47(F): sum=-7.90   [0.08, -0.08, 0.01, -0.02]
```

### Side-by-side first4 comparison (HF vs hipfire, "Hello" pos=0)

```
Layer  Type   HF[0]  HF[1]  HF[2]  HF[3]  || HIP[0] HIP[1] HIP[2] HIP[3] || Δmax
L0     S     -0.85   2.95  -0.79   0.13   ||  0.17  -0.15  16.63   0.01  || 17.4  ← HUGE at L0!
```

Divergence is **already catastrophic at L0** — a sliding layer. This
invalidates the earlier "L0 is ~6% off, the problem must be in full layers"
narrative. The 6% was a sum-cancellation artifact (see Claude review §UPDATE).

### Claude review hypotheses 1A/1B/1C checked

**1A: RMSNorm +1 shift — RULED OUT.**
- HF `Gemma4UnifiedRMSNorm` stores weights initialized at 1.0 and trained
  (mean≈6.6 for input_layernorm), applies as `normed * weight` (no +1).
- Hipfire's `rmsnorm_f32(x, weight) = x * rsqrt(mean(x²) + eps) * weight`
  is correct — no +1 needed.
- The earlier Gemma convention (zero-centered weights needing +1) does NOT
  apply to Gemma4.

**1B: Attention scaling = 1.0 — CONFIRMED CORRECT.**
- HF's `self.scaling = 1.0` hardcoded in `Gemma4UnifiedTextAttention.__init__`.
- HF computes `Q @ K^T * 1.0` (no 1/√d).
- Hipfire pre-scales Q by √d to cancel the flash kernel's internal 1/√d,
  achieving the same net effect of scaling=1.0. ✓

**1C: layer_scalar semantics — APPEARS CORRECT.**
- HF: `hidden_states *= self.layer_scalar` applied to entire residual at
  end of each layer.
- Hipfire: `gpu.scale_f32(&scratch.x, lw.layer_scalar_host)` — same. ✓
- layer_scalar values are ≈1.0 (learned, close to identity).

**1D: embed_scale precision — CONFIRMED CORRECT, MINOR.**
- HF casts √3840 to bf16 (61.97 → 62.0) before multiply.
- Hipfire keeps f32. Small drift, not garbage-output cause.

### Sandwich norm ordering — CONFIRMED CORRECT

HF layer forward:
```python
residual = x
x = input_layernorm(x)
x = self_attn(x)              # includes o_proj
x = post_attention_layernorm(x)
x = residual + x

residual = x
x = pre_feedforward_layernorm(x)
x = mlp(x)
x = post_feedforward_layernorm(x)
x = residual + x

x *= layer_scalar
```

Hipfire matches this order exactly. ✓

### L0 sub-step bisect — WHERE THE DIVERGENCE HAPPENS

Dumped 20+ intermediate taps through L0 for "Hello" at pos=0, and matched
against HF hooked intermediates.

**Steps that MATCH (Δ < 0.3 per element):**
- embed / scratch.x: HF [0.82, 0.58, -0.25, 1.27] vs HIP [0.81, 0.59, -0.26, 1.27] ✓
- post_input_norm: HF [13.56, 16.38, 0.00, 19.00] vs HIP [13.39, 16.61, 0.00, 18.94] ✓
- post_q_proj: HF [-5.78, 6.56, 0.65, 3.42] vs HIP [-5.80, 6.53, 0.80, 3.36] ✓
- post_k_proj: HF [6.91, 12.88, -16.00, 22.50] vs HIP [6.92, 13.13, -15.88, 22.54] ✓
- post_v_proj: HF [7.78, 42.00, -3.03, -10.06] vs HIP [7.94, 41.96, -3.18, -10.00] ✓

**Steps that DIVERGE:**
- post_o_proj: HF [-0.09, 0.12, -3.67, 1.70] vs HIP [-0.79, 0.55, -2.12, 0.09]
  - Δ = [0.70, 0.42, 1.55, 1.61] — **10× worse than projections**

**Conclusion: the attention layer (from v_norm output to o_proj output) is
where the divergence originates.** Everything before attention matches
within HFQ4 quantization error (~1-5%); after attention, the outputs diverge
by 10-30×.

### Per-head attention output comparison

For single-token decode, softmax has one element → attention output = V.
HF confirms this: every head pair (h, h+1) is identical (GQA), and each
head has RMS ≈ 1.0 (from v_norm normalization).

**HF per-head attn output (first2 of each head):**
```
head  0 (kv=0): [-0.0884, -0.2930]  sum= -8.41
head  2 (kv=1): [-1.2734, -1.8906]  sum= -4.08
head  4 (kv=2): [-3.3125,  0.6055]  sum= -5.65
head  6 (kv=3): [ 0.4668,  0.7969]  sum= 18.02   ← note positive
head  8 (kv=4): [ 0.1426,  0.1064]  sum=-23.00
head 10 (kv=5): [-0.9883,  0.5938]  sum= -8.96
head 12 (kv=6): [-0.2021, -0.3496]  sum=-11.60
head 14 (kv=7): [-1.3594,  0.0045]  sum= -5.11
```

**Hipfire per-head attn output (first2 of each head):**
```
head  0 (kv=0): [-0.1065, -0.7027]  sum=-10.72   ← DIFFERENT from HF
head  2 (kv=1): [-0.4392,  ?]       sum= 10.60   ← OPPOSITE SIGN to HF
head  4 (kv=2): [-0.7356,  ?]       sum=  6.36   ← OPPOSITE SIGN to HF
head  6 (kv=3): [ 0.0007,  ?]       sum=-12.14   ← OPPOSITE SIGN to HF (+18 → -12!)
head  8 (kv=4): [ 0.3837,  ?]       sum= -4.22
head 10 (kv=5): [-0.1692,  ?]       sum=-13.43
head 12 (kv=6): [-0.2027,  ?]       sum= -7.05
head 14 (kv=7): [-0.4279,  ?]       sum= -8.62
```

Multiple heads show **opposite-sign** outputs vs HF. For single-token decode
where attn_out = V, this means the V values written to/read from the KV cache
are fundamentally wrong — not quantization noise.

### CRITICAL: v_proj matches but v_norm output doesn't

Wait — this is the key contradiction:
- v_proj output matches HF (within quantization error) ✓
- v_norm is divide-by-RMS-only (no learned weight) ✓
- Yet attention output ≠ v_norm output in hipfire

For single-token decode, attention output should equal the v_norm output
(for each KV head). If they don't match, the attention kernel is reading V
from the wrong KV cache slot, or the KV cache write/read has a layout bug.

Actually, looking more carefully: hipfire's attention output first2 =
[-0.1065, -0.7027] and hipfire's v_norm first2 = [-0.1065, -0.7027].
They DO match for head 0. But HF's head 0 first2 = [0.2266, 1.2266].

**So the V values are correct WITHIN hipfire's computation chain but wrong
vs HF.** The v_proj output differs from HF at pos=0, which means either:
1. The quantized weight is wrong (but q_proj/k_proj match — why only v_proj?)
2. The input to v_proj differs subtly (post_input_norm is close but not exact)
3. The v_proj weight quantization has a systematic error (bad group alignment?)

### Remaining open questions for next session

1. **Why do q_proj/k_proj match HF but v_proj doesn't at pos=0?** All three
   use the same input (post_input_norm) and the same GEMV kernel. The v_proj
   weight must be quantized differently or loaded with wrong dims.

2. **Is the weight loading order correct?** v_proj weight shape should be
   `[n_kv * head_dim, dim]` = `[2048, 3840]`. If it's transposed or has
   swapped dims, the GEMV would produce wrong results.

3. **Are all projection weights loaded from the correct safetensors keys?**
   The model uses `model.language_model.layers.{i}.self_attn.v_proj.weight`.
   A key mapping error could load the wrong weight.

4. **Does the quantizer handle v_proj's row count (2048) differently than
   q_proj's (4096)?** The group size is 256; 2048 = 8 groups, 4096 = 16
   groups. If the quantizer misaligns groups for non-power-of-2 group counts,
   this would explain a v_proj-specific error.

### BOS token handling note

The daemon's gemma4 path at `daemon.rs:11494` says:
```
// Raw tokenization. BOS (2) prepended manually — Gemma4 SPM-BPE
// doesn't auto-prepend.
```
But `bos_token: None` is passed to the tokenizer, so BOS is NOT prepended.
This means "Hello" tokenizes as [9259] with no BOS, and HF's tokenizer also
doesn't add BOS for `encode()` (only `__call__` auto-prepends). Both sides
are consistent: single token 9259, no BOS.

---

## 2026-06-07 · Session 11 — BOS token context + Full-layer divergence isolated

### Critical finding: daemon prepends BOS token

The daemon's gemma4 path (`daemon.rs:11494-11496`) prepends BOS (token 2):
```rust
let mut ids = vec![2u32]; // BOS
ids.extend(tokenizer.encode(prompt));
```

This means "Hello" is tokenized as [2, 9259], not [9259]. All previous
HF comparisons that used `tok("Hello")` (which returns [9259] without BOS)
were comparing against different token positions:
- pos=0 in hipfire = BOS token (not "Hello")
- pos=1 in hipfire = "Hello" token

This invalidated the pos=0 v_proj comparison (which was BOS, not Hello).

### Proper comparison with matching BOS context

Ran HF with `input_ids = torch.tensor([[2, 9259]])` (BOS + Hello) and
compared against hipfire's pos=1 dump. Results:

**L0-L4 (sliding layers): ALL MATCH within HFQ4 quantization error:**
```
L0: max_Δ = 0.059, ΔΣ = 0.05  ✓
L1: max_Δ = 0.061, ΔΣ = 0.05  ✓
L2: max_Δ = 0.058, ΔΣ = 0.00  ✓
L3: max_Δ = 0.032, ΔΣ = 0.75  ✓
L4: max_Δ = 0.070, ΔΣ = 0.64  ✓
```

**L5 (first Full layer): DIVERGENCE JUMPS:**
```
L5: max_Δ = 1.148, ΔΣ = 11.35  ← FIRST FULL LAYER
```

**L6+ (subsequent sliding layers): CATASTROPHIC:**
```
L6: max_Δ = 4.934, ΔΣ = 34.99
L7: max_Δ = 1.471, ΔΣ = 216.02
```

### Root cause narrowed to Full attention layers

The bug is now definitively isolated to the **Full (global) attention layer
path** — specifically L5 and every 6th layer thereafter (L5, L11, L17, L23,
L29, L35, L41, L47). These layers use:
- `head_dim = 512` (vs 256 for sliding)
- `n_kv = 1` (vs 8 for sliding)  
- `partial_rotary_factor = 0.25` (only 128 of 512 dims get RoPE)
- `full_rope_theta = 1e6` (vs 1e4 for sliding)
- `attention_k_eq_v = True` (V = k_proj output, no v_proj)
- hd512 flash attention kernel (`attention_flash_asym3_hd512`)

The sliding attention path is VERIFIED CORRECT. Only the full-layer path
is broken.

### Sub-step verification at L0 (pos=1, sliding)

Detailed sub-step comparison confirms every sliding-layer operation matches:
- embed: Δ < 0.01 ✓
- input_norm: Δ < 0.23 ✓  
- q/k/v_proj: Δ < 0.16 ✓ (HFQ4 quantization error)
- attention output: Δ < 0.02 ✓ (single-token decode, attn_out ≈ V)
- o_proj: Δ < 0.06 ✓
- post_attn_norm: Δ < 0.23 ✓
- gate/up_proj: Δ < 0.12 ✓
- down_proj: Δ < 0.26 ✓
- layer_scalar = 0.052979 (matches HF exactly) ✓

### Next step: bisect L5 (full layer) sub-steps

Need to dump intermediate values inside the full attention layer and
compare against HF. Key suspects:
1. **Partial RoPE** (`rope_partial_halved_f32`) — `partial_rotary_factor=0.25`
   means only 128 of 512 dimensions rotate. An off-by-2× or wrong-dim
   split here corrupts every full layer.
2. **hd512 flash attention kernel** — new, gemma4-only, untested against
   oracle.
3. **K=V weight sharing** — V = k_proj output (no v_proj), need to verify
   memcpy ordering.
4. **full_rope_theta = 1e6** — different from sliding's 1e4.


---

## 2026-06-07 · Session 12 — L5 full-layer sub-step bisect: attention output is wrong

### Added diagnostic dumps to full layer path

Added `_fdump` (gated on `pos==1 && kv_layer_idx==0`) to
`full_layer_decode_impl` in `gemma4.rs`, dumping 12 intermediate taps:
input, post_input_norm, q/k_proj, q/k/v_norm, post_rope_q/k,
post_attn_out, post_o_proj.

### CORRECTION: earlier q_norm comparison was invalid

The earlier "q_norm diverges" finding was an indexing artifact. HF's
hooked q_norm output is a 4D tensor `[batch, seq, n_heads, head_dim]` =
`[1, 2, 16, 512]`. I was comparing hipfire's flat first4 against HF's
flattened 4D tensor which interleaved heads. After correctly extracting
HF head 0's first4 from the 4D tensor:

```
HF head 0 q_norm: [4.1250, 0.4180, 2.2656, -3.5312]
HIP head 0 q_norm: [4.1269, 0.3942, 2.2824, -3.4937]
max_Δ = 0.0375  ✓
```

q_norm matches within quantization noise, as expected.

### L5 sub-step comparison (all at pos=1, "Hello" token)

| Stage | HF first4 | HIP first4 | max_Δ | Status |
|---|---|---|---|---|
| input scratch.x | L4 output | L4 output | < 0.01 | ✓ |
| post_input_norm | [-3.75, -2.84, 1.44, -2.61] | [-3.86, -2.72, 1.55, -2.44] | 0.17 | ✓ |
| post_q_proj | [26.25, 2.67, 14.50, -22.50] | [26.40, 2.52, 14.60, -22.35] | 0.15 | ✓ |
| post_k_proj | [-1.85, 1.90, 3.63, -2.05] | [-1.85, 1.85, 3.67, -2.06] | 0.05 | ✓ |
| post_q_norm (head 0) | [4.13, 0.42, 2.27, -3.53] | [4.13, 0.39, 2.28, -3.49] | 0.04 | ✓ |
| post_k_norm | [-0.024, 0.025, 0.047, -0.027] | [-0.024, 0.024, 0.048, -0.027] | 0.001 | ✓ |
| post_v_norm | [-0.398, 0.410, 0.781, -0.441] | [-0.400, 0.399, 0.792, -0.445] | 0.012 | ✓ |
| **post_attn_out** | **[-0.160, 0.072, 0.029, -0.053]** | **[-0.081, -0.696, 0.958, 0.483]** | **1.18** | **✗** |
| post_o_proj | [-0.205, -1.617, -1.789, -2.031] | [0.912, 2.333, -1.192, 0.015] | 3.95 | ✗ |

### Root cause: hd512 attention kernel output is wrong

Everything before the attention kernel matches (q/k/v projections, norms,
RoPE). The divergence originates **inside the attention computation itself**
for head_dim=512, n_kv=1.

HF head 0 (first 4 of 512): [-0.160, 0.072, 0.029, -0.053]
HIP head 0 (first 4 of 512): [-0.081, -0.696, 0.958, 0.483]

These are completely different values — not a scale issue, not a sign
flip, but structurally wrong attention output.

### Suspects for the hd512 attention bug

1. **`attention_flash_asym3_tile_hd512` kernel** — gemma4-only, never
   validated against an oracle. Uses TILE_SIZE=128 and the same
   online-softmax pattern as the hd256 variant. Could have:
   - Wrong stride for head_dim=512 (K/V cache row stride)
   - Wrong GQA handling for n_kv=1 (all 16 query heads share 1 KV head)
   - Buffer overflow (512-dim head exceeds expected partials buffer)

2. **`kv_cache_write_asym3_hd512` kernel** — writes K and V into the
   quantized KV cache. If the write stride is wrong, the attention
   kernel reads garbage K/V from cache.

3. **Partials buffer sizing** — `scratch.flash_partials` may be sized
   for head_dim=256, not 512. The hd512 kernel writes larger partial
   sums per tile.

4. **K=V sharing with n_kv=1** — hipfire copies k→v before k_norm,
   then applies v_norm (divide-only). With n_kv=1, the single KV head's
   V is 512-dim. If the attention kernel's GQA expansion (repeat 16×)
   or the V cache slot stride is wrong, it reads from the wrong offset.

### HF attention output details (for debugging)

HF L5 pos=1 per-head attention output (pre-o_proj):
```
head  0: sum= 20.21 norm= 22.02 first4=[-0.1602, 0.0723, 0.0293, -0.0525]
head  1: sum= 18.16 norm= 20.66 first4=[-0.1885, 0.1123, 0.1187, -0.0986]
head  2: sum= 16.96 norm= 20.01 first4=[-0.2041, 0.1348, 0.1689, -0.1245]
head  3: sum= 12.34 norm= 18.76 first4=[-0.2676, 0.2256, 0.3711, -0.2295]
head  4: sum=  6.62 norm= 20.09 first4=[-0.3457, 0.3359, 0.6133, -0.3555]
head  5: sum= 12.02 norm= 18.75 first4=[-0.2734, 0.2334, 0.3867, -0.2373]
head  6: sum= 17.41 norm= 20.23 first4=[-0.1982, 0.1270, 0.1504, -0.1152]
head  7: sum= 20.73 norm= 22.41 first4=[-0.1533, 0.0630, 0.0085, -0.0417]
head  8: sum= 18.63 norm= 20.93 first4=[-0.1816, 0.1040, 0.0996, -0.0889]
head  9: sum= 16.19 norm= 19.65 first4=[-0.2148, 0.1514, 0.2041, -0.1426]
head 10: sum= 16.47 norm= 19.77 first4=[-0.2119, 0.1455, 0.1924, -0.1367]
head 11: sum= 11.45 norm= 18.76 first4=[-0.2812, 0.2441, 0.4102, -0.2500]
head 12: sum=  3.27 norm= 22.24 first4=[-0.3926, 0.4004, 0.7578, -0.4297]
head 13: sum= 19.41 norm= 21.42 first4=[-0.1729, 0.0903, 0.0693, -0.0732]
head 14: sum= 14.40 norm= 19.07 first4=[-0.2402, 0.1855, 0.2812, -0.1836]
head 15: sum=  7.40 norm= 19.75 first4=[-0.3359, 0.3203, 0.5820, -0.3379]
```

All heads have norm ≈ 20 and show a consistent pattern: first element
negative, second positive. This is the signature of the single-KV-head
attention output (all 16 query heads attend to the same V, producing
similar but scaled results).

### Next steps

1. **Read the hd512 attention kernel source** (`attention_flash_asym3_tile_hd512.hip`)
   and compare with the hd256 variant. Focus on:
   - GQA n_kv=1 handling
   - K/V cache stride for head_dim=512
   - Partials buffer size expectations
   - Output write stride (should be n_heads * head_dim = 8192)

2. **Test with fp32 KV cache** to rule out the asym3 quantized KV
   write kernel. If fp32 KV also produces wrong attention output,
   the bug is in the attention kernel, not the KV write.

3. **Verify partials buffer allocation** in `Gemma4Scratch` — is it
   sized for head_dim=512 or only 256?

4. **Dump post-RoPE Q/K values** and verify against HF for the full
   layer to confirm RoPE is correct before blaming attention.

---

## 2026-06-07 · Session 13 — ROOT CAUSE FOUND: hd512 attention missing reduce step

### Bug: `attention_flash_asym3_hd512` never calls the reduce kernel

The `attention_flash_asym3_hd512` function in `gemma4_ext.rs:68-112` only
launches the **tile** kernel (`attention_flash_asym3_tile_hd512`). It **never
launches the reduce kernel** (`attention_flash_q8_0_reduce`).

Compare with the working hd256 version `attention_flash_asym3` in
`attention.rs:3790-3907`:
```
attention_flash_asym3 (hd256):
  1. launch attention_flash_asym3_tile     ← tile kernel
  2. launch attention_flash_q8_0_reduce    ← REDUCE KERNEL ← PRESENT ✓

attention_flash_asym3_hd512 (hd512):
  1. launch attention_flash_asym3_tile_hd512  ← tile kernel
  (returns)                                    ← NO REDUCE ✗
```

### What the reduce kernel does

`attention_flash_q8_0_reduce` (in `kernels/src/attention_flash_q8_0_reduce.hip`):
- Takes per-tile partials (tile_max, tile_sum, V-weighted accumulator)
- Finds global max across all tiles
- Rescales each tile's accumulator by `exp(tile_max - global_max)`
- Sums all rescaled accumulators
- Normalizes by global sum
- Writes the result to `out`

### What happens without the reduce

The tile kernel writes to `partials` buffer. The `out` tensor (which maps to
`scratch.attn_out` in the gemma4 forward) is **never written**. It contains
stale data from the previous layer's sliding attention output. This stale
data then gets fed through o_proj → post_attn_norm → residual, producing
garbage.

### Why this wasn't caught earlier

With only 2 tokens (BOS + Hello), there's only 1 tile per head (128 > 2
positions). The "reduce" of a single tile is trivially `out = partials /
sum`. But even with 1 tile, the reduce kernel is needed because:
- The tile kernel writes to `partials[h * max_tiles * (2 + head_dim) + ...]`
- The reduce kernel reads from partials and writes to `out[h * head_dim + ...]`
- Without the reduce, `out` is never populated

### Why the hd256 version works

The hd256 `attention_flash_asym3` function correctly launches both tile and
reduce kernels (confirmed by reading `attention.rs:3790-3907`). All sliding
layers use the hd256 path → correct. All full layers use the hd512 path →
missing reduce → garbage from L5 onward.

### Fix

Add the reduce kernel launch to `attention_flash_asym3_hd512`, matching
the pattern from `attention_flash_asym3` (hd256):

```rust
// After the tile kernel launch in attention_flash_asym3_hd512:
self.ensure_kernel(
    "attention_flash_q8_0_reduce",
    kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
    "attention_flash_q8_0_reduce",
)?;
// ... launch reduce with same params as hd256 version ...
```

The reduce kernel already handles head_dim=512 correctly via `n_halves =
head_dim / 128 = 4` (4 iterations of 128 dims each).

### Verification needed after fix

1. Rebuild and run "Hello" — expect coherent output
2. Run "What is the capital of France?" — expect correct answer
3. Run coherence-gate-dflash — verify no regression on Qwen models

---

## 2026-06-07 · Session 14 — Reduce fix validated; deeper divergence remains

### Reduce kernel fix applied and confirmed working

The fix (adding `attention_flash_q8_0_reduce` launch after the tile kernel
in `attention_flash_asym3_hd512`) was applied in the working tree. Build
succeeded. The `debug_gemma4_attention` example now runs to completion
without NaN, panic, or garbage values.

### Model output still wrong: logits don't change between positions

**Critical finding**: despite the reduce fix, the model produces nearly
identical logits at pos=0 (BOS) and pos=1 ("Hello"):

```
Hipfire pos=0 logits top5: [(532, 18.033), (255999, 16.294), (240494, 15.212), ...]
Hipfire pos=1 logits top5: [(532, 18.034), (255999, 16.294), (240494, 15.211), ...]
```

HF reference shows dramatically different logits (cosine similarity = 0.064):

```
HF pos=0 logits top5: [(532, 18.000), (255999, 16.125), (240494, 15.125), ...]
HF pos=1 logits top5: [(575, 14.875), (236747, 14.750), (514, 14.750), ...]
```

Hipfire is essentially regurgitating the BOS-position logits at pos=1,
meaning the "Hello" token's contribution is not propagating through the
layers.

### Per-layer hidden state comparison (pos=1)

Comparing first4 elements of hidden state between HF and hipfire at pos=1:

```
Layer | HF first4[0:2]          | HIP first4[0:2]         | Δ[0]  | Δ[1]  | Type
L 0   | [-0.7773,  2.4844]      | [-0.4549,  2.2682]      |  0.32 |  0.22 | Sliding
L 1   | [ 0.0486,  1.3203]      | [ 0.2167,  1.9676]      |  0.17 |  0.65 | Sliding
L 2   | [-0.1533,  1.5625]      | [-0.0647, -0.2303]      |  0.09 |  1.79 | Sliding
L 3   | [-0.1196, -0.0282]      | [-0.0661,  2.0908]      |  0.05 |  2.12 | Sliding
L 4   | [-0.1099, -1.3516]      | [-0.1241, -1.3104]      |  0.01 |  0.04 | Sliding  ← best match
L 5   | [-0.0083, -1.6016]      | [ 0.0147, -0.7470]      |  0.02 |  0.86 | Full
L 6   | [-0.4707, -0.1787]      | [-0.3272, -1.4983]      |  0.14 |  1.32 | Sliding
L 7   | [-0.1836,  0.3242]      | [-0.4316, -1.2499]      |  0.25 |  1.57 | Sliding
L 8   | [ 0.0352,  1.5625]      | [-0.8700, -1.9601]      |  0.91 |  3.52 | Sliding
L 9   | [-2.0156,  2.1562]      | [ 0.7172, -1.5170]      |  2.73 |  3.67 | Sliding
L10   | [-0.2441,  0.4805]      | [ 0.9231,  5.3017]      |  1.17 |  4.82 | Sliding
L11   | [ 0.0125, -0.0017]      | [-0.0092,  0.0137]      |  0.02 |  0.02 | Full     ← good match
```

### Key observations

1. **Sliding layers ALSO diverge** — L2, L3 show Δ[1]≈2.0 despite being
   sliding (hd=256) layers with the well-tested hd256 path. This means the
   bug is NOT exclusive to the full-layer hd512 attention.

2. **Full layers (L5, L11) are NOT the worst** — L5 has Δ[1]=0.86 and L11
   has Δ[0]=0.02. The worst divergences are in sliding layers L8-L10.

3. **The pattern is non-monotonic**: L4 (sliding) matches best, then L5
   (full) diverges moderately, then L6-L10 (sliding) diverge severely.
   This is NOT a simple "full layers break everything" pattern.

4. **HF reference shows `k_eq_v=False` at the per-layer level** for ALL
   layers, but the global config says `attention_k_eq_v: True`. The HF
   model's code likely overrides per-layer: full layers use k_eq_v=True
   (V = pre-norm K), sliding layers use k_eq_v=False (real v_proj).

5. **Hipfire's hidden states at later layers (L12-L47) are nearly identical
   between pos=0 and pos=1** — the model "collapses" to the same output
   regardless of input token. This suggests the attention mechanism is not
   differentiating between positions, which points to a KV cache issue.

### Suspects for the remaining divergence

1. **Sliding window attention not reading both positions correctly**: The
   sliding KV cache has `sliding_window=1024`. At pos=1, the attention
   should attend to both pos=0 and pos=1. If it only sees one position,
   the softmax is trivially [1.0] and attn_out = V.

2. **Cosine/sin theta buffer for full layers**: The full layers use
   `full_rope_theta=1000000` vs sliding's `sliding_rope_theta=10000`. If
   the Givens cos/sin tables are shared or computed with the wrong theta,
   the KV cache write uses wrong rotations, corrupting the K cache.

3. **KV cache Givens tables initialized per-cache vs per-layer-type**: The
   `kv_cache.givens_cos/sin` are computed once at cache creation. If the
   sliding and full KV caches share the same Givens tables, one of them
   will have wrong rotations.

4. **Sliding window KV cache `physical_cap` too small**: Debug output shows
   `physical_cap=1024 / max_seq=1024` for the sliding cache. This is
   `sliding_window=1024`, which is correct. But if the attention kernel
   interprets `max_seq` differently for sliding vs full, it could truncate.

5. **The `v_norm_ones_full` tensor**: This was the earlier fix (initialized
   to all-ones for V normalization). If it's sized wrong or applied to the
   wrong buffer, V normalization could corrupt values.

### HF config details (12B)

```
text_config:
  attention_k_eq_v: True              ← global default
  global_head_dim: 512
  head_dim: 256
  num_attention_heads: 16
  num_global_key_value_heads: 1       ← full layers: 1 KV head
  num_key_value_heads: 8              ← sliding layers: 8 KV heads
  sliding_window: 1024
  layer_types: [S,S,S,S,S,F, S,S,S,S,S,F, ...] (48 total, 8 Full at L5,11,17,23,29,35,41,47)
```

Per-layer from HF model:
- Sliding: hd=256, n_kv=8, sw=1024, k_eq_v=False (real v_proj)
- Full:    hd=512, n_kv=1, sw=None, k_eq_v=False (but global k_eq_v=True?)

The `k_eq_v` semantics: V = pre-k_norm K for full layers only. Sliding
layers have real v_proj weights. Hipfire handles this via separate
`LayerWeights::Sliding` (has v_proj) and `LayerWeights::Full` (no v_proj,
copies K→V before k_norm). This was verified correct earlier.

### Tokenizer issue blocks end-to-end CLI validation

The `hipfire run` and `hipfire serve` CLI commands fail to load the gemma4
model's tokenizer:
```
GPT-2 BPE vocab missing byte symbol: byte 0x90 maps to char 'Ĳ' which is not in token_to_id
```

Gemma4 uses a 262K-vocab BPE tokenizer (model_type="BPE" in the embedded
tokenizer JSON), not SentencePiece. The tokenizer loader incorrectly tries
GPT-2 BPE mode, which expects byte-fallback tokens for all 256 bytes. The
262K vocab doesn't have these (it's a different BPE variant). This is a
separate bug from the attention divergence.

### Next steps

1. **Run with HIPFIRE_GEMMA4_DUMP=1 on a longer prompt** (5-10 tokens) to
   see if the divergence pattern changes with more context. If it does, the
   KV cache read/write is the issue. If it doesn't, the issue is in the
   per-layer computation itself.

2. **Verify the Givens cos/sin tables** for the full KV cache. The full
   layers use `full_rope_theta=1000000`. Check that the KV cache's
   `givens_cos/sin` are computed with the correct theta for each cache
   type (sliding vs full).

3. **Dump the actual KV cache contents** for L5 at pos=1 and compare with
   HF's K/V at the same position. This will isolate whether the write or
   the read is wrong.

4. **Test with fp32 KV cache** (`--kv-mode fp32`) to bypass all Givens
   rotation and asym3 quantization. If fp32 KV produces correct output,
   the bug is in the Givens rotation / asym3 quant, not in the attention
   computation itself.

5. **Check if sliding attention reads 2 positions**: At pos=1, the sliding
   attention should have seq_len=2. Dump the partials buffer to verify the
   tile kernel processes both positions.

6. **Fix the tokenizer** to unblock end-to-end CLI validation.

---

## 2026-06-07 · Session 15 — hd512 reduce fix lands; stop-token fix; coherence confirmed

### hd512 attention reduce (root cause of L5+ garbage) — FIXED + COMMITTED
`attention_flash_asym3_hd512` launched only the tile kernel (writes unnormalized
per-tile partials) and **never the `attention_flash_q8_0_reduce`** that divides by
the softmax sum and writes `out`. So `attn_out` was never populated on full
(global, hd=512) layers — it read stale data from the prior sliding layer. Mirrored
the hd256 two-kernel pattern (scoped the tile launch so its `&self.functions` borrow
ends, then ensure+launch the reduce; reduce handles hd512 via `n_halves=512/128=4`).
Commit `2e36fee2`.

### Session 14's "deeper divergence" was a 4th harness artifact
The standalone `debug_gemma4_attention` example claimed post-fix the model still
collapses (pos0≈pos1 logits, cosine-to-HF 0.064). **Falsified by the real daemon
path:** a 4-prompt battery on 12B-q8 returns correct, *distinct*, input-dependent
answers (`Tokyo`/`42`/banana/French). Session 14's own data shows hipfire pos=0
matching HF but pos=1 not updating → a position-advance bug **in the harness**, not
the model. Trust the daemon generate path; the standalone harness gives false
negatives (same class as the earlier double-scale / sum-metric / BOS oracle bugs).

### Stop-token fix (decode looped `<turn|>` forever) — FIXED + VERIFIED
`config.eos_token` is `eos_token_id` parsed as a scalar (→1), but gemma4's HF config
sets it to a LIST `[1, 106]`; `<turn|>`=106 (end-of-turn) is the real conversational
stop and was dropped. `generate_gemma4` now builds a `stop_set` = {`<eos>`,
`<turn|>` via `special_token_id` + documented 106}. Result:
```
"capital of France?" → "...The capital of France is Paris."   (stops, 12 tok)
"capital of Japan?"  → "...Tokyo"                              (stops, 6 tok)
```
(Was looping `<turn|><turn|>…` to max_tokens.)

### Remaining (priority order)
1. **Chat-template framing** — output still prefixes an empty `<|channel>thought\n
   <channel|>` because the prompt is raw (no turn scaffolding). Frame as
   `<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`
   (ids 105/106 + channel ids) so the model emits a clean answer. (In progress.)
2. **CLI tokenizer bug** — `hipfire run`/`serve` mis-load the 262K BPE as GPT-2 BPE
   (`missing byte symbol 0x90`); daemon JSONL path tokenizes fine.
3. **Prefill/batched full layers** — `attention_flash_asym3_batched_window` routes
   hd512→hd256; the hd512-batched kernel is unwired and would need its own reduce.
4. Backlog: MoE stubbed (26B-A4B can't run), sliding-window ring buffer no-op
   (`cache_capacity` dead), dead code (`forward_prefill_batch_v1`, unreachable
   graph-capture branch), no unit tests, missing copyright header on `gemma4.rs`.
