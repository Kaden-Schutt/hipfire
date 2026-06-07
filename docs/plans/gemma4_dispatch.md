# Gemma 4 dispatch-unification migration plan

Branch: `feat/dispatch-unification-gemma4` (off `upstream/integration/dispatch-unification`
@ `a7902234` — Ship 5.2 tip, 2026-06-07).

**Adversarially reviewed:** 2026-06-07 by Gemini 3.5 Flash, Claude Opus 4.8, glm5.
Consolidated findings in `findings/gemma4_dispatch_plan_consolidated_rev.md`.
All accepted findings incorporated below.

**Config-audited:** 2026-06-07 against real BF16 safetensors + configs from HuggingFace
(`google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it`). Corrections applied inline.

---

## 1 · Context

Two branches converge here:

| | **gemma4** (`feat/gemma4-128k-ring-buffer`) | **dispatch-unification** (`integration/dispatch-unification`) |
|---|---|---|
| **Base** | Merged into master (`9b206438`) | **1674 commits** ahead of master, Ships 1–5.2 |
| **New crate** | `hipfire-arch-gemma4` (2654-line `gemma4.rs`) | `hipfire-dispatch` (centralized kernel families) |
| **Dispatch pattern** | Old-style: direct `weight_gemv()`, `gpu.kv_cache_write_*()`, `gpu.attention_flash_*()`, inline MoE loops | New-style: `execute_steps()`, `Step::*`, `GemvFamily`, `AttentionFamily`, `MoeFamily`, `FUSED_TABLE` |
| **Kernel files** | 12 new HIP files + modifications to 13 existing | All existing kernels; no gemma4-specific ones |
| **rdna-compute** | Adds `cache_capacity` params, hd512 attn, MoE kernels, `rope_partial_halved`, `logit_softcap` | Completely rewritten `attention.rs` (9615 lines); has own dispatch helpers |
| **Runtime wiring** | Daemon: `Gemma4Config`/`Weights`/`Scratch` fields, `arch_id=7` dispatch, cross-request KV caching, SPM-BPE tokenizer | Has LFM2MoE, MiniMax, DeepSeek4, Qwen2, DotsOCR — **no gemma4** |

### 1.1 · Gemma 4 model variants (from real configs)

BF16 safetensors land in `/local/models/google/` (four variants). Only configs
examined below — MoE shards are incomplete and being re-downloaded.

| Variant | HF repo | Layers | `hidden_size` | `intermediate_size` | MoE | Heads (Q / KV / global-KV) | Vision | Audio |
|---|---|---|---|---|---|---|---|---|
| **12B-it** (dense) | `google/gemma-4-12B-it` | 48 | 3840 | 15360 | No | 16 / 8 / 1 | ✓ (27L, hd=72) | ✓ (hd=640) |
| **26B-A4B-it** (MoE) | `google/gemma-4-26B-A4B-it` | 30 | 2816 | 2112 | 128 experts, k=8, `moe_intermediate=704` | 16 / 8 / 2 | ✓ (27L, hd=72) | No |
| **31B-it** (dense) | `google/gemma-4-31B-it` | TBD | TBD | TBD | No | TBD | TBD | TBD |
| **E4B-it** (Any-to-Any) | `google/gemma-4-E4B-it` | TBD | TBD | TBD | TBD | TBD | ✓ | ✓ |
| **E2B-it** (Any-to-Any) | `google/gemma-4-E2B-it` | TBD | TBD | TBD | TBD | TBD | ✓ | ✓ |

**Shared across all variants:**
- `head_dim=256` (sliding layers), `global_head_dim=512` (full-attention layers)
- `sliding_window=1024`, `max_position_embeddings=262144`
- `hidden_activation=gelu_pytorch_tanh` (SwiGLU activation in dense + MoE FFN)
- `final_logit_softcapping=30.0`
- `tie_word_embeddings=true`, `vocab_size=262144`
- `rms_norm_eps=1e-6`
- RoPE: sliding uses `rope_theta=10000` (default); full uses `rope_theta=1e6` (proportional, `partial_rotary_factor=0.25`)
- `attention_k_eq_v=true` (K and V share same per-head dimension)
- Per-head Q/K norms (`q_norm`, `k_norm` weights) — applied after projection
- `layer_scalar` per layer (learned scalar multiplier)
- Layer pattern: 5× sliding → 1× full → 5× sliding → 1× full … (6-layer cadence)

**12B vs 26B-A4B `model_type` difference:** The 12B uses `"gemma4_unified"`
(with audio + vision + text towers, `transformers_version: "5.10.0.dev0"`);
the 26B-A4B uses `"gemma4"` (text + vision, no audio,
`transformers_version: "5.5.0.dev0"`). The `text_config` sub-object holds
the language-model parameters for both; the `model_type` field inside
`text_config` is `"gemma4_unified_text"` (12B) / `"gemma4_text"` (26B).

**Per-layer weight structure (26B-A4B, from `model.safetensors.index.json`):**
```
layer.{i}.input_layernorm.weight
layer.{i}.self_attn.{q,k,v,o}_proj.weight
layer.{i}.self_attn.{q,k}_norm.weight       # Per-head norms (post-projection)
layer.{i}.post_attention_layernorm.weight
layer.{i}.pre_feedforward_layernorm.weight   # Dense FFN pre-norm
layer.{i}.pre_feedforward_layernorm_2.weight # MoE pre-norm
layer.{i}.mlp.{gate,up,down}_proj.weight     # Dense FFN
layer.{i}.post_feedforward_layernorm.weight  # Dense FFN post-norm
layer.{i}.post_feedforward_layernorm_1.weight # MoE post-norm (route A?)
layer.{i}.post_feedforward_layernorm_2.weight # MoE post-norm (route B?)
layer.{i}.experts.gate_up_proj               # 128-expert stacked gate+up
layer.{i}.experts.down_proj                  # 128-expert stacked down
layer.{i}.router.proj.weight                 # Router projection
layer.{i}.router.scale                       # Router logit scale
layer.{i}.router.per_expert_scale            # Per-expert scaling factor
layer.{i}.layer_scalar                       # Learned scalar multiplier
```

**Parallel dense + MoE structure (confirmed):** Every MoE layer runs a
dense FFN (gate/up/down with `gelu_tanh` activation) in parallel with the
routed expert FFN, then sums the outputs. Norm paths are separate
(`pre_feedforward_layernorm` for dense, `pre_feedforward_layernorm_2` for
MoE router input; `post_feedforward_layernorm`/`_1`/`_2` for outputs).
This is structurally different from qwen35 A3B's serial MoE→shared-expert
pattern — see §7.

### 1.2 · Model artifact status

All models land in `/local/models/google/`. Status as of 2026-06-07:

| Variant | Config | Tokenizer | Safetensors |
|---|---|---|---|
| 12B-it | ✓ | ✓ (GemmaTokenizer, SPM-BPE, 262K vocab, 32 MB) | Incoming |
| 26B-A4B-it | ✓ | ✓ | **Incomplete** — only shard 2/2 present; needs re-download |
| 31B-it | — | — | Incoming |
| E4B-it | — | — | Incoming |
| E2B-it | — | — | Incoming |

**Tokenization notes:** GemmaTokenizer is a SentencePiece BPE tokenizer with
262,144 tokens, `<bos>` prepend (id=2), `▁`-space prefix normalization.
Special tokens: `<|turn>` (105), `<turn|>` (106, also an EOS token),
`<|image>` (255999 BOI / 258880 image_token_id),
`<image|>` (258882 EOI), `<|audio>` (256000 BOA / 258881),
`<audio|>` (258883 EOA), `<|tool_call>` / `<tool_call|>` /
`<|tool_response>` / `<tool|>`, `<|channel>` / `<channel|>`,
`<|video|>`, `<|think|>`. The `response_schema` in `tokenizer_config.json`
uses regex-based tool-call parsing with `gemma4-tool-call` parser type.

### 1.3 · Conflict surface

13 crate files touched by **both** branches:

```
crates/hipfire-quantize/src/main.rs
crates/hipfire-runtime/Cargo.toml
crates/hipfire-runtime/examples/daemon.rs
crates/hipfire-runtime/src/hfq.rs
crates/hipfire-runtime/src/llama.rs
crates/hipfire-runtime/src/tokenizer.rs
crates/rdna-compute/src/attention.rs
crates/rdna-compute/src/dispatch.rs
crates/rdna-compute/src/gemv.rs
crates/rdna-compute/src/graph.rs
crates/rdna-compute/src/kernels.rs
crates/rdna-compute/src/moe.rs
crates/rdna-compute/src/norm.rs
```

The dispatch-unification branch has **deleted or rewritten** many dispatch
helpers gemma4 depends on. Merging dispatch-unification *into* gemma4 would
create catastrophic conflicts (1674 commits × 457 files × major refactors to
`attention.rs`, etc.).

**Decision: carry gemma4 piece by piece into dispatch-unification.** The gemma4
model crate is entirely new — zero conflicts with the dispatch crate. The old
`weight_gemv()` / `weight_gemm()` / `weight_gemv_prerotated()` /
`weight_gemv_residual()` / `weight_gemv_swiglu_residual()` still exist on the
dispatch branch as backward-compatible wrappers (they already route through
`GemvFamily` internally), so we can scaffold with old-style dispatch first,
then migrate incrementally.

**Backward compatibility:** not a concern. Gemma4 models have never been used
in production. Old `arch_id=7` artifacts require re-quantization with the new
`arch_id=12`; no runtime shim is needed.

---

## 2 · Phase 0 — prerequisites (before any gemma4 code lands)

These items thread gemma4's requirements through the shared dispatch surface.
They must complete and pass validation on existing models before gemma4 code
enters the branch.

### 0a · Thread `cache_capacity` through the KV + attention dispatch surface

Gemma4's sliding-window layers use ring-buffer KV caches where
`slot = pos % cache_capacity` (with `cache_capacity = sliding_window = 1024`)
rather than `slot = pos` (the identity used by every model today).

Currently `cache_capacity` does not exist on the dispatch branch — zero hits
in `attention.rs`, zero in any kernel HIP file, zero in `KvTierInputs` or
`AttnParams`. **The signature of `kv_cache_write_asym3_fused` itself lacks the
parameter** — adding it breaks the dispatch crate's attention path for all archs.

**Decision (review consensus, all 3 reviewers): modify the shared surface.**
Add `cache_capacity: u32` to every KV-write and flash-attention dispatch
function in `rdna-compute/src/attention.rs`, to `KvTierInputs`, to
`KvTierPlan`, and to `AttnParams`. All existing callers pass `physical_cap`
(the non-wrapping identity). Kernel HIP files get a default-zero
`cache_capacity` argument. This is a **mechanical, file-wide parameter addition**
to ~40 kernel launch sites and 2 struct definitions — no behavioral change
for existing archs.

**Blast radius:**
- `crates/rdna-compute/src/attention.rs` — every `kv_cache_write_*` and
  `attention_flash_*` method signature
- `kernels/src/` — every asym2/3/4, q8_0, fwht, and batched KV kernel
- `crates/hipfire-dispatch/src/families/attention.rs` — `dispatch_kv_write`
  and `dispatch_attend` call sites
- `crates/hipfire-dispatch/src/families/kv_tier.rs` — `KvTierInputs`,
  `KvTierPlan`, `KvTierPlan::derive`
- `crates/hipfire-runtime/src/llama.rs` — `prefill_forward` KV write calls
  (currently pass `0` explicitly for `cache_capacity` on existing call sites)

**Gate:** coherence-gate pass on qwen35 MQ4 + A3B MQ4 + llama Q4K before
proceeding.

### 0b · Add `head_dim` routing to `KvTierInputs` and attention resolution

Gemma4 uses both `head_dim=256` (sliding layers) and `head_dim=512` (full
layers) within the same model. The attention dispatch must select hd512
kernels for full layers and hd256 for sliding layers.

Add `head_dim: usize` to `KvTierInputs`. Plumb through `ShapeInfo` (which
already has a `head_dim` field, but it's not currently used by attention
resolution). Register hd512 kernel variants under the **existing**
`KernelKey` arms (`KvWriteAsym3`, `AttnFlashAsym3`, etc.) with
`ShapePredicate::HeadDimEq(512)`. In `dispatch_kv_write`/`dispatch_attend`,
branch on `io.head_dim` to launch the hd512 kernel when needed.

**No new `KernelKey` variants needed.** This keeps the key enum clean —
quantization tier and head dimension are orthogonal dispatch axes.

### 0c · Quantize gemma4 model weights + gate script rows

Before any phase gate can run, we need:
1. Wait for complete BF16 safetensors to land in `/local/models/google/`
   (26B-A4B is incomplete — shard 2 only; needs re-download. 12B, 31B,
   E4B, E2B are incoming.)
2. Quantize gemma4 model weights to HFQ/MQ4 format via `hipfire-quantize`
   with `arch_id=12`
3. Symlink quantized artifacts into `~/.hipfire/models/`
4. Add gemma4 `(model|id|prompt|max)` rows to `coherence-gate.sh` and
   `speed-gate.sh` matrices

The scripts currently have zero gemma4 rows; `--gemma4` flags do not exist.
Initial target: 12B dense (simplest forward path, no MoE).

### 0d · Verify Phase 0 doesn't regress existing models

```bash
./scripts/coherence-gate.sh          # dense models
./scripts/coherence-gate.sh --full   # + A3B MoE
./scripts/coherence-gate-dflash.sh   # spec-decode
cargo test -p hipfire-dispatch
cargo test -p hipfire-dispatch-tests
```

---

## 3 · Overall sequence

```
Phase 0 ── prerequisites (cache_capacity, head_dim, gate scripts)
    │
Phase 1 ── scaffold (old dispatch, compiles, decodes coherently)
    │
Phase 2 ── migrate decode to execute_steps / AttentionFamily
    │
Phase 3 ── migrate prefill to GemmFamily / AttentionFamily
    │
Phase 4 ── migrate MoE to MoeFamily / run_moe_decode_gemma4
    │
Phase 5 ── validation (coherence, perf A/B, coverage gate)
```

Each phase gate: `cargo check --workspace` clean + coherence-gate pass on
gemma4 weights, measured independently before proceeding.

---

## 4 · Phase 1 — Scaffold

**Goal:** bring gemma4 crate, kernel files, and rdna-compute additions over
using old dispatch patterns. Gemma4 decodes coherently on gfx1100 + gfx1201.

**Exit criterion (mergeable checkpoint):** gemma4 decodes coherently at
≥90% of the gemma4 branch's tok/s baseline. Phase 1 is intended for merge
into the integration branch; Phases 2–5 follow as incremental PRs.

### 1a · Add `hipfire-arch-gemma4` crate to workspace

Most files can be ported directly from `feat/gemma4-128k-ring-buffer`:
- `crates/hipfire-arch-gemma4/Cargo.toml` — depends on `hipfire-runtime`, `rdna-compute`, `hip-bridge`, `hipfire-dispatch`
- `crates/hipfire-arch-gemma4/src/lib.rs` — re-exports
- `crates/hipfire-arch-gemma4/src/arch.rs` — `Architecture` trait impl. **Update `arch_id()` from `7` → `12`.**
- `crates/hipfire-arch-gemma4/src/gemma4.rs` — forward pass (old-style dispatch initially)
- `crates/hipfire-arch-gemma4/src/gemma4_vision.rs` — vision tower placeholder (out of scope for Ships 1–5)
- `crates/hipfire-arch-gemma4/examples/` — bench + smoke + verify tools
- Register in workspace `Cargo.toml`

### 1b · Port gemma4 kernel files

**Only genuinely new kernels** (not already on the dispatch branch):

| Kernel file | Used by |
|---|---|
| `attention_flash_asym3_tile_hd512.hip` | Full-attention layers, decode |
| `attention_flash_asym3_tile_hd512_batched.hip` | Full-attention layers, prefill |
| `kv_cache_write_asym_k_givens3_hd512.hip` | Full-attention KV write, decode |
| `kv_cache_write_asym_k_givens3_hd512_batched.hip` | Full-attention KV write, prefill |
| `gemv_mq4g256_moe_gate_up_k8_indexed.hip` | MoE expert gate+up, decode |
| `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed.hip` | MoE expert down-proj, decode (HFQ4G128 weights) |
| `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed_batched.hip` | MoE expert down-proj, prefill (HFQ4G128) |
| `gemv_q8_0_moe_down_residual_scaled_k8_indexed.hip` | MoE expert down-proj, decode (Q8_0 weights) |
| `gemv_hfq4g256_moe_gate_up_bucketed.hip` | MoE prefill bucketed gate+up (HFQ4G256) |
| `gemv_hfq4g128_moe_down_residual_scaled_bucketed.hip` | MoE prefill bucketed down-proj (HFQ4G128) |
| `logit_softcap.hip` | Final logit softcap |
| `moe_bucket_build.hip` | Token→expert grouping for bucketed prefill |
| `rope_partial_halved.hip` | Proportional RoPE on full-attention layers |

**Existing kernel modifications** — the `cache_capacity` parameter threading
(Phase 0a) handles these. The gemma4 branch's kernel HIP modifications
(added `cache_capacity` param + slot = pos % capacity indexing) are absorbed
into the Phase 0a kernel edits and do NOT need separate porting.

### 1c · Add kernel declarations

In `crates/rdna-compute/src/kernels.rs`: declare each new kernel symbol from
§1b with its precompiled binary slice (follow the `ensure_kernel` lazy-load
pattern).

### 1d · Add gemma4 dispatch helpers to `rdna-compute/src/`

Only **net-new** dispatch functions. Symbols already present on this branch
are listed as "exists" and omitted.

**`attention.rs`** — no new functions needed after Phase 0a. The existing
`attention_flash_asym3` / `kv_cache_write_asym3_fused` etc. already have
`cache_capacity` threaded through. The hd512 path is selected by the
existing function body inspecting `head_dim` (routed via `dispatch_kv_write`
/ `dispatch_attend` in `AttentionFamily`).

**`dispatch.rs`** — net-new:
- `rope_partial_halved_f32()` — proportional partial RoPE (first 64 of
  512 dims rotate; rest are NoPE/identity). Kernel exists at `rope_partial_halved.hip`.

**`norm.rs`** — net-new:
- `logit_softcap_f32()` — final logit softcap: `tanh(x/cap)*cap`. Kernel
  exists at `logit_softcap.hip`.

Already present (omit from checklist): `gelu_tanh_f32` (norm.rs:2160),
`rmsnorm_batched` (norm.rs:71), `kv_cache_write_q8_0` (attention.rs:1327),
`embedding_lookup_hfq4g256` (embedding.rs:95).

**`gemv.rs`** — net-new:
- `gemv_mq4g256_moe_gate_up_k8_indexed()` — MoE expert gate+up, decode
- `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed()` — MoE down-proj, decode
- `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed_batched()` — MoE down-proj, prefill
- `gemv_q8_0_moe_down_residual_scaled_k8_indexed()` — MoE down-proj (Q8_0), decode
- `gemv_hfq4g256_moe_gate_up_bucketed()` — MoE prefill bucketed gate+up
- `gemv_hfq4g128_moe_down_residual_scaled_bucketed()` — MoE prefill bucketed down-proj

**`moe.rs`** — net-new:
- `moe_bucket_build()` — token→expert grouping kernel for bucketed prefill

### 1e · Wire gemma4 into daemon

In `crates/hipfire-runtime/examples/daemon.rs`:

**Imports:**
```rust
use hipfire_arch_gemma4::gemma4;
use hipfire_arch_gemma4::Gemma4;
```

**`LoadedModel` struct** — add fields:
```rust
gemma4_config: Option<gemma4::Gemma4Config>,
gemma4_weights: Option<gemma4::Gemma4Weights>,
gemma4_scratch: Option<gemma4::Gemma4Scratch>,
gemma4_kv_sliding: Option<llama::KvCache>,
gemma4_kv_full: Option<llama::KvCache>,
```

**`arch_id` dispatch** — add `12 => "gemma4"` at **every** `arch_id` match
site. Known sites (≥16):

| Line | Context | Gemma4 action |
|------|---------|--------------|
| :1854 | Main arch match | `12 => "gemma4"` |
| :1936 | Cache-capable check | Add `12` to matches (gemma4 has KV cache) |
| :2196 | Default temp/top_p | No special defaults (use generic) |
| :2327 | is_dots_ocr | `false` for gemma4 |
| :2343 | VL routing | Gemma4 has vision tower — gate like qwen35-VL |
| :2590 | Cursor rewind | Add `12 => rewind gemma4_kv_sliding + gemma4_kv_full` |
| :2686 | Reset dispatch | Add `12 => gemma4` arm |
| :2833–2902 | Generate dispatch | Add `12 => generate_gemma4(...)` |
| :3158, :3516 | VL-gating | Gemma4 vision tower — explicit treatment |

> ⚠️ **`arch_id=7` collision resolved:** The gemma4 branch originally claimed
> `arch_id=7`, but the dispatch-unification branch's `hipfire-arch-qwen2`
> already occupies 7. Current arch_id registry on this branch:
>
> | ID | Arch |
> |----|------|
> | 0  | LLaMA |
> | 5  | Qwen3.5 dense |
> | 6  | Qwen3.5/3.6 MoE (A3B) |
> | 7  | Qwen2 |
> | 8  | Dots-OCR (Qwen2-VL) |
> | 9  | DeepSeek V4 |
> | 10 | MiniMax-M2 |
> | 11 | LFM2MoE |
> | **12** | **Gemma4** |
>
> **Action items:**
> - `Gemma4::arch_id()` → 12 (in `arch.rs`)
> - `docs/architecture-ids.md` → add gemma4=12, backfill minimax=10 + lfm2moe=11
> - Re-quantize any gemma4 artifacts that carry `arch_id=7` → stamp `arch_id=12`
> - Old `arch_id=7` gemma4 files are incompatible; re-quantization required

**Model load path** — add `load_model_gemma4()` routing through
`Gemma4::config_from_hfq`, `load_weights`, `new_state`. Allocate dual KV
caches: `kv_sliding` sized at `sliding_window`, `kv_full` at `max_seq`.

**Generate path** — add `generate_gemma4()` calling `gemma4::forward_scratch()`
and `gemma4::forward_prefill_batch()`. Per-layer KV cache selection via
`config.layer_types[layer_idx]`:
- `LayerType::Sliding` → use `gemma4_kv_sliding`
- `LayerType::Full` → use `gemma4_kv_full`

**Cross-request KV caching** — gemma4 (unlike qwen35) keeps `seq_pos` and
`conversation_tokens` across `reset` requests, using LCP-match to decide
whether to reuse cached KV. Port from gemma4 branch.

**Prompt framing** — gemma4 uses `<|turn|>` / `<turn|>` special tokens
(ids 105/106 in the shipped tokenizer). Wire through the existing prompt-frame
path (or special-case until trait-dispatched framing lands).

### 1f · Wire gemma4 into `llama.rs`

- `weight_gemm`: add MQ4G256 arm (FWHT-rotate batch → `gemm_hfq4g256`).
  This arm exists on the gemma4 branch but is absent on the dispatch branch.
- `prefill_forward` path: add gemma4 batched prefill call
- Weight loading: add `Gemma4Weights` load path through `Gemma4::load_weights()`
- KV cache alloc: sliding KV sized at `sliding_window` (ring buffer), full KV at `max_seq`

### 1g · Wire GemmaTokenizer (SPM-BPE) support

In `crates/hipfire-runtime/src/tokenizer.rs`:
- Gemma 4 uses `GemmaTokenizer` — a SentencePiece BPE tokenizer with
  `vocab_size=262144`, `<bos>` prepend (id=2), `▁`-space prefix normalization.
  Port the tokenizer extensions from the gemma4 branch.
- Special token IDs (confirmed from `tokenizer_config.json`):
  - `<bos>` = 2, `<eos>` = 1, `<pad>` = 0
  - `<|turn>` = 105 (start of turn), `<turn|>` = 106 (end of turn, also EOS)
  - `<|image>` = 255999 (BOI), `<image|>` = 258882 (EOI); `image_token_id` = 258880
  - `<|audio>` = 256000 (BOA), `<audio|>` = 258883 (EOA); `audio_token_id` = 258881
  - `<|tool_call>` / `<tool_call|>` / `<|tool_response>` / `<|tool>`
  - `<|channel>` / `<channel|>` / `<|think|>` / `<|video|>`
- The `response_schema` uses regex-based tool-call extraction with
  `"x-parser": "gemma4-tool-call"` — the daemon's `parseToolCalls` path
  needs a gemma4 arm (different regex pattern from qwen35's XML-tag form).
- Register `tokenizer_type: "spm-bpe"` or detect from HFQ metadata.
- Audit against the dispatch branch's tokenizer extensions (dots-ocr, qwen2-VL)
  for insertion-point conflicts.

### 1h · Wire gemma4 into quantizer

In `crates/hipfire-quantize/src/main.rs`:
- Port gemma4-specific config parsing, weight layout, MoE expert table handling.
- Gate under `arch_id = 12` branches.

### Phase 1 gate

```bash
cargo check --workspace --all-targets
cargo test -p hipfire-dispatch
cargo test -p hipfire-dispatch-tests
# Gemma4 decodes coherently on gfx1100 + gfx1201
# Decode tok/s ≥ 90% of gemma4 branch baseline
```

---

## 5 · Phase 2 — Migrate decode path to dispatch framework

**Goal:** gemma4 single-token decode uses `execute_steps` and `AttentionFamily`
for every projection. Old `weight_gemv` direct calls removed from the decode
hot path.

### 2a · GEMV projections through `execute_steps`

Convert sliding + full layer decode. The old `weight_gemv` calls already
route through `GemvFamily` on this branch (they're backward-compatible
wrappers). The migration gains **fusion** (QKV/gate-up through `FUSED_TABLE`)
and makes gemma4 use the same `execute_steps` interpreter as qwen35/llama/qwen2.

**After (real API, verified against this branch):**
```rust
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_dispatch::types::{dtype_rotation_plan, RotationPlan};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;

let ctx = DispatchCtx::new(gpu);
let rotation = dtype_rotation_plan(lw.q_proj.gpu_dtype);
let steps = [
    Step::RmsnormAutomatic {
        x: &scratch.x,
        norm_weight: &lw.input_layernorm,
        x_plain: &scratch.tmp_plain,   // rmsnorm intermediate
        out: &scratch.tmp,             // final activation
        awq_scale: None,
        k: config.dim,
        eps: config.norm_eps,
        rotation,
    },
    Step::Gemv { w: &wr_q, input: GemvInput::Prerotated(&scratch.tmp), out: &scratch.q },
    Step::Gemv { w: &wr_k, input: GemvInput::Prerotated(&scratch.tmp), out: &scratch.k },
    Step::Gemv { w: &wr_v, input: GemvInput::Prerotated(&scratch.tmp), out: &scratch.v },
];
execute_steps(gpu, &ctx, &steps)?;
```

Same pattern for o_proj (with `GemvResidual`), gate/up/down.

**Operations that stay as direct GPU calls in Phase 2:**
- Per-head Q/K norm (`rmsnorm_batched` across heads) — no `Step` variant yet
- RoPE (`rope_f32` / `rope_partial_halved_f32`) — no `Step` variant yet
- `gelu_tanh_f32` + `mul_f32` (SwiGLU activation) — no `Step` variant yet
- `scale_f32` (layer scalar) — trivial, non-fusible

These will gain `Step` variants in a follow-up (enables future fusion
patterns like `[SiluMul, RmsnormAutomatic, Gemv]`). For now they remain
as direct GPU calls within the gemma4 layer function, same as qwen35's
embed lookup + scale outside `execute_steps`.

### 2b · Attention through `Step::Attend`

**After (real API):**
```rust
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::families::attention::AttnParams;

let tier_inputs = KvTierInputs {
    quant_asym4: kv_cache.quant_asym4,
    quant_asym3: kv_cache.quant_asym3,
    quant_asym2: kv_cache.quant_asym2,
    quant_q8: kv_cache.quant_q8,
    quant_fwht: kv_cache.quant_fwht,
    quant_hfq4: false,
    quant_q4: false,
    v_mode_bits: kv_cache.v_mode_bits,
    pos,
    flash_mode: kv_cache.flash_mode,
    capture_mode: gpu.graphs.capture_mode,
    batch_size: 1,
    is_tree: false,
    is_boundary: false,
    // ── Phase 0a additions ──
    cache_capacity: kv_cache.physical_cap,  // sliding: window; full: max_seq
    // ── Phase 0b addition ──
    head_dim,
};

let plan = KvTierPlan::derive(tier_inputs)
    .map_err(|e| hip_bridge::HipError::new(0, &format!("{:?}", e)))?;

let attn_io = AttnParams {
    q: &scratch.q, k: &scratch.k, v: &scratch.v,
    k_cache: &kv_cache.k_gpu[kv_layer_idx],
    v_cache: &kv_cache.v_gpu[kv_layer_idx],
    k_scales: None, v_scales: None,
    pos_buf: &scratch.pos_buf, pos,
    positions: None,
    n_heads, n_kv_heads, head_dim,
    physical_cap: plan.cache_capacity as usize,
    batch_size: 1,
    max_ctx_len: 0,
    flash_partials: Some(&scratch.fa_partials),
    givens_cos: kv_cache.givens_cos.as_ref(),
    givens_sin: kv_cache.givens_sin.as_ref(),
    tree_bias: None, block_start: 0, block_cols: 0,
    output: &scratch.attn_out,
};

let steps = [Step::Attend { plan, io: attn_io }];
execute_steps(gpu, &ctx, &steps)?;
```

### 2c · hd512 attention routing

hd512 kernels are registered under the **existing** `KernelKey` arms
(`KvWriteAsym3`, `AttnFlashAsym3`, etc.) with `ShapePredicate::HeadDimEq(512)`.
No new `KernelKey` variants. The `head_dim` field in `KvTierInputs` (added
in Phase 0b) flows through `KvTierPlan` and `AttnParams`, and
`dispatch_kv_write`/`dispatch_attend` branch on `io.head_dim` to launch
the hd512 kernel when needed.

### 2d · Fused QKV entries

Gemma4 uses Q/K norms *after* projection (batched per-head rmsnorm), so
QKV fusion is unlikely — the fused kernel bypasses per-head norms. Mark
as non-goal.

### Phase 2 gate

```bash
./scripts/coherence-gate.sh --gemma4   # coherence pass + KLD ≤ threshold
./scripts/speed-gate.sh --gemma4       # decode tok/s within ±3% of Phase 1
```

---

## 6 · Phase 3 — Migrate prefill path

**Goal:** batched prefill uses `GemmFamily` for GEMM projections and
`AttentionFamily` for batched attention. Preserve the v2 prefill structure
(`forward_prefill_batch_v2`) — it was the result of significant optimization
(token-batched projections, bucketed MoE GEMM, batched flash-attention
with dual KV caches and hd512). Adapt dispatchers within the existing
structure rather than rewriting to the qwen35 prefill template.

### 3a · GEMM through `GemmFamily::run_key()`

Convert `weight_gemm()` calls to `GemmFamily::resolve()` + `run()`.

The MQ4G256 batched GEMM path (FWHT-rotate input batch → `gemm_hfq4g256`)
must be registered as a `GemmFamily` variant with the pre-rotate expressed
as a separate `PipelineOp` or handled by the family's `run()` method.

### 3b · Batched attention

Wire through `AttentionFamily` with `batch_size > 1`, `positions` tensor,
`max_ctx_len`. The `cache_capacity` param threads through `KvTierPlan`
into both write and attend kernels. Sliding layers use
`cache_capacity = sliding_window` (ring-buffer wrap); full layers use
`cache_capacity = physical_cap` (no wrap).

### Phase 3 gate

Prefill tok/s within ±3% of Phase 1 baseline on gfx1100 + gfx1201.

---

## 7 · Phase 4 — MoE path migration

**Goal:** gemma4 MoE (26B-A4B: 128 experts, k=8, per-expert SwiGLU FFN
with `gelu_tanh` activation) goes through `MoeFamily` /
`run_moe_decode_gemma4`.

### 4a · Structural differences from qwen35 A3B MoE

Confirmed against `google/gemma-4-26B-A4B-it` config + safetensors index
(2026-06-07).

| Feature | Qwen35 A3B MoE | Gemma4 26B-A4B MoE |
|---|---|---|
| **Hidden size** | 2048 | 2816 |
| **Intermediate (dense)** | 11008 | 2112 |
| **MoE intermediate** | 2560 (shared) | 704 (per expert) |
| **Experts / top-k** | 128 / 8 | 128 / 8 |
| **Activation** | SiLU | `gelu_pytorch_tanh` |
| **Shared expert** | Yes (separate FFN) | No |
| **Combine** | MoE + shared expert → post-norm | MoE + **dense FFN** → post-FFN-norm |
| **Dense path** | Separate from MoE (selective layers) | **Parallel** to MoE (every MoE layer also runs dense FFN) |
| **Norm structure** | Single pre/post FFN norm | Separate norms: `pre_feedforward_layernorm` (dense) + `pre_feedforward_layernorm_2` (MoE router); 3 post-norms (`post_feedforward_layernorm`, `_1`, `_2`) |
| **KV heads (global)** | N/A (no dual attn) | `num_global_key_value_heads=2` (full layers use 2 KV heads × hd512) |
| **Decode kernels** | `gemv_*_k8_indexed` | `gemv_mq4g256_moe_gate_up_k8_indexed` + `gemv_*_moe_down_residual_scaled_k8_indexed` |
| **Prefill kernels** | Indexed batched GEMV | Indexed (default) + bucketed (opt-in, -5.7% prefill regression) |

Because `run_moe_decode` in `hipfire-dispatch` hardcodes SiLU activation
and the A3B shared-expert combine pattern, gemma4 needs its own executor:
`run_moe_decode_gemma4` in `pipeline/mod.rs`. This is a fork, not an
extension — the activation function, combine semantics, and parallel dense
path are too different to unify cleanly in this ship.

**Deferred unification:** when DeepSeek V4 MoE (bias-aware k=6) lands its
own executor, the three MoE patterns can be unified under a common
`MoeStrategy` abstraction. Not in scope for gemma4 Ships 1–5.

### 4b · Gate MoE path selection through `FeatureFlags`

The bucketed vs indexed MoE prefill path is selected by a new feature flag
`moe_bucketed: bool` in `FeatureFlags` (parsed once at startup from
`HIPFIRE_MOE_BUCKETED=1`), **not** by `std::env::var` at dispatch time.
This aligns with the existing `force_unfused` pattern and keeps the
dispatch resolver deterministic for graph capture.

### 4c · `PipelineOp` additions

Add `PipelineOp::GeluTanhMul` for the `gelu_tanh(gate) * up` activation
pattern (used by both dense FFN and MoE in gemma4). This replaces the
existing `SiluMul` for gemma4 layers.

### 4d · Register MoE kernels

Add to `moe_table.rs`:
- `gemv_mq4g256_moe_gate_up_k8_indexed` — decode gate+up
- `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed` — decode down-proj (HFQ4G128)
- `gemv_q8_0_moe_down_residual_scaled_k8_indexed` — decode down-proj (Q8_0)
- `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed_batched` — prefill down-proj
- `gemv_hfq4g256_moe_gate_up_bucketed` — prefill bucketed gate+up
- `gemv_hfq4g128_moe_down_residual_scaled_bucketed` — prefill bucketed down-proj
- `moe_bucket_build` — token→expert grouping

**Correctness constraint:** preserve exact fixed-order combine (no ULP drift
from reordering expert outputs — same constraint as #397 Step 6 for A3B).

### Phase 4 gate

MoE coherence (26B-A4B weights), decode tok/s parity with Phase 1.

---

## 8 · Phase 5 — Validation

### 5a · Coherence gate

```bash
./scripts/coherence-gate.sh --gemma4     # all gemma4 variants
./scripts/coherence-gate-dflash.sh       # if DFlash wired
```

### 5b · Perf A/B

```bash
./scripts/probe_commits.sh <phase1-commit> HEAD
```
±1–3% on gfx1100 + gfx1201 for decode, prefill, MoE.

### 5c · Coverage gate

Add gemma4 rows to `hipfire-dispatch-tests`:
```rust
// Sliding layers: asym3 + hd=256
assert_resolves(KernelKey::KvWriteAsym3, ArchPredicate::Always, gfx1100,
    ShapePredicate::HeadDimEq(256));
// Full layers: asym3 + givens + hd=512
assert_resolves(KernelKey::KvWriteAsym3, ArchPredicate::Always, gfx1100,
    ShapePredicate::HeadDimEq(512));
// MoE decode
assert_resolves(KernelKey::MoeGroupedGemv, ArchPredicate::Always, gfx1100);
// ...
```

### 5d · Cleanup

- Delete old `weight_gemv` / `weight_gemm` direct calls from gemma4.rs
- Remove `#[allow(dead_code)]` from migrated helpers
- Delete `HIPFIRE_DISPATCH_OLD/NEW` selector if still present
- SPDX/copyright headers on all new files

---

## 9 · Phase 0 contracts compliance

Gemma4 participates in the [#402](https://github.com/Kaden-Schutt/hipfire/pull/402)
Phase 0 contracts as follows:

| Contract | Gemma4 compliance |
|---|---|
| **Resolve-cache exemption** (0.1) | `KvTierPlan::derive()` called per token (not cached), consistent with the live-resolve exemption for KV-tier families. Gemma4's dual-cache model doesn't change this. |
| **Scratch ownership** (0.2) | `Gemma4Scratch` owns all tensors. Families take `&mut` refs at call time. No arch scratch moves into `hipfire-dispatch`. |
| **Paired write-then-attend** (0.3) | Gemma4 attention is inherently paired — KV write + flash-attention share derived state (givens cos/sin, cache_capacity, head_dim). `KvTierPlan::derive()` produces both keys from one input. `debug_assert` write-tier == attend-tier. |
| **gfx12 ladder** (0.4) | `HasWmma` predicate (collapsed from `HasWmmaW32` + `HasWmmaW32Gfx12`) covers gemma4's WMMA-eligible kernels. No new predicates needed. |
| **Family API doc** (0.5) | Gemma4 registers its kernels following the 7-item checklist from `families/mod.rs`. |
| **Pipeline-first verification** (0.6) | `probe_commits.sh` A/B on gfx1100 + gfx1201. `coherence-gate.sh` full matrix. Per-arch coverage tests. |

---

## 10 · Key dispatch framework additions for gemma4

| Feature | Why gemma4 needs it | Framework impact | Phase |
|---|---|---|---|
| **`cache_capacity` on KV + attn** | Sliding-window ring-buffer KV | Thread through `KvTierInputs` → `KvTierPlan` → `AttnParams` + all kernel signatures | 0a |
| **`head_dim` routing in attention** | Full layers use hd=512; sliding uses hd=256 | Add to `KvTierInputs`; gate via `ShapePredicate::HeadDimEq(512)` under existing keys | 0b |
| **`rope_partial_halved_f32`** | Proportional RoPE on full-attention layers | New kernel + dispatch; direct GPU call in Phase 1–2; `Step` variant in follow-up | 1d, 2a |
| **`logit_softcap_f32`** | Final logit softcap before sampling | New kernel + dispatch; direct GPU call (non-fusible, 1×/token) | 1d |
| **`gelu_tanh_f32` activation** | SwiGLU FFN uses gelu_pytorch_tanh | Already exists on dispatch branch (norm.rs:2160) | — |
| **MoE bucket-build** | Token→expert grouping for batched prefill | New `PipelineOp::GeluTanhMul` + `MoeFamily` entries; `FeatureFlags::moe_bucketed` gate | 4b–4d |
| **`run_moe_decode_gemma4`** | gelu_tanh + no shared expert + parallel dense FFN | Forked executor in `pipeline/mod.rs`; deferred unification with A3B path | 4a |
| **SPM-BPE tokenizer** | Gemma 4 vocab=262144, BOS-prepend, ▁-space | Tokenizer module extension; audit against dots-ocr extensions | 1g |
| **Dual KV cache** | Separate sliding + full KV caches | `LoadedModel` carries two `KvCache` fields; per-layer dispatch helper | 1e |

---

## 11 · Risk register

| Risk | Phase | Mitigation |
|------|-------|------------|
| `arch_id=7` collision (gemma4 vs qwen2) | 1e | ✅ Resolved: gemma4 → 12. Re-quantize artifacts. Update `docs/architecture-ids.md`. |
| `cache_capacity` breaks existing models | 0a | All callers pass `physical_cap` (identity). Full coherence gate on qwen35/A3B/llama before proceeding. |
| `head_dim` routing selects wrong kernel | 0b, 2c | `ShapePredicate::HeadDimEq(512)` gates hd512 variants. Coverage test per arch. |
| `gelu_tanh` vs SiLU activation mismatch | 4a | Forked `run_moe_decode_gemma4` executor. Deferred unification with A3B path. |
| hd512 kernels not precompiled for all archs | 1b/5b | Compile + validate per-arch; add to coverage gate. |
| `rope_partial_halved` not in dispatch framework | 2a | Direct GPU call in Phase 1–2; `Step::RopePartial` in follow-up. |
| Old `weight_gemv` deleted from `llama.rs` | 1f | ✅ Still present on this branch (llama.rs:702+). Phase 1 is safe. |
| MoE bucket-build differs from A3B indexed dispatch | 4a | Forked executor + `FeatureFlags::moe_bucketed` gate. |
| `kv_cache_write_asym3_fused` signature change breaks dispatch crate | 0a | `dispatch_kv_write` in `attention.rs` updated to pass `plan.cache_capacity`. |
| `KvTierPlan::derive` drops `cache_capacity` | 0a | Field added to `KvTierInputs` → flows through `derive` → `KvTierPlan` → `AttnParams`. |
| Graph capture pointer staleness for direct GPU calls | 2a | Gemma4 uses its own warmup-then-capture pattern (same as qwen35). `Step` variants for rope/logit_softcap/gelu_tanh added in follow-up. |
| Phase gate scripts have no gemma4 rows | 0c | Quantize weights + add gemma4 rows to gate script matrices before Phase 1 gate. Start with 12B dense. |
| 26B-A4B safetensors incomplete (shard 1 missing) | 0c | Re-download from HF. Gate: verify md5 of all shards before quantizing. |
| 12B/31B/E-class safetensors not yet on disk | 0c | Blocked on incoming BF16 files. Phase 0a–0b and 1a–1d (code-only) proceed in parallel. | |
| Daemon `arch_id` wildcard fallback misroutes gemma4 | 1e | Audit all 16+ sites; add explicit `12 =>` arms. VL-gating sites aware of vision tower. |
| SPM-BPE tokenizer conflicts with dots-ocr extensions | 1g | Audit insertion points against dispatch branch's tokenizer before porting. |

---

## 12 · Reference

- Dispatch unification roadmap: [#397](https://github.com/Kaden-Schutt/hipfire/issues/397)
- Dispatch unification PR (DRAFT): [#393](https://github.com/Kaden-Schutt/hipfire/pull/393)
- Phase 0 contracts: [#402](https://github.com/Kaden-Schutt/hipfire/pull/402)
- Canonical branch: `Kaden-Schutt/hipfire:integration/dispatch-unification`
- Gemma4 branch: `feat/gemma4-128k-ring-buffer` (merged to master @ `9b206438`)
- Consolidated adversarial review: `findings/gemma4_dispatch_plan_consolidated_rev.md`
- Individual reviews: `findings/gemm4_dispatch_plan_rev_gemini.md`,
  `findings/gemm4_dispatch_plan_rev_claude.md`,
  `findings/gemma4_dispatch_plan_rev_glm5.md`
- Model sources (HuggingFace):
  - `google/gemma-4-12B-it` — 12B dense, unified (audio+vision+text)
  - `google/gemma-4-26B-A4B-it` — 26B MoE A4B, text+vision
  - `google/gemma-4-31B-it` — 31B dense (TBD)
  - `google/gemma-4-E4B-it` / `google/gemma-4-E2B-it` — Any-to-Any (TBD)
- Local artifacts: `/local/models/google/gemma-4-{12B,26B,31B,E4B,E2B}-it/`

---

## 13 · Out of scope

- **Gemma4 vision tower** (`gemma4_vision.rs`) — placeholder file. Wire into VL
  dispatch path in a follow-up (Phase N+1). Not part of Ships 1–5.
- **Gemma4 audio tower** (12B/E-class unified variants) — not part of Ships 1–5.
- **DFlash / spec-decode for gemma4** — no draft model exists. Deferred.
- **DDTree for gemma4** — no tree-attention integration. Deferred.
- **MoE unification with A3B/DeepSeekV4** — deferred to post-Ship-5 work.
- **`Step` variants for rope/logit_softcap/gelu_tanh** — deferred to Phase 2
  follow-up. Not blocking Phase 1 or initial Phase 2 decode migration.
- **E-class Any-to-Any variants** (E4B, E2B) — not in Ships 1–5. Config and
  forward-pass differences TBD once safetensors land.

---

*Plan authored 2026-06-07. Updated 2026-06-07 with consolidated adversarial
review findings. Config-audited 2026-06-07 against real BF16 model configs.
Update as phases complete.*
