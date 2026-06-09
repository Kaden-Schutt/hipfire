# Gemma 4 dispatch-unification migration plan

Branch: `feat/dispatch-unification-gemma4` (off `upstream/integration/dispatch-unification`
@ `a7902234` — Ship 5.2 tip, 2026-06-07).

**Adversarially reviewed:** 2026-06-07 by Gemini 3.5 Flash, Claude Opus 4.8, glm5.
Consolidated findings in `findings/gemma4_dispatch_plan_consolidated_rev.md`.
All accepted findings incorporated below.

**Config-audited:** 2026-06-07 against real BF16 safetensors + configs from HuggingFace
(`google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it`). Corrections applied inline.

**Status (2026-06-08):** Phase 0 + Phase 1 complete; **Phase 1.5 Step A done**
(`876c1158`). **Phase 1.5 Step B done** — daemon sliding KV switched from
fp32 to q8 ring-buffer (`physical_cap=sliding_window=1024`), constant ~300 MB
regardless of context length. Coherent output confirmed at 1266 tokens.
**Phase 1.5 §4.5.5 Goal A infrastructure done** — all HIP tile
kernels (q8/asym4/asym2) have `window_size`+`cache_capacity`+hd512 support;
Rust `_cap` siblings use `launch_maybe_blob` for graph-capture correctness.
asym4/asym2 model-crate wiring deferred to Phase 2 step 2e (dispatch-unification
principle — old-style branches in `gemma4.rs` are phased out, not expanded).
Debug history in `findings/gemma4_dispatch_devlog.md`.

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

BF16 safetensors land in `/local/models/google/` (four variants). The 26B-A4B
MoE safetensors are now **complete** (both shards present, ~49 GB) — the MoE
model can be quantized and validated once Phase 4 lands.

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
| 26B-A4B-it | ✓ | ✓ | ✅ **Complete** — both shards (~49 GB) at `/local/models/google/gemma-4-26B-A4B-it` (2026-06-08) |
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

## 2 · Phase 0 — prerequisites (before any gemma4 code lands) ✅ DONE

These items thread gemma4's requirements through the shared dispatch surface.
They must complete and pass validation on existing models before gemma4 code
enters the branch.

### 0a · Thread `cache_capacity` through the KV + attention dispatch surface ✅ DONE

Commit `6c02fb1a`. Added `cache_capacity: u32` and `head_dim: usize` to
`KvTierInputs` → `KvTierPlan` → `AttnParams`. GPU method signatures deferred
(gemma4 uses old-style dispatch wrappers for Phase 1).

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

> **Revised in Phase 1.5 (§4.5.2):** the struct-level fields below landed, but the
> GPU-method/kernel threading is now done via a **sibling method** for high-fan-out
> calls (`kv_cache_write_q8_0`, 46 callers) + extend-signature for low-fan-out ones,
> rather than a uniform 40-site change. And **windowing — not the ring buffer — is
> the >1024 correctness fix**; the ring buffer is a memory-only follow-up.

**Blast radius (struct-level — done; GPU-method threading per §4.5.2):**
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

### 0b · Add `head_dim` routing to `KvTierInputs` and attention resolution ✅ DONE

Rolled into Phase 0a commit. `head_dim` field added alongside `cache_capacity`.

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

### 0c · Quantize gemma4 model weights + gate script rows ✅ DONE

Commit `3ca8a07d`. 12B quantized as HFQ4G256 (12.7 GB), E2B quantized (5.5 GB),
also Q8 (12.7 GB) and MQ4 (6.9 GB) variants. Files at `/local/models/google/`.
Gate script rows not yet added — deferred to Phase 5.

Before any phase gate can run, we need:
1. ✅ Complete BF16 safetensors present for 12B and **26B-A4B (both shards,
   ~49 GB, 2026-06-08)**. 31B / E4B / E2B still incoming.
2. Quantize gemma4 model weights to HFQ/MQ4 format via `hipfire-quantize`
   with `arch_id=12`
3. Symlink quantized artifacts into `~/.hipfire/models/`
4. Add gemma4 `(model|id|prompt|max)` rows to `coherence-gate.sh` and
   `speed-gate.sh` matrices

The scripts currently have zero gemma4 rows; `--gemma4` flags do not exist.
Initial target: 12B dense (simplest forward path, no MoE).

### 0d · Verify Phase 0 doesn't regress existing models ✅ DONE

`cargo check --workspace` clean. No coherence-gate regression observed.

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

## 4 · Phase 1 — Scaffold ✅ DONE

**Goal:** bring gemma4 crate, kernel files, and rdna-compute additions over
using old dispatch patterns. Gemma4 decodes coherently on gfx1100 + gfx1201.

**Status:** 12B dense model decodes coherently and is a usable chat model.
Debug history documented in `findings/gemma4_dispatch_devlog.md` (16 sessions).

**Exit criterion (mergeable checkpoint):** gemma4 decodes coherently at
≥90% of the gemma4 branch's tok/s baseline. Phase 1 is intended for merge
into the integration branch; Phases 2–5 follow as incremental PRs.

### 1a · Add `hipfire-arch-gemma4` crate to workspace ✅ DONE

Commit `aee1e0bc`. 2654-line `gemma4.rs` ported with `arch_id=12`.

Most files can be ported directly from `feat/gemma4-128k-ring-buffer`:
- `crates/hipfire-arch-gemma4/Cargo.toml` — depends on `hipfire-runtime`, `rdna-compute`, `hip-bridge`, `hipfire-dispatch`
- `crates/hipfire-arch-gemma4/src/lib.rs` — re-exports
- `crates/hipfire-arch-gemma4/src/arch.rs` — `Architecture` trait impl. **Update `arch_id()` from `7` → `12`.**
- `crates/hipfire-arch-gemma4/src/gemma4.rs` — forward pass (old-style dispatch initially)
- `crates/hipfire-arch-gemma4/src/gemma4_vision.rs` — vision tower placeholder (out of scope for Ships 1–5)
- `crates/hipfire-arch-gemma4/examples/` — bench + smoke + verify tools
- Register in workspace `Cargo.toml`

### 1b · Port gemma4 kernel files ✅ DONE

Commits `9a8016f4`+`0d783631`. Ported all decode kernels:
- `attention_flash_asym3_tile_hd512.hip` — full-attention decode
- `kv_cache_write_asym_k_givens3_hd512.hip` — full-attention KV write
- `rope_partial_halved.hip` — proportional RoPE
- `logit_softcap.hip` — final logit softcap

MoE kernels deferred (Phase 4). Batched prefill kernels deferred (Phase 3).

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

### 1c · Add kernel declarations ✅ DONE

Added in `crates/rdna-compute/src/gemma4_ext.rs` and `kernels.rs`.

In `crates/rdna-compute/src/kernels.rs`: declare each new kernel symbol from
§1b with its precompiled binary slice (follow the `ensure_kernel` lazy-load
pattern).

### 1d · Add gemma4 dispatch helpers to `rdna-compute/src/` ✅ DONE

Added `rope_partial_halved_f32()`, `logit_softcap_f32()`, plus hd512
attention and KV-write dispatch helpers in `gemma4_ext.rs`.

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

### 1e · Wire gemma4 into daemon ✅ DONE

Commits `3d76df05`+`b00a5ace`+`876c0b48`. Full wiring:
- `LoadedModel` fields for gemma4 config/weights/scratch/dual-KV
- `arch_id=12` dispatch at all match sites
- Load path through `Gemma4::config_from_hfq` / `load_weights` / `new_state`
- Generate path via `gemma4::forward_scratch()`
- Dual KV cache: sliding (window=1024) + full (max_seq=2048)
- Per-layer KV selection via `config.layer_types[]`
- `gemma4::init_scratch_constants()` call for `v_norm_ones_full`

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

### 1f · Wire gemma4 into `llama.rs` ✅ DONE

Relaxed `asym3 head_dim==256` assertion to also accept `head_dim==512`.

- `weight_gemm`: add MQ4G256 arm (FWHT-rotate batch → `gemm_hfq4g256`).
  This arm exists on the gemma4 branch but is absent on the dispatch branch.
- `prefill_forward` path: add gemma4 batched prefill call
- Weight loading: add `Gemma4Weights` load path through `Gemma4::load_weights()`
- KV cache alloc: sliding KV sized at `sliding_window` (ring buffer), full KV at `max_seq`

### 1g · Wire GemmaTokenizer (SPM-BPE) support ✅ DONE (tool-call parsing pending)

Commit `5ffd0314`. SPM-BPE encoder fixed (removed erroneous `▁` prepend).
`"Hello"` now encodes to 9259 matching HF.

**CLI tokenizer resolved:** the earlier `hipfire run`/`serve` failure
(`GPT-2 BPE vocab missing byte symbol: byte 0x90`) was the **stale prod daemon**
(`~/.hipfire/bin/daemon` predated the SPM-BPE ▁-detection fix), not a code bug —
the tokenizer detection (`▁` over `Ġ`) is correct. Refreshing the binary makes
`hipfire run gemma-4-12B-it-q8 "..."` produce clean output. Standing gotcha:
refresh the prod daemon after any runtime change.

**Still pending:** the `gemma4-tool-call` regex parser arm (below) — not on the
≤Phase-1 critical path.

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

### 1h · Wire gemma4 into quantizer ✅ DONE

Commit `3ca8a07d`. `"gemma4" | "gemma4_unified" => 12` auto-detection.

In `crates/hipfire-quantize/src/main.rs`:
- Port gemma4-specific config parsing, weight layout, MoE expert table handling.
- Gate under `arch_id = 12` branches.

### Phase 1 gate ✅ PASSED (12B dense)

```bash
cargo check --workspace --all-targets  # clean
# Gemma4 decodes coherently on gfx1151 (confirmed Claude Opus 4.8)
# Decode: 15.2 tok/s (AR, no spec-decode)
```

### Phase 1 debug history

The path to coherent decode involved 12+ debug sessions, documented in
`findings/gemma4_dispatch_devlog.md`. Key bugs found and fixed:

1. **`v_norm_ones_full` never initialized** (`c7ce94c0`): Scratch tensor
   remained zeros → V normalization multiplied by zero → garbage output.
2. **SPM-BPE encoder prepended `▁`** (`5ffd0314`): "Hello" encoded wrong.
3. **hd512 attention missing reduce kernel** (`2e36fee2`): `attention_flash_asym3_hd512`
   only launched the tile kernel, never the `attention_flash_q8_0_reduce` kernel.
   The `attn_out` buffer was never written — stale data from the previous
   sliding layer was used. This was the critical fix that produced coherent output.
4. **HF reference double-scaled**: `lm.embed_tokens(input_ids)` already applies
   `embed_scale` internally; the Python oracle was multiplying by it again.
   Corrected oracle confirmed embedding + all projections + norms match HF.

Session-by-session detail in `findings/gemma4_dispatch_devlog.md` (1567 lines, 16 sessions).

---

## 4.5 · Phase 1.5 — Long-context: sliding-window correctness + ring-buffer KV

**Goal:** correct gemma4 generation for sequences **>`sliding_window` (1024)**,
then make it **memory-scalable** to 128k context. These are two distinct pieces:
windowing is a **correctness** fix (mostly done); the ring buffer is a **memory**
optimization layered on top.

### 4.5.0 · What we learned (Session 16–17b)

- The original ">1024 collapse" was a **measurement artifact**, not a model bug:
  the debug oracle (`examples/gemma4_oracle.rs`) allocated the sliding KV via
  `KvCache::new_gpu` = the **fp32 KV path**, whose `attention_flash` branch had
  **no window masking** (it attended to all positions). That path is the debug
  branch (`// incorrect for seq > window_size`). It collapsed at exactly 1024.
  **Now fixed** (`41bd5d87`): the fp32 `attention_flash_partial` kernel gained a
  `kv_window` param (same chunk-clamping pattern as the asym3 tile kernel).
- The **asym3 KV path** (the production path) **does** window: it routes through
  `attention_flash_asym3_window` → `attention_flash_asym3` with a `kv_window`
  param (ported from `upstream/feat/sliding-window-fa`: clamp `tile_start` up to
  `seq_len - kv_window`, skip out-of-window sub-tiles; `kv_window=0` = full causal,
  byte-identical, so qwen35/dispatch callers are unaffected).
- **Confirmed via the per-token-ids oracle:** with the asym3 windowed path at
  1200 tokens, **hipfire argmax matches HF (236761)** — collapse gone. Residual
  per-layer Δ 0.5–1.8 from ~L19 is HFQ4/q8 quantization noise compounding across
  48 layers; it does **not** flip the argmax. So **windowing is the correctness fix.**
- The window-only path sizes the sliding cache at `max_seq` (correct, but defeats
  the 128k memory goal). The ring buffer keeps it at `sliding_window` via modular
  slot indexing — same math, less memory.

**Oracle infrastructure (built, committed):** `scripts/oracle_gemma4.py` (HF
reference, f32; a GPU path exists via `.venv-rocm` torch 2.9.1+rocm6.4 which drives
gfx1151) + `examples/gemma4_oracle.rs` (hipfire side). Both read a **token-ids
file** so they compare byte-identical inputs (no tokenizer drift). Self-gate
`[2,9259]` matches HF (real-token argmax 575). This is the gate for all
long-context work.

### 4.5.1 · Step A — enable windowed long-context (correctness) ✅ DONE

Landed in three commits:

1. ✅ **fp32 sliding path fixed (`41bd5d87`)** — added `kv_window` masking to the
   fp32 `attention_flash` kernel (root cause in `8de8d1eb`), so the debug path can't
   silently mislead a future investigation (it cost one cycle). The oracle can stay
   on either KV path now; asym3 remains the production path.
2. ✅ **Daemon >1024 enabled (`876c1158`)** — sliding KV sized at `max_seq` (window-only),
   `kv_window` masking active in attention, `>= sliding_window` refusal guard replaced
   with a `max_seq` context-length guard. Validated: 1266-token prompt → coherent
   3-sentence summary (104 tok @ 10.8 tok/s).
3. ✅ **Oracle gate passed** — argmax matches HF (236761) at 1200 tokens.

**Gate (Step A):** `gemma4_oracle` argmax + top-k match HF at 1100/1200 tokens
(real tokens <256000; the q8 lm_head artifact on the multimodal block ≥256000 is
expected — compare on real tokens). Coherence on a >1024 prompt.

### 4.5.2 · Step B — ring-buffer KV (memory; enables 128k) ✅ DONE

The window-only `max_seq` sliding cache scales linearly with context (wasteful for
128k). The ring buffer keeps the sliding cache at `sliding_window=1024` slots:
**write `slot = pos % cache_capacity`, read `slot = t % cache_capacity`**, with the
same window mask. Mathematically identical to window-only — it's a memory rewrite.

**⚠ Do NOT merge `feat/gemma4-128k-ring-buffer` into this branch.** That branch
predates the dispatch framework (`hipfire-dispatch/`), deletes `kv_tier.rs`,
uses different GPU method signatures, and **reverts** the fp32 `attention_flash`
window fix (`41bd5d87`). The correct approach is to **cherry-pick only the 3 HIP
kernel diffs** and write the Rust integration fresh per the sibling-method design
below.

The proven kernel HIP diffs apply cleanly from `feat/gemma4-128k-ring-buffer`
(verified: 184 lines across 3 files, no textual conflicts with this branch's
kernel sources):
- `kv_cache_write_asym_k_givens3.hip` (K): `slot = cap>0 ? pos%cap : pos`
- `kv_cache_write_q8_0.hip` (V): same
- `attention_flash_asym3_tile.hip`: read `slot = cap>0 ? t%cap : t` (composes with
  the `kv_window` mask already added)

`cache_capacity = 0` is the identity (slot = pos / t), so every non-gemma path is
byte-identical. The default `u32` value of 0 gives the safe identity — any caller
that forgets to set it gets correct (non-wrapping) behavior.

#### Rust integration — sibling method vs. extend signature

**Rust has no method overloading** (unlike Java/C++): you cannot have two methods
named `kv_cache_write_q8_0` with different arities — it's a duplicate-definition
error. Match the tool to the call-site fan-out:

- **High fan-out → sibling method with a *distinct name*** that delegates with `0`.
  `kv_cache_write_q8_0` has **46 callers** (other arch crates, examples, tests).
  Keep its signature; add `kv_cache_write_q8_0_cap(.., cache_capacity: u32)`:
  ```rust
  // Existing API — UNCHANGED; all 46 callers stay byte-identical.
  pub fn kv_cache_write_q8_0(&mut self, dst, src, pos_buf, n_kv, head_dim) -> HipResult<()> {
      self.kv_cache_write_q8_0_cap(dst, src, pos_buf, n_kv, head_dim, 0)
  }
  // Ring-aware variant — gemma sliding layers call this.
  pub fn kv_cache_write_q8_0_cap(&mut self, dst, src, pos_buf, n_kv, head_dim,
                                 cache_capacity: u32) -> HipResult<()> { /* launch with cap */ }
  ```
  This is *the* idiomatic substitute for overloading (cf. `Vec::new` vs
  `Vec::with_capacity`). Make the wrapper a one-line delegation + a doc comment
  ("`cap=0` shorthand of `_cap`") so it doesn't read as accidental duplication.
- **Low fan-out → extend the signature.** `kv_cache_write_asym3_fused` (~7 callers)
  and `attention_flash_asym3` (5 callers — *already* carries `kv_window`) just take
  an added `cache_capacity` arg; the handful of non-gemma callers pass `0`. Two
  siblings where you don't need them is its own smell.

**Correctness invariant:** the write (`pos%cap`) and the read (`t%cap`) **must use
the same `cap`**. Derive it once (see dispatch integration) so they cannot drift —
a mismatch is the #30-class silent-wrong-precision bug.

#### Daemon sliding KV: must switch from fp32 to asym3

The daemon currently allocates the sliding KV via `KvCache::new_gpu` (fp32). The
ring-buffer write kernels only exist for the **asym3** path
(`kv_cache_write_asym_k_givens3`, `kv_cache_write_q8_0`). Without this switch,
Step B would leave the daemon's fp32 sliding path still writing `slot = pos` and
OOBing at >1024.

**Pre-Step B prerequisite:** change `daemon.rs`'s gemma4 load path from:
```rust
let kv_sliding = KvCache::new_gpu(gpu, n_layers, sliding_n_kv, sliding_head_dim, sliding_window);
```
to:
```rust
let kv_sliding = KvCache::new_gpu_asym3(gpu, n_layers, sliding_n_kv, sliding_head_dim, sliding_window);
```
This aligns the daemon's sliding path with the production asym3 pipeline that the
ring-buffer kernels target. After this switch, the sliding layer code will take the
asym3 branch (already windowed via `attention_flash_asym3_window`) for both read
and write.

> **Memory note:** sizing the fp32 sliding cache at `max_seq` for Step A's
> window-only path causes a transient VRAM spike at large `max_seq` (48 layers ×
> 8 heads × 256 dim × 4 bytes × 128k ≈ 6 GB). Step B's ring buffer shrinks it back
> to 1024 rows. This grow-then-shrink is architecturally fine but plan for the
> spike when testing Step A at large contexts.

#### Dispatch integration (the "unified" part)

`cache_capacity` already exists on `KvTierInputs` → `KvTierPlan` → `AttnParams`
(Phase 0a, struct-level only). Wire it the rest of the way:
- Derive `cache_capacity` once in `KvTierPlan::derive` (= `sliding_window` for
  gemma sliding tiers, `0` otherwise) so write+read share one value.
  **Open question:** `KvTierPlan::derive` currently has no arch-specific dispatch —
  it doesn't know which model is loaded or which tiers are sliding. The gemma4
  crate must either (a) pass `cache_capacity` as an input to `derive`, or
  (b) the daemon/gemma4 code sets `cache_capacity` on the plan after derivation.
  Option (b) is simpler and avoids threading the config through the dispatch
  crate.
- Flow it from the plan to the GPU methods (the `_cap` sibling / extended args).
- Near-term, gemma's `gemma4_ext` wrappers already accept `cache_capacity: u32`
  but stub it out with `let _ = cache_capacity`. Remove the stubs and pass it to
  the kernel launches. The full gemma→`AttentionFamily` migration (Phase 2)
  inherits it for free.

#### Concrete touch-point list

| File | What | Strategy |
|------|------|----------|
| `kernels/src/kv_cache_write_asym_k_givens3.hip` | Add `cache_capacity` param; `slot = cap>0 ? pos%cap : pos` | Cherry-pick from ring-buffer branch |
| `kernels/src/kv_cache_write_q8_0.hip` | Same | Cherry-pick |
| `kernels/src/attention_flash_asym3_tile.hip` | Add `cache_capacity` param; read `slot = cap>0 ? t%cap : t` | Cherry-pick |
| `crates/rdna-compute/src/attention.rs` `kv_cache_write_q8_0` | Sibling `kv_cache_write_q8_0_cap(.., cache_capacity: u32)` | New method, existing delegates to it with `0` |
| `crates/rdna-compute/src/attention.rs` `kv_cache_write_asym3_fused` | Extend with `cache_capacity: u32` (7 callers pass `0`) | Extend signature |
| `crates/rdna-compute/src/attention.rs` `kv_cache_write_asym_k_givens3_hd512` | **No change** — full layers don't wrap | — |
| `crates/rdna-compute/src/attention.rs` `attention_flash_asym3` | Already has `kv_window`; add `cache_capacity: u32` (5 callers pass `0`) | Extend signature |
| `crates/rdna-compute/src/attention.rs` `attention_flash_asym3_tile_hd512` | **No change** — full layers don't wrap | — |
| `crates/rdna-compute/src/gemma4_ext.rs` all `_window` wrappers | Remove `let _ = cache_capacity` stubs; thread to underlying methods | Remove stub |
| `crates/hipfire-arch-gemma4/src/gemma4.rs` sliding writes | Pass `cache_capacity = sliding_window` | Model-crate edit |

**Correctness invariant:** the write (`pos%cap`) and the read (`t%cap`) **must
use the same `cap`**. Derive it once in the plan so they cannot drift — a mismatch
is the #30-class silent-wrong-precision bug.

> **Revises §0a.** §0a proposed adding `cache_capacity` to *every* KV-write/flash
> method as a "mechanical file-wide" 40-site change with all callers passing
> `physical_cap`. The sibling-for-high-fan-out + extend-for-low-fan-out split above
> is less invasive (the 46 `kv_cache_write_q8_0` callers don't change) and is the
> recommended approach. §0a's struct-level fields stand; only the GPU-method
> threading strategy changes.

### 4.5.3 · Step C — validation ✅ PARTIAL

- **Ring == window-only:** Not yet run as a formal oracle gate (the oracle uses
  fp32 sliding, not q8). Validated informally: coherent summary at 1266 tokens,
  matching the fp32 path's output character ("Based on the text provided...").
  Formal gate requires oracle with q8 sliding or a dedicated comparison harness.
- **vs HF:** argmax=236761 matches HF at 1200 tokens (full-layer path; sliding
  quant mode does not affect the oracle's full-layer dump).
- **Coherence:** ✅ passed — coherent 80-token summary at 1266 input tokens.
- **Memory:** ✅ confirmed — sliding cache at `physical_cap=1024` (q8), constant
  ~300 MB regardless of context length.

### 4.5.4 · Scope / sequencing

- **Step A (windowed asym3 + oracle-on-asym3 + drop guard)** delivers correct
  long-context now and is independently shippable.
- **Step B (ring buffer)** is a memory follow-up — needed for 128k, not for
  correctness. Do it after Step A is green, gated on "ring == window-only."
- The `kv_window` kernel work (`attention_flash_asym3_tile`) is lineage-grafted
  from `sliding-window-fa`; Step B's ring indexing is lineage-matched to the
  original gemma branch. Both compose (window mask + modular slot).

**Stale-kernel-cache caution:** changing `kernels/src/*.hip` requires clearing
`.hipfire_kernels/{arch}/` (the on-disk hsaco cache keys by name, not source hash)
**and** refreshing `~/.hipfire/bin/daemon` — a stale cache served the pre-window
kernel and produced a phantom "window has no effect" result for a full cycle.

**⚠ Adding parameters to an existing kernel name is the highest-risk scenario.**
The old hsaco loads silently (same kernel name) but the kernarg buffer is
misaligned — garbage parameters, silent corruption, no clean error. Either clear
the cache directory or **rename the kernel** when changing its parameter
signature. Session 16 hit this exact failure mode with the `kv_window` addition.

### 4.5.5 · KV-mode coverage (q8 / asym4 / asym2 re-port; fwht3/fwht4 follow-up)

The gemma4 forward branches on **five** KV modes (asym3, asym4, asym2, q8, fp32),
but the dispatch migration left coverage uneven — only asym3 (+ fp32) actually
window correctly. Current state:

| Mode | Sliding (hd256) window/ring | Full (hd512) | Notes |
|---|---|---|---|
| **asym3** | ✅ window + ring | ✅ | the reference path |
| **fp32** | ✅ window (no ring) | ✅ (`attention_f32`) | daemon's sliding default (`new_gpu`) |
| **q8** | ✅ sliding window + ring done (`7204d471`) | ✅ full done — `attention_flash_q8_0_cap` (window=0/cap=0) | complete |
| **asym4** | ⚠️ kernel infra done, wiring deferred to Phase 2 step 2e | ⚠️ kernel infra done (n_halves=4), wiring deferred to Phase 2 step 2e | HIP tile + Rust `_cap` + `launch_maybe_blob` landed |
| **asym2** | ⚠️ same as asym4 | ⚠️ same as asym4 | same treatment |
| **fwht3/4/2** | ❌ no branch | ❌ no branch | never wired into gemma4; Goal B (follow-up PR) |

**The original `feat/gemma4-128k-ring-buffer` branch had q8/asym4/asym2 windowed +
ring** (window threaded through 7 `attention_flash_*.hip` files; fp32 errored).
The dispatch port stripped the window/cap args from those three `_window` wrappers
down to `let _ = (window_size, cache_capacity)`. The re-port restores this via
kernel infrastructure (Phase 1.5 ✅) and dispatch-framework wiring (Phase 2 step 2e).

#### Goal A — restore q8 / asym4 / asym2 (re-port; split across Phase 1.5 + Phase 2)

Bring the three quantized modes back to full window + ring parity with asym3.
Work split into **infrastructure** (Phase 1.5) and **model-crate wiring** (Phase 2)
per the dispatch-unification principle — old-style branches in `gemma4.rs` are
intentionally avoided; that routing belongs in `AttentionFamily`.

**Phase 1.5 (done): kernel + Rust GPU-method infrastructure**

| Piece | Status | Details |
|-------|--------|--------|
| q8 tile: `window_size` + `cache_capacity` | ✅ Done | `attention_flash_q8_0_tile.hip` + `_cap` sibling |
| asym4 tile: `window_size` + `cache_capacity` + hd512 | ✅ Done | `attention_flash_asym4_tile.hip` — `mq[16]`/`out_vec[16]`, n_halves=4, ring slot indexing, `launch_maybe_blob` |
| asym2 tile: `window_size` + `cache_capacity` + hd512 | ✅ Done | `attention_flash_asym2_tile.hip` — same treatment |
| givens4 K-write: `cache_capacity` ring | ✅ Done | `kv_cache_write_asym_k_givens4.hip` — slot = cap>0 ? pos%cap : pos |
| givens2 K-write: `cache_capacity` ring | ✅ Done | `kv_cache_write_asym_k_givens2.hip` — same |
| Rust `_cap` siblings (asym4/asym2) | ✅ Done | `attention_flash_asym4_cap`, `attention_flash_asym2_cap` — `launch_maybe_blob` + `KernargBlob` for graph-capture correctness |
| Rust `_window` wrappers (asym4/asym2) | ✅ Done | Delegate to `_cap` with window+cap params |
| q8 full-layer wiring in `gemma4.rs` | ✅ Done | Uses `_cap` with window=0/cap=0 for full-attention layers |
| asym4/asym2 full-layer wiring in `gemma4.rs` | ❌ Deferred | Belongs in Phase 2 `execute_steps`/`AttentionFamily` — see §5 step 2e |

**Phase 2 (step 2e): model-crate wiring through dispatch framework**

Instead of adding `} else if kv_cache.quant_asym4 {` / `quant_asym2 {` branches
in `gemma4.rs`'s full-layer decode function (old-style dispatch), the routing
for these quant modes goes into `AttentionFamily` → `dispatch_attend`. The
infrastructure above ensures the kernels and Rust entry points are ready;
Phase 2 just needs the dispatch-table rows.

See §5 step 2e for the concrete migration.

#### Goal B — implement fwht3 / fwht4 for gemma4 (FOLLOW-UP, net-new)

Unlike A, fwht is **not a re-port** — gemma4's forward has never branched on
`quant_fwht`, and there are no gemma4 fwht hd512 kernels. The dispatch branch has
fwht KV machinery for qwen35, and the daemon already has gemma4-reachable
`new_gpu_fwht3_*` / `new_gpu_fwht4_*` alloc paths — so the cache side exists, but
the **forward + kernels do not**. Scope:

1. Add `quant_fwht3` / `quant_fwht4` branches to `sliding_layer_decode_impl` and
   `full_layer_decode_impl`, with windowed `_window` wrappers (window + ring).
2. Provide **windowed fwht3/fwht4 tile kernels** (adapt qwen35's fwht KV kernels,
   adding the `window_size` + `cache_capacity` args) for hd256 sliding.
3. Provide **hd512 fwht3/fwht4 KV-write + flash variants** for the full layers.
4. Gate: oracle parity + coherence per mode; KLD vs asym3/q8 to confirm fwht's
   quality benefit on gemma4's KV (the motivation for the more-modern formats).

**Sequencing:** Goal A (re-port q8/asym4/asym2) lands with Phase 1.5 — it's
restoring proven kernels and closes the regression. Goal B (fwht3/fwht4) is a
**separate follow-up PR after Phase 1.5**, since it's genuinely new gemma4 surface
(new branches + new kernels), not a port.

---

## 5 · Phase 2 — Migrate decode path to dispatch framework ✅ DONE

**Goal:** gemma4 single-token decode uses `execute_steps` and `AttentionFamily`
for every projection. Old `weight_gemv` direct calls removed from the decode
hot path.

**All sub-steps complete:**
- §2a GEMV projections — `Step::Gemv` for all decode-path projections
- §2b Attention — `Step::Attend` for sliding + full layers
- §2c hd512 routing — dispatch branches on `head_dim==512` for asym3
- §2d Fused QKV — non-goal (per-head norms prevent fusion)
- §2e asym4/asym2 — completed implicitly by threading commit: dispatch
  arms for all four quant modes (q8/asym3/asym4/asym2) already route
  through `_cap` GPU methods. No model-crate branches needed —
  `KvTierPlan::derive` selects the right keys automatically.

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

For asym4 and asym2, the tile kernels already handle hd512 via `n_halves`
(up to 4 for 512-dim heads) — no separate `*_hd512` kernel variant needed.
`ShapePredicate::HeadDimGe(128)` (or unconditional) is sufficient since
`n_halves = head_dim / 128` is computed inside the kernel.

### 2d · Fused QKV entries

Gemma4 uses Q/K norms *after* projection (batched per-head rmsnorm), so
QKV fusion is unlikely — the fused kernel bypasses per-head norms. Mark
as non-goal.

### 2e · Wire asym4 / asym2 through `AttentionFamily` (deferred from Phase 1.5)

The kernel infrastructure (window+ring+hd512 tile kernels, `launch_maybe_blob`
Rust `_cap` siblings, givens4/2 K-write ring slots) landed in Phase 1.5 §4.5.5
Goal A. This step wires it into the dispatch framework instead of adding
old-style branches to `gemma4.rs`:

1. **Register asym4/asym2 in `dispatch_kv_write`**: add `KvWriteAsym4` /
   `KvWriteAsym2` arms that call `kv_cache_write_asym4_fused` /
   `kv_cache_write_asym2_fused` (these already exist as GPU methods). The K-side
   ring-buffer `cache_capacity` threads through `AttnParams.physical_cap`.
2. **Register asym4/asym2 in `dispatch_attend`**: add `AttnFlashAsym4` /
   `AttnFlashAsym2` arms that call `attention_flash_asym4_cap` /
   `attention_flash_asym2_cap` with `window_size` + `cache_capacity` from
   `AttnParams`. The hd512 case needs no special handling — `n_halves` is
   computed inside the tile kernel.
3. **Remove the `// NOTE: asym4/asym2 full-layer branches intentionally omitted`
   comment** from `gemma4.rs` full-layer decode — those modes now route
   through `Step::Attend` like asym3/q8/fp32.
4. **Sliding layers**: same `AttentionFamily` path, just with
   `cache_capacity = sliding_window` and `window_size = sliding_window`.
5. **Gate**: `gemma4_oracle` logits match fp32 path at 1200 tokens for both
   asym4 and asym2; coherence gate pass.

### Phase 2 gate

```bash
./scripts/coherence-gate.sh --gemma4   # coherence pass + KLD ≤ threshold
./scripts/speed-gate.sh --gemma4       # decode tok/s within ±3% of Phase 1
```

---

## 6 · Phase 3 — Migrate prefill path ✅ DONE

**Goal:** batched prefill uses `GemmFamily` for GEMM projections and
`AttentionFamily` for batched attention.

**All sub-steps complete:**
- §3a GEMM through `GemmFamily::run_key()` — `run_prefill_gemm` helper
  routes 15 `weight_gemm` calls through the dispatch framework with
  explicit dispatcher-entry keys (GemmHfq4G256 / GemmHfq4G128).
- §3b Batched attention through `AttentionFamily` — sliding and full
  layer KV-write + attention migrated to `Step::Attend` with
  `batch_size = n_batch`. Dispatch routes to
  `AttnFlashAsym3BatchedMasked(tree_bias=None)` which calls the same
  underlying `attention_flash_asym3_tile_batched` HIP kernel.

**Preserved:** the v2 prefill structure (token-batched projections,
per-head Q/K/V norms, dual KV caches, hd512). No structural changes
— only the dispatch routing changed.

### Phase 3 gate

Prefill tok/s within ±3% of Phase 1 baseline on gfx1100 + gfx1201.

---

## 7 · Phase 4 — MoE path migration 🔄 IN PROGRESS

**Note:** MoE GPU methods are stubbed (return `HipError`) in Phase 1, so the
26B-A4B model cannot run *yet* — all 8 MoE GEMV stubs need real implementations.
The **artifact is no longer a blocker**: the full 26B-A4B BF16 safetensors are
present (`/local/models/google/gemma-4-26B-A4B-it`, both shards, ~49 GB, 2026-06-08).
First Phase-4 step is to quantize it (`hipfire-quantize`, `arch_id=12`, MoE expert
table handling) and symlink it for the coherence gate; then implement the stubs.

**Status:**
- Quantizer support: ✅ (expert 3D split, router Q8, router.scale fix)
- 26B-A4B quantized: ✅ (15.6 GB, `/local/models/google/gemma-4-26B-A4B-it-mq4.hfq`)
- Model loads: ✅ (~3 min, 30 layers × 128 experts)
- MoE decode via legacy path: ✅ (per-expert CPU loop, 11.6 tok/s)
- MoE prefill via legacy path: ✅ (per-token loop, slow but correct)
- Output quality: ❓ (garbled — needs investigation)
- Fused MoE GEMV kernels: 🔲 (stubs remain, deferred)

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

## 8 · Phase 5 — Validation 🔲 NOT STARTED

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

## 8.5 · Phase 6 — gemma4 kernel performance (FOLLOW-UP)

A dedicated optimization pass on the gemma4-specific kernels, distinct from the
correctness/regression perf gates in Phase 5. Follow
`docs/methodology/perf-benchmarking.md` throughout (warm DPM + kernel cache; warm
each A/B cell; `scripts/probe_commits.sh` across fresh processes; ±1–3% band;
Δ≥5% is real signal).

### 6.0 · Baseline from jukefr's perf work (issue #270)

jukefr profiled the **original** `feat/gemma4-128k-ring-buffer` branch on **RDNA4
(gfx1201), 26B-A4B MoE, MQ4** — the most complete prior perf data. Final state vs
llama.cpp:

| Test | llama.cpp | hipfire (jukefr) | gap |
|---|---|---|---|
| **tg128 (decode)** | 77 t/s | **71 t/s** | ~0.92× — competitive |
| **pp1270 (prefill)** | 3925 t/s | **342 t/s** | **~0.09× — 11× behind** |

Key takeaways that **reframe this phase**:
- **Decode is NOT the problem on capable HW.** 71 vs 77 t/s on RDNA4 is close to
  parity. The ~15 tok/s seen on gfx1151/12B-q8 is mostly the Strix Halo APU's
  ~4× lower memory bandwidth (115 GB/s LPDDR5X vs RDNA4 GDDR6) — **not directly
  comparable**, and per [[project_gfx1100_is_primary_deploy_target]] perf should be
  judged on gfx1100/gfx1201, not gfx1151. Re-baseline decode on gfx1100/1201 before
  treating it as subpar.
- **Prefill is the dominant gap (11×).** jukefr took prefill 60→342 t/s via
  **token-batched prefill wiring (+38.5%)**, **batched dense projections (+55%)**,
  and **sliding-layer batching (→300, →350 with full-attention batching)**. The
  dispatch daemon currently does **per-token prefill** (`forward_scratch` loop) —
  i.e. it **regressed jukefr's batched prefill**. Recovering it (Phase 3) is the
  single biggest perf win, not a follow-up nicety.
- jukefr's numbers were MoE/MQ4 on the original branch; reproduce on the dispatch
  branch + dense 12B before/after each change.

### 6a · Occupancy audit after the kernel extensions ⚠️ do first

The hot attention kernels were **extended in place** during the long-context work
(`attention_flash_asym3_tile` gained `window_size` + `cache_capacity` params, an
out-of-window predicate, an early-return tile path, and `slot = t % cap` indexing;
the same will land on the q8/asym4/asym2 tiles in §4.5.5 Goal A). Added registers
and branches can raise VGPR pressure → lower occupancy → slower decode, or cause
spills. **Audit before assuming the extension was free.**

Use the **`gfx-kernel-metadata` skill** to extract VGPR/SGPR/LDS/spill counts from
the compiled `.hsaco` and compute theoretical occupancy, for each extended kernel:
- `attention_flash_asym3_tile` (+ the hd512 variant) — pre vs post `kv_window`/`cache_capacity`.
- `attention_flash_q8_0` (now `_cap`, n_halves path), asym4/asym2 tiles after Goal A.
- `kv_cache_write_*` ring siblings.

**Requirements:** **zero spills**; occupancy not regressed vs the pre-extension
baseline. If VGPR pressure rose, consider hoisting the window/cap math out of the
inner loop, `__launch_bounds__` tuning, or splitting a windowed kernel variant from
the full-causal one rather than branching at runtime (avoids the always-resident
predicate cost on the hot ≤1024 path).

### 6b · Speed-up — prefill first, then decode kernels

**Priority 1 — prefill, in two milestones.** The 11× gap is the headline.

- **Milestone 1 — recover jukefr's state (~342 t/s).** Wire Phase 3's batched
  prefill (`forward_prefill_batch_v2`) into the daemon and re-port jukefr's
  optimizations (token-batched prefill, batched dense projections, sliding-layer +
  full-attention batching → ~342 t/s on the original branch). The dispatch daemon's
  per-token prefill is the regression; this is the immediate win. Gated on the
  batched full-layer kernels (Phase 3 §3b).
- **Milestone 2 — close to within 10% of llama.cpp (FOLLOW-UP).** ~342 t/s is still
  only ~9% of llama.cpp's ~3925 t/s (pp1270, RDNA4). Target **≥ ~3530 t/s** (within
  10%) — a further ~10× over Milestone 1, a major optimization effort distinct from
  the recovery. Likely levers: **WMMA/MFMA-accelerated prefill GEMMs** (QKV, gate/up/down,
  MoE grouped-GEMM — the prefill is GEMM-bound, and llama.cpp's RDNA4 throughput
  implies hardware-matmul utilization hipfire's `gemm_hfq4g256`/indexed-MoE path may
  not be reaching), larger effective batch / better tiling + occupancy, kernel fusion
  to cut launch + memory traffic, and attention-prefill (PFlash-style) tiling. Profile
  to find whether the gap is GEMM efficiency, attention, or MoE routing before
  committing. Gate each step with `probe_commits.sh` on gfx1100/1201 + coherence.

**Priority 2 — decode kernels (re-baseline on gfx1100/1201 first; likely close to
parity already).** Profile (`rocprof` / gfx1151 PMC per `project_gfx1151_pmc_works`,
but **judge on gfx1100/1201**) and attack bottlenecks only if a real gap remains:
- **hd512 full-attention** (`attention_flash_asym3_hd512` + reduce) — 512-dim heads,
  2560 B LDS; check tile size / occupancy / bandwidth utilization.
- **Per-head Q/K/V norms** (`rmsnorm_batched` ×3/layer) and **RoPE** (`rope_f32` /
  `rope_partial_halved_f32`) — many small launches/token; fusion/batching
  opportunities (the "stay as direct GPU calls" ops from §2a).
- **Per-token dispatch overhead** — 48 forward_scratch calls/token with per-layer
  KV-mode branching; quantify launch overhead vs compute.

**Gate:** report tok/s deltas with prompt md5 + warm-cache protocol; reproduce on
the dispatch branch + dense 12B; any win must pass `coherence-gate.sh` (a tok/s
gain that ships an attractor is not a win — see the perf methodology's
synth-win→prod-falsify log).

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

| Risk | Phase | Status | Mitigation |
|------|-------|--------|------------|
| `arch_id=7` collision (gemma4 vs qwen2) | 1e | ✅ Resolved | gemma4 → 12. Re-quantized artifacts. |
| `cache_capacity` breaks existing models | 0a | ✅ Resolved | All callers pass `physical_cap`. |
| `head_dim` routing selects wrong kernel | 0b, 2c | ✅ Resolved | Struct-level field added. |
| `v_norm_ones_full` never initialized | 1e | ✅ Resolved (`c7ce94c0`) | Added `init_scratch_constants()` call. |
| SPM-BPE encoder prepended `▁` | 1g | ✅ Resolved (`5ffd0314`) | Removed erroneous prepend. |
| hd512 attention missing reduce kernel | 1d | ✅ Resolved (`2e36fee2`) | Added reduce kernel launch after tile. |
| HF reference double-scaled | debug | ✅ Resolved | `embed_tokens` already scales; oracle corrected. |
| Decode looped `<turn|>` forever (no stop) | 1e | ✅ Resolved | `eos_token_id` is `[1,106]` parsed as scalar 1; `generate_gemma4` now stops on a set incl. `<turn|>`=106. 12B-q8 stops cleanly. |
| "deeper divergence remains" (Session 14) | debug | ✅ Not a bug | Standalone `debug_gemma4_attention` harness has a position-advance bug; daemon battery returns correct distinct answers (Tokyo/42/banana/FR). 4th measurement artifact. |
| Chat-template framing (empty `<\|channel>thought`) | 1e | ✅ Resolved | `generate_gemma4` frames `<bos><\|turn>user\n{p}<turn\|>\n<\|turn>model\n<\|channel>thought\n<channel\|>` (guarded on the 4 special tokens; raw fallback). Output is now clean: "The capital of France is Paris." / valid haiku / "7 times 6 is 42." |
| >1024 correctness (sliding window) | 4.5.1 | ✅ Done (`876c1158`) | fp32 `attention_flash` window fix (`41bd5d87`) + daemon sliding KV sized at `max_seq` + refusal guard dropped. 1266-token prompt coherent. Oracle argmax=236761 matches HF at 1200 tok. |
| Sliding-window **ring buffer** (memory, 128k) | 4.5.2 | ✅ Done | Daemon sliding KV switched to q8 ring-buffer (`new_gpu_q8_capped`, `physical_cap=sliding_window=1024`). Constant ~300 MB regardless of context. Coherent at 1266 tokens. |
| q8 / asym4 / asym2 window+ring+hd512 | 4.5.5 | ✅ Infra done, wiring deferred | HIP tile kernels + Rust `_cap` siblings landed for all three (window+ring+hd512). q8 sliding+full wired in gemma4.rs. asym4/asym2 model-crate wiring deferred to Phase 2 step 2e (dispatch-unification principle). |
| fwht3 / fwht4 KV not wired for gemma4 | 4.5.5 | 🔲 Follow-up | gemma4 forward never branches on `quant_fwht`; no gemma4 fwht hd512 kernels. Net-new (not a re-port): new forward branches + windowed fwht tile kernels + hd512 variants. Daemon `new_gpu_fwht{3,4}_*` alloc already exists. Separate PR after Phase 1.5 (§4.5.5 Goal B). |
| Kernel-extension occupancy regression | 6a | 🔲 Follow-up | `attention_flash_asym3_tile` (+ q8/asym4/asym2 after Goal A) gained params/branches in place; verify no VGPR spills / occupancy drop via the `gfx-kernel-metadata` skill before assuming free. §8.5 Phase 6a. |
| gemma4 prefill: recover jukefr (~342 t/s) | 6b | 🔲 Follow-up | Milestone 1. Dispatch daemon regressed to per-token prefill — wire Phase 3 batched prefill + re-port jukefr's batching (#270). §8.5 Phase 6b. |
| gemma4 prefill: within 10% of llama.cpp | 6b | 🔲 Follow-up | Milestone 2 (separate, ambitious). Target ≥ ~3530 t/s (pp1270) vs jukefr's 342 — a further ~10×. Needs WMMA/MFMA prefill GEMMs + MoE grouped-GEMM + fusion/tiling. **Dominant perf gap.** §8.5 Phase 6b. |
| gemma4 decode perf (re-baseline needed) | 6b | 🔲 Verify | ~15 tok/s on gfx1151/12B-q8, but jukefr shows **71 vs 77 t/s (near parity)** on RDNA4/26B — the gfx1151 number is mostly the APU's 4× lower bandwidth. Re-baseline on gfx1100/1201 before treating decode as subpar. §8.5 Phase 6.0. |
| `gelu_tanh` vs SiLU activation mismatch | 4a | Open | Forked `run_moe_decode_gemma4` executor. |
| hd512 kernels not precompiled for all archs | 1b/5b | Open (gfx1151 works) | Compile + validate per-arch. |
| `rope_partial_halved` not in dispatch framework | 2a | Open | Direct GPU call in Phase 1. |
| MoE GPU methods stubbed | 4 | Open | 26B-A4B cannot run. Deferred to Phase 4. |
| CLI tokenizer fails for gemma4 262K BPE | 1g | ✅ Resolved | Was the **stale prod daemon** (`~/.hipfire/bin/daemon`, predated the SPM-BPE ▁-detection fix), not a code bug. Refreshed binary → `hipfire run gemma-4-12B-it-q8 "..."` → "The capital of France is **Paris**." See [[feedback_hipfire_run_uses_prod_daemon]]. |
| Phase gate scripts have no gemma4 rows | 0c | ✅ Partial | Added `gemma-4-12B-it-q8.hfq` cap row to `coherence-gate.sh` SHORT_TESTS (skip-if-missing). Exercises hd512 reduce + `<turn\|>` stop + framing. DFlash/A3B rows pending (MoE not portable). |
| 26B-A4B safetensors incomplete | 0c | ✅ Resolved | Full BF16 (both shards, ~49 GB) present at `/local/models/google/gemma-4-26B-A4B-it` (2026-06-08). MoE blocker is now code (stubs), not artifact. |
| Graph capture pointer staleness | 2a | Open | Deferred to Phase 2. |
| Daemon `arch_id` wildcard fallback misroutes | 1e | ✅ Resolved | Explicit `12 =>` arms at all 16+ sites. |
| SPM-BPE tokenizer conflicts with dots-ocr | 1g | ✅ Resolved | No conflicts observed. |

---

## 12 · Reference

- Dispatch unification roadmap: [#397](https://github.com/Kaden-Schutt/hipfire/issues/397)
- Dispatch unification PR (DRAFT): [#393](https://github.com/Kaden-Schutt/hipfire/pull/393)
- Phase 0 contracts: [#402](https://github.com/Kaden-Schutt/hipfire/pull/402)
- Canonical branch: `Kaden-Schutt/hipfire:integration/dispatch-unification`
- Gemma4 branch: `feat/gemma4-128k-ring-buffer` (merged to master @ `9b206438`)
- Working branch: `feat/dispatch-unification-gemma4` (tip `41bd5d87` as of 2026-06-08)
- Consolidated adversarial review: `findings/gemma4_dispatch_plan_consolidated_rev.md`
- Individual reviews: `findings/gemm4_dispatch_plan_rev_gemini.md`,
  `findings/gemm4_dispatch_plan_rev_claude.md`,
  `findings/gemma4_dispatch_plan_rev_glm5.md`
- Code review (Claude): `findings/gemma4_dispatch_code_rev_claude.md`
- Code review (Gemini): `findings/gemma4_dispatch_code_rev_gemini.md`
- Debug log: `findings/gemma4_dispatch_devlog.md` (1567 lines, 16 sessions)
- Model sources (HuggingFace):
  - `google/gemma-4-12B-it` — 12B dense, unified (audio+vision+text)
  - `google/gemma-4-26B-A4B-it` — 26B MoE A4B, text+vision
  - `google/gemma-4-31B-it` — 31B dense (TBD)
  - `google/gemma-4-E4B-it` / `google/gemma-4-E2B-it` — Any-to-Any (TBD)
- Local artifacts: `/local/models/google/gemma-4-{12B,26B,31B,E4B,E2B}-it/`
- Quantized artifacts: `/local/models/google/gemma-4-12B-it.{hfq,mq4,q8}`

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

## 14 · Unported jukefr kernels (tuning backlog)

Kate (jukefr) authored several MoE prefill kernels on `feat/gemma4-128k-ring-buffer`
that were **not ported** to the dispatch-unification branch. These are opt-in
paths that need tuning before production use. Listed here for future work.

| Kernel file | Commit | Purpose | Status | Notes |
|---|---|---|---|---|
| `gemv_hfq4g128_moe_down_residual_scaled_bucketed.hip` | `959fb0a` → `6d3a44b` | Routing-bucketed MoE down-proj (HFQ4G128) | **NOT PORTED** | v1 had -5.7% prefill regression; v2 (LDS-staged) closed it. Needs re-benchmark on dispatch branch. |
| `gemv_hfq4g256_moe_gate_up_bucketed.hip` | `959fb0a` → `6d3a44b` | Routing-bucketed MoE gate+up (HFQ4G256) | **NOT PORTED** | Same bucketed path as above. |
| `moe_bucket_build.hip` | `959fb0a` | Token→expert histogram + scatter offsets | **NOT PORTED** | Prerequisite for bucketed path. |
| `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed.hip` | `21a2cedd` | Indexed MoE down for HFQ4G128 weights (decode) | **NOT PORTED** | Used for MQ4 gate_up + HFQ4G128 down model. Our production quant uses Q8_0 down instead, so this kernel isn't load-bearing yet. |
| `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed_batched.hip` | `5012a0c` | Batched indexed MoE down for prefill | **NOT PORTED** | Scaffolding for prefill batching; depends on the indexed kernel above. |

**What we DO have ported** (for comparison):

| Kernel file | Author | Status |
|---|---|---|
| `gemv_q8_0_moe_gate_up_k8_indexed.hip` | Kevin Read | ✅ Production decode path |
| `gemv_q8_0_moe_down_residual_scaled_k8_indexed.hip` | Kevin Read | ✅ Production decode path |
| `gemv_hfq4g256_moe_gate_up_indexed.hip` (contains `k8_indexed`) | Kaden Schutt | ✅ Used for MQ4G256 indexed gate_up |
| `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed.hip` | Kaden Schutt | ✅ Existing HFQ4G128 indexed down (different from Kate's version) |
| `attention_flash_asym3_tile_hd512_batched.hip` | Kate | ✅ Batched hd512 flash-attn |

**Tuning priority:** The bucketed MoE path (`959fb0a`/`6d3a44b`) is the most
impactful — it groups tokens by expert to reduce redundant weight loads during
prefill. On the original branch it closed a prefill regression. Needs re-benchmark
on the dispatch branch to confirm the win still holds with the current MoE
indexed-fast decode path.

---

*Plan authored 2026-06-07. Updated 2026-06-07 with consolidated adversarial
review findings. Config-audited 2026-06-07 against real BF16 model configs.
Status updated 2026-06-07: Phase 0 + Phase 1 done, 12B dense coherent.
Update as phases complete.*
