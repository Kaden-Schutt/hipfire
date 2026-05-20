# dots.ocr (Qwen2-VL family) + Qwen2 text decoder implementation plan

Status: rev 5 — phase 1 closed (PR #297 merged upstream), phase 2a +
2b landed, phase 2c sub-tasks 1-4 of 5 landed. 2026-05-20.
`feat/dots-ocr-qwen2-phase-2` is the active branch; phase 2c-5
(vision_forward assembly + per-stage validation against the 2c-1
.npy refs + end-to-end Qwen2-prefill logit match) is the remaining
work before phase 3 (daemon plumbing) can start.

Filename note: originally filed as `qwen_2.5_vlm.md` matching the
initial request; renamed to `qwen_2.0_vlm_plus_dots_ocr.md` once
verification against the safetensors index confirmed the backbone is
plain Qwen2 (not Qwen2.5). Both crates — `hipfire-arch-qwen2` and
`hipfire-arch-dots-ocr` — land under the Qwen2 family. Generalising
to true Qwen2.5-VL is a stretch goal once dots.ocr lands.

This revision folds in adversarial review findings from three
reviewers (Claude / Gemini / GLM5) on the original draft. The
critical corrections all stem from direct verification against the
dots.ocr safetensors index, `tokenizer_config.json` /
`generation_config.json`, vllm's `dots_ocr.py`, the daemon dispatch
code, and the LLaMA-arch crate header.

## 0. Progress log

Live tracker, updated as commits land on `feat/dots-ocr-qwen2`. Detailed
status per phase in §5.

| commit | scope |
|---|---|
| `c6d4e539` | Bootstrap `hipfire-arch-qwen2` crate (skeleton from toy), `docs/architecture-ids.md`, rev-2 plan |
| `8ab7ec62` | Real `Qwen2Config::from_hfq` parser + Qwen2-1.5B manifest + 4 unit tests |
| `4bf9f6d4` | HFQ4 quantisation of Qwen2-1.5B validated (820 MB, 100% coverage); `inspect_hfq` example |
| `e034c44b` | Real `Qwen2Weights::load` — 28 layers + tied-lm_head + Q/K/V bias; cross-arch TODO markers on both sides |
| `45913eb0` | Rev-2 review fold-in: §2/§5/§6 amendments, tied-F16 lm_head fix (B1), EOS array semantics, lib.rs / doc refreshes, plan rename |
| `9477fbbb` | R1: `hipfire-quantize --arch-id <u32>` flag + re-quantised `qwen2-1.5b.arch7.hfq4` (arch_id byte = `0x07` verified, inspect_hfq --load succeeds) |
| `51e05b99` | R2: LLaMA-family loader (`load_weights_hfq`) now hard-fails when `q_proj.bias` is in the manifest — closes the silent-wrong-output footgun for mis-tagged Qwen2 HFQ files |
| `d7a2ebab` | Phase 0 items 6 + 7: HF reference captured at `benchmarks/references/qwen2_1p5b_instruct_smoke.json` (transformers 5.5.1; 25 KB artifact with first-16 completion IDs + top-100 logits at pos 0/8/14) |
| `00d406af` | R3 mitigation: `infer_qwen2.rs` driver binary — wires the bring-up triple end-to-end. Tokenizer parity confirmed (hipfire's Rust BPE produces byte-identical token IDs to HF on the smoke prompt). |
| `afd4b059` | Phase 1 forward pass: real `Qwen2State` (KV cache + per-step scratch) + `forward_step` / `forward_step_greedy` (28 layers: RMSNorm → fused QKV + bias adds → RoPE → KV cache → attention → o_proj → residual → FFN norm → SwiGLU → residual; final norm + lm_head). HFQ4G256 path landed 9/16 top-1 matches with 7/7 prefix + fluent output (synonym-position divergence consistent with 4-bit quant noise). |
| `9bd083f6` | Phase 1 precision sweep: re-quantised at Q8F16. End-to-end run: **16/16 top-1 matches vs HF F32 reference** — definitive correctness lock-in. Forward in 303 ms (140 ms prefill + 163 ms greedy decode of 16 tokens). Confirms (a) the implementation is correct end-to-end, (b) the HFQ4G256 divergence was 4-bit quant noise, not implementation error. Phase 1 closed. |
| `806680b2` | R3 resolved: daemon arm for `arch_id=7`. Wired `hipfire-arch-qwen2` as a runtime dev-dependency (gated behind new `arch-qwen2` feature, default-on); added `qwen2_config / qwen2_weights / qwen2_state` fields to `LoadedModel` with matching `free_gpu` impls; new load arm constructs the bring-up triple via the `Architecture` trait; new `generate_qwen2` function does encode → prefill → greedy decode → JSON `{type:"token"}` stream → `{type:"done"}`. Verified end-to-end: `hipfire run` (production CLI) against `qwen2-1.5b.arch7.q8.hfq` emits the same continuation as the Q8 precision-sweep run (`"A transformer's attention mechanism is a crucial component of its architecture, which is designed to ..."`) at ~96 tok/s for 137 generated tokens (single-shot, not warmed). Scope-limited bring-up — DFlash / CASK / PFlash / VL / ChatML scaffolding / repeat penalty / top-p / `<think>` budgeting / multi-GPU are all explicitly refused or skipped on this path. |
| `2226bbcf` | Pre-PR fixes from rev-3 review fold-in (Claude+Gemini+GLM-5): A1 (bench_prefill arch_id=7 panic) + A2 (reset event missed Qwen2State.next_pos) + B1 comment (bias path was option (a) not (c) as claimed) + D1 (dead `let _ = decode_t0`) + MINOR doc refreshes across arch.rs / qwen2.rs / README.md / infer_qwen2.rs. `Qwen2State::reset()` helper added and called from both the daemon's `reset` event AND the `bench_prefill` cold-start path. End-to-end regression run (load + bench_prefill[32] + reset + generate[16] + unload) confirms forward output matches the Q8 reference byte-for-byte. |
| _PR open_ | [Kaden-Schutt/hipfire#297](https://github.com/Kaden-Schutt/hipfire/pull/297) — phase 1 deliverable. Awaiting maintainer review + merge. |
| `f6b28a12` | **Phase 2a** — bootstrap `hipfire-arch-dots-ocr` crate (arch_id=8). Mirrors phase 1 sequence from `hipfire-arch-qwen2`: typed Config/Weights/State + Architecture trait impl with dots.ocr-specific EOS overrides. `DotsOcrConfig` wraps Qwen2Config (text) + DotsVisionConfig (vision); `DotsOcrWeights` wraps Qwen2Weights + DotsVisionWeights side-by-side. Vision weight load + `vision_forward` are phase-2c stubs. Tests: 5 passing. |
| `acd75473` | Merge `upstream/master` into `feat/dots-ocr-qwen2`: 47 commits / 779 files / +23.8k lines (license relicense, HFQ6 family, BF16 loader, MoE grouped-WMMA, gfx94x MFMA, Qwen3.5 MoE norm fix). 1 real conflict resolved: `qwen35.rs` dropped `load_norm_weight_raw` (superseded by upstream PR #228 GemmaRMSNorm convention fix). PR #297 mergeable. |
| `544822b4` | **R5** — quantiser packs `generation_config.json` into HFQ metadata. Qwen2 parser walks fallback (`config.eos_token_id` → `generation_config.eos_token_id` → default). Fixes dots.ocr's silent EOS-default-to-`<\|im_end\|>` (151645, never fires) by surfacing the real `[151643, 151673]`. Tests: 8 passing (+2 new for the new fallback path). |
| `1115486a` | **Phase 0 item 2** — read `modeling_dots_ocr.py` + `modeling_dots_vision.py` end-to-end. No plan contradictions. New §2.9 captures: attention scale = plain `1/sqrt(head_dim)`, multi-image batching packs into single flattened sequence (i32 cu_seqlens), bf16 cast at vision forward entry, vision-text integration via `masked_scatter()` with no projection layer, all 42 blocks structurally identical, no dropout/droppath in inference. Gotchas mirrored on `vision_forward` doc. |
| `bfe1f56d` | **Phase 2b** — image preprocessing complete. `smart_resize` (28-divisible, beta scaling, AR>200 guard, zero-dim guard), `clip_normalise` (RGB→CHW f32, CLIP constants), `extract_patches` (the §2.7 silent-failure trap — 2×2-grouped-block-major enumeration), `preprocess_image` top-level wrapper with RGBA→RGB compositing over white. 14 unit tests, no GPU needed. Silent-failure-trap gated by `extract_patches_uses_grid_block_order` against a per-pixel-tagged 28×56 synthetic input. |
| `bc4640b7` | Merge `upstream/master` into `feat/dots-ocr-qwen2` (round 2): 2 commits / 1 conflict in `daemon.rs` (PR #312 OpenAI vision refactored generate_vl to GenerateVLParams; kept our generate_qwen2 + json_escape unchanged, took upstream's new signature). PR #297 still mergeable. |
| `8c04ba46` | **Phase 0 item 5** — dots.ocr HF reference captured across two complementary artifacts. (a) `dots_ocr_smoke_001.json` via HF/CPU/bf16/eager — prefill logits at positions 0/32/128/5094 (last prompt position; top-1='[' with +11 nats gap, confirming valid forward pass through prompt + image embeds). Greedy decode degrades after 5 tokens (documented CPU+bf16 limitation). (b) `dots_ocr_smoke_001_vllm.json` via vLLM 0.13.0/GPU/bf16/flash_attn — 13-element layout JSON, parse_status ok, real ground truth for the phase-4 OCR coherence gate. Image is `dots_ocr/demo/demo_image1.jpg` (1700×2250 medical-paper page, md5 `a434c567a2dfa0664ce75291508bad85`). |
| `7ba3c749` | **Phase 2c-1** — captured 6 intermediate vision-tower activations via PyTorch forward hooks on the same HF/CPU/bf16/eager run: `patch_embed`, blocks 0/21/41, `post_trunk_norm`, `merger`. Full tensors (~120 MB each, [19520, 1536] for patch-shape stages and [4880, 1536] for the merger output) saved to `/data/cache/hipfire/dots_ocr_activations_full/` for local-only use. A uniform 256-row (64 for merger) linspace sample plus the sample-indices list is committed to `benchmarks/references/dots_ocr_smoke_001_activations/` for git (7.9 MB total). Phase 2c per-stage validation reads from the committed sample; full-tensor consumers regenerate locally via the capture script. Statistics across the stack look healthy (std 0.25 → 0.29 → 0.56 → ...; expected residual-network growth). |
| `7051d6e9` | **Phase 2c-2** — vision weight loader. Wires all 17 vision-tensor names: patch_embed (Conv2d → linear reshape [embed_dim,3,14,14] → [1536, 588], bias, RMSNorm), 42× DotsVisionBlock (norm1, attn.qkv, attn.proj, norm2, fc13_proj load-time concat, fc2), post_trunk_norm, merger (ln_q + bias, mlp.{0,2} + biases). Load-time SwiGLU fusion via fc1+fc3 → fc13_proj concat (initially byte-level on raw quant; refactored to F16 byte-level in 2c-4). 4 helpers (`load_norm_weight_raw`, `load_bias_f32`, `load_weight_tensor`, `load_weight_tensor_concat_rows`) all carry `TODO(transformer-extraction)` markers. Tests: 19 still passing. |
| `22d47330` | **Phase 2c-3** — 2-D RoPE prep helper (`rope::build_rope_2d_tables`). Ports `get_pos_ids_by_grid` + `VisionRotaryEmbedding` + `apply_rotary_pos_emb_vision` quarter-repeat layout from `modeling_dots_vision.py` into a single CPU function emitting per-patch [N_patches, head_dim] cos/sin tables in dots.ocr's exact `[hc, wc, hc, wc]` layout. Patch enumeration in 2×2-block-major order matches `image::extract_patches`. 7 new tests including a hand-computed 2×2/head_dim=8 case and a reshape-permute-flatten equivalence check on a 4×6 grid. 26 tests total. |
| `9f738911` | **Phase 2c-4** — vision GPU primitives. (1) New kernel `rope_2d_halfsplit_f32` (`kernels/src/rope_2d_halfsplit.hip` + dispatch fn) — applies the 2c-3 precomputed cos/sin tables to Q/K in-place; halfsplit pairs `(d, d+head_dim/2)`. (2) Loader refactor: vision linear weights now stored as F16 GpuTensor on GPU (HFQ4 / Q8 / F32 source quant types dequantise to F16 at load time per the qwen35-vl pattern — N=~20k patches makes batched HFQ4 GEMM the bottleneck; dequant-on-load + gemm_f16 sidesteps it). DotsVisionBlockWeights + DotsVisionWeights field types changed from WeightTensor → GpuTensor; load_f16_or_dequant + load_f16_or_dequant_concat_rows + dequant_hfq4 helpers added. (3) `linear_f16` + `linear_f16_no_bias` private helpers in dots_ocr.rs (the latter for the use_bias=false vision-block linears). Vision-shape primitive audit confirmed: `rmsnorm_f32`, `silu_mul_f32`, `vit_attention_f32` (non-causal), `bias_add_f32`, `gelu_tanh_f32`, `layernorm_batched`, `gemm_f16`, `add_inplace_f32`, `transpose_f32` all accept [N_patches, hidden] strides — CAVEAT: large-N attention needs `vit_attention_opt` instead of `vit_attention_f32` (the latter materialises N² scores in shared mem, ~77 KB at N≈19520 exceeds RDNA per-CU SLM cap). 26 tests still passing. |
| _wip_ | **Phase 2 review fold-in (rev-claude + rev-glm5 + rev-gemini)** — three-reviewer pass on `f6b28a12..9f738911`. Adjudication: rev-claude A1 (out_hidden_size fallback) VALIDATED → fixed; rev-claude B1 (smart_resize upscale re-clamp) VALIDATED → fixed; rev-claude B2/B3 (rope_2d dispatch missing launch_maybe_blob + profile + guards) VALIDATED → fixed (now wired through `launch_maybe_blob` + `begin_timer` like neighbouring kernels, `head_dim % 4` assert matches table builder); rev-claude B4 (qwen2 `load_norm_weight_raw` missing length assert) VALIDATED → harmonised; rev-claude C2 (concat_rows missing length asserts) VALIDATED → added; rev-claude C5 (TPS>1 doc) VALIDATED → added; rev-glm5 A1 (rope kernel head bounds) REJECTED (body work IS inside guards — no OOB; GLM-5 misread); rev-glm5 A2 (vision_forward GPU signature) VALIDATED, merged with rev-claude C3 → stub now takes `&GpuTensor` patches and returns `HipResult<GpuTensor>`; rev-glm5 A3 (load_f16_or_dequant qt=3) PARTIAL → panic message improved (matches qwen35-vl gap; defer qt=3 arm to phase 5); rev-gemini 3.1 (multi-image attention leakage in vit_attention_opt) DOCUMENTED → vision_forward now has explicit single-image-only doc + plan §5 phase 3 spec calls for per-image loop in daemon; rev-gemini 3.2 (IMGPAD count assertion) DOCUMENTED in plan §5 phase 3 as a hard splice-site assert; rev-gemini 3.3 (R5 still listed in §6.1 deferred) FIXED → §6.1 R5 marked resolved with reference to §6. 34 tests passing (26 dots-ocr + 8 qwen2). Review scaffolding files dropped per `feedback_drop_review_files_after_fold_in`. |

Verified at *load* time on gfx1151 via `inspect_hfq --load`
(no forward pass, no token output yet):
- Quantiser handles Qwen2 layer naming (closes risk M9 in §6).
- Config parser yields exact match with Qwen2-1.5B-Instruct on all 13
  fields, including the two non-trivial defaults (`attention_bias=true`
  via Qwen2 modeling default; `tie_word_embeddings=true` extracted).
- All 28 layers + tied lm_head + Q/K/V bias upload to GPU without
  error; dimensions match config (wq 1536×1536, wk 256×1536, lm_head
  151936×1536).
- Tied-embedding detection works; lm_head re-uploads embed_tokens
  (~117 MB extra) since `GpuTensor` is not `Clone` — documented as a
  follow-up consolidation candidate (see §6).

**Not yet verified**: forward pass correctness against HF, logit /
token-id match, coherence gate. These are blocked on phase 0 items
5/6/7 (HF reference) plus the forward-pass port plus a standalone
forward driver (since the crate isn't wired into the daemon — see R3
in §6).

Discovered during implementation (not in original plan):
- `HipResult` lives in `hip_bridge::error`, not `rdna_compute`.
- `GpuTensor` doesn't implement `Clone`; tied embeddings cost ~117 MB
  VRAM duplication on Qwen2-1.5B at HFQ4. Resolvable via
  `GpuTensor::shallow_clone()` or `Arc<GpuTensor>` in the
  Transformer-extraction PR.
- The qwen35 weight-loading helpers (`load_norm_weight*`,
  `load_weight_tensor*`) are all private. Cross-arch reuse currently
  requires duplication; both sides marked with
  `TODO(transformer-extraction)` for the future consolidation PR.
- `hipfire-quantize` auto-assigns `arch_id=1` to Qwen2 inputs (existing
  Qwen2/3 default). Our `hipfire-arch-qwen2` claims `arch_id=7` to
  avoid colliding with the LLaMA crate, so the HFQ needs a per-file
  arch_id remap (or a quantiser CLI flag) before daemon dispatch.
- `hipfire_runtime::llama::EmbeddingFormat` has no `F16` variant — the
  llama / qwen35 loaders always expand F16 source → F32 on host before
  upload for tied embeddings. The qwen2 loader must follow the same
  pattern; a naive `upload_raw(&data, ...)` on F16 bytes paired with
  `gpu_dtype: F32` produces a corrupted lm_head (caught and fixed in
  the rev-2 patch; was latent because all current HFQ files use
  HFQ4G256 for embeddings).
- The dots.ocr `eos_token_id` lives in `generation_config.json`, not
  `config.json`. `hipfire-quantize` does **not** pack
  `generation_config.json` into HFQ metadata, so the config parser's
  EOS lookup silently mis-defaults for dots.ocr (Qwen2-1.5B-Instruct
  is unaffected because its `config.json` does carry `eos_token_id`).
  Phase 3 must either teach the quantiser to merge `generation_config`
  or special-case the EOS via `eos_filter_overrides`.

## 1. Goal and scope

Two crates land in this plan:

1. **`hipfire-arch-qwen2`** — plain Qwen2 text decoder. Validated
   against the downloaded Qwen2-1.5B-Instruct checkpoint at
   `/home/kread/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/`
   (config: hidden=1536, 28 layers, 12 heads, 2 KV heads, head_dim=128,
   intermediate=8960, vocab=151936, `tie_word_embeddings=True`). The
   transformer body shape is identical to dots.ocr's text backbone,
   so getting Qwen2-1.5B to run correctly in hipfire solves the
   dots.ocr text path. Claims **`arch_id = 7`** (next-free slot;
   `arch_id = 1` is already covered by `hipfire-arch-llama`'s
   Qwen2/Qwen3 branch — see §3a).
2. **`hipfire-arch-dots-ocr`** — the dots.ocr-specific vision tower
   plus a text trait impl that delegates to `hipfire-arch-qwen2`
   directly (no weight remap needed — dots.ocr stores text weights
   as `model.*`, identical to plain Qwen2). Claims **`arch_id = 8`**
   (Qwen2-VL family). Minimum success: load the bf16 weights from
   `/data/cache/huggingface/hub/models--rednote-hilab--dots.ocr`,
   process one page image, emit the JSON layout output that the
   upstream model produces. Quantised path (HFQ4 / MQ4) is a stretch
   phase.

True Qwen2.5-VL coverage (m-rope + window/full attention split + the
Qwen2.5 text backbone) is treated as a follow-on.

## 2. Architecture (verified)

The HF modeling code in the snapshot
(`models--rednote-hilab--dots.ocr/snapshots/c0111ce6.../modeling_dots_ocr.py`)
inherits **`Qwen2ForCausalLM`**, not Qwen2.5. vllm's `dots_ocr.py`
uses `Qwen2_5_VLProcessor` for chat-template plumbing only — the
model class is Qwen2-derived. Implications:

- Text decoder = plain Qwen2 (GQA, RMSNorm, SwiGLU, 1-D RoPE, no
  m-rope, no DeltaNet recurrence, no MoE, **`attention_bias = true`
  on Q/K/V projections**).
- Vision tower = custom `DotsVisionTransformer` (RMSNorm + SwiGLU FFN,
  full attention, non-causal).
- Image-token wrapper sequence inside the chat is
  `<|img|> <|imgpad|>×N <|endofimg|>` (151666 / 151665 / 151667).
- Chat framing is **custom, not ChatML** (see §2.5).

### 2.1. Text backbone (from dots.ocr config.json)

| field | value | notes |
|---|---|---|
| hidden_size | 1536 | |
| num_hidden_layers | 28 | |
| num_attention_heads | 12 | |
| num_key_value_heads | 2 | GQA, 6:1 |
| head_dim | 128 | |
| intermediate_size | 8960 | |
| vocab_size | 151936 | |
| max_position_embeddings | 131072 | |
| rope_theta | 1_000_000 | |
| rope_scaling | null | 1-D RoPE, no m-rope |
| attention_bias | true | Q/K/V projections have bias |
| hidden_act | silu | |
| rms_norm_eps | 1e-6 | |
| tie_word_embeddings | false | **separate lm_head.weight on disk** |
| use_sliding_window | false | SWA disabled; plan ignores it |
| torch_dtype | bfloat16 | |

For comparison, **Qwen2-1.5B-Instruct** has the same transformer
body shape with two deltas: `tie_word_embeddings = true` (no
separate lm_head tensor; reuses embed_tokens) and `attention_bias`
not set in config (Qwen2 modeling-code default is `true`, same as
dots.ocr). The `hipfire-arch-qwen2` loader must detect
`tie_word_embeddings` from config and either alias lm_head to
embed_tokens or copy at load time.

Parameter count verified at ~3.04B total (~2.37B text + ~670M
vision); the earlier "~760M" agent estimate was wrong.

### 2.2. Vision tower (from `vision_config`)

| field | value |
|---|---|
| embed_dim | 1536 |
| num_hidden_layers | 42 |
| num_attention_heads | 12 |
| intermediate_size | 4224 |
| patch_size | 14 |
| spatial_merge_size | 2 |
| temporal_patch_size | 1 |
| num_channels | 3 |
| use_bias | false — applies to **both** attention (qkv + proj) AND SwiGLU FFN (fc1, fc2, fc3). Only `patch_embed.proj` and the merger linears carry bias. |
| post_norm | true |
| rms_norm_eps | 1e-5 |
| hidden_size (post-merger out) | 1536 (matches LM) |

The `use_bias` flag controls every `nn.Linear` in `DotsVisionBlock`,
not just attention. Verified at
`modeling_dots_vision.py:329-333` (`DotsSwiGLUFFN.__init__`):

```python
bias = config.use_bias
self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)
```

and confirmed against the manifest (no `vision_tower.blocks.*.mlp.fc*.bias`
or `*.attn.{qkv,proj}.bias` entries — only the merger and patch_embed
have biases on disk).

Block layout: pre-norm RMSNorm → attention (qkv merged, **non-causal**)
→ residual → pre-norm RMSNorm → SwiGLU MLP → residual. Final RMSNorm
if `post_norm=true`.

### 2.3. Patch embedding — has bias; weight is 4-D

The vision config sets `use_bias: false` but this **does not** apply
to the patch embed: `patch_embed.patchifier.proj` is an
`nn.Conv2d(num_channels, embed_dim, kernel=patch_size,
stride=patch_size)` constructed independently of `use_bias`, and the
manifest confirms a bias tensor on disk:

```
vision_tower.patch_embed.patchifier.proj.weight   [1536, 3, 14, 14]   ← 4-D
vision_tower.patch_embed.patchifier.proj.bias     [1536]
vision_tower.patch_embed.patchifier.norm.weight   [1536]   (RMSNorm)
```

The weight is the standard 4-D `[out_channels, in_channels, kH, kW]`
Conv2d weight; reshape directly to `[1536, 588]` for the GEMM, no
`.squeeze(2)` needed.

The `temporal_patch_size=1` axis lives on the **input** tensor, not on
the weight. `modeling_dots_vision.py:357-358` applies it to the input
right before the Conv2d call:

```python
x = x.view(-1, self.num_channels, self.temporal_patch_size,
           self.patch_size, self.patch_size)[:, :, 0]
x = self.proj(x).view(-1, self.embed_dim)
```

For `temporal_patch_size=1` (dots.ocr's setting), the `[:, :, 0]` slice
is a no-op the porter can skip — patch tensors already arrive in
`[N, C, H, W]` layout when the upstream image processor uses
`temporal_patch_size=1`.

### 2.4. PatchMerger — LayerNorm (NOT RMSNorm), MLP has bias

The merger's pre-norm is **LayerNorm with eps=1e-6**, not RMSNorm.
Verified directly: safetensors index contains
`vision_tower.merger.ln_q.bias` (LayerNorm has bias; RMSNorm does
not). vllm `dots_ocr.py:184-198` defaults `pre_norm="layernorm"`.

The merger MLP has bias on both linears:
```
vision_tower.merger.ln_q.weight     [1536]
vision_tower.merger.ln_q.bias       [1536]
vision_tower.merger.mlp.0.weight    [6144, 6144]
vision_tower.merger.mlp.0.bias      [6144]
vision_tower.merger.mlp.2.weight    [1536, 6144]
vision_tower.merger.mlp.2.bias      [1536]
```

Forward: `LayerNorm(x) → view(-1, 6144) → linear(6144→6144) + bias
→ GELU → linear(6144→1536) + bias`.

Use `gpu.layernorm_batched` (already exists in
`kernels/src/layernorm.hip`) for the pre-norm. Apply bias via
`bias_add_f32` after each linear (precedent: vision QKV bias in
`qwen35_vl.rs:52`).

### 2.5. Chat template — custom framing, NOT ChatML; primary EOS = 151673

The chat template from `tokenizer_config.json` (Python `repr()` of the
JSON string):

```jinja2
{%- for m in messages %}
  {%- if m.role == 'system' %}
    {{- '<|system|>' + m.content + '<|endofsystem|>\n' }}
  {%- elif m.role == 'user' %}
    {{- '<|user|>' + m.content + '<|endofuser|>' }}
  {%- elif m.role == 'assistant' %}
    {{- '<|assistant|>' + m.content }}
    {%- if not loop.last %}{{- '<|endofassistant|>' }}{%- endif %}
  {%- endif %}
{%- endfor %}
{%- if messages[-1].role != 'assistant' %}{{- '<|assistant|>' }}{%- endif %}
```

**Framing token taxonomy — special vs BPE-fragmented.** The six framing
literals split into two groups against `added_tokens_decoder`:

| literal in template | special-token ID | how the runtime emits it |
|---|---|---|
| `<\|user\|>`          | 151670 | single special-token ID |
| `<\|endofuser\|>`     | 151671 | single special-token ID |
| `<\|assistant\|>`     | 151672 | single special-token ID (note: includes the closing `>`; verified verbatim against the JSON) |
| `<\|endofassistant\|>`| 151673 | single special-token ID |
| `<\|system\|>`        | **NOT in vocab** | emit raw bytes, let BPE fragment |
| `<\|endofsystem\|>`   | **NOT in vocab** | emit raw bytes, let BPE fragment |

The `<|systemprompt|>` (151668) and `<|endofsystemprompt|>` (151669)
tokens *do* exist in `added_tokens_decoder`, but the chat template
does **not** use them. They are vestigial; do not emit either.

**EOS.** `generation_config.json` declares `eos_token_id: [151643,
151673]`. The array form means *either* token terminates a turn,
but the chat template's `<|endofassistant|>` (151673) is the
load-bearing one for assistant turn-end. 151643 `<|endoftext|>` is
the wire-EOS used outside of chat (e.g., raw completion). Both must
be in the runtime's stop-set; neither alone is sufficient.

Token 151645 `<|im_end|>` exists in the vocab but the dots.ocr chat
template does not use it — the default hipfire EOS filter (which
looks for `<|im_end|>`) will never fire on a correct dots.ocr
response.

Both `prompt_frame_overrides` and `eos_filter_overrides` **must be
customised** for dots.ocr — they cannot stay at the qwen35 default.

**BPE-fragmentation verification.** Phase 3 must include an empirical
test that hipfire's Rust BPE produces the same token ID sequence for
the literal byte strings `<|system|>` and `<|endofsystem|>` as the HF
tokenizer does. If they diverge, the prefill substring won't match
the training distribution and the model output will degrade in a way
that's hard to localise. Cheap to test (round-trip a known prompt
through both tokenizers); only needs to run once per tokenizer change.

### 2.6. RoPE

- **Text:** standard 1-D RoPE, theta=1e6. Reuse hipfire's existing
  text-side RoPE infrastructure.
- **Vision:** 2-D spatial RoPE over the (H, W) patch grid, theta=10000.
  hipfire does not currently have a 2-D RoPE kernel; the existing
  `rope_partial_halfsplit_f32` is 1-D text RoPE. Need either a new
  `rope_2d_f32` kernel variant or a careful CPU-side cos/sin table
  generator that emits per-patch frequencies for the head_dim halves
  (one half rotated by h-frequency, the other by w-frequency).

  Critically: position IDs must be **reshape-permute-flattened**
  before RoPE application. Per `dots_ocr.py:572-597`:

  ```python
  hpos_ids = hpos_ids.reshape(h//sm, sm, w//sm, sm)
  hpos_ids = hpos_ids.permute(0, 2, 1, 3)   # group 2×2 neighbours
  hpos_ids = hpos_ids.flatten()
  # same for wpos_ids
  ```

  This groups 2×2 spatial neighbours to be contiguous in the
  sequence dimension *before* the merger's `view(-1, 6144)`. Without
  this permutation: RoPE applies wrong frequencies to wrong patches
  **and** the merger groups the wrong 4 patches together → garbage
  visual tokens that still produce plausible-looking JSON.

### 2.7. Image preprocessing

**Patch extraction order (critical — silent-failure trap).** The HF
`Qwen2VLImageProcessor` (which dots.ocr uses per
`preprocessor_config.json:image_processor_type:"Qwen2VLImageProcessor"`)
performs a non-obvious reshape+transpose on the raw pixel array
*before* tokens leave the image processor. From
`image_processing_qwen2_vl.py:281-295`:

```python
patches = patches.reshape(
    grid_t, temporal_patch_size, channel,
    grid_h // merge_size, merge_size, patch_size,
    grid_w // merge_size, merge_size, patch_size,
)
patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
flatten_patches = patches.reshape(
    grid_t * grid_h * grid_w,
    channel * temporal_patch_size * patch_size * patch_size,
)
```

The `transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)` is the **patch-side**
counterpart to the position-ID permute in §2.6: both reorder
raster-scan neighbours into grid-block order so the merger's later
`view(-1, 6144)` groups the right 2×2 blocks.

If the hipfire image preprocessor emits patches in raw raster order
without this transpose, the merger will fuse horizontally-adjacent
patches instead of 2×2 spatial blocks. The model still produces
plausible-looking JSON output but **bounding-box coordinates are
wrong by a 2×2-tile-sized offset that varies by image width**. This
is the canonical silent-failure mode for this model family.

The phase 4 coherence gate's box-IoU check (§5 phase 4) catches this
empirically. The phase 2 implementation should also write a unit test
that, given a known synthetic input, reproduces the HF processor's
exact `flatten_patches` byte sequence — independent of any forward
pass.

**Smart-resize** (`dots_ocr/utils/image_utils.py:29-63`):

- IMAGE_FACTOR = 28 (= 2 × patch_size); resized H and W must both be
  divisible by 28.
- Clamp total pixels to `[min_pixels=3136, max_pixels=11_289_600]`.
- Algorithm: `round_by_factor` to 28-multiples first; if the result
  exceeds `max_pixels`, apply a `beta` scaling factor to bring it
  back under the limit while preserving aspect ratio.
- Preserve aspect ratio; reject AR > 200:1.
- Normalise to CLIP-style mean
  `[0.48145466, 0.4578275, 0.40821073]` / std
  `[0.26862954, 0.26130258, 0.27577711]`.
- RGB; convert RGBA → RGB on white background.

Port the exact algorithm (including beta scaling and AR guard) —
off-by-pixel rounding will misalign bboxes in the JSON output.

### 2.8. Output

Pure text generation. Layout JSON, Markdown, or SVG depending on the
prompt template (`dots_ocr/utils/prompts.py`). No separate detection
head — tokens encode bboxes, categories, and content directly.

### 2.9. Phase-0 item-2 findings (modeling_dots_ocr.py + modeling_dots_vision.py)

End-to-end read of the HF source at
`/data/cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots/c0111ce6.../`
captured these subtleties not already covered by §2.1–§2.8. None
contradict the rev-2 plan; all are pre-Phase-2c gotchas.

**Vision tower (modeling_dots_vision.py):**

- **Attention scale = standard `1/sqrt(head_dim)`** across every
  attention impl (eager / eager_v2 / flash_attention_2 / sdpa /
  ascend_fa — lines 130, 202, 246). No learned scale, no qk-norm.
  Use plain `1/sqrt(128)` for the dots.ocr vision attention.
- **Block uniformity**: all 42 blocks are structurally identical
  (pre-norm RMSNorm → attn → residual → pre-norm RMSNorm → SwiGLU
  → residual; line 399-400). First block has residual, last block
  uses the same norm convention. No depth-conditional branches.
- **`image_grid_thw` batch handling** (line 499): for batch_size > 1
  the parser **concatenates** image patches into a single flattened
  sequence (`pos_ids = torch.cat(pos_ids, dim=0)` at line 486;
  cu_seqlens is image-major: cumsum of `repeat_interleave(grid_thw[:,1] * grid_thw[:,2], grid_thw[:,0])`).
  Multi-image batching is NOT a 4-D batched tensor — it's a 2-D
  packed sequence with cu_seqlens marking image boundaries.
  Rust port: keep the same packing order; image-major, not patch-
  major. cu_seqlens must be `i32` for FA correctness (line 501).
- **bf16 cast at vision forward entry** (line 493-494): vision
  `hidden_states` are unconditionally cast to bf16 at the top of
  `VisionTransformerPretrainedModel.forward` if `bf16=True` (the
  default). Don't treat this as training-only — it changes the
  numerics of every downstream activation in the vision tower.
  Rust port: compute the vision tower at f32, cast the final
  merged output to f16 (already in plan).
- **Dropout / DropPath**: none. `dropout_p=0.0` is hardcoded in
  every attention call (lines 289, 324, 339). SwiGLU is plain
  `F.silu()` with no dropout wrapper. Inference and train modes
  produce identical activations in the vision tower.
- **PatchMerger init quirk**: `init_merger_std` zero-inits biases
  and normal-inits weights with stddev `init_merger_std` (lines
  87-91). Conv2d patch_embed inherits PyTorch's default uniform
  init. Neither matters at inference — trained weights are what
  load from disk — but document so a future from-scratch trainer
  doesn't introduce divergence.

**Text + assembly (modeling_dots_ocr.py):**

- **Vision-token splicing uses `masked_scatter()`** (line 61). The
  daemon prefill loop in hipfire should follow the same pattern as
  qwen35-vl: look up text-embed for every `input_id`, then *overwrite*
  the `<|imgpad|>` (151665) positions with the corresponding merged
  visual tokens. The `img_mask = input_ids == IMGPAD_ID` shape is
  `[B, L]` (boolean). NO projection layer between the merger output
  and the text embed space — the merger already emits `[N_patches/4,
  text_hidden_size]` directly.
- **Vision-text dtype mismatch on the integration boundary** (line
  63): vision embeddings are cast to `inputs_embeds.dtype` before
  the `masked_scatter`. If the vision tower runs in bf16 (default)
  but the text decoder runs in f16 or f32, the cast quantises.
  Rust port: ensure the merged visual tokens are at the same dtype
  as the text embedding's GPU buffer before splicing.
- **No projection layer, no KV pre-allocation for image tokens**:
  the assembly is "lookup text tokens → overwrite image-pad slots →
  feed straight to Qwen2 decoder". Standard KV cache allocation —
  no special handling.

**Plan divergences found: none.** §2.1 line 153 (`tie_word_embeddings=false`
on dots.ocr) is confirmed against `configuration_dots_ocr.py:153` and
the manifest's separate `lm_head.weight [151936, 1536]`. §2.2 vision
`use_bias=false` confirmed against `modeling_dots_vision.py:332-333`
(SwiGLU FFN) and `:392` (VisionAttention qkv/proj). §2.4 LayerNorm-
based merger confirmed.

## 3. Reusable hipfire infrastructure

### 3a. The `Architecture` trait (bring-up contract)

`crates/hipfire-runtime/src/arch.rs` is the spec. Forward is **not**
on the trait (static dispatch in hot path), so the arch crate
exposes its own typed forward functions.

Required methods:
- `arch_id() -> u32`, `name() -> &'static str`
- `config_from_hfq`, `load_weights`, `new_state`

Existing arch_id assignments (verified):
- `0` = LLaMA / Mistral (covered by `hipfire-arch-llama`)
- `1` = plain Qwen3 / Qwen2 (also covered by `hipfire-arch-llama`
  per its `lib.rs:4`; the daemon dispatch at `daemon.rs:1494` routes
  everything `< 5` to LLaMA's path).
- `5` = Qwen3.5 dense
- `6` = Qwen3.5/3.6 MoE
- `0xFF` = toy

**Implication for this plan:** `arch_id = 1` is **not free**.
Taking the next-free slots `7` (qwen2) and `8` (qwen2-vl / dots.ocr)
avoids restructuring the LLaMA crate's dispatch. The id=1 slot can
be migrated to the new Qwen2 crate as a follow-on once it's proven.

Optional override hooks (defaults are qwen35 conventions —
ChatML + `<|im_end|>` EOS):
- `loop_guard_overrides`, `sampler_overrides`,
  `prompt_frame_overrides`, `eos_filter_overrides`.

dots.ocr MUST customise both `prompt_frame_overrides` (custom
framing tokens, not ChatML) and `eos_filter_overrides` (151673 as
primary stop, not `<|im_end|>`). Possibly also
`loop_guard_overrides` if layout JSON's short repeats trip the
default n-gram threshold (verify in phase 4).

### 3b. The template: `crates/hipfire-arch-toy/`

`hipfire-arch-toy` is the worked-out trait template. Per
`crates/hipfire-arch-toy/README.md:30-56`:

1. Copy `crates/hipfire-arch-toy/` → `crates/hipfire-arch-<name>/`.
2. Update `Cargo.toml` (`name`, `description`) and add to workspace
   `Cargo.toml` `members`.
3. Replace stub types in `src/toy_model.rs`.
4. Update `src/arch.rs` `impl Architecture` calls.
5. Implement forward as free functions in the model module.
6. Claim an `arch_id` and record it in `docs/architecture-ids.md`
   (create this file as part of phase 1; both the toy README and
   the trait doc reference it as future).

### 3c. Production reference: `hipfire-arch-qwen35-vl`

Closest analog for the **vision** side. Reusable:

- **Trait-impl split pattern** — qwen35-vl has its own `Architecture`
  impl with `type State = ()`; the text decoder is a separate impl.
  Same pattern here: one trait impl for Qwen2 text decoder, one for
  dots.ocr vision tower (which holds the text-side delegation).
- **Image preprocessing skeleton** in
  `crates/hipfire-arch-qwen35-vl/src/image.rs` — generic over
  `patch_size` and `spatial_merge_size`. Need to swap the resize
  policy to dots.ocr's 28-divisible + beta-scaling algorithm.
  Verify whether the [R, B, G] channel reordering quirk asserted by
  `tests/channel_order.rs:44-81` applies — dots.ocr's vision encoder
  is different, so likely **not**, but inspect the patch_embed
  tensor layout before assuming.
- **Daemon vision plumbing** (`daemon.rs:240-242, 4077-4194`) — the
  IMAGE_PAD_ID / VISION_START_ID / VISION_END_ID constants and the
  prefill substitution loop. We add a new triple
  `(IMGPAD=151665, IMG_START=151666, IMG_END=151667)` for dots.ocr.
  **However**, the entire VL path is currently gated `arch_id == 5
  || 6` and the `LoadedModel` struct only holds `q35_*` fields. See
  §5 phase 3 for the daemon plumbing work this entails.
- **HFQ4 quantize pipeline** at
  `crates/hipfire-quantize/src/main.rs:5076-5122` already understands
  `--include-vision` with `--vision-quant {hfq4|bf16}`. Same flow
  applies here, modulo verifying Qwen2 layer naming is supported.
- **rdna-compute kernels** in scope: `GEMM_F16`, `layernorm_batched`,
  `gelu_tanh_f32`, `vit_attention_f32`, `transpose_f32`. Text-side
  RMSNorm/SwiGLU/RoPE live in `hipfire-arch-qwen35`. New kernels
  required: 2-D RoPE for vision, fused QKV+bias for text (or a
  separate bias_add pass — see §5 phase 1, M-level finding).

## 4. Gaps vs. qwen35-vl

The dots.ocr vision tower differs from qwen35-vl's enough that this
is a fork, not a parametrisation:

1. **RMSNorm in vision blocks** (eps=1e-5) — qwen35-vl uses
   `layernorm_batched` (LayerNorm). Vision-shape RMSNorm is **new
   code**, not reuse. The text-side RMSNorm kernel may need a
   variant for `[N_patches, embed_dim]` strides.
2. **SwiGLU in vision FFN** — qwen35-vl uses GELU. Vision-shape
   SwiGLU is new code; check the text-side kernel's shape
   assumptions.
3. **Full non-causal attention** — qwen35-vl already does this via
   `vit_attention_f32`. Direct reuse; verify the kernel supports
   non-causal explicitly (config has `is_causal: false`).
4. **PatchMerger uses LayerNorm + bias, not RMSNorm** — see §2.4.
5. **PatchMerger MLP has bias** — see §2.4.
6. **2-D vision RoPE with position-ID permutation** — see §2.6. Both
   the kernel/algorithm and the permute are new.
7. **Different text backbone** — qwen35-vl is bolted onto the
   `Qwen35` decoder (DeltaNet hybrid). dots.ocr uses plain Qwen2.
   **Decision: fork** to a new `hipfire-arch-qwen2` crate. Qwen2 vs
   Qwen3 differ in two ways that would otherwise turn into config-
   flag bloat in qwen35:
   (a) Qwen3 applies q/k RMSNorm before RoPE; Qwen2 doesn't.
   (b) Qwen2 has `attention_bias=true` on Q/K/V projections; qwen35
   likely hardcodes `bias=false` in `fused_qkv_hfq4g256`. See §5
   phase 1 for kernel-variant choices.
8. **Custom token IDs and chat template** — see §2.5.
9. **Smart-resize policy** — 28-divisible H/W with min/max-pixels
   clamp and beta scaling. New code in image preprocessing.
10. **Patch_embed has bias** — see §2.3.
11. **Tied embeddings differ between Qwen2-1.5B-Instruct
    (tie=True) and dots.ocr (tie=False)** — the qwen2 crate loader
    must handle both cases. See §2.1.

## 5. Phased implementation

Time estimates are revised 2-3× upward from the original draft per
review consensus. Treat this as a 1-2 week effort, not a long
weekend.

### Phase 0 — verify ground truth + capture references (4-6 hr, blocking)

Goal: capture HF reference outputs that subsequent phases compare
against. Most "verify X" sub-steps below are already done during
plan review; this phase makes the artifacts reproducible.

1. **[DONE]** Verified during plan review:
   - Merger uses LayerNorm + bias (§2.4)
   - Merger MLP has bias on both linears (§2.4)
   - Chat template is custom, EOS = 151673 (§2.5)
   - Vision position IDs use permute(0,2,1,3) (§2.6)
   - Patch embed Conv2d has bias (§2.3)
   - dots.ocr text weights stored as `model.*` (no remap needed)
   - arch_id=1 already claimed by LLaMA crate (using 7 instead)

2. **[DONE]** Read dots_ocr.py + dots_vision.py end-to-end. Findings
   folded into §2.9. No contradictions with the rev-2 plan; main
   pre-Phase-2c gotchas captured: attention scale is plain
   `1/sqrt(head_dim)` (no qk-norm, no learned scale), `image_grid_thw`
   for batch>1 packs into a single flattened sequence with image-major
   cu_seqlens (i32), bf16 cast at vision forward entry, vision-text
   integration via `masked_scatter()` with no projection layer
   (vision output already at `text_hidden_size`), dtype cast at
   integration boundary. No dropout/droppath in inference.

3. **[DONE]** dots.ocr safetensors manifest captured at
   `docs/plans/qwen_2.0_vlm_plus_dots_ocr.dots_ocr_manifest.txt` (642 lines, 338
   tensors). Param count to be confirmed during phase 2 weight load.

4. **[DONE]** Qwen2-1.5B safetensors manifest captured at
   `docs/plans/qwen_2.0_vlm_plus_dots_ocr.qwen2_1p5b_manifest.txt` (339 lines, 338
   tensors). No `lm_head.weight` confirms `tie_word_embeddings=true`;
   `q/k/v_proj.bias` entries confirm `attention_bias=true`.

5. **[DONE]** End-to-end dots.ocr reference captured on
   `benchmarks/images/dots_ocr_smoke_001.jpg` (md5
   `a434c567a2dfa0664ce75291508bad85`, sourced from
   `dots.ocr/demo/demo_image1.jpg`, a 1700×2250 medical-paper page).
   Captured across TWO complementary artifacts, each addressing a
   different downstream consumer:

   **(a) HF transformers / CPU+bf16 — prefill logits ground truth.**
   `benchmarks/references/dots_ocr_smoke_001.json` (~91 KB) via
   `scripts/capture_dots_ocr_reference.py`. transformers 5.5.1, torch
   2.12.0+cu130 (CPU runtime), `attn_implementation=eager`,
   `dtype=torch.bfloat16` (matches the model's native bf16 cast at
   vision forward entry — loading at f32 produces a Conv2d dtype
   mismatch). Captures: full input_token_ids (5095 tokens with vision
   embeds, padded image-token wrapper, layout prompt), image_grid_thw
   `[1, 160, 122]`, top-100 logits at positions 0 / 32 / 128 / 5094.
   Prefill logits are coherent (pos_0='task'/+13 nats top-1 gap,
   pos_5094='['/+11 nats — both confidence-strong, sensible
   first-completion-token prediction). The captured greedy decode
   collapses into a repeated-token attractor after ~5 tokens (correct
   JSON skeleton, then numerical drift) — this is a known CPU+bf16
   limitation for this model. The `decode_quality` field +
   `decode_quality_note` field in the artifact document this so
   downstream consumers don't mistake the completion_token_ids for a
   usable layout reference; the per-position LOGITS are the actual
   ground truth for hipfire phase-2c forward-pass validation.

   Three runtime patches were needed against transformers 5.5.1
   (recorded in the script for future-resilience):
   - dots.ocr's `prepare_inputs_for_generation` dereferences
     `cache_position[0]` without a None guard; transformers 5.5.1's
     `_prefill` calls it with cache_position=None. Patched via
     MRO bypass to `GenerationMixin.prepare_inputs_for_generation`
     plus first-step detection on the parent-populated
     cache_position.
   - dots.ocr's processor returns `mm_token_type_ids`; transformers'
     `_validate_model_kwargs` rejects it. Filtered explicitly
     before calling `generate`.
   - The validator's introspection misses dots.ocr's `pixel_values`/
     `image_grid_thw`/`attention_mask` forward params (likely picks
     up a parent's forward signature via reflection). No-op'd
     `_validate_model_kwargs` since we manually filter
     `mm_token_type_ids` ourselves.

   **(b) vLLM 0.13.0 / GPU+bf16+flash_attn — Phase 4 OCR gate
   ground truth.** `benchmarks/references/dots_ocr_smoke_001_vllm.json`
   (~32 KB) via `scripts/capture_dots_ocr_vllm.py
   http://<host>:8000`. Sends the same image + canonical
   `prompt_layout_all_en` prompt as an OpenAI-compatible
   `/v1/chat/completions` POST. Captures the real layout output: 13
   well-formed layout elements (`{"bbox": [x1, y1, x2, y2], "category":
   "...", "text": "..."}`), 6 Text / 3 Table / 2 Page-header / 2
   Caption — realistic for this medical-paper page. parse_status: ok.
   4634 completion tokens, 5097 prompt tokens (matches HF capture's
   5095 ±2, attributable to chat-template detail). 82s wall on the
   user's RTX 3060 vLLM instance.

   The two artifacts are kept side-by-side rather than consolidated:
   the HF path gives prefill logits (phase 2c); the vLLM path gives
   end-to-end layout JSON (phase 4 coherence gate). Neither path
   alone covers both needs.

6. **[DONE]** End-to-end Qwen2-1.5B-Instruct HF reference captured.
   - Prompt at `benchmarks/prompts/qwen2_smoke.txt` (83 bytes,
     md5 `4800a2ddde4312e40d692bd4d6ac193f`).
   - Capture script `scripts/capture_qwen2_reference.py`.
   - Reference artifact `benchmarks/references/qwen2_1p5b_instruct_smoke.json`
     (25 KB) records: prompt token IDs (15 tokens), greedy-decoded
     continuation (32 tokens), top-100 logits at positions 0 / 8 / 14
     (= n_prompt-1, the first-completion-token predictor), generation
     config, transformers version, torch version, snapshot path.
   - Note: actual transformers version is 5.5.1 (not 4.56.1 — the
     installed venv was newer; reference re-capture needed only if
     transformers version drift changes outputs).
   - Sanity check: top-1 at `pos_14` = token 362, which matches
     `first_16_completion_token_ids[0]` (= 362). Greedy + logit dump
     are self-consistent.

7. **[DONE]** venv setup: `.venv` already exists at repo root from
   a prior session, with transformers 5.5.1 / torch 2.6.0 (CPU) /
   safetensors 0.7.0 installed. Per
   `feedback_use_venv_for_python_installs`.

**Contingency:** if any phase-0 assumption fails verification, halt
phase 1 and amend §2 before proceeding.

### Phase 1 — Qwen2 text decoder, standalone crate (1-3 days)

Bring up `hipfire-arch-qwen2` against Qwen2-1.5B-Instruct first.
arch_id = 7. Independently useful (fills the long-empty plain-Qwen2
slot) and unblocks the dots.ocr text path.

**Bring-up:** [DONE in `c6d4e539`]
- ✅ Copied `crates/hipfire-arch-toy/` → `crates/hipfire-arch-qwen2/`.
- ✅ Added to workspace `members`.
- ✅ Created `docs/architecture-ids.md` recording slots
  0/1/5/6/7/8/0xFF (slot 8 reserved for dots.ocr in phase 3).
- ✅ `Architecture` trait impl with `arch_id() = 7`, `name() =
  "qwen2"`, `eos_filter_overrides` setting `strip_think: Some(false)`.

**Config parser:** [DONE in `8ab7ec62`]
- ✅ Real `Qwen2Config::from_hfq` parsing 13 fields with sensible
  defaults (`attention_bias` → `true`, `tie_word_embeddings` →
  `false`, `rope_theta` → `1_000_000`, `eos_token_id` accepts both
  int and array forms).
- ✅ Split into testable `config_from_metadata_json(&str)` plus
  trait-facing `config_from_hfq(&HfqFile)`.
- ✅ 4 unit tests pass (1.5B-Instruct fixture, dots.ocr text-config
  fixture, missing-required-field, defaults-only).

**Quantisation:** [DONE in `4bf9f6d4`]
- ✅ `hipfire-quantize --format hfq4` on Qwen2-1.5B-Instruct →
  ~820 MB HFQ output, 100% param coverage, q/k/v bias preserved in
  F16. Resolves §6 risk M9. No `--dry-run` flag exists in the CLI;
  ran the real thing.
- ⚠️ HFQ emits `arch_id=1` (existing Qwen2/Qwen3 default).
  `hipfire-arch-qwen2` claims `arch_id=7`. Need per-file remap or a
  `hipfire-quantize --arch-id` flag before daemon can dispatch our
  HFQ to our crate. See §6 new risk R1.

**Weight loader:** [DONE in `e034c44b`]
- ✅ Real `Qwen2Weights::load` ports the qwen35 pattern:
  embed_tokens → final norm → tied lm_head re-upload → 28 layers
  (input_layernorm + qkv with bias + o_proj + post_attention_layernorm
  + gate/up/down).
- ✅ Verified end-to-end on gfx1151: 28 layers loaded, dims match
  Qwen2-1.5B exactly (wq 1536×1536, wk 256×1536, lm_head
  151936×1536). Tied-lm_head correctly detected.
- ✅ `TODO(transformer-extraction)` markers on every cross-arch
  duplicate, on both sides (`hipfire-arch-qwen2::qwen2` and
  `hipfire-arch-qwen35::qwen35`).
- ⚠️ Loader currently handles quant_types {1 (F16), 6 (HFQ4G256),
  7 (HFQ4G128)}. Add MQ4/MQ3/etc. on demand.
- ⚠️ Tied embeddings re-upload the embedding bytes for the lm_head
  GpuTensor (`GpuTensor` is not `Clone`) — costs ~117 MB extra VRAM
  on Qwen2-1.5B at HFQ4. Resolvable via `GpuTensor::shallow_clone()`
  or `Arc<GpuTensor>` during the Transformer-extraction PR.

**Forward-pass port** from `crates/hipfire-arch-qwen35/src/qwen35.rs`:
[DONE in `afd4b059`; closed at Q8F16 16/16 top-1 in `9bd083f6`]
- ✅ **Remove** q/k-RMSNorm-pre-RoPE (Qwen3-only).
- ✅ **Remove** DeltaNet hybrid LA layers — Qwen2 is pure FA.
- ✅ **Remove** MoE expert routing — Qwen2 is dense FFN.
- ✅ **RMSNorm convention:** uses standard `weight * x * rsqrt(...)`,
  no `+= 1.0` offset (via `load_norm_weight_raw`).
- ⚠️ **Add** bias terms to Q/K/V projections — shipped as
  **option (a)** (3 separate `bias_add_f32` per layer = 84 launches
  per decode step), not the originally-recommended option (c).
  Promoting to (c) or (b) is on the deferred-perf list in §6.1
  pending the Δ ≥ 5% rule check. The earlier "option (c)" comment
  in the source was wrong (caught in rev-3 review, fixed in
  `2226bbcf`).
- ✅ **Tied-embeddings forward path:** loader produces a usable
  `WeightTensor` for `weights.output` whether tied or not; the
  forward path uses it uniformly.
- ✅ **Keep**: GQA attention, 1-D RoPE (theta=1e6), SwiGLU FFN,
  RMSNorm (eps=1e-6), KV cache layout.

**Real `Qwen2State`:** [DONE in `afd4b059`] allocates the full
per-step scratch graph (x, tmp, q, k, v, attn_out, o, gate, up,
ffn_hidden, ffn_out, logits, pos_buf) plus F32 KV cache
(`num_hidden_layers × 2 × max_seq × kv_dim`). `Qwen2State::new`
takes the default `max_seq=512`; `new_with_max_seq` for explicit
budgets. `reset()` rewinds `next_pos` to 0 (added in `2226bbcf`
for use by the daemon's `reset` event and `bench_prefill` cold
start). `free_gpu` releases all buffers + the pos i32. KV
quantisation modes are listed in §6.1 as deferred follow-on.

**Maintenance note** (deferrable to a future PR): every cross-arch
primitive duplicated from qwen35 carries a
`TODO(transformer-extraction)` marker on both sides (qwen35 source
and qwen2 destination). The future consolidation PR can
`git grep TODO(transformer-extraction)` to find every duplicate
and lift the shared primitives into
`hipfire_runtime::transformer::*` per upstream's plan. Identified
consolidation candidates so far: `load_norm_weight*`,
`load_weight_tensor*`, the tied-lm_head pattern, and (when added)
`KvCache` allocation.

**Layer-dump infrastructure (highly recommended):** [SKIPPED —
not needed]. The Q8 16/16 top-1 match in `9bd083f6` arrived on
first run without per-layer diffing, so the layer-dump hook was
never wired. Keep on the deferred list for phase 2 (vision tower)
where the qwen35-vl experience suggests it's worth more — image
preprocessing + 42-block ViT has more places to go wrong silently
than a 28-block dense decoder.

**Validation:** [DONE]
- ✅ Smoke prompt at `benchmarks/prompts/qwen2_smoke.txt` (83
  bytes, md5 `4800a2ddde4312e40d692bd4d6ac193f`).
- ✅ HF reference captured at
  `benchmarks/references/qwen2_1p5b_instruct_smoke.json`
  (transformers 5.5.1, F32, CPU; top-100 logits at pos 0/8/14;
  first-16 + 32-token greedy continuation).
- ✅ **Q8F16 (qt=3): 16/16 top-1 match** on the committed reference
  — definitive correctness lock-in.
- ✅ HFQ4G256: 9/16 with a perfect 7/7 prefix — synonym-position
  divergence consistent with 4-bit quant noise, not an
  implementation bug (Q8 sweep above confirms).
- ✅ Both `infer_qwen2` (standalone) and `hipfire run` (daemon
  path) reproduce the same continuation byte-for-byte.
- Coherence gate (`scripts/coherence-gate.sh`) intentionally not
  run on this branch — the new code is gated by a new `arch_id`
  and doesn't change existing model outputs. The gate is mandatory
  for changes that affect existing kernels/dispatch/forward; this
  is additive only. Maintainer call on whether to require it
  before merge of Kaden-Schutt/hipfire#297.

### Phase 2 — dots.ocr vision tower (16-30 hr)

- New crate `hipfire-arch-dots-ocr` depending on `hipfire-arch-qwen2`.
- **Text-side trait impl** is a thin delegation to
  `hipfire-arch-qwen2::Qwen2::load_weights` / `forward_scratch`. No
  weight-key remap required (dots.ocr stores text weights as
  `model.*`, same as plain Qwen2). The dots.ocr Weights struct
  contains a `Qwen2Weights` plus the vision-tower weights side-by-
  side.
- **Vision trait impl** owns the custom `DotsVisionTransformer`
  encoder. `type State = ()` (one-shot encoder, no KV cache).

**Image preprocessing:** [DONE in `bfe1f56d` (phase 2b)]
- ✅ `image.rs` ported from qwen35-vl as skeleton, swapped to
  dots.ocr's resize policy.
- ✅ `smart_resize` matches `dots_ocr/utils/image_utils.py` exactly:
  28-divisible, `[3136, 11_289_600]` pixel clamp, beta scaling for
  both over-max and under-min cases, AR > 200:1 guard, zero-dim
  guard.
- ✅ `clip_normalise` applies the CLIP constants
  `mean=[0.48145466, 0.4578275, 0.40821073]`,
  `std=[0.26862954, 0.26130258, 0.27577711]` per-channel, RGB.
- ✅ `extract_patches` mirrors `transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)`:
  iteration nesting `outer_y → outer_x → inner_y → inner_x →
  channel → tps → patch_y → patch_x` produces the 2×2-grouped-block-
  major patch ordering with channel-major inner element layout.
  Asserts catch `h`/`w` not multiples of `PATCH_SIZE`, and grid not
  multiple of `SPATIAL_MERGE_SIZE` — no silent truncation.
- ✅ Top-level `preprocess_image(path)` / `preprocess_dynamic_image(img)`
  wrap the full pipeline with RGBA→RGB compositing over white
  background (matches HF's `PIL.Image.convert("RGB")` on alpha-source).
- ✅ Unit tests (14 in `image::tests`, no GPU needed):
  - 5 covering smart_resize: 28-multiples, AR guard rejection,
    zero-dim rejection, downscale (8000×8000), upscale (20×20).
  - 1 covering CLIP normalisation with a hand-computed 1×1 RGB
    reference (3 channels × 1 pixel).
  - 3 covering extract_patches: 2×2-grouped order verified against
    a per-patch-tagged synthetic 28×56 input (the canonical §2.7
    silent-failure test), patch-interior layout, and the two assert
    panics on misaligned input.
  - 1 helper test for `PreprocessedImage` helpers.
  - 3 covering RGBA compositing: alpha=128 mid-blend (hand-computed),
    alpha=0 yields pure white, alpha=255 preserves source.

The patch-order silent-failure trap is gated by
`image::tests::extract_patches_uses_grid_block_order`. The test
explicitly computes both the "raster" and "grid-block" expected
sequences and asserts on the latter — any drift to raster order
fails with a diagnostic that names both candidates.

**`vision_forward(gpu, weights, &patches, image_grid_thw) -> Vec<f16>`:**
*(Sub-tasks 2c-1 through 2c-4 landed; 2c-5 assembly + per-stage
validation is the remaining work — see progress log entries above
for commit-by-commit detail.)*

- **Patch embed** — patch_embed_w stored as F16 [1536, 588] on GPU
  (the 4-D `[1536, 3, 14, 14]` Conv2d weight reshapes to 2-D linear
  at load via `load_f16_or_dequant`). Forward: `linear_f16` (GEMM +
  bias_add) → `rmsnorm_f32` with `eps=1e-5`. *(Loader: 2c-2 / 2c-4;
  forward call: 2c-5.)*
- **2-D RoPE preparation** [DONE in 2c-3 / 2c-4]:
  - ✅ CPU table builder `rope::build_rope_2d_tables(grid_h, grid_w,
    head_dim=128, sm=2, theta=10000)` emits per-patch [N_patches,
    head_dim] cos/sin in dots.ocr's exact `[hc, wc, hc, wc]` quarter-
    repeat layout. Patch enumeration matches `image::extract_patches`.
  - ✅ GPU application kernel `rope_2d_halfsplit_f32` +
    `Gpu::rope_2d_halfsplit_f32` dispatch fn — applies the
    precomputed tables to Q/K in-place via the halfsplit rotation
    convention. Decision made: new fused kernel rather than re-using
    `rope_partial_halfsplit_f32` because the latter generates
    cos/sin from a single position counter and can't accept
    precomputed tables.
- **42× block:**
  - RMSNorm (eps=1e-5)
  - QKV via single GEMM (no bias — vision QKV has `use_bias=false`).
    Use `linear_f16_no_bias` against the F16 qkv_w on GPU.
  - Apply 2-D RoPE via `gpu.rope_2d_halfsplit_f32(q, k, cos_tab, sin_tab, ...)`.
  - **`vit_attention_opt`** (NOT `vit_attention_f32`) — the basic
    variant materialises an N²-scores buffer in shared memory
    (~77 KB at N≈19520 for the smoke image; exceeds RDNA per-CU SLM
    cap). The `_opt` variant uses tiled K/V loading + 4 queries per
    block. Both are non-causal.
  - Output projection (no bias) via `linear_f16_no_bias`
  - Residual via `add_inplace_f32`
  - RMSNorm
  - SwiGLU MLP: `linear_f16_no_bias(fc13_proj)` → `silu_mul_f32(y[:H], y[H:])`
    → `linear_f16_no_bias(fc2)`. The `fc13_proj` row concatenation
    happens at load time (option (a) of phase 2c) — verified
    landed in 2c-4 via `load_f16_or_dequant_concat_rows`.
  - Residual
- **Post-norm** [DONE — weight loaded in 2c-2] — `rmsnorm_f32`
  against `post_trunk_norm` with `eps=1e-5`.
- **Merger:**
  - Reshape: contiguous 2×2 groups are already adjacent thanks to
    the position-ID permutation in `image::extract_patches` +
    `rope::build_rope_2d_tables` — `view(-1, 6144)` gives the right
    grouping.
  - `gpu.layernorm_batched(merger_ln_w, merger_ln_b, ..., eps=1e-6)`.
  - `linear_f16(merger_fc1_w, merger_fc1_b, ...)` → `gelu_tanh_f32`
    → `linear_f16(merger_fc2_w, merger_fc2_b, ...)`.

**Kernel work required:**
- ✅ Vision-shape RMSNorm — existing `gpu.rmsnorm_f32` accepts the
  vision `[N_patches, embed_dim]` shape directly (uses
  `x.shape[0]` as batch). Confirmed in 2c-4 audit.
- ✅ Vision-shape SwiGLU — existing `gpu.silu_mul_f32` is
  element-wise / shape-agnostic. Confirmed in 2c-4 audit.
- ✅ 2-D RoPE — landed in 2c-3 (`rope::build_rope_2d_tables` CPU
  builder) + 2c-4 (`rope_2d_halfsplit_f32` GPU apply kernel). The
  plan's "CPU pre-comp first, fused kernel later" pattern carried
  out exactly.
- ⚠️ `vit_attention_f32` materialises `(n + block_size) * 4` bytes
  in shared mem; for the smoke image's N=19520 that's ~77 KB,
  over RDNA per-CU SLM caps. Use `vit_attention_opt` (tiled K/V,
  ~3-5× faster anyway) in 2c-5 instead. Both are non-causal.

**Validation [PENDING — 2c-5]:**
- HF reference activations captured in 2c-1 at: `patch_embed`,
  blocks 0/21/41, `post_trunk_norm`, `merger`. Sampled refs (256 /
  64 rows) at `benchmarks/references/dots_ocr_smoke_001_activations/`;
  full tensors at `/data/cache/hipfire/dots_ocr_activations_full/`.
- 2c-5 will diff hipfire output against HF at each capture point.
  Tolerance: absolute < 1e-2 OR cosine > 0.999 (bf16 → F16 cast
  loss allows some slack at deeper layers).
- Channel-order test: load the patch_embed weight, run on a known
  test image with known expected first-block output, verify [R, G, B]
  matches (NOT [R, B, G] — that quirk is qwen35-vl-specific; verify
  it doesn't apply here).

### Phase 3 — assembly + daemon plumbing (6-10 hr)

Daemon currently has VL infrastructure only for arch_id 5/6. The
`LoadedModel` struct holds `q35_config`, `q35_weights`,
`q35_scratch` and `generate_vl()` unwraps them. arch_id=8 needs
parallel plumbing — not "add a match arm" but new fields + new
dispatch arms.

**arch_id registry:**
- Add `arch_id = 8` for `hipfire-arch-dots-ocr` to
  `docs/architecture-ids.md`.

**Daemon `LoadedModel` extension:**
- Add `dots_ocr_config`, `dots_ocr_weights`, `dots_ocr_scratch`
  (or Qwen2 equivalents — the text-side delegation pattern means
  we hold a `Qwen2Weights` plus dots.ocr vision weights). Decide:
  one combined struct or two parallel structs.

**Daemon dispatch:**
- New arm in `load_model` (currently `daemon.rs:672-677,
  :1494, :1719, :3158, :3516`) for arch_id == 8.
- New `generate_vl_dots_ocr()` (or refactor `generate_vl()` into a
  generic dispatcher branching on arch_id).

**Token-id constants:** `IMGPAD_ID = 151665`, `IMG_START_ID = 151666`,
`IMG_END_ID = 151667`. New triple alongside the existing qwen35-vl
triple.

**Architecture trait overrides (MANDATORY, not optional):**
- `prompt_frame_overrides`: emit the custom dots.ocr framing per
  §2.5. **Decision point:** does the daemon's chat-template
  renderer evaluate arbitrary Jinja2, or only ChatML?
  - If Jinja2: register the dots.ocr template, done.
  - If ChatML-only: hardcode the framing in the override
    (`<|system|>...<|endofsystem|>\n<|user|>...<|endofuser|>...`).
- `eos_filter_overrides`:
  ```rust
  stop_at: vec![b"<|endofassistant|>".to_vec()],   // primary EOS (151673)
  holdback_prefixes: vec![b"<|end".to_vec()],
  strip_think: Some(false),                         // OCR model, no <think>
  ```
  Plus add 151643 (`<|endoftext|>`) and 151673 to the runtime's
  blocked-EOS-for-streaming list.

**Vision token splicing:**
- Hook the daemon prefill loop the same way qwen35-vl does
  (`daemon.rs:4178-4186` template): on IMGPAD_ID, call
  `forward_scratch_embed` with the next merged visual embedding.
- Plumb `image_grid_thw` through the daemon's generate request
  payload (currently no path for this in non-Qwen3.5 VL).
- **IMGPAD count assertion (Gemini review).** The merger emits exactly
  `(grid_h / sm) * (grid_w / sm)` visual tokens per image. The prompt
  framer MUST insert exactly that count of `<|imgpad|>` (151665)
  tokens between `<|img|>` and `<|endofimg|>` for each image. Add a
  hard assert at the splice site:
  ```rust
  assert_eq!(
      img_mask.count_ones(),
      merged_vision_tokens.len(),
      "dots-ocr: prompt has {} <|imgpad|> slots but vision_forward \
       produced {} merged tokens — prompt framer mismatch",
      img_mask.count_ones(), merged_vision_tokens.len(),
  );
  ```
  Mismatched counts either truncate vision tokens silently or leave
  unresolved IMGPAD positions in the text context — both are
  silent-failure traps.
- **Multi-image attention leakage (Gemini review, HIGH RISK).**
  `vit_attention_opt` is a standard dense ViT attention kernel; it
  does NOT support FlashAttention's `cu_seqlens` block-diagonal
  masking that HF's `flash_attn_varlen_func` uses for multi-image
  concatenation. Two paths:
  1. Per-image loop in the daemon: call `vision_forward` once per
     image (batch=1), concatenate merged tokens AFTER the vision
     pass but BEFORE the text-side `masked_scatter`. Safer, no
     kernel changes.
  2. Add `cu_seqlens` masking to `vit_attention_opt`. More work,
     better throughput at multi-image scale.
  Phase 3 implements (1) — `vision_forward` itself already documents
  single-image-only semantics. Phase 6+ (perf) may revisit (2).

**Example binary:**
- `crates/hipfire-runtime/examples/infer_dots_ocr.rs`: takes
  `--image path.png --prompt-template layout-all-en`, emits JSON
  layout to stdout.

### Phase 4 — correctness gate (8-12 hr)

OCR-specific gate. Fluent ≠ correct.

**Coherence gate (mandatory):**
- `./scripts/coherence-gate.sh` — pre-commit hook triggers on
  kernel/forward-pass changes.

**New `scripts/coherence-gate-dots-ocr.sh`:**
- Inputs: committed 2-image reference set
  (`benchmarks/images/dots_ocr_smoke_001.png`,
  `benchmarks/images/dots_ocr_smoke_002.png`), md5s recorded.
- Run hipfire dots.ocr forward on each image with layout-all-en
  prompt. Save JSON output.
- Compare against phase-0 HF reference JSON:
  - **Parse failure** (invalid JSON) → HARD FAIL.
  - **Box matching:** Hungarian assignment by IoU between HF and
    hipfire box sets; require ≥80% of HF boxes paired with IoU ≥
    0.85.
  - **Category equivalence:** Jaccard on category strings per pair
    ≥ 0.9 (allows for `"title"` vs `"heading"` near-equivalence;
    can be tightened later).
  - **Coverage:** unpaired HF boxes count > 20% → HARD FAIL.
- Reject the commit if any HARD FAIL fires.
- Δ ≥ 5% perf investigation rule applies as usual.

**Reference image set selection:**
- Pick one English-text-heavy academic-paper-style page and one
  mixed-content page (figures, tables, text). Commit at original
  resolution so smart-resize is exercised end-to-end.

### Phase 5 — quantisation (stretch, post-merge)

- Run `hipfire-quantize --include-vision --vision-quant bf16` for
  text-quantised + vision-bf16 first cut.
- Re-run `coherence-gate-dots-ocr.sh`. Hard-fail thresholds
  unchanged.
- Only then attempt `--vision-quant hfq4`. Per
  `project_mq3_lm_head_awq_regresses_kld_2026_05_19`, sub-4-bit
  quant can break structured output even when PPL looks fine.

## 6. Risks and unknowns

- **[RESOLVED — R1] HFQ `arch_id` mismatch.** `hipfire-quantize`
  now accepts `--arch-id <u32>` (added in the R1 commit; see §0
  progress log) which overrides the auto-detected id at HFQ write
  time. Applied to both the GGUF and safetensors entry paths via the
  shared `parse_arch_id_override()` helper. Re-quantised
  `qwen2-1.5b.arch7.hfq4` at `/data/cache/hipfire/` carries
  `arch_id = 0x07` at header offset 0x08 (`xxd` verified); load
  through `hipfire-arch-qwen2::inspect_hfq --load` completes
  successfully with the correct config (`arch_id=7`, 28 layers,
  tied lm_head, HFQ4G256 embeddings).
  Other paths that were on the menu but not taken:
  - in-place `hfq-rewrite-arch-id` tool — superseded by the CLI flag
  - daemon dispatcher change routing arch_id=1 → qwen2 — still on
    the table for the eventual id=1 retirement (a follow-on once
    the new crate has shipped a forward pass); deferred per §8.
- **[RESOLVED — R2] LLaMA path silently drops Qwen2 Q/K/V bias.**
  Shipped the hard-guard path in `51e05b99`:
  `load_weights_hfq` now checks for
  `model.layers.0.self_attn.q_proj.bias` in the manifest and
  hard-refuses with an error pointing at the `--arch-id 7` flag
  and the standalone `inspect_hfq` example. Defense-in-depth for
  any legacy `arch_id=1` Qwen2 HFQ files still on operators' disks.
  The "bias-aware LLaMA loader" path stays deferred — only worth
  doing if `arch_id=1` is eventually migrated to the new crate
  (see §8 deferred item).
- **[RESOLVED — R3] Daemon wired for arch_id=7.** Standalone
  `infer_qwen2.rs` driver binary landed first (commit `00d406af`),
  then full daemon arm: load_model arm + LoadedModel fields +
  `generate_qwen2` JSON-stream emit + `arch-qwen2` cargo feature.
  `hipfire run` (production CLI) on `qwen2-1.5b.arch7.q8.hfq`
  generates coherent text at 96.3 tok/s. Bring-up scope —
  DFlash / CASK / PFlash / VL / ChatML / repeat penalty / top-p /
  `<think>` / multi-GPU are explicitly refused or skipped.
- **[RESOLVED — R4] Tied F16 lm_head corruption (latent).**
  `load_lm_head` originally took `gpu.upload_raw(&data, ...)` for
  the `quant_type == 1` (F16) tied-embedding branch while the
  `WeightTensor.gpu_dtype` was set to `F32` — kernel would read F16
  bytes as F32 → garbage. Caught at rev-2 review; fixed in
  `45913eb0` by mirroring qwen35's host-side F16→F32 expansion.
  Latent (didn't fire) because the current `qwen2-1.5b.hfq4` uses
  HFQ4G256 for the embedding, not F16. Tagged here so the test
  matrix doesn't regress.
- **[RESOLVED — R5] dots.ocr EOS via generation_config.json.**
  Shipped in `544822b4`: `hipfire-quantize` now reads
  `generation_config.json` alongside `tokenizer_config.json` (if the
  file exists in the input directory) and packs it into HFQ metadata
  under a top-level `generation_config` key. Parser side
  (`Qwen2Config::from_hfq`) walks the three-layer fallback
  `text_config.eos_token_id` / `config.eos_token_id` →
  `generation_config.eos_token_id` → default `[151645]`. Tests: 8
  passing including `eos_falls_back_to_generation_config_when_absent_from_config`
  (dots.ocr's real shape) and
  `eos_in_config_takes_precedence_over_generation_config` (guards
  against ordering ambiguity). Field docs in
  `crates/hipfire-arch-qwen2/src/qwen2.rs:46-57` describe the
  lookup order explicitly.
- **[RESOLVED — R6] Daemon arch_id=7 event-handler gaps.**
  The R3 commit (`806680b2`) added the `load_model` / `generate`
  arms for arch_id=7 but missed two other daemon event handlers:
  `bench_prefill` (panicked on arch_id=7 because the LLaMA-else
  fallthrough unwrapped `m.llama_config`); the `reset` event
  (cleared `seq_pos` and KV `compact_offset` but never touched
  `Qwen2State.next_pos`, leaking prior-turn KV into the next
  prefill). Both fixed in `2226bbcf` — `bench_prefill` got a new
  arch_id=7 arm calling `qwen2::forward_step` per token, and
  `reset` now calls a new `Qwen2State::reset()` helper that also
  fires from `bench_prefill`'s cold-start path. End-to-end
  regression run (load + bench_prefill[32] + reset + generate[16] +
  unload) confirmed both fixes plus no regression on the Q8 16/16
  match. Caught by the rev-3 pre-PR self-review; missed by both
  external reviewers (Gemini concluded "no blockers", GLM-5 said
  "ship it"). Lesson for future arch-arm work: when adding a new
  `arch_id` branch, grep `daemon.rs` for every site that switches
  on `m.arch_id` and confirm each one has the new arm.
- **[RESOLVED — M9] Quantiser support for Qwen2 layer naming.**
  Verified working in `4bf9f6d4` — Qwen2-1.5B quantises to HFQ4 at
  100% param coverage. Q/K/V bias tensors correctly preserved in
  F16. No quantiser-side changes needed.
- **2-D RoPE kernel + position-ID permute.** New algorithm + likely
  new kernel variant. Budget per phase 2.
- **Vision-shape RMSNorm/SwiGLU kernels.** Text-side kernels may
  not handle vision strides. May need new variants.
- **`vit_attention_f32` N² scaling at max image size.** ~14400
  post-merge patches → 800 MB fp32 attention matrix per block if
  materialised. Verify tiling before committing to max image size.
- **QKV bias kernel path.** Three options (a/b/c) in phase 1;
  recommend (c) for initial bring-up. Promote to (b) only under the
  Δ ≥ 5% rule.
- **Quantiser support for Qwen2 layer naming.** Dry-run check in
  phase 1.
- **Daemon dispatch restructuring.** Phase 3 is wiring-heavy, not a
  trivial branch addition.
- **Chat-template Jinja2 vs ChatML-only renderer.** Decision point
  in phase 3; affects how `prompt_frame_overrides` is implemented.
- **F16 vs HF-F32 reference debugging.** Without layer-dump
  infrastructure (recommended in phase 1), single bugs cost 3-5 hr
  to localise.
- **Maintenance divergence cost of forking qwen35.rs.** Document
  the divergence point; consider a follow-on PR for `qwen_common`
  refactor.
- **bf16 → F16 cast loss on RoPE inv_freq.** Keep inv_freq tables in
  F32; F16 underflow risk at small magnitudes.
- **Memory budget on gfx1151.** dots.ocr ~6 GB F16 weights + ~1.8 GB
  KV cache @ 128K context + vision activations at max image. Unified
  memory means host pressure visible too. Back-of-envelope budget
  before phase 2.
- **Smart-resize off-by-pixel.** Replicate the Python algorithm
  exactly; bbox accuracy depends on identical (H, W) selection.

## 6.1 Deferred follow-ons after pre-PR review fold-in (rev-3)

Captured from the Claude / Gemini / GLM-5 reviews of commits
`9477fbbb..806680b2` (see `qwen2_post_phase1_rev_claude.md` for the
synthesis). Real items, ranked, each tagged with the phase that
ought to absorb them. None blocks the rev-3 PR.

**Perf (Phase 1.5 / post-PR optimisation pass):**

- **Bias-add fusion** — code is currently option (a) from §5 (3
  separate `bias_add_f32` per QKV per layer = 84 launches per
  decode step). Promote to option (c) — single batched bias-add of
  Q/K/V per layer (~28 launches per decode step) — or option (b)
  fused into `fused_qkv_hfq4g256_bias`. Apply Δ ≥ 5% rule before
  picking which one ships. (Gemini §3.2, Claude rev-3 B1)
- **`gemv_hfq4g256_residual` fusion** — o_proj + residual and
  ffn_out + residual currently run as `weight_gemv` + `add_inplace_f32`
  (2 launches each). The LLaMA path uses the fused residual variant
  for both sites; same upgrade saves ~56 launches/decode on Qwen2
  at HFQ4G256 weights. (Claude rev-3 B2)
- **`argmax_f32` per-call malloc** — `Gpu::argmax_f32` allocates a
  4-byte result buffer on every invocation. Greedy decode pays one
  malloc + memset + memcpy per token. Move to a persistent
  scratch on `Qwen2State` (`argmax_result: DeviceBuffer`). Cross-arch
  fix — qwen35 / llama would benefit too. (Claude rev-3 D6)
- **Prefill batching** — `forward_step` is per-token, so a 2048-token
  prompt costs ~2048× single-step decode time (Q8 baseline ≈ 10 ms
  per token at 1.5B). Production serving needs a GEMM-based batched
  prefill variant (`forward_prefill_batch` analog). Required before
  Qwen2 ships at non-bring-up scale. (Gemini §3.1, GLM-5 CAVEAT-3,
  Claude rev-3 B2-adjacent)
- **KV cache quantisation** — currently F32 (~28 MB at seq=512 for
  the 1.5B). Wire HFQ4 / HFQ8 / Q8 / asym-N modes (the qwen35 path's
  kv_mode story) for memory-constrained serving. (Gemini §3.3,
  Claude rev-3 F-rec)
- **Tied-embedding VRAM aliasing** — tied `lm_head` re-uploads the
  embedding bytes (~117 MB on Qwen2-1.5B at HFQ4) because `GpuTensor`
  is not `Clone`. Resolve via `Arc<GpuTensor>` / shallow-clone in the
  Transformer-extraction PR. (Gemini §3.4, originally rev-1 B5,
  acknowledged in Phase 0 progress log.)
- **Perf-claim hygiene** — the rev-3 commit log compared an HFQ4
  first-run (4153 ms, JIT contaminated) against a Q8 warm-run
  (303 ms) without re-running HFQ4 warm. Re-measure both paths fresh
  before any tok/s ratio enters a perf doc. Single-shot tok/s
  reported with 2-decimal precision (`96.34 tok/s`) violates the
  CLAUDE.md ±10-15% noise guard. (Claude rev-3 C1, C2)

**Daemon-arm feature parity (Phase 3 — pre-GA wave):**

- **Chat-template framing on arch_id=7.** `generate_qwen2`
  short-circuits before the daemon's `prompt_frame::apply_chatml_frame`
  pipeline runs. `hipfire run` against a Qwen2 model produces
  continuation, not instruction-following. Wire `apply_chatml_frame`
  before tokenizing once the `prompt_frame_overrides` taxonomy is
  finalised for Qwen2-1.5B-Instruct. (GLM-5 CAVEAT-1)
- **Sampling beyond greedy.** `temp` / `top_p` / `repeat_penalty` /
  `repeat_window` are all underscored params on `generate_qwen2`.
  Greedy is the validation contract; non-greedy is a feature gap.
  Add a sampler call (port the LLaMA sample_top_p path or use the
  shared sampler infrastructure). (GLM-5 CAVEAT-2)
- **`pp > 1` + arch_id=7.** Currently falls through to `load_model_pp`
  which doesn't have an arch_id=7 arm and errors with "non-Qwen3.5
  architectures". UX-fix: refuse upstream with a Qwen2-specific
  message; functional fix is multi-GPU pp for Qwen2, which is a
  separate large task.

**Dots.ocr-specific fixes (Phase 2):**

- **§2.3 patch_embed weight is 4-D `[1536, 3, 14, 14]`, not 5-D.**
  rev-1 BUG-1, still deferred per rev-2 plan. (GLM-5 §4)
- **§2.5 `<|endofsystem|>` token-string handling.** Not in
  `added_tokens_decoder` — must be emitted as raw bytes that the BPE
  tokenizer fragments. rev-1 BUG-2 deferred. (GLM-5 §4)
- **[RESOLVED] R5 dots.ocr EOS in `generation_config.json`.** Landed
  in `544822b4` — quantiser packs `generation_config.json` into HFQ
  metadata; Qwen2 parser walks fallback chain
  `text_config.eos_token_id` / `config.eos_token_id` →
  `generation_config.eos_token_id` → default `[151645]`. See §6 R5
  for the full resolution narrative.

**Cleanup / nice-to-have (any future PR):**

- `Qwen2State` could expose a `pub fn argmax(&mut self, gpu) -> u32`
  to internalise the logits→token step (currently the daemon does
  `gpu.argmax_f32(&state.logits, cfg.vocab_size)` directly).
- `infer_qwen2.rs` could print the decoded English at the end (not
  just the token IDs).
- `m.conversation_tokens.push(tok)` is filled by `generate_qwen2`
  but never read on the arch_id=7 path (it's a repeat-penalty input
  for qwen35/llama). Either remove or leave for the future sampler
  wiring.
- `chat_template = resolve_chat_template(...)` is loaded on the
  arch_id=7 path but never consulted by `generate_qwen2`. Same
  status — load now, consume when chat-template framing lands.
- `parse_arch_id_override` could move from a `unwrap_or_else` with
  `!`-return to an `if let Some(..) else` pattern. Pure style.
- `scripts/capture_qwen2_reference.py` hardcodes the HF snapshot
  path. Acceptable for a phase-0 one-time capture but won't reproduce
  on another machine without editing.

## 7. File layout (target)

```
crates/
  hipfire-arch-qwen2/                 # phase 1
    Cargo.toml
    README.md
    src/
      lib.rs            # crate root
      arch.rs           # impl Architecture for Qwen2 (arch_id=7)
      qwen2.rs          # Qwen2Config, Qwen2Weights, Qwen2State
                        # forward + forward_scratch free fns
  hipfire-arch-dots-ocr/              # phase 2-3
    Cargo.toml
    README.md
    src/
      lib.rs
      arch.rs           # impl Architecture for DotsOcr (arch_id=8)
      dots_ocr.rs       # DotsOcrConfig (text-cfg + vision-cfg),
                        # DotsOcrWeights (qwen2-weights + vision-weights),
                        # vision_forward + helpers
      image.rs          # smart-resize + patch extraction
    tests/
      smart_resize.rs
      channel_order.rs  # if needed
crates/hipfire-runtime/
  examples/
    infer_qwen2.rs                    # phase 1 text-only smoke binary
    infer_dots_ocr.rs                 # phase 3 end-to-end image+text
benchmarks/
  prompts/
    qwen2_smoke.txt                   # phase 0/1 committed prompt
  images/
    dots_ocr_smoke_001.png            # phase 0/4 reference image 1
    dots_ocr_smoke_002.png            # phase 0/4 reference image 2
docs/
  architecture-ids.md                 # phase 1: record ids 7, 8
docs/plans/
  qwen_2.0_vlm_plus_dots_ocr.md                     # this file
  qwen_2.0_vlm_plus_dots_ocr.dots_ocr_manifest.txt  # phase 0
  qwen_2.0_vlm_plus_dots_ocr.qwen2_1p5b_manifest.txt # phase 0
scripts/
  coherence-gate-dots-ocr.sh          # phase 4
```

## 8. Out of scope (for now)

- True Qwen2.5-VL (m-rope, window/full attention split, Qwen2.5
  text backbone). Defer until dots.ocr is green.
- Video / temporal patching. dots.ocr has `temporal_patch_size=1`;
  the code paths shouldn't assume it but no t-axis support needed.
- Vulkan / cross-vendor backend. Out of scope project-wide per
  CLAUDE.md rule 7.
- Training. Inference only.
- Migrating arch_id=1 from LLaMA to the new Qwen2 crate — keep
  separate slots (7, 8) for the initial bring-up; consolidation is
  a follow-on PR.
- `qwen_common` shared-primitives extraction — document the
  divergence in phase 1, refactor later.
