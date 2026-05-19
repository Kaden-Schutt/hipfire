# dots.ocr (Qwen2-VL family) + Qwen2 text decoder implementation plan

Status: rev 4 (in progress on `feat/dots-ocr-qwen2`), 2026-05-19
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
| _pending_ | Phase 1 forward pass: real `Qwen2State` (KV cache + per-step scratch) + `forward_step` / `forward_step_greedy` (28 layers: RMSNorm → fused QKV + bias adds → RoPE → KV cache → attention → o_proj → residual → FFN norm → SwiGLU → residual; final norm + lm_head). End-to-end run: 9/16 top-1 matches vs HF F32 reference, with **7/7 on the prefix and a perfect match on pos 0** (the strongest diagnostic position). Hipfire's HFQ4G256 output decodes to `" A transformer's attention mechanism is a key component of natural language processing (NLP"` vs HF F32's `" ... is a crucial component of its architecture, ..."` — divergence at synonym positions ("key" vs "crucial"), which is the expected behavior of 4-bit weight quant against F32 reference. Implementation correctness signal is strong; quant-precision-induced divergence is not a phase-1 blocker. |

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

2. **[PENDING]** Read dots_ocr.py end-to-end for subtleties not
   surfaced in review (e.g. how `image_grid_thw` is constructed for
   batch sizes > 1, attention scale factor, weight-init quirks).
   Deferrable until phase 2 (vision tower) starts.

3. **[DONE]** dots.ocr safetensors manifest captured at
   `docs/plans/qwen_2.0_vlm_plus_dots_ocr.dots_ocr_manifest.txt` (642 lines, 338
   tensors). Param count to be confirmed during phase 2 weight load.

4. **[DONE]** Qwen2-1.5B safetensors manifest captured at
   `docs/plans/qwen_2.0_vlm_plus_dots_ocr.qwen2_1p5b_manifest.txt` (339 lines, 338
   tensors). No `lm_head.weight` confirms `tie_word_embeddings=true`;
   `q/k/v_proj.bias` entries confirm `attention_bias=true`.

5. **[PENDING]** End-to-end run dots.ocr under HF transformers on a
   committed page image. Commit image bytes to
   `benchmarks/images/dots_ocr_smoke_001.png`, record md5. Use
   `transformers==4.56.1`, `trust_remote_code=True`. Capture token
   IDs (first 200 positions), logits at 0/32/128, parsed JSON
   output. Required for phase 4 OCR gate.

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
[PENDING — largest remaining chunk in phase 1]
- **Remove** q/k-RMSNorm-pre-RoPE (Qwen3-only).
- **Remove** DeltaNet hybrid LA layers — Qwen2 is pure FA.
- **Remove** MoE expert routing — Qwen2 is dense FFN.
- **RMSNorm convention:** Qwen2 uses standard `weight * x *
  rsqrt(...)`, NOT Qwen3.5's `(1 + weight) * ...`. The loader
  already loads norm weights without the `+= 1.0` offset (see
  `load_norm_weight_raw` in qwen2.rs); the forward path must apply
  norm without the offset to match.
- **Add** bias terms to Q/K/V projections. `fused_qkv_hfq4g256`
  doesn't currently accept bias. Three options, recommend (c) for
  initial bring-up:
  - (a) Separate `bias_add_f32` after each QKV: 3 extra launches per
    layer × 28 = 84 extra launches per decode step.
  - (b) New fused `fused_qkv_hfq4g256_bias` kernel: zero runtime cost
    but more upfront work.
  - (c) Batched bias-add of Q/K/V in one launch after fused QKV: ~28
    extra launches per decode step, ~half the cost of (a).
  Promote to (b) only if (c) measurably regresses tok/s under the
  Δ ≥ 5% rule.
- **Tied-embeddings forward path:** loader already detects
  `tied_lm_head` and produces a usable `WeightTensor` for the
  lm_head whether tied or not. Forward path can use
  `weights.output` uniformly without branching on the tied flag.
- **Keep**: GQA attention, 1-D RoPE (theta=1e6), SwiGLU FFN,
  RMSNorm (eps=1e-6), KV cache layout.

**Real `Qwen2State`:** [PENDING] currently a stub `token_count`
counter. Real impl allocates KV cache buffers
(`num_key_value_heads × head_dim × max_seq_len × num_hidden_layers`)
plus attention scratch. Mirror `ForwardScratch::new` in
`hipfire-runtime::llama`. `KvCache` is on the upstream consolidation
list — mark with `TODO(transformer-extraction)`.

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

**Layer-dump infrastructure (highly recommended):** [PENDING]
before debugging logit mismatches, add a
`HIPFIRE_DUMP_ACTIVATIONS=path` env-gated hook in the qwen2
forward path that writes per-layer activations as `.npy` for
layer-by-layer diffing against HF reference. Without this, the
first logit mismatch takes 3-5 hr to localise. With it, ~30 min.

**Validation:** [PENDING — blocked on phase 0 items 5/6/7]
- Smoke test: feed prompt from `benchmarks/prompts/qwen2_smoke.txt`
  with temperature=0 + greedy, decode 32 tokens.
- **Pass criterion: top-1 token match against the phase-0 HF
  reference for the first 16 positions.** Logit absolute diff is
  diagnostic-only; max(abs_diff) < 5e-2 OR cosine similarity > 0.999
  at positions 0/8/16 is the diagnostic threshold (F16 accumulation
  drift over 28 layers can reach ~2-5e-2 even when correct, so the
  original 1e-2 criterion would produce false negatives).
- Run on both fp16 and HFQ4 paths.
- Run `./scripts/coherence-gate.sh` — pre-commit hook will trigger
  it anyway.

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

**Image preprocessing:**
- Port `image.rs` from qwen35-vl as a starting skeleton.
- Swap resize to the dots.ocr 28-divisible + beta-scaling algorithm
  exactly (§2.7). Include AR > 200:1 guard.
- CLIP-style normalisation constants (§2.7).
- **Apply the patch-extraction reshape+transpose from §2.7 exactly**
  (`reshape(grid_t, tps, c, gh/sm, sm, ps, gw/sm, sm, ps)` followed by
  `transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)`). This puts patches in 2×2
  grid-block order so the merger groups correctly. Silent-failure trap
  if skipped — see §2.7.
- Unit tests:
  - smart-resize clamps within bounds and lands on 28-multiples for a
    100×3000 input; AR guard rejects 1×500.
  - patch-order test: synthetic 56×28 RGB input (1×4 patch grid with
    `merge_size=2`) produces flatten_patches byte-identical to the HF
    `Qwen2VLImageProcessor` reference for the same input. Catches the
    transpose bug independently of any GPU code.

**`vision_forward(gpu, weights, &patches, image_grid_thw) -> Vec<f16>`:**

- **Patch embed.** Reshape the 5-D weight `[1536,3,1,14,14]` to
  `[1536, 588]`. Linear projection via GEMM_F16; apply patch-embed
  bias via `bias_add_f32`; then patch-embed RMSNorm (eps=1e-5).
- **2-D RoPE preparation.** Generate hpos/wpos via the
  reshape-permute-flatten in §2.6. Compute cos/sin tables for each
  axis. Choose between a new `rope_2d_f32` kernel and a CPU-side
  pre-computation + the existing `rope_partial_halfsplit_f32`
  applied to head_dim halves. Start with CPU pre-comp + existing
  kernel; promote to a fused kernel only under the Δ ≥ 5% rule.
- **42× block:**
  - RMSNorm (eps=1e-5)
  - QKV via single GEMM (no bias — vision QKV has `use_bias: false`)
  - Apply 2-D RoPE to Q and K
  - `vit_attention_f32` with `causal=false` (verify the kernel
    supports this mode; qwen35-vl already uses it non-causally so
    likely fine)
  - Output projection (no bias)
  - Residual
  - RMSNorm
  - SwiGLU MLP. On disk the FFN has three separate linears
    (`mlp.fc1.weight [4224, 1536]`, `mlp.fc3.weight [4224, 1536]`,
    `mlp.fc2.weight [1536, 4224]`); all have no bias since
    `use_bias=false` covers FFN per §2.2. Two implementation choices:
    (a) **load-time fuse** fc1+fc3 into a single `fc13_proj [8448, 1536]`
    (vllm pattern via `stacked_params_mapping`), then SwiGLU = `silu(y[:H]) * y[H:]`
    where `y = fc13(x)`, then `fc2`; or
    (b) keep fc1 and fc3 separate and run two GEMMs per block in the
    forward pass.
    Prefer (a) for fewer launches; pick (b) only if quantisation per
    half ends up different. Note: the merge is a tensor concat at load
    time, not a runtime fusion choice.
  - Residual
- **Post-norm** (RMSNorm) since `post_norm=true`.
- **Merger:**
  - Reshape: contiguous 2×2 groups are already adjacent thanks to
    the position-ID permutation — `view(-1, 6144)` gives the right
    grouping.
  - LayerNorm with bias (eps=1e-6) — `gpu.layernorm_batched`.
  - linear(6144→6144) + bias → GELU → linear(6144→1536) + bias.

**Kernel work required:**
- Vision-shape RMSNorm (verify text kernel handles `[N, embed_dim]`
  strides; if not, thin variant).
- Vision-shape SwiGLU (same question).
- 2-D RoPE (CPU pre-comp variant first, kernel later if needed).
- Confirm `vit_attention_f32` (a) supports non-causal and (b)
  handles up to ~14400 post-merge patches without materialising the
  full N² attention matrix at fp32 (memory budget at fp32 is
  N²×4 ≈ 800 MB for N=14400).

**Validation:**
- Capture HF reference activations at: first block output, final
  block pre-merger, post-merger. Save as `.npy` per phase 0.
- Diff hipfire output against HF at each capture point. Tolerance:
  absolute < 1e-2 OR cosine > 0.999 (bf16 → F16 cast loss allows
  some slack at deeper layers).
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
- **[NEW — R2] LLaMA path silently drops Qwen2 Q/K/V bias.**
  `hipfire_runtime::llama::LayerWeights`
  (`crates/hipfire-runtime/src/llama.rs:525-537`) has no bias fields,
  and `load_weights_hfq` (`hfq.rs:646-731`) never reads
  `q_proj.bias` / `k_proj.bias` / `v_proj.bias`. The quantiser tags
  every Qwen2 HFQ as `arch_id=1`, which the daemon today routes
  through the LLaMA crate — so every existing Qwen2 HFQ on disk runs
  without bias and produces wrong outputs without any warning.
  Resolution paths (pick one when R1 lands):
  - **Hard guard in `load_weights_hfq`** that refuses to load when
    `arch_id == 1` AND `q_proj.bias` is present in the manifest,
    with a message pointing at `hipfire-arch-qwen2`. ~5 lines.
  - **Bias-aware LLaMA loader** that reads the biases when present
    and threads them through forward. Larger; only worth it if we
    decide to migrate `arch_id=1` to the new crate via R1 path 3.
- **[NEW — R3] New crate is daemon-unwired.** `hipfire-arch-qwen2` is
  a workspace member but nothing depends on it
  (`grep -l hipfire-arch-qwen2 crates/*/Cargo.toml` lists only the
  crate itself). The only entry point is `inspect_hfq`. Phase 1's
  acceptance criterion (top-1 token match vs HF) cannot fire from
  here. Mitigation: add a `crates/hipfire-arch-qwen2/examples/
  infer_qwen2.rs` driver binary in phase 1 that loads + forwards +
  greedy-samples in-process (no daemon), bypassing R1 entirely for
  bring-up correctness work. Defer daemon wiring to phase 3.
- **[NEW — R4] Tied F16 lm_head corruption (latent).**
  `load_lm_head` in `qwen2.rs` originally took `gpu.upload_raw(&data,
  ...)` for the `quant_type == 1` (F16) tied-embedding branch while
  the `WeightTensor.gpu_dtype` was set to `F32` (because the embedding
  upstream is promoted to F32 — `EmbeddingFormat` has no `F16`
  variant). The kernel would have read F16 bytes as F32 → garbage.
  Caught at rev-2 review; fixed in the rev-2 patch by mirroring
  qwen35's host-side F16→F32 expansion. Latent (didn't fire) because
  the current `qwen2-1.5b.hfq4` uses HFQ4G256 for the embedding, not
  F16. Tagged here so the test matrix doesn't regress.
- **[NEW — R5] dots.ocr `eos_token_id` is in `generation_config.json`,
  not `config.json`.** The quantiser doesn't load
  `generation_config.json`; the config parser then defaults
  `eos_token_id` to 151645 (ChatML `<|im_end|>`), which is wrong for
  dots.ocr (correct primary is 151673 `<|endofassistant|>`). Two
  resolution paths (pick one before phase 3):
  - Extend `hipfire-quantize` to also pack `generation_config.json`
    into metadata when present.
  - Special-case dots.ocr's EOS in `eos_filter_overrides` (already
    planned per §2.5) and accept that the `cfg.eos_token_id` scalar
    is for sampler-level termination only, not for streaming EOS.
  Plus a parser improvement: keep the *full* `eos_token_id` array
  rather than collapsing to the first scalar. The daemon's stop-set
  wants a multi-element set, and the current `arr.first()` semantics
  silently pick `151643` over `151673` on dots.ocr.
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
