# dots.ocr (Qwen2-VL family) — PRD

**Status:** Phase 2 merged to master (2026-05-25). Phase 3 (daemon
serving) **substantially landed** on `feat/dots-ocr-phase-3-daemon`:
the daemon serves dots.ocr over its JSONL protocol (arch_id=8 load arm,
`generate_vl_dots_ocr`, file-path + base64 image input), validated
13/13 exact-match vs the vLLM reference, with batched prefill (per-token
46.8s → 0.35s). Remaining phase-3 items: multi-image, non-greedy
sampling. This PRD is the durable design and decision record; for
commit-by-commit progress see [`dots-ocr-devlog.md`](dots-ocr-devlog.md).

## Goal

Two-crate end-to-end implementation of [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)
in hipfire:

1. **`hipfire-arch-qwen2`** — plain Qwen2 text decoder
   (`arch_id = 7`). Validated against `Qwen2-1.5B-Instruct` at HFQ4 +
   Q8 quants on gfx1151. Text body shape is identical to dots.ocr's
   text backbone, so getting Qwen2-1.5B correct solves the dots.ocr
   text path.

2. **`hipfire-arch-dots-ocr`** — dots.ocr's 42-block
   `DotsVisionTransformer` + spatial-merger + image preprocessing
   (`arch_id = 8`). Text-side trait impl delegates to
   `hipfire-arch-qwen2` (dots.ocr stores text weights under `model.*`,
   identical key layout).

Minimum success: load the dots.ocr safetensors via the quantiser,
process one page image, emit the layout JSON that the upstream model
produces. **Achieved 2026-05-21** — byte-exact vs vLLM reference on
the smoke image (F1 1.000, 13/13 text exact-match, max 1-pixel bbox
L1 deviation).

True Qwen2.5-VL coverage (m-rope + window/full attention split +
Qwen2.5 text backbone) is a follow-on, not in scope here.

## 1. Architecture (verified)

All dimensions in this section are pulled from the model's
`config.json` and the safetensors manifests at
[`dots-ocr.dots_ocr_manifest.txt`](dots-ocr.dots_ocr_manifest.txt) and
[`dots-ocr.qwen2_1p5b_manifest.txt`](dots-ocr.qwen2_1p5b_manifest.txt).

### 1.1. Text backbone (from dots.ocr `config.json`)

Plain Qwen2 1.5B-class decoder:

| field | value |
|---|---|
| `hidden_size` | 1536 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 12 |
| `num_key_value_heads` | 2 (GQA) |
| `head_dim` | 128 |
| `intermediate_size` | 8960 |
| `vocab_size` | 151_936 |
| `max_position_embeddings` | 131_072 |
| `rope_theta` | 1_000_000 |
| `rms_norm_eps` | 1e-6 |
| `attention_bias` | **true** (modeling default; Qwen2 distinct from Qwen3) |
| `tie_word_embeddings` | **false** — separate `lm_head.weight [151936, 1536]` on disk |

Q/K/V projections carry a bias; o_proj and the FFN linears do not.

### 1.2. Vision tower (from `vision_config`)

Custom `DotsVisionTransformer` (NOT a SigLIP-2 like Qwen2.5-VL's
vision tower):

| field | value |
|---|---|
| `embed_dim` | 1536 |
| `num_hidden_layers` | 42 |
| `num_attention_heads` | 12 |
| `head_dim` | 128 |
| `intermediate_size` | 4224 |
| `out_hidden_size` | 1536 (must equal text-decoder hidden_size for splicing) |
| `patch_size` | 14 |
| `spatial_merge_size` | 2 |
| `temporal_patch_size` | 1 (image, not video) |
| `use_bias` | **false** (attention QKV/proj + FFN linears) |
| `post_norm` | **true** (RMSNorm after all 42 blocks) |
| `rms_norm_eps` | 1e-5 |

Pre-norm RMSNorm → attention → residual → pre-norm RMSNorm → SwiGLU
FFN → residual, identical across all 42 blocks. No dropout / droppath
in inference.

### 1.3. Patch embedding

Weight is 4-D on disk: `[1536, 3, 14, 14]` (Conv2d) — reshaped at
load time to 2-D linear `[1536, 588]`. **Has bias.** Followed by
`patch_embed.post_layernorm` (RMSNorm, eps=1e-5).

### 1.4. PatchMerger

Spatial merger that combines 2×2 patch groups into one visual token
before feeding the text decoder:

```
LayerNorm(eps=1e-6, with bias)   ← NOT RMSNorm
free 2×2 reshape                  ← thanks to patch-order permutation in §1.7
Linear(6144 → 6144, bias=true)
GELU(tanh approximation)
Linear(6144 → 1536, bias=true)
```

`6144 = embed_dim × (spatial_merge_size ** 2) = 1536 × 4`. The output
dim equals the text decoder's `hidden_size`, so the merger's output
splices directly into the text-decoder embedding stream — no
projection layer between them. (Differs from many other VLMs where a
separate `mm_projector` linear lives here.)

### 1.5. Chat template + EOS

dots.ocr's chat template is custom, **NOT ChatML**. Three image-pad
framing tokens (verified against `tokenizer_config.json`):

| token | id | role |
|---|---|---|
| `<\|imgpad\|>` | 151_665 | image-pad slot, marks where merged visual tokens splice in |
| `<\|img\|>` | 151_666 | image-start framing |
| `<\|endofimg\|>` | 151_667 | image-end framing |

EOS set:

| token | id | role |
|---|---|---|
| `<\|endoftext\|>` | 151_643 | secondary EOS |
| `<\|endofassistant\|>` | 151_673 | primary EOS |

Both must be in the daemon's stop-set; `<\|endofassistant\|>` is the
canonical end-of-turn. **Neither lives in dots.ocr's `config.json`**
— the array `[151643, 151673]` lives in `generation_config.json`,
which the quantiser packs into HFQ metadata as of `544822b4`. Parser
walks `text_config.eos_token_id` → `config.eos_token_id` →
`generation_config.eos_token_id` → default `[151645]`.

### 1.6. 2-D RoPE

The vision tower uses a 2-D RoPE (height position × width position)
with quarter-repeat layout `[hc, wc, hc, wc]` over `head_dim`:

- `head_dim = 128`, split into four quarters of 32 each.
- Quarter 0 + quarter 2 carry the height-axis cos/sin.
- Quarter 1 + quarter 3 carry the width-axis cos/sin.
- `theta = 10_000`, `inv_freq[k] = theta ** (-2k / (head_dim/2))` for
  `k ∈ [0, head_dim/4)`.

Patch position IDs are generated via a reshape-permute-flatten that
puts neighbouring 2×2 patch groups adjacent — required so the
merger's `view(-1, 6144)` groups correctly. CPU pre-compute builds
the tables once per image; GPU apply kernel
(`rope_2d_halfsplit_qkv_interleaved_f32`) rotates Q/K in-place inside
the fused `[N, 3h]` QKV buffer.

### 1.7. Image preprocessing

Ported from `dots_ocr/utils/image_utils.py`:

- **Smart-resize.** Snap both H and W to multiples of
  `IMAGE_FACTOR = 28 = patch_size × spatial_merge_size`. Clamp total
  pixels to `[3136, 11_289_600]` via beta scaling on either bound.
  Reject aspect ratios `> 200:1`.
- **CLIP normalisation.** Mean `[0.48145466, 0.4578275, 0.40821073]`
  / std `[0.26862954, 0.26130258, 0.27577711]` per RGB channel.
- **RGBA → RGB.** Composite over white background (matches HF's
  `PIL.Image.convert("RGB")` on alpha sources).
- **Resize filter.** PIL `BICUBIC` upstream; closest in the Rust
  `image` crate is `CatmullRom`. Triangle (bilinear) was tried and
  drifted patch_embed cosine vs HF dumps by ~0.001 — fixed in
  `21ed91e1` (see devlog).
- **Patch extraction order.** `reshape(grid_t, tps, c, gh/sm, sm, ps,
  gw/sm, sm, ps)` followed by `transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)`.
  This puts patches in 2×2-grouped-block-major order with
  channel-major inner element layout. **Skipping this transpose is a
  silent failure** — the model still runs, but produces bounding-box
  coordinates offset by sub-patch shifts. Gated by
  `image::tests::extract_patches_uses_grid_block_order` against a
  per-pixel-tagged synthetic input.

### 1.8. Output format

Pure text generation; layout JSON, Markdown, or SVG depending on the
prompt template (`dots_ocr/utils/prompts.py`). No separate detection
head — tokens encode bboxes, categories, and content directly.
Canonical prompt is `prompt_layout_all_en` for layout extraction.

### 1.9. Subtleties from end-to-end HF source read

End-to-end read of `modeling_dots_ocr.py` + `modeling_dots_vision.py`
during phase-0 surfaced these non-obvious points (none contradict the
architecture above):

- **Attention scale = `1/sqrt(head_dim)`** across every attention
  impl (eager / eager_v2 / flash_attention_2 / sdpa / ascend_fa). No
  learned scale, no qk-norm.
- **Block uniformity** — all 42 vision blocks are structurally
  identical (no depth-conditional branches). First block has
  residual; last block uses the same norm convention.
- **`image_grid_thw` batch handling** (multi-image). For
  `batch_size > 1` the parser concatenates image patches into a
  single flattened sequence with `cu_seqlens` (image-major: cumsum of
  `repeat_interleave(grid_thw[:,1] * grid_thw[:,2], grid_thw[:,0])`).
  Multi-image batching is a 2-D packed sequence, not a 4-D batched
  tensor. `cu_seqlens` must be `i32` for FA correctness.
- **bf16 cast at vision forward entry.** Vision `hidden_states` are
  unconditionally cast to bf16 at the top of
  `VisionTransformerPretrainedModel.forward` when `bf16=True` (the
  default). This is the source of HF dumps drifting from F32
  reference — see §2 on the bf16-oracle lesson.
- **Dropout / DropPath**: none. `dropout_p=0.0` hardcoded; SwiGLU is
  plain `F.silu()` with no dropout wrapper. Inference and train
  modes produce identical activations.
- **Vision-token splicing uses `masked_scatter()`**. The daemon
  prefill loop overwrites text embedding rows at every
  `input_id == <\|imgpad\|>` position with the corresponding merged
  visual token. The merger output is `[N_patches/4, text_hidden_size]`
  — splices directly, NO projection layer between merger and text.
- **Vision-text dtype on the integration boundary.** Vision
  embeddings cast to `inputs_embeds.dtype` before `masked_scatter`. If
  vision is bf16 and text is f16/f32, the cast quantises. hipfire
  side runs everything F32; cast is a no-op.

## 2. Operational rules

### 2.1. The bf16-oracle lesson (load-bearing)

**For any model trained at bf16 (dots.ocr, future Qwen2-VL siblings,
likely most modern VLMs), per-stage activation cosine vs HF tensor
dumps is NOT a valid correctness oracle.**

HF's bf16 forward path truncates at every linear / layernorm output,
accumulating ~1% per-layer drift from the F32 reference algorithm. A
correctness-first F32-everywhere pipeline (hipfire's) produces *more*
numerically accurate activations than HF — but the two diverge by
~1% per layer in opposite directions, so the per-stage cosine LOOKS
like a bug when it isn't.

Concrete numbers from the smoke image at vision block 1:

| comparison | mean cos | observation |
|---|---|---|
| numpy F32 ref using OUR qkv vs OUR GPU pre-proj attn | **1.00000** | our attention kernel is bit-equivalent to F32 numpy |
| numpy(our pre @ proj_w.T) vs OUR GPU attn_out | **1.00000** | our proj GEMM is bit-equivalent to F32 numpy |
| numpy(HF qkv @ proj_w.T) vs HF attn_out (HF's bf16) | 0.99013 | HF's bf16 deviates from F32 reference by ~1% mean / 27° worst row |
| numpy(HF qkv @ proj_w.T) vs OUR attn_out | **0.99942** | we are 6× closer to F32 reference than HF |

### 2.2. The correctness gate

End-to-end task output, not per-stage cosine. For dots-ocr:

1. Run `cargo run --release --example ocr_e2e -p hipfire-arch-dots-ocr`
   on a committed reference image with the canonical layout-all-en
   prompt.
2. Score the output against the vLLM reference at
   `benchmarks/references/dots_ocr_smoke_001_vllm.json` (and future
   sibling references) via `scripts/grade_dots_ocr_e2e.py`.
3. PASS = `region_F1 > 0.9 AND mean_text_distance < 0.10` at IoU
   threshold 0.5.

Smoke image current PASS: F1 1.000, 13/13 text exact-match, max
1-pixel bbox L1 deviation.

### 2.3. Per-stage HF-diff as diagnostic only

The per-stage dump scaffolding
(`HIPFIRE_DOTS_OCR_DUMP_DIR=...` +
`scripts/diff_dots_ocr_stages.py`) is retained as a diagnostic — if
something IS broken, it usually shows up in early-block cosine too.
But cos < 0.99 at block N is NOT itself a failure unless end-to-end
output is also bad. When in doubt about whether a divergence is "us
wrong" or "HF bf16 drift", run
`scripts/numpy_attention_replay.py` on captured QKV: if our GPU
output matches numpy F32 and HF doesn't, the divergence is HF's bf16
drift not our bug.

## 3. Reusable hipfire infrastructure

### 3.1. The `Architecture` trait (bring-up contract)

`crates/hipfire-runtime/src/arch.rs` defines the spec. Required
methods: `arch_id() -> u32`, `name() -> &'static str`,
`config_from_hfq`, `load_weights`, `new_state`. Forward is **not** on
the trait (kept static-dispatch in the hot path); each arch crate
exposes its own typed forward functions.

Optional overrides for VLMs:
- `sampler_overrides` — per-arch greedy / top-p defaults
- `prompt_frame_overrides` — chat-template handling
- `eos_filter_overrides` — multi-element stop-set, holdback prefixes
- `loop_guard_overrides` — n-gram repetition guard tuning

### 3.2. Production reference: `hipfire-arch-qwen35-vl`

The closest production-shaped sibling. Layout:

```
hipfire-arch-qwen35-vl/
  src/
    arch.rs                # Architecture trait impl (vision-side bring-up only)
    qwen35_vl.rs           # VisionConfig, VisionWeights, vision_forward
    image.rs               # SigLIP-2-style preprocessing
```

dots-ocr follows the same shape (`hipfire-arch-dots-ocr/`). One key
difference: qwen35-vl's text side is the hybrid DeltaNet
`hipfire-arch-qwen35`; dots-ocr's text side is plain
`hipfire-arch-qwen2`.

## 4. Architecture-trait identity

| arch_id | family | crate | notes |
|---|---|---|---|
| 7 | Qwen2 dense (standalone) | `hipfire-arch-qwen2` | Phase 1 closed |
| 8 | Qwen2-VL family (dots.ocr) | `hipfire-arch-dots-ocr` | Phase 2 closed |

dots.ocr stores text weights under `model.*` — identical key layout
to plain Qwen2 — so the text-side trait impl is a thin delegation.
The Weights struct contains a `Qwen2Weights` plus the vision-tower
weights side-by-side. `type State = ()` for the vision impl (one-shot
encoder, no KV cache).

## 5. Phased roadmap

Phases 0-2 are closed (see devlog). Phases 3-5 are forward-looking.

### Phase 3 — daemon plumbing (6-10 hr)

> **Status (2026-05-25): substantially landed** on
> `feat/dots-ocr-phase-3-daemon` — see devlog `d9e00e4e`, `3d2412b5`,
> `a3389fc2`, `995d7449`. Workstream items below:
> 1. Token-id constants + arch-trait overrides — **done** (were already
>    present in `arch.rs`; `IMGPAD/IMG_START/IMG_END/USER/ENDOFUSER/
>    ASSISTANT/ENDOFASSISTANT/ENDOFTEXT` constants exported from
>    `dots_ocr.rs`).
> 2. Chat-template framing — **done & decided.** The image-OCR turn is
>    NOT wrapped in `<|user|>`/`<|endofuser|>` (the text-only template
>    branch is; the image branch isn't). `dots_ocr::build_prompt_ids`
>    hand-rolls the framing `220 <|img|> N×<|imgpad|> <|endofimg|>
>    <prompt> <|assistant|>` and is verified **byte-exact** against the
>    HF `apply_chat_template` capture. (The Jinja renderer exists, but
>    image-token expansion must be hand-rolled regardless — pure Jinja
>    emits one `<|imgpad|>`, not N; same pattern as qwen35-vl.)
> 3. `LoadedModel` fields + dispatch arm — **done.** Added
>    `dots_ocr_config` + `dots_ocr_weights`; text decode state reuses
>    the existing `qwen2_state`. `load_model` arm for `arch_id == 8`.
> 4. Splice + IMGPAD assertion — **done.** `generate_vl_dots_ocr` with
>    the merger-count-vs-IMGPAD-slot hard guard.
> 5. Multi-image per-image loop — **deferred** (single-image ships).
>
> Also landed beyond the original list: **base64 image input**
> (`image::preprocess_image_bytes`), correct `loaded`-event metadata,
> and **batched prefill** (`qwen2::forward_prefill_batch_embeds` +
> `kv_cache_write_f32_batched.hip`) cutting prefill 46.8s → 0.35s.
> Validated 13/13 exact-match vs vLLM over the daemon path (both
> file-path and base64). **Still open:** multi-image, non-greedy
> sampling. Decode (~55 tok/s) is now the dominant request cost.

Promote the splice pattern from
`crates/hipfire-arch-dots-ocr/examples/ocr_e2e.rs` (commit
`1f94da31`) into the daemon's serving path. The example demonstrates:
load HFQ → `vision_forward` → download merger output → prefill with
`forward_step_with_embed` at `<\|imgpad\|>` slots,
`forward_step(token)` otherwise → greedy decode. Phase 3 wraps that
in the daemon's request/response shape.

**Suggested workstream order** (the items below have a natural
dependency chain — tackle in this order to minimise rework):

1. **Token-id constants + arch-trait overrides.** Cheapest;
   everything else depends on having `IMGPAD_ID` plumbed and the EOS
   stop-set + prompt-frame override registered. No daemon-state
   changes yet.
2. **Chat-template framing decision + render.** The Jinja2 vs
   ChatML-only call is a fork point — picks the override
   implementation shape and decides whether `apply_chatml_frame` is
   reused or bypassed.
3. **`LoadedModel` fields + dispatch arm.** Add struct fields, wire
   `load_model` arm for `arch_id == 8`, plumb `image_grid_thw`
   through the request payload.
4. **Splice + IMGPAD assertion.** Implement `generate_vl_dots_ocr`
   (or generic-dispatch refactor) with the splice loop + hard
   IMGPAD-count assert from the §"Vision token splicing" subsection
   below.
5. **Multi-image per-image loop.** Only matters once multi-image
   requests need to work; the daemon's existing single-image arms
   are sufficient for phase-3 ship if multi-image is deferred to a
   follow-up.

**Daemon `LoadedModel` extension.** Add `dots_ocr_config`,
`dots_ocr_weights`, `dots_ocr_state` fields (mirroring the `q35_*`
pattern). The text-side delegation means we hold a `Qwen2Weights`
plus dots.ocr vision weights side-by-side.

**Dispatch arms.** New arm in `load_model` (daemon.rs:672-677, :1494,
:1719, :3158, :3516) for `arch_id == 8`. New `generate_vl_dots_ocr`
or refactor `generate_vl` into a generic dispatcher branching on
arch_id.

**Token-id constants.** `IMGPAD_ID = 151665`, `IMG_START_ID = 151666`,
`IMG_END_ID = 151667` (already exported from
`dots_ocr.rs:1306-1314`).

**Architecture trait overrides (MANDATORY).**

- `prompt_frame_overrides`: emit the custom dots.ocr framing per
  §1.5. **Decision point:** does the daemon's chat-template renderer
  evaluate arbitrary Jinja2, or only ChatML?
  - Jinja2: register the dots.ocr template, done.
  - ChatML-only: hardcode the framing in the override.
- `eos_filter_overrides`:
  ```rust
  stop_at: vec![b"<|endofassistant|>".to_vec()],   // primary EOS (151673)
  holdback_prefixes: vec![b"<|end".to_vec()],
  strip_think: Some(false),                         // OCR model, no <think>
  ```
  Plus add 151643 (`<\|endoftext\|>`) and 151673 to the runtime's
  blocked-EOS-for-streaming list.

**Vision token splicing.**
- Hook the daemon prefill loop the same way qwen35-vl does
  (`daemon.rs:4178-4186` template): on `IMGPAD_ID`, call
  `qwen2::forward_step_with_embed` with the next merged visual
  embedding row.
- Plumb `image_grid_thw` through the daemon's generate request
  payload (no path for this in non-Qwen3.5 VL today).
- **IMGPAD count assertion.** The merger emits exactly
  `(grid_h / sm) * (grid_w / sm)` visual tokens per image. The
  prompt framer MUST insert exactly that count of IMGPAD between
  IMG_START and IMG_END. Hard assert at the splice site:
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
  unresolved IMGPAD positions in the text context — both are silent
  failures.
- **Multi-image attention leakage (HIGH RISK).** `vit_attention_opt`
  /`attention_dflash_*` is dense ViT attention; it does NOT support
  FlashAttention's `cu_seqlens` block-diagonal masking that HF's
  `flash_attn_varlen_func` uses for multi-image concatenation. Two
  paths:
  1. Per-image loop in the daemon: call `vision_forward` once per
     image (batch=1), concatenate merged tokens AFTER the vision
     pass but BEFORE the text-side splice. Safer, no kernel changes.
  2. Add `cu_seqlens` masking to the attention kernel. More work,
     better throughput at multi-image scale.

  Phase 3 implements (1) — `vision_forward` already documents
  single-image-only semantics. Phase 6+ (perf) may revisit (2).

**Example binary.** `crates/hipfire-runtime/examples/infer_dots_ocr.rs`
takes `--image path.png --prompt-template layout-all-en`, emits JSON
layout to stdout via the daemon path (vs the standalone phase-2
`examples/ocr_e2e.rs`).

### Phase 4 — correctness gate (8-12 hr)

OCR-specific coherence gate. Fluent ≠ correct.

**Already shipped** (commit `1f94da31`):
- `examples/ocr_e2e.rs` — runs hipfire end-to-end OCR on one image.
- `scripts/grade_dots_ocr_e2e.py` — bbox IoU + Levenshtein text
  distance + PASS/SOFT-PASS/FAIL verdict.
- Reference: `benchmarks/references/dots_ocr_smoke_001_vllm.json`
  (vLLM, parse_status ok, 13 regions).

**Remaining:**
- **Broaden the reference set.** Capture vLLM references for 4-9
  more diverse images (multi-column papers, forms, tables-heavy,
  scanned docs, dense PDFs). Commit at original resolution so
  smart-resize is exercised end-to-end.
- **`scripts/coherence-gate-dots-ocr.sh`** that runs `ocr_e2e` +
  grader in a loop over the reference set; PASS = all-images F1 >
  0.9, mean text-distance < 0.10.
- **Pre-commit hook trigger.** Add to `.githooks/pre-commit` trigger
  list when `crates/hipfire-arch-dots-ocr/`, dots-ocr-relevant
  kernels (vision RoPE, QKV split, ViT-shape attention, GELU tanh),
  or `hipfire-arch-qwen2` change.
- **Failure modes to gate on:**
  - **Parse failure** (invalid JSON) → HARD FAIL.
  - **Box matching:** Hungarian assignment by IoU between vLLM and
    hipfire box sets; ≥80% of vLLM boxes paired at IoU ≥ 0.85.
  - **Category equivalence:** Jaccard on category strings per pair
    ≥ 0.9.
  - **Coverage:** unpaired vLLM boxes > 20% → HARD FAIL.

### Phase 5 — quantisation (stretch, post-phase-3-merge)

Vision tower currently runs full F16 weights (linear weights are F16
on GPU; norms / biases are F32; activations everywhere F32). Now
that correctness is locked, candidates for quant:

- **Q8 on vision linear weights.** 50% memory savings vs F16 with
  near-zero quality impact at this layer count (42 blocks of small
  GEMMs).
- **HFQ4 on vision linear weights.** ~75% savings; might trigger a
  small per-block precision regression but the model's bf16-drift
  tolerance suggests it'll survive.
- **MQ4/MQ3-Lloyd on vision linear weights.** Aggressive sub-4-bit;
  high risk of crossing the trained-model precision floor.

Gate each step end-to-end via the phase-4 coherence gate. Don't
quantise norms, biases, or the merger's two MLP linears (small
parameter count, big sensitivity).

## 6. Open risks

(See devlog for the closed risk register: R1-R6, M9 — all resolved
during phases 1-2.)

- **Memory budget on gfx1151.** dots.ocr ~6 GB F16 weights + ~1.8 GB
  KV cache @ 128K context + vision activations at max image. Unified
  memory means host pressure visible too. Back-of-envelope check
  before pushing context length up.
- **Chat-template Jinja2 vs ChatML-only renderer.** ~~Decision in
  phase 3~~ **RESOLVED (phase 3):** the daemon has a minijinja renderer,
  but image-token expansion can't come from Jinja (it emits one
  `<|imgpad|>`, not N). The image-OCR turn is hand-rolled by
  `dots_ocr::build_prompt_ids` and verified byte-exact against the HF
  `apply_chat_template` capture. `prompt_frame_overrides` stays at
  default.
- **Multi-image attention leakage in `vit_attention_*`.** No
  cu_seqlens masking. Phase 3 mitigates with a per-image loop; an
  attention-kernel fix is phase 6+ perf work.
- **Smart-resize off-by-pixel.** Replicate the Python algorithm
  exactly; bbox accuracy depends on identical (H, W) selection.
  Currently passes byte-exact on the smoke image; broaden coverage
  in phase 4.

## 7. File layout

```
crates/
  hipfire-arch-qwen2/                 # phase 1
    Cargo.toml
    src/
      lib.rs
      arch.rs           # impl Architecture for Qwen2 (arch_id=7)
      qwen2.rs          # Qwen2Config, Qwen2Weights, Qwen2State,
                        # forward_step, forward_step_with_embed
    examples/
      infer_qwen2.rs    # text-only smoke binary

  hipfire-arch-dots-ocr/              # phase 2
    Cargo.toml
    src/
      lib.rs
      arch.rs           # impl Architecture for DotsOcr (arch_id=8)
      dots_ocr.rs       # DotsOcrConfig, DotsOcrWeights, vision_forward
      image.rs          # smart-resize + patch extraction
      rope.rs           # 2-D RoPE table builder
    examples/
      infer_dots_ocr.rs # vision-only validation driver
      ocr_e2e.rs        # end-to-end OCR (phase-2 throwaway, obsoleted by phase 3)
      dump_proj_weight.rs # numpy-replay diagnostic

kernels/src/
  rope_2d_halfsplit.hip
  rope_2d_halfsplit_qkv_interleaved.hip
  qkv_split_interleaved.hip
  attention_dflash_wmma.hip            # FlashAttention-2 + RDNA3 WMMA
  bf16_round_trip.hip                  # diagnostic, env-gated

benchmarks/
  images/
    dots_ocr_smoke_001.jpg             # phase 0 reference image
  references/
    dots_ocr_smoke_001.json            # HF/CPU/bf16 prefill-logit ref
    dots_ocr_smoke_001_vllm.json       # vLLM/GPU layout-JSON ref (THE gate)
    dots_ocr_smoke_001_activations/    # sampled per-stage HF activations

scripts/
  capture_dots_ocr_reference.py        # HF reference capture
  capture_dots_ocr_vllm.py             # vLLM reference capture
  capture_dots_ocr_activations.py      # per-stage activation capture
  diff_dots_ocr_stages.py              # per-stage cosine diff (diagnostic)
  numpy_attention_replay.py            # F32 reference attention (oracle for bf16 lesson)
  grade_dots_ocr_e2e.py                # F1 + Levenshtein scorer
  coherence-gate-dots-ocr.sh           # phase 4 (TODO)

docs/
  architecture-ids.md                  # arch_id 7 + 8 registered

docs/plans/
  dots-ocr-prd.md                      # this file
  dots-ocr-devlog.md                   # progress log
  dots-ocr.dots_ocr_manifest.txt       # phase 0
  dots-ocr.qwen2_1p5b_manifest.txt     # phase 0
```

## 8. Out of scope

- **True Qwen2.5-VL** (m-rope, window/full attention split, Qwen2.5
  text backbone). Defer until dots.ocr is shipped.
- **Video / temporal patching.** dots.ocr has `temporal_patch_size=1`;
  code paths don't assume it but no t-axis support needed.
- **Vulkan / cross-vendor backend.** Out of scope project-wide per
  CLAUDE.md rule 7.
- **Training.** Inference only.
- **Migrating `arch_id=1` from LLaMA to the Qwen2 crate.** Keep
  separate slots (7, 8) for the initial bring-up; consolidation is a
  follow-on PR.
- **`qwen_common` shared-primitives extraction.** Cross-arch
  TODO(transformer-extraction) markers placed during phase 2; refactor
  in a dedicated PR, not here.
