# Krea-2-Turbo DiT diffusion pipeline (plan)

Status: proposed (2026-07-02). Branch: `chaingun`.

Bring Krea-2-Turbo — a flow-matching MMDiT image model — into
`hipfire-diffusion`. This is the crate's first **transformer (DiT)** pipeline;
the existing pipeline is Stable-Diffusion-shaped (UNet + CLIP + KL-VAE +
DDPM/Euler-ε). The work is a new forward + scheduler + VAE + text-encoder
wiring, not a config addition. It aligns unusually well with the in-flight
oq4/oq8 diffusion quant work (Phases 1–4): the DiT is **all linear layers**, so
the oq weight ladder + progressive per-step precision schedule apply directly,
with none of the conv-quant awkwardness.

Source model: `/srv/huggingface/usb256GB/Krea-2-Turbo` (diffusers format,
`_diffusers_version 0.39.0.dev0`, `is_distilled: true`, `patch_size: 2`).

## 0. Implementation status (2026-07-02)

Discovery during P0 revealed the crate already carries **substantial Qwen-Image /
Krea2 DiT scaffolding** (in `transformer.rs`, ~2.25k lines, from the diffusion
WIP lineage). The gap is smaller than a from-scratch build — the goal is to
*upgrade/wire* the existing pieces, not write a new pipeline. Current state:

**Done / working:**
- Import + metadata: `import_diffusers_to_hfq` recognizes `Krea2Pipeline`,
  the sharded transformer (`parse_sharded_safetensors_state_dict`),
  `Krea2Transformer2DModel`, `AutoencoderKLQwenImage`,
  `FlowMatchEulerDiscreteScheduler`. **P0 metadata fidelity fixed this session:**
  `latent_channels` now = VAE `z_dim` (16, not the patchified 64); tokenizer
  detected as `qwen2-bpe` with `tokenizer.json`/`chat_template.jinja` captured;
  text-encoder class read from `architectures` (`Qwen3VLModel`). All 17 import
  tests green.
- Topology detection: `TransformerDenoiserFamily::Krea2`, patch-size 2,
  `has_text_fusion`, block enumeration.
- Per-component Krea2 loaders: `NativeTransformerDenoiserIo` (img_in/txt_in/
  final_layer), `NativeTransformerTimestepEmbedding` (`time_embed`),
  `NativeTransformerBlockModulation` (`scale_shift_table` +
  `krea_scale_shift_with_runtime_context`), `krea_swiglu_from_hfq`,
  attention QKV/RoPE/`attend_image_text`.
- `AutoencoderKLQwenImage` accepted by the native VAE support gate.

**Not yet wired (the remaining work):**
- **Denoiser assembly + forward gate Krea2 out.** `NativeTransformerDenoiser::
  from_hfq` and both `forward_qwen_with_runtime_context` (block + denoiser)
  hard-reject non-`QwenImage`. Krea2 needs its own **single-stream** block
  forward: concat `[text, image]`, `Krea2RMSNorm` (weighted, eps 1e-5), 6-chunk
  adaLN in `[prescale,preshift,pregate,postscale,postshift,postgate]` order via
  `krea_scale_shift`, **sigmoid `to_gate`** on the SDPA output, GQA 48/12,
  SwiGLU, then split + `final_layer`. (Contrast: the existing QwenImage block is
  double-stream, LayerNorm-no-affine eps 1e-6, GEGLU, per-stream modulation.)
- **text_fusion** (§Q3): 12-layer stack → 2×layerwise (attend layer axis) →
  `projector[12→1]` → 2×refiner (attend token axis) → `txt_in`. Only the
  `has_text_fusion` flag exists; no forward.
- **Qwen3-VL text-encoder hidden-state extraction** (§4): produce the 12 selected
  layers `[2,5,…,35]`. Not wired.
- **Flow-matching scheduler** (§3.8) and **Qwen-Image 3D VAE decode** (§5).

Revised phasing: **P1** = Krea2 single-stream block + denoiser forward, unit-
tested against a tiny Krea2 fixture with synthetic text_hidden (mirrors the
existing tiny QwenImage fixtures — no 8.9 GB encoder needed). **P2** = text_fusion
+ Qwen3-VL encoder. **P3** = flow scheduler + VAE decode → end-to-end. **P4** =
oq quant. This front-loads the self-contained DiT-forward work.

## 1. What the model is

`Krea2Pipeline` components (from `model_index.json`):

| Component | Class | Size | hipfire status |
|---|---|---|---|
| Transformer | `Krea2Transformer2DModel` | ~26 GB, 3 shards, 430 tensors | new (DiT) |
| Text encoder | `Qwen3VLModel` (`qwen3_vl_text`, 36L, hidden 2560) | 8.9 GB | reuse `hipfire-arch-qwen35-vl` |
| VAE | `AutoencoderKLQwenImage` | 0.5 GB | conv-KL, new variant |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | — | new (flow matching) |
| Tokenizer | `Qwen2Tokenizer` | — | reuse existing tokenizer path |

`turbo.safetensors` (26 GB, single file) is the all-in-one variant; prefer the
sharded diffusers layout for conversion (per-component control).

### 1.1 Transformer forward (Krea2Transformer2DModel)

Config: `num_layers 28`, `num_attention_heads 48`, `num_key_value_heads 12`
(GQA 4:1), `attention_head_dim 128` (hidden = 6144), `intermediate_size 16384`
(SwiGLU FFN), `axes_dims_rope [32,48,48]` (3D RoPE, sums to head_dim 128),
`rope_theta 1000`, `in_channels 64`, `patch_size 2`, `timestep_embed_dim 256`.
Text side: `num_text_layers 12`, `text_hidden_dim 2560`,
`text_intermediate_size 6912`, `num_layerwise_text_blocks 2`,
`num_refiner_text_blocks 2`.

Graph (from the tensor manifest, `N` = layer index):

- **Input embed:** `img_in` (linear, 64·patch² → 6144); `txt_in`
  (`linear_1 → norm → linear_2`); `time_embed` (`linear_1 → linear_2`) +
  `time_mod_proj` (adaLN modulation source).
- **`text_fusion`** — fuses the 12 selected Qwen3-VL hidden states
  (`text_encoder_select_layers = [2,5,8,…,35]`) into the image stream:
  `projector` (2560 → model dim) + 2 `layerwise_blocks` + 2 `refiner_blocks`.
  Each fusion block = attention (`to_q/k/v/gate/to_out.0`, QK-norm
  `norm_q/norm_k`) + SwiGLU `ff.{gate,up,down}` + `norm1/norm2`.
- **28 `transformer_blocks`** (main stack), each:
  - `attn`: `to_q/to_k/to_v` (GQA 48/12 heads), `norm_q/norm_k` (QK RMSNorm),
    3D RoPE, `to_gate` (gated attention output), `to_out.0`.
  - `ff`: `gate/up/down` (SwiGLU, intermediate 16384).
  - `norm1/norm2` (RMSNorm) + `scale_shift_table` (adaLN-zero modulation from
    the timestep embedding: shift/scale/gate per sub-layer).
- **`final_layer`:** `norm` + `scale_shift_table` + `linear` → unpatchify to
  `in_channels` latent.

This is a Flux/Qwen-Image-family single-stream adaLN MMDiT with QK-norm and
gated attention. Nothing exotic per-op; the novelty for hipfire is the
**assembly** (adaLN modulation, patchify, 3D RoPE, GQA) and the **text fusion**.

## 2. Reuse map (what already exists)

`hipfire-diffusion::gpu_ops` resident primitives that carry over directly:

- `linear_optional_bias_resident` — every DiT linear (img_in, txt_in, all
  to_*, ff.*, final_layer.linear).
- `scaled_dot_product_attention_resident` — the flash-style SDPA from the
  Phase-1b resident work; DiT attention core.
- `timestep_embedding_hip_on_gpu` — sinusoidal timestep embed.
- `silu_resident` + a gate multiply — SwiGLU (or add a fused `swiglu_resident`
  mirroring `geglu_gate_3d_resident`).
- VAE: `conv2d_nchw_wmma_resident`, `group_norm_nchw_resident`,
  `upsample_nearest2d_nchw_resident`, `add_channel_bias_nchw_resident`,
  `nchw_to_bsc`/`bsc_to_nchw` — the KL-VAE decode scaffold.
- HFQ container + loader: `DiffusionPipeline::open_hfq`, `DiffusionHfqMetadata`,
  `summarize_hfq`, `inspect_hfq_with_runtime_support` (`lib.rs`).
- Quant: the Phase 1–4 oq path — `quant_encode.rs` (oq4/oq8 encoders),
  `quant_decode.rs` (`decode_oq4g256_slice`/`decode_oq8g256_slice`),
  `quant_calib.rs` (activation capture → `.calib.hfq`), progressive per-step
  precision schedule.

Lift from `rdna-compute` (already in-tree, portable RDNA2/3/4):

- RoPE: `dispatch/rope.rs` (`rope_f32`, `rope_batched_f32`,
  `rope_2d_halfsplit_f32`) — extend/compose for the 3-axis `[32,48,48]` split.
- RMSNorm: the dispatch rmsnorm kernel (referenced by `profile::rmsnorm_bytes`).

## 3. New work (gaps)

DiT-specific ops to add to `gpu_ops.rs` (+ HIP kernels where needed):

1. **RMSNorm resident** (`rms_norm_resident`) — for `norm1/norm2` and QK-norm.
   Prefer wiring the existing rdna-compute rmsnorm over a new kernel.
2. **3D RoPE** for `axes_dims_rope [32,48,48]` — image tokens carry a
   (frame,h,w) position; compose per-axis rotation over the 128-dim head.
   Reuse the rope dispatch; add the 3-axis position/section handling.
3. **adaLN modulation** (`adaln_modulate_resident`) — 6-chunk Wan/PixArt form
   (see §10): `modulation = temb.unflatten(-1,(6,-1)) + scale_shift_table`, then
   `[prescale, preshift, pregate, postscale, postshift, postgate]`. Applied as
   `attn_out = attn((1+prescale)·norm1(x) + preshift)`; `x += pregate·attn_out`;
   `ff_out = ff((1+postscale)·norm2(x) + postshift)`; `x += postgate·ff_out`.
   `temb` = shared `time_mod_proj(time_embed)` (6·6144), added to each block's
   own `scale_shift_table [6,6144]`. `final_layer` uses a `[2,6144]` (shift,
   scale) table, no gate.
4. **GQA head-repeat** — expand 12 KV heads → 48 Q heads (repeat_kv 4×) before
   `scaled_dot_product_attention_resident`.
5. **Sigmoid attention gate** (`to_gate`) — `attn = attn · sigmoid(to_gate(xm))`
   on the flattened SDPA output (`xm` = the modulated norm1 input), before
   `to_out.0`. Not the adaLN gate (that is `pregate`, applied to the residual).
6. **Patchify / unpatchify** (patch_size 2) — latent [B,64,H,W] ↔ token
   sequence [B, H/2·W/2, 6144]; a layout kernel (see
   `launch_diffusion_layout_kernel`).
7. **SwiGLU resident** — SiLU-gated FFN (silu(gate)·up → down).
8. **Flow-matching scheduler** in `scheduler.rs`:
   `FlowMatchEulerDiscreteScheduler` — sigma/timestep schedule with
   `use_dynamic_shifting`, `base_shift 0.5`/`max_shift 1.15`, exponential time
   shift, `max_image_seq_len 6400`. The velocity update replaces the ε/v-pred
   `euler_step`; add `flow_match_step_hip_on_gpu` alongside the existing
   `euler_step_hip_on_gpu`.

## 4. Text encoder (Qwen3-VL) — the reuse seam

The text encoder is a full `qwen3_vl_text` LM (36 layers, mRoPE interleaved,
GQA 32/8). hipfire already runs this arch (`hipfire-arch-qwen35-vl`), but as an
autoregressive *generator*. Krea-2 needs it as a **feature extractor**: run the
prompt through it and capture hidden states at layers `[2,5,8,…,35]`, then feed
those 12 tensors to `text_fusion`.

Plan item: add a **hidden-state-extraction seam** to the qwen3_vl forward — a
prefill-only path returning selected mid-layer hidden states (no lm_head, no
decode loop). This mirrors the activation-capture hook the Hessian collector /
oq calibration already use, so it likely reuses `ActivationCapture` wiring.
Scope this concretely in the text-encoder phase (its exact API needs a read of
the current qwen35-vl forward). This keeps the heavy 8.9 GB encoder on the
proven runtime path rather than reimplementing it in the diffusion crate.

Per AGENTS.md, weight import/conversion is coexistence tooling — the
safetensors→`.hfq` conversion for all three components belongs in
`hipfire-coexistence` / `hipfire-quantize`, not the inference path.

## 5. VAE (AutoencoderKLQwenImage) — resolved

Confirmed from `vae/config.json`: this is the **Qwen-Image / Wan-lineage 3D
causal VAE**, not an SD 2D VAE:

- `z_dim: 16` latent channels → the DiT `in_channels 64` = 16 × (2×2 patch). ✓
- `base_dim 96`, `dim_mult [1,2,4,4]` → 4 stages / 3 spatial downsamples =
  **/8 spatial**. `num_res_blocks 2`, `input_channels 3`, `attn_scales []`.
- `temperal_downsample [false, true, true]` → **3D causal conv** decoder
  (video-capable). For still-image T2I the temporal length is 1, but the conv
  stack is 3D-causal, so the reused `conv2d` path must generalize to a
  degenerate-time `conv3d` (or a 2D specialization when T=1).
- **Latent normalization is per-channel**, not a scalar `scaling_factor`:
  `latents_mean`/`latents_std` are 16-vectors. Decode does
  `z = latent · latents_std + latents_mean` before the conv decoder; encode
  inverts it. This must be wired explicitly (a common SD-porting bug).

Reuse the `vae.rs` conv/group_norm/upsample scaffold with a T=1 specialization;
decode-only first (generation), encode only for img2img/inpaint.

## 6. Quantization fit (the payoff)

The DiT is GEMM-dominated, so the oq work applies with no new format:

- **Weights:** oq4/oq4++/oq8 on all `transformer_blocks` linears (to_*, ff.*).
  `text_fusion` and `img_in/txt_in/final_layer` likely stay higher precision
  (small, sensitive) — decide via the calib fidelity harness.
- **Calibration:** reuse `quant_calib.rs` activation capture on the DiT forward
  (the capture sites are linear inputs, same as the LLM path).
- **Progressive per-step schedule (Phase 4):** directly reusable — early
  flow-matching steps at higher activation precision (W4A8), later steps W4A4.
  Distilled turbo has few steps, so the schedule is short and each step's
  precision choice matters more; this is a good stress test for the schedule.
- **Artifact naming (AGENTS.md):** e.g.
  `Krea-2-Turbo-24B.dit.oq4++.gfx1151.hfq` — `family` Krea, `version` 2, `turbo`
  tag, `.dit` role/feature group before the quant token, `oq4++` for
  Hessian/LDLQ, arch sidecar. Confirm parameter count for the size field during
  conversion. VAE and text-encoder ship as their own `.hfq` sidecars.

## 7. Phasing

Mirrors the diffusion-quant cadence; each phase is independently validatable on
gfx1103/gfx1151 (coordinate GPU via `hipfire lock`).

- **P0 — Convert + inspect.** safetensors→`.hfq` for transformer/VAE/text
  encoder via the coexistence/quantize path; extend `DiffusionHfqMetadata` to
  carry the DiT config (heads, layers, rope axes, patch, text-fusion counts).
  `summarize_hfq` reports the new pipeline kind. No forward yet.
- **P1 — Text encoder.** qwen3_vl hidden-state-extraction seam; dump the 12
  selected layers for a prompt; parity vs a diffusers reference capture. Fully
  self-contained (already-supported arch).
- **P2 — DiT block (CPU/GPU parity).** One `transformer_blocks` layer end to
  end: RMSNorm, QK-norm, 3D RoPE, GQA SDPA, gated out, SwiGLU, adaLN. CPU
  reference + GPU parity on a tiny fixture before scaling to 28 layers.
- **P3 — Full DiT forward + flow-matching scheduler.** img_in→28 blocks→
  final_layer→unpatchify; flow-match sigma schedule + step; produce a latent.
- **P4 — VAE decode.** Qwen-Image KL-VAE latent→RGB; end-to-end image out.
- **P5 — Quantize.** oq4/oq8 weights + calibration + progressive per-step
  schedule; fidelity harness (PSNR/perceptual) vs bf16, per the Phase-2/3 quant
  findings (oq8 ~lossless, naive q4 degrades — expect oq4++ needed here).

## 8. Risks / remaining unknowns

The three "read from reference, don't assume" gates are now **resolved** (§10).
What remains:

- **`scale_rope` flag** — whether Krea centers image H/W RoPE positions
  (neg/pos split) or uses `[0..H)`. Read from the `QwenEmbedRope(...)` ctor args
  in the Krea2 `__init__` at conversion time; a one-line check, affects position
  ids only.
- **`time_embed` sinusoidal convention** — the exact timestep-embedding
  frequency layout feeding `linear_1 [6144,256]` (256 = `timestep_embed_dim`);
  match `timestep_embedding_hip_on_gpu`'s convention or adapt. Verify by parity
  on the timestep path in P2.
- **Distilled turbo guidance/steps** — `is_distilled: true`; turbo models are
  often guidance-distilled (no CFG). Confirm whether the pipeline runs CFG at
  all (affects the batched-CFG path and the per-step precision schedule length).
- **Qwen3-VL prompt template** — the exact chat/template wrapping the encoder
  expects (`tokenizer/chat_template.jinja` present); parity in P1.
- **Memory** — 26 GB transformer + 8.9 GB encoder + VAE. bf16 fits halo
  (128 GB); gfx1103 (64 GB UMA) needs oq4/oq8 weights to be comfortable — a
  direct motivation for P5 and for weight paging if bf16 is wanted on nix1.

## 9. First step

P0 conversion + `DiffusionHfqMetadata` extension. The three reference-forward
gates are resolved (§10), so P0 is unblocked; carry the §10 spec into the
metadata schema (heads 48/12, head_dim 128, 28 blocks, rope axes [32,48,48]
θ=1000, patch 2, z_dim 16, per-channel latent mean/std, text-fusion
2×layerwise + projector[12→1] + 2×refiner, select_layers [2,5,…,35]).

## 10. Reference-forward resolutions (2026-07-02)

Resolved from tensor shapes + the `Krea2Transformer2DModel` / `QwenEmbedRope` /
`AutoencoderKLQwenImage` reference (diffusers main; local 0.38 QwenImage +
fetched 0.39-dev Krea2). All three gating questions answered:

### Q1 — VAE — see §5 (z=16, /8 spatial, 3D causal, per-channel mean/std).

### Q2 — 3D RoPE (`QwenEmbedRope`, axes [32,48,48], θ=1000)

- Per-axis freqs `polar(1, outer(index, θ^(-2i/dim_axis)))`; the three axes
  (frame, height, width) get 32/48/48 dims (Σ = head_dim 128).
- Image tokens: `video_fhw = (frame=1, H_grid, W_grid)` where the grid is the
  **post-patchify lattice** = (latent /8, then patch /2) = (H_img/16, W_img/16).
  Frame index constant; per-token freqs = concat of frame/height/width freqs
  expanded over the lattice → `[seq_img, 64]` complex.
- **Text tokens are RoPE-positioned after the image**:
  `txt_freqs = pos_freqs[max_vid_index : max_vid_index + txt_len]`.
- Applied to Q and K (both image and text) as complex rotation before attention.

### Q3 — adaLN block + attention + text_fusion

Single-stream joint-attention DiT block (`x` = image tokens; text prepended):

```
# once per forward, shared:
temb  = time_mod_proj(time_embed(sinusoid(t, 256)))      # [B, 6*6144]
# joint sequence for the 28 blocks:
seq   = cat([text_tokens_6144, image_tokens_6144], dim=1)

# per block:
mod = temb.unflatten(-1,(6,-1)) + block.scale_shift_table  # [B,6,6144]
prescale, preshift, pregate, postscale, postshift, postgate = mod.unbind(-2)
xm  = (1+prescale) * RMSNorm(seq, norm1.weight) + preshift
q,k,v = to_q(xm), to_k(xm), to_v(xm)          # 48 / 12 / 12 heads, head_dim 128
q = RMSNorm(q, norm_q); k = RMSNorm(k, norm_k)   # QK-norm on head_dim
q,k = rope(q,k)                                   # §Q2, GQA: repeat_kv 4x
a   = sdpa(q,k,v, enable_gqa=True)                # joint over [text; image]
a   = a * sigmoid(to_gate(xm))                    # sigmoid attention gate
a   = to_out.0(a)
seq = seq + pregate * a
xm2 = (1+postscale) * RMSNorm(seq, norm2.weight) + postshift
f   = down( silu(gate(xm2)) * up(xm2) )           # SwiGLU, intermediate 16384
seq = seq + postgate * f

# after 28 blocks, image part only:
img = seq[:, txt_len:, :]
mod2 = final_layer.scale_shift_table + proj(temb)  # [2,6144] -> shift, scale
out = final_layer.linear((1+scale)*RMSNorm(img, final.norm) + shift)  # ->64
latent = unpatchify(out)                           # [B,16, H/8, W/8]
```

- `norm1/norm2` = `Krea2RMSNorm` (weighted, eps 1e-5). `to_gate` gate is
  **sigmoid** on the flattened SDPA output — distinct from the adaLN `pregate`
  residual gate.
- `img_in` is a plain `Linear(64→6144)` on already-patchified latent (patchify =
  a layout/reshape of the 16-ch latent into 2×2 blocks, done before `img_in`).

**text_fusion** (produces the text tokens fed to `txt_in`, a pre-pass — it does
NOT interleave with the 28 blocks):

```
h = stack(select_layers)            # [B, seq, 12, 2560]  (layers 2,5,8,…,35)
for blk in layerwise_blocks(2): h = blk(h)   # attend over the 12-LAYER axis per token
h = projector(h).squeeze(-1)        # [1,12] collapses 12 layers -> [B, seq, 2560]
for blk in refiner_blocks(2): h = blk(h)     # attend over the TOKEN axis (self-attn)
text_tokens = txt_in.linear_2(txt_in.linear_1(RMSNorm(h, txt_in.norm)))  # 2560->6144
```

Fusion/refiner blocks are the same attn(QK-norm)+SwiGLU shape at dim 2560
(20 heads, no GQA), intermediate 6912.
