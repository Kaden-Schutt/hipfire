# Arch roster & feature matrix (multi-family bring-up)

Status: **reference** — drives the family-seam refactor + per-family plans.
Built 2026-06-19 from the actual `config.json`s under `/srv/huggingface`.

## Why this exists

The family roster isn't "a few more transformers" — it spans AR transformers,
hybrid SSM+attention, pure SSM, and block-diffusion. The seam
(`docs/plans/2026-06-19-daemon-family-seam.md`) must be designed against this
diversity, not against transformers alone. This doc is the shared picture.

## Feature matrix (verified configs)

| family | model_type | mixer layers | FFN | generation | new infra forced |
|---|---|---|---|---|---|
| gemma3 *(WIP)* | gemma3_text | SWA + full attn, dual-θ (5:1) | GeGLU | AR | GeGLU, (1+w) norm, dual-θ SWA |
| gemma4 | gemma4_text | SWA+full via `layer_types` list | GeGLU **or MoE** (A4B, 128 exp) | AR | MoE-on-gemma |
| diffusion_gemma | diffusion_gemma_text | SWA+full attn (`layer_types`) | MoE (128 exp, A4B) | **block diffusion** | block-diffusion loop |
| nemotron_h | nemotron_h | **Mamba2 + attn** interleaved (`hybrid_override_pattern`) | dense **or MoE** (128–512 exp) | AR | **Mamba2 SSM+conv kernels** |
| mamba2 | ssm_cfg.layer=Mamba2 | **pure Mamba2** (no attn/KV) | — | AR | **Mamba2 SSM+conv kernels** |
| LFM2 *(WIP, arch_id 11)* | lfm2 / lfm2_moe | **short-conv** + attn | dense or MoE (32 exp) | AR | short-conv state |

Key per-config specifics:
- **gemma4**: head_dim 256, gelu_pytorch_tanh, sliding_window 512/1024, layer
  types as an explicit `layer_types: [sliding_attention|full_attention]` list
  (vs gemma3's `sliding_window_pattern` int). Dense 31B ≈ gemma3-31B; 26B-A4B
  + E2B/E4B are MoE. `Gemma4ForConditionalGeneration` (multimodal wrapper).
- **diffusion_gemma** (`DiffusionGemmaForBlockDiffusion`, 26B-A4B): gemma-shaped
  transformer layers (head_dim 256, GeGLU, SWA, MoE 128) but **block-diffusion**
  generation. Forward is reusable from gemma4; the loop is the novelty.
- **nemotron_h** (`NemotronHForCausalLM`): `hybrid_override_pattern` of M
  (Mamba2) / `*` (attention) / `-` (dense MLP) / E (MoE) per layer.
  mamba_head_dim 64–80, mamba_num_heads 64–128, ssm_state_size 128, conv_kernel
  4. Nano-4B dense-FFN; 30B-A3B (128 exp) + Super-120B (512 exp) MoE; Super also
  has an MTP head (`mtp_hybrid_override_pattern`). 131072 vocab.
- **mamba2** (state-spaces, 130m/2.7b): original (non-HF-transformers) checkpoint
  format — `ssm_cfg={'layer':'Mamba2'}`, no `architectures`, GPT-NeoX vocab
  50277. Pure SSM, the clean kernel-validation vehicle.
- **LFM2.5**: `conv` mixer layers (`conv_L_cache=3`, `conv_dim`) interleaved with
  attention; `lfm2_moe` (8B-A1B) adds 32-expert MoE.

## Two truths this forces on the seam

1. **Generation strategy ⊥ model-forward.** diffusion_gemma proves it: gemma
   transformer layers + block-diffusion loop. So the forward (run the layer
   stack → hidden/logits) must be **separable from the loop**. `SimpleAr` is the
   *AR strategy over a forward*, not the owner of the forward; a `BlockDiffusion`
   strategy reuses the same forward. `ServingBackend` (the boxed seam) is the
   strategy; `GenerateCtx` must not assume causal/KV.
2. **The layer stack is a heterogeneous per-layer mixer list.** mixer ∈
   {full-attn, SWA, Mamba2, short-conv} × FFN ∈ {SwiGLU, GeGLU, MoE}, selected
   per layer (nemotron `hybrid_override_pattern`, lfm2/gemma `layer_types`).
   qwen35's LA/FA hybrid is the existing precedent to generalize.

## Two big NEW infra investments (rest is composition)

- **Mamba2 SSM + conv1d kernels** — `mamba2` and `nemotron_h`. Build/validate on
  **pure mamba2 first** (no attn/MoE confounds), then nemotron_h composes them.
- **Block-diffusion generation loop** — `diffusion_gemma`, reusing gemma4 forward.

## Dependency-aware bring-up order

1. **gemma3** (WIP) — GeGLU/SWA/(1+w), `SimpleAr` + `ServingBackend`.
2. **seam wiring** — route qwen2 + gemma3 through it.
3. **mamba2 (pure)** — land SSM+conv kernels in isolation; validate `SimpleAr`
   on a no-KV recurrent arch.
4. **gemma4** — gemma3 + MoE (reuse qwen35-MoE) + `layer_types` form; cheap.
5. **nemotron_h** — Mamba2(3) + attn + MoE(4) + per-layer hybrid dispatch.
6. **diffusion_gemma** — gemma4 forward(4) + block-diffusion `ServingBackend`.
7. **LFM2** finish + shared-loader / Option-soup cleanup folded across.

Builds each new capability in its cheapest isolating context, composes upward,
and stresses the seam early (no-KV at #3, non-AR at #6).

## Cleanup that the roster justifies

- **Shared transformer loader**: `load_weight_tensor` / `load_norm` /
  `load_embed` / `load_lm_head` are now duplicated across qwen35 / qwen2 /
  dots-ocr / gemma3 (all carry `TODO(transformer-extraction)`). With 4+ more
  transformer families coming, extract into `hipfire_runtime::transformer` after
  gemma3 decodes (generalize on the working instance, not before).
- **Shared MoE FFN**: qwen35-MoE, gemma4, nemotron_h, lfm2_moe, diffusion_gemma
  all need MoE — converge on one expert-FFN building block (watch differing
  expert tensor layouts: stacked-3D vs per-expert-split).

---

# Part 2 — the rest of /srv/huggingface (beyond text decoders)

Surveyed 2026-06-19 (130 model dirs). Most are variants of the Part-1 families
(Qwen2.5/3/3.5/3.6, gemma4, nemotron, lfm2, llama, deepseek4, minimax). The
genuinely new work clusters into a few **machinery classes** layered on top of —
or beside — the text core.

## The tiered shape this reveals

An arch is no longer just "a forward + an AR loop." It decomposes into:

```
INPUT ADAPTERS              CORE FORWARD                OUTPUT STRATEGY
  text tokenizer       →    layer stack (the         →    AR decode (text)        ← ServingBackend
  vision encoder       →    Part-1 mixer/FFN          →    pooling (embedding)
  audio encoder        →    families)                 →    scoring (rerank)
                                                       →    speech tokens → codec→wav (TTS/omni)
                                                       →    block diffusion (diffusion_gemma)
                                                       →    image-latent diffusion (Qwen-Image, separate)
```

`ServingBackend` is **one output strategy among several**; input adapters
(vision/audio → embeddings spliced into the core, à la `hipfire-arch-qwen35-vl`)
and output heads are orthogonal axes. This is the generalization the seam must
allow for, even though near-term we only wire the AR strategy.

## Family taxonomy (the remainder)

| group | families | disposition |
|---|---|---|
| **A. reuse existing text family** | Llama-3.x/3.1/3.2/3.3 (llama 0), MiniCPM5 (`llama`), Qwen2/2.5 (qwen2 7), Qwen3 dense/MoE (llama/qwen35), MiniMax-M2.7 (`minimax_m2`, arch 10), all PARO/DFlash z-lab variants | loader/config only; no new arch |
| **B. new text-arch crate** | **llama4** (`llama4_text`: 48L, 16-expert MoE, iRoPE + early-fusion vision), **hrm_text** (hierarchical recurrent reasoning, H/L cycles — not a flat layer stack), **zaya** (Zyphra, 80L hybrid — study at bring-up) | new crates; llama4 also needs vision |
| **C. multimodal input** | medgemma-*-it / gemma4-mm (vision), **sensenova neo_chat/MoT** (qwen3 LLM + `neo_vision`), llama4 vision | vision-encoder adapter (extend qwen35-vl) + existing/new core |
| **D. omni (in + out)** | **Qwen2.5-Omni** (thinker + talker + token2wav), **Qwen3-Omni-MoE** (thinker + talker + code2wav), **Nemotron-Omni** (parakeet sound + nemotron_h llm + vision) | orchestration of C + audio-in + TTS-out |
| **E. non-generative heads** | **Qwen3-Embedding** (0.6/4/8B), **Qwen3-Reranker** (0.6/4/8B) — `Qwen3ForCausalLM` backbone, no AR loop | cheap: reuse qwen3 forward, add encode→pool / score head |
| **F. audio subsystems** | STT: **kyutai-stt**, **parakeet-tdt/ctc** (.nemo), **conformer-ctc** (.nemo), parakeet-realtime/multitalker. TTS: **Kokoro-82M**, **kyutai pocket-tts**, **supertonic-3** (onnx), **personaplex**. VAD/diarization: **pyannote** (config.yaml) | large new subsystems (see below) |
| **G. image generation** | **Qwen-Image**, **Qwen-Image-Edit** (`model_index.json` — diffusers MMDiT+VAE pipeline) | separate engine; out of near-term text-inference scope |
| **misc / not models** | froggeric (chat templates), tiny-random-qwen3-moe (test fixture), Qwen SAE-Res (interpretability SAE) | skip |

## New machinery subsystems (scope + reuse)

1. **Vision encoder** — extend the `hipfire-arch-qwen35-vl` splice pattern
   (ViT → projector → image tokens into the decoder input). Consumers:
   gemma multimodal, llama4_vision (34L), neo_vision, omni vision.
2. **Audio encoder / ASR front-end** — mel-spectrogram features → Conformer/
   Whisper-style encoder → CTC / TDT / RNN-T decode. Consumers: parakeet,
   kyutai-stt, conformer, nemotron-omni `sound_config(parakeet)`, and the
   omni **thinker** `audio_encoder`. Big new subsystem; build/validate on a
   standalone STT (conformer-ctc) first.
3. **Speech synthesis (TTS)** — G2P → acoustic model → neural codec/vocoder.
   Consumers: Kokoro, pocket-tts, supertonic, personaplex, and the omni
   **talker** (speech-token LM) + **code2wav/token2wav** (codec → waveform).
4. **Omni orchestration** — thinker (multimodal LLM, ingests text+vision+audio)
   → talker (autoregressive speech-token LM conditioned on thinker hidden) →
   codec vocoder. Composes (1)+(2)+(3) over a text core. Qwen2.5/3-Omni,
   Nemotron-Omni.
5. **Non-generative output heads** — `encode → pool` (embedding) and pairwise
   `score` (rerank). The **cheapest** new capability: the qwen3 forward already
   exists; add a pooled/scoring output path + skip the AR loop. Good first proof
   that `ServingBackend` spans non-AR *output* (complements diffusion's non-AR
   *generation*).
6. **Image generation** — MMDiT + VAE + text-encoder diffusers pipeline
   (Qwen-Image). A distinct engine (image-latent diffusion); treat as out of the
   text/audio inference scope unless explicitly prioritized.
7. **VAD / diarization** — pyannote segmentation models; preprocessing for
   streaming STT/omni, not generation.

## Non-HFQ formats (need dedicated loaders, or are out of the safetensors path)

- **`.nemo`** (parakeet, conformer) — NVIDIA NeMo tarball (weights + yaml).
- **`model_index.json`** (Qwen-Image) — diffusers multi-component pipeline.
- **`config.yaml`** (pyannote) — pyannote framework.
- **`model_type=onnx`** (supertonic) — ONNX runtime graph.

These don't flow through the HFQ/safetensors quantizer ingest; each needs a
bespoke loader (or stays out of scope).

## Scope framing (honest)

hipfire is HIP/ROCm **text-inference-first**. Of the above, the near-term-cheap
wins that fit the current engine are **(E) embedding/rerank heads** (qwen3
forward exists) and **(C) vision input** (qwen35-vl precedent). **(B) llama4**
is a normal-but-large new text+vision arch. **(F) audio STT/TTS** and **(D) omni**
are major multi-quarter subsystems; **(G) image-gen** is a separate engine. The
family-seam refactor (output-strategy + input-adapter decomposition) is exactly
what keeps these addable later without re-architecting the core.
