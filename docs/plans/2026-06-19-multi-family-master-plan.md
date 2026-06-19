# Multi-family master plan — architecture + roadmap

Status: **active master plan** — supersedes the sequencing in the per-family
docs; they become sub-plans this orders. Built 2026-06-19.

Reference data: `2026-06-19-arch-roster-feature-matrix.md` (the full survey).
Sub-plans: `2026-06-19-gemma3-bringup.md`, `2026-06-19-daemon-family-seam.md`.

## What's settled (the constraints this plan must honor)

1. **The roster is wide** — AR transformers (gemma3/4, llama4, qwen*), hybrid
   SSM+attn (nemotron_h), pure SSM (mamba2), short-conv hybrid (lfm2),
   block-diffusion (diffusion_gemma), plus non-text machinery: vision/audio
   input, embedding/rerank heads, TTS/STT, omni, image-gen.
2. **Generation strategy ⊥ model-forward** — diffusion_gemma reuses gemma layers
   with a diffusion loop; embedding/rerank reuse qwen3 layers with no loop. The
   forward must be separable from the loop.
3. **The core is a heterogeneous per-layer mixer stack** — mixer ∈ {full-attn,
   SWA, Mamba2, short-conv, DeltaNet} × FFN ∈ {SwiGLU, GeGLU, MoE}, selected per
   layer. qwen35's LA/FA hybrid is the precedent to generalize.
4. **Runtime is HFQ-only** — every foreign format (.nemo, diffusers, pyannote,
   onnx, safetensors, GGUF) converts to HFQ offline via `hipfire-quantize`
   front-ends. The loader surface never grows per format.
5. **Realtime is a serving mode** — streaming/duplex audio (parakeet-realtime,
   kyutai, Qwen-Omni) is latency-bounded (RTF<1, per-frame deadline); the seam
   must admit a streaming I/O mode, not only batch prompt→N-tokens.
6. **Daemon de-qwen is owned here** (upstream branch abandoned): collapse the
   `LoadedModel` Option-soup → `Box<dyn ServingBackend>`; delete per-arch
   `generate_*`.

## Target architecture — three tiers

An arch decomposes into orthogonal tiers wired by `arch_id` + HFQ metadata:

```
 ┌ Tier 1: INPUT ADAPTERS ┐   ┌ Tier 2: CORE FORWARD ┐   ┌ Tier 3: OUTPUT STRATEGY ┐
   text tokenizer            │  heterogeneous layer    │   ArDecode (text, batch/stream)
   vision encoder  ──embeds──▶  stack: per-layer        ──▶ Pool (embedding) / Score (rerank)
   audio encoder             │  (Mixer × Ffn);          │   BlockDiffusion
   (splice at placeholders)  │  separable from any loop │   SpeechDecode + Codec (stream, RTF<1)
 └────────────────────────┘   └──────────────────────┘   └─────────────────────────┘
```

Intended Rust shapes (finalized in Epoch 2 — P0's `SimpleAr`/`ArchCaps` is the
seed; this refines it):

```rust
// Tier 1 — modality → core-input embeddings, spliced at placeholder positions.
trait InputAdapter { fn encode(&mut self, gpu: &mut Gpu, input: ModalInput) -> HipResult<Embeds>; }
//   text (exists), VisionEncoder (qwen35-vl), AudioEncoder (+ streaming/cache variant)

// Tier 2 — the forward, SEPARABLE from the loop. A stack of (Mixer, Ffn).
trait CoreModel {
    fn embed(&mut self, gpu, tokens, adapters) -> HipResult<Hidden>;
    fn forward(&mut self, gpu, hidden, pos, state) -> HipResult<Hidden>; // the layer stack
    fn logits(&mut self, gpu, hidden) -> HipResult<&GpuTensor>;
}
enum Mixer { FullAttn, Swa{window}, Mamba2, ShortConv, DeltaNet } // + per-layer RoPE θ, qk-norm
enum Ffn   { DenseSwiGLU, DenseGeGLU, Moe(MoeSpec) }

// Tier 3 — object-safe daemon handle; one strategy among several.
trait ServingBackend: Send {
    fn caps(&self) -> ArchCaps;            // dflash/mtp/vision/audio/streaming/...
    fn serve(&mut self, gpu, ctx: &mut ServeCtx) -> ServeResult; // batch OR streaming
    fn reset_session(&mut self, gpu, id: &str) -> Result<(),String>;
    fn unload(self: Box<Self>, gpu);
}
//   ArDecode (uses SimpleAr-style prefill/decode_step over a CoreModel),
//   Pool, Score, BlockDiffusion, SpeechDecode+Codec.
```

`ServeCtx` generalizes `GenerateCtx`: input is a prompt **or** a frame stream;
output is N tokens **or** a latency-SLA stream. `LoadedModel { arch_id,
backend: Box<dyn ServingBackend> }`.

## Roadmap (dependency-ordered epochs)

Principle: validate the seam against diversity **early**, build shared infra on
**working** examples (generalize on the 3rd+ instance), respect dependencies,
defer the heavy audio/omni/streaming until the text+seam foundation is proven.

- **E0 — gemma3 ingest/config/loader** ✅ (Phase 0/1.1/1.2 landed).
- **E1 — gemma3 decodes.** Forward (embed √scale, Q pre-scale, dual-θ, 4-norm
  residual, QK-norm, GQA) + fused `gelu_mul` + `ArDecode`/`SimpleAr` impl. First
  working new family + first real `ServingBackend`. *(next GPU block)*
- **E2 — the seam.** Refine P0 into the three-tier abstraction:
  `CoreModel` (forward separable from loop), `ServingBackend.serve` (batch AR
  first; `ServeCtx` stream-aware but only batch implemented), route **qwen2 +
  gemma3** through it; begin the `LoadedModel` Option-soup collapse for those
  two. Lands the design truths in code.
- **E3 — shared building blocks** (cleanup, now justified by 4 working text
  archs): extract `hipfire_runtime::transformer` shared loader (qwen35/qwen2/
  dots-ocr/gemma3), the `Mixer`/`Ffn` layer-stack, and one shared MoE FFN.
- **E4 — Mamba2 (pure).** SSM + conv1d kernels in isolation (no attn/MoE
  confounds); validates `Mixer::Mamba2` and a **no-KV** `CoreModel`/`ArDecode`.
- **E5 — compose transformers+SSM.** gemma4 (gemma3 + shared MoE + `layer_types`
  — cheap) → nemotron_h (Mamba2 + attn + MoE hybrid stack — the big composition).
- **E6 — non-AR output strategies.** Embedding/rerank `Pool`/`Score` heads
  (cheap; qwen3 forward exists) validate non-AR *output*; diffusion_gemma
  `BlockDiffusion` over the gemma4 forward validates non-AR *generation*.
- **E7 — input adapters / multimodal-in.** Generalize the vision encoder
  (gemma-mm, sensenova MoT, llama4-vision); llama4 text+vision.
- **E8 — audio + streaming (long horizon).** `.nemo`/onnx → HFQ converter
  front-ends; `AudioEncoder` (Conformer/Whisper, streaming/cache); STT output;
  then TTS (codec/vocoder); then omni orchestration (thinker+talker+codec) on the
  **Streaming** serving mode + a latency/RTF gate in the perf methodology.
- **E9 — image-gen (optional, separate engine).** Qwen-Image MMDiT+VAE.
- **Anytime:** exotic text archs (hrm_text, zaya) slot in once the shared
  building blocks exist; reuse-only families (llama3.x/minicpm5/qwen2.x) are
  loader/config tasks done on demand.

## Immediate next step

Begin **E1 — the gemma3 forward** (GPU-iteration block): `forward_step`, the
fused `gelu_mul_f32` kernel, and the `Architecture` + `ArDecode`/`SimpleAr`
impls, then decode medgemma-27b-text-it and pass coherence. That delivers the
first working new family and the first concrete `ServingBackend`, which E2
generalizes.

## Why this order is safe

Each epoch builds one new capability in its cheapest isolating context and
composes upward: gemma3 proves the dense-AR path; E2 proves the seam on two
archs; E4 proves the SSM mixer alone before E5 composes it; E6 proves both
non-AR axes; E8 proves streaming last, when the foundation can't wobble under
it. Shared infra (E3) is extracted only after ≥3 working instances exist.
