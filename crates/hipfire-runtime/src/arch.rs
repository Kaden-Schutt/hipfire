// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The bring-up contract for a hipfire architecture. Implement this
//! trait in your arch crate (e.g. `hipfire-arch-qwen35`) to plug a
//! model into the runtime. Generation, sampling, eviction, spec
//! decode, paging, prompt framing, and EOS filtering all live in
//! the runtime crate; the arch contributes only the model-specific
//! pieces.
//!
//! Default impls cover the Qwen3.5 family conventions. Override only
//! what diverges for your arch.
//!
//! # Worked examples
//!
//! - `crates/hipfire-arch-toy/` — minimum-viable stub, ~50 lines of
//!   trait-impl with explanatory comments. Copy-paste this directory
//!   as a starting point for a new arch.
//! - `crates/hipfire-arch-qwen35/src/arch.rs` — full production impl
//!   for the Qwen3.5 hybrid DeltaNet + MoE family. Read this for the
//!   bar: how `config_from_hfq` walks the JSON metadata, how
//!   `load_weights` drives the weight pager, how `new_state` allocates
//!   GPU scratch.
//! - `crates/hipfire-arch-llama/src/arch.rs` — second impl, dense
//!   LLaMA / Mistral / plain-Qwen3 family. Demonstrates the trait at
//!   facade-stage (forward body still in `hipfire-runtime::llama`,
//!   PR 14 will physically split).
//!
//! # Why forward isn't on the trait
//!
//! Forward-pass dispatch is intentionally NOT routed through this
//! trait. Reasons:
//!   1. Forward signatures vary heavily across arches (number of
//!      buffers, KV layout, hybrid-vs-dense paths, vision conditioning,
//!      MoE expert management). Forcing one trait shape would either
//!      bloat the contract or hide essential parameters behind opaque
//!      slots.
//!   2. Forward dispatch is hot-path. Static dispatch via concrete-type
//!      function calls keeps the call graph fully inlinable; dyn-trait
//!      dispatch in the inner loop costs measurable tok/s on small
//!      models.
//!   3. The trait's job is BRING-UP scaffolding (load → instantiate →
//!      generation-loop wiring), not runtime polymorphism. Once an arch
//!      is loaded, the daemon/CLI knows the concrete type at compile
//!      time.

use crate::hfq::HfqFile;
use rdna_compute::{Gpu, GpuTensor};

/// Bring-up contract for a hipfire architecture.
///
/// Implementors live in their own arch crate (`hipfire-arch-<name>`)
/// and provide the three required types (Config / Weights / State)
/// plus five required methods. The four optional override hooks let
/// an arch deviate from Qwen3.5 family defaults without growing a
/// per-`arch_id` `match` ladder in the daemon.
///
/// # Required: associated types
///
/// - `Config` — model-shape constants parsed from HFQ metadata.
///   Cheap to clone, sent across threads. Example: `Qwen35Config`
///   in `hipfire-arch-qwen35` carries dim, n_layers, head counts,
///   MoE topology, RoPE params.
/// - `Weights` — GPU-resident model weights. Owns `WeightTensor`
///   handles plus any host-side metadata for the weight pager.
/// - `State` — GPU-resident per-decode scratch (KV cache, attention
///   workspace, recurrent state for hybrid archs).
///
/// # Required: methods
///
/// See per-method docs below.
///
/// # Optional: override hooks
///
/// `loop_guard_overrides`, `sampler_overrides`, `prompt_frame_overrides`,
/// `eos_filter_overrides`. Default impls match Qwen3.5 conventions.
/// Override per-arch when the arch's prompt format / sampling
/// requirements / end-of-turn markers diverge.
pub trait Architecture: Send + 'static {
    type Weights;
    type State;
    type Config: Clone + Send + 'static;

    /// Canonical arch_id marker for this family. Existing IDs:
    /// 0 = LLaMA / Mistral, 1 = plain Qwen3 / Qwen2,
    /// 5 = Qwen3.5 dense, 6 = Qwen3.5/3.6 MoE.
    ///
    /// The actual id loaded at runtime is `HfqFile::arch_id` and may
    /// differ from this canonical marker for families that span
    /// multiple ids (e.g. `Llama::arch_id() == 0` but covers both 0
    /// and 1; the dense-vs-Qwen3-norm distinction is read off the HFQ
    /// metadata inside `config_from_hfq`).
    fn arch_id() -> u32;

    /// Human-readable arch tag for logs and CLI dispatch (e.g. `"qwen35"`,
    /// `"llama"`).
    fn name() -> &'static str;

    /// Parse model-shape constants out of `hfq.metadata_json`.
    ///
    /// Returns a typed `Config` or an error string. Implementations
    /// generally use `serde_json` to walk the metadata blob and branch
    /// on `hfq.arch_id` for variants within the family (e.g. dense vs
    /// MoE, with-vs-without DeltaNet).
    ///
    /// # Worked example: Qwen3.5
    ///
    /// `hipfire_arch_qwen35::qwen35::config_from_hfq` parses the
    /// metadata, branches `arch_id == 5` (dense) vs `arch_id == 6`
    /// (MoE) for expert-count fields, fills defaults for missing
    /// keys (e.g. `partial_rotary_factor`), and returns a
    /// `Qwen35Config` with the full per-layer shape.
    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String>;

    /// Load model weights from an HFQ file into GPU memory.
    ///
    /// PR 8 note: signature changed from `&mut HfqFile` (PR 7
    /// scaffold) to `&HfqFile`. The mmap-backed HfqFile is read-only
    /// at the syscall level and Qwen35::load_weights only reads
    /// tensor data. Weight-pager state mutations happen on the
    /// returned Weights object via interior mutability
    /// (`RefCell<WeightPager>`), not on the file.
    ///
    /// # Worked example: Qwen3.5
    ///
    /// `hipfire_arch_qwen35::qwen35::load_weights` walks every layer's
    /// QKV / output / FFN / norm tensors, hands each to
    /// `WeightTensor::from_hfq_tensor` (which dispatches on the
    /// HFQ quant_type to upload Q4F16G64 / F16 / F32 to GPU), and
    /// assembles per-layer `LayerWeights` arrays. The weight pager
    /// (lazy load + LRU eviction for >VRAM models) is wired through
    /// `WeightTensor` and is not arch-specific.
    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String>;

    /// Allocate per-decode GPU scratch for this arch.
    ///
    /// Returns the `State` object the daemon's generation loop holds
    /// for the lifetime of a session. Sized by `cfg`.
    ///
    /// # Worked examples
    ///
    /// - Hybrid LA + FA (`DeltaNetState::new` in
    ///   `hipfire-arch-qwen35`) — KV cache for FA layers, recurrent
    ///   state buffers for DeltaNet (LA) layers, plus shared
    ///   attention scratch.
    /// - Dense FA-only (`ForwardScratch::new` in
    ///   `hipfire-runtime::llama`) — KV cache plus attention
    ///   workspace; no recurrent state.
    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String>;

    // Forward pass shapes are arch-specific; declare the surface but
    // don't constrain types in this trait — concrete arch crates
    // expose their own typed forward methods. The runtime's generic
    // generation loop holds an `impl Architecture`-bound model and
    // uses arch crate-specific call sites.
    //
    // Future PRs may tighten the forward signatures once we see what
    // the qwen35 / qwen35-vl / llama splits actually need. For PR 7
    // the trait is intentionally minimal — just enough scaffolding for
    // a canary arch crate to implement and the runtime to type-check.

    /// Override loop-guard config for this arch. Default is None on
    /// every field, falling back to runtime/env defaults.
    ///
    /// Override when a base or instruct-tuned model legitimately
    /// emits short repeating sequences (e.g. structured output, code
    /// boilerplate) that the default n-gram threshold would falsely
    /// flag. See `LoopGuardOverrides` for fields.
    fn loop_guard_overrides(_cfg: &Self::Config) -> LoopGuardOverrides {
        LoopGuardOverrides::default()
    }

    /// Override sampler config for this arch. Default is empty on
    /// `blocked_tokens`, None on `repeat_penalty`.
    ///
    /// Override to add arch-specific blocked tokens (e.g. a special
    /// `<tool_call>` opener that the model emits in attractor loops)
    /// or to set a per-arch default `repeat_penalty`.
    fn sampler_overrides(_cfg: &Self::Config) -> SamplerOverrides {
        SamplerOverrides::default()
    }

    /// Override prompt framing for this arch. Default assumes ChatML
    /// (`<|im_start|>` / `<|im_end|>` markers).
    ///
    /// Override `raw: Some(true)` for a non-ChatML completion model.
    fn prompt_frame_overrides(_cfg: &Self::Config) -> PromptFrameOverrides {
        PromptFrameOverrides::default()
    }

    /// Override EOS handling for this arch. Default uses ChatML
    /// `<|im_end|>` plus the `<think>` strip policy from runtime.
    ///
    /// Override to add arch-specific stop sequences (e.g. Gemma's
    /// `<end_of_turn>`) and matching `holdback_prefixes` so the
    /// stream doesn't leak the marker bytes to the visible output.
    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        EosFilterOverrides::default()
    }
}

/// Per-arch overrides for the loop-guard n-gram blocker.
///
/// The generation loop guard (`hipfire_generate::loop_guard`) detects
/// repeated n-grams in the recent decode window and blocks the
/// repeating token before sampler draws it. Defaults come from env
/// (`HIPFIRE_NGRAM_THRESHOLD`, `HIPFIRE_NGRAM_WINDOW`); per-arch
/// overrides take precedence.
#[derive(Debug, Clone, Default)]
pub struct LoopGuardOverrides {
    /// If `Some`, replace the env-derived n-gram threshold (count of
    /// repeats before block fires). Lower = more aggressive blocking.
    pub ngram_threshold: Option<usize>,
    /// If `Some`, replace the env-derived window length (recent-token
    /// span the n-gram detector scans).
    pub ngram_window: Option<usize>,
}

/// Per-arch overrides for the sampler.
///
/// `hipfire-generate` owns sampler policy. `hipfire_runtime::sampler` owns
/// top-p / top-k / repeat-penalty execution mechanics. Per-arch overrides add
/// to (don't replace) the runtime config.
#[derive(Debug, Clone, Default)]
pub struct SamplerOverrides {
    /// Tokens to add to `SamplerConfig::blocked_tokens` for this arch
    /// (e.g. arch-specific `<tool_call>` opener IDs that the model
    /// emits in attractor loops). Appended to the runtime list, not
    /// replacing it.
    pub blocked_tokens: Vec<u32>,
    /// If `Some`, override the repeat penalty for this arch. Use
    /// sparingly — `1.05` is the user-validated default floor; values
    /// >1.3 cause MQ4/MQ6 gibberish at low temperature.
    pub repeat_penalty: Option<f32>,
}

/// Per-arch overrides for prompt framing.
///
/// `hipfire-prompt` owns the `<|im_start|>` / `<|im_end|>`
/// scaffolding plus `<think>` injection for thinking-mode models.
#[derive(Debug, Clone, Default)]
pub struct PromptFrameOverrides {
    /// If `Some`, override the assistant prefix scheme. `Some(true)`
    /// disables ChatML framing entirely (raw completion, no
    /// `<|im_start|>assistant`); `Some(false)` forces ChatML even if
    /// the runtime would otherwise auto-detect raw.
    pub raw: Option<bool>,
}

/// Per-arch overrides for EOS / end-of-turn filtering.
///
/// `hipfire_generate::eos_filter` owns visible-stream EOS detection.
/// The default implementation handles ChatML `<|im_end|>` plus
/// `<think>` strip; per-arch overrides extend to additional markers.
#[derive(Debug, Clone, Default)]
pub struct EosFilterOverrides {
    /// Byte sequences that signal end-of-turn for this arch. Streaming
    /// stops (and the marker is not emitted) when the decoded byte
    /// stream contains any sequence here.
    /// Examples: Gemma4's `<end_of_turn>` (when forward-ported).
    pub stop_at: Vec<Vec<u8>>,
    /// Byte prefixes the streamer holds back until disambiguated.
    /// Required so a partial decode of a `stop_at` marker doesn't leak
    /// its initial bytes (e.g. holding back `<end_` until we see
    /// either `<end_of_turn>` to stop or `<end_of_something_else>` to
    /// flush).
    pub holdback_prefixes: Vec<Vec<u8>>,
    /// If `Some`, override whether to strip `<think>...</think>` blocks
    /// from the visible stream. Default is on for thinking-mode arches.
    pub strip_think: Option<bool>,
}

// ── Serving seam (de-qwen-ification) ────────────────────────────────
// See docs/plans/2026-06-19-daemon-family-seam.md. The daemon's per-arch
// `generate_*` functions and the `LoadedModel` Option-soup are being
// migrated behind an object-safe serving trait so new families integrate
// without editing the 18k-line daemon at ~1000 sites.
//
// P0 (this): `ArchCaps` + `SimpleAr` — the stable keystone a new dense
// arch implements. The full `ServingBackend` boxed trait + `GenerateCtx`
// land in P2, once the daemon extraction pins the exact context surface.

/// Optional fast-path capabilities a serving backend advertises.
///
/// The daemon checks these instead of branching on `arch_id` (e.g.
/// `if caps.dflash { ... }` rather than `if is_qwen35_family_arch_id(id) { ... }`).
/// A backend that lacks a path leaves the flag `false` and the daemon falls
/// back to the plain autoregressive loop ([`SimpleAr`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArchCaps {
    /// DDTree / DFlash speculative decode (draft + tree verify).
    pub dflash: bool,
    /// Multi-token-prediction head (e.g. DeepSeek V4, qwen35 MTP).
    pub mtp: bool,
    /// Pipeline-parallel multi-GPU serving (`pp > 1`).
    pub pipeline_parallel: bool,
    /// Grouped-MoE batched prefill/decode fast path.
    pub grouped_moe_batch: bool,
    /// Vision conditioning (image tokens spliced into the decoder input).
    pub vision: bool,
    /// Paged KV cache (block-table allocator) rather than a flat buffer.
    pub paged_kv: bool,
}

/// Plain autoregressive serving surface.
///
/// A dense, full-attention-only family (gemma3 text, llama, plain qwen2)
/// implements **only** this; the daemon drives the shared
/// prefill → sample → stream → decode_step loop (sampler, EOS filter,
/// loop-guard, penalties all stay daemon-side and are reused by every arch).
/// Families with a bespoke loop (qwen35 DFlash/MTP, deepseek4 MTP, VL splice)
/// override the full serving entry point instead (P2's `ServingBackend`).
///
/// Object-safe by construction: methods take `&mut self`, `&mut Gpu`, slices
/// and scalars, and hand back the logits tensor by reference for the daemon's
/// sampler — no associated types, no generics, no `Self`-by-value. So the
/// daemon can hold `Box<dyn SimpleAr>`.
///
/// The dyn boundary here is **coarse** — one virtual call per prefill and per
/// decode step (i.e. per token), not per layer. The per-layer forward stays
/// monomorphized inside the implementor, so this does not reintroduce the
/// inner-loop dyn cost the [`Architecture`] docs warn about.
pub trait SimpleAr {
    /// Run the prompt through the model, populating the KV cache and leaving
    /// the final-position logits available via [`SimpleAr::logits`]. `tokens`
    /// is the full prompt (already framed/templated by the daemon).
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String>;

    /// Advance one autoregressive step: feed `token` at absolute position
    /// `pos`, updating the KV cache and refreshing [`SimpleAr::logits`].
    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String>;

    /// The most recent step's logits (`[vocab_size]`), for the daemon sampler.
    fn logits(&self) -> &GpuTensor;

    /// Vocabulary length (logits width), so the daemon sizes its sampler.
    fn vocab_size(&self) -> usize;
}

/// Why a serving run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StopReason {
    /// Hit the model's EOS / end-of-turn token.
    #[default]
    Eos,
    /// Reached `GenerateCtx::max_tokens`.
    MaxTokens,
    /// Matched a `GenerateCtx::stop_sequences` entry.
    StopSequence,
}

/// Result of a [`ServingBackend::serve`] run.
#[derive(Debug, Clone, Copy, Default)]
pub struct ServeOutcome {
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub stop_reason: StopReason,
}

/// Serving-infra parameters the daemon owns and hands to a backend's `serve`
/// loop — the decoupled, object-safe replacement for `generate()`'s 28-arg
/// surface. **Prompt framing is done daemon-side before `serve`** (chat
/// template, think-mode, tools, system prompt all resolved into `prompt`), so
/// those daemon/prompt types stay out of this ctx. Per-arch fast-path state
/// (pflash/MTP/drafter) is NOT here either — it lives in the backend, gated by
/// [`ArchCaps`]. Visible tokens stream to `sink`.
pub struct GenerateCtx<'a> {
    /// Request id (echoed in streamed events).
    pub id: &'a str,
    /// The fully-framed prompt text (post chat-template / think-mode).
    pub prompt: &'a str,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub max_think_tokens: usize,
    /// Visible-stream stop sequences (in addition to EOS).
    pub stop_sequences: &'a [String],
    /// Raw encoded image bytes for a multimodal request; the backend decodes +
    /// preprocesses (None for a text-only request). The `caps().vision` flag
    /// says whether a backend can consume it.
    pub image_bytes: Option<&'a [u8]>,
    /// Visible-token streaming sink (the daemon's JSONL stdout writer).
    pub sink: &'a mut dyn std::io::Write,
}

/// Object-safe serving handle the daemon holds per loaded model
/// (`Box<dyn ServingBackend>`), replacing the per-arch `generate_*` dispatch and
/// the `LoadedModel` Option-soup. It is **one output strategy among several**:
/// dense-AR families run the shared `run_simple_ar` loop over their [`SimpleAr`];
/// families with a bespoke loop (qwen35 DFlash/MTP, the VL splice,
/// block-diffusion) override [`ServingBackend::serve`] directly. The dyn
/// boundary is per-request, so it costs nothing in the per-token/per-layer hot
/// path.
pub trait ServingBackend: Send {
    /// Loaded `HfqFile::arch_id` (e.g. 5/6 qwen35, 12 gemma3, 13 gemma3-vl).
    fn arch_id(&self) -> u32;
    /// Optional fast-path capabilities the daemon checks instead of branching
    /// on `arch_id`.
    fn caps(&self) -> ArchCaps;
    /// The model's EOS / end-of-turn token id.
    fn eos_token(&self) -> u32;
    /// Run one full generation, streaming visible tokens to `ctx.sink`. `tok`
    /// is the daemon-owned tokenizer (kept out of the backend to avoid a
    /// self-borrow conflict when delegating to [`run_simple_ar`]).
    fn serve(
        &mut self,
        gpu: &mut Gpu,
        tok: &crate::tokenizer::Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String>;
    /// Reset multi-turn session state (e.g. the KV-cache cursor) for a session.
    fn reset_session(&mut self, gpu: &mut Gpu, session_id: &str) -> Result<(), String>;
    /// Release GPU resources (consumes the boxed backend on unload).
    fn unload(self: Box<Self>, gpu: &mut Gpu);
}

/// Shared dense-AR serving loop: tokenize the (pre-framed) prompt → prefill →
/// greedy-decode, streaming JSONL `token` events to `ctx.sink` and a final
/// `done`, stopping on EOS / `max_tokens` / a `stop_sequences` match. Every
/// `SimpleAr` backend's `ServingBackend::serve` delegates here, so the loop
/// lives in ONE place instead of per-arch `generate_*` copies.
///
/// Greedy for now (matches the bring-up examples); top-p / repeat-penalty /
/// loop-guard thread in as the daemon's sampler is migrated behind this driver.
pub fn run_simple_ar(
    gpu: &mut Gpu,
    backend: &mut dyn SimpleAr,
    tok: &crate::tokenizer::Tokenizer,
    eos: u32,
    ctx: &mut GenerateCtx,
) -> Result<ServeOutcome, String> {
    let ids = tok.encode(ctx.prompt);
    if ids.is_empty() {
        return Err("run_simple_ar: empty prompt after tokenize".to_string());
    }
    backend.prefill(gpu, &ids)?;
    decode_loop(gpu, backend, tok, eos, ctx, ids.len(), ids.len())
}

/// Shared post-prefill greedy decode loop: from the just-prefilled
/// `backend.logits()`, argmax → stream a JSONL `token` event → `decode_step`,
/// until EOS / `max_tokens` / a `stop_sequences` match, then a final `done`.
///
/// Split out of [`run_simple_ar`] so backends whose **prefill** differs from the
/// plain token-stream path — e.g. the gemma3-vl image-embedding splice, which
/// feeds projected vision rows via `forward_step_with_embed` — can run their own
/// prefill and still share this one streaming/stop loop. `start_pos` is the
/// absolute KV position after prefill; `prompt_tokens` is reported back in the
/// [`ServeOutcome`] (the two differ when image rows expand the prompt).
///
/// Greedy for now (matches the bring-up examples); top-p / repeat-penalty /
/// loop-guard thread in as the daemon's sampler is migrated behind this driver.
pub fn decode_loop(
    gpu: &mut Gpu,
    backend: &mut dyn SimpleAr,
    tok: &crate::tokenizer::Tokenizer,
    eos: u32,
    ctx: &mut GenerateCtx,
    start_pos: usize,
    prompt_tokens: usize,
) -> Result<ServeOutcome, String> {
    let vocab = backend.vocab_size();

    let mut pos = start_pos;
    let mut next = gpu
        .argmax_f32(backend.logits(), vocab)
        .map_err(|e| format!("argmax: {e:?}"))?;
    let mut generated = 0usize;
    let mut text = String::new();
    let mut stop = StopReason::MaxTokens;

    while generated < ctx.max_tokens {
        if next == eos {
            stop = StopReason::Eos;
            break;
        }
        let frag = tok.decode(&[next]);
        text.push_str(&frag);
        if ctx
            .stop_sequences
            .iter()
            .any(|s| !s.is_empty() && text.contains(s.as_str()))
        {
            stop = StopReason::StopSequence;
            break;
        }
        let ev = serde_json::json!({ "type": "token", "id": ctx.id, "text": frag });
        let _ = writeln!(ctx.sink, "{ev}");
        let _ = ctx.sink.flush();
        generated += 1;

        backend.decode_step(gpu, next, pos)?;
        pos += 1;
        next = gpu
            .argmax_f32(backend.logits(), vocab)
            .map_err(|e| format!("argmax: {e:?}"))?;
    }

    let done = serde_json::json!({ "type": "done", "id": ctx.id, "tokens": generated });
    let _ = writeln!(ctx.sink, "{done}");
    let _ = ctx.sink.flush();

    Ok(ServeOutcome {
        prompt_tokens,
        tokens_generated: generated,
        stop_reason: stop,
    })
}
