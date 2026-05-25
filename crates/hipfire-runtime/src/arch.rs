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
use rdna_compute::Gpu;

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
/// The runtime's loop guard (`hipfire_runtime::loop_guard`) detects
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
/// `hipfire_runtime::sampler` owns top-p / top-k / temperature / repeat-
/// penalty / blocked-token mechanics. Per-arch overrides add to (don't
/// replace) the runtime config.
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
/// `hipfire_runtime::prompt_frame` owns the `<|im_start|>` / `<|im_end|>`
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
/// `hipfire_runtime::eos_filter` owns visible-stream EOS detection.
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
