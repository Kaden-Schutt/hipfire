// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The daemon's in-memory model representation and its satellites.
//!
//! `LoadedModel` is the per-loaded-model state the request loop drives — an
//! arch-tagged bundle of the typed config/weights/state for whichever family is
//! resident (the Option-soup the E2 `ServingBackend` seam will eventually
//! collapse). Extracted verbatim from the former `main.rs` monolith (no behavior
//! change); fields are `pub` so the load/generate/memory/session modules
//! that still live in `main.rs` (and sibling modules) can construct and read it.

use hip_bridge::HipResult;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_gemma3::Gemma3Backend;
use hipfire_arch_gemma3_vl::Gemma3VlBackend;
#[cfg(feature = "arch-lfm2moe")]
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::{DeltaNetState, Qwen35ScratchSet};
use hipfire_arch_qwen35::speculative::{
    DdtreeScratch, DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer, VerifyScratch,
};
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_prompt as prompt_frame;
use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::kv;
use hipfire_runtime::llama;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::sequence_state::SequenceState;
use hipfire_runtime::triattn::EvictionCtx;
use hipfire_state::ModelArtifactMemory;

#[cfg(feature = "arch-lfm2moe")]
use crate::session::Lfm2RequestSessionState;
use crate::session::Qwen35RequestSessionState;

/// CASK/TriAttention params forwarded by the CLI at load time. Zero-initialized
/// CaskConfig{sidecar: None, ..} means no eviction — matches 0.1.7-alpha behavior.
#[derive(Default)]
pub struct CaskConfig {
    pub sidecar: Option<String>,
    /// true = CASK m-folding; false = plain TriAttention drop-eviction.
    pub cask_m_folding: bool,
    pub budget: usize,
    pub beta: usize,
    pub core_frac: f32,
    pub fold_m: usize,
}

/// Eviction policy wrapper — dispatches to plain TriAttention or CASK m-folding.
pub enum Eviction {
    Plain(EvictionCtx),
    Cask(CaskCtx),
}

impl Eviction {
    /// Run one eviction pass when the physical KV occupancy warrants it,
    /// dispatching to the active policy; `Ok(None)` when nothing was evicted.
    pub fn maybe_evict(
        &self,
        gpu: &mut rdna_compute::Gpu,
        kv: &mut kv::KvCache,
        physical: usize,
    ) -> HipResult<Option<hipfire_runtime::triattn::EvictionResult>> {
        match self {
            Eviction::Plain(c) => c.maybe_evict(gpu, kv, physical),
            Eviction::Cask(c) => c.maybe_evict(gpu, kv, physical),
        }
    }
    /// Protected-core budget (slots kept before eviction begins).
    pub fn budget(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.budget,
            Eviction::Cask(c) => c.base.budget,
        }
    }
    /// Recency window (`beta`) preserved alongside the budget.
    pub fn beta(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.beta,
            Eviction::Cask(c) => c.base.beta,
        }
    }
    /// Release the policy's GPU-side scratch on unload.
    pub fn free_gpu(self, gpu: &mut rdna_compute::Gpu) {
        match self {
            Eviction::Plain(c) => c.free_gpu(gpu),
            Eviction::Cask(c) => c.free_gpu(gpu),
        }
    }
}

/// Optional DFlash speculative-decoding state. Populated when `load` supplies
/// a matching draft (.hfq arch=20) via `params.draft`. Used by the daemon's
/// `generate` fast path when temperature == 0 — falls back to AR sampling
/// otherwise (DFlash is greedy-only in this integration).
pub struct DflashState {
    pub draft_config: DflashConfig,
    pub draft_weights: DflashWeights,
    pub draft_scratch: DflashScratch,
    pub hidden_rb: HiddenStateRingBuffer,
    pub verify_scratch: VerifyScratch,
    pub target_snap: DeltaNetSnapshot,
    pub gdn_tape: GdnTape,
    /// CPU-side ring of target hidden states (num_extract × hidden per pos)
    /// — seeded from the prompt, extended by each verify's accepted rows.
    /// Drives the draft's diffusion forward.
    pub target_hidden_host: Vec<f32>,
    /// Max ctx the draft was initialized for (ring buffer cap).
    pub ctx_capacity: usize,
    /// Block size the draft was trained at.
    pub block_size: usize,
    /// Optional DDTree state. Populated only when `HIPFIRE_DDTREE_BUDGET` is
    /// set to a positive integer at daemon startup. None = DDTree disabled,
    /// the decode loop falls through to `spec_step_dflash` (chain mode).
    /// See `spec_step_ddtree_batched` for the tree-verify path.
    pub ddtree: Option<DdtreeState>,
}

/// Optional LFM2 DFlash speculative-decoding state. LFM2 has no DeltaNet
/// recurrent state, so it carries the generic DFlash draft plus an arch-local
/// target snapshot and host hidden-history rows.
#[cfg(feature = "arch-lfm2moe")]
pub struct Lfm2DflashState {
    pub draft_config: DflashConfig,
    pub draft_weights: DflashWeights,
    pub draft_scratch: DflashScratch,
    pub target_snap: lfm2moe::Lfm2DflashTargetSnapshot,
    pub target_hidden_host: Vec<f32>,
    pub ctx_capacity: usize,
    pub block_size: usize,
}

/// Side state for DDTree-mode speculative decoding. Allocated alongside
/// the rest of `DflashState` at model-load time when DDTree is enabled,
/// reused across all decode cycles.
pub struct DdtreeState {
    /// Second DeltaNetSnapshot used by `spec_step_ddtree_batched`: snap0 =
    /// pre-seed (lives in `DflashState::target_snap`), snap1 = post-seed.
    /// The batched verify forward uses both to bracket the tree-verify pass.
    pub post_seed_snap: DeltaNetSnapshot,
    /// Persistent tree-verify scratch (attn_bias, parent_indices, kv-gather
    /// staging, pre-RoPE K capture). Sized for `budget` non-root nodes.
    pub scratch: DdtreeScratch,
    /// Maximum non-root tree nodes per cycle. Read once at daemon startup
    /// from `HIPFIRE_DDTREE_BUDGET` (positive integer required to enable).
    pub budget: usize,
    /// Per-position top-K width fed into the DDTree builder. Read from
    /// `HIPFIRE_DDTREE_TOPK` (default 4 — matches paper Algorithm 1's
    /// typical setting on dense Qwen targets).
    pub topk: usize,
    /// Path C Phase 2 auxiliary snapshots. Used only when
    /// `HIPFIRE_DDTREE_PATH_C=phase2`. Allocated unconditionally when DDTree
    /// is enabled — DN state buffers are small (a few KB each on 27B) and
    /// avoiding the gate keeps allocation deterministic at session start.
    /// See `speculative::Phase2Snapshots` for what each snapshot holds.
    pub path_c_parent_pre_snap: DeltaNetSnapshot,
    pub path_c_main_end_snap: DeltaNetSnapshot,
}

thread_local! {
    /// Per-request raw-prompt override, parsed from the generate message's
    /// optional `"raw"` field. `None` = use the auto default (raw iff the model
    /// has no chat_template). Set at the top of the generate handler (the daemon
    /// processes generate messages synchronously on one thread), read by
    /// `effective_raw`. Reset every generate request, so no cross-request leak.
    pub static RAW_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Effective raw-prompt flag for prompt framing. An explicit request `"raw"`
/// wins; otherwise default to raw for base/completion models (no chat_template)
/// and framed for chat models (has a chat_template).
pub fn effective_raw(m: &LoadedModel) -> bool {
    RAW_OVERRIDE
        .with(|c| c.get())
        .unwrap_or(m.chat_template.is_none())
}

/// Everything resident for the currently-loaded model: an `arch_id` tag plus the
/// per-family typed config/weights/state behind `Option` fields (only the active
/// arch's are `Some`), the KV cache + eviction policy, multi-turn conversation
/// bookkeeping, and the optional DFlash/vision/chat-template side state. The
/// request loop drives whichever arch is populated. The E2 `ServingBackend` seam
/// will eventually collapse this Option-soup into one boxed backend.
pub struct LoadedModel {
    pub arch_id: u32,
    /// Pipeline-parallel degree. 1 = single-GPU (all existing fields below in
    /// use, q35_scratch populated). >1 = multi-GPU (pp_gpus + pp_scratch_set
    /// populated; q35_scratch stays None; kv_cache + dn_state still hold the
    /// per-layer-routed tensors since the struct types are the same as
    /// single-GPU). Refusal contracts in load_model_pp keep DFlash, CASK,
    /// PFlash, VL and arch_id < 5 out of this branch.
    pub pp: usize,
    /// Owned multi-GPU orchestrator when `pp > 1`. The single-GPU path
    /// continues to use the daemon's main `Gpu` directly.
    pub pp_gpus: Option<Gpus>,
    /// Per-device scratch when `pp > 1`. Replaces `q35_scratch`.
    pub pp_scratch_set: Option<Qwen35ScratchSet>,
    /// LA-layer → device map returned by `DeltaNetState::new_with_quant_multi`,
    /// kept so `unload_model` and the reset handler can route per-layer
    /// memsets to the correct device.
    pub pp_dn_la_to_device: Option<Vec<u8>>,
    // Qwen3.5 state
    pub q35_config: Option<qwen35::Qwen35Config>,
    pub q35_weights: Option<qwen35::Qwen35Weights>,
    pub q35_scratch: Option<qwen35::Qwen35Scratch>,
    /// Active session's live decode state (KV cache + DeltaNet recurrent state)
    /// as one unified container. `None` when no session is resident or the arch
    /// carries no such state. P2c Slice 3: replaces the former separate
    /// `kv_cache: Option<KvCache>` + `dn_state: Option<DeltaNetState>` fields.
    pub sequence_state: Option<SequenceState>,
    pub q35_kv_mode: Option<String>,
    pub q35_state_quant: Option<hipfire_arch_qwen35::qwen35::StateQuant>,
    pub q35_sessions: std::collections::HashMap<String, Qwen35RequestSessionState>,
    pub q35_active_session_id: Option<String>,
    pub q35_active_state_allocation_epoch: u64,
    pub q35_active_prefilled_generated_suffix_len: usize,
    // Qwen3 state
    pub llama_config: Option<llama::LlamaConfig>,
    pub llama_weights: Option<llama::LlamaWeights>,
    pub llama_scratch: Option<llama::ForwardScratch>,
    pub llama_kv: Option<kv::KvCache>,
    /// Assembled LLaMA/Qwen3 serving backend (arch_id 0/1), driven through the
    /// shared `ServingBackend::serve` seam — mirrors `qwen2_backend`/`gemma3_text`.
    /// Owns its own config/weights/scratch/KV; the separate `llama_*` fields above
    /// stay `None` on the backend path. P3.2.
    pub llama_backend: Option<hipfire_arch_llama::LlamaBackend>,
    /// Assembled Mamba-capable backend (nemotron_h arch_id 14 or pure Mamba-2
    /// arch_id 15), driven through the shared `ServingBackend::serve` seam.
    /// `NemotronModel` owns its own weights + per-block recurrent/KV state;
    /// there are no separate `nemotron_*` Option fields. N5b.
    pub nemotron_backend: Option<hipfire_arch_nemotron::model::NemotronModel>,
    // Qwen2 state (arch_id=7 — hipfire-arch-qwen2 standalone). The
    // KV cache lives inside Qwen2State, so there's no separate
    // qwen2_kv field. None on every other arch path.
    pub qwen2_config: Option<qwen2::Qwen2Config>,
    pub qwen2_weights: Option<qwen2::Qwen2Weights>,
    pub qwen2_state: Option<qwen2::Qwen2State>,
    /// Assembled Qwen2 serving backend (arch_id=7), driven through the shared
    /// `ServingBackend::serve` seam — mirrors `gemma3_text`. Owns its own
    /// config/weights/state; the separate `qwen2_*` fields above stay `None` on
    /// the arch-7 path and are retained only for dots-ocr's reuse of
    /// `qwen2_state`. P3.1.
    pub qwen2_backend: Option<hipfire_arch_qwen2::Qwen2Backend>,
    // DeepSeek V4 Flash state (arch_id=9 — hipfire-arch-deepseek4).
    // Hyper-Connections + compressed-KV indexer + tail-only RoPE + raw
    // SWA cache. KV cache lives inside DeepseekV4State; no separate
    // deepseek4_kv field. None on every other arch path.
    pub deepseek4_config: Option<deepseek4::DeepseekV4Config>,
    pub deepseek4_weights: Option<deepseek4::DeepseekV4Weights>,
    pub deepseek4_state: Option<deepseek4::DeepseekV4State>,
    /// Pre-allocated PrefillBatchScratch sized to `HIPFIRE_DEEPSEEK4_PP_BATCH`
    /// (default 64). Used by both batched prefill and the MTP spec-decode
    /// verify pass. Lazy-allocated on first arch_id=9 load — None on every
    /// other arch path.
    pub deepseek4_pbs: Option<deepseek4::forward::PrefillBatchScratch>,
    /// Cached `<｜end▁of▁sentence｜>` token id resolved at load time.
    /// Falls back to 1 (DeepSeek family default) if the tokenizer lacks
    /// the special-token entry.
    pub deepseek4_eos_tok: u32,
    /// MTP config — parsed from load-message params, read at generate time.
    /// Arch-agnostic: currently only DeepSeek V4 (arch_id=9) evaluates these,
    /// but the namespace is intentionally not deepseek4-specific.
    pub mtp_mode: String,
    /// Draft tokens per spec-decode window (1-10, default 3).
    pub mtp_k: usize,
    /// Whether MTP head weights were found at load time. Set by the sibling-
    /// file scan (e.g. `<stem>-mtp.*`) or bundled MTP detection. Used by
    /// `mtp_mode = "auto"` to decide whether to enable spec-decode.
    pub mtp_weights_present: bool,
    // MiniMax-M2 state (arch_id=10 — hipfire-arch-minimax). Mixtral-style
    // MoE: GQA + per-layer QK-norm + partial RoPE + sigmoid-bias top-k
    // routing, no shared expert. KV cache lives inside MiniMaxState; no
    // separate field. NO PrefillBatchScratch — prefill is the per-token
    // `decode_step` loop. None on every other arch path.
    pub minimax_config: Option<minimax::MiniMaxConfig>,
    pub minimax_weights: Option<minimax::MiniMaxWeights>,
    pub minimax_state: Option<minimax::MiniMaxState>,
    /// Cached EOS token id resolved at load time. Falls back to 1 if the
    /// tokenizer lacks the special-token entry.
    pub minimax_eos_tok: u32,
    // LFM2.5-8B-A1B state (arch_id=11 — hipfire-arch-lfm2moe). Hybrid:
    // double-gated LIV short-conv mixers interleaved with GQA+QK-norm
    // attention, feeding a DeepSeek-style sigmoid-bias top-4 MoE FFN (or
    // dense SwiGLU on the first num_dense_layers). KV cache + conv-state
    // cache both live inside Lfm2MoeState; no separate field. NO
    // PrefillBatchScratch — prefill is the per-token `decode_step` loop.
    // None on every other arch path. Structurally mirrors MiniMax (10).
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2moe_config: Option<lfm2moe::config::Lfm2MoeConfig>,
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2moe_weights: Option<lfm2moe::lfm2moe::Lfm2MoeWeights>,
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2moe_state: Option<lfm2moe::lfm2moe::Lfm2MoeState>,
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2_sessions: std::collections::HashMap<String, Lfm2RequestSessionState>,
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2_active_session_id: Option<String>,
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2_active_state_allocation_epoch: u64,
    /// Cached EOS token id resolved at load time. Falls back to 1 if the
    /// tokenizer lacks the special-token entry.
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2moe_eos_tok: u32,
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2_dflash: Option<Lfm2DflashState>,
    // dots.ocr state (arch_id=8 — Qwen2-VL family). The text decoder is
    // Qwen2: `dots_ocr_config.text` / `dots_ocr_weights.text` feed
    // `qwen2::forward_step*`, and the per-step decode state reuses the
    // `qwen2_state` field above. `dots_ocr_weights.vision` holds the
    // resident vision-tower weights for `dots_ocr::vision_forward`.
    pub dots_ocr_config: Option<dots_ocr::DotsOcrConfig>,
    pub dots_ocr_weights: Option<dots_ocr::DotsOcrWeights>,
    // Vision state (VL models only)
    pub vision_config: Option<qwen35_vl::VisionConfig>,
    pub vision_weights: Option<qwen35_vl::VisionWeights>,
    // Gemma3-VL (medgemma, arch_id=13). Self-contained multimodal backend
    // (gemma3 text decoder + SigLIP tower + projector + decode state) behind
    // the object-safe `ServingBackend::serve` seam — its own KV/decode state
    // lives inside, so there is no separate kv_cache/scratch field. The
    // `has_vl` gate keys off `gemma3_vl.is_some()` for arch 13 (rather than the
    // qwen35-typed `vision_config`). None on every other arch path.
    pub gemma3_vl: Option<Gemma3VlBackend>,
    // Gemma3 text (medgemma-*-text, arch_id=12). The splice-free text decoder
    // behind the same `ServingBackend::serve` seam (delegates to `run_simple_ar`);
    // its KV/decode state lives inside `Gemma3Backend`. None on every other arch.
    pub gemma3_text: Option<Gemma3Backend>,
    // Shared
    pub tokenizer: Option<hipfire_model::tokenizer::Tokenizer>,
    // Multi-turn conversation state
    //
    // `seq_pos` is the *physical* write position in the KV cache (the value
    // passed to `forward_scratch(..., pos, ...)`). With no eviction, physical
    // == absolute, so seq_pos simply grows. Under eviction, seq_pos is bounded
    // to `physical_cap`; absolute position = seq_pos + kv.compact_offset.
    pub seq_pos: usize,
    /// Advertised context window — client-facing capacity, the upper bound on
    /// absolute conversation length. Without eviction this equals
    /// `physical_cap` (the buffer size); under eviction it can be much larger.
    pub max_seq: usize,
    /// Physical KV buffer capacity, in slots. Allocators size per-layer K/V
    /// for this many tokens. Under eviction, budget+beta <= physical_cap;
    /// without eviction, physical_cap may be lower than max_seq and grows by
    /// loading a larger worker.
    pub physical_cap: usize,
    /// When Some(_), the daemon calls `maybe_evict` after every prefill-chunk
    /// and every decode-forward so the physical cache stays bounded by
    /// `physical_cap` even when `max_seq` advertises a much larger window.
    pub eviction: Option<Eviction>,
    pub conversation_tokens: Vec<u32>, // full token history for repeat penalty

    /// Per-turn token cache for V4F prefix-cache stability.
    ///
    /// Maps a stable fingerprint of an assistant message — `(role,
    /// content_text, tool_calls_canonical_json)` — to the token IDs the
    /// model ACTUALLY emitted for that turn. When the next request
    /// replays the same assistant message in its `messages` history, the
    /// V4F render loop uses these cached tokens verbatim instead of
    /// re-encoding via `render_assistant_tool_calls` + tokenizer.encode.
    ///
    /// Why this matters: BPE is not bijective. The model can emit a
    /// 2-token DSML tool_call (multi-char special tokens picked
    /// greedily); our re-encode of the same text via Jinja-style
    /// rendering may produce 67 tokens covering the same string. The
    /// resulting prompt diverges from the prior turn's KV slots at
    /// the assistant-turn boundary, capping the prefix-cache LCP at
    /// the divergence point. Caching the emitted tokens restores
    /// byte-identical replay and lets LCP extend through all prior
    /// assistant turns.
    ///
    /// Cleared on model unload (LoadedModel destruction). Bounded by
    /// the natural lifetime of a session — entries that never come
    /// back in a `messages` history will linger but never affect
    /// correctness (worst case: VRAM-free Vec<u32> memory growth on
    /// the host).
    pub asst_turn_cache: std::collections::HashMap<u64, Vec<u32>>,

    /// Lazily-built decoded-vocab cache for grammar-guided sampling.
    /// `tokenizer.decode(&[id])` for every id ∈ `0..vocab_size`. Built
    /// once on first tool-using V4F request, reused for every subsequent
    /// request on the same model. Without this cache, each generate
    /// rebuilt all ~129k entries at request entry (one tokenizer.decode
    /// allocation per id), adding tens of milliseconds of pure overhead
    /// to every tool-using turn. `None` until first build; cleared by
    /// `unload_model` via `LoadedModel` drop.
    pub decoded_vocab: Option<std::sync::Arc<Vec<String>>>,
    // Target model file path — cached so the DFlash fast path can reopen the
    // HfqFile mmap to construct a transient ModelSlot without reloading
    // weights. `HfqFile::open` is a cheap mmap operation.
    pub model_path: String,
    pub memory: ModelArtifactMemory,
    // DFlash speculative decoding state (populated when load supplied a draft).
    pub dflash: Option<DflashState>,
    // Upstream HF Jinja chat_template, extracted from the HFQ
    // `tokenizer_config.chat_template` at load time. `None` when the source
    // model didn't ship one (rare for instruct models). Only consumed when
    // `HIPFIRE_JINJA_CHAT=1` is set; otherwise the daemon's hand-rolled
    // `prompt_frame::ChatFrame::Plain` scaffolding is used as today.
    //
    // Stage 2 partial: AR generate() path only. DFlash, multi-GPU PP>1, and
    // VL paths still hit the Plain scaffold.
    pub chat_template: Option<String>,
    pub chat_template_profile: Option<prompt_frame::ChatTemplateProfile>,
}

impl LoadedModel {
    /// Active session's KV cache, if any. Replaces the former `kv_cache.as_ref()`
    /// on the unified `sequence_state`. Sites needing KV **and** DeltaNet
    /// simultaneously bind `sequence_state.as_mut()` once, then borrow its
    /// disjoint `.kv` / `.recurrent` fields.
    pub fn kv_cache(&self) -> Option<&kv::KvCache> {
        self.sequence_state.as_ref().and_then(|s| s.kv())
    }
    /// Mutable active KV cache (single-access only — see [`Self::kv_cache`]).
    pub fn kv_cache_mut(&mut self) -> Option<&mut kv::KvCache> {
        self.sequence_state.as_mut().and_then(|s| s.kv_mut())
    }
    /// Active session's DeltaNet recurrent state, if any.
    pub fn dn_state(&self) -> Option<&DeltaNetState> {
        self.sequence_state
            .as_ref()
            .and_then(|s| s.recurrent_as::<DeltaNetState>())
    }
    /// Mutable active DeltaNet recurrent state (single-access only).
    pub fn dn_state_mut(&mut self) -> Option<&mut DeltaNetState> {
        self.sequence_state
            .as_mut()
            .and_then(|s| s.recurrent_as_mut::<DeltaNetState>())
    }
}
