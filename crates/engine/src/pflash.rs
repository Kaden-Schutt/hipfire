//! PFlash: speculative prefill compression for long-context inputs.
//!
//! Top-of-prefill compression stage. Runs a small drafter model over the
//! source prompt, scores attention importance per source block, keeps the
//! highest-scoring spans plus mandatory anchors (sink + recent + chat
//! boundaries), and hands the compressed token stream to the target's
//! existing prefill path. Decode (DFlash / DDTree / AR) is unchanged.
//!
//! See `docs/plans/pflash-speculative-prefill.prd` for design rationale.
//!
//! Phase 1.0 status: scaffolding only. `maybe_compress_prompt` always
//! returns `Bypass` regardless of mode. Drafter loading + scoring +
//! selection land in subsequent phases.

use crate::hfq::{self, HfqFile};
use crate::llama::{self, ForwardScratch, KvCache, LlamaConfig, LlamaWeights};
use crate::tokenizer::Tokenizer;
use hip_bridge::HipResult;
use rdna_compute::Gpu;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PflashMode {
    /// Disabled. `maybe_compress_prompt` always returns `Bypass`.
    Off,
    /// Compress only when the source token count exceeds `threshold_tokens`.
    Auto,
    /// Always attempt compression; useful for benchmarking / research.
    Always,
}

impl PflashMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "off" | "0" | "false" | "no" => Some(PflashMode::Off),
            "auto" | "1" | "true" | "yes" => Some(PflashMode::Auto),
            "always" | "2" | "force" => Some(PflashMode::Always),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            PflashMode::Off => "off",
            PflashMode::Auto => "auto",
            PflashMode::Always => "always",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PflashConfig {
    pub mode: PflashMode,
    pub threshold_tokens: usize,
    pub keep_ratio: f32,
    pub alpha: f32,
    pub min_keep_tokens: usize,
    pub sink_tokens: usize,
    pub recent_tokens: usize,
    pub block_size: usize,
    pub profile: bool,
    pub drafter_path: Option<String>,
}

impl Default for PflashConfig {
    fn default() -> Self {
        Self {
            mode: PflashMode::Off,
            threshold_tokens: 32768,
            keep_ratio: 0.05,
            alpha: 0.85,
            min_keep_tokens: 2048,
            sink_tokens: 256,
            recent_tokens: 1024,
            block_size: 128,
            profile: false,
            drafter_path: None,
        }
    }
}

impl PflashConfig {
    /// Hydrate config from `HIPFIRE_PREFILL_*` env vars. Any missing var
    /// falls back to the default. Invalid values panic with a clear
    /// message rather than silently degrading.
    pub fn from_env() -> Self {
        let mut cfg = PflashConfig::default();
        if let Ok(v) = std::env::var("HIPFIRE_PREFILL_COMPRESSION") {
            cfg.mode = PflashMode::parse(&v)
                .unwrap_or_else(|| panic!("HIPFIRE_PREFILL_COMPRESSION={v} not in {{off,auto,always}}"));
        }
        if let Ok(v) = std::env::var("HIPFIRE_PREFILL_THRESHOLD") {
            cfg.threshold_tokens = v.parse()
                .unwrap_or_else(|_| panic!("HIPFIRE_PREFILL_THRESHOLD={v} not a usize"));
        }
        if let Ok(v) = std::env::var("HIPFIRE_PREFILL_KEEP_RATIO") {
            cfg.keep_ratio = v.parse()
                .unwrap_or_else(|_| panic!("HIPFIRE_PREFILL_KEEP_RATIO={v} not f32"));
            assert!(cfg.keep_ratio > 0.0 && cfg.keep_ratio <= 1.0,
                "HIPFIRE_PREFILL_KEEP_RATIO must be in (0, 1], got {}", cfg.keep_ratio);
        }
        if let Ok(v) = std::env::var("HIPFIRE_PREFILL_ALPHA") {
            cfg.alpha = v.parse()
                .unwrap_or_else(|_| panic!("HIPFIRE_PREFILL_ALPHA={v} not f32"));
        }
        if let Ok(v) = std::env::var("HIPFIRE_PREFILL_MIN_KEEP") {
            cfg.min_keep_tokens = v.parse()
                .unwrap_or_else(|_| panic!("HIPFIRE_PREFILL_MIN_KEEP={v} not usize"));
        }
        if let Ok(v) = std::env::var("HIPFIRE_PREFILL_SINK") {
            cfg.sink_tokens = v.parse()
                .unwrap_or_else(|_| panic!("HIPFIRE_PREFILL_SINK={v} not usize"));
        }
        if let Ok(v) = std::env::var("HIPFIRE_PREFILL_RECENT") {
            cfg.recent_tokens = v.parse()
                .unwrap_or_else(|_| panic!("HIPFIRE_PREFILL_RECENT={v} not usize"));
        }
        if let Ok(v) = std::env::var("HIPFIRE_PREFILL_BLOCK") {
            cfg.block_size = v.parse()
                .unwrap_or_else(|_| panic!("HIPFIRE_PREFILL_BLOCK={v} not usize"));
        }
        if std::env::var("HIPFIRE_PREFILL_PROFILE").ok().as_deref() == Some("1") {
            cfg.profile = true;
        }
        if let Ok(v) = std::env::var("HIPFIRE_PREFILL_DRAFTER") {
            cfg.drafter_path = Some(v);
        }
        cfg
    }
}

/// Carry-over state across requests: drafter model + tokenizer + scratch.
///
/// Drafter loading is opt-in via `load_drafter`. While `drafter_loaded == false`
/// the GPU-bearing fields are `None`, so this struct stays cheap to construct
/// even when PFlash is disabled. Tokenizer-compat checking against the target
/// happens at load time and any mismatch surfaces as `BypassReason::TokenizerMismatch`.
pub struct PflashState {
    pub drafter_path: Option<String>,
    pub drafter_loaded: bool,
    pub drafter_config: Option<LlamaConfig>,
    pub drafter_weights: Option<LlamaWeights>,
    pub drafter_tokenizer: Option<Tokenizer>,
    pub drafter_scratch: Option<ForwardScratch>,
    pub drafter_kv: Option<KvCache>,
    /// True only if `drafter_tokenizer.vocab_size() == target_tokenizer.vocab_size()`
    /// AND a fixed probe phrase round-trips identically through both. Set by
    /// `load_drafter`; if false at request time, `decide_bypass` returns
    /// `BypassReason::TokenizerMismatch`.
    pub tokenizer_compat: bool,
}

impl PflashState {
    pub fn new(cfg: &PflashConfig) -> Self {
        Self {
            drafter_path: cfg.drafter_path.clone(),
            drafter_loaded: false,
            drafter_config: None,
            drafter_weights: None,
            drafter_tokenizer: None,
            drafter_scratch: None,
            drafter_kv: None,
            tokenizer_compat: false,
        }
    }

    /// Drop drafter GPU resources back to the pool. Idempotent.
    pub fn unload_drafter(&mut self, gpu: &mut Gpu) {
        if let Some(w) = self.drafter_weights.take() {
            w.free_gpu(gpu);
        }
        if let Some(s) = self.drafter_scratch.take() {
            s.free_gpu(gpu);
        }
        // KvCache has its own buffers; let drop handle them.
        self.drafter_kv = None;
        self.drafter_config = None;
        self.drafter_tokenizer = None;
        self.drafter_loaded = false;
        self.tokenizer_compat = false;
    }
}

/// Probe phrase used to verify drafter and target BPE merges agree. Picked to
/// hit common BPE seams (whitespace, mixed case, punctuation, code-shape
/// tokens, a multi-byte glyph). If both tokenizers produce the same id sequence
/// then they share a vocab and merges and are interchangeable for compression.
const TOKENIZER_COMPAT_PROBE: &str = "Hello, world! 0xCAFEf00d def fn() {} \u{2014}";

/// Compare drafter vs target tokenizers for compression compatibility.
/// Returns `true` only if both tokenizers have the same `vocab_size` AND
/// produce the same encoding for `TOKENIZER_COMPAT_PROBE`.
pub fn tokenizers_compatible(target: &Tokenizer, draft: &Tokenizer) -> bool {
    if target.vocab_size() != draft.vocab_size() {
        return false;
    }
    let a = target.encode(TOKENIZER_COMPAT_PROBE);
    let b = draft.encode(TOKENIZER_COMPAT_PROBE);
    a == b
}

/// Load a Qwen3-family drafter from `path` (HFQ artifact) onto `gpu` and
/// stash it inside `state`. Verifies tokenizer compatibility against
/// `target_tokenizer`; mismatch is surfaced via `tokenizer_compat = false`
/// rather than a hard error so the caller can still bypass cleanly.
///
/// Allocates a small KV cache sized for `max_kv_seq` tokens (the drafter
/// itself never sees more than the source prompt length, but the cache must
/// be large enough for the longest context the daemon will ever score).
///
/// Bumps `state.drafter_loaded = true` only when:
///   - HFQ opens cleanly,
///   - LlamaConfig parses,
///   - tokenizer parses,
///   - weights load,
///   - tokenizer_compat passes (otherwise loaded=true but compat=false; the
///     caller sees BypassReason::TokenizerMismatch downstream).
pub fn load_drafter(
    state: &mut PflashState,
    gpu: &mut Gpu,
    path: &Path,
    target_tokenizer: &Tokenizer,
    max_kv_seq: usize,
) -> HipResult<()> {
    let hfq = HfqFile::open(path).map_err(|e| hip_bridge::HipError::new(0, &format!(
        "pflash: open drafter HFQ at {}: {e}", path.display(),
    )))?;
    let config = hfq::config_from_hfq(&hfq).ok_or_else(|| hip_bridge::HipError::new(0,
        "pflash: drafter HFQ has no recoverable LlamaConfig (model_type missing or unsupported)",
    ))?;
    let drafter_tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).ok_or_else(||
        hip_bridge::HipError::new(0, "pflash: drafter HFQ has no embedded tokenizer metadata")
    )?;
    let weights = hfq::load_weights_hfq(&hfq, &config, gpu)?;
    let scratch = ForwardScratch::new(gpu, &config)?;
    // Default to Q8 KV — minimal-risk format, batched-eligible, supported
    // across all targeted RDNA archs. Future iterations can pick asym3
    // when scoring quality at long context demands the K-rotation.
    let kv = KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_kv_seq)?;

    let compat = tokenizers_compatible(target_tokenizer, &drafter_tokenizer);

    state.drafter_path = Some(path.display().to_string());
    state.drafter_config = Some(config);
    state.drafter_weights = Some(weights);
    state.drafter_tokenizer = Some(drafter_tokenizer);
    state.drafter_scratch = Some(scratch);
    state.drafter_kv = Some(kv);
    state.tokenizer_compat = compat;
    state.drafter_loaded = true;
    Ok(())
}

/// Approximate VRAM cost of a drafter load *before* committing to it.
/// Returns bytes of all GPU buffers a `load_drafter` call would touch
/// (weights + scratch + KV cache). Useful for the daemon's parking
/// decision in Phase 4.
pub fn drafter_vram_estimate_bytes(config: &LlamaConfig, max_kv_seq: usize) -> usize {
    // Weights: rough HFQ4G256 = 0.5 bytes/element + ~32 bytes/group overhead.
    // Approximate as 0.6 bytes/element for the dense Qwen3 portion.
    let n_params = {
        let dim = config.dim;
        let hd = config.hidden_dim;
        let kvd = config.n_kv_heads * config.head_dim;
        let qd = config.n_heads * config.head_dim;
        let per_layer = dim * (qd + kvd + kvd) + qd * dim + dim * (hd + hd) + hd * dim;
        per_layer * config.n_layers + 2 * config.vocab_size * dim
    };
    let weights_bytes = (n_params * 6) / 10;
    // Scratch: a few [dim] + [hidden_dim] buffers plus partials, FP32. Bound
    // by max(dim, hidden_dim) * 32.
    let scratch_bytes = std::cmp::max(config.dim, config.hidden_dim) * 4 * 32;
    // Q8 KV cache: 136 bytes per 128-element head (Q8 block stride).
    let kv_bytes_per_pos = config.n_kv_heads * 136;
    let kv_bytes = max_kv_seq * kv_bytes_per_pos * 2;
    weights_bytes + scratch_bytes + kv_bytes
}

/// Why a request bypassed compression. Logged so operators can
/// distinguish "below threshold" from "tool call" from "tokenizer mismatch".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BypassReason {
    /// Compression mode is `Off`.
    ModeOff,
    /// Source token count is below `threshold_tokens` and mode is `Auto`.
    BelowThreshold { source_tokens: usize, threshold: usize },
    /// Vision request — image-bearing prompts always bypass for now.
    VisionRequest,
    /// Tool-calling request or prompt with structured JSON tool definitions.
    ToolCallRequest,
    /// Drafter not loaded; nothing to score with.
    DrafterUnavailable,
    /// Drafter and target tokenizers do not match.
    TokenizerMismatch,
    /// Architecture / KV / model shape unsupported by the current drafter.
    UnsupportedDrafter { reason: String },
}

impl BypassReason {
    pub fn as_str(&self) -> String {
        match self {
            BypassReason::ModeOff => "mode_off".to_string(),
            BypassReason::BelowThreshold { source_tokens, threshold } =>
                format!("below_threshold ({source_tokens} < {threshold})"),
            BypassReason::VisionRequest => "vision_request".to_string(),
            BypassReason::ToolCallRequest => "tool_call_request".to_string(),
            BypassReason::DrafterUnavailable => "drafter_unavailable".to_string(),
            BypassReason::TokenizerMismatch => "tokenizer_mismatch".to_string(),
            BypassReason::UnsupportedDrafter { reason } =>
                format!("unsupported_drafter: {reason}"),
        }
    }
}

/// Hint about what kind of request this is, for bypass decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    /// Plain text generation.
    Text,
    /// Vision / multimodal request.
    Vision,
    /// Tool-calling request with schema definitions.
    ToolCall,
}

/// Per-stage timings for compression. All wall-clock ms.
#[derive(Debug, Default, Clone)]
pub struct PflashTimings {
    pub drafter_prefill_ms: u128,
    pub score_ms: u128,
    pub select_ms: u128,
    pub gather_ms: u128,
    pub total_ms: u128,
}

#[derive(Debug, Clone)]
pub struct CompressedPrompt {
    pub source_tokens: usize,
    pub kept_tokens: usize,
    pub token_ids: Vec<u32>,
    pub kept_spans: Vec<(usize, usize)>,
    pub source_md5: String,
    pub compressed_md5: String,
    pub timings: PflashTimings,
}

#[derive(Debug)]
pub enum PflashDecision {
    Bypass { reason: BypassReason },
    Compressed(CompressedPrompt),
}

/// Pure-CPU bypass decision. Returns `Some(reason)` to bypass compression,
/// `None` to proceed to drafter scoring. Split from `maybe_compress_prompt`
/// so tests can exercise gating logic without faking a `Gpu`.
pub fn decide_bypass(
    state: &PflashState,
    cfg: &PflashConfig,
    token_ids: &[u32],
    request_kind: RequestKind,
) -> Option<BypassReason> {
    if cfg.mode == PflashMode::Off {
        return Some(BypassReason::ModeOff);
    }
    if request_kind == RequestKind::Vision {
        return Some(BypassReason::VisionRequest);
    }
    if request_kind == RequestKind::ToolCall {
        return Some(BypassReason::ToolCallRequest);
    }
    if cfg.mode == PflashMode::Auto && token_ids.len() < cfg.threshold_tokens {
        return Some(BypassReason::BelowThreshold {
            source_tokens: token_ids.len(),
            threshold: cfg.threshold_tokens,
        });
    }
    if !state.drafter_loaded {
        return Some(BypassReason::DrafterUnavailable);
    }
    if !state.tokenizer_compat {
        return Some(BypassReason::TokenizerMismatch);
    }
    None
}

/// Top-level compression entry point. Decides bypass vs compress and
/// dispatches accordingly.
///
/// Phase 1.0: only the bypass paths are wired. When `decide_bypass` returns
/// `None` (mode==Always or Auto-over-threshold + drafter loaded), we fall
/// through to a placeholder that returns `UnsupportedDrafter` so callers
/// log honestly. Drafter scoring lands in Phase 1.1+.
pub fn maybe_compress_prompt(
    _gpu: &mut rdna_compute::Gpu,
    state: &mut PflashState,
    cfg: &PflashConfig,
    token_ids: &[u32],
    request_kind: RequestKind,
) -> HipResult<PflashDecision> {
    if let Some(reason) = decide_bypass(state, cfg, token_ids, request_kind) {
        return Ok(PflashDecision::Bypass { reason });
    }
    // Phase 1.0: scoring not yet implemented.
    Ok(PflashDecision::Bypass {
        reason: BypassReason::UnsupportedDrafter {
            reason: "drafter scoring not yet implemented (Phase 1.1+)".to_string(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_auto(threshold: usize) -> PflashConfig {
        PflashConfig {
            mode: PflashMode::Auto,
            threshold_tokens: threshold,
            ..Default::default()
        }
    }

    #[test]
    fn mode_parses_known_strings() {
        assert_eq!(PflashMode::parse("off"), Some(PflashMode::Off));
        assert_eq!(PflashMode::parse("AUTO"), Some(PflashMode::Auto));
        assert_eq!(PflashMode::parse("always"), Some(PflashMode::Always));
        assert_eq!(PflashMode::parse("force"), Some(PflashMode::Always));
        assert_eq!(PflashMode::parse("garbage"), None);
    }

    #[test]
    fn bypass_when_off() {
        let cfg = PflashConfig { mode: PflashMode::Off, ..Default::default() };
        let state = PflashState::new(&cfg);
        let tokens = vec![1u32; 50_000];
        let r = decide_bypass(&state, &cfg, &tokens, RequestKind::Text);
        assert_eq!(r, Some(BypassReason::ModeOff));
    }

    #[test]
    fn bypass_below_threshold_in_auto() {
        let cfg = cfg_auto(32_768);
        let state = PflashState::new(&cfg);
        let tokens = vec![1u32; 8_000];
        let r = decide_bypass(&state, &cfg, &tokens, RequestKind::Text);
        assert_eq!(r, Some(BypassReason::BelowThreshold {
            source_tokens: 8_000, threshold: 32_768,
        }));
    }

    #[test]
    fn bypass_vision_and_tool_call() {
        let cfg = PflashConfig { mode: PflashMode::Always, ..Default::default() };
        let state = PflashState::new(&cfg);
        let tokens = vec![1u32; 100_000];
        let r1 = decide_bypass(&state, &cfg, &tokens, RequestKind::Vision);
        let r2 = decide_bypass(&state, &cfg, &tokens, RequestKind::ToolCall);
        assert_eq!(r1, Some(BypassReason::VisionRequest));
        assert_eq!(r2, Some(BypassReason::ToolCallRequest));
    }

    #[test]
    fn bypass_when_drafter_unavailable_at_threshold() {
        let cfg = cfg_auto(1_000);
        let state = PflashState::new(&cfg);
        assert!(!state.drafter_loaded);
        let tokens = vec![1u32; 5_000];
        let r = decide_bypass(&state, &cfg, &tokens, RequestKind::Text);
        assert_eq!(r, Some(BypassReason::DrafterUnavailable));
    }

    fn synthetic_loaded(compat: bool) -> PflashState {
        PflashState {
            drafter_path: Some("synthetic".into()),
            drafter_loaded: true,
            drafter_config: None,
            drafter_weights: None,
            drafter_tokenizer: None,
            drafter_scratch: None,
            drafter_kv: None,
            tokenizer_compat: compat,
        }
    }

    #[test]
    fn no_bypass_when_always_with_loaded_drafter_over_threshold() {
        let cfg = PflashConfig { mode: PflashMode::Always, ..Default::default() };
        let state = synthetic_loaded(true);
        let tokens = vec![1u32; 100];
        let r = decide_bypass(&state, &cfg, &tokens, RequestKind::Text);
        assert_eq!(r, None, "always mode + drafter loaded + compat must reach scoring");
    }

    #[test]
    fn bypass_on_tokenizer_mismatch() {
        let cfg = PflashConfig { mode: PflashMode::Always, ..Default::default() };
        let state = synthetic_loaded(false);
        let tokens = vec![1u32; 100];
        let r = decide_bypass(&state, &cfg, &tokens, RequestKind::Text);
        assert_eq!(r, Some(BypassReason::TokenizerMismatch));
    }
}
