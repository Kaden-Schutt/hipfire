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

use hip_bridge::HipResult;

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
/// Phase 1.0: stub. Phase 1.1 fills in actual drafter model loading.
pub struct PflashState {
    pub drafter_path: Option<String>,
    pub drafter_loaded: bool,
}

impl PflashState {
    pub fn new(cfg: &PflashConfig) -> Self {
        Self {
            drafter_path: cfg.drafter_path.clone(),
            drafter_loaded: false,
        }
    }
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

    #[test]
    fn no_bypass_when_always_with_loaded_drafter_over_threshold() {
        // Always mode skips the threshold check; drafter loaded → fall through.
        let cfg = PflashConfig { mode: PflashMode::Always, ..Default::default() };
        let state = PflashState { drafter_path: Some("synthetic".into()), drafter_loaded: true };
        let tokens = vec![1u32; 100];
        let r = decide_bypass(&state, &cfg, &tokens, RequestKind::Text);
        assert_eq!(r, None, "always mode + drafter loaded must reach scoring");
    }
}
