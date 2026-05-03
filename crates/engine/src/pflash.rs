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
use rdna_compute::{DType, Gpu};
use std::path::Path;

#[cfg(feature = "deltanet")]
use crate::qwen35;

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
    /// Phase 3 (sparse drafter forward) threshold. Source-token counts
    /// below this drop to dense `forward_prefill_batch` regardless; at or
    /// above, the scoring path may dispatch the sparse kernel once it
    /// lands. Default 32768 per PRD §6 Phase 3 ("Fall back to dense
    /// drafter attention below a configurable threshold, initially 32K").
    /// Phase 3.1 wires the actual sparse attention kernel; until then this
    /// field is plumbing-only and the dense path is always used.
    pub sparse_threshold: usize,
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
            sparse_threshold: 32768,
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
        if let Ok(v) = std::env::var("HIPFIRE_PREFILL_SPARSE_THRESHOLD") {
            cfg.sparse_threshold = v.parse()
                .unwrap_or_else(|_| panic!("HIPFIRE_PREFILL_SPARSE_THRESHOLD={v} not usize"));
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

/// Drafter model variant. Plain Qwen3 / LLaMA-family loads via `llama::*`;
/// Qwen3.5 hybrid (DeltaNet + FullAttn + optional MoE) loads via `qwen35::*`.
/// PFlash dispatches at runtime based on which variant is held.
///
/// The matched-tokenizer story for Qwen3.5 targets requires a Qwen3.5-vocab
/// drafter (vocab=248320). qwen3-0.6b has vocab=151743 and is incompatible.
/// qwen3.5-0.8b is the matched smallest, hence the Hybrid variant.
pub enum DrafterModel {
    Plain {
        config: LlamaConfig,
        weights: LlamaWeights,
        scratch: ForwardScratch,
    },
    #[cfg(feature = "deltanet")]
    Hybrid {
        config: qwen35::Qwen35Config,
        weights: qwen35::Qwen35Weights,
        scratch: qwen35::Qwen35Scratch,
        dn_state: qwen35::DeltaNetState,
    },
}

impl DrafterModel {
    /// Common config-derived metrics needed by the K-cache layout.
    pub fn n_layers(&self) -> usize {
        match self {
            DrafterModel::Plain { config, .. } => config.n_layers,
            #[cfg(feature = "deltanet")]
            DrafterModel::Hybrid { config, .. } => config.n_layers,
        }
    }
    pub fn n_kv_heads(&self) -> usize {
        match self {
            DrafterModel::Plain { config, .. } => config.n_kv_heads,
            #[cfg(feature = "deltanet")]
            DrafterModel::Hybrid { config, .. } => config.n_kv_heads,
        }
    }
    pub fn head_dim(&self) -> usize {
        match self {
            DrafterModel::Plain { config, .. } => config.head_dim,
            #[cfg(feature = "deltanet")]
            DrafterModel::Hybrid { config, .. } => config.head_dim,
        }
    }
    pub fn variant_name(&self) -> &'static str {
        match self {
            DrafterModel::Plain { .. } => "plain",
            #[cfg(feature = "deltanet")]
            DrafterModel::Hybrid { .. } => "hybrid",
        }
    }

    /// Smallest layer index whose K cache is populated by the drafter
    /// forward (i.e. a FullAttention layer for hybrids; any layer for
    /// plain since plain is FullAttention everywhere).
    ///
    /// Used by `compute_scores_batched_gpu` as the source layer for
    /// scoring. Picking the SHALLOWEST FullAttention layer dodges the
    /// long-context RoPE-OOD NaN cascade observed on small drafters
    /// (MANUAL_REVIEW.md): deep layers accumulate NaN once positions
    /// exceed the drafter's trained window, but the first FullAttn
    /// layer's K is still finite at 21K source tokens. The shallow
    /// layer carries enough positional + content signal for the cosine
    /// scoring math the PRD specifies.
    ///
    /// Returns `None` only on the pathological case of a hybrid drafter
    /// with zero FullAttention layers, which would mean no Q8 K cache
    /// to score against; caller should refuse load there.
    pub fn score_layer_idx(&self) -> Option<usize> {
        match self {
            DrafterModel::Plain { .. } => Some(0),
            #[cfg(feature = "deltanet")]
            DrafterModel::Hybrid { config, .. } => {
                config.layer_types.iter().enumerate()
                    .find(|(_, t)| **t == qwen35::LayerType::FullAttention)
                    .map(|(i, _)| i)
            }
        }
    }
    pub fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            DrafterModel::Plain { weights, scratch, .. } => {
                weights.free_gpu(gpu);
                scratch.free_gpu(gpu);
            }
            #[cfg(feature = "deltanet")]
            DrafterModel::Hybrid { weights, scratch, dn_state, .. } => {
                weights.free_gpu(gpu);
                scratch.free_gpu(gpu);
                dn_state.free_gpu(gpu);
            }
        }
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
    pub drafter_model: Option<DrafterModel>,
    pub drafter_tokenizer: Option<Tokenizer>,
    pub drafter_kv: Option<KvCache>,
    /// True only if drafter and target tokenizers match per the
    /// `tokenizers_compatible` contract. Mismatch surfaces as
    /// `BypassReason::TokenizerMismatch` at request time.
    pub tokenizer_compat: bool,
}

impl PflashState {
    pub fn new(cfg: &PflashConfig) -> Self {
        Self {
            drafter_path: cfg.drafter_path.clone(),
            drafter_loaded: false,
            drafter_model: None,
            drafter_tokenizer: None,
            drafter_kv: None,
            tokenizer_compat: false,
        }
    }

    /// Drop drafter GPU resources back to the pool. Idempotent.
    /// Order matters: unload tensors INTO the pool, then upstream
    /// `unload_model` drains the pool to actually release VRAM. Call
    /// this BEFORE the next `unload_model`.
    pub fn unload_drafter(&mut self, gpu: &mut Gpu) {
        if let Some(m) = self.drafter_model.take() {
            m.free_gpu(gpu);
        }
        if let Some(kv) = self.drafter_kv.take() {
            kv.free_gpu(gpu);
        }
        self.drafter_tokenizer = None;
        self.drafter_loaded = false;
        self.tokenizer_compat = false;
    }
}

/// Probe phrase exercised after the structural compatibility check. Hits
/// common BPE seams (whitespace, mixed case, punctuation, code-shape tokens,
/// a multi-byte glyph) so a same-sized but merge-divergent vocab still gets
/// caught. Two tokenizers that match on signature AND produce the same
/// probe encoding can be used interchangeably for compression.
const TOKENIZER_COMPAT_PROBE: &str = "Hello, world! 0xCAFEf00d def fn() {} \u{2014}";

/// Vocab / special-token slot strings that are legitimately allowed to
/// differ between matched-tokenizer pairs in the Qwen3.5 family. Smaller
/// family members (e.g. 0.8B) ship the high-end audio/TTS reserved slots
/// as empty strings; larger members (e.g. 27B) populate them with the
/// real specials. Both sides agree those positions are unreachable from
/// normal text input, so divergence on them does not break PFlash's
/// "drafter and target encode identically" invariant.
fn is_audio_tts_padding(s: &str) -> bool {
    s.is_empty()
        || matches!(
            s,
            "<|audio_start|>"
                | "<|audio_end|>"
                | "<|audio_pad|>"
                | "<tts_pad>"
                | "<tts_text_bos>"
                | "<tts_text_eod>"
                | "<tts_text_bos_single>"
        )
}

/// Stable hash of the §5.3-relevant tokenizer state, EXCLUDING the audio/TTS
/// padding band per `is_audio_tts_padding`. Two tokenizers that pass
/// `tokenizers_compatible` produce identical `compat_signature`s; this lets
/// pretokenized fixtures travel safely across same-family members of
/// different sizes (e.g. tokens written by qwen3.5-0.8b's tokenizer can be
/// consumed by qwen3.5-27b's tokenizer because both produce the same encoding
/// for any reachable text).
///
/// To catch foreign tokenizers that happen to match on a vocab subset, the
/// hash also incorporates:
///   - `vocab_size()` — different total counts always diverge.
///   - encode(TOKENIZER_COMPAT_PROBE) — the same probe used by
///     `tokenizers_compatible`'s structural check, baked into the static
///     signature. A foreign tokenizer would have to produce the same probe
///     encoding AND the same non-padding vocab to falsely match.
///
/// Cost: O(vocab_size) plus a one-time sort of specials and one probe
/// encoding (~20 byte input). Called once per drafter load and once per
/// pretok read.
pub fn tokenizer_compat_signature(t: &Tokenizer) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    fn mix(bytes: &[u8], h: &mut u64) {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
        *h ^= 0xff;
        *h = h.wrapping_mul(0x100000001b3);
    }
    // Vocab size is part of the §5.3 contract; folding it in catches a
    // foreign tokenizer that happens to share a vocab prefix but differs in
    // total slot count.
    mix(&(t.vocab_size() as u64).to_le_bytes(), &mut h);
    // Vocab in id order. Padding slots get a stable marker rather than
    // being skipped: skipping would lose positional information and let
    // `[FOO, PAD, BAR]` hash-match `[FOO, BAR, PAD]` even though those
    // tokenizers map id=1 to different strings, so identical text would
    // tokenize differently. Mixing the marker IN PLACE preserves slot
    // alignment, while still allowing two §5.3-compatible tokenizers to
    // match when they each have padding (empty / audio / tts specials)
    // at the same slot.
    const PADDING_MARKER: &[u8] = b"\x00PFLASH_AUDIO_TTS_PAD\x00";
    for tok in t.vocab() {
        if is_audio_tts_padding(tok) {
            mix(PADDING_MARKER, &mut h);
        } else {
            mix(tok.as_bytes(), &mut h);
        }
    }
    // Specials: filter out audio/tts entries, then sort by (name, id) for
    // canonical order. The filter mirrors `tokenizers_compatible`'s exception
    // (extras on either side are allowed iff their string is in the padding
    // band), and the sort means storage-order divergence between two
    // compatible tokenizers doesn't cause a false mismatch. We mix the count
    // in too so a tokenizer with FEWER non-padding specials can't hash-match
    // one with MORE (the surplus simply wouldn't be hashed otherwise).
    let mut specials: Vec<&(String, u32)> = t
        .special_tokens()
        .iter()
        .filter(|(s, _)| !is_audio_tts_padding(s))
        .collect();
    specials.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    mix(&(specials.len() as u64).to_le_bytes(), &mut h);
    for (s, id) in specials {
        mix(s.as_bytes(), &mut h);
        mix(&id.to_le_bytes(), &mut h);
    }
    mix(&t.bos_id.to_le_bytes(), &mut h);
    mix(&t.eos_id.to_le_bytes(), &mut h);
    mix(&t.eot_id.unwrap_or(u32::MAX).to_le_bytes(), &mut h);
    // Bake the probe encoding into the signature so a structural BPE
    // divergence (different merges yielding different token sequences for
    // common text) is caught even if the static vocab/specials happen to
    // match. This mirrors the probe step in `tokenizers_compatible`.
    let probe_tokens = t.encode(TOKENIZER_COMPAT_PROBE);
    mix(&(probe_tokens.len() as u64).to_le_bytes(), &mut h);
    for tok in &probe_tokens {
        mix(&tok.to_le_bytes(), &mut h);
    }
    h
}

/// Compare drafter vs target tokenizers for compression compatibility.
///
/// PRD §5.3 contract enforced directly:
///   - same vocab size (catches Qwen3 vocab=151743 vs Qwen3.5 vocab=248144).
///   - same byte string at every vocab slot, with one documented exception:
///     slots in the audio/TTS padding band (per `is_audio_tts_padding`) are
///     allowed to be empty on one side and populated on the other. These
///     slots are unreachable from normal text encoding.
///   - same `(string, id)` set in `special_tokens()`, with the same audio/
///     TTS exception (extras on one side allowed iff their string is in
///     the documented padding band).
///   - bos / eos / eot ids equal.
///   - probe-encoding equal as a final structural sanity check on common
///     BPE seams (whitespace, punctuation, code shapes, multi-byte glyph).
///
/// The audio/TTS exception is what lets `qwen3.5-0.8b` drive `qwen3.5-27b`:
/// the family shares the BPE vocab + merges but the smaller members leave
/// the audio specials as empty padding. No other slot divergence is
/// tolerated.
pub fn tokenizers_compatible(target: &Tokenizer, draft: &Tokenizer) -> bool {
    if target.vocab_size() != draft.vocab_size() {
        return false;
    }
    if target.bos_id != draft.bos_id || target.eos_id != draft.eos_id {
        return false;
    }
    if target.eot_id != draft.eot_id {
        return false;
    }
    for (a, b) in target.vocab().iter().zip(draft.vocab().iter()) {
        if a == b {
            continue;
        }
        if is_audio_tts_padding(a) && is_audio_tts_padding(b) {
            continue;
        }
        return false;
    }
    let specials_subset_ok = |a: &[(String, u32)], b: &[(String, u32)]| -> bool {
        for (s, id) in a {
            match b.iter().find(|(s2, _)| s2 == s) {
                Some((_, id2)) if id2 == id => {}
                Some(_) => return false,
                None => {
                    if !is_audio_tts_padding(s) {
                        return false;
                    }
                }
            }
        }
        true
    };
    if !specials_subset_ok(target.special_tokens(), draft.special_tokens()) {
        return false;
    }
    if !specials_subset_ok(draft.special_tokens(), target.special_tokens()) {
        return false;
    }
    target.encode(TOKENIZER_COMPAT_PROBE) == draft.encode(TOKENIZER_COMPAT_PROBE)
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
    let drafter_tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).ok_or_else(||
        hip_bridge::HipError::new(0, "pflash: drafter HFQ has no embedded tokenizer metadata")
    )?;

    // Detect drafter family via the HFQ header's `arch_id` (set at
    // quantize time by hipfire-quantize):
    //   1 = plain Qwen3 / LLaMA-family (loads via llama::*)
    //   5 = Qwen3.5 / 3.6 dense hybrid (DeltaNet + FullAttn)
    //   6 = Qwen3.5 / 3.6 MoE-A3B hybrid
    //
    // Matched-tokenizer pairing: Qwen3.5 / 3.6 targets (vocab 248320)
    // need a Qwen3.5-vocab drafter. qwen3.5-0.8b (arch_id=5) is the
    // smallest matched option and routes through the Hybrid branch.
    // qwen3-0.6b (arch_id=1, vocab 151743) routes through Plain and is
    // suitable for plain-Qwen3 targets only.
    let is_hybrid = hfq.arch_id == 5 || hfq.arch_id == 6;
    #[cfg(feature = "deltanet")]
    {
        if is_hybrid {
            let q35_cfg = qwen35::config_from_hfq(&hfq).ok_or_else(||
                hip_bridge::HipError::new(0, "pflash: hybrid tensors detected but qwen35 config parse failed"))?;
            let weights = qwen35::load_weights(&hfq, &q35_cfg, gpu)?;
            let scratch = qwen35::Qwen35Scratch::new_with_kv_max(gpu, &q35_cfg, 128, max_kv_seq)?;
            let dn_state = qwen35::DeltaNetState::new(gpu, &q35_cfg)?;
            let kv = KvCache::new_gpu_q8(gpu, q35_cfg.n_layers, q35_cfg.n_kv_heads, q35_cfg.head_dim, max_kv_seq)?;
            let compat = tokenizers_compatible(target_tokenizer, &drafter_tokenizer);
            state.drafter_path = Some(path.display().to_string());
            state.drafter_model = Some(DrafterModel::Hybrid {
                config: q35_cfg, weights, scratch, dn_state,
            });
            state.drafter_tokenizer = Some(drafter_tokenizer);
            state.drafter_kv = Some(kv);
            state.tokenizer_compat = compat;
            state.drafter_loaded = true;
            return Ok(());
        }
    }

    let config = hfq::config_from_hfq(&hfq).ok_or_else(|| hip_bridge::HipError::new(0,
        "pflash: drafter HFQ has no recoverable config (neither qwen35 hybrid nor plain LlamaConfig)",
    ))?;
    let weights = hfq::load_weights_hfq(&hfq, &config, gpu)?;
    let scratch = ForwardScratch::new(gpu, &config)?;
    let kv = KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_kv_seq)?;
    let compat = tokenizers_compatible(target_tokenizer, &drafter_tokenizer);
    state.drafter_path = Some(path.display().to_string());
    state.drafter_model = Some(DrafterModel::Plain { config, weights, scratch });
    state.drafter_tokenizer = Some(drafter_tokenizer);
    state.drafter_kv = Some(kv);
    state.tokenizer_compat = compat;
    state.drafter_loaded = true;
    Ok(())
}

/// Per-block scoring output. `scores[b]` is the importance score for source
/// block `b`; higher means "more relevant" by the drafter's last-layer
/// K-similarity heuristic. `block_size` and `n_blocks` are the layout used
/// for selection so caller can map back to source positions.
#[derive(Debug, Clone)]
pub struct BlockScores {
    pub scores: Vec<f32>,
    pub block_size: usize,
    pub n_blocks: usize,
    pub source_tokens: usize,
}

/// Dequantize a single position's Q8_0 K cache slice into f32. Layout per
/// PRD-aligned KvCache::new_gpu_q8: head_dim must be a multiple of 32 and
/// each head is `(head_dim / 32) * 34` bytes (`f16 scale | int8 x 32` per
/// block). Output length = `n_kv_heads * head_dim`.
///
/// Pure CPU helper; pulls Q8 K out of a downloaded cache buffer for
/// pflash scoring without needing a HIP kernel.
fn dequant_q8_kv_position(
    bytes: &[u8],
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(head_dim % 32, 0, "Q8 KV cache requires head_dim multiple of 32");
    let blocks_per_head = head_dim / 32;
    let bytes_per_head = blocks_per_head * 34;
    debug_assert_eq!(bytes.len(), n_kv_heads * bytes_per_head);
    debug_assert_eq!(out.len(), n_kv_heads * head_dim);
    for h in 0..n_kv_heads {
        let head_bytes = &bytes[h * bytes_per_head..(h + 1) * bytes_per_head];
        let head_out = &mut out[h * head_dim..(h + 1) * head_dim];
        for blk in 0..blocks_per_head {
            let bb = &head_bytes[blk * 34..(blk + 1) * 34];
            let scale_bits = u16::from_le_bytes([bb[0], bb[1]]);
            let scale = crate::llama::f16_to_f32(scale_bits);
            for j in 0..32 {
                let v = bb[2 + j] as i8;
                head_out[blk * 32 + j] = (v as f32) * scale;
            }
        }
    }
}

/// Run the drafter forward over `source_tokens` (start_pos = 0), populating
/// the drafter's Q8 K cache for positions [0, n). Dispatches on
/// `DrafterModel::{Plain, Hybrid}` so a Qwen3.5-vocab matched drafter
/// (e.g. qwen3.5-0.8b) goes through the qwen35 hybrid path automatically.
/// Returns `(n_layers, n_kv_heads, head_dim)` for downstream cache-layout
/// math. Caller still owns `state.drafter_kv`.
///
/// Preconditions: drafter_loaded, drafter_kv quant_q8, n <= physical_cap,
/// head_dim multiple of 32.
fn drafter_prefill(
    state: &mut PflashState,
    gpu: &mut Gpu,
    source_tokens: &[u32],
) -> HipResult<(usize, usize, usize)> {
    let n = source_tokens.len();
    let model = state.drafter_model.as_mut().expect("loaded -> drafter_model");
    let kv = state.drafter_kv.as_mut().expect("loaded -> kv");
    assert!(kv.quant_q8, "drafter_prefill: drafter KV must be Q8_0");
    assert!(n <= kv.physical_cap, "drafter_prefill: source {n} > physical_cap {}", kv.physical_cap);

    match model {
        DrafterModel::Plain { config, weights, scratch } => {
            assert!(config.head_dim % 32 == 0, "drafter_prefill: head_dim must be multiple of 32");
            llama::forward_prefill_batch(gpu, weights, config, source_tokens, 0, kv, scratch, None)?;
            Ok((config.n_layers, config.n_kv_heads, config.head_dim))
        }
        #[cfg(feature = "deltanet")]
        DrafterModel::Hybrid { config, weights, scratch, dn_state } => {
            assert!(config.head_dim % 32 == 0, "drafter_prefill: head_dim must be multiple of 32");
            // DeltaNet recurrent state (s_matrices, s_scales, conv_states)
            // is advanced by every forward pass, so a previous scoring run
            // on a different prompt would leave stale GDN trajectory in
            // the state. Zero it before this run -- pflash always starts
            // from a fresh prompt at start_pos = 0, so there is no prior
            // state to keep. Plain LLaMA path has no recurrent state, so
            // this is hybrid-only.
            for s in &dn_state.s_matrices { gpu.hip.memset(&s.buf, 0, s.buf.size())?; }
            for s in &dn_state.s_scales   { gpu.hip.memset(&s.buf, 0, s.buf.size())?; }
            for s in &dn_state.conv_states { gpu.hip.memset(&s.buf, 0, s.buf.size())?; }
            // qwen35 batched prefill writes the same Q8_0 K cache layout
            // as llama::forward_prefill_batch. None on hidden_rb /
            // per_token_hidden_out / gdn_tape / tree_verify -- pflash
            // doesn't need DFlash hidden capture or DDTree state.
            qwen35::forward_prefill_batch(
                gpu, weights, config, source_tokens, 0, kv, dn_state, scratch,
                None, None, None, None,
            )?;
            Ok((config.n_layers, config.n_kv_heads, config.head_dim))
        }
    }
}

/// Phase 2.0 fast path: drafter scoring via batched prefill + Q8 cache
/// dequant. Replaces the Phase 1.2 per-token loop (~3 ms/token on
/// qwen3-0.6b at gfx1100, dominated by decode-mode forward pass) with one
/// `forward_prefill_batch` call (~3000+ tok/s prefill) plus a single CPU
/// dequant + mean-pool over the chosen scoring layer's K cache.
///
/// Algorithm (same heuristic as Phase 1.2; only the FORWARD path changed):
///   1. Run llama::forward_prefill_batch on source_tokens at start_pos=0.
///   2. Download the chosen layer's Q8 K cache for [0, source_tokens).
///   3. CPU dequant per position into [N × kv_dim] f32.
///   4. Mean-pool K per block, score = cosine(block_mean, last_K).
///
/// Preconditions:
///   - drafter loaded
///   - drafter_kv quant_q8 (the Phase 1.1 default)
///   - source_tokens.len() <= drafter_kv.physical_cap
///
/// Mutates drafter_kv (overwrites positions 0..source_tokens.len()). Stale
/// data past the source length is ignored by subsequent calls because
/// we always start at position 0.
pub fn compute_scores_batched(
    state: &mut PflashState,
    gpu: &mut Gpu,
    source_tokens: &[u32],
    block_size: usize,
) -> HipResult<BlockScores> {
    let n = source_tokens.len();
    assert!(n > 0, "compute_scores_batched: empty source");
    assert!(block_size > 0, "compute_scores_batched: block_size must be > 0");
    assert!(state.drafter_loaded, "compute_scores_batched: drafter not loaded");

    let (n_layers, n_kv_heads, head_dim) = drafter_prefill(state, gpu, source_tokens)?;
    // Same auto-pick + env-override policy as compute_scores_batched_gpu so
    // the CPU reference path scores from the same layer as the GPU path.
    let auto_layer = state.drafter_model.as_ref()
        .and_then(|m| m.score_layer_idx())
        .unwrap_or(n_layers - 1);
    let layer_idx = std::env::var("HIPFIRE_PFLASH_SCORE_LAYER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&i| i < n_layers)
        .unwrap_or(auto_layer);
    let kv = state.drafter_kv.as_mut().expect("loaded -> kv");

    let blocks_per_head = head_dim / 32;
    let bytes_per_head = blocks_per_head * 34;
    let bytes_per_pos = n_kv_heads * bytes_per_head;
    let kv_dim = n_kv_heads * head_dim;
    let cache_f32 = gpu.download_f32(&kv.k_gpu[layer_idx])?;
    let cache_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(cache_f32.as_ptr() as *const u8, cache_f32.len() * 4)
    };

    // 4. Dequant per position into a flat [n × kv_dim] f32 buffer.
    let mut k_per_pos = vec![0.0f32; n * kv_dim];
    for pos in 0..n {
        let pos_in_bytes = &cache_bytes[pos * bytes_per_pos..(pos + 1) * bytes_per_pos];
        let pos_out = &mut k_per_pos[pos * kv_dim..(pos + 1) * kv_dim];
        dequant_q8_kv_position(pos_in_bytes, n_kv_heads, head_dim, pos_out);
    }

    // 5. Per-block mean K + cosine vs last-position K. Same scoring math
    // as Phase 1.2 compute_scores_cpu, just over the batched-captured
    // K instead of per-token.
    let n_blocks = (n + block_size - 1) / block_size;
    let mut scores = vec![0.0f32; n_blocks];
    let last_k = &k_per_pos[(n - 1) * kv_dim..n * kv_dim];
    let last_norm = norm_l2(last_k);
    for b in 0..n_blocks {
        let start = b * block_size;
        let end = ((b + 1) * block_size).min(n);
        let mut block_mean = vec![0.0f32; kv_dim];
        for pos in start..end {
            let row = &k_per_pos[pos * kv_dim..(pos + 1) * kv_dim];
            for d in 0..kv_dim {
                block_mean[d] += row[d];
            }
        }
        let len_inv = 1.0 / (end - start).max(1) as f32;
        for d in 0..kv_dim {
            block_mean[d] *= len_inv;
        }
        let block_norm = norm_l2(&block_mean);
        let denom = (last_norm * block_norm).max(1e-12);
        let mut dot = 0.0f32;
        for d in 0..kv_dim {
            dot += block_mean[d] * last_k[d];
        }
        scores[b] = dot / denom;
    }

    Ok(BlockScores { scores, block_size, n_blocks, source_tokens: n })
}

/// Phase 2.1 GPU fast path: drafter scoring entirely on the GPU. Same
/// algorithm as `compute_scores_batched`, but the per-block mean +
/// cosine reduce runs on a single HIP launch
/// (`gpu.pflash_score_q8_kv`) reading the Q8 K cache in place. Returns
/// the same `BlockScores` as the CPU paths so it's a drop-in upgrade.
///
/// Pre-conditions match `compute_scores_batched`. On unsupported configs
/// (head_dim not a multiple of 32, non-Q8 KV) caller should fall back to
/// the CPU path. The CPU dequant + reduce in `compute_scores_batched`
/// stays as the reference for cross-checking GPU results.
pub fn compute_scores_batched_gpu(
    state: &mut PflashState,
    gpu: &mut Gpu,
    source_tokens: &[u32],
    block_size: usize,
) -> HipResult<BlockScores> {
    let n = source_tokens.len();
    assert!(n > 0, "compute_scores_batched_gpu: empty source");
    assert!(block_size > 0, "compute_scores_batched_gpu: block_size must be > 0");
    assert!(state.drafter_loaded, "compute_scores_batched_gpu: drafter not loaded");

    let (n_layers, n_kv_heads, head_dim) = drafter_prefill(state, gpu, source_tokens)?;
    let kv = state.drafter_kv.as_ref().expect("loaded -> kv");
    // Source layer for scoring. Default: shallowest FullAttention layer
    // returned by `DrafterModel::score_layer_idx` (layer 0 for plain
    // Qwen3, the first FullAttn slot in the hybrid layer pattern for
    // Qwen3.5/3.6). This dodges the long-context RoPE-OOD NaN cascade
    // documented in MANUAL_REVIEW.md: deep layers accumulate NaN once
    // positions exceed the small drafter's trained window, but the
    // shallowest FullAttn layer's K is still finite at 21K source.
    //
    // HIPFIRE_PFLASH_SCORE_LAYER preserves an escape hatch for
    // operators bisecting the issue further or experimenting with
    // alternative scoring layers; if set and in range it overrides the
    // auto-pick.
    let auto_layer = state.drafter_model.as_ref()
        .and_then(|m| m.score_layer_idx())
        .unwrap_or(n_layers - 1);
    let layer_idx = std::env::var("HIPFIRE_PFLASH_SCORE_LAYER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&i| i < n_layers)
        .unwrap_or(auto_layer);
    let n_blocks = (n + block_size - 1) / block_size;

    let scores_buf = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
    gpu.pflash_score_q8_kv(
        &kv.k_gpu[layer_idx],
        &scores_buf,
        n,
        n_kv_heads,
        head_dim,
        block_size,
        n_blocks,
        n - 1, // last_pos
    )?;
    let scores = gpu.download_f32(&scores_buf)?;
    let _ = gpu.free_tensor(scores_buf);

    Ok(BlockScores { scores, block_size, n_blocks, source_tokens: n })
}

/// Run the drafter over `source_tokens` token by token, capturing post-RoPE
/// K from the last layer at every position into a host buffer, then build
/// per-block scores via mean-pooled K . last-position-K.
///
/// This is the Phase 1.2 MVP -- uses the existing `forward_scratch_compute`
/// per-token path so no llama.rs surface needs Q/K capture hooks. For 8K
/// source on Qwen3-0.6B at gfx1100 this runs in ~30 s; Phase 2+ replaces it
/// with batched scoring on GPU.
///
/// Heuristic: score(block b) = cos_sim(mean_K_block_b, last_K). Picks blocks
/// whose attention key direction aligns with the autoregressive position's
/// own key, matching what the model would attend to at the next token.
/// Cheap, deterministic, NO GPU kernel changes. Phase 2 replaces with
/// proper tail-Q × source-K attention scoring.
///
/// Preconditions:
///   - `state.drafter_loaded == true`
///   - `state.drafter_kv` is sized for at least `source_tokens.len()` positions
///   - drafter is plain Qwen3 (no DeltaNet / MoE)
///
/// Mutates `state.drafter_kv` (advances by `source_tokens.len()`). Caller is
/// responsible for resetting / recreating before reuse if scoring is to be
/// repeated.
pub fn compute_scores_cpu(
    state: &mut PflashState,
    gpu: &mut Gpu,
    source_tokens: &[u32],
    block_size: usize,
) -> HipResult<BlockScores> {
    let n = source_tokens.len();
    assert!(n > 0, "compute_scores_cpu: empty source_tokens");
    assert!(block_size > 0, "compute_scores_cpu: block_size must be > 0");
    assert!(state.drafter_loaded, "compute_scores_cpu: drafter not loaded");

    // Phase 1.2 per-token path is Plain-only (uses llama::forward_scratch_*
    // which doesn't have a Qwen3.5 hybrid equivalent that captures K the
    // same way). Hybrid drafters route through compute_scores_batched(_gpu).
    let model = state.drafter_model.as_ref().expect("drafter loaded -> model");
    let (cfg, weights, scratch) = match model {
        DrafterModel::Plain { config, weights, scratch } =>
            (config.clone(), weights, scratch),
        #[cfg(feature = "deltanet")]
        DrafterModel::Hybrid { .. } => panic!(
            "compute_scores_cpu: Plain-only path; hybrid drafters must call compute_scores_batched_gpu"
        ),
    };
    let kv = state.drafter_kv.as_mut().expect("drafter loaded -> kv");
    assert!(n <= kv.physical_cap,
        "compute_scores_cpu: source {n} exceeds drafter kv physical_cap {}", kv.physical_cap);

    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    let mut k_per_pos: Vec<f32> = Vec::with_capacity(n * kv_dim);

    for (pos, &tok) in source_tokens.iter().enumerate() {
        llama::forward_scratch_embed(gpu, weights, &cfg, tok, pos, scratch)?;
        llama::forward_scratch_compute(gpu, weights, &cfg, pos, kv, scratch)?;
        // scratch.k now holds the post-RoPE K for the LAST processed layer
        // at position `pos`. Download to host. The buffer is sized to kv_dim
        // f32 elements regardless of cache quantization (it's the pre-quant
        // K that gets fed into the cache write).
        let k_row = gpu.download_f32(&scratch.k)?;
        debug_assert_eq!(k_row.len(), kv_dim,
            "scratch.k size {} != expected kv_dim {kv_dim}", k_row.len());
        k_per_pos.extend_from_slice(&k_row);
    }

    let n_blocks = (n + block_size - 1) / block_size;
    let mut scores = vec![0.0f32; n_blocks];

    // Last-position K is the proxy for "what the model would attend to next"
    // -- used as the query direction against block-mean Ks.
    let last_k = &k_per_pos[(n - 1) * kv_dim..n * kv_dim];
    let last_norm = norm_l2(last_k);

    for b in 0..n_blocks {
        let start = b * block_size;
        let end = ((b + 1) * block_size).min(n);
        // Mean-pool K over positions in this block.
        let mut block_mean = vec![0.0f32; kv_dim];
        for pos in start..end {
            let row = &k_per_pos[pos * kv_dim..(pos + 1) * kv_dim];
            for d in 0..kv_dim {
                block_mean[d] += row[d];
            }
        }
        let len_inv = 1.0 / (end - start).max(1) as f32;
        for d in 0..kv_dim {
            block_mean[d] *= len_inv;
        }
        let block_norm = norm_l2(&block_mean);
        // Cosine similarity. Avoids favoring large-magnitude blocks just
        // because they're high-norm. Returns NaN-free even if a vector is
        // all zeros (rare; clamp denominator).
        let denom = (last_norm * block_norm).max(1e-12);
        let mut dot = 0.0f32;
        for d in 0..kv_dim {
            dot += block_mean[d] * last_k[d];
        }
        scores[b] = dot / denom;
    }

    Ok(BlockScores { scores, block_size, n_blocks, source_tokens: n })
}

fn norm_l2(v: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// Pick which source positions survive compression. Combines mandatory
/// anchors (prefix sink + recent tail + caller-supplied `must_keep` spans)
/// with top-scoring middle blocks under a token budget set by `keep_ratio`
/// × `source_tokens`, floor'd at `min_keep_tokens`. Returns kept spans
/// `[ (start, end_exclusive) ... ]` in source order, with overlapping
/// spans coalesced.
///
/// PRD §5.4 selection rules:
///   - always keep `sink_tokens` from the front,
///   - always keep `recent_tokens` from the back,
///   - always keep every span in `must_keep_spans` (chat boundaries,
///     system message frames, tool-defs, role markers -- anything the
///     prompt parser needs to find or the model treats as a control
///     token). Caller is responsible for locating these positions in
///     source-order using the target tokenizer's special-token IDs.
///   - select highest-scoring middle blocks by descending score until the
///     overall budget is met,
///   - coalesce overlapping / adjacent spans so the emitted token stream
///     stays span-coherent (no single-token scatter).
///
/// `must_keep_spans` is consumed verbatim -- caller passes empty slice when
/// the prompt has no chat boundaries (raw-text completion). Spans outside
/// `[0, source_tokens)` are clamped, not rejected.
///
/// Pure CPU. Deterministic for a fixed input tuple.
pub fn select_spans(
    scores: &BlockScores,
    sink_tokens: usize,
    recent_tokens: usize,
    keep_ratio: f32,
    min_keep_tokens: usize,
    must_keep_spans: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    assert!(keep_ratio > 0.0 && keep_ratio <= 1.0,
        "keep_ratio {keep_ratio} must be in (0, 1]");
    let n = scores.source_tokens;
    let bs = scores.block_size;
    let n_blocks = scores.n_blocks;

    let target_kept = std::cmp::max(
        min_keep_tokens,
        (n as f32 * keep_ratio).ceil() as usize,
    ).min(n);

    // If the prompt is shorter than the floor, keep everything.
    if target_kept >= n {
        return vec![(0, n)];
    }

    let sink_end = sink_tokens.min(n);
    let recent_start = if recent_tokens >= n { 0 } else { n - recent_tokens };

    // Build the mandatory-keep set up-front so it counts against budget.
    // Clamp must_keep entries to [0, n) and drop empty / inverted spans.
    let mut anchors: Vec<(usize, usize)> = Vec::with_capacity(2 + must_keep_spans.len());
    if sink_end > 0 {
        anchors.push((0, sink_end));
    }
    if recent_start < n {
        anchors.push((recent_start, n));
    }
    for &(s, e) in must_keep_spans {
        let s = s.min(n);
        let e = e.min(n);
        if s < e {
            anchors.push((s, e));
        }
    }
    let anchors = coalesce(anchors);
    let anchored: usize = anchors.iter().map(|(s, e)| e - s).sum();

    if anchored >= target_kept {
        // Anchors alone meet the budget. Return them coalesced; nothing more
        // to add. Coalesce already collapsed any overlap among sink / recent
        // / must_keep.
        return anchors;
    }

    // Budget for middle-block selection.
    let middle_budget = target_kept - anchored;

    // Rank ALL blocks by descending score. A scored block may partially
    // overlap an anchor (e.g., a single ChatML boundary token sitting
    // inside an otherwise interesting block); the overlapping positions
    // are already covered, but the block's NON-anchored positions still
    // matter and the block itself stays span-coherent when emitted.
    //
    // The earlier "filter out any block touching an anchor" approach
    // could starve selection: with ChatML boundaries every few hundred
    // tokens, every middle block is touched and the budget never fills.
    // Fix: include all blocks, count INCREMENTAL coverage against the
    // budget. Blocks fully covered by anchors contribute zero increment
    // and add no tokens, so they're skipped naturally.
    fn block_incremental(anchors: &[(usize, usize)], start: usize, end: usize) -> usize {
        // Tokens in [start, end) not already covered by any anchor.
        let mut covered = 0usize;
        for &(a_s, a_e) in anchors.iter() {
            let lo = std::cmp::max(start, a_s);
            let hi = std::cmp::min(end, a_e);
            if lo < hi {
                covered += hi - lo;
            }
        }
        (end - start).saturating_sub(covered)
    }
    let mut middle: Vec<(usize, f32)> = (0..n_blocks)
        .map(|b| (b, scores.scores[b]))
        .collect();
    // Stable sort by descending score; tie-break on block index for
    // determinism.
    middle.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        .then(a.0.cmp(&b.0)));

    let mut spans = anchors;
    let mut middle_kept = 0usize;
    for (b, _score) in middle {
        if middle_kept >= middle_budget {
            break;
        }
        let start = b * bs;
        let end = ((b + 1) * bs).min(n);
        let incr = block_incremental(&spans, start, end);
        if incr == 0 {
            // Fully anchored already; nothing to add.
            continue;
        }
        spans.push((start, end));
        middle_kept += incr;
    }

    coalesce(spans)
}

/// Sort + merge overlapping or adjacent half-open spans. `[a, b) ∪ [b, c)`
/// merges into `[a, c)`. Required by the selection output to be span-coherent.
fn coalesce(mut spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if spans.is_empty() {
        return spans;
    }
    spans.sort_by_key(|&(s, _)| s);
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    out.push(spans[0]);
    for &(s, e) in &spans[1..] {
        let last = out.last_mut().unwrap();
        if s <= last.1 {
            last.1 = std::cmp::max(last.1, e);
        } else {
            out.push((s, e));
        }
    }
    out
}

/// Emit the compressed token stream by gathering the tokens from every kept
/// span in source order. `spans` must be coalesced + sorted (output of
/// `select_spans`); this function does not re-validate.
pub fn emit_compressed(source_tokens: &[u32], kept_spans: &[(usize, usize)]) -> Vec<u32> {
    let total: usize = kept_spans.iter().map(|(s, e)| e.saturating_sub(*s)).sum();
    let mut out = Vec::with_capacity(total);
    for &(s, e) in kept_spans {
        let start = s.min(source_tokens.len());
        let end = e.min(source_tokens.len());
        if start < end {
            out.extend_from_slice(&source_tokens[start..end]);
        }
    }
    out
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
    /// Vision request -- image-bearing prompts always bypass for now.
    VisionRequest,
    /// Tool-calling request or prompt with structured JSON tool definitions.
    ToolCallRequest,
    /// Drafter not loaded; nothing to score with.
    DrafterUnavailable,
    /// Drafter and target tokenizers do not match.
    TokenizerMismatch,
    /// Architecture / KV / model shape unsupported by the current drafter.
    UnsupportedDrafter { reason: String },
    /// Scorer returned non-finite or all-zero scores. Compressing on those
    /// would silently corrupt span selection (NaN sorts unstably, all-zero
    /// gives meaningless ranking), so we bypass loudly instead.
    ScoringDegenerate { detail: String },
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
            BypassReason::ScoringDegenerate { detail } =>
                format!("scoring_degenerate: {detail}"),
        }
    }
}

impl BlockScores {
    /// Cheap health check: scores must be finite and at least one must be
    /// nonzero. Returns `Err(reason)` describing the failure, `Ok(())` on
    /// healthy output. Used by `maybe_compress_prompt` to bypass loudly
    /// rather than ship a CompressedPrompt built from junk scores (NaN
    /// sorts unstably, all-zero gives ill-defined ranking).
    pub fn well_formed(&self) -> Result<(), String> {
        if self.scores.is_empty() {
            return Err("scores vector is empty".to_string());
        }
        let mut n_nan = 0usize;
        let mut n_inf = 0usize;
        let mut any_nonzero = false;
        for &s in &self.scores {
            if s.is_nan() {
                n_nan += 1;
            } else if s.is_infinite() {
                n_inf += 1;
            } else if s.abs() > 0.0 {
                any_nonzero = true;
            }
        }
        if n_nan > 0 || n_inf > 0 {
            return Err(format!("non-finite scores: {n_nan} NaN, {n_inf} inf"));
        }
        if !any_nonzero {
            return Err("all scores are exactly 0.0".to_string());
        }
        Ok(())
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

/// Stable hex md5 of a slice of u32 token ids (LE bytes). Used for the
/// `source_md5` and `compressed_md5` fields in `CompressedPrompt` per PRD
/// §5.3.3 reproducibility contract.
fn token_md5(tokens: &[u32]) -> String {
    use std::process::Command;
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let mut child = match Command::new("md5sum")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return String::new(), // Best-effort: empty hash if md5sum missing.
    };
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        let _ = stdin.write_all(&bytes);
    }
    match child.wait_with_output() {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout);
            s.split_whitespace().next().unwrap_or("").to_string()
        }
        Err(_) => String::new(),
    }
}

/// Top-level compression entry point. Decides bypass vs compress and
/// dispatches accordingly.
///
/// Compress path:
///   1. score blocks via `compute_scores_cpu` (drafter K-capture)
///   2. `select_spans` with cfg's sink/recent/ratio/min_keep + caller
///      `must_keep_spans` (chat boundaries / role markers / tool defs)
///   3. `emit_compressed` to materialize the kept token stream
///   4. populate `CompressedPrompt` with md5s + per-stage timings
///
/// Returns `Bypass(reason)` whenever the gating logic in `decide_bypass`
/// short-circuits, AND when scoring/selection cannot meaningfully
/// compress (e.g., budget would keep the whole prompt). Caller hands the
/// `Compressed` variant's `token_ids` to the target prefill path.
pub fn maybe_compress_prompt(
    gpu: &mut rdna_compute::Gpu,
    state: &mut PflashState,
    cfg: &PflashConfig,
    token_ids: &[u32],
    request_kind: RequestKind,
    must_keep_spans: &[(usize, usize)],
) -> HipResult<PflashDecision> {
    if let Some(reason) = decide_bypass(state, cfg, token_ids, request_kind) {
        return Ok(PflashDecision::Bypass { reason });
    }
    let n = token_ids.len();
    let t_total = std::time::Instant::now();

    // 1. Score blocks via the Phase 2.1 GPU path. The CPU paths
    // (compute_scores_batched / compute_scores_cpu) remain public for
    // tests and for archs where head_dim % 32 != 0 (the GPU kernel
    // requires it).
    let t_score = std::time::Instant::now();
    let bs = compute_scores_batched_gpu(state, gpu, token_ids, cfg.block_size)?;
    let score_ms = t_score.elapsed().as_millis();
    if let Err(detail) = bs.well_formed() {
        // Bypass loudly: select_spans treats NaN as Equal, so a broken
        // scorer would otherwise produce plausible-looking output that's
        // actually meaningless ranking. Surface the concrete failure so
        // operators can spot scorer regressions in logs.
        return Ok(PflashDecision::Bypass {
            reason: BypassReason::ScoringDegenerate { detail },
        });
    }

    // 2. Select spans.
    let t_select = std::time::Instant::now();
    let kept_spans = select_spans(
        &bs, cfg.sink_tokens, cfg.recent_tokens, cfg.keep_ratio,
        cfg.min_keep_tokens, must_keep_spans,
    );
    let select_ms = t_select.elapsed().as_millis();

    // 3. Gather.
    let t_gather = std::time::Instant::now();
    let compressed: Vec<u32> = emit_compressed(token_ids, &kept_spans);
    let gather_ms = t_gather.elapsed().as_millis();

    let total_ms = t_total.elapsed().as_millis();

    // If budget kept (effectively) the whole prompt, bypass downstream so
    // the daemon doesn't double-prefill the same tokens.
    if compressed.len() >= n {
        return Ok(PflashDecision::Bypass {
            reason: BypassReason::BelowThreshold {
                source_tokens: n,
                threshold: cfg.threshold_tokens,
            },
        });
    }

    let source_md5 = token_md5(token_ids);
    let compressed_md5 = token_md5(&compressed);

    Ok(PflashDecision::Compressed(CompressedPrompt {
        source_tokens: n,
        kept_tokens: compressed.len(),
        token_ids: compressed,
        kept_spans,
        source_md5,
        compressed_md5,
        timings: PflashTimings {
            drafter_prefill_ms: 0, // counted inside score_ms; no separate clock yet
            score_ms,
            select_ms,
            gather_ms,
            total_ms,
        },
    }))
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
            drafter_model: None,
            drafter_tokenizer: None,
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

    fn synthetic_scores(scores_vec: Vec<f32>, block_size: usize) -> BlockScores {
        let n_blocks = scores_vec.len();
        BlockScores {
            scores: scores_vec,
            block_size,
            n_blocks,
            source_tokens: n_blocks * block_size,
        }
    }

    #[test]
    fn select_returns_full_when_under_min_keep() {
        let scores = synthetic_scores(vec![0.1; 4], 8); // 32 tokens
        let spans = select_spans(&scores, 8, 8, 0.5, 64, &[]);
        assert_eq!(spans, vec![(0, 32)], "min_keep>n must return full span");
    }

    #[test]
    fn select_picks_top_blocks_with_anchors() {
        // 16 blocks of 8 tokens = 128 source tokens. Make middle block 7
        // (positions 56..64) the highest-scoring; expect it to survive
        // alongside sink + recent.
        let mut s = vec![0.1f32; 16];
        s[7] = 5.0;
        let scores = synthetic_scores(s, 8);
        // sink=16, recent=16. keep_ratio=0.25 → target=32 tokens. Anchors cover
        // 32 already, so middle budget = 0 -- block 7 is NOT picked because
        // anchors alone meet the budget. This is the documented behavior.
        let spans = select_spans(&scores, 16, 16, 0.25, 0, &[]);
        // Expected: [0, 16) ∪ [112, 128).
        assert_eq!(spans, vec![(0, 16), (112, 128)]);

        // Bump keep_ratio to 0.40 → target=52 → middle budget=20 → block 7
        // (8 tokens, score 5.0) survives first; the remaining ~12 token
        // budget pulls the next two tied-score blocks (2 and 3, ascending
        // index tie-break) which coalesce with the sink into [0, 32).
        let spans = select_spans(&scores, 16, 16, 0.40, 0, &[]);
        assert_eq!(spans, vec![(0, 32), (56, 64), (112, 128)],
            "block 7 must survive on score; ties pull lowest-index first → \
             blocks 2+3 coalesce with sink");
    }

    #[test]
    fn select_coalesces_adjacent_spans() {
        // Two adjacent middle blocks both top-scoring → single coalesced span.
        let mut s = vec![0.1f32; 16];
        s[7] = 5.0;
        s[8] = 5.0;
        let scores = synthetic_scores(s, 8);
        let spans = select_spans(&scores, 16, 16, 0.50, 0, &[]);
        assert!(spans.iter().any(|&(a, b)| a == 56 && b == 72),
            "blocks 7+8 (56..64 + 64..72) must coalesce, got {spans:?}");
    }

    #[test]
    fn select_preserves_must_keep_chat_boundaries() {
        // Simulate a ChatML prompt: <|im_start|> at pos 50, <|im_end|> at
        // pos 51. Even with low scores in that region the boundaries must
        // survive. The middle block containing those positions has score
        // 0.0 so it would never be picked otherwise.
        let s = vec![0.1f32; 16]; // 128 tokens, all near-zero
        let scores = synthetic_scores(s, 8);
        // Must-keep two single-token spans. They sit in block 6 (48..56)
        // which would not be picked by the scoring loop.
        let must = vec![(50, 51), (51, 52)];
        let spans = select_spans(&scores, 4, 4, 0.05, 0, &must);
        // Boundaries must show up in the output.
        assert!(spans.iter().any(|&(a, b)| a <= 50 && 51 <= b),
            "must_keep position 50 dropped, spans = {spans:?}");
        assert!(spans.iter().any(|&(a, b)| a <= 51 && 52 <= b),
            "must_keep position 51 dropped, spans = {spans:?}");
    }

    #[test]
    fn select_coalesces_must_keep_with_anchors() {
        // Must-keep span [14, 18) overlaps the sink end (sink=16). The
        // coalesce contract is "no two output spans overlap or touch":
        // sink (0..16) + must_keep (14..18) MUST merge into a single
        // contiguous prefix range (which may extend further if the budget
        // pulls in adjacent blocks under the new incremental-coverage
        // selection).
        let s = vec![0.1f32; 16];
        let scores = synthetic_scores(s, 8);
        let spans = select_spans(&scores, 16, 16, 0.30, 0, &[(14, 18)]);
        // Whatever extra blocks come in, the prefix must still be one
        // contiguous span starting at 0 and covering the must-keep range.
        let prefix = spans.iter().find(|&&(s, _)| s == 0)
            .copied()
            .unwrap_or_else(|| panic!("missing prefix anchor in {spans:?}"));
        assert!(prefix.1 >= 18, "prefix {prefix:?} must cover must_keep end 18");
        // No two output spans may overlap or be adjacent without merging.
        for w in spans.windows(2) {
            assert!(w[0].1 < w[1].0, "spans must be disjoint and gapped, got {spans:?}");
        }
    }

    #[test]
    fn select_does_not_starve_when_every_block_touches_anchor() {
        // 16 blocks of 8 tokens = 128 source. Boundary token at every 8th
        // position (0, 8, 16, ...) -- that's one inside every block. With
        // the old overlap-filter design every middle block would be
        // disqualified and `middle` would be empty, leaving budget unmet.
        // Regression: incremental-coverage selection must still pull
        // high-score blocks despite the boundary touch.
        let mut s = vec![0.1f32; 16];
        s[7] = 5.0; // block 7 covers [56, 64); we want this to survive.
        let scores = synthetic_scores(s, 8);
        let must: Vec<(usize, usize)> = (0..16).map(|b| (b * 8, b * 8 + 1)).collect();
        let spans = select_spans(&scores, 0, 0, 0.50, 0, &must);
        let total: usize = spans.iter().map(|(a, b)| b - a).sum();
        // 50% of 128 = 64 token target. Anchors alone contribute 16
        // incremental tokens, so middle selection MUST add ~48 more.
        assert!(total >= 64, "selector starved -- expected >=64 kept, got {total} (spans = {spans:?})");
        // Block 7 is the highest-scored, so it should appear.
        assert!(spans.iter().any(|&(a, b)| a <= 56 && 64 <= b),
            "block 7 (highest score) must survive despite anchor overlap, got {spans:?}");
    }

    #[test]
    fn compat_signature_detects_slot_shuffle_with_padding() {
        // Property under test: skipping padding slots when hashing would let
        // [FOO, PAD, BAR] hash-match [FOO, BAR, PAD] even though those
        // tokenizers map id=1 to different strings (the second has BAR at
        // id=1, the first has padding there). The fix is to mix a stable
        // marker IN PLACE for padding slots so positional information is
        // preserved.
        //
        // We exercise this via two synthetic HF tokenizer JSON blobs that
        // share size + non-padding strings but differ in slot ordering of
        // an audio/tts padding token.
        let json_a = r#"{
            "model": {
                "vocab": {
                    "FOO": 0,
                    "<|audio_pad|>": 1,
                    "BAR": 2,
                    "<|endoftext|>": 3
                },
                "merges": []
            },
            "added_tokens": [
                {"id": 3, "content": "<|endoftext|>", "special": true}
            ]
        }"#;
        let json_b = r#"{
            "model": {
                "vocab": {
                    "FOO": 0,
                    "BAR": 1,
                    "<|audio_pad|>": 2,
                    "<|endoftext|>": 3
                },
                "merges": []
            },
            "added_tokens": [
                {"id": 3, "content": "<|endoftext|>", "special": true}
            ]
        }"#;
        let tok_a = Tokenizer::from_hf_json(json_a).expect("tokenizer A");
        let tok_b = Tokenizer::from_hf_json(json_b).expect("tokenizer B");
        // tokenizers_compatible must reject (slot 1 diverges: padding vs BAR).
        assert!(!tokenizers_compatible(&tok_a, &tok_b),
            "tokenizers_compatible must reject slot-shuffle pairs");
        // compat_signature must also diverge (otherwise pretok would silently
        // load tokens authored by A under B).
        assert_ne!(tokenizer_compat_signature(&tok_a),
                   tokenizer_compat_signature(&tok_b),
                   "compat signature must encode slot position, not just \
                    the set of non-padding strings");
    }

    #[test]
    fn compat_signature_matches_self() {
        // Sanity: a tokenizer is always compat with itself, and its
        // signature is deterministic across calls.
        let json = r#"{
            "model": {
                "vocab": {"FOO": 0, "BAR": 1, "<|endoftext|>": 2},
                "merges": []
            },
            "added_tokens": [
                {"id": 2, "content": "<|endoftext|>", "special": true}
            ]
        }"#;
        let tok = Tokenizer::from_hf_json(json).expect("tokenizer");
        assert!(tokenizers_compatible(&tok, &tok));
        assert_eq!(tokenizer_compat_signature(&tok),
                   tokenizer_compat_signature(&tok));
    }

    #[test]
    fn block_scores_well_formed_rejects_nan_inf_and_all_zero() {
        // NaN -> reject.
        let bs = synthetic_scores(vec![0.5, f32::NAN, 0.3, 0.1], 8);
        assert!(bs.well_formed().unwrap_err().contains("non-finite"));
        // Inf -> reject.
        let bs = synthetic_scores(vec![0.5, f32::INFINITY, 0.3, 0.1], 8);
        assert!(bs.well_formed().unwrap_err().contains("non-finite"));
        // All-zero -> reject.
        let bs = synthetic_scores(vec![0.0; 4], 8);
        assert!(bs.well_formed().unwrap_err().contains("all scores"));
        // Empty -> reject.
        let bs = BlockScores { scores: vec![], block_size: 8, n_blocks: 0, source_tokens: 0 };
        assert!(bs.well_formed().unwrap_err().contains("empty"));
        // Healthy -> ok.
        let bs = synthetic_scores(vec![0.1, 0.2, 0.3, 0.4], 8);
        assert!(bs.well_formed().is_ok());
    }

    #[test]
    fn select_clamps_oob_must_keep() {
        let s = vec![0.1f32; 4];
        let scores = synthetic_scores(s, 8); // 32 tokens
        let spans = select_spans(&scores, 0, 0, 0.5, 0, &[(100, 200), (5, 1000)]);
        // First span clamps to nothing (start > n), second clamps to (5, 32).
        assert!(spans.iter().all(|&(_, e)| e <= 32), "spans must stay in range, got {spans:?}");
        assert!(spans.iter().any(|&(a, b)| a == 5 && b == 32),
            "expected clamped (5, 32), got {spans:?}");
    }

    #[test]
    fn emit_concatenates_in_order() {
        let src: Vec<u32> = (0..20).collect();
        let spans = vec![(0, 4), (10, 14)];
        let out = emit_compressed(&src, &spans);
        assert_eq!(out, vec![0, 1, 2, 3, 10, 11, 12, 13]);
    }

    #[test]
    fn emit_clamps_out_of_bounds_spans() {
        let src: Vec<u32> = (0..5).collect();
        let spans = vec![(0, 100), (200, 300)];
        let out = emit_compressed(&src, &spans);
        assert_eq!(out, vec![0, 1, 2, 3, 4]);
    }
}
