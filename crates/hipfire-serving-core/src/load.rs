// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Model load / unload: turning an HFQ (or safetensors) path + load-message
//! params into a [`LoadedModel`], and tearing it down.
//!
//! Covers the per-arch single-GPU `load_model`, the safetensors path
//! (`load_model_safetensors`), the multi-GPU pipeline-parallel `load_model_pp`,
//! `unload_model`, the optional DFlash drafter (`load_dflash_state`), and the
//! small config/metadata helpers (chat-template resolution, state-quant parsing,
//! parameter counting, tiny-model bring-up). Extracted verbatim from the former
//! `main.rs` monolith (no behavior change); items called from `main.rs` are
//! `pub`.

use std::path::Path;

use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_gemma3::{Gemma3Backend, Gemma3State};
use hipfire_arch_gemma3_vl::{load_vl, Gemma3VlBackend, LoadedVl};
#[cfg(feature = "arch-lfm2moe")]
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_llama::Llama;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::{DeltaNetState, LayerType, Qwen35ScratchSet};
use hipfire_arch_qwen35::speculative::{
    DdtreeScratch, DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer, VerifyScratch,
};
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_model::{
    arch_features, is_qwen35_dense_arch_id, is_qwen35_family_arch_id, FeatureSupport,
    ARCH_ID_DEEPSEEK4_FLASH, ARCH_ID_DOTS_OCR, ARCH_ID_GEMMA3_TEXT, ARCH_ID_GEMMA3_VL,
    ARCH_ID_LFM2_MOE, ARCH_ID_LLAMA_MISTRAL, ARCH_ID_MAMBA2, ARCH_ID_MINIMAX_M2,
    ARCH_ID_NEMOTRON_H, ARCH_ID_QWEN2, ARCH_ID_QWEN3_QWEN2_LEGACY, ARCH_ID_ZAYA,
};
use hipfire_prompt as prompt_frame;
use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv;
use hipfire_runtime::llama;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::triattn::{EvictionCtx, TriAttnCenters};

use crate::memory::{hfq_model_memory, unknown_model_memory};
use crate::model::CaskConfig;
#[cfg(feature = "arch-lfm2moe")]
use crate::model::Lfm2DflashState;
use crate::model::{DdtreeState, DflashState, DsparkState, Eviction, LoadedModel, ResidentSession};
use crate::session::{
    next_qwen35_state_allocation_epoch, SessionRegistry, QWEN35_LEGACY_SESSION_ID,
};
use hipfire_runtime::sequence_state::SequenceState;

/// Matrix-backed admission gate: refuse a request for `feature` on a model whose
/// arch capability matrix (the generated `arch_features`, source
/// `docs/model-support.toml`) does not mark it fully supported. Returns a clean,
/// operator-facing reason. This is the single authority — adding an arch to the
/// matrix as feature-capable enables it here without touching this code.
fn require_arch_feature(
    arch_id: u32,
    feature: &str,
    support: FeatureSupport,
) -> Result<(), String> {
    if support.is_full() {
        return Ok(());
    }
    let f = arch_features(arch_id);
    Err(format!(
        "{feature} requested but arch {} (arch_id={arch_id}) does not support it \
         (capability matrix: {}={}). Reload without it. See MODEL-SUPPORT.md / \
         docs/model-support.toml.",
        f.label,
        feature,
        support.mark()
    ))
}

#[cfg(feature = "arch-lfm2moe")]
fn lfm2_triattn_kv_layer_ids(config: &lfm2moe::config::Lfm2MoeConfig) -> Vec<usize> {
    (0..config.num_attention_layers()).collect()
}

/// Resolve the effective chat template for a model: the HFQ-embedded
/// `tokenizer_config.chat_template`, with sidecar/path fallbacks. `None` when
/// the model ships no template (base/completion model).
pub fn resolve_chat_template(
    hfq: &hipfire_runtime::hfq::HfqFile,
    model_path: &str,
) -> Option<String> {
    match prompt_frame::resolve_chat_template(model_path, hfq.chat_template()) {
        Some(resolved) => {
            prompt_frame::log_resolved_chat_template_source(&resolved.source);
            Some(resolved.template)
        }
        None => {
            // Defensive tripwire. A `None` here means `effective_raw()` will
            // default to true (see model::effective_raw) and the daemon serves
            // the prompt RAW/unframed — correct for a base/completion model, but
            // a disaster for a chat model: a bare prompt has no
            // `<|im_start|>assistant` decode anchor and collapses heavily-quantized
            // chat models into a token attractor. The normal quantize path always
            // embeds the template (folding `chat_template.jinja`, and requant
            // carries metadata forward), so a chat arch arriving template-less
            // means a hand-built / corrupted HFQ. Warn loudly rather than
            // silently mis-serving. Scoped to the qwen3.5/3.6 family (arch_id
            // 5/6), the known chat arch this guards; base models of other arches
            // legitimately resolve no template and must not be warned on.
            if hipfire_model::is_qwen35_family_arch_id(hfq.arch_id) {
                eprintln!(
                    "[chat_template] WARNING: chat arch (arch_id={}) resolved NO chat template \
                     for {model_path} — the daemon will serve RAW/unframed prompts, which can \
                     collapse a chat model into a token attractor. Provide one via \
                     HIPFIRE_CHAT_TEMPLATE_FILE, ~/.hipfire/templates/<model-basename>.j2, or \
                     re-quantize from a source that embeds tokenizer_config.chat_template.",
                    hfq.arch_id
                );
            }
            None
        }
    }
}

/// Parse a resolved chat-template string into a `ChatTemplateProfile` (the
/// stop-token / holdback / framing metadata the output filter and prompt
/// framing consume).
pub fn profile_chat_template(
    chat_template: Option<String>,
    tokenizer: Option<&hipfire_model::tokenizer::Tokenizer>,
) -> (Option<String>, Option<prompt_frame::ChatTemplateProfile>) {
    let profile = match (chat_template.as_deref(), tokenizer) {
        (Some(template), Some(tokenizer)) => {
            match prompt_frame::ChatTemplateProfile::from_template(tokenizer, template) {
                Ok(profile) => Some(profile),
                Err(e) => {
                    eprintln!("[chat_template] failed to profile template ({e}); using fallback stop policy");
                    None
                }
            }
        }
        _ => None,
    };
    (chat_template, profile)
}

/// Parse the DeltaNet state-quant mode from a load-message param string
/// (e.g. `q8`/`fp16`), falling back to the arch default when absent/unknown.
pub fn parse_state_quant(
    mode: Option<&str>,
) -> Result<hipfire_arch_qwen35::qwen35::StateQuant, String> {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    match mode.unwrap_or("q8").to_ascii_lowercase().as_str() {
        "" | "auto" | "q8" | "int8" => Ok(StateQuant::Q8),
        "fp32" | "f32" => Ok(StateQuant::FP32),
        "q4" | "int4" => Ok(StateQuant::Q4),
        other => Err(format!(
            "unsupported DeltaNet state_quant '{other}' (expected q8|fp32|q4)"
        )),
    }
}

/// Human-readable label for a `StateQuant` (for the `loaded` event / status).
pub fn state_quant_label(q: hipfire_arch_qwen35::qwen35::StateQuant) -> &'static str {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    match q {
        StateQuant::FP32 => "FP32",
        StateQuant::Q8 => "Q8",
        StateQuant::Q4 => "Q4",
    }
}

/// Total parameter count across an HFQ's tensors (summed element counts), for
/// the reported model size.
pub fn hfq_parameter_count(hfq: &HfqFile) -> u128 {
    hfq.tensors()
        .iter()
        .map(|t| {
            t.shape
                .iter()
                .fold(1u128, |acc, &dim| acc.saturating_mul(dim as u128))
        })
        .sum()
}

/// True if any tensor in the HFQ is stored as bf16 — gates the bf16-capable
/// load path / dtype handling.
pub fn hfq_has_bf16_weights(hfq: &HfqFile) -> bool {
    hfq.tensors().iter().any(|t| t.quant_type == 16)
}

/// True only when the model is *predominantly* BF16 (a full-precision artifact),
/// not merely a quantized model that keeps a few small tensors (norms) at BF16.
/// Decided on the 2-D weight tensors (the matmul projections): a full BF16 model
/// has them all BF16 (qt==16); an MQ4/Q8 model has them quantized with only 1-D
/// norms left at BF16. Used to decide whether to FORCE fp32 KV — quantized
/// models must NOT be forced (it locks them out of batched prefill).
pub fn hfq_is_bf16_dominant(hfq: &HfqFile) -> bool {
    let (mut bf16_2d, mut total_2d) = (0usize, 0usize);
    for t in hfq.tensors() {
        if t.shape.len() == 2 {
            total_2d += 1;
            if t.quant_type == 16 {
                bf16_2d += 1;
            }
        }
    }
    total_2d > 0 && bf16_2d * 2 > total_2d
}

/// Read the model's trained context window (`max_position_embeddings`) from the
/// HFQ metadata JSON. Handles the multimodal wrapper shape (gemma3-vl etc.)
/// where the decoder config is nested under `text_config`.
fn model_max_position_embeddings(metadata_json: &str) -> Option<usize> {
    let v: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
    let read = |obj: &serde_json::Value| {
        obj.get("max_position_embeddings")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
    };
    // HFQ metadata wraps the HF config under `config`; multimodal wrappers
    // (gemma3-vl etc.) nest the decoder shape under `config.text_config`. Also
    // accept the field at the top level for formats that hoist it. Mirrors
    // `hipfire_arch_gemma3::config_from_metadata_json`.
    let config = v.get("config");
    None.or_else(|| config.and_then(|c| c.get("text_config")).and_then(read))
        .or_else(|| config.and_then(read))
        .or_else(|| v.get("text_config").and_then(read))
        .or_else(|| read(&v))
}

/// Clamp a requested `max_seq` to the model's trained context window. Allocating
/// KV for more than `max_position_embeddings` wastes memory (and on RDNA APUs
/// can OOM the shared GTT pool) for context the model was never trained to use.
/// An operator can opt out with `HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1` (a warning is
/// printed either way).
fn clamp_max_seq_to_model_context(max_seq: usize, metadata_json: &str) -> usize {
    match model_max_position_embeddings(metadata_json) {
        Some(model_max) if max_seq > model_max => {
            if std::env::var("HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "  WARNING: max_seq={max_seq} exceeds model max_position_embeddings={model_max}; \
                     HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1 set — proceeding with {max_seq} \
                     (may exceed trained context and/or OOM the KV allocation)"
                );
                max_seq
            } else {
                eprintln!(
                    "  WARNING: max_seq={max_seq} exceeds model max_position_embeddings={model_max}; \
                     clamping to {model_max}. Set HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1 to force the larger value."
                );
                model_max
            }
        }
        _ => max_seq,
    }
}

/// Budget cap for gemma3 context on shared-GTT RDNA APUs.
///
/// gemma3 now uses sliding-window attention: the 5-of-6 LOCAL layers keep only a
/// `sliding_window` (1024) ring, so their KV is fixed and tiny. The remaining
/// term is the ~10 GLOBAL layers, which still carry a full-context cache. At
/// medgemma's default F32 KV that is ~1.6 MB/token across the global layers, so
/// full 128K would be ~21 GB of global KV + ~15 GB weights — still over the
/// ~32 GB effective budget (the 43 GB GTT is shared with the host OS). q8 global
/// KV (`kv_mode=q8`) drops that ~4× and reaches full context; until that is the
/// default, cap F32 gemma3 at a budget-safe context. Operators can force the
/// full value with `HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1`.
///
/// 65536: ~10.7 GB global F32 KV + ~0.9 GB local rings + ~15 GB weights ≈ 27 GB.
const GEMMA3_STOPGAP_MAX_SEQ: usize = 65_536;

/// Map an operator `kv_mode` string to the gemma3 KV cache mode + KVarN bit
/// width. gemma3's state wires F32, Q8 (`q8`/`int8`), and KVarN (variance-
/// normalized K + Q8 V) at `kvarn2`/`kvarn`(=4)/`kvarn8`; the rotated asym/fwht
/// tiers have no gemma3 kernel yet and fall back to F32. The `usize` is the
/// KVarN K bit width (meaningful only for the Kvarn mode; 4 otherwise).
fn gemma3_kv_mode(kv_mode: &str) -> (hipfire_runtime::kv::KvQuantMode, usize) {
    use hipfire_runtime::kv::KvQuantMode;
    match kv_mode {
        "q8" | "int8" => (KvQuantMode::Q8, 4),
        "kvarn2" => (KvQuantMode::Kvarn, 2),
        "kvarn" | "kvarn4" => (KvQuantMode::Kvarn, 4),
        "kvarn8" => (KvQuantMode::Kvarn, 8),
        _ => (KvQuantMode::Unquantized, 4),
    }
}

fn cap_gemma3_stopgap_max_seq(max_seq: usize, arch_id: u32, kv_mode: &str) -> usize {
    let is_gemma3 = arch_id == ARCH_ID_GEMMA3_TEXT || arch_id == ARCH_ID_GEMMA3_VL;
    if !is_gemma3 {
        return max_seq;
    }
    // q8/int8/kvarn global KV is ~4× smaller than F32, so with sliding-window
    // local layers the full trained context fits — no gemma3-specific cap needed
    // (the model-context clamp still applies). Only F32 global KV needs the cap.
    let quantized_global = matches!(kv_mode, "q8" | "int8" | "kvarn");
    let cap = if quantized_global {
        max_seq
    } else {
        GEMMA3_STOPGAP_MAX_SEQ
    };
    if max_seq <= cap {
        return max_seq;
    }
    if std::env::var("HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "  WARNING: gemma3 max_seq={max_seq} — the global-layer F32 KV alone may OOM the \
             shared GTT pool at this context; HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1 set — proceeding \
             (use kv_mode=q8 for full 128K)."
        );
        max_seq
    } else {
        eprintln!(
            "  NOTE: capping gemma3 max_seq {max_seq} -> {GEMMA3_STOPGAP_MAX_SEQ} to fit the GTT \
             pool (sliding-window KV bounds local layers; global F32 KV still scales with context \
             — use kv_mode=q8 for full 128K). Override: HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1."
        );
        GEMMA3_STOPGAP_MAX_SEQ
    }
}

// Auto-upgrade DeltaNet state to FP32 for low-redundancy models when the caller
// has not made an explicit non-default choice. Q8/Q4 state accumulates quality
// drift on long outputs; the recurrent state is the model's numerical anchor
// (quant error compounds across the sequence), and low-redundancy models lack
// the head capacity to absorb it. Gate keys on DeltaNet state redundancy
// (`linear_key_head_dim × linear_num_value_heads`) rather than parameter count
// — this directly measures state capacity (0.8B=2048, 9B=4096, 27B=6144) and
// scales with the actual sensitivity. Threshold: HIPFIRE_DN_STATE_FP32_BELOW
// (default usize::MAX ⇒ FP32 for all current models; see TODO.md / NEXT-STEPS).
// An explicit override (state_quant="q8"/"q4" in JSON params) is honoured.
/// Bring-up helper for the tiny smoke-test models: resolve their abbreviated
/// config into the concrete state needed to load them.
pub fn resolve_tiny_model_state(
    hfq: &HfqFile,
    override_str: Option<&str>,
    q: hipfire_arch_qwen35::qwen35::StateQuant,
) -> hipfire_arch_qwen35::qwen35::StateQuant {
    use hipfire_arch_qwen35::qwen35::{
        config_from_hfq, deltanet_state_fp32_below, deltanet_state_redundancy, StateQuant,
    };
    // "Explicit" means the caller passed a non-default token; system default
    // tokens (None, "", "auto", "q8", "int8") all count as unspecified.
    let explicit = matches!(override_str,
        Some(s) if !matches!(s.to_ascii_lowercase().as_str(), "" | "auto" | "q8" | "int8"));
    if explicit || q == StateQuant::FP32 {
        return q;
    }
    let threshold = deltanet_state_fp32_below();
    // Prefer the redundancy gate; fall back to param count if the qwen35 config
    // can't be parsed (non-hybrid artifact).
    if let Some(cfg) = config_from_hfq(hfq) {
        let redundancy = deltanet_state_redundancy(&cfg);
        if redundancy < threshold {
            eprintln!(
                "  DeltaNet state: auto-upgraded to FP32 (redundancy {redundancy} = \
                 head_dim×n_value_heads < {threshold}; recurrent state is the numerical \
                 anchor — pass state_quant=q8 to override)",
            );
            return StateQuant::FP32;
        }
        return q;
    }
    const TINY_MODEL_PARAMS: u128 = 2_000_000_000;
    let params = hfq_parameter_count(hfq);
    if params < TINY_MODEL_PARAMS {
        eprintln!(
            "  DeltaNet state: auto-upgraded to FP32 ({:.2}B params, config unparsed — \
             FP32 stable below 2B; pass state_quant=q8 to override)",
            params as f64 / 1.0e9,
        );
        StateQuant::FP32
    } else {
        q
    }
}

/// Load a model from an HFQ path + load-message params into a [`LoadedModel`]:
/// detect the arch, parse its config, upload weights and allocate the forward
/// scratch/KV/state for that family, resolve the chat template and eviction
/// policy, and wire any optional DFlash drafter. The single-GPU entry point
/// (multi-GPU goes through [`load_model_pp`]).
pub fn load_model(
    path: &str,
    max_seq: usize,
    requested_physical_cap: Option<usize>,
    draft_path: Option<&str>,
    kv_mode_override: Option<&str>,
    state_quant_override: Option<&str>,
    cask: &CaskConfig,
    pp: usize,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<LoadedModel, String> {
    if pp > 1 {
        // Refusal contracts (DFlash, CASK sidecar) are enforced upstream in
        // the "load" event handler so the operator gets a structured error
        // before any HFQ open / weight allocation. By the time we get here
        // with pp>1, draft_path is None and cask.sidecar is None.
        let _ = (draft_path, cask);
        return load_model_pp(
            path,
            max_seq,
            kv_mode_override,
            state_quant_override,
            pp,
            gpu,
        );
    }
    // Per-load kv_mode (sent in load message params) overrides the env var.
    // Lets the CLI set size-aware defaults — e.g. Qwen3.5-27B prefers asym4
    // since layer-count compounding of asym3 noise flips argmax at decision
    // boundaries on deep stacks.
    let mut kv_mode = kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
    // ─── ParoQuant / safetensors directory path ────────────────────────────
    // If the path is a directory with config.json, try loading as a
    // SafetensorsSource (ParoQuant, AWQ, etc.) instead of HFQ.
    if Path::new(path).is_dir() {
        return load_model_safetensors(path, max_seq, &kv_mode, gpu);
    }

    let mut hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let max_seq = clamp_max_seq_to_model_context(max_seq, &hfq.metadata_json);
    let max_seq = cap_gemma3_stopgap_max_seq(max_seq, hfq.arch_id, &kv_mode);
    let model_memory = hfq_model_memory(path, &hfq);
    // Optimization notice: canonical OQ4 weights (quant_type 34) are repacked
    // into the arch-combined device layout on EVERY load. `hipfire optimize`
    // prebakes that layout (34 -> 37) so later loads upload verbatim with no
    // per-load repack. Index-level scan only — no GPU work, once per load.
    let canonical_oq4 = hfq
        .tensors()
        .iter()
        .filter(|t| t.quant_type == hipfire_runtime::hfq::OQ4_CANONICAL_QT)
        .count();
    if canonical_oq4 > 0 {
        eprintln!(
            "[optimize] '{path}' has {canonical_oq4} canonical OQ4 tensor(s) repacked per load; \
             run `hipfire optimize {path}` to prebake the arch-optimal layout and skip the per-load repack"
        );
    }
    // Whether ANY tensor is BF16 — used to keep the DeltaNet *state* at FP32
    // (the recurrent state's cumulative-error sensitivity; orthogonal to KV).
    let is_bf16_artifact = hfq_has_bf16_weights(&hfq);
    // KV precision policy:
    //   * BF16-DOMINANT model (full-precision artifact) -> force fp32 KV (mixing
    //     a quantized KV under bf16 weights is a precision mismatch).
    //   * Quantized model (MQ4/Q8 weights, only norms BF16) -> honor an explicit
    //     kv_mode; otherwise default to fp32 for now. A quantized KV (q8/asym/
    //     KVarN) makes the model batched-prefill eligible (~32x prefill); the
    //     prior rule wrongly force-fp32'd these via the BF16 norms, locking them
    //     to the per-token path. Default flips to KVarN once it's a runtime mode.
    // KV precision: respect an explicit kv_mode (config/CLI/JSON) for ALL
    // models, including BF16-dominant ones. Default to fp32 only when the
    // caller left it unspecified. (Previously BF16-dominant artifacts were
    // force-overridden to fp32 even when the operator asked for q8/asym/KVarN
    // — that silently discarded the requested KV quant. fp32 remains the
    // safe default; quantizing KV under bf16 weights is now an opt-in the
    // operator owns.)
    if kv_mode.is_empty() {
        kv_mode = "fp32".to_string();
    }
    let tokenizer = hipfire_model::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;

    // DFlash speculative-decode requires the target's lm_head to have a
    // batched-GEMM kernel (used for verify and DDTree top-K). Only
    // Q8_0 (qt=3) / HFQ4G256 (qt=6) / MQ4G256 (qt=13) are wired into
    // speculative.rs's `try_batched` predicate (lines 2083-2087,
    // 2606-2609); every other dtype falls through to a per-row sequential
    // GEMV path that hangs spec verify (observed: 1 token in 240 s on
    // 27B MQ3 + mq4.dflash draft).
    //
    // Refuse fast at the HFQ-index level — BEFORE any weight upload, KV
    // alloc, or scratch alloc — so we don't strand ~12 GB of VRAM in the
    // pool when the operator passed a draft against an unsupported target.
    // Read the lm_head tensor's `quant_type` byte directly from the index
    // (no GPU work). lm_head can be a separate tensor or tied to
    // embed_tokens, and the tensor names differ by arch:
    //   - Qwen3.5/3.6 separate: "lm_head.weight" or "model.language_model.lm_head.weight"
    //   - Qwen3.5/3.6 tied:     "model.language_model.embed_tokens.weight"
    //   - LLaMA separate:       "lm_head.weight"
    //   - LLaMA tied:           "model.embed_tokens.weight"
    // Cover all four; the order mirrors what qwen35::load_weights /
    // hfq::load_weights_hfq do at runtime, so the qt we read here is the
    // qt that will end up driving `weights.output.gpu_dtype`.
    if draft_path.is_some() {
        if hfq.arch_id == ARCH_ID_LFM2_MOE {
            #[cfg(not(feature = "arch-lfm2moe"))]
            {
                return Err(
                    "LFM2 DFlash draft requested but arch-lfm2moe is not compiled in".to_string(),
                );
            }
            if std::env::var("HIPFIRE_LFM2_DFLASH").ok().as_deref() != Some("1") {
                return Err(
                    "LFM2 DFlash is experimental; set HIPFIRE_LFM2_DFLASH=1 to load a draft"
                        .to_string(),
                );
            }
            eprintln!(
                "  WARNING: LFM2 DFlash is experimental; admission is gated by HIPFIRE_LFM2_DFLASH=1"
            );
        } else {
            // Arch-level capability gate FIRST (matrix-backed). DFlash spec-decode
            // only runs on archs whose matrix marks it Full — the generate() router
            // requires it. Without this an operator could attach a draft to a
            // non-DFlash arch, pass the lm_head dtype check below, then silently get
            // plain AR decode (a no-op draft). Refuse up front with the matrix reason.
            require_arch_feature(
                hfq.arch_id,
                "DFlash spec-decode",
                arch_features(hfq.arch_id).dflash,
            )?;

            let lm_qt = hfq
                .tensor_data("lm_head.weight")
                .or_else(|| hfq.tensor_data("model.language_model.lm_head.weight"))
                .or_else(|| hfq.tensor_data("model.language_model.embed_tokens.weight"))
                .or_else(|| hfq.tensor_data("model.embed_tokens.weight"))
                .map(|(info, _)| info.quant_type);
            // MQ3 (qt=17) batched lm_head + WMMA prefill kernels exist on gfx11
            // only (`gemm_hfq3g256_batched_lmhead` + `is_batchable_la` admits MQ3
            // for gfx1100/1101/1102/1150/1151). On other archs, MQ3 lm_head still
            // falls through to per-row GEMV that hangs verify. Whitelist:
            //   - Always: Q8_0=3, HFQ4G256=6, MQ4G256=13
            //   - gfx11 only: MQ3G256=17
            // MQ2 (qt=18) is not yet wired into speculative.rs match arms.
            // MQ3 WMMA family is ported to gfx11 (RDNA3) and gfx12 (RDNA4).
            // Keep them grouped under the same flag — the builtin name differs
            // (_w32 vs _w32_gfx12) but the dispatch wrappers route per-arch.
            let arch_is_gfx11 = matches!(
                gpu.arch.as_str(),
                "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1200" | "gfx1201"
            );
            let supported = match lm_qt {
                Some(3 | 6 | 13) => true,
                Some(17) => arch_is_gfx11,
                _ => false,
            };
            if !supported {
                let qt_desc = match lm_qt {
                    Some(qt) => format!("quant_type={qt}"),
                    None => "no lm_head/embed_tokens tensor found at any known name".to_string(),
                };
                return Err(format!(
                    "DFlash draft requested but target lm_head {} is not \
                     supported by speculative.rs's batched GEMM paths on this arch \
                     ({}). Supported: Q8_0 (qt=3), HFQ4G256 (qt=6), MQ4G256 (qt=13) \
                     always; MQ3G256 (qt=17) on gfx11 only. Other dtypes \
                     (MQ2 qt=18, MQ6/MQ8, HFQ3/HFQ2, HFQ4G128, HFQ6, F16, …) fall \
                     through to a per-row GEMV that hangs verify. Reload without a \
                     draft, or use an MQ4 / HFQ4 / Q8 target. (PRD Phase 2: extend \
                     speculative.rs match arms + add gemm_*_batched_lmhead kernels \
                     for the remaining dtypes.)",
                    qt_desc, gpu.arch
                ));
            }

            // Defense-in-depth: refuse if any body weight is MQ2 (qt=18). MQ3
            // is now allowed on gfx11 dense (arch_id=5) because the WMMA prefill
            // family (qkvza/qkv/gate_up/residual hfq3) and
            // `gemm_hfq3g256_batched_lmhead` are wired. MQ3 is REFUSED on:
            //   - non-gfx11 archs (no batched WMMA prefill kernels)
            //   - MoE/A3B targets (arch_id=6) — the MoE LA/FA prefill branches
            //     and `moe_ffn_all_mq4` predicate are MQ4-only; MQ3 weights
            //     would silently fall through to HFQ4 kernels with the wrong
            //     104-vs-136 byte stride. (Future: wire MQ3 into the MoE
            //     batched branches and the MoE FFN expert kernels.)
            // MQ2 body still has no batched WMMA kernels anywhere.
            let arch_is_dense_qwen35 = is_qwen35_dense_arch_id(hfq.arch_id);
            let mq3_supported = arch_is_gfx11 && arch_is_dense_qwen35;
            let mq_unsupported = hfq
                .first_tensor_with_quant_type(18)
                .map(|n| ("MQ2 (qt=18)", n));
            let mq_unsupported = mq_unsupported.or_else(|| {
                if !mq3_supported {
                    hfq.first_tensor_with_quant_type(17)
                        .map(|n| ("MQ3 (qt=17)", n))
                } else {
                    None
                }
            });
            if let Some((qt_label, name)) = mq_unsupported {
                let arch_reason = if !arch_is_dense_qwen35 && qt_label.starts_with("MQ3") {
                    format!(
                        "arch_id={} (MoE/A3B-class) has no MQ3 MoE kernels",
                        hfq.arch_id
                    )
                } else {
                    format!(
                        "arch={} lacks the corresponding batched WMMA prefill family",
                        gpu.arch
                    )
                };
                return Err(format!(
                    "DFlash draft requested but model contains {qt_label} weight \
                     `{name}` and {arch_reason}. The prefill fast-path falls back \
                     to per-token `forward_scratch` for every spec verify cycle \
                     (or worse, a kernel-stride mismatch on MoE) — defeating \
                     DFlash's speedup. Reload without a draft, or use an MQ4 / \
                     HFQ4 / Q8 target. (Future: port MQ3/MQ2 to MoE branches and \
                     additional archs.)"
                ));
            }
        }
    }

    // Derive physical_cap. With eviction (cask.sidecar set), the physical
    // buffer only needs to hold budget+beta+safety slots; max_seq is the
    // advertised window the client targets. Without eviction, the server may
    // still request a smaller initial allocation and reload a larger worker
    // on demand; max_seq remains the logical context-window limit.
    //
    // The `HIPFIRE_KV_PHYSICAL_CAP` env var is an explicit operator override —
    // useful for ablations or reproducing dflash_spec_demo settings.
    let physical_cap = if cask.sidecar.is_some() {
        let env_override = std::env::var("HIPFIRE_KV_PHYSICAL_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let safety = 256usize;
        let floor = cask.budget + cask.beta + 4;
        let derived = cask.budget + cask.beta + safety;
        env_override.unwrap_or(derived).clamp(floor, max_seq)
    } else {
        let requested = requested_physical_cap
            .or_else(|| {
                std::env::var("HIPFIRE_KV_PHYSICAL_CAP")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .unwrap_or(max_seq);
        requested.clamp(512.min(max_seq), max_seq)
    };

    if hfq.arch_id == ARCH_ID_QWEN2 {
        // Qwen2 dense (hipfire-arch-qwen2). Standalone bring-up — no
        // eviction, no DFlash, no PFlash, no VL. The Architecture
        // trait surface gives us config + weights + state in three
        // calls; forward is direct `qwen2::forward_step` below.
        if draft_path.is_some() {
            return Err(
                "DFlash not supported on arch_id=7 (hipfire-arch-qwen2 bring-up). \
                       Reload without a draft."
                    .to_string(),
            );
        }
        if cask.sidecar.is_some() {
            return Err(
                "CASK eviction not supported on arch_id=7 (hipfire-arch-qwen2 bring-up). \
                       Reload without --cask-sidecar."
                    .to_string(),
            );
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        use hipfire_arch_qwen2::Qwen2;
        use hipfire_runtime::arch::Architecture;
        let config = <Qwen2 as Architecture>::config_from_hfq(&hfq)?;
        let weights = <Qwen2 as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        let state = qwen2::Qwen2State::new_with_max_seq(gpu, &config, max_seq)
            .map_err(|e| format!("qwen2: Qwen2State::new_with_max_seq failed: {e:?}"))?;
        let qwen2_backend = hipfire_arch_qwen2::Qwen2Backend::new(config, weights, state);
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            qwen2_backend: Some(qwen2_backend),
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_ZAYA {
        // ZAYA1 (CCA attention + EDA/MoD-routed MoE). Served through the shared
        // ServingBackend seam on ZayaModel (O(1) KV-cache decode).
        if draft_path.is_some() {
            return Err("DFlash not supported on arch_id=16 (zaya).".to_string());
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("zaya metadata parse: {e}"))?;
        let cfg_json = meta
            .get("config")
            .ok_or("zaya: metadata_json missing 'config'")?;
        let cfg = hipfire_arch_zaya::ZayaConfig::from_json(cfg_json)
            .map_err(|e| format!("zaya config: {e}"))?;
        eprintln!(
            "  zaya: hidden={}, blocks={}, experts={}, vocab={}, eos={}",
            cfg.hidden_size, cfg.num_blocks, cfg.moe.num_experts, cfg.vocab_size, cfg.eos_token_id,
        );
        let model = hipfire_arch_zaya::arch::ZayaModel::from_hfq(gpu, &hfq, cfg, max_seq)
            .map_err(|e| format!("ZayaModel::from_hfq: {e}"))?;
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: Some(model),
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            qwen2_backend: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_NEMOTRON_H || hfq.arch_id == ARCH_ID_MAMBA2 {
        // nemotron_h (hybrid Mamba-2 + attention/MLP/MoE) and pure Mamba-2
        // from quantized (or bf16) .hfq artifacts, driven through the same
        // Mamba-capable ServingBackend seam.
        if draft_path.is_some() {
            return Err(format!(
                "DFlash not supported on arch_id={} ({}). Reload without a draft.",
                hfq.arch_id,
                if hfq.arch_id == ARCH_ID_MAMBA2 {
                    "mamba2"
                } else {
                    "nemotron_h"
                }
            ));
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("nemotron metadata parse: {e}"))?;
        let cfg_json = meta
            .get("config")
            .ok_or("nemotron: metadata_json missing 'config'")?;
        let mut cfg = if hfq.arch_id == ARCH_ID_MAMBA2 {
            hipfire_arch_nemotron::NemotronHConfig::from_mamba2_json(cfg_json)
                .map_err(|e| format!("mamba2 config: {e}"))?
        } else {
            hipfire_arch_nemotron::NemotronHConfig::from_json(cfg_json)
                .map_err(|e| format!("nemotron config: {e}"))?
        };
        if hfq.arch_id == ARCH_ID_MAMBA2 {
            if let Some(eot) = tokenizer.special_token_id("<|endoftext|>") {
                cfg.eos_token_id = eot;
            }
        } else if let Some(im_end) = tokenizer.special_token_id("<|im_end|>") {
            // Chat serving stops on the ChatML turn delimiter `<|im_end|>`, not
            // the base `eos_token_id` (`</s>` = 2 for Nano).
            cfg.eos_token_id = im_end;
        }
        eprintln!(
            "  {}: hidden={}, layers={} ({} M / {} * / {} - / {} E), vocab={}, eos={}",
            if hfq.arch_id == ARCH_ID_MAMBA2 {
                "mamba2"
            } else {
                "nemotron_h"
            },
            cfg.hidden_size,
            cfg.num_layers,
            cfg.count(hipfire_arch_nemotron::BlockKind::Mamba2),
            cfg.count(hipfire_arch_nemotron::BlockKind::Attention),
            cfg.count(hipfire_arch_nemotron::BlockKind::Mlp),
            cfg.count(hipfire_arch_nemotron::BlockKind::Moe),
            cfg.vocab_size,
            cfg.eos_token_id,
        );
        let model = hipfire_arch_nemotron::model::NemotronModel::from_hfq(gpu, &hfq, cfg, max_seq)
            .map_err(|e| format!("mamba-capable NemotronModel::from_hfq: {e}"))?;
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: Some(model),
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            qwen2_backend: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_GEMMA3_TEXT {
        // Gemma3 text (medgemma-*-text). Plain dense-AR: the gemma3 decoder +
        // its own decode state in `Gemma3Backend`, served via the same
        // `ServingBackend::serve` seam (delegates to `run_simple_ar`). No
        // vision, eviction, DFlash, CASK, or PP.
        if draft_path.is_some() {
            return Err(
                "DFlash not supported on arch_id=12 (gemma3 text). Reload without a draft."
                    .to_string(),
            );
        }
        if cask.sidecar.is_some() {
            return Err("CASK eviction not supported on arch_id=12 (gemma3 text). \
                       Reload without --cask-sidecar."
                .to_string());
        }
        // gemma3 KV: F32 by default; honor an explicit q8/int8/kvarn kv_mode
        // (all ~4x smaller than F32, letting larger contexts fit). Other quant
        // modes (asym/fwht) have no gemma3 kernel yet and fall back to F32.
        let (kv_mode_g3, kvarn_bits_g3) = gemma3_kv_mode(&kv_mode);
        let _ = state_quant_override;
        let cfg = hipfire_arch_gemma3::config_from_hfq(&hfq)
            .ok_or_else(|| "gemma3: failed to parse Gemma3Config".to_string())?;
        let weights = hipfire_arch_gemma3::load_weights(&mut hfq, &cfg, gpu)
            .map_err(|e| format!("gemma3: load_weights failed: {e:?}"))?;
        let state = Gemma3State::new_with_max_seq(gpu, &cfg, max_seq, kv_mode_g3, kvarn_bits_g3)
            .map_err(|e| format!("gemma3: Gemma3State::new_with_max_seq failed: {e:?}"))?;
        let backend = Gemma3Backend::new(cfg, weights, state);
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            qwen2_backend: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: Some(backend),
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_GEMMA3_VL {
        // Gemma3-VL (medgemma). Self-contained multimodal backend: the gemma3
        // text decoder (loaded from the `language_model.` prefix) + the SigLIP
        // vision tower + the projector, plus its own decode state — all owned by
        // `Gemma3VlBackend`, which serves via `ServingBackend::serve` →
        // `decode_loop` (greedy). No eviction / DFlash / CASK / PP, and not the
        // qwen35-VL `vision_config` splice path (that field stays None for 13;
        // the `has_vl` gate keys off `gemma3_vl.is_some()`).
        if draft_path.is_some() {
            return Err(
                "DFlash not supported on arch_id=13 (gemma3-vl). Reload without a draft."
                    .to_string(),
            );
        }
        if cask.sidecar.is_some() {
            return Err("CASK eviction not supported on arch_id=13 (gemma3-vl). \
                       Reload without --cask-sidecar."
                .to_string());
        }
        // gemma3-vl KV: F32 by default; honor an explicit q8/int8/kvarn kv_mode.
        // Lets medgemma run a much larger context before exhausting the GTT pool.
        let (kv_mode_g3, kvarn_bits_g3) = gemma3_kv_mode(&kv_mode);
        let _ = state_quant_override;
        let LoadedVl {
            text_cfg,
            vl_cfg,
            weights,
            vision_tier,
            vision_source_id,
        } = load_vl(&mut hfq, gpu)?;
        let state =
            Gemma3State::new_with_max_seq(gpu, &text_cfg, max_seq, kv_mode_g3, kvarn_bits_g3)
                .map_err(|e| format!("gemma3-vl: Gemma3State::new_with_max_seq failed: {e:?}"))?;
        let backend = Gemma3VlBackend::new(
            text_cfg,
            vl_cfg,
            weights,
            state,
            vision_tier,
            vision_source_id,
        );
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            qwen2_backend: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: Some(backend),
            gemma3_text: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_DOTS_OCR {
        // dots.ocr (Qwen2-VL family). Text decoder is Qwen2; vision tower
        // is the 42-block DotsVisionTransformer. Both load side-by-side in
        // DotsOcrWeights and stay resident. Single-image, greedy decode at
        // bring-up — no eviction, DFlash, CASK, or PP.
        if draft_path.is_some() {
            return Err(
                "DFlash not supported on arch_id=8 (dots.ocr). Reload without a draft.".to_string(),
            );
        }
        if cask.sidecar.is_some() {
            return Err("CASK eviction not supported on arch_id=8 (dots.ocr). Reload without --cask-sidecar.".to_string());
        }
        if pp > 1 {
            return Err(
                "pipeline-parallel (pp>1) not supported on arch_id=8 (dots.ocr).".to_string(),
            );
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        use hipfire_arch_dots_ocr::DotsOcr;
        use hipfire_runtime::arch::Architecture;
        let config = <DotsOcr as Architecture>::config_from_hfq(&hfq)?;
        let weights = <DotsOcr as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        // Size the decode KV cache to the requested window (the trait's
        // new_state uses a default max_seq; OCR prompts are long).
        let state = qwen2::Qwen2State::new_with_max_seq(gpu, &config.text, max_seq)
            .map_err(|e| format!("dots-ocr: Qwen2State::new_with_max_seq failed: {e:?}"))?;
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: Some(state),
            qwen2_backend: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: Some(config),
            dots_ocr_weights: Some(weights),
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_DEEPSEEK4_FLASH {
        // DeepSeek V4 Flash (hipfire-arch-deepseek4). Standalone bring-up —
        // no eviction, no DFlash drafter, no PFlash, no VL. The
        // Architecture trait gives us config + weights + state in three
        // calls; forward goes through `deepseek4::forward::forward_prefill_*` /
        // `decode_step` in the generate hot path.
        if draft_path.is_some() {
            return Err("DFlash not supported on arch_id=9 (DeepSeek V4 Flash). \
                       Reload without a draft."
                .to_string());
        }
        if cask.sidecar.is_some() {
            return Err(
                "CASK eviction not supported on arch_id=9 (DeepSeek V4 Flash). \
                       Reload without --cask-sidecar."
                    .to_string(),
            );
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        use hipfire_runtime::arch::Architecture;
        let config = <deepseek4::DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
        let weights =
            <deepseek4::DeepseekV4 as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        let state = deepseek4::DeepseekV4State::new(&config)?;
        // Pre-allocate PrefillBatchScratch. Default B=1024 (bumped from 64
        // on 2026-05-26). PP_BATCH sweep on the 2.1k-tok bench (3 trials/cell):
        //   PP=256: 46.4 tps   PP=512: 48.3 tps
        //   PP=1024: 49.3 tps  PP=2048: 49.0 tps
        // 1024 captures the L2-amortization peak; 2048 plateaus from PBS
        // memory footprint exceeding effective L2/Inf-cache reuse window.
        // PBS sits in (UMA) GPU memory for the model's lifetime — ~600 MB
        // at B=1024 on V4-Flash, well within 128 GB. Override via
        // HIPFIRE_DEEPSEEK4_PP_BATCH.
        let pbs_max_batch: usize = std::env::var("HIPFIRE_DEEPSEEK4_PP_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let pbs = deepseek4::forward::PrefillBatchScratch::new(gpu, &config, pbs_max_batch)?;
        // Cache EOS token id. DeepSeek family uses `<｜end▁of▁sentence｜>`;
        // fall back to 1 if tokenizer lacks the entry.
        let eos_tok: u32 = {
            let ids = tokenizer.encode("<｜end▁of▁sentence｜>");
            if ids.len() == 1 {
                ids[0]
            } else {
                1
            }
        };
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            qwen2_backend: None,
            deepseek4_config: Some(config),
            deepseek4_weights: Some(weights),
            deepseek4_state: Some(state),
            deepseek4_pbs: Some(pbs),
            deepseek4_eos_tok: eos_tok,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_MINIMAX_M2 {
        // MiniMax-M2 (hipfire-arch-minimax). Standalone bring-up — no
        // eviction, no DFlash drafter, no PFlash, no VL, no spec-decode.
        // The Architecture trait gives us config + weights + state in three
        // calls; prefill + decode both go through the per-token
        // `minimax::forward::decode_step` in the generate hot path. There
        // is NO PrefillBatchScratch (no batched prefill kernel).
        if draft_path.is_some() {
            return Err("DFlash not supported on arch_id=10 (MiniMax-M2). \
                       Reload without a draft."
                .to_string());
        }
        if cask.sidecar.is_some() {
            return Err("CASK eviction not supported on arch_id=10 (MiniMax-M2). \
                       Reload without --cask-sidecar."
                .to_string());
        }
        if pp > 1 {
            return Err(
                "pipeline-parallel (pp>1) not supported on arch_id=10 (MiniMax-M2).".to_string(),
            );
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        use hipfire_runtime::arch::Architecture;
        let config = <minimax::MiniMaxM2 as Architecture>::config_from_hfq(&hfq)?;
        let weights = <minimax::MiniMaxM2 as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        // Size the KV cache to the requested window (the trait's new_state
        // caps at 8192; honour the caller's max_seq when it's larger/smaller).
        let state = minimax::MiniMaxState::new_with_max_seq(gpu, &config, max_seq)
            .map_err(|e| format!("minimax: MiniMaxState::new_with_max_seq failed: {e}"))?;
        // Resolve EOS via the tokenizer. MiniMax-M2 does NOT use ChatML —
        // its end-of-turn marker is the added token `[e~[` (id 200020 in the
        // 200k vocab; tokenizer_config.json eos_token = `[e~[`,
        // generation_config.json eos_token_id = 200020). The earlier ChatML
        // probes (`<|im_end|>` etc.) are absent from this vocab and silently
        // fell back to token 1, so generate_minimax never hit EOS: every turn
        // ran to max_tokens and the model spammed `[e~[` trying to end the
        // turn. Probe the real marker first; keep the ChatML fallbacks for
        // safety on any future tokenizer variant.
        let eos_tok: u32 = {
            let try_one = |s: &str| -> Option<u32> {
                let ids = tokenizer.encode(s);
                if ids.len() == 1 {
                    Some(ids[0])
                } else {
                    None
                }
            };
            try_one("[e~[")
                .or_else(|| try_one("<|im_end|>"))
                .or_else(|| try_one("</s>"))
                .or_else(|| try_one("<|endoftext|>"))
                .unwrap_or(1)
        };
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            qwen2_backend: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: Some(config),
            minimax_weights: Some(weights),
            minimax_state: Some(state),
            minimax_eos_tok: eos_tok,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_LFM2_MOE {
        // LFM2.5-8B-A1B (hipfire-arch-lfm2moe). Standalone bring-up — no
        // DFlash drafter, no PFlash, no VL, no spec-decode. CASK/TriAttention
        // is wired through the shared KvCache using attention-ordinal sidecar
        // indices (0..num_attention_layers), because LFM2 stores KV only for
        // attention layers rather than allocating one KV slot per model layer.
        // Hybrid LIV short-conv + GQA attention feeding a top-4 MoE FFN.
        // config + weights + state come from the crate's direct API
        // (it does not implement the Architecture trait); generate prefill
        // routes through `lfm2moe::forward::prefill_batch` and decode through
        // `lfm2moe::forward::decode_step`. Structurally mirrors MiniMax (10)
        // while carrying arch-local prefill scratch in the forward module.
        #[cfg(not(feature = "arch-lfm2moe"))]
        {
            let _ = (
                &mut hfq,
                path,
                max_seq,
                draft_path,
                kv_mode,
                state_quant_override,
                &cask,
                pp,
                gpu,
                tokenizer,
            );
            return Err(
                "lfm2moe arch (id 11) not compiled in (enable feature arch-lfm2moe)".to_string(),
            );
        }
        #[cfg(feature = "arch-lfm2moe")]
        {
            if pp > 1 {
                return Err(
                    "pipeline-parallel (pp>1) not supported on arch_id=11 (LFM2.5-MoE)."
                        .to_string(),
                );
            }
            if draft_path.is_some() && cask.sidecar.is_some() {
                return Err(
                    "LFM2 DFlash does not support CASK/TriAttention eviction yet; \
                     reload without the draft or without the CASK sidecar"
                        .to_string(),
                );
            }
            let _ = kv_mode;
            let _ = state_quant_override;
            let config = lfm2moe::config::Lfm2MoeConfig::from_hfq(&hfq)?;
            let weights = lfm2moe::lfm2moe::Lfm2MoeWeights::load(&mut hfq, &config, gpu)?;
            // Size the KV + conv-state cache to the requested window.
            let state = lfm2moe::lfm2moe::Lfm2MoeState::new_with_physical_cap(
                gpu,
                &config,
                max_seq,
                physical_cap,
            )
            .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_physical_cap failed: {e}"))?;
            let lfm2_dflash = if let Some(dp) = draft_path {
                Some(load_lfm2_dflash_state(
                    dp,
                    physical_cap,
                    &config,
                    &state,
                    gpu,
                )?)
            } else {
                None
            };
            let eviction = if let Some(ref sidecar_path) = cask.sidecar {
                let centers = TriAttnCenters::load(Path::new(sidecar_path)).map_err(|e| {
                    use std::io::ErrorKind;
                    let p = Path::new(sidecar_path);
                    let why = match e.kind() {
                        ErrorKind::NotFound if p.symlink_metadata().is_ok() => {
                            format!("dangling symlink (target absent): {sidecar_path}")
                        }
                        ErrorKind::NotFound => format!("file not found: {sidecar_path}"),
                        ErrorKind::InvalidData => format!("bad format ({e}): {sidecar_path}"),
                        ErrorKind::UnexpectedEof => {
                            format!("truncated/corrupt sidecar: {sidecar_path}")
                        }
                        _ => format!("read error ({e}): {sidecar_path}"),
                    };
                    format!("lfm2moe cask sidecar load failed — {why} (regen: hipfire sidecar-gen, or HIPFIRE_CASK_OFF=1)")
                })?;
                let n_attn = config.num_attention_layers();
                if centers.n_layers != n_attn
                    || centers.n_heads != config.num_attention_heads
                    || centers.head_dim != config.head_dim
                {
                    return Err(format!(
                        "lfm2moe cask sidecar shape mismatch: sidecar layers={} heads={} head_dim={} but model attention_layers={} heads={} head_dim={}. \
                         LFM2 sidecars must be calibrated with attention-ordinal layer indices.",
                        centers.n_layers,
                        centers.n_heads,
                        centers.head_dim,
                        n_attn,
                        config.num_attention_heads,
                        config.head_dim
                    ));
                }
                if (centers.rope_theta - config.rope_theta).abs()
                    > (config.rope_theta.abs().max(1.0) * 1e-4)
                {
                    return Err(format!(
                        "lfm2moe cask sidecar rope_theta mismatch: sidecar={} model={}",
                        centers.rope_theta, config.rope_theta
                    ));
                }
                let fa_layer_ids = lfm2_triattn_kv_layer_ids(&config);
                let base = EvictionCtx::new(
                    gpu,
                    &centers,
                    fa_layer_ids,
                    cask.budget,
                    cask.beta,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.head_dim,
                    config.rope_theta,
                    physical_cap,
                )
                .map_err(|e| format!("lfm2moe build EvictionCtx: {e}"))?;
                if cask.cask_m_folding {
                    eprintln!(
                        "  lfm2moe eviction: CASK α={:.2} m={} budget={} β={} physical_cap={}",
                        cask.core_frac, cask.fold_m, cask.budget, cask.beta, physical_cap,
                    );
                    Some(Eviction::Cask(CaskCtx::new(
                        base,
                        cask.core_frac,
                        cask.fold_m,
                    )))
                } else {
                    eprintln!(
                        "  lfm2moe eviction: TriAttention (plain drop) budget={} β={} physical_cap={}",
                        cask.budget, cask.beta, physical_cap,
                    );
                    Some(Eviction::Plain(base))
                }
            } else {
                None
            };
            // Resolve EOS via the tokenizer. LFM2.5 uses the standard
            // ChatML-ish `<|im_end|>`; fall back to common alternates, then 1.
            let eos_tok: u32 = {
                let try_one = |s: &str| -> Option<u32> {
                    let ids = tokenizer.encode(s);
                    if ids.len() == 1 {
                        Some(ids[0])
                    } else {
                        None
                    }
                };
                try_one("<|im_end|>")
                    .or_else(|| try_one("</s>"))
                    .or_else(|| try_one("<|endoftext|>"))
                    .unwrap_or(1)
            };
            let chat_template = resolve_chat_template(&hfq, path);
            let (chat_template, chat_template_profile) =
                profile_chat_template(chat_template, Some(&tokenizer));
            return Ok(LoadedModel {
                arch_id: hfq.arch_id,
                pp: 1,
                pp_gpus: None,
                pp_scratch_set: None,
                pp_dn_la_to_device: None,
                q35_config: None,
                q35_weights: None,
                q35_scratch: None,
                q35_kv_mode: None,
                q35_state_quant: None,
                q35_registry: SessionRegistry::default(),
                llama_config: None,
                llama_weights: None,
                llama_scratch: None,
                llama_kv: None,
                llama_backend: None,
                nemotron_backend: None,
                zaya_backend: None,
                qwen2_config: None,
                qwen2_weights: None,
                qwen2_state: None,
                qwen2_backend: None,
                deepseek4_config: None,
                deepseek4_weights: None,
                deepseek4_state: None,
                deepseek4_pbs: None,
                deepseek4_eos_tok: 0,
                mtp_mode: "auto".to_string(),
                mtp_k: 3,
                mtp_weights_present: false,
                minimax_config: None,
                minimax_weights: None,
                minimax_state: None,
                minimax_eos_tok: 0,
                lfm2moe_config: Some(config),
                lfm2moe_weights: Some(weights),
                lfm2_registry: SessionRegistry {
                    sessions: std::collections::HashMap::new(),
                    active_session_id: Some(crate::session::LFM2_LEGACY_SESSION_ID.to_string()),
                    allocation_epoch: next_qwen35_state_allocation_epoch(),
                },
                lfm2moe_eos_tok: eos_tok,
                dots_ocr_config: None,
                dots_ocr_weights: None,
                vision_config: None,
                vision_weights: None,
                gemma3_vl: None,
                gemma3_text: None,
                tokenizer: Some(tokenizer),
                active: ResidentSession {
                    lfm2moe_state: Some(state),
                    ..Default::default()
                },
                max_seq,
                physical_cap,
                eviction,
                asst_turn_cache: std::collections::HashMap::new(),
                decoded_vocab: None,
                model_path: path.to_string(),
                memory: model_memory,
                #[cfg(feature = "arch-lfm2moe")]
                lfm2_dflash,
                dflash: None,
                dspark: None,
                chat_template,
                chat_template_profile,
            });
        }
    }

    if is_qwen35_family_arch_id(hfq.arch_id) {
        // Qwen3.5 DeltaNet (arch=5 dense, arch=6 MoE/A3B). PR 8: dispatch
        // through the `Architecture` trait for the bring-up triple
        // (config → load → state). Forward passes below still call
        // `qwen35::*` directly — see crates/hipfire-arch-qwen35/src/arch.rs
        // for why static dispatch wins for the hot path.
        use hipfire_arch_qwen35::Qwen35;
        use hipfire_arch_qwen35_vl::Qwen35Vl;
        use hipfire_runtime::arch::Architecture;
        let config = <Qwen35 as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;

        // Detect VL model: vision_config presence (from HFQ metadata) AND
        // actual vision tensors are required. Text-only Qwen3.5 models can
        // have vision_config in metadata without the patch_embed weights.
        // PR 9: bring-up triple now goes through the Qwen35Vl trait impl;
        // forward (`qwen35_vl::vision_forward`) stays a direct static call.
        let has_vision_tensors = hfq
            .tensor_data("model.visual.patch_embed.proj.weight")
            .is_some();
        let vision_config = <Qwen35Vl as Architecture>::config_from_hfq(&hfq).ok();
        let (vision_config, vision_weights) = if let Some(vc) = vision_config {
            if has_vision_tensors {
                let vw = <Qwen35Vl as Architecture>::load_weights(&mut hfq, &vc, gpu)
                    .map_err(|e| e.to_string())?;
                eprintln!(
                    "  VL model: vision encoder (hidden={}, layers={})",
                    vc.hidden_size, vc.num_layers
                );
                (Some(vc), Some(vw))
            } else {
                (None, None) // text-only model, no vision tensors
            }
        } else {
            (None, None)
        };

        let weights = <Qwen35 as Architecture>::load_weights(&mut hfq, &config, gpu)?;

        // MMQ per-weight screening (#87): pre-screen all weight matrices at
        // load time so the first prefill doesn't pay the screening overhead.
        // Results are cached by device pointer in gpu.mmq_screen_cache.
        // Disabled by default on all arches; opt-in via mmq_screen=true or
        // HIPFIRE_MMQ_SCREEN=1. gfx906 is included for the opt-in case so
        // its ~700 µs/weight screening-reference dispatch doesn't surprise
        // first prefill if a user enables it.
        if gpu.mmq_screen
            && matches!(
                gpu.arch.as_str(),
                "gfx906"
                    | "gfx1100"
                    | "gfx1101"
                    | "gfx1102"
                    | "gfx1103"
                    | "gfx1150"
                    | "gfx1151"
                    | "gfx1152"
            )
        {
            let t0 = std::time::Instant::now();
            let (n_safe, n_unsafe) = screen_weights_qwen35(&weights, gpu);
            let elapsed = t0.elapsed();
            eprintln!(
                "  MMQ screening: {n_safe} safe, {n_unsafe} unsafe (threshold={:.2}, {:.1}ms)",
                gpu.mmq_screen_threshold,
                elapsed.as_secs_f64() * 1000.0,
            );
        }

        // KV cache modes (RotorQuant-style asymmetric: K rotated + V Q8):
        //   asym3 (default) — K at 3-bit rotated, V at Q8_0. 5.5× vs fp32.
        //                     Best quality/compression tradeoff — RotorQuant "planar3".
        //   asym4 — K at 4-bit rotated, V at Q8_0. 5.1× (slightly safer).
        //   asym2 — K at 2-bit rotated, V at Q8_0. 6.0× (loses rare-token tail).
        //   q8    — K+V both Q8_0. 3.76× (reference quality).
        //
        // Legacy "turbo{2,3,4}" aliases map to asym{2,3,4} for backward compat.
        //
        // All allocators go through the `_capped` entry points with
        // physical_cap derived above. Without eviction, physical_cap may still
        // be smaller than max_seq; the server reloads a larger worker when a
        // request needs more physical rows.
        let is_kv_layer = crate::session::qwen35_mixer_profile(&config.layer_types).kv_layer_mask();
        let kv = match kv_mode.as_str() {
            "fp32" | "f32" => kv::KvCache::new_gpu_capped_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                physical_cap,
            )
            .map_err(|e| format!("{e}"))?,
            "q8" => {
                eprintln!("  KV cache: Q8");
                kv::KvCache::new_gpu_q8_capped(
                    gpu,
                    config.n_layers,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq,
                    physical_cap,
                )
                .map_err(|e| format!("{e}"))?
            }
            "asym4" | "turbo4" => kv::KvCache::new_gpu_asym4_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                physical_cap,
            )
            .map_err(|e| format!("{e}"))?,
            "kvarn" => kv::KvCache::new_gpu_kvarn_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                physical_cap,
                4,
            )
            .map_err(|e| format!("{e}"))?,
            "asym2" | "turbo2" => kv::KvCache::new_gpu_asym2_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                physical_cap,
            )
            .map_err(|e| format!("{e}"))?,
            "asym3" | "turbo3" | "turbo" | "auto" | "" => kv::KvCache::new_gpu_asym3_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                physical_cap,
            )
            .map_err(|e| format!("{e}"))?,
            other => {
                eprintln!("  KV cache: unrecognized '{other}', defaulting to asym3");
                kv::KvCache::new_gpu_asym3_capped(
                    gpu,
                    config.n_layers,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq,
                    physical_cap,
                )
                .map_err(|e| format!("{e}"))?
            }
        };
        // Q8 DeltaNet state can accumulate quality drift on long generation.
        // The load-time override exists for coherence A/B probes.
        let dn_quant = if is_bf16_artifact {
            hipfire_arch_qwen35::qwen35::StateQuant::FP32
        } else {
            let parsed = parse_state_quant(state_quant_override)?;
            resolve_tiny_model_state(&hfq, state_quant_override, parsed)
        };
        eprintln!("  DeltaNet state: {}", state_quant_label(dn_quant));
        let dn =
            DeltaNetState::new_with_quant(gpu, &config, dn_quant).map_err(|e| format!("{e}"))?;
        // Flash partials size with physical_cap (bounds the max_tiles the
        // flash kernel must address). When physical_cap == max_seq this is
        // identical to sizing-by-max_seq; otherwise it follows the worker's
        // current physical allocation.
        // Keep the request default at 128, but allocate enough history for
        // clients that explicitly ask for a wider repeat / OpenAI penalty
        // window.
        let scratch = qwen35::Qwen35Scratch::new_with_kv_max(gpu, &config, 2048, physical_cap)
            .map_err(|e| format!("{e}"))?;

        // Build eviction policy if the operator supplied a sidecar. Qwen3 (arch_id < 5)
        // lacks the FA/LA hybrid wiring TriAttention needs, so sidecars only take
        // effect on arch_id 5/6 — see the cask.rs docs for why CASK targets full-
        // attention layers only.
        let eviction = if let Some(ref sidecar_path) = cask.sidecar {
            let centers = TriAttnCenters::load(Path::new(sidecar_path)).map_err(|e| {
                use std::io::ErrorKind;
                let p = Path::new(sidecar_path);
                let why = match e.kind() {
                    // os error 2: open failed. Disambiguate missing vs dangling symlink.
                    ErrorKind::NotFound if p.symlink_metadata().is_ok() =>
                        format!("dangling symlink (target absent): {sidecar_path}"),
                    ErrorKind::NotFound => format!("file not found: {sidecar_path}"),
                    ErrorKind::InvalidData => format!("bad format ({e}): {sidecar_path}"),
                    ErrorKind::UnexpectedEof => format!("truncated/corrupt sidecar: {sidecar_path}"),
                    _ => format!("read error ({e}): {sidecar_path}"),
                };
                format!("cask sidecar load failed — {why} (regen: hipfire sidecar-gen, or HIPFIRE_CASK_OFF=1)")
            })?;
            let fa_layer_ids =
                crate::session::qwen35_mixer_profile(&config.layer_types).kv_layer_indices();
            if fa_layer_ids.is_empty() {
                eprintln!("  cask_sidecar set but model has no FullAttention layers — ignoring");
                None
            } else {
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                let base = EvictionCtx::new(
                    gpu,
                    &centers,
                    fa_layer_ids,
                    cask.budget,
                    cask.beta,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                    physical_cap,
                )
                .map_err(|e| format!("build EvictionCtx: {e}"))?;
                if cask.cask_m_folding {
                    eprintln!(
                        "  eviction: CASK α={:.2} m={} budget={} β={} physical_cap={}",
                        cask.core_frac, cask.fold_m, cask.budget, cask.beta, physical_cap,
                    );
                    Some(Eviction::Cask(CaskCtx::new(
                        base,
                        cask.core_frac,
                        cask.fold_m,
                    )))
                } else {
                    eprintln!(
                        "  eviction: TriAttention (plain drop) budget={} β={} physical_cap={}",
                        cask.budget, cask.beta, physical_cap,
                    );
                    Some(Eviction::Plain(base))
                }
            }
        } else {
            None
        };
        // Optional DFlash draft: load the draft model's weights + a fresh set
        // of per-cycle scratch buffers (hidden ring, verify scratch, GdnTape,
        // DeltaNetSnapshot) sized for the target's max_seq. If the draft file
        // is missing or arch-mismatched, we log and continue without DFlash
        // (temp==0 requests will fall back to AR sampling).
        let dflash = if let Some(dp) = draft_path {
            // DFlash state (hidden_rb + target_hidden_host) sizes linearly with
            // the ctx_capacity argument. Pass `physical_cap` instead of
            // `max_seq` so eviction's smaller buffer caps VRAM: a 128K-advertised
            // model with physical_cap=896 allocates an 896-slot ring, not 128K.
            // Without eviction, callers may now choose physical_cap < max_seq;
            // pass the actual allocation size so the draft ring matches.
            match load_dflash_state(dp, physical_cap, &config, &dn, gpu) {
                Ok(state) => {
                    eprintln!(
                        "  DFlash draft loaded: {} (layers={}, hidden={}, block={})",
                        dp,
                        state.draft_config.n_layers,
                        state.draft_config.hidden,
                        state.draft_config.block_size,
                    );
                    Some(state)
                }
                Err(e) => {
                    eprintln!(
                        "  DFlash draft load failed ({}): {} — falling back to AR only",
                        dp, e
                    );
                    None
                }
            }
        } else {
            None
        };

        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        let sequence_state = Some(SequenceState::new(
            crate::session::qwen35_mixer_profile(&config.layer_types),
            Some(kv),
            Some(Box::new(dn)),
        ));
        Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: Some(config),
            q35_weights: Some(weights),
            q35_scratch: Some(scratch),
            q35_kv_mode: Some(kv_mode.clone()),
            q35_state_quant: Some(dn_quant),
            q35_registry: SessionRegistry {
                sessions: std::collections::HashMap::new(),
                active_session_id: Some(QWEN35_LEGACY_SESSION_ID.to_string()),
                allocation_epoch: next_qwen35_state_allocation_epoch(),
            },
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            qwen2_backend: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config,
            vision_weights,
            gemma3_vl: None,
            gemma3_text: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession {
                sequence_state,
                ..Default::default()
            },
            max_seq,
            physical_cap,
            eviction,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash,
            dspark: None,
            chat_template,
            chat_template_profile,
        })
    } else {
        // Qwen3 / LLaMA — no eviction supported on this path (TriAttention needs
        // the FA/LA hybrid wiring from arch_id 5/6). physical_cap == max_seq.
        // PR 11: dispatch through the `Architecture` trait for the bring-up
        // triple (config → load → scratch). Forward passes below still call
        // `llama::*` directly — see crates/hipfire-arch-llama/src/arch.rs
        // for why static dispatch wins for the hot path.
        use hipfire_runtime::arch::Architecture;
        let config = <Llama as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
        let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        // Honor the requested kv_mode. LLaMA-family head_dim is typically 64/128,
        // so the rotated-K asym{2,3,4} caches (head_dim ∈ {128,256}) do not apply
        // here; surface that as an explicit error rather than silently building a
        // Q8 cache under an asym label. Supported menu: fp32 | q8.
        let kv = match kv_mode.as_str() {
            "fp32" | "f32" => {
                eprintln!("  KV cache: FP32");
                kv::KvCache::new_gpu(
                    gpu,
                    config.n_layers,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq,
                )
                .map_err(|e| format!("{e}"))?
            }
            "q8" | "int8" | "auto" | "" => {
                eprintln!("  KV cache: Q8");
                kv::KvCache::new_gpu_q8(
                    gpu,
                    config.n_layers,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq,
                )
                .map_err(|e| format!("{e}"))?
            }
            other => {
                return Err(format!(
                    "kv_mode '{other}' is not supported for the LLaMA arch \
                     (head_dim={}); rotated-K asym{{2,3,4}} require head_dim ∈ {{128,256}}. \
                     Use fp32 or q8.",
                    config.head_dim
                ));
            }
        };
        let scratch = <Llama as Architecture>::new_state(gpu, &config)?;
        // P3.2: assemble the ServingBackend (owns config/weights/scratch/kv); the
        // separate llama_* fields stay None. (HFQ load path — mirrors the
        // safetensors path below.)
        let mut llama_backend =
            hipfire_arch_llama::LlamaBackend::new(hfq.arch_id, config, weights, scratch, kv);
        // DSpark sidecar discovery (dense LLaMA/Qwen3, arch 0/1). Mirrors the
        // DFlash draft wiring: look for a `<stem>-<quant>.dspark.hfq` next to the
        // target and, when found, load the drafter + build the greedy speculator.
        // `None` (no sidecar / disabled / non-0-1 arch) leaves the AR path
        // byte-unchanged.
        let dspark = maybe_load_dspark(&mut llama_backend, hfq.arch_id, path, max_seq, gpu);
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: Some(llama_backend),
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            qwen2_backend: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            dspark,
            chat_template,
            chat_template_profile,
        })
    }
}

/// Load a model from a HuggingFace safetensors directory (ParoQuant, AWQ, etc.).
pub fn load_model_safetensors(
    path: &str,
    max_seq: usize,
    kv_mode: &str,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<LoadedModel, String> {
    use hipfire_model::ModelSource;
    use hipfire_runtime::safetensors_source::SafetensorsSource;

    eprintln!("  opening safetensors directory: {path}");
    let model_memory = unknown_model_memory(path);
    let source =
        SafetensorsSource::open(Path::new(path)).map_err(|e| format!("safetensors open: {e}"))?;

    let arch_id = source.arch_id();
    let qm = source
        .quant_config()
        .map(|q| q.method.as_str())
        .unwrap_or("none");
    eprintln!("  arch_id={arch_id}, quant_method={qm}");

    // Tokenizer from tokenizer.json
    let tokenizer = if let Some(tok_path) = source.tokenizer_json_path() {
        hipfire_model::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
            .map_err(|e| format!("failed to parse tokenizer at {}: {e}", tok_path.display()))?
            .ok_or_else(|| format!("failed to load tokenizer from {}", tok_path.display()))?
    } else {
        return Err("no tokenizer.json found in model directory".to_string());
    };

    // HF safetensors use half-split RoPE convention (rotate_half)
    // — upstream now defaults to halfsplit, no flag needed
    let chat_template = source.chat_template();

    if arch_id == ARCH_ID_LLAMA_MISTRAL || arch_id == ARCH_ID_QWEN3_QWEN2_LEGACY {
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        // LLaMA / Qwen3 — standard attention, no DeltaNet
        let config = hipfire_runtime::hfq::config_from_safetensors_llama(&source)
            .ok_or("failed to parse LLaMA/Qwen3 config from config.json")?;

        eprintln!(
            "  LLaMA/Qwen3: dim={}, layers={}, heads={}, kv_heads={}, head_dim={}, qk_norm={}",
            config.dim,
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.has_qk_norm
        );

        let weights = hipfire_runtime::hfq::load_weights_paroquant_llama(&source, &config, gpu)
            .map_err(|e| format!("load_weights_paroquant_llama: {e:?}"))?;

        // asym3 K-cache asserts head_dim==256 (Qwen 3.5/3.6 family). Qwen3
        // dense checkpoints (e.g. shisa-Qwen3-0.6B-PARO, head_dim=128) need
        // q8 for auto/default selection; explicit "asym3" still routes to
        // the panicking constructor so caller-misconfigured runs surface.
        let asym3_auto = matches!(kv_mode, "turbo3" | "turbo" | "auto" | "");
        let kv = match kv_mode {
            "q8" => kv::KvCache::new_gpu_q8_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                max_seq,
            ),
            "asym4" | "turbo4" => kv::KvCache::new_gpu_asym4_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                max_seq,
            ),
            "kvarn" => kv::KvCache::new_gpu_kvarn_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                max_seq,
                4,
            ),
            "asym3" => kv::KvCache::new_gpu_asym3_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                max_seq,
            ),
            _ if asym3_auto && config.head_dim == 256 => kv::KvCache::new_gpu_asym3_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                max_seq,
            ),
            _ => kv::KvCache::new_gpu_q8_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                max_seq,
            ),
        }
        .map_err(|e| format!("KvCache: {e}"))?;

        let scratch = llama::ForwardScratch::new(gpu, &config)
            .map_err(|e| format!("ForwardScratch::new: {e:?}"))?;

        // P3.2: route arch 0/1 through the ServingBackend seam — assemble the
        // backend (owns config/weights/scratch/kv); the separate llama_* fields
        // stay None.
        let llama_backend =
            hipfire_arch_llama::LlamaBackend::new(arch_id, config, weights, scratch, kv);

        return Ok(LoadedModel {
            arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            qwen2_backend: None,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: Some(llama_backend),
            nemotron_backend: None,
            zaya_backend: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if arch_id == ARCH_ID_NEMOTRON_H || arch_id == ARCH_ID_MAMBA2 {
        // nemotron_h (hybrid Mamba-2 + attention/MLP/MoE) and pure Mamba-2 are
        // routed through the same Mamba-capable ServingBackend seam.
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        // The HFQ-compatible metadata wraps config.json under the "config" key.
        let meta: serde_json::Value = serde_json::from_str(source.metadata_json())
            .map_err(|e| format!("nemotron metadata parse: {e}"))?;
        let cfg_json = meta
            .get("config")
            .ok_or("nemotron: metadata_json missing 'config'")?;
        let mut cfg = if arch_id == ARCH_ID_MAMBA2 {
            hipfire_arch_nemotron::NemotronHConfig::from_mamba2_json(cfg_json)
                .map_err(|e| format!("mamba2 config: {e}"))?
        } else {
            hipfire_arch_nemotron::NemotronHConfig::from_json(cfg_json)
                .map_err(|e| format!("nemotron config: {e}"))?
        };
        if arch_id == ARCH_ID_MAMBA2 {
            if let Some(eot) = tokenizer.special_token_id("<|endoftext|>") {
                cfg.eos_token_id = eot;
            }
        } else if let Some(im_end) = tokenizer.special_token_id("<|im_end|>") {
            // Chat serving stops on the ChatML turn delimiter `<|im_end|>`, not
            // the base `eos_token_id` (`</s>` = 2 for Nano). Resolve it from the
            // tokenizer; fall back to the config eos if the model isn't ChatML.
            cfg.eos_token_id = im_end;
        }
        eprintln!(
            "  {}: hidden={}, layers={} ({} M / {} * / {} - / {} E), vocab={}, eos={}",
            if arch_id == ARCH_ID_MAMBA2 {
                "mamba2"
            } else {
                "nemotron_h"
            },
            cfg.hidden_size,
            cfg.num_layers,
            cfg.count(hipfire_arch_nemotron::BlockKind::Mamba2),
            cfg.count(hipfire_arch_nemotron::BlockKind::Attention),
            cfg.count(hipfire_arch_nemotron::BlockKind::Mlp),
            cfg.count(hipfire_arch_nemotron::BlockKind::Moe),
            cfg.vocab_size,
            cfg.eos_token_id,
        );
        let weights = hipfire_arch_nemotron::loader::load_nemotron_weights(&source, &cfg)?;
        let model = hipfire_arch_nemotron::model::NemotronModel::new(gpu, cfg, &weights, max_seq)
            .map_err(|e| format!("mamba-capable NemotronModel::new: {e:?}"))?;

        return Ok(LoadedModel {
            arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            qwen2_backend: None,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: Some(model),
            zaya_backend: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if !is_qwen35_family_arch_id(arch_id) {
        return Err(format!("safetensors loading only supports LLaMA/Qwen3 (arch_id 0/1) and Qwen3.5/3.6 (arch_id 5/6), got {arch_id}"));
    }

    // Parse config (reuse Qwen35's config parser via metadata_json)
    let config = qwen35::config_from_safetensors(&source)
        .ok_or("failed to parse Qwen3.5 config from config.json")?;

    eprintln!(
        "  Qwen3.5/3.6: dim={}, layers={}, heads={}",
        config.dim, config.n_layers, config.n_heads
    );

    // Load weights via ParoQuant path
    let weights = qwen35::load_weights_paroquant(&source, &config, gpu)
        .map_err(|e| format!("load_weights_paroquant: {e:?}"))?;

    // KV cache: default to asym3 (matches the main Qwen35 path)
    let effective_max_seq = max_seq;
    let kv_cache = match kv_mode {
        "q8" => kv::KvCache::new_gpu_q8_capped(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        ),
        "asym4" | "turbo4" => kv::KvCache::new_gpu_asym4_capped(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        ),
        "kvarn" => kv::KvCache::new_gpu_kvarn_capped(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
            4,
        ),
        _ => kv::KvCache::new_gpu_asym3_capped(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        ),
    }
    .map_err(|e| format!("KvCache: {e}"))?;
    let dn_state =
        DeltaNetState::new(gpu, &config).map_err(|e| format!("DeltaNetState::new: {e:?}"))?;
    let scratch = qwen35::Qwen35Scratch::new(gpu, &config, 256)
        .map_err(|e| format!("Qwen35Scratch::new: {e:?}"))?;
    let (chat_template, chat_template_profile) =
        profile_chat_template(chat_template, Some(&tokenizer));

    let sequence_state = Some(SequenceState::new(
        crate::session::qwen35_mixer_profile(&config.layer_types),
        Some(kv_cache),
        Some(Box::new(dn_state)),
    ));
    Ok(LoadedModel {
        arch_id,
        pp: 1,
        pp_gpus: None,
        pp_scratch_set: None,
        pp_dn_la_to_device: None,
        q35_config: Some(config),
        q35_weights: Some(weights),
        q35_scratch: Some(scratch),
        qwen2_config: None,
        qwen2_weights: None,
        qwen2_state: None,
        qwen2_backend: None,
        dots_ocr_config: None,
        dots_ocr_weights: None,
        q35_kv_mode: Some(kv_mode.to_string()),
        q35_state_quant: Some(hipfire_arch_qwen35::qwen35::StateQuant::Q8),
        q35_registry: SessionRegistry {
            sessions: std::collections::HashMap::new(),
            active_session_id: Some(QWEN35_LEGACY_SESSION_ID.to_string()),
            allocation_epoch: next_qwen35_state_allocation_epoch(),
        },
        llama_config: None,
        llama_weights: None,
        llama_scratch: None,
        llama_kv: None,
        llama_backend: None,
        nemotron_backend: None,
        zaya_backend: None,
        deepseek4_config: None,
        deepseek4_weights: None,
        deepseek4_state: None,
        deepseek4_pbs: None,
        deepseek4_eos_tok: 0,
        mtp_mode: "auto".to_string(),
        mtp_k: 3,
        mtp_weights_present: false,
        minimax_config: None,
        minimax_weights: None,
        minimax_state: None,
        minimax_eos_tok: 0,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2moe_config: None,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2moe_weights: None,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2_registry: SessionRegistry::default(),
        #[cfg(feature = "arch-lfm2moe")]
        lfm2moe_eos_tok: 0,
        vision_config: None,
        vision_weights: None,
        gemma3_vl: None,
        gemma3_text: None,
        tokenizer: Some(tokenizer),
        active: ResidentSession {
            sequence_state,
            ..Default::default()
        },
        max_seq: effective_max_seq,
        physical_cap: effective_max_seq,
        eviction: None,
        asst_turn_cache: std::collections::HashMap::new(),
        decoded_vocab: None,
        model_path: path.to_string(),
        memory: model_memory,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2_dflash: None,
        dflash: None,
        dspark: None,
        chat_template,
        chat_template_profile,
    })
}

/// Multi-GPU pipeline-parallel load path (Stage 7 of #58). Refuses VL,
/// non-Qwen3.5 architectures and (transitively, via the upstream "load"
/// handler) DFlash, CASK and PFlash. Returns a `LoadedModel` with `pp_gpus`,
/// `pp_scratch_set` and `pp_dn_la_to_device` populated; the daemon's primary
/// `gpu` parameter is unused on this path. Eviction is refused at this layer
/// because TriAttention/CASK/PFlash live on a single device and are not v1
/// targets for pp>1 — physical_cap == max_seq accordingly.
pub fn load_model_pp(
    path: &str,
    max_seq: usize,
    kv_mode_override: Option<&str>,
    state_quant_override: Option<&str>,
    pp: usize,
    _gpu: &mut hipfire_rdna::Gpu,
) -> Result<LoadedModel, String> {
    let mut kv_mode = kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let max_seq = clamp_max_seq_to_model_context(max_seq, &hfq.metadata_json);
    let model_memory = hfq_model_memory(path, &hfq);
    // Whether ANY tensor is BF16 — used to keep the DeltaNet *state* at FP32
    // (the recurrent state's cumulative-error sensitivity; orthogonal to KV).
    let is_bf16_artifact = hfq_has_bf16_weights(&hfq);
    // KV precision policy:
    //   * BF16-DOMINANT model (full-precision artifact) -> force fp32 KV (mixing
    //     a quantized KV under bf16 weights is a precision mismatch).
    //   * Quantized model (MQ4/Q8 weights, only norms BF16) -> honor an explicit
    //     kv_mode; otherwise default to fp32 for now. A quantized KV (q8/asym/
    //     KVarN) makes the model batched-prefill eligible (~32x prefill); the
    //     prior rule wrongly force-fp32'd these via the BF16 norms, locking them
    //     to the per-token path. Default flips to KVarN once it's a runtime mode.
    // KV precision: respect an explicit kv_mode (config/CLI/JSON) for ALL
    // models, including BF16-dominant ones. Default to fp32 only when the
    // caller left it unspecified. (Previously BF16-dominant artifacts were
    // force-overridden to fp32 even when the operator asked for q8/asym/KVarN
    // — that silently discarded the requested KV quant. fp32 remains the
    // safe default; quantizing KV under bf16 weights is now an opt-in the
    // operator owns.)
    if kv_mode.is_empty() {
        kv_mode = "fp32".to_string();
    }
    let tokenizer = hipfire_model::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;

    if !is_qwen35_family_arch_id(hfq.arch_id) {
        return Err(format!(
            "pp>1 supports Qwen3.5 dense (arch_id=5) and Qwen3.5-MoE / \
             Qwen3.6-A3B (arch_id=6) only; got arch_id={}. LLaMA / Qwen3 \
             dense (arch_id<5) is pp=1 only.",
            hfq.arch_id
        ));
    }
    if qwen35_vl::vision_config_from_hfq(&hfq).is_some()
        && hfq
            .tensor_data("model.visual.patch_embed.proj.weight")
            .is_some()
    {
        return Err(
            "pp>1 does not support VL models in v1; see issue #58 v1.1 roadmap".to_string(),
        );
    }

    let config = qwen35::config_from_hfq(&hfq).ok_or("failed to read Qwen3.5 config")?;

    // HIPFIRE_PP_LAYERS="a,b,..." overrides uniform split. Length must equal
    // pp; sum must equal n_layers; each entry >= 1. Used to shift layers off
    // dev 0 when token_embd asymmetry caps max_seq under uniform split.
    let mut gpus = match std::env::var("HIPFIRE_PP_LAYERS")
        .ok()
        .filter(|s| !s.is_empty())
    {
        Some(spec) => {
            let counts: Result<Vec<usize>, _> =
                spec.split(',').map(|s| s.trim().parse::<usize>()).collect();
            let counts = counts.map_err(|e| format!("HIPFIRE_PP_LAYERS parse: {e}"))?;
            if counts.len() != pp {
                return Err(format!(
                    "HIPFIRE_PP_LAYERS has {} entries, expected pp={}",
                    counts.len(),
                    pp
                ));
            }
            let sum: usize = counts.iter().sum();
            if sum != config.n_layers {
                return Err(format!(
                    "HIPFIRE_PP_LAYERS sum={} != n_layers={}",
                    sum, config.n_layers
                ));
            }
            eprintln!("  HIPFIRE_PP_LAYERS override: {:?}", counts);
            Gpus::init_layers(&counts).map_err(|e| format!("{e}"))?
        }
        None => Gpus::init_uniform(pp, config.n_layers).map_err(|e| format!("{e}"))?,
    };

    let weights =
        qwen35::load_weights_multi(&hfq, &config, &mut gpus).map_err(|e| format!("{e}"))?;

    // KV cache (asym3 default, q8/asym4/asym2/fwht{4,3,2} selectable).
    // physical_cap == max_seq on this path — eviction is refused at load.
    let kv = match kv_mode.as_str() {
        "fp32" | "f32" => kv::KvCache::new_gpu_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "q8" => kv::KvCache::new_gpu_q8_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "asym4" | "turbo4" => kv::KvCache::new_gpu_asym4_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "asym2" | "turbo2" => kv::KvCache::new_gpu_asym2_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "asym3" | "turbo3" | "turbo" | "auto" | "" => kv::KvCache::new_gpu_asym3_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "fwht4" => kv::KvCache::new_gpu_fwht4_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "fwht3" => kv::KvCache::new_gpu_fwht3_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "fwht2" => kv::KvCache::new_gpu_fwht2_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        other => {
            eprintln!("  KV cache: unrecognized '{other}', defaulting to asym3");
            kv::KvCache::new_gpu_asym3_capped_multi(
                &mut gpus,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                max_seq,
            )
            .map_err(|e| format!("{e}"))?
        }
    };

    // Mirror the pp=1 state-mode parser so pp parity probes can force the
    // same DeltaNet state representation.
    let dn_quant = if is_bf16_artifact {
        hipfire_arch_qwen35::qwen35::StateQuant::FP32
    } else {
        let parsed = parse_state_quant(state_quant_override)?;
        resolve_tiny_model_state(&hfq, state_quant_override, parsed)
    };
    eprintln!("  DeltaNet state: {}", state_quant_label(dn_quant));
    let (dn, la_to_device) = DeltaNetState::new_with_quant_multi(&mut gpus, &config, dn_quant)
        .map_err(|e| format!("{e}"))?;

    let scratch_set = Qwen35ScratchSet::new_with_kv_max_multi(&mut gpus, &config, 2048, max_seq)
        .map_err(|e| format!("{e}"))?;

    // ROCm 6.4.3 gotcha: enable_peer_access AFTER all allocations are live.
    // See multi_gpu.rs::enable_peer_all docstring for the silent-success bug
    // when the call precedes hipMalloc.
    let _peer = gpus
        .enable_peer_all()
        .map_err(|e| format!("enable_peer_all: {e}"))?;

    eprintln!(
        "  pp={pp} loaded: layer_to_device={:?}, output_device={}, peer_access={}",
        gpus.layer_to_device, gpus.output_device, gpus.peer_access_enabled,
    );

    let chat_template = resolve_chat_template(&hfq, path);
    let (chat_template, chat_template_profile) =
        profile_chat_template(chat_template, Some(&tokenizer));

    let sequence_state = Some(SequenceState::new(
        crate::session::qwen35_mixer_profile(&config.layer_types),
        Some(kv),
        Some(Box::new(dn)),
    ));
    Ok(LoadedModel {
        arch_id: hfq.arch_id,
        pp,
        pp_gpus: Some(gpus),
        pp_scratch_set: Some(scratch_set),
        pp_dn_la_to_device: Some(la_to_device),
        q35_config: Some(config),
        q35_weights: Some(weights),
        q35_scratch: None,
        q35_kv_mode: None,
        q35_state_quant: None,
        q35_registry: SessionRegistry::default(),
        llama_config: None,
        llama_weights: None,
        llama_scratch: None,
        llama_kv: None,
        llama_backend: None,
        nemotron_backend: None,
        zaya_backend: None,
        qwen2_config: None,
        qwen2_weights: None,
        qwen2_state: None,
        qwen2_backend: None,
        deepseek4_config: None,
        deepseek4_weights: None,
        deepseek4_state: None,
        deepseek4_pbs: None,
        deepseek4_eos_tok: 0,
        mtp_mode: "auto".to_string(),
        mtp_k: 3,
        mtp_weights_present: false,
        minimax_config: None,
        minimax_weights: None,
        minimax_state: None,
        minimax_eos_tok: 0,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2moe_config: None,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2moe_weights: None,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2_registry: SessionRegistry::default(),
        #[cfg(feature = "arch-lfm2moe")]
        lfm2moe_eos_tok: 0,
        dots_ocr_config: None,
        dots_ocr_weights: None,
        vision_config: None,
        vision_weights: None,
        gemma3_vl: None,
        gemma3_text: None,
        tokenizer: Some(tokenizer),
        active: ResidentSession {
            sequence_state,
            ..Default::default()
        },
        max_seq,
        physical_cap: max_seq,
        eviction: None,
        asst_turn_cache: std::collections::HashMap::new(),
        decoded_vocab: None,
        model_path: path.to_string(),
        memory: model_memory,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2_dflash: None,
        dflash: None,
        dspark: None,
        chat_template,
        chat_template_profile,
    })
}

/// Pre-screen all Qwen3.5/3.6 weight matrices for MMQ safety (#87).
/// Returns (n_safe, n_unsafe). Results are cached in gpu.mmq_screen_cache.
pub fn screen_weights_qwen35(
    weights: &qwen35::Qwen35Weights,
    gpu: &mut hipfire_rdna::Gpu,
) -> (usize, usize) {
    use hipfire_arch_qwen35::qwen35::LayerWeights;
    let mut n_safe = 0usize;
    let mut n_unsafe = 0usize;

    for layer in &weights.layers {
        // Collect all weight tensors for this layer that could use MMQ
        let wts: Vec<(&hipfire_runtime::weights::WeightTensor, &str)> = match layer {
            LayerWeights::DeltaNet(l) => vec![
                (&l.wqkv, "qkvza.qkv"),
                (&l.wz, "qkvza.z"),
                (&l.w_beta, "qkvza.beta"),
                (&l.w_alpha, "qkvza.alpha"),
                (&l.w_gate, "gate_up.gate"),
                (&l.w_up, "gate_up.up"),
                (&l.wo, "residual"),
            ],
            LayerWeights::FullAttn(l) => vec![
                (&l.wq, "qkv.q"),
                (&l.wk, "qkv.k"),
                (&l.wv, "qkv.v"),
                (&l.w_gate, "gate_up.gate"),
                (&l.w_up, "gate_up.up"),
                (&l.wo, "residual"),
            ],
            LayerWeights::DeltaNetMoe(l) => vec![
                (&l.wqkv, "qkvza.qkv"),
                (&l.wz, "qkvza.z"),
                (&l.w_beta, "qkvza.beta"),
                (&l.w_alpha, "qkvza.alpha"),
                (&l.wo, "residual"),
            ],
            LayerWeights::FullAttnMoe(l) => vec![
                (&l.wq, "qkv.q"),
                (&l.wk, "qkv.k"),
                (&l.wv, "qkv.v"),
                (&l.wo, "residual"),
            ],
        };

        for (wt, _name) in wts {
            // MMQ kernels only operate on HFQ4G256 weights. Other formats
            // (MQ3, MQ2, HFQ6, etc.) use different dispatch paths and must
            // not be fed to the HFQ4-specific screening kernels — buffer
            // layout mismatch would read past the end. See PR #106.
            if !matches!(
                wt.gpu_dtype,
                hipfire_rdna::DType::HFQ4G256 | hipfire_rdna::DType::MQ4G256
            ) {
                continue;
            }
            if gpu.mmq_screen_weight(&wt.buf, wt.m, wt.k) {
                n_safe += 1;
            } else {
                n_unsafe += 1;
            }
        }
    }

    (n_safe, n_unsafe)
}

/// Free all GPU resources held by a loaded model (weights, scratch, KV/state,
/// eviction scratch, DFlash drafter) by consuming it. Per-arch teardown mirrors
/// whichever Option fields are populated.
pub fn unload_model(mut m: LoadedModel, gpu: &mut hipfire_rdna::Gpu) {
    // Multi-GPU branch (Stage 7 of #58). Frees per-device tensors through the
    // Gpus orchestrator, then invalidates per-device caches so the next load
    // can't inherit stale verdicts at recycled device addresses. Order
    // matches the alloc order in load_model_pp reversed: scratch → kv → dn →
    // weights, so each free targets a still-live owner.
    if m.pp > 1 {
        let mut gpus = m.pp_gpus.expect("pp>1 must carry pp_gpus");
        if let Some(scratch_set) = m.pp_scratch_set {
            scratch_set.free_gpu_multi(&mut gpus);
        }
        if let Some(ss) = m.active.sequence_state.take() {
            let (kv, recurrent) = ss.into_parts();
            if let Some(kv) = kv {
                kv.free_gpu_multi(&mut gpus);
            }
            if let Some(r) = recurrent {
                let dn = *r
                    .into_any()
                    .downcast::<DeltaNetState>()
                    .expect("qwen35 recurrent state is DeltaNetState");
                let la_to_device = m.pp_dn_la_to_device.expect("pp>1 must carry la_to_device");
                dn.free_gpu_multi(&mut gpus, &la_to_device);
            }
        }
        if let Some(w) = m.q35_weights {
            w.free_gpu_multi(&mut gpus);
        }
        for g in gpus.devices.iter_mut() {
            g.invalidate_weight_caches();
            g.invalidate_graph_state();
            g.drain_pool();
        }
        let _ = gpu;
        return;
    }
    // DFlash state: free every GPU-resident component explicitly. No `Drop` on
    // GpuTensor/DeviceBuffer, so the ring buffer, verify scratch, snapshot, tape,
    // and (optional) DDTree state must each be returned to the pool — otherwise a
    // mid-session load/unload cycle strands them until daemon exit.
    if let Some(df) = m.dflash {
        df.draft_weights.free_gpu(gpu);
        df.draft_scratch.free_gpu(gpu);
        df.hidden_rb.free_gpu(gpu);
        df.verify_scratch.free_gpu(gpu);
        df.target_snap.free_gpu(gpu);
        df.gdn_tape.free_gpu(gpu);
        if let Some(ddtree) = df.ddtree {
            ddtree.free_gpu(gpu);
        }
    }
    // DSpark speculator: release every GPU buffer the drafter owns (drafter body
    // weights/scratch + block KV + main-hidden cache). `Speculator::free` is a
    // required trait method, so a forgotten buffer is a compile error, not a leak.
    if let Some(ds) = m.dspark {
        ds.speculator.free(gpu);
    }
    #[cfg(feature = "arch-lfm2moe")]
    if let Some(df) = m.lfm2_dflash {
        df.draft_weights.free_gpu(gpu);
        df.draft_scratch.free_gpu(gpu);
        df.target_snap.free_gpu(gpu);
    }
    // Free eviction context (centers + scratch tensors) if active.
    if let Some(ev) = m.eviction {
        ev.free_gpu(gpu);
    }
    // Free KV cache + DeltaNet state + scratch first (small fraction of VRAM).
    if let Some(ss) = m.active.sequence_state.take() {
        let (kv, recurrent) = ss.into_parts();
        if let Some(kv) = kv {
            kv.free_gpu(gpu);
        }
        if let Some(r) = recurrent {
            let dn = *r
                .into_any()
                .downcast::<DeltaNetState>()
                .expect("qwen35 recurrent state is DeltaNetState");
            dn.free_gpu(gpu);
        }
    }
    if let Some(s) = m.q35_scratch {
        s.free_gpu(gpu);
    }
    if let Some(kv) = m.llama_kv {
        kv.free_gpu(gpu);
    }
    if let Some(s) = m.llama_scratch {
        s.free_gpu(gpu);
    }
    // Qwen2 state holds both the per-step scratch AND the KV cache — one
    // free_gpu call handles both. (Compare LLaMA where ForwardScratch and
    // KvCache are separate fields.)
    if let Some(s) = m.qwen2_state {
        s.free_gpu(gpu);
    }
    // V4F (arch_id=9) per-session scratch + per-layer SWA/indexer/
    // compressor caches. Without these `unload_model` would leak ~tens
    // of MB of state buffers per load/unload cycle, defeating idle
    // eviction.
    if let Some(s) = m.deepseek4_state {
        s.free_gpu(gpu);
    }
    if let Some(pbs) = m.deepseek4_pbs {
        pbs.free_gpu(gpu);
    }
    // MiniMax-M2 (arch_id=10): state (KV + scratch + device pos scalar) then
    // weights (the VRAM bulk). Both expose free_gpu so a load/unload cycle
    // returns their device buffers to the pool instead of leaking.
    if let Some(s) = m.minimax_state {
        s.free_gpu(gpu);
    }
    if let Some(w) = m.minimax_weights {
        w.free_gpu(gpu);
    }
    // LFM2.5-MoE (arch_id=11): state (KV + conv ring + scratch + pos scalar)
    // then weights. Same explicit-free contract as minimax.
    #[cfg(feature = "arch-lfm2moe")]
    if let Some(s) = m.active.lfm2moe_state {
        s.free_gpu(gpu);
    }
    #[cfg(feature = "arch-lfm2moe")]
    if let Some(w) = m.lfm2moe_weights {
        w.free_gpu(gpu);
    }
    // Weights are the bulk of VRAM (~80%). Free them too so idle eviction
    // actually returns VRAM to the system, not just the cache.
    if let Some(w) = m.q35_weights {
        w.free_gpu(gpu);
    }
    if let Some(w) = m.llama_weights {
        w.free_gpu(gpu);
    }
    if let Some(w) = m.qwen2_weights {
        w.free_gpu(gpu);
    }
    if let Some(w) = m.vision_weights {
        w.free_gpu(gpu);
    }
    // Gemma3-VL (arch_id=13): the backend owns the text/vision/projector weights
    // and its decode state — free both (mirrors Gemma3VlBackend::unload).
    if let Some(b) = m.gemma3_vl {
        b.weights.free_gpu(gpu);
        b.state.free_gpu(gpu);
    }
    // Gemma3 text (arch_id=12): backend owns the decoder weights + decode state.
    if let Some(b) = m.gemma3_text {
        b.weights.free_gpu(gpu);
        b.state.free_gpu(gpu);
    }
    if let Some(w) = m.deepseek4_weights {
        w.free_gpu(gpu);
    }
    // Assembled ServingBackend wrappers (arch_id 0/1 llama, 7 qwen2, 14/15
    // nemotron, 16 zaya) own their GPU weights + decode state internally and
    // have no Drop. Unlike the loose-slot paths above, dropping the LoadedModel
    // does NOT return their device buffers to the pool — so without an explicit
    // unload they leak across a load/unload cycle. For a bf16 reference (~34 GB)
    // that OOMs the very next load (the multi-model quality-battery failure).
    // Each backend's `ServingBackend::unload` frees exactly what it allocated;
    // the loose `*_weights`/`*_state` frees above are no-ops on these paths
    // (those fields stay None when the backend wrapper is populated).
    {
        use hipfire_runtime::arch::ServingBackend;
        if let Some(b) = m.zaya_backend {
            Box::new(b).unload(gpu);
        }
        if let Some(b) = m.nemotron_backend {
            Box::new(b).unload(gpu);
        }
        if let Some(b) = m.qwen2_backend {
            Box::new(b).unload(gpu);
        }
        if let Some(b) = m.llama_backend {
            Box::new(b).unload(gpu);
        }
    }
    // Drop pointer-keyed caches whose keys point at weight buffers that are
    // about to be returned to the pool. Without this, the next model loaded
    // can land at the same device address and silently inherit stale
    // verdicts (mmq_screen_cache) or leaked FP16 shadows (fp16_shadow_cache).
    gpu.invalidate_weight_caches();
    // Tear down any captured hipGraphs (single-slot AR forward graph plus
    // DFlash verify and replay graph caches). These bake KV-cache, scratch,
    // and draft-weight pointers into kernarg memory at capture time; the
    // tensors backing those pointers are freed above, so replaying after
    // a model swap would dispatch against dangling or wrong-content
    // memory.
    gpu.invalidate_graph_state();
    gpu.drain_pool();
}

/// Load the experimental LFM2 DFlash drafter. Unlike Qwen DFlash, this path
/// does not allocate DeltaNet/ring-buffer/DDTree state; LFM2 verifies through
/// its arch-local batched prefill and snapshots KV + short-conv target state.
#[cfg(feature = "arch-lfm2moe")]
pub fn load_lfm2_dflash_state(
    draft_path: &str,
    ctx_capacity: usize,
    target_config: &lfm2moe::config::Lfm2MoeConfig,
    target_state: &lfm2moe::lfm2moe::Lfm2MoeState,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<Lfm2DflashState, String> {
    let hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("open LFM2 draft: {e}"))?;
    let draft_config = DflashConfig::from_hfq(&hfq).ok_or("parse LFM2 DflashConfig")?;
    lfm2moe::validate_dflash_contract(target_config, &draft_config)
        .map_err(|e| format!("LFM2 DFlash draft contract: {e}"))?;
    let use_f16_weights = lfm2moe::lfm2_dflash_use_f16_weights();
    let draft_weights = DflashWeights::load_with_f16(gpu, &hfq, &draft_config, use_f16_weights)
        .map_err(|e| format!("load LFM2 draft weights: {e}"))?;
    let sync_gemm = lfm2moe::lfm2_dflash_sync_gemm();
    let draft_scratch = DflashScratch::new_with_mq_and_sync(
        gpu,
        &draft_config,
        draft_config.block_size,
        ctx_capacity,
        draft_weights.has_mq,
        sync_gemm,
    )
    .map_err(|e| format!("LFM2 draft scratch: {e}"))?;
    let target_snap =
        lfm2moe::Lfm2DflashTargetSnapshot::new_for(gpu, target_state, draft_config.block_size)
            .map_err(|e| format!("LFM2 target snapshot: {e:?}"))?;
    let target_hidden_host: Vec<f32> =
        Vec::with_capacity(ctx_capacity * draft_config.num_extract() * draft_config.hidden);
    let block_size = draft_config.block_size;
    eprintln!(
        "  LFM2 DFlash draft loaded: block={} extract_layers={:?} hidden={} ctx_capacity={} f16_weights={} sync_gemm={}",
        block_size, draft_config.target_layer_ids, draft_config.hidden, ctx_capacity, use_f16_weights, sync_gemm
    );

    Ok(Lfm2DflashState {
        draft_config,
        draft_weights,
        draft_scratch,
        target_snap,
        target_hidden_host,
        ctx_capacity,
        block_size,
    })
}

/// DSpark analog of the DFlash `draft_path.is_some()` gate: discover a
/// `<stem>-<quant>.dspark.hfq` sidecar next to a dense LLaMA/Qwen3 target
/// (arch 0/1) and, when present, load it + build the greedy speculator. Returns
/// `None` for any non-0/1 arch, when `HIPFIRE_DSPARK=0`, when no sidecar exists,
/// or when the sidecar fails to load (logged, non-fatal — the model still loads
/// AR-only). Keeps non-DSpark loads byte-identical.
fn maybe_load_dspark(
    backend: &mut hipfire_arch_llama::LlamaBackend,
    arch_id: u32,
    target_path: &str,
    ctx_capacity: usize,
    gpu: &mut hipfire_rdna::Gpu,
) -> Option<DsparkState> {
    if !matches!(arch_id, 0 | 1) {
        return None;
    }
    if std::env::var("HIPFIRE_DSPARK").ok().as_deref() == Some("0") {
        return None;
    }
    let sidecar = hipfire_model::discover_dspark_draft_for_model(Path::new(target_path))?;
    let sidecar = sidecar.to_string_lossy().to_string();
    eprintln!("  llama: DSpark sidecar discovered: {sidecar}");
    match load_dspark_state(&sidecar, backend, ctx_capacity, gpu) {
        Ok(state) => {
            eprintln!(
                "  llama DSpark speculator enabled (sidecar, block={})",
                state.speculator.block_size()
            );
            Some(state)
        }
        Err(e) => {
            eprintln!("  llama: WARNING DSpark sidecar load failed: {e}");
            None
        }
    }
}

/// Load a DSpark drafter sidecar and build the arch-generic greedy speculator.
///
/// Mirrors [`load_dflash_state`] for the dense LLaMA/Qwen3 arch: opens the
/// sidecar HFQ, loads the Qwen3 drafter body + DSpark globals via
/// `hipfire_arch_llama::dspark_body::load_qwen3_dspark`, arms the target's
/// extract-layer capture (`set_dflash_extract_layers`), then builds the
/// `Box<dyn Speculator>` (`build_qwen3_dspark_body` + `build_dspark_speculator`).
///
/// `stage_norm` / `lm_head` are NON-OWNING aliases of the drafter's
/// `output_norm` / `output` tensors: the speculator body owns the primaries and
/// frees them in `Speculator::free`; `DsparkDrafter::mtp_free` never frees these
/// aliases, so there is no double-free (this matches the source carrier's
/// `shallow_clone` contract, expressed here via `GpuTensor::sub_offset`).
pub fn load_dspark_state(
    sidecar_path: &str,
    backend: &mut hipfire_arch_llama::LlamaBackend,
    ctx_capacity: usize,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<DsparkState, String> {
    let mut hfq =
        HfqFile::open(Path::new(sidecar_path)).map_err(|e| format!("open dspark sidecar: {e}"))?;
    // The loader reads via the pread-backed `tensor_data_vec`; drop the mmap
    // first (mirrors the load-smoke example + the source carrier).
    hfq.drop_mmap();
    let (dspark_weights, dspark_assets) =
        hipfire_arch_llama::dspark_body::load_qwen3_dspark(&hfq, gpu)?
            .ok_or("dspark sidecar has no dspark_* metadata")?;

    // Stash on the target then take back to build — the LlamaBackend fields are
    // the canonical home for the discovered sidecar (mirrors the source
    // carrier's `bundle.dspark_weights = Some(..)` / `.take()` two-phase shape),
    // and arm the target's extract-layer capture with the drafter's
    // `target_layer_ids`.
    let target_layers = dspark_weights.cfg.target_layer_ids.clone();
    backend.set_dflash_extract_layers(target_layers);
    backend.dspark_weights = Some(dspark_weights);
    backend.dspark_assets = Some(dspark_assets);
    let dspark_weights = backend.dspark_weights.take().unwrap();
    let dspark_assets = backend.dspark_assets.take().unwrap();

    let block = dspark_weights.cfg.block_size;
    // conf_threshold ladder: env > 0.1 default (source's sweep-tuned default).
    let conf_threshold = std::env::var("HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1f32);
    let vocab = dspark_assets.config.vocab_size;

    // Non-owning aliases (body owns/frees the primaries). `stage_norm` keeps the
    // drafter final-norm dtype/shape; `lm_head` is re-tagged F16/[vocab] because
    // `output.buf` was uploaded raw but its layout is F16 and `run_heads`
    // dispatches on `GpuTensor.dtype` + reads `lm_head.shape[0]` as the vocab.
    let stage_norm = {
        let n = &dspark_assets.weights.output_norm;
        n.sub_offset(0, n.numel())
    };
    let mut lm_head = {
        let o = &dspark_assets.weights.output.buf;
        o.sub_offset(0, o.numel())
    };
    lm_head.dtype = hipfire_rdna::DType::F16;
    lm_head.shape = vec![vocab];

    let body = hipfire_arch_llama::dspark_body::build_qwen3_dspark_body(
        dspark_assets,
        &dspark_weights.cfg,
        gpu,
    )
    .map_err(|e| format!("build dspark body: {e}"))?;

    // Greedy-only MVP: `supports_temp=false` ⇒ the speculator advertises
    // `requires_greedy()`, and `generate_llama` only drives it when temp≈0.
    let speculator = hipfire_specdecode_dspark::dspark_core::build_dspark_speculator(
        body,
        dspark_weights,
        stage_norm,
        lm_head,
        block,
        ctx_capacity,
        conf_threshold,
        false,
    );
    Ok(DsparkState { speculator })
}

/// Load the optional DFlash speculative-decoding drafter for a model: the draft
/// weights/config/scratch, the hidden-state ring buffer + verify scratch, and
/// (when `HIPFIRE_DDTREE_BUDGET` is set) the DDTree tree-verify side state.
pub fn load_dflash_state(
    draft_path: &str,
    ctx_capacity: usize,
    target_config: &qwen35::Qwen35Config,
    target_dn: &DeltaNetState,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<DflashState, String> {
    let hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("open draft: {e}"))?;
    let draft_config = DflashConfig::from_hfq(&hfq).ok_or("parse DflashConfig")?;
    let draft_weights =
        DflashWeights::load(gpu, &hfq, &draft_config).map_err(|e| format!("load weights: {e}"))?;
    let draft_scratch = DflashScratch::new_with_mq(
        gpu,
        &draft_config,
        draft_config.block_size,
        ctx_capacity,
        draft_weights.has_mq,
    )
    .map_err(|e| format!("draft scratch: {e}"))?;

    // Hidden ring: one row per target-layer selected by the draft config,
    // captured during each target forward. Sized so the whole context plus
    // one block fits without aliasing. Cheap (< 100 MB) next to the draft
    // weights themselves.
    let hidden_rb = HiddenStateRingBuffer::new(
        gpu,
        target_config.n_layers,
        draft_config.num_extract(),
        draft_config.hidden,
        ctx_capacity + draft_config.block_size,
        hipfire_arch_qwen35::qwen35::PREFILL_MAX_BATCH.max(draft_config.block_size),
    )
    .map_err(|e| format!("hidden_rb: {e}"))?;

    let target_snap =
        DeltaNetSnapshot::new_for(gpu, target_dn).map_err(|e| format!("target_snap: {e}"))?;

    // Read DDTree budget env-var BEFORE sizing GdnTape / VerifyScratch.
    // When DDTree is enabled, both must be sized for `1 + budget` nodes
    // per cycle (the linearized tree includes one root slot plus all
    // tree nodes), not just `block_size`. Reading the env-var here keeps
    // a single source of truth and avoids re-allocating these scratches
    // after the model is on GPU.
    //
    // DdtreeScratch::attn_bias is sized `max_n²` (max_n = 1 + budget),
    // so the allocation is quadratic in budget. The paper's Algorithm 1
    // typically uses budget ≤ 22; we cap at 256 to leave huge headroom
    // while killing the OOM cliff from a typo'd budget value (`=10000`
    // would request 400 MB just for attn_bias; `=100000` would OOM most
    // GPUs). Invalid / out-of-range values warn loudly and disable
    // DDTree rather than silently falling through.
    const DDTREE_BUDGET_MAX: usize = 256;
    let ddtree_budget_env: usize = match std::env::var("HIPFIRE_DDTREE_BUDGET").ok() {
        None => 0,
        Some(s) if s.is_empty() => 0,
        Some(s) => match s.parse::<usize>() {
            Ok(0) => 0,
            Ok(n) if n <= DDTREE_BUDGET_MAX => n,
            Ok(n) => {
                eprintln!(
                    "[hipfire-daemon] HIPFIRE_DDTREE_BUDGET={} exceeds cap {DDTREE_BUDGET_MAX} \
                     (attn_bias is O(budget²); typical values are 12-22). Disabling DDTree.",
                    n
                );
                0
            }
            Err(_) => {
                eprintln!(
                    "[hipfire-daemon] HIPFIRE_DDTREE_BUDGET={:?} is not a non-negative integer. \
                     Disabling DDTree.",
                    s
                );
                0
            }
        },
    };
    let scratch_max_n = if ddtree_budget_env > 0 {
        std::cmp::max(draft_config.block_size, 1 + ddtree_budget_env)
    } else {
        draft_config.block_size
    };

    let gdn_tape = GdnTape::new_for_config(gpu, target_config, scratch_max_n)
        .map_err(|e| format!("gdn_tape: {e}"))?;
    let verify_scratch = VerifyScratch::with_prefill(
        gpu,
        scratch_max_n,
        target_config.dim,
        target_config.vocab_size,
        target_config.dim,
        target_config,
    )
    .map_err(|e| format!("verify_scratch: {e}"))?;

    let target_hidden_host: Vec<f32> =
        Vec::with_capacity(ctx_capacity * draft_config.num_extract() * draft_config.hidden);
    let block_size = draft_config.block_size;

    // Optional DDTree allocation. `HIPFIRE_DDTREE_BUDGET=<n>` (positive
    // integer) wires the decode loop to `spec_step_ddtree_batched` instead
    // of `spec_step_dflash`. `HIPFIRE_DDTREE_TOPK=<k>` controls the
    // per-position top-K (default 4). Anything else, or budget=0, leaves
    // the existing DFlash chain-mode path untouched.
    let ddtree = match Some(ddtree_budget_env).filter(|&n| n > 0) {
        Some(budget) => {
            // topk caps the per-position branching factor in the tree
            // builder. Algorithm 1's typical setting is 4; the active
            // kernel `run_dflash_draft_for_topk_gpu` (called by both
            // `spec_step_ddtree_batched` and `spec_step_ddtree_path_c`)
            // asserts `k >= 1 && k <= 8` at speculative.rs:3302 and panics
            // outside that range. Take the kernel's bound as authoritative;
            // anything looser would let env-var values pass daemon
            // validation but blow up at the first decode cycle.
            //
            // Two upper bounds:
            //   - DDTREE_TOPK_KERNEL_MAX = 8 — kernel's hardcoded assert.
            //   - vocab_size — extra correctness cap for tiny-vocab /
            //     character-level targets where vocab can be < 8.
            //
            // Effective cap = min(8, vocab_size). Default = min(4, vocab_size).
            const DDTREE_TOPK_KERNEL_MAX: usize = 8;
            let vocab = target_config.vocab_size;
            let effective_topk_max = std::cmp::min(DDTREE_TOPK_KERNEL_MAX, vocab);
            let default_topk = std::cmp::min(4usize, vocab.max(1));
            let topk = match std::env::var("HIPFIRE_DDTREE_TOPK").ok() {
                None => default_topk,
                Some(s) if s.is_empty() => default_topk,
                Some(s) => match s.parse::<usize>() {
                    Ok(k) if k >= 1 && k <= effective_topk_max => k,
                    Ok(k) => {
                        eprintln!(
                            "[hipfire-daemon] HIPFIRE_DDTREE_TOPK={k} out of range [1, {effective_topk_max}] \
                             (vocab_size={vocab}). Falling back to default topk={default_topk}."
                        );
                        default_topk
                    }
                    Err(_) => {
                        eprintln!(
                            "[hipfire-daemon] HIPFIRE_DDTREE_TOPK={:?} is not a positive integer. \
                             Falling back to default topk={default_topk}.",
                            s
                        );
                        default_topk
                    }
                },
            };
            let post_seed_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
                .map_err(|e| format!("ddtree post_seed_snap: {e}"))?;
            let path_c_parent_pre_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
                .map_err(|e| format!("ddtree path_c_parent_pre_snap: {e}"))?;
            let path_c_main_end_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
                .map_err(|e| format!("ddtree path_c_main_end_snap: {e}"))?;
            let n_fa_layers = target_config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count();
            // qkv_dim mirrors GdnTape::new_for_config: linear-attention
            // qkv row width (k_dim × 2 + v_dim).
            let k_dim = target_config.linear_num_key_heads * target_config.linear_key_head_dim;
            let v_dim = target_config.linear_num_value_heads * target_config.linear_value_head_dim;
            let qkv_dim = k_dim * 2 + v_dim;
            let scratch = DdtreeScratch::new(
                gpu,
                budget,
                target_config.n_kv_heads,
                target_config.head_dim,
                qkv_dim,
                n_fa_layers,
            )
            .map_err(|e| format!("ddtree scratch: {e}"))?;
            eprintln!(
                "[hipfire-daemon] DDTree enabled: budget={budget}, topk={topk}, n_fa_layers={n_fa_layers}"
            );
            Some(DdtreeState {
                post_seed_snap,
                scratch,
                budget,
                topk,
                path_c_parent_pre_snap,
                path_c_main_end_snap,
            })
        }
        None => None,
    };

    Ok(DflashState {
        draft_config,
        draft_weights,
        draft_scratch,
        hidden_rb,
        verify_scratch,
        target_snap,
        gdn_tape,
        target_hidden_host,
        ctx_capacity,
        block_size,
        ddtree,
    })
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    /// The DFlash load gate is driven by the generated capability matrix, not a
    /// hard-coded arch list: qwen3.5 (5/6) is admitted; archs whose matrix marks
    /// dflash != full are refused with an operator-facing, matrix-derived reason.
    /// Guards the silent-no-op gap (draft attached to a non-DFlash arch → load
    /// succeeds, then generate() falls through to plain AR).
    #[test]
    fn dflash_admission_is_matrix_backed() {
        for arch in [5u32, 6] {
            assert!(
                require_arch_feature(arch, "DFlash spec-decode", arch_features(arch).dflash)
                    .is_ok(),
                "qwen3.5 arch {arch} must admit DFlash"
            );
        }
        // llama (0) and gemma3 (12) have dflash=none → refused.
        let e = require_arch_feature(0, "DFlash spec-decode", arch_features(0).dflash)
            .expect_err("llama must be refused");
        assert!(
            e.contains("llama") && e.contains("does not support"),
            "msg: {e}"
        );
        assert!(require_arch_feature(12, "DFlash spec-decode", arch_features(12).dflash).is_err());
    }

    #[cfg(feature = "arch-lfm2moe")]
    #[test]
    fn lfm2_triattn_sidecars_use_attention_ordinal_indices() {
        let config = lfm2moe::config::Lfm2MoeConfig::from_config_value(&serde_json::json!({
            "vocab_size": 256,
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "layer_types": ["conv", "full_attention", "conv", "full_attention"]
        }))
        .unwrap();

        assert_eq!(config.num_attention_layers(), 2);
        assert_eq!(lfm2_triattn_kv_layer_ids(&config), vec![0, 1]);
    }
}
