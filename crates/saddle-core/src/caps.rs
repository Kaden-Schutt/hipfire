// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Architecture capability contract.
//!
//! STUB — filled by lean-up map item **C3**. See
//! `docs/governance/2026-08-15-hipfire-leanup-map.md` § 5b for the file
//! ownership contract governing this module.
//!
//! Capabilities are **declared** per [`crate::caps::ArchCaps`] by the
//! carrier, never inferred from an identifier inside this crate. There is no
//! identifier matching here — the daemon maps `arch id -> &dyn Carrier ->
//! caps()` once at the call site and then queries the resulting struct.

/// Which DFlash diffusion-drafter family, if any, an architecture participates in.
///
/// Qwen (5,6) and LLaMA (0,1) both use the `Qwen35Emit` / `Qwen35Dflash`
/// diffusion path but with different generation routes (`QwenDflash` vs
/// `LlamaSpec`). All other arches return `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DflashKind {
    /// Qwen family (arch 5/6) — `GenerationRoute::QwenDflash` / `QwenAr` / `QwenMtp`.
    Qwen,
    /// LLaMA/Mistral family (arch 0/1) — `GenerationRoute::LlamaSpec` / `LlamaAr`.
    Llama,
}

/// Native reasoning-control protocol declared by a model architecture.
///
/// Effort selects prompt semantics. Any explicit thinking-token limit remains
/// an independent inference policy and is never derived from this enum.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReasoningContract {
    #[default]
    Unsupported,
    QwenJinja,
    DeepSeek4,
    GemmaBoolean,
    MuseGlimmer,
}

impl ReasoningContract {
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::QwenJinja => "qwen_jinja",
            Self::DeepSeek4 => "deepseek4",
            Self::GemmaBoolean => "gemma_boolean",
            Self::MuseGlimmer => "muse_glimmer",
        }
    }

    pub fn from_wire_name(value: &str) -> Option<Self> {
        match value {
            "unsupported" => Some(Self::Unsupported),
            "qwen_jinja" => Some(Self::QwenJinja),
            "deepseek4" => Some(Self::DeepSeek4),
            "gemma_boolean" => Some(Self::GemmaBoolean),
            "muse_glimmer" => Some(Self::MuseGlimmer),
            _ => None,
        }
    }
}

/// Declared capabilities of one architecture.
///
/// Every field is set by the carrier that claims the identifier; the daemon
/// replaces `arch id ==` branches with queries of this struct. Fields are
/// deliberately named after the *functional* property they gate, not after
/// the family name — e.g. `spec_excludes_adaptive` rather than `is_qwen`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchCaps {
    /// Native reasoning and thinking control protocol.
    pub reasoning_contract: ReasoningContract,
    /// Single-GPU continuous batching (independent decode lanes).
    ///
    /// True for Qwen3.5 (5,6) and LFM2.5 (11). See
    /// `daemon.rs:1584` and `is_batch_eligible` (identifiers 5/6/11).
    pub supports_continuous_batch: bool,

    /// Expert-parallel (TP=4) continuous batching for Qwen35.
    ///
    /// True for Qwen3.5 (5,6) only. See `daemon.rs:3938`
    /// (`is_qwen_ep_batch_request_eligible`).
    pub supports_ep_batch: bool,

    /// DFlash diffusion-drafter family. `Some(Qwen)` for 5/6, `Some(Llama)`
    /// for 0/1, `None` otherwise. Covers `daemon.rs:23874,23876,23883,24874`.
    pub dflash: Option<DflashKind>,

    /// Native Qwen MTP heads (NextN). True for 5,6 only. See `daemon.rs:23859`.
    pub supports_mtp: bool,

    /// Whether speculative decoding is incompatible with adaptive KV for this
    /// arch. When true, `has_spec && kv_adaptive` forces the AR fallback.
    /// True for Qwen3.5 (5,6) only. See `daemon.rs:23881`.
    pub spec_excludes_adaptive: bool,

    /// Semantic contract version advertised on `gen_start`. `Some(2)` for
    /// Qwen (5,6) and Muse Glimmer (14), `None` otherwise. See
    /// `daemon.rs:10599-10601` (`gen_start_contract_version_for_arch`) and
    /// `daemon.rs:19684` (Qwen path is `has_deltanet && Some(2)`).
    pub semantic_contract_version: Option<u32>,

    /// Whether this arch's AR path uses DeltaNet recurrent state (the Qwen
    /// family). True for 5,6 only. Replaces the Qwen AR
    /// execution branch at `daemon.rs:25894` and the `qwen_semantic_v2`
    /// predicate at `:19684`.
    pub has_deltanet: bool,

    /// Whether this architecture can accept image input (vision).
    ///
    /// True for Qwen3.5-VL (5,6) when a vision tower is present and
    /// dots.ocr (8). The daemon's `has_image && !has_vl` gate and the
    /// `VisionRoute::None` check become queries of this field; the
    /// arch_id table in `vision_route` remains only to discriminate which
    /// vision implementation to run (QwenVl vs DotsOcr have distinct generate
    /// bodies). Default `false` so every text-only arch is unaffected.
    pub supports_images: bool,
}

impl ArchCaps {
    /// Convenience: does this arch participate in any DFlash route?
    pub fn supports_dflash(&self) -> bool {
        self.dflash.is_some()
    }

    /// True if this is the Qwen DFlash family (5,6).
    pub fn is_qwen_dflash(&self) -> bool {
        matches!(self.dflash, Some(DflashKind::Qwen))
    }

    /// True if this is the LLaMA DFlash family (0,1).
    pub fn is_llama_dflash(&self) -> bool {
        matches!(self.dflash, Some(DflashKind::Llama))
    }

    /// True if this arch advertises semantic-v2 contract (`Some(2)`).
    pub fn supports_semantic_v2(&self) -> bool {
        self.semantic_contract_version == Some(2)
    }

    /// True for the Qwen semantic-v2 path (DeltaNet + v2). Covers
    /// `daemon.rs:19684`.
    pub fn qwen_semantic_v2(&self) -> bool {
        self.has_deltanet && self.semantic_contract_version == Some(2)
    }
}

impl Default for ArchCaps {
    fn default() -> Self {
        Self {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            reasoning_contract: ReasoningContract::Unsupported,
            supports_images: false,
        }
    }
}

/// Request-derived flags for [`is_batch_eligible`] style checks.
///
/// Consumed by `hipfire_engine::scheduler::is_batch_eligible` together with
/// [`ArchCaps`]: each field snapshots one request/topology input (`pp`,
/// `ep`, image/tools/stop payloads, `speculator`/`kv_adaptive`/`pflash`
/// activity, single-user history, think mode, continuous-batch opt-in and
/// size). Nothing in production constructs this struct today — it is built
/// in unit tests; the daemon's live request path decides through
/// `hipfire_generate::batch::is_batch_request_eligible`, which reads the
/// JSON message and `LoadedModel` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchEligibilityRequest {
    /// Pipeline-parallel degree from `LoadedModel::pp`.
    pub pp: usize,
    /// Whether expert-parallel state is present (`m.ep.is_some()`).
    pub ep_is_some: bool,
    /// Whether the request carries an image / `image_base64`.
    pub has_image: bool,
    /// Whether the request carries non-empty `tools`.
    pub has_tools: bool,
    /// Whether the request carries a custom `stop` sequence.
    pub has_stop: bool,
    /// Whether a speculator is loaded (`m.speculator.is_some()`).
    pub has_speculator: bool,
    /// Whether adaptive KV is active (`m.kv_adaptive.is_some()`).
    pub has_adaptive: bool,
    /// Whether PFlash is active for this request (`pflash_active`).
    pub has_pflash: bool,
    /// Whether `messages` is not a single user turn (multi-turn history).
    pub has_messages_history: bool,
    /// Whether `ThinkMode` is `NonThink` (batch requires non-thinking).
    pub think_mode_is_nonthink: bool,
    /// Whether the request opted into `serve_continuous_batch`.
    pub serve_continuous_batch: bool,
    /// Value of `continuous_batch_size` (must be >1 for batch).
    pub continuous_batch_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_text_only_no_images() {
        let caps = ArchCaps::default();
        assert!(
            !caps.supports_images,
            "text-only default must not support images"
        );
        assert_eq!(
            caps.reasoning_contract,
            ReasoningContract::Unsupported,
            "default must not claim a reasoning protocol"
        );
        // All other caps also false/None for text-only baseline.
        assert!(!caps.supports_continuous_batch);
        assert!(!caps.supports_ep_batch);
        assert!(caps.dflash.is_none());
        assert!(!caps.supports_mtp);
        assert!(!caps.spec_excludes_adaptive);
        assert!(caps.semantic_contract_version.is_none());
        assert!(!caps.has_deltanet);
    }

    #[test]
    fn vl_arch_caps_declare_images() {
        // Mirrors Qwen35Carrier and DotsOcrCarrier declarations.
        let qwen35_vl = ArchCaps {
            reasoning_contract: ReasoningContract::QwenJinja,
            supports_continuous_batch: true,
            supports_ep_batch: true,
            dflash: Some(DflashKind::Qwen),
            supports_mtp: true,
            spec_excludes_adaptive: true,
            semantic_contract_version: Some(2),
            has_deltanet: true,
            supports_images: true,
        };
        let dots_ocr = ArchCaps {
            supports_images: true,
            ..ArchCaps::default()
        };
        assert!(
            qwen35_vl.supports_images,
            "Qwen3.5-VL must declare image support"
        );
        assert!(
            dots_ocr.supports_images,
            "dots.ocr must declare image support"
        );

        // Text-only archetypes must stay false.
        let llama = ArchCaps {
            dflash: Some(DflashKind::Llama),
            ..ArchCaps::default()
        };
        let qwen2 = ArchCaps::default();
        assert!(
            !llama.supports_images,
            "Llama (text-only) must not declare image support"
        );
        assert!(
            !qwen2.supports_images,
            "Qwen2 (text-only) must not declare image support"
        );
    }

    #[test]
    fn reasoning_contract_wire_names_round_trip() {
        for contract in [
            ReasoningContract::Unsupported,
            ReasoningContract::QwenJinja,
            ReasoningContract::DeepSeek4,
            ReasoningContract::GemmaBoolean,
            ReasoningContract::MuseGlimmer,
        ] {
            assert_eq!(
                ReasoningContract::from_wire_name(contract.wire_name()),
                Some(contract)
            );
        }
        assert_eq!(ReasoningContract::from_wire_name("unknown"), None);
    }
}
