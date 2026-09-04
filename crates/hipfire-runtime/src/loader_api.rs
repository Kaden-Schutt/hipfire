// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Arch-agnostic loader contract. Concrete `ModelState`/`LoadedModel`
//! and the registry live top-of-DAG in `hipfire-loader`; this module
//! holds only what the arch crates need to implement a carrier.

use crate::hfq::HfqFile;
use crate::kv_backend::KvBackend;
use crate::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;
use std::path::Path;

/// A model on disk, before we know its arch. Carries either a parsed
/// HFQ header or a directory (safetensors/ParoQuant — probed later).
pub enum ModelSource {
    Hfq(HfqFile),
    Dir(SafetensorsSource),
}

impl ModelSource {
    /// Open a model from either an HFQ file or a safetensors directory
    /// based on whether `path` is a file or directory.
    pub fn from_path(path: &str) -> Result<Self, String> {
        if Path::new(path).is_dir() {
            Ok(ModelSource::Dir(
                SafetensorsSource::open(Path::new(path)).map_err(|e| format!("{e:?}"))?,
            ))
        } else {
            Ok(ModelSource::Hfq(
                HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?,
            ))
        }
    }

    /// The HFQ or safetensors arch_id.
    pub fn arch_id(&self) -> Option<u32> {
        match self {
            ModelSource::Hfq(h) => Some(h.arch_id),
            ModelSource::Dir(s) => Some(s.arch_id()),
        }
    }

    /// Whether this source is a safetensors directory (vs an HFQ file).
    /// Carriers route on this because the HFQ and `derive_arch_id`
    /// namespaces are distinct (e.g. Qwen2 is HFQ id 7 but dir id 1).
    pub fn is_dir(&self) -> bool {
        matches!(self, ModelSource::Dir(_))
    }

    /// Human-readable description for logging.
    pub fn describe(&self) -> String {
        match self {
            ModelSource::Hfq(h) => format!("HFQ arch_id={}", h.arch_id),
            ModelSource::Dir(s) => format!("safetensors-dir arch_id={}", s.arch_id()),
        }
    }
}

/// Everything a carrier's `load` needs beyond the source itself.
pub struct LoadCtx<'a> {
    pub path: &'a str,
    pub max_seq: usize,
    /// DeepSeek V4-only physical compute placement. The default is `Single`;
    /// other carriers must ignore it.
    pub deepseek4_compute_placement: hipfire_config::Deepseek4ComputePlacement,
    /// DeepSeek V4-only routed-expert fanout override. `None` preserves the
    /// checkpoint value; other carriers must ignore it.
    pub deepseek4_experts_per_token: Option<usize>,
    pub draft_path: Option<&'a str>,
    pub kv_mode_override: Option<&'a str>,
    pub kv_backend: KvBackend,
    pub kv_adaptive_override: Option<&'a str>,
    pub state_quant_override: Option<&'a str>,
    pub cask: &'a CaskConfig,
    pub pp: usize,
    pub spec: SpecLoadCfg,
    pub gpu: &'a mut Gpu,
    /// Gemma4 EAGLE drafter path (arch 22 `gemma4_unified_assistant`), separate
    /// from `draft_path` (Qwen DFlash) so a DFlash .hfq can never be routed
    /// into the EAGLE loader by accident. `None` = no drafter (AR-only).
    /// Only `Gemma4Carrier` reads this.
    pub gemma4_drafter_path: Option<&'a str>,
    /// Gemma4 EAGLE draft_len (verify block = draft_len + 1). Validated at
    /// load time via `gemma4_eagle_spec_len` (1..=5, default 3). Meaningful
    /// only when `gemma4_drafter_path` is `Some`.
    pub gemma4_draft_len: usize,
}

/// Per-load model-free n-gram speculator settings, resolved by the CLI through
/// the config ladder (env > flag > per-model > global) and forwarded in the
/// `load` message params. `None` fields mean "the CLI said nothing" — the loader
/// then falls back to the legacy env vars (`HIPFIRE_NGRAM_DRAFT*`) so a daemon
/// driven directly (no hipfire CLI) keeps working. Env always *wins* over these
/// when set, matching the top of the ladder.
///
/// The master `speculation` selector lives entirely CLI-side: it is lowered into
/// the per-mechanism signals (`dflash_mode`/`draft`, `mtp_mode`, and this), so
/// `build_speculator`'s first-match cascade (dflash > mtp > n-gram) naturally
/// yields the chosen mechanism without the loader needing a selector of its own.
#[derive(Clone, Copy, Default)]
pub struct SpecLoadCfg {
    /// Enable the model-free n-gram drafter for this load. `None` = unspecified.
    pub ngram_draft: Option<bool>,
    /// n-gram draft window K (`HIPFIRE_NGRAM_DRAFT_K`). `None` = loader default.
    pub ngram_k: Option<usize>,
    /// n-gram min match count (`HIPFIRE_NGRAM_MIN_COUNT`). `None` = loader default.
    pub ngram_min_count: Option<u32>,
    /// DDTree verify budget — max tree nodes (`HIPFIRE_DDTREE_BUDGET`). `None` =
    /// loader default (0 = chain-mode DFlash, no ddtree). Mirrors `ngram_k`: a
    /// CLI-forwarded draft tuning knob, env-wins-else-param in the loader.
    pub ddtree_budget: Option<usize>,
    /// DDTree per-position top-K width (`HIPFIRE_DDTREE_TOPK`). `None` = default.
    pub ddtree_topk: Option<usize>,
    /// DSpark draft module (deepseek4 `-dspark` sidecar) enable, lowered from the
    /// `speculation` selector: `Some(true)` = `dspark` mode (load + force),
    /// `Some(false)` = another mechanism selected (skip load + build),
    /// `None` = `auto` (load if the sidecar exists, prefer over in-trunk MTP).
    /// Replaces the old `HIPFIRE_DEEPSEEK4_DSPARK` / `HIPFIRE_DEEPSEEK4_LOAD_DSPARK`
    /// env gates — both fold into this one mode.
    pub dspark: Option<bool>,
    /// DSpark confidence-truncation threshold (`--dspark-conf-threshold`),
    /// forwarded ONLY when the user set it. `None` = use the per-arch carrier
    /// default (qwen3 0.1, deepseek4 0.3) — the CLI no longer imposes a global
    /// default that would shadow those. Env `HIPFIRE_{QWEN3,DEEPSEEK4}_DSPARK_CONF_THRESHOLD`
    /// still wins over this in the builder.
    pub dspark_conf_threshold: Option<f32>,
    /// Qwen MTP (NextN) enable, lowered from the `speculation` selector:
    /// `Some(true)` = `mtp` mode (load + force), `Some(false)` = another
    /// mechanism selected (skip load + build), `None` = `auto` (load if a
    /// bundled trailer or `.mtp` sidecar exists).
    pub mtp: Option<bool>,
    /// MTP draft window K. `None` = runtime default (`HIPFIRE_MTP_K`).
    pub mtp_k: Option<usize>,
    /// Qwen DFlash draft enable, lowered from `dflash_mode`: `Some(true)` =
    /// `on` (fail the load when the draft is missing or unloadable),
    /// `Some(false)` = `off` (skip), `None` = `auto` (load when present,
    /// log-and-AR fallback otherwise).
    pub dflash: Option<bool>,
}

/// CASK/TriAttention params forwarded by the CLI at load time.
#[derive(Default)]
pub struct CaskConfig {
    pub sidecar: Option<String>,
    pub cask_m_folding: bool,
    /// One-way adaptive-KV -> plain TriAttention handoff position. Zero keeps
    /// the legacy mutual exclusion between adaptive KV and eviction.
    pub handoff_tokens: usize,
    pub budget: usize,
    pub beta: usize,
    pub core_frac: f32,
    pub fold_m: usize,
}

impl CaskConfig {
    /// Resolve the bounded physical KV window used by CASK/TriAttention.
    ///
    /// Keep this next to the shared load contract so the architecture carrier
    /// allocates KV with the exact same cap later used to size the eviction
    /// context. A mismatch is especially dangerous for VMM: the virtual reserve
    /// and mapped-prefix guards must agree with the eviction trigger before the
    /// first device allocation.
    pub fn physical_cap(&self, max_seq: usize) -> Result<usize, String> {
        let override_cap = hipfire_config::developer_var("HIPFIRE_KV_PHYSICAL_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        self.physical_cap_with_override(max_seq, override_cap)
    }

    /// Pure resolution seam used by CPU tests. An explicit override is clamped
    /// to the safe CASK window, preserving the historical environment-variable
    /// behavior without mutating process-global state in tests.
    pub fn physical_cap_with_override(
        &self,
        max_seq: usize,
        override_cap: Option<usize>,
    ) -> Result<usize, String> {
        if self.sidecar.is_none() {
            return Ok(max_seq);
        }
        // A staged adaptive cache reserves its K/V arenas at the adaptive
        // floor for the full advertised context. It cannot use the ordinary
        // eviction-bounded physical cap before the handoff has completed.
        if self.handoff_tokens != 0 {
            return Ok(max_seq);
        }
        let trigger = self
            .budget
            .checked_add(self.beta)
            .ok_or_else(|| "CASK budget + beta overflowed".to_string())?;
        let floor = trigger
            .checked_add(4)
            .ok_or_else(|| "CASK physical-cap safety margin overflowed".to_string())?;
        if floor > max_seq {
            return Err(format!(
                "CASK requires max_seq >= budget + beta + 4 (got max_seq={max_seq}, budget={}, beta={})",
                self.budget, self.beta
            ));
        }
        let derived = trigger.saturating_add(256).min(max_seq);
        Ok(override_cap.unwrap_or(derived).clamp(floor, max_seq))
    }
}

#[cfg(test)]
mod cask_config_tests {
    use super::CaskConfig;

    fn enabled() -> CaskConfig {
        CaskConfig {
            sidecar: Some("centers.bin".into()),
            budget: 4096,
            beta: 128,
            ..CaskConfig::default()
        }
    }

    #[test]
    fn disabled_cask_keeps_full_physical_cap() {
        assert_eq!(
            CaskConfig::default()
                .physical_cap_with_override(32_768, Some(1024))
                .unwrap(),
            32_768
        );
    }

    #[test]
    fn enabled_cask_derives_and_clamps_physical_cap() {
        let cask = enabled();
        assert_eq!(cask.physical_cap_with_override(32_768, None).unwrap(), 4480);
        assert_eq!(
            cask.physical_cap_with_override(32_768, Some(1)).unwrap(),
            4228
        );
        assert_eq!(
            cask.physical_cap_with_override(5000, Some(99_999)).unwrap(),
            5000
        );
    }

    #[test]
    fn enabled_cask_rejects_an_impossible_window() {
        let err = enabled()
            .physical_cap_with_override(4200, None)
            .unwrap_err();
        assert!(err.contains("budget + beta + 4"), "{err}");
    }

    #[test]
    fn staged_handoff_keeps_full_adaptive_physical_cap() {
        let mut cask = enabled();
        cask.handoff_tokens = 8192;
        assert_eq!(
            cask.physical_cap_with_override(32_768, Some(1024)).unwrap(),
            32_768,
            "adaptive floor reservation must not be clamped to the eviction window"
        );
    }
}
