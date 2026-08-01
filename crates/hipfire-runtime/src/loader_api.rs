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
    pub draft_path: Option<&'a str>,
    pub kv_mode_override: Option<&'a str>,
    pub kv_backend: KvBackend,
    pub kv_adaptive_override: Option<&'a str>,
    pub state_quant_override: Option<&'a str>,
    pub cask: &'a CaskConfig,
    pub pp: usize,
    pub spec: SpecLoadCfg,
    pub gpu: &'a mut Gpu,
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
}

/// One arch's load contract. Object-safe — usable as `&dyn Carrier`.
/// Implementations live in `hipfire-loader::carriers`.
pub trait Carrier {
    fn name(&self) -> &'static str;
    fn probe(&self, src: &ModelSource) -> bool;
}

/// CASK/TriAttention params forwarded by the CLI at load time.
#[derive(Default)]
pub struct CaskConfig {
    pub sidecar: Option<String>,
    pub cask_m_folding: bool,
    pub budget: usize,
    pub beta: usize,
    pub core_frac: f32,
    pub fold_m: usize,
}
