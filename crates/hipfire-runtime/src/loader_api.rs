// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Arch-agnostic loader contract. Concrete `ModelState`/`LoadedModel`
//! and the registry live top-of-DAG in `hipfire-loader`; this module
//! holds only what the arch crates need to implement a carrier.

use crate::hfq::HfqFile;
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
    /// Gemma 4 EAGLE drafter path (arch-22 `gemma4_unified_assistant`),
    /// supplied via `params.drafter` on an arch_id=13 target. None on every
    /// other arch / when no drafter requested.
    pub gemma4_drafter: Option<&'a str>,
    /// Gemma 4 EAGLE draft length (drafts per round). Defaulted by the caller
    /// to the validated value when `params.spec` is absent.
    pub gemma4_eagle_draft_len: usize,
    pub kv_mode_override: Option<&'a str>,
    pub kv_adaptive_override: Option<&'a str>,
    pub state_quant_override: Option<&'a str>,
    pub cask: &'a CaskConfig,
    pub pp: usize,
    pub gpu: &'a mut Gpu,
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
