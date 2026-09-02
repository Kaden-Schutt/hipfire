// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Source-aware admission (device-mesh G2).
//!
//! Classifies a retained source once and decides one effective topology BEFORE
//! any destructive side effect (prior-model teardown, VMM init, remap, GPU
//! allocation, carrier entry, collective creation). The load route consumes the
//! [`SourceAdmission`]'s already-open [`ModelSource`] — never re-opening or
//! re-classifying the path. A refusal here leaves whatever model is currently
//! loaded untouched: no teardown, no allocation, no carrier entry, no cache
//! mutation.

use crate::Carrier;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::loader_api::ModelSource;

/// The one effective topology admitted for a load. `tp>1` (expert-parallel) and
/// `pp>1` (pipeline-parallel) are mutually exclusive; both default to 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveTopology {
    Single,
    Pipeline(usize),
    Expert(usize),
}

/// A source admitted before any destructive side effect.
pub struct SourceAdmission {
    /// The already-open source the load route must consume (no second open).
    pub source: ModelSource,
    pub arch_id: u32,
    pub is_dir: bool,
    /// Tower-tensor presence decides text-vs-VL; config metadata alone never
    /// does (remediation contract `179a20d7f`).
    pub has_vision: bool,
    pub topology: EffectiveTopology,
    pub kv_backend: KvBackend,
    /// The resolved carrier (single/pp path). `None` for expert-parallel, which
    /// dispatches on `arch_id` directly rather than through the registry.
    pub carrier: Option<&'static dyn Carrier>,
}

/// Pure text-vs-VL decision. The vision tower tensor decides; configuration
/// metadata alone never does. Every Qwen3.5-family HF config embeds
/// `vision_config` even for text-only quantized artifacts, so a config marker
/// without the tower is the text backbone, not a refusal.
///
/// LFM2 is the one exception that *refuses*: a tower tensor with no parseable
/// `vision_config` metadata is malformed (carriers.rs:1553-1558) and fails
/// closed rather than silently loading as text.
///
/// Contract provenance: remediation commit `179a20d7f` ("classify Qwen3.5/LFM2
/// sources by vision tower tensor, not config markers").
pub fn classify_vision(
    arch_id: u32,
    has_vision_tensor: bool,
    has_vision_config: bool,
) -> Result<bool, String> {
    match arch_id {
        // Qwen3.5 dense (5) / MoE (6): the tower tensor alone decides.
        5 | 6 => Ok(has_vision_tensor),
        // LFM2 (11): tower + config both required; tower-without-config refuses.
        11 => {
            if has_vision_tensor && !has_vision_config {
                return Err(
                    "lfm2moe: artifact carries vision tensors but no vision_config \
                     metadata — requantize with --include-vision"
                        .into(),
                );
            }
            Ok(has_vision_tensor)
        }
        _ => Ok(false),
    }
}

/// Read the vision-tower probes out of an already-open source and fold them
/// through [`classify_vision`]. Read-only: probes the HFQ tensor index and (for
/// LFM2) parses `vision_config` metadata; touches no GPU state.
fn probe_vision(src: &ModelSource, arch_id: u32) -> Result<bool, String> {
    let ModelSource::Hfq(hfq) = src else {
        return classify_vision(arch_id, false, false);
    };
    let (has_tensor, has_config) = match arch_id {
        5 | 6 => (
            hfq.tensor_data("model.visual.patch_embed.proj.weight")
                .is_some(),
            // Qwen3.5 does not use the config in the decision; config parse is
            // soft (carriers.rs:527-544). A dummy `false` is never read.
            false,
        ),
        11 => (
            hfq.tensor_data("model.vision_tower.vision_model.embeddings.patch_embedding.weight")
                .is_some(),
            hipfire_arch_lfm2_vl::vision_config_from_hfq(hfq).is_some(),
        ),
        _ => (false, false),
    };
    classify_vision(arch_id, has_tensor, has_config)
}

/// Resolve the single carrier that claims a source, refusing no-carrier and
/// ambiguous-carrier sources exactly as the load entries do.
fn resolve_carrier(src: &ModelSource) -> Result<&'static dyn Carrier, String> {
    let mut matches = crate::REGISTRY.iter().copied().filter(|c| c.probe(src));
    let carrier = matches
        .next()
        .ok_or_else(|| format!("no carrier for {}", src.describe()))?;
    if let Some(other) = matches.next() {
        return Err(format!(
            "ambiguous carrier dispatch for {}: '{}' and '{}' both claim it",
            src.describe(),
            carrier.name(),
            other.name()
        ));
    }
    Ok(carrier)
}

/// Read-only DFlash lm-head quant refusal: a draft is attached but the target's
/// lm_head/embed quant type is not admitted for the batched GEMM verify paths.
/// Mirrors the gemma4-entry pre-allocation check (lib.rs) so the refusal fires
/// at admission instead of after prior-model teardown.
fn df_lash_lm_head_admission(
    hfq: &hipfire_runtime::hfq::HfqFile,
    draft_path: Option<&str>,
    gpu_arch: &str,
) -> Result<(), String> {
    if draft_path.is_none() {
        return Ok(());
    }
    let lm_qt = hfq
        .tensor_data("lm_head.weight")
        .or_else(|| hfq.tensor_data("model.language_model.lm_head.weight"))
        .or_else(|| hfq.tensor_data("model.language_model.embed_tokens.weight"))
        .or_else(|| hfq.tensor_data("model.embed_tokens.weight"))
        .map(|(info, _)| info.quant_type);
    if !crate::dflash_lm_head_quant_supported(lm_qt, gpu_arch) {
        let qt_desc = match lm_qt {
            Some(qt) => format!("quant_type={qt}"),
            None => "no lm_head/embed_tokens tensor found".to_string(),
        };
        return Err(format!(
            "DFlash draft requested but target lm_head {qt_desc} is not supported \
             on gfx11+gfx12 WMMA ({gpu_arch})."
        ));
    }
    Ok(())
}

/// Read-only source admission: open the source, classify `arch_id` + vision,
/// decide the effective topology, and refuse every unsupported/contradictory
/// combination — without touching GPU state, VMM, or any prior model.
///
/// Refusals mirror the current-master daemon/loader refusals so no
/// currently-served route changes; they simply fire before destructive work.
pub fn admit_source(
    path: &str,
    tp: usize,
    pp: usize,
    kv_backend_override: Option<&str>,
    draft_path: Option<&str>,
    gpu_arch: &str,
) -> Result<SourceAdmission, String> {
    let source = ModelSource::from_path(path)?;
    let arch_id = source
        .arch_id()
        .ok_or_else(|| format!("unrecognized source: {}", source.describe()))?;
    let is_dir = source.is_dir();
    let kv_backend: KvBackend = kv_backend_override
        .unwrap_or("contiguous")
        .parse()
        .map_err(|err| format!("{err}"))?;

    let (topology, carrier) = if tp > 1 {
        // Expert-parallel admission (HFQ-only). Mirrors
        // `load_model_ep_with_kv_mode`'s arch_id dispatch + VMM refusal.
        if is_dir {
            return Err(
                "EP not supported for safetensors directory sources (load as a single HFQ file)"
                    .into(),
            );
        }
        if !matches!(arch_id, 5 | 6 | 9 | 10) {
            return Err(format!(
                "EP not supported for arch_id={arch_id} (expected 5|6 for Qwen3.5, 9 for DeepSeek V4 or 10 for MiniMax)"
            ));
        }
        if kv_backend == KvBackend::Vmm {
            return Err(format!(
                "KV backend '{}' requires tp=1",
                kv_backend.as_str()
            ));
        }
        (EffectiveTopology::Expert(tp), None)
    } else {
        // Single / pipeline-parallel via the carrier registry.
        let carrier = resolve_carrier(&source)?;
        if kv_backend == KvBackend::Vmm
            && !matches!(carrier.name(), "qwen35" | "deepseek4" | "muse_glimmer")
        {
            return Err(format!(
                "KV backend 'vmm' currently supports qwen3.5, deepseek4, and Muse Glimmer only (selected carrier: {})",
                carrier.name()
            ));
        }
        if kv_backend == KvBackend::Vmm && pp > 1 {
            return Err(
                "KV backend 'vmm' is single-device and does not support pipeline parallelism (pp>1); \
                 use a different kv_cache backend or load with pp=1"
                    .to_string(),
            );
        }
        carrier.admit_topology(arch_id, is_dir, pp, kv_backend)?;
        let topology = if pp > 1 {
            EffectiveTopology::Pipeline(pp)
        } else {
            EffectiveTopology::Single
        };
        (topology, Some(carrier))
    };
    let has_vision = probe_vision(&source, arch_id)?;
    if let ModelSource::Hfq(hfq) = &source {
        df_lash_lm_head_admission(hfq, draft_path, gpu_arch)?;
    }

    Ok(SourceAdmission {
        source,
        arch_id,
        is_dir,
        has_vision,
        topology,
        kv_backend,
        carrier,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every Qwen3.5-family HF config embeds `vision_config` even for text-only
    /// quantized artifacts (the 27B/A3B production files all do). The tower
    /// tensor decides; config markers alone classify as the text backbone,
    /// never refuse. This is the contract from remediation `179a20d7f`.
    #[test]
    fn qwen35_config_marker_without_tower_is_text() {
        assert_eq!(classify_vision(5, false, true).unwrap(), false); // dense
        assert_eq!(classify_vision(6, false, true).unwrap(), false); // MoE
    }

    #[test]
    fn qwen35_tower_tensor_decides_vl() {
        assert_eq!(classify_vision(5, true, false).unwrap(), true);
        assert_eq!(classify_vision(6, true, false).unwrap(), true);
    }

    #[test]
    fn lfm2_tower_without_config_refuses() {
        assert!(classify_vision(11, true, false).is_err());
    }

    #[test]
    fn lfm2_config_without_tower_is_text() {
        assert_eq!(classify_vision(11, false, true).unwrap(), false);
    }

    #[test]
    fn non_vision_archs_are_never_vl() {
        for arch in [0u32, 1, 7, 9, 10, 22] {
            assert_eq!(classify_vision(arch, true, true).unwrap(), false);
        }
    }
}
