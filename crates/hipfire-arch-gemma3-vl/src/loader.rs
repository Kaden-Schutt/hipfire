// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3-VL multimodal weight bundle. See LICENSE / NOTICE.

//! `Gemma3VlWeights` — the full multimodal model: the gemma3 text decoder
//! (loaded from the `language_model.` prefix), the SigLIP vision tower, and the
//! projector. `load_vl` returns the configs + the bundle from one HFQ.

use hipfire_arch_gemma3::{config_from_hfq, load_weights_prefixed, Gemma3Config, Gemma3Weights};
use hipfire_rdna::Gpu;
use hipfire_runtime::hfq::HfqFile;

use crate::config::{vl_config_from_hfq, Gemma3VlConfig};
use crate::projector::ProjectorWeights;
use crate::vision::SigLipWeights;

/// Everything needed to run a Gemma3 multimodal forward.
pub struct Gemma3VlWeights {
    pub text: Gemma3Weights,
    pub vision: SigLipWeights,
    pub projector: ProjectorWeights,
}

impl Gemma3VlWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.text.free_gpu(gpu);
        self.vision.free_gpu(gpu);
        self.projector.free_gpu(gpu);
    }
}

/// Parsed configs + loaded weights for a gemma3 multimodal HFQ.
pub struct LoadedVl {
    pub text_cfg: Gemma3Config,
    pub vl_cfg: Gemma3VlConfig,
    pub weights: Gemma3VlWeights,
    /// Precision of the vision tower (min stored bits over its matrix weights;
    /// bf16/f16 → 16, Q8/Oq8 → 8, …). Drives precision-monotone vision-cache
    /// reuse: a variant only consumes a cached embedding produced at `>=` its
    /// own tier.
    pub vision_tier: u16,
    /// Stable identity of the *source* vision tower (the pre-quant `source_hfq`
    /// path when present, else empty). Quant variants of one base share this, so
    /// their vision embeddings share a cache namespace.
    pub vision_source_id: String,
}

/// Effective stored bits for a vision matrix weight's quant type (higher = more
/// precise). Conservative default (4) for unrecognized codes so an unknown
/// tower is never over-shared to a higher-precision reader.
fn vision_bits(qt: u8) -> u16 {
    match qt {
        2 => 32,         // F32
        1 | 16 => 16,    // F16 / BF16
        3 | 5 | 35 => 8, // Q8F16 / Q8HFQ / Oq8G256
        40 => 6,         // Oq6G256
        38 => 3,         // Oq3G256
        39 => 2,         // Oq2G256
        _ => 4,          // Oq4/OqPlus/HFQ4/MQ4… and unknown → conservative
    }
}

/// Min stored precision (bits) over the vision tower + projector **2D matrix**
/// weights — the tensors whose quantization actually shapes the embedding.
/// Biases/norms (1-D, always float) are excluded so they don't mask a quantized
/// matrix. Empty/absent → 16 (assume float).
fn vision_precision_tier(hfq: &HfqFile) -> u16 {
    hfq.tensors()
        .iter()
        .filter(|t| {
            t.shape.len() == 2
                && (t.name.starts_with("vision_tower.")
                    || t.name.starts_with("multi_modal_projector."))
        })
        .map(|t| vision_bits(t.quant_type))
        .min()
        .unwrap_or(16)
}

/// `source_hfq.path` from the HFQ metadata (the pre-quant source), or `""`.
fn vision_source_id(hfq: &HfqFile) -> String {
    serde_json::from_str::<serde_json::Value>(&hfq.metadata_json)
        .ok()
        .and_then(|m| {
            m.get("source_hfq")
                .and_then(|s| s.get("path"))
                .and_then(|p| p.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default()
}

/// Load a gemma3 multimodal model from `hfq`: text decoder (under
/// `language_model.`), SigLIP tower, and projector.
pub fn load_vl(hfq: &mut HfqFile, gpu: &mut Gpu) -> Result<LoadedVl, String> {
    // Gemma3Config parses the decoder shape from `config.text_config` (its
    // parser prefers the nested block), so it is correct for the multimodal
    // wrapper. Gemma3VlConfig parses vision_config + the mm/splice fields.
    let text_cfg = config_from_hfq(hfq)
        .ok_or_else(|| "gemma3-vl: failed to parse text Gemma3Config".to_string())?;
    let vl_cfg = vl_config_from_hfq(hfq)
        .ok_or_else(|| "gemma3-vl: failed to parse Gemma3VlConfig".to_string())?;

    // Cache identity/precision captured from the index+metadata before the
    // heavy weight loads (cheap: `tensors()` is the in-memory index).
    let vision_tier = vision_precision_tier(hfq);
    let vision_source_id = vision_source_id(hfq);

    let text = load_weights_prefixed(hfq, &text_cfg, gpu, "language_model.")
        .map_err(|e| format!("gemma3-vl: text load failed: {e:?}"))?;
    let vision = SigLipWeights::load(hfq, &vl_cfg.vision, gpu)?;
    let projector = ProjectorWeights::load(hfq, &vl_cfg, gpu)
        .map_err(|e| format!("gemma3-vl: projector load failed: {e:?}"))?;

    Ok(LoadedVl {
        text_cfg,
        vl_cfg,
        weights: Gemma3VlWeights {
            text,
            vision,
            projector,
        },
        vision_tier,
        vision_source_id,
    })
}
