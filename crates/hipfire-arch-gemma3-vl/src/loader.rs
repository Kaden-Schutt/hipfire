// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3-VL multimodal weight bundle. See LICENSE / NOTICE.

//! `Gemma3VlWeights` — the full multimodal model: the gemma3 text decoder
//! (loaded from the `language_model.` prefix), the SigLIP vision tower, and the
//! projector. `load_vl` returns the configs + the bundle from one HFQ.

use hipfire_arch_gemma3::{config_from_hfq, load_weights_prefixed, Gemma3Config, Gemma3Weights};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

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
    })
}
