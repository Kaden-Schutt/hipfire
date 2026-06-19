// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3-VL config parsing. See LICENSE / NOTICE.

//! [`SigLipConfig`] (the vision tower) and [`Gemma3VlConfig`] (vision +
//! projector + splice params), parsed from the HFQ-embedded `config.json`
//! (`vision_config` + the top-level multimodal fields). The text-decoder shape
//! comes from `hipfire_arch_gemma3::Gemma3Config` over the same metadata.

use hipfire_runtime::hfq::HfqFile;

/// SigLIP vision-tower shape constants (`vision_config`, model_type
/// `siglip_vision_model`). Standard ViT: LayerNorm(+bias), bidirectional
/// attention, gelu-tanh MLP, learned position embeddings, no pooling head.
#[derive(Debug, Clone, PartialEq)]
pub struct SigLipConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub num_channels: usize,
    pub layer_norm_eps: f32,
}

impl SigLipConfig {
    /// Per-side patch count: `image_size / patch_size` (896/14 = 64).
    pub fn grid_side(&self) -> usize {
        self.image_size / self.patch_size
    }
    /// Total patches = `grid_side²` (64² = 4096) — the encoder sequence length
    /// and the learned position-embedding table size.
    pub fn num_patches(&self) -> usize {
        let g = self.grid_side();
        g * g
    }
    /// Attention head dim = `hidden_size / num_attention_heads` (1152/16 = 72).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Full Gemma3 multimodal config: the SigLIP tower + the projector/splice
/// parameters. The text decoder uses `hipfire_arch_gemma3::Gemma3Config` parsed
/// from the same metadata (`text_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma3VlConfig {
    pub vision: SigLipConfig,
    /// Image soft-tokens per image after pooling (= projector output rows). 256.
    pub mm_tokens_per_image: usize,
    /// Placeholder token id in the text stream that image embeddings replace.
    pub image_token_index: u32,
    /// Begin/end-of-image delimiter token ids.
    pub boi_token_index: u32,
    pub eoi_token_index: u32,
    /// Projector output dim = text hidden_size (`mm_input_projection`: 1152 → 2560).
    pub text_hidden_size: usize,
    /// `(1+w)` offset baked into the projector's `mm_soft_emb_norm` RMSNorm at
    /// ingest (gemma convention); `0.0` if absent.
    pub gemma_norm_offset: f32,
}

impl Gemma3VlConfig {
    /// Pool factor per side mapping the `grid_side²` patches down to
    /// `mm_tokens_per_image` (64² → 16² ⇒ factor 4). Gemma3 avg-pools the
    /// vision grid by this factor before the projector.
    pub fn pool_side(&self) -> usize {
        // sqrt(mm_tokens_per_image), e.g. sqrt(256) = 16.
        (self.mm_tokens_per_image as f64).sqrt() as usize
    }
    pub fn pool_factor(&self) -> usize {
        self.vision.grid_side() / self.pool_side()
    }
}

/// Parse the multimodal config from an HFQ file's embedded metadata.
pub fn vl_config_from_hfq(hfq: &HfqFile) -> Option<Gemma3VlConfig> {
    vl_config_from_metadata_json(&hfq.metadata_json)
}

/// Inner parser, decoupled from `HfqFile` for unit testability.
pub fn vl_config_from_metadata_json(metadata_json: &str) -> Option<Gemma3VlConfig> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
    let config = meta.get("config")?;
    let vc = config.get("vision_config")?;

    let hidden_size = vc.get("hidden_size")?.as_u64()? as usize;
    let vision = SigLipConfig {
        hidden_size,
        num_hidden_layers: vc.get("num_hidden_layers")?.as_u64()? as usize,
        num_attention_heads: vc
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize,
        intermediate_size: vc.get("intermediate_size")?.as_u64()? as usize,
        patch_size: vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(14) as usize,
        image_size: vc.get("image_size").and_then(|v| v.as_u64()).unwrap_or(896) as usize,
        num_channels: vc.get("num_channels").and_then(|v| v.as_u64()).unwrap_or(3) as usize,
        layer_norm_eps: vc
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32,
    };

    let mm_tokens_per_image = config
        .get("mm_tokens_per_image")
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as usize;
    let image_token_index = config
        .get("image_token_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(262144) as u32;
    let boi_token_index = config
        .get("boi_token_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(255999) as u32;
    let eoi_token_index = config
        .get("eoi_token_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(256000) as u32;
    let text_hidden_size = config
        .get("text_config")
        .and_then(|tc| tc.get("hidden_size"))
        .and_then(|v| v.as_u64())? as usize;
    let gemma_norm_offset = meta
        .get("gemma_norm_offset")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;

    Some(Gemma3VlConfig {
        vision,
        mm_tokens_per_image,
        image_token_index,
        boi_token_index,
        eoi_token_index,
        text_hidden_size,
        gemma_norm_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn medgemma_4b_metadata() -> String {
        serde_json::json!({
            "architecture": "gemma3",
            "gemma_norm_offset": 1.0,
            "config": {
                "model_type": "gemma3",
                "image_token_index": 262144,
                "boi_token_index": 255999,
                "eoi_token_index": 256000,
                "mm_tokens_per_image": 256,
                "vision_config": {
                    "model_type": "siglip_vision_model",
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "image_size": 896,
                    "patch_size": 14,
                    "num_channels": 3,
                    "layer_norm_eps": 1e-6
                },
                "text_config": { "hidden_size": 2560 }
            }
        })
        .to_string()
    }

    #[test]
    fn parses_medgemma_4b_vision() {
        let c = vl_config_from_metadata_json(&medgemma_4b_metadata()).unwrap();
        assert_eq!(c.vision.hidden_size, 1152);
        assert_eq!(c.vision.num_hidden_layers, 27);
        assert_eq!(c.vision.head_dim(), 72); // 1152 / 16
        assert_eq!(c.vision.grid_side(), 64); // 896 / 14
        assert_eq!(c.vision.num_patches(), 4096); // 64²
        assert_eq!(c.mm_tokens_per_image, 256);
        assert_eq!(c.image_token_index, 262144);
        assert_eq!(c.text_hidden_size, 2560);
        assert_eq!(c.gemma_norm_offset, 1.0);
    }

    #[test]
    fn pooling_4096_to_256() {
        let c = vl_config_from_metadata_json(&medgemma_4b_metadata()).unwrap();
        assert_eq!(c.pool_side(), 16); // sqrt(256)
        assert_eq!(c.pool_factor(), 4); // 64 / 16 — avg-pool 4x4 over the grid
    }
}
