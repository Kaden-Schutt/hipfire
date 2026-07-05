// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Diffusion request / batch / plan value types: the user-facing batch and
//! img2img requests, prompts, conditioning batches, progress/output structs,
//! and the CPU-side latent/image batch buffers. Plain data + pure buffer
//! helpers; re-exported at the crate root (3.8 Part 2 split).

use crate::{
    box_muller_pair, shape4, CpuTensor, DiffusionResult, DiffusionRuntimeKind, DiffusionSchedule,
    SplitMix64,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiffusionBatchRequest {
    pub prompts: Vec<DiffusionPrompt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conditioning: Option<DiffusionExternalConditioningBatch>,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub original_width: Option<u32>,
    #[serde(default)]
    pub original_height: Option<u32>,
    #[serde(default)]
    pub target_width: Option<u32>,
    #[serde(default)]
    pub target_height: Option<u32>,
    #[serde(default)]
    pub seed_resize_from_width: Option<u32>,
    #[serde(default)]
    pub seed_resize_from_height: Option<u32>,
    #[serde(default)]
    pub crop_x: u32,
    #[serde(default)]
    pub crop_y: u32,
    pub steps: u32,
    pub cfg_scale: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distilled_guidance_scale: Option<f32>,
    pub scheduler: String,
    #[serde(default)]
    pub subseed_strength: f32,
    pub send_images: bool,
    pub save_images: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiffusionExternalConditioningBatch {
    pub prompt_embeddings: CpuTensor,
    pub negative_embeddings: CpuTensor,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_attention_mask: Option<CpuTensor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_attention_mask: Option<CpuTensor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_pooled_embeddings: Option<CpuTensor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_pooled_embeddings: Option<CpuTensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiffusionImg2ImgRequest {
    pub batch: DiffusionBatchRequest,
    pub init_image: RgbImageBatch,
    #[serde(default)]
    pub mask: Option<RgbImageBatch>,
    #[serde(default)]
    pub inpainting_fill: Option<u32>,
    #[serde(default)]
    pub resize_mode: DiffusionImg2ImgResizeMode,
    pub denoising_strength: f32,
    /// When set, overrides `denoising_strength` with an explicit MrFlow
    /// direct-sigma refine schedule: the high-resolution refine pass of staged
    /// sampling (low-res generate -> pixel-space SR -> re-encode -> short
    /// refine). Flow-match backbones only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine_sigma: Option<RefineSigmaSchedule>,
}

/// MrFlow "direct sigma" refine schedule parameters. See
/// [`DiffusionSchedule::refine_direct_sigma`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RefineSigmaSchedule {
    /// Direct start sigma for the refine pass; `0 < first_sigma < 1`. MrFlow
    /// Krea-2 presets use `0.11`-`0.16`.
    pub first_sigma: f32,
    /// Number of refine denoise steps. MrFlow uses `1`.
    pub steps: u32,
    /// Use the flow-match shifted interior schedule. Only differs from the
    /// linear ramp when `steps > 1`.
    #[serde(default)]
    pub shifted: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DiffusionImg2ImgResizeMode {
    #[default]
    Image,
    Latent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiffusionPrompt {
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: i64,
    #[serde(default)]
    pub subseed: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionBatchOutput {
    pub images: Vec<String>,
    pub info: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionProgress {
    pub completed_steps: usize,
    pub total_steps: usize,
    pub timestep: usize,
    pub preview_latents: Option<LatentBatch>,
}

pub struct MaskedDenoiseReference<'a> {
    pub init_latents: &'a LatentBatch,
    pub noise: &'a [f32],
    pub mask_weights: &'a [f32],
    pub source_schedule: &'a DiffusionSchedule,
    pub start_step: usize,
}

pub struct InpaintDenoiseConditioning {
    pub mask_weights: Vec<f32>,
    pub masked_image_latents: LatentBatch,
}

pub struct SdxlDenoiseConditioning<'a> {
    pub text_embeds: &'a CpuTensor,
    pub time_ids: &'a CpuTensor,
}

pub(crate) struct DenoiseLatentsOutput {
    pub(crate) latents: LatentBatch,
    pub(crate) runtime_kind: DiffusionRuntimeKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionConditioningBatch {
    pub prompt_tokens: Vec<Vec<u32>>,
    pub negative_tokens: Vec<Vec<u32>>,
    pub prompt_tokens_2: Option<Vec<Vec<u32>>>,
    pub negative_tokens_2: Option<Vec<Vec<u32>>>,
    pub prompt_embeddings: Option<CpuTensor>,
    pub negative_embeddings: Option<CpuTensor>,
    pub prompt_embeddings_2: Option<CpuTensor>,
    pub negative_embeddings_2: Option<CpuTensor>,
    pub prompt_cross_attention_embeddings: Option<CpuTensor>,
    pub negative_cross_attention_embeddings: Option<CpuTensor>,
    pub prompt_attention_mask: Option<CpuTensor>,
    pub negative_attention_mask: Option<CpuTensor>,
    pub prompt_pooled_embeddings: Option<CpuTensor>,
    pub negative_pooled_embeddings: Option<CpuTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionLatentShape {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionRunPlan {
    pub latent_shape: DiffusionLatentShape,
    pub latents: LatentBatch,
    pub schedule: DiffusionSchedule,
    pub conditioning: DiffusionConditioningBatch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RgbImageBatch {
    pub batch: usize,
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatentBatch {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl LatentBatch {
    pub fn seeded_normal(
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        seeds: &[i64],
    ) -> Self {
        let mut data = Vec::with_capacity(batch * channels * height * width);
        for b in 0..batch {
            let mut rng = SplitMix64::new(seeds.get(b).copied().unwrap_or(-1) as u64);
            let count = channels * height * width;
            let mut i = 0;
            while i < count {
                let (a, next) = box_muller_pair(&mut rng);
                data.push(a);
                i += 1;
                if i < count {
                    data.push(next);
                    i += 1;
                }
            }
        }
        Self {
            batch,
            channels,
            height,
            width,
            data,
        }
    }

    pub fn len_per_batch(&self) -> usize {
        self.channels * self.height * self.width
    }

    pub fn as_nchw_tensor(&self) -> CpuTensor {
        CpuTensor {
            shape: vec![self.batch, self.channels, self.height, self.width],
            data: self.data.clone(),
        }
    }

    pub fn from_nchw_tensor(tensor: CpuTensor) -> DiffusionResult<Self> {
        let [batch, channels, height, width] = shape4(&tensor)?;
        Ok(Self {
            batch,
            channels,
            height,
            width,
            data: tensor.data,
        })
    }
}
