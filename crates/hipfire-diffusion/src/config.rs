// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Diffusion runtime configuration value types: the pipeline / text-encoder /
//! unet / transformer-denoiser / vae / scheduler configs and their derived
//! helpers. Plain data + pure logic, re-exported at the crate root (3.8 Part 2).

use crate::{DiffusionError, DiffusionResult};

#[derive(Debug, Clone, PartialEq)]
pub struct StableDiffusionConfig {
    pub pipeline_class: String,
    pub text_encoder: TextEncoderConfig,
    pub text_encoder_2: Option<TextEncoderConfig>,
    pub unet: UnetConfig,
    pub transformer: Option<TransformerDenoiserConfig>,
    pub vae: VaeConfig,
    pub scheduler: SchedulerConfig,
    pub latent_channels: usize,
    pub latent_height: Option<usize>,
    pub latent_width: Option<usize>,
    pub vae_scale_factor: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextEncoderConfig {
    pub class_name: String,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub vocab_size: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct UnetConfig {
    pub class_name: String,
    pub sample_size: Option<usize>,
    pub in_channels: Option<usize>,
    pub out_channels: Option<usize>,
    pub cross_attention_dim: Option<usize>,
    pub attention_head_dim: Vec<usize>,
    pub block_out_channels: Vec<usize>,
    pub down_block_types: Vec<String>,
    pub up_block_types: Vec<String>,
    pub layers_per_block: Option<usize>,
    pub norm_num_groups: Option<usize>,
    pub norm_eps: Option<f32>,
    pub center_input_sample: bool,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f32,
    pub addition_embed_type: Option<String>,
    pub addition_time_embed_dim: Option<usize>,
    pub projection_class_embeddings_input_dim: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TransformerDenoiserConfig {
    pub class_name: String,
    pub in_channels: Option<usize>,
    pub out_channels: Option<usize>,
    pub patch_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub attention_head_dim: Option<usize>,
    pub cross_attention_dim: Option<usize>,
    pub caption_projection_dim: Option<usize>,
    pub pooled_projection_dim: Option<usize>,
    pub axes_dims_rope: Vec<usize>,
    pub guidance_embeds: Option<bool>,
    pub intermediate_size: Option<usize>,
    pub norm_eps: Option<f32>,
    pub text_hidden_dim: Option<usize>,
    pub text_intermediate_size: Option<usize>,
    pub text_num_attention_heads: Option<usize>,
    pub text_num_key_value_heads: Option<usize>,
    pub num_text_layers: Option<usize>,
    pub num_refiner_text_blocks: Option<usize>,
    pub num_layerwise_text_blocks: Option<usize>,
    pub timestep_embed_dim: Option<usize>,
    pub rope_theta: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransformerDenoiserFamily {
    QwenImage,
    Krea2,
    Unknown,
}

impl TransformerDenoiserFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::QwenImage => "qwen-image-mmdit",
            Self::Krea2 => "krea2-mmdit",
            Self::Unknown => "unknown-transformer",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransformerDenoiserWeightTopology {
    pub(crate) family: TransformerDenoiserFamily,
    pub(crate) block_count: usize,
    pub(crate) has_input_projection: bool,
    pub(crate) has_output_projection: bool,
    pub(crate) has_text_modulation: bool,
    pub(crate) has_text_fusion: bool,
}

impl TransformerDenoiserWeightTopology {
    pub(crate) fn diagnostic_label(&self) -> String {
        let mut features = Vec::new();
        if self.has_input_projection {
            features.push("img_in");
        }
        if self.has_output_projection {
            features.push("output");
        }
        if self.has_text_modulation {
            features.push("text_modulation");
        }
        if self.has_text_fusion {
            features.push("text_fusion");
        }
        let feature_label = if features.is_empty() {
            "no recognized transformer weights".to_string()
        } else {
            features.join(",")
        };
        format!(
            "{} blocks={} features={feature_label}",
            self.family.as_str(),
            self.block_count
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VaeConfig {
    pub class_name: String,
    pub latent_channels: Option<usize>,
    pub z_dim: Option<usize>,
    pub scaling_factor: Option<f32>,
    pub shift_factor: Option<f32>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub block_out_channels: Vec<usize>,
    pub down_block_types: Vec<String>,
    pub up_block_types: Vec<String>,
    pub norm_num_groups: Option<usize>,
    pub norm_eps: Option<f32>,
}

/// Latent-space normalization for a VAE.
///
/// SD/SD2/SDXL VAEs use a single scalar `scaling_factor` with `shift_factor == 0`.
/// Flux/SD3-class VAEs add a non-zero `shift_factor`. Qwen-Image/Wan-class VAEs
/// (`AutoencoderKLQwenImage`) instead publish per-channel `latents_mean`/`latents_std`
/// and carry no scalar scaling factor. When per-channel statistics are present they
/// take precedence and the scalar factors are ignored.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VaeLatentNorm {
    pub(crate) scaling_factor: f32,
    pub(crate) shift_factor: f32,
    pub(crate) latents_mean: Vec<f32>,
    pub(crate) latents_std: Vec<f32>,
}

impl VaeLatentNorm {
    pub(crate) fn from_config(config: &VaeConfig) -> DiffusionResult<Self> {
        let latents_mean = config.latents_mean.clone();
        let latents_std = config.latents_std.clone();
        if latents_mean.len() != latents_std.len() {
            return Err(DiffusionError::InvalidMetadata(format!(
                "VAE latents_mean ({}) and latents_std ({}) must have matching length",
                latents_mean.len(),
                latents_std.len()
            )));
        }
        if latents_std
            .iter()
            .any(|value| value.abs() <= f32::MIN_POSITIVE)
        {
            return Err(DiffusionError::InvalidMetadata(
                "VAE latents_std entries must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            scaling_factor: config.scaling_factor.unwrap_or(0.18215),
            shift_factor: config.shift_factor.unwrap_or(0.0),
            latents_mean,
            latents_std,
        })
    }

    /// Scalar normalization with the given factor (used by tests and kernel probes).
    pub(crate) fn scalar(scaling_factor: f32) -> Self {
        Self {
            scaling_factor,
            shift_factor: 0.0,
            latents_mean: Vec::new(),
            latents_std: Vec::new(),
        }
    }

    /// Per-channel mean/std normalization (Qwen-Image/Wan). When false, the scalar
    /// `scaling_factor`/`shift_factor` path applies.
    pub(crate) fn is_per_channel(&self) -> bool {
        !self.latents_mean.is_empty()
    }

    /// `true` when decode reduces to a single reciprocal scale (the SD/SDXL fast path
    /// that the fused HIP kernel covers).
    pub(crate) fn is_scalar_scale_only(&self) -> bool {
        !self.is_per_channel() && self.shift_factor == 0.0
    }

    fn validate_channels(&self, latent_channels: usize) -> DiffusionResult<()> {
        if self.is_per_channel() && self.latents_mean.len() != latent_channels {
            return Err(DiffusionError::InvalidMetadata(format!(
                "VAE latents_mean length {} does not match latent channel count {latent_channels}",
                self.latents_mean.len()
            )));
        }
        Ok(())
    }

    /// Map a raw VAE distribution mean into latent space, in place. `data` is laid out
    /// NCHW with `latent_channels` channels and `plane` (= H*W) elements per channel.
    pub(crate) fn apply_encode(
        &self,
        data: &mut [f32],
        latent_channels: usize,
        plane: usize,
    ) -> DiffusionResult<()> {
        self.validate_channels(latent_channels)?;
        if self.is_per_channel() {
            let stride = plane.max(1);
            for (idx, value) in data.iter_mut().enumerate() {
                let channel = (idx / stride) % latent_channels;
                *value = (*value - self.latents_mean[channel]) / self.latents_std[channel];
            }
        } else {
            let scale = self.scaling_factor.max(f32::MIN_POSITIVE);
            for value in data.iter_mut() {
                *value = (*value - self.shift_factor) * scale;
            }
        }
        Ok(())
    }

    /// Invert [`apply_encode`]: map latents back into VAE input space, in place.
    pub(crate) fn apply_decode(
        &self,
        data: &mut [f32],
        latent_channels: usize,
        plane: usize,
    ) -> DiffusionResult<()> {
        self.validate_channels(latent_channels)?;
        if self.is_per_channel() {
            let stride = plane.max(1);
            for (idx, value) in data.iter_mut().enumerate() {
                let channel = (idx / stride) % latent_channels;
                *value = *value * self.latents_std[channel] + self.latents_mean[channel];
            }
        } else {
            let scale = self.scaling_factor.max(f32::MIN_POSITIVE);
            for value in data.iter_mut() {
                *value = *value / scale + self.shift_factor;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SchedulerConfig {
    pub class_name: String,
    pub beta_start: Option<f32>,
    pub beta_end: Option<f32>,
    pub beta_schedule: Option<String>,
    pub num_train_timesteps: Option<usize>,
    pub prediction_type: Option<String>,
    pub algorithm_type: Option<String>,
    pub solver_order: Option<usize>,
    pub solver_type: Option<String>,
    pub lower_order_final: Option<bool>,
    pub thresholding: Option<bool>,
    pub dynamic_thresholding_ratio: Option<f32>,
    pub sample_max_value: Option<f32>,
    pub timestep_spacing: Option<String>,
    pub steps_offset: Option<i32>,
    pub use_karras_sigmas: Option<bool>,
    pub set_alpha_to_one: Option<bool>,
    pub shift: Option<f32>,
    pub shift_terminal: Option<f32>,
    pub invert_sigmas: Option<bool>,
    pub use_dynamic_shifting: Option<bool>,
    pub time_shift_type: Option<String>,
    // Resolution-dependent dynamic shifting (FlowMatchEuler): the shift `mu` is
    // interpolated between `base_shift`/`max_shift` over the image token count
    // `[base_image_seq_len, max_image_seq_len]`.
    pub base_shift: Option<f32>,
    pub max_shift: Option<f32>,
    pub base_image_seq_len: Option<usize>,
    pub max_image_seq_len: Option<usize>,
}

impl SchedulerConfig {
    pub fn resolve_request_scheduler(&self, requested: &str) -> DiffusionResult<Self> {
        let normalized = normalize_scheduler_name(requested);
        let karras = normalized.contains(" karras") || normalized == "karras";
        let normalized = normalized.replace(" karras", "");
        if normalized.is_empty()
            || matches!(
                normalized.as_str(),
                "automatic" | "auto" | "default" | "dpm++ 2m" | "dpmpp 2m" | "dpm++2m" | "dpmpp2m"
            )
        {
            let mut config = self.clone();
            if karras {
                config.use_karras_sigmas = Some(true);
            }
            return Ok(config);
        }
        if matches!(
            normalized.as_str(),
            "dpm++ 3m" | "dpmpp 3m" | "dpm++3m" | "dpmpp3m"
        ) {
            let mut config = self.clone();
            config.class_name = "DPMSolverMultistepScheduler".to_string();
            config.algorithm_type = Some("dpmsolver++".to_string());
            config.solver_order = Some(3);
            config.solver_type = config.solver_type.or_else(|| Some("midpoint".to_string()));
            config.lower_order_final = config.lower_order_final.or(Some(true));
            config.thresholding = config.thresholding.or(Some(false));
            if karras {
                config.use_karras_sigmas = Some(true);
            }
            return Ok(config);
        }
        if matches!(
            normalized.as_str(),
            "euler" | "euler a" | "euler ancestral" | "euler_a"
        ) {
            let mut config = self.clone();
            config.class_name = if matches!(
                normalized.as_str(),
                "euler a" | "euler ancestral" | "euler_a"
            ) {
                "EulerAncestralDiscreteScheduler".to_string()
            } else {
                "EulerDiscreteScheduler".to_string()
            };
            config.algorithm_type = None;
            config.solver_order = None;
            config.solver_type = None;
            config.lower_order_final = None;
            config.thresholding = None;
            config.timestep_spacing = None;
            config.steps_offset = None;
            config.use_karras_sigmas = karras.then_some(true);
            config.set_alpha_to_one = None;
            return Ok(config);
        }
        if normalized == "ddim" {
            let mut config = self.clone();
            config.class_name = "DDIMScheduler".to_string();
            config.algorithm_type = None;
            config.solver_order = None;
            config.solver_type = None;
            config.lower_order_final = None;
            config.thresholding = None;
            config.use_karras_sigmas = karras.then_some(true);
            config.set_alpha_to_one = config.set_alpha_to_one.or(Some(true));
            return Ok(config);
        }
        Err(DiffusionError::InvalidRequest(format!(
            "unsupported scheduler {requested:?}; supported schedulers are Automatic, DPM++ 2M, DPM++ 2M Karras, DPM++ 3M, DPM++ 3M Karras, Euler, Euler a, Euler Karras, and DDIM"
        )))
    }
}

pub(crate) fn normalize_scheduler_name(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
