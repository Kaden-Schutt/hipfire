// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Diffusion model support for HFQ-backed Hipfire serving.
//!
//! This crate owns the stable metadata and batched runtime API for diffusion
//! artifacts. The first importer preserves Diffusers component weights as HFQ
//! role entries; later importer phases can replace those entries with decoded
//! and quantized tensors without changing server routing.

use base64::Engine;
use hipfire_runtime::hfq::{write_hfqm_package_streaming, HfqFile, HfqStreamEntry};
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const DIFFUSION_ARTIFACT_KIND: &str = "diffusion";
pub const DIFFUSION_SCHEMA_VERSION: u32 = 1;
/// Reserved arch_id for HFQ diffusion containers. The value is ASCII-ish
/// "DIF0", outside the existing small integer LLM architecture ids.
pub const HFQ_ARCH_DIFFUSION: u32 = 0x3046_4944;

pub const QT_DIFFUSION_JSON: u8 = 240;
pub const QT_DIFFUSION_TOKENIZER: u8 = 241;
pub const QT_DIFFUSION_SOURCE_WEIGHTS: u8 = 242;
pub const QT_DIFFUSION_TENSOR_Q4F16_G64: u8 = 0;
pub const QT_DIFFUSION_TENSOR_F16: u8 = 1;
pub const QT_DIFFUSION_TENSOR_F32: u8 = 2;
pub const QT_DIFFUSION_TENSOR_Q8F16: u8 = 3;
pub const QT_DIFFUSION_TENSOR_Q4_K: u8 = 4;
pub const QT_DIFFUSION_TENSOR_HFQ4_G256: u8 = 6;
pub const QT_DIFFUSION_TENSOR_HFQ4_G128: u8 = 7;
pub const QT_DIFFUSION_TENSOR_HFQ6_G256: u8 = 8;
pub const QT_DIFFUSION_TENSOR_BF16: u8 = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionHfqMetadata {
    pub artifact_kind: String,
    pub schema_version: u32,
    pub pipeline: DiffusionPipelineMetadata,
    #[serde(default)]
    pub tokenizer: DiffusionTokenizerMetadata,
    #[serde(default)]
    pub tokenizer_2: Option<DiffusionTokenizerMetadata>,
    #[serde(default)]
    pub batch: DiffusionBatchMetadata,
    #[serde(default)]
    pub quantization: DiffusionQuantizationMetadata,
    #[serde(default)]
    pub components: BTreeMap<String, DiffusionComponentMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionPipelineMetadata {
    pub class_name: String,
    pub source: String,
    #[serde(default)]
    pub model_name: String,
    #[serde(default)]
    pub latent_channels: Option<u32>,
    #[serde(default)]
    pub latent_height: Option<u32>,
    #[serde(default)]
    pub latent_width: Option<u32>,
    #[serde(default)]
    pub supported_widths: Vec<u32>,
    #[serde(default)]
    pub supported_heights: Vec<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionTokenizerMetadata {
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub max_length: Option<u32>,
    #[serde(default)]
    pub entries: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionBatchMetadata {
    pub max_batch: u32,
    pub batched_runtime: bool,
}

impl Default for DiffusionBatchMetadata {
    fn default() -> Self {
        Self {
            max_batch: 1,
            batched_runtime: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionQuantizationMetadata {
    pub weight_format: String,
    pub activation_format: String,
    pub tensor_roles_version: u32,
}

impl Default for DiffusionQuantizationMetadata {
    fn default() -> Self {
        Self {
            weight_format: "source".to_string(),
            activation_format: "fp16".to_string(),
            tensor_roles_version: 1,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionComponentMetadata {
    #[serde(default)]
    pub class_name: Option<String>,
    #[serde(default)]
    pub config_entry: Option<String>,
    #[serde(default)]
    pub weight_entries: Vec<String>,
    #[serde(default)]
    pub tensor_roles: Vec<DiffusionTensorRole>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionTensorRole {
    pub role: String,
    pub entry: String,
    pub dtype: String,
    #[serde(default)]
    pub quant_format: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffusionModelSummary {
    pub path: PathBuf,
    pub title: String,
    pub model_name: String,
    pub pipeline_class: String,
    pub max_batch: u32,
    pub weight_format: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionRuntimeKind {
    CpuSourceReference,
    RocmHybridReference,
}

impl DiffusionRuntimeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuSourceReference => "cpu-source-reference",
            Self::RocmHybridReference => "rocm-hybrid-reference",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffusionRuntimeCapabilities {
    pub kind: DiffusionRuntimeKind,
    pub weight_format: String,
    pub activation_format: String,
    pub tensor_roles_version: u32,
    pub max_batch: u32,
    pub supports_img2img: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionHipRuntimeOptions {
    pub device_id: i32,
}

impl Default for DiffusionHipRuntimeOptions {
    fn default() -> Self {
        Self { device_id: 0 }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionGenerationRuntimeOptions {
    pub rocm_device_id: Option<i32>,
}

impl DiffusionGenerationRuntimeOptions {
    pub fn cpu_reference() -> Self {
        Self {
            rocm_device_id: None,
        }
    }

    pub fn rocm_hybrid(device_id: i32) -> Self {
        Self {
            rocm_device_id: Some(device_id),
        }
    }
}

struct DiffusionGenerationRuntimeContext {
    options: DiffusionGenerationRuntimeOptions,
    #[cfg(feature = "rocm")]
    rocm_gpu: Option<rdna_compute::Gpu>,
    #[cfg(feature = "rocm")]
    rocm_gpu_init_count: usize,
}

impl DiffusionGenerationRuntimeContext {
    fn new(options: DiffusionGenerationRuntimeOptions) -> Self {
        Self {
            options,
            #[cfg(feature = "rocm")]
            rocm_gpu: None,
            #[cfg(feature = "rocm")]
            rocm_gpu_init_count: 0,
        }
    }

    fn rocm_device_id(&self) -> Option<i32> {
        self.options.rocm_device_id
    }

    #[cfg(feature = "rocm")]
    fn with_rocm_gpu<T>(
        &mut self,
        f: impl FnOnce(&mut rdna_compute::Gpu) -> DiffusionResult<T>,
    ) -> DiffusionResult<T> {
        let Some(device_id) = self.options.rocm_device_id else {
            return Err(DiffusionError::BackendUnavailable(
                "ROCm runtime context was requested without a device id".to_string(),
            ));
        };
        if self.rocm_gpu.is_none() {
            let gpu = rdna_compute::Gpu::init_with_device(device_id)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            self.rocm_gpu = Some(gpu);
            self.rocm_gpu_init_count += 1;
        }
        let gpu = self.rocm_gpu.as_mut().ok_or_else(|| {
            DiffusionError::BackendUnavailable(
                "ROCm runtime context failed to retain initialized GPU".to_string(),
            )
        })?;
        f(gpu)
    }

    #[cfg(all(feature = "rocm", test))]
    fn rocm_gpu_init_count(&self) -> usize {
        self.rocm_gpu_init_count
    }
}

fn runtime_kind_for_context(
    runtime_context: &DiffusionGenerationRuntimeContext,
) -> DiffusionRuntimeKind {
    if runtime_context.rocm_device_id().is_some() {
        DiffusionRuntimeKind::RocmHybridReference
    } else {
        DiffusionRuntimeKind::CpuSourceReference
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionHipMemoryPlan {
    pub latent_shape: DiffusionLatentShape,
    pub latent_bytes: usize,
    pub denoise_input_bytes: usize,
    pub conditioning_bytes: usize,
    pub vae_decode_bytes: usize,
    pub rgb_bytes: usize,
    pub scheduler_scratch_bytes: usize,
    pub total_device_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionHipPreflight {
    pub device_id: i32,
    pub arch: String,
    pub integrated: bool,
    pub memory_plan: DiffusionHipMemoryPlan,
    pub probe_bytes: usize,
    pub kernel_probe: DiffusionHipKernelProbe,
    pub rgb_to_vae_tensor_kernel_probe: DiffusionHipKernelProbe,
    pub latent_mask_weights_kernel_probe: DiffusionHipKernelProbe,
    pub masked_rgb_inpaint_kernel_probe: DiffusionHipKernelProbe,
    pub blend_latents_with_mask_kernel_probe: DiffusionHipKernelProbe,
    pub model_input_kernel_probe: DiffusionHipKernelProbe,
    pub guidance_kernel_probe: DiffusionHipKernelProbe,
    pub scheduler_kernel_probe: DiffusionHipKernelProbe,
    pub center_unet_input_kernel_probe: DiffusionHipKernelProbe,
    pub timestep_embedding_kernel_probe: DiffusionHipKernelProbe,
    pub clip_token_position_embedding_kernel_probe: DiffusionHipKernelProbe,
    pub tensor_add_kernel_probe: DiffusionHipKernelProbe,
    pub add_channel_bias_kernel_probe: DiffusionHipKernelProbe,
    pub nchw_to_bsc_kernel_probe: DiffusionHipKernelProbe,
    pub bsc_to_nchw_kernel_probe: DiffusionHipKernelProbe,
    pub concat_channels_kernel_probe: DiffusionHipKernelProbe,
    pub concat_last_dim_2d_kernel_probe: DiffusionHipKernelProbe,
    pub concat_last_dim_3d_kernel_probe: DiffusionHipKernelProbe,
    pub conv2d_kernel_probe: DiffusionHipKernelProbe,
    pub group_norm_kernel_probe: DiffusionHipKernelProbe,
    pub silu_kernel_probe: DiffusionHipKernelProbe,
    pub quick_gelu_kernel_probe: DiffusionHipKernelProbe,
    pub upsample_kernel_probe: DiffusionHipKernelProbe,
    pub linear_kernel_probe: DiffusionHipKernelProbe,
    pub layer_norm_kernel_probe: DiffusionHipKernelProbe,
    pub softmax_kernel_probe: DiffusionHipKernelProbe,
    pub sdpa_kernel_probe: DiffusionHipKernelProbe,
    pub clip_causal_attention_kernel_probe: DiffusionHipKernelProbe,
    pub geglu_gate_kernel_probe: DiffusionHipKernelProbe,
    pub vae_moments_to_latents_kernel_probe: DiffusionHipKernelProbe,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionHipKernelProbe {
    pub name: String,
    pub input_elements: usize,
    pub output_bytes: usize,
    pub matched_cpu_reference: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffusionRuntimeSupport {
    pub supported: bool,
    pub runtime_kind: Option<DiffusionRuntimeKind>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffusionHfqInspection {
    pub summary: DiffusionModelSummary,
    pub runtime_support: DiffusionRuntimeSupport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiffusionBatchRequest {
    pub prompts: Vec<DiffusionPrompt>,
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
    pub scheduler: String,
    #[serde(default)]
    pub subseed_strength: f32,
    pub send_images: bool,
    pub save_images: bool,
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

struct DenoiseLatentsOutput {
    latents: LatentBatch,
    runtime_kind: DiffusionRuntimeKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StableDiffusionConfig {
    pub pipeline_class: String,
    pub text_encoder: TextEncoderConfig,
    pub text_encoder_2: Option<TextEncoderConfig>,
    pub unet: UnetConfig,
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
pub struct VaeConfig {
    pub class_name: String,
    pub latent_channels: Option<usize>,
    pub z_dim: Option<usize>,
    pub scaling_factor: Option<f32>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub block_out_channels: Vec<usize>,
    pub down_block_types: Vec<String>,
    pub up_block_types: Vec<String>,
    pub norm_num_groups: Option<usize>,
    pub norm_eps: Option<f32>,
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

fn normalize_scheduler_name(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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

fn seeded_latents_for_request(
    config: &StableDiffusionConfig,
    request: &DiffusionBatchRequest,
    latent_shape: &DiffusionLatentShape,
    seeds: &[i64],
) -> DiffusionResult<LatentBatch> {
    let scale = config.vae_scale_factor.max(1) as u32;
    let seed_width = request.seed_resize_from_width.unwrap_or(request.width);
    let seed_height = request.seed_resize_from_height.unwrap_or(request.height);
    if seed_width == 0 || seed_height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "seed resize dimensions must be positive".to_string(),
        ));
    }
    if seed_width % scale != 0 || seed_height % scale != 0 {
        return Err(DiffusionError::InvalidRequest(format!(
            "seed resize dimensions {seed_width}x{seed_height} must be divisible by VAE scale factor {scale}",
        )));
    }
    let seed_latent_width = (seed_width / scale) as usize;
    let seed_latent_height = (seed_height / scale) as usize;
    let latents = LatentBatch::seeded_normal(
        latent_shape.batch,
        latent_shape.channels,
        seed_latent_height,
        seed_latent_width,
        seeds,
    );
    resize_latent_batch_nearest(&latents, latent_shape.height, latent_shape.width)
}

fn blend_subseed_latents(
    config: &StableDiffusionConfig,
    latents: &mut LatentBatch,
    request: &DiffusionBatchRequest,
    latent_shape: &DiffusionLatentShape,
) -> DiffusionResult<()> {
    let strength = request.subseed_strength.clamp(0.0, 1.0);
    if strength <= 0.0
        || request
            .prompts
            .iter()
            .all(|prompt| prompt.subseed.is_none())
    {
        return Ok(());
    }
    let subseeds = request
        .prompts
        .iter()
        .map(|prompt| prompt.subseed.unwrap_or(prompt.seed))
        .collect::<Vec<_>>();
    let subseed_latents = seeded_latents_for_request(config, request, latent_shape, &subseeds)?;
    let image_len = latents.len_per_batch();
    for (batch_idx, prompt) in request.prompts.iter().enumerate() {
        if prompt.subseed.is_none() {
            continue;
        }
        let offset = batch_idx * image_len;
        for idx in offset..offset + image_len {
            latents.data[idx] =
                latents.data[idx] * (1.0 - strength) + subseed_latents.data[idx] * strength;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionSchedule {
    pub timesteps: Vec<f32>,
    pub sigmas: Vec<f32>,
    pub prediction_type: SchedulerPredictionType,
    pub input_scaling: SchedulerInputScaling,
    pub solver: SchedulerSolver,
    train_timesteps: Vec<usize>,
    alpha_t: Vec<f32>,
    sigma_t: Vec<f32>,
    lambda_t: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SchedulerSolver {
    Euler,
    FlowMatchEuler,
    Ddim {
        set_alpha_to_one: bool,
    },
    DpmSolverMultistep {
        algorithm_type: DpmSolverAlgorithm,
        solver_order: usize,
        solver_type: DpmSolverType,
        lower_order_final: bool,
        thresholding: bool,
        dynamic_thresholding_ratio: f32,
        sample_max_value: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpmSolverAlgorithm {
    DpmSolverPlusPlus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpmSolverType {
    Midpoint,
    Heun,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SchedulerStepState {
    model_outputs: Vec<Vec<f32>>,
    lower_order_nums: usize,
}

impl SchedulerSolver {
    fn from_config(config: &SchedulerConfig) -> DiffusionResult<Self> {
        if config.class_name == "FlowMatchEulerDiscreteScheduler" {
            return Ok(Self::FlowMatchEuler);
        }
        if config.class_name == "DDIMScheduler" {
            return Ok(Self::Ddim {
                set_alpha_to_one: config.set_alpha_to_one.unwrap_or(true),
            });
        }
        if config.class_name != "DPMSolverMultistepScheduler" {
            return Ok(Self::Euler);
        }
        let algorithm_type = match config.algorithm_type.as_deref().unwrap_or("dpmsolver++") {
            "dpmsolver++" => DpmSolverAlgorithm::DpmSolverPlusPlus,
            other => {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "unsupported DPM-Solver algorithm_type {other:?}"
                )));
            }
        };
        let solver_type = match config.solver_type.as_deref().unwrap_or("midpoint") {
            "midpoint" => DpmSolverType::Midpoint,
            "heun" => DpmSolverType::Heun,
            other => {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "unsupported DPM-Solver solver_type {other:?}"
                )));
            }
        };
        let solver_order = config.solver_order.unwrap_or(2);
        if !(1..=3).contains(&solver_order) {
            return Err(DiffusionError::InvalidMetadata(format!(
                "unsupported DPM-Solver solver_order {solver_order}; only 1, 2, and 3 are implemented"
            )));
        }
        Ok(Self::DpmSolverMultistep {
            algorithm_type,
            solver_order,
            solver_type,
            lower_order_final: config.lower_order_final.unwrap_or(true),
            thresholding: config.thresholding.unwrap_or(false),
            dynamic_thresholding_ratio: normalize_dynamic_thresholding_ratio(
                config.dynamic_thresholding_ratio,
            ),
            sample_max_value: normalize_dynamic_thresholding_sample_max(config.sample_max_value),
        })
    }
}

fn normalize_dynamic_thresholding_ratio(value: Option<f32>) -> f32 {
    match value {
        Some(value) if value.is_finite() => value.clamp(0.0, 1.0),
        _ => 0.995,
    }
}

fn normalize_dynamic_thresholding_sample_max(value: Option<f32>) -> f32 {
    match value {
        Some(value) if value.is_finite() => value.max(1.0),
        _ => 1.0,
    }
}

fn dynamic_threshold_sample(
    data: &mut [f32],
    shape: &[usize],
    ratio: f32,
    sample_max_value: f32,
) -> DiffusionResult<()> {
    let batch = shape.first().copied().ok_or_else(|| {
        DiffusionError::InvalidMetadata(
            "DPM-Solver dynamic thresholding requires a batch dimension".to_string(),
        )
    })?;
    if batch == 0 || data.is_empty() {
        return Ok(());
    }
    if !data.len().is_multiple_of(batch) {
        return Err(DiffusionError::InvalidMetadata(format!(
            "DPM-Solver dynamic thresholding data length {} is not divisible by batch {batch}",
            data.len()
        )));
    }
    let values_per_batch = data.len() / batch;
    if values_per_batch == 0 {
        return Ok(());
    }

    let ratio = normalize_dynamic_thresholding_ratio(Some(ratio));
    let sample_max_value = normalize_dynamic_thresholding_sample_max(Some(sample_max_value));
    let mut sorted_abs = Vec::with_capacity(values_per_batch);
    for chunk in data.chunks_mut(values_per_batch) {
        sorted_abs.clear();
        sorted_abs.extend(chunk.iter().map(|value| value.abs()));
        sorted_abs.sort_by(|left, right| left.total_cmp(right));

        let threshold = if sorted_abs.len() == 1 {
            sorted_abs[0]
        } else {
            let rank = ratio * (sorted_abs.len() - 1) as f32;
            let lower = rank.floor() as usize;
            let upper = rank.ceil() as usize;
            let frac = rank - lower as f32;
            sorted_abs[lower] + (sorted_abs[upper] - sorted_abs[lower]) * frac
        };
        let threshold = threshold.clamp(1.0, sample_max_value);
        for value in chunk {
            *value = value.clamp(-threshold, threshold) / threshold;
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerPredictionType {
    Epsilon,
    Sample,
    VPrediction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerInputScaling {
    None,
    Sigma,
}

impl SchedulerInputScaling {
    fn from_scheduler_class(class_name: &str) -> Self {
        match class_name {
            "EulerDiscreteScheduler" | "EulerAncestralDiscreteScheduler" => Self::Sigma,
            _ => Self::None,
        }
    }
}

impl SchedulerPredictionType {
    fn from_config(value: Option<&str>) -> DiffusionResult<Self> {
        match value.unwrap_or("epsilon") {
            "epsilon" => Ok(Self::Epsilon),
            "sample" => Ok(Self::Sample),
            "v_prediction" => Ok(Self::VPrediction),
            other => Err(DiffusionError::InvalidMetadata(format!(
                "unsupported scheduler prediction_type {other:?}"
            ))),
        }
    }
}

impl DiffusionSchedule {
    pub fn linear(steps: u32) -> DiffusionResult<Self> {
        if steps == 0 {
            return Err(DiffusionError::InvalidRequest(
                "scheduler steps must be greater than zero".to_string(),
            ));
        }
        let steps = steps as usize;
        let mut timesteps = Vec::with_capacity(steps);
        let mut sigmas = Vec::with_capacity(steps + 1);
        for idx in 0..steps {
            let frac = if steps == 1 {
                1.0
            } else {
                1.0 - idx as f32 / (steps - 1) as f32
            };
            timesteps.push(frac);
            sigmas.push(frac);
        }
        sigmas.push(0.0);
        Ok(Self {
            timesteps,
            sigmas,
            prediction_type: SchedulerPredictionType::Epsilon,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::Euler,
            train_timesteps: Vec::new(),
            alpha_t: Vec::new(),
            sigma_t: Vec::new(),
            lambda_t: Vec::new(),
        })
    }

    pub fn from_config(config: &SchedulerConfig, steps: u32) -> DiffusionResult<Self> {
        if steps == 0 {
            return Err(DiffusionError::InvalidRequest(
                "scheduler steps must be greater than zero".to_string(),
            ));
        }
        if config.class_name == "FlowMatchEulerDiscreteScheduler" {
            return Self::flow_match_euler(config, steps);
        }
        let (Some(beta_start), Some(beta_end), Some(num_train_timesteps)) = (
            config.beta_start,
            config.beta_end,
            config.num_train_timesteps,
        ) else {
            let mut schedule = Self::linear(steps)?;
            schedule.prediction_type =
                SchedulerPredictionType::from_config(config.prediction_type.as_deref())?;
            schedule.input_scaling =
                SchedulerInputScaling::from_scheduler_class(&config.class_name);
            return Ok(schedule);
        };
        if beta_start <= 0.0 || beta_end <= 0.0 || num_train_timesteps == 0 {
            let mut schedule = Self::linear(steps)?;
            schedule.prediction_type =
                SchedulerPredictionType::from_config(config.prediction_type.as_deref())?;
            schedule.input_scaling =
                SchedulerInputScaling::from_scheduler_class(&config.class_name);
            return Ok(schedule);
        }
        let prediction_type =
            SchedulerPredictionType::from_config(config.prediction_type.as_deref())?;
        let input_scaling = SchedulerInputScaling::from_scheduler_class(&config.class_name);
        let solver = SchedulerSolver::from_config(config)?;
        let betas = scheduler_betas(
            beta_start,
            beta_end,
            num_train_timesteps,
            config.beta_schedule.as_deref().unwrap_or("linear"),
        )?;
        let alpha_cumprod = betas
            .iter()
            .scan(1.0f32, |acc, beta| {
                *acc *= 1.0 - beta;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        let mut train_indices =
            inference_train_timesteps(config, num_train_timesteps, steps as usize)?;
        let mut timesteps = Vec::with_capacity(train_indices.len());
        let mut sigmas = Vec::with_capacity(train_indices.len() + 1);
        for idx in &train_indices {
            let alpha = alpha_cumprod[*idx].clamp(f32::MIN_POSITIVE, 1.0);
            timesteps.push(*idx as f32);
            sigmas.push(((1.0 - alpha) / alpha).max(0.0).sqrt());
        }
        sigmas.push(0.0);
        if config.use_karras_sigmas.unwrap_or(false) && sigmas.len() > 1 {
            let training_sigmas = alpha_cumprod
                .iter()
                .map(|alpha| {
                    let alpha = alpha.clamp(f32::MIN_POSITIVE, 1.0);
                    ((1.0 - alpha) / alpha).max(0.0).sqrt()
                })
                .collect::<Vec<_>>();
            sigmas = karras_sigmas(&sigmas[..sigmas.len() - 1]);
            train_indices = sigmas[..sigmas.len() - 1]
                .iter()
                .map(|sigma| nearest_training_timestep_for_sigma(&training_sigmas, *sigma))
                .collect();
            timesteps = train_indices.iter().map(|idx| *idx as f32).collect();
        }
        let mut alpha_t = Vec::with_capacity(alpha_cumprod.len());
        let mut sigma_t = Vec::with_capacity(alpha_cumprod.len());
        let mut lambda_t = Vec::with_capacity(alpha_cumprod.len());
        for alpha_cumprod in &alpha_cumprod {
            let alpha = alpha_cumprod.clamp(f32::MIN_POSITIVE, 1.0).sqrt();
            let sigma = (1.0 - alpha_cumprod).max(f32::MIN_POSITIVE).sqrt();
            alpha_t.push(alpha);
            sigma_t.push(sigma);
            lambda_t.push(alpha.ln() - sigma.ln());
        }
        Ok(Self {
            timesteps,
            sigmas,
            prediction_type,
            input_scaling,
            solver,
            train_timesteps: train_indices,
            alpha_t,
            sigma_t,
            lambda_t,
        })
    }

    fn flow_match_euler(config: &SchedulerConfig, steps: u32) -> DiffusionResult<Self> {
        let steps = steps as usize;
        let train_timesteps = config.num_train_timesteps.unwrap_or(1000).max(1);
        let shift = config.shift.unwrap_or(1.0).max(f32::MIN_POSITIVE);
        let mut sigmas = Vec::with_capacity(steps + 1);
        for idx in 0..steps {
            let frac = if steps == 1 {
                1.0
            } else {
                1.0 - idx as f32 / (steps - 1) as f32
            };
            let sigma = if (shift - 1.0).abs() <= f32::EPSILON {
                frac
            } else {
                (shift * frac) / (1.0 + (shift - 1.0) * frac)
            };
            sigmas.push(sigma.clamp(0.0, 1.0));
        }
        if config.invert_sigmas.unwrap_or(false) {
            for sigma in &mut sigmas {
                *sigma = 1.0 - *sigma;
            }
            sigmas.reverse();
        }
        if let Some(terminal) = config.shift_terminal {
            rescale_sigmas_to_terminal(&mut sigmas, terminal.clamp(0.0, 1.0));
        }
        let timesteps = sigmas
            .iter()
            .map(|sigma| sigma * train_timesteps as f32)
            .collect::<Vec<_>>();
        sigmas.push(0.0);
        Ok(Self {
            timesteps,
            sigmas,
            prediction_type: SchedulerPredictionType::Sample,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::FlowMatchEuler,
            train_timesteps: Vec::new(),
            alpha_t: Vec::new(),
            sigma_t: Vec::new(),
            lambda_t: Vec::new(),
        })
    }

    pub fn scale_model_input(&self, sample: &CpuTensor, step: usize) -> DiffusionResult<CpuTensor> {
        match self.input_scaling {
            SchedulerInputScaling::None => Ok(sample.clone()),
            SchedulerInputScaling::Sigma => {
                let sigma = *self.sigmas.get(step).ok_or_else(|| {
                    DiffusionError::InvalidRequest(format!("missing sigma for step {step}"))
                })?;
                let scale = (sigma * sigma + 1.0).sqrt().recip();
                Ok(CpuTensor {
                    shape: sample.shape.clone(),
                    data: sample.data.iter().map(|value| value * scale).collect(),
                })
            }
        }
    }

    pub fn initial_noise_sigma(&self) -> f32 {
        match self.input_scaling {
            SchedulerInputScaling::None => 1.0,
            SchedulerInputScaling::Sigma => {
                self.sigmas.iter().copied().fold(0.0, f32::max).max(1.0)
            }
        }
    }

    pub fn scale_initial_latents(&self, latents: &mut LatentBatch) {
        let sigma = self.initial_noise_sigma();
        if (sigma - 1.0).abs() <= f32::EPSILON {
            return;
        }
        for value in &mut latents.data {
            *value *= sigma;
        }
    }

    pub fn add_noise_to_latents(
        &self,
        latents: &mut LatentBatch,
        noise: &[f32],
        step: usize,
    ) -> DiffusionResult<()> {
        if noise.len() != latents.data.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "noise length {} != latent length {}",
                noise.len(),
                latents.data.len()
            )));
        }
        if let Some(timestep) = self.train_timesteps.get(step).copied() {
            let alpha = self.scheduler_alpha(timestep)?;
            let sigma = self.scheduler_sigma(timestep)?;
            for (latent, noise) in latents.data.iter_mut().zip(noise) {
                *latent = *latent * alpha + *noise * sigma;
            }
            return Ok(());
        }
        let sigma = *self.sigmas.get(step).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing sigma for step {step}"))
        })?;
        for (latent, noise) in latents.data.iter_mut().zip(noise) {
            *latent += *noise * sigma;
        }
        Ok(())
    }

    pub fn slice_from_step(&self, start_step: usize) -> DiffusionResult<Self> {
        if start_step > self.timesteps.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "scheduler start step {start_step} exceeds {} steps",
                self.timesteps.len()
            )));
        }
        Ok(Self {
            timesteps: self.timesteps[start_step..].to_vec(),
            sigmas: self.sigmas[start_step..].to_vec(),
            prediction_type: self.prediction_type,
            input_scaling: self.input_scaling,
            solver: self.solver,
            train_timesteps: if self.train_timesteps.is_empty() {
                Vec::new()
            } else {
                self.train_timesteps[start_step..].to_vec()
            },
            alpha_t: self.alpha_t.clone(),
            sigma_t: self.sigma_t.clone(),
            lambda_t: self.lambda_t.clone(),
        })
    }

    pub fn euler_step(
        &self,
        latents: &mut LatentBatch,
        noise_pred: &[f32],
        step: usize,
    ) -> DiffusionResult<()> {
        if noise_pred.len() != latents.data.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "noise prediction length {} != latent length {}",
                noise_pred.len(),
                latents.data.len()
            )));
        }
        let sigma = *self.sigmas.get(step).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing sigma for step {step}"))
        })?;
        let next_sigma = *self.sigmas.get(step + 1).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing next sigma for step {step}"))
        })?;
        let dt = next_sigma - sigma;
        for (latent, model_output) in latents.data.iter_mut().zip(noise_pred) {
            let derivative =
                scheduler_derivative(*latent, *model_output, sigma, self.prediction_type);
            *latent += derivative * dt;
        }
        Ok(())
    }

    pub fn step(
        &self,
        latents: &mut LatentBatch,
        noise_pred: &[f32],
        step: usize,
        state: &mut SchedulerStepState,
    ) -> DiffusionResult<()> {
        match self.solver {
            SchedulerSolver::Euler => self.euler_step(latents, noise_pred, step),
            SchedulerSolver::FlowMatchEuler => {
                self.flow_match_euler_step(latents, noise_pred, step)
            }
            SchedulerSolver::Ddim { .. } => self.ddim_step(latents, noise_pred, step),
            SchedulerSolver::DpmSolverMultistep { .. } => {
                self.dpm_solver_multistep_step(latents, noise_pred, step, state)
            }
        }
    }

    fn flow_match_euler_step(
        &self,
        latents: &mut LatentBatch,
        model_output: &[f32],
        step: usize,
    ) -> DiffusionResult<()> {
        if model_output.len() != latents.data.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "noise prediction length {} != latent length {}",
                model_output.len(),
                latents.data.len()
            )));
        }
        let sigma = *self.sigmas.get(step).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing sigma for step {step}"))
        })?;
        let next_sigma = *self.sigmas.get(step + 1).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing next sigma for step {step}"))
        })?;
        let dt = next_sigma - sigma;
        for (latent, output) in latents.data.iter_mut().zip(model_output) {
            *latent += output * dt;
        }
        Ok(())
    }

    fn ddim_step(
        &self,
        latents: &mut LatentBatch,
        model_output: &[f32],
        step: usize,
    ) -> DiffusionResult<()> {
        if model_output.len() != latents.data.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "noise prediction length {} != latent length {}",
                model_output.len(),
                latents.data.len()
            )));
        }
        let timestep = *self.train_timesteps.get(step).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing DDIM timestep for step {step}"))
        })?;
        let alpha = self.scheduler_alpha(timestep)?;
        let sigma = self.scheduler_sigma(timestep)?;
        let SchedulerSolver::Ddim { set_alpha_to_one } = self.solver else {
            return self.euler_step(latents, model_output, step);
        };
        let (prev_alpha, prev_sigma) =
            if let Some(prev_timestep) = self.train_timesteps.get(step + 1) {
                (
                    self.scheduler_alpha(*prev_timestep)?,
                    self.scheduler_sigma(*prev_timestep)?,
                )
            } else if set_alpha_to_one {
                (1.0, 0.0)
            } else {
                (self.scheduler_alpha(0)?, self.scheduler_sigma(0)?)
            };
        for (sample, output) in latents.data.iter_mut().zip(model_output) {
            let (pred_original, pred_epsilon) = match self.prediction_type {
                SchedulerPredictionType::Epsilon => ((*sample - sigma * output) / alpha, *output),
                SchedulerPredictionType::Sample => {
                    let epsilon = if sigma.abs() <= f32::MIN_POSITIVE {
                        0.0
                    } else {
                        (*sample - alpha * output) / sigma
                    };
                    (*output, epsilon)
                }
                SchedulerPredictionType::VPrediction => {
                    let pred_original = alpha * *sample - sigma * output;
                    let pred_epsilon = alpha * output + sigma * *sample;
                    (pred_original, pred_epsilon)
                }
            };
            *sample = prev_alpha * pred_original + prev_sigma * pred_epsilon;
        }
        Ok(())
    }

    fn dpm_solver_multistep_step(
        &self,
        latents: &mut LatentBatch,
        model_output: &[f32],
        step: usize,
        state: &mut SchedulerStepState,
    ) -> DiffusionResult<()> {
        let SchedulerSolver::DpmSolverMultistep {
            solver_order,
            lower_order_final: use_lower_order_final,
            ..
        } = self.solver
        else {
            return self.euler_step(latents, model_output, step);
        };
        if model_output.len() != latents.data.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "noise prediction length {} != latent length {}",
                model_output.len(),
                latents.data.len()
            )));
        }
        let timestep = *self.train_timesteps.get(step).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing DPM timestep for step {step}"))
        })?;
        let prev_timestep = if step + 1 == self.train_timesteps.len() {
            0
        } else {
            self.train_timesteps[step + 1]
        };
        let sample = latents.as_nchw_tensor();
        let converted = self.dpm_convert_model_output(model_output, timestep, &sample)?;
        state.model_outputs.push(converted);
        if state.model_outputs.len() > solver_order {
            state.model_outputs.remove(0);
        }

        let lower_order_final = step + 1 == self.train_timesteps.len()
            && use_lower_order_final
            && self.train_timesteps.len() < 15;
        let lower_order_second = step + 2 == self.train_timesteps.len()
            && use_lower_order_final
            && self.train_timesteps.len() < 15;

        let prev_sample =
            if solver_order == 1 || state.lower_order_nums < 1 || lower_order_final {
                self.dpm_first_order_update(
                    state.model_outputs.last().unwrap(),
                    timestep,
                    prev_timestep,
                    &sample,
                )?
            } else if solver_order == 2 || state.lower_order_nums < 2 || lower_order_second {
                let previous_timestep = *self
                    .train_timesteps
                    .get(step.wrapping_sub(1))
                    .ok_or_else(|| {
                        DiffusionError::InvalidRequest("missing previous DPM timestep".to_string())
                    })?;
                self.dpm_second_order_update(
                    previous_timestep,
                    timestep,
                    prev_timestep,
                    &sample,
                    state,
                )?
            } else {
                let previous_timestep = *self
                    .train_timesteps
                    .get(step.wrapping_sub(1))
                    .ok_or_else(|| {
                        DiffusionError::InvalidRequest("missing previous DPM timestep".to_string())
                    })?;
                let previous_previous_timestep = *self
                    .train_timesteps
                    .get(step.wrapping_sub(2))
                    .ok_or_else(|| {
                        DiffusionError::InvalidRequest(
                            "missing second previous DPM timestep".to_string(),
                        )
                    })?;
                self.dpm_third_order_update(
                    previous_previous_timestep,
                    previous_timestep,
                    timestep,
                    prev_timestep,
                    &sample,
                    state,
                )?
            };

        latents.data = prev_sample.data;
        if state.lower_order_nums < solver_order {
            state.lower_order_nums += 1;
        }
        Ok(())
    }

    fn dpm_convert_model_output(
        &self,
        model_output: &[f32],
        timestep: usize,
        sample: &CpuTensor,
    ) -> DiffusionResult<Vec<f32>> {
        let SchedulerSolver::DpmSolverMultistep {
            algorithm_type,
            thresholding,
            dynamic_thresholding_ratio,
            sample_max_value,
            ..
        } = self.solver
        else {
            return Ok(model_output.to_vec());
        };
        let alpha = self.scheduler_alpha(timestep)?;
        let sigma = self.scheduler_sigma(timestep)?;
        let mut output = match algorithm_type {
            DpmSolverAlgorithm::DpmSolverPlusPlus => match self.prediction_type {
                SchedulerPredictionType::Epsilon => sample
                    .data
                    .iter()
                    .zip(model_output)
                    .map(|(sample, noise)| (sample - sigma * noise) / alpha)
                    .collect(),
                SchedulerPredictionType::Sample => model_output.to_vec(),
                SchedulerPredictionType::VPrediction => sample
                    .data
                    .iter()
                    .zip(model_output)
                    .map(|(sample, value)| alpha * sample - sigma * value)
                    .collect(),
            },
        };
        if thresholding {
            dynamic_threshold_sample(
                &mut output,
                &sample.shape,
                dynamic_thresholding_ratio,
                sample_max_value,
            )?;
        }
        Ok(output)
    }

    fn dpm_first_order_update(
        &self,
        model_output: &[f32],
        timestep: usize,
        prev_timestep: usize,
        sample: &CpuTensor,
    ) -> DiffusionResult<CpuTensor> {
        let lambda_t = self.scheduler_lambda(prev_timestep)?;
        let lambda_s = self.scheduler_lambda(timestep)?;
        let alpha_t = self.scheduler_alpha(prev_timestep)?;
        let sigma_t = self.scheduler_sigma(prev_timestep)?;
        let sigma_s = self.scheduler_sigma(timestep)?;
        let h = lambda_t - lambda_s;
        let data = sample
            .data
            .iter()
            .zip(model_output)
            .map(|(sample, model_output)| {
                (sigma_t / sigma_s) * sample - (alpha_t * ((-h).exp() - 1.0)) * model_output
            })
            .collect();
        Ok(CpuTensor {
            shape: sample.shape.clone(),
            data,
        })
    }

    fn dpm_second_order_update(
        &self,
        previous_timestep: usize,
        timestep: usize,
        prev_timestep: usize,
        sample: &CpuTensor,
        state: &SchedulerStepState,
    ) -> DiffusionResult<CpuTensor> {
        let SchedulerSolver::DpmSolverMultistep { solver_type, .. } = self.solver else {
            unreachable!("DPM second-order update called for non-DPM scheduler");
        };
        let m0 = state.model_outputs.last().ok_or_else(|| {
            DiffusionError::InvalidRequest("missing current DPM model output".to_string())
        })?;
        let m1 = state
            .model_outputs
            .get(state.model_outputs.len().saturating_sub(2))
            .ok_or_else(|| {
                DiffusionError::InvalidRequest("missing previous DPM model output".to_string())
            })?;
        let lambda_t = self.scheduler_lambda(prev_timestep)?;
        let lambda_s0 = self.scheduler_lambda(timestep)?;
        let lambda_s1 = self.scheduler_lambda(previous_timestep)?;
        let alpha_t = self.scheduler_alpha(prev_timestep)?;
        let sigma_t = self.scheduler_sigma(prev_timestep)?;
        let sigma_s0 = self.scheduler_sigma(timestep)?;
        let h = lambda_t - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        if h.abs() <= f32::MIN_POSITIVE || h0.abs() <= f32::MIN_POSITIVE {
            return self.dpm_first_order_update(m0, timestep, prev_timestep, sample);
        }
        let r0 = h0 / h;
        let data = sample
            .data
            .iter()
            .zip(m0.iter().zip(m1))
            .map(|(sample, (m0, m1))| {
                let d1 = (m0 - m1) / r0;
                match solver_type {
                    DpmSolverType::Midpoint => {
                        (sigma_t / sigma_s0) * sample
                            - (alpha_t * ((-h).exp() - 1.0)) * m0
                            - 0.5 * (alpha_t * ((-h).exp() - 1.0)) * d1
                    }
                    DpmSolverType::Heun => {
                        (sigma_t / sigma_s0) * sample - (alpha_t * ((-h).exp() - 1.0)) * m0
                            + (alpha_t * (((-h).exp() - 1.0) / h + 1.0)) * d1
                    }
                }
            })
            .collect();
        Ok(CpuTensor {
            shape: sample.shape.clone(),
            data,
        })
    }

    fn dpm_third_order_update(
        &self,
        previous_previous_timestep: usize,
        previous_timestep: usize,
        timestep: usize,
        prev_timestep: usize,
        sample: &CpuTensor,
        state: &SchedulerStepState,
    ) -> DiffusionResult<CpuTensor> {
        let m0 = state.model_outputs.last().ok_or_else(|| {
            DiffusionError::InvalidRequest("missing current DPM model output".to_string())
        })?;
        let m1 = state
            .model_outputs
            .get(state.model_outputs.len().saturating_sub(2))
            .ok_or_else(|| {
                DiffusionError::InvalidRequest("missing previous DPM model output".to_string())
            })?;
        let m2 = state
            .model_outputs
            .get(state.model_outputs.len().saturating_sub(3))
            .ok_or_else(|| {
                DiffusionError::InvalidRequest(
                    "missing second previous DPM model output".to_string(),
                )
            })?;
        let lambda_t = self.scheduler_lambda(prev_timestep)?;
        let lambda_s0 = self.scheduler_lambda(timestep)?;
        let lambda_s1 = self.scheduler_lambda(previous_timestep)?;
        let lambda_s2 = self.scheduler_lambda(previous_previous_timestep)?;
        let alpha_t = self.scheduler_alpha(prev_timestep)?;
        let sigma_t = self.scheduler_sigma(prev_timestep)?;
        let sigma_s0 = self.scheduler_sigma(timestep)?;
        let h = lambda_t - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        let h1 = lambda_s1 - lambda_s2;
        if h.abs() <= f32::MIN_POSITIVE
            || h0.abs() <= f32::MIN_POSITIVE
            || h1.abs() <= f32::MIN_POSITIVE
            || (h0 + h1).abs() <= f32::MIN_POSITIVE
        {
            return self.dpm_second_order_update(
                previous_timestep,
                timestep,
                prev_timestep,
                sample,
                state,
            );
        }
        let r0 = h0 / h;
        let r1 = h1 / h;
        if r0.abs() <= f32::MIN_POSITIVE
            || r1.abs() <= f32::MIN_POSITIVE
            || (r0 + r1).abs() <= f32::MIN_POSITIVE
        {
            return self.dpm_second_order_update(
                previous_timestep,
                timestep,
                prev_timestep,
                sample,
                state,
            );
        }
        let exp_neg_h = (-h).exp();
        let data = sample
            .data
            .iter()
            .zip(m0.iter().zip(m1.iter().zip(m2)))
            .map(|(sample, (m0, (m1, m2)))| {
                let d0 = *m0;
                let d1_0 = (m0 - m1) / r0;
                let d1_1 = (m1 - m2) / r1;
                let d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1);
                let d2 = (d1_0 - d1_1) / (r0 + r1);
                (sigma_t / sigma_s0) * sample - (alpha_t * (exp_neg_h - 1.0)) * d0
                    + (alpha_t * ((exp_neg_h - 1.0) / h + 1.0)) * d1
                    - (alpha_t * ((exp_neg_h - 1.0 + h) / (h * h) - 0.5)) * d2
            })
            .collect();
        Ok(CpuTensor {
            shape: sample.shape.clone(),
            data,
        })
    }

    fn scheduler_alpha(&self, timestep: usize) -> DiffusionResult<f32> {
        self.alpha_t.get(timestep).copied().ok_or_else(|| {
            DiffusionError::InvalidRequest(format!(
                "missing scheduler alpha for timestep {timestep}"
            ))
        })
    }

    fn scheduler_sigma(&self, timestep: usize) -> DiffusionResult<f32> {
        self.sigma_t.get(timestep).copied().ok_or_else(|| {
            DiffusionError::InvalidRequest(format!(
                "missing scheduler sigma for timestep {timestep}"
            ))
        })
    }

    fn scheduler_lambda(&self, timestep: usize) -> DiffusionResult<f32> {
        self.lambda_t.get(timestep).copied().ok_or_else(|| {
            DiffusionError::InvalidRequest(format!(
                "missing scheduler lambda for timestep {timestep}"
            ))
        })
    }
}

fn scheduler_derivative(
    sample: f32,
    model_output: f32,
    sigma: f32,
    prediction_type: SchedulerPredictionType,
) -> f32 {
    if sigma.abs() <= f32::MIN_POSITIVE {
        return model_output;
    }
    match prediction_type {
        SchedulerPredictionType::Epsilon => model_output,
        SchedulerPredictionType::Sample => (sample - model_output) / sigma,
        SchedulerPredictionType::VPrediction => {
            let sigma_sq = sigma * sigma;
            let denom = sigma_sq + 1.0;
            let pred_original_sample = model_output * (-sigma / denom.sqrt()) + sample / denom;
            (sample - pred_original_sample) / sigma
        }
    }
}

fn scheduler_betas(
    beta_start: f32,
    beta_end: f32,
    num_train_timesteps: usize,
    schedule: &str,
) -> DiffusionResult<Vec<f32>> {
    if num_train_timesteps == 1 {
        return Ok(vec![beta_end.clamp(0.0, 0.999)]);
    }
    match schedule {
        "linear" => Ok((0..num_train_timesteps)
            .map(|idx| {
                let frac = idx as f32 / (num_train_timesteps - 1) as f32;
                beta_start + (beta_end - beta_start) * frac
            })
            .collect()),
        "scaled_linear" => {
            let start = beta_start.sqrt();
            let end = beta_end.sqrt();
            Ok((0..num_train_timesteps)
                .map(|idx| {
                    let frac = idx as f32 / (num_train_timesteps - 1) as f32;
                    let value = start + (end - start) * frac;
                    value * value
                })
                .collect())
        }
        "squaredcos_cap_v2" => Ok(betas_for_alpha_bar(num_train_timesteps)),
        other => Err(DiffusionError::InvalidMetadata(format!(
            "unsupported scheduler beta_schedule {other:?}"
        ))),
    }
}

fn betas_for_alpha_bar(num_train_timesteps: usize) -> Vec<f32> {
    fn alpha_bar(time: f32) -> f32 {
        let value = (time + 0.008) / 1.008 * std::f32::consts::FRAC_PI_2;
        value.cos().powi(2)
    }
    (0..num_train_timesteps)
        .map(|idx| {
            let t1 = idx as f32 / num_train_timesteps as f32;
            let t2 = (idx + 1) as f32 / num_train_timesteps as f32;
            (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(0.999)
        })
        .collect()
}

fn karras_sigmas(base_sigmas: &[f32]) -> Vec<f32> {
    if base_sigmas.is_empty() {
        return vec![0.0];
    }
    let rho = 7.0f32;
    let sigma_max = base_sigmas
        .first()
        .copied()
        .unwrap_or(0.0)
        .max(f32::MIN_POSITIVE);
    let sigma_min = base_sigmas
        .last()
        .copied()
        .unwrap_or(sigma_max)
        .max(f32::MIN_POSITIVE);
    let min_inv_rho = sigma_min.powf(1.0 / rho);
    let max_inv_rho = sigma_max.powf(1.0 / rho);
    let denom = base_sigmas.len().saturating_sub(1).max(1) as f32;
    let mut sigmas = (0..base_sigmas.len())
        .map(|idx| {
            let ramp = idx as f32 / denom;
            (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)).powf(rho)
        })
        .collect::<Vec<_>>();
    sigmas.push(0.0);
    sigmas
}

fn rescale_sigmas_to_terminal(sigmas: &mut [f32], terminal: f32) {
    let Some(first) = sigmas.first().copied() else {
        return;
    };
    let Some(last) = sigmas.last().copied() else {
        return;
    };
    let denom = first - last;
    if denom.abs() <= f32::EPSILON {
        return;
    }
    for sigma in sigmas {
        let normalized = (*sigma - last) / denom;
        *sigma = terminal + normalized * (first - terminal);
    }
}

fn nearest_training_timestep_for_sigma(training_sigmas: &[f32], sigma: f32) -> usize {
    training_sigmas
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            let left_delta = (*left - sigma).abs();
            let right_delta = (*right - sigma).abs();
            left_delta
                .partial_cmp(&right_delta)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn inference_train_timesteps(
    config: &SchedulerConfig,
    num_train_timesteps: usize,
    steps: usize,
) -> DiffusionResult<Vec<usize>> {
    if steps == 1 {
        return Ok(vec![num_train_timesteps - 1]);
    }
    if config.class_name == "DPMSolverMultistepScheduler" {
        return dpm_solver_train_timesteps(config, num_train_timesteps, steps);
    }
    Ok((0..steps)
        .map(|idx| {
            let frac = idx as f32 / (steps - 1) as f32;
            ((num_train_timesteps - 1) as f32 * (1.0 - frac)).round() as usize
        })
        .collect())
}

fn dpm_solver_train_timesteps(
    config: &SchedulerConfig,
    num_train_timesteps: usize,
    steps: usize,
) -> DiffusionResult<Vec<usize>> {
    let last_timestep = num_train_timesteps;
    let spacing = config.timestep_spacing.as_deref().unwrap_or("linspace");
    let offset = config.steps_offset.unwrap_or(0);
    let mut timesteps = match spacing {
        "linspace" => (0..=steps)
            .map(|idx| {
                let frac = idx as f32 / steps as f32;
                ((last_timestep - 1) as f32 * frac).round() as i32
            })
            .rev()
            .take(steps)
            .collect::<Vec<_>>(),
        "leading" => {
            let step_ratio = last_timestep / (steps + 1);
            (0..=steps)
                .map(|idx| (idx * step_ratio) as i32 + offset)
                .rev()
                .take(steps)
                .collect()
        }
        "trailing" => {
            let step_ratio = num_train_timesteps as f32 / steps as f32;
            (0..steps)
                .map(|idx| (last_timestep as f32 - idx as f32 * step_ratio).round() as i32 - 1)
                .collect()
        }
        other => {
            return Err(DiffusionError::InvalidMetadata(format!(
                "unsupported scheduler timestep_spacing {other:?}"
            )));
        }
    };
    timesteps.dedup();
    let mut out = Vec::with_capacity(timesteps.len());
    for timestep in timesteps {
        if timestep < 0 || timestep as usize >= num_train_timesteps {
            return Err(DiffusionError::InvalidMetadata(format!(
                "DPM-Solver timestep {timestep} is outside 0..{num_train_timesteps}"
            )));
        }
        out.push(timestep as usize);
    }
    Ok(out)
}

pub fn denoise_latents_with_cfg(
    latents: LatentBatch,
    schedule: &DiffusionSchedule,
    cfg_scale: f32,
    positive_embeddings: &CpuTensor,
    negative_embeddings: &CpuTensor,
    mut predict_noise: impl FnMut(&CpuTensor, &[f32], &CpuTensor) -> DiffusionResult<CpuTensor>,
) -> DiffusionResult<LatentBatch> {
    denoise_latents_with_cfg_progress(
        latents,
        schedule,
        cfg_scale,
        positive_embeddings,
        negative_embeddings,
        |sample, timesteps, encoder_states, _sdxl| predict_noise(sample, timesteps, encoder_states),
        None,
        None,
        None,
        None,
        None,
    )
}

pub fn denoise_latents_with_cfg_progress(
    latents: LatentBatch,
    schedule: &DiffusionSchedule,
    cfg_scale: f32,
    positive_embeddings: &CpuTensor,
    negative_embeddings: &CpuTensor,
    mut predict_noise: impl FnMut(
        &CpuTensor,
        &[f32],
        &CpuTensor,
        Option<&SdxlDenoiseConditioning<'_>>,
    ) -> DiffusionResult<CpuTensor>,
    positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
    negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
    inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
    masked_reference: Option<&MaskedDenoiseReference<'_>>,
    progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
) -> DiffusionResult<LatentBatch> {
    denoise_latents_with_cfg_progress_and_runtime_options(
        latents,
        schedule,
        cfg_scale,
        positive_embeddings,
        negative_embeddings,
        |sample, timesteps, encoder_states, sdxl_conditioning, _runtime_context| {
            predict_noise(sample, timesteps, encoder_states, sdxl_conditioning)
        },
        positive_sdxl_conditioning,
        negative_sdxl_conditioning,
        inpaint_conditioning,
        masked_reference,
        DiffusionGenerationRuntimeOptions::default(),
        progress,
    )
    .map(|output| output.latents)
}

fn denoise_latents_with_cfg_progress_and_runtime_options(
    latents: LatentBatch,
    schedule: &DiffusionSchedule,
    cfg_scale: f32,
    positive_embeddings: &CpuTensor,
    negative_embeddings: &CpuTensor,
    predict_noise: impl FnMut(
        &CpuTensor,
        &[f32],
        &CpuTensor,
        Option<&SdxlDenoiseConditioning<'_>>,
        &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor>,
    positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
    negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
    inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
    masked_reference: Option<&MaskedDenoiseReference<'_>>,
    runtime_options: DiffusionGenerationRuntimeOptions,
    progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
) -> DiffusionResult<DenoiseLatentsOutput> {
    let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
    denoise_latents_with_cfg_progress_and_runtime_context(
        latents,
        schedule,
        cfg_scale,
        positive_embeddings,
        negative_embeddings,
        predict_noise,
        positive_sdxl_conditioning,
        negative_sdxl_conditioning,
        inpaint_conditioning,
        masked_reference,
        &mut runtime_context,
        progress,
    )
}

fn denoise_latents_with_cfg_progress_and_runtime_context(
    mut latents: LatentBatch,
    schedule: &DiffusionSchedule,
    cfg_scale: f32,
    positive_embeddings: &CpuTensor,
    negative_embeddings: &CpuTensor,
    mut predict_noise: impl FnMut(
        &CpuTensor,
        &[f32],
        &CpuTensor,
        Option<&SdxlDenoiseConditioning<'_>>,
        &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor>,
    positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
    negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
    inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
    masked_reference: Option<&MaskedDenoiseReference<'_>>,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
    mut progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
) -> DiffusionResult<DenoiseLatentsOutput> {
    validate_conditioning_for_latents(&latents, positive_embeddings)?;
    validate_conditioning_for_latents(&latents, negative_embeddings)?;
    if let Some(inpaint_conditioning) = inpaint_conditioning {
        validate_inpaint_denoise_conditioning(&latents, inpaint_conditioning)?;
    }
    if let Some(masked_reference) = masked_reference {
        validate_masked_denoise_reference(&latents, masked_reference)?;
    }
    let mut scheduler_state = SchedulerStepState::default();
    let mut runtime_kind = DiffusionRuntimeKind::CpuSourceReference;
    for step in 0..schedule.timesteps.len() {
        let (sample, scale_runtime_kind) = scale_model_input_with_runtime_context(
            schedule,
            &latents.as_nchw_tensor(),
            step,
            runtime_context,
        )?;
        runtime_kind = merge_runtime_kind(runtime_kind, scale_runtime_kind);
        let model_sample = if let Some(inpaint_conditioning) = inpaint_conditioning {
            append_inpaint_conditioning(&sample, inpaint_conditioning)?
        } else {
            sample
        };
        let timestep = schedule.timesteps[step];
        let timesteps = vec![timestep; latents.batch];
        let negative_pred = predict_noise(
            &model_sample,
            &timesteps,
            negative_embeddings,
            negative_sdxl_conditioning,
            runtime_context,
        )?;
        let positive_pred = predict_noise(
            &model_sample,
            &timesteps,
            positive_embeddings,
            positive_sdxl_conditioning,
            runtime_context,
        )?;
        validate_noise_prediction(&latents, &negative_pred)?;
        validate_noise_prediction(&latents, &positive_pred)?;
        let (guided, guidance_runtime_kind) = cfg_guidance_with_runtime_context(
            &negative_pred,
            &positive_pred,
            cfg_scale,
            runtime_context,
        )?;
        runtime_kind = merge_runtime_kind(runtime_kind, guidance_runtime_kind);
        let step_runtime_kind = scheduler_step_with_runtime_context(
            schedule,
            &mut latents,
            &guided.data,
            step,
            &mut scheduler_state,
            runtime_context,
        )?;
        runtime_kind = merge_runtime_kind(runtime_kind, step_runtime_kind);
        if let Some(masked_reference) = masked_reference {
            let masked_reference_runtime_kind =
                apply_masked_denoise_reference_with_runtime_context(
                    &mut latents,
                    masked_reference,
                    step,
                    runtime_context,
                )?;
            runtime_kind = merge_runtime_kind(runtime_kind, masked_reference_runtime_kind);
        }
        if let Some(progress) = progress.as_deref_mut() {
            progress(DiffusionProgress {
                completed_steps: step + 1,
                total_steps: schedule.timesteps.len(),
                timestep: timestep.round().max(0.0) as usize,
                preview_latents: Some(latents.clone()),
            })?;
        }
    }
    Ok(DenoiseLatentsOutput {
        latents,
        runtime_kind,
    })
}

fn validate_conditioning_for_latents(
    latents: &LatentBatch,
    embeddings: &CpuTensor,
) -> DiffusionResult<()> {
    let [batch, seq, width] = shape3(embeddings)?;
    if batch != latents.batch {
        return Err(DiffusionError::InvalidRequest(format!(
            "conditioning batch {batch} != latent batch {}",
            latents.batch
        )));
    }
    if seq == 0 || width == 0 {
        return Err(DiffusionError::InvalidRequest(
            "conditioning embeddings must have non-empty sequence and width".to_string(),
        ));
    }
    Ok(())
}

fn validate_inpaint_denoise_conditioning(
    latents: &LatentBatch,
    conditioning: &InpaintDenoiseConditioning,
) -> DiffusionResult<()> {
    if latents.batch != conditioning.masked_image_latents.batch
        || latents.channels != conditioning.masked_image_latents.channels
        || latents.height != conditioning.masked_image_latents.height
        || latents.width != conditioning.masked_image_latents.width
    {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpaint masked-image latent shape [{}x{}x{}x{}] != latent shape [{}x{}x{}x{}]",
            conditioning.masked_image_latents.batch,
            conditioning.masked_image_latents.channels,
            conditioning.masked_image_latents.height,
            conditioning.masked_image_latents.width,
            latents.batch,
            latents.channels,
            latents.height,
            latents.width
        )));
    }
    let expected_mask = latents.batch * latents.height * latents.width;
    if conditioning.mask_weights.len() != expected_mask {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpaint mask has {} weights, expected {expected_mask}",
            conditioning.mask_weights.len()
        )));
    }
    Ok(())
}

fn append_inpaint_conditioning(
    sample: &CpuTensor,
    conditioning: &InpaintDenoiseConditioning,
) -> DiffusionResult<CpuTensor> {
    let [batch, channels, height, width] = shape4(sample)?;
    if batch != conditioning.masked_image_latents.batch
        || channels != conditioning.masked_image_latents.channels
        || height != conditioning.masked_image_latents.height
        || width != conditioning.masked_image_latents.width
    {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpaint sample shape {:?} != masked-image latent shape [{}x{}x{}x{}]",
            sample.shape,
            conditioning.masked_image_latents.batch,
            conditioning.masked_image_latents.channels,
            conditioning.masked_image_latents.height,
            conditioning.masked_image_latents.width
        )));
    }
    let expected_mask = batch * height * width;
    if conditioning.mask_weights.len() != expected_mask {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpaint mask has {} weights, expected {expected_mask}",
            conditioning.mask_weights.len()
        )));
    }
    let out_channels = channels + 1 + conditioning.masked_image_latents.channels;
    let mut out = CpuTensor::zeros(&[batch, out_channels, height, width]);
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    out.data[nchw_idx(b, c, y, x, out_channels, height, width)] =
                        sample.data[nchw_idx(b, c, y, x, channels, height, width)];
                }
            }
        }
        for y in 0..height {
            for x in 0..width {
                let mask_idx = (b * height + y) * width + x;
                out.data[nchw_idx(b, channels, y, x, out_channels, height, width)] =
                    conditioning.mask_weights[mask_idx];
            }
        }
        for c in 0..conditioning.masked_image_latents.channels {
            for y in 0..height {
                for x in 0..width {
                    out.data[nchw_idx(b, channels + 1 + c, y, x, out_channels, height, width)] =
                        conditioning.masked_image_latents.data
                            [nchw_idx(b, c, y, x, channels, height, width)];
                }
            }
        }
    }
    Ok(out)
}

fn validate_masked_denoise_reference(
    latents: &LatentBatch,
    reference: &MaskedDenoiseReference<'_>,
) -> DiffusionResult<()> {
    if latents.batch != reference.init_latents.batch
        || latents.channels != reference.init_latents.channels
        || latents.height != reference.init_latents.height
        || latents.width != reference.init_latents.width
    {
        return Err(DiffusionError::InvalidRequest(format!(
            "masked denoise latent shape [{}x{}x{}x{}] != init latent shape [{}x{}x{}x{}]",
            latents.batch,
            latents.channels,
            latents.height,
            latents.width,
            reference.init_latents.batch,
            reference.init_latents.channels,
            reference.init_latents.height,
            reference.init_latents.width
        )));
    }
    if reference.noise.len() != latents.data.len() {
        return Err(DiffusionError::InvalidRequest(format!(
            "masked denoise noise length {} != latent length {}",
            reference.noise.len(),
            latents.data.len()
        )));
    }
    let expected_mask = latents.batch * latents.height * latents.width;
    if reference.mask_weights.len() != expected_mask {
        return Err(DiffusionError::InvalidRequest(format!(
            "masked denoise mask has {} weights, expected {expected_mask}",
            reference.mask_weights.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
fn apply_masked_denoise_reference(
    latents: &mut LatentBatch,
    reference: &MaskedDenoiseReference<'_>,
    sliced_step: usize,
) -> DiffusionResult<()> {
    let mut reference_latents = reference.init_latents.clone();
    let source_step = reference.start_step + sliced_step + 1;
    if source_step < reference.source_schedule.timesteps.len() {
        reference.source_schedule.add_noise_to_latents(
            &mut reference_latents,
            reference.noise,
            source_step,
        )?;
    }
    blend_latents_with_mask(latents, &reference_latents, reference.mask_weights)
}

fn apply_masked_denoise_reference_with_runtime_context(
    latents: &mut LatentBatch,
    reference: &MaskedDenoiseReference<'_>,
    sliced_step: usize,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<DiffusionRuntimeKind> {
    let mut reference_latents = reference.init_latents.clone();
    let source_step = reference.start_step + sliced_step + 1;
    if source_step < reference.source_schedule.timesteps.len() {
        reference.source_schedule.add_noise_to_latents(
            &mut reference_latents,
            reference.noise,
            source_step,
        )?;
    }
    blend_latents_with_mask_with_runtime_context(
        latents,
        &reference_latents,
        reference.mask_weights,
        runtime_context,
    )
}

fn validate_noise_prediction(latents: &LatentBatch, noise: &CpuTensor) -> DiffusionResult<()> {
    let expected = [
        latents.batch,
        latents.channels,
        latents.height,
        latents.width,
    ];
    let actual = shape4(noise)?;
    if actual != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "noise prediction shape {:?} != latent shape {:?}",
            noise.shape, expected
        )));
    }
    Ok(())
}

fn cfg_guidance(
    negative_pred: &CpuTensor,
    positive_pred: &CpuTensor,
    cfg_scale: f32,
) -> DiffusionResult<CpuTensor> {
    if negative_pred.shape != positive_pred.shape {
        return Err(DiffusionError::InvalidRequest(format!(
            "CFG prediction shape mismatch {:?} vs {:?}",
            negative_pred.shape, positive_pred.shape
        )));
    }
    Ok(CpuTensor {
        shape: negative_pred.shape.clone(),
        data: negative_pred
            .data
            .iter()
            .zip(&positive_pred.data)
            .map(|(negative, positive)| negative + cfg_scale * (positive - negative))
            .collect(),
    })
}

#[cfg(not(feature = "rocm"))]
fn rocm_hybrid_unavailable_error() -> DiffusionError {
    DiffusionError::BackendUnavailable(
        "ROCm hybrid diffusion generation requested, but hipfire-diffusion was built without the rocm feature"
            .to_string(),
    )
}

fn scale_model_input_with_runtime_context(
    schedule: &DiffusionSchedule,
    sample: &CpuTensor,
    step: usize,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<(CpuTensor, DiffusionRuntimeKind)> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return Ok((
            schedule.scale_model_input(sample, step)?,
            DiffusionRuntimeKind::CpuSourceReference,
        ));
    };
    match schedule.input_scaling {
        SchedulerInputScaling::None => {
            Ok((sample.clone(), DiffusionRuntimeKind::CpuSourceReference))
        }
        SchedulerInputScaling::Sigma => {
            #[cfg(feature = "rocm")]
            {
                let sigma = *schedule.sigmas.get(step).ok_or_else(|| {
                    DiffusionError::InvalidRequest(format!("missing sigma for step {step}"))
                })?;
                let scale = (sigma * sigma + 1.0).sqrt().recip();
                let data = runtime_context
                    .with_rocm_gpu(|gpu| scale_model_input_hip_on_gpu(gpu, &sample.data, scale))?;
                Ok((
                    CpuTensor {
                        shape: sample.shape.clone(),
                        data,
                    },
                    DiffusionRuntimeKind::RocmHybridReference,
                ))
            }
            #[cfg(not(feature = "rocm"))]
            {
                let _ = _device_id;
                Err(rocm_hybrid_unavailable_error())
            }
        }
    }
}

fn cfg_guidance_with_runtime_context(
    negative_pred: &CpuTensor,
    positive_pred: &CpuTensor,
    cfg_scale: f32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<(CpuTensor, DiffusionRuntimeKind)> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return Ok((
            cfg_guidance(negative_pred, positive_pred, cfg_scale)?,
            DiffusionRuntimeKind::CpuSourceReference,
        ));
    };
    if negative_pred.shape != positive_pred.shape {
        return Err(DiffusionError::InvalidRequest(format!(
            "CFG prediction shape mismatch {:?} vs {:?}",
            negative_pred.shape, positive_pred.shape
        )));
    }
    #[cfg(feature = "rocm")]
    {
        let data = runtime_context.with_rocm_gpu(|gpu| {
            cfg_guidance_hip_on_gpu(gpu, &negative_pred.data, &positive_pred.data, cfg_scale)
        })?;
        Ok((
            CpuTensor {
                shape: negative_pred.shape.clone(),
                data,
            },
            DiffusionRuntimeKind::RocmHybridReference,
        ))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn scheduler_step_with_runtime_context(
    schedule: &DiffusionSchedule,
    latents: &mut LatentBatch,
    noise_pred: &[f32],
    step: usize,
    state: &mut SchedulerStepState,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<DiffusionRuntimeKind> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        schedule.step(latents, noise_pred, step, state)?;
        return Ok(DiffusionRuntimeKind::CpuSourceReference);
    };
    if schedule.solver != SchedulerSolver::Euler {
        schedule.step(latents, noise_pred, step, state)?;
        return Ok(DiffusionRuntimeKind::CpuSourceReference);
    }
    if noise_pred.len() != latents.data.len() {
        return Err(DiffusionError::InvalidRequest(format!(
            "noise prediction length {} != latent length {}",
            noise_pred.len(),
            latents.data.len()
        )));
    }
    #[cfg(feature = "rocm")]
    {
        let sigma = *schedule.sigmas.get(step).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing sigma for step {step}"))
        })?;
        let next_sigma = *schedule.sigmas.get(step + 1).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing next sigma for step {step}"))
        })?;
        latents.data = runtime_context.with_rocm_gpu(|gpu| {
            euler_step_hip_on_gpu(
                gpu,
                &latents.data,
                noise_pred,
                sigma,
                next_sigma,
                schedule.prediction_type,
            )
        })?;
        Ok(DiffusionRuntimeKind::RocmHybridReference)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn maybe_center_unet_input_with_runtime_context(
    sample: &CpuTensor,
    center_input_sample: bool,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return Ok(maybe_center_unet_input(sample, center_input_sample));
    };
    if !center_input_sample {
        return Ok(sample.clone());
    }
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| {
            maybe_center_unet_input_hip_on_gpu(gpu, sample, center_input_sample)
        })
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn timestep_embedding_with_runtime_context(
    timesteps: &[f32],
    dim: usize,
    flip_sin_to_cos: bool,
    freq_shift: f32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return timestep_embedding(timesteps, dim, flip_sin_to_cos, freq_shift);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| {
            timestep_embedding_hip_on_gpu(gpu, timesteps, dim, flip_sin_to_cos, freq_shift)
        })
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn scale_tensor_with_runtime_context(
    input: &CpuTensor,
    scale: f32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return Ok(CpuTensor {
            shape: input.shape.clone(),
            data: input.data.iter().map(|value| value * scale).collect(),
        });
    };
    #[cfg(feature = "rocm")]
    {
        let data = runtime_context
            .with_rocm_gpu(|gpu| scale_model_input_hip_on_gpu(gpu, &input.data, scale))?;
        Ok(CpuTensor {
            shape: input.shape.clone(),
            data,
        })
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn linear_optional_bias_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return linear_optional_bias(input, weight, bias);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context
            .with_rocm_gpu(|gpu| linear_optional_bias_hip_on_gpu(gpu, input, weight, bias))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn linear_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    if runtime_context.rocm_device_id().is_none() {
        return linear(input, weight, bias);
    }
    linear_optional_bias_with_runtime_context(input, weight, Some(bias), runtime_context)
}

fn silu_with_runtime_context(
    input: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return Ok(tensor_map(input, silu));
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| silu_hip_on_gpu(gpu, input))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn quick_gelu_with_runtime_context(
    input: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return Ok(tensor_map(input, quick_gelu));
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| quick_gelu_hip_on_gpu(gpu, input))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn clip_token_position_embeddings(
    token_embedding: &CpuTensor,
    position_embedding: &CpuTensor,
    tokens: &[u32],
) -> DiffusionResult<CpuTensor> {
    let (vocab, hidden) = token_embedding.rows_cols()?;
    let (max_positions, position_hidden) = position_embedding.rows_cols()?;
    if position_hidden != hidden {
        return Err(DiffusionError::InvalidMetadata(format!(
            "CLIP position embedding hidden size {position_hidden} != token hidden size {hidden}"
        )));
    }
    if tokens.len() > max_positions {
        return Err(DiffusionError::InvalidRequest(format!(
            "CLIP token length {} exceeds position embedding length {max_positions}",
            tokens.len()
        )));
    }
    let seq = tokens.len();
    let mut x = CpuTensor::zeros(&[seq, hidden]);
    for (pos, &token) in tokens.iter().enumerate() {
        let token = token as usize;
        if token >= vocab {
            return Err(DiffusionError::InvalidRequest(format!(
                "CLIP token id {token} exceeds vocab {vocab}"
            )));
        }
        let dst = pos * hidden;
        let token_src = token * hidden;
        let pos_src = pos * hidden;
        for col in 0..hidden {
            x.data[dst + col] =
                token_embedding.data[token_src + col] + position_embedding.data[pos_src + col];
        }
    }
    Ok(x)
}

fn clip_token_position_embeddings_with_runtime_context(
    token_embedding: &CpuTensor,
    position_embedding: &CpuTensor,
    tokens: &[u32],
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return clip_token_position_embeddings(token_embedding, position_embedding, tokens);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| {
            clip_token_position_embeddings_hip_on_gpu(
                gpu,
                token_embedding,
                position_embedding,
                tokens,
            )
        })
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn tensor_add_with_runtime_context(
    a: &CpuTensor,
    b: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return tensor_add(a, b);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| tensor_add_hip_on_gpu(gpu, a, b))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn concat_last_dim_2d_with_runtime_context(
    a: &CpuTensor,
    b: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return concat_last_dim_2d(a, b);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| concat_last_dim_2d_hip_on_gpu(gpu, a, b))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn concat_last_dim_3d_with_runtime_context(
    a: &CpuTensor,
    b: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return concat_last_dim_3d(a, b);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| concat_last_dim_3d_hip_on_gpu(gpu, a, b))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn conv2d_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    padding: usize,
    stride: usize,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return conv2d_nchw_with_stride(input, weight, bias, padding, stride);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context
            .with_rocm_gpu(|gpu| conv2d_nchw_hip_on_gpu(gpu, input, weight, bias, padding, stride))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn group_norm_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    groups: usize,
    eps: f32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return group_norm_nchw(input, weight, bias, groups, eps);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context
            .with_rocm_gpu(|gpu| group_norm_nchw_hip_on_gpu(gpu, input, weight, bias, groups, eps))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn add_channel_bias_nchw_with_runtime_context(
    input: &mut CpuTensor,
    bias: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<()> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return add_channel_bias_nchw(input, bias);
    };
    #[cfg(feature = "rocm")]
    {
        *input = runtime_context
            .with_rocm_gpu(|gpu| add_channel_bias_nchw_hip_on_gpu(gpu, input, bias))?;
        Ok(())
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn concat_channels_nchw_with_runtime_context(
    a: &CpuTensor,
    b: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return concat_channels_nchw(a, b);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| concat_channels_nchw_hip_on_gpu(gpu, a, b))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn upsample_nearest2d_nchw_with_runtime_context(
    input: &CpuTensor,
    scale: usize,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return upsample_nearest2d_nchw(input, scale);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| upsample_nearest2d_nchw_hip_on_gpu(gpu, input, scale))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn nchw_to_bsc_with_runtime_context(
    input: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return nchw_to_bsc(input);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| nchw_to_bsc_hip_on_gpu(gpu, input))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn bsc_to_nchw_with_runtime_context(
    input: &CpuTensor,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return bsc_to_nchw(input, batch, channels, height, width);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context
            .with_rocm_gpu(|gpu| bsc_to_nchw_hip_on_gpu(gpu, input, batch, channels, height, width))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn linear_3d_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    if runtime_context.rocm_device_id().is_none() {
        return linear_3d(input, weight, bias);
    }
    let [batch, seq, in_features] = shape3(input)?;
    let flat = CpuTensor {
        shape: vec![batch * seq, in_features],
        data: input.data.clone(),
    };
    let out = linear_optional_bias_with_runtime_context(&flat, weight, bias, runtime_context)?;
    let [rows, out_features] = shape2(&out)?;
    if rows != batch * seq {
        return Err(DiffusionError::InvalidMetadata(format!(
            "linear_3d row count {rows} != batch*seq {}",
            batch * seq
        )));
    }
    Ok(CpuTensor {
        shape: vec![batch, seq, out_features],
        data: out.data,
    })
}

fn layer_norm_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    eps: f32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return layer_norm(input, weight, bias, eps);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| layer_norm_hip_on_gpu(gpu, input, weight, bias, eps))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn layer_norm_3d_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    eps: f32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    if runtime_context.rocm_device_id().is_none() {
        return layer_norm_3d(input, weight, bias, eps);
    }
    let [batch, seq, width] = shape3(input)?;
    let flat = CpuTensor {
        shape: vec![batch * seq, width],
        data: input.data.clone(),
    };
    let out = layer_norm_with_runtime_context(&flat, weight, bias, eps, runtime_context)?;
    Ok(CpuTensor {
        shape: vec![batch, seq, width],
        data: out.data,
    })
}

fn scaled_dot_product_attention_with_runtime_context(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    heads: usize,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return scaled_dot_product_attention(q, k, v, heads);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context
            .with_rocm_gpu(|gpu| scaled_dot_product_attention_hip_on_gpu(gpu, q, k, v, heads))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn clip_causal_self_attention_with_runtime_context(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    n_heads: usize,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return clip_causal_self_attention(q, k, v, n_heads);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context
            .with_rocm_gpu(|gpu| clip_causal_self_attention_hip_on_gpu(gpu, q, k, v, n_heads))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn geglu_gate_3d_with_runtime_context(
    projected: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return geglu_gate_3d(projected);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| geglu_gate_3d_hip_on_gpu(gpu, projected))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

#[cfg(feature = "rocm")]
fn f32_slices_close(actual: &[f32], expected: &[f32], tolerance: f32) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= tolerance)
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        ((value as f32) + 0.5) / ((1u64 << 24) as f32)
    }
}

fn box_muller_pair(rng: &mut SplitMix64) -> (f32, f32) {
    let u1 = rng.next_unit().max(f32::MIN_POSITIVE);
    let u2 = rng.next_unit();
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (radius * theta.cos(), radius * theta.sin())
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl CpuTensor {
    pub fn from_hfq(hfq: &HfqFile, name: &str) -> DiffusionResult<Self> {
        let (info, bytes) = hfq.tensor_data_vec(name).ok_or_else(|| {
            DiffusionError::InvalidMetadata(format!("tensor {name:?} is missing"))
        })?;
        let elem_count = info
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim as usize))
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata(format!("tensor {name:?} shape overflows"))
            })?;
        let data = match info.quant_type {
            QT_DIFFUSION_TENSOR_Q4F16_G64 => decode_q4f16_g64_slice(name, &bytes, elem_count)?,
            QT_DIFFUSION_TENSOR_F16 => decode_f16_slice(&bytes),
            QT_DIFFUSION_TENSOR_BF16 => decode_bf16_slice(&bytes),
            QT_DIFFUSION_TENSOR_F32 => decode_f32_slice(&bytes),
            QT_DIFFUSION_TENSOR_Q8F16 => decode_q8f16_slice(name, &bytes, elem_count)?,
            QT_DIFFUSION_TENSOR_Q4_K => decode_q4_k_slice(name, &bytes, elem_count)?,
            QT_DIFFUSION_TENSOR_HFQ4_G256 => {
                decode_hfq4_slice(name, &bytes, elem_count, 256, 136, "HFQ4G256")?
            }
            QT_DIFFUSION_TENSOR_HFQ4_G128 => {
                decode_hfq4_slice(name, &bytes, elem_count, 128, 72, "HFQ4G128")?
            }
            QT_DIFFUSION_TENSOR_HFQ6_G256 => decode_hfq6_g256_slice(name, &bytes, elem_count)?,
            other => {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "tensor {name:?} has unsupported quant_type {other}; native diffusion tensor decoding currently supports Q4F16_G64, f16, bf16, f32, Q8F16, Q4_K, HFQ4G256, HFQ4G128, and HFQ6G256 tensor payloads. Other packed or quantized payloads require a diffusion dequantizer/runtime implementation"
                )))
            }
        };
        if data.len() != elem_count {
            return Err(DiffusionError::InvalidMetadata(format!(
                "tensor {name:?} decoded {} elements but shape expects {elem_count}",
                data.len()
            )));
        }
        Ok(Self {
            shape: info.shape.iter().map(|&dim| dim as usize).collect(),
            data,
        })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let len = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            data: vec![0.0; len],
        }
    }

    pub fn rows_cols(&self) -> DiffusionResult<(usize, usize)> {
        match self.shape.as_slice() {
            [rows, cols] => Ok((*rows, *cols)),
            _ => Err(DiffusionError::InvalidMetadata(format!(
                "expected 2-D tensor, got shape {:?}",
                self.shape
            ))),
        }
    }
}

fn decode_f16_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

fn decode_bf16_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16))
        .collect()
}

fn decode_q4f16_g64_slice(
    name: &str,
    bytes: &[u8],
    elem_count: usize,
) -> DiffusionResult<Vec<f32>> {
    let expected_blocks = elem_count.div_ceil(64);
    let expected_bytes = expected_blocks * 36;
    if bytes.len() < expected_bytes {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Q4F16_G64 tensor {name:?} has {} bytes but shape requires at least {expected_bytes}",
            bytes.len()
        )));
    }
    let mut out = vec![0.0f32; elem_count];
    for block in 0..expected_blocks {
        let offset = block * 36;
        let scale = f16_bits_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
        let min = f16_bits_to_f32(u16::from_le_bytes([bytes[offset + 2], bytes[offset + 3]]));
        for idx in 0..32 {
            let packed = bytes[offset + 4 + idx];
            let lo = (packed & 0x0f) as f32;
            let hi = (packed >> 4) as f32;
            let lo_idx = block * 64 + idx;
            let hi_idx = lo_idx + 32;
            if lo_idx < elem_count {
                out[lo_idx] = min + lo * scale;
            }
            if hi_idx < elem_count {
                out[hi_idx] = min + hi * scale;
            }
        }
    }
    Ok(out)
}

fn decode_q8f16_slice(name: &str, bytes: &[u8], elem_count: usize) -> DiffusionResult<Vec<f32>> {
    let expected_blocks = elem_count.div_ceil(32);
    let expected_bytes = expected_blocks * 34;
    if bytes.len() < expected_bytes {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Q8F16 tensor {name:?} has {} bytes but shape requires at least {expected_bytes}",
            bytes.len()
        )));
    }
    Ok(hipfire_runtime::quant::dequantize_q8_0(bytes, elem_count))
}

fn decode_q4_k_slice(name: &str, bytes: &[u8], elem_count: usize) -> DiffusionResult<Vec<f32>> {
    let expected_blocks = elem_count.div_ceil(256);
    let expected_bytes = expected_blocks * 144;
    if bytes.len() < expected_bytes {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Q4_K tensor {name:?} has {} bytes but shape requires at least {expected_bytes}",
            bytes.len()
        )));
    }
    Ok(hipfire_runtime::quant::dequantize_q4_k(bytes, elem_count))
}

fn decode_hfq4_slice(
    name: &str,
    bytes: &[u8],
    elem_count: usize,
    group_size: usize,
    block_bytes: usize,
    label: &str,
) -> DiffusionResult<Vec<f32>> {
    let expected_blocks = elem_count.div_ceil(group_size);
    let expected_bytes = expected_blocks * block_bytes;
    if bytes.len() < expected_bytes {
        return Err(DiffusionError::InvalidMetadata(format!(
            "{label} tensor {name:?} has {} bytes but shape requires at least {expected_bytes}",
            bytes.len()
        )));
    }
    let mut out = vec![0.0f32; elem_count];
    for block in 0..expected_blocks {
        let offset = block * block_bytes;
        let scale = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        let min = f32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        for idx in 0..(group_size / 2) {
            let packed = bytes[offset + 8 + idx];
            let lo_idx = block * group_size + idx * 2;
            let hi_idx = lo_idx + 1;
            if lo_idx < elem_count {
                out[lo_idx] = min + (packed & 0x0f) as f32 * scale;
            }
            if hi_idx < elem_count {
                out[hi_idx] = min + (packed >> 4) as f32 * scale;
            }
        }
    }
    Ok(out)
}

fn decode_hfq6_g256_slice(
    name: &str,
    bytes: &[u8],
    elem_count: usize,
) -> DiffusionResult<Vec<f32>> {
    let expected_blocks = elem_count.div_ceil(256);
    let expected_bytes = expected_blocks * 200;
    if bytes.len() < expected_bytes {
        return Err(DiffusionError::InvalidMetadata(format!(
            "HFQ6G256 tensor {name:?} has {} bytes but shape requires at least {expected_bytes}",
            bytes.len()
        )));
    }
    let mut out = vec![0.0f32; elem_count];
    for block in 0..expected_blocks {
        let offset = block * 200;
        let scale = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        let min = f32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        for i in (0..256).step_by(4) {
            let byte_offset = offset + 8 + (i / 4) * 3;
            let b0 = bytes[byte_offset];
            let b1 = bytes[byte_offset + 1];
            let b2 = bytes[byte_offset + 2];
            let values = [
                b0 & 0x3f,
                ((b0 >> 6) | ((b1 & 0x0f) << 2)) & 0x3f,
                ((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3f,
                (b2 >> 2) & 0x3f,
            ];
            for (lane, value) in values.into_iter().enumerate() {
                let idx = block * 256 + i + lane;
                if idx < elem_count {
                    out[idx] = min + value as f32 * scale;
                }
            }
        }
    }
    Ok(out)
}

fn decode_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac_norm = frac;
            let mut exp_norm = -14i32;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                exp_norm -= 1;
            }
            frac_norm &= 0x03ff;
            sign | (((exp_norm + 127) as u32) << 23) | (frac_norm << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | (((exp - 15 + 127) as u32) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mant = mant | 0x80_0000;
        let shift = (14 - half_exp) as u32;
        let rounded = (mant + (1 << (shift - 1))) >> shift;
        return sign | rounded as u16;
    }
    let rounded = mant + 0x1000;
    sign | ((half_exp as u16) << 10) | ((rounded >> 13) as u16 & 0x03ff)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffusionError {
    InvalidMetadata(String),
    InvalidRequest(String),
    BackendUnavailable(String),
    Interrupted(String),
    Io(String),
}

impl std::fmt::Display for DiffusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMetadata(message) => write!(f, "invalid diffusion metadata: {message}"),
            Self::InvalidRequest(message) => write!(f, "invalid diffusion request: {message}"),
            Self::BackendUnavailable(message) => {
                write!(f, "diffusion backend unavailable: {message}")
            }
            Self::Interrupted(message) => write!(f, "diffusion interrupted: {message}"),
            Self::Io(message) => write!(f, "diffusion I/O error: {message}"),
        }
    }
}

impl std::error::Error for DiffusionError {}

pub type DiffusionResult<T> = Result<T, DiffusionError>;

pub struct DiffusionPipeline {
    summary: DiffusionModelSummary,
    metadata: DiffusionHfqMetadata,
    config: StableDiffusionConfig,
    tokenizer: Option<ClipTokenizer>,
    tokenizer_2: Option<ClipTokenizer>,
    text_encoder: Option<ClipTextEncoder>,
    text_encoder_2: Option<ClipTextEncoder>,
    native_runtime: Option<NativeDiffusionRuntime>,
    native_runtime_error: Option<String>,
}

impl DiffusionPipeline {
    pub fn open_hfq(path: impl AsRef<Path>) -> DiffusionResult<Self> {
        let path = path.as_ref();
        let hfq =
            HfqFile::open_index_only(path).map_err(|err| DiffusionError::Io(err.to_string()))?;
        let metadata = parse_diffusion_metadata(&hfq.metadata_json)?;
        validate_diffusion_hfq(&hfq, &metadata)?;
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata)?;
        let tokenizer = ClipTokenizer::from_hfq_file(&hfq).ok();
        let tokenizer_2 = ClipTokenizer::from_hfq_file_with_prefix(&hfq, "tokenizer_2").ok();
        let text_encoder = ClipTextEncoder::from_hfq_file_with_heads(
            &hfq,
            config.text_encoder.num_attention_heads.unwrap_or(12),
        )
        .ok();
        let text_encoder_2 = config.text_encoder_2.as_ref().and_then(|config| {
            ClipTextEncoder::from_hfq_file_with_prefix_and_heads(
                &hfq,
                "text_encoder_2",
                config.num_attention_heads.unwrap_or(20),
            )
            .ok()
        });
        let runtime_support_error = native_runtime_metadata_support_error(&metadata);
        let (native_runtime, native_runtime_error) = if let Some(error) = runtime_support_error {
            (None, Some(error))
        } else {
            match NativeDiffusionRuntime::from_hfq(&hfq, &config) {
                Ok(runtime) => (Some(runtime), None),
                Err(error) => (None, Some(error.to_string())),
            }
        };
        let summary = summarize_hfq(path, &metadata);
        Ok(Self {
            summary,
            metadata,
            config,
            tokenizer,
            tokenizer_2,
            text_encoder,
            text_encoder_2,
            native_runtime,
            native_runtime_error,
        })
    }

    pub fn summary(&self) -> &DiffusionModelSummary {
        &self.summary
    }

    pub fn metadata(&self) -> &DiffusionHfqMetadata {
        &self.metadata
    }

    pub fn config(&self) -> &StableDiffusionConfig {
        &self.config
    }

    pub fn supports_img2img(&self) -> bool {
        self.native_runtime
            .as_ref()
            .and_then(|runtime| runtime.encoder.as_ref())
            .is_some()
    }

    pub fn runtime_capabilities(&self) -> Option<DiffusionRuntimeCapabilities> {
        let runtime = self.native_runtime.as_ref()?;
        Some(DiffusionRuntimeCapabilities {
            kind: runtime.kind,
            weight_format: self.metadata.quantization.weight_format.clone(),
            activation_format: self.metadata.quantization.activation_format.clone(),
            tensor_roles_version: self.metadata.quantization.tensor_roles_version,
            max_batch: self.metadata.batch.max_batch,
            supports_img2img: runtime.encoder.is_some(),
        })
    }

    pub fn hip_memory_plan(
        &self,
        request: &DiffusionBatchRequest,
    ) -> DiffusionResult<DiffusionHipMemoryPlan> {
        diffusion_hip_memory_plan(&self.config, request)
    }

    #[cfg(feature = "rocm")]
    pub fn preflight_hip_runtime(
        &self,
        request: &DiffusionBatchRequest,
        options: DiffusionHipRuntimeOptions,
    ) -> DiffusionResult<DiffusionHipPreflight> {
        validate_batch_request(&self.metadata, request)?;
        let memory_plan = self.hip_memory_plan(request)?;
        let mut gpu = rdna_compute::Gpu::init_with_device(options.device_id)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        gpu.bind_thread()
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;

        for bytes in [
            memory_plan.latent_bytes,
            memory_plan.denoise_input_bytes,
            memory_plan.conditioning_bytes,
            memory_plan.vae_decode_bytes,
            memory_plan.rgb_bytes,
            memory_plan.scheduler_scratch_bytes,
        ] {
            if bytes == 0 {
                continue;
            }
            let buffer = gpu
                .hip
                .malloc(bytes)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            gpu.hip
                .memset(&buffer, 0, bytes)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            gpu.hip
                .free(buffer)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        }

        let probe = b"hipfire-diffusion-rocm-preflight";
        let probe_buffer = gpu
            .hip
            .malloc(probe.len())
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        gpu.hip
            .memcpy_htod(&probe_buffer, probe)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        let mut roundtrip = vec![0u8; probe.len()];
        gpu.hip
            .memcpy_dtoh(&mut roundtrip, &probe_buffer)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        gpu.hip
            .free(probe_buffer)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        gpu.hip
            .device_synchronize()
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        if roundtrip != probe {
            return Err(DiffusionError::BackendUnavailable(
                "HIP preflight probe roundtrip mismatch".to_string(),
            ));
        }
        let kernel_probe_tensor = CpuTensor {
            shape: vec![1, 3, 2, 2],
            data: vec![
                -1.0, 0.0, 1.0, 0.25, -0.5, 0.5, -0.25, 0.75, 1.0, -1.0, 0.1, -0.1,
            ],
        };
        let cpu_reference = rgb_tensor_to_u8(&kernel_probe_tensor)?;
        let gpu_output = rgb_tensor_to_u8_hip_on_gpu(&mut gpu, &kernel_probe_tensor)?;
        let kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_rgb_tensor_to_u8".to_string(),
            input_elements: kernel_probe_tensor.data.len(),
            output_bytes: gpu_output.data.len(),
            matched_cpu_reference: gpu_output == cpu_reference,
        };
        if !kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion RGB kernel output differed from CPU reference".to_string(),
            ));
        }
        let vae_image_probe = RgbImageBatch {
            batch: 2,
            width: 2,
            height: 2,
            data: vec![
                0, 128, 255, 255, 0, 128, 32, 64, 96, 192, 224, 16, 10, 20, 30, 40, 50, 60, 70, 80,
                90, 100, 110, 120,
            ],
        };
        let rgb_to_vae_tensor_cpu_reference = rgb_batch_to_vae_tensor(&vae_image_probe)?;
        let rgb_to_vae_tensor_gpu_output =
            rgb_batch_to_vae_tensor_hip_on_gpu(&mut gpu, &vae_image_probe)?;
        let rgb_to_vae_tensor_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_rgb_u8_to_vae_nchw_f32".to_string(),
            input_elements: vae_image_probe.data.len(),
            output_bytes: rgb_to_vae_tensor_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: rgb_to_vae_tensor_gpu_output.shape
                == rgb_to_vae_tensor_cpu_reference.shape
                && f32_slices_close(
                    &rgb_to_vae_tensor_gpu_output.data,
                    &rgb_to_vae_tensor_cpu_reference.data,
                    1e-6,
                ),
        };
        if !rgb_to_vae_tensor_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion RGB-to-VAE kernel output differed from CPU reference".to_string(),
            ));
        }
        let inpaint_mask_probe = RgbImageBatch {
            batch: 2,
            width: 4,
            height: 4,
            data: (0..96)
                .map(|idx| ((idx * 37 + 11) % 256) as u8)
                .collect::<Vec<_>>(),
        };
        let inpaint_image_probe = RgbImageBatch {
            batch: 2,
            width: 4,
            height: 4,
            data: (0..96)
                .map(|idx| ((idx * 19 + 5) % 256) as u8)
                .collect::<Vec<_>>(),
        };
        let inpaint_latent_probe = LatentBatch {
            batch: 2,
            channels: 2,
            height: 2,
            width: 2,
            data: (0..16)
                .map(|idx| idx as f32 / 7.0 - 1.0)
                .collect::<Vec<_>>(),
        };
        let latent_mask_weights_cpu_reference =
            latent_mask_weights_from_rgb_batch(&inpaint_mask_probe, &inpaint_latent_probe)?;
        let latent_mask_weights_gpu_output = latent_mask_weights_from_rgb_batch_hip_on_gpu(
            &mut gpu,
            &inpaint_mask_probe,
            &inpaint_latent_probe,
        )?;
        let latent_mask_weights_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_latent_mask_weights_from_rgb_f32".to_string(),
            input_elements: inpaint_mask_probe.data.len(),
            output_bytes: latent_mask_weights_gpu_output.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: f32_slices_close(
                &latent_mask_weights_gpu_output,
                &latent_mask_weights_cpu_reference,
                1e-6,
            ),
        };
        if !latent_mask_weights_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion latent-mask kernel output differed from CPU reference".to_string(),
            ));
        }
        let masked_rgb_inpaint_cpu_reference =
            masked_rgb_batch_for_inpaint(&inpaint_image_probe, &inpaint_mask_probe)?;
        let masked_rgb_inpaint_gpu_output = masked_rgb_batch_for_inpaint_hip_on_gpu(
            &mut gpu,
            &inpaint_image_probe,
            &inpaint_mask_probe,
        )?;
        let masked_rgb_inpaint_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_masked_rgb_for_inpaint_u8".to_string(),
            input_elements: inpaint_image_probe.data.len() + inpaint_mask_probe.data.len(),
            output_bytes: masked_rgb_inpaint_gpu_output.data.len(),
            matched_cpu_reference: masked_rgb_inpaint_gpu_output
                == masked_rgb_inpaint_cpu_reference,
        };
        if !masked_rgb_inpaint_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion masked-RGB kernel output differed from CPU reference".to_string(),
            ));
        }
        let generated_latents_probe = LatentBatch {
            batch: 2,
            channels: 2,
            height: 2,
            width: 2,
            data: (0..16)
                .map(|idx| (idx as f32 % 9.0 - 4.0) / 3.0)
                .collect::<Vec<_>>(),
        };
        let mut blend_latents_cpu_reference = generated_latents_probe.clone();
        blend_latents_with_mask(
            &mut blend_latents_cpu_reference,
            &inpaint_latent_probe,
            &latent_mask_weights_cpu_reference,
        )?;
        let blend_latents_gpu_output = blend_latents_with_mask_hip_on_gpu(
            &mut gpu,
            &generated_latents_probe,
            &inpaint_latent_probe,
            &latent_mask_weights_cpu_reference,
        )?;
        let blend_latents_with_mask_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_blend_latents_with_mask_f32".to_string(),
            input_elements: generated_latents_probe.data.len()
                + inpaint_latent_probe.data.len()
                + latent_mask_weights_cpu_reference.len(),
            output_bytes: blend_latents_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: blend_latents_gpu_output.batch
                == blend_latents_cpu_reference.batch
                && blend_latents_gpu_output.channels == blend_latents_cpu_reference.channels
                && blend_latents_gpu_output.height == blend_latents_cpu_reference.height
                && blend_latents_gpu_output.width == blend_latents_cpu_reference.width
                && f32_slices_close(
                    &blend_latents_gpu_output.data,
                    &blend_latents_cpu_reference.data,
                    1e-6,
                ),
        };
        if !blend_latents_with_mask_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion latent-blend kernel output differed from CPU reference".to_string(),
            ));
        }
        let model_input_sample = vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125];
        let model_input_scale = 0.5;
        let model_input_cpu_reference = model_input_sample
            .iter()
            .map(|sample| sample * model_input_scale)
            .collect::<Vec<_>>();
        let model_input_gpu_output =
            scale_model_input_hip_on_gpu(&mut gpu, &model_input_sample, model_input_scale)?;
        let model_input_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_scale_model_input_f32".to_string(),
            input_elements: model_input_sample.len(),
            output_bytes: model_input_gpu_output.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: f32_slices_close(
                &model_input_gpu_output,
                &model_input_cpu_reference,
                1e-6,
            ),
        };
        if !model_input_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion model-input scaling kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let guidance_negative = vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125];
        let guidance_positive = vec![0.5, -0.5, 0.25, -0.25, 1.5, -1.0, 0.75, -0.125];
        let guidance_scale = 7.5;
        let guidance_cpu_reference = guidance_negative
            .iter()
            .zip(&guidance_positive)
            .map(|(negative, positive)| negative + guidance_scale * (positive - negative))
            .collect::<Vec<_>>();
        let guidance_gpu_output = cfg_guidance_hip_on_gpu(
            &mut gpu,
            &guidance_negative,
            &guidance_positive,
            guidance_scale,
        )?;
        let guidance_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_cfg_guidance_f32".to_string(),
            input_elements: guidance_negative.len(),
            output_bytes: guidance_gpu_output.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: f32_slices_close(
                &guidance_gpu_output,
                &guidance_cpu_reference,
                1e-6,
            ),
        };
        if !guidance_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion CFG guidance kernel output differed from CPU reference".to_string(),
            ));
        }
        let scheduler_sample = vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125];
        let scheduler_model_output = vec![0.5, -0.5, 0.25, -0.25, 1.5, -1.0, 0.75, -0.125];
        let scheduler_sigma = 1.0;
        let scheduler_next_sigma = 0.5;
        let scheduler_cpu_reference = scheduler_sample
            .iter()
            .zip(&scheduler_model_output)
            .map(|(sample, model_output)| {
                sample
                    + scheduler_derivative(
                        *sample,
                        *model_output,
                        scheduler_sigma,
                        SchedulerPredictionType::Epsilon,
                    ) * (scheduler_next_sigma - scheduler_sigma)
            })
            .collect::<Vec<_>>();
        let scheduler_gpu_output = euler_step_hip_on_gpu(
            &mut gpu,
            &scheduler_sample,
            &scheduler_model_output,
            scheduler_sigma,
            scheduler_next_sigma,
            SchedulerPredictionType::Epsilon,
        )?;
        let scheduler_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_euler_step_f32".to_string(),
            input_elements: scheduler_sample.len(),
            output_bytes: scheduler_gpu_output.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: scheduler_gpu_output == scheduler_cpu_reference,
        };
        if !scheduler_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion scheduler kernel output differed from CPU reference".to_string(),
            ));
        }
        let center_input = CpuTensor {
            shape: vec![1, 2, 2, 2],
            data: vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125],
        };
        let center_unet_input_cpu_reference = maybe_center_unet_input(&center_input, true);
        let center_unet_input_gpu_output =
            maybe_center_unet_input_hip_on_gpu(&mut gpu, &center_input, true)?;
        let center_unet_input_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_center_unet_input_f32".to_string(),
            input_elements: center_input.data.len(),
            output_bytes: center_unet_input_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: center_unet_input_gpu_output.shape
                == center_unet_input_cpu_reference.shape
                && center_unet_input_gpu_output.data == center_unet_input_cpu_reference.data,
        };
        if !center_unet_input_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion centered UNet input kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let timestep_values = vec![999.0, 500.5, 0.25];
        let timestep_embedding_dim = 7;
        let timestep_embedding_cpu_reference =
            timestep_embedding(&timestep_values, timestep_embedding_dim, true, 1.0)?;
        let timestep_embedding_gpu_output = timestep_embedding_hip_on_gpu(
            &mut gpu,
            &timestep_values,
            timestep_embedding_dim,
            true,
            1.0,
        )?;
        let timestep_embedding_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_timestep_embedding_f32".to_string(),
            input_elements: timestep_values.len(),
            output_bytes: timestep_embedding_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: timestep_embedding_gpu_output.shape
                == timestep_embedding_cpu_reference.shape
                && f32_slices_close(
                    &timestep_embedding_gpu_output.data,
                    &timestep_embedding_cpu_reference.data,
                    1e-5,
                ),
        };
        if !timestep_embedding_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion timestep embedding kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let tensor_add_left = CpuTensor {
            shape: vec![1, 2, 2, 3],
            data: (0..12)
                .map(|idx| idx as f32 / 8.0 - 0.75)
                .collect::<Vec<_>>(),
        };
        let tensor_add_right = CpuTensor {
            shape: vec![1, 2, 2, 3],
            data: (0..12)
                .map(|idx| (idx as f32 % 5.0 - 2.0) / 3.0)
                .collect::<Vec<_>>(),
        };
        let tensor_add_cpu_reference = tensor_add(&tensor_add_left, &tensor_add_right)?;
        let tensor_add_gpu_output =
            tensor_add_hip_on_gpu(&mut gpu, &tensor_add_left, &tensor_add_right)?;
        let tensor_add_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_tensor_add_f32".to_string(),
            input_elements: tensor_add_left.data.len() + tensor_add_right.data.len(),
            output_bytes: tensor_add_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: tensor_add_gpu_output.shape == tensor_add_cpu_reference.shape
                && f32_slices_close(
                    &tensor_add_gpu_output.data,
                    &tensor_add_cpu_reference.data,
                    1e-6,
                ),
        };
        if !tensor_add_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion tensor-add kernel output differed from CPU reference".to_string(),
            ));
        }
        let channel_bias_input = CpuTensor {
            shape: vec![2, 3, 2, 2],
            data: (0..24)
                .map(|idx| idx as f32 / 10.0 - 1.0)
                .collect::<Vec<_>>(),
        };
        let channel_bias = CpuTensor {
            shape: vec![2, 3],
            data: vec![0.25, -0.5, 0.75, -1.0, 0.5, -0.25],
        };
        let mut add_channel_bias_cpu_reference = channel_bias_input.clone();
        add_channel_bias_nchw(&mut add_channel_bias_cpu_reference, &channel_bias)?;
        let add_channel_bias_gpu_output =
            add_channel_bias_nchw_hip_on_gpu(&mut gpu, &channel_bias_input, &channel_bias)?;
        let add_channel_bias_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_add_channel_bias_nchw_f32".to_string(),
            input_elements: channel_bias_input.data.len() + channel_bias.data.len(),
            output_bytes: add_channel_bias_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: add_channel_bias_gpu_output.shape
                == add_channel_bias_cpu_reference.shape
                && f32_slices_close(
                    &add_channel_bias_gpu_output.data,
                    &add_channel_bias_cpu_reference.data,
                    1e-6,
                ),
        };
        if !add_channel_bias_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion channel-bias kernel output differed from CPU reference".to_string(),
            ));
        }
        let layout_input = CpuTensor {
            shape: vec![2, 3, 2, 4],
            data: (0..48)
                .map(|idx| idx as f32 / 11.0 - 1.75)
                .collect::<Vec<_>>(),
        };
        let nchw_to_bsc_cpu_reference = nchw_to_bsc(&layout_input)?;
        let nchw_to_bsc_gpu_output = nchw_to_bsc_hip_on_gpu(&mut gpu, &layout_input)?;
        let nchw_to_bsc_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_nchw_to_bsc_f32".to_string(),
            input_elements: layout_input.data.len(),
            output_bytes: nchw_to_bsc_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: nchw_to_bsc_gpu_output.shape == nchw_to_bsc_cpu_reference.shape
                && nchw_to_bsc_gpu_output.data == nchw_to_bsc_cpu_reference.data,
        };
        if !nchw_to_bsc_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion NCHW-to-BSC kernel output differed from CPU reference".to_string(),
            ));
        }
        let bsc_to_nchw_cpu_reference = bsc_to_nchw(&nchw_to_bsc_cpu_reference, 2, 3, 2, 4)?;
        let bsc_to_nchw_gpu_output =
            bsc_to_nchw_hip_on_gpu(&mut gpu, &nchw_to_bsc_cpu_reference, 2, 3, 2, 4)?;
        let bsc_to_nchw_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_bsc_to_nchw_f32".to_string(),
            input_elements: nchw_to_bsc_cpu_reference.data.len(),
            output_bytes: bsc_to_nchw_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: bsc_to_nchw_gpu_output.shape == bsc_to_nchw_cpu_reference.shape
                && bsc_to_nchw_gpu_output.data == bsc_to_nchw_cpu_reference.data,
        };
        if !bsc_to_nchw_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion BSC-to-NCHW kernel output differed from CPU reference".to_string(),
            ));
        }
        let concat_channels_left = CpuTensor {
            shape: vec![1, 2, 2, 3],
            data: (0..12)
                .map(|idx| idx as f32 / 7.0 - 0.75)
                .collect::<Vec<_>>(),
        };
        let concat_channels_right = CpuTensor {
            shape: vec![1, 3, 2, 3],
            data: (0..18)
                .map(|idx| (idx as f32 % 11.0 - 5.0) / 6.0)
                .collect::<Vec<_>>(),
        };
        let concat_channels_cpu_reference =
            concat_channels_nchw(&concat_channels_left, &concat_channels_right)?;
        let concat_channels_gpu_output = concat_channels_nchw_hip_on_gpu(
            &mut gpu,
            &concat_channels_left,
            &concat_channels_right,
        )?;
        let concat_channels_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_concat_channels_nchw_f32".to_string(),
            input_elements: concat_channels_left.data.len() + concat_channels_right.data.len(),
            output_bytes: concat_channels_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: concat_channels_gpu_output.shape
                == concat_channels_cpu_reference.shape
                && concat_channels_gpu_output.data == concat_channels_cpu_reference.data,
        };
        if !concat_channels_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion channel-concat kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let concat_2d_left = CpuTensor {
            shape: vec![3, 2],
            data: vec![0.25, -0.5, 0.75, 1.0, -1.25, 0.5],
        };
        let concat_2d_right = CpuTensor {
            shape: vec![3, 3],
            data: vec![-1.0, 0.0, 1.0, 0.5, -0.25, 0.75, 1.25, -0.75, 0.25],
        };
        let concat_last_dim_2d_cpu_reference =
            concat_last_dim_2d(&concat_2d_left, &concat_2d_right)?;
        let concat_last_dim_2d_gpu_output =
            concat_last_dim_2d_hip_on_gpu(&mut gpu, &concat_2d_left, &concat_2d_right)?;
        let concat_last_dim_2d_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_concat_last_dim_2d_f32".to_string(),
            input_elements: concat_2d_left.data.len() + concat_2d_right.data.len(),
            output_bytes: concat_last_dim_2d_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: concat_last_dim_2d_gpu_output.shape
                == concat_last_dim_2d_cpu_reference.shape
                && concat_last_dim_2d_gpu_output.data == concat_last_dim_2d_cpu_reference.data,
        };
        if !concat_last_dim_2d_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion 2D last-dim concat kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let concat_3d_left = CpuTensor {
            shape: vec![2, 2, 2],
            data: (0..8).map(|idx| idx as f32 / 4.0 - 0.5).collect::<Vec<_>>(),
        };
        let concat_3d_right = CpuTensor {
            shape: vec![2, 2, 3],
            data: (0..12)
                .map(|idx| (idx as f32 % 7.0 - 3.0) / 5.0)
                .collect::<Vec<_>>(),
        };
        let concat_last_dim_3d_cpu_reference =
            concat_last_dim_3d(&concat_3d_left, &concat_3d_right)?;
        let concat_last_dim_3d_gpu_output =
            concat_last_dim_3d_hip_on_gpu(&mut gpu, &concat_3d_left, &concat_3d_right)?;
        let concat_last_dim_3d_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_concat_last_dim_3d_f32".to_string(),
            input_elements: concat_3d_left.data.len() + concat_3d_right.data.len(),
            output_bytes: concat_last_dim_3d_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: concat_last_dim_3d_gpu_output.shape
                == concat_last_dim_3d_cpu_reference.shape
                && concat_last_dim_3d_gpu_output.data == concat_last_dim_3d_cpu_reference.data,
        };
        if !concat_last_dim_3d_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion 3D last-dim concat kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let conv2d_input = CpuTensor {
            shape: vec![1, 2, 3, 4],
            data: (0..24)
                .map(|idx| idx as f32 / 8.0 - 1.5)
                .collect::<Vec<_>>(),
        };
        let conv2d_weight = CpuTensor {
            shape: vec![3, 2, 3, 2],
            data: (0..36)
                .map(|idx| (idx as f32 % 7.0 - 3.0) / 5.0)
                .collect::<Vec<_>>(),
        };
        let conv2d_bias = CpuTensor {
            shape: vec![3],
            data: vec![0.25, -0.5, 0.75],
        };
        let conv2d_cpu_reference =
            conv2d_nchw_with_stride(&conv2d_input, &conv2d_weight, Some(&conv2d_bias), 1, 2)?;
        let conv2d_gpu_output = conv2d_nchw_hip_on_gpu(
            &mut gpu,
            &conv2d_input,
            &conv2d_weight,
            Some(&conv2d_bias),
            1,
            2,
        )?;
        let conv2d_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_conv2d_nchw_f32".to_string(),
            input_elements: conv2d_input.data.len() + conv2d_weight.data.len(),
            output_bytes: conv2d_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: conv2d_gpu_output.shape == conv2d_cpu_reference.shape
                && f32_slices_close(&conv2d_gpu_output.data, &conv2d_cpu_reference.data, 1e-5),
        };
        if !conv2d_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion Conv2D kernel output differed from CPU reference".to_string(),
            ));
        }
        let group_norm_input = CpuTensor {
            shape: vec![1, 4, 2, 3],
            data: (0..24)
                .map(|idx| idx as f32 / 7.0 - 1.5)
                .collect::<Vec<_>>(),
        };
        let group_norm_weight = CpuTensor {
            shape: vec![4],
            data: vec![1.0, 0.5, -1.0, 1.5],
        };
        let group_norm_bias = CpuTensor {
            shape: vec![4],
            data: vec![0.0, 0.25, -0.5, 0.75],
        };
        let group_norm_cpu_reference = group_norm_nchw(
            &group_norm_input,
            &group_norm_weight,
            &group_norm_bias,
            2,
            1e-5,
        )?;
        let group_norm_gpu_output = group_norm_nchw_hip_on_gpu(
            &mut gpu,
            &group_norm_input,
            &group_norm_weight,
            &group_norm_bias,
            2,
            1e-5,
        )?;
        let group_norm_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_group_norm_nchw_f32".to_string(),
            input_elements: group_norm_input.data.len()
                + group_norm_weight.data.len()
                + group_norm_bias.data.len(),
            output_bytes: group_norm_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: group_norm_gpu_output.shape == group_norm_cpu_reference.shape
                && f32_slices_close(
                    &group_norm_gpu_output.data,
                    &group_norm_cpu_reference.data,
                    1e-5,
                ),
        };
        if !group_norm_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion GroupNorm kernel output differed from CPU reference".to_string(),
            ));
        }
        let silu_input = CpuTensor {
            shape: vec![1, 2, 2, 2],
            data: vec![-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0],
        };
        let silu_cpu_reference = tensor_map(&silu_input, silu);
        let silu_gpu_output = silu_hip_on_gpu(&mut gpu, &silu_input)?;
        let silu_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_silu_f32".to_string(),
            input_elements: silu_input.data.len(),
            output_bytes: silu_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: silu_gpu_output.shape == silu_cpu_reference.shape
                && f32_slices_close(&silu_gpu_output.data, &silu_cpu_reference.data, 1e-6),
        };
        if !silu_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion SiLU kernel output differed from CPU reference".to_string(),
            ));
        }
        let quick_gelu_input = CpuTensor {
            shape: vec![2, 4],
            data: vec![-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0],
        };
        let quick_gelu_cpu_reference = tensor_map(&quick_gelu_input, quick_gelu);
        let quick_gelu_gpu_output = quick_gelu_hip_on_gpu(&mut gpu, &quick_gelu_input)?;
        let quick_gelu_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_quick_gelu_f32".to_string(),
            input_elements: quick_gelu_input.data.len(),
            output_bytes: quick_gelu_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: quick_gelu_gpu_output.shape == quick_gelu_cpu_reference.shape
                && f32_slices_close(
                    &quick_gelu_gpu_output.data,
                    &quick_gelu_cpu_reference.data,
                    1e-6,
                ),
        };
        if !quick_gelu_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion QuickGELU kernel output differed from CPU reference".to_string(),
            ));
        }
        let clip_token_embedding_probe = CpuTensor {
            shape: vec![4, 3],
            data: (0..12)
                .map(|idx| idx as f32 / 10.0 - 0.4)
                .collect::<Vec<_>>(),
        };
        let clip_position_embedding_probe = CpuTensor {
            shape: vec![3, 3],
            data: (0..9)
                .map(|idx| (idx as f32 % 5.0 - 2.0) / 7.0)
                .collect::<Vec<_>>(),
        };
        let clip_token_probe = vec![0, 3, 1];
        let clip_token_position_embedding_cpu_reference = clip_token_position_embeddings(
            &clip_token_embedding_probe,
            &clip_position_embedding_probe,
            &clip_token_probe,
        )?;
        let clip_token_position_embedding_gpu_output = clip_token_position_embeddings_hip_on_gpu(
            &mut gpu,
            &clip_token_embedding_probe,
            &clip_position_embedding_probe,
            &clip_token_probe,
        )?;
        let clip_token_position_embedding_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_clip_token_position_embedding_f32".to_string(),
            input_elements: clip_token_embedding_probe.data.len()
                + clip_position_embedding_probe.data.len()
                + clip_token_probe.len(),
            output_bytes: clip_token_position_embedding_gpu_output.data.len()
                * std::mem::size_of::<f32>(),
            matched_cpu_reference: clip_token_position_embedding_gpu_output.shape
                == clip_token_position_embedding_cpu_reference.shape
                && f32_slices_close(
                    &clip_token_position_embedding_gpu_output.data,
                    &clip_token_position_embedding_cpu_reference.data,
                    1e-6,
                ),
        };
        if !clip_token_position_embedding_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion CLIP token-position embedding kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let upsample_input = CpuTensor {
            shape: vec![1, 2, 2, 3],
            data: (0..12)
                .map(|idx| idx as f32 / 5.0 - 1.0)
                .collect::<Vec<_>>(),
        };
        let upsample_cpu_reference = upsample_nearest2d_nchw(&upsample_input, 2)?;
        let upsample_gpu_output = upsample_nearest2d_nchw_hip_on_gpu(&mut gpu, &upsample_input, 2)?;
        let upsample_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_upsample_nearest2d_nchw_f32".to_string(),
            input_elements: upsample_input.data.len(),
            output_bytes: upsample_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: upsample_gpu_output.shape == upsample_cpu_reference.shape
                && upsample_gpu_output.data == upsample_cpu_reference.data,
        };
        if !upsample_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion nearest-upsample kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let linear_input = CpuTensor {
            shape: vec![3, 4],
            data: (0..12)
                .map(|idx| idx as f32 / 6.0 - 1.0)
                .collect::<Vec<_>>(),
        };
        let linear_weight = CpuTensor {
            shape: vec![5, 4],
            data: (0..20)
                .map(|idx| (idx as f32 % 11.0 - 5.0) / 7.0)
                .collect::<Vec<_>>(),
        };
        let linear_bias = CpuTensor {
            shape: vec![5],
            data: vec![0.0, 0.25, -0.5, 0.75, -1.0],
        };
        let linear_cpu_reference = linear(&linear_input, &linear_weight, &linear_bias)?;
        let linear_gpu_output = linear_optional_bias_hip_on_gpu(
            &mut gpu,
            &linear_input,
            &linear_weight,
            Some(&linear_bias),
        )?;
        let linear_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_linear_f32".to_string(),
            input_elements: linear_input.data.len()
                + linear_weight.data.len()
                + linear_bias.data.len(),
            output_bytes: linear_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: linear_gpu_output.shape == linear_cpu_reference.shape
                && f32_slices_close(&linear_gpu_output.data, &linear_cpu_reference.data, 1e-5),
        };
        if !linear_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion linear kernel output differed from CPU reference".to_string(),
            ));
        }
        let layer_norm_input = CpuTensor {
            shape: vec![3, 4],
            data: (0..12)
                .map(|idx| idx as f32 / 5.0 - 1.25)
                .collect::<Vec<_>>(),
        };
        let layer_norm_weight = CpuTensor {
            shape: vec![4],
            data: vec![1.0, 0.5, -1.0, 1.5],
        };
        let layer_norm_bias = CpuTensor {
            shape: vec![4],
            data: vec![0.0, 0.25, -0.5, 0.75],
        };
        let layer_norm_cpu_reference = layer_norm(
            &layer_norm_input,
            &layer_norm_weight,
            &layer_norm_bias,
            1e-5,
        )?;
        let layer_norm_gpu_output = layer_norm_hip_on_gpu(
            &mut gpu,
            &layer_norm_input,
            &layer_norm_weight,
            &layer_norm_bias,
            1e-5,
        )?;
        let layer_norm_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_layer_norm_f32".to_string(),
            input_elements: layer_norm_input.data.len()
                + layer_norm_weight.data.len()
                + layer_norm_bias.data.len(),
            output_bytes: layer_norm_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: layer_norm_gpu_output.shape == layer_norm_cpu_reference.shape
                && f32_slices_close(
                    &layer_norm_gpu_output.data,
                    &layer_norm_cpu_reference.data,
                    1e-5,
                ),
        };
        if !layer_norm_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion LayerNorm kernel output differed from CPU reference".to_string(),
            ));
        }
        let softmax_input = CpuTensor {
            shape: vec![3, 4],
            data: vec![
                1.0, 2.0, 3.0, 4.0, -2.0, -0.5, 0.25, 1.5, 10.0, 9.0, 8.0, 7.0,
            ],
        };
        let mut softmax_cpu_reference = softmax_input.clone();
        for row in softmax_cpu_reference.data.chunks_mut(4) {
            softmax_in_place(row);
        }
        let softmax_gpu_output = softmax_rows_hip_on_gpu(&mut gpu, &softmax_input)?;
        let softmax_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_softmax_rows_f32".to_string(),
            input_elements: softmax_input.data.len(),
            output_bytes: softmax_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: softmax_gpu_output.shape == softmax_cpu_reference.shape
                && f32_slices_close(&softmax_gpu_output.data, &softmax_cpu_reference.data, 1e-6),
        };
        if !softmax_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion softmax kernel output differed from CPU reference".to_string(),
            ));
        }
        let sdpa_q = CpuTensor {
            shape: vec![1, 2, 4],
            data: vec![0.2, -0.4, 0.6, 1.0, -1.2, 0.3, 0.5, -0.7],
        };
        let sdpa_k = CpuTensor {
            shape: vec![1, 3, 4],
            data: vec![
                -0.5, 0.25, 0.75, -1.0, 1.25, -0.75, 0.5, 0.0, 0.1, 0.9, -0.3, 0.4,
            ],
        };
        let sdpa_v = CpuTensor {
            shape: vec![1, 3, 4],
            data: vec![
                0.5, -1.0, 0.25, 0.75, -0.4, 0.6, -0.8, 1.2, 1.0, 0.2, -0.5, -0.1,
            ],
        };
        let sdpa_cpu_reference = scaled_dot_product_attention(&sdpa_q, &sdpa_k, &sdpa_v, 2)?;
        let sdpa_gpu_output =
            scaled_dot_product_attention_hip_on_gpu(&mut gpu, &sdpa_q, &sdpa_k, &sdpa_v, 2)?;
        let sdpa_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_sdpa_3d_f32".to_string(),
            input_elements: sdpa_q.data.len() + sdpa_k.data.len() + sdpa_v.data.len(),
            output_bytes: sdpa_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: sdpa_gpu_output.shape == sdpa_cpu_reference.shape
                && f32_slices_close(&sdpa_gpu_output.data, &sdpa_cpu_reference.data, 1e-5),
        };
        if !sdpa_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion SDPA kernel output differed from CPU reference".to_string(),
            ));
        }
        let clip_attn_q = CpuTensor {
            shape: vec![4, 4],
            data: vec![
                0.2, -0.4, 0.6, 1.0, -1.2, 0.3, 0.5, -0.7, 1.0, 0.0, -0.5, 0.25, -0.25, 0.75, -1.0,
                0.5,
            ],
        };
        let clip_attn_k = CpuTensor {
            shape: vec![4, 4],
            data: vec![
                -0.5, 0.25, 0.75, -1.0, 1.25, -0.75, 0.5, 0.0, 0.1, 0.9, -0.3, 0.4, 0.7, -0.2, 0.3,
                -0.8,
            ],
        };
        let clip_attn_v = CpuTensor {
            shape: vec![4, 4],
            data: vec![
                0.5, -1.0, 0.25, 0.75, -0.4, 0.6, -0.8, 1.2, 1.0, 0.2, -0.5, -0.1, -0.9, 0.3, 0.8,
                -0.2,
            ],
        };
        let clip_causal_attention_cpu_reference =
            clip_causal_self_attention(&clip_attn_q, &clip_attn_k, &clip_attn_v, 2)?;
        let clip_causal_attention_gpu_output = clip_causal_self_attention_hip_on_gpu(
            &mut gpu,
            &clip_attn_q,
            &clip_attn_k,
            &clip_attn_v,
            2,
        )?;
        let clip_causal_attention_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_clip_causal_attention_f32".to_string(),
            input_elements: clip_attn_q.data.len()
                + clip_attn_k.data.len()
                + clip_attn_v.data.len(),
            output_bytes: clip_causal_attention_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: clip_causal_attention_gpu_output.shape
                == clip_causal_attention_cpu_reference.shape
                && f32_slices_close(
                    &clip_causal_attention_gpu_output.data,
                    &clip_causal_attention_cpu_reference.data,
                    1e-5,
                ),
        };
        if !clip_causal_attention_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion CLIP causal attention kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let geglu_projected = CpuTensor {
            shape: vec![1, 3, 6],
            data: vec![
                0.5, -0.25, 1.0, -1.5, 0.2, 0.75, -0.4, 0.9, -1.1, 0.6, -0.8, 1.25, 1.5, -1.0, 0.3,
                0.0, 1.1, -0.6,
            ],
        };
        let geglu_gate_cpu_reference = geglu_gate_3d(&geglu_projected)?;
        let geglu_gate_gpu_output = geglu_gate_3d_hip_on_gpu(&mut gpu, &geglu_projected)?;
        let geglu_gate_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_geglu_gate_3d_f32".to_string(),
            input_elements: geglu_projected.data.len(),
            output_bytes: geglu_gate_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: geglu_gate_gpu_output.shape == geglu_gate_cpu_reference.shape
                && f32_slices_close(
                    &geglu_gate_gpu_output.data,
                    &geglu_gate_cpu_reference.data,
                    1e-5,
                ),
        };
        if !geglu_gate_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion GeGLU gate kernel output differed from CPU reference".to_string(),
            ));
        }
        let vae_moments = CpuTensor {
            shape: vec![2, 4, 2, 2],
            data: (0..32)
                .map(|idx| idx as f32 / 9.0 - 1.5)
                .collect::<Vec<_>>(),
        };
        let vae_moments_to_latents_cpu_reference = vae_moments_to_latents(&vae_moments, 0.18215)?;
        let vae_moments_to_latents_gpu_output =
            vae_moments_to_latents_hip_on_gpu(&mut gpu, &vae_moments, 0.18215)?;
        let vae_moments_to_latents_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_vae_moments_to_latents_f32".to_string(),
            input_elements: vae_moments.data.len(),
            output_bytes: vae_moments_to_latents_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: vae_moments_to_latents_gpu_output.batch
                == vae_moments_to_latents_cpu_reference.batch
                && vae_moments_to_latents_gpu_output.channels
                    == vae_moments_to_latents_cpu_reference.channels
                && vae_moments_to_latents_gpu_output.height
                    == vae_moments_to_latents_cpu_reference.height
                && vae_moments_to_latents_gpu_output.width
                    == vae_moments_to_latents_cpu_reference.width
                && f32_slices_close(
                    &vae_moments_to_latents_gpu_output.data,
                    &vae_moments_to_latents_cpu_reference.data,
                    1e-6,
                ),
        };
        if !vae_moments_to_latents_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion VAE moments-to-latents kernel output differed from CPU reference"
                    .to_string(),
            ));
        }

        Ok(DiffusionHipPreflight {
            device_id: gpu.device_id,
            arch: gpu.arch.clone(),
            integrated: gpu.integrated,
            memory_plan,
            probe_bytes: probe.len(),
            kernel_probe,
            rgb_to_vae_tensor_kernel_probe,
            latent_mask_weights_kernel_probe,
            masked_rgb_inpaint_kernel_probe,
            blend_latents_with_mask_kernel_probe,
            model_input_kernel_probe,
            guidance_kernel_probe,
            scheduler_kernel_probe,
            center_unet_input_kernel_probe,
            timestep_embedding_kernel_probe,
            tensor_add_kernel_probe,
            add_channel_bias_kernel_probe,
            nchw_to_bsc_kernel_probe,
            bsc_to_nchw_kernel_probe,
            concat_channels_kernel_probe,
            concat_last_dim_2d_kernel_probe,
            concat_last_dim_3d_kernel_probe,
            conv2d_kernel_probe,
            group_norm_kernel_probe,
            silu_kernel_probe,
            quick_gelu_kernel_probe,
            clip_token_position_embedding_kernel_probe,
            upsample_kernel_probe,
            linear_kernel_probe,
            layer_norm_kernel_probe,
            softmax_kernel_probe,
            sdpa_kernel_probe,
            clip_causal_attention_kernel_probe,
            geglu_gate_kernel_probe,
            vae_moments_to_latents_kernel_probe,
        })
    }

    pub fn generate_batch(
        &self,
        request: DiffusionBatchRequest,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_batch_inner(request, DiffusionGenerationRuntimeOptions::default(), None)
    }

    pub fn generate_batch_with_progress(
        &self,
        request: DiffusionBatchRequest,
        progress: &mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_batch_inner(
            request,
            DiffusionGenerationRuntimeOptions::default(),
            Some(progress),
        )
    }

    pub fn generate_batch_with_runtime_options(
        &self,
        request: DiffusionBatchRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_batch_inner(request, runtime_options, None)
    }

    pub fn generate_batch_with_progress_and_runtime_options(
        &self,
        request: DiffusionBatchRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
        progress: &mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_batch_inner(request, runtime_options, Some(progress))
    }

    fn generate_batch_inner(
        &self,
        request: DiffusionBatchRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        validate_batch_request(&self.metadata, &request)?;
        let runtime = self.native_runtime()?;
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        let plan = self.prepare_run_plan_with_runtime_context(&request, &mut runtime_context)?;
        let positive_embeddings = plan
            .conditioning
            .prompt_cross_attention_embeddings
            .as_ref()
            .or(plan.conditioning.prompt_embeddings.as_ref())
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain usable native text conditioning".to_string(),
                )
            })?;
        let negative_embeddings = plan
            .conditioning
            .negative_cross_attention_embeddings
            .as_ref()
            .or(plan.conditioning.negative_embeddings.as_ref())
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain usable native text conditioning".to_string(),
                )
            })?;
        let _primary_positive_embeddings = plan
            .conditioning
            .prompt_embeddings
            .as_ref()
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain a usable native CLIP text encoder".to_string(),
                )
            })?;
        let _primary_negative_embeddings = plan
            .conditioning
            .negative_embeddings
            .as_ref()
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain a usable native CLIP text encoder".to_string(),
                )
            })?;
        let sdxl_time_ids = sdxl_time_ids_for_request(&request)?;
        let positive_sdxl_conditioning =
            build_sdxl_denoise_conditioning(&plan.conditioning, &sdxl_time_ids, true)?;
        let negative_sdxl_conditioning =
            build_sdxl_denoise_conditioning(&plan.conditioning, &sdxl_time_ids, false)?;
        if is_sdxl_pipeline_class(&self.config.pipeline_class)
            && (positive_sdxl_conditioning.is_none() || negative_sdxl_conditioning.is_none())
        {
            return Err(DiffusionError::BackendUnavailable(
                "SDXL generation requires dual text encoders, pooled text embeddings, and time IDs"
                    .to_string(),
            ));
        }
        let denoise_output = runtime.noise.denoise_latents_with_runtime_context(
            plan.latents,
            &plan.schedule,
            request.cfg_scale,
            positive_embeddings,
            negative_embeddings,
            positive_sdxl_conditioning.as_ref(),
            negative_sdxl_conditioning.as_ref(),
            None,
            None,
            &mut runtime_context,
            progress,
        )?;
        let latents = denoise_output.latents;
        let mut generation_runtime_kind =
            merge_runtime_kind(runtime.kind, denoise_output.runtime_kind);
        let images = if request.send_images {
            let (rgb, image_runtime_kind) = decode_to_rgb8_with_runtime_context(
                runtime.decoder.as_ref(),
                &latents,
                &mut runtime_context,
            )?;
            generation_runtime_kind =
                merge_runtime_kind(generation_runtime_kind, image_runtime_kind);
            encode_rgb_batch_png_base64(&rgb)?
        } else {
            Vec::new()
        };
        Ok(DiffusionBatchOutput {
            images,
            info: diffusion_generation_info(
                self.summary(),
                generation_runtime_kind,
                &request,
                &plan.latent_shape,
            ),
        })
    }

    pub fn generate_img2img_batch(
        &self,
        request: DiffusionImg2ImgRequest,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_img2img_batch_inner(
            request,
            DiffusionGenerationRuntimeOptions::default(),
            None,
        )
    }

    pub fn generate_img2img_batch_with_progress(
        &self,
        request: DiffusionImg2ImgRequest,
        progress: &mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_img2img_batch_inner(
            request,
            DiffusionGenerationRuntimeOptions::default(),
            Some(progress),
        )
    }

    pub fn generate_img2img_batch_with_runtime_options(
        &self,
        request: DiffusionImg2ImgRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_img2img_batch_inner(request, runtime_options, None)
    }

    pub fn generate_img2img_batch_with_progress_and_runtime_options(
        &self,
        request: DiffusionImg2ImgRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
        progress: &mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_img2img_batch_inner(request, runtime_options, Some(progress))
    }

    fn generate_img2img_batch_inner(
        &self,
        request: DiffusionImg2ImgRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        validate_img2img_request(&self.metadata, &request)?;
        let runtime = self.native_runtime()?;
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        let plan =
            self.prepare_run_plan_with_runtime_context(&request.batch, &mut runtime_context)?;
        let positive_embeddings = plan
            .conditioning
            .prompt_cross_attention_embeddings
            .as_ref()
            .or(plan.conditioning.prompt_embeddings.as_ref())
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain usable native text conditioning".to_string(),
                )
            })?;
        let negative_embeddings = plan
            .conditioning
            .negative_cross_attention_embeddings
            .as_ref()
            .or(plan.conditioning.negative_embeddings.as_ref())
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain usable native text conditioning".to_string(),
                )
            })?;
        let _primary_positive_embeddings = plan
            .conditioning
            .prompt_embeddings
            .as_ref()
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain a usable native CLIP text encoder".to_string(),
                )
            })?;
        let _primary_negative_embeddings = plan
            .conditioning
            .negative_embeddings
            .as_ref()
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain a usable native CLIP text encoder".to_string(),
                )
            })?;
        let sdxl_time_ids = sdxl_time_ids_for_request(&request.batch)?;
        let positive_sdxl_conditioning =
            build_sdxl_denoise_conditioning(&plan.conditioning, &sdxl_time_ids, true)?;
        let negative_sdxl_conditioning =
            build_sdxl_denoise_conditioning(&plan.conditioning, &sdxl_time_ids, false)?;
        if is_sdxl_pipeline_class(&self.config.pipeline_class)
            && (positive_sdxl_conditioning.is_none() || negative_sdxl_conditioning.is_none())
        {
            return Err(DiffusionError::BackendUnavailable(
                "SDXL generation requires dual text encoders, pooled text embeddings, and time IDs"
                    .to_string(),
            ));
        }
        let encoder = runtime.encoder.as_ref().ok_or_else(|| {
            DiffusionError::BackendUnavailable(
                "diffusion HFQ does not contain a usable native VAE encoder".to_string(),
            )
        })?;
        let mut generation_runtime_kind = runtime.kind;
        let expanded_init_image =
            expand_rgb_batch_for_prompts(&request.init_image, request.batch.prompts.len())?;
        let init_image = match request.resize_mode {
            DiffusionImg2ImgResizeMode::Image => resize_rgb_batch_nearest(
                &expanded_init_image,
                request.batch.width,
                request.batch.height,
            )?,
            DiffusionImg2ImgResizeMode::Latent => expanded_init_image,
        };
        let (encoded_init_latents, init_encode_kind) =
            encode_to_latents_with_runtime_context(encoder, &init_image, &mut runtime_context)?;
        generation_runtime_kind = merge_runtime_kind(generation_runtime_kind, init_encode_kind);
        let init_latents = match request.resize_mode {
            DiffusionImg2ImgResizeMode::Image => encoded_init_latents,
            DiffusionImg2ImgResizeMode::Latent => resize_latent_batch_nearest(
                &encoded_init_latents,
                plan.latent_shape.height,
                plan.latent_shape.width,
            )?,
        };
        let mut denoise_init_latents = init_latents.clone();
        if denoise_init_latents.batch != plan.latent_shape.batch
            || denoise_init_latents.channels != plan.latent_shape.channels
            || denoise_init_latents.height != plan.latent_shape.height
            || denoise_init_latents.width != plan.latent_shape.width
        {
            return Err(DiffusionError::InvalidRequest(format!(
                "encoded init latent shape [{}x{}x{}x{}] != requested latent shape [{}x{}x{}x{}]",
                denoise_init_latents.batch,
                denoise_init_latents.channels,
                denoise_init_latents.height,
                denoise_init_latents.width,
                plan.latent_shape.batch,
                plan.latent_shape.channels,
                plan.latent_shape.height,
                plan.latent_shape.width
            )));
        }
        let strength = request.denoising_strength.clamp(0.0, 1.0);
        let denoise_steps = ((plan.schedule.timesteps.len() as f32) * strength).ceil() as usize;
        let start_step = plan.schedule.timesteps.len().saturating_sub(denoise_steps);
        let schedule = plan.schedule.slice_from_step(start_step)?;
        let expanded_mask = if let Some(mask) = request.mask.as_ref() {
            let mask = expand_rgb_batch_for_prompts(mask, request.batch.prompts.len())?;
            let target_width = match request.resize_mode {
                DiffusionImg2ImgResizeMode::Image => request.batch.width,
                DiffusionImg2ImgResizeMode::Latent => {
                    u32::try_from(init_image.width).map_err(|_| {
                        DiffusionError::InvalidRequest(
                            "init image width is out of range".to_string(),
                        )
                    })?
                }
            };
            let target_height = match request.resize_mode {
                DiffusionImg2ImgResizeMode::Image => request.batch.height,
                DiffusionImg2ImgResizeMode::Latent => {
                    u32::try_from(init_image.height).map_err(|_| {
                        DiffusionError::InvalidRequest(
                            "init image height is out of range".to_string(),
                        )
                    })?
                }
            };
            Some(resize_rgb_batch_nearest(
                &mask,
                target_width,
                target_height,
            )?)
        } else {
            None
        };
        let mask_weights = if let Some(mask) = expanded_mask.as_ref() {
            let (weights, mask_kind) = latent_mask_weights_with_runtime_context(
                mask,
                &denoise_init_latents,
                &mut runtime_context,
            )?;
            generation_runtime_kind = merge_runtime_kind(generation_runtime_kind, mask_kind);
            Some(weights)
        } else {
            None
        };
        let inpaint_conditioning = if let Some(mask) = expanded_mask.as_ref() {
            let (conditioning, inpaint_kind) = build_inpaint_conditioning_if_supported(
                runtime.noise.as_ref(),
                encoder,
                &init_image,
                mask,
                &denoise_init_latents,
                mask_weights.as_deref(),
                &mut runtime_context,
            )?;
            generation_runtime_kind = merge_runtime_kind(generation_runtime_kind, inpaint_kind);
            conditioning
        } else {
            None
        };
        let inpainting_fill = request.inpainting_fill.unwrap_or(0);
        let applied_inpainting_fill = if let Some(mask_weights) = mask_weights.as_ref() {
            apply_inpainting_fill_to_latents(
                &mut denoise_init_latents,
                &plan.latents,
                mask_weights,
                inpainting_fill,
            )?
        } else {
            false
        };
        let mut latents = denoise_init_latents.clone();
        if !schedule.timesteps.is_empty() {
            let noise = plan.latents;
            plan.schedule
                .add_noise_to_latents(&mut latents, &noise.data, start_step)?;
            let masked_reference =
                mask_weights
                    .as_ref()
                    .map(|mask_weights| MaskedDenoiseReference {
                        init_latents: &denoise_init_latents,
                        noise: &noise.data,
                        mask_weights,
                        source_schedule: &plan.schedule,
                        start_step,
                    });
            let denoise_output = runtime.noise.denoise_latents_with_runtime_context(
                latents,
                &schedule,
                request.batch.cfg_scale,
                positive_embeddings,
                negative_embeddings,
                positive_sdxl_conditioning.as_ref(),
                negative_sdxl_conditioning.as_ref(),
                inpaint_conditioning.as_ref(),
                masked_reference.as_ref(),
                &mut runtime_context,
                progress,
            )?;
            latents = denoise_output.latents;
            generation_runtime_kind =
                merge_runtime_kind(generation_runtime_kind, denoise_output.runtime_kind);
        }
        let masked = if let Some(mask_weights) = mask_weights.as_ref() {
            let blend_kind = blend_latents_with_mask_with_runtime_context(
                &mut latents,
                &init_latents,
                mask_weights,
                &mut runtime_context,
            )?;
            generation_runtime_kind = merge_runtime_kind(generation_runtime_kind, blend_kind);
            true
        } else {
            false
        };
        let images = if request.batch.send_images {
            let (rgb, image_runtime_kind) = decode_to_rgb8_with_runtime_context(
                runtime.decoder.as_ref(),
                &latents,
                &mut runtime_context,
            )?;
            generation_runtime_kind =
                merge_runtime_kind(generation_runtime_kind, image_runtime_kind);
            encode_rgb_batch_png_base64(&rgb)?
        } else {
            Vec::new()
        };
        let mut info = diffusion_generation_info(
            self.summary(),
            generation_runtime_kind,
            &request.batch,
            &plan.latent_shape,
        );
        if let Value::Object(map) = &mut info {
            map.insert("mode".to_string(), Value::String("img2img".to_string()));
            map.insert(
                "denoising_strength".to_string(),
                json!(request.denoising_strength),
            );
            map.insert("start_step".to_string(), json!(start_step));
            map.insert("denoise_steps".to_string(), json!(schedule.timesteps.len()));
            map.insert("masked".to_string(), json!(masked));
            if request.resize_mode == DiffusionImg2ImgResizeMode::Latent {
                map.insert(
                    "resize_mode".to_string(),
                    Value::String("latent".to_string()),
                );
                map.insert("latent_resize".to_string(), json!(true));
            }
            if request.inpainting_fill.is_some() || applied_inpainting_fill {
                map.insert("inpainting_fill".to_string(), json!(inpainting_fill));
            }
            if applied_inpainting_fill {
                let masked_content = match inpainting_fill {
                    2 => "latent noise",
                    3 => "latent nothing",
                    _ => unreachable!("only latent inpaint fill modes are applied"),
                };
                map.insert(
                    "masked_content".to_string(),
                    Value::String(masked_content.to_string()),
                );
            }
        }
        Ok(DiffusionBatchOutput { images, info })
    }

    fn native_runtime(&self) -> DiffusionResult<&NativeDiffusionRuntime> {
        self.native_runtime.as_ref().ok_or_else(|| {
            DiffusionError::BackendUnavailable(self.native_runtime_error.clone().unwrap_or_else(
                || {
                    "native UNet/VAE runtime is not available for this diffusion HFQ artifact"
                        .to_string()
                },
            ))
        })
    }

    pub fn decode_preview_latents_png_base64_with_runtime_options(
        &self,
        latents: &LatentBatch,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<String> {
        let runtime = self.native_runtime()?;
        let (rgb, _) = decode_to_rgb8_with_runtime_options(
            runtime.decoder.as_ref(),
            latents,
            runtime_options,
        )?;
        encode_rgb_batch_png_base64(&rgb)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                DiffusionError::InvalidRequest(
                    "preview latent batch did not decode any images".to_string(),
                )
            })
    }

    pub fn prepare_run_plan(
        &self,
        request: &DiffusionBatchRequest,
    ) -> DiffusionResult<DiffusionRunPlan> {
        self.prepare_run_plan_with_runtime_options(
            request,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn prepare_run_plan_with_runtime_options(
        &self,
        request: &DiffusionBatchRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<DiffusionRunPlan> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.prepare_run_plan_with_runtime_context(request, &mut runtime_context)
    }

    fn prepare_run_plan_with_runtime_context(
        &self,
        request: &DiffusionBatchRequest,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<DiffusionRunPlan> {
        let conditioning =
            self.prepare_conditioning_batch_with_runtime_context(request, runtime_context)?;
        let latent_shape = latent_shape_for_request(&self.config, request)?;
        let seeds = request
            .prompts
            .iter()
            .map(|prompt| prompt.seed)
            .collect::<Vec<_>>();
        let mut latents = seeded_latents_for_request(&self.config, request, &latent_shape, &seeds)?;
        blend_subseed_latents(&self.config, &mut latents, request, &latent_shape)?;
        let scheduler_config = self
            .config
            .scheduler
            .resolve_request_scheduler(&request.scheduler)?;
        let schedule = DiffusionSchedule::from_config(&scheduler_config, request.steps)?;
        schedule.scale_initial_latents(&mut latents);
        Ok(DiffusionRunPlan {
            latent_shape,
            latents,
            schedule,
            conditioning,
        })
    }

    pub fn prepare_conditioning_batch(
        &self,
        request: &DiffusionBatchRequest,
    ) -> DiffusionResult<DiffusionConditioningBatch> {
        self.prepare_conditioning_batch_with_runtime_options(
            request,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn prepare_conditioning_batch_with_runtime_options(
        &self,
        request: &DiffusionBatchRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<DiffusionConditioningBatch> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.prepare_conditioning_batch_with_runtime_context(request, &mut runtime_context)
    }

    fn prepare_conditioning_batch_with_runtime_context(
        &self,
        request: &DiffusionBatchRequest,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<DiffusionConditioningBatch> {
        validate_batch_request(&self.metadata, request)?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            DiffusionError::BackendUnavailable(
                "diffusion HFQ does not contain a usable CLIP tokenizer".to_string(),
            )
        })?;
        let prompt_tokens = request
            .prompts
            .iter()
            .map(|prompt| tokenizer.encode_padded(&prompt.prompt))
            .collect::<Vec<_>>();
        let negative_tokens = request
            .prompts
            .iter()
            .map(|prompt| tokenizer.encode_padded(&prompt.negative_prompt))
            .collect::<Vec<_>>();
        let (prompt_tokens_2, negative_tokens_2) =
            if let Some(tokenizer_2) = self.tokenizer_2.as_ref() {
                (
                    Some(
                        request
                            .prompts
                            .iter()
                            .map(|prompt| tokenizer_2.encode_padded(&prompt.prompt))
                            .collect::<Vec<_>>(),
                    ),
                    Some(
                        request
                            .prompts
                            .iter()
                            .map(|prompt| tokenizer_2.encode_padded(&prompt.negative_prompt))
                            .collect::<Vec<_>>(),
                    ),
                )
            } else {
                (None, None)
            };
        let (prompt_embeddings, negative_embeddings) =
            if let Some(text_encoder) = self.text_encoder.as_ref() {
                (
                    Some(encode_token_batch_with_runtime_context(
                        text_encoder,
                        &prompt_tokens,
                        runtime_context,
                    )?),
                    Some(encode_token_batch_with_runtime_context(
                        text_encoder,
                        &negative_tokens,
                        runtime_context,
                    )?),
                )
            } else {
                (None, None)
            };
        let (
            prompt_embeddings_2,
            negative_embeddings_2,
            prompt_cross_attention_embeddings,
            negative_cross_attention_embeddings,
            prompt_pooled_embeddings,
            negative_pooled_embeddings,
        ) = if let (
            Some(text_encoder_2),
            Some(tokenizer_2),
            Some(prompt_tokens_2),
            Some(negative_tokens_2),
        ) = (
            self.text_encoder_2.as_ref(),
            self.tokenizer_2.as_ref(),
            prompt_tokens_2.as_ref(),
            negative_tokens_2.as_ref(),
        ) {
            let (prompt_embeddings_2, prompt_pooled_embeddings) =
                encode_token_batch_with_pooled_and_runtime_context(
                    text_encoder_2,
                    prompt_tokens_2,
                    tokenizer_2.end_token_id(),
                    runtime_context,
                )?;
            let (negative_embeddings_2, negative_pooled_embeddings) =
                encode_token_batch_with_pooled_and_runtime_context(
                    text_encoder_2,
                    negative_tokens_2,
                    tokenizer_2.end_token_id(),
                    runtime_context,
                )?;
            let prompt_cross_attention_embeddings = prompt_embeddings
                .as_ref()
                .map(|prompt_embeddings| {
                    concat_last_dim_3d_with_runtime_context(
                        prompt_embeddings,
                        &prompt_embeddings_2,
                        runtime_context,
                    )
                })
                .transpose()?;
            let negative_cross_attention_embeddings = negative_embeddings
                .as_ref()
                .map(|negative_embeddings| {
                    concat_last_dim_3d_with_runtime_context(
                        negative_embeddings,
                        &negative_embeddings_2,
                        runtime_context,
                    )
                })
                .transpose()?;
            (
                Some(prompt_embeddings_2),
                Some(negative_embeddings_2),
                prompt_cross_attention_embeddings,
                negative_cross_attention_embeddings,
                Some(prompt_pooled_embeddings),
                Some(negative_pooled_embeddings),
            )
        } else {
            (None, None, None, None, None, None)
        };
        Ok(DiffusionConditioningBatch {
            prompt_tokens,
            negative_tokens,
            prompt_tokens_2,
            negative_tokens_2,
            prompt_embeddings,
            negative_embeddings,
            prompt_embeddings_2,
            negative_embeddings_2,
            prompt_cross_attention_embeddings,
            negative_cross_attention_embeddings,
            prompt_pooled_embeddings,
            negative_pooled_embeddings,
        })
    }
}

trait DiffusionNoiseBackend: Send + Sync {
    fn model_input_channels(&self) -> usize;

    fn denoise_latents_with_runtime_context(
        &self,
        latents: LatentBatch,
        schedule: &DiffusionSchedule,
        cfg_scale: f32,
        positive_embeddings: &CpuTensor,
        negative_embeddings: &CpuTensor,
        positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
        masked_reference: Option<&MaskedDenoiseReference<'_>>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<DenoiseLatentsOutput>;
}

impl DiffusionNoiseBackend for NativeUnet2DConditionModel {
    fn model_input_channels(&self) -> usize {
        self.input_channels()
    }

    fn denoise_latents_with_runtime_context(
        &self,
        latents: LatentBatch,
        schedule: &DiffusionSchedule,
        cfg_scale: f32,
        positive_embeddings: &CpuTensor,
        negative_embeddings: &CpuTensor,
        positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
        masked_reference: Option<&MaskedDenoiseReference<'_>>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<DenoiseLatentsOutput> {
        NativeUnet2DConditionModel::denoise_latents_with_runtime_context(
            self,
            latents,
            schedule,
            cfg_scale,
            positive_embeddings,
            negative_embeddings,
            positive_sdxl_conditioning,
            negative_sdxl_conditioning,
            inpaint_conditioning,
            masked_reference,
            runtime_context,
            progress,
        )
    }
}

trait DiffusionImageDecoder: Send + Sync {
    fn decode_to_rgb_tensor(&self, latents: &LatentBatch) -> DiffusionResult<CpuTensor>;

    fn decode_to_rgb_tensor_with_runtime_context(
        &self,
        latents: &LatentBatch,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let _ = runtime_context;
        self.decode_to_rgb_tensor(latents)
    }
}

impl DiffusionImageDecoder for NativeVaeDecoder {
    fn decode_to_rgb_tensor(&self, latents: &LatentBatch) -> DiffusionResult<CpuTensor> {
        NativeVaeDecoder::decode_latents(self, latents)
    }

    fn decode_to_rgb_tensor_with_runtime_context(
        &self,
        latents: &LatentBatch,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        NativeVaeDecoder::decode_latents_with_runtime_context(self, latents, runtime_context)
    }
}

fn decode_to_rgb8_with_runtime_options(
    decoder: &dyn DiffusionImageDecoder,
    latents: &LatentBatch,
    runtime_options: DiffusionGenerationRuntimeOptions,
) -> DiffusionResult<(RgbImageBatch, DiffusionRuntimeKind)> {
    let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
    decode_to_rgb8_with_runtime_context(decoder, latents, &mut runtime_context)
}

fn decode_to_rgb8_with_runtime_context(
    decoder: &dyn DiffusionImageDecoder,
    latents: &LatentBatch,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<(RgbImageBatch, DiffusionRuntimeKind)> {
    let decoded = decoder.decode_to_rgb_tensor_with_runtime_context(latents, runtime_context)?;
    let rgb = rgb_tensor_to_u8_with_runtime_context(&decoded, runtime_context)?;
    Ok((rgb, runtime_kind_for_context(runtime_context)))
}

fn encode_to_latents_with_runtime_context(
    encoder: &NativeVaeEncoder,
    image: &RgbImageBatch,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<(LatentBatch, DiffusionRuntimeKind)> {
    let latents = encoder.encode_to_latents_with_runtime_context(image, runtime_context)?;
    Ok((latents, runtime_kind_for_context(runtime_context)))
}

fn latent_mask_weights_with_runtime_context(
    mask: &RgbImageBatch,
    latents: &LatentBatch,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<(Vec<f32>, DiffusionRuntimeKind)> {
    if let Some(_device_id) = runtime_context.rocm_device_id() {
        #[cfg(feature = "rocm")]
        {
            let weights = runtime_context.with_rocm_gpu(|gpu| {
                latent_mask_weights_from_rgb_batch_hip_on_gpu(gpu, mask, latents)
            })?;
            return Ok((weights, DiffusionRuntimeKind::RocmHybridReference));
        }
        #[cfg(not(feature = "rocm"))]
        {
            return Err(rocm_hybrid_unavailable_error());
        }
    }
    Ok((
        latent_mask_weights_from_rgb_batch(mask, latents)?,
        DiffusionRuntimeKind::CpuSourceReference,
    ))
}

fn masked_rgb_batch_for_inpaint_with_runtime_context(
    image: &RgbImageBatch,
    mask: &RgbImageBatch,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<(RgbImageBatch, DiffusionRuntimeKind)> {
    if let Some(_device_id) = runtime_context.rocm_device_id() {
        #[cfg(feature = "rocm")]
        {
            let masked = runtime_context
                .with_rocm_gpu(|gpu| masked_rgb_batch_for_inpaint_hip_on_gpu(gpu, image, mask))?;
            return Ok((masked, DiffusionRuntimeKind::RocmHybridReference));
        }
        #[cfg(not(feature = "rocm"))]
        {
            return Err(rocm_hybrid_unavailable_error());
        }
    }
    Ok((
        masked_rgb_batch_for_inpaint(image, mask)?,
        DiffusionRuntimeKind::CpuSourceReference,
    ))
}

fn blend_latents_with_mask_with_runtime_context(
    generated: &mut LatentBatch,
    init: &LatentBatch,
    mask_weights: &[f32],
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<DiffusionRuntimeKind> {
    if let Some(_device_id) = runtime_context.rocm_device_id() {
        #[cfg(feature = "rocm")]
        {
            *generated = runtime_context.with_rocm_gpu(|gpu| {
                blend_latents_with_mask_hip_on_gpu(gpu, generated, init, mask_weights)
            })?;
            return Ok(DiffusionRuntimeKind::RocmHybridReference);
        }
        #[cfg(not(feature = "rocm"))]
        {
            return Err(rocm_hybrid_unavailable_error());
        }
    }
    blend_latents_with_mask(generated, init, mask_weights)?;
    Ok(DiffusionRuntimeKind::CpuSourceReference)
}

fn merge_runtime_kind(
    current: DiffusionRuntimeKind,
    observed: DiffusionRuntimeKind,
) -> DiffusionRuntimeKind {
    if current == DiffusionRuntimeKind::RocmHybridReference
        || observed == DiffusionRuntimeKind::RocmHybridReference
    {
        DiffusionRuntimeKind::RocmHybridReference
    } else {
        DiffusionRuntimeKind::CpuSourceReference
    }
}

struct NativeDiffusionRuntime {
    kind: DiffusionRuntimeKind,
    noise: Box<dyn DiffusionNoiseBackend>,
    encoder: Option<NativeVaeEncoder>,
    decoder: Box<dyn DiffusionImageDecoder>,
}

impl NativeDiffusionRuntime {
    fn from_hfq(hfq: &HfqFile, config: &StableDiffusionConfig) -> DiffusionResult<Self> {
        Ok(Self {
            kind: DiffusionRuntimeKind::CpuSourceReference,
            noise: Box::new(NativeUnet2DConditionModel::from_hfq(hfq, &config.unet)?),
            encoder: NativeVaeEncoder::from_hfq(hfq, &config.vae).ok(),
            decoder: Box::new(NativeVaeDecoder::from_hfq(hfq, &config.vae)?),
        })
    }
}

fn diffusion_generation_info(
    summary: &DiffusionModelSummary,
    runtime_kind: DiffusionRuntimeKind,
    request: &DiffusionBatchRequest,
    latent_shape: &DiffusionLatentShape,
) -> Value {
    json!({
        "compat": "stable-diffusion-webui",
        "backend": "hipfire-diffusion-hfq",
        "runtime": runtime_kind.as_str(),
        "model": summary.model_name,
        "pipeline": summary.pipeline_class,
        "weight_format": summary.weight_format,
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "cfg_scale": request.cfg_scale,
        "scheduler": request.scheduler,
        "batch_size": request.prompts.len(),
        "seeds": request.prompts.iter().map(|prompt| prompt.seed).collect::<Vec<_>>(),
        "subseeds": request.prompts.iter().map(|prompt| prompt.subseed).collect::<Vec<_>>(),
        "subseed_strength": request.subseed_strength,
        "seed_resize_from_w": request.seed_resize_from_width,
        "seed_resize_from_h": request.seed_resize_from_height,
        "latent_shape": {
            "batch": latent_shape.batch,
            "channels": latent_shape.channels,
            "height": latent_shape.height,
            "width": latent_shape.width,
        },
    })
}

impl StableDiffusionConfig {
    pub fn from_hfq(hfq: &HfqFile, metadata: &DiffusionHfqMetadata) -> DiffusionResult<Self> {
        let text_json = component_json(hfq, metadata, "text_encoder")?.unwrap_or_else(|| json!({}));
        let text_2_json = component_json(hfq, metadata, "text_encoder_2")?;
        let unet_json = component_json(hfq, metadata, "unet")?.unwrap_or_else(|| json!({}));
        let vae_json = component_json(hfq, metadata, "vae")?.unwrap_or_else(|| json!({}));
        let scheduler_json =
            component_json(hfq, metadata, "scheduler")?.unwrap_or_else(|| json!({}));

        let text_encoder = TextEncoderConfig {
            class_name: json_string(&text_json, "_class_name"),
            hidden_size: json_usize(&text_json, "hidden_size"),
            intermediate_size: json_usize(&text_json, "intermediate_size"),
            num_hidden_layers: json_usize(&text_json, "num_hidden_layers"),
            num_attention_heads: json_usize(&text_json, "num_attention_heads"),
            max_position_embeddings: json_usize(&text_json, "max_position_embeddings")
                .or(metadata.tokenizer.max_length.map(|value| value as usize)),
            vocab_size: json_usize(&text_json, "vocab_size"),
        };
        let text_encoder_2 = text_2_json.as_ref().map(|text_json| TextEncoderConfig {
            class_name: json_string(text_json, "_class_name"),
            hidden_size: json_usize(text_json, "hidden_size"),
            intermediate_size: json_usize(text_json, "intermediate_size"),
            num_hidden_layers: json_usize(text_json, "num_hidden_layers"),
            num_attention_heads: json_usize(text_json, "num_attention_heads"),
            max_position_embeddings: json_usize(text_json, "max_position_embeddings").or(metadata
                .tokenizer_2
                .as_ref()
                .and_then(|tokenizer| tokenizer.max_length)
                .map(|value| value as usize)),
            vocab_size: json_usize(text_json, "vocab_size"),
        });
        let unet = UnetConfig {
            class_name: json_string(&unet_json, "_class_name"),
            sample_size: json_usize(&unet_json, "sample_size"),
            in_channels: json_usize(&unet_json, "in_channels"),
            out_channels: json_usize(&unet_json, "out_channels"),
            cross_attention_dim: json_usize(&unet_json, "cross_attention_dim"),
            attention_head_dim: json_usize_vec(&unet_json, "attention_head_dim"),
            block_out_channels: json_usize_vec(&unet_json, "block_out_channels"),
            down_block_types: json_string_vec(&unet_json, "down_block_types"),
            up_block_types: json_string_vec(&unet_json, "up_block_types"),
            layers_per_block: json_usize(&unet_json, "layers_per_block"),
            norm_num_groups: json_usize(&unet_json, "norm_num_groups"),
            norm_eps: json_f32(&unet_json, "norm_eps"),
            center_input_sample: json_bool(&unet_json, "center_input_sample").unwrap_or(false),
            flip_sin_to_cos: json_bool(&unet_json, "flip_sin_to_cos").unwrap_or(false),
            freq_shift: json_f32(&unet_json, "freq_shift").unwrap_or(0.0),
            addition_embed_type: json_optional_string(&unet_json, "addition_embed_type"),
            addition_time_embed_dim: json_usize(&unet_json, "addition_time_embed_dim"),
            projection_class_embeddings_input_dim: json_usize(
                &unet_json,
                "projection_class_embeddings_input_dim",
            ),
        };
        let vae = VaeConfig {
            class_name: json_string(&vae_json, "_class_name"),
            latent_channels: json_usize(&vae_json, "latent_channels"),
            z_dim: json_usize(&vae_json, "z_dim"),
            scaling_factor: json_f32(&vae_json, "scaling_factor"),
            latents_mean: json_f32_vec(&vae_json, "latents_mean"),
            latents_std: json_f32_vec(&vae_json, "latents_std"),
            block_out_channels: json_usize_vec(&vae_json, "block_out_channels"),
            down_block_types: json_string_vec(&vae_json, "down_block_types"),
            up_block_types: json_string_vec(&vae_json, "up_block_types"),
            norm_num_groups: json_usize(&vae_json, "norm_num_groups"),
            norm_eps: json_f32(&vae_json, "norm_eps"),
        };
        let scheduler = SchedulerConfig {
            class_name: json_string(&scheduler_json, "_class_name"),
            beta_start: json_f32(&scheduler_json, "beta_start"),
            beta_end: json_f32(&scheduler_json, "beta_end"),
            beta_schedule: json_optional_string(&scheduler_json, "beta_schedule"),
            num_train_timesteps: json_usize(&scheduler_json, "num_train_timesteps"),
            prediction_type: json_optional_string(&scheduler_json, "prediction_type"),
            algorithm_type: json_optional_string(&scheduler_json, "algorithm_type"),
            solver_order: json_usize(&scheduler_json, "solver_order"),
            solver_type: json_optional_string(&scheduler_json, "solver_type"),
            lower_order_final: json_bool(&scheduler_json, "lower_order_final"),
            thresholding: json_bool(&scheduler_json, "thresholding"),
            dynamic_thresholding_ratio: json_f32(&scheduler_json, "dynamic_thresholding_ratio"),
            sample_max_value: json_f32(&scheduler_json, "sample_max_value"),
            timestep_spacing: json_optional_string(&scheduler_json, "timestep_spacing"),
            steps_offset: json_i32(&scheduler_json, "steps_offset"),
            use_karras_sigmas: json_bool(&scheduler_json, "use_karras_sigmas"),
            set_alpha_to_one: json_bool(&scheduler_json, "set_alpha_to_one"),
            shift: json_f32(&scheduler_json, "shift"),
            shift_terminal: json_f32(&scheduler_json, "shift_terminal"),
            invert_sigmas: json_bool(&scheduler_json, "invert_sigmas"),
            use_dynamic_shifting: json_bool(&scheduler_json, "use_dynamic_shifting"),
            time_shift_type: json_optional_string(&scheduler_json, "time_shift_type"),
        };
        let latent_channels = metadata
            .pipeline
            .latent_channels
            .map(|value| value as usize)
            .or(unet.in_channels)
            .or(vae.latent_channels)
            .or(vae.z_dim)
            .unwrap_or(4);
        let latent_height = metadata
            .pipeline
            .latent_height
            .map(|value| value as usize)
            .or(unet.sample_size);
        let latent_width = metadata
            .pipeline
            .latent_width
            .map(|value| value as usize)
            .or(unet.sample_size);
        let vae_scale_factor = vae
            .block_out_channels
            .len()
            .checked_sub(1)
            .map(|power| 1usize << power)
            .unwrap_or(8);
        Ok(Self {
            pipeline_class: metadata.pipeline.class_name.clone(),
            text_encoder,
            text_encoder_2,
            unet,
            vae,
            scheduler,
            latent_channels,
            latent_height,
            latent_width,
            vae_scale_factor,
        })
    }
}

fn encode_token_batch_with_runtime_context(
    text_encoder: &ClipTextEncoder,
    token_batch: &[Vec<u32>],
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let mut encoded = Vec::new();
    let mut shape = None;
    for tokens in token_batch {
        let tensor = text_encoder.encode_tokens_with_runtime_context(tokens, runtime_context)?;
        let [seq, hidden] = shape2(&tensor)?;
        if let Some((expected_seq, expected_hidden)) = shape {
            if (seq, hidden) != (expected_seq, expected_hidden) {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "CLIP batch embedding shape mismatch [{seq}, {hidden}] vs [{expected_seq}, {expected_hidden}]"
                )));
            }
        } else {
            shape = Some((seq, hidden));
        }
        encoded.extend_from_slice(&tensor.data);
    }
    let (seq, hidden) = shape.unwrap_or((0, 0));
    Ok(CpuTensor {
        shape: vec![token_batch.len(), seq, hidden],
        data: encoded,
    })
}

fn encode_token_batch_with_pooled_and_runtime_context(
    text_encoder: &ClipTextEncoder,
    token_batch: &[Vec<u32>],
    end_token: u32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<(CpuTensor, CpuTensor)> {
    let mut encoded = Vec::new();
    let mut pooled = Vec::new();
    let mut hidden_shape = None;
    let mut pooled_width = None;
    for tokens in token_batch {
        let (hidden_states, pooled_embedding) = text_encoder
            .encode_tokens_with_pooled_and_runtime_context(tokens, end_token, runtime_context)?;
        let [seq, hidden] = shape2(&hidden_states)?;
        if let Some((expected_seq, expected_hidden)) = hidden_shape {
            if (seq, hidden) != (expected_seq, expected_hidden) {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "CLIP batch embedding shape mismatch [{seq}, {hidden}] vs [{expected_seq}, {expected_hidden}]"
                )));
            }
        } else {
            hidden_shape = Some((seq, hidden));
        }
        let pooled_embedding = pooled_embedding.ok_or_else(|| {
            DiffusionError::InvalidMetadata("CLIP pooled embedding is missing".to_string())
        })?;
        if let Some(expected_width) = pooled_width {
            if pooled_embedding.len() != expected_width {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "CLIP pooled embedding width {} != expected {expected_width}",
                    pooled_embedding.len()
                )));
            }
        } else {
            pooled_width = Some(pooled_embedding.len());
        }
        encoded.extend(hidden_states.data);
        pooled.extend(pooled_embedding);
    }
    let (seq, hidden) = hidden_shape.unwrap_or((0, 0));
    let pooled_width = pooled_width.unwrap_or(0);
    Ok((
        CpuTensor {
            shape: vec![token_batch.len(), seq, hidden],
            data: encoded,
        },
        CpuTensor {
            shape: vec![token_batch.len(), pooled_width],
            data: pooled,
        },
    ))
}

fn concat_last_dim_3d(a: &CpuTensor, b: &CpuTensor) -> DiffusionResult<CpuTensor> {
    let [batch, seq, a_width] = shape3(a)?;
    let [b_batch, b_seq, b_width] = shape3(b)?;
    if batch != b_batch || seq != b_seq {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate 3-D tensors with shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let out_width = a_width + b_width;
    let mut out = CpuTensor::zeros(&[batch, seq, out_width]);
    for b_idx in 0..batch {
        for s in 0..seq {
            let a_base = (b_idx * seq + s) * a_width;
            let b_base = (b_idx * seq + s) * b_width;
            let out_base = (b_idx * seq + s) * out_width;
            out.data[out_base..out_base + a_width]
                .copy_from_slice(&a.data[a_base..a_base + a_width]);
            out.data[out_base + a_width..out_base + out_width]
                .copy_from_slice(&b.data[b_base..b_base + b_width]);
        }
    }
    Ok(out)
}

fn concat_last_dim_2d(a: &CpuTensor, b: &CpuTensor) -> DiffusionResult<CpuTensor> {
    let [batch, a_width] = shape2(a)?;
    let [b_batch, b_width] = shape2(b)?;
    if batch != b_batch {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate 2-D tensors with shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let out_width = a_width + b_width;
    let mut out = CpuTensor::zeros(&[batch, out_width]);
    for row in 0..batch {
        let a_base = row * a_width;
        let b_base = row * b_width;
        let out_base = row * out_width;
        out.data[out_base..out_base + a_width].copy_from_slice(&a.data[a_base..a_base + a_width]);
        out.data[out_base + a_width..out_base + out_width]
            .copy_from_slice(&b.data[b_base..b_base + b_width]);
    }
    Ok(out)
}

pub fn latent_shape_for_request(
    config: &StableDiffusionConfig,
    request: &DiffusionBatchRequest,
) -> DiffusionResult<DiffusionLatentShape> {
    let scale = config.vae_scale_factor.max(1) as u32;
    if request.width % scale != 0 || request.height % scale != 0 {
        return Err(DiffusionError::InvalidRequest(format!(
            "width/height {}x{} must be divisible by VAE scale factor {scale}",
            request.width, request.height
        )));
    }
    Ok(DiffusionLatentShape {
        batch: request.prompts.len(),
        channels: config.latent_channels,
        height: (request.height / scale) as usize,
        width: (request.width / scale) as usize,
    })
}

pub fn diffusion_hip_memory_plan(
    config: &StableDiffusionConfig,
    request: &DiffusionBatchRequest,
) -> DiffusionResult<DiffusionHipMemoryPlan> {
    let latent_shape = latent_shape_for_request(config, request)?;
    let latent_elements = checked_shape_elements(
        "latent",
        &[
            latent_shape.batch,
            latent_shape.channels,
            latent_shape.height,
            latent_shape.width,
        ],
    )?;
    let latent_bytes = checked_bytes("latent", latent_elements, 4)?;
    let denoise_channels = config
        .unet
        .in_channels
        .unwrap_or(config.latent_channels)
        .max(config.latent_channels);
    let denoise_elements = checked_shape_elements(
        "denoise input",
        &[
            latent_shape.batch,
            denoise_channels,
            latent_shape.height,
            latent_shape.width,
        ],
    )?;
    let denoise_input_bytes = checked_bytes("denoise input", denoise_elements, 4)?;
    let max_position_embeddings = config
        .text_encoder
        .max_position_embeddings
        .unwrap_or(77)
        .max(1);
    let cross_attention_dim = config
        .unet
        .cross_attention_dim
        .or(config.text_encoder.hidden_size)
        .unwrap_or(768)
        .max(1);
    let text_encoder_count = if config.text_encoder_2.is_some() {
        2
    } else {
        1
    };
    let conditioning_elements = checked_shape_elements(
        "conditioning",
        &[
            latent_shape.batch,
            2,
            text_encoder_count,
            max_position_embeddings,
            cross_attention_dim,
        ],
    )?;
    let conditioning_bytes = checked_bytes("conditioning", conditioning_elements, 4)?;
    let vae_decode_bytes = latent_bytes;
    let rgb_elements = checked_shape_elements(
        "rgb",
        &[
            latent_shape.batch,
            request.height as usize,
            request.width as usize,
            3,
        ],
    )?;
    let rgb_bytes = checked_bytes("rgb", rgb_elements, 1)?;
    let scheduler_scratch_bytes = latent_bytes
        .checked_add(denoise_input_bytes)
        .ok_or_else(|| DiffusionError::InvalidRequest("scheduler scratch bytes overflow".into()))?;
    let total_device_bytes = [
        latent_bytes,
        denoise_input_bytes,
        conditioning_bytes,
        vae_decode_bytes,
        rgb_bytes,
        scheduler_scratch_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |acc, bytes| {
        acc.checked_add(bytes).ok_or_else(|| {
            DiffusionError::InvalidRequest("HIP diffusion memory plan overflow".into())
        })
    })?;
    Ok(DiffusionHipMemoryPlan {
        latent_shape,
        latent_bytes,
        denoise_input_bytes,
        conditioning_bytes,
        vae_decode_bytes,
        rgb_bytes,
        scheduler_scratch_bytes,
        total_device_bytes,
    })
}

fn checked_shape_elements(label: &str, dims: &[usize]) -> DiffusionResult<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("{label} shape element count overflows"))
        })
    })
}

fn checked_bytes(label: &str, elements: usize, element_bytes: usize) -> DiffusionResult<usize> {
    elements
        .checked_mul(element_bytes)
        .ok_or_else(|| DiffusionError::InvalidRequest(format!("{label} byte size overflows")))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Conv2dLayer {
    pub weight: CpuTensor,
    pub bias: Option<CpuTensor>,
    pub padding: usize,
    pub stride: usize,
}

impl Conv2dLayer {
    pub fn from_hfq(
        hfq: &HfqFile,
        weight_entry: &str,
        bias_entry: Option<&str>,
        padding: usize,
    ) -> DiffusionResult<Self> {
        let weight = CpuTensor::from_hfq(hfq, weight_entry)?;
        let bias = bias_entry
            .map(|entry| CpuTensor::from_hfq(hfq, entry))
            .transpose()?;
        Ok(Self {
            weight,
            bias,
            padding,
            stride: 1,
        })
    }

    pub fn from_hfq_with_stride(
        hfq: &HfqFile,
        weight_entry: &str,
        bias_entry: Option<&str>,
        padding: usize,
        stride: usize,
    ) -> DiffusionResult<Self> {
        let mut layer = Self::from_hfq(hfq, weight_entry, bias_entry, padding)?;
        if stride == 0 {
            return Err(DiffusionError::InvalidMetadata(
                "conv2d stride must be positive".to_string(),
            ));
        }
        layer.stride = stride;
        Ok(layer)
    }

    pub fn forward(&self, input: &CpuTensor) -> DiffusionResult<CpuTensor> {
        conv2d_nchw_with_stride(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.padding,
            self.stride,
        )
    }

    fn forward_with_runtime_context(
        &self,
        input: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        conv2d_with_runtime_context(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.padding,
            self.stride,
            runtime_context,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupNormLayer {
    pub weight: CpuTensor,
    pub bias: CpuTensor,
    pub groups: usize,
    pub eps: f32,
}

impl GroupNormLayer {
    pub fn from_hfq(
        hfq: &HfqFile,
        weight_entry: &str,
        bias_entry: &str,
        groups: usize,
        eps: f32,
    ) -> DiffusionResult<Self> {
        Ok(Self {
            weight: CpuTensor::from_hfq(hfq, weight_entry)?,
            bias: CpuTensor::from_hfq(hfq, bias_entry)?,
            groups,
            eps,
        })
    }

    pub fn forward(&self, input: &CpuTensor) -> DiffusionResult<CpuTensor> {
        group_norm_nchw(input, &self.weight, &self.bias, self.groups, self.eps)
    }

    fn forward_with_runtime_context(
        &self,
        input: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        group_norm_with_runtime_context(
            input,
            &self.weight,
            &self.bias,
            self.groups,
            self.eps,
            runtime_context,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResnetBlock2D {
    pub norm1: GroupNormLayer,
    pub conv1: Conv2dLayer,
    pub norm2: GroupNormLayer,
    pub conv2: Conv2dLayer,
    pub shortcut: Option<Conv2dLayer>,
}

impl ResnetBlock2D {
    pub fn from_hfq(hfq: &HfqFile, prefix: &str, groups: usize) -> DiffusionResult<Self> {
        let norm1 = GroupNormLayer::from_hfq(
            hfq,
            &format!("{prefix}.norm1.weight"),
            &format!("{prefix}.norm1.bias"),
            groups,
            1e-6,
        )?;
        let conv1 = Conv2dLayer::from_hfq(
            hfq,
            &format!("{prefix}.conv1.weight"),
            Some(&format!("{prefix}.conv1.bias")),
            1,
        )?;
        let norm2 = GroupNormLayer::from_hfq(
            hfq,
            &format!("{prefix}.norm2.weight"),
            &format!("{prefix}.norm2.bias"),
            groups,
            1e-6,
        )?;
        let conv2 = Conv2dLayer::from_hfq(
            hfq,
            &format!("{prefix}.conv2.weight"),
            Some(&format!("{prefix}.conv2.bias")),
            1,
        )?;
        let shortcut_weight = format!("{prefix}.conv_shortcut.weight");
        let shortcut_bias = format!("{prefix}.conv_shortcut.bias");
        let shortcut = if hfq.find_tensor_info(&shortcut_weight).is_some() {
            Some(Conv2dLayer::from_hfq(
                hfq,
                &shortcut_weight,
                Some(&shortcut_bias),
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        })
    }

    pub fn forward(&self, input: &CpuTensor) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(input, DiffusionGenerationRuntimeOptions::default())
    }

    fn forward_with_runtime_options(
        &self,
        input: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(input, &mut runtime_context)
    }

    fn forward_with_runtime_context(
        &self,
        input: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let hidden = self
            .norm1
            .forward_with_runtime_context(input, runtime_context)?;
        let hidden = silu_with_runtime_context(&hidden, runtime_context)?;
        let hidden = self
            .conv1
            .forward_with_runtime_context(&hidden, runtime_context)?;
        let hidden = self
            .norm2
            .forward_with_runtime_context(&hidden, runtime_context)?;
        let hidden = silu_with_runtime_context(&hidden, runtime_context)?;
        let hidden = self
            .conv2
            .forward_with_runtime_context(&hidden, runtime_context)?;
        let residual = if let Some(shortcut) = &self.shortcut {
            shortcut.forward_with_runtime_context(input, runtime_context)?
        } else {
            input.clone()
        };
        tensor_add_with_runtime_context(&hidden, &residual, runtime_context)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnetResnetBlock2D {
    pub norm1: GroupNormLayer,
    pub conv1: Conv2dLayer,
    pub time_emb_proj_weight: CpuTensor,
    pub time_emb_proj_bias: CpuTensor,
    pub norm2: GroupNormLayer,
    pub conv2: Conv2dLayer,
    pub shortcut: Option<Conv2dLayer>,
}

impl UnetResnetBlock2D {
    pub fn from_hfq(hfq: &HfqFile, prefix: &str, groups: usize, eps: f32) -> DiffusionResult<Self> {
        let shortcut_weight = format!("{prefix}.conv_shortcut.weight");
        let shortcut_bias = format!("{prefix}.conv_shortcut.bias");
        let shortcut = if hfq.find_tensor_info(&shortcut_weight).is_some() {
            Some(Conv2dLayer::from_hfq(
                hfq,
                &shortcut_weight,
                Some(&shortcut_bias),
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: GroupNormLayer::from_hfq(
                hfq,
                &format!("{prefix}.norm1.weight"),
                &format!("{prefix}.norm1.bias"),
                groups,
                eps,
            )?,
            conv1: Conv2dLayer::from_hfq(
                hfq,
                &format!("{prefix}.conv1.weight"),
                Some(&format!("{prefix}.conv1.bias")),
                1,
            )?,
            time_emb_proj_weight: CpuTensor::from_hfq(
                hfq,
                &format!("{prefix}.time_emb_proj.weight"),
            )?,
            time_emb_proj_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.time_emb_proj.bias"))?,
            norm2: GroupNormLayer::from_hfq(
                hfq,
                &format!("{prefix}.norm2.weight"),
                &format!("{prefix}.norm2.bias"),
                groups,
                eps,
            )?,
            conv2: Conv2dLayer::from_hfq(
                hfq,
                &format!("{prefix}.conv2.weight"),
                Some(&format!("{prefix}.conv2.bias")),
                1,
            )?,
            shortcut,
        })
    }

    pub fn forward(
        &self,
        input: &CpuTensor,
        time_embedding: &CpuTensor,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(
            input,
            time_embedding,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        input: &CpuTensor,
        time_embedding: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(input, time_embedding, &mut runtime_context)
    }

    fn forward_with_runtime_context(
        &self,
        input: &CpuTensor,
        time_embedding: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let [batch, _, _, _] = shape4(input)?;
        let [time_batch, _] = shape2(time_embedding)?;
        if time_batch != batch {
            return Err(DiffusionError::InvalidMetadata(format!(
                "UNet ResNet time embedding batch {time_batch} != input batch {batch}"
            )));
        }
        let hidden = self
            .norm1
            .forward_with_runtime_context(input, runtime_context)?;
        let hidden = silu_with_runtime_context(&hidden, runtime_context)?;
        let mut hidden = self
            .conv1
            .forward_with_runtime_context(&hidden, runtime_context)?;
        let projected_time = linear_with_runtime_context(
            &silu_with_runtime_context(time_embedding, runtime_context)?,
            &self.time_emb_proj_weight,
            &self.time_emb_proj_bias,
            runtime_context,
        )?;
        add_channel_bias_nchw_with_runtime_context(&mut hidden, &projected_time, runtime_context)?;
        let hidden = self
            .norm2
            .forward_with_runtime_context(&hidden, runtime_context)?;
        let hidden = silu_with_runtime_context(&hidden, runtime_context)?;
        let hidden = self
            .conv2
            .forward_with_runtime_context(&hidden, runtime_context)?;
        let residual = if let Some(shortcut) = &self.shortcut {
            shortcut.forward_with_runtime_context(input, runtime_context)?
        } else {
            input.clone()
        };
        tensor_add_with_runtime_context(&hidden, &residual, runtime_context)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnetTimeEmbedding {
    pub linear_1_weight: CpuTensor,
    pub linear_1_bias: CpuTensor,
    pub linear_2_weight: CpuTensor,
    pub linear_2_bias: CpuTensor,
}

impl UnetTimeEmbedding {
    pub fn from_hfq(hfq: &HfqFile) -> DiffusionResult<Self> {
        Ok(Self {
            linear_1_weight: CpuTensor::from_hfq(
                hfq,
                "unet/tensors/time_embedding.linear_1.weight",
            )?,
            linear_1_bias: CpuTensor::from_hfq(hfq, "unet/tensors/time_embedding.linear_1.bias")?,
            linear_2_weight: CpuTensor::from_hfq(
                hfq,
                "unet/tensors/time_embedding.linear_2.weight",
            )?,
            linear_2_bias: CpuTensor::from_hfq(hfq, "unet/tensors/time_embedding.linear_2.bias")?,
        })
    }

    pub fn forward(
        &self,
        timesteps: &[f32],
        flip_sin_to_cos: bool,
        freq_shift: f32,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(
            timesteps,
            flip_sin_to_cos,
            freq_shift,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        timesteps: &[f32],
        flip_sin_to_cos: bool,
        freq_shift: f32,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(
            timesteps,
            flip_sin_to_cos,
            freq_shift,
            &mut runtime_context,
        )
    }

    fn forward_with_runtime_context(
        &self,
        timesteps: &[f32],
        flip_sin_to_cos: bool,
        freq_shift: f32,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let (_, embedding_dim) = self.linear_1_weight.rows_cols()?;
        let input = timestep_embedding_with_runtime_context(
            timesteps,
            embedding_dim,
            flip_sin_to_cos,
            freq_shift,
            runtime_context,
        )?;
        let hidden = linear_with_runtime_context(
            &input,
            &self.linear_1_weight,
            &self.linear_1_bias,
            runtime_context,
        )?;
        let hidden = silu_with_runtime_context(&hidden, runtime_context)?;
        linear_with_runtime_context(
            &hidden,
            &self.linear_2_weight,
            &self.linear_2_bias,
            runtime_context,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnetTextTimeEmbedding {
    pub addition_time_embed_dim: usize,
    pub linear_1_weight: CpuTensor,
    pub linear_1_bias: CpuTensor,
    pub linear_2_weight: CpuTensor,
    pub linear_2_bias: CpuTensor,
}

impl UnetTextTimeEmbedding {
    pub fn from_hfq(hfq: &HfqFile, config: &UnetConfig) -> DiffusionResult<Option<Self>> {
        if config.addition_embed_type.as_deref() != Some("text_time") {
            return Ok(None);
        }
        let addition_time_embed_dim = config.addition_time_embed_dim.ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "UNet text_time addition requires addition_time_embed_dim".to_string(),
            )
        })?;
        let linear_1 = "unet/tensors/add_embedding.linear_1.weight";
        if hfq.find_tensor_info(linear_1).is_none() {
            return Err(DiffusionError::InvalidMetadata(
                "UNet text_time addition is configured but add_embedding weights are missing"
                    .to_string(),
            ));
        }
        Ok(Some(Self {
            addition_time_embed_dim,
            linear_1_weight: CpuTensor::from_hfq(hfq, linear_1)?,
            linear_1_bias: CpuTensor::from_hfq(hfq, "unet/tensors/add_embedding.linear_1.bias")?,
            linear_2_weight: CpuTensor::from_hfq(
                hfq,
                "unet/tensors/add_embedding.linear_2.weight",
            )?,
            linear_2_bias: CpuTensor::from_hfq(hfq, "unet/tensors/add_embedding.linear_2.bias")?,
        }))
    }

    pub fn forward(
        &self,
        text_embeds: &CpuTensor,
        time_ids: &CpuTensor,
        flip_sin_to_cos: bool,
        freq_shift: f32,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(
            text_embeds,
            time_ids,
            flip_sin_to_cos,
            freq_shift,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        text_embeds: &CpuTensor,
        time_ids: &CpuTensor,
        flip_sin_to_cos: bool,
        freq_shift: f32,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(
            text_embeds,
            time_ids,
            flip_sin_to_cos,
            freq_shift,
            &mut runtime_context,
        )
    }

    fn forward_with_runtime_context(
        &self,
        text_embeds: &CpuTensor,
        time_ids: &CpuTensor,
        flip_sin_to_cos: bool,
        freq_shift: f32,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let [batch, _] = shape2(text_embeds)?;
        let [time_batch, time_id_count] = shape2(time_ids)?;
        if time_batch != batch {
            return Err(DiffusionError::InvalidRequest(format!(
                "SDXL time_ids batch {time_batch} != text_embeds batch {batch}"
            )));
        }
        let time_embeds = timestep_embedding_with_runtime_context(
            &time_ids.data,
            self.addition_time_embed_dim,
            flip_sin_to_cos,
            freq_shift,
            runtime_context,
        )?;
        let [flat_time, time_width] = shape2(&time_embeds)?;
        if flat_time != batch * time_id_count {
            return Err(DiffusionError::InvalidMetadata(format!(
                "SDXL time embedding rows {flat_time} != batch*time_ids {}",
                batch * time_id_count
            )));
        }
        let mut flat = CpuTensor::zeros(&[batch, time_id_count * time_width]);
        for b in 0..batch {
            for t in 0..time_id_count {
                let src = (b * time_id_count + t) * time_width;
                let dst = b * time_id_count * time_width + t * time_width;
                flat.data[dst..dst + time_width]
                    .copy_from_slice(&time_embeds.data[src..src + time_width]);
            }
        }
        let input = concat_last_dim_2d_with_runtime_context(text_embeds, &flat, runtime_context)?;
        let hidden = linear_with_runtime_context(
            &input,
            &self.linear_1_weight,
            &self.linear_1_bias,
            runtime_context,
        )?;
        let hidden = silu_with_runtime_context(&hidden, runtime_context)?;
        linear_with_runtime_context(
            &hidden,
            &self.linear_2_weight,
            &self.linear_2_bias,
            runtime_context,
        )
    }
}

pub fn timestep_embedding(
    timesteps: &[f32],
    dim: usize,
    flip_sin_to_cos: bool,
    freq_shift: f32,
) -> DiffusionResult<CpuTensor> {
    if dim == 0 {
        return Err(DiffusionError::InvalidRequest(
            "timestep embedding dimension must be positive".to_string(),
        ));
    }
    let half = dim / 2;
    if half == 0 {
        return Ok(CpuTensor {
            shape: vec![timesteps.len(), dim],
            data: vec![0.0; timesteps.len() * dim],
        });
    }
    let denom = (half as f32 - freq_shift).max(1.0);
    let frequencies = (0..half)
        .map(|idx| (-10000.0f32.ln() * idx as f32 / denom).exp())
        .collect::<Vec<_>>();
    let mut out = CpuTensor::zeros(&[timesteps.len(), dim]);
    for (row, timestep) in timesteps.iter().enumerate() {
        let base = row * dim;
        for (idx, frequency) in frequencies.iter().enumerate() {
            let value = timestep * frequency;
            let (first, second) = if flip_sin_to_cos {
                (value.cos(), value.sin())
            } else {
                (value.sin(), value.cos())
            };
            out.data[base + idx] = first;
            out.data[base + half + idx] = second;
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionLayer {
    pub to_q_weight: CpuTensor,
    pub to_q_bias: Option<CpuTensor>,
    pub to_k_weight: CpuTensor,
    pub to_k_bias: Option<CpuTensor>,
    pub to_v_weight: CpuTensor,
    pub to_v_bias: Option<CpuTensor>,
    pub to_out_weight: CpuTensor,
    pub to_out_bias: Option<CpuTensor>,
    pub heads: usize,
}

impl AttentionLayer {
    pub fn from_hfq(hfq: &HfqFile, prefix: &str, heads: usize) -> DiffusionResult<Self> {
        Ok(Self {
            to_q_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.to_q.weight"))?,
            to_q_bias: optional_tensor(hfq, &format!("{prefix}.to_q.bias"))?,
            to_k_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.to_k.weight"))?,
            to_k_bias: optional_tensor(hfq, &format!("{prefix}.to_k.bias"))?,
            to_v_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.to_v.weight"))?,
            to_v_bias: optional_tensor(hfq, &format!("{prefix}.to_v.bias"))?,
            to_out_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.to_out.0.weight"))?,
            to_out_bias: optional_tensor(hfq, &format!("{prefix}.to_out.0.bias"))?,
            heads: heads.max(1),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &CpuTensor,
        encoder_states: Option<&CpuTensor>,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(
            hidden_states,
            encoder_states,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        hidden_states: &CpuTensor,
        encoder_states: Option<&CpuTensor>,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(hidden_states, encoder_states, &mut runtime_context)
    }

    fn forward_with_runtime_context(
        &self,
        hidden_states: &CpuTensor,
        encoder_states: Option<&CpuTensor>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let context = encoder_states.unwrap_or(hidden_states);
        let q = linear_3d_with_runtime_context(
            hidden_states,
            &self.to_q_weight,
            self.to_q_bias.as_ref(),
            runtime_context,
        )?;
        let k = linear_3d_with_runtime_context(
            context,
            &self.to_k_weight,
            self.to_k_bias.as_ref(),
            runtime_context,
        )?;
        let v = linear_3d_with_runtime_context(
            context,
            &self.to_v_weight,
            self.to_v_bias.as_ref(),
            runtime_context,
        )?;
        let attended = scaled_dot_product_attention_with_runtime_context(
            &q,
            &k,
            &v,
            self.heads,
            runtime_context,
        )?;
        linear_3d_with_runtime_context(
            &attended,
            &self.to_out_weight,
            self.to_out_bias.as_ref(),
            runtime_context,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeGluFeedForward {
    pub proj_weight: CpuTensor,
    pub proj_bias: CpuTensor,
    pub out_weight: CpuTensor,
    pub out_bias: CpuTensor,
}

impl GeGluFeedForward {
    pub fn from_hfq(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Self> {
        Ok(Self {
            proj_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.ff.net.0.proj.weight"))?,
            proj_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.ff.net.0.proj.bias"))?,
            out_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.ff.net.2.weight"))?,
            out_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.ff.net.2.bias"))?,
        })
    }

    pub fn forward(&self, hidden_states: &CpuTensor) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(
            hidden_states,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        hidden_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(hidden_states, &mut runtime_context)
    }

    fn forward_with_runtime_context(
        &self,
        hidden_states: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let projected = linear_3d_with_runtime_context(
            hidden_states,
            &self.proj_weight,
            Some(&self.proj_bias),
            runtime_context,
        )?;
        let gated = geglu_gate_3d_with_runtime_context(&projected, runtime_context)?;
        linear_3d_with_runtime_context(
            &gated,
            &self.out_weight,
            Some(&self.out_bias),
            runtime_context,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasicTransformerBlock {
    pub norm1_weight: CpuTensor,
    pub norm1_bias: CpuTensor,
    pub attn1: AttentionLayer,
    pub norm2_weight: CpuTensor,
    pub norm2_bias: CpuTensor,
    pub attn2: AttentionLayer,
    pub norm3_weight: CpuTensor,
    pub norm3_bias: CpuTensor,
    pub feed_forward: GeGluFeedForward,
}

impl BasicTransformerBlock {
    pub fn from_hfq(hfq: &HfqFile, prefix: &str, heads: usize) -> DiffusionResult<Self> {
        Ok(Self {
            norm1_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.norm1.weight"))?,
            norm1_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.norm1.bias"))?,
            attn1: AttentionLayer::from_hfq(hfq, &format!("{prefix}.attn1"), heads)?,
            norm2_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.norm2.weight"))?,
            norm2_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.norm2.bias"))?,
            attn2: AttentionLayer::from_hfq(hfq, &format!("{prefix}.attn2"), heads)?,
            norm3_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.norm3.weight"))?,
            norm3_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.norm3.bias"))?,
            feed_forward: GeGluFeedForward::from_hfq(hfq, prefix)?,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &CpuTensor,
        encoder_states: &CpuTensor,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(
            hidden_states,
            encoder_states,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        hidden_states: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(hidden_states, encoder_states, &mut runtime_context)
    }

    fn forward_with_runtime_context(
        &self,
        hidden_states: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let normed = layer_norm_3d_with_runtime_context(
            hidden_states,
            &self.norm1_weight,
            &self.norm1_bias,
            1e-5,
            runtime_context,
        )?;
        let attn = self
            .attn1
            .forward_with_runtime_context(&normed, None, runtime_context)?;
        let hidden_states = tensor_add_with_runtime_context(hidden_states, &attn, runtime_context)?;

        let normed = layer_norm_3d_with_runtime_context(
            &hidden_states,
            &self.norm2_weight,
            &self.norm2_bias,
            1e-5,
            runtime_context,
        )?;
        let attn = self.attn2.forward_with_runtime_context(
            &normed,
            Some(encoder_states),
            runtime_context,
        )?;
        let hidden_states =
            tensor_add_with_runtime_context(&hidden_states, &attn, runtime_context)?;

        let normed = layer_norm_3d_with_runtime_context(
            &hidden_states,
            &self.norm3_weight,
            &self.norm3_bias,
            1e-5,
            runtime_context,
        )?;
        let ff = self
            .feed_forward
            .forward_with_runtime_context(&normed, runtime_context)?;
        tensor_add_with_runtime_context(&hidden_states, &ff, runtime_context)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transformer2DModel {
    pub norm: GroupNormLayer,
    pub proj_in: Conv2dLayer,
    pub block: BasicTransformerBlock,
    pub proj_out: Conv2dLayer,
}

impl Transformer2DModel {
    pub fn from_hfq(
        hfq: &HfqFile,
        prefix: &str,
        heads: usize,
        norm_groups: usize,
        norm_eps: f32,
    ) -> DiffusionResult<Self> {
        Ok(Self {
            norm: GroupNormLayer::from_hfq(
                hfq,
                &format!("{prefix}.norm.weight"),
                &format!("{prefix}.norm.bias"),
                norm_groups,
                norm_eps,
            )?,
            proj_in: Conv2dLayer::from_hfq(
                hfq,
                &format!("{prefix}.proj_in.weight"),
                Some(&format!("{prefix}.proj_in.bias")),
                0,
            )?,
            block: BasicTransformerBlock::from_hfq(
                hfq,
                &format!("{prefix}.transformer_blocks.0"),
                heads,
            )?,
            proj_out: Conv2dLayer::from_hfq(
                hfq,
                &format!("{prefix}.proj_out.weight"),
                Some(&format!("{prefix}.proj_out.bias")),
                0,
            )?,
        })
    }

    pub fn forward(
        &self,
        input: &CpuTensor,
        encoder_states: &CpuTensor,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(
            input,
            encoder_states,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        input: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(input, encoder_states, &mut runtime_context)
    }

    fn forward_with_runtime_context(
        &self,
        input: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let residual = input.clone();
        let hidden = self
            .norm
            .forward_with_runtime_context(input, runtime_context)?;
        let hidden = self
            .proj_in
            .forward_with_runtime_context(&hidden, runtime_context)?;
        let [batch, channels, height, width] = shape4(&hidden)?;
        let hidden = nchw_to_bsc_with_runtime_context(&hidden, runtime_context)?;
        let hidden =
            self.block
                .forward_with_runtime_context(&hidden, encoder_states, runtime_context)?;
        let hidden = bsc_to_nchw_with_runtime_context(
            &hidden,
            batch,
            channels,
            height,
            width,
            runtime_context,
        )?;
        let hidden = self
            .proj_out
            .forward_with_runtime_context(&hidden, runtime_context)?;
        tensor_add_with_runtime_context(&hidden, &residual, runtime_context)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnetDownBlock2D {
    pub resnets: Vec<UnetResnetBlock2D>,
    pub attentions: Vec<Transformer2DModel>,
    pub downsampler: Option<Conv2dLayer>,
}

impl UnetDownBlock2D {
    pub fn from_hfq(
        hfq: &HfqFile,
        block_idx: usize,
        layers_per_block: usize,
        heads: usize,
        norm_groups: usize,
        norm_eps: f32,
    ) -> DiffusionResult<Self> {
        let prefix = format!("unet/tensors/down_blocks.{block_idx}");
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();
        for layer_idx in 0..layers_per_block {
            let resnet_prefix = format!("{prefix}.resnets.{layer_idx}");
            if hfq
                .find_tensor_info(&format!("{resnet_prefix}.norm1.weight"))
                .is_none()
            {
                break;
            }
            resnets.push(UnetResnetBlock2D::from_hfq(
                hfq,
                &resnet_prefix,
                norm_groups,
                norm_eps,
            )?);
            let attention_prefix = format!("{prefix}.attentions.{layer_idx}");
            if hfq
                .find_tensor_info(&format!("{attention_prefix}.norm.weight"))
                .is_some()
            {
                attentions.push(Transformer2DModel::from_hfq(
                    hfq,
                    &attention_prefix,
                    heads,
                    norm_groups,
                    norm_eps,
                )?);
            }
        }
        if resnets.is_empty() {
            return Err(DiffusionError::InvalidMetadata(format!(
                "UNet down block {block_idx} has no resnets"
            )));
        }
        let down_weight = format!("{prefix}.downsamplers.0.conv.weight");
        let down_bias = format!("{prefix}.downsamplers.0.conv.bias");
        let downsampler = if hfq.find_tensor_info(&down_weight).is_some() {
            Some(Conv2dLayer::from_hfq_with_stride(
                hfq,
                &down_weight,
                Some(&down_bias),
                1,
                2,
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            attentions,
            downsampler,
        })
    }

    pub fn forward(
        &self,
        hidden: CpuTensor,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
    ) -> DiffusionResult<(CpuTensor, Vec<CpuTensor>)> {
        self.forward_with_runtime_options(
            hidden,
            time_embedding,
            encoder_states,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        hidden: CpuTensor,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<(CpuTensor, Vec<CpuTensor>)> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(
            hidden,
            time_embedding,
            encoder_states,
            &mut runtime_context,
        )
    }

    fn forward_with_runtime_context(
        &self,
        mut hidden: CpuTensor,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<(CpuTensor, Vec<CpuTensor>)> {
        let mut skips = Vec::new();
        for (idx, resnet) in self.resnets.iter().enumerate() {
            hidden =
                resnet.forward_with_runtime_context(&hidden, time_embedding, runtime_context)?;
            if let Some(attention) = self.attentions.get(idx) {
                hidden = attention.forward_with_runtime_context(
                    &hidden,
                    encoder_states,
                    runtime_context,
                )?;
            }
            skips.push(hidden.clone());
        }
        if let Some(downsampler) = &self.downsampler {
            hidden = downsampler.forward_with_runtime_context(&hidden, runtime_context)?;
            skips.push(hidden.clone());
        }
        Ok((hidden, skips))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnetDownPath {
    pub conv_in: Conv2dLayer,
    pub blocks: Vec<UnetDownBlock2D>,
}

impl UnetDownPath {
    pub fn from_hfq(hfq: &HfqFile, config: &UnetConfig) -> DiffusionResult<Self> {
        let conv_in = Conv2dLayer::from_hfq(
            hfq,
            "unet/tensors/conv_in.weight",
            Some("unet/tensors/conv_in.bias"),
            1,
        )?;
        let block_count = if config.down_block_types.is_empty() {
            config.block_out_channels.len()
        } else {
            config.down_block_types.len()
        };
        if block_count == 0 {
            return Err(DiffusionError::InvalidMetadata(
                "UNet config has no down blocks".to_string(),
            ));
        }
        let layers_per_block = config.layers_per_block.unwrap_or(1).max(1);
        let norm_groups = config.norm_num_groups.unwrap_or(32);
        let norm_eps = config.norm_eps.unwrap_or(1e-5);
        let mut blocks = Vec::new();
        for block_idx in 0..block_count {
            let channels = config
                .block_out_channels
                .get(block_idx)
                .copied()
                .unwrap_or_else(|| config.block_out_channels.last().copied().unwrap_or(1));
            let head_dim = config
                .attention_head_dim
                .get(block_idx)
                .copied()
                .or_else(|| config.attention_head_dim.first().copied())
                .unwrap_or(channels);
            let heads = (channels / head_dim.max(1)).max(1);
            blocks.push(UnetDownBlock2D::from_hfq(
                hfq,
                block_idx,
                layers_per_block,
                heads,
                norm_groups,
                norm_eps,
            )?);
        }
        Ok(Self { conv_in, blocks })
    }

    pub fn forward(
        &self,
        sample: &CpuTensor,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
    ) -> DiffusionResult<(CpuTensor, Vec<CpuTensor>)> {
        self.forward_with_runtime_options(
            sample,
            time_embedding,
            encoder_states,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        sample: &CpuTensor,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<(CpuTensor, Vec<CpuTensor>)> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(
            sample,
            time_embedding,
            encoder_states,
            &mut runtime_context,
        )
    }

    fn forward_with_runtime_context(
        &self,
        sample: &CpuTensor,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<(CpuTensor, Vec<CpuTensor>)> {
        let mut hidden = self
            .conv_in
            .forward_with_runtime_context(sample, runtime_context)?;
        let mut skips = vec![hidden.clone()];
        for block in &self.blocks {
            let (next, mut block_skips) = block.forward_with_runtime_context(
                hidden,
                time_embedding,
                encoder_states,
                runtime_context,
            )?;
            hidden = next;
            skips.append(&mut block_skips);
        }
        Ok((hidden, skips))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnetUpBlock2D {
    pub resnets: Vec<UnetResnetBlock2D>,
    pub attentions: Vec<Transformer2DModel>,
    pub upsampler: Option<Conv2dLayer>,
}

impl UnetUpBlock2D {
    pub fn from_hfq(
        hfq: &HfqFile,
        block_idx: usize,
        heads: usize,
        norm_groups: usize,
        norm_eps: f32,
    ) -> DiffusionResult<Self> {
        let prefix = format!("unet/tensors/up_blocks.{block_idx}");
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();
        for layer_idx in 0.. {
            let resnet_prefix = format!("{prefix}.resnets.{layer_idx}");
            if hfq
                .find_tensor_info(&format!("{resnet_prefix}.norm1.weight"))
                .is_none()
            {
                break;
            }
            resnets.push(UnetResnetBlock2D::from_hfq(
                hfq,
                &resnet_prefix,
                norm_groups,
                norm_eps,
            )?);
            let attention_prefix = format!("{prefix}.attentions.{layer_idx}");
            if hfq
                .find_tensor_info(&format!("{attention_prefix}.norm.weight"))
                .is_some()
            {
                attentions.push(Transformer2DModel::from_hfq(
                    hfq,
                    &attention_prefix,
                    heads,
                    norm_groups,
                    norm_eps,
                )?);
            }
        }
        if resnets.is_empty() {
            return Err(DiffusionError::InvalidMetadata(format!(
                "UNet up block {block_idx} has no resnets"
            )));
        }
        let up_weight = format!("{prefix}.upsamplers.0.conv.weight");
        let up_bias = format!("{prefix}.upsamplers.0.conv.bias");
        let upsampler = if hfq.find_tensor_info(&up_weight).is_some() {
            Some(Conv2dLayer::from_hfq(hfq, &up_weight, Some(&up_bias), 1)?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            attentions,
            upsampler,
        })
    }

    pub fn forward(
        &self,
        hidden: CpuTensor,
        skips: &mut Vec<CpuTensor>,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(
            hidden,
            skips,
            time_embedding,
            encoder_states,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        hidden: CpuTensor,
        skips: &mut Vec<CpuTensor>,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(
            hidden,
            skips,
            time_embedding,
            encoder_states,
            &mut runtime_context,
        )
    }

    fn forward_with_runtime_context(
        &self,
        mut hidden: CpuTensor,
        skips: &mut Vec<CpuTensor>,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        for (idx, resnet) in self.resnets.iter().enumerate() {
            let skip = skips.pop().ok_or_else(|| {
                DiffusionError::InvalidMetadata("UNet up block ran out of skip tensors".to_string())
            })?;
            hidden = concat_channels_nchw_with_runtime_context(&hidden, &skip, runtime_context)?;
            hidden =
                resnet.forward_with_runtime_context(&hidden, time_embedding, runtime_context)?;
            if let Some(attention) = self.attentions.get(idx) {
                hidden = attention.forward_with_runtime_context(
                    &hidden,
                    encoder_states,
                    runtime_context,
                )?;
            }
        }
        if let Some(upsampler) = &self.upsampler {
            hidden = upsample_nearest2d_nchw_with_runtime_context(&hidden, 2, runtime_context)?;
            hidden = upsampler.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnetUpPath {
    pub blocks: Vec<UnetUpBlock2D>,
}

impl UnetUpPath {
    pub fn from_hfq(hfq: &HfqFile, config: &UnetConfig) -> DiffusionResult<Self> {
        let block_count = if config.up_block_types.is_empty() {
            config.block_out_channels.len()
        } else {
            config.up_block_types.len()
        };
        if block_count == 0 {
            return Err(DiffusionError::InvalidMetadata(
                "UNet config has no up blocks".to_string(),
            ));
        }
        let norm_groups = config.norm_num_groups.unwrap_or(32);
        let norm_eps = config.norm_eps.unwrap_or(1e-5);
        let mut blocks = Vec::new();
        for block_idx in 0..block_count {
            let channels = config
                .block_out_channels
                .iter()
                .rev()
                .nth(block_idx)
                .copied()
                .unwrap_or_else(|| config.block_out_channels.first().copied().unwrap_or(1));
            let head_dim = config
                .attention_head_dim
                .iter()
                .rev()
                .nth(block_idx)
                .copied()
                .or_else(|| config.attention_head_dim.first().copied())
                .unwrap_or(channels);
            let heads = (channels / head_dim.max(1)).max(1);
            blocks.push(UnetUpBlock2D::from_hfq(
                hfq,
                block_idx,
                heads,
                norm_groups,
                norm_eps,
            )?);
        }
        Ok(Self { blocks })
    }

    pub fn forward(
        &self,
        hidden: CpuTensor,
        skips: &mut Vec<CpuTensor>,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(
            hidden,
            skips,
            time_embedding,
            encoder_states,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        hidden: CpuTensor,
        skips: &mut Vec<CpuTensor>,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(
            hidden,
            skips,
            time_embedding,
            encoder_states,
            &mut runtime_context,
        )
    }

    fn forward_with_runtime_context(
        &self,
        mut hidden: CpuTensor,
        skips: &mut Vec<CpuTensor>,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        for block in &self.blocks {
            hidden = block.forward_with_runtime_context(
                hidden,
                skips,
                time_embedding,
                encoder_states,
                runtime_context,
            )?;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnetMidBlock2DCrossAttn {
    pub resnet_0: UnetResnetBlock2D,
    pub attention: Option<Transformer2DModel>,
    pub resnet_1: Option<UnetResnetBlock2D>,
}

impl UnetMidBlock2DCrossAttn {
    pub fn from_hfq(hfq: &HfqFile, config: &UnetConfig) -> DiffusionResult<Option<Self>> {
        let prefix = "unet/tensors/mid_block";
        let resnet_0_prefix = format!("{prefix}.resnets.0");
        if hfq
            .find_tensor_info(&format!("{resnet_0_prefix}.norm1.weight"))
            .is_none()
        {
            return Ok(None);
        }
        let norm_groups = config.norm_num_groups.unwrap_or(32);
        let norm_eps = config.norm_eps.unwrap_or(1e-5);
        let channels = config.block_out_channels.last().copied().unwrap_or(1);
        let head_dim = config
            .attention_head_dim
            .last()
            .copied()
            .or_else(|| config.attention_head_dim.first().copied())
            .unwrap_or(channels);
        let heads = (channels / head_dim.max(1)).max(1);
        let attention_prefix = format!("{prefix}.attentions.0");
        let attention = if hfq
            .find_tensor_info(&format!("{attention_prefix}.norm.weight"))
            .is_some()
        {
            Some(Transformer2DModel::from_hfq(
                hfq,
                &attention_prefix,
                heads,
                norm_groups,
                norm_eps,
            )?)
        } else {
            None
        };
        let resnet_1_prefix = format!("{prefix}.resnets.1");
        let resnet_1 = if hfq
            .find_tensor_info(&format!("{resnet_1_prefix}.norm1.weight"))
            .is_some()
        {
            Some(UnetResnetBlock2D::from_hfq(
                hfq,
                &resnet_1_prefix,
                norm_groups,
                norm_eps,
            )?)
        } else {
            None
        };
        Ok(Some(Self {
            resnet_0: UnetResnetBlock2D::from_hfq(hfq, &resnet_0_prefix, norm_groups, norm_eps)?,
            attention,
            resnet_1,
        }))
    }

    pub fn forward(
        &self,
        hidden: CpuTensor,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(
            hidden,
            time_embedding,
            encoder_states,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn forward_with_runtime_options(
        &self,
        hidden: CpuTensor,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(
            hidden,
            time_embedding,
            encoder_states,
            &mut runtime_context,
        )
    }

    fn forward_with_runtime_context(
        &self,
        mut hidden: CpuTensor,
        time_embedding: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        hidden =
            self.resnet_0
                .forward_with_runtime_context(&hidden, time_embedding, runtime_context)?;
        if let Some(attention) = &self.attention {
            hidden =
                attention.forward_with_runtime_context(&hidden, encoder_states, runtime_context)?;
        }
        if let Some(resnet) = &self.resnet_1 {
            hidden =
                resnet.forward_with_runtime_context(&hidden, time_embedding, runtime_context)?;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeUnet2DConditionModel {
    pub time_embedding: UnetTimeEmbedding,
    pub add_embedding: Option<UnetTextTimeEmbedding>,
    pub down_path: UnetDownPath,
    pub mid_block: Option<UnetMidBlock2DCrossAttn>,
    pub up_path: UnetUpPath,
    pub conv_norm_out: GroupNormLayer,
    pub conv_out: Conv2dLayer,
    pub center_input_sample: bool,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f32,
}

impl NativeUnet2DConditionModel {
    pub fn from_hfq(hfq: &HfqFile, config: &UnetConfig) -> DiffusionResult<Self> {
        let norm_groups = config.norm_num_groups.unwrap_or(32);
        let norm_eps = config.norm_eps.unwrap_or(1e-5);
        Ok(Self {
            time_embedding: UnetTimeEmbedding::from_hfq(hfq)?,
            add_embedding: UnetTextTimeEmbedding::from_hfq(hfq, config)?,
            down_path: UnetDownPath::from_hfq(hfq, config)?,
            mid_block: UnetMidBlock2DCrossAttn::from_hfq(hfq, config)?,
            up_path: UnetUpPath::from_hfq(hfq, config)?,
            conv_norm_out: GroupNormLayer::from_hfq(
                hfq,
                "unet/tensors/conv_norm_out.weight",
                "unet/tensors/conv_norm_out.bias",
                norm_groups,
                norm_eps,
            )?,
            conv_out: Conv2dLayer::from_hfq(
                hfq,
                "unet/tensors/conv_out.weight",
                Some("unet/tensors/conv_out.bias"),
                1,
            )?,
            center_input_sample: config.center_input_sample,
            flip_sin_to_cos: config.flip_sin_to_cos,
            freq_shift: config.freq_shift,
        })
    }

    pub fn input_channels(&self) -> usize {
        self.down_path
            .conv_in
            .weight
            .shape
            .get(1)
            .copied()
            .unwrap_or(0)
    }

    pub fn forward(
        &self,
        sample: &CpuTensor,
        timesteps: &[f32],
        encoder_states: &CpuTensor,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_sdxl_conditioning(sample, timesteps, encoder_states, None)
    }

    pub fn forward_with_sdxl_conditioning(
        &self,
        sample: &CpuTensor,
        timesteps: &[f32],
        encoder_states: &CpuTensor,
        sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_sdxl_conditioning_and_runtime_options(
            sample,
            timesteps,
            encoder_states,
            sdxl_conditioning,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    pub fn forward_with_runtime_options(
        &self,
        sample: &CpuTensor,
        timesteps: &[f32],
        encoder_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        self.forward_with_sdxl_conditioning_and_runtime_options(
            sample,
            timesteps,
            encoder_states,
            None,
            runtime_options,
        )
    }

    fn forward_with_sdxl_conditioning_and_runtime_options(
        &self,
        sample: &CpuTensor,
        timesteps: &[f32],
        encoder_states: &CpuTensor,
        sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_sdxl_conditioning_and_runtime_context(
            sample,
            timesteps,
            encoder_states,
            sdxl_conditioning,
            &mut runtime_context,
        )
    }

    fn forward_with_sdxl_conditioning_and_runtime_context(
        &self,
        sample: &CpuTensor,
        timesteps: &[f32],
        encoder_states: &CpuTensor,
        sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let [batch, _, _, _] = shape4(sample)?;
        if timesteps.len() != batch {
            return Err(DiffusionError::InvalidRequest(format!(
                "UNet timestep batch {} != sample batch {batch}",
                timesteps.len()
            )));
        }
        let [encoder_batch, _, _] = shape3(encoder_states)?;
        if encoder_batch != batch {
            return Err(DiffusionError::InvalidRequest(format!(
                "UNet encoder batch {encoder_batch} != sample batch {batch}"
            )));
        }
        let sample = maybe_center_unet_input_with_runtime_context(
            sample,
            self.center_input_sample,
            runtime_context,
        )?;
        let mut time_embedding = self.time_embedding.forward_with_runtime_context(
            timesteps,
            self.flip_sin_to_cos,
            self.freq_shift,
            runtime_context,
        )?;
        if let Some(sdxl_conditioning) = sdxl_conditioning {
            let add_embedding = self.add_embedding.as_ref().ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "SDXL text_time conditioning requires UNet add_embedding weights".to_string(),
                )
            })?;
            let added = add_embedding.forward_with_runtime_context(
                sdxl_conditioning.text_embeds,
                sdxl_conditioning.time_ids,
                self.flip_sin_to_cos,
                self.freq_shift,
                runtime_context,
            )?;
            time_embedding =
                tensor_add_with_runtime_context(&time_embedding, &added, runtime_context)?;
        }
        let (hidden, mut skips) = self.down_path.forward_with_runtime_context(
            &sample,
            &time_embedding,
            encoder_states,
            runtime_context,
        )?;
        let hidden = if let Some(mid_block) = &self.mid_block {
            mid_block.forward_with_runtime_context(
                hidden,
                &time_embedding,
                encoder_states,
                runtime_context,
            )?
        } else {
            hidden
        };
        let hidden = self.up_path.forward_with_runtime_context(
            hidden,
            &mut skips,
            &time_embedding,
            encoder_states,
            runtime_context,
        )?;
        let hidden = self
            .conv_norm_out
            .forward_with_runtime_context(&hidden, runtime_context)?;
        let hidden = silu_with_runtime_context(&hidden, runtime_context)?;
        self.conv_out
            .forward_with_runtime_context(&hidden, runtime_context)
    }

    pub fn denoise_latents(
        &self,
        latents: LatentBatch,
        schedule: &DiffusionSchedule,
        cfg_scale: f32,
        positive_embeddings: &CpuTensor,
        negative_embeddings: &CpuTensor,
        positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
        masked_reference: Option<&MaskedDenoiseReference<'_>>,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<LatentBatch> {
        self.denoise_latents_with_runtime_options(
            latents,
            schedule,
            cfg_scale,
            positive_embeddings,
            negative_embeddings,
            positive_sdxl_conditioning,
            negative_sdxl_conditioning,
            inpaint_conditioning,
            masked_reference,
            DiffusionGenerationRuntimeOptions::default(),
            progress,
        )
        .map(|output| output.latents)
    }

    fn denoise_latents_with_runtime_options(
        &self,
        latents: LatentBatch,
        schedule: &DiffusionSchedule,
        cfg_scale: f32,
        positive_embeddings: &CpuTensor,
        negative_embeddings: &CpuTensor,
        positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
        masked_reference: Option<&MaskedDenoiseReference<'_>>,
        runtime_options: DiffusionGenerationRuntimeOptions,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<DenoiseLatentsOutput> {
        denoise_latents_with_cfg_progress_and_runtime_options(
            latents,
            schedule,
            cfg_scale,
            positive_embeddings,
            negative_embeddings,
            |sample, timesteps, encoder_states, sdxl_conditioning, runtime_context| {
                self.forward_with_sdxl_conditioning_and_runtime_context(
                    sample,
                    timesteps,
                    encoder_states,
                    sdxl_conditioning,
                    runtime_context,
                )
            },
            positive_sdxl_conditioning,
            negative_sdxl_conditioning,
            inpaint_conditioning,
            masked_reference,
            runtime_options,
            progress,
        )
    }

    fn denoise_latents_with_runtime_context(
        &self,
        latents: LatentBatch,
        schedule: &DiffusionSchedule,
        cfg_scale: f32,
        positive_embeddings: &CpuTensor,
        negative_embeddings: &CpuTensor,
        positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
        masked_reference: Option<&MaskedDenoiseReference<'_>>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<DenoiseLatentsOutput> {
        denoise_latents_with_cfg_progress_and_runtime_context(
            latents,
            schedule,
            cfg_scale,
            positive_embeddings,
            negative_embeddings,
            |sample, timesteps, encoder_states, sdxl_conditioning, runtime_context| {
                self.forward_with_sdxl_conditioning_and_runtime_context(
                    sample,
                    timesteps,
                    encoder_states,
                    sdxl_conditioning,
                    runtime_context,
                )
            },
            positive_sdxl_conditioning,
            negative_sdxl_conditioning,
            inpaint_conditioning,
            masked_reference,
            runtime_context,
            progress,
        )
    }
}

fn maybe_center_unet_input(sample: &CpuTensor, center_input_sample: bool) -> CpuTensor {
    if center_input_sample {
        CpuTensor {
            shape: sample.shape.clone(),
            data: sample.data.iter().map(|value| value * 2.0 - 1.0).collect(),
        }
    } else {
        sample.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VaeAttentionBlock {
    pub norm: GroupNormLayer,
    pub attention: AttentionLayer,
}

impl VaeAttentionBlock {
    pub fn from_hfq(hfq: &HfqFile, prefix: &str, groups: usize, eps: f32) -> DiffusionResult<Self> {
        Ok(Self {
            norm: GroupNormLayer::from_hfq(
                hfq,
                &format!("{prefix}.group_norm.weight"),
                &format!("{prefix}.group_norm.bias"),
                groups,
                eps,
            )?,
            attention: AttentionLayer::from_hfq(hfq, prefix, 1)?,
        })
    }

    pub fn forward(&self, input: &CpuTensor) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(input, DiffusionGenerationRuntimeOptions::default())
    }

    fn forward_with_runtime_options(
        &self,
        input: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(input, &mut runtime_context)
    }

    fn forward_with_runtime_context(
        &self,
        input: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let residual = input.clone();
        let [batch, channels, height, width] = shape4(input)?;
        let hidden = self
            .norm
            .forward_with_runtime_context(input, runtime_context)?;
        let hidden = nchw_to_bsc_with_runtime_context(&hidden, runtime_context)?;
        let hidden = self
            .attention
            .forward_with_runtime_context(&hidden, None, runtime_context)?;
        let hidden = bsc_to_nchw_with_runtime_context(
            &hidden,
            batch,
            channels,
            height,
            width,
            runtime_context,
        )?;
        tensor_add_with_runtime_context(&hidden, &residual, runtime_context)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VaeEncoderDownBlock {
    pub resnets: Vec<ResnetBlock2D>,
    pub downsampler: Option<Conv2dLayer>,
}

impl VaeEncoderDownBlock {
    pub fn from_hfq(hfq: &HfqFile, block_idx: usize, groups: usize) -> DiffusionResult<Self> {
        let prefix = format!("vae/tensors/encoder.down_blocks.{block_idx}");
        let mut resnets = Vec::new();
        for layer_idx in 0.. {
            let resnet_prefix = format!("{prefix}.resnets.{layer_idx}");
            if hfq
                .find_tensor_info(&format!("{resnet_prefix}.norm1.weight"))
                .is_none()
            {
                break;
            }
            resnets.push(ResnetBlock2D::from_hfq(hfq, &resnet_prefix, groups)?);
        }
        if resnets.is_empty() {
            return Err(DiffusionError::InvalidMetadata(format!(
                "VAE encoder down block {block_idx} has no resnets"
            )));
        }
        let down_weight = format!("{prefix}.downsamplers.0.conv.weight");
        let down_bias = format!("{prefix}.downsamplers.0.conv.bias");
        let downsampler = if hfq.find_tensor_info(&down_weight).is_some() {
            Some(Conv2dLayer::from_hfq_with_stride(
                hfq,
                &down_weight,
                Some(&down_bias),
                1,
                2,
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsampler,
        })
    }

    pub fn forward(&self, hidden: CpuTensor) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(hidden, DiffusionGenerationRuntimeOptions::default())
    }

    fn forward_with_runtime_options(
        &self,
        hidden: CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(hidden, &mut runtime_context)
    }

    fn forward_with_runtime_context(
        &self,
        mut hidden: CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        for resnet in &self.resnets {
            hidden = resnet.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        if let Some(downsampler) = &self.downsampler {
            hidden = downsampler.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeVaeEncoder {
    pub conv_in: Conv2dLayer,
    pub down_blocks: Vec<VaeEncoderDownBlock>,
    pub mid_resnet_0: Option<ResnetBlock2D>,
    pub mid_attention: Option<VaeAttentionBlock>,
    pub mid_resnet_1: Option<ResnetBlock2D>,
    pub conv_norm_out: GroupNormLayer,
    pub conv_out: Conv2dLayer,
    pub quant_conv: Option<Conv2dLayer>,
    pub scaling_factor: f32,
}

impl NativeVaeEncoder {
    pub fn from_hfq(hfq: &HfqFile, config: &VaeConfig) -> DiffusionResult<Self> {
        let groups = config.norm_num_groups.unwrap_or(32);
        let eps = config.norm_eps.unwrap_or(1e-6);
        let block_count = config
            .down_block_types
            .len()
            .max(config.block_out_channels.len());
        if block_count == 0 {
            return Err(DiffusionError::InvalidMetadata(
                "VAE encoder config has no down blocks".to_string(),
            ));
        }
        let mut down_blocks = Vec::new();
        for block_idx in 0..block_count {
            down_blocks.push(VaeEncoderDownBlock::from_hfq(hfq, block_idx, groups)?);
        }
        let mid_resnet_0_prefix = "vae/tensors/encoder.mid_block.resnets.0";
        let mid_resnet_0 = if hfq
            .find_tensor_info(&format!("{mid_resnet_0_prefix}.norm1.weight"))
            .is_some()
        {
            Some(ResnetBlock2D::from_hfq(hfq, mid_resnet_0_prefix, groups)?)
        } else {
            None
        };
        let mid_attention_prefix = "vae/tensors/encoder.mid_block.attentions.0";
        let mid_attention = if hfq
            .find_tensor_info(&format!("{mid_attention_prefix}.group_norm.weight"))
            .is_some()
        {
            Some(VaeAttentionBlock::from_hfq(
                hfq,
                mid_attention_prefix,
                groups,
                eps,
            )?)
        } else {
            None
        };
        let mid_resnet_1_prefix = "vae/tensors/encoder.mid_block.resnets.1";
        let mid_resnet_1 = if hfq
            .find_tensor_info(&format!("{mid_resnet_1_prefix}.norm1.weight"))
            .is_some()
        {
            Some(ResnetBlock2D::from_hfq(hfq, mid_resnet_1_prefix, groups)?)
        } else {
            None
        };
        let quant_conv = if hfq
            .find_tensor_info("vae/tensors/quant_conv.weight")
            .is_some()
        {
            Some(Conv2dLayer::from_hfq(
                hfq,
                "vae/tensors/quant_conv.weight",
                Some("vae/tensors/quant_conv.bias"),
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv_in: Conv2dLayer::from_hfq(
                hfq,
                "vae/tensors/encoder.conv_in.weight",
                Some("vae/tensors/encoder.conv_in.bias"),
                1,
            )?,
            down_blocks,
            mid_resnet_0,
            mid_attention,
            mid_resnet_1,
            conv_norm_out: GroupNormLayer::from_hfq(
                hfq,
                "vae/tensors/encoder.conv_norm_out.weight",
                "vae/tensors/encoder.conv_norm_out.bias",
                groups,
                eps,
            )?,
            conv_out: Conv2dLayer::from_hfq(
                hfq,
                "vae/tensors/encoder.conv_out.weight",
                Some("vae/tensors/encoder.conv_out.bias"),
                1,
            )?,
            quant_conv,
            scaling_factor: config.scaling_factor.unwrap_or(0.18215),
        })
    }

    pub fn encode_tensor_moments(&self, image: &CpuTensor) -> DiffusionResult<CpuTensor> {
        self.encode_tensor_moments_with_runtime_options(
            image,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn encode_tensor_moments_with_runtime_options(
        &self,
        image: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.encode_tensor_moments_with_runtime_context(image, &mut runtime_context)
    }

    fn encode_tensor_moments_with_runtime_context(
        &self,
        image: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let mut hidden = self
            .conv_in
            .forward_with_runtime_context(image, runtime_context)?;
        for block in &self.down_blocks {
            hidden = block.forward_with_runtime_context(hidden, runtime_context)?;
        }
        if let Some(resnet) = &self.mid_resnet_0 {
            hidden = resnet.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        if let Some(attention) = &self.mid_attention {
            hidden = attention.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        if let Some(resnet) = &self.mid_resnet_1 {
            hidden = resnet.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        hidden = self
            .conv_norm_out
            .forward_with_runtime_context(&hidden, runtime_context)?;
        hidden = silu_with_runtime_context(&hidden, runtime_context)?;
        hidden = self
            .conv_out
            .forward_with_runtime_context(&hidden, runtime_context)?;
        if let Some(quant_conv) = &self.quant_conv {
            hidden = quant_conv.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        Ok(hidden)
    }

    pub fn encode_to_latents(&self, image: &RgbImageBatch) -> DiffusionResult<LatentBatch> {
        let image = rgb_batch_to_vae_tensor(image)?;
        let moments = self.encode_tensor_moments(&image)?;
        vae_moments_to_latents(&moments, self.scaling_factor)
    }

    #[cfg(all(test, feature = "rocm"))]
    fn encode_to_latents_with_runtime_options(
        &self,
        image: &RgbImageBatch,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<LatentBatch> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.encode_to_latents_with_runtime_context(image, &mut runtime_context)
    }

    fn encode_to_latents_with_runtime_context(
        &self,
        image: &RgbImageBatch,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<LatentBatch> {
        let image_tensor = rgb_batch_to_vae_tensor_with_runtime_context(image, runtime_context)?;
        let moments =
            self.encode_tensor_moments_with_runtime_context(&image_tensor, runtime_context)?;
        vae_moments_to_latents_with_runtime_context(&moments, self.scaling_factor, runtime_context)
    }
}

fn vae_moments_to_latents(
    moments: &CpuTensor,
    scaling_factor: f32,
) -> DiffusionResult<LatentBatch> {
    let [batch, channels, height, width] = shape4(moments)?;
    if channels % 2 != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "VAE encoder moments channel count {channels} is not even"
        )));
    }
    let latent_channels = channels / 2;
    let mut data = Vec::with_capacity(batch * latent_channels * height * width);
    let scale = scaling_factor.max(f32::MIN_POSITIVE);
    for b in 0..batch {
        for c in 0..latent_channels {
            for y in 0..height {
                for x in 0..width {
                    data.push(moments.data[nchw_idx(b, c, y, x, channels, height, width)] * scale);
                }
            }
        }
    }
    Ok(LatentBatch {
        batch,
        channels: latent_channels,
        height,
        width,
        data,
    })
}

fn rgb_batch_to_vae_tensor_with_runtime_context(
    image: &RgbImageBatch,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return rgb_batch_to_vae_tensor(image);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| rgb_batch_to_vae_tensor_hip_on_gpu(gpu, image))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

fn vae_moments_to_latents_with_runtime_context(
    moments: &CpuTensor,
    scaling_factor: f32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<LatentBatch> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return vae_moments_to_latents(moments, scaling_factor);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context
            .with_rocm_gpu(|gpu| vae_moments_to_latents_hip_on_gpu(gpu, moments, scaling_factor))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VaeDecoderUpBlock {
    pub resnets: Vec<ResnetBlock2D>,
    pub upsampler: Option<Conv2dLayer>,
}

impl VaeDecoderUpBlock {
    pub fn from_hfq(hfq: &HfqFile, block_idx: usize, groups: usize) -> DiffusionResult<Self> {
        let prefix = format!("vae/tensors/decoder.up_blocks.{block_idx}");
        let mut resnets = Vec::new();
        for layer_idx in 0.. {
            let resnet_prefix = format!("{prefix}.resnets.{layer_idx}");
            if hfq
                .find_tensor_info(&format!("{resnet_prefix}.norm1.weight"))
                .is_none()
            {
                break;
            }
            resnets.push(ResnetBlock2D::from_hfq(hfq, &resnet_prefix, groups)?);
        }
        if resnets.is_empty() {
            return Err(DiffusionError::InvalidMetadata(format!(
                "VAE decoder up block {block_idx} has no resnets"
            )));
        }
        let up_weight = format!("{prefix}.upsamplers.0.conv.weight");
        let up_bias = format!("{prefix}.upsamplers.0.conv.bias");
        let upsampler = if hfq.find_tensor_info(&up_weight).is_some() {
            Some(Conv2dLayer::from_hfq(hfq, &up_weight, Some(&up_bias), 1)?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }

    pub fn forward(&self, hidden: CpuTensor) -> DiffusionResult<CpuTensor> {
        self.forward_with_runtime_options(hidden, DiffusionGenerationRuntimeOptions::default())
    }

    fn forward_with_runtime_options(
        &self,
        hidden: CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(hidden, &mut runtime_context)
    }

    fn forward_with_runtime_context(
        &self,
        mut hidden: CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        for resnet in &self.resnets {
            hidden = resnet.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        if let Some(upsampler) = &self.upsampler {
            hidden = upsample_nearest2d_nchw_with_runtime_context(&hidden, 2, runtime_context)?;
            hidden = upsampler.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeVaeDecoder {
    pub post_quant_conv: Option<Conv2dLayer>,
    pub conv_in: Conv2dLayer,
    pub mid_resnet_0: Option<ResnetBlock2D>,
    pub mid_attention: Option<VaeAttentionBlock>,
    pub mid_resnet_1: Option<ResnetBlock2D>,
    pub up_blocks: Vec<VaeDecoderUpBlock>,
    pub conv_norm_out: GroupNormLayer,
    pub conv_out: Conv2dLayer,
    pub scaling_factor: f32,
}

impl NativeVaeDecoder {
    pub fn from_hfq(hfq: &HfqFile, config: &VaeConfig) -> DiffusionResult<Self> {
        let groups = config.norm_num_groups.unwrap_or(32);
        let eps = config.norm_eps.unwrap_or(1e-6);
        let post_quant_conv = if hfq
            .find_tensor_info("vae/tensors/post_quant_conv.weight")
            .is_some()
        {
            Some(Conv2dLayer::from_hfq(
                hfq,
                "vae/tensors/post_quant_conv.weight",
                Some("vae/tensors/post_quant_conv.bias"),
                0,
            )?)
        } else {
            None
        };
        let mid_resnet_0_prefix = "vae/tensors/decoder.mid_block.resnets.0";
        let mid_resnet_0 = if hfq
            .find_tensor_info(&format!("{mid_resnet_0_prefix}.norm1.weight"))
            .is_some()
        {
            Some(ResnetBlock2D::from_hfq(hfq, mid_resnet_0_prefix, groups)?)
        } else {
            None
        };
        let mid_attention_prefix = "vae/tensors/decoder.mid_block.attentions.0";
        let mid_attention = if hfq
            .find_tensor_info(&format!("{mid_attention_prefix}.group_norm.weight"))
            .is_some()
        {
            Some(VaeAttentionBlock::from_hfq(
                hfq,
                mid_attention_prefix,
                groups,
                eps,
            )?)
        } else {
            None
        };
        let mid_resnet_1_prefix = "vae/tensors/decoder.mid_block.resnets.1";
        let mid_resnet_1 = if hfq
            .find_tensor_info(&format!("{mid_resnet_1_prefix}.norm1.weight"))
            .is_some()
        {
            Some(ResnetBlock2D::from_hfq(hfq, mid_resnet_1_prefix, groups)?)
        } else {
            None
        };

        let block_count = config
            .up_block_types
            .len()
            .max(config.block_out_channels.len());
        if block_count == 0 {
            return Err(DiffusionError::InvalidMetadata(
                "VAE decoder config has no up blocks".to_string(),
            ));
        }
        let mut up_blocks = Vec::new();
        for block_idx in 0..block_count {
            up_blocks.push(VaeDecoderUpBlock::from_hfq(hfq, block_idx, groups)?);
        }

        Ok(Self {
            post_quant_conv,
            conv_in: Conv2dLayer::from_hfq(
                hfq,
                "vae/tensors/decoder.conv_in.weight",
                Some("vae/tensors/decoder.conv_in.bias"),
                1,
            )?,
            mid_resnet_0,
            mid_attention,
            mid_resnet_1,
            up_blocks,
            conv_norm_out: GroupNormLayer::from_hfq(
                hfq,
                "vae/tensors/decoder.conv_norm_out.weight",
                "vae/tensors/decoder.conv_norm_out.bias",
                groups,
                eps,
            )?,
            conv_out: Conv2dLayer::from_hfq(
                hfq,
                "vae/tensors/decoder.conv_out.weight",
                Some("vae/tensors/decoder.conv_out.bias"),
                1,
            )?,
            scaling_factor: config.scaling_factor.unwrap_or(0.18215),
        })
    }

    pub fn decode_latents(&self, latents: &LatentBatch) -> DiffusionResult<CpuTensor> {
        self.decode_latents_with_runtime_options(
            latents,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn decode_latents_with_runtime_options(
        &self,
        latents: &LatentBatch,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.decode_latents_with_runtime_context(latents, &mut runtime_context)
    }

    fn decode_latents_with_runtime_context(
        &self,
        latents: &LatentBatch,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let mut hidden = latents.as_nchw_tensor();
        let scale = self.scaling_factor.max(f32::MIN_POSITIVE);
        hidden = scale_tensor_with_runtime_context(&hidden, scale.recip(), runtime_context)?;
        if let Some(post_quant_conv) = &self.post_quant_conv {
            hidden = post_quant_conv.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        hidden = self
            .conv_in
            .forward_with_runtime_context(&hidden, runtime_context)?;
        if let Some(resnet) = &self.mid_resnet_0 {
            hidden = resnet.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        if let Some(attention) = &self.mid_attention {
            hidden = attention.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        if let Some(resnet) = &self.mid_resnet_1 {
            hidden = resnet.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        for block in &self.up_blocks {
            hidden = block.forward_with_runtime_context(hidden, runtime_context)?;
        }
        hidden = self
            .conv_norm_out
            .forward_with_runtime_context(&hidden, runtime_context)?;
        hidden = silu_with_runtime_context(&hidden, runtime_context)?;
        self.conv_out
            .forward_with_runtime_context(&hidden, runtime_context)
    }

    pub fn decode_to_rgb8(&self, latents: &LatentBatch) -> DiffusionResult<RgbImageBatch> {
        let decoded = self.decode_latents(latents)?;
        rgb_tensor_to_u8(&decoded)
    }
}

pub fn rgb_tensor_to_u8(tensor: &CpuTensor) -> DiffusionResult<RgbImageBatch> {
    let [batch, channels, height, width] = shape4(tensor)?;
    if channels != 3 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "expected RGB tensor with 3 channels, got {channels}"
        )));
    }
    let mut data = Vec::with_capacity(batch * height * width * 3);
    for b in 0..batch {
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    let value = tensor.data[nchw_idx(b, c, y, x, channels, height, width)];
                    let value = (value * 0.5 + 0.5).clamp(0.0, 1.0);
                    data.push((value * 255.0).round() as u8);
                }
            }
        }
    }
    Ok(RgbImageBatch {
        batch,
        width,
        height,
        data,
    })
}

fn rgb_tensor_to_u8_with_runtime_context(
    tensor: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<RgbImageBatch> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return rgb_tensor_to_u8(tensor);
    };
    #[cfg(feature = "rocm")]
    {
        runtime_context.with_rocm_gpu(|gpu| rgb_tensor_to_u8_hip_on_gpu(gpu, tensor))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = _device_id;
        Err(rocm_hybrid_unavailable_error())
    }
}

#[cfg(feature = "rocm")]
const DIFFUSION_RGB_TENSOR_TO_U8_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_rgb_tensor_to_u8(
    const float* input,
    unsigned char* output,
    int total_pixels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) {
        return;
    }
    int pixels_per_batch = height * width;
    int b = idx / pixels_per_batch;
    int rem = idx - b * pixels_per_batch;
    int y = rem / width;
    int x = rem - y * width;
    for (int c = 0; c < 3; ++c) {
        int input_idx = ((b * 3 + c) * height + y) * width + x;
        float value = input[input_idx] * 0.5f + 0.5f;
        value = fminf(fmaxf(value, 0.0f), 1.0f);
        output[idx * 3 + c] = (unsigned char)floorf(value * 255.0f + 0.5f);
    }
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_VAE_BOUNDARY_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_rgb_u8_to_vae_nchw_f32(
    const unsigned char* input,
    float* output,
    int total_outputs,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % 3;
    int b = t / 3;
    int rgb_idx = (b * height * width + y * width + x) * 3 + c;
    output[idx] = ((float)input[rgb_idx]) / 127.5f - 1.0f;
}

extern "C" __global__ void diffusion_vae_moments_to_latents_f32(
    const float* moments,
    float* output,
    int total_outputs,
    int moments_channels,
    int latent_channels,
    int height,
    int width,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % latent_channels;
    int b = t / latent_channels;
    int moments_idx = ((b * moments_channels + c) * height + y) * width + x;
    output[idx] = moments[moments_idx] * scale;
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_INPAINT_MASK_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <math.h>

extern "C" __global__ void diffusion_latent_mask_weights_from_rgb_f32(
    const unsigned char* mask,
    float* output,
    int total_outputs,
    int mask_height,
    int mask_width,
    int latent_height,
    int latent_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int latent_pixels = latent_height * latent_width;
    int b = idx / latent_pixels;
    int rem = idx - b * latent_pixels;
    int y = rem / latent_width;
    int x = rem - y * latent_width;
    int source_y = (y * mask_height) / latent_height;
    int source_x = (x * mask_width) / latent_width;
    int max_y = mask_height > 0 ? mask_height - 1 : 0;
    int max_x = mask_width > 0 ? mask_width - 1 : 0;
    source_y = source_y < max_y ? source_y : max_y;
    source_x = source_x < max_x ? source_x : max_x;
    int mask_idx = (b * mask_height * mask_width + source_y * mask_width + source_x) * 3;
    float luma = ((float)mask[mask_idx] + (float)mask[mask_idx + 1] + (float)mask[mask_idx + 2])
        / (3.0f * 255.0f);
    output[idx] = fminf(fmaxf(luma, 0.0f), 1.0f);
}

extern "C" __global__ void diffusion_masked_rgb_for_inpaint_u8(
    const unsigned char* image,
    const unsigned char* mask,
    unsigned char* output,
    int total_pixels
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= total_pixels) {
        return;
    }
    int idx = pixel * 3;
    float weight = ((float)mask[idx] + (float)mask[idx + 1] + (float)mask[idx + 2])
        / (3.0f * 255.0f);
    float keep = 1.0f - fminf(fmaxf(weight, 0.0f), 1.0f);
    output[idx] = (unsigned char)fminf(fmaxf(floorf((float)image[idx] * keep + 0.5f), 0.0f), 255.0f);
    output[idx + 1] = (unsigned char)fminf(fmaxf(floorf((float)image[idx + 1] * keep + 0.5f), 0.0f), 255.0f);
    output[idx + 2] = (unsigned char)fminf(fmaxf(floorf((float)image[idx + 2] * keep + 0.5f), 0.0f), 255.0f);
}

extern "C" __global__ void diffusion_blend_latents_with_mask_f32(
    const float* generated,
    const float* init,
    const float* mask,
    float* output,
    int total_outputs,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % channels;
    int b = t / channels;
    int mask_idx = (b * height + y) * width + x;
    float weight = mask[mask_idx];
    output[idx] = init[idx] * (1.0f - weight) + generated[idx] * weight;
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_EULER_STEP_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <float.h>

extern "C" __global__ void diffusion_euler_step_f32(
    const float* sample,
    const float* model_output,
    float* output,
    int n,
    float sigma,
    float next_sigma,
    int prediction_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float s = sample[idx];
    float m = model_output[idx];
    float derivative = m;
    if (fabsf(sigma) > FLT_MIN) {
        if (prediction_type == 1) {
            derivative = (s - m) / sigma;
        } else if (prediction_type == 2) {
            float sigma_sq = sigma * sigma;
            float denom = sigma_sq + 1.0f;
            float pred_original_sample = m * (-sigma / sqrtf(denom)) + s / denom;
            derivative = (s - pred_original_sample) / sigma;
        }
    }
    output[idx] = s + derivative * (next_sigma - sigma);
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_DENOISE_VECTOR_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_scale_model_input_f32(
    const float* sample,
    float* output,
    int n,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    output[idx] = sample[idx] * scale;
}

extern "C" __global__ void diffusion_cfg_guidance_f32(
    const float* negative_pred,
    const float* positive_pred,
    float* output,
    int n,
    float cfg_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float negative = negative_pred[idx];
    float positive = positive_pred[idx];
    output[idx] = negative + cfg_scale * (positive - negative);
}

extern "C" __global__ void diffusion_tensor_add_f32(
    const float* a,
    const float* b,
    float* output,
    int n,
    float unused
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    output[idx] = a[idx] + b[idx];
}

extern "C" __global__ void diffusion_center_unet_input_f32(
    const float* sample,
    float* output,
    int n,
    float unused
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    output[idx] = sample[idx] * 2.0f - 1.0f;
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_TIMESTEP_EMBEDDING_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_timestep_embedding_f32(
    const float* timesteps,
    float* output,
    int total_outputs,
    int dim,
    int half,
    int flip_sin_to_cos,
    float freq_shift
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int col = idx % dim;
    int row = idx / dim;
    if (half <= 0 || col >= half * 2) {
        output[idx] = 0.0f;
        return;
    }
    int frequency_idx = col < half ? col : col - half;
    float denom = fmaxf((float)half - freq_shift, 1.0f);
    float frequency = expf(-logf(10000.0f) * (float)frequency_idx / denom);
    float value = timesteps[row] * frequency;
    if (col < half) {
        output[idx] = flip_sin_to_cos ? cosf(value) : sinf(value);
    } else {
        output[idx] = flip_sin_to_cos ? sinf(value) : cosf(value);
    }
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_CONV2D_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_conv2d_nchw_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int total_outputs,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int padding,
    int stride,
    int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int ox = idx % out_w;
    int t = idx / out_w;
    int oy = t % out_h;
    t /= out_h;
    int oc = t % out_channels;
    int b = t / out_channels;

    float acc = has_bias ? bias[oc] : 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_h; ++ky) {
            int iy_with_pad = oy * stride + ky;
            if (iy_with_pad < padding || iy_with_pad >= in_h + padding) {
                continue;
            }
            int iy = iy_with_pad - padding;
            for (int kx = 0; kx < kernel_w; ++kx) {
                int ix_with_pad = ox * stride + kx;
                if (ix_with_pad < padding || ix_with_pad >= in_w + padding) {
                    continue;
                }
                int ix = ix_with_pad - padding;
                int input_idx = ((b * in_channels + ic) * in_h + iy) * in_w + ix;
                int weight_idx = ((oc * in_channels + ic) * kernel_h + ky) * kernel_w + kx;
                acc += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[idx] = acc;
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_GROUP_NORM_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_group_norm_nchw_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int total_elements,
    int channels,
    int height,
    int width,
    int groups,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    int t = idx / width;
    t /= height;
    int c = t % channels;
    int b = t / channels;
    int channels_per_group = channels / groups;
    int group = c / channels_per_group;
    int c_start = group * channels_per_group;
    int c_end = c_start + channels_per_group;
    int elems_per_group = channels_per_group * height * width;

    float sum = 0.0f;
    for (int gc = c_start; gc < c_end; ++gc) {
        for (int gy = 0; gy < height; ++gy) {
            for (int gx = 0; gx < width; ++gx) {
                int sample_idx = ((b * channels + gc) * height + gy) * width + gx;
                sum += input[sample_idx];
            }
        }
    }
    float mean = sum / (float)elems_per_group;

    float var_sum = 0.0f;
    for (int gc = c_start; gc < c_end; ++gc) {
        for (int gy = 0; gy < height; ++gy) {
            for (int gx = 0; gx < width; ++gx) {
                int sample_idx = ((b * channels + gc) * height + gy) * width + gx;
                float centered = input[sample_idx] - mean;
                var_sum += centered * centered;
            }
        }
    }
    float inv_std = rsqrtf(var_sum / (float)elems_per_group + eps);
    output[idx] = (input[idx] - mean) * inv_std * weight[c] + bias[c];
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_SILU_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_silu_f32(
    const float* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float value = input[idx];
    output[idx] = value / (1.0f + expf(-value));
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_QUICK_GELU_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_quick_gelu_f32(
    const float* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float value = input[idx];
    output[idx] = value / (1.0f + expf(-1.702f * value));
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_CLIP_EMBEDDINGS_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_clip_token_position_embedding_f32(
    const float* token_embedding,
    const float* position_embedding,
    const unsigned int* tokens,
    float* output,
    int total_outputs,
    int hidden
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int col = idx % hidden;
    int pos = idx / hidden;
    unsigned int token = tokens[pos];
    output[idx] = token_embedding[token * hidden + col] + position_embedding[pos * hidden + col];
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_UPSAMPLE_NEAREST2D_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_upsample_nearest2d_nchw_f32(
    const float* input,
    float* output,
    int total_outputs,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int ox = idx % out_w;
    int t = idx / out_w;
    int oy = t % out_h;
    t /= out_h;
    int c = t % channels;
    int b = t / channels;
    int iy = oy / scale;
    int ix = ox / scale;
    int input_idx = ((b * channels + c) * in_h + iy) * in_w + ix;
    output[idx] = input[input_idx];
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_LAYOUT_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_add_channel_bias_nchw_f32(
    const float* input,
    const float* bias,
    float* output,
    int total_elements,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    int t = idx / width;
    t /= height;
    int c = t % channels;
    int b = t / channels;
    output[idx] = input[idx] + bias[b * channels + c];
}

extern "C" __global__ void diffusion_nchw_to_bsc_f32(
    const float* input,
    float* output,
    int total_elements,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % channels;
    int b = t / channels;
    int seq = height * width;
    int s = y * width + x;
    output[(b * seq + s) * channels + c] = input[idx];
}

extern "C" __global__ void diffusion_bsc_to_nchw_f32(
    const float* input,
    float* output,
    int total_elements,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % channels;
    int b = t / channels;
    int seq = height * width;
    int s = y * width + x;
    output[idx] = input[(b * seq + s) * channels + c];
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_CONCAT_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_concat_channels_nchw_f32(
    const float* a,
    const float* b,
    float* output,
    int total_outputs,
    int a_channels,
    int b_channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int out_channels = a_channels + b_channels;
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % out_channels;
    int batch = t / out_channels;
    if (c < a_channels) {
        output[idx] = a[((batch * a_channels + c) * height + y) * width + x];
    } else {
        int bc = c - a_channels;
        output[idx] = b[((batch * b_channels + bc) * height + y) * width + x];
    }
}

extern "C" __global__ void diffusion_concat_last_dim_f32(
    const float* a,
    const float* b,
    float* output,
    int total_outputs,
    int left_width,
    int right_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int out_width = left_width + right_width;
    int col = idx % out_width;
    int row = idx / out_width;
    if (col < left_width) {
        output[idx] = a[row * left_width + col];
    } else {
        output[idx] = b[row * right_width + (col - left_width)];
    }
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_LINEAR_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_linear_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int total_outputs,
    int in_features,
    int out_features,
    int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int out_col = idx % out_features;
    int row = idx / out_features;
    int input_row = row * in_features;
    int weight_row = out_col * in_features;
    float acc = has_bias ? bias[out_col] : 0.0f;
    for (int k = 0; k < in_features; ++k) {
        acc += input[input_row + k] * weight[weight_row + k];
    }
    output[idx] = acc;
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_LAYER_NORM_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_layer_norm_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int total_outputs,
    int cols,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int col = idx % cols;
    int row = idx / cols;
    int base = row * cols;

    float sum = 0.0f;
    for (int k = 0; k < cols; ++k) {
        sum += input[base + k];
    }
    float mean = sum / (float)cols;

    float var_sum = 0.0f;
    for (int k = 0; k < cols; ++k) {
        float centered = input[base + k] - mean;
        var_sum += centered * centered;
    }
    float inv_std = rsqrtf(var_sum / (float)cols + eps);
    output[idx] = (input[idx] - mean) * inv_std * weight[col] + bias[col];
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_SOFTMAX_ROWS_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_softmax_rows_f32(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * cols;
    float max_value = input[base];
    for (int col = 1; col < cols; ++col) {
        max_value = fmaxf(max_value, input[base + col]);
    }

    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        float value = expf(input[base + col] - max_value);
        output[base + col] = value;
        sum += value;
    }
    if (sum > 0.0f) {
        for (int col = 0; col < cols; ++col) {
            output[base + col] /= sum;
        }
    }
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_SDPA_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_sdpa_3d_f32(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int total_outputs,
    int q_seq,
    int k_seq,
    int hidden,
    int heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int d = idx % hidden;
    int t = idx / hidden;
    int qi = t % q_seq;
    int b = t / q_seq;
    int head = d / head_dim;
    int head_off = head * head_dim;
    int local_d = d - head_off;
    float scale = rsqrtf((float)head_dim);

    float max_score = -INFINITY;
    for (int ki = 0; ki < k_seq; ++ki) {
        float dot = 0.0f;
        for (int hd = 0; hd < head_dim; ++hd) {
            int q_idx = ((b * q_seq + qi) * hidden) + head_off + hd;
            int k_idx = ((b * k_seq + ki) * hidden) + head_off + hd;
            dot += q[q_idx] * k[k_idx];
        }
        float score = dot * scale;
        max_score = fmaxf(max_score, score);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int ki = 0; ki < k_seq; ++ki) {
        float dot = 0.0f;
        for (int hd = 0; hd < head_dim; ++hd) {
            int q_idx = ((b * q_seq + qi) * hidden) + head_off + hd;
            int k_idx = ((b * k_seq + ki) * hidden) + head_off + hd;
            dot += q[q_idx] * k[k_idx];
        }
        float weight = expf(dot * scale - max_score);
        int v_idx = ((b * k_seq + ki) * hidden) + head_off + local_d;
        acc += weight * v[v_idx];
        sum += weight;
    }
    output[idx] = sum > 0.0f ? acc / sum : 0.0f;
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_CLIP_CAUSAL_ATTENTION_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_clip_causal_attention_f32(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int total_outputs,
    int seq,
    int hidden,
    int heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int d = idx % hidden;
    int qi = idx / hidden;
    int head = d / head_dim;
    int head_off = head * head_dim;
    int local_d = d - head_off;
    float scale = rsqrtf((float)head_dim);

    float max_score = -INFINITY;
    for (int ki = 0; ki <= qi; ++ki) {
        float dot = 0.0f;
        for (int hd = 0; hd < head_dim; ++hd) {
            dot += q[qi * hidden + head_off + hd] * k[ki * hidden + head_off + hd];
        }
        max_score = fmaxf(max_score, dot * scale);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int ki = 0; ki <= qi; ++ki) {
        float dot = 0.0f;
        for (int hd = 0; hd < head_dim; ++hd) {
            dot += q[qi * hidden + head_off + hd] * k[ki * hidden + head_off + hd];
        }
        float weight = expf(dot * scale - max_score);
        acc += weight * v[ki * hidden + head_off + local_d];
        sum += weight;
    }
    output[idx] = sum > 0.0f ? acc / sum : 0.0f;
}
"#;

#[cfg(feature = "rocm")]
const DIFFUSION_GEGLU_GATE_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_geglu_gate_3d_f32(
    const float* input,
    float* output,
    int total_outputs,
    int inner,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int col = idx % inner;
    int row = idx / inner;
    int src = row * width;
    float value = input[src + col];
    float gate_value = input[src + inner + col];
    float gelu_arg = 1.1283791670955126f * (gate_value + 0.044715f * gate_value * gate_value * gate_value);
    float gate = 0.5f * gate_value * (1.0f + tanhf(gelu_arg));
    output[idx] = value * gate;
}
"#;

#[cfg(feature = "rocm")]
fn ensure_and_launch_diffusion_kernel(
    gpu: &mut rdna_compute::Gpu,
    module_name: &str,
    source: &str,
    func_name: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    kernargs: &mut hip_bridge::KernargBlob,
) -> DiffusionResult<()> {
    gpu.ensure_kernel_public(module_name, source, func_name)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.launch_kernel_blob(func_name, grid, block, shared_mem, kernargs.as_mut_slice())
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))
}

#[cfg(feature = "rocm")]
fn rgb_tensor_to_u8_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    tensor: &CpuTensor,
) -> DiffusionResult<RgbImageBatch> {
    let [batch, channels, height, width] = shape4(tensor)?;
    if channels != 3 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "expected RGB tensor with 3 channels, got {channels}"
        )));
    }
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input = gpu
        .upload_f32(&tensor.data, &tensor.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_bytes = batch
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(width))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidMetadata("RGB output size overflows".to_string()))?;
    let output = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_rgb_tensor_to_u8";
    let function_name = "diffusion_rgb_tensor_to_u8";
    let kernel_source = DIFFUSION_RGB_TENSOR_TO_U8_HIP_SRC;
    let total_pixels = batch
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(width))
        .ok_or_else(|| DiffusionError::InvalidMetadata("RGB pixel count overflows".to_string()))?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(output.as_ptr());
    kernargs.push_i32(total_pixels as i32);
    kernargs.push_i32(height as i32);
    kernargs.push_i32(width as i32);
    kernargs.pad_to(16);
    let grid = [((total_pixels as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut data = vec![0u8; output_bytes];
    gpu.hip
        .memcpy_dtoh(&mut data, &output)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(output)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(RgbImageBatch {
        batch,
        width,
        height,
        data,
    })
}

#[cfg(feature = "rocm")]
fn rgb_batch_to_vae_tensor_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    batch: &RgbImageBatch,
) -> DiffusionResult<CpuTensor> {
    let bytes_per_image = batch
        .width
        .checked_mul(batch.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("image dimensions overflow".to_string()))?;
    let expected = bytes_per_image
        .checked_mul(batch.batch)
        .ok_or_else(|| DiffusionError::InvalidRequest("image batch size overflows".to_string()))?;
    if batch.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "RGB image batch has {} bytes, expected {expected}",
            batch.data.len()
        )));
    }
    let output_shape = [batch.batch, 3, batch.height, batch.width];
    let output_elements = checked_shape_elements("RGB-to-VAE tensor output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("RGB-to-VAE tensor output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .hip
        .malloc(batch.data.len())
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .memcpy_htod(&input_gpu, &batch.data)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_rgb_u8_to_vae_nchw_f32";
    let function_name = "diffusion_rgb_u8_to_vae_nchw_f32";
    let kernel_source = DIFFUSION_VAE_BOUNDARY_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "RGB-to-VAE output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("RGB-to-VAE height", batch.height)?);
    kernargs.push_i32(i32_kernel_dim("RGB-to-VAE width", batch.width)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(input_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn vae_moments_to_latents_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    moments: &CpuTensor,
    scaling_factor: f32,
) -> DiffusionResult<LatentBatch> {
    let [batch, channels, height, width] = shape4(moments)?;
    if channels % 2 != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "VAE encoder moments channel count {channels} is not even"
        )));
    }
    let latent_channels = channels / 2;
    let output_shape = [batch, latent_channels, height, width];
    let output_elements = checked_shape_elements("VAE moments-to-latents output", &output_shape)?;
    if output_elements == 0 {
        return Ok(LatentBatch {
            batch,
            channels: latent_channels,
            height,
            width,
            data: Vec::new(),
        });
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "VAE moments-to-latents output size overflows".to_string(),
            )
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let moments_gpu = gpu
        .upload_f32(&moments.data, &moments.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_vae_moments_to_latents_f32";
    let function_name = "diffusion_vae_moments_to_latents_f32";
    let kernel_source = DIFFUSION_VAE_BOUNDARY_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(moments_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "VAE moments-to-latents output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("VAE moments channels", channels)?);
    kernargs.push_i32(i32_kernel_dim("VAE latent channels", latent_channels)?);
    kernargs.push_i32(i32_kernel_dim("VAE latent height", height)?);
    kernargs.push_i32(i32_kernel_dim("VAE latent width", width)?);
    kernargs.push_f32(scaling_factor.max(f32::MIN_POSITIVE));
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(LatentBatch {
        batch,
        channels: latent_channels,
        height,
        width,
        data,
    })
}

#[cfg(feature = "rocm")]
fn latent_mask_weights_from_rgb_batch_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    mask: &RgbImageBatch,
    latents: &LatentBatch,
) -> DiffusionResult<Vec<f32>> {
    if mask.batch != latents.batch {
        return Err(DiffusionError::InvalidRequest(format!(
            "mask batch {} != latent batch {}",
            mask.batch, latents.batch
        )));
    }
    let bytes_per_image = mask
        .width
        .checked_mul(mask.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("mask dimensions overflow".to_string()))?;
    let expected = bytes_per_image.checked_mul(mask.batch).ok_or_else(|| {
        DiffusionError::InvalidRequest("mask batch dimensions overflow".to_string())
    })?;
    if mask.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "mask has {} bytes, expected {expected}",
            mask.data.len()
        )));
    }
    let output_elements = latents
        .batch
        .checked_mul(latents.height)
        .and_then(|pixels| pixels.checked_mul(latents.width))
        .ok_or_else(|| DiffusionError::InvalidRequest("latent mask size overflows".to_string()))?;
    if output_elements == 0 {
        return Ok(Vec::new());
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("latent mask output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mask_gpu = gpu
        .hip
        .malloc(mask.data.len())
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .memcpy_htod(&mask_gpu, &mask.data)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_latent_mask_weights_from_rgb_f32";
    let function_name = "diffusion_latent_mask_weights_from_rgb_f32";
    let kernel_source = DIFFUSION_INPAINT_MASK_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(mask_gpu.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "latent mask output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("latent mask source height", mask.height)?);
    kernargs.push_i32(i32_kernel_dim("latent mask source width", mask.width)?);
    kernargs.push_i32(i32_kernel_dim("latent mask output height", latents.height)?);
    kernargs.push_i32(i32_kernel_dim("latent mask output width", latents.width)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(mask_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(output)
}

#[cfg(feature = "rocm")]
fn masked_rgb_batch_for_inpaint_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    image: &RgbImageBatch,
    mask: &RgbImageBatch,
) -> DiffusionResult<RgbImageBatch> {
    if image.batch != mask.batch || image.width != mask.width || image.height != mask.height {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpaint image shape [{}x{}x{}] != mask shape [{}x{}x{}]",
            image.batch, image.width, image.height, mask.batch, mask.width, mask.height
        )));
    }
    let expected = image
        .batch
        .checked_mul(image.width)
        .and_then(|pixels| pixels.checked_mul(image.height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("image dimensions overflow".to_string()))?;
    if image.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "image has {} bytes, expected {expected}",
            image.data.len()
        )));
    }
    if mask.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "mask has {} bytes, expected {expected}",
            mask.data.len()
        )));
    }
    if expected == 0 {
        return Ok(RgbImageBatch {
            batch: image.batch,
            width: image.width,
            height: image.height,
            data: Vec::new(),
        });
    }
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let image_gpu = gpu
        .hip
        .malloc(image.data.len())
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .memcpy_htod(&image_gpu, &image.data)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mask_gpu = gpu
        .hip
        .malloc(mask.data.len())
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .memcpy_htod(&mask_gpu, &mask.data)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(expected)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_masked_rgb_for_inpaint_u8";
    let function_name = "diffusion_masked_rgb_for_inpaint_u8";
    let kernel_source = DIFFUSION_INPAINT_MASK_HIP_SRC;
    let total_pixels = expected / 3;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(image_gpu.as_ptr());
    kernargs.push_ptr(mask_gpu.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("masked RGB pixels", total_pixels)?);
    kernargs.pad_to(16);
    let grid = [((total_pixels as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut data = vec![0u8; expected];
    gpu.hip
        .memcpy_dtoh(&mut data, &output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(mask_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(image_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(RgbImageBatch {
        batch: image.batch,
        width: image.width,
        height: image.height,
        data,
    })
}

#[cfg(feature = "rocm")]
fn blend_latents_with_mask_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    generated: &LatentBatch,
    init: &LatentBatch,
    mask_weights: &[f32],
) -> DiffusionResult<LatentBatch> {
    if generated.batch != init.batch
        || generated.channels != init.channels
        || generated.height != init.height
        || generated.width != init.width
    {
        return Err(DiffusionError::InvalidRequest(format!(
            "generated latent shape [{}x{}x{}x{}] != init latent shape [{}x{}x{}x{}]",
            generated.batch,
            generated.channels,
            generated.height,
            generated.width,
            init.batch,
            init.channels,
            init.height,
            init.width
        )));
    }
    let expected_mask = generated.batch * generated.height * generated.width;
    if mask_weights.len() != expected_mask {
        return Err(DiffusionError::InvalidRequest(format!(
            "latent mask has {} weights, expected {expected_mask}",
            mask_weights.len()
        )));
    }
    let output_elements = generated
        .batch
        .checked_mul(generated.channels)
        .and_then(|elements| elements.checked_mul(generated.height))
        .and_then(|elements| elements.checked_mul(generated.width))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("latent output size overflows".to_string())
        })?;
    if generated.data.len() != output_elements || init.data.len() != output_elements {
        return Err(DiffusionError::InvalidRequest(format!(
            "latent data length mismatch for shape [{}x{}x{}x{}]",
            generated.batch, generated.channels, generated.height, generated.width
        )));
    }
    if output_elements == 0 {
        return Ok(generated.clone());
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("latent blend output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let generated_gpu = gpu
        .upload_f32(&generated.data, &[output_elements])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let init_gpu = gpu
        .upload_f32(&init.data, &[output_elements])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mask_gpu = gpu
        .upload_f32(mask_weights, &[mask_weights.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_blend_latents_with_mask_f32";
    let function_name = "diffusion_blend_latents_with_mask_f32";
    let kernel_source = DIFFUSION_INPAINT_MASK_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(generated_gpu.buf.as_ptr());
    kernargs.push_ptr(init_gpu.buf.as_ptr());
    kernargs.push_ptr(mask_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "latent blend output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("latent blend channels", generated.channels)?);
    kernargs.push_i32(i32_kernel_dim("latent blend height", generated.height)?);
    kernargs.push_i32(i32_kernel_dim("latent blend width", generated.width)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(LatentBatch {
        batch: generated.batch,
        channels: generated.channels,
        height: generated.height,
        width: generated.width,
        data,
    })
}

#[cfg(feature = "rocm")]
fn launch_diffusion_vector_kernel(
    gpu: &mut rdna_compute::Gpu,
    function_name: &str,
    source: &str,
    output_gpu: &hip_bridge::DeviceBuffer,
    input_a: &rdna_compute::GpuTensor,
    input_b: Option<&rdna_compute::GpuTensor>,
    n: i32,
    scalar: f32,
) -> DiffusionResult<()> {
    let module_name = function_name;
    let kernel_source = source;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_a.buf.as_ptr());
    if let Some(input_b) = input_b {
        kernargs.push_ptr(input_b.buf.as_ptr());
    }
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(n);
    kernargs.push_f32(scalar);
    kernargs.pad_to(16);
    let grid = [((n as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))
}

#[cfg(feature = "rocm")]
fn download_f32_buffer(
    gpu: &mut rdna_compute::Gpu,
    buffer: &hip_bridge::DeviceBuffer,
    elements: usize,
) -> DiffusionResult<Vec<f32>> {
    let output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| DiffusionError::InvalidMetadata("f32 output size overflows".to_string()))?;
    let mut raw = vec![0u8; output_bytes];
    gpu.hip
        .memcpy_dtoh(&mut raw, buffer)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut output = Vec::with_capacity(elements);
    for chunk in raw.chunks_exact(std::mem::size_of::<f32>()) {
        output.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(output)
}

#[cfg(feature = "rocm")]
fn scale_model_input_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    sample: &[f32],
    scale: f32,
) -> DiffusionResult<Vec<f32>> {
    if sample.is_empty() {
        return Ok(Vec::new());
    }
    let n = i32::try_from(sample.len()).map_err(|_| {
        DiffusionError::InvalidRequest(format!("model input length {} exceeds i32", sample.len()))
    })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let sample_gpu = gpu
        .upload_f32(sample, &[sample.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_bytes = sample
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("model input output size overflows".to_string())
        })?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_vector_kernel(
        gpu,
        "diffusion_scale_model_input_f32",
        DIFFUSION_DENOISE_VECTOR_HIP_SRC,
        &output_gpu,
        &sample_gpu,
        None,
        n,
        scale,
    )?;
    let output = download_f32_buffer(gpu, &output_gpu, sample.len())?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(output)
}

#[cfg(feature = "rocm")]
fn cfg_guidance_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    negative_pred: &[f32],
    positive_pred: &[f32],
    cfg_scale: f32,
) -> DiffusionResult<Vec<f32>> {
    if negative_pred.len() != positive_pred.len() {
        return Err(DiffusionError::InvalidRequest(format!(
            "negative prediction length {} != positive prediction length {}",
            negative_pred.len(),
            positive_pred.len()
        )));
    }
    if negative_pred.is_empty() {
        return Ok(Vec::new());
    }
    let n = i32::try_from(negative_pred.len()).map_err(|_| {
        DiffusionError::InvalidRequest(format!(
            "CFG prediction length {} exceeds i32",
            negative_pred.len()
        ))
    })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let negative_gpu = gpu
        .upload_f32(negative_pred, &[negative_pred.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let positive_gpu = gpu
        .upload_f32(positive_pred, &[positive_pred.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_bytes = negative_pred
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| DiffusionError::InvalidMetadata("CFG output size overflows".to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_vector_kernel(
        gpu,
        "diffusion_cfg_guidance_f32",
        DIFFUSION_DENOISE_VECTOR_HIP_SRC,
        &output_gpu,
        &negative_gpu,
        Some(&positive_gpu),
        n,
        cfg_scale,
    )?;
    let output = download_f32_buffer(gpu, &output_gpu, negative_pred.len())?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(output)
}

#[cfg(feature = "rocm")]
fn tensor_add_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    a: &CpuTensor,
    b: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    if a.shape != b.shape {
        return Err(DiffusionError::InvalidMetadata(format!(
            "tensor_add shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    if a.data.is_empty() {
        return Ok(CpuTensor::zeros(&a.shape));
    }
    let n = i32::try_from(a.data.len()).map_err(|_| {
        DiffusionError::InvalidRequest(format!("tensor_add length {} exceeds i32", a.data.len()))
    })?;
    let output_bytes = a
        .data
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("tensor_add output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let a_gpu = gpu
        .upload_f32(&a.data, &a.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let b_gpu = gpu
        .upload_f32(&b.data, &b.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_vector_kernel(
        gpu,
        "diffusion_tensor_add_f32",
        DIFFUSION_DENOISE_VECTOR_HIP_SRC,
        &output_gpu,
        &a_gpu,
        Some(&b_gpu),
        n,
        0.0,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, a.data.len())?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: a.shape.clone(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn maybe_center_unet_input_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    sample: &CpuTensor,
    center_input_sample: bool,
) -> DiffusionResult<CpuTensor> {
    if !center_input_sample {
        return Ok(sample.clone());
    }
    if sample.data.is_empty() {
        return Ok(CpuTensor::zeros(&sample.shape));
    }
    let n = i32::try_from(sample.data.len()).map_err(|_| {
        DiffusionError::InvalidRequest(format!(
            "UNet input length {} exceeds i32",
            sample.data.len()
        ))
    })?;
    let output_bytes = sample
        .data
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("UNet centered input size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let sample_gpu = gpu
        .upload_f32(&sample.data, &sample.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_vector_kernel(
        gpu,
        "diffusion_center_unet_input_f32",
        DIFFUSION_DENOISE_VECTOR_HIP_SRC,
        &output_gpu,
        &sample_gpu,
        None,
        n,
        0.0,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, sample.data.len())?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: sample.shape.clone(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn i32_kernel_dim(label: &str, value: usize) -> DiffusionResult<i32> {
    i32::try_from(value)
        .map_err(|_| DiffusionError::InvalidRequest(format!("{label} value {value} exceeds i32")))
}

#[cfg(feature = "rocm")]
fn launch_diffusion_layout_kernel(
    gpu: &mut rdna_compute::Gpu,
    function_name: &str,
    input_gpu: &rdna_compute::GpuTensor,
    bias_gpu: Option<&rdna_compute::GpuTensor>,
    output_gpu: &hip_bridge::DeviceBuffer,
    output_elements: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> DiffusionResult<()> {
    let module_name = function_name;
    let kernel_source = DIFFUSION_LAYOUT_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    if let Some(bias_gpu) = bias_gpu {
        kernargs.push_ptr(bias_gpu.buf.as_ptr());
    }
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("layout output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("layout channels", channels)?);
    kernargs.push_i32(i32_kernel_dim("layout height", height)?);
    kernargs.push_i32(i32_kernel_dim("layout width", width)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))
}

#[cfg(feature = "rocm")]
fn add_channel_bias_nchw_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    input: &CpuTensor,
    bias: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, channels, height, width] = shape4(input)?;
    if bias.shape.as_slice() != [batch, channels] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "channel bias shape {:?} != [{batch}, {channels}]",
            bias.shape
        )));
    }
    let output_elements = checked_shape_elements("channel-bias output", &input.shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&input.shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("channel-bias output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let bias_gpu = gpu
        .upload_f32(&bias.data, &bias.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_layout_kernel(
        gpu,
        "diffusion_add_channel_bias_nchw_f32",
        &input_gpu,
        Some(&bias_gpu),
        &output_gpu,
        output_elements,
        channels,
        height,
        width,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: input.shape.clone(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn nchw_to_bsc_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    input: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, channels, height, width] = shape4(input)?;
    let seq = height
        .checked_mul(width)
        .ok_or_else(|| DiffusionError::InvalidMetadata("BSC sequence overflows".to_string()))?;
    let output_shape = [batch, seq, channels];
    let output_elements = checked_shape_elements("NCHW-to-BSC output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("NCHW-to-BSC output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_layout_kernel(
        gpu,
        "diffusion_nchw_to_bsc_f32",
        &input_gpu,
        None,
        &output_gpu,
        output_elements,
        channels,
        height,
        width,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn bsc_to_nchw_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    input: &CpuTensor,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> DiffusionResult<CpuTensor> {
    let [input_batch, seq, input_channels] = shape3(input)?;
    if input_batch != batch || input_channels != channels || seq != height * width {
        return Err(DiffusionError::InvalidMetadata(format!(
            "BSC tensor shape {:?} cannot reshape to [{batch}, {channels}, {height}, {width}]",
            input.shape
        )));
    }
    let output_shape = [batch, channels, height, width];
    let output_elements = checked_shape_elements("BSC-to-NCHW output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("BSC-to-NCHW output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_layout_kernel(
        gpu,
        "diffusion_bsc_to_nchw_f32",
        &input_gpu,
        None,
        &output_gpu,
        output_elements,
        channels,
        height,
        width,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn launch_diffusion_concat_kernel(
    gpu: &mut rdna_compute::Gpu,
    function_name: &str,
    a_gpu: &rdna_compute::GpuTensor,
    b_gpu: &rdna_compute::GpuTensor,
    output_gpu: &hip_bridge::DeviceBuffer,
    kernargs_tail: impl FnOnce(&mut hip_bridge::KernargBlob) -> DiffusionResult<()>,
    output_elements: usize,
) -> DiffusionResult<()> {
    let module_name = function_name;
    let kernel_source = DIFFUSION_CONCAT_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(a_gpu.buf.as_ptr());
    kernargs.push_ptr(b_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("concat output elements", output_elements)?);
    kernargs_tail(&mut kernargs)?;
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))
}

#[cfg(feature = "rocm")]
fn concat_channels_nchw_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    a: &CpuTensor,
    b: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, a_channels, height, width] = shape4(a)?;
    let [b_batch, b_channels, b_height, b_width] = shape4(b)?;
    if batch != b_batch || height != b_height || width != b_width {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate NCHW tensors with shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let out_channels = a_channels.checked_add(b_channels).ok_or_else(|| {
        DiffusionError::InvalidMetadata("concat channel count overflows".to_string())
    })?;
    let output_shape = [batch, out_channels, height, width];
    let output_elements = checked_shape_elements("NCHW channel concat output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("NCHW channel concat output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let a_gpu = gpu
        .upload_f32(&a.data, &a.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let b_gpu = gpu
        .upload_f32(&b.data, &b.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_concat_kernel(
        gpu,
        "diffusion_concat_channels_nchw_f32",
        &a_gpu,
        &b_gpu,
        &output_gpu,
        |kernargs| {
            kernargs.push_i32(i32_kernel_dim("concat left channels", a_channels)?);
            kernargs.push_i32(i32_kernel_dim("concat right channels", b_channels)?);
            kernargs.push_i32(i32_kernel_dim("concat height", height)?);
            kernargs.push_i32(i32_kernel_dim("concat width", width)?);
            Ok(())
        },
        output_elements,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn concat_last_dim_2d_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    a: &CpuTensor,
    b: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [rows, left_width] = shape2(a)?;
    let [b_rows, right_width] = shape2(b)?;
    if rows != b_rows {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate 2D tensors with shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let output_width = left_width.checked_add(right_width).ok_or_else(|| {
        DiffusionError::InvalidMetadata("2D concat output width overflows".to_string())
    })?;
    concat_last_dim_hip_on_gpu(gpu, a, b, &[rows, output_width], left_width, right_width)
}

#[cfg(feature = "rocm")]
fn concat_last_dim_3d_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    a: &CpuTensor,
    b: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, left_width] = shape3(a)?;
    let [b_batch, b_seq, right_width] = shape3(b)?;
    if batch != b_batch || seq != b_seq {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate 3D tensors with shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let output_width = left_width.checked_add(right_width).ok_or_else(|| {
        DiffusionError::InvalidMetadata("3D concat output width overflows".to_string())
    })?;
    concat_last_dim_hip_on_gpu(
        gpu,
        a,
        b,
        &[batch, seq, output_width],
        left_width,
        right_width,
    )
}

#[cfg(feature = "rocm")]
fn concat_last_dim_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    a: &CpuTensor,
    b: &CpuTensor,
    output_shape: &[usize],
    left_width: usize,
    right_width: usize,
) -> DiffusionResult<CpuTensor> {
    let output_elements = checked_shape_elements("last-dim concat output", output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("last-dim concat output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let a_gpu = gpu
        .upload_f32(&a.data, &a.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let b_gpu = gpu
        .upload_f32(&b.data, &b.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_concat_kernel(
        gpu,
        "diffusion_concat_last_dim_f32",
        &a_gpu,
        &b_gpu,
        &output_gpu,
        |kernargs| {
            kernargs.push_i32(i32_kernel_dim("concat left width", left_width)?);
            kernargs.push_i32(i32_kernel_dim("concat right width", right_width)?);
            Ok(())
        },
        output_elements,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn conv2d_nchw_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    padding: usize,
    stride: usize,
) -> DiffusionResult<CpuTensor> {
    if stride == 0 {
        return Err(DiffusionError::InvalidRequest(
            "conv2d stride must be positive".to_string(),
        ));
    }
    let [batch, in_channels, in_h, in_w] = shape4(input)?;
    let [out_channels, weight_in_channels, kernel_h, kernel_w] = shape4(weight)?;
    if in_channels != weight_in_channels {
        return Err(DiffusionError::InvalidMetadata(format!(
            "conv2d input channels {in_channels} != weight input channels {weight_in_channels}"
        )));
    }
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [out_channels] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "conv2d bias shape {:?} != [{out_channels}]",
                bias.shape
            )));
        }
    }
    let padded_h = in_h + 2 * padding;
    let padded_w = in_w + 2 * padding;
    if kernel_h > padded_h || kernel_w > padded_w {
        return Err(DiffusionError::InvalidMetadata(format!(
            "conv2d kernel [{kernel_h}, {kernel_w}] is larger than padded input [{padded_h}, {padded_w}]"
        )));
    }
    let out_h = (padded_h - kernel_h) / stride + 1;
    let out_w = (padded_w - kernel_w) / stride + 1;
    let output_elements =
        checked_shape_elements("conv2d output", &[batch, out_channels, out_h, out_w])?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&[batch, out_channels, out_h, out_w]));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("conv2d output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let weight_gpu = gpu
        .upload_f32(&weight.data, &weight.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let bias_gpu = bias
        .map(|bias| gpu.upload_f32(&bias.data, &bias.shape))
        .transpose()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_conv2d_nchw_f32";
    let function_name = "diffusion_conv2d_nchw_f32";
    let kernel_source = DIFFUSION_CONV2D_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(weight_gpu.buf.as_ptr());
    if let Some(bias_gpu) = bias_gpu.as_ref() {
        kernargs.push_ptr(bias_gpu.buf.as_ptr());
    } else {
        kernargs.push_ptr(std::ptr::null());
    }
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("conv2d output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("conv2d batch", batch)?);
    kernargs.push_i32(i32_kernel_dim("conv2d input channels", in_channels)?);
    kernargs.push_i32(i32_kernel_dim("conv2d input height", in_h)?);
    kernargs.push_i32(i32_kernel_dim("conv2d input width", in_w)?);
    kernargs.push_i32(i32_kernel_dim("conv2d output channels", out_channels)?);
    kernargs.push_i32(i32_kernel_dim("conv2d output height", out_h)?);
    kernargs.push_i32(i32_kernel_dim("conv2d output width", out_w)?);
    kernargs.push_i32(i32_kernel_dim("conv2d kernel height", kernel_h)?);
    kernargs.push_i32(i32_kernel_dim("conv2d kernel width", kernel_w)?);
    kernargs.push_i32(i32_kernel_dim("conv2d padding", padding)?);
    kernargs.push_i32(i32_kernel_dim("conv2d stride", stride)?);
    kernargs.push_i32(if bias.is_some() { 1 } else { 0 });
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: vec![batch, out_channels, out_h, out_w],
        data,
    })
}

#[cfg(feature = "rocm")]
fn group_norm_nchw_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    groups: usize,
    eps: f32,
) -> DiffusionResult<CpuTensor> {
    let [_batch, channels, height, width] = shape4(input)?;
    if groups == 0 || channels % groups != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "group_norm channels {channels} not divisible by groups {groups}"
        )));
    }
    if weight.shape.as_slice() != [channels] || bias.shape.as_slice() != [channels] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "group_norm weight/bias shapes {:?}/{:?} != [{channels}]",
            weight.shape, bias.shape
        )));
    }
    let output_elements = checked_shape_elements("group_norm output", &input.shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&input.shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("group_norm output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let weight_gpu = gpu
        .upload_f32(&weight.data, &weight.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let bias_gpu = gpu
        .upload_f32(&bias.data, &bias.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_group_norm_nchw_f32";
    let function_name = "diffusion_group_norm_nchw_f32";
    let kernel_source = DIFFUSION_GROUP_NORM_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(weight_gpu.buf.as_ptr());
    kernargs.push_ptr(bias_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "group_norm output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("group_norm channels", channels)?);
    kernargs.push_i32(i32_kernel_dim("group_norm height", height)?);
    kernargs.push_i32(i32_kernel_dim("group_norm width", width)?);
    kernargs.push_i32(i32_kernel_dim("group_norm groups", groups)?);
    kernargs.push_f32(eps);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: input.shape.clone(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn silu_hip_on_gpu(gpu: &mut rdna_compute::Gpu, input: &CpuTensor) -> DiffusionResult<CpuTensor> {
    let elements = checked_shape_elements("SiLU input", &input.shape)?;
    if elements == 0 {
        return Ok(CpuTensor::zeros(&input.shape));
    }
    let n = i32_kernel_dim("SiLU elements", elements)?;
    let output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| DiffusionError::InvalidMetadata("SiLU output size overflows".to_string()))?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_silu_f32";
    let function_name = "diffusion_silu_f32";
    let kernel_source = DIFFUSION_SILU_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(n);
    kernargs.pad_to(16);
    let grid = [((elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: input.shape.clone(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn quick_gelu_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    input: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let elements = checked_shape_elements("QuickGELU input", &input.shape)?;
    if elements == 0 {
        return Ok(CpuTensor::zeros(&input.shape));
    }
    let n = i32_kernel_dim("QuickGELU elements", elements)?;
    let output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("QuickGELU output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_quick_gelu_f32";
    let function_name = "diffusion_quick_gelu_f32";
    let kernel_source = DIFFUSION_QUICK_GELU_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(n);
    kernargs.pad_to(16);
    let grid = [((elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: input.shape.clone(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn clip_token_position_embeddings_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    token_embedding: &CpuTensor,
    position_embedding: &CpuTensor,
    tokens: &[u32],
) -> DiffusionResult<CpuTensor> {
    let (vocab, hidden) = token_embedding.rows_cols()?;
    let (max_positions, position_hidden) = position_embedding.rows_cols()?;
    if position_hidden != hidden {
        return Err(DiffusionError::InvalidMetadata(format!(
            "CLIP position embedding hidden size {position_hidden} != token hidden size {hidden}"
        )));
    }
    if tokens.len() > max_positions {
        return Err(DiffusionError::InvalidRequest(format!(
            "CLIP token length {} exceeds position embedding length {max_positions}",
            tokens.len()
        )));
    }
    for &token in tokens {
        let token = token as usize;
        if token >= vocab {
            return Err(DiffusionError::InvalidRequest(format!(
                "CLIP token id {token} exceeds vocab {vocab}"
            )));
        }
    }
    let output_shape = [tokens.len(), hidden];
    let output_elements =
        checked_shape_elements("CLIP token-position embedding output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "CLIP token-position embedding output size overflows".to_string(),
            )
        })?;
    let token_bytes = tokens
        .iter()
        .flat_map(|token| token.to_ne_bytes())
        .collect::<Vec<_>>();
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let token_embedding_gpu = gpu
        .upload_f32(&token_embedding.data, &token_embedding.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let position_embedding_gpu = gpu
        .upload_f32(&position_embedding.data, &position_embedding.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let tokens_gpu = gpu
        .upload_raw(&token_bytes, &[tokens.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_clip_token_position_embedding_f32";
    let function_name = "diffusion_clip_token_position_embedding_f32";
    let kernel_source = DIFFUSION_CLIP_EMBEDDINGS_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(token_embedding_gpu.buf.as_ptr());
    kernargs.push_ptr(position_embedding_gpu.buf.as_ptr());
    kernargs.push_ptr(tokens_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "CLIP token-position embedding output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim(
        "CLIP token-position embedding hidden size",
        hidden,
    )?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn upsample_nearest2d_nchw_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    input: &CpuTensor,
    scale: usize,
) -> DiffusionResult<CpuTensor> {
    if scale == 0 {
        return Err(DiffusionError::InvalidRequest(
            "upsample scale must be positive".to_string(),
        ));
    }
    let [batch, channels, in_h, in_w] = shape4(input)?;
    let out_h = in_h.checked_mul(scale).ok_or_else(|| {
        DiffusionError::InvalidRequest("upsample output height overflows".to_string())
    })?;
    let out_w = in_w.checked_mul(scale).ok_or_else(|| {
        DiffusionError::InvalidRequest("upsample output width overflows".to_string())
    })?;
    let output_shape = [batch, channels, out_h, out_w];
    let output_elements = checked_shape_elements("upsample output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("upsample output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_upsample_nearest2d_nchw_f32";
    let function_name = "diffusion_upsample_nearest2d_nchw_f32";
    let kernel_source = DIFFUSION_UPSAMPLE_NEAREST2D_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("upsample output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("upsample channels", channels)?);
    kernargs.push_i32(i32_kernel_dim("upsample input height", in_h)?);
    kernargs.push_i32(i32_kernel_dim("upsample input width", in_w)?);
    kernargs.push_i32(i32_kernel_dim("upsample output height", out_h)?);
    kernargs.push_i32(i32_kernel_dim("upsample output width", out_w)?);
    kernargs.push_i32(i32_kernel_dim("upsample scale", scale)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn linear_optional_bias_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
) -> DiffusionResult<CpuTensor> {
    let (rows, in_features) = input.rows_cols()?;
    let (out_features, weight_in) = weight.rows_cols()?;
    if in_features != weight_in {
        return Err(DiffusionError::InvalidMetadata(format!(
            "linear input width {in_features} != weight input width {weight_in}"
        )));
    }
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [out_features] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "linear bias shape {:?} != [{out_features}]",
                bias.shape
            )));
        }
    }
    let output_shape = [rows, out_features];
    let output_elements = checked_shape_elements("linear output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("linear output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let weight_gpu = gpu
        .upload_f32(&weight.data, &weight.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let bias_gpu = bias
        .map(|bias| gpu.upload_f32(&bias.data, &bias.shape))
        .transpose()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_linear_f32";
    let function_name = "diffusion_linear_f32";
    let kernel_source = DIFFUSION_LINEAR_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(weight_gpu.buf.as_ptr());
    if let Some(bias_gpu) = bias_gpu.as_ref() {
        kernargs.push_ptr(bias_gpu.buf.as_ptr());
    } else {
        kernargs.push_ptr(std::ptr::null());
    }
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("linear output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("linear input features", in_features)?);
    kernargs.push_i32(i32_kernel_dim("linear output features", out_features)?);
    kernargs.push_i32(if bias.is_some() { 1 } else { 0 });
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn layer_norm_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    eps: f32,
) -> DiffusionResult<CpuTensor> {
    let (rows, cols) = input.rows_cols()?;
    if weight.shape.as_slice() != [cols] || bias.shape.as_slice() != [cols] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "layer_norm weight/bias shapes {:?}/{:?} do not match width {cols}",
            weight.shape, bias.shape
        )));
    }
    let output_shape = [rows, cols];
    let output_elements = checked_shape_elements("layer_norm output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("layer_norm output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let weight_gpu = gpu
        .upload_f32(&weight.data, &weight.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let bias_gpu = gpu
        .upload_f32(&bias.data, &bias.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_layer_norm_f32";
    let function_name = "diffusion_layer_norm_f32";
    let kernel_source = DIFFUSION_LAYER_NORM_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(weight_gpu.buf.as_ptr());
    kernargs.push_ptr(bias_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "layer_norm output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("layer_norm width", cols)?);
    kernargs.push_f32(eps);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn softmax_rows_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    input: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let (rows, cols) = input.rows_cols()?;
    let output_elements = checked_shape_elements("softmax output", &input.shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&input.shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("softmax output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_softmax_rows_f32";
    let function_name = "diffusion_softmax_rows_f32";
    let kernel_source = DIFFUSION_SOFTMAX_ROWS_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("softmax rows", rows)?);
    kernargs.push_i32(i32_kernel_dim("softmax cols", cols)?);
    kernargs.pad_to(16);
    let grid = [((rows as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: input.shape.clone(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn scaled_dot_product_attention_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    heads: usize,
) -> DiffusionResult<CpuTensor> {
    let [batch, q_seq, hidden] = shape3(q)?;
    let [k_batch, k_seq, k_hidden] = shape3(k)?;
    let [v_batch, v_seq, v_hidden] = shape3(v)?;
    if batch != k_batch || batch != v_batch || k_seq != v_seq || k_hidden != v_hidden {
        return Err(DiffusionError::InvalidMetadata(format!(
            "attention q/k/v shapes {:?}/{:?}/{:?} are incompatible",
            q.shape, k.shape, v.shape
        )));
    }
    if heads == 0 || hidden != k_hidden || hidden % heads != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "attention hidden size {hidden} is incompatible with key size {k_hidden} and heads {heads}"
        )));
    }
    let head_dim = hidden / heads;
    let output_shape = [batch, q_seq, hidden];
    let output_elements = checked_shape_elements("SDPA output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| DiffusionError::InvalidMetadata("SDPA output size overflows".to_string()))?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let q_gpu = gpu
        .upload_f32(&q.data, &q.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let k_gpu = gpu
        .upload_f32(&k.data, &k.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let v_gpu = gpu
        .upload_f32(&v.data, &v.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_sdpa_3d_f32";
    let function_name = "diffusion_sdpa_3d_f32";
    let kernel_source = DIFFUSION_SDPA_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(q_gpu.buf.as_ptr());
    kernargs.push_ptr(k_gpu.buf.as_ptr());
    kernargs.push_ptr(v_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("SDPA output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("SDPA query sequence", q_seq)?);
    kernargs.push_i32(i32_kernel_dim("SDPA key sequence", k_seq)?);
    kernargs.push_i32(i32_kernel_dim("SDPA hidden size", hidden)?);
    kernargs.push_i32(i32_kernel_dim("SDPA heads", heads)?);
    kernargs.push_i32(i32_kernel_dim("SDPA head dim", head_dim)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn clip_causal_self_attention_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    n_heads: usize,
) -> DiffusionResult<CpuTensor> {
    let (seq, hidden) = q.rows_cols()?;
    if k.shape.as_slice() != [seq, hidden] || v.shape.as_slice() != [seq, hidden] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "CLIP causal attention q/k/v shapes {:?}/{:?}/{:?} are incompatible",
            q.shape, k.shape, v.shape
        )));
    }
    if n_heads == 0 || hidden % n_heads != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "CLIP hidden size {hidden} is not divisible by {n_heads} heads"
        )));
    }
    let head_dim = hidden / n_heads;
    let output_shape = [seq, hidden];
    let output_elements = checked_shape_elements("CLIP causal attention output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "CLIP causal attention output size overflows".to_string(),
            )
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let q_gpu = gpu
        .upload_f32(&q.data, &q.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let k_gpu = gpu
        .upload_f32(&k.data, &k.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let v_gpu = gpu
        .upload_f32(&v.data, &v.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_clip_causal_attention_f32";
    let function_name = "diffusion_clip_causal_attention_f32";
    let kernel_source = DIFFUSION_CLIP_CAUSAL_ATTENTION_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(q_gpu.buf.as_ptr());
    kernargs.push_ptr(k_gpu.buf.as_ptr());
    kernargs.push_ptr(v_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "CLIP causal attention output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention sequence", seq)?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention hidden size", hidden)?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention heads", n_heads)?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention head dim", head_dim)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn geglu_gate_3d_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    projected: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, width] = shape3(projected)?;
    if width % 2 != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "GEGLU projection width {width} is not even"
        )));
    }
    let inner = width / 2;
    let output_shape = [batch, seq, inner];
    let output_elements = checked_shape_elements("GeGLU gate output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("GeGLU gate output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&projected.data, &projected.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_geglu_gate_3d_f32";
    let function_name = "diffusion_geglu_gate_3d_f32";
    let kernel_source = DIFFUSION_GEGLU_GATE_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "GeGLU gate output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("GeGLU gate inner width", inner)?);
    kernargs.push_i32(i32_kernel_dim("GeGLU gate projected width", width)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn timestep_embedding_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    timesteps: &[f32],
    dim: usize,
    flip_sin_to_cos: bool,
    freq_shift: f32,
) -> DiffusionResult<CpuTensor> {
    if dim == 0 {
        return Err(DiffusionError::InvalidRequest(
            "timestep embedding dimension must be positive".to_string(),
        ));
    }
    let output_shape = [timesteps.len(), dim];
    let output_elements = checked_shape_elements("timestep embedding output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("timestep embedding output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let timesteps_gpu = gpu
        .upload_f32(timesteps, &[timesteps.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_timestep_embedding_f32";
    let function_name = "diffusion_timestep_embedding_f32";
    let kernel_source = DIFFUSION_TIMESTEP_EMBEDDING_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(timesteps_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "timestep embedding output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("timestep embedding dimension", dim)?);
    kernargs.push_i32(i32_kernel_dim(
        "timestep embedding half dimension",
        dim / 2,
    )?);
    kernargs.push_i32(if flip_sin_to_cos { 1 } else { 0 });
    kernargs.push_f32(freq_shift);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[cfg(feature = "rocm")]
fn scheduler_prediction_type_id(prediction_type: SchedulerPredictionType) -> i32 {
    match prediction_type {
        SchedulerPredictionType::Epsilon => 0,
        SchedulerPredictionType::Sample => 1,
        SchedulerPredictionType::VPrediction => 2,
    }
}

#[cfg(feature = "rocm")]
fn euler_step_hip_on_gpu(
    gpu: &mut rdna_compute::Gpu,
    sample: &[f32],
    model_output: &[f32],
    sigma: f32,
    next_sigma: f32,
    prediction_type: SchedulerPredictionType,
) -> DiffusionResult<Vec<f32>> {
    if sample.len() != model_output.len() {
        return Err(DiffusionError::InvalidRequest(format!(
            "sample length {} != model output length {}",
            sample.len(),
            model_output.len()
        )));
    }
    if sample.is_empty() {
        return Ok(Vec::new());
    }
    let n = i32::try_from(sample.len()).map_err(|_| {
        DiffusionError::InvalidRequest(format!(
            "scheduler input length {} exceeds i32",
            sample.len()
        ))
    })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let sample_gpu = gpu
        .upload_f32(sample, &[sample.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let model_output_gpu = gpu
        .upload_f32(model_output, &[model_output.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_bytes = sample
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("scheduler output size overflows".to_string())
        })?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_euler_step_f32";
    let function_name = "diffusion_euler_step_f32";
    let kernel_source = DIFFUSION_EULER_STEP_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(sample_gpu.buf.as_ptr());
    kernargs.push_ptr(model_output_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(n);
    kernargs.push_f32(sigma);
    kernargs.push_f32(next_sigma);
    kernargs.push_i32(scheduler_prediction_type_id(prediction_type));
    kernargs.pad_to(16);
    let grid = [((sample.len() as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut raw = vec![0u8; output_bytes];
    gpu.hip
        .memcpy_dtoh(&mut raw, &output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut output = Vec::with_capacity(sample.len());
    for chunk in raw.chunks_exact(std::mem::size_of::<f32>()) {
        output.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(output)
}

pub fn rgb_batch_to_vae_tensor(batch: &RgbImageBatch) -> DiffusionResult<CpuTensor> {
    let bytes_per_image = batch
        .width
        .checked_mul(batch.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("image dimensions overflow".to_string()))?;
    let expected = bytes_per_image
        .checked_mul(batch.batch)
        .ok_or_else(|| DiffusionError::InvalidRequest("image batch size overflows".to_string()))?;
    if batch.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "RGB image batch has {} bytes, expected {expected}",
            batch.data.len()
        )));
    }
    let mut out = CpuTensor::zeros(&[batch.batch, 3, batch.height, batch.width]);
    for b in 0..batch.batch {
        let image_base = b * bytes_per_image;
        for y in 0..batch.height {
            for x in 0..batch.width {
                let rgb_base = image_base + (y * batch.width + x) * 3;
                for c in 0..3 {
                    out.data[nchw_idx(b, c, y, x, 3, batch.height, batch.width)] =
                        batch.data[rgb_base + c] as f32 / 127.5 - 1.0;
                }
            }
        }
    }
    Ok(out)
}

pub fn encode_rgb_batch_png_base64(batch: &RgbImageBatch) -> DiffusionResult<Vec<String>> {
    let width = u32::try_from(batch.width).map_err(|_| {
        DiffusionError::InvalidRequest(format!("image width {} exceeds u32", batch.width))
    })?;
    let height = u32::try_from(batch.height).map_err(|_| {
        DiffusionError::InvalidRequest(format!("image height {} exceeds u32", batch.height))
    })?;
    let bytes_per_image = batch
        .width
        .checked_mul(batch.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("image dimensions overflow".to_string()))?;
    let expected = bytes_per_image
        .checked_mul(batch.batch)
        .ok_or_else(|| DiffusionError::InvalidRequest("image batch size overflows".to_string()))?;
    if batch.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "RGB image batch has {} bytes, expected {expected}",
            batch.data.len()
        )));
    }

    let mut encoded = Vec::with_capacity(batch.batch);
    for idx in 0..batch.batch {
        let start = idx * bytes_per_image;
        let end = start + bytes_per_image;
        let mut png = Vec::new();
        PngEncoder::new(&mut png)
            .write_image(
                &batch.data[start..end],
                width,
                height,
                ColorType::Rgb8.into(),
            )
            .map_err(|err| DiffusionError::Io(format!("PNG encode failed: {err}")))?;
        encoded.push(base64::engine::general_purpose::STANDARD.encode(png));
    }
    Ok(encoded)
}

fn nchw_to_bsc(input: &CpuTensor) -> DiffusionResult<CpuTensor> {
    let [batch, channels, height, width] = shape4(input)?;
    let seq = height * width;
    let mut out = CpuTensor::zeros(&[batch, seq, channels]);
    for b in 0..batch {
        for y in 0..height {
            for x in 0..width {
                let s = y * width + x;
                for c in 0..channels {
                    out.data[(b * seq + s) * channels + c] =
                        input.data[nchw_idx(b, c, y, x, channels, height, width)];
                }
            }
        }
    }
    Ok(out)
}

fn bsc_to_nchw(
    input: &CpuTensor,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> DiffusionResult<CpuTensor> {
    let [input_batch, seq, input_channels] = shape3(input)?;
    if input_batch != batch || input_channels != channels || seq != height * width {
        return Err(DiffusionError::InvalidMetadata(format!(
            "BSC tensor shape {:?} cannot reshape to [{batch}, {channels}, {height}, {width}]",
            input.shape
        )));
    }
    let mut out = CpuTensor::zeros(&[batch, channels, height, width]);
    for b in 0..batch {
        for y in 0..height {
            for x in 0..width {
                let s = y * width + x;
                for c in 0..channels {
                    out.data[nchw_idx(b, c, y, x, channels, height, width)] =
                        input.data[(b * seq + s) * channels + c];
                }
            }
        }
    }
    Ok(out)
}

fn optional_tensor(hfq: &HfqFile, entry: &str) -> DiffusionResult<Option<CpuTensor>> {
    if hfq.find_tensor_info(entry).is_some() {
        CpuTensor::from_hfq(hfq, entry).map(Some)
    } else {
        Ok(None)
    }
}

fn linear_3d(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, in_features] = shape3(input)?;
    let flat = CpuTensor {
        shape: vec![batch * seq, in_features],
        data: input.data.clone(),
    };
    let out = linear_optional_bias(&flat, weight, bias)?;
    let [rows, out_features] = shape2(&out)?;
    if rows != batch * seq {
        return Err(DiffusionError::InvalidMetadata(format!(
            "linear_3d row count {rows} != batch*seq {}",
            batch * seq
        )));
    }
    Ok(CpuTensor {
        shape: vec![batch, seq, out_features],
        data: out.data,
    })
}

fn layer_norm_3d(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    eps: f32,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, width] = shape3(input)?;
    let flat = CpuTensor {
        shape: vec![batch * seq, width],
        data: input.data.clone(),
    };
    let out = layer_norm(&flat, weight, bias, eps)?;
    Ok(CpuTensor {
        shape: vec![batch, seq, width],
        data: out.data,
    })
}

fn scaled_dot_product_attention(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    heads: usize,
) -> DiffusionResult<CpuTensor> {
    let [batch, q_seq, hidden] = shape3(q)?;
    let [k_batch, k_seq, k_hidden] = shape3(k)?;
    let [v_batch, v_seq, v_hidden] = shape3(v)?;
    if batch != k_batch || batch != v_batch || k_seq != v_seq || k_hidden != v_hidden {
        return Err(DiffusionError::InvalidMetadata(format!(
            "attention q/k/v shapes {:?}/{:?}/{:?} are incompatible",
            q.shape, k.shape, v.shape
        )));
    }
    if hidden != k_hidden || hidden % heads != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "attention hidden size {hidden} is incompatible with key size {k_hidden} and heads {heads}"
        )));
    }
    let head_dim = hidden / heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = CpuTensor::zeros(&[batch, q_seq, hidden]);
    for b in 0..batch {
        for head in 0..heads {
            let head_off = head * head_dim;
            for qi in 0..q_seq {
                let mut scores = vec![0.0f32; k_seq];
                for ki in 0..k_seq {
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += q.data[((b * q_seq + qi) * hidden) + head_off + d]
                            * k.data[((b * k_seq + ki) * hidden) + head_off + d];
                    }
                    scores[ki] = dot * scale;
                }
                softmax_in_place(&mut scores);
                for d in 0..head_dim {
                    let mut acc = 0.0;
                    for ki in 0..k_seq {
                        acc += scores[ki] * v.data[((b * k_seq + ki) * hidden) + head_off + d];
                    }
                    out.data[((b * q_seq + qi) * hidden) + head_off + d] = acc;
                }
            }
        }
    }
    Ok(out)
}

fn gelu(value: f32) -> f32 {
    0.5 * value
        * (1.0
            + (std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044_715 * value * value * value))
                .tanh())
}

#[derive(Debug, Clone)]
pub struct ClipTextEncoder {
    token_embedding: CpuTensor,
    position_embedding: CpuTensor,
    layers: Vec<ClipEncoderLayer>,
    final_layer_norm_weight: CpuTensor,
    final_layer_norm_bias: CpuTensor,
    text_projection: Option<CpuTensor>,
    hidden_size: usize,
    max_length: usize,
    n_heads: usize,
}

#[derive(Debug, Clone)]
struct ClipEncoderLayer {
    q_proj_weight: CpuTensor,
    q_proj_bias: CpuTensor,
    k_proj_weight: CpuTensor,
    k_proj_bias: CpuTensor,
    v_proj_weight: CpuTensor,
    v_proj_bias: CpuTensor,
    out_proj_weight: CpuTensor,
    out_proj_bias: CpuTensor,
    layer_norm1_weight: CpuTensor,
    layer_norm1_bias: CpuTensor,
    fc1_weight: CpuTensor,
    fc1_bias: CpuTensor,
    fc2_weight: CpuTensor,
    fc2_bias: CpuTensor,
    layer_norm2_weight: CpuTensor,
    layer_norm2_bias: CpuTensor,
}

impl ClipTextEncoder {
    pub fn from_hfq_file(hfq: &HfqFile) -> DiffusionResult<Self> {
        Self::from_hfq_file_with_heads(hfq, 12)
    }

    pub fn from_hfq_file_with_heads(hfq: &HfqFile, n_heads: usize) -> DiffusionResult<Self> {
        Self::from_hfq_file_with_prefix_and_heads(hfq, "text_encoder", n_heads)
    }

    pub fn from_hfq_file_with_prefix_and_heads(
        hfq: &HfqFile,
        component: &str,
        n_heads: usize,
    ) -> DiffusionResult<Self> {
        let token_embedding = CpuTensor::from_hfq(
            hfq,
            &format!("{component}/tensors/text_model.embeddings.token_embedding.weight"),
        )?;
        let position_embedding = CpuTensor::from_hfq(
            hfq,
            &format!("{component}/tensors/text_model.embeddings.position_embedding.weight"),
        )?;
        let (_, hidden_size) = token_embedding.rows_cols()?;
        let (max_length, position_hidden) = position_embedding.rows_cols()?;
        if position_hidden != hidden_size {
            return Err(DiffusionError::InvalidMetadata(format!(
                "CLIP position embedding hidden size {position_hidden} != token hidden size {hidden_size}"
            )));
        }
        let mut layers = Vec::new();
        for layer_idx in 0.. {
            let prefix = format!("{component}/tensors/text_model.encoder.layers.{layer_idx}");
            if hfq
                .find_tensor_info(&format!("{prefix}.self_attn.q_proj.weight"))
                .is_none()
            {
                break;
            }
            layers.push(ClipEncoderLayer {
                q_proj_weight: CpuTensor::from_hfq(
                    hfq,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                )?,
                q_proj_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.self_attn.q_proj.bias"))?,
                k_proj_weight: CpuTensor::from_hfq(
                    hfq,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                )?,
                k_proj_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.self_attn.k_proj.bias"))?,
                v_proj_weight: CpuTensor::from_hfq(
                    hfq,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                )?,
                v_proj_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.self_attn.v_proj.bias"))?,
                out_proj_weight: CpuTensor::from_hfq(
                    hfq,
                    &format!("{prefix}.self_attn.out_proj.weight"),
                )?,
                out_proj_bias: CpuTensor::from_hfq(
                    hfq,
                    &format!("{prefix}.self_attn.out_proj.bias"),
                )?,
                layer_norm1_weight: CpuTensor::from_hfq(
                    hfq,
                    &format!("{prefix}.layer_norm1.weight"),
                )?,
                layer_norm1_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.layer_norm1.bias"))?,
                fc1_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.mlp.fc1.weight"))?,
                fc1_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.mlp.fc1.bias"))?,
                fc2_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.mlp.fc2.weight"))?,
                fc2_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.mlp.fc2.bias"))?,
                layer_norm2_weight: CpuTensor::from_hfq(
                    hfq,
                    &format!("{prefix}.layer_norm2.weight"),
                )?,
                layer_norm2_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.layer_norm2.bias"))?,
            });
        }
        if layers.is_empty() {
            return Err(DiffusionError::InvalidMetadata(
                "CLIP text encoder has no transformer layers".to_string(),
            ));
        }
        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm_weight: CpuTensor::from_hfq(
                hfq,
                &format!("{component}/tensors/text_model.final_layer_norm.weight"),
            )?,
            final_layer_norm_bias: CpuTensor::from_hfq(
                hfq,
                &format!("{component}/tensors/text_model.final_layer_norm.bias"),
            )?,
            text_projection: CpuTensor::from_hfq(
                hfq,
                &format!("{component}/tensors/text_projection.weight"),
            )
            .ok(),
            hidden_size,
            max_length,
            n_heads,
        })
    }

    pub fn encode_tokens(&self, tokens: &[u32]) -> DiffusionResult<CpuTensor> {
        self.encode_tokens_with_runtime_options(
            tokens,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn encode_tokens_with_runtime_options(
        &self,
        tokens: &[u32],
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.encode_tokens_with_runtime_context(tokens, &mut runtime_context)
    }

    fn encode_tokens_with_runtime_context(
        &self,
        tokens: &[u32],
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        self.encode_tokens_internal_with_runtime_context(tokens, runtime_context)
    }

    pub fn encode_tokens_with_pooled(
        &self,
        tokens: &[u32],
        end_token: u32,
    ) -> DiffusionResult<(CpuTensor, Option<Vec<f32>>)> {
        self.encode_tokens_with_pooled_and_runtime_options(
            tokens,
            end_token,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    fn encode_tokens_with_pooled_and_runtime_options(
        &self,
        tokens: &[u32],
        end_token: u32,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<(CpuTensor, Option<Vec<f32>>)> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.encode_tokens_with_pooled_and_runtime_context(tokens, end_token, &mut runtime_context)
    }

    fn encode_tokens_with_pooled_and_runtime_context(
        &self,
        tokens: &[u32],
        end_token: u32,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<(CpuTensor, Option<Vec<f32>>)> {
        let hidden_states =
            self.encode_tokens_internal_with_runtime_context(tokens, runtime_context)?;
        let pooled = self.pooled_text_embedding_with_runtime_context(
            &hidden_states,
            tokens,
            end_token,
            runtime_context,
        )?;
        Ok((hidden_states, Some(pooled)))
    }

    fn encode_tokens_internal_with_runtime_context(
        &self,
        tokens: &[u32],
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        if tokens.len() > self.max_length {
            return Err(DiffusionError::InvalidRequest(format!(
                "CLIP token length {} exceeds max_length {}",
                tokens.len(),
                self.max_length
            )));
        }
        let mut x = clip_token_position_embeddings_with_runtime_context(
            &self.token_embedding,
            &self.position_embedding,
            tokens,
            runtime_context,
        )?;
        if x.shape.as_slice() != [tokens.len(), self.hidden_size] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "CLIP embedding output shape {:?} does not match [{}, {}]",
                x.shape,
                tokens.len(),
                self.hidden_size
            )));
        }
        for layer in &self.layers {
            x = layer.forward_with_runtime_context(&x, self.n_heads, runtime_context)?;
        }
        layer_norm_with_runtime_context(
            &x,
            &self.final_layer_norm_weight,
            &self.final_layer_norm_bias,
            1e-5,
            runtime_context,
        )
    }

    fn pooled_text_embedding_with_runtime_context(
        &self,
        hidden_states: &CpuTensor,
        tokens: &[u32],
        end_token: u32,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<Vec<f32>> {
        let [seq, hidden] = shape2(hidden_states)?;
        let token_idx = tokens
            .iter()
            .position(|token| *token == end_token)
            .unwrap_or_else(|| tokens.len().saturating_sub(1))
            .min(seq.saturating_sub(1));
        let base = token_idx * hidden;
        let pooled = hidden_states.data[base..base + hidden].to_vec();
        if let Some(projection) = &self.text_projection {
            matmul_vector_with_runtime_context(&pooled, projection, runtime_context)
        } else {
            Ok(pooled)
        }
    }
}

impl ClipEncoderLayer {
    fn forward_with_runtime_context(
        &self,
        x: &CpuTensor,
        n_heads: usize,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let norm1 = layer_norm_with_runtime_context(
            x,
            &self.layer_norm1_weight,
            &self.layer_norm1_bias,
            1e-5,
            runtime_context,
        )?;
        let attn = self.self_attention_with_runtime_context(&norm1, n_heads, runtime_context)?;
        let residual1 = tensor_add_with_runtime_context(x, &attn, runtime_context)?;
        let norm2 = layer_norm_with_runtime_context(
            &residual1,
            &self.layer_norm2_weight,
            &self.layer_norm2_bias,
            1e-5,
            runtime_context,
        )?;
        let hidden =
            linear_with_runtime_context(&norm2, &self.fc1_weight, &self.fc1_bias, runtime_context)?;
        let activated = quick_gelu_with_runtime_context(&hidden, runtime_context)?;
        let mlp = linear_with_runtime_context(
            &activated,
            &self.fc2_weight,
            &self.fc2_bias,
            runtime_context,
        )?;
        tensor_add_with_runtime_context(&residual1, &mlp, runtime_context)
    }

    fn self_attention_with_runtime_context(
        &self,
        x: &CpuTensor,
        n_heads: usize,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let q = linear_with_runtime_context(
            x,
            &self.q_proj_weight,
            &self.q_proj_bias,
            runtime_context,
        )?;
        let k = linear_with_runtime_context(
            x,
            &self.k_proj_weight,
            &self.k_proj_bias,
            runtime_context,
        )?;
        let v = linear_with_runtime_context(
            x,
            &self.v_proj_weight,
            &self.v_proj_bias,
            runtime_context,
        )?;
        let context =
            clip_causal_self_attention_with_runtime_context(&q, &k, &v, n_heads, runtime_context)?;
        linear_with_runtime_context(
            &context,
            &self.out_proj_weight,
            &self.out_proj_bias,
            runtime_context,
        )
    }
}

fn clip_causal_self_attention(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    n_heads: usize,
) -> DiffusionResult<CpuTensor> {
    let (seq, hidden) = q.rows_cols()?;
    if k.shape.as_slice() != [seq, hidden] || v.shape.as_slice() != [seq, hidden] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "CLIP causal attention q/k/v shapes {:?}/{:?}/{:?} are incompatible",
            q.shape, k.shape, v.shape
        )));
    }
    if n_heads == 0 || hidden % n_heads != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "CLIP hidden size {hidden} is not divisible by {n_heads} heads"
        )));
    }
    let head_dim = hidden / n_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut context = CpuTensor::zeros(&[seq, hidden]);
    for head in 0..n_heads {
        let head_off = head * head_dim;
        for i in 0..seq {
            let mut scores = vec![0.0f32; seq];
            for j in 0..seq {
                if j > i {
                    scores[j] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0;
                for d in 0..head_dim {
                    dot += q.data[i * hidden + head_off + d] * k.data[j * hidden + head_off + d];
                }
                scores[j] = dot * scale;
            }
            softmax_in_place(&mut scores);
            for d in 0..head_dim {
                let mut acc = 0.0;
                for j in 0..seq {
                    acc += scores[j] * v.data[j * hidden + head_off + d];
                }
                context.data[i * hidden + head_off + d] = acc;
            }
        }
    }
    Ok(context)
}

fn linear(input: &CpuTensor, weight: &CpuTensor, bias: &CpuTensor) -> DiffusionResult<CpuTensor> {
    linear_optional_bias(input, weight, Some(bias))
}

fn linear_optional_bias(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
) -> DiffusionResult<CpuTensor> {
    let (rows, in_features) = input.rows_cols()?;
    let (out_features, weight_in) = weight.rows_cols()?;
    if in_features != weight_in {
        return Err(DiffusionError::InvalidMetadata(format!(
            "linear input width {in_features} != weight input width {weight_in}"
        )));
    }
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [out_features] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "linear bias shape {:?} != [{out_features}]",
                bias.shape
            )));
        }
    }
    let mut out = CpuTensor::zeros(&[rows, out_features]);
    for row in 0..rows {
        for out_col in 0..out_features {
            let mut acc = bias.map(|bias| bias.data[out_col]).unwrap_or(0.0);
            let weight_row = out_col * in_features;
            let input_row = row * in_features;
            for k in 0..in_features {
                acc += input.data[input_row + k] * weight.data[weight_row + k];
            }
            out.data[row * out_features + out_col] = acc;
        }
    }
    Ok(out)
}

fn layer_norm(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    eps: f32,
) -> DiffusionResult<CpuTensor> {
    let (rows, cols) = input.rows_cols()?;
    if weight.shape.as_slice() != [cols] || bias.shape.as_slice() != [cols] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "layer_norm weight/bias shapes {:?}/{:?} do not match width {cols}",
            weight.shape, bias.shape
        )));
    }
    let mut out = CpuTensor::zeros(&[rows, cols]);
    for row in 0..rows {
        let base = row * cols;
        let mean = input.data[base..base + cols].iter().sum::<f32>() / cols as f32;
        let var = input.data[base..base + cols]
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / cols as f32;
        let inv_std = (var + eps).sqrt().recip();
        for col in 0..cols {
            out.data[base + col] =
                (input.data[base + col] - mean) * inv_std * weight.data[col] + bias.data[col];
        }
    }
    Ok(out)
}

fn tensor_add(a: &CpuTensor, b: &CpuTensor) -> DiffusionResult<CpuTensor> {
    if a.shape != b.shape {
        return Err(DiffusionError::InvalidMetadata(format!(
            "tensor_add shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    Ok(CpuTensor {
        shape: a.shape.clone(),
        data: a.data.iter().zip(&b.data).map(|(a, b)| a + b).collect(),
    })
}

fn geglu_gate_3d(projected: &CpuTensor) -> DiffusionResult<CpuTensor> {
    let [batch, seq, width] = shape3(projected)?;
    if width % 2 != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "GEGLU projection width {width} is not even"
        )));
    }
    let inner = width / 2;
    let mut gated = CpuTensor::zeros(&[batch, seq, inner]);
    for b in 0..batch {
        for s in 0..seq {
            let src = (b * seq + s) * width;
            let dst = (b * seq + s) * inner;
            for col in 0..inner {
                let value = projected.data[src + col];
                let gate = gelu(projected.data[src + inner + col]);
                gated.data[dst + col] = value * gate;
            }
        }
    }
    Ok(gated)
}

fn tensor_map(input: &CpuTensor, f: impl Fn(f32) -> f32) -> CpuTensor {
    CpuTensor {
        shape: input.shape.clone(),
        data: input.data.iter().copied().map(f).collect(),
    }
}

fn add_channel_bias_nchw(input: &mut CpuTensor, bias: &CpuTensor) -> DiffusionResult<()> {
    let [batch, channels, height, width] = shape4(input)?;
    if bias.shape.as_slice() != [batch, channels] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "channel bias shape {:?} != [{batch}, {channels}]",
            bias.shape
        )));
    }
    for b in 0..batch {
        for c in 0..channels {
            let value = bias.data[b * channels + c];
            for y in 0..height {
                for x in 0..width {
                    input.data[nchw_idx(b, c, y, x, channels, height, width)] += value;
                }
            }
        }
    }
    Ok(())
}

fn concat_channels_nchw(a: &CpuTensor, b: &CpuTensor) -> DiffusionResult<CpuTensor> {
    let [batch, a_channels, height, width] = shape4(a)?;
    let [b_batch, b_channels, b_height, b_width] = shape4(b)?;
    if batch != b_batch || height != b_height || width != b_width {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate NCHW tensors with shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let out_channels = a_channels + b_channels;
    let mut out = CpuTensor::zeros(&[batch, out_channels, height, width]);
    for b_idx in 0..batch {
        for c in 0..a_channels {
            for y in 0..height {
                for x in 0..width {
                    out.data[nchw_idx(b_idx, c, y, x, out_channels, height, width)] =
                        a.data[nchw_idx(b_idx, c, y, x, a_channels, height, width)];
                }
            }
        }
        for c in 0..b_channels {
            for y in 0..height {
                for x in 0..width {
                    out.data[nchw_idx(b_idx, a_channels + c, y, x, out_channels, height, width)] =
                        b.data[nchw_idx(b_idx, c, y, x, b_channels, height, width)];
                }
            }
        }
    }
    Ok(out)
}

fn quick_gelu(value: f32) -> f32 {
    value / (1.0 + (-1.702 * value).exp())
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn matmul_vector(vector: &[f32], matrix: &CpuTensor) -> DiffusionResult<Vec<f32>> {
    let [rows, cols] = shape2(matrix)?;
    if vector.len() == rows {
        let mut out = vec![0.0; cols];
        for (row, value) in vector.iter().enumerate() {
            let base = row * cols;
            for col in 0..cols {
                out[col] += value * matrix.data[base + col];
            }
        }
        return Ok(out);
    }
    if vector.len() == cols {
        let mut out = vec![0.0; rows];
        for (row, out_value) in out.iter_mut().enumerate() {
            let base = row * cols;
            for (col, value) in vector.iter().enumerate() {
                *out_value += value * matrix.data[base + col];
            }
        }
        return Ok(out);
    }
    Err(DiffusionError::InvalidMetadata(format!(
        "vector length {} does not match projection matrix shape {:?}",
        vector.len(),
        matrix.shape
    )))
}

fn matmul_vector_with_runtime_context(
    vector: &[f32],
    matrix: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<Vec<f32>> {
    let [rows, cols] = shape2(matrix)?;
    if runtime_context.rocm_device_id().is_some() && vector.len() == cols {
        let input = CpuTensor {
            shape: vec![1, cols],
            data: vector.to_vec(),
        };
        let output =
            linear_optional_bias_with_runtime_context(&input, matrix, None, runtime_context)?;
        return Ok(output.data);
    }
    if runtime_context.rocm_device_id().is_some() && vector.len() != rows {
        return Err(DiffusionError::InvalidMetadata(format!(
            "vector length {} does not match projection matrix shape {:?}",
            vector.len(),
            matrix.shape
        )));
    }
    matmul_vector(vector, matrix)
}

pub fn conv2d_nchw(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    padding: usize,
) -> DiffusionResult<CpuTensor> {
    conv2d_nchw_with_stride(input, weight, bias, padding, 1)
}

pub fn conv2d_nchw_with_stride(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    padding: usize,
    stride: usize,
) -> DiffusionResult<CpuTensor> {
    if stride == 0 {
        return Err(DiffusionError::InvalidRequest(
            "conv2d stride must be positive".to_string(),
        ));
    }
    let [batch, in_channels, in_h, in_w] = shape4(input)?;
    let [out_channels, weight_in_channels, kernel_h, kernel_w] = shape4(weight)?;
    if in_channels != weight_in_channels {
        return Err(DiffusionError::InvalidMetadata(format!(
            "conv2d input channels {in_channels} != weight input channels {weight_in_channels}"
        )));
    }
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [out_channels] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "conv2d bias shape {:?} != [{out_channels}]",
                bias.shape
            )));
        }
    }
    let padded_h = in_h + 2 * padding;
    let padded_w = in_w + 2 * padding;
    if kernel_h > padded_h || kernel_w > padded_w {
        return Err(DiffusionError::InvalidMetadata(format!(
            "conv2d kernel [{kernel_h}, {kernel_w}] is larger than padded input [{padded_h}, {padded_w}]"
        )));
    }
    let out_h = (padded_h - kernel_h) / stride + 1;
    let out_w = (padded_w - kernel_w) / stride + 1;
    let mut out = CpuTensor::zeros(&[batch, out_channels, out_h, out_w]);
    for b in 0..batch {
        for oc in 0..out_channels {
            let out_base = ((b * out_channels + oc) * out_h) * out_w;
            if let Some(bias) = bias {
                out.data[out_base..out_base + out_h * out_w].fill(bias.data[oc]);
            }
            for ic in 0..in_channels {
                let input_base = ((b * in_channels + ic) * in_h) * in_w;
                let weight_base = ((oc * in_channels + ic) * kernel_h) * kernel_w;
                for ky in 0..kernel_h {
                    for oy in 0..out_h {
                        let iy_with_pad = oy * stride + ky;
                        if iy_with_pad < padding || iy_with_pad >= in_h + padding {
                            continue;
                        }
                        let iy = iy_with_pad - padding;
                        let input_row = input_base + iy * in_w;
                        let output_row = out_base + oy * out_w;
                        for kx in 0..kernel_w {
                            let ix_offset = kx;
                            let weight_value = weight.data[weight_base + ky * kernel_w + kx];
                            if weight_value == 0.0 {
                                continue;
                            }
                            for ox in 0..out_w {
                                let ix_with_pad = ox * stride + ix_offset;
                                if ix_with_pad < padding || ix_with_pad >= in_w + padding {
                                    continue;
                                }
                                let ix = ix_with_pad - padding;
                                out.data[output_row + ox] +=
                                    input.data[input_row + ix] * weight_value;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn group_norm_nchw(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    groups: usize,
    eps: f32,
) -> DiffusionResult<CpuTensor> {
    let [batch, channels, height, width] = shape4(input)?;
    if groups == 0 || channels % groups != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "group_norm channels {channels} not divisible by groups {groups}"
        )));
    }
    if weight.shape.as_slice() != [channels] || bias.shape.as_slice() != [channels] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "group_norm weight/bias shapes {:?}/{:?} != [{channels}]",
            weight.shape, bias.shape
        )));
    }
    let mut out = CpuTensor::zeros(&input.shape);
    let channels_per_group = channels / groups;
    let elems_per_group = channels_per_group * height * width;
    for b in 0..batch {
        for group in 0..groups {
            let c_start = group * channels_per_group;
            let c_end = c_start + channels_per_group;
            let mut sum = 0.0;
            for c in c_start..c_end {
                for y in 0..height {
                    for x in 0..width {
                        sum += input.data[nchw_idx(b, c, y, x, channels, height, width)];
                    }
                }
            }
            let mean = sum / elems_per_group as f32;
            let mut var_sum = 0.0;
            for c in c_start..c_end {
                for y in 0..height {
                    for x in 0..width {
                        let centered =
                            input.data[nchw_idx(b, c, y, x, channels, height, width)] - mean;
                        var_sum += centered * centered;
                    }
                }
            }
            let inv_std = (var_sum / elems_per_group as f32 + eps).sqrt().recip();
            for c in c_start..c_end {
                for y in 0..height {
                    for x in 0..width {
                        let idx = nchw_idx(b, c, y, x, channels, height, width);
                        out.data[idx] =
                            (input.data[idx] - mean) * inv_std * weight.data[c] + bias.data[c];
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn upsample_nearest2d_nchw(input: &CpuTensor, scale: usize) -> DiffusionResult<CpuTensor> {
    if scale == 0 {
        return Err(DiffusionError::InvalidRequest(
            "upsample scale must be positive".to_string(),
        ));
    }
    let [batch, channels, height, width] = shape4(input)?;
    let out_h = height * scale;
    let out_w = width * scale;
    let mut out = CpuTensor::zeros(&[batch, channels, out_h, out_w]);
    for b in 0..batch {
        for c in 0..channels {
            for oy in 0..out_h {
                let iy = oy / scale;
                for ox in 0..out_w {
                    let ix = ox / scale;
                    out.data[nchw_idx(b, c, oy, ox, channels, out_h, out_w)] =
                        input.data[nchw_idx(b, c, iy, ix, channels, height, width)];
                }
            }
        }
    }
    Ok(out)
}

fn shape4(tensor: &CpuTensor) -> DiffusionResult<[usize; 4]> {
    match tensor.shape.as_slice() {
        [a, b, c, d] => Ok([*a, *b, *c, *d]),
        _ => Err(DiffusionError::InvalidMetadata(format!(
            "expected 4-D NCHW tensor, got {:?}",
            tensor.shape
        ))),
    }
}

fn shape2(tensor: &CpuTensor) -> DiffusionResult<[usize; 2]> {
    match tensor.shape.as_slice() {
        [a, b] => Ok([*a, *b]),
        _ => Err(DiffusionError::InvalidMetadata(format!(
            "expected 2-D tensor, got {:?}",
            tensor.shape
        ))),
    }
}

fn shape3(tensor: &CpuTensor) -> DiffusionResult<[usize; 3]> {
    match tensor.shape.as_slice() {
        [a, b, c] => Ok([*a, *b, *c]),
        _ => Err(DiffusionError::InvalidMetadata(format!(
            "expected 3-D tensor, got {:?}",
            tensor.shape
        ))),
    }
}

fn nchw_idx(
    batch: usize,
    channel: usize,
    y: usize,
    x: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> usize {
    (((batch * channels + channel) * height + y) * width) + x
}

#[derive(Debug, Clone)]
pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    byte_encoder: Vec<String>,
    start_token: u32,
    end_token: u32,
    pad_token: u32,
    max_length: usize,
    pattern: Regex,
}

impl ClipTokenizer {
    pub fn from_hfq_file(hfq: &HfqFile) -> DiffusionResult<Self> {
        Self::from_hfq_file_with_prefix(hfq, "tokenizer")
    }

    pub fn from_hfq_file_with_prefix(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Self> {
        let vocab_entry = format!("{prefix}/vocab.json");
        let merges_entry = format!("{prefix}/merges.txt");
        let (_, vocab_bytes) = hfq
            .tensor_data_vec(&vocab_entry)
            .ok_or_else(|| DiffusionError::InvalidMetadata(format!("{vocab_entry} is missing")))?;
        let (_, merges_bytes) = hfq
            .tensor_data_vec(&merges_entry)
            .ok_or_else(|| DiffusionError::InvalidMetadata(format!("{merges_entry} is missing")))?;
        Self::from_bytes(&vocab_bytes, &merges_bytes, 77)
    }

    pub fn from_bytes(
        vocab_json: &[u8],
        merges_txt: &[u8],
        max_length: usize,
    ) -> DiffusionResult<Self> {
        let vocab: HashMap<String, u32> = serde_json::from_slice(vocab_json)
            .map_err(|err| DiffusionError::InvalidMetadata(format!("invalid CLIP vocab: {err}")))?;
        let mut merges = HashMap::new();
        let merges_text = String::from_utf8_lossy(merges_txt);
        for line in merges_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(left) = parts.next() else {
                continue;
            };
            let Some(right) = parts.next() else {
                continue;
            };
            let rank = merges.len();
            merges.insert((left.to_string(), right.to_string()), rank);
        }
        let start_token = *vocab.get("<|startoftext|>").ok_or_else(|| {
            DiffusionError::InvalidMetadata("CLIP vocab missing start token".to_string())
        })?;
        let end_token = *vocab.get("<|endoftext|>").ok_or_else(|| {
            DiffusionError::InvalidMetadata("CLIP vocab missing end token".to_string())
        })?;
        Ok(Self {
            vocab,
            merges,
            byte_encoder: clip_byte_encoder(),
            start_token,
            end_token,
            pad_token: end_token,
            max_length,
            pattern: Regex::new(
                r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+",
            )
            .map_err(|err| DiffusionError::InvalidMetadata(format!("invalid CLIP regex: {err}")))?,
        })
    }

    pub fn encode_padded(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(self.max_length);
        tokens.push(self.start_token);
        for piece in self.tokenize(text) {
            if tokens.len() + 1 >= self.max_length {
                break;
            }
            tokens.push(piece);
        }
        tokens.push(self.end_token);
        tokens.resize(self.max_length, self.pad_token);
        tokens
    }

    pub fn end_token_id(&self) -> u32 {
        self.end_token
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let cleaned = whitespace_clean(text).to_lowercase();
        for mat in self.pattern.find_iter(&cleaned) {
            let token = mat.as_str();
            if let Some(&id) = self.vocab.get(token) {
                out.push(id);
                continue;
            }
            let mut encoded = String::new();
            for byte in token.as_bytes() {
                encoded.push_str(&self.byte_encoder[*byte as usize]);
            }
            for bpe_token in self.bpe(&encoded) {
                if let Some(&id) = self.vocab.get(&bpe_token) {
                    out.push(id);
                }
            }
        }
        out
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word = token.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        if word.len() == 1 {
            return word;
        }
        loop {
            let Some((best_idx, _)) = word
                .windows(2)
                .enumerate()
                .filter_map(|(idx, pair)| {
                    self.merges
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .map(|rank| (idx, *rank))
                })
                .min_by_key(|(_, rank)| *rank)
            else {
                break;
            };
            let merged = format!("{}{}", word[best_idx], word[best_idx + 1]);
            word.splice(best_idx..=best_idx + 1, [merged]);
            if word.len() == 1 {
                break;
            }
        }
        word
    }
}

fn whitespace_clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn clip_byte_encoder() -> Vec<String> {
    let mut bs = Vec::new();
    bs.extend(b'!'..=b'~');
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);
    let mut cs = bs.iter().map(|&b| b as u32).collect::<Vec<_>>();
    let mut n = 0u32;
    for b in 0u32..=255 {
        if !bs.contains(&(b as u8)) {
            bs.push(b as u8);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = vec![String::new(); 256];
    for (byte, codepoint) in bs.into_iter().zip(cs.into_iter()) {
        out[byte as usize] = char::from_u32(codepoint).unwrap().to_string();
    }
    out
}

pub fn inspect_hfq(path: impl AsRef<Path>) -> DiffusionResult<DiffusionModelSummary> {
    Ok(inspect_hfq_with_runtime_support(path)?.summary)
}

pub fn inspect_hfq_with_runtime_support(
    path: impl AsRef<Path>,
) -> DiffusionResult<DiffusionHfqInspection> {
    let path = path.as_ref();
    let hfq = HfqFile::open_index_only(path).map_err(|err| DiffusionError::Io(err.to_string()))?;
    let metadata = parse_diffusion_metadata(&hfq.metadata_json)?;
    validate_diffusion_hfq(&hfq, &metadata)?;
    let runtime_support = match native_runtime_metadata_support_error(&metadata) {
        Some(reason) => DiffusionRuntimeSupport {
            supported: false,
            runtime_kind: None,
            reason: Some(reason),
        },
        None => DiffusionRuntimeSupport {
            supported: true,
            runtime_kind: Some(DiffusionRuntimeKind::CpuSourceReference),
            reason: None,
        },
    };
    Ok(DiffusionHfqInspection {
        summary: summarize_hfq(path, &metadata),
        runtime_support,
    })
}

pub fn is_diffusion_hfq(path: impl AsRef<Path>) -> bool {
    inspect_hfq(path).is_ok()
}

pub fn parse_diffusion_metadata(metadata_json: &str) -> DiffusionResult<DiffusionHfqMetadata> {
    let metadata: DiffusionHfqMetadata = serde_json::from_str(metadata_json)
        .map_err(|err| DiffusionError::InvalidMetadata(err.to_string()))?;
    if metadata.artifact_kind != DIFFUSION_ARTIFACT_KIND {
        return Err(DiffusionError::InvalidMetadata(format!(
            "artifact_kind must be {DIFFUSION_ARTIFACT_KIND:?}"
        )));
    }
    if metadata.schema_version != DIFFUSION_SCHEMA_VERSION {
        return Err(DiffusionError::InvalidMetadata(format!(
            "unsupported schema_version {}",
            metadata.schema_version
        )));
    }
    if metadata.pipeline.class_name.is_empty() {
        return Err(DiffusionError::InvalidMetadata(
            "pipeline.class_name is required".to_string(),
        ));
    }
    Ok(metadata)
}

fn validate_diffusion_hfq(hfq: &HfqFile, metadata: &DiffusionHfqMetadata) -> DiffusionResult<()> {
    for (component_name, component) in &metadata.components {
        if let Some(config_entry) = &component.config_entry {
            if hfq.find_tensor_info(config_entry).is_none() {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "component {component_name} config entry {config_entry:?} is missing"
                )));
            }
        }
        for entry in &component.weight_entries {
            if hfq.find_tensor_info(entry).is_none() {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "component {component_name} weight entry {entry:?} is missing"
                )));
            }
        }
        for role in &component.tensor_roles {
            if hfq.find_tensor_info(&role.entry).is_none() {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "tensor role {} entry {:?} is missing",
                    role.role, role.entry
                )));
            }
        }
    }
    Ok(())
}

fn component_json(
    hfq: &HfqFile,
    metadata: &DiffusionHfqMetadata,
    component: &str,
) -> DiffusionResult<Option<Value>> {
    let Some(component) = metadata.components.get(component) else {
        return Ok(None);
    };
    let Some(entry) = component.config_entry.as_deref() else {
        return Ok(None);
    };
    let (_, bytes) = hfq.tensor_data_vec(entry).ok_or_else(|| {
        DiffusionError::InvalidMetadata(format!("component config entry {entry:?} is missing"))
    })?;
    let text = std::str::from_utf8(&bytes).map_err(|err| {
        DiffusionError::InvalidMetadata(format!(
            "component config entry {entry:?} is not utf-8: {err}"
        ))
    })?;
    parse_json_lenient(text).map(Some).map_err(|err| {
        DiffusionError::InvalidMetadata(format!(
            "component config entry {entry:?} is invalid json: {err}"
        ))
    })
}

fn parse_json_lenient(text: &str) -> serde_json::Result<Value> {
    match serde_json::from_str(text) {
        Ok(value) => Ok(value),
        Err(first_error) => {
            let sanitized = text
                .replace("-Infinity", "null")
                .replace("Infinity", "null")
                .replace("NaN", "null");
            serde_json::from_str(&sanitized).map_err(|_| first_error)
        }
    }
}

fn json_string(value: &Value, key: &str) -> String {
    json_optional_string(value, key).unwrap_or_default()
}

fn json_optional_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn json_usize(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn json_i32(value: &Value, key: &str) -> Option<i32> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
}

fn json_f32(value: &Value, key: &str) -> Option<f32> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
}

fn json_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn json_f32_vec(value: &Value, key: &str) -> Vec<f32> {
    match value.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_f64)
            .map(|value| value as f32)
            .collect(),
        _ => Vec::new(),
    }
}

fn json_usize_vec(value: &Value, key: &str) -> Vec<usize> {
    match value.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_u64)
            .filter_map(|value| usize::try_from(value).ok())
            .collect(),
        Some(Value::Number(number)) => number
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

fn json_string_vec(value: &Value, key: &str) -> Vec<String> {
    match value.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

fn is_sdxl_pipeline_class(class_name: &str) -> bool {
    matches!(
        class_name,
        "StableDiffusionXLPipeline"
            | "StableDiffusionXLImg2ImgPipeline"
            | "StableDiffusionXLInpaintPipeline"
    )
}

fn is_native_unet_pipeline_class(class_name: &str) -> bool {
    matches!(
        class_name,
        "StableDiffusionPipeline"
            | "StableDiffusionImg2ImgPipeline"
            | "StableDiffusionInpaintPipeline"
            | "StableDiffusionXLPipeline"
            | "StableDiffusionXLImg2ImgPipeline"
            | "StableDiffusionXLInpaintPipeline"
    )
}

fn validate_batch_request(
    metadata: &DiffusionHfqMetadata,
    request: &DiffusionBatchRequest,
) -> DiffusionResult<()> {
    if request.prompts.is_empty() {
        return Err(DiffusionError::InvalidRequest(
            "at least one prompt is required".to_string(),
        ));
    }
    if request.prompts.len() as u32 > metadata.batch.max_batch {
        return Err(DiffusionError::InvalidRequest(format!(
            "batch size {} exceeds model max_batch {}",
            request.prompts.len(),
            metadata.batch.max_batch
        )));
    }
    if request.width == 0 || request.height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "width and height must be positive".to_string(),
        ));
    }
    if request.steps == 0 {
        return Err(DiffusionError::InvalidRequest(
            "steps must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

pub fn sdxl_time_ids_for_request(request: &DiffusionBatchRequest) -> DiffusionResult<CpuTensor> {
    let batch = request.prompts.len();
    let original_height = request.original_height.unwrap_or(request.height);
    let original_width = request.original_width.unwrap_or(request.width);
    let target_height = request.target_height.unwrap_or(request.height);
    let target_width = request.target_width.unwrap_or(request.width);
    let values = [
        original_height,
        original_width,
        request.crop_y,
        request.crop_x,
        target_height,
        target_width,
    ];
    if [original_height, original_width, target_height, target_width].contains(&0) {
        return Err(DiffusionError::InvalidRequest(
            "SDXL original/target dimensions must be positive".to_string(),
        ));
    }
    let mut data = Vec::with_capacity(batch * values.len());
    for _ in 0..batch {
        data.extend(values.iter().map(|value| *value as f32));
    }
    Ok(CpuTensor {
        shape: vec![batch, values.len()],
        data,
    })
}

fn build_sdxl_denoise_conditioning<'a>(
    conditioning: &'a DiffusionConditioningBatch,
    time_ids: &'a CpuTensor,
    positive: bool,
) -> DiffusionResult<Option<SdxlDenoiseConditioning<'a>>> {
    let (cross_attention, pooled) = if positive {
        (
            conditioning.prompt_cross_attention_embeddings.as_ref(),
            conditioning.prompt_pooled_embeddings.as_ref(),
        )
    } else {
        (
            conditioning.negative_cross_attention_embeddings.as_ref(),
            conditioning.negative_pooled_embeddings.as_ref(),
        )
    };
    match (cross_attention, pooled) {
        (Some(_), Some(text_embeds)) => Ok(Some(SdxlDenoiseConditioning {
            text_embeds,
            time_ids,
        })),
        (None, None) => Ok(None),
        _ => Err(DiffusionError::BackendUnavailable(
            "SDXL denoise conditioning requires both combined cross-attention embeddings and pooled text embeddings".to_string(),
        )),
    }
}

fn validate_img2img_request(
    metadata: &DiffusionHfqMetadata,
    request: &DiffusionImg2ImgRequest,
) -> DiffusionResult<()> {
    validate_batch_request(metadata, &request.batch)?;
    if !request.denoising_strength.is_finite() || !(0.0..=1.0).contains(&request.denoising_strength)
    {
        return Err(DiffusionError::InvalidRequest(format!(
            "denoising_strength {} must be between 0 and 1",
            request.denoising_strength
        )));
    }
    if let Some(inpainting_fill) = request.inpainting_fill {
        if inpainting_fill > 3 {
            return Err(DiffusionError::InvalidRequest(format!(
                "inpainting_fill {inpainting_fill} must be 0, 1, 2, or 3"
            )));
        }
    }
    if request.init_image.batch == 0 {
        return Err(DiffusionError::InvalidRequest(
            "init image batch must be non-empty".to_string(),
        ));
    }
    if request.init_image.width == 0 || request.init_image.height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "init image dimensions must be positive".to_string(),
        ));
    }
    if request.init_image.batch != 1 && request.init_image.batch != request.batch.prompts.len() {
        return Err(DiffusionError::InvalidRequest(format!(
            "init image batch {} must be 1 or match prompt batch {}",
            request.init_image.batch,
            request.batch.prompts.len()
        )));
    }
    if let Some(mask) = &request.mask {
        if mask.batch == 0 {
            return Err(DiffusionError::InvalidRequest(
                "mask batch must be non-empty".to_string(),
            ));
        }
        if mask.width == 0 || mask.height == 0 {
            return Err(DiffusionError::InvalidRequest(
                "mask dimensions must be positive".to_string(),
            ));
        }
        if mask.batch != 1 && mask.batch != request.batch.prompts.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "mask batch {} must be 1 or match prompt batch {}",
                mask.batch,
                request.batch.prompts.len()
            )));
        }
        if mask.width != request.init_image.width || mask.height != request.init_image.height {
            return Err(DiffusionError::InvalidRequest(format!(
                "mask dimensions {}x{} do not match init image {}x{}",
                mask.width, mask.height, request.init_image.width, request.init_image.height
            )));
        }
    }
    Ok(())
}

fn latent_mask_weights_from_rgb_batch(
    mask: &RgbImageBatch,
    latents: &LatentBatch,
) -> DiffusionResult<Vec<f32>> {
    if mask.batch != latents.batch {
        return Err(DiffusionError::InvalidRequest(format!(
            "mask batch {} != latent batch {}",
            mask.batch, latents.batch
        )));
    }
    let bytes_per_image = mask
        .width
        .checked_mul(mask.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("mask dimensions overflow".to_string()))?;
    let expected = bytes_per_image.checked_mul(mask.batch).ok_or_else(|| {
        DiffusionError::InvalidRequest("mask batch dimensions overflow".to_string())
    })?;
    if mask.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "mask has {} bytes, expected {expected}",
            mask.data.len()
        )));
    }
    let mut weights = Vec::with_capacity(latents.batch * latents.height * latents.width);
    for b in 0..latents.batch {
        let image_offset = b * bytes_per_image;
        for y in 0..latents.height {
            let source_y = ((y * mask.height) / latents.height).min(mask.height.saturating_sub(1));
            for x in 0..latents.width {
                let source_x = ((x * mask.width) / latents.width).min(mask.width.saturating_sub(1));
                let idx = image_offset + ((source_y * mask.width + source_x) * 3);
                let luma =
                    (mask.data[idx] as f32 + mask.data[idx + 1] as f32 + mask.data[idx + 2] as f32)
                        / (3.0 * 255.0);
                weights.push(luma.clamp(0.0, 1.0));
            }
        }
    }
    Ok(weights)
}

fn apply_inpainting_fill_to_latents(
    init_latents: &mut LatentBatch,
    noise_latents: &LatentBatch,
    mask_weights: &[f32],
    inpainting_fill: u32,
) -> DiffusionResult<bool> {
    match inpainting_fill {
        0 | 1 => return Ok(false),
        2 | 3 => {}
        _ => {
            return Err(DiffusionError::InvalidRequest(format!(
                "inpainting_fill {inpainting_fill} must be 0, 1, 2, or 3"
            )));
        }
    }
    if noise_latents.batch != init_latents.batch
        || noise_latents.channels != init_latents.channels
        || noise_latents.height != init_latents.height
        || noise_latents.width != init_latents.width
    {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpainting_fill noise latent shape [{}x{}x{}x{}] != init latent shape [{}x{}x{}x{}]",
            noise_latents.batch,
            noise_latents.channels,
            noise_latents.height,
            noise_latents.width,
            init_latents.batch,
            init_latents.channels,
            init_latents.height,
            init_latents.width
        )));
    }
    let expected_weights = init_latents
        .batch
        .checked_mul(init_latents.height)
        .and_then(|value| value.checked_mul(init_latents.width))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("inpainting_fill mask dimensions overflow".to_string())
        })?;
    if mask_weights.len() != expected_weights {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpainting_fill mask has {} latent weights, expected {expected_weights}",
            mask_weights.len()
        )));
    }
    for batch in 0..init_latents.batch {
        for y in 0..init_latents.height {
            for x in 0..init_latents.width {
                let mask_idx = (batch * init_latents.height + y) * init_latents.width + x;
                let weight = mask_weights[mask_idx].clamp(0.0, 1.0);
                if weight == 0.0 {
                    continue;
                }
                for channel in 0..init_latents.channels {
                    let idx = ((batch * init_latents.channels + channel) * init_latents.height + y)
                        * init_latents.width
                        + x;
                    let replacement = if inpainting_fill == 2 {
                        noise_latents.data[idx]
                    } else {
                        0.0
                    };
                    init_latents.data[idx] =
                        init_latents.data[idx] * (1.0 - weight) + replacement * weight;
                }
            }
        }
    }
    Ok(true)
}

fn build_inpaint_conditioning_if_supported(
    noise: &dyn DiffusionNoiseBackend,
    encoder: &NativeVaeEncoder,
    init_image: &RgbImageBatch,
    mask: &RgbImageBatch,
    latents: &LatentBatch,
    mask_weights: Option<&[f32]>,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<(Option<InpaintDenoiseConditioning>, DiffusionRuntimeKind)> {
    let base_channels = latents.channels;
    let model_channels = noise.model_input_channels();
    if model_channels == base_channels {
        return Ok((None, DiffusionRuntimeKind::CpuSourceReference));
    }
    let inpaint_channels = base_channels
        .checked_mul(2)
        .and_then(|channels| channels.checked_add(1))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("inpaint channel count overflow".to_string())
        })?;
    if model_channels != inpaint_channels {
        return Err(DiffusionError::InvalidMetadata(format!(
            "UNet input channels {model_channels} do not match latent channels {base_channels} or inpaint channels {inpaint_channels}"
        )));
    }
    let mask_weights = mask_weights.ok_or_else(|| {
        DiffusionError::InvalidRequest("inpaint conditioning requires a mask".to_string())
    })?;
    let (masked_image, masked_image_kind) =
        masked_rgb_batch_for_inpaint_with_runtime_context(init_image, mask, runtime_context)?;
    let (masked_image_latents, masked_latents_kind) =
        encode_to_latents_with_runtime_context(encoder, &masked_image, runtime_context)?;
    let masked_image_latents = if masked_image_latents.batch == latents.batch
        && masked_image_latents.channels == latents.channels
        && (masked_image_latents.height != latents.height
            || masked_image_latents.width != latents.width)
    {
        resize_latent_batch_nearest(&masked_image_latents, latents.height, latents.width)?
    } else {
        masked_image_latents
    };
    if masked_image_latents.batch != latents.batch
        || masked_image_latents.channels != latents.channels
        || masked_image_latents.height != latents.height
        || masked_image_latents.width != latents.width
    {
        return Err(DiffusionError::InvalidRequest(format!(
            "encoded masked-image latent shape [{}x{}x{}x{}] != init latent shape [{}x{}x{}x{}]",
            masked_image_latents.batch,
            masked_image_latents.channels,
            masked_image_latents.height,
            masked_image_latents.width,
            latents.batch,
            latents.channels,
            latents.height,
            latents.width
        )));
    }
    Ok((
        Some(InpaintDenoiseConditioning {
            mask_weights: mask_weights.to_vec(),
            masked_image_latents,
        }),
        merge_runtime_kind(masked_image_kind, masked_latents_kind),
    ))
}

fn masked_rgb_batch_for_inpaint(
    image: &RgbImageBatch,
    mask: &RgbImageBatch,
) -> DiffusionResult<RgbImageBatch> {
    if image.batch != mask.batch || image.width != mask.width || image.height != mask.height {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpaint image shape [{}x{}x{}] != mask shape [{}x{}x{}]",
            image.batch, image.width, image.height, mask.batch, mask.width, mask.height
        )));
    }
    let expected = image
        .batch
        .checked_mul(image.width)
        .and_then(|pixels| pixels.checked_mul(image.height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("image dimensions overflow".to_string()))?;
    if image.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "image has {} bytes, expected {expected}",
            image.data.len()
        )));
    }
    if mask.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "mask has {} bytes, expected {expected}",
            mask.data.len()
        )));
    }
    let mut data = Vec::with_capacity(image.data.len());
    for pixel in 0..(image.batch * image.width * image.height) {
        let idx = pixel * 3;
        let weight =
            (mask.data[idx] as f32 + mask.data[idx + 1] as f32 + mask.data[idx + 2] as f32)
                / (3.0 * 255.0);
        let keep = 1.0 - weight.clamp(0.0, 1.0);
        data.push((image.data[idx] as f32 * keep).round().clamp(0.0, 255.0) as u8);
        data.push(
            (image.data[idx + 1] as f32 * keep)
                .round()
                .clamp(0.0, 255.0) as u8,
        );
        data.push(
            (image.data[idx + 2] as f32 * keep)
                .round()
                .clamp(0.0, 255.0) as u8,
        );
    }
    Ok(RgbImageBatch {
        batch: image.batch,
        width: image.width,
        height: image.height,
        data,
    })
}

fn blend_latents_with_mask(
    generated: &mut LatentBatch,
    init: &LatentBatch,
    mask_weights: &[f32],
) -> DiffusionResult<()> {
    if generated.batch != init.batch
        || generated.channels != init.channels
        || generated.height != init.height
        || generated.width != init.width
    {
        return Err(DiffusionError::InvalidRequest(format!(
            "generated latent shape [{}x{}x{}x{}] != init latent shape [{}x{}x{}x{}]",
            generated.batch,
            generated.channels,
            generated.height,
            generated.width,
            init.batch,
            init.channels,
            init.height,
            init.width
        )));
    }
    let expected = generated.batch * generated.height * generated.width;
    if mask_weights.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "latent mask has {} weights, expected {expected}",
            mask_weights.len()
        )));
    }
    for b in 0..generated.batch {
        for c in 0..generated.channels {
            for y in 0..generated.height {
                for x in 0..generated.width {
                    let latent_idx = (((b * generated.channels + c) * generated.height + y)
                        * generated.width)
                        + x;
                    let mask_idx = (b * generated.height + y) * generated.width + x;
                    let weight = mask_weights[mask_idx];
                    generated.data[latent_idx] = init.data[latent_idx] * (1.0 - weight)
                        + generated.data[latent_idx] * weight;
                }
            }
        }
    }
    Ok(())
}

fn expand_rgb_batch_for_prompts(
    image: &RgbImageBatch,
    target_batch: usize,
) -> DiffusionResult<RgbImageBatch> {
    if image.batch == target_batch {
        return Ok(image.clone());
    }
    if image.batch != 1 {
        return Err(DiffusionError::InvalidRequest(format!(
            "cannot expand image batch {} to prompt batch {target_batch}",
            image.batch
        )));
    }
    let bytes_per_image = image
        .width
        .checked_mul(image.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("image dimensions overflow".to_string()))?;
    if image.data.len() != bytes_per_image {
        return Err(DiffusionError::InvalidRequest(format!(
            "single RGB image has {} bytes, expected {bytes_per_image}",
            image.data.len()
        )));
    }
    let mut data = Vec::with_capacity(bytes_per_image * target_batch);
    for _ in 0..target_batch {
        data.extend_from_slice(&image.data);
    }
    Ok(RgbImageBatch {
        batch: target_batch,
        width: image.width,
        height: image.height,
        data,
    })
}

pub fn resize_rgb_batch_nearest(
    image: &RgbImageBatch,
    target_width: u32,
    target_height: u32,
) -> DiffusionResult<RgbImageBatch> {
    let target_width = usize::try_from(target_width).map_err(|_| {
        DiffusionError::InvalidRequest("target image width does not fit usize".to_string())
    })?;
    let target_height = usize::try_from(target_height).map_err(|_| {
        DiffusionError::InvalidRequest("target image height does not fit usize".to_string())
    })?;
    if target_width == 0 || target_height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "target image dimensions must be positive".to_string(),
        ));
    }
    if image.width == 0 || image.height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "source image dimensions must be positive".to_string(),
        ));
    }
    let source_bytes = image
        .batch
        .checked_mul(image.width)
        .and_then(|pixels| pixels.checked_mul(image.height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("image dimensions overflow".to_string()))?;
    if image.data.len() != source_bytes {
        return Err(DiffusionError::InvalidRequest(format!(
            "RGB image batch has {} bytes, expected {source_bytes}",
            image.data.len()
        )));
    }
    if image.width == target_width && image.height == target_height {
        return Ok(image.clone());
    }
    let target_bytes = image
        .batch
        .checked_mul(target_width)
        .and_then(|pixels| pixels.checked_mul(target_height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("target image dimensions overflow".to_string())
        })?;
    let mut data = vec![0u8; target_bytes];
    let source_image_bytes = image.width * image.height * 3;
    let target_image_bytes = target_width * target_height * 3;
    for batch_idx in 0..image.batch {
        let source_batch_offset = batch_idx * source_image_bytes;
        let target_batch_offset = batch_idx * target_image_bytes;
        for y in 0..target_height {
            let source_y = (y * image.height / target_height).min(image.height.saturating_sub(1));
            for x in 0..target_width {
                let source_x = (x * image.width / target_width).min(image.width.saturating_sub(1));
                let source_idx = source_batch_offset + ((source_y * image.width + source_x) * 3);
                let target_idx = target_batch_offset + ((y * target_width + x) * 3);
                data[target_idx..target_idx + 3]
                    .copy_from_slice(&image.data[source_idx..source_idx + 3]);
            }
        }
    }
    Ok(RgbImageBatch {
        batch: image.batch,
        width: target_width,
        height: target_height,
        data,
    })
}

fn resize_latent_batch_nearest(
    latents: &LatentBatch,
    target_height: usize,
    target_width: usize,
) -> DiffusionResult<LatentBatch> {
    if target_width == 0 || target_height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "target latent dimensions must be positive".to_string(),
        ));
    }
    if latents.width == 0 || latents.height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "source latent dimensions must be positive".to_string(),
        ));
    }
    let source_values = latents
        .batch
        .checked_mul(latents.channels)
        .and_then(|values| values.checked_mul(latents.height))
        .and_then(|values| values.checked_mul(latents.width))
        .ok_or_else(|| DiffusionError::InvalidRequest("latent dimensions overflow".to_string()))?;
    if latents.data.len() != source_values {
        return Err(DiffusionError::InvalidRequest(format!(
            "latent batch has {} values, expected {source_values}",
            latents.data.len()
        )));
    }
    if latents.width == target_width && latents.height == target_height {
        return Ok(latents.clone());
    }
    let target_values = latents
        .batch
        .checked_mul(latents.channels)
        .and_then(|values| values.checked_mul(target_height))
        .and_then(|values| values.checked_mul(target_width))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("target latent dimensions overflow".to_string())
        })?;
    let mut data = vec![0.0f32; target_values];
    let source_image_values = latents.channels * latents.height * latents.width;
    let target_image_values = latents.channels * target_height * target_width;
    for batch_idx in 0..latents.batch {
        let source_batch_offset = batch_idx * source_image_values;
        let target_batch_offset = batch_idx * target_image_values;
        for channel in 0..latents.channels {
            let source_channel_offset =
                source_batch_offset + channel * latents.height * latents.width;
            let target_channel_offset =
                target_batch_offset + channel * target_height * target_width;
            for y in 0..target_height {
                let source_y =
                    (y * latents.height / target_height).min(latents.height.saturating_sub(1));
                for x in 0..target_width {
                    let source_x =
                        (x * latents.width / target_width).min(latents.width.saturating_sub(1));
                    let source_idx = source_channel_offset + source_y * latents.width + source_x;
                    let target_idx = target_channel_offset + y * target_width + x;
                    data[target_idx] = latents.data[source_idx];
                }
            }
        }
    }
    Ok(LatentBatch {
        batch: latents.batch,
        channels: latents.channels,
        height: target_height,
        width: target_width,
        data,
    })
}

pub fn resize_rgb_batch_to_cover_nearest(
    image: &RgbImageBatch,
    target_width: u32,
    target_height: u32,
) -> DiffusionResult<RgbImageBatch> {
    let target_width = usize::try_from(target_width).map_err(|_| {
        DiffusionError::InvalidRequest("target image width does not fit usize".to_string())
    })?;
    let target_height = usize::try_from(target_height).map_err(|_| {
        DiffusionError::InvalidRequest("target image height does not fit usize".to_string())
    })?;
    if target_width == 0 || target_height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "target image dimensions must be positive".to_string(),
        ));
    }
    if image.width == 0 || image.height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "source image dimensions must be positive".to_string(),
        ));
    }

    let source_w = image.width as u128;
    let source_h = image.height as u128;
    let target_w = target_width as u128;
    let target_h = target_height as u128;
    let (cover_width, cover_height) = if source_w * target_h < target_w * source_h {
        let height = ((target_w * source_h) / source_w).max(target_h);
        (target_w, height)
    } else {
        let width = ((target_h * source_w) / source_h).max(target_w);
        (width, target_h)
    };
    let cover_width_u32 = u32::try_from(cover_width).map_err(|_| {
        DiffusionError::InvalidRequest("cover image width is out of range".to_string())
    })?;
    let cover_height_u32 = u32::try_from(cover_height).map_err(|_| {
        DiffusionError::InvalidRequest("cover image height is out of range".to_string())
    })?;
    let resized = resize_rgb_batch_nearest(image, cover_width_u32, cover_height_u32)?;
    crop_rgb_batch_center(&resized, target_width, target_height)
}

pub fn resize_rgb_batch_to_contain_fill_nearest(
    image: &RgbImageBatch,
    target_width: u32,
    target_height: u32,
) -> DiffusionResult<RgbImageBatch> {
    let target_width = usize::try_from(target_width).map_err(|_| {
        DiffusionError::InvalidRequest("target image width does not fit usize".to_string())
    })?;
    let target_height = usize::try_from(target_height).map_err(|_| {
        DiffusionError::InvalidRequest("target image height does not fit usize".to_string())
    })?;
    if target_width == 0 || target_height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "target image dimensions must be positive".to_string(),
        ));
    }
    if image.width == 0 || image.height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "source image dimensions must be positive".to_string(),
        ));
    }

    let source_w = image.width as u128;
    let source_h = image.height as u128;
    let target_w = target_width as u128;
    let target_h = target_height as u128;
    let (fit_width, fit_height) = if target_w * source_h < source_w * target_h {
        let height = ((target_w * source_h) / source_w).max(1);
        (target_w, height)
    } else {
        let width = ((target_h * source_w) / source_h).max(1);
        (width, target_h)
    };
    let fit_width = usize::try_from(fit_width).map_err(|_| {
        DiffusionError::InvalidRequest("contained image width is out of range".to_string())
    })?;
    let fit_height = usize::try_from(fit_height).map_err(|_| {
        DiffusionError::InvalidRequest("contained image height is out of range".to_string())
    })?;
    let fit_width_u32 = u32::try_from(fit_width).map_err(|_| {
        DiffusionError::InvalidRequest("contained image width is out of range".to_string())
    })?;
    let fit_height_u32 = u32::try_from(fit_height).map_err(|_| {
        DiffusionError::InvalidRequest("contained image height is out of range".to_string())
    })?;
    let resized = resize_rgb_batch_nearest(image, fit_width_u32, fit_height_u32)?;
    if fit_width == target_width && fit_height == target_height {
        return Ok(resized);
    }

    let paste_x = (target_width - fit_width) / 2;
    let paste_y = (target_height - fit_height) / 2;
    let target_image_bytes = target_width
        .checked_mul(target_height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("target image dimensions overflow".to_string())
        })?;
    let resized_image_bytes = fit_width
        .checked_mul(fit_height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("contained image dimensions overflow".to_string())
        })?;
    let mut data = vec![0u8; image.batch * target_image_bytes];
    for batch_idx in 0..image.batch {
        let resized_batch_offset = batch_idx * resized_image_bytes;
        let target_batch_offset = batch_idx * target_image_bytes;
        for y in 0..target_height {
            let resized_y = y.saturating_sub(paste_y).min(fit_height - 1);
            for x in 0..target_width {
                let resized_x = x.saturating_sub(paste_x).min(fit_width - 1);
                let source_idx = resized_batch_offset + ((resized_y * fit_width + resized_x) * 3);
                let target_idx = target_batch_offset + ((y * target_width + x) * 3);
                data[target_idx..target_idx + 3]
                    .copy_from_slice(&resized.data[source_idx..source_idx + 3]);
            }
        }
    }
    Ok(RgbImageBatch {
        batch: image.batch,
        width: target_width,
        height: target_height,
        data,
    })
}

fn crop_rgb_batch_center(
    image: &RgbImageBatch,
    target_width: usize,
    target_height: usize,
) -> DiffusionResult<RgbImageBatch> {
    if target_width == 0 || target_height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "target image dimensions must be positive".to_string(),
        ));
    }
    if target_width > image.width || target_height > image.height {
        return Err(DiffusionError::InvalidRequest(format!(
            "cannot crop image {}x{} to larger target {}x{}",
            image.width, image.height, target_width, target_height
        )));
    }
    let source_bytes = image
        .batch
        .checked_mul(image.width)
        .and_then(|pixels| pixels.checked_mul(image.height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("image dimensions overflow".to_string()))?;
    if image.data.len() != source_bytes {
        return Err(DiffusionError::InvalidRequest(format!(
            "RGB image batch has {} bytes, expected {source_bytes}",
            image.data.len()
        )));
    }
    if image.width == target_width && image.height == target_height {
        return Ok(image.clone());
    }
    let source_x = (image.width - target_width) / 2;
    let source_y = (image.height - target_height) / 2;
    let target_image_bytes = target_width
        .checked_mul(target_height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("target image dimensions overflow".to_string())
        })?;
    let mut data = vec![0u8; image.batch * target_image_bytes];
    let source_image_bytes = image.width * image.height * 3;
    for batch_idx in 0..image.batch {
        let source_batch_offset = batch_idx * source_image_bytes;
        let target_batch_offset = batch_idx * target_image_bytes;
        for y in 0..target_height {
            let source_row = source_batch_offset + ((source_y + y) * image.width + source_x) * 3;
            let target_row = target_batch_offset + y * target_width * 3;
            let bytes = target_width * 3;
            data[target_row..target_row + bytes]
                .copy_from_slice(&image.data[source_row..source_row + bytes]);
        }
    }
    Ok(RgbImageBatch {
        batch: image.batch,
        width: target_width,
        height: target_height,
        data,
    })
}

fn summarize_hfq(path: &Path, metadata: &DiffusionHfqMetadata) -> DiffusionModelSummary {
    let file_name = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("diffusion");
    let model_name = if metadata.pipeline.model_name.is_empty() {
        file_name.to_string()
    } else {
        metadata.pipeline.model_name.clone()
    };
    DiffusionModelSummary {
        path: path.to_path_buf(),
        title: format!("{model_name}:{}", metadata.pipeline.class_name),
        model_name,
        pipeline_class: metadata.pipeline.class_name.clone(),
        max_batch: metadata.batch.max_batch,
        weight_format: metadata.quantization.weight_format.clone(),
    }
}

fn native_runtime_metadata_support_error(metadata: &DiffusionHfqMetadata) -> Option<String> {
    let quantization = &metadata.quantization;
    if quantization.weight_format == "metadata-only" {
        return Some(
            "diffusion HFQ contains metadata only; import without --metadata-only or attach converted weights before serving"
                .to_string(),
        );
    }
    if !is_native_unet_pipeline_class(&metadata.pipeline.class_name) {
        let denoiser = if metadata.components.contains_key("transformer") {
            "transformer denoiser"
        } else {
            "unsupported denoiser"
        };
        return Some(format!(
            "native diffusion runtime currently supports Stable Diffusion UNet-family pipelines only; artifact pipeline {:?} uses a {denoiser} and requires a matching diffusion runtime",
            metadata.pipeline.class_name
        ));
    }
    if let Some(unet) = metadata
        .components
        .get("unet")
        .and_then(|component| component.class_name.as_deref())
    {
        if unet != "UNet2DConditionModel" {
            return Some(format!(
                "native diffusion runtime supports UNet2DConditionModel denoisers only; artifact unet class {unet:?} is unsupported"
            ));
        }
    }
    if let Some(vae) = metadata
        .components
        .get("vae")
        .and_then(|component| component.class_name.as_deref())
    {
        if vae != "AutoencoderKL" {
            return Some(format!(
                "native diffusion runtime supports AutoencoderKL VAEs only; artifact vae class {vae:?} is unsupported"
            ));
        }
    }
    if let Some(text_encoder) = metadata
        .components
        .get("text_encoder")
        .and_then(|component| component.class_name.as_deref())
    {
        if text_encoder != "CLIPTextModel" && text_encoder != "CLIPTextModelWithProjection" {
            return Some(format!(
                "native diffusion runtime supports CLIP text encoders only; artifact text_encoder class {text_encoder:?} is unsupported"
            ));
        }
    }
    if !matches!(quantization.activation_format.as_str(), "fp16" | "fp32") {
        return Some(format!(
            "native diffusion runtime currently supports fp16/fp32 activation metadata only; artifact activation_format {:?} is unsupported",
            quantization.activation_format
        ));
    }
    if quantization.tensor_roles_version != 1 {
        return Some(format!(
            "native diffusion runtime supports tensor_roles_version 1; artifact tensor_roles_version {} is unsupported",
            quantization.tensor_roles_version
        ));
    }
    None
}

#[derive(Debug, Clone)]
pub struct DiffusersImportOptions {
    pub source: PathBuf,
    pub output: PathBuf,
    pub model_name: Option<String>,
    pub max_batch: u32,
    pub metadata_only: bool,
}

pub fn import_diffusers_to_hfq(
    options: DiffusersImportOptions,
) -> anyhow::Result<DiffusionModelSummary> {
    let source = options.source.canonicalize()?;
    if source.is_file() {
        if options.metadata_only {
            anyhow::bail!("--metadata-only is only supported for Diffusers snapshot directories");
        }
        return import_single_file_checkpoint_to_hfq(
            source,
            options.output,
            options.model_name,
            options.max_batch,
        );
    }
    let output = options.output;
    let model_index = read_json(source.join("model_index.json"))?;
    let class_name = model_index
        .get("_class_name")
        .and_then(Value::as_str)
        .unwrap_or("DiffusionPipeline")
        .to_string();

    let mut entries = Vec::new();
    let mut components = BTreeMap::new();
    let mut tokenizer_entries = Vec::new();
    let mut tokenizer_2_entries = Vec::new();
    let weight_files_enabled = !options.metadata_only;

    push_import_file_entry(
        &mut entries,
        "diffusers/model_index.json",
        QT_DIFFUSION_JSON,
        source.join("model_index.json"),
    )?;
    add_component(
        &source,
        &mut entries,
        &mut components,
        "text_encoder",
        if weight_files_enabled {
            &[
                "model.safetensors",
                "pytorch_model.safetensors",
                "pytorch_model.bin",
            ]
        } else {
            &[]
        },
    )?;
    add_component(
        &source,
        &mut entries,
        &mut components,
        "text_encoder_2",
        if weight_files_enabled {
            &[
                "model.safetensors",
                "pytorch_model.safetensors",
                "pytorch_model.bin",
            ]
        } else {
            &[]
        },
    )?;
    add_component(
        &source,
        &mut entries,
        &mut components,
        "unet",
        if weight_files_enabled {
            &[
                "diffusion_pytorch_model.safetensors",
                "model.safetensors",
                "diffusion_pytorch_model.bin",
            ]
        } else {
            &[]
        },
    )?;
    add_component(
        &source,
        &mut entries,
        &mut components,
        "transformer",
        if weight_files_enabled {
            &[
                "diffusion_pytorch_model.safetensors",
                "model.safetensors",
                "diffusion_pytorch_model.bin",
            ]
        } else {
            &[]
        },
    )?;
    add_component(
        &source,
        &mut entries,
        &mut components,
        "vae",
        if weight_files_enabled {
            &[
                "diffusion_pytorch_model.safetensors",
                "model.safetensors",
                "diffusion_pytorch_model.bin",
            ]
        } else {
            &[]
        },
    )?;
    add_component(&source, &mut entries, &mut components, "scheduler", &[])?;

    for name in [
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let path = source.join("tokenizer").join(name);
        if path.is_file() {
            let entry_name = format!("tokenizer/{name}");
            push_import_file_entry(&mut entries, &entry_name, QT_DIFFUSION_TOKENIZER, path)?;
            tokenizer_entries.push(entry_name);
        }
    }
    for name in [
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let path = source.join("tokenizer_2").join(name);
        if path.is_file() {
            let entry_name = format!("tokenizer_2/{name}");
            push_import_file_entry(&mut entries, &entry_name, QT_DIFFUSION_TOKENIZER, path)?;
            tokenizer_2_entries.push(entry_name);
        }
    }

    let unet_config = read_json(source.join("unet/config.json")).unwrap_or_else(|_| json!({}));
    let transformer_config =
        read_json(source.join("transformer/config.json")).unwrap_or_else(|_| json!({}));
    let vae_config = read_json(source.join("vae/config.json")).unwrap_or_else(|_| json!({}));
    let latent_channels = unet_config
        .get("in_channels")
        .and_then(Value::as_u64)
        .or_else(|| {
            transformer_config
                .get("out_channels")
                .and_then(Value::as_u64)
        })
        .or_else(|| {
            transformer_config
                .get("in_channels")
                .and_then(Value::as_u64)
        })
        .or_else(|| vae_config.get("latent_channels").and_then(Value::as_u64))
        .or_else(|| vae_config.get("z_dim").and_then(Value::as_u64))
        .map(|value| value as u32);
    let latent_size = unet_config
        .get("sample_size")
        .and_then(Value::as_u64)
        .map(|value| value as u32);
    let model_name = options
        .model_name
        .or_else(|| {
            source
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "diffusion".to_string());
    let max_batch = options.max_batch.max(1);
    let metadata = DiffusionHfqMetadata {
        artifact_kind: DIFFUSION_ARTIFACT_KIND.to_string(),
        schema_version: DIFFUSION_SCHEMA_VERSION,
        pipeline: DiffusionPipelineMetadata {
            class_name,
            source: source.to_string_lossy().into_owned(),
            model_name,
            latent_channels,
            latent_height: latent_size,
            latent_width: latent_size,
            supported_widths: Vec::new(),
            supported_heights: Vec::new(),
        },
        tokenizer: DiffusionTokenizerMetadata {
            kind: "clip-bpe".to_string(),
            max_length: Some(77),
            entries: tokenizer_entries,
        },
        tokenizer_2: (!tokenizer_2_entries.is_empty()).then_some(DiffusionTokenizerMetadata {
            kind: "clip-bpe".to_string(),
            max_length: Some(77),
            entries: tokenizer_2_entries,
        }),
        batch: DiffusionBatchMetadata {
            max_batch,
            batched_runtime: true,
        },
        quantization: if options.metadata_only {
            DiffusionQuantizationMetadata {
                weight_format: "metadata-only".to_string(),
                ..DiffusionQuantizationMetadata::default()
            }
        } else {
            DiffusionQuantizationMetadata::default()
        },
        components,
    };
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    write_import_entries_to_hfq(&output, &metadata_json, &entries)?;
    inspect_hfq(output).map_err(anyhow::Error::from)
}

fn import_single_file_checkpoint_to_hfq(
    source: PathBuf,
    output: PathBuf,
    model_name: Option<String>,
    max_batch: u32,
) -> anyhow::Result<DiffusionModelSummary> {
    let parsed_safetensors = source
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"));
    let tensors = if parsed_safetensors {
        parse_safetensors_state_dict(&source).unwrap_or_default()
    } else {
        Vec::new()
    };
    let mut entries = Vec::new();
    let mut components = BTreeMap::<String, DiffusionComponentMetadata>::new();
    let mut tokenizer_entries = Vec::new();
    let mut tokenizer_2_entries = Vec::new();
    let pipeline_class = infer_single_file_pipeline_class(&tensors);
    let latent_channels = infer_single_file_latent_channels(&tensors);

    if tensors.is_empty() {
        push_import_file_entry(
            &mut entries,
            "checkpoint/source_weights",
            QT_DIFFUSION_SOURCE_WEIGHTS,
            source.clone(),
        )?;
        components.insert(
            "checkpoint".to_string(),
            DiffusionComponentMetadata {
                class_name: Some("SingleFileCheckpoint".to_string()),
                config_entry: None,
                weight_entries: vec!["checkpoint/source_weights".to_string()],
                tensor_roles: Vec::new(),
            },
        );
    } else {
        for tensor in tensors {
            let component = single_file_tensor_component(&tensor.name);
            let entry_name = format!("{component}/checkpoint_tensors/{}", tensor.name);
            let native_entry_name = native_entry_for_single_file_tensor(&tensor.name);
            let metadata = components.entry(component.to_string()).or_insert_with(|| {
                DiffusionComponentMetadata {
                    class_name: Some(single_file_component_class_name(component).to_string()),
                    config_entry: None,
                    weight_entries: Vec::new(),
                    tensor_roles: Vec::new(),
                }
            });
            metadata.tensor_roles.push(DiffusionTensorRole {
                role: tensor.name.clone(),
                entry: entry_name.clone(),
                dtype: tensor.dtype.clone(),
                quant_format: None,
            });
            metadata.weight_entries.push(entry_name.clone());
            if let Some(native_entry_name) = native_entry_name {
                metadata.tensor_roles.push(DiffusionTensorRole {
                    role: tensor.name.clone(),
                    entry: native_entry_name.clone(),
                    dtype: tensor.dtype.clone(),
                    quant_format: None,
                });
                metadata.weight_entries.push(native_entry_name.clone());
                entries.push(DiffusionImportEntry {
                    name: native_entry_name,
                    quant_type: tensor.quant_type,
                    shape: tensor.shape.clone(),
                    group_size: 0,
                    source: DiffusionImportSource::FileSlice {
                        path: tensor.source_path.clone(),
                        offset: tensor.data_offset,
                        len: tensor.data_len,
                    },
                });
            }
            entries.push(DiffusionImportEntry {
                name: entry_name,
                quant_type: tensor.quant_type,
                shape: tensor.shape,
                group_size: 0,
                source: DiffusionImportSource::FileSlice {
                    path: tensor.source_path,
                    offset: tensor.data_offset,
                    len: tensor.data_len,
                },
            });
        }
    }

    add_single_file_generated_configs(&mut entries, &mut components, &pipeline_class)?;
    add_single_file_tokenizer_sidecars(
        &source,
        &mut entries,
        &mut tokenizer_entries,
        &mut tokenizer_2_entries,
    )?;

    let model_name = model_name
        .or_else(|| {
            source
                .file_stem()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "diffusion-checkpoint".to_string());
    let metadata = DiffusionHfqMetadata {
        artifact_kind: DIFFUSION_ARTIFACT_KIND.to_string(),
        schema_version: DIFFUSION_SCHEMA_VERSION,
        pipeline: DiffusionPipelineMetadata {
            class_name: pipeline_class,
            source: source.to_string_lossy().into_owned(),
            model_name,
            latent_channels,
            latent_height: None,
            latent_width: None,
            supported_widths: Vec::new(),
            supported_heights: Vec::new(),
        },
        tokenizer: DiffusionTokenizerMetadata {
            kind: "clip-bpe".to_string(),
            max_length: Some(77),
            entries: tokenizer_entries,
        },
        tokenizer_2: (!tokenizer_2_entries.is_empty()).then_some(DiffusionTokenizerMetadata {
            kind: "clip-bpe".to_string(),
            max_length: Some(77),
            entries: tokenizer_2_entries,
        }),
        batch: DiffusionBatchMetadata {
            max_batch: max_batch.max(1),
            batched_runtime: true,
        },
        quantization: DiffusionQuantizationMetadata::default(),
        components,
    };
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    write_import_entries_to_hfq(&output, &metadata_json, &entries)?;
    inspect_hfq(output).map_err(anyhow::Error::from)
}

fn add_single_file_generated_configs(
    entries: &mut Vec<DiffusionImportEntry>,
    components: &mut BTreeMap<String, DiffusionComponentMetadata>,
    pipeline_class: &str,
) -> anyhow::Result<()> {
    let is_sdxl = pipeline_class == "StableDiffusionXLPipeline";
    if components.contains_key("unet") {
        push_single_file_component_config(
            entries,
            components,
            "unet",
            "config.json",
            if is_sdxl {
                json!({
                    "_class_name": "UNet2DConditionModel",
                    "sample_size": 128,
                    "in_channels": 4,
                    "out_channels": 4,
                    "cross_attention_dim": 2048,
                    "attention_head_dim": [5, 10, 20],
                    "block_out_channels": [320, 640, 1280],
                    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
                    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
                    "layers_per_block": 2,
                    "norm_num_groups": 32,
                    "norm_eps": 1e-5,
                    "center_input_sample": false,
                    "flip_sin_to_cos": true,
                    "freq_shift": 0,
                    "addition_embed_type": "text_time",
                    "addition_time_embed_dim": 256,
                    "projection_class_embeddings_input_dim": 2816
                })
            } else {
                json!({
                    "_class_name": "UNet2DConditionModel",
                    "sample_size": 64,
                    "in_channels": 4,
                    "out_channels": 4,
                    "cross_attention_dim": 768,
                    "attention_head_dim": 8,
                    "block_out_channels": [320, 640, 1280, 1280],
                    "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
                    "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
                    "layers_per_block": 2,
                    "norm_num_groups": 32,
                    "norm_eps": 1e-5,
                    "center_input_sample": false,
                    "flip_sin_to_cos": true,
                    "freq_shift": 0
                })
            },
        )?;
    }
    if components.contains_key("vae") {
        push_single_file_component_config(
            entries,
            components,
            "vae",
            "config.json",
            json!({
                "_class_name": "AutoencoderKL",
                "latent_channels": 4,
                "scaling_factor": 0.18215,
                "block_out_channels": [128, 256, 512, 512],
                "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
                "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
                "norm_num_groups": 32,
                "norm_eps": 1e-6
            }),
        )?;
    }
    if components.contains_key("text_encoder") {
        push_single_file_component_config(
            entries,
            components,
            "text_encoder",
            "config.json",
            json!({
                "_class_name": "CLIPTextModel",
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "max_position_embeddings": 77,
                "vocab_size": 49408
            }),
        )?;
    }
    if components.contains_key("text_encoder_2") {
        push_single_file_component_config(
            entries,
            components,
            "text_encoder_2",
            "config.json",
            json!({
                "_class_name": "CLIPTextModelWithProjection",
                "hidden_size": 1280,
                "intermediate_size": 5120,
                "num_hidden_layers": 32,
                "num_attention_heads": 20,
                "max_position_embeddings": 77,
                "vocab_size": 49408
            }),
        )?;
    }
    push_single_file_component_config(
        entries,
        components,
        "scheduler",
        "scheduler_config.json",
        json!({
            "_class_name": "EulerDiscreteScheduler",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "timestep_spacing": "linspace",
            "steps_offset": 1
        }),
    )?;
    Ok(())
}

fn push_single_file_component_config(
    entries: &mut Vec<DiffusionImportEntry>,
    components: &mut BTreeMap<String, DiffusionComponentMetadata>,
    component: &str,
    config_name: &str,
    config: Value,
) -> anyhow::Result<()> {
    let entry_name = format!("{component}/{config_name}");
    let data = serde_json::to_vec_pretty(&config)?;
    push_import_inline_entry(entries, &entry_name, QT_DIFFUSION_JSON, data);
    let metadata = components.entry(component.to_string()).or_default();
    metadata.class_name = config
        .get("_class_name")
        .and_then(Value::as_str)
        .map(str::to_string);
    metadata.config_entry = Some(entry_name);
    Ok(())
}

fn add_single_file_tokenizer_sidecars(
    source: &Path,
    entries: &mut Vec<DiffusionImportEntry>,
    tokenizer_entries: &mut Vec<String>,
    tokenizer_2_entries: &mut Vec<String>,
) -> anyhow::Result<()> {
    let Some(parent) = source.parent() else {
        return Ok(());
    };
    add_single_file_tokenizer_sidecar(parent, "tokenizer", entries, tokenizer_entries)?;
    add_single_file_tokenizer_sidecar(parent, "tokenizer_2", entries, tokenizer_2_entries)?;
    Ok(())
}

fn add_single_file_tokenizer_sidecar(
    parent: &Path,
    sidecar_dir: &str,
    entries: &mut Vec<DiffusionImportEntry>,
    tokenizer_entries: &mut Vec<String>,
) -> anyhow::Result<()> {
    for name in [
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let path = parent.join(sidecar_dir).join(name);
        if path.is_file() {
            let entry_name = format!("{sidecar_dir}/{name}");
            push_import_file_entry(entries, &entry_name, QT_DIFFUSION_TOKENIZER, path)?;
            tokenizer_entries.push(entry_name);
        }
    }
    Ok(())
}

fn infer_single_file_pipeline_class(tensors: &[SafetensorsTensorEntry]) -> String {
    if tensors.iter().any(|tensor| {
        tensor.name.starts_with("conditioner.embedders.1.")
            || tensor.name.starts_with("conditioner.embedders.0.model.")
    }) {
        "StableDiffusionXLPipeline".to_string()
    } else {
        "StableDiffusionPipeline".to_string()
    }
}

fn infer_single_file_latent_channels(tensors: &[SafetensorsTensorEntry]) -> Option<u32> {
    tensors
        .iter()
        .find(|tensor| {
            (tensor.name == "model.diffusion_model.input_blocks.0.0.weight"
                || tensor.name == "diffusion_model.input_blocks.0.0.weight"
                || tensor.name == "conv_in.weight")
                && tensor.shape.len() == 4
                && tensor.shape[1] > 0
        })
        .map(|tensor| tensor.shape[1])
}

fn single_file_tensor_component(name: &str) -> &'static str {
    if name.starts_with("model.diffusion_model.") || name.starts_with("diffusion_model.") {
        "unet"
    } else if name.starts_with("first_stage_model.") {
        "vae"
    } else if name.starts_with("cond_stage_model.") || name.starts_with("conditioner.embedders.0.")
    {
        "text_encoder"
    } else if name.starts_with("conditioner.embedders.1.") {
        "text_encoder_2"
    } else {
        "checkpoint"
    }
}

fn single_file_component_class_name(component: &str) -> &'static str {
    match component {
        "unet" => "UNet2DConditionModel",
        "vae" => "AutoencoderKL",
        "text_encoder" => "CLIPTextModel",
        "text_encoder_2" => "CLIPTextModelWithProjection",
        _ => "SingleFileCheckpoint",
    }
}

fn native_entry_for_single_file_tensor(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("first_stage_model.") {
        let mapped = ldm_vae_native_tensor_name(rest).unwrap_or_else(|| rest.to_string());
        return Some(format!("vae/tensors/{mapped}"));
    }
    if let Some(rest) = name.strip_prefix("cond_stage_model.transformer.") {
        return Some(format!("text_encoder/tensors/{rest}"));
    }
    if let Some(rest) = name.strip_prefix("conditioner.embedders.0.transformer.") {
        return Some(format!("text_encoder/tensors/{rest}"));
    }
    if let Some(rest) = name.strip_prefix("conditioner.embedders.1.model.") {
        return Some(format!("text_encoder_2/tensors/{rest}"));
    }
    let unet_name = name
        .strip_prefix("model.diffusion_model.")
        .or_else(|| name.strip_prefix("diffusion_model."))?;
    ldm_unet_native_tensor_name(unet_name).map(|mapped| format!("unet/tensors/{mapped}"))
}

fn ldm_vae_native_tensor_name(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("encoder.down.") {
        return ldm_vae_encoder_down_tensor_name(rest);
    }
    if let Some(rest) = name.strip_prefix("encoder.mid.") {
        return ldm_vae_mid_tensor_name("encoder", rest);
    }
    if let Some(rest) = name.strip_prefix("decoder.mid.") {
        return ldm_vae_mid_tensor_name("decoder", rest);
    }
    if let Some(rest) = name.strip_prefix("decoder.up.") {
        return ldm_vae_decoder_up_tensor_name(rest);
    }
    let mapped = match name {
        "encoder.norm_out.weight" => "encoder.conv_norm_out.weight",
        "encoder.norm_out.bias" => "encoder.conv_norm_out.bias",
        "decoder.norm_out.weight" => "decoder.conv_norm_out.weight",
        "decoder.norm_out.bias" => "decoder.conv_norm_out.bias",
        _ => return None,
    };
    Some(mapped.to_string())
}

fn ldm_vae_encoder_down_tensor_name(rest: &str) -> Option<String> {
    let (block_idx, rest) = split_usize_prefix(rest)?;
    if let Some(rest) = rest.strip_prefix("block.") {
        let (layer_idx, rest) = split_usize_prefix(rest)?;
        return map_ldm_vae_resnet_suffix(rest)
            .map(|suffix| format!("encoder.down_blocks.{block_idx}.resnets.{layer_idx}.{suffix}"));
    }
    rest.strip_prefix("downsample.conv.")
        .map(|suffix| format!("encoder.down_blocks.{block_idx}.downsamplers.0.conv.{suffix}"))
}

fn ldm_vae_decoder_up_tensor_name(rest: &str) -> Option<String> {
    const STANDARD_LDM_VAE_MAX_LEVEL: usize = 3;
    let (ldm_block_idx, rest) = split_usize_prefix(rest)?;
    let block_idx = STANDARD_LDM_VAE_MAX_LEVEL.checked_sub(ldm_block_idx)?;
    if let Some(rest) = rest.strip_prefix("block.") {
        let (layer_idx, rest) = split_usize_prefix(rest)?;
        return map_ldm_vae_resnet_suffix(rest)
            .map(|suffix| format!("decoder.up_blocks.{block_idx}.resnets.{layer_idx}.{suffix}"));
    }
    rest.strip_prefix("upsample.conv.")
        .map(|suffix| format!("decoder.up_blocks.{block_idx}.upsamplers.0.conv.{suffix}"))
}

fn ldm_vae_mid_tensor_name(side: &str, rest: &str) -> Option<String> {
    if let Some(rest) = rest.strip_prefix("block_1.") {
        return map_ldm_vae_resnet_suffix(rest)
            .map(|suffix| format!("{side}.mid_block.resnets.0.{suffix}"));
    }
    if let Some(rest) = rest.strip_prefix("attn_1.") {
        return map_ldm_vae_attention_suffix(rest)
            .map(|suffix| format!("{side}.mid_block.attentions.0.{suffix}"));
    }
    if let Some(rest) = rest.strip_prefix("block_2.") {
        return map_ldm_vae_resnet_suffix(rest)
            .map(|suffix| format!("{side}.mid_block.resnets.1.{suffix}"));
    }
    None
}

fn map_ldm_vae_resnet_suffix(rest: &str) -> Option<String> {
    let mapped = match rest {
        "norm1.weight" => "norm1.weight",
        "norm1.bias" => "norm1.bias",
        "conv1.weight" => "conv1.weight",
        "conv1.bias" => "conv1.bias",
        "norm2.weight" => "norm2.weight",
        "norm2.bias" => "norm2.bias",
        "conv2.weight" => "conv2.weight",
        "conv2.bias" => "conv2.bias",
        "nin_shortcut.weight" => "conv_shortcut.weight",
        "nin_shortcut.bias" => "conv_shortcut.bias",
        _ => return None,
    };
    Some(mapped.to_string())
}

fn map_ldm_vae_attention_suffix(rest: &str) -> Option<String> {
    let mapped = match rest {
        "norm.weight" => "group_norm.weight",
        "norm.bias" => "group_norm.bias",
        "q.weight" => "to_q.weight",
        "q.bias" => "to_q.bias",
        "k.weight" => "to_k.weight",
        "k.bias" => "to_k.bias",
        "v.weight" => "to_v.weight",
        "v.bias" => "to_v.bias",
        "proj_out.weight" => "to_out.0.weight",
        "proj_out.bias" => "to_out.0.bias",
        _ => return None,
    };
    Some(mapped.to_string())
}

fn ldm_unet_native_tensor_name(name: &str) -> Option<String> {
    let mapped = match name {
        "input_blocks.0.0.weight" => return Some("conv_in.weight".to_string()),
        "input_blocks.0.0.bias" => return Some("conv_in.bias".to_string()),
        "time_embed.0.weight" => return Some("time_embedding.linear_1.weight".to_string()),
        "time_embed.0.bias" => return Some("time_embedding.linear_1.bias".to_string()),
        "time_embed.2.weight" => return Some("time_embedding.linear_2.weight".to_string()),
        "time_embed.2.bias" => return Some("time_embedding.linear_2.bias".to_string()),
        "out.0.weight" => return Some("conv_norm_out.weight".to_string()),
        "out.0.bias" => return Some("conv_norm_out.bias".to_string()),
        "out.2.weight" => return Some("conv_out.weight".to_string()),
        "out.2.bias" => return Some("conv_out.bias".to_string()),
        _ => name,
    };
    ldm_input_block_native_tensor_name(mapped)
        .or_else(|| ldm_middle_block_native_tensor_name(mapped))
        .or_else(|| ldm_output_block_native_tensor_name(mapped))
}

fn ldm_input_block_native_tensor_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("input_blocks.")?;
    let (block_idx, rest) = split_usize_prefix(rest)?;
    if block_idx == 0 {
        return None;
    }
    if [3, 6, 9].contains(&block_idx) {
        let down_block = (block_idx / 3).saturating_sub(1);
        let sampler_rest = rest
            .strip_prefix("0.op.")
            .or_else(|| rest.strip_prefix("0.conv."))?;
        return Some(format!(
            "down_blocks.{down_block}.downsamplers.0.conv.{sampler_rest}"
        ));
    }
    let down_block = (block_idx - 1) / 3;
    let layer_idx = (block_idx - 1) % 3;
    if layer_idx >= 2 {
        return None;
    }
    let (submodule_idx, rest) = split_usize_prefix(rest)?;
    match submodule_idx {
        0 => map_ldm_resnet_suffix(rest)
            .map(|suffix| format!("down_blocks.{down_block}.resnets.{layer_idx}.{suffix}")),
        1 => Some(format!(
            "down_blocks.{down_block}.attentions.{layer_idx}.{rest}"
        )),
        _ => None,
    }
}

fn ldm_middle_block_native_tensor_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("middle_block.")?;
    let (block_idx, rest) = split_usize_prefix(rest)?;
    match block_idx {
        0 => map_ldm_resnet_suffix(rest).map(|suffix| format!("mid_block.resnets.0.{suffix}")),
        1 => Some(format!("mid_block.attentions.0.{rest}")),
        2 => map_ldm_resnet_suffix(rest).map(|suffix| format!("mid_block.resnets.1.{suffix}")),
        _ => None,
    }
}

fn ldm_output_block_native_tensor_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("output_blocks.")?;
    let (block_idx, rest) = split_usize_prefix(rest)?;
    let up_block = block_idx / 3;
    let layer_idx = block_idx % 3;
    let (submodule_idx, rest) = split_usize_prefix(rest)?;
    if layer_idx == 2 {
        if let Some(sampler_rest) = rest
            .strip_prefix("op.")
            .or_else(|| rest.strip_prefix("conv."))
        {
            return Some(format!(
                "up_blocks.{up_block}.upsamplers.0.conv.{sampler_rest}"
            ));
        }
    }
    match submodule_idx {
        0 => map_ldm_resnet_suffix(rest)
            .map(|suffix| format!("up_blocks.{up_block}.resnets.{layer_idx}.{suffix}")),
        1 => Some(format!(
            "up_blocks.{up_block}.attentions.{layer_idx}.{rest}"
        )),
        2 if layer_idx == 2 => rest
            .strip_prefix("op.")
            .or_else(|| rest.strip_prefix("conv."))
            .map(|sampler_rest| format!("up_blocks.{up_block}.upsamplers.0.conv.{sampler_rest}")),
        _ => None,
    }
}

fn map_ldm_resnet_suffix(rest: &str) -> Option<String> {
    let mapped = match rest {
        "in_layers.0.weight" => "norm1.weight",
        "in_layers.0.bias" => "norm1.bias",
        "in_layers.2.weight" => "conv1.weight",
        "in_layers.2.bias" => "conv1.bias",
        "emb_layers.1.weight" => "time_emb_proj.weight",
        "emb_layers.1.bias" => "time_emb_proj.bias",
        "out_layers.0.weight" => "norm2.weight",
        "out_layers.0.bias" => "norm2.bias",
        "out_layers.3.weight" => "conv2.weight",
        "out_layers.3.bias" => "conv2.bias",
        "skip_connection.weight" => "conv_shortcut.weight",
        "skip_connection.bias" => "conv_shortcut.bias",
        _ => return None,
    };
    Some(mapped.to_string())
}

fn split_usize_prefix(value: &str) -> Option<(usize, &str)> {
    let (head, tail) = value.split_once('.')?;
    Some((head.parse().ok()?, tail))
}

fn add_component(
    source: &Path,
    entries: &mut Vec<DiffusionImportEntry>,
    components: &mut BTreeMap<String, DiffusionComponentMetadata>,
    component: &str,
    weight_files: &[&str],
) -> anyhow::Result<()> {
    let component_dir = source.join(component);
    let config_name =
        if component == "scheduler" && component_dir.join("scheduler_config.json").is_file() {
            "scheduler_config.json"
        } else {
            "config.json"
        };
    let config_path = component_dir.join(config_name);
    let mut metadata = DiffusionComponentMetadata::default();
    if config_path.is_file() {
        let entry_name = format!("{component}/{config_name}");
        let config = read_json(&config_path).unwrap_or_else(|_| json!({}));
        metadata.class_name = config
            .get("_class_name")
            .and_then(Value::as_str)
            .map(str::to_string);
        metadata.config_entry = Some(entry_name.clone());
        push_import_file_entry(entries, &entry_name, QT_DIFFUSION_JSON, config_path)?;
    }
    if let Some(weight_file) = weight_files
        .iter()
        .filter(|candidate| candidate.ends_with(".safetensors"))
        .map(|candidate| format!("{candidate}.index.json"))
        .find(|candidate| component_dir.join(candidate).is_file())
    {
        let index_path = component_dir.join(&weight_file);
        match parse_sharded_safetensors_state_dict(&component_dir, &index_path) {
            Ok(tensors) if !tensors.is_empty() => {
                for tensor in tensors {
                    let entry_name = format!("{component}/tensors/{}", tensor.name);
                    metadata.tensor_roles.push(DiffusionTensorRole {
                        role: tensor.name.clone(),
                        entry: entry_name.clone(),
                        dtype: tensor.dtype.clone(),
                        quant_format: None,
                    });
                    metadata.weight_entries.push(entry_name.clone());
                    entries.push(DiffusionImportEntry {
                        name: entry_name,
                        quant_type: tensor.quant_type,
                        shape: tensor.shape,
                        group_size: 0,
                        source: DiffusionImportSource::FileSlice {
                            path: tensor.source_path,
                            offset: tensor.data_offset,
                            len: tensor.data_len,
                        },
                    });
                }
            }
            _ => {
                let entry_name = format!("{component}/{weight_file}");
                metadata.weight_entries.push(entry_name.clone());
                push_import_file_entry(
                    entries,
                    &entry_name,
                    QT_DIFFUSION_SOURCE_WEIGHTS,
                    index_path,
                )?;
            }
        }
    } else if let Some(weight_file) = weight_files
        .iter()
        .find(|candidate| component_dir.join(candidate).is_file())
    {
        let weight_path = component_dir.join(weight_file);
        if weight_path.is_file() {
            if weight_file.ends_with(".safetensors") {
                match parse_safetensors_state_dict(&weight_path) {
                    Ok(tensors) if !tensors.is_empty() => {
                        for tensor in tensors {
                            let entry_name = format!("{component}/tensors/{}", tensor.name);
                            metadata.tensor_roles.push(DiffusionTensorRole {
                                role: tensor.name.clone(),
                                entry: entry_name.clone(),
                                dtype: tensor.dtype.clone(),
                                quant_format: None,
                            });
                            metadata.weight_entries.push(entry_name.clone());
                            entries.push(DiffusionImportEntry {
                                name: entry_name,
                                quant_type: tensor.quant_type,
                                shape: tensor.shape,
                                group_size: 0,
                                source: DiffusionImportSource::FileSlice {
                                    path: tensor.source_path,
                                    offset: tensor.data_offset,
                                    len: tensor.data_len,
                                },
                            });
                        }
                    }
                    _ => {
                        let entry_name = format!("{component}/{weight_file}");
                        metadata.weight_entries.push(entry_name.clone());
                        push_import_file_entry(
                            entries,
                            &entry_name,
                            QT_DIFFUSION_SOURCE_WEIGHTS,
                            weight_path,
                        )?;
                    }
                }
            } else {
                match parse_pytorch_state_dict(&weight_path) {
                    Ok(tensors) if !tensors.is_empty() => {
                        for tensor in tensors {
                            let entry_name = format!("{component}/tensors/{}", tensor.name);
                            metadata.tensor_roles.push(DiffusionTensorRole {
                                role: tensor.name.clone(),
                                entry: entry_name.clone(),
                                dtype: tensor.dtype.clone(),
                                quant_format: None,
                            });
                            metadata.weight_entries.push(entry_name.clone());
                            entries.push(DiffusionImportEntry {
                                name: entry_name,
                                quant_type: tensor.quant_type,
                                shape: tensor.shape,
                                group_size: 0,
                                source: DiffusionImportSource::ZipMember {
                                    archive_path: weight_path.clone(),
                                    member_name: tensor.member_name,
                                },
                            });
                        }
                    }
                    _ => {
                        let entry_name = format!("{component}/{weight_file}");
                        metadata.weight_entries.push(entry_name.clone());
                        push_import_file_entry(
                            entries,
                            &entry_name,
                            QT_DIFFUSION_SOURCE_WEIGHTS,
                            weight_path,
                        )?;
                    }
                }
            }
        }
    }
    if metadata.config_entry.is_some() || !metadata.weight_entries.is_empty() {
        components.insert(component.to_string(), metadata);
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct DiffusionImportEntry {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    source: DiffusionImportSource,
}

#[derive(Debug, Clone)]
enum DiffusionImportSource {
    Inline(Vec<u8>),
    File(PathBuf),
    FileSlice {
        path: PathBuf,
        offset: u64,
        len: u64,
    },
    ZipMember {
        archive_path: PathBuf,
        member_name: String,
    },
}

fn push_import_file_entry(
    entries: &mut Vec<DiffusionImportEntry>,
    name: &str,
    quant_type: u8,
    source_path: PathBuf,
) -> anyhow::Result<()> {
    let data_size = fs::metadata(&source_path)?.len();
    entries.push(DiffusionImportEntry {
        name: name.to_string(),
        quant_type,
        shape: vec![data_size.min(u32::MAX as u64) as u32],
        group_size: 0,
        source: DiffusionImportSource::File(source_path),
    });
    Ok(())
}

fn push_import_inline_entry(
    entries: &mut Vec<DiffusionImportEntry>,
    name: &str,
    quant_type: u8,
    data: Vec<u8>,
) {
    entries.push(DiffusionImportEntry {
        name: name.to_string(),
        quant_type,
        shape: vec![data.len().min(u32::MAX as usize) as u32],
        group_size: 0,
        source: DiffusionImportSource::Inline(data),
    });
}

fn write_import_entries_to_hfq(
    output: &Path,
    metadata_json: &str,
    entries: &[DiffusionImportEntry],
) -> anyhow::Result<()> {
    let stream_entries = entries
        .iter()
        .map(|entry| {
            let data_len = match &entry.source {
                DiffusionImportSource::Inline(data) => data.len() as u64,
                DiffusionImportSource::File(path) => fs::metadata(path)?.len(),
                DiffusionImportSource::FileSlice { len, .. } => *len,
                DiffusionImportSource::ZipMember {
                    archive_path,
                    member_name,
                } => {
                    MiniZipArchive::open(archive_path)?
                        .entry(member_name)?
                        .uncompressed_size
                }
            };
            Ok(HfqStreamEntry {
                name: entry.name.clone(),
                quant_type: entry.quant_type,
                shape: entry.shape.clone(),
                group_size: entry.group_size,
                data_len,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    write_hfqm_package_streaming(
        output,
        HFQ_ARCH_DIFFUSION,
        metadata_json,
        &stream_entries,
        |i, writer| write_import_entry_payload(&entries[i], writer),
    )?;
    Ok(())
}

fn write_import_entry_payload(
    entry: &DiffusionImportEntry,
    writer: &mut dyn Write,
) -> std::io::Result<()> {
    match &entry.source {
        DiffusionImportSource::Inline(data) => {
            writer.write_all(data)?;
        }
        DiffusionImportSource::File(path) => {
            let mut file = fs::File::open(path)?;
            std::io::copy(&mut file, writer)?;
        }
        DiffusionImportSource::FileSlice { path, offset, len } => {
            let mut file = fs::File::open(path)?;
            file.seek(SeekFrom::Start(*offset))?;
            std::io::copy(&mut file.take(*len), writer)?;
        }
        DiffusionImportSource::ZipMember {
            archive_path,
            member_name,
        } => {
            let archive = MiniZipArchive::open(archive_path).map_err(anyhow_to_io)?;
            archive.copy_entry_to(member_name, writer)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct PytorchTensorEntry {
    name: String,
    member_name: String,
    dtype: String,
    quant_type: u8,
    shape: Vec<u32>,
}

#[derive(Debug, Clone)]
struct SafetensorsTensorEntry {
    name: String,
    dtype: String,
    quant_type: u8,
    shape: Vec<u32>,
    source_path: PathBuf,
    data_offset: u64,
    data_len: u64,
}

fn parse_sharded_safetensors_state_dict(
    component_dir: &Path,
    index_path: &Path,
) -> anyhow::Result<Vec<SafetensorsTensorEntry>> {
    let index = read_json(index_path)?;
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("safetensors index missing weight_map"))?;
    let mut shard_to_tensors: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (tensor, shard) in weight_map {
        let shard = shard
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("safetensors shard for {tensor:?} is not a string"))?;
        shard_to_tensors
            .entry(shard.to_string())
            .or_default()
            .push(tensor.clone());
    }
    let mut tensors = Vec::new();
    for (shard, wanted) in shard_to_tensors {
        let shard_path = component_dir.join(&shard);
        let parsed = parse_safetensors_state_dict(&shard_path)?;
        let parsed_by_name = parsed
            .into_iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect::<BTreeMap<_, _>>();
        for name in wanted {
            let tensor = parsed_by_name.get(&name).ok_or_else(|| {
                anyhow::anyhow!("safetensors index references missing tensor {name:?} in {shard:?}")
            })?;
            tensors.push(tensor.clone());
        }
    }
    Ok(tensors)
}

fn parse_safetensors_state_dict(path: &Path) -> anyhow::Result<Vec<SafetensorsTensorEntry>> {
    let mut file = fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    let data_start = 8u64
        .checked_add(header_len)
        .ok_or_else(|| anyhow::anyhow!("safetensors header length overflow"))?;
    if data_start > file_len {
        anyhow::bail!(
            "safetensors header extends past end of file: header bytes {header_len}, file bytes {file_len}"
        );
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let header: Value = serde_json::from_slice(&header)?;
    let object = header
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("safetensors header must be a JSON object"))?;
    let mut tensors = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let tensor = value.as_object().ok_or_else(|| {
            anyhow::anyhow!("safetensors tensor {name:?} metadata is not an object")
        })?;
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} missing dtype"))?;
        let (dtype, quant_type, byte_width) = safetensors_dtype_info(dtype)
            .ok_or_else(|| anyhow::anyhow!("unsupported safetensors dtype {dtype:?}"))?;
        let shape_values = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} missing shape"))?;
        let shape = shape_values
            .iter()
            .map(|dim| {
                let dim = dim.as_u64().ok_or_else(|| {
                    anyhow::anyhow!("safetensors tensor {name:?} has non-u64 shape dim")
                })?;
                u32::try_from(dim).map_err(|_| {
                    anyhow::anyhow!("safetensors tensor {name:?} shape dim {dim} exceeds u32")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let offsets = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} missing data_offsets"))?;
        if offsets.len() != 2 {
            anyhow::bail!("safetensors tensor {name:?} data_offsets must have two entries");
        }
        let start = offsets[0].as_u64().ok_or_else(|| {
            anyhow::anyhow!("safetensors tensor {name:?} start offset is not u64")
        })?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} end offset is not u64"))?;
        if end < start {
            anyhow::bail!("safetensors tensor {name:?} end offset precedes start offset");
        }
        let data_len = end - start;
        let absolute_start = data_start.checked_add(start).ok_or_else(|| {
            anyhow::anyhow!("safetensors tensor {name:?} absolute offset overflow")
        })?;
        let absolute_end = data_start
            .checked_add(end)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} absolute end overflow"))?;
        if absolute_end > file_len {
            anyhow::bail!(
                "safetensors tensor {name:?} extends past end of file: end {absolute_end}, file {file_len}"
            );
        }
        let elem_count = shape.iter().try_fold(1u64, |acc, &dim| {
            acc.checked_mul(dim as u64)
                .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} shape overflows"))
        })?;
        let expected_len = elem_count
            .checked_mul(byte_width)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} byte length overflows"))?;
        if data_len != expected_len {
            anyhow::bail!(
                "safetensors tensor {name:?} has {data_len} data bytes but shape/dtype expect {expected_len}"
            );
        }
        tensors.push(SafetensorsTensorEntry {
            name: name.clone(),
            dtype: dtype.to_string(),
            quant_type,
            shape,
            source_path: path.to_path_buf(),
            data_offset: absolute_start,
            data_len,
        });
    }
    Ok(tensors)
}

fn safetensors_dtype_info(dtype: &str) -> Option<(&'static str, u8, u64)> {
    match dtype {
        "F16" => Some(("F16", QT_DIFFUSION_TENSOR_F16, 2)),
        "BF16" => Some(("BF16", QT_DIFFUSION_TENSOR_BF16, 2)),
        "F32" => Some(("F32", QT_DIFFUSION_TENSOR_F32, 4)),
        _ => None,
    }
}

fn parse_pytorch_state_dict(path: &Path) -> anyhow::Result<Vec<PytorchTensorEntry>> {
    let archive = MiniZipArchive::open(path)?;
    let data_pkl_name = archive
        .entries
        .keys()
        .find(|name| name.ends_with("/data.pkl") || name.as_str() == "data.pkl")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("PyTorch archive missing data.pkl"))?;
    let root = data_pkl_name
        .strip_suffix("data.pkl")
        .unwrap_or("")
        .trim_end_matches('/');
    let pickle = archive.read_entry(&data_pkl_name)?;
    let mut tensors = parse_pytorch_pickle_tensor_index(&pickle)?;
    tensors.retain(|tensor| {
        archive
            .entry(&format!("{root}/data/{}", tensor.storage_key))
            .is_ok()
    });
    Ok(tensors
        .into_iter()
        .map(|tensor| PytorchTensorEntry {
            name: tensor.name,
            member_name: format!("{root}/data/{}", tensor.storage_key),
            dtype: tensor.dtype,
            quant_type: tensor.quant_type,
            shape: tensor.shape,
        })
        .collect())
}

#[derive(Debug, Clone)]
struct ParsedPytorchTensor {
    name: String,
    storage_key: String,
    dtype: String,
    quant_type: u8,
    shape: Vec<u32>,
}

#[derive(Debug, Clone)]
enum PickleValue {
    Mark,
    Str(String),
    Int(i64),
    Bool(()),
    Tuple(Vec<PickleValue>),
    Global {
        module: String,
        name: String,
    },
    StorageRef {
        key: String,
        dtype: String,
        quant_type: u8,
    },
    Tensor(ParsedPytorchTensor),
    Other,
}

fn parse_pytorch_pickle_tensor_index(data: &[u8]) -> anyhow::Result<Vec<ParsedPytorchTensor>> {
    let mut pos = 0usize;
    let mut stack: Vec<PickleValue> = Vec::new();
    let mut memo: BTreeMap<u32, PickleValue> = BTreeMap::new();
    let mut tensors = Vec::new();
    while pos < data.len() {
        let op = data[pos];
        pos += 1;
        match op {
            0x80 => pos += 1, // PROTO
            b'c' => {
                let module = read_pickle_line(data, &mut pos)?;
                let name = read_pickle_line(data, &mut pos)?;
                stack.push(PickleValue::Global { module, name });
            }
            b'q' => {
                let idx = read_u8(data, &mut pos)? as u32;
                if let Some(value) = stack.last().cloned() {
                    memo.insert(idx, value);
                }
            }
            b'r' => {
                let idx = read_u32(data, &mut pos)?;
                if let Some(value) = stack.last().cloned() {
                    memo.insert(idx, value);
                }
            }
            b'h' => {
                let idx = read_u8(data, &mut pos)? as u32;
                stack.push(memo.get(&idx).cloned().unwrap_or(PickleValue::Other));
            }
            b'j' => {
                let idx = read_u32(data, &mut pos)?;
                stack.push(memo.get(&idx).cloned().unwrap_or(PickleValue::Other));
            }
            b'(' => stack.push(PickleValue::Mark),
            b')' => stack.push(PickleValue::Tuple(Vec::new())),
            b'X' => {
                let len = read_u32(data, &mut pos)? as usize;
                let bytes = read_bytes(data, &mut pos, len)?;
                stack.push(PickleValue::Str(String::from_utf8_lossy(bytes).to_string()));
            }
            b'U' => {
                let len = read_u8(data, &mut pos)? as usize;
                let bytes = read_bytes(data, &mut pos, len)?;
                stack.push(PickleValue::Str(String::from_utf8_lossy(bytes).to_string()));
            }
            b'K' => stack.push(PickleValue::Int(read_u8(data, &mut pos)? as i64)),
            b'M' => stack.push(PickleValue::Int(read_u16(data, &mut pos)? as i64)),
            b'J' => stack.push(PickleValue::Int(read_i32(data, &mut pos)? as i64)),
            0x88 => stack.push(PickleValue::Bool(())),
            0x89 => stack.push(PickleValue::Bool(())),
            b'N' => stack.push(PickleValue::Other),
            b'}' | b']' => stack.push(PickleValue::Other),
            b't' => {
                let values = pop_to_mark(&mut stack);
                stack.push(PickleValue::Tuple(values));
            }
            0x85 => {
                let Some(value) = stack.pop() else {
                    anyhow::bail!("pickle TUPLE1 stack underflow");
                };
                stack.push(PickleValue::Tuple(vec![value]));
            }
            0x86 => {
                let b = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle TUPLE2 stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle TUPLE2 stack underflow"))?;
                stack.push(PickleValue::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle TUPLE3 stack underflow"))?;
                let b = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle TUPLE3 stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle TUPLE3 stack underflow"))?;
                stack.push(PickleValue::Tuple(vec![a, b, c]));
            }
            b'Q' => {
                let value = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle BINPERSID stack underflow"))?;
                stack.push(storage_ref_from_pickle_value(value));
            }
            b'R' => {
                let args = stack.pop().unwrap_or(PickleValue::Other);
                let callable = stack.pop().unwrap_or(PickleValue::Other);
                let reduced = reduce_pickle_value(callable, args);
                if let PickleValue::Tensor(mut tensor) = reduced.clone() {
                    if let Some(name) = previous_stack_string(&stack) {
                        tensor.name = name;
                        tensors.push(tensor.clone());
                    }
                    stack.push(PickleValue::Tensor(tensor));
                } else {
                    stack.push(reduced);
                }
            }
            b'b' => {
                let _ = stack.pop();
            }
            b's' => {
                let _ = stack.pop();
                let _ = stack.pop();
            }
            b'u' | b'e' => {
                let _ = pop_to_mark(&mut stack);
            }
            b'.' => break,
            _ => {
                anyhow::bail!("unsupported pickle opcode 0x{op:02x} at {}", pos - 1);
            }
        }
    }
    Ok(tensors)
}

fn reduce_pickle_value(callable: PickleValue, args: PickleValue) -> PickleValue {
    match callable {
        PickleValue::Global { module, name }
            if module == "torch._utils" && name == "_rebuild_tensor_v2" =>
        {
            tensor_from_rebuild_args(args)
                .map(PickleValue::Tensor)
                .unwrap_or(PickleValue::Other)
        }
        _ => PickleValue::Other,
    }
}

fn tensor_from_rebuild_args(args: PickleValue) -> Option<ParsedPytorchTensor> {
    let PickleValue::Tuple(items) = args else {
        return None;
    };
    let storage = match items.first()? {
        PickleValue::StorageRef {
            key,
            dtype,
            quant_type,
        } => (key.clone(), dtype.clone(), *quant_type),
        _ => return None,
    };
    let shape = tuple_ints(items.get(2)?)?
        .into_iter()
        .map(|dim| dim as u32)
        .collect();
    Some(ParsedPytorchTensor {
        name: String::new(),
        storage_key: storage.0,
        dtype: storage.1,
        quant_type: storage.2,
        shape,
    })
}

fn storage_ref_from_pickle_value(value: PickleValue) -> PickleValue {
    let PickleValue::Tuple(items) = value else {
        return PickleValue::Other;
    };
    if items.len() < 5 {
        return PickleValue::Other;
    }
    let PickleValue::Global { module, name } = &items[1] else {
        return PickleValue::Other;
    };
    if module != "torch" {
        return PickleValue::Other;
    }
    let Some((dtype, quant_type)) = torch_storage_dtype(name) else {
        return PickleValue::Other;
    };
    let PickleValue::Str(key) = &items[2] else {
        return PickleValue::Other;
    };
    PickleValue::StorageRef {
        key: key.clone(),
        dtype: dtype.to_string(),
        quant_type,
    }
}

fn torch_storage_dtype(storage: &str) -> Option<(&'static str, u8)> {
    match storage {
        "HalfStorage" => Some(("F16", QT_DIFFUSION_TENSOR_F16)),
        "FloatStorage" => Some(("F32", QT_DIFFUSION_TENSOR_F32)),
        "BFloat16Storage" => Some(("BF16", QT_DIFFUSION_TENSOR_BF16)),
        _ => None,
    }
}

fn previous_stack_string(stack: &[PickleValue]) -> Option<String> {
    stack.iter().rev().find_map(|value| match value {
        PickleValue::Str(value) => Some(value.clone()),
        _ => None,
    })
}

fn tuple_ints(value: &PickleValue) -> Option<Vec<i64>> {
    let PickleValue::Tuple(items) = value else {
        return None;
    };
    items
        .iter()
        .map(|item| match item {
            PickleValue::Int(value) => Some(*value),
            _ => None,
        })
        .collect()
}

fn pop_to_mark(stack: &mut Vec<PickleValue>) -> Vec<PickleValue> {
    let mut values = Vec::new();
    while let Some(value) = stack.pop() {
        if matches!(value, PickleValue::Mark) {
            values.reverse();
            return values;
        }
        values.push(value);
    }
    values.reverse();
    values
}

fn read_pickle_line(data: &[u8], pos: &mut usize) -> anyhow::Result<String> {
    let start = *pos;
    while *pos < data.len() && data[*pos] != b'\n' {
        *pos += 1;
    }
    if *pos >= data.len() {
        anyhow::bail!("pickle line extends past end of data");
    }
    let line = String::from_utf8_lossy(&data[start..*pos]).to_string();
    *pos += 1;
    Ok(line)
}

fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, len: usize) -> anyhow::Result<&'a [u8]> {
    if *pos + len > data.len() {
        anyhow::bail!("read past end");
    }
    let out = &data[*pos..*pos + len];
    *pos += len;
    Ok(out)
}

fn read_u8(data: &[u8], pos: &mut usize) -> anyhow::Result<u8> {
    Ok(read_bytes(data, pos, 1)?[0])
}

fn read_u16(data: &[u8], pos: &mut usize) -> anyhow::Result<u16> {
    Ok(u16::from_le_bytes(read_bytes(data, pos, 2)?.try_into()?))
}

fn read_u32(data: &[u8], pos: &mut usize) -> anyhow::Result<u32> {
    Ok(u32::from_le_bytes(read_bytes(data, pos, 4)?.try_into()?))
}

fn read_i32(data: &[u8], pos: &mut usize) -> anyhow::Result<i32> {
    Ok(i32::from_le_bytes(read_bytes(data, pos, 4)?.try_into()?))
}

#[derive(Debug, Clone)]
struct MiniZipEntry {
    compressed_size: u64,
    uncompressed_size: u64,
    data_offset: u64,
    compression_method: u16,
}

#[derive(Debug, Clone)]
struct MiniZipArchive {
    path: PathBuf,
    entries: BTreeMap<String, MiniZipEntry>,
}

impl MiniZipArchive {
    fn open(path: &Path) -> anyhow::Result<Self> {
        let mut file = fs::File::open(path)?;
        let len = file.metadata()?.len();
        let tail_len = len.min(66_000) as usize;
        file.seek(SeekFrom::End(-(tail_len as i64)))?;
        let mut tail = vec![0u8; tail_len];
        file.read_exact(&mut tail)?;
        let eocd_pos = tail
            .windows(4)
            .rposition(|window| window == b"PK\x05\x06")
            .ok_or_else(|| anyhow::anyhow!("zip EOCD not found in {}", path.display()))?;
        let eocd = &tail[eocd_pos..];
        if eocd.len() < 22 {
            anyhow::bail!("truncated zip EOCD");
        }
        let central_size = u32::from_le_bytes(eocd[12..16].try_into()?) as u64;
        let central_offset = u32::from_le_bytes(eocd[16..20].try_into()?) as u64;
        let mut central = vec![0u8; central_size as usize];
        file.seek(SeekFrom::Start(central_offset))?;
        file.read_exact(&mut central)?;
        let mut pos = 0usize;
        let mut entries = BTreeMap::new();
        while pos + 46 <= central.len() {
            if &central[pos..pos + 4] != b"PK\x01\x02" {
                break;
            }
            let compression_method = u16::from_le_bytes(central[pos + 10..pos + 12].try_into()?);
            let compressed_size =
                u32::from_le_bytes(central[pos + 20..pos + 24].try_into()?) as u64;
            let uncompressed_size =
                u32::from_le_bytes(central[pos + 24..pos + 28].try_into()?) as u64;
            let name_len = u16::from_le_bytes(central[pos + 28..pos + 30].try_into()?) as usize;
            let extra_len = u16::from_le_bytes(central[pos + 30..pos + 32].try_into()?) as usize;
            let comment_len = u16::from_le_bytes(central[pos + 32..pos + 34].try_into()?) as usize;
            let local_offset = u32::from_le_bytes(central[pos + 42..pos + 46].try_into()?) as u64;
            let name_start = pos + 46;
            let name_end = name_start + name_len;
            if name_end > central.len() {
                anyhow::bail!("truncated zip central directory name");
            }
            let name = String::from_utf8_lossy(&central[name_start..name_end]).to_string();
            let data_offset = local_data_offset(&mut file, local_offset)?;
            entries.insert(
                name,
                MiniZipEntry {
                    compressed_size,
                    uncompressed_size,
                    data_offset,
                    compression_method,
                },
            );
            pos = name_end + extra_len + comment_len;
        }
        Ok(Self {
            path: path.to_path_buf(),
            entries,
        })
    }

    fn entry(&self, name: &str) -> anyhow::Result<&MiniZipEntry> {
        self.entries
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("zip entry {name:?} not found"))
    }

    fn read_entry(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        let entry = self.entry(name)?;
        if entry.compression_method != 0 {
            anyhow::bail!("zip entry {name:?} is compressed; only stored entries are supported");
        }
        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(entry.data_offset))?;
        let mut data = vec![0u8; entry.uncompressed_size as usize];
        file.read_exact(&mut data)?;
        Ok(data)
    }

    fn copy_entry_to(&self, name: &str, writer: &mut dyn Write) -> std::io::Result<()> {
        let entry = self.entry(name).map_err(anyhow_to_io)?;
        if entry.compression_method != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("zip entry {name:?} is compressed"),
            ));
        }
        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(entry.data_offset))?;
        let mut limited = file.take(entry.compressed_size);
        std::io::copy(&mut limited, writer)?;
        Ok(())
    }
}

fn local_data_offset(file: &mut fs::File, local_offset: u64) -> anyhow::Result<u64> {
    let mut header = [0u8; 30];
    file.seek(SeekFrom::Start(local_offset))?;
    file.read_exact(&mut header)?;
    if &header[0..4] != b"PK\x03\x04" {
        anyhow::bail!("invalid zip local header");
    }
    let name_len = u16::from_le_bytes(header[26..28].try_into()?) as u64;
    let extra_len = u16::from_le_bytes(header[28..30].try_into()?) as u64;
    Ok(local_offset + 30 + name_len + extra_len)
}

fn anyhow_to_io(error: anyhow::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
}

fn read_json(path: impl AsRef<Path>) -> anyhow::Result<Value> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

    const DEFAULT_TINY_SD_HFQ: &str = "/tmp/hipfire-tiny-sd-diffusion.hfq";

    fn tiny_sd_hfq_path() -> PathBuf {
        std::env::var_os("HIPFIRE_TINY_SD_HFQ")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_TINY_SD_HFQ))
    }

    fn skip_missing_tiny_sd(path: &Path) -> bool {
        if path.exists() {
            false
        } else {
            eprintln!(
                "skip: set HIPFIRE_TINY_SD_HFQ or create {}",
                DEFAULT_TINY_SD_HFQ
            );
            true
        }
    }

    #[test]
    fn parses_diffusion_metadata() {
        let metadata = minimal_metadata();
        let json = serde_json::to_string(&metadata).unwrap();
        assert_eq!(parse_diffusion_metadata(&json).unwrap(), metadata);
    }

    #[test]
    fn native_runtime_metadata_support_reports_runtime_boundaries() {
        let mut metadata = minimal_metadata();
        assert!(native_runtime_metadata_support_error(&metadata).is_none());

        metadata.quantization.weight_format = "metadata-only".to_string();
        let error = native_runtime_metadata_support_error(&metadata).unwrap();
        assert!(error.contains("metadata only"));

        metadata.quantization.weight_format = "oq4".to_string();
        assert!(native_runtime_metadata_support_error(&metadata).is_none());

        metadata.quantization.activation_format = "fp8".to_string();
        let error = native_runtime_metadata_support_error(&metadata).unwrap();
        assert!(error.contains("activation_format"));
        assert!(error.contains("fp8"));

        metadata.quantization.activation_format = "fp16".to_string();
        metadata.quantization.tensor_roles_version = 2;
        let error = native_runtime_metadata_support_error(&metadata).unwrap();
        assert!(error.contains("tensor_roles_version 2"));
    }

    #[test]
    fn native_source_runtime_support_rejects_transformer_pipeline_classes() {
        let mut metadata = minimal_metadata();
        metadata.pipeline.class_name = "Krea2Pipeline".to_string();
        metadata.components.remove("unet");
        metadata.components.insert(
            "transformer".to_string(),
            DiffusionComponentMetadata {
                class_name: Some("Krea2Transformer2DModel".to_string()),
                config_entry: Some("transformer/config.json".to_string()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );

        let error = native_runtime_metadata_support_error(&metadata).unwrap();

        assert!(error.contains("Stable Diffusion UNet-family"));
        assert!(error.contains("Krea2Pipeline"));
        assert!(error.contains("transformer denoiser"));
    }

    #[test]
    fn inspect_hfq_reports_metadata_runtime_support_without_loading_runtime() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-runtime-support-inspect-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let source_path = dir.join("source.hfq");
        let quantized_path = dir.join("quantized.hfq");
        let source_metadata = tiny_runtime_metadata();
        let mut quantized_metadata = tiny_runtime_metadata();
        quantized_metadata.quantization.weight_format = "oq4".to_string();
        write_hfqm_package_mem(
            &source_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&source_metadata).unwrap(),
            &tiny_complete_runtime_tensors(),
        )
        .unwrap();
        write_hfqm_package_mem(
            &quantized_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&quantized_metadata).unwrap(),
            &tiny_complete_runtime_tensors(),
        )
        .unwrap();

        let source = inspect_hfq_with_runtime_support(&source_path).unwrap();
        let quantized = inspect_hfq_with_runtime_support(&quantized_path).unwrap();

        assert!(source.runtime_support.supported);
        assert_eq!(
            source.runtime_support.runtime_kind,
            Some(DiffusionRuntimeKind::CpuSourceReference)
        );
        assert_eq!(source.runtime_support.reason, None);
        assert!(quantized.runtime_support.supported);
        assert_eq!(
            quantized.runtime_support.runtime_kind,
            Some(DiffusionRuntimeKind::CpuSourceReference)
        );
        assert_eq!(quantized.runtime_support.reason, None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_non_diffusion_metadata() {
        let err = parse_diffusion_metadata(r#"{"artifact_kind":"llm","schema_version":1,"pipeline":{"class_name":"x","source":"x"}}"#)
            .unwrap_err();
        assert!(err.to_string().contains("artifact_kind"));
    }

    #[test]
    fn lenient_config_json_accepts_diffusers_non_finite_tokens() {
        let parsed = parse_json_lenient(
            r#"{"_class_name":"DPMSolverMultistepScheduler","lambda_min_clipped":-Infinity}"#,
        )
        .unwrap();
        assert_eq!(
            parsed.get("_class_name").and_then(Value::as_str),
            Some("DPMSolverMultistepScheduler")
        );
        assert!(parsed.get("lambda_min_clipped").unwrap().is_null());
    }

    #[test]
    fn validates_batched_request_limits() {
        let metadata = minimal_metadata();
        let request = DiffusionBatchRequest {
            prompts: vec![
                DiffusionPrompt {
                    prompt: "a".into(),
                    negative_prompt: String::new(),
                    seed: 1,
                    subseed: None,
                },
                DiffusionPrompt {
                    prompt: "b".into(),
                    negative_prompt: String::new(),
                    seed: 2,
                    subseed: None,
                },
            ],
            width: 512,
            height: 512,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 20,
            cfg_scale: 7.0,
            scheduler: "DPM++ 2M".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };
        assert!(validate_batch_request(&metadata, &request).is_ok());
    }

    #[test]
    fn inspect_hfq_detects_diffusion_container() {
        let dir =
            std::env::temp_dir().join(format!("hipfire-diffusion-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        fs::write(&config_path, b"{}").unwrap();
        let hfq_path = dir.join("model.hfq");
        let metadata = minimal_metadata();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &[HfqMemTensor {
                name: "unet/config.json".into(),
                quant_type: QT_DIFFUSION_JSON,
                shape: vec![2],
                group_size: 0,
                data: b"{}".to_vec(),
            }],
        )
        .unwrap();
        let summary = inspect_hfq(&hfq_path).unwrap();
        assert_eq!(summary.pipeline_class, "StableDiffusionPipeline");
        assert!(is_diffusion_hfq(&hfq_path));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn imports_minimal_diffusers_snapshot_to_hfq() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-import-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let source = dir.join("snapshot");
        fs::create_dir_all(source.join("text_encoder")).unwrap();
        fs::create_dir_all(source.join("unet")).unwrap();
        fs::create_dir_all(source.join("vae")).unwrap();
        fs::create_dir_all(source.join("scheduler")).unwrap();
        fs::create_dir_all(source.join("tokenizer")).unwrap();
        fs::write(
            source.join("model_index.json"),
            br#"{"_class_name":"StableDiffusionPipeline"}"#,
        )
        .unwrap();
        fs::write(
            source.join("unet/config.json"),
            br#"{"_class_name":"UNet2DConditionModel","sample_size":64,"in_channels":4}"#,
        )
        .unwrap();
        fs::write(
            source.join("vae/config.json"),
            br#"{"_class_name":"AutoencoderKL","latent_channels":4}"#,
        )
        .unwrap();
        fs::write(
            source.join("text_encoder/config.json"),
            br#"{"_class_name":"CLIPTextModel"}"#,
        )
        .unwrap();
        fs::write(
            source.join("scheduler/scheduler_config.json"),
            br#"{"_class_name":"DPMSolverMultistepScheduler"}"#,
        )
        .unwrap();
        fs::write(source.join("tokenizer/vocab.json"), b"{}").unwrap();
        fs::write(source.join("unet/diffusion_pytorch_model.bin"), b"unet").unwrap();
        fs::write(source.join("vae/diffusion_pytorch_model.bin"), b"vae").unwrap();
        fs::write(source.join("text_encoder/pytorch_model.bin"), b"text").unwrap();

        let output = dir.join("tiny.hfq");
        let summary = import_diffusers_to_hfq(DiffusersImportOptions {
            source,
            output: output.clone(),
            model_name: Some("tiny-sd".into()),
            max_batch: 3,
            metadata_only: false,
        })
        .unwrap();

        assert_eq!(summary.model_name, "tiny-sd");
        assert_eq!(summary.max_batch, 3);
        assert!(is_diffusion_hfq(&output));

        let hfq = HfqFile::open_index_only(&output).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        assert_eq!(config.pipeline_class, "StableDiffusionPipeline");
        assert_eq!(config.unet.sample_size, Some(64));
        assert_eq!(config.latent_channels, 4);
        assert_eq!(config.scheduler.class_name, "DPMSolverMultistepScheduler");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn imports_transformer_pipeline_metadata_without_marking_runtime_supported() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-transformer-import-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let source = dir.join("snapshot");
        fs::create_dir_all(source.join("text_encoder")).unwrap();
        fs::create_dir_all(source.join("transformer")).unwrap();
        fs::create_dir_all(source.join("vae")).unwrap();
        fs::create_dir_all(source.join("scheduler")).unwrap();
        fs::write(
            source.join("model_index.json"),
            br#"{"_class_name":"Krea2Pipeline","text_encoder":["transformers","Qwen3VLModel"],"transformer":["diffusers","Krea2Transformer2DModel"],"vae":["diffusers","AutoencoderKLQwenImage"],"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"]}"#,
        )
        .unwrap();
        fs::write(
            source.join("text_encoder/config.json"),
            br#"{"_class_name":"Qwen3VLModel","hidden_size":2560}"#,
        )
        .unwrap();
        fs::write(
            source.join("transformer/config.json"),
            br#"{"_class_name":"Krea2Transformer2DModel","in_channels":64,"out_channels":16,"num_layers":28}"#,
        )
        .unwrap();
        fs::write(
            source.join("vae/config.json"),
            br#"{"_class_name":"AutoencoderKLQwenImage","z_dim":16,"latents_mean":[-0.75,0.25],"latents_std":[2.0,1.5]}"#,
        )
        .unwrap();
        fs::write(
            source.join("scheduler/scheduler_config.json"),
            br#"{"_class_name":"FlowMatchEulerDiscreteScheduler","num_train_timesteps":1000,"shift":1.0,"shift_terminal":0.02,"invert_sigmas":false,"use_dynamic_shifting":true,"time_shift_type":"exponential"}"#,
        )
        .unwrap();

        let output = dir.join("krea.hfq");
        let summary = import_diffusers_to_hfq(DiffusersImportOptions {
            source,
            output: output.clone(),
            model_name: Some("krea".into()),
            max_batch: 1,
            metadata_only: false,
        })
        .unwrap();
        let hfq = HfqFile::open_index_only(&output).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let inspection = inspect_hfq_with_runtime_support(&output).unwrap();

        assert_eq!(summary.pipeline_class, "Krea2Pipeline");
        assert_eq!(metadata.pipeline.latent_channels, Some(16));
        assert!(metadata.components.contains_key("transformer"));
        assert_eq!(
            metadata.components["transformer"].class_name.as_deref(),
            Some("Krea2Transformer2DModel")
        );
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        assert_eq!(
            config.scheduler.class_name,
            "FlowMatchEulerDiscreteScheduler"
        );
        assert_eq!(config.latent_channels, 16);
        assert_eq!(config.vae.z_dim, Some(16));
        assert_eq!(config.vae.latents_mean, vec![-0.75, 0.25]);
        assert_eq!(config.vae.latents_std, vec![2.0, 1.5]);
        assert_eq!(config.scheduler.shift, Some(1.0));
        assert_eq!(config.scheduler.shift_terminal, Some(0.02));
        assert_eq!(config.scheduler.invert_sigmas, Some(false));
        assert_eq!(config.scheduler.use_dynamic_shifting, Some(true));
        assert_eq!(
            config.scheduler.time_shift_type.as_deref(),
            Some("exponential")
        );
        assert!(!inspection.runtime_support.supported);
        assert!(inspection
            .runtime_support
            .reason
            .as_deref()
            .unwrap()
            .contains("transformer denoiser"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_only_import_skips_weights_and_reports_non_runnable_artifact() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-metadata-only-import-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let source = dir.join("snapshot");
        fs::create_dir_all(source.join("transformer")).unwrap();
        fs::create_dir_all(source.join("vae")).unwrap();
        fs::create_dir_all(source.join("scheduler")).unwrap();
        fs::write(
            source.join("model_index.json"),
            br#"{"_class_name":"QwenImagePipeline","transformer":["diffusers","QwenImageTransformer2DModel"],"vae":["diffusers","AutoencoderKLQwenImage"],"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"]}"#,
        )
        .unwrap();
        fs::write(
            source.join("transformer/config.json"),
            br#"{"_class_name":"QwenImageTransformer2DModel","in_channels":64,"out_channels":16}"#,
        )
        .unwrap();
        fs::write(
            source.join("transformer/diffusion_pytorch_model.safetensors.index.json"),
            br#"{"metadata":{"total_size":4},"weight_map":{"x":"missing.safetensors"}}"#,
        )
        .unwrap();
        fs::write(
            source.join("vae/config.json"),
            br#"{"_class_name":"AutoencoderKLQwenImage","z_dim":16}"#,
        )
        .unwrap();
        fs::write(
            source.join("scheduler/scheduler_config.json"),
            br#"{"_class_name":"FlowMatchEulerDiscreteScheduler","num_train_timesteps":1000,"shift":1.0}"#,
        )
        .unwrap();

        let output = dir.join("qwen-metadata.hfq");
        let summary = import_diffusers_to_hfq(DiffusersImportOptions {
            source,
            output: output.clone(),
            model_name: Some("qwen-image".into()),
            max_batch: 1,
            metadata_only: true,
        })
        .unwrap();
        let hfq = HfqFile::open_index_only(&output).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let inspection = inspect_hfq_with_runtime_support(&output).unwrap();

        assert_eq!(summary.weight_format, "metadata-only");
        assert_eq!(metadata.quantization.weight_format, "metadata-only");
        assert_eq!(metadata.pipeline.latent_channels, Some(16));
        assert!(metadata.components["transformer"].weight_entries.is_empty());
        assert_eq!(
            metadata.components["transformer"].config_entry.as_deref(),
            Some("transformer/config.json")
        );
        assert!(!inspection.runtime_support.supported);
        assert!(inspection
            .runtime_support
            .reason
            .as_deref()
            .unwrap()
            .contains("metadata only"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn imports_sdxl_secondary_text_encoder_and_tokenizer_metadata() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-sdxl-import-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let source = dir.join("snapshot");
        fs::create_dir_all(source.join("text_encoder")).unwrap();
        fs::create_dir_all(source.join("text_encoder_2")).unwrap();
        fs::create_dir_all(source.join("unet")).unwrap();
        fs::create_dir_all(source.join("vae")).unwrap();
        fs::create_dir_all(source.join("scheduler")).unwrap();
        fs::create_dir_all(source.join("tokenizer")).unwrap();
        fs::create_dir_all(source.join("tokenizer_2")).unwrap();
        fs::write(
            source.join("model_index.json"),
            br#"{"_class_name":"StableDiffusionXLPipeline"}"#,
        )
        .unwrap();
        fs::write(
            source.join("unet/config.json"),
            br#"{"_class_name":"UNet2DConditionModel","sample_size":128,"in_channels":4,"addition_embed_type":"text_time"}"#,
        )
        .unwrap();
        fs::write(
            source.join("vae/config.json"),
            br#"{"_class_name":"AutoencoderKL","latent_channels":4}"#,
        )
        .unwrap();
        fs::write(
            source.join("text_encoder/config.json"),
            br#"{"_class_name":"CLIPTextModel","hidden_size":768}"#,
        )
        .unwrap();
        fs::write(
            source.join("text_encoder_2/config.json"),
            br#"{"_class_name":"CLIPTextModelWithProjection","hidden_size":1280,"projection_dim":1280}"#,
        )
        .unwrap();
        fs::write(
            source.join("scheduler/scheduler_config.json"),
            br#"{"_class_name":"EulerDiscreteScheduler"}"#,
        )
        .unwrap();
        fs::write(source.join("tokenizer/vocab.json"), b"{}").unwrap();
        fs::write(source.join("tokenizer_2/vocab.json"), b"{}").unwrap();
        fs::write(source.join("text_encoder/pytorch_model.bin"), b"text").unwrap();
        fs::write(source.join("text_encoder_2/pytorch_model.bin"), b"text2").unwrap();
        fs::write(source.join("unet/diffusion_pytorch_model.bin"), b"unet").unwrap();
        fs::write(source.join("vae/diffusion_pytorch_model.bin"), b"vae").unwrap();

        let output = dir.join("tiny-sdxl.hfq");
        import_diffusers_to_hfq(DiffusersImportOptions {
            source,
            output: output.clone(),
            model_name: Some("tiny-sdxl".into()),
            max_batch: 2,
            metadata_only: false,
        })
        .unwrap();

        let hfq = HfqFile::open_index_only(&output).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        assert_eq!(metadata.pipeline.class_name, "StableDiffusionXLPipeline");
        assert!(metadata.components.contains_key("text_encoder_2"));
        assert_eq!(
            metadata.components["text_encoder_2"]
                .config_entry
                .as_deref(),
            Some("text_encoder_2/config.json")
        );
        assert_eq!(
            metadata.tokenizer_2.as_ref().unwrap().entries,
            vec!["tokenizer_2/vocab.json"]
        );
        assert!(hfq.find_tensor_info("text_encoder_2/config.json").is_some());
        assert!(hfq.find_tensor_info("tokenizer_2/vocab.json").is_some());

        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        assert_eq!(
            config.text_encoder_2.as_ref().unwrap().class_name,
            "CLIPTextModelWithProjection"
        );
        let pipeline = DiffusionPipeline::open_hfq(&output).unwrap();
        assert!(pipeline.native_runtime.is_none());
        let native_runtime_error = pipeline.native_runtime_error.as_deref().unwrap();
        assert!(!native_runtime_error.contains("dual-text-encoder"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn imports_diffusers_safetensors_as_hfq_tensor_entries() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-safetensors-import-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let source = dir.join("snapshot");
        fs::create_dir_all(source.join("text_encoder")).unwrap();
        fs::create_dir_all(source.join("unet")).unwrap();
        fs::create_dir_all(source.join("vae")).unwrap();
        fs::create_dir_all(source.join("scheduler")).unwrap();
        fs::write(
            source.join("model_index.json"),
            br#"{"_class_name":"StableDiffusionPipeline"}"#,
        )
        .unwrap();
        fs::write(
            source.join("unet/config.json"),
            br#"{"_class_name":"UNet2DConditionModel","sample_size":2,"in_channels":1}"#,
        )
        .unwrap();
        fs::write(
            source.join("vae/config.json"),
            br#"{"_class_name":"AutoencoderKL","latent_channels":1}"#,
        )
        .unwrap();
        fs::write(
            source.join("text_encoder/config.json"),
            br#"{"_class_name":"CLIPTextModel"}"#,
        )
        .unwrap();
        fs::write(
            source.join("scheduler/scheduler_config.json"),
            br#"{"_class_name":"EulerDiscreteScheduler"}"#,
        )
        .unwrap();
        write_safetensors_fixture(
            &source.join("unet/diffusion_pytorch_model.safetensors"),
            &[("conv_in.weight", "F32", &[1, 1], &[0x00, 0x00, 0xc0, 0x3f])],
        );
        write_safetensors_fixture(
            &source.join("vae/diffusion_pytorch_model.safetensors"),
            &[("post_quant_conv.weight", "F16", &[1], &[0x00, 0x3c])],
        );
        write_safetensors_fixture(
            &source.join("text_encoder/model.safetensors"),
            &[(
                "text_model.final_layer_norm.weight",
                "BF16",
                &[1],
                &[0x80, 0x3f],
            )],
        );

        let output = dir.join("safe.hfq");
        import_diffusers_to_hfq(DiffusersImportOptions {
            source,
            output: output.clone(),
            model_name: Some("safe-sd".into()),
            max_batch: 1,
            metadata_only: false,
        })
        .unwrap();

        let hfq = HfqFile::open_index_only(&output).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        assert_eq!(
            metadata.components["unet"].weight_entries,
            vec!["unet/tensors/conv_in.weight"]
        );
        assert_eq!(metadata.components["unet"].tensor_roles[0].dtype, "F32");
        assert_eq!(metadata.components["vae"].tensor_roles[0].dtype, "F16");
        assert_eq!(
            metadata.components["text_encoder"].tensor_roles[0].dtype,
            "BF16"
        );
        let unet = CpuTensor::from_hfq(&hfq, "unet/tensors/conv_in.weight").unwrap();
        let vae = CpuTensor::from_hfq(&hfq, "vae/tensors/post_quant_conv.weight").unwrap();
        let text = CpuTensor::from_hfq(
            &hfq,
            "text_encoder/tensors/text_model.final_layer_norm.weight",
        )
        .unwrap();
        assert_eq!(unet.shape, vec![1, 1]);
        assert_eq!(unet.data, vec![1.5]);
        assert_eq!(vae.data, vec![1.0]);
        assert_eq!(text.data, vec![1.0]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn importer_prefers_safetensors_over_legacy_bin_when_both_exist() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-safetensors-precedence-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let source = dir.join("snapshot");
        fs::create_dir_all(source.join("unet")).unwrap();
        fs::create_dir_all(source.join("scheduler")).unwrap();
        fs::write(
            source.join("model_index.json"),
            br#"{"_class_name":"StableDiffusionPipeline"}"#,
        )
        .unwrap();
        fs::write(
            source.join("unet/config.json"),
            br#"{"_class_name":"UNet2DConditionModel","sample_size":2,"in_channels":1}"#,
        )
        .unwrap();
        fs::write(
            source.join("scheduler/scheduler_config.json"),
            br#"{"_class_name":"EulerDiscreteScheduler"}"#,
        )
        .unwrap();
        fs::write(source.join("unet/diffusion_pytorch_model.bin"), b"opaque").unwrap();
        write_safetensors_fixture(
            &source.join("unet/diffusion_pytorch_model.safetensors"),
            &[("conv_in.bias", "F32", &[1], &[0x00, 0x00, 0x20, 0x40])],
        );

        let output = dir.join("precedence.hfq");
        import_diffusers_to_hfq(DiffusersImportOptions {
            source,
            output: output.clone(),
            model_name: None,
            max_batch: 1,
            metadata_only: false,
        })
        .unwrap();

        let hfq = HfqFile::open_index_only(&output).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        assert_eq!(
            metadata.components["unet"].weight_entries,
            vec!["unet/tensors/conv_in.bias"]
        );
        let tensor = CpuTensor::from_hfq(&hfq, "unet/tensors/conv_in.bias").unwrap();
        assert_eq!(tensor.data, vec![2.5]);
        assert!(hfq
            .tensor_data_vec("unet/diffusion_pytorch_model.bin")
            .is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn imports_diffusers_sharded_safetensors_index_as_hfq_tensor_entries() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-sharded-safetensors-import-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let source = dir.join("snapshot");
        fs::create_dir_all(source.join("unet")).unwrap();
        fs::create_dir_all(source.join("scheduler")).unwrap();
        fs::write(
            source.join("model_index.json"),
            br#"{"_class_name":"StableDiffusionPipeline"}"#,
        )
        .unwrap();
        fs::write(
            source.join("unet/config.json"),
            br#"{"_class_name":"UNet2DConditionModel","sample_size":2,"in_channels":1}"#,
        )
        .unwrap();
        fs::write(
            source.join("scheduler/scheduler_config.json"),
            br#"{"_class_name":"EulerDiscreteScheduler"}"#,
        )
        .unwrap();
        write_safetensors_fixture(
            &source.join("unet/diffusion_pytorch_model-00001-of-00002.safetensors"),
            &[("conv_in.weight", "F32", &[1], &[0x00, 0x00, 0xc0, 0x3f])],
        );
        write_safetensors_fixture(
            &source.join("unet/diffusion_pytorch_model-00002-of-00002.safetensors"),
            &[("conv_out.bias", "F32", &[1], &[0x00, 0x00, 0x20, 0x40])],
        );
        fs::write(
            source.join("unet/diffusion_pytorch_model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 8},
                "weight_map": {
                    "conv_in.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
                    "conv_out.bias": "diffusion_pytorch_model-00002-of-00002.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let output = dir.join("sharded.hfq");
        import_diffusers_to_hfq(DiffusersImportOptions {
            source,
            output: output.clone(),
            model_name: None,
            max_batch: 1,
            metadata_only: false,
        })
        .unwrap();

        let hfq = HfqFile::open_index_only(&output).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let entries = &metadata.components["unet"].weight_entries;
        assert_eq!(entries.len(), 2);
        assert!(entries.contains(&"unet/tensors/conv_in.weight".to_string()));
        assert!(entries.contains(&"unet/tensors/conv_out.bias".to_string()));
        let conv_in = CpuTensor::from_hfq(&hfq, "unet/tensors/conv_in.weight").unwrap();
        let conv_out = CpuTensor::from_hfq(&hfq, "unet/tensors/conv_out.bias").unwrap();
        assert_eq!(conv_in.data, vec![1.5]);
        assert_eq!(conv_out.data, vec![2.5]);
        assert!(hfq
            .tensor_data_vec("unet/diffusion_pytorch_model.safetensors.index.json")
            .is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn imports_single_file_safetensors_checkpoint_as_component_tensors() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-single-file-import-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::create_dir_all(dir.join("tokenizer")).unwrap();
        fs::write(
            dir.join("tokenizer/vocab.json"),
            br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "cat": 3
            }"#,
        )
        .unwrap();
        fs::write(dir.join("tokenizer/merges.txt"), b"#version: 0.2\n").unwrap();
        let source = dir.join("webui-model.safetensors");
        write_safetensors_fixture(
            &source,
            &[
                (
                    "model.diffusion_model.input_blocks.0.0.weight",
                    "F32",
                    &[1, 4, 1, 1],
                    &[
                        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40,
                        0x00, 0x00, 0x80, 0x40,
                    ],
                ),
                (
                    "first_stage_model.decoder.conv_in.weight",
                    "F16",
                    &[1],
                    &[0x00, 0x3c],
                ),
                (
                    "cond_stage_model.transformer.text_model.final_layer_norm.weight",
                    "BF16",
                    &[1],
                    &[0x80, 0x3f],
                ),
                (
                    "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
                    "F32",
                    &[1],
                    &[0x00, 0x00, 0x80, 0x3f],
                ),
                (
                    "model.diffusion_model.input_blocks.1.1.norm.weight",
                    "F32",
                    &[1],
                    &[0x00, 0x00, 0x00, 0x40],
                ),
                (
                    "model.diffusion_model.input_blocks.3.0.op.weight",
                    "F32",
                    &[1],
                    &[0x00, 0x00, 0x40, 0x40],
                ),
                (
                    "model.diffusion_model.middle_block.0.out_layers.3.bias",
                    "F32",
                    &[1],
                    &[0x00, 0x00, 0x80, 0x40],
                ),
                (
                    "model.diffusion_model.middle_block.1.proj_in.weight",
                    "F32",
                    &[1],
                    &[0x00, 0x00, 0xa0, 0x40],
                ),
                (
                    "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
                    "F32",
                    &[1],
                    &[0x00, 0x00, 0xc0, 0x40],
                ),
                (
                    "model.diffusion_model.output_blocks.2.2.conv.weight",
                    "F32",
                    &[1],
                    &[0x00, 0x00, 0xe0, 0x40],
                ),
            ],
        );

        let output = dir.join("webui-model.hfq");
        let summary = import_diffusers_to_hfq(DiffusersImportOptions {
            source,
            output: output.clone(),
            model_name: None,
            max_batch: 2,
            metadata_only: false,
        })
        .unwrap();

        assert_eq!(summary.model_name, "webui-model");
        assert_eq!(summary.pipeline_class, "StableDiffusionPipeline");
        assert_eq!(summary.max_batch, 2);
        let hfq = HfqFile::open_index_only(&output).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        assert_eq!(metadata.pipeline.latent_channels, Some(4));
        assert_eq!(config.pipeline_class, "StableDiffusionPipeline");
        assert_eq!(config.unet.sample_size, Some(64));
        assert_eq!(config.unet.block_out_channels, vec![320, 640, 1280, 1280]);
        assert_eq!(config.vae.scaling_factor, Some(0.18215));
        assert_eq!(config.scheduler.class_name, "EulerDiscreteScheduler");
        assert_eq!(
            metadata.components["unet"].config_entry.as_deref(),
            Some("unet/config.json")
        );
        assert_eq!(
            metadata.tokenizer.entries,
            vec!["tokenizer/vocab.json", "tokenizer/merges.txt"]
        );
        assert!(metadata.components.contains_key("unet"));
        assert!(metadata.components.contains_key("vae"));
        assert!(metadata.components.contains_key("text_encoder"));
        assert!(metadata.components["unet"].weight_entries.contains(
            &"unet/checkpoint_tensors/model.diffusion_model.input_blocks.0.0.weight".to_string()
        ));
        assert!(metadata.components["unet"]
            .weight_entries
            .contains(&"unet/tensors/conv_in.weight".to_string()));
        assert!(metadata.components["vae"]
            .weight_entries
            .contains(&"vae/tensors/decoder.conv_in.weight".to_string()));
        assert!(metadata.components["text_encoder"]
            .weight_entries
            .contains(&"text_encoder/tensors/text_model.final_layer_norm.weight".to_string()));
        for expected in [
            "unet/tensors/down_blocks.0.resnets.0.norm1.weight",
            "unet/tensors/down_blocks.0.attentions.0.norm.weight",
            "unet/tensors/down_blocks.0.downsamplers.0.conv.weight",
            "unet/tensors/mid_block.resnets.0.conv2.bias",
            "unet/tensors/mid_block.attentions.0.proj_in.weight",
            "unet/tensors/up_blocks.0.resnets.0.conv_shortcut.weight",
            "unet/tensors/up_blocks.0.upsamplers.0.conv.weight",
        ] {
            assert!(
                metadata.components["unet"]
                    .weight_entries
                    .contains(&expected.to_string()),
                "missing projected native entry {expected}"
            );
        }
        let checkpoint_tensor = CpuTensor::from_hfq(
            &hfq,
            "unet/checkpoint_tensors/model.diffusion_model.input_blocks.0.0.weight",
        )
        .unwrap();
        let native_tensor = CpuTensor::from_hfq(&hfq, "unet/tensors/conv_in.weight").unwrap();
        let tokenizer = ClipTokenizer::from_hfq_file(&hfq).unwrap();
        let tokens = tokenizer.encode_padded("a cat");
        let down_resnet =
            CpuTensor::from_hfq(&hfq, "unet/tensors/down_blocks.0.resnets.0.norm1.weight").unwrap();
        let upsample =
            CpuTensor::from_hfq(&hfq, "unet/tensors/up_blocks.0.upsamplers.0.conv.weight").unwrap();
        assert_eq!(checkpoint_tensor.shape, vec![1, 4, 1, 1]);
        assert_eq!(checkpoint_tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(native_tensor.shape, checkpoint_tensor.shape);
        assert_eq!(native_tensor.data, checkpoint_tensor.data);
        assert_eq!(&tokens[..4], &[0, 2, 3, 1]);
        assert_eq!(down_resnet.data, vec![1.0]);
        assert_eq!(upsample.data, vec![7.0]);
        let pipeline = DiffusionPipeline::open_hfq(&output).unwrap();
        assert!(pipeline.native_runtime.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn imports_single_file_sdxl_safetensors_checkpoint_metadata() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-single-file-sdxl-import-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::create_dir_all(dir.join("tokenizer_2")).unwrap();
        fs::write(
            dir.join("tokenizer_2/vocab.json"),
            br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "wide": 2
            }"#,
        )
        .unwrap();
        fs::write(dir.join("tokenizer_2/merges.txt"), b"#version: 0.2\n").unwrap();
        let source = dir.join("webui-sdxl.safetensors");
        write_safetensors_fixture(
            &source,
            &[
                (
                    "model.diffusion_model.input_blocks.0.0.weight",
                    "F32",
                    &[1, 4, 1, 1],
                    &[
                        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40,
                        0x00, 0x00, 0x80, 0x40,
                    ],
                ),
                (
                    "conditioner.embedders.1.model.text_model.final_layer_norm.weight",
                    "F32",
                    &[1],
                    &[0x00, 0x00, 0x80, 0x3f],
                ),
            ],
        );

        let output = dir.join("webui-sdxl.hfq");
        let summary = import_diffusers_to_hfq(DiffusersImportOptions {
            source,
            output: output.clone(),
            model_name: Some("webui-sdxl".into()),
            max_batch: 1,
            metadata_only: false,
        })
        .unwrap();

        assert_eq!(summary.pipeline_class, "StableDiffusionXLPipeline");
        let hfq = HfqFile::open_index_only(&output).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        assert_eq!(
            config.unet.addition_embed_type.as_deref(),
            Some("text_time")
        );
        assert_eq!(
            config.text_encoder_2.as_ref().unwrap().hidden_size,
            Some(1280)
        );
        assert!(metadata.components.contains_key("text_encoder_2"));
        assert_eq!(
            metadata.tokenizer_2.as_ref().unwrap().entries,
            vec!["tokenizer_2/vocab.json", "tokenizer_2/merges.txt"]
        );
        assert!(metadata.components["text_encoder_2"].weight_entries.contains(
            &"text_encoder_2/checkpoint_tensors/conditioner.embedders.1.model.text_model.final_layer_norm.weight".to_string()
        ));
        assert!(metadata.components["text_encoder_2"]
            .weight_entries
            .contains(&"text_encoder_2/tensors/text_model.final_layer_norm.weight".to_string()));
        let tokenizer_2 = ClipTokenizer::from_hfq_file_with_prefix(&hfq, "tokenizer_2").unwrap();
        assert_eq!(&tokenizer_2.encode_padded("wide")[..3], &[0, 2, 1]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ldm_unet_native_tensor_name_maps_standard_sd_blocks() {
        let cases = [
            (
                "input_blocks.0.0.weight",
                Some("conv_in.weight".to_string()),
            ),
            (
                "input_blocks.2.0.emb_layers.1.bias",
                Some("down_blocks.0.resnets.1.time_emb_proj.bias".to_string()),
            ),
            (
                "input_blocks.6.0.op.bias",
                Some("down_blocks.1.downsamplers.0.conv.bias".to_string()),
            ),
            (
                "middle_block.2.skip_connection.weight",
                Some("mid_block.resnets.1.conv_shortcut.weight".to_string()),
            ),
            (
                "output_blocks.4.0.out_layers.3.weight",
                Some("up_blocks.1.resnets.1.conv2.weight".to_string()),
            ),
            (
                "output_blocks.5.1.op.bias",
                Some("up_blocks.1.upsamplers.0.conv.bias".to_string()),
            ),
            ("input_blocks.3.1.norm.weight", None),
        ];

        for (input, expected) in cases {
            assert_eq!(ldm_unet_native_tensor_name(input), expected, "{input}");
        }
    }

    #[test]
    fn ldm_vae_native_tensor_name_maps_standard_sd_blocks() {
        let cases = [
            (
                "encoder.down.0.block.1.norm1.weight",
                Some("encoder.down_blocks.0.resnets.1.norm1.weight".to_string()),
            ),
            (
                "encoder.down.2.downsample.conv.bias",
                Some("encoder.down_blocks.2.downsamplers.0.conv.bias".to_string()),
            ),
            (
                "encoder.mid.attn_1.proj_out.weight",
                Some("encoder.mid_block.attentions.0.to_out.0.weight".to_string()),
            ),
            (
                "decoder.mid.block_2.nin_shortcut.bias",
                Some("decoder.mid_block.resnets.1.conv_shortcut.bias".to_string()),
            ),
            (
                "decoder.up.3.block.0.conv2.weight",
                Some("decoder.up_blocks.0.resnets.0.conv2.weight".to_string()),
            ),
            (
                "decoder.up.1.upsample.conv.weight",
                Some("decoder.up_blocks.2.upsamplers.0.conv.weight".to_string()),
            ),
            (
                "decoder.norm_out.bias",
                Some("decoder.conv_norm_out.bias".to_string()),
            ),
            ("decoder.up.4.block.0.norm1.weight", None),
        ];

        for (input, expected) in cases {
            assert_eq!(ldm_vae_native_tensor_name(input), expected, "{input}");
        }
    }

    #[test]
    fn single_file_checkpoint_projection_loads_tiny_native_unet() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-single-file-native-unet-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("tiny-ldm.safetensors");
        write_tiny_ldm_unet_safetensors(&source);

        let output = dir.join("tiny-ldm.hfq");
        import_diffusers_to_hfq(DiffusersImportOptions {
            source,
            output: output.clone(),
            model_name: Some("tiny-ldm".into()),
            max_batch: 1,
            metadata_only: false,
        })
        .unwrap();

        let hfq = HfqFile::open_index_only(&output).unwrap();
        let config = tiny_runtime_config();
        let unet = NativeUnet2DConditionModel::from_hfq(&hfq, &config.unet).unwrap();
        let encoder = NativeVaeEncoder::from_hfq(&hfq, &config.vae).unwrap();
        let decoder = NativeVaeDecoder::from_hfq(&hfq, &config.vae).unwrap();
        let sample = CpuTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![0.25, -0.25, 0.5, -0.5],
        };
        let encoder_states = CpuTensor {
            shape: vec![1, 4, 2],
            data: vec![0.0; 8],
        };

        let output = unet.forward(&sample, &[0.0], &encoder_states).unwrap();

        assert_eq!(output.shape, sample.shape);
        assert!(output.data.iter().all(|value| value.is_finite()));
        let latents = encoder
            .encode_to_latents(&RgbImageBatch {
                batch: 1,
                width: 2,
                height: 2,
                data: vec![255; 12],
            })
            .unwrap();
        assert_eq!(latents.batch, 1);
        assert_eq!(latents.channels, 1);
        assert_eq!(latents.height, 2);
        assert_eq!(latents.width, 2);
        assert!(latents.data.iter().all(|value| value.is_finite()));
        let rgb = decoder
            .decode_to_rgb8(&LatentBatch {
                batch: 1,
                channels: 1,
                height: 2,
                width: 2,
                data: output.data,
            })
            .unwrap();
        assert_eq!(rgb.batch, 1);
        assert_eq!(rgb.width, 2);
        assert_eq!(rgb.height, 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn single_file_checkpoint_projection_loads_tiny_text_conditioning() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-single-file-text-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("tokenizer")).unwrap();
        fs::write(
            dir.join("tokenizer/vocab.json"),
            br#"{"<|startoftext|>":0,"<|endoftext|>":1,"cat":2}"#,
        )
        .unwrap();
        fs::write(dir.join("tokenizer/merges.txt"), b"#version: 0.2\n").unwrap();
        let source = dir.join("tiny-ldm.safetensors");
        write_tiny_ldm_unet_safetensors(&source);

        let output = dir.join("tiny-ldm.hfq");
        import_diffusers_to_hfq(DiffusersImportOptions {
            source,
            output: output.clone(),
            model_name: Some("tiny-ldm".into()),
            max_batch: 1,
            metadata_only: false,
        })
        .unwrap();

        let hfq = HfqFile::open_index_only(&output).unwrap();
        let tokenizer = ClipTokenizer::from_hfq_file(&hfq).unwrap();
        let tokens = tokenizer.encode_padded("cat");
        assert_eq!(&tokens[..3], &[0, 2, 1]);
        let text_encoder = ClipTextEncoder::from_hfq_file_with_heads(&hfq, 1).unwrap();
        let hidden_states = text_encoder.encode_tokens(&tokens).unwrap();
        assert_eq!(hidden_states.shape, vec![77, 2]);
        assert!(hidden_states.data.iter().all(|value| value.is_finite()));
        let (hidden_states, pooled) = text_encoder
            .encode_tokens_with_pooled(&tokens, tokenizer.end_token_id())
            .unwrap();
        assert_eq!(hidden_states.shape, vec![77, 2]);
        let pooled = pooled.unwrap();
        assert_eq!(pooled.len(), 2);
        assert!(pooled.iter().all(|value| value.is_finite()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_tiny_sd_pytorch_tensor_indexes_when_cache_is_present() {
        let root = Path::new("/srv/huggingface/models--segmind--tiny-sd/snapshots/cad0bd7495fa6c4bcca01b19a723dc91627fe84f");
        if !root.exists() {
            eprintln!("skip: tiny-sd cache not present");
            return;
        }

        let unet =
            parse_pytorch_state_dict(&root.join("unet/diffusion_pytorch_model.bin")).unwrap();
        let vae = parse_pytorch_state_dict(&root.join("vae/diffusion_pytorch_model.bin")).unwrap();
        let text = parse_pytorch_state_dict(&root.join("text_encoder/pytorch_model.bin")).unwrap();

        assert!(unet
            .iter()
            .any(|tensor| tensor.name == "conv_in.weight" && tensor.shape == [320, 4, 3, 3]));
        assert!(vae
            .iter()
            .any(|tensor| tensor.name == "decoder.conv_out.weight"));
        assert!(text
            .iter()
            .any(|tensor| tensor.name == "text_model.embeddings.token_embedding.weight"));
    }

    #[test]
    fn clip_tokenizer_pads_and_keeps_special_tokens() {
        let vocab = br#"{
            "<|startoftext|>": 49406,
            "<|endoftext|>": 49407,
            "a</w>": 10,
            "cat</w>": 11
        }"#;
        let merges = b"#version: 0.2\nc a\nca t</w>\n";
        let tokenizer = ClipTokenizer::from_bytes(vocab, merges, 6).unwrap();
        let encoded = tokenizer.encode_padded("a cat");

        assert_eq!(encoded[0], 49406);
        assert_eq!(encoded[1], 10);
        assert_eq!(encoded[2], 11);
        assert_eq!(encoded[3], 49407);
        assert_eq!(encoded[4], 49407);
        assert_eq!(encoded[5], 49407);
    }

    #[test]
    fn tiny_sd_clip_tokenizer_files_encode_prompt_when_cache_is_present() {
        let root = Path::new("/srv/huggingface/models--segmind--tiny-sd/snapshots/cad0bd7495fa6c4bcca01b19a723dc91627fe84f/tokenizer");
        if !root.exists() {
            eprintln!("skip: tiny-sd tokenizer cache not present");
            return;
        }
        let tokenizer = ClipTokenizer::from_bytes(
            &fs::read(root.join("vocab.json")).unwrap(),
            &fs::read(root.join("merges.txt")).unwrap(),
            77,
        )
        .unwrap();
        let encoded = tokenizer.encode_padded("a red robot");

        assert_eq!(encoded.len(), 77);
        assert_eq!(encoded[0], 49406);
        assert!(encoded[1..10].iter().any(|&token| token != 49407));
        assert!(encoded.contains(&49407));
    }

    #[test]
    fn cpu_tensor_loads_supported_source_and_packed_formats_from_hfq() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-tensor-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tensors.hfq");
        let metadata = minimal_metadata();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &[
                HfqMemTensor {
                    name: "unet/config.json".into(),
                    quant_type: QT_DIFFUSION_JSON,
                    shape: vec![2],
                    group_size: 0,
                    data: b"{}".to_vec(),
                },
                HfqMemTensor {
                    name: "f16".into(),
                    quant_type: QT_DIFFUSION_TENSOR_F16,
                    shape: vec![2],
                    group_size: 0,
                    data: [
                        f32_to_f16_bits(1.5).to_le_bytes(),
                        f32_to_f16_bits(-2.0).to_le_bytes(),
                    ]
                    .concat(),
                },
                HfqMemTensor {
                    name: "bf16".into(),
                    quant_type: QT_DIFFUSION_TENSOR_BF16,
                    shape: vec![1],
                    group_size: 0,
                    data: (((3.0f32).to_bits() >> 16) as u16).to_le_bytes().to_vec(),
                },
                HfqMemTensor {
                    name: "f32".into(),
                    quant_type: QT_DIFFUSION_TENSOR_F32,
                    shape: vec![1],
                    group_size: 0,
                    data: 4.25f32.to_le_bytes().to_vec(),
                },
                HfqMemTensor {
                    name: "q8".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q8F16,
                    shape: vec![3],
                    group_size: 32,
                    data: [
                        f32_to_f16_bits(0.5).to_le_bytes().as_slice(),
                        &[2u8, (-4i8) as u8, 7u8],
                        &[0u8; 29],
                    ]
                    .concat(),
                },
                HfqMemTensor {
                    name: "q4".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q4F16_G64,
                    shape: vec![4],
                    group_size: 64,
                    data: [
                        f32_to_f16_bits(0.25).to_le_bytes().as_slice(),
                        f32_to_f16_bits(-1.0).to_le_bytes().as_slice(),
                        &[0x00u8, 0x08u8, 0x04u8, 0x0bu8],
                        &[0u8; 28],
                    ]
                    .concat(),
                },
                HfqMemTensor {
                    name: "q4k".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q4_K,
                    shape: vec![4],
                    group_size: 256,
                    data: q4k_test_block(&[4, 8, 0, 7]),
                },
                hfq4_mem_tensor(
                    "hfq4g128",
                    QT_DIFFUSION_TENSOR_HFQ4_G128,
                    &[4],
                    128,
                    &[0, 8, 4, 11],
                ),
                hfq4_mem_tensor(
                    "hfq4g256",
                    QT_DIFFUSION_TENSOR_HFQ4_G256,
                    &[4],
                    256,
                    &[0, 8, 4, 11],
                ),
                hfq6_mem_tensor("hfq6g256", &[4], &[0, 8, 4, 11]),
            ],
        )
        .unwrap();

        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "f16").unwrap().data,
            vec![1.5, -2.0]
        );
        assert_eq!(CpuTensor::from_hfq(&hfq, "bf16").unwrap().data, vec![3.0]);
        assert_eq!(CpuTensor::from_hfq(&hfq, "f32").unwrap().data, vec![4.25]);
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "q8").unwrap().data,
            vec![1.0, -2.0, 3.5]
        );
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "q4").unwrap().data,
            vec![-1.0, 1.0, 0.0, 1.75]
        );
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "q4k").unwrap().data,
            vec![1.0, 2.0, 0.0, 1.75]
        );
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "hfq4g128").unwrap().data,
            vec![-1.0, 1.0, 0.0, 1.75]
        );
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "hfq4g256").unwrap().data,
            vec![-1.0, 1.0, 0.0, 1.75]
        );
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "hfq6g256").unwrap().data,
            vec![-1.0, 1.0, 0.0, 1.75]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cpu_tensor_rejects_truncated_packed_payloads() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-truncated-packed-tensor-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("truncated-tensors.hfq");
        let metadata = minimal_metadata();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &[
                bytes_mem_tensor("unet/config.json", QT_DIFFUSION_JSON, b"{}"),
                HfqMemTensor {
                    name: "bad_q4".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q4F16_G64,
                    shape: vec![64],
                    group_size: 64,
                    data: vec![0u8; 35],
                },
                HfqMemTensor {
                    name: "bad_q8".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q8F16,
                    shape: vec![32],
                    group_size: 32,
                    data: vec![0u8; 33],
                },
                HfqMemTensor {
                    name: "bad_q4k".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q4_K,
                    shape: vec![256],
                    group_size: 256,
                    data: vec![0u8; 143],
                },
                HfqMemTensor {
                    name: "bad_hfq4g128".into(),
                    quant_type: QT_DIFFUSION_TENSOR_HFQ4_G128,
                    shape: vec![128],
                    group_size: 128,
                    data: vec![0u8; 71],
                },
                HfqMemTensor {
                    name: "bad_hfq4g256".into(),
                    quant_type: QT_DIFFUSION_TENSOR_HFQ4_G256,
                    shape: vec![256],
                    group_size: 256,
                    data: vec![0u8; 135],
                },
                HfqMemTensor {
                    name: "bad_hfq6g256".into(),
                    quant_type: QT_DIFFUSION_TENSOR_HFQ6_G256,
                    shape: vec![256],
                    group_size: 256,
                    data: vec![0u8; 199],
                },
            ],
        )
        .unwrap();

        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let q4_error = CpuTensor::from_hfq(&hfq, "bad_q4").unwrap_err();
        assert!(q4_error.to_string().contains("Q4F16_G64"));
        assert!(q4_error.to_string().contains("requires at least 36"));
        let q8_error = CpuTensor::from_hfq(&hfq, "bad_q8").unwrap_err();
        assert!(q8_error.to_string().contains("Q8F16"));
        assert!(q8_error.to_string().contains("requires at least 34"));
        let q4k_error = CpuTensor::from_hfq(&hfq, "bad_q4k").unwrap_err();
        assert!(q4k_error.to_string().contains("Q4_K"));
        assert!(q4k_error.to_string().contains("requires at least 144"));
        let hfq4g128_error = CpuTensor::from_hfq(&hfq, "bad_hfq4g128").unwrap_err();
        assert!(hfq4g128_error.to_string().contains("HFQ4G128"));
        assert!(hfq4g128_error.to_string().contains("requires at least 72"));
        let hfq4g256_error = CpuTensor::from_hfq(&hfq, "bad_hfq4g256").unwrap_err();
        assert!(hfq4g256_error.to_string().contains("HFQ4G256"));
        assert!(hfq4g256_error.to_string().contains("requires at least 136"));
        let hfq6g256_error = CpuTensor::from_hfq(&hfq, "bad_hfq6g256").unwrap_err();
        assert!(hfq6g256_error.to_string().contains("HFQ6G256"));
        assert!(hfq6g256_error.to_string().contains("requires at least 200"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cpu_linear_layer_norm_and_softmax_are_stable() {
        let input = CpuTensor {
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let weight = CpuTensor {
            shape: vec![2, 2],
            data: vec![1.0, 0.0, 0.0, 1.0],
        };
        let bias = CpuTensor {
            shape: vec![2],
            data: vec![0.5, -0.5],
        };
        let out = linear(&input, &weight, &bias).unwrap();
        assert_eq!(out.data, vec![1.5, 1.5, 3.5, 3.5]);

        let norm_weight = CpuTensor {
            shape: vec![2],
            data: vec![1.0, 1.0],
        };
        let norm_bias = CpuTensor {
            shape: vec![2],
            data: vec![0.0, 0.0],
        };
        let normed = layer_norm(&input, &norm_weight, &norm_bias, 1e-5).unwrap();
        assert!(normed.data[0] < -0.99 && normed.data[1] > 0.99);

        let mut logits = vec![1.0, 2.0, 3.0];
        softmax_in_place(&mut logits);
        let sum = logits.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(logits[2] > logits[1] && logits[1] > logits[0]);

        assert_eq!(quick_gelu(0.0), 0.0);
        assert!((quick_gelu(1.0) - 0.845795).abs() < 1e-5);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for QuickGELU routing test: {error}");
            } else {
                let cpu = tensor_map(&input, quick_gelu);
                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip = quick_gelu_with_runtime_context(&input, &mut runtime_context).unwrap();
                assert_eq!(hip.shape, cpu.shape);
                assert!(f32_slices_close(&hip.data, &cpu.data, 1e-6));
            }
        }
    }

    #[test]
    fn seeded_latents_are_deterministic_and_batched() {
        let a = LatentBatch::seeded_normal(2, 4, 2, 2, &[123, 456]);
        let b = LatentBatch::seeded_normal(2, 4, 2, 2, &[123, 456]);
        let c = LatentBatch::seeded_normal(2, 4, 2, 2, &[123, 789]);

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.batch, 2);
        assert_eq!(a.len_per_batch(), 16);
        assert!(a.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn seed_resize_generates_source_latents_and_resizes_to_target_shape() {
        let config = StableDiffusionConfig {
            pipeline_class: "StableDiffusionPipeline".into(),
            text_encoder: TextEncoderConfig::default(),
            text_encoder_2: None,
            unet: UnetConfig::default(),
            vae: VaeConfig::default(),
            scheduler: SchedulerConfig::default(),
            latent_channels: 1,
            latent_height: None,
            latent_width: None,
            vae_scale_factor: 1,
        };
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a".into(),
                negative_prompt: String::new(),
                seed: 123,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: Some(1),
            seed_resize_from_height: Some(1),
            crop_x: 0,
            crop_y: 0,
            steps: 2,
            cfg_scale: 7.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };
        let latent_shape = latent_shape_for_request(&config, &request).unwrap();

        let resized = seeded_latents_for_request(&config, &request, &latent_shape, &[123]).unwrap();
        let source = LatentBatch::seeded_normal(1, 1, 1, 1, &[123]);
        let direct = LatentBatch::seeded_normal(1, 1, 2, 2, &[123]);

        assert_eq!(resized, resize_latent_batch_nearest(&source, 2, 2).unwrap());
        assert_ne!(resized, direct);
    }

    #[test]
    fn subseed_strength_blends_only_prompt_latents_with_subseeds() {
        let mut latents = LatentBatch::seeded_normal(2, 1, 1, 2, &[10, 20]);
        let original = latents.clone();
        let subseed = LatentBatch::seeded_normal(2, 1, 1, 2, &[30, 20]);
        let config = StableDiffusionConfig {
            pipeline_class: "StableDiffusionPipeline".into(),
            text_encoder: TextEncoderConfig::default(),
            text_encoder_2: None,
            unet: UnetConfig::default(),
            vae: VaeConfig::default(),
            scheduler: SchedulerConfig::default(),
            latent_channels: 1,
            latent_height: None,
            latent_width: None,
            vae_scale_factor: 1,
        };
        let request = DiffusionBatchRequest {
            prompts: vec![
                DiffusionPrompt {
                    prompt: "a".into(),
                    negative_prompt: String::new(),
                    seed: 10,
                    subseed: Some(30),
                },
                DiffusionPrompt {
                    prompt: "b".into(),
                    negative_prompt: String::new(),
                    seed: 20,
                    subseed: None,
                },
            ],
            width: 2,
            height: 1,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 2,
            cfg_scale: 7.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.25,
            send_images: true,
            save_images: false,
        };
        let latent_shape = latent_shape_for_request(&config, &request).unwrap();

        blend_subseed_latents(&config, &mut latents, &request, &latent_shape).unwrap();

        assert_eq!(latents.batch, 2);
        for idx in 0..2 {
            let expected = original.data[idx] * 0.75 + subseed.data[idx] * 0.25;
            assert!((latents.data[idx] - expected).abs() < 1e-6);
        }
        assert_eq!(&latents.data[2..], &original.data[2..]);
    }

    #[test]
    fn linear_scheduler_euler_step_moves_toward_next_sigma() {
        let schedule = DiffusionSchedule::linear(2).unwrap();
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            data: vec![1.0, -1.0],
        };

        schedule.euler_step(&mut latents, &[0.25, -0.5], 0).unwrap();

        assert_eq!(schedule.timesteps, vec![1.0, 0.0]);
        assert_eq!(schedule.sigmas, vec![1.0, 0.0, 0.0]);
        assert_eq!(latents.data, vec![0.75, -0.5]);
    }

    #[test]
    fn scheduler_config_uses_diffusers_beta_sigmas_and_train_timesteps() {
        let config = SchedulerConfig {
            class_name: "EulerDiscreteScheduler".into(),
            beta_start: Some(0.0001),
            beta_end: Some(0.02),
            beta_schedule: Some("linear".into()),
            num_train_timesteps: Some(10),
            prediction_type: Some("epsilon".into()),
            ..SchedulerConfig::default()
        };

        let schedule = DiffusionSchedule::from_config(&config, 3).unwrap();

        assert_eq!(schedule.timesteps, vec![9.0, 5.0, 0.0]);
        assert_eq!(schedule.sigmas.len(), 4);
        assert!(schedule.sigmas[0] > schedule.sigmas[1]);
        assert!(schedule.sigmas[1] > schedule.sigmas[2]);
        assert_eq!(schedule.sigmas[3], 0.0);
    }

    #[test]
    fn dpm_solver_config_uses_diffusers_linspace_timesteps() {
        let config = SchedulerConfig {
            class_name: "DPMSolverMultistepScheduler".into(),
            beta_start: Some(0.00085),
            beta_end: Some(0.012),
            beta_schedule: Some("scaled_linear".into()),
            num_train_timesteps: Some(1000),
            prediction_type: Some("epsilon".into()),
            algorithm_type: Some("dpmsolver++".into()),
            solver_order: Some(2),
            solver_type: Some("midpoint".into()),
            lower_order_final: Some(true),
            thresholding: Some(false),
            timestep_spacing: Some("linspace".into()),
            steps_offset: Some(1),
            use_karras_sigmas: Some(false),
            set_alpha_to_one: None,
            ..SchedulerConfig::default()
        };

        let schedule = DiffusionSchedule::from_config(&config, 3).unwrap();

        assert_eq!(schedule.train_timesteps, vec![999, 666, 333]);
        assert_eq!(schedule.timesteps, vec![999.0, 666.0, 333.0]);
        assert_eq!(
            schedule.solver,
            SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 2,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: true,
                thresholding: false,
                dynamic_thresholding_ratio: 0.995,
                sample_max_value: 1.0,
            }
        );
        assert_eq!(schedule.input_scaling, SchedulerInputScaling::None);
        assert_eq!(schedule.initial_noise_sigma(), 1.0);
    }

    #[test]
    fn dpm_solver_config_preserves_dynamic_thresholding_settings() {
        let config = SchedulerConfig {
            class_name: "DPMSolverMultistepScheduler".into(),
            beta_start: Some(0.00085),
            beta_end: Some(0.012),
            beta_schedule: Some("scaled_linear".into()),
            num_train_timesteps: Some(1000),
            prediction_type: Some("epsilon".into()),
            algorithm_type: Some("dpmsolver++".into()),
            solver_order: Some(2),
            solver_type: Some("midpoint".into()),
            thresholding: Some(true),
            dynamic_thresholding_ratio: Some(0.9),
            sample_max_value: Some(2.0),
            ..SchedulerConfig::default()
        };

        let schedule = DiffusionSchedule::from_config(&config, 2).unwrap();

        assert_eq!(
            schedule.solver,
            SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 2,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: true,
                thresholding: true,
                dynamic_thresholding_ratio: 0.9,
                sample_max_value: 2.0,
            }
        );
    }

    #[test]
    fn flow_match_euler_scheduler_uses_shifted_sigmas_and_terminal_rescale() {
        let config = SchedulerConfig {
            class_name: "FlowMatchEulerDiscreteScheduler".into(),
            num_train_timesteps: Some(1000),
            shift: Some(1.0),
            shift_terminal: Some(0.02),
            invert_sigmas: Some(false),
            ..SchedulerConfig::default()
        };

        let schedule = DiffusionSchedule::from_config(&config, 3).unwrap();

        assert_eq!(schedule.solver, SchedulerSolver::FlowMatchEuler);
        assert_eq!(schedule.input_scaling, SchedulerInputScaling::None);
        assert_eq!(schedule.sigmas, vec![1.0, 0.51, 0.02, 0.0]);
        assert_eq!(schedule.timesteps, vec![1000.0, 510.0, 20.0]);
        assert_eq!(schedule.initial_noise_sigma(), 1.0);
    }

    #[test]
    fn flow_match_euler_step_uses_model_output_as_velocity() {
        let config = SchedulerConfig {
            class_name: "FlowMatchEulerDiscreteScheduler".into(),
            num_train_timesteps: Some(1000),
            shift: Some(1.0),
            ..SchedulerConfig::default()
        };
        let schedule = DiffusionSchedule::from_config(&config, 2).unwrap();
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            data: vec![1.0, -1.0],
        };
        let mut state = SchedulerStepState::default();

        schedule
            .step(&mut latents, &[0.25, -0.5], 0, &mut state)
            .unwrap();

        assert_eq!(schedule.sigmas, vec![1.0, 0.0, 0.0]);
        assert_eq!(latents.data, vec![0.75, -0.5]);
    }

    #[test]
    fn karras_scheduler_uses_power_law_sigmas_and_nearest_train_timesteps() {
        let mut config = SchedulerConfig {
            class_name: "DPMSolverMultistepScheduler".into(),
            beta_start: Some(0.00085),
            beta_end: Some(0.012),
            beta_schedule: Some("scaled_linear".into()),
            num_train_timesteps: Some(1000),
            prediction_type: Some("epsilon".into()),
            algorithm_type: Some("dpmsolver++".into()),
            solver_order: Some(2),
            solver_type: Some("midpoint".into()),
            lower_order_final: Some(true),
            thresholding: Some(false),
            timestep_spacing: Some("linspace".into()),
            steps_offset: Some(1),
            use_karras_sigmas: Some(false),
            set_alpha_to_one: None,
            ..SchedulerConfig::default()
        };
        let normal = DiffusionSchedule::from_config(&config, 4).unwrap();
        config.use_karras_sigmas = Some(true);

        let karras = DiffusionSchedule::from_config(&config, 4).unwrap();

        assert_eq!(karras.sigmas.len(), 5);
        assert!((karras.sigmas[0] - normal.sigmas[0]).abs() / normal.sigmas[0].max(1.0) < 1e-4);
        assert_eq!(karras.sigmas[4], 0.0);
        assert!(karras.sigmas[0] > karras.sigmas[1]);
        assert!(karras.sigmas[1] > karras.sigmas[2]);
        assert!(karras.sigmas[2] > karras.sigmas[3]);
        assert_ne!(karras.sigmas, normal.sigmas);
        assert_eq!(karras.train_timesteps.len(), 4);
        assert!(karras
            .train_timesteps
            .windows(2)
            .all(|pair| pair[0] >= pair[1]));
    }

    #[test]
    fn scheduler_request_aliases_select_actual_sampler_config() {
        let config = tiny_sd_scheduler_config_for_tests();

        let dpm = config.resolve_request_scheduler("DPM++ 2M").unwrap();
        let dpm_karras = config.resolve_request_scheduler("DPM++ 2M Karras").unwrap();
        let dpm3 = config.resolve_request_scheduler("DPM++ 3M").unwrap();
        let dpm3_karras = config.resolve_request_scheduler("DPM++ 3M Karras").unwrap();
        let euler = config.resolve_request_scheduler("Euler").unwrap();
        let euler_karras = config.resolve_request_scheduler("Euler Karras").unwrap();
        let euler_a = config.resolve_request_scheduler("Euler a").unwrap();
        let ddim = config.resolve_request_scheduler("DDIM").unwrap();

        assert_eq!(dpm.class_name, "DPMSolverMultistepScheduler");
        assert_eq!(dpm_karras.class_name, "DPMSolverMultistepScheduler");
        assert_eq!(dpm_karras.use_karras_sigmas, Some(true));
        assert_eq!(dpm3.class_name, "DPMSolverMultistepScheduler");
        assert_eq!(dpm3.algorithm_type.as_deref(), Some("dpmsolver++"));
        assert_eq!(dpm3.solver_order, Some(3));
        assert_eq!(dpm3_karras.solver_order, Some(3));
        assert_eq!(dpm3_karras.use_karras_sigmas, Some(true));
        assert_eq!(euler.class_name, "EulerDiscreteScheduler");
        assert_eq!(euler.algorithm_type, None);
        assert_eq!(euler_karras.class_name, "EulerDiscreteScheduler");
        assert_eq!(euler_karras.use_karras_sigmas, Some(true));
        assert_eq!(euler_a.class_name, "EulerAncestralDiscreteScheduler");
        assert_eq!(ddim.class_name, "DDIMScheduler");
        assert!(config.resolve_request_scheduler("not a sampler").is_err());
    }

    #[test]
    fn scheduler_request_alias_changes_run_plan_schedule() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-scheduler-alias-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-scheduler-alias.hfq");
        let metadata = tiny_runtime_metadata();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tiny_complete_runtime_tensors(),
        )
        .unwrap();
        let mut pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        pipeline.config.scheduler = tiny_sd_scheduler_config_for_tests();
        let mut request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a".into(),
                negative_prompt: String::new(),
                seed: 1,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 2,
            cfg_scale: 1.0,
            scheduler: "DPM++ 2M".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let dpm_plan = pipeline.prepare_run_plan(&request).unwrap();
        request.scheduler = "Euler".into();
        let euler_plan = pipeline.prepare_run_plan(&request).unwrap();
        request.scheduler = "DDIM".into();
        let ddim_plan = pipeline.prepare_run_plan(&request).unwrap();

        assert!(matches!(
            dpm_plan.schedule.solver,
            SchedulerSolver::DpmSolverMultistep { .. }
        ));
        assert_eq!(euler_plan.schedule.solver, SchedulerSolver::Euler);
        assert_eq!(
            euler_plan.schedule.input_scaling,
            SchedulerInputScaling::Sigma
        );
        assert_eq!(
            ddim_plan.schedule.solver,
            SchedulerSolver::Ddim {
                set_alpha_to_one: true
            }
        );
        assert_eq!(
            ddim_plan.schedule.input_scaling,
            SchedulerInputScaling::None
        );
        assert_ne!(dpm_plan.latents.data, euler_plan.latents.data);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ddim_scheduler_step_matches_deterministic_epsilon_update() {
        let schedule = DiffusionSchedule {
            timesteps: vec![2.0, 1.0],
            sigmas: vec![0.8, 0.6, 0.0],
            prediction_type: SchedulerPredictionType::Epsilon,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::Ddim {
                set_alpha_to_one: true,
            },
            train_timesteps: vec![2, 1],
            alpha_t: vec![1.0, 0.8, 0.6],
            sigma_t: vec![0.0, 0.6, 0.8],
            lambda_t: Vec::new(),
        };
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![1.4],
        };
        let mut state = SchedulerStepState::default();

        schedule.step(&mut latents, &[0.5], 0, &mut state).unwrap();

        let pred_original = (1.4 - 0.8 * 0.5) / 0.6;
        let expected = 0.8 * pred_original + 0.6 * 0.5;
        assert!((latents.data[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn dpm_solver_multistep_updates_with_model_output_history() {
        let lambda = |alpha: f32, sigma: f32| alpha.ln() - sigma.ln();
        let schedule = DiffusionSchedule {
            timesteps: vec![2.0, 1.0],
            sigmas: vec![0.3, 0.2, 0.0],
            prediction_type: SchedulerPredictionType::Epsilon,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 2,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: false,
                thresholding: false,
                dynamic_thresholding_ratio: 0.995,
                sample_max_value: 1.0,
            },
            train_timesteps: vec![2, 1],
            alpha_t: vec![0.9, 0.8, 0.7],
            sigma_t: vec![0.1, 0.2, 0.3],
            lambda_t: vec![lambda(0.9, 0.1), lambda(0.8, 0.2), lambda(0.7, 0.3)],
        };
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![1.0],
        };
        let mut state = SchedulerStepState::default();

        schedule.step(&mut latents, &[0.5], 0, &mut state).unwrap();
        let first = latents.data[0];
        schedule.step(&mut latents, &[0.25], 1, &mut state).unwrap();

        assert_eq!(state.lower_order_nums, 2);
        assert_eq!(state.model_outputs.len(), 2);
        assert!(first.is_finite());
        assert!(latents.data[0].is_finite());
        assert_ne!(latents.data[0], first);
    }

    #[test]
    fn dpm_solver_dynamic_thresholding_clips_predicted_original_sample() {
        let schedule = DiffusionSchedule {
            timesteps: vec![0.0],
            sigmas: vec![0.0, 0.0],
            prediction_type: SchedulerPredictionType::Sample,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 2,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: true,
                thresholding: true,
                dynamic_thresholding_ratio: 1.0,
                sample_max_value: 4.0,
            },
            train_timesteps: vec![0],
            alpha_t: vec![1.0],
            sigma_t: vec![0.0],
            lambda_t: vec![0.0],
        };
        let sample = CpuTensor {
            shape: vec![2, 1, 1, 4],
            data: vec![0.0; 8],
        };
        let model_output = [-0.5, 0.5, 2.0, -4.0, 0.2, -3.0, 6.0, -9.0];

        let output = schedule
            .dpm_convert_model_output(&model_output, 0, &sample)
            .unwrap();

        assert_eq!(
            output,
            vec![-0.125, 0.125, 0.5, -1.0, 0.05, -0.75, 1.0, -1.0]
        );
    }

    #[test]
    fn dpm_solver_third_order_update_matches_diffusers_formula() {
        let lambda = |alpha: f32, sigma: f32| alpha.ln() - sigma.ln();
        let schedule = DiffusionSchedule {
            timesteps: vec![3.0, 2.0, 1.0],
            sigmas: vec![0.4, 0.3, 0.2, 0.0],
            prediction_type: SchedulerPredictionType::Sample,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 3,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: false,
                thresholding: false,
                dynamic_thresholding_ratio: 0.995,
                sample_max_value: 1.0,
            },
            train_timesteps: vec![3, 2, 1],
            alpha_t: vec![0.95, 0.85, 0.75, 0.65],
            sigma_t: vec![0.10, 0.20, 0.30, 0.40],
            lambda_t: vec![
                lambda(0.95, 0.10),
                lambda(0.85, 0.20),
                lambda(0.75, 0.30),
                lambda(0.65, 0.40),
            ],
        };
        let sample = CpuTensor {
            shape: vec![1, 1, 1, 1],
            data: vec![1.25],
        };
        let state = SchedulerStepState {
            model_outputs: vec![vec![0.20], vec![0.40], vec![0.70]],
            lower_order_nums: 2,
        };

        let next = schedule
            .dpm_third_order_update(3, 2, 1, 0, &sample, &state)
            .unwrap();

        let lambda_t = schedule.scheduler_lambda(0).unwrap();
        let lambda_s0 = schedule.scheduler_lambda(1).unwrap();
        let lambda_s1 = schedule.scheduler_lambda(2).unwrap();
        let lambda_s2 = schedule.scheduler_lambda(3).unwrap();
        let h = lambda_t - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        let h1 = lambda_s1 - lambda_s2;
        let r0 = h0 / h;
        let r1 = h1 / h;
        let m0 = 0.70;
        let m1 = 0.40;
        let m2 = 0.20;
        let d1_0 = (m0 - m1) / r0;
        let d1_1 = (m1 - m2) / r1;
        let d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1);
        let d2 = (d1_0 - d1_1) / (r0 + r1);
        let exp_neg_h = (-h).exp();
        let expected = (schedule.scheduler_sigma(0).unwrap()
            / schedule.scheduler_sigma(1).unwrap())
            * sample.data[0]
            - (schedule.scheduler_alpha(0).unwrap() * (exp_neg_h - 1.0)) * m0
            + (schedule.scheduler_alpha(0).unwrap() * ((exp_neg_h - 1.0) / h + 1.0)) * d1
            - (schedule.scheduler_alpha(0).unwrap() * ((exp_neg_h - 1.0 + h) / (h * h) - 0.5)) * d2;

        assert!((next.data[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn dpm_solver_order_three_step_uses_third_order_history() {
        let lambda = |alpha: f32, sigma: f32| alpha.ln() - sigma.ln();
        let schedule = DiffusionSchedule {
            timesteps: vec![3.0, 2.0, 1.0],
            sigmas: vec![0.4, 0.3, 0.2, 0.0],
            prediction_type: SchedulerPredictionType::Sample,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 3,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: false,
                thresholding: false,
                dynamic_thresholding_ratio: 0.995,
                sample_max_value: 1.0,
            },
            train_timesteps: vec![3, 2, 1],
            alpha_t: vec![0.95, 0.85, 0.75, 0.65],
            sigma_t: vec![0.10, 0.20, 0.30, 0.40],
            lambda_t: vec![
                lambda(0.95, 0.10),
                lambda(0.85, 0.20),
                lambda(0.75, 0.30),
                lambda(0.65, 0.40),
            ],
        };
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![1.0],
        };
        let mut state = SchedulerStepState::default();

        schedule.step(&mut latents, &[0.20], 0, &mut state).unwrap();
        schedule.step(&mut latents, &[0.40], 1, &mut state).unwrap();
        let second = latents.data[0];
        schedule.step(&mut latents, &[0.70], 2, &mut state).unwrap();

        assert_eq!(state.lower_order_nums, 3);
        assert_eq!(state.model_outputs.len(), 3);
        assert!(latents.data[0].is_finite());
        assert_ne!(latents.data[0], second);
    }

    #[test]
    fn scheduler_config_falls_back_to_linear_when_beta_metadata_is_missing() {
        let schedule = DiffusionSchedule::from_config(&SchedulerConfig::default(), 2).unwrap();

        assert_eq!(schedule.timesteps, vec![1.0, 0.0]);
        assert_eq!(schedule.sigmas, vec![1.0, 0.0, 0.0]);
        assert_eq!(schedule.prediction_type, SchedulerPredictionType::Epsilon);
        assert_eq!(schedule.input_scaling, SchedulerInputScaling::None);
    }

    #[test]
    fn denoise_progress_callback_can_interrupt_generation() {
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![0.0],
        };
        let schedule = DiffusionSchedule::from_config(&SchedulerConfig::default(), 2).unwrap();
        let positive = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.0],
        };
        let negative = positive.clone();
        let mut events = Vec::new();

        let error = denoise_latents_with_cfg_progress(
            latents,
            &schedule,
            1.0,
            &positive,
            &negative,
            |sample, _timesteps, _encoder_states, _sdxl_conditioning| {
                Ok(CpuTensor {
                    shape: sample.shape.clone(),
                    data: vec![0.0; sample.data.len()],
                })
            },
            None,
            None,
            None,
            None,
            Some(&mut |progress| {
                events.push(progress);
                Err(DiffusionError::Interrupted("test interrupt".to_string()))
            }),
        )
        .unwrap_err();

        assert!(matches!(error, DiffusionError::Interrupted(_)));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].completed_steps, 1);
        assert_eq!(events[0].total_steps, 2);
    }

    #[test]
    fn scheduler_scales_model_input_for_euler_class() {
        let config = SchedulerConfig {
            class_name: "EulerDiscreteScheduler".into(),
            beta_start: Some(0.0001),
            beta_end: Some(0.02),
            beta_schedule: Some("linear".into()),
            num_train_timesteps: Some(1000),
            prediction_type: Some("epsilon".into()),
            ..SchedulerConfig::default()
        };
        let schedule = DiffusionSchedule::from_config(&config, 1).unwrap();
        let sample = CpuTensor {
            shape: vec![1, 1, 1, 1],
            data: vec![2.0],
        };

        let scaled = schedule.scale_model_input(&sample, 0).unwrap();

        assert_eq!(schedule.input_scaling, SchedulerInputScaling::Sigma);
        assert!(scaled.data[0] < sample.data[0]);
    }

    #[test]
    fn scheduler_scales_initial_latents_for_euler_class() {
        let config = SchedulerConfig {
            class_name: "EulerDiscreteScheduler".into(),
            beta_start: Some(0.0001),
            beta_end: Some(0.02),
            beta_schedule: Some("linear".into()),
            num_train_timesteps: Some(1000),
            prediction_type: Some("epsilon".into()),
            ..SchedulerConfig::default()
        };
        let schedule = DiffusionSchedule::from_config(&config, 2).unwrap();
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            data: vec![1.0, -2.0],
        };
        let sigma = schedule.initial_noise_sigma();

        schedule.scale_initial_latents(&mut latents);

        assert!(sigma > 1.0);
        assert_eq!(latents.data, vec![sigma, -2.0 * sigma]);
    }

    #[test]
    fn scheduler_step_supports_sample_prediction_type() {
        let mut schedule = DiffusionSchedule::linear(1).unwrap();
        schedule.prediction_type = SchedulerPredictionType::Sample;
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![2.0],
        };

        schedule.euler_step(&mut latents, &[1.5], 0).unwrap();

        assert_eq!(latents.data, vec![1.5]);
    }

    #[test]
    fn scheduler_step_supports_v_prediction_type() {
        let mut schedule = DiffusionSchedule::linear(1).unwrap();
        schedule.prediction_type = SchedulerPredictionType::VPrediction;
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![2.0],
        };

        schedule.euler_step(&mut latents, &[0.5], 0).unwrap();

        let expected =
            2.0 - scheduler_derivative(2.0, 0.5, 1.0, SchedulerPredictionType::VPrediction);
        assert!((latents.data[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn unet_input_centering_matches_diffusers_config() {
        let sample = CpuTensor {
            shape: vec![1, 1, 1, 3],
            data: vec![0.0, 0.5, 1.0],
        };

        let centered = maybe_center_unet_input(&sample, true);
        let unchanged = maybe_center_unet_input(&sample, false);

        assert_eq!(centered.shape, sample.shape);
        assert_eq!(centered.data, vec![-1.0, 0.0, 1.0]);
        assert_eq!(unchanged, sample);
    }

    #[test]
    fn denoise_loop_applies_classifier_free_guidance() {
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            data: vec![1.0, -1.0],
        };
        let schedule = DiffusionSchedule::linear(1).unwrap();
        let positive = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![1.0],
        };
        let negative = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.0],
        };
        let out = denoise_latents_with_cfg(
            latents,
            &schedule,
            2.0,
            &positive,
            &negative,
            |_sample, _timesteps, encoder| {
                if encoder.data[0] == 0.0 {
                    Ok(CpuTensor {
                        shape: vec![1, 1, 1, 2],
                        data: vec![0.25, -0.25],
                    })
                } else {
                    Ok(CpuTensor {
                        shape: vec![1, 1, 1, 2],
                        data: vec![0.75, 0.25],
                    })
                }
            },
        )
        .unwrap();

        assert_eq!(out.data, vec![-0.25, -1.75]);
    }

    #[test]
    fn denoise_loop_uses_scheduler_model_input_scaling() {
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![2.0],
        };
        let mut schedule = DiffusionSchedule::linear(1).unwrap();
        schedule.input_scaling = SchedulerInputScaling::Sigma;
        let positive = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![1.0],
        };
        let negative = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.0],
        };
        let mut seen_sample = None;
        let _ = denoise_latents_with_cfg(
            latents,
            &schedule,
            1.0,
            &positive,
            &negative,
            |sample, _timesteps, _encoder| {
                seen_sample.get_or_insert(sample.data[0]);
                Ok(CpuTensor {
                    shape: vec![1, 1, 1, 1],
                    data: vec![0.0],
                })
            },
        )
        .unwrap();

        assert!((seen_sample.unwrap() - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn denoise_loop_rejects_bad_conditioning_and_noise_shapes() {
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![0.0],
        };
        let schedule = DiffusionSchedule::linear(1).unwrap();
        let positive = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![1.0],
        };
        let negative_bad_batch = CpuTensor {
            shape: vec![2, 1, 1],
            data: vec![0.0, 0.0],
        };
        assert!(denoise_latents_with_cfg(
            latents.clone(),
            &schedule,
            1.0,
            &positive,
            &negative_bad_batch,
            |_sample, _timesteps, _encoder| unreachable!(),
        )
        .is_err());

        let negative = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.0],
        };
        assert!(denoise_latents_with_cfg(
            latents,
            &schedule,
            1.0,
            &positive,
            &negative,
            |_sample, _timesteps, _encoder| Ok(CpuTensor {
                shape: vec![1, 1, 1, 2],
                data: vec![0.0, 0.0],
            }),
        )
        .is_err());
    }

    #[test]
    fn latent_shape_uses_vae_scale_factor() {
        let mut config = StableDiffusionConfig {
            pipeline_class: "StableDiffusionPipeline".into(),
            text_encoder: TextEncoderConfig::default(),
            text_encoder_2: None,
            unet: UnetConfig::default(),
            vae: VaeConfig::default(),
            scheduler: SchedulerConfig::default(),
            latent_channels: 4,
            latent_height: Some(64),
            latent_width: Some(64),
            vae_scale_factor: 8,
        };
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a".into(),
                negative_prompt: String::new(),
                seed: 1,
                subseed: None,
            }],
            width: 512,
            height: 512,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 20,
            cfg_scale: 7.0,
            scheduler: "DPM++ 2M".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };
        let shape = latent_shape_for_request(&config, &request).unwrap();
        assert_eq!(
            shape,
            DiffusionLatentShape {
                batch: 1,
                channels: 4,
                height: 64,
                width: 64
            }
        );

        config.vae_scale_factor = 7;
        assert!(latent_shape_for_request(&config, &request).is_err());
    }

    #[test]
    fn hip_memory_plan_accounts_for_diffusion_buffers() {
        let mut config = StableDiffusionConfig {
            pipeline_class: "StableDiffusionPipeline".into(),
            text_encoder: TextEncoderConfig {
                hidden_size: Some(32),
                max_position_embeddings: Some(4),
                ..TextEncoderConfig::default()
            },
            text_encoder_2: None,
            unet: UnetConfig {
                in_channels: Some(9),
                cross_attention_dim: Some(48),
                ..UnetConfig::default()
            },
            vae: VaeConfig::default(),
            scheduler: SchedulerConfig::default(),
            latent_channels: 4,
            latent_height: None,
            latent_width: None,
            vae_scale_factor: 8,
        };
        let request = DiffusionBatchRequest {
            prompts: vec![
                DiffusionPrompt {
                    prompt: "a".into(),
                    negative_prompt: String::new(),
                    seed: 1,
                    subseed: None,
                },
                DiffusionPrompt {
                    prompt: "b".into(),
                    negative_prompt: String::new(),
                    seed: 2,
                    subseed: None,
                },
            ],
            width: 64,
            height: 32,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 2,
            cfg_scale: 1.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let plan = diffusion_hip_memory_plan(&config, &request).unwrap();

        assert_eq!(
            plan.latent_shape,
            DiffusionLatentShape {
                batch: 2,
                channels: 4,
                height: 4,
                width: 8
            }
        );
        assert_eq!(plan.latent_bytes, 2 * 4 * 4 * 8 * 4);
        assert_eq!(plan.denoise_input_bytes, 2 * 9 * 4 * 8 * 4);
        assert_eq!(plan.conditioning_bytes, 2 * 2 * 1 * 4 * 48 * 4);
        assert_eq!(plan.vae_decode_bytes, plan.latent_bytes);
        assert_eq!(plan.rgb_bytes, 2 * 32 * 64 * 3);
        assert_eq!(
            plan.total_device_bytes,
            plan.latent_bytes
                + plan.denoise_input_bytes
                + plan.conditioning_bytes
                + plan.vae_decode_bytes
                + plan.rgb_bytes
                + plan.scheduler_scratch_bytes
        );

        config.text_encoder_2 = Some(TextEncoderConfig::default());
        let sdxl_plan = diffusion_hip_memory_plan(&config, &request).unwrap();
        assert_eq!(sdxl_plan.conditioning_bytes, plan.conditioning_bytes * 2);
    }

    #[test]
    fn hip_memory_plan_rejects_invalid_latent_dimensions() {
        let config = StableDiffusionConfig {
            pipeline_class: "StableDiffusionPipeline".into(),
            text_encoder: TextEncoderConfig::default(),
            text_encoder_2: None,
            unet: UnetConfig::default(),
            vae: VaeConfig::default(),
            scheduler: SchedulerConfig::default(),
            latent_channels: 4,
            latent_height: None,
            latent_width: None,
            vae_scale_factor: 8,
        };
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a".into(),
                negative_prompt: String::new(),
                seed: 1,
                subseed: None,
            }],
            width: 63,
            height: 64,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 1,
            cfg_scale: 1.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let error = diffusion_hip_memory_plan(&config, &request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("VAE scale factor 8"));
    }

    #[test]
    fn timestep_embedding_matches_diffusers_ordering_flags() {
        let flipped = timestep_embedding(&[0.0], 4, true, 0.0).unwrap();
        assert_eq!(flipped.shape, vec![1, 4]);
        assert_eq!(flipped.data, vec![1.0, 1.0, 0.0, 0.0]);

        let unflipped = timestep_embedding(&[0.0], 4, false, 0.0).unwrap();
        assert_eq!(unflipped.data, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn sdxl_time_ids_default_to_requested_size_and_crop() {
        let request = DiffusionBatchRequest {
            prompts: vec![
                DiffusionPrompt {
                    prompt: "a".into(),
                    negative_prompt: String::new(),
                    seed: 1,
                    subseed: None,
                },
                DiffusionPrompt {
                    prompt: "b".into(),
                    negative_prompt: String::new(),
                    seed: 2,
                    subseed: None,
                },
            ],
            width: 768,
            height: 512,
            original_width: Some(1024),
            original_height: Some(768),
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 8,
            crop_y: 16,
            steps: 1,
            cfg_scale: 7.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: false,
            save_images: false,
        };

        let time_ids = sdxl_time_ids_for_request(&request).unwrap();

        assert_eq!(time_ids.shape, vec![2, 6]);
        assert_eq!(
            time_ids.data,
            vec![
                768.0, 1024.0, 16.0, 8.0, 512.0, 768.0, //
                768.0, 1024.0, 16.0, 8.0, 512.0, 768.0,
            ]
        );
    }

    #[test]
    fn unet_text_time_embedding_projects_pooled_text_and_time_ids() {
        let add_embedding = UnetTextTimeEmbedding {
            addition_time_embed_dim: 2,
            linear_1_weight: CpuTensor {
                shape: vec![2, 14],
                data: vec![0.0; 28],
            },
            linear_1_bias: CpuTensor {
                shape: vec![2],
                data: vec![1.0, -1.0],
            },
            linear_2_weight: CpuTensor {
                shape: vec![2, 2],
                data: vec![1.0, 0.0, 0.0, 1.0],
            },
            linear_2_bias: CpuTensor {
                shape: vec![2],
                data: vec![0.0, 0.0],
            },
        };
        let text_embeds = CpuTensor {
            shape: vec![1, 2],
            data: vec![0.5, -0.25],
        };
        let time_ids = CpuTensor {
            shape: vec![1, 6],
            data: vec![512.0, 512.0, 0.0, 0.0, 512.0, 512.0],
        };

        let output = add_embedding
            .forward(&text_embeds, &time_ids, true, 0.0)
            .unwrap();

        assert_eq!(output.shape, vec![1, 2]);
        assert!((output.data[0] - silu(1.0)).abs() < 1e-6);
        assert!((output.data[1] - silu(-1.0)).abs() < 1e-6);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!(
                    "skip: ROCm GPU unavailable for UNet text-time embedding routing test: {error}"
                );
            } else {
                let hip = add_embedding
                    .forward_with_runtime_options(
                        &text_embeds,
                        &time_ids,
                        true,
                        0.0,
                        DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                    )
                    .unwrap();
                assert_eq!(hip.shape, output.shape);
                assert!(f32_slices_close(&hip.data, &output.data, 1e-5));
            }
        }
    }

    #[test]
    fn conv2d_groupnorm_silu_and_upsample_primitives_work() {
        let input = CpuTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let weight = CpuTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![1.0, 0.0, 0.0, -1.0],
        };
        let bias = CpuTensor {
            shape: vec![1],
            data: vec![0.5],
        };
        let conv = conv2d_nchw(&input, &weight, Some(&bias), 0).unwrap();
        assert_eq!(conv.shape, vec![1, 1, 1, 1]);
        assert_eq!(conv.data, vec![-2.5]);

        let padded = conv2d_nchw(&input, &weight, None, 1).unwrap();
        assert_eq!(padded.shape, vec![1, 1, 3, 3]);
        assert_eq!(padded.data[0], -1.0);

        let gn_input = CpuTensor {
            shape: vec![1, 2, 1, 2],
            data: vec![1.0, 3.0, 10.0, 14.0],
        };
        let affine = CpuTensor {
            shape: vec![2],
            data: vec![1.0, 1.0],
        };
        let zeros = CpuTensor {
            shape: vec![2],
            data: vec![0.0, 0.0],
        };
        let normed = group_norm_nchw(&gn_input, &affine, &zeros, 2, 1e-5).unwrap();
        assert!(normed.data[0] < -0.99 && normed.data[1] > 0.99);
        assert!(normed.data[2] < -0.99 && normed.data[3] > 0.99);

        assert!((silu(1.0) - 0.7310586).abs() < 1e-6);

        let up = upsample_nearest2d_nchw(&input, 2).unwrap();
        assert_eq!(up.shape, vec![1, 1, 4, 4]);
        assert_eq!(&up.data[0..4], &[1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn resnet_block_loads_from_hfq_and_preserves_residual_shape() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-resnet-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("resnet.hfq");
        let prefix = "vae/tensors/decoder.up_blocks.0.resnets.0";
        let metadata = minimal_metadata();
        let tensors = [
            f32_mem_tensor(&format!("{prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{prefix}.conv1.weight"), &[1, 1, 3, 3], &[0.0; 9]),
            f32_mem_tensor(&format!("{prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{prefix}.conv2.weight"), &[1, 1, 3, 3], &[0.0; 9]),
            f32_mem_tensor(&format!("{prefix}.conv2.bias"), &[1], &[0.0]),
        ];
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let block = ResnetBlock2D::from_hfq(&hfq, prefix, 1).unwrap();
        let input = CpuTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let output = block.forward(&input).unwrap();
        assert_eq!(output.shape, input.shape);
        assert_eq!(output.data, input.data);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for VAE ResNet context test: {error}");
            } else {
                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip = block
                    .forward_with_runtime_context(&input, &mut runtime_context)
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert_eq!(hip.shape, output.shape);
                assert!(f32_slices_close(&hip.data, &output.data, 1e-5));
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unet_resnet_block_loads_time_projection_from_hfq() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-unet-resnet-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("unet-resnet.hfq");
        let prefix = "unet/tensors/down_blocks.0.resnets.0";
        let metadata = minimal_metadata();
        let identity_conv = center_identity_conv2(2);
        let tensors = [
            f32_mem_tensor(&format!("{prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{prefix}.conv1.weight"),
                &[2, 2, 3, 3],
                &identity_conv,
            ),
            f32_mem_tensor(&format!("{prefix}.conv1.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{prefix}.time_emb_proj.weight"),
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor(&format!("{prefix}.time_emb_proj.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{prefix}.conv2.weight"),
                &[2, 2, 3, 3],
                &identity_conv,
            ),
            f32_mem_tensor(&format!("{prefix}.conv2.bias"), &[2], &[0.0, 0.0]),
        ];
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let block = UnetResnetBlock2D::from_hfq(&hfq, prefix, 1, 1e-5).unwrap();
        let input = CpuTensor {
            shape: vec![1, 2, 1, 1],
            data: vec![0.0, 2.0],
        };
        let time_a = CpuTensor {
            shape: vec![1, 2],
            data: vec![0.0, 0.0],
        };
        let time_b = CpuTensor {
            shape: vec![1, 2],
            data: vec![2.0, 0.0],
        };
        let out_a = block.forward(&input, &time_a).unwrap();
        let out_b = block.forward(&input, &time_b).unwrap();
        assert_eq!(out_a.shape, input.shape);
        assert_eq!(out_b.shape, input.shape);
        assert_ne!(out_a.data, out_b.data);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for UNet ResNet context test: {error}");
            } else {
                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip = block
                    .forward_with_runtime_context(&input, &time_b, &mut runtime_context)
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert_eq!(hip.shape, out_b.shape);
                assert!(f32_slices_close(&hip.data, &out_b.data, 1e-5));
            }
        }

        let bad_time = CpuTensor {
            shape: vec![2, 2],
            data: vec![0.0; 4],
        };
        assert!(block.forward(&input, &bad_time).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unet_down_block_forward_collects_skips_and_downsamples() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-down-block-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("down-block.hfq");
        let metadata = minimal_metadata();
        let block_prefix = "unet/tensors/down_blocks.0";
        let resnet_prefix = format!("{block_prefix}.resnets.0");
        let attention_prefix = format!("{block_prefix}.attentions.0");
        let block = format!("{attention_prefix}.transformer_blocks.0");
        let identity_conv = center_identity_conv2(2);
        let mut tensors = vec![
            f32_mem_tensor(&format!("{resnet_prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{resnet_prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{resnet_prefix}.conv1.weight"),
                &[2, 2, 3, 3],
                &identity_conv,
            ),
            f32_mem_tensor(&format!("{resnet_prefix}.conv1.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{resnet_prefix}.time_emb_proj.weight"),
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                &format!("{resnet_prefix}.time_emb_proj.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{resnet_prefix}.norm2.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{resnet_prefix}.norm2.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{resnet_prefix}.conv2.weight"),
                &[2, 2, 3, 3],
                &[0.0; 36],
            ),
            f32_mem_tensor(&format!("{resnet_prefix}.conv2.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{block_prefix}.downsamplers.0.conv.weight"),
                &[2, 2, 3, 3],
                &identity_conv,
            ),
            f32_mem_tensor(
                &format!("{block_prefix}.downsamplers.0.conv.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                &format!("{attention_prefix}.norm.weight"),
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(&format!("{attention_prefix}.norm.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{attention_prefix}.proj_in.weight"),
                &[2, 2, 1, 1],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                &format!("{attention_prefix}.proj_in.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                &format!("{attention_prefix}.proj_out.weight"),
                &[2, 2, 1, 1],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                &format!("{attention_prefix}.proj_out.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{block}.norm1.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{block}.norm1.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(&format!("{block}.norm2.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{block}.norm2.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(&format!("{block}.norm3.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{block}.norm3.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(&format!("{block}.ff.net.0.proj.weight"), &[4, 2], &[0.0; 8]),
            f32_mem_tensor(&format!("{block}.ff.net.0.proj.bias"), &[4], &[0.0; 4]),
            f32_mem_tensor(&format!("{block}.ff.net.2.weight"), &[2, 2], &[0.0; 4]),
            f32_mem_tensor(&format!("{block}.ff.net.2.bias"), &[2], &[0.0; 2]),
        ];
        push_zero_attention_tensors(&mut tensors, &format!("{block}.attn1"), 2, 2);
        push_zero_attention_tensors(&mut tensors, &format!("{block}.attn2"), 2, 3);
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let block = UnetDownBlock2D::from_hfq(&hfq, 0, 1, 1, 1, 1e-5).unwrap();
        let input = CpuTensor {
            shape: vec![1, 2, 4, 4],
            data: (0..32).map(|value| value as f32).collect(),
        };
        let time = CpuTensor {
            shape: vec![1, 2],
            data: vec![0.0, 0.0],
        };
        let encoder = CpuTensor {
            shape: vec![1, 1, 3],
            data: vec![0.0; 3],
        };
        #[cfg(feature = "rocm")]
        let input_for_hip = input.clone();
        let (hidden, skips) = block.forward(input, &time, &encoder).unwrap();
        assert_eq!(skips.len(), 2);
        assert_eq!(skips[0].shape, vec![1, 2, 4, 4]);
        assert_eq!(skips[1].shape, vec![1, 2, 2, 2]);
        assert_eq!(hidden.shape, vec![1, 2, 2, 2]);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for UNet down block context test: {error}");
            } else {
                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let (hip_hidden, hip_skips) = block
                    .forward_with_runtime_context(
                        input_for_hip,
                        &time,
                        &encoder,
                        &mut runtime_context,
                    )
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert_eq!(hip_hidden.shape, hidden.shape);
                assert!(f32_slices_close(&hip_hidden.data, &hidden.data, 1e-5));
                assert_eq!(hip_skips.len(), skips.len());
                for (hip_skip, cpu_skip) in hip_skips.iter().zip(&skips) {
                    assert_eq!(hip_skip.shape, cpu_skip.shape);
                    assert!(f32_slices_close(&hip_skip.data, &cpu_skip.data, 1e-5));
                }
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tiny_sd_unet_down_path_loads_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let hfq = HfqFile::open_index_only(&path).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        let down_path = UnetDownPath::from_hfq(&hfq, &config.unet).unwrap();

        assert_eq!(down_path.conv_in.weight.shape, vec![320, 4, 3, 3]);
        assert_eq!(down_path.blocks.len(), 3);
        assert!(down_path.blocks[0].downsampler.is_some());
        assert!(down_path.blocks[1].downsampler.is_some());
        assert!(down_path.blocks[2].downsampler.is_none());
        assert_eq!(
            down_path.blocks[2].resnets[0].conv2.weight.shape,
            vec![1280, 1280, 3, 3]
        );
    }

    #[test]
    fn unet_up_block_pops_skip_and_upsamples() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-up-block-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("up-block.hfq");
        let metadata = minimal_metadata();
        let prefix = "unet/tensors/up_blocks.0";
        let resnet_prefix = format!("{prefix}.resnets.0");
        let tensors = [
            f32_mem_tensor(&format!("{resnet_prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{resnet_prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{resnet_prefix}.conv1.weight"),
                &[1, 2, 3, 3],
                &[0.0; 18],
            ),
            f32_mem_tensor(&format!("{resnet_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{resnet_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{resnet_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{resnet_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{resnet_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{resnet_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{resnet_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{resnet_prefix}.conv_shortcut.weight"),
                &[1, 2, 1, 1],
                &[1.0, 0.0],
            ),
            f32_mem_tensor(&format!("{resnet_prefix}.conv_shortcut.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{prefix}.upsamplers.0.conv.weight"),
                &[1, 1, 3, 3],
                &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{prefix}.upsamplers.0.conv.bias"), &[1], &[0.0]),
        ];
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let block = UnetUpBlock2D::from_hfq(&hfq, 0, 1, 1, 1e-5).unwrap();
        let hidden = CpuTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let mut skips = vec![CpuTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![10.0, 20.0, 30.0, 40.0],
        }];
        let time = CpuTensor {
            shape: vec![1, 2],
            data: vec![0.0, 0.0],
        };
        let encoder = CpuTensor {
            shape: vec![1, 1, 3],
            data: vec![0.0; 3],
        };
        #[cfg(feature = "rocm")]
        let hidden_for_hip = hidden.clone();
        #[cfg(feature = "rocm")]
        let mut skips_for_hip = skips.clone();
        let output = block.forward(hidden, &mut skips, &time, &encoder).unwrap();
        assert!(skips.is_empty());
        assert_eq!(output.shape, vec![1, 1, 4, 4]);
        assert_eq!(&output.data[0..4], &[1.0, 1.0, 2.0, 2.0]);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for UNet up block context test: {error}");
            } else {
                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip_output = block
                    .forward_with_runtime_context(
                        hidden_for_hip,
                        &mut skips_for_hip,
                        &time,
                        &encoder,
                        &mut runtime_context,
                    )
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert!(skips_for_hip.is_empty());
                assert_eq!(hip_output.shape, output.shape);
                assert!(f32_slices_close(&hip_output.data, &output.data, 1e-5));
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tiny_sd_unet_up_path_loads_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let hfq = HfqFile::open_index_only(&path).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        let up_path = UnetUpPath::from_hfq(&hfq, &config.unet).unwrap();

        assert_eq!(up_path.blocks.len(), 3);
        assert_eq!(up_path.blocks[0].resnets.len(), 2);
        assert_eq!(up_path.blocks[1].resnets.len(), 2);
        assert_eq!(up_path.blocks[2].resnets.len(), 2);
        assert!(up_path.blocks[0].upsampler.is_some());
        assert!(up_path.blocks[1].upsampler.is_some());
        assert!(up_path.blocks[2].upsampler.is_none());
        assert_eq!(
            up_path.blocks[0].resnets[0].conv1.weight.shape,
            vec![1280, 2560, 3, 3]
        );
        assert_eq!(
            up_path.blocks[2].resnets[1].conv2.weight.shape,
            vec![320, 320, 3, 3]
        );
    }

    #[test]
    fn tiny_sd_unet_mid_block_loads_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let hfq = HfqFile::open_index_only(&path).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        let Some(mid_block) = UnetMidBlock2DCrossAttn::from_hfq(&hfq, &config.unet).unwrap() else {
            eprintln!("skip: imported tiny-sd artifact has no UNet mid_block tensors");
            return;
        };

        assert!(mid_block.attention.is_some());
        assert!(mid_block.resnet_1.is_some());
        assert_eq!(
            mid_block.resnet_0.conv1.weight.shape,
            vec![1280, 1280, 3, 3]
        );
        assert_eq!(
            mid_block.attention.as_ref().unwrap().proj_in.weight.shape,
            vec![1280, 1280, 1, 1]
        );
    }

    #[test]
    fn unet_mid_block_loads_attention_and_resnets_from_hfq() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-mid-block-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("mid-block.hfq");
        let metadata = minimal_metadata();
        let identity1 = center_identity_conv(1);
        let mid0_prefix = "unet/tensors/mid_block.resnets.0";
        let mid1_prefix = "unet/tensors/mid_block.resnets.1";
        let attention_prefix = "unet/tensors/mid_block.attentions.0";
        let block_prefix = format!("{attention_prefix}.transformer_blocks.0");
        let mut tensors = vec![
            f32_mem_tensor(&format!("{mid0_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid0_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid0_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor(&format!("{mid0_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid0_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{mid0_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{mid0_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid0_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid0_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{mid0_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid1_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor(&format!("{mid1_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid1_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{mid1_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid1_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{mid1_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{attention_prefix}.norm.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{attention_prefix}.norm.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{attention_prefix}.proj_in.weight"),
                &[1, 1, 1, 1],
                &[0.0],
            ),
            f32_mem_tensor(&format!("{attention_prefix}.proj_in.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{attention_prefix}.proj_out.weight"),
                &[1, 1, 1, 1],
                &[0.0],
            ),
            f32_mem_tensor(&format!("{attention_prefix}.proj_out.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{block_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{block_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{block_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{block_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{block_prefix}.norm3.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{block_prefix}.norm3.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{block_prefix}.ff.net.0.proj.weight"),
                &[2, 1],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                &format!("{block_prefix}.ff.net.0.proj.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{block_prefix}.ff.net.2.weight"), &[1, 1], &[0.0]),
            f32_mem_tensor(&format!("{block_prefix}.ff.net.2.bias"), &[1], &[0.0]),
        ];
        push_zero_attention_tensors(&mut tensors, &format!("{block_prefix}.attn1"), 1, 1);
        push_zero_attention_tensors(&mut tensors, &format!("{block_prefix}.attn2"), 1, 1);
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let config = UnetConfig {
            class_name: "UNet2DConditionModel".into(),
            sample_size: Some(2),
            in_channels: Some(1),
            out_channels: Some(1),
            cross_attention_dim: Some(1),
            attention_head_dim: vec![1],
            block_out_channels: vec![1],
            down_block_types: vec!["DownBlock2D".into()],
            up_block_types: vec!["UpBlock2D".into()],
            layers_per_block: Some(1),
            norm_num_groups: Some(1),
            norm_eps: Some(1e-5),
            center_input_sample: true,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
            addition_embed_type: None,
            addition_time_embed_dim: None,
            projection_class_embeddings_input_dim: None,
        };
        let mid_block = UnetMidBlock2DCrossAttn::from_hfq(&hfq, &config)
            .unwrap()
            .unwrap();
        let input = CpuTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let time = CpuTensor {
            shape: vec![1, 2],
            data: vec![0.0, 0.0],
        };
        let encoder = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.0],
        };
        #[cfg(feature = "rocm")]
        let input_for_hip = input.clone();

        let output = mid_block.forward(input, &time, &encoder).unwrap();

        assert!(mid_block.attention.is_some());
        assert!(mid_block.resnet_1.is_some());
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        assert!(output.data.iter().all(|value| value.is_finite()));

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for UNet mid block context test: {error}");
            } else {
                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip_output = mid_block
                    .forward_with_runtime_context(
                        input_for_hip,
                        &time,
                        &encoder,
                        &mut runtime_context,
                    )
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert_eq!(hip_output.shape, output.shape);
                assert!(f32_slices_close(&hip_output.data, &output.data, 1e-5));
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn native_unet_forward_runs_synthetic_graph() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-native-unet-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("native-unet.hfq");
        let metadata = minimal_metadata();
        let identity1 = center_identity_conv(1);
        let down_prefix = "unet/tensors/down_blocks.0.resnets.0";
        let mid0_prefix = "unet/tensors/mid_block.resnets.0";
        let mid1_prefix = "unet/tensors/mid_block.resnets.1";
        let up_prefix = "unet/tensors/up_blocks.0.resnets.0";
        let tensors = [
            f32_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &identity1),
            f32_mem_tensor("unet/tensors/conv_in.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_1.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_1.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_2.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_2.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{down_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{down_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{down_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor(&format!("{down_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{down_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{down_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{down_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{down_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{down_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{down_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{mid0_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid0_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid0_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor(&format!("{mid0_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid0_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{mid0_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{mid0_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid0_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid0_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{mid0_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid1_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor(&format!("{mid1_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid1_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{mid1_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid1_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{mid1_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{up_prefix}.conv1.weight"),
                &[1, 2, 3, 3],
                &[0.0; 18],
            ),
            f32_mem_tensor(&format!("{up_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{up_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{up_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{up_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{up_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{up_prefix}.conv_shortcut.weight"),
                &[1, 2, 1, 1],
                &[1.0, 0.0],
            ),
            f32_mem_tensor(&format!("{up_prefix}.conv_shortcut.bias"), &[1], &[0.0]),
            f32_mem_tensor("unet/tensors/conv_norm_out.weight", &[1], &[1.0]),
            f32_mem_tensor("unet/tensors/conv_norm_out.bias", &[1], &[0.0]),
            f32_mem_tensor("unet/tensors/conv_out.weight", &[1, 1, 3, 3], &identity1),
            f32_mem_tensor("unet/tensors/conv_out.bias", &[1], &[0.0]),
        ];
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let config = UnetConfig {
            class_name: "UNet2DConditionModel".into(),
            sample_size: Some(2),
            in_channels: Some(1),
            out_channels: Some(1),
            cross_attention_dim: Some(1),
            attention_head_dim: vec![1],
            block_out_channels: vec![1],
            down_block_types: vec!["DownBlock2D".into()],
            up_block_types: vec!["UpBlock2D".into()],
            layers_per_block: Some(1),
            norm_num_groups: Some(1),
            norm_eps: Some(1e-5),
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
            addition_embed_type: None,
            addition_time_embed_dim: None,
            projection_class_embeddings_input_dim: None,
        };
        let unet = NativeUnet2DConditionModel::from_hfq(&hfq, &config).unwrap();
        assert!(unet.mid_block.is_some());
        let sample = CpuTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let encoder = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.0],
        };
        let output = unet.forward(&sample, &[0.0], &encoder).unwrap();
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        assert!(output.data.iter().all(|value| value.is_finite()));

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for UNet forward routing test: {error}");
            } else {
                let hip = unet
                    .forward_with_runtime_options(
                        &sample,
                        &[0.0],
                        &encoder,
                        DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                    )
                    .unwrap();
                assert_eq!(hip.shape, output.shape);
                assert!(f32_slices_close(&hip.data, &output.data, 1e-4));
            }
        }

        let bad_encoder = CpuTensor {
            shape: vec![2, 1, 1],
            data: vec![0.0, 0.0],
        };
        assert!(unet.forward(&sample, &[0.0], &bad_encoder).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tiny_sd_native_unet_loads_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let hfq = HfqFile::open_index_only(&path).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        let unet = NativeUnet2DConditionModel::from_hfq(&hfq, &config.unet).unwrap();

        assert_eq!(unet.down_path.blocks.len(), 3);
        assert_eq!(unet.up_path.blocks.len(), 3);
        assert_eq!(unet.conv_norm_out.weight.shape, vec![320]);
        assert_eq!(unet.conv_out.weight.shape, vec![4, 320, 3, 3]);
    }

    #[test]
    fn rgb_tensor_to_u8_maps_model_range_to_pixels() {
        let tensor = CpuTensor {
            shape: vec![1, 3, 1, 2],
            data: vec![-1.0, 1.0, 0.0, 2.0, -2.0, 0.5],
        };
        let image = rgb_tensor_to_u8(&tensor).unwrap();
        assert_eq!(image.batch, 1);
        assert_eq!(image.width, 2);
        assert_eq!(image.height, 1);
        assert_eq!(image.data, vec![0, 128, 0, 255, 255, 191]);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_rgb_tensor_to_u8_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for RGB kernel parity test: {error}");
                return;
            }
        };
        let tensor = CpuTensor {
            shape: vec![1, 3, 2, 2],
            data: vec![
                -1.0, 0.0, 1.0, 0.25, -0.5, 0.5, -0.25, 0.75, 1.0, -1.0, 0.1, -0.1,
            ],
        };
        let cpu = rgb_tensor_to_u8(&tensor).unwrap();
        let hip = rgb_tensor_to_u8_hip_on_gpu(&mut gpu, &tensor).unwrap();

        assert_eq!(hip, cpu);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_vae_boundary_transforms_match_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for VAE boundary kernel parity test: {error}"
                );
                return;
            }
        };
        let image = RgbImageBatch {
            batch: 2,
            width: 2,
            height: 2,
            data: vec![
                0, 128, 255, 255, 0, 128, 32, 64, 96, 192, 224, 16, 10, 20, 30, 40, 50, 60, 70, 80,
                90, 100, 110, 120,
            ],
        };
        let cpu_tensor = rgb_batch_to_vae_tensor(&image).unwrap();
        let hip_tensor = rgb_batch_to_vae_tensor_hip_on_gpu(&mut gpu, &image).unwrap();

        assert_eq!(hip_tensor.shape, cpu_tensor.shape);
        for (index, (actual, expected)) in hip_tensor.data.iter().zip(&cpu_tensor.data).enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "RGB-to-VAE mismatch at {index}: hip={actual} cpu={expected}"
            );
        }

        let moments = CpuTensor {
            shape: vec![2, 4, 2, 2],
            data: (0..32)
                .map(|idx| idx as f32 / 9.0 - 1.5)
                .collect::<Vec<_>>(),
        };
        let cpu_latents = vae_moments_to_latents(&moments, 0.18215).unwrap();
        let hip_latents = vae_moments_to_latents_hip_on_gpu(&mut gpu, &moments, 0.18215).unwrap();

        assert_eq!(hip_latents.batch, cpu_latents.batch);
        assert_eq!(hip_latents.channels, cpu_latents.channels);
        assert_eq!(hip_latents.height, cpu_latents.height);
        assert_eq!(hip_latents.width, cpu_latents.width);
        for (index, (actual, expected)) in
            hip_latents.data.iter().zip(&cpu_latents.data).enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "VAE moments-to-latents mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_inpaint_mask_ops_match_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for inpaint mask kernel parity test: {error}"
                );
                return;
            }
        };
        let image = RgbImageBatch {
            batch: 2,
            width: 4,
            height: 4,
            data: (0..96)
                .map(|idx| ((idx * 19 + 5) % 256) as u8)
                .collect::<Vec<_>>(),
        };
        let mask = RgbImageBatch {
            batch: 2,
            width: 4,
            height: 4,
            data: (0..96)
                .map(|idx| ((idx * 37 + 11) % 256) as u8)
                .collect::<Vec<_>>(),
        };
        let init_latents = LatentBatch {
            batch: 2,
            channels: 2,
            height: 2,
            width: 2,
            data: (0..16)
                .map(|idx| idx as f32 / 7.0 - 1.0)
                .collect::<Vec<_>>(),
        };
        let generated_latents = LatentBatch {
            batch: 2,
            channels: 2,
            height: 2,
            width: 2,
            data: (0..16)
                .map(|idx| (idx as f32 % 9.0 - 4.0) / 3.0)
                .collect::<Vec<_>>(),
        };

        let cpu_weights = latent_mask_weights_from_rgb_batch(&mask, &init_latents).unwrap();
        let hip_weights =
            latent_mask_weights_from_rgb_batch_hip_on_gpu(&mut gpu, &mask, &init_latents).unwrap();
        for (index, (actual, expected)) in hip_weights.iter().zip(&cpu_weights).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "latent mask mismatch at {index}: hip={actual} cpu={expected}"
            );
        }

        let cpu_masked = masked_rgb_batch_for_inpaint(&image, &mask).unwrap();
        let hip_masked = masked_rgb_batch_for_inpaint_hip_on_gpu(&mut gpu, &image, &mask).unwrap();
        assert_eq!(hip_masked, cpu_masked);

        let mut cpu_blend = generated_latents.clone();
        blend_latents_with_mask(&mut cpu_blend, &init_latents, &cpu_weights).unwrap();
        let hip_blend = blend_latents_with_mask_hip_on_gpu(
            &mut gpu,
            &generated_latents,
            &init_latents,
            &cpu_weights,
        )
        .unwrap();
        assert_eq!(hip_blend.batch, cpu_blend.batch);
        assert_eq!(hip_blend.channels, cpu_blend.channels);
        assert_eq!(hip_blend.height, cpu_blend.height);
        assert_eq!(hip_blend.width, cpu_blend.width);
        for (index, (actual, expected)) in hip_blend.data.iter().zip(&cpu_blend.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "latent blend mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_euler_step_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for Euler kernel parity test: {error}");
                return;
            }
        };
        let sample = vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125];
        let model_output = vec![0.5, -0.5, 0.25, -0.25, 1.5, -1.0, 0.75, -0.125];
        let sigma = 1.0;
        let next_sigma = 0.5;
        for prediction_type in [
            SchedulerPredictionType::Epsilon,
            SchedulerPredictionType::Sample,
            SchedulerPredictionType::VPrediction,
        ] {
            let cpu = sample
                .iter()
                .zip(&model_output)
                .map(|(sample, model_output)| {
                    sample
                        + scheduler_derivative(*sample, *model_output, sigma, prediction_type)
                            * (next_sigma - sigma)
                })
                .collect::<Vec<_>>();
            let hip = euler_step_hip_on_gpu(
                &mut gpu,
                &sample,
                &model_output,
                sigma,
                next_sigma,
                prediction_type,
            )
            .unwrap();

            for (index, (actual, expected)) in hip.iter().zip(&cpu).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1e-6,
                    "{prediction_type:?} mismatch at {index}: hip={actual} cpu={expected}"
                );
            }
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_denoise_vector_ops_match_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for denoise vector parity test: {error}");
                return;
            }
        };
        let sample = vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125];
        let scale = 0.5;
        let cpu_scaled = sample
            .iter()
            .map(|sample| sample * scale)
            .collect::<Vec<_>>();
        let hip_scaled = scale_model_input_hip_on_gpu(&mut gpu, &sample, scale).unwrap();
        assert!(f32_slices_close(&hip_scaled, &cpu_scaled, 1e-6));

        let negative = sample;
        let positive = vec![0.5, -0.5, 0.25, -0.25, 1.5, -1.0, 0.75, -0.125];
        let cfg_scale = 7.5;
        let cpu_guided = cfg_guidance(
            &CpuTensor {
                shape: vec![1, 2, 2, 2],
                data: negative.clone(),
            },
            &CpuTensor {
                shape: vec![1, 2, 2, 2],
                data: positive.clone(),
            },
            cfg_scale,
        )
        .unwrap();
        let hip_guided =
            cfg_guidance_hip_on_gpu(&mut gpu, &negative, &positive, cfg_scale).unwrap();
        assert!(f32_slices_close(&hip_guided, &cpu_guided.data, 1e-6));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_denoise_loop_runtime_options_route_vector_stages_when_gpu_is_available() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for denoise loop routing test: {error}");
            return;
        }
        let schedule = DiffusionSchedule {
            timesteps: vec![1.0, 0.0],
            sigmas: vec![1.0, 0.5, 0.0],
            prediction_type: SchedulerPredictionType::Epsilon,
            input_scaling: SchedulerInputScaling::Sigma,
            solver: SchedulerSolver::Euler,
            train_timesteps: Vec::new(),
            alpha_t: Vec::new(),
            sigma_t: Vec::new(),
            lambda_t: Vec::new(),
        };
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 2,
            width: 2,
            data: vec![-0.75, -0.25, 0.5, 1.25],
        };
        let positive_embeddings = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.2],
        };
        let negative_embeddings = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![-0.1],
        };
        let predict_noise =
            |sample: &CpuTensor,
             _timesteps: &[f32],
             encoder_states: &CpuTensor,
             _sdxl: Option<&SdxlDenoiseConditioning<'_>>,
             _runtime_context: &mut DiffusionGenerationRuntimeContext| {
                let bias = encoder_states.data[0];
                Ok(CpuTensor {
                    shape: sample.shape.clone(),
                    data: sample
                        .data
                        .iter()
                        .map(|value| value * 0.25 + bias)
                        .collect(),
                })
            };
        let cpu = denoise_latents_with_cfg_progress_and_runtime_options(
            latents.clone(),
            &schedule,
            2.0,
            &positive_embeddings,
            &negative_embeddings,
            predict_noise,
            None,
            None,
            None,
            None,
            DiffusionGenerationRuntimeOptions::default(),
            None,
        )
        .unwrap();
        let hip = denoise_latents_with_cfg_progress_and_runtime_options(
            latents,
            &schedule,
            2.0,
            &positive_embeddings,
            &negative_embeddings,
            predict_noise,
            None,
            None,
            None,
            None,
            DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            None,
        )
        .unwrap();

        assert_eq!(cpu.runtime_kind, DiffusionRuntimeKind::CpuSourceReference);
        assert_eq!(hip.runtime_kind, DiffusionRuntimeKind::RocmHybridReference);
        assert_eq!(hip.latents.batch, cpu.latents.batch);
        assert_eq!(hip.latents.channels, cpu.latents.channels);
        assert_eq!(hip.latents.height, cpu.latents.height);
        assert_eq!(hip.latents.width, cpu.latents.width);
        assert!(f32_slices_close(&hip.latents.data, &cpu.latents.data, 1e-5));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_denoise_vector_runtime_context_reuses_single_gpu() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for denoise context reuse test: {error}");
            return;
        }
        let schedule = DiffusionSchedule {
            timesteps: vec![1.0],
            sigmas: vec![1.0, 0.5],
            prediction_type: SchedulerPredictionType::Epsilon,
            input_scaling: SchedulerInputScaling::Sigma,
            solver: SchedulerSolver::Euler,
            train_timesteps: Vec::new(),
            alpha_t: Vec::new(),
            sigma_t: Vec::new(),
            lambda_t: Vec::new(),
        };
        let sample = CpuTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![-0.75, -0.25, 0.5, 1.25],
        };
        let negative_pred = CpuTensor {
            shape: sample.shape.clone(),
            data: vec![0.1, -0.2, 0.3, -0.4],
        };
        let positive_pred = CpuTensor {
            shape: sample.shape.clone(),
            data: vec![0.4, -0.1, 0.6, -0.2],
        };
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 2,
            width: 2,
            data: sample.data.clone(),
        };
        let mut scheduler_state = SchedulerStepState::default();
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(
            DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
        );

        let (_scaled, scale_kind) =
            scale_model_input_with_runtime_context(&schedule, &sample, 0, &mut runtime_context)
                .unwrap();
        let (guided, guidance_kind) = cfg_guidance_with_runtime_context(
            &negative_pred,
            &positive_pred,
            2.0,
            &mut runtime_context,
        )
        .unwrap();
        let step_kind = scheduler_step_with_runtime_context(
            &schedule,
            &mut latents,
            &guided.data,
            0,
            &mut scheduler_state,
            &mut runtime_context,
        )
        .unwrap();

        assert_eq!(scale_kind, DiffusionRuntimeKind::RocmHybridReference);
        assert_eq!(guidance_kind, DiffusionRuntimeKind::RocmHybridReference);
        assert_eq!(step_kind, DiffusionRuntimeKind::RocmHybridReference);
        assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
        assert!(latents.data.iter().all(|value| value.is_finite()));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_center_unet_input_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for centered UNet input parity test: {error}"
                );
                return;
            }
        };
        let input = CpuTensor {
            shape: vec![2, 2, 2, 2],
            data: (0..16)
                .map(|idx| idx as f32 / 7.0 - 1.0)
                .collect::<Vec<_>>(),
        };

        let cpu_centered = maybe_center_unet_input(&input, true);
        let hip_centered = maybe_center_unet_input_hip_on_gpu(&mut gpu, &input, true).unwrap();
        assert_eq!(hip_centered, cpu_centered);

        let cpu_passthrough = maybe_center_unet_input(&input, false);
        let hip_passthrough = maybe_center_unet_input_hip_on_gpu(&mut gpu, &input, false).unwrap();
        assert_eq!(hip_passthrough, cpu_passthrough);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_timestep_embedding_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for timestep embedding parity test: {error}");
                return;
            }
        };
        let timesteps = [999.0, 500.5, 0.25];
        for (dim, flip_sin_to_cos, freq_shift) in [(7, true, 1.0), (6, false, 0.0), (1, true, 0.0)]
        {
            let cpu = timestep_embedding(&timesteps, dim, flip_sin_to_cos, freq_shift).unwrap();
            let hip = timestep_embedding_hip_on_gpu(
                &mut gpu,
                &timesteps,
                dim,
                flip_sin_to_cos,
                freq_shift,
            )
            .unwrap();

            assert_eq!(hip.shape, cpu.shape);
            for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1e-5,
                    "timestep embedding mismatch at {index}: dim={dim} flip={flip_sin_to_cos} shift={freq_shift} hip={actual} cpu={expected}"
                );
            }
            if dim % 2 == 1 {
                for row in 0..timesteps.len() {
                    assert_eq!(hip.data[row * dim + dim - 1], 0.0);
                }
            }
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_conv2d_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for Conv2D kernel parity test: {error}");
                return;
            }
        };
        let input = CpuTensor {
            shape: vec![2, 2, 3, 4],
            data: (0..48)
                .map(|idx| idx as f32 / 11.0 - 2.0)
                .collect::<Vec<_>>(),
        };
        let weight = CpuTensor {
            shape: vec![3, 2, 3, 2],
            data: (0..36)
                .map(|idx| (idx as f32 % 9.0 - 4.0) / 6.0)
                .collect::<Vec<_>>(),
        };
        let bias = CpuTensor {
            shape: vec![3],
            data: vec![0.25, -0.5, 0.75],
        };
        for bias in [Some(&bias), None] {
            let cpu = conv2d_nchw_with_stride(&input, &weight, bias, 1, 2).unwrap();
            let hip = conv2d_nchw_hip_on_gpu(&mut gpu, &input, &weight, bias, 1, 2).unwrap();

            assert_eq!(hip.shape, cpu.shape);
            for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1e-5,
                    "Conv2D mismatch at {index}: hip={actual} cpu={expected}"
                );
            }
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_group_norm_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for GroupNorm kernel parity test: {error}");
                return;
            }
        };
        let input = CpuTensor {
            shape: vec![2, 4, 2, 3],
            data: (0..48)
                .map(|idx| idx as f32 / 13.0 - 1.75)
                .collect::<Vec<_>>(),
        };
        let weight = CpuTensor {
            shape: vec![4],
            data: vec![1.0, 0.5, -1.0, 1.5],
        };
        let bias = CpuTensor {
            shape: vec![4],
            data: vec![0.0, 0.25, -0.5, 0.75],
        };

        let cpu = group_norm_nchw(&input, &weight, &bias, 2, 1e-5).unwrap();
        let hip = group_norm_nchw_hip_on_gpu(&mut gpu, &input, &weight, &bias, 2, 1e-5).unwrap();

        assert_eq!(hip.shape, cpu.shape);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "GroupNorm mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_silu_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for SiLU kernel parity test: {error}");
                return;
            }
        };
        let input = CpuTensor {
            shape: vec![2, 2, 2, 2],
            data: vec![
                -8.0, -4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 8.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0,
                0.125,
            ],
        };

        let cpu = tensor_map(&input, silu);
        let hip = silu_hip_on_gpu(&mut gpu, &input).unwrap();

        assert_eq!(hip.shape, cpu.shape);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "SiLU mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_tensor_add_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for tensor-add kernel parity test: {error}");
                return;
            }
        };
        let left = CpuTensor {
            shape: vec![2, 2, 2, 3],
            data: (0..24)
                .map(|idx| idx as f32 / 9.0 - 1.25)
                .collect::<Vec<_>>(),
        };
        let right = CpuTensor {
            shape: vec![2, 2, 2, 3],
            data: (0..24)
                .map(|idx| (idx as f32 % 7.0 - 3.0) / 5.0)
                .collect::<Vec<_>>(),
        };

        let cpu = tensor_add(&left, &right).unwrap();
        let hip = tensor_add_hip_on_gpu(&mut gpu, &left, &right).unwrap();

        assert_eq!(hip.shape, cpu.shape);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "tensor-add mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_add_channel_bias_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for channel-bias kernel parity test: {error}"
                );
                return;
            }
        };
        let input = CpuTensor {
            shape: vec![2, 3, 2, 3],
            data: (0..36)
                .map(|idx| idx as f32 / 13.0 - 1.5)
                .collect::<Vec<_>>(),
        };
        let bias = CpuTensor {
            shape: vec![2, 3],
            data: vec![0.25, -0.5, 0.75, -1.0, 0.5, -0.25],
        };
        let mut cpu = input.clone();
        add_channel_bias_nchw(&mut cpu, &bias).unwrap();
        let hip = add_channel_bias_nchw_hip_on_gpu(&mut gpu, &input, &bias).unwrap();

        assert_eq!(hip.shape, cpu.shape);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "channel-bias mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_nchw_bsc_layout_transforms_match_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for layout kernel parity test: {error}");
                return;
            }
        };
        let input = CpuTensor {
            shape: vec![2, 3, 2, 4],
            data: (0..48)
                .map(|idx| idx as f32 / 17.0 - 1.25)
                .collect::<Vec<_>>(),
        };

        let cpu_bsc = nchw_to_bsc(&input).unwrap();
        let hip_bsc = nchw_to_bsc_hip_on_gpu(&mut gpu, &input).unwrap();
        assert_eq!(hip_bsc, cpu_bsc);

        let cpu_nchw = bsc_to_nchw(&cpu_bsc, 2, 3, 2, 4).unwrap();
        let hip_nchw = bsc_to_nchw_hip_on_gpu(&mut gpu, &cpu_bsc, 2, 3, 2, 4).unwrap();
        assert_eq!(hip_nchw, cpu_nchw);
        assert_eq!(hip_nchw, input);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_concat_channels_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for channel-concat kernel parity test: {error}"
                );
                return;
            }
        };
        let left = CpuTensor {
            shape: vec![2, 2, 2, 3],
            data: (0..24)
                .map(|idx| idx as f32 / 9.0 - 1.0)
                .collect::<Vec<_>>(),
        };
        let right = CpuTensor {
            shape: vec![2, 3, 2, 3],
            data: (0..36)
                .map(|idx| (idx as f32 % 13.0 - 6.0) / 7.0)
                .collect::<Vec<_>>(),
        };

        let cpu = concat_channels_nchw(&left, &right).unwrap();
        let hip = concat_channels_nchw_hip_on_gpu(&mut gpu, &left, &right).unwrap();

        assert_eq!(hip, cpu);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_concat_last_dim_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for last-dim concat kernel parity test: {error}"
                );
                return;
            }
        };
        let left_2d = CpuTensor {
            shape: vec![4, 2],
            data: (0..8).map(|idx| idx as f32 / 5.0 - 0.75).collect(),
        };
        let right_2d = CpuTensor {
            shape: vec![4, 3],
            data: (0..12).map(|idx| (idx as f32 % 7.0 - 3.0) / 4.0).collect(),
        };
        let cpu_2d = concat_last_dim_2d(&left_2d, &right_2d).unwrap();
        let hip_2d = concat_last_dim_2d_hip_on_gpu(&mut gpu, &left_2d, &right_2d).unwrap();
        assert_eq!(hip_2d, cpu_2d);

        let left_3d = CpuTensor {
            shape: vec![2, 3, 2],
            data: (0..12).map(|idx| idx as f32 / 6.0 - 1.0).collect(),
        };
        let right_3d = CpuTensor {
            shape: vec![2, 3, 4],
            data: (0..24).map(|idx| (idx as f32 % 11.0 - 5.0) / 8.0).collect(),
        };
        let cpu_3d = concat_last_dim_3d(&left_3d, &right_3d).unwrap();
        let hip_3d = concat_last_dim_3d_hip_on_gpu(&mut gpu, &left_3d, &right_3d).unwrap();
        assert_eq!(hip_3d, cpu_3d);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_upsample_nearest2d_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for nearest-upsample kernel parity test: {error}"
                );
                return;
            }
        };
        let input = CpuTensor {
            shape: vec![2, 2, 2, 3],
            data: (0..24)
                .map(|idx| idx as f32 / 7.0 - 1.25)
                .collect::<Vec<_>>(),
        };

        for scale in [2, 3] {
            let cpu = upsample_nearest2d_nchw(&input, scale).unwrap();
            let hip = upsample_nearest2d_nchw_hip_on_gpu(&mut gpu, &input, scale).unwrap();

            assert_eq!(hip.shape, cpu.shape);
            assert_eq!(hip.data, cpu.data);
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_linear_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for linear kernel parity test: {error}");
                return;
            }
        };
        let input = CpuTensor {
            shape: vec![4, 3],
            data: (0..12)
                .map(|idx| idx as f32 / 5.0 - 1.1)
                .collect::<Vec<_>>(),
        };
        let weight = CpuTensor {
            shape: vec![5, 3],
            data: (0..15)
                .map(|idx| (idx as f32 % 7.0 - 3.0) / 4.0)
                .collect::<Vec<_>>(),
        };
        let bias = CpuTensor {
            shape: vec![5],
            data: vec![0.25, -0.5, 0.75, -1.0, 1.25],
        };

        for bias in [Some(&bias), None] {
            let cpu = linear_optional_bias(&input, &weight, bias).unwrap();
            let hip = linear_optional_bias_hip_on_gpu(&mut gpu, &input, &weight, bias).unwrap();

            assert_eq!(hip.shape, cpu.shape);
            for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1e-5,
                    "linear mismatch at {index}: hip={actual} cpu={expected}"
                );
            }
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_layer_norm_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for LayerNorm kernel parity test: {error}");
                return;
            }
        };
        let input = CpuTensor {
            shape: vec![4, 5],
            data: (0..20)
                .map(|idx| idx as f32 / 6.0 - 1.75)
                .collect::<Vec<_>>(),
        };
        let weight = CpuTensor {
            shape: vec![5],
            data: vec![1.0, 0.5, -1.0, 1.5, -0.25],
        };
        let bias = CpuTensor {
            shape: vec![5],
            data: vec![0.0, 0.25, -0.5, 0.75, -1.0],
        };

        let cpu = layer_norm(&input, &weight, &bias, 1e-5).unwrap();
        let hip = layer_norm_hip_on_gpu(&mut gpu, &input, &weight, &bias, 1e-5).unwrap();

        assert_eq!(hip.shape, cpu.shape);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "LayerNorm mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_softmax_rows_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for softmax kernel parity test: {error}");
                return;
            }
        };
        let input = CpuTensor {
            shape: vec![4, 5],
            data: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, -3.0, -1.0, -0.5, 0.25, 2.5, 10.0, 9.5, 8.0, 7.25, 6.0,
                100.0, 99.0, 98.0, 97.0, 96.0,
            ],
        };
        let mut cpu = input.clone();
        for row in cpu.data.chunks_mut(5) {
            softmax_in_place(row);
        }
        let hip = softmax_rows_hip_on_gpu(&mut gpu, &input).unwrap();

        assert_eq!(hip.shape, cpu.shape);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "softmax mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
        for row in hip.data.chunks(5) {
            let sum = row.iter().sum::<f32>();
            assert!((sum - 1.0).abs() <= 1e-6, "softmax row sum {sum}");
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_sdpa_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for SDPA kernel parity test: {error}");
                return;
            }
        };
        let q = CpuTensor {
            shape: vec![2, 2, 4],
            data: vec![
                0.2, -0.4, 0.6, 1.0, -1.2, 0.3, 0.5, -0.7, 1.0, 0.0, -0.5, 0.25, -0.25, 0.75, -1.0,
                0.5,
            ],
        };
        let k = CpuTensor {
            shape: vec![2, 3, 4],
            data: vec![
                -0.5, 0.25, 0.75, -1.0, 1.25, -0.75, 0.5, 0.0, 0.1, 0.9, -0.3, 0.4, 0.7, -0.2, 0.3,
                -0.8, -0.6, 1.1, 0.2, 0.9, 1.5, -1.0, 0.4, -0.1,
            ],
        };
        let v = CpuTensor {
            shape: vec![2, 3, 4],
            data: vec![
                0.5, -1.0, 0.25, 0.75, -0.4, 0.6, -0.8, 1.2, 1.0, 0.2, -0.5, -0.1, -0.9, 0.3, 0.8,
                -0.2, 0.4, -0.7, 1.1, 0.6, -1.3, 0.5, 0.0, 0.9,
            ],
        };

        let cpu = scaled_dot_product_attention(&q, &k, &v, 2).unwrap();
        let hip = scaled_dot_product_attention_hip_on_gpu(&mut gpu, &q, &k, &v, 2).unwrap();

        assert_eq!(hip.shape, cpu.shape);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "SDPA mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_geglu_gate_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for GeGLU gate parity test: {error}");
                return;
            }
        };
        let projected = CpuTensor {
            shape: vec![2, 3, 6],
            data: (0..36)
                .map(|idx| (idx as f32 % 13.0 - 6.0) / 4.0)
                .collect::<Vec<_>>(),
        };

        let cpu = geglu_gate_3d(&projected).unwrap();
        let hip = geglu_gate_3d_hip_on_gpu(&mut gpu, &projected).unwrap();

        assert_eq!(hip.shape, cpu.shape);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "GeGLU gate mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_clip_causal_attention_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for CLIP causal attention parity test: {error}"
                );
                return;
            }
        };
        let q = CpuTensor {
            shape: vec![4, 4],
            data: vec![
                0.2, -0.4, 0.6, 1.0, -1.2, 0.3, 0.5, -0.7, 1.0, 0.0, -0.5, 0.25, -0.25, 0.75, -1.0,
                0.5,
            ],
        };
        let k = CpuTensor {
            shape: vec![4, 4],
            data: vec![
                -0.5, 0.25, 0.75, -1.0, 1.25, -0.75, 0.5, 0.0, 0.1, 0.9, -0.3, 0.4, 0.7, -0.2, 0.3,
                -0.8,
            ],
        };
        let v = CpuTensor {
            shape: vec![4, 4],
            data: vec![
                0.5, -1.0, 0.25, 0.75, -0.4, 0.6, -0.8, 1.2, 1.0, 0.2, -0.5, -0.1, -0.9, 0.3, 0.8,
                -0.2,
            ],
        };

        let cpu = clip_causal_self_attention(&q, &k, &v, 2).unwrap();
        let hip = clip_causal_self_attention_hip_on_gpu(&mut gpu, &q, &k, &v, 2).unwrap();

        assert_eq!(hip.shape, cpu.shape);
        assert_eq!(&cpu.data[0..4], &v.data[0..4]);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "CLIP causal attention mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }

    #[test]
    fn rgb_batch_to_vae_tensor_maps_pixels_to_model_range() {
        let image = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 1,
            data: vec![0, 128, 255, 255, 0, 128],
        };

        let tensor = rgb_batch_to_vae_tensor(&image).unwrap();

        assert_eq!(tensor.shape, vec![1, 3, 1, 2]);
        assert!((tensor.data[nchw_idx(0, 0, 0, 0, 3, 1, 2)] + 1.0).abs() < 1e-6);
        assert!((tensor.data[nchw_idx(0, 1, 0, 0, 3, 1, 2)] - 0.003921628).abs() < 1e-6);
        assert!((tensor.data[nchw_idx(0, 2, 0, 0, 3, 1, 2)] - 1.0).abs() < 1e-6);
        assert!((tensor.data[nchw_idx(0, 0, 0, 1, 3, 1, 2)] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vae_moments_to_latents_selects_mean_channels_and_scales() {
        let moments = CpuTensor {
            shape: vec![1, 4, 1, 2],
            data: vec![1.0, -2.0, 3.0, -4.0, 10.0, 20.0, 30.0, 40.0],
        };

        let latents = vae_moments_to_latents(&moments, 0.5).unwrap();

        assert_eq!(latents.batch, 1);
        assert_eq!(latents.channels, 2);
        assert_eq!(latents.height, 1);
        assert_eq!(latents.width, 2);
        assert_eq!(latents.data, vec![0.5, -1.0, 1.5, -2.0]);
    }

    #[test]
    fn rgb_batch_encodes_to_decodeable_png_base64_images() {
        let batch = RgbImageBatch {
            batch: 2,
            width: 1,
            height: 1,
            data: vec![255, 0, 0, 0, 255, 0],
        };

        let images = encode_rgb_batch_png_base64(&batch).unwrap();

        assert_eq!(images.len(), 2);
        for image in images {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(image)
                .unwrap();
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
            let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
            assert_eq!(decoded.dimensions(), (1, 1));
        }
    }

    #[test]
    fn rgb_batch_resize_nearest_preserves_batch_items() {
        let image = RgbImageBatch {
            batch: 2,
            width: 1,
            height: 2,
            data: vec![
                10, 20, 30, //
                40, 50, 60, //
                70, 80, 90, //
                100, 110, 120,
            ],
        };

        let resized = resize_rgb_batch_nearest(&image, 2, 4).unwrap();

        assert_eq!(resized.batch, 2);
        assert_eq!(resized.width, 2);
        assert_eq!(resized.height, 4);
        assert_eq!(
            resized.data,
            vec![
                10, 20, 30, 10, 20, 30, //
                10, 20, 30, 10, 20, 30, //
                40, 50, 60, 40, 50, 60, //
                40, 50, 60, 40, 50, 60, //
                70, 80, 90, 70, 80, 90, //
                70, 80, 90, 70, 80, 90, //
                100, 110, 120, 100, 110, 120, //
                100, 110, 120, 100, 110, 120,
            ]
        );
    }

    #[test]
    fn rgb_batch_resize_to_cover_center_crops_aspect_mismatch() {
        let image = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 4,
            data: vec![
                10, 10, 10, 10, 10, 10, //
                20, 20, 20, 20, 20, 20, //
                30, 30, 30, 30, 30, 30, //
                40, 40, 40, 40, 40, 40, //
            ],
        };

        let resized = resize_rgb_batch_to_cover_nearest(&image, 4, 4).unwrap();

        assert_eq!(resized.batch, 1);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
        assert_eq!(
            resized.data,
            vec![
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
                30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, //
                30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, //
            ]
        );
    }

    #[test]
    fn rgb_batch_resize_to_contain_fill_extends_edges() {
        let image = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 4,
            data: vec![
                10, 10, 10, 11, 11, 11, //
                20, 20, 20, 21, 21, 21, //
                30, 30, 30, 31, 31, 31, //
                40, 40, 40, 41, 41, 41, //
            ],
        };

        let resized = resize_rgb_batch_to_contain_fill_nearest(&image, 4, 4).unwrap();

        assert_eq!(resized.batch, 1);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
        assert_eq!(
            resized.data,
            vec![
                10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, //
                20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, //
                30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, //
                40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, //
            ]
        );
    }

    #[test]
    fn latent_mask_weights_downsample_rgb_luma_to_latent_shape() {
        let mask = RgbImageBatch {
            batch: 1,
            width: 4,
            height: 4,
            data: vec![
                0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, //
                0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, //
                128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64, //
                128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64,
            ],
        };
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 2,
            width: 2,
            data: vec![0.0; 4],
        };

        let weights = latent_mask_weights_from_rgb_batch(&mask, &latents).unwrap();

        assert_eq!(weights.len(), 4);
        assert_eq!(weights[0], 0.0);
        assert_eq!(weights[1], 1.0);
        assert!((weights[2] - (128.0 / 255.0)).abs() < 1e-6);
        assert!((weights[3] - (64.0 / 255.0)).abs() < 1e-6);
    }

    #[test]
    fn masked_rgb_batch_for_inpaint_zeroes_white_mask_pixels() {
        let image = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 1,
            data: vec![10, 20, 30, 100, 120, 140],
        };
        let mask = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 1,
            data: vec![0, 0, 0, 255, 255, 255],
        };

        let masked = masked_rgb_batch_for_inpaint(&image, &mask).unwrap();

        assert_eq!(masked.data, vec![10, 20, 30, 0, 0, 0]);
    }

    #[test]
    fn append_inpaint_conditioning_concatenates_latents_mask_and_masked_latents() {
        let sample = CpuTensor {
            shape: vec![1, 2, 1, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let conditioning = InpaintDenoiseConditioning {
            mask_weights: vec![0.25, 0.75],
            masked_image_latents: LatentBatch {
                batch: 1,
                channels: 2,
                height: 1,
                width: 2,
                data: vec![5.0, 6.0, 7.0, 8.0],
            },
        };

        let conditioned = append_inpaint_conditioning(&sample, &conditioning).unwrap();

        assert_eq!(conditioned.shape, vec![1, 5, 1, 2]);
        assert_eq!(
            conditioned.data,
            vec![1.0, 2.0, 3.0, 4.0, 0.25, 0.75, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn blend_latents_with_mask_preserves_black_and_uses_generated_white() {
        let mut generated = LatentBatch {
            batch: 1,
            channels: 2,
            height: 1,
            width: 2,
            data: vec![10.0, 20.0, 30.0, 40.0],
        };
        let init = LatentBatch {
            batch: 1,
            channels: 2,
            height: 1,
            width: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };

        blend_latents_with_mask(&mut generated, &init, &[0.0, 1.0]).unwrap();

        assert_eq!(generated.data, vec![1.0, 20.0, 3.0, 40.0]);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_img2img_boundary_helpers_reuse_single_runtime_context() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for img2img boundary reuse test: {error}");
            return;
        }
        let image = RgbImageBatch {
            batch: 1,
            width: 4,
            height: 4,
            data: (0..48).map(|idx| (idx * 3 % 256) as u8).collect(),
        };
        let mask = RgbImageBatch {
            batch: 1,
            width: 4,
            height: 4,
            data: (0..16)
                .flat_map(|idx| {
                    let value = if idx % 3 == 0 { 255 } else { 64 };
                    [value, value, value]
                })
                .collect(),
        };
        let init = LatentBatch {
            batch: 1,
            channels: 2,
            height: 2,
            width: 2,
            data: (0..8).map(|idx| idx as f32 / 8.0).collect(),
        };
        let mut generated = LatentBatch {
            batch: 1,
            channels: 2,
            height: 2,
            width: 2,
            data: (0..8).map(|idx| 1.0 - idx as f32 / 8.0).collect(),
        };
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(
            DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
        );

        let (weights, mask_kind) =
            latent_mask_weights_with_runtime_context(&mask, &init, &mut runtime_context).unwrap();
        let (masked, masked_kind) =
            masked_rgb_batch_for_inpaint_with_runtime_context(&image, &mask, &mut runtime_context)
                .unwrap();
        let blend_kind = blend_latents_with_mask_with_runtime_context(
            &mut generated,
            &init,
            &weights,
            &mut runtime_context,
        )
        .unwrap();

        assert_eq!(mask_kind, DiffusionRuntimeKind::RocmHybridReference);
        assert_eq!(masked_kind, DiffusionRuntimeKind::RocmHybridReference);
        assert_eq!(blend_kind, DiffusionRuntimeKind::RocmHybridReference);
        assert_eq!(masked.batch, image.batch);
        assert_eq!(masked.width, image.width);
        assert_eq!(masked.height, image.height);
        assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
    }

    #[test]
    fn masked_denoise_reference_reprojects_noised_init_latents_per_step() {
        let source_schedule = DiffusionSchedule::linear(3).unwrap();
        let init = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            data: vec![10.0, 20.0],
        };
        let noise = vec![2.0, 4.0];
        let mut generated = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            data: vec![100.0, 200.0],
        };
        let reference = MaskedDenoiseReference {
            init_latents: &init,
            noise: &noise,
            mask_weights: &[0.0, 1.0],
            source_schedule: &source_schedule,
            start_step: 0,
        };

        apply_masked_denoise_reference(&mut generated, &reference, 0).unwrap();

        assert_eq!(generated.data, vec![11.0, 200.0]);
    }

    #[test]
    fn diffusion_pipeline_generate_batch_returns_sdapi_png_images_with_test_backend() {
        let metadata = tiny_runtime_metadata();
        let config = tiny_runtime_config();
        let tokenizer = ClipTokenizer::from_bytes(
            br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "cat</w>": 3
            }"#,
            b"#version: 0.2\n",
            4,
        )
        .unwrap();
        let text_encoder = ClipTextEncoder {
            token_embedding: CpuTensor {
                shape: vec![4, 2],
                data: vec![0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
            },
            position_embedding: CpuTensor {
                shape: vec![4, 2],
                data: vec![0.0; 8],
            },
            layers: Vec::new(),
            final_layer_norm_weight: CpuTensor {
                shape: vec![2],
                data: vec![1.0, 1.0],
            },
            final_layer_norm_bias: CpuTensor {
                shape: vec![2],
                data: vec![0.0, 0.0],
            },
            text_projection: None,
            hidden_size: 2,
            max_length: 4,
            n_heads: 1,
        };
        let pipeline = DiffusionPipeline {
            summary: summarize_hfq(Path::new("/tmp/tiny-runtime.hfq"), &metadata),
            metadata,
            config,
            tokenizer: Some(tokenizer),
            tokenizer_2: None,
            text_encoder: Some(text_encoder),
            text_encoder_2: None,
            native_runtime: Some(NativeDiffusionRuntime {
                kind: DiffusionRuntimeKind::CpuSourceReference,
                noise: Box::new(TestNoiseBackend),
                encoder: None,
                decoder: Box::new(TestImageDecoder),
            }),
            native_runtime_error: None,
        };
        let request = DiffusionBatchRequest {
            prompts: vec![
                DiffusionPrompt {
                    prompt: "a cat".into(),
                    negative_prompt: String::new(),
                    seed: 7,
                    subseed: None,
                },
                DiffusionPrompt {
                    prompt: "a cat".into(),
                    negative_prompt: "blur".into(),
                    seed: 8,
                    subseed: None,
                },
            ],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 2,
            cfg_scale: 7.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let mut progress_events = Vec::new();
        let output = pipeline
            .generate_batch_with_progress(request, &mut |progress| {
                progress_events.push(progress);
                Ok(())
            })
            .unwrap();

        assert_eq!(output.images.len(), 2);
        assert_eq!(progress_events.len(), 2);
        assert_eq!(progress_events[0].completed_steps, 1);
        assert_eq!(progress_events[0].total_steps, 2);
        let first_preview = progress_events[0].preview_latents.as_ref().unwrap();
        assert_eq!(first_preview.batch, 2);
        assert_eq!(first_preview.channels, 1);
        assert_eq!(first_preview.height, 2);
        assert_eq!(first_preview.width, 2);
        assert_eq!(progress_events[1].completed_steps, 2);
        assert_eq!(progress_events[1].total_steps, 2);
        assert!(progress_events[1].preview_latents.is_some());
        assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
        assert_eq!(output.info["runtime"], "cpu-source-reference");
        assert_eq!(output.info["latent_shape"]["batch"], 2);
        let capabilities = pipeline.runtime_capabilities().unwrap();
        assert_eq!(capabilities.kind, DiffusionRuntimeKind::CpuSourceReference);
        assert_eq!(capabilities.weight_format, "source");
        assert!(!capabilities.supports_img2img);
        for image in output.images {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(image)
                .unwrap();
            let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
            assert_eq!(decoded.dimensions(), (2, 2));
        }
    }

    #[test]
    fn runtime_options_default_decode_uses_cpu_rgb_conversion() {
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 2,
            width: 2,
            data: vec![0.0, 0.25, 0.5, 0.75],
        };
        let (rgb, runtime_kind) = decode_to_rgb8_with_runtime_options(
            &SolidTensorImageDecoder,
            &latents,
            DiffusionGenerationRuntimeOptions::default(),
        )
        .unwrap();

        assert_eq!(runtime_kind, DiffusionRuntimeKind::CpuSourceReference);
        assert_eq!(rgb, SolidTensorImageDecoder::expected_rgb(&latents));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn runtime_options_rocm_hybrid_decode_matches_cpu_when_gpu_is_available() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for hybrid decode test: {error}");
            return;
        }
        let latents = LatentBatch {
            batch: 2,
            channels: 1,
            height: 2,
            width: 3,
            data: vec![0.0; 12],
        };
        let (rgb, runtime_kind) = decode_to_rgb8_with_runtime_options(
            &SolidTensorImageDecoder,
            &latents,
            DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
        )
        .unwrap();

        assert_eq!(runtime_kind, DiffusionRuntimeKind::RocmHybridReference);
        assert_eq!(rgb, SolidTensorImageDecoder::expected_rgb(&latents));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn generate_batch_runtime_options_surface_rocm_hybrid_runtime_when_gpu_is_available() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for hybrid generation test: {error}");
            return;
        }
        let pipeline = tiny_txt2img_test_pipeline(Box::new(SolidTensorImageDecoder));
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 7,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 2,
            cfg_scale: 7.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let output = pipeline
            .generate_batch_with_runtime_options(
                request,
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            )
            .unwrap();

        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["runtime"], "rocm-hybrid-reference");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        assert_eq!(decoded.get_pixel(0, 0).0, [32, 128, 224]);
    }

    #[test]
    fn diffusion_pipeline_prepares_secondary_clip_conditioning_when_available() {
        let mut metadata = tiny_runtime_metadata();
        metadata.pipeline.class_name = "StableDiffusionXLPipeline".into();
        metadata.tokenizer_2 = Some(DiffusionTokenizerMetadata {
            kind: "clip-bpe".into(),
            max_length: Some(4),
            entries: vec!["tokenizer_2/vocab.json".into()],
        });
        let mut config = tiny_runtime_config();
        config.pipeline_class = "StableDiffusionXLPipeline".into();
        config.text_encoder_2 = Some(TextEncoderConfig {
            class_name: "CLIPTextModelWithProjection".into(),
            hidden_size: Some(2),
            intermediate_size: Some(4),
            num_hidden_layers: Some(0),
            num_attention_heads: Some(1),
            max_position_embeddings: Some(4),
            vocab_size: Some(4),
        });
        let tokenizer = ClipTokenizer::from_bytes(
            br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "cat</w>": 3
            }"#,
            b"#version: 0.2\n",
            4,
        )
        .unwrap();
        let text_encoder = ClipTextEncoder {
            token_embedding: CpuTensor {
                shape: vec![4, 2],
                data: vec![0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
            },
            position_embedding: CpuTensor {
                shape: vec![4, 2],
                data: vec![0.0; 8],
            },
            layers: Vec::new(),
            final_layer_norm_weight: CpuTensor {
                shape: vec![2],
                data: vec![1.0, 1.0],
            },
            final_layer_norm_bias: CpuTensor {
                shape: vec![2],
                data: vec![0.0, 0.0],
            },
            text_projection: Some(CpuTensor {
                shape: vec![2, 2],
                data: vec![1.0, 0.0, 0.0, 1.0],
            }),
            hidden_size: 2,
            max_length: 4,
            n_heads: 1,
        };
        let pipeline = DiffusionPipeline {
            summary: summarize_hfq(Path::new("/tmp/tiny-sdxl-runtime.hfq"), &metadata),
            metadata,
            config,
            tokenizer: Some(tokenizer.clone()),
            tokenizer_2: Some(tokenizer),
            text_encoder: Some(text_encoder.clone()),
            text_encoder_2: Some(text_encoder),
            native_runtime: None,
            native_runtime_error: Some("dual encoder test".into()),
        };
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 7,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 2,
            cfg_scale: 7.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: false,
            save_images: false,
        };

        let conditioning = pipeline.prepare_conditioning_batch(&request).unwrap();

        assert!(conditioning.prompt_tokens_2.is_some());
        assert_eq!(
            conditioning.prompt_embeddings_2.as_ref().unwrap().shape,
            vec![1, 4, 2]
        );
        assert_eq!(
            conditioning
                .prompt_pooled_embeddings
                .as_ref()
                .unwrap()
                .shape,
            vec![1, 2]
        );
        assert_eq!(
            conditioning
                .negative_pooled_embeddings
                .as_ref()
                .unwrap()
                .shape,
            vec![1, 2]
        );
    }

    #[test]
    fn diffusion_pipeline_passes_sdxl_conditioning_to_noise_backend() {
        let mut metadata = tiny_runtime_metadata();
        metadata.pipeline.class_name = "StableDiffusionXLPipeline".into();
        metadata.tokenizer_2 = Some(DiffusionTokenizerMetadata {
            kind: "clip-bpe".into(),
            max_length: Some(4),
            entries: vec!["tokenizer_2/vocab.json".into()],
        });
        let mut config = tiny_runtime_config();
        config.pipeline_class = "StableDiffusionXLPipeline".into();
        config.text_encoder_2 = Some(TextEncoderConfig {
            class_name: "CLIPTextModelWithProjection".into(),
            hidden_size: Some(2),
            intermediate_size: Some(4),
            num_hidden_layers: Some(0),
            num_attention_heads: Some(1),
            max_position_embeddings: Some(4),
            vocab_size: Some(4),
        });
        let tokenizer = ClipTokenizer::from_bytes(
            br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "cat</w>": 3
            }"#,
            b"#version: 0.2\n",
            4,
        )
        .unwrap();
        let text_encoder = ClipTextEncoder {
            token_embedding: CpuTensor {
                shape: vec![4, 2],
                data: vec![0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
            },
            position_embedding: CpuTensor {
                shape: vec![4, 2],
                data: vec![0.0; 8],
            },
            layers: Vec::new(),
            final_layer_norm_weight: CpuTensor {
                shape: vec![2],
                data: vec![1.0, 1.0],
            },
            final_layer_norm_bias: CpuTensor {
                shape: vec![2],
                data: vec![0.0, 0.0],
            },
            text_projection: Some(CpuTensor {
                shape: vec![2, 2],
                data: vec![1.0, 0.0, 0.0, 1.0],
            }),
            hidden_size: 2,
            max_length: 4,
            n_heads: 1,
        };
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pipeline = DiffusionPipeline {
            summary: summarize_hfq(Path::new("/tmp/tiny-sdxl-runtime.hfq"), &metadata),
            metadata,
            config,
            tokenizer: Some(tokenizer.clone()),
            tokenizer_2: Some(tokenizer),
            text_encoder: Some(text_encoder.clone()),
            text_encoder_2: Some(text_encoder),
            native_runtime: Some(NativeDiffusionRuntime {
                kind: DiffusionRuntimeKind::CpuSourceReference,
                noise: Box::new(TestSdxlNoiseBackend {
                    called: called.clone(),
                }),
                encoder: None,
                decoder: Box::new(TestImageDecoder),
            }),
            native_runtime_error: None,
        };
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 7,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: Some(128),
            original_height: Some(256),
            target_width: Some(32),
            target_height: Some(64),
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 8,
            crop_y: 4,
            steps: 2,
            cfg_scale: 7.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: false,
            save_images: false,
        };

        let output = pipeline.generate_batch(request).unwrap();

        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
        assert!(output.images.is_empty());
        assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
    }

    #[test]
    fn diffusion_pipeline_img2img_uses_inpaint_conditioning_for_inpaint_channel_model() {
        let (pipeline, called, dir) = tiny_inpaint_test_pipeline(
            "hipfire-diffusion-inpaint-routing-test",
            Box::new(TestImageDecoder),
        );
        let request = DiffusionImg2ImgRequest {
            batch: DiffusionBatchRequest {
                prompts: vec![DiffusionPrompt {
                    prompt: "a cat".into(),
                    negative_prompt: String::new(),
                    seed: 7,
                    subseed: None,
                }],
                width: 2,
                height: 2,
                original_width: None,
                original_height: None,
                target_width: None,
                target_height: None,
                seed_resize_from_width: None,
                seed_resize_from_height: None,
                crop_x: 0,
                crop_y: 0,
                steps: 2,
                cfg_scale: 7.0,
                scheduler: "Euler".into(),
                subseed_strength: 0.0,
                send_images: true,
                save_images: false,
            },
            init_image: tiny_rgb_image_batch(1, 2, 2),
            mask: Some(tiny_mask_image_batch(1, 2, 2)),
            inpainting_fill: None,
            resize_mode: DiffusionImg2ImgResizeMode::Image,
            denoising_strength: 1.0,
        };

        let output = pipeline.generate_img2img_batch(request).unwrap();

        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["masked"], true);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn diffusion_pipeline_img2img_resizes_init_and_mask_to_request_dimensions() {
        let (pipeline, called, dir) = tiny_inpaint_test_pipeline(
            "hipfire-diffusion-inpaint-resize-routing-test",
            Box::new(TestImageDecoder),
        );
        let request = DiffusionImg2ImgRequest {
            batch: DiffusionBatchRequest {
                prompts: vec![DiffusionPrompt {
                    prompt: "a cat".into(),
                    negative_prompt: String::new(),
                    seed: 7,
                    subseed: None,
                }],
                width: 2,
                height: 2,
                original_width: None,
                original_height: None,
                target_width: None,
                target_height: None,
                seed_resize_from_width: None,
                seed_resize_from_height: None,
                crop_x: 0,
                crop_y: 0,
                steps: 2,
                cfg_scale: 7.0,
                scheduler: "Euler".into(),
                subseed_strength: 0.0,
                send_images: true,
                save_images: false,
            },
            init_image: tiny_rgb_image_batch(1, 1, 1),
            mask: Some(tiny_mask_image_batch(1, 1, 1)),
            inpainting_fill: None,
            resize_mode: DiffusionImg2ImgResizeMode::Image,
            denoising_strength: 1.0,
        };

        let output = pipeline.generate_img2img_batch(request).unwrap();

        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["mode"], "img2img");
        assert_eq!(output.info["masked"], true);
        assert_eq!(output.info["width"], 2);
        assert_eq!(output.info["height"], 2);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn diffusion_pipeline_img2img_latent_resize_mode_resizes_encoded_latents() {
        let (pipeline, called, dir) = tiny_inpaint_test_pipeline(
            "hipfire-diffusion-inpaint-latent-resize-routing-test",
            Box::new(TestImageDecoder),
        );
        let request = DiffusionImg2ImgRequest {
            batch: DiffusionBatchRequest {
                prompts: vec![DiffusionPrompt {
                    prompt: "a cat".into(),
                    negative_prompt: String::new(),
                    seed: 7,
                    subseed: None,
                }],
                width: 2,
                height: 2,
                original_width: None,
                original_height: None,
                target_width: None,
                target_height: None,
                seed_resize_from_width: None,
                seed_resize_from_height: None,
                crop_x: 0,
                crop_y: 0,
                steps: 2,
                cfg_scale: 7.0,
                scheduler: "Euler".into(),
                subseed_strength: 0.0,
                send_images: true,
                save_images: false,
            },
            init_image: tiny_rgb_image_batch(1, 1, 1),
            mask: Some(tiny_mask_image_batch(1, 1, 1)),
            inpainting_fill: None,
            resize_mode: DiffusionImg2ImgResizeMode::Latent,
            denoising_strength: 1.0,
        };

        let output = pipeline.generate_img2img_batch(request).unwrap();

        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["mode"], "img2img");
        assert_eq!(output.info["masked"], true);
        assert_eq!(output.info["resize_mode"], "latent");
        assert_eq!(output.info["latent_resize"], true);
        assert_eq!(output.info["width"], 2);
        assert_eq!(output.info["height"], 2);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resize_latent_batch_nearest_resizes_spatial_axes_per_channel() {
        let latents = LatentBatch {
            batch: 1,
            channels: 2,
            height: 2,
            width: 2,
            data: vec![
                1.0, 2.0, //
                3.0, 4.0, //
                10.0, 20.0, //
                30.0, 40.0,
            ],
        };

        let resized = resize_latent_batch_nearest(&latents, 1, 4).unwrap();

        assert_eq!(resized.batch, 1);
        assert_eq!(resized.channels, 2);
        assert_eq!(resized.height, 1);
        assert_eq!(resized.width, 4);
        assert_eq!(
            resized.data,
            vec![1.0, 1.0, 2.0, 2.0, 10.0, 10.0, 20.0, 20.0]
        );
    }

    #[test]
    fn inpainting_fill_latent_noise_replaces_masked_latents() {
        let mut init = LatentBatch {
            batch: 1,
            channels: 2,
            height: 1,
            width: 2,
            data: vec![10.0, 20.0, 30.0, 40.0],
        };
        let noise = LatentBatch {
            batch: 1,
            channels: 2,
            height: 1,
            width: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };

        let applied = apply_inpainting_fill_to_latents(&mut init, &noise, &[0.0, 1.0], 2).unwrap();

        assert!(applied);
        assert_eq!(init.data, vec![10.0, 2.0, 30.0, 4.0]);
    }

    #[test]
    fn inpainting_fill_latent_nothing_zeros_masked_latents() {
        let mut init = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            data: vec![10.0, 20.0],
        };
        let noise = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            data: vec![1.0, 2.0],
        };

        let applied = apply_inpainting_fill_to_latents(&mut init, &noise, &[1.0, 0.25], 3).unwrap();

        assert!(applied);
        assert_eq!(init.data, vec![0.0, 15.0]);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn generate_img2img_runtime_options_route_vae_mask_boundaries_when_gpu_is_available() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for hybrid img2img generation test: {error}");
            return;
        }
        let (pipeline, called, dir) = tiny_inpaint_test_pipeline(
            "hipfire-diffusion-inpaint-hybrid-routing-test",
            Box::new(SolidTensorImageDecoder),
        );
        let request = DiffusionImg2ImgRequest {
            batch: DiffusionBatchRequest {
                prompts: vec![DiffusionPrompt {
                    prompt: "a cat".into(),
                    negative_prompt: String::new(),
                    seed: 7,
                    subseed: None,
                }],
                width: 2,
                height: 2,
                original_width: None,
                original_height: None,
                target_width: None,
                target_height: None,
                seed_resize_from_width: None,
                seed_resize_from_height: None,
                crop_x: 0,
                crop_y: 0,
                steps: 2,
                cfg_scale: 7.0,
                scheduler: "Euler".into(),
                subseed_strength: 0.0,
                send_images: true,
                save_images: false,
            },
            init_image: tiny_rgb_image_batch(1, 2, 2),
            mask: Some(tiny_mask_image_batch(1, 2, 2)),
            inpainting_fill: None,
            resize_mode: DiffusionImg2ImgResizeMode::Image,
            denoising_strength: 1.0,
        };

        let output = pipeline
            .generate_img2img_batch_with_runtime_options(
                request,
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            )
            .unwrap();

        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["runtime"], "rocm-hybrid-reference");
        assert_eq!(output.info["masked"], true);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        assert_eq!(decoded.get_pixel(0, 0).0, [32, 128, 224]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn diffusion_pipeline_open_hfq_generates_png_with_native_tiny_components() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-complete-pipeline-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-complete.hfq");
        let metadata = tiny_runtime_metadata();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tiny_complete_runtime_tensors(),
        )
        .unwrap();
        let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 9,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 1,
            cfg_scale: 1.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let output = pipeline.generate_batch(request).unwrap();

        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_preflight_reports_clip_token_position_embedding_probe() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for diffusion preflight test: {error}");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-preflight-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-preflight.hfq");
        let metadata = tiny_runtime_metadata();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tiny_complete_runtime_tensors(),
        )
        .unwrap();
        let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 9,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 1,
            cfg_scale: 1.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: false,
            save_images: false,
        };

        let preflight = pipeline
            .preflight_hip_runtime(&request, DiffusionHipRuntimeOptions { device_id: 0 })
            .unwrap();

        assert_eq!(
            preflight.clip_token_position_embedding_kernel_probe.name,
            "diffusion_clip_token_position_embedding_f32"
        );
        assert!(
            preflight
                .clip_token_position_embedding_kernel_probe
                .matched_cpu_reference
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn diffusion_pipeline_runs_quantized_metadata_with_float_tensor_payloads() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-quantized-float-runtime-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-quantized.hfq");
        let mut metadata = tiny_runtime_metadata();
        metadata.quantization.weight_format = "oq4".to_string();
        metadata.quantization.activation_format = "fp16".to_string();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tiny_complete_runtime_tensors(),
        )
        .unwrap();
        let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        assert!(pipeline.native_runtime.is_some());
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 9,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 1,
            cfg_scale: 1.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let output = pipeline.generate_batch(request).unwrap();

        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["weight_format"], "oq4");
        assert_eq!(output.info["runtime"], "cpu-source-reference");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn diffusion_pipeline_runs_with_q8f16_tensor_payloads() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-q8-runtime-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-q8.hfq");
        let mut metadata = tiny_runtime_metadata();
        metadata.quantization.weight_format = "q8".to_string();
        let mut tensors = tiny_complete_runtime_tensors();
        let tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
            .unwrap();
        *tensor = q8f16_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &[0.0; 9]);
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        assert!(pipeline.native_runtime.is_some());
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 9,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 1,
            cfg_scale: 1.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let output = pipeline.generate_batch(request).unwrap();

        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["weight_format"], "q8");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn diffusion_pipeline_runs_with_q4f16_g64_tensor_payloads() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-q4-runtime-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-q4.hfq");
        let mut metadata = tiny_runtime_metadata();
        metadata.quantization.weight_format = "q4f16".to_string();
        let mut tensors = tiny_complete_runtime_tensors();
        let tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
            .unwrap();
        *tensor = q4f16_g64_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &[0.0; 9]);
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        assert!(pipeline.native_runtime.is_some());
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 9,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 1,
            cfg_scale: 1.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let output = pipeline.generate_batch(request).unwrap();

        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["weight_format"], "q4f16");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn diffusion_pipeline_runs_with_q4k_tensor_payloads() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-q4k-runtime-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-q4k.hfq");
        let mut metadata = tiny_runtime_metadata();
        metadata.quantization.weight_format = "q4k".to_string();
        let mut tensors = tiny_complete_runtime_tensors();
        let tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
            .unwrap();
        *tensor = q4k_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &[0; 9]);
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        assert!(pipeline.native_runtime.is_some());
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 9,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 1,
            cfg_scale: 1.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let output = pipeline.generate_batch(request).unwrap();

        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["weight_format"], "q4k");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn diffusion_pipeline_runs_with_hfq4_tensor_payloads() {
        for (label, quant_type, group_size) in [
            ("hfq4g128", QT_DIFFUSION_TENSOR_HFQ4_G128, 128usize),
            ("hfq4g256", QT_DIFFUSION_TENSOR_HFQ4_G256, 256usize),
        ] {
            let dir = std::env::temp_dir().join(format!(
                "hipfire-diffusion-{label}-runtime-test-{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            let hfq_path = dir.join(format!("tiny-{label}.hfq"));
            let mut metadata = tiny_runtime_metadata();
            metadata.quantization.weight_format = label.to_string();
            let mut tensors = tiny_complete_runtime_tensors();
            let tensor = tensors
                .iter_mut()
                .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
                .unwrap();
            *tensor = hfq4_mem_tensor(
                "unet/tensors/conv_in.weight",
                quant_type,
                &[1, 1, 3, 3],
                group_size,
                &[0; 9],
            );
            write_hfqm_package_mem(
                &hfq_path,
                HFQ_ARCH_DIFFUSION,
                &serde_json::to_string(&metadata).unwrap(),
                &tensors,
            )
            .unwrap();
            let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
            assert!(pipeline.native_runtime.is_some());
            let request = DiffusionBatchRequest {
                prompts: vec![DiffusionPrompt {
                    prompt: "a cat".into(),
                    negative_prompt: String::new(),
                    seed: 9,
                    subseed: None,
                }],
                width: 2,
                height: 2,
                original_width: None,
                original_height: None,
                target_width: None,
                target_height: None,
                seed_resize_from_width: None,
                seed_resize_from_height: None,
                crop_x: 0,
                crop_y: 0,
                steps: 1,
                cfg_scale: 1.0,
                scheduler: "Euler".into(),
                subseed_strength: 0.0,
                send_images: true,
                save_images: false,
            };

            let output = pipeline.generate_batch(request).unwrap();

            assert_eq!(output.images.len(), 1);
            assert_eq!(output.info["weight_format"], label);
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&output.images[0])
                .unwrap();
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
            let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
            assert_eq!(decoded.dimensions(), (2, 2));
            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn diffusion_pipeline_runs_with_hfq6_tensor_payloads() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-hfq6-runtime-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-hfq6g256.hfq");
        let mut metadata = tiny_runtime_metadata();
        metadata.quantization.weight_format = "hfq6g256".to_string();
        let mut tensors = tiny_complete_runtime_tensors();
        let tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
            .unwrap();
        *tensor = hfq6_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &[0; 9]);
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        assert!(pipeline.native_runtime.is_some());
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 9,
                subseed: None,
            }],
            width: 2,
            height: 2,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 1,
            cfg_scale: 1.0,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let output = pipeline.generate_batch(request).unwrap();

        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["weight_format"], "hfq6g256");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn diffusion_pipeline_rejects_packed_quant_tensor_payload_without_dequantizer() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-packed-quant-runtime-boundary-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-packed-quant.hfq");
        let mut metadata = tiny_runtime_metadata();
        metadata.quantization.weight_format = "oq4".to_string();
        metadata.quantization.activation_format = "fp16".to_string();
        let mut tensors = tiny_complete_runtime_tensors();
        let tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
            .unwrap();
        tensor.quant_type = 99;
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();

        let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();

        assert!(pipeline.native_runtime.is_none());
        let error = pipeline.native_runtime_error.as_deref().unwrap();
        assert!(error.contains("unsupported quant_type 99"));
        assert!(error.contains("diffusion dequantizer/runtime"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn diffusion_pipeline_open_hfq_generates_img2img_png_with_native_tiny_components() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-complete-img2img-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-complete-img2img.hfq");
        let metadata = tiny_runtime_metadata();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tiny_complete_runtime_tensors(),
        )
        .unwrap();
        let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        assert!(pipeline.supports_img2img());
        let request = DiffusionImg2ImgRequest {
            batch: DiffusionBatchRequest {
                prompts: vec![DiffusionPrompt {
                    prompt: "a cat".into(),
                    negative_prompt: String::new(),
                    seed: 9,
                    subseed: None,
                }],
                width: 2,
                height: 2,
                original_width: None,
                original_height: None,
                target_width: None,
                target_height: None,
                seed_resize_from_width: None,
                seed_resize_from_height: None,
                crop_x: 0,
                crop_y: 0,
                steps: 1,
                cfg_scale: 1.0,
                scheduler: "Euler".into(),
                subseed_strength: 0.0,
                send_images: true,
                save_images: false,
            },
            init_image: RgbImageBatch {
                batch: 1,
                width: 2,
                height: 2,
                data: vec![
                    255, 0, 0, 128, 0, 0, //
                    64, 0, 0, 0, 0, 0,
                ],
            },
            mask: None,
            inpainting_fill: None,
            resize_mode: DiffusionImg2ImgResizeMode::Image,
            denoising_strength: 1.0,
        };

        let output = pipeline.generate_img2img_batch(request).unwrap();

        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
        assert_eq!(output.info["mode"], "img2img");
        assert_eq!(output.info["masked"], false);
        assert_eq!(output.info["denoise_steps"], 1);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn native_vae_decoder_decodes_synthetic_latents_to_rgb8() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-native-vae-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("native-vae.hfq");
        let metadata = minimal_metadata();
        let identity1 = center_identity_conv(1);
        let resnet_prefix = "vae/tensors/decoder.up_blocks.0.resnets.0";
        let tensors = [
            f32_mem_tensor("vae/tensors/post_quant_conv.weight", &[1, 1, 1, 1], &[1.0]),
            f32_mem_tensor("vae/tensors/post_quant_conv.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/decoder.conv_in.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor("vae/tensors/decoder.conv_in.bias", &[1], &[0.0]),
            f32_mem_tensor(&format!("{resnet_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{resnet_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{resnet_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{resnet_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{resnet_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{resnet_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{resnet_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{resnet_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor("vae/tensors/decoder.conv_norm_out.weight", &[1], &[1.0]),
            f32_mem_tensor("vae/tensors/decoder.conv_norm_out.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/decoder.conv_out.weight",
                &[3, 1, 3, 3],
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            f32_mem_tensor("vae/tensors/decoder.conv_out.bias", &[3], &[0.0, 0.0, 0.0]),
        ];
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let config = VaeConfig {
            class_name: "AutoencoderKL".into(),
            latent_channels: Some(1),
            z_dim: None,
            scaling_factor: Some(1.0),
            latents_mean: Vec::new(),
            latents_std: Vec::new(),
            block_out_channels: vec![1],
            down_block_types: Vec::new(),
            up_block_types: vec!["UpDecoderBlock2D".into()],
            norm_num_groups: Some(1),
            norm_eps: Some(1e-6),
        };
        let decoder = NativeVaeDecoder::from_hfq(&hfq, &config).unwrap();
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 2,
            width: 2,
            data: vec![0.0, 0.5, -0.5, 1.0],
        };
        let decoded = decoder.decode_latents(&latents).unwrap();
        assert_eq!(decoded.shape, vec![1, 3, 2, 2]);
        assert!(decoded.data.iter().all(|value| value.is_finite()));
        let image = decoder.decode_to_rgb8(&latents).unwrap();
        assert_eq!(image.batch, 1);
        assert_eq!(image.width, 2);
        assert_eq!(image.height, 2);
        assert_eq!(image.data.len(), 12);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for VAE decoder routing test: {error}");
            } else {
                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip_context_decoded = decoder
                    .decode_latents_with_runtime_context(&latents, &mut runtime_context)
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert_eq!(hip_context_decoded.shape, decoded.shape);
                assert!(f32_slices_close(
                    &hip_context_decoded.data,
                    &decoded.data,
                    1e-5
                ));
                let hip_decoded = decoder
                    .decode_latents_with_runtime_options(
                        &latents,
                        DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                    )
                    .unwrap();
                assert_eq!(hip_decoded.shape, decoded.shape);
                assert!(f32_slices_close(&hip_decoded.data, &decoded.data, 1e-5));
                let (hip_image, runtime_kind) = decode_to_rgb8_with_runtime_options(
                    &decoder,
                    &latents,
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                )
                .unwrap();
                assert_eq!(runtime_kind, DiffusionRuntimeKind::RocmHybridReference);
                assert_eq!(hip_image, image);
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn native_vae_encoder_encodes_synthetic_image_to_latents() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-native-vae-encoder-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("native-vae-encoder.hfq");
        let metadata = minimal_metadata();
        let prefix = "vae/tensors/encoder.down_blocks.0.resnets.0";
        let identity1 = center_identity_conv(1);
        let mut conv_in = vec![0.0; 1 * 3 * 3 * 3];
        conv_in[1 * 3 + 1] = 1.0;
        let mut conv_out = vec![0.0; 2 * 1 * 3 * 3];
        conv_out[1 * 3 + 1] = 1.0;
        let tensors = vec![
            f32_mem_tensor(
                "vae/tensors/encoder.conv_in.weight",
                &[1, 3, 3, 3],
                &conv_in,
            ),
            f32_mem_tensor("vae/tensors/encoder.conv_in.bias", &[1], &[0.0]),
            f32_mem_tensor(&format!("{prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{prefix}.conv1.weight"), &[1, 1, 3, 3], &identity1),
            f32_mem_tensor(&format!("{prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{prefix}.conv2.weight"), &[1, 1, 3, 3], &[0.0; 9]),
            f32_mem_tensor(&format!("{prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor("vae/tensors/encoder.conv_norm_out.weight", &[1], &[1.0]),
            f32_mem_tensor("vae/tensors/encoder.conv_norm_out.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/encoder.conv_out.weight",
                &[2, 1, 3, 3],
                &conv_out,
            ),
            f32_mem_tensor("vae/tensors/encoder.conv_out.bias", &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                "vae/tensors/quant_conv.weight",
                &[2, 2, 1, 1],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor("vae/tensors/quant_conv.bias", &[2], &[0.0, 0.0]),
        ];
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let config = VaeConfig {
            class_name: "AutoencoderKL".into(),
            latent_channels: Some(1),
            z_dim: None,
            scaling_factor: Some(0.5),
            latents_mean: Vec::new(),
            latents_std: Vec::new(),
            block_out_channels: vec![1],
            down_block_types: vec!["DownEncoderBlock2D".into()],
            up_block_types: Vec::new(),
            norm_num_groups: Some(1),
            norm_eps: Some(1e-6),
        };
        let encoder = NativeVaeEncoder::from_hfq(&hfq, &config).unwrap();
        let image = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 2,
            data: vec![255; 12],
        };

        let latents = encoder.encode_to_latents(&image).unwrap();

        assert_eq!(latents.batch, 1);
        assert_eq!(latents.channels, 1);
        assert_eq!(latents.height, 2);
        assert_eq!(latents.width, 2);
        assert!(latents.data.iter().all(|value| value.is_finite()));

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for VAE encoder routing test: {error}");
            } else {
                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip_context_latents = encoder
                    .encode_to_latents_with_runtime_context(&image, &mut runtime_context)
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert_eq!(hip_context_latents.batch, latents.batch);
                assert_eq!(hip_context_latents.channels, latents.channels);
                assert_eq!(hip_context_latents.height, latents.height);
                assert_eq!(hip_context_latents.width, latents.width);
                assert!(f32_slices_close(
                    &hip_context_latents.data,
                    &latents.data,
                    1e-5
                ));
                let hip_latents = encoder
                    .encode_to_latents_with_runtime_options(
                        &image,
                        DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                    )
                    .unwrap();
                assert_eq!(hip_latents.batch, latents.batch);
                assert_eq!(hip_latents.channels, latents.channels);
                assert_eq!(hip_latents.height, latents.height);
                assert_eq!(hip_latents.width, latents.width);
                assert!(f32_slices_close(&hip_latents.data, &latents.data, 1e-5));
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tiny_sd_native_vae_decoder_loads_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let hfq = HfqFile::open_index_only(&path).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        let decoder = NativeVaeDecoder::from_hfq(&hfq, &config.vae).unwrap();

        assert_eq!(decoder.conv_in.weight.shape, vec![512, 4, 3, 3]);
        assert_eq!(decoder.up_blocks.len(), 4);
        assert!(decoder.up_blocks[0].upsampler.is_some());
        assert!(decoder.up_blocks[1].upsampler.is_some());
        assert!(decoder.up_blocks[2].upsampler.is_some());
        assert!(decoder.up_blocks[3].upsampler.is_none());
        assert_eq!(decoder.conv_out.weight.shape, vec![3, 128, 3, 3]);
    }

    #[test]
    fn tiny_sd_native_vae_encoder_loads_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let hfq = HfqFile::open_index_only(&path).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        let encoder = NativeVaeEncoder::from_hfq(&hfq, &config.vae).unwrap();

        assert_eq!(encoder.conv_in.weight.shape, vec![128, 3, 3, 3]);
        assert_eq!(encoder.down_blocks.len(), 4);
        assert!(encoder.down_blocks[0].downsampler.is_some());
        assert!(encoder.down_blocks[1].downsampler.is_some());
        assert!(encoder.down_blocks[2].downsampler.is_some());
        assert!(encoder.down_blocks[3].downsampler.is_none());
        assert_eq!(encoder.conv_out.weight.shape, vec![8, 512, 3, 3]);
        assert!(encoder.quant_conv.is_some());
    }

    #[test]
    fn tiny_sd_unet_resnet_block_loads_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let hfq = HfqFile::open_index_only(&path).unwrap();
        let block =
            UnetResnetBlock2D::from_hfq(&hfq, "unet/tensors/down_blocks.0.resnets.0", 32, 1e-5)
                .unwrap();

        assert_eq!(block.conv1.weight.shape, vec![320, 320, 3, 3]);
        assert_eq!(block.time_emb_proj_weight.shape, vec![320, 1280]);
        assert!(block.shortcut.is_none());
    }

    #[test]
    fn unet_time_embedding_loads_from_hfq() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-time-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("time.hfq");
        let metadata = minimal_metadata();
        let identity = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let tensors = [
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_1.weight",
                &[4, 4],
                &identity,
            ),
            f32_mem_tensor("unet/tensors/time_embedding.linear_1.bias", &[4], &[0.0; 4]),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_2.weight",
                &[4, 4],
                &identity,
            ),
            f32_mem_tensor("unet/tensors/time_embedding.linear_2.bias", &[4], &[0.0; 4]),
        ];
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let time_embedding = UnetTimeEmbedding::from_hfq(&hfq).unwrap();
        let output = time_embedding.forward(&[0.0, 1.0], true, 0.0).unwrap();
        assert_eq!(output.shape, vec![2, 4]);
        assert!(output.data.iter().all(|value| value.is_finite()));
        assert!(output.data[0] > 0.73 && output.data[2] == 0.0);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!(
                    "skip: ROCm GPU unavailable for UNet time embedding routing test: {error}"
                );
            } else {
                let hip = time_embedding
                    .forward_with_runtime_options(
                        &[0.0, 1.0],
                        true,
                        0.0,
                        DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                    )
                    .unwrap();
                assert_eq!(hip.shape, output.shape);
                assert!(f32_slices_close(&hip.data, &output.data, 1e-5));

                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip_context = time_embedding
                    .forward_with_runtime_context(&[0.0, 1.0], true, 0.0, &mut runtime_context)
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert_eq!(hip_context.shape, output.shape);
                assert!(f32_slices_close(&hip_context.data, &output.data, 1e-5));
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn attention_layer_runs_biasless_self_and_cross_attention() {
        let identity = CpuTensor {
            shape: vec![2, 2],
            data: vec![1.0, 0.0, 0.0, 1.0],
        };
        let attention = AttentionLayer {
            to_q_weight: identity.clone(),
            to_q_bias: None,
            to_k_weight: identity.clone(),
            to_k_bias: None,
            to_v_weight: identity.clone(),
            to_v_bias: None,
            to_out_weight: identity,
            to_out_bias: None,
            heads: 1,
        };
        let hidden = CpuTensor {
            shape: vec![1, 2, 2],
            data: vec![1.0, 0.0, 0.0, 1.0],
        };
        let self_out = attention.forward(&hidden, None).unwrap();
        assert_eq!(self_out.shape, hidden.shape);
        assert!(self_out.data.iter().all(|value| value.is_finite()));

        let encoder = CpuTensor {
            shape: vec![1, 1, 2],
            data: vec![0.25, 0.75],
        };
        let cross_out = attention.forward(&hidden, Some(&encoder)).unwrap();
        assert_eq!(cross_out.shape, hidden.shape);
        assert_eq!(cross_out.data, vec![0.25, 0.75, 0.25, 0.75]);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for attention routing test: {error}");
            } else {
                let runtime_options = DiffusionGenerationRuntimeOptions::rocm_hybrid(0);
                let hip_self = attention
                    .forward_with_runtime_options(&hidden, None, runtime_options)
                    .unwrap();
                assert_eq!(hip_self.shape, self_out.shape);
                assert!(f32_slices_close(&hip_self.data, &self_out.data, 1e-5));

                let hip_cross = attention
                    .forward_with_runtime_options(&hidden, Some(&encoder), runtime_options)
                    .unwrap();
                assert_eq!(hip_cross.shape, cross_out.shape);
                assert!(f32_slices_close(&hip_cross.data, &cross_out.data, 1e-5));

                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip_context_self = attention
                    .forward_with_runtime_context(&hidden, None, &mut runtime_context)
                    .unwrap();
                let hip_context_cross = attention
                    .forward_with_runtime_context(&hidden, Some(&encoder), &mut runtime_context)
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert_eq!(hip_context_self.shape, self_out.shape);
                assert!(f32_slices_close(
                    &hip_context_self.data,
                    &self_out.data,
                    1e-5
                ));
                assert_eq!(hip_context_cross.shape, cross_out.shape);
                assert!(f32_slices_close(
                    &hip_context_cross.data,
                    &cross_out.data,
                    1e-5
                ));
            }
        }
    }

    #[test]
    fn transformer_block_loads_from_hfq_and_preserves_residual_with_zero_weights() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-transformer-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("transformer.hfq");
        let metadata = minimal_metadata();
        let prefix = "unet/tensors/down_blocks.0.attentions.0.transformer_blocks.0";
        let mut tensors = vec![
            f32_mem_tensor(&format!("{prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(&format!("{prefix}.norm3.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.norm3.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{prefix}.ff.net.0.proj.weight"),
                &[4, 2],
                &[0.0; 8],
            ),
            f32_mem_tensor(&format!("{prefix}.ff.net.0.proj.bias"), &[4], &[0.0; 4]),
            f32_mem_tensor(&format!("{prefix}.ff.net.2.weight"), &[2, 2], &[0.0; 4]),
            f32_mem_tensor(&format!("{prefix}.ff.net.2.bias"), &[2], &[0.0; 2]),
        ];
        push_zero_attention_tensors(&mut tensors, &format!("{prefix}.attn1"), 2, 2);
        push_zero_attention_tensors(&mut tensors, &format!("{prefix}.attn2"), 2, 3);
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let block = BasicTransformerBlock::from_hfq(&hfq, prefix, 1).unwrap();
        let hidden = CpuTensor {
            shape: vec![1, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let encoder = CpuTensor {
            shape: vec![1, 1, 3],
            data: vec![0.5, 0.25, -0.5],
        };
        let output = block.forward(&hidden, &encoder).unwrap();
        assert_eq!(output.shape, hidden.shape);
        assert_eq!(output.data, hidden.data);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for transformer block routing test: {error}");
            } else {
                let hip = block
                    .forward_with_runtime_options(
                        &hidden,
                        &encoder,
                        DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                    )
                    .unwrap();
                assert_eq!(hip.shape, output.shape);
                assert!(f32_slices_close(&hip.data, &output.data, 1e-5));

                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip_context = block
                    .forward_with_runtime_context(&hidden, &encoder, &mut runtime_context)
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert_eq!(hip_context.shape, output.shape);
                assert!(f32_slices_close(&hip_context.data, &output.data, 1e-5));
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn transformer2d_model_loads_from_hfq_and_preserves_residual_with_zero_weights() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-transformer2d-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("transformer2d.hfq");
        let metadata = minimal_metadata();
        let prefix = "unet/tensors/down_blocks.0.attentions.0";
        let block = format!("{prefix}.transformer_blocks.0");
        let mut tensors = vec![
            f32_mem_tensor(&format!("{prefix}.norm.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.norm.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{prefix}.proj_in.weight"),
                &[2, 2, 1, 1],
                &[0.0; 4],
            ),
            f32_mem_tensor(&format!("{prefix}.proj_in.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{prefix}.proj_out.weight"),
                &[2, 2, 1, 1],
                &[0.0; 4],
            ),
            f32_mem_tensor(&format!("{prefix}.proj_out.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(&format!("{block}.norm1.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{block}.norm1.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(&format!("{block}.norm2.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{block}.norm2.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(&format!("{block}.norm3.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{block}.norm3.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(&format!("{block}.ff.net.0.proj.weight"), &[4, 2], &[0.0; 8]),
            f32_mem_tensor(&format!("{block}.ff.net.0.proj.bias"), &[4], &[0.0; 4]),
            f32_mem_tensor(&format!("{block}.ff.net.2.weight"), &[2, 2], &[0.0; 4]),
            f32_mem_tensor(&format!("{block}.ff.net.2.bias"), &[2], &[0.0; 2]),
        ];
        push_zero_attention_tensors(&mut tensors, &format!("{block}.attn1"), 2, 2);
        push_zero_attention_tensors(&mut tensors, &format!("{block}.attn2"), 2, 3);
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let model = Transformer2DModel::from_hfq(&hfq, prefix, 1, 1, 1e-5).unwrap();
        let input = CpuTensor {
            shape: vec![1, 2, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
        };
        let encoder = CpuTensor {
            shape: vec![1, 1, 3],
            data: vec![0.5, 0.25, -0.5],
        };
        let output = model.forward(&input, &encoder).unwrap();
        assert_eq!(output.shape, input.shape);
        assert_eq!(output.data, input.data);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for transformer2d routing test: {error}");
            } else {
                let hip = model
                    .forward_with_runtime_options(
                        &input,
                        &encoder,
                        DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                    )
                    .unwrap();
                assert_eq!(hip.shape, output.shape);
                assert!(f32_slices_close(&hip.data, &output.data, 1e-5));

                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip_context = model
                    .forward_with_runtime_context(&input, &encoder, &mut runtime_context)
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert_eq!(hip_context.shape, output.shape);
                assert!(f32_slices_close(&hip_context.data, &output.data, 1e-5));
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthetic_clip_text_encoder_forward_is_finite() {
        let hidden = 12usize;
        let encoder = ClipTextEncoder {
            token_embedding: CpuTensor {
                shape: vec![3, hidden],
                data: (0..3 * hidden).map(|idx| idx as f32 * 0.01).collect(),
            },
            position_embedding: CpuTensor {
                shape: vec![2, hidden],
                data: vec![0.0; 2 * hidden],
            },
            layers: vec![zero_clip_layer(hidden)],
            final_layer_norm_weight: CpuTensor {
                shape: vec![hidden],
                data: vec![1.0; hidden],
            },
            final_layer_norm_bias: CpuTensor {
                shape: vec![hidden],
                data: vec![0.0; hidden],
            },
            text_projection: None,
            hidden_size: hidden,
            max_length: 2,
            n_heads: 3,
        };
        let encoded = encoder.encode_tokens(&[0, 1]).unwrap();

        assert_eq!(encoded.shape, vec![2, hidden]);
        assert!(encoded.data.iter().all(|value| value.is_finite()));
        assert!(encoded.data.iter().any(|value| value.abs() > 0.001));

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for CLIP encoder routing test: {error}");
            } else {
                let hip = encoder
                    .encode_tokens_with_runtime_options(
                        &[0, 1],
                        DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                    )
                    .unwrap();
                assert_eq!(hip.shape, encoded.shape);
                assert!(f32_slices_close(&hip.data, &encoded.data, 1e-5));
            }
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_clip_text_encoder_runtime_context_reuses_single_gpu() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for CLIP context reuse test: {error}");
            return;
        }
        let hidden = 12usize;
        let encoder = ClipTextEncoder {
            token_embedding: CpuTensor {
                shape: vec![3, hidden],
                data: (0..3 * hidden).map(|idx| idx as f32 * 0.01).collect(),
            },
            position_embedding: CpuTensor {
                shape: vec![2, hidden],
                data: vec![0.0; 2 * hidden],
            },
            layers: vec![zero_clip_layer(hidden)],
            final_layer_norm_weight: CpuTensor {
                shape: vec![hidden],
                data: vec![1.0; hidden],
            },
            final_layer_norm_bias: CpuTensor {
                shape: vec![hidden],
                data: vec![0.0; hidden],
            },
            text_projection: Some(CpuTensor {
                shape: vec![hidden, hidden],
                data: (0..hidden * hidden)
                    .map(|idx| {
                        let row = idx / hidden;
                        let col = idx % hidden;
                        if row == col {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            }),
            hidden_size: hidden,
            max_length: 2,
            n_heads: 3,
        };
        let (cpu_hidden, cpu_pooled) = encoder.encode_tokens_with_pooled(&[0, 1], 1).unwrap();
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(
            DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
        );
        let (hip_hidden, hip_pooled) = encoder
            .encode_tokens_with_pooled_and_runtime_context(&[0, 1], 1, &mut runtime_context)
            .unwrap();

        assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
        assert_eq!(hip_hidden.shape, cpu_hidden.shape);
        assert!(f32_slices_close(&hip_hidden.data, &cpu_hidden.data, 1e-5));
        assert!(f32_slices_close(
            &hip_pooled.unwrap(),
            &cpu_pooled.unwrap(),
            1e-5
        ));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hip_clip_token_position_embeddings_match_cpu_reference() {
        let token_embedding = CpuTensor {
            shape: vec![4, 5],
            data: (0..20).map(|idx| idx as f32 / 13.0 - 0.6).collect(),
        };
        let position_embedding = CpuTensor {
            shape: vec![3, 5],
            data: (0..15).map(|idx| (idx as f32 % 7.0 - 3.0) / 11.0).collect(),
        };
        let tokens = [3, 0, 2];
        let cpu =
            clip_token_position_embeddings(&token_embedding, &position_embedding, &tokens).unwrap();
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for CLIP embedding routing test: {error}");
                return;
            }
        };
        let hip = clip_token_position_embeddings_hip_on_gpu(
            &mut gpu,
            &token_embedding,
            &position_embedding,
            &tokens,
        )
        .unwrap();

        assert_eq!(hip.shape, cpu.shape);
        assert!(f32_slices_close(&hip.data, &cpu.data, 1e-6));
    }

    #[test]
    fn clip_text_encoder_pools_eos_hidden_state_and_applies_projection() {
        let encoder = ClipTextEncoder {
            token_embedding: CpuTensor {
                shape: vec![3, 2],
                data: vec![0.0, 0.0, 1.0, -1.0, 0.5, 0.5],
            },
            position_embedding: CpuTensor {
                shape: vec![3, 2],
                data: vec![0.0; 6],
            },
            layers: Vec::new(),
            final_layer_norm_weight: CpuTensor {
                shape: vec![2],
                data: vec![1.0, 1.0],
            },
            final_layer_norm_bias: CpuTensor {
                shape: vec![2],
                data: vec![0.0, 0.0],
            },
            text_projection: Some(CpuTensor {
                shape: vec![2, 2],
                data: vec![2.0, 0.0, 0.0, 3.0],
            }),
            hidden_size: 2,
            max_length: 3,
            n_heads: 1,
        };

        let (hidden, pooled) = encoder.encode_tokens_with_pooled(&[0, 1, 2], 1).unwrap();
        let pooled = pooled.unwrap();

        assert_eq!(hidden.shape, vec![3, 2]);
        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 2.0).abs() < 1e-4);
        assert!((pooled[1] + 3.0).abs() < 1e-4);

        #[cfg(feature = "rocm")]
        {
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for CLIP pooled routing test: {error}");
            } else {
                let (hip_hidden, hip_pooled) = encoder
                    .encode_tokens_with_pooled_and_runtime_options(
                        &[0, 1, 2],
                        1,
                        DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                    )
                    .unwrap();
                let hip_pooled = hip_pooled.unwrap();
                assert_eq!(hip_hidden.shape, hidden.shape);
                assert!(f32_slices_close(&hip_hidden.data, &hidden.data, 1e-5));
                assert!(f32_slices_close(&hip_pooled, &pooled, 1e-5));
            }
        }
    }

    #[test]
    #[ignore = "naive CPU CLIP forward over tiny-sd is a correctness smoke, not a normal unit test"]
    fn tiny_sd_clip_text_encoder_loads_and_encodes_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let hfq = HfqFile::open_index_only(&path).unwrap();
        let tokenizer = ClipTokenizer::from_hfq_file(&hfq).unwrap();
        let text_encoder = ClipTextEncoder::from_hfq_file(&hfq).unwrap();
        let tokens = tokenizer.encode_padded("a red robot");
        let encoded = text_encoder.encode_tokens(&tokens).unwrap();

        assert_eq!(encoded.shape, vec![77, 768]);
        assert!(encoded.data.iter().all(|value| value.is_finite()));
        assert!(encoded.data.iter().any(|value| value.abs() > 0.001));
    }

    #[test]
    #[ignore = "real Tiny-SD end-to-end generation is an admission smoke; the naive CPU runtime is slow"]
    fn tiny_sd_pipeline_generates_one_step_png_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let pipeline = DiffusionPipeline::open_hfq(&path).unwrap();
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a red robot".into(),
                negative_prompt: String::new(),
                seed: 123,
                subseed: None,
            }],
            width: 64,
            height: 64,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 1,
            cfg_scale: 1.0,
            scheduler: "DPM++ 2M".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let output = pipeline.generate_batch(request).unwrap();

        assert_eq!(output.images.len(), 1);
        assert!(output.images[0].starts_with("iVBORw0KGgo"));
        assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
    }

    #[test]
    #[ignore = "real Tiny-SD img2img is an admission smoke; run in release mode under an external timeout"]
    fn tiny_sd_pipeline_generates_one_step_img2img_png_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let pipeline = DiffusionPipeline::open_hfq(&path).unwrap();
        if !pipeline.supports_img2img() {
            eprintln!("skip: {} has no native VAE encoder", path.display());
            return;
        }
        let request = DiffusionImg2ImgRequest {
            batch: DiffusionBatchRequest {
                prompts: vec![DiffusionPrompt {
                    prompt: "a red robot".into(),
                    negative_prompt: String::new(),
                    seed: 123,
                    subseed: None,
                }],
                width: 64,
                height: 64,
                original_width: None,
                original_height: None,
                target_width: None,
                target_height: None,
                seed_resize_from_width: None,
                seed_resize_from_height: None,
                crop_x: 0,
                crop_y: 0,
                steps: 1,
                cfg_scale: 1.0,
                scheduler: "DPM++ 2M".into(),
                subseed_strength: 0.0,
                send_images: true,
                save_images: false,
            },
            init_image: tiny_rgb_image_batch(1, 64, 64),
            mask: Some(tiny_mask_image_batch(1, 64, 64)),
            inpainting_fill: None,
            resize_mode: DiffusionImg2ImgResizeMode::Image,
            denoising_strength: 1.0,
        };

        let output = pipeline.generate_img2img_batch(request).unwrap();

        assert_eq!(output.images.len(), 1);
        assert!(output.images[0].starts_with("iVBORw0KGgo"));
        assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
        assert_eq!(output.info["mode"], "img2img");
        assert_eq!(output.info["masked"], true);
    }

    #[test]
    #[ignore = "diagnostic real-model phase timing; run with --nocapture under an external timeout"]
    fn tiny_sd_pipeline_phase_timings_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let request = DiffusionBatchRequest {
            prompts: vec![DiffusionPrompt {
                prompt: "a red robot".into(),
                negative_prompt: String::new(),
                seed: 123,
                subseed: None,
            }],
            width: 64,
            height: 64,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: 1,
            cfg_scale: 1.0,
            scheduler: "DPM++ 2M".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let total = std::time::Instant::now();
        let phase = std::time::Instant::now();
        let pipeline = DiffusionPipeline::open_hfq(&path).unwrap();
        eprintln!("phase open_hfq {:?}", phase.elapsed());

        let phase = std::time::Instant::now();
        let plan = pipeline.prepare_run_plan(&request).unwrap();
        eprintln!("phase prepare_run_plan {:?}", phase.elapsed());

        let runtime = pipeline.native_runtime.as_ref().unwrap();
        let positive = plan.conditioning.prompt_embeddings.as_ref().unwrap();
        let negative = plan.conditioning.negative_embeddings.as_ref().unwrap();
        let mut runtime_context =
            DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());
        let phase = std::time::Instant::now();
        let latents = runtime
            .noise
            .denoise_latents_with_runtime_context(
                plan.latents,
                &plan.schedule,
                request.cfg_scale,
                positive,
                negative,
                None,
                None,
                None,
                None,
                &mut runtime_context,
                None,
            )
            .unwrap();
        eprintln!("phase denoise {:?}", phase.elapsed());

        let hfq = HfqFile::open_index_only(&path).unwrap();
        let decoder = NativeVaeDecoder::from_hfq(&hfq, &pipeline.config.vae).unwrap();
        let phase = std::time::Instant::now();
        let decoded = decoder.decode_latents(&latents.latents).unwrap();
        eprintln!("phase decode_latents {:?}", phase.elapsed());

        let phase = std::time::Instant::now();
        let rgb = rgb_tensor_to_u8(&decoded).unwrap();
        eprintln!("phase rgb_tensor_to_u8 {:?}", phase.elapsed());

        let phase = std::time::Instant::now();
        let images = encode_rgb_batch_png_base64(&rgb).unwrap();
        eprintln!("phase png_base64 {:?}", phase.elapsed());
        eprintln!("phase total {:?}", total.elapsed());

        assert_eq!(images.len(), 1);
        assert!(images[0].starts_with("iVBORw0KGgo"));
    }

    fn zero_clip_layer(hidden: usize) -> ClipEncoderLayer {
        let square = CpuTensor {
            shape: vec![hidden, hidden],
            data: vec![0.0; hidden * hidden],
        };
        let bias = CpuTensor {
            shape: vec![hidden],
            data: vec![0.0; hidden],
        };
        let norm_weight = CpuTensor {
            shape: vec![hidden],
            data: vec![1.0; hidden],
        };
        let norm_bias = bias.clone();
        ClipEncoderLayer {
            q_proj_weight: square.clone(),
            q_proj_bias: bias.clone(),
            k_proj_weight: square.clone(),
            k_proj_bias: bias.clone(),
            v_proj_weight: square.clone(),
            v_proj_bias: bias.clone(),
            out_proj_weight: square.clone(),
            out_proj_bias: bias.clone(),
            layer_norm1_weight: norm_weight.clone(),
            layer_norm1_bias: norm_bias.clone(),
            fc1_weight: square.clone(),
            fc1_bias: bias.clone(),
            fc2_weight: square,
            fc2_bias: bias,
            layer_norm2_weight: norm_weight,
            layer_norm2_bias: norm_bias,
        }
    }

    fn tiny_rgb_image_batch(batch: usize, width: usize, height: usize) -> RgbImageBatch {
        let mut data = Vec::with_capacity(batch * width * height * 3);
        for batch_idx in 0..batch {
            for y in 0..height {
                for x in 0..width {
                    let red = ((x * 255) / width.max(1)) as u8;
                    let green = ((y * 255) / height.max(1)) as u8;
                    let blue = if batch_idx % 2 == 0 { 32 } else { 96 };
                    data.extend_from_slice(&[red, green, blue]);
                }
            }
        }
        RgbImageBatch {
            batch,
            width,
            height,
            data,
        }
    }

    fn tiny_mask_image_batch(batch: usize, width: usize, height: usize) -> RgbImageBatch {
        let mut data = Vec::with_capacity(batch * width * height * 3);
        for _ in 0..batch {
            for y in 0..height {
                for x in 0..width {
                    let value = if (x + y) % 2 == 0 { 255 } else { 0 };
                    data.extend_from_slice(&[value, value, value]);
                }
            }
        }
        RgbImageBatch {
            batch,
            width,
            height,
            data,
        }
    }

    fn minimal_metadata() -> DiffusionHfqMetadata {
        let mut components = BTreeMap::new();
        components.insert(
            "unet".to_string(),
            DiffusionComponentMetadata {
                class_name: Some("UNet2DConditionModel".into()),
                config_entry: Some("unet/config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        DiffusionHfqMetadata {
            artifact_kind: DIFFUSION_ARTIFACT_KIND.to_string(),
            schema_version: DIFFUSION_SCHEMA_VERSION,
            pipeline: DiffusionPipelineMetadata {
                class_name: "StableDiffusionPipeline".into(),
                source: "/tmp/model".into(),
                model_name: "tiny-sd".into(),
                latent_channels: Some(4),
                latent_height: Some(64),
                latent_width: Some(64),
                supported_widths: vec![512],
                supported_heights: vec![512],
            },
            tokenizer: DiffusionTokenizerMetadata::default(),
            tokenizer_2: None,
            batch: DiffusionBatchMetadata {
                max_batch: 2,
                batched_runtime: true,
            },
            quantization: DiffusionQuantizationMetadata::default(),
            components,
        }
    }

    fn tiny_sd_scheduler_config_for_tests() -> SchedulerConfig {
        SchedulerConfig {
            class_name: "DPMSolverMultistepScheduler".into(),
            beta_start: Some(0.00085),
            beta_end: Some(0.012),
            beta_schedule: Some("scaled_linear".into()),
            num_train_timesteps: Some(1000),
            prediction_type: Some("epsilon".into()),
            algorithm_type: Some("dpmsolver++".into()),
            solver_order: Some(2),
            solver_type: Some("midpoint".into()),
            lower_order_final: Some(true),
            thresholding: Some(false),
            timestep_spacing: Some("linspace".into()),
            steps_offset: Some(1),
            use_karras_sigmas: Some(false),
            set_alpha_to_one: None,
            ..SchedulerConfig::default()
        }
    }

    fn tiny_runtime_metadata() -> DiffusionHfqMetadata {
        let mut metadata = minimal_metadata();
        metadata.pipeline.model_name = "tiny-runtime".into();
        metadata.pipeline.latent_channels = Some(1);
        metadata.pipeline.latent_height = Some(2);
        metadata.pipeline.latent_width = Some(2);
        metadata.pipeline.supported_widths = vec![2];
        metadata.pipeline.supported_heights = vec![2];
        metadata.batch.max_batch = 4;
        metadata.components.insert(
            "text_encoder".into(),
            DiffusionComponentMetadata {
                class_name: Some("CLIPTextModel".into()),
                config_entry: Some("text_encoder/config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        metadata.components.insert(
            "vae".into(),
            DiffusionComponentMetadata {
                class_name: Some("AutoencoderKL".into()),
                config_entry: Some("vae/config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        metadata.components.insert(
            "scheduler".into(),
            DiffusionComponentMetadata {
                class_name: Some("EulerDiscreteScheduler".into()),
                config_entry: Some("scheduler/scheduler_config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        metadata
    }

    fn tiny_runtime_config() -> StableDiffusionConfig {
        StableDiffusionConfig {
            pipeline_class: "StableDiffusionPipeline".into(),
            text_encoder: TextEncoderConfig {
                class_name: "CLIPTextModel".into(),
                hidden_size: Some(2),
                intermediate_size: Some(4),
                num_hidden_layers: Some(0),
                num_attention_heads: Some(1),
                max_position_embeddings: Some(4),
                vocab_size: Some(4),
            },
            text_encoder_2: None,
            unet: UnetConfig {
                class_name: "UNet2DConditionModel".into(),
                sample_size: Some(2),
                in_channels: Some(1),
                out_channels: Some(1),
                cross_attention_dim: Some(2),
                attention_head_dim: vec![1],
                block_out_channels: vec![1],
                down_block_types: vec!["DownBlock2D".into()],
                up_block_types: vec!["UpBlock2D".into()],
                layers_per_block: Some(1),
                norm_num_groups: Some(1),
                norm_eps: Some(1e-5),
                center_input_sample: false,
                flip_sin_to_cos: true,
                freq_shift: 0.0,
                addition_embed_type: None,
                addition_time_embed_dim: None,
                projection_class_embeddings_input_dim: None,
            },
            vae: VaeConfig {
                class_name: "AutoencoderKL".into(),
                latent_channels: Some(1),
                z_dim: None,
                scaling_factor: Some(1.0),
                latents_mean: Vec::new(),
                latents_std: Vec::new(),
                block_out_channels: vec![1],
                down_block_types: vec!["DownEncoderBlock2D".into()],
                up_block_types: vec!["UpDecoderBlock2D".into()],
                norm_num_groups: Some(1),
                norm_eps: Some(1e-6),
            },
            scheduler: SchedulerConfig::default(),
            latent_channels: 1,
            latent_height: Some(2),
            latent_width: Some(2),
            vae_scale_factor: 1,
        }
    }

    fn tiny_complete_runtime_tensors() -> Vec<HfqMemTensor> {
        let identity1 = center_identity_conv(1);
        let mut vae_encoder_conv_in = vec![0.0; 1 * 3 * 3 * 3];
        vae_encoder_conv_in[1 * 3 + 1] = 1.0;
        let mut vae_encoder_conv_out = vec![0.0; 2 * 1 * 3 * 3];
        vae_encoder_conv_out[1 * 3 + 1] = 1.0;
        let down_prefix = "unet/tensors/down_blocks.0.resnets.0";
        let mid0_prefix = "unet/tensors/mid_block.resnets.0";
        let mid1_prefix = "unet/tensors/mid_block.resnets.1";
        let up_prefix = "unet/tensors/up_blocks.0.resnets.0";
        let vae_resnet_prefix = "vae/tensors/decoder.up_blocks.0.resnets.0";
        let vae_encoder_resnet_prefix = "vae/tensors/encoder.down_blocks.0.resnets.0";
        vec![
            bytes_mem_tensor(
                "text_encoder/config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"CLIPTextModel","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"max_position_embeddings":77,"vocab_size":4}"#,
            ),
            bytes_mem_tensor(
                "unet/config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"UNet2DConditionModel","sample_size":2,"in_channels":1,"out_channels":1,"cross_attention_dim":2,"attention_head_dim":[1],"block_out_channels":[1],"down_block_types":["DownBlock2D"],"up_block_types":["UpBlock2D"],"layers_per_block":1,"norm_num_groups":1,"norm_eps":0.00001,"flip_sin_to_cos":true,"freq_shift":0.0}"#,
            ),
            bytes_mem_tensor(
                "vae/config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"AutoencoderKL","latent_channels":1,"scaling_factor":1.0,"block_out_channels":[1],"down_block_types":["DownEncoderBlock2D"],"up_block_types":["UpDecoderBlock2D"],"norm_num_groups":1,"norm_eps":0.000001}"#,
            ),
            bytes_mem_tensor(
                "scheduler/scheduler_config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"EulerDiscreteScheduler"}"#,
            ),
            bytes_mem_tensor(
                "tokenizer/vocab.json",
                QT_DIFFUSION_TOKENIZER,
                br#"{"<|startoftext|>":0,"<|endoftext|>":1,"a</w>":2,"cat</w>":3}"#,
            ),
            bytes_mem_tensor("tokenizer/merges.txt", QT_DIFFUSION_TOKENIZER, b"#version: 0.2\n"),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.embeddings.token_embedding.weight",
                &[4, 2],
                &[0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.embeddings.position_embedding.weight",
                &[77, 2],
                &[0.0; 154],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.final_layer_norm.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.final_layer_norm.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.q_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.q_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.k_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.k_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.v_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.v_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.out_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.out_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm1.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm1.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc1.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc1.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc2.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc2.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm2.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm2.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &identity1),
            f32_mem_tensor("unet/tensors/conv_in.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_1.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_1.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_2.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_2.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{down_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{down_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{down_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor(&format!("{down_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{down_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{down_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{down_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{down_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{down_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{down_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{mid0_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid0_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid0_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor(&format!("{mid0_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid0_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{mid0_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{mid0_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid0_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid0_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{mid0_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid1_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor(&format!("{mid1_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid1_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{mid1_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{mid1_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{mid1_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{mid1_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                &format!("{up_prefix}.conv1.weight"),
                &[1, 2, 3, 3],
                &[0.0; 18],
            ),
            f32_mem_tensor(&format!("{up_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{up_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{up_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{up_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{up_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{up_prefix}.conv_shortcut.weight"),
                &[1, 2, 1, 1],
                &[1.0, 0.0],
            ),
            f32_mem_tensor(&format!("{up_prefix}.conv_shortcut.bias"), &[1], &[0.0]),
            f32_mem_tensor("unet/tensors/conv_norm_out.weight", &[1], &[1.0]),
            f32_mem_tensor("unet/tensors/conv_norm_out.bias", &[1], &[0.0]),
            f32_mem_tensor("unet/tensors/conv_out.weight", &[1, 1, 3, 3], &identity1),
            f32_mem_tensor("unet/tensors/conv_out.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/encoder.conv_in.weight",
                &[1, 3, 3, 3],
                &vae_encoder_conv_in,
            ),
            f32_mem_tensor("vae/tensors/encoder.conv_in.bias", &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm1.weight"),
                &[1],
                &[1.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm1.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv1.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm2.weight"),
                &[1],
                &[1.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm2.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv2.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor("vae/tensors/encoder.conv_norm_out.weight", &[1], &[1.0]),
            f32_mem_tensor("vae/tensors/encoder.conv_norm_out.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/encoder.conv_out.weight",
                &[2, 1, 3, 3],
                &vae_encoder_conv_out,
            ),
            f32_mem_tensor(
                "vae/tensors/encoder.conv_out.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "vae/tensors/quant_conv.weight",
                &[2, 2, 1, 1],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor("vae/tensors/quant_conv.bias", &[2], &[0.0, 0.0]),
            f32_mem_tensor("vae/tensors/post_quant_conv.weight", &[1, 1, 1, 1], &[1.0]),
            f32_mem_tensor("vae/tensors/post_quant_conv.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/decoder.conv_in.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor("vae/tensors/decoder.conv_in.bias", &[1], &[0.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{vae_resnet_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{vae_resnet_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor("vae/tensors/decoder.conv_norm_out.weight", &[1], &[1.0]),
            f32_mem_tensor("vae/tensors/decoder.conv_norm_out.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/decoder.conv_out.weight",
                &[3, 1, 3, 3],
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            f32_mem_tensor("vae/tensors/decoder.conv_out.bias", &[3], &[0.0, 0.0, 0.0]),
        ]
    }

    struct TestNoiseBackend;

    impl DiffusionNoiseBackend for TestNoiseBackend {
        fn model_input_channels(&self) -> usize {
            1
        }

        fn denoise_latents_with_runtime_context(
            &self,
            mut latents: LatentBatch,
            schedule: &DiffusionSchedule,
            cfg_scale: f32,
            positive_embeddings: &CpuTensor,
            negative_embeddings: &CpuTensor,
            _positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            _negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            _inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
            _masked_reference: Option<&MaskedDenoiseReference<'_>>,
            _runtime_context: &mut DiffusionGenerationRuntimeContext,
            mut progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
        ) -> DiffusionResult<DenoiseLatentsOutput> {
            assert_eq!(schedule.timesteps.len(), 2);
            assert_eq!(cfg_scale, 7.0);
            assert_eq!(positive_embeddings.shape[0], latents.batch);
            assert_eq!(negative_embeddings.shape[0], latents.batch);
            for (idx, value) in latents.data.iter_mut().enumerate() {
                *value = (idx as f32 % 4.0) / 3.0;
            }
            for step in 0..schedule.timesteps.len() {
                if let Some(progress) = progress.as_deref_mut() {
                    progress(DiffusionProgress {
                        completed_steps: step + 1,
                        total_steps: schedule.timesteps.len(),
                        timestep: schedule.timesteps[step].round().max(0.0) as usize,
                        preview_latents: Some(latents.clone()),
                    })?;
                }
            }
            Ok(DenoiseLatentsOutput {
                latents,
                runtime_kind: DiffusionRuntimeKind::CpuSourceReference,
            })
        }
    }

    struct TestSdxlNoiseBackend {
        called: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl DiffusionNoiseBackend for TestSdxlNoiseBackend {
        fn model_input_channels(&self) -> usize {
            1
        }

        fn denoise_latents_with_runtime_context(
            &self,
            latents: LatentBatch,
            schedule: &DiffusionSchedule,
            cfg_scale: f32,
            positive_embeddings: &CpuTensor,
            negative_embeddings: &CpuTensor,
            positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            _inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
            _masked_reference: Option<&MaskedDenoiseReference<'_>>,
            _runtime_context: &mut DiffusionGenerationRuntimeContext,
            _progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
        ) -> DiffusionResult<DenoiseLatentsOutput> {
            assert_eq!(schedule.timesteps.len(), 2);
            assert_eq!(cfg_scale, 7.0);
            assert_eq!(positive_embeddings.shape, vec![1, 4, 4]);
            assert_eq!(negative_embeddings.shape, vec![1, 4, 4]);
            let positive = positive_sdxl_conditioning.expect("positive SDXL conditioning");
            let negative = negative_sdxl_conditioning.expect("negative SDXL conditioning");
            assert_eq!(positive.text_embeds.shape, vec![1, 2]);
            assert_eq!(negative.text_embeds.shape, vec![1, 2]);
            assert_eq!(positive.time_ids.shape, vec![1, 6]);
            assert_eq!(negative.time_ids.shape, vec![1, 6]);
            assert_eq!(
                positive.time_ids.data,
                vec![256.0, 128.0, 4.0, 8.0, 64.0, 32.0]
            );
            assert_eq!(negative.time_ids.data, positive.time_ids.data);
            self.called.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(DenoiseLatentsOutput {
                latents,
                runtime_kind: DiffusionRuntimeKind::CpuSourceReference,
            })
        }
    }

    struct TestInpaintNoiseBackend {
        called: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl DiffusionNoiseBackend for TestInpaintNoiseBackend {
        fn model_input_channels(&self) -> usize {
            3
        }

        fn denoise_latents_with_runtime_context(
            &self,
            latents: LatentBatch,
            schedule: &DiffusionSchedule,
            cfg_scale: f32,
            positive_embeddings: &CpuTensor,
            negative_embeddings: &CpuTensor,
            _positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            _negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
            _masked_reference: Option<&MaskedDenoiseReference<'_>>,
            _runtime_context: &mut DiffusionGenerationRuntimeContext,
            mut progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
        ) -> DiffusionResult<DenoiseLatentsOutput> {
            assert_eq!(schedule.timesteps.len(), 2);
            assert_eq!(cfg_scale, 7.0);
            assert_eq!(positive_embeddings.shape[0], latents.batch);
            assert_eq!(negative_embeddings.shape[0], latents.batch);
            let conditioning = inpaint_conditioning.expect("inpaint conditioning is required");
            assert_eq!(
                conditioning.mask_weights.len(),
                latents.batch * latents.height * latents.width
            );
            assert_eq!(conditioning.masked_image_latents.batch, latents.batch);
            assert_eq!(conditioning.masked_image_latents.channels, latents.channels);
            assert_eq!(conditioning.masked_image_latents.height, latents.height);
            assert_eq!(conditioning.masked_image_latents.width, latents.width);
            self.called.store(true, std::sync::atomic::Ordering::SeqCst);
            for step in 0..schedule.timesteps.len() {
                if let Some(progress) = progress.as_deref_mut() {
                    progress(DiffusionProgress {
                        completed_steps: step + 1,
                        total_steps: schedule.timesteps.len(),
                        timestep: schedule.timesteps[step].round().max(0.0) as usize,
                        preview_latents: Some(latents.clone()),
                    })?;
                }
            }
            Ok(DenoiseLatentsOutput {
                latents,
                runtime_kind: DiffusionRuntimeKind::CpuSourceReference,
            })
        }
    }

    struct TestImageDecoder;

    impl DiffusionImageDecoder for TestImageDecoder {
        fn decode_to_rgb_tensor(&self, latents: &LatentBatch) -> DiffusionResult<CpuTensor> {
            let mut data = Vec::with_capacity(latents.batch * latents.height * latents.width * 3);
            let image_len = latents.len_per_batch();
            for batch in 0..latents.batch {
                let mut red = Vec::with_capacity(latents.height * latents.width);
                let mut green = Vec::with_capacity(latents.height * latents.width);
                let mut blue = Vec::with_capacity(latents.height * latents.width);
                for pixel in 0..(latents.height * latents.width) {
                    let value = (latents.data[batch * image_len + pixel] * 255.0).round() as u8;
                    red.push(rgb_byte_to_model_value(value));
                    green.push(rgb_byte_to_model_value(255u8.saturating_sub(value)));
                    blue.push(rgb_byte_to_model_value(value / 2));
                }
                data.extend(red);
                data.extend(green);
                data.extend(blue);
            }
            Ok(CpuTensor {
                shape: vec![latents.batch, 3, latents.height, latents.width],
                data,
            })
        }
    }

    fn rgb_byte_to_model_value(value: u8) -> f32 {
        (value as f32) / 127.5 - 1.0
    }

    struct SolidTensorImageDecoder;

    impl DiffusionImageDecoder for SolidTensorImageDecoder {
        fn decode_to_rgb_tensor(&self, latents: &LatentBatch) -> DiffusionResult<CpuTensor> {
            let pixels = latents.batch * latents.height * latents.width;
            let mut data = Vec::with_capacity(pixels * 3);
            let pixels_per_batch = latents.height * latents.width;
            for _ in 0..latents.batch {
                data.extend(std::iter::repeat(rgb_byte_to_model_value(32)).take(pixels_per_batch));
                data.extend(std::iter::repeat(rgb_byte_to_model_value(128)).take(pixels_per_batch));
                data.extend(std::iter::repeat(rgb_byte_to_model_value(224)).take(pixels_per_batch));
            }
            Ok(CpuTensor {
                shape: vec![latents.batch, 3, latents.height, latents.width],
                data,
            })
        }
    }

    impl SolidTensorImageDecoder {
        fn expected_rgb(latents: &LatentBatch) -> RgbImageBatch {
            let pixels = latents.batch * latents.height * latents.width;
            let mut data = Vec::with_capacity(pixels * 3);
            for _ in 0..pixels {
                data.extend_from_slice(&[32, 128, 224]);
            }
            RgbImageBatch {
                batch: latents.batch,
                width: latents.width,
                height: latents.height,
                data,
            }
        }
    }

    #[cfg(feature = "rocm")]
    fn tiny_txt2img_test_pipeline(decoder: Box<dyn DiffusionImageDecoder>) -> DiffusionPipeline {
        let metadata = tiny_runtime_metadata();
        let config = tiny_runtime_config();
        let tokenizer = ClipTokenizer::from_bytes(
            br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "cat</w>": 3
            }"#,
            b"#version: 0.2\n",
            4,
        )
        .unwrap();
        let text_encoder = ClipTextEncoder {
            token_embedding: CpuTensor {
                shape: vec![4, 2],
                data: vec![0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
            },
            position_embedding: CpuTensor {
                shape: vec![4, 2],
                data: vec![0.0; 8],
            },
            layers: Vec::new(),
            final_layer_norm_weight: CpuTensor {
                shape: vec![2],
                data: vec![1.0, 1.0],
            },
            final_layer_norm_bias: CpuTensor {
                shape: vec![2],
                data: vec![0.0, 0.0],
            },
            text_projection: None,
            hidden_size: 2,
            max_length: 4,
            n_heads: 1,
        };
        DiffusionPipeline {
            summary: summarize_hfq(Path::new("/tmp/tiny-runtime.hfq"), &metadata),
            metadata,
            config,
            tokenizer: Some(tokenizer),
            tokenizer_2: None,
            text_encoder: Some(text_encoder),
            text_encoder_2: None,
            native_runtime: Some(NativeDiffusionRuntime {
                kind: DiffusionRuntimeKind::CpuSourceReference,
                noise: Box::new(TestNoiseBackend),
                encoder: None,
                decoder,
            }),
            native_runtime_error: None,
        }
    }

    fn tiny_inpaint_test_pipeline(
        temp_label: &str,
        decoder: Box<dyn DiffusionImageDecoder>,
    ) -> (
        DiffusionPipeline,
        std::sync::Arc<std::sync::atomic::AtomicBool>,
        PathBuf,
    ) {
        let dir = std::env::temp_dir().join(format!("{temp_label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-complete.hfq");
        let metadata = tiny_runtime_metadata();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tiny_complete_runtime_tensors(),
        )
        .unwrap();
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let config = tiny_runtime_config();
        let encoder = NativeVaeEncoder::from_hfq(&hfq, &config.vae).unwrap();
        let tokenizer = ClipTokenizer::from_bytes(
            br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "cat</w>": 3
            }"#,
            b"#version: 0.2\n",
            4,
        )
        .unwrap();
        let text_encoder = ClipTextEncoder {
            token_embedding: CpuTensor {
                shape: vec![4, 2],
                data: vec![0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
            },
            position_embedding: CpuTensor {
                shape: vec![4, 2],
                data: vec![0.0; 8],
            },
            layers: Vec::new(),
            final_layer_norm_weight: CpuTensor {
                shape: vec![2],
                data: vec![1.0, 1.0],
            },
            final_layer_norm_bias: CpuTensor {
                shape: vec![2],
                data: vec![0.0, 0.0],
            },
            text_projection: None,
            hidden_size: 2,
            max_length: 4,
            n_heads: 1,
        };
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pipeline = DiffusionPipeline {
            summary: summarize_hfq(Path::new("/tmp/tiny-inpaint.hfq"), &metadata),
            metadata,
            config,
            tokenizer: Some(tokenizer),
            tokenizer_2: None,
            text_encoder: Some(text_encoder),
            text_encoder_2: None,
            native_runtime: Some(NativeDiffusionRuntime {
                kind: DiffusionRuntimeKind::CpuSourceReference,
                noise: Box::new(TestInpaintNoiseBackend {
                    called: called.clone(),
                }),
                encoder: Some(encoder),
                decoder,
            }),
            native_runtime_error: None,
        };
        (pipeline, called, dir)
    }

    fn f32_mem_tensor(name: &str, shape: &[u32], data: &[f32]) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_DIFFUSION_TENSOR_F32,
            shape: shape.to_vec(),
            group_size: 0,
            data: data
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        }
    }

    fn q4f16_g64_mem_tensor(name: &str, shape: &[u32], data: &[f32]) -> HfqMemTensor {
        let mut bytes = Vec::new();
        for group in data.chunks(64) {
            let min = group.iter().copied().fold(f32::INFINITY, f32::min);
            let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = if max > min { (max - min) / 15.0 } else { 1.0 };
            bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
            bytes.extend_from_slice(&f32_to_f16_bits(min).to_le_bytes());
            for idx in 0..32 {
                let lo = group.get(idx).copied().unwrap_or(min);
                let hi = group.get(idx + 32).copied().unwrap_or(min);
                let lo_q = ((lo - min) / scale).round().clamp(0.0, 15.0) as u8;
                let hi_q = ((hi - min) / scale).round().clamp(0.0, 15.0) as u8;
                bytes.push(lo_q | (hi_q << 4));
            }
        }
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_DIFFUSION_TENSOR_Q4F16_G64,
            shape: shape.to_vec(),
            group_size: 64,
            data: bytes,
        }
    }

    fn q4k_mem_tensor(name: &str, shape: &[u32], low_nibbles: &[u8]) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_DIFFUSION_TENSOR_Q4_K,
            shape: shape.to_vec(),
            group_size: 256,
            data: q4k_test_block(low_nibbles),
        }
    }

    fn q4k_test_block(low_nibbles: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; 144];
        bytes[0..2].copy_from_slice(&f32_to_f16_bits(0.25).to_le_bytes());
        bytes[2..4].copy_from_slice(&f32_to_f16_bits(0.0).to_le_bytes());
        bytes[4] = 1;
        bytes[5] = 1;
        for (idx, value) in low_nibbles.iter().copied().take(32).enumerate() {
            bytes[16 + idx] = value.min(15);
        }
        bytes
    }

    fn hfq4_mem_tensor(
        name: &str,
        quant_type: u8,
        shape: &[u32],
        group_size: usize,
        low_nibbles: &[u8],
    ) -> HfqMemTensor {
        let block_bytes = match group_size {
            128 => 72,
            256 => 136,
            _ => panic!("unsupported test HFQ4 group size {group_size}"),
        };
        let mut bytes = vec![0u8; block_bytes];
        bytes[0..4].copy_from_slice(&0.25f32.to_le_bytes());
        bytes[4..8].copy_from_slice(&(-1.0f32).to_le_bytes());
        for idx in 0..(group_size / 2) {
            let lo = low_nibbles.get(idx * 2).copied().unwrap_or(0).min(15);
            let hi = low_nibbles.get(idx * 2 + 1).copied().unwrap_or(0).min(15);
            bytes[8 + idx] = lo | (hi << 4);
        }
        HfqMemTensor {
            name: name.to_string(),
            quant_type,
            shape: shape.to_vec(),
            group_size: group_size as u32,
            data: bytes,
        }
    }

    fn hfq6_mem_tensor(name: &str, shape: &[u32], values: &[u8]) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_DIFFUSION_TENSOR_HFQ6_G256,
            shape: shape.to_vec(),
            group_size: 256,
            data: hfq6_test_block(values),
        }
    }

    fn hfq6_test_block(values: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; 200];
        bytes[0..4].copy_from_slice(&0.25f32.to_le_bytes());
        bytes[4..8].copy_from_slice(&(-1.0f32).to_le_bytes());
        for i in (0..256).step_by(4) {
            let q0 = values.get(i).copied().unwrap_or(0).min(63);
            let q1 = values.get(i + 1).copied().unwrap_or(0).min(63);
            let q2 = values.get(i + 2).copied().unwrap_or(0).min(63);
            let q3 = values.get(i + 3).copied().unwrap_or(0).min(63);
            let offset = 8 + (i / 4) * 3;
            bytes[offset] = q0 | (q1 << 6);
            bytes[offset + 1] = (q1 >> 2) | (q2 << 4);
            bytes[offset + 2] = (q2 >> 4) | (q3 << 2);
        }
        bytes
    }

    fn q8f16_mem_tensor(name: &str, shape: &[u32], data: &[f32]) -> HfqMemTensor {
        let mut bytes = Vec::new();
        for group in data.chunks(32) {
            let max_abs = group.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
            for idx in 0..32 {
                let value = group.get(idx).copied().unwrap_or(0.0);
                let quantized = (value / scale).round().clamp(-128.0, 127.0) as i8;
                bytes.push(quantized as u8);
            }
        }
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_DIFFUSION_TENSOR_Q8F16,
            shape: shape.to_vec(),
            group_size: 32,
            data: bytes,
        }
    }

    fn bytes_mem_tensor(name: &str, quant_type: u8, data: &[u8]) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type,
            shape: vec![data.len() as u32],
            group_size: 0,
            data: data.to_vec(),
        }
    }

    fn write_safetensors_fixture(path: &Path, tensors: &[(&str, &str, &[u64], &[u8])]) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut offset = 0u64;
        for (name, dtype, shape, data) in tensors {
            let end = offset + data.len() as u64;
            header.insert(
                (*name).to_string(),
                json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, end],
                }),
            );
            payload.extend_from_slice(data);
            offset = end;
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        fs::write(path, bytes).unwrap();
    }

    fn write_safetensors_fixture_owned(
        path: &Path,
        tensors: &[(String, String, Vec<u64>, Vec<u8>)],
    ) {
        let borrowed = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    name.as_str(),
                    dtype.as_str(),
                    shape.as_slice(),
                    data.as_slice(),
                )
            })
            .collect::<Vec<_>>();
        write_safetensors_fixture(path, &borrowed);
    }

    fn f32_safetensors_tensor(
        name: &str,
        shape: &[u64],
        data: &[f32],
    ) -> (String, String, Vec<u64>, Vec<u8>) {
        (
            name.to_string(),
            "F32".to_string(),
            shape.to_vec(),
            data.iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }

    fn write_tiny_ldm_unet_safetensors(path: &Path) {
        let identity1 = center_identity_conv(1);
        let mut vae_encoder_conv_in = vec![0.0; 1 * 3 * 3 * 3];
        vae_encoder_conv_in[1 * 3 + 1] = 1.0;
        let mut vae_encoder_conv_out = vec![0.0; 2 * 1 * 3 * 3];
        vae_encoder_conv_out[1 * 3 + 1] = 1.0;
        let mut tensors = vec![
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.0.0.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_safetensors_tensor("model.diffusion_model.input_blocks.0.0.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "model.diffusion_model.time_embed.0.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor("model.diffusion_model.time_embed.0.bias", &[2], &[0.0, 0.0]),
            f32_safetensors_tensor(
                "model.diffusion_model.time_embed.2.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor("model.diffusion_model.time_embed.2.bias", &[2], &[0.0, 0.0]),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.in_layers.0.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.in_layers.2.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.in_layers.2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight",
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.emb_layers.1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.out_layers.0.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.out_layers.0.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.out_layers.3.weight",
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.out_layers.3.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.in_layers.0.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.in_layers.0.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.in_layers.2.weight",
                &[1, 2, 3, 3],
                &[0.0; 18],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.in_layers.2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.emb_layers.1.weight",
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.emb_layers.1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.out_layers.0.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.out_layers.0.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.out_layers.3.weight",
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.out_layers.3.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
                &[1, 2, 1, 1],
                &[1.0, 0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.skip_connection.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor("model.diffusion_model.out.0.weight", &[1], &[1.0]),
            f32_safetensors_tensor("model.diffusion_model.out.0.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "model.diffusion_model.out.2.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_safetensors_tensor("model.diffusion_model.out.2.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.post_quant_conv.weight",
                &[1, 1, 1, 1],
                &[1.0],
            ),
            f32_safetensors_tensor("first_stage_model.post_quant_conv.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.encoder.conv_in.weight",
                &[1, 3, 3, 3],
                &vae_encoder_conv_in,
            ),
            f32_safetensors_tensor("first_stage_model.encoder.conv_in.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.norm1.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.norm1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.conv1.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.conv1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.norm2.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.norm2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.conv2.weight",
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.conv2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor("first_stage_model.encoder.norm_out.weight", &[1], &[1.0]),
            f32_safetensors_tensor("first_stage_model.encoder.norm_out.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.encoder.conv_out.weight",
                &[2, 1, 3, 3],
                &vae_encoder_conv_out,
            ),
            f32_safetensors_tensor("first_stage_model.encoder.conv_out.bias", &[2], &[0.0, 0.0]),
            f32_safetensors_tensor(
                "first_stage_model.quant_conv.weight",
                &[2, 2, 1, 1],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor("first_stage_model.quant_conv.bias", &[2], &[0.0, 0.0]),
            f32_safetensors_tensor(
                "first_stage_model.decoder.conv_in.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_safetensors_tensor("first_stage_model.decoder.conv_in.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.norm1.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.norm1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.conv1.weight",
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.conv1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.norm2.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.norm2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.conv2.weight",
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.conv2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor("first_stage_model.decoder.norm_out.weight", &[1], &[1.0]),
            f32_safetensors_tensor("first_stage_model.decoder.norm_out.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.decoder.conv_out.weight",
                &[3, 1, 3, 3],
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.conv_out.bias",
                &[3],
                &[0.0, 0.0, 0.0],
            ),
        ];
        push_tiny_ldm_clip_text_encoder_tensors(&mut tensors);
        write_safetensors_fixture_owned(path, &tensors);
    }

    fn push_tiny_ldm_clip_text_encoder_tensors(
        tensors: &mut Vec<(String, String, Vec<u64>, Vec<u8>)>,
    ) {
        let prefix = "cond_stage_model.transformer.text_model";
        tensors.extend([
            f32_safetensors_tensor(
                &format!("{prefix}.embeddings.token_embedding.weight"),
                &[3, 2],
                &[0.0, 0.0, 0.5, -0.5, 1.0, 0.25],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.embeddings.position_embedding.weight"),
                &[77, 2],
                &vec![0.0; 77 * 2],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.q_proj.weight"),
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.q_proj.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.k_proj.weight"),
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.k_proj.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.v_proj.weight"),
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.v_proj.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.out_proj.weight"),
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.out_proj.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.layer_norm1.weight"),
                &[2],
                &[1.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.layer_norm1.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.mlp.fc1.weight"),
                &[4, 2],
                &[0.0; 8],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.mlp.fc1.bias"),
                &[4],
                &[0.0; 4],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.mlp.fc2.weight"),
                &[2, 4],
                &[0.0; 8],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.mlp.fc2.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.layer_norm2.weight"),
                &[2],
                &[1.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.layer_norm2.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.final_layer_norm.weight"),
                &[2],
                &[1.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.final_layer_norm.bias"),
                &[2],
                &[0.0, 0.0],
            ),
        ]);
    }

    fn center_identity_conv2(channels: usize) -> Vec<f32> {
        center_identity_conv(channels)
    }

    fn center_identity_conv(channels: usize) -> Vec<f32> {
        let mut data = vec![0.0; channels * channels * 3 * 3];
        for channel in 0..channels {
            data[(((channel * channels + channel) * 3 + 1) * 3) + 1] = 1.0;
        }
        data
    }

    fn push_zero_attention_tensors(
        tensors: &mut Vec<HfqMemTensor>,
        prefix: &str,
        hidden: u32,
        context: u32,
    ) {
        tensors.push(f32_mem_tensor(
            &format!("{prefix}.to_q.weight"),
            &[hidden, hidden],
            &vec![0.0; (hidden * hidden) as usize],
        ));
        tensors.push(f32_mem_tensor(
            &format!("{prefix}.to_k.weight"),
            &[hidden, context],
            &vec![0.0; (hidden * context) as usize],
        ));
        tensors.push(f32_mem_tensor(
            &format!("{prefix}.to_v.weight"),
            &[hidden, context],
            &vec![0.0; (hidden * context) as usize],
        ));
        tensors.push(f32_mem_tensor(
            &format!("{prefix}.to_out.0.weight"),
            &[hidden, hidden],
            &vec![0.0; (hidden * hidden) as usize],
        ));
        tensors.push(f32_mem_tensor(
            &format!("{prefix}.to_out.0.bias"),
            &[hidden],
            &vec![0.0; hidden as usize],
        ));
    }
}
