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
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
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

    /// Build runtime options for the daemon-resolved `device_id`. hipfire is
    /// HIP/ROCm-first, so the GPU is the default; the CPU reference path is an
    /// opt-in correctness oracle (too slow for real generation) enabled only via
    /// the `HIPFIRE_DIFFUSION_CPU_REFERENCE` environment variable.
    pub fn for_device(device_id: i32) -> Self {
        if Self::cpu_reference_requested() {
            Self::cpu_reference()
        } else {
            Self::rocm_hybrid(device_id)
        }
    }

    /// Whether the CPU reference oracle was requested via
    /// `HIPFIRE_DIFFUSION_CPU_REFERENCE`. Frontends should consult this before
    /// resolving a GPU device so a CPU-only run does not require a GPU.
    pub fn cpu_reference_requested() -> bool {
        cpu_reference_env_enabled(
            std::env::var("HIPFIRE_DIFFUSION_CPU_REFERENCE")
                .ok()
                .as_deref(),
        )
    }
}

/// Pure predicate for the `HIPFIRE_DIFFUSION_CPU_REFERENCE` toggle: unset, empty,
/// `0`, `false`, or `no` mean "GPU"; any other value means "CPU reference".
fn cpu_reference_env_enabled(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        None | Some("") | Some("0") => false,
        Some(v) => !v.eq_ignore_ascii_case("false") && !v.eq_ignore_ascii_case("no"),
    }
}

/// Device-resident cache for VAE/UNet weights. Each weight tensor is uploaded
/// once and reused across every denoise step and CFG pass instead of being
/// re-copied to the device on every op call. Keyed by the host data pointer plus
/// length, which is stable for the lifetime of the owning layer (weights live in
/// the pipeline runtime and are not moved mid-generation). The cache lives for one
/// generation (the runtime context is created per `generate_*` call), so resident
/// buffers are released when the GPU/context tears down.
#[derive(Default)]
struct RocmWeightCache {
    entries: std::collections::HashMap<(usize, usize), rdna_compute::GpuTensor>,
}

impl RocmWeightCache {
    /// Return the raw device pointer for `tensor`, uploading it once on first use.
    fn resident_ptr(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        tensor: &CpuTensor,
    ) -> DiffusionResult<*mut std::ffi::c_void> {
        let key = (tensor.data.as_ptr() as usize, tensor.data.len());
        if !self.entries.contains_key(&key) {
            let resident = gpu
                .upload_f32(&tensor.data, &tensor.shape)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            self.entries.insert(key, resident);
        }
        Ok(self
            .entries
            .get(&key)
            .expect("weight just inserted")
            .buf
            .as_ptr())
    }
}

struct DiffusionGenerationRuntimeContext {
    options: DiffusionGenerationRuntimeOptions,
    rocm_gpu: Option<rdna_compute::Gpu>,
    rocm_gpu_init_count: usize,
    rocm_weights: RocmWeightCache,
}

impl DiffusionGenerationRuntimeContext {
    fn new(options: DiffusionGenerationRuntimeOptions) -> Self {
        Self {
            options,
            rocm_gpu: None,
            rocm_gpu_init_count: 0,
            rocm_weights: RocmWeightCache::default(),
        }
    }

    fn rocm_device_id(&self) -> Option<i32> {
        self.options.rocm_device_id
    }

    fn ensure_rocm_gpu(&mut self) -> DiffusionResult<()> {
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
        Ok(())
    }

    fn with_rocm_gpu<T>(
        &mut self,
        f: impl FnOnce(&mut rdna_compute::Gpu) -> DiffusionResult<T>,
    ) -> DiffusionResult<T> {
        self.ensure_rocm_gpu()?;
        let gpu = self.rocm_gpu.as_mut().ok_or_else(|| {
            DiffusionError::BackendUnavailable(
                "ROCm runtime context failed to retain initialized GPU".to_string(),
            )
        })?;
        f(gpu)
    }

    /// Like [`with_rocm_gpu`], but also exposes the device weight cache so
    /// weight-bearing ops (conv, linear) can reuse resident weights instead of
    /// re-uploading them on every call.
    fn with_rocm_gpu_weighted<T>(
        &mut self,
        f: impl FnOnce(&mut rdna_compute::Gpu, &mut RocmWeightCache) -> DiffusionResult<T>,
    ) -> DiffusionResult<T> {
        self.ensure_rocm_gpu()?;
        let gpu = self.rocm_gpu.as_mut().ok_or_else(|| {
            DiffusionError::BackendUnavailable(
                "ROCm runtime context failed to retain initialized GPU".to_string(),
            )
        })?;
        f(gpu, &mut self.rocm_weights)
    }

    #[cfg(test)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transformer_denoiser: Option<DiffusionTransformerDenoiserPlan>,
    pub latent_bytes: usize,
    pub denoise_input_bytes: usize,
    pub conditioning_bytes: usize,
    pub vae_decode_bytes: usize,
    pub rgb_bytes: usize,
    pub scheduler_scratch_bytes: usize,
    pub total_device_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionTransformerDenoiserPlan {
    pub representation: String,
    pub batch: usize,
    pub sequence_length: usize,
    pub token_width: usize,
    pub patch_size: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    pub patch_height: usize,
    pub patch_width: usize,
    pub output_channels: usize,
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
enum TransformerDenoiserFamily {
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
struct TransformerDenoiserWeightTopology {
    family: TransformerDenoiserFamily,
    block_count: usize,
    has_input_projection: bool,
    has_output_projection: bool,
    has_text_modulation: bool,
    has_text_fusion: bool,
}

impl TransformerDenoiserWeightTopology {
    fn diagnostic_label(&self) -> String {
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
struct VaeLatentNorm {
    scaling_factor: f32,
    shift_factor: f32,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

impl VaeLatentNorm {
    fn from_config(config: &VaeConfig) -> DiffusionResult<Self> {
        let latents_mean = config.latents_mean.clone();
        let latents_std = config.latents_std.clone();
        if latents_mean.len() != latents_std.len() {
            return Err(DiffusionError::InvalidMetadata(format!(
                "VAE latents_mean ({}) and latents_std ({}) must have matching length",
                latents_mean.len(),
                latents_std.len()
            )));
        }
        if latents_std.iter().any(|value| value.abs() <= f32::MIN_POSITIVE) {
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
    fn scalar(scaling_factor: f32) -> Self {
        Self {
            scaling_factor,
            shift_factor: 0.0,
            latents_mean: Vec::new(),
            latents_std: Vec::new(),
        }
    }

    /// Per-channel mean/std normalization (Qwen-Image/Wan). When false, the scalar
    /// `scaling_factor`/`shift_factor` path applies.
    fn is_per_channel(&self) -> bool {
        !self.latents_mean.is_empty()
    }

    /// `true` when decode reduces to a single reciprocal scale (the SD/SDXL fast path
    /// that the fused HIP kernel covers).
    fn is_scalar_scale_only(&self) -> bool {
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
    fn apply_encode(
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
    fn apply_decode(
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

mod scheduler;
pub use scheduler::*;

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
        |sample, timesteps, encoder_states, _attention_mask, _sdxl| {
            predict_noise(sample, timesteps, encoder_states)
        },
        None,
        None,
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
        Option<&CpuTensor>,
        Option<&SdxlDenoiseConditioning<'_>>,
    ) -> DiffusionResult<CpuTensor>,
    positive_attention_mask: Option<&CpuTensor>,
    negative_attention_mask: Option<&CpuTensor>,
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
        |sample, timesteps, encoder_states, attention_mask, sdxl_conditioning, _runtime_context| {
            predict_noise(
                sample,
                timesteps,
                encoder_states,
                attention_mask,
                sdxl_conditioning,
            )
        },
        positive_attention_mask,
        negative_attention_mask,
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
        Option<&CpuTensor>,
        Option<&SdxlDenoiseConditioning<'_>>,
        &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor>,
    positive_attention_mask: Option<&CpuTensor>,
    negative_attention_mask: Option<&CpuTensor>,
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
        positive_attention_mask,
        negative_attention_mask,
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
        Option<&CpuTensor>,
        Option<&SdxlDenoiseConditioning<'_>>,
        &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor>,
    positive_attention_mask: Option<&CpuTensor>,
    negative_attention_mask: Option<&CpuTensor>,
    positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
    negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
    inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
    masked_reference: Option<&MaskedDenoiseReference<'_>>,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
    mut progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
) -> DiffusionResult<DenoiseLatentsOutput> {
    validate_conditioning_for_latents(&latents, positive_embeddings)?;
    validate_conditioning_for_latents(&latents, negative_embeddings)?;
    if let Some(mask) = positive_attention_mask {
        let [batch, seq, _] = shape3(positive_embeddings)?;
        validate_text_attention_mask(mask, batch, seq, "positive conditioning")?;
    }
    if let Some(mask) = negative_attention_mask {
        let [batch, seq, _] = shape3(negative_embeddings)?;
        validate_text_attention_mask(mask, batch, seq, "negative conditioning")?;
    }
    if let Some(inpaint_conditioning) = inpaint_conditioning {
        validate_inpaint_denoise_conditioning(&latents, inpaint_conditioning)?;
    }
    if let Some(masked_reference) = masked_reference {
        validate_masked_denoise_reference(&latents, masked_reference)?;
    }
    let mut scheduler_state = SchedulerStepState::default();
    let mut runtime_kind = DiffusionRuntimeKind::CpuSourceReference;
    let cfg_is_identity = classifier_free_guidance_is_identity(cfg_scale);
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
        let positive_pred = predict_noise(
            &model_sample,
            &timesteps,
            positive_embeddings,
            positive_attention_mask,
            positive_sdxl_conditioning,
            runtime_context,
        )?;
        validate_noise_prediction(&latents, &positive_pred)?;
        let guided = if cfg_is_identity {
            positive_pred
        } else {
            let negative_pred = predict_noise(
                &model_sample,
                &timesteps,
                negative_embeddings,
                negative_attention_mask,
                negative_sdxl_conditioning,
                runtime_context,
            )?;
            validate_noise_prediction(&latents, &negative_pred)?;
            let (guided, guidance_runtime_kind) = cfg_guidance_with_runtime_context(
                &negative_pred,
                &positive_pred,
                cfg_scale,
                runtime_context,
            )?;
            runtime_kind = merge_runtime_kind(runtime_kind, guidance_runtime_kind);
            guided
        };
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

fn validate_text_attention_mask(
    mask: &CpuTensor,
    batch: usize,
    seq: usize,
    context: &str,
) -> DiffusionResult<()> {
    let [mask_batch, mask_seq] = shape2(mask)?;
    if mask_batch != batch || mask_seq != seq {
        return Err(DiffusionError::InvalidRequest(format!(
            "{context} attention mask shape {:?} != [{batch}, {seq}]",
            mask.shape
        )));
    }
    Ok(())
}

fn qwen_joint_key_mask(
    text_attention_mask: Option<&CpuTensor>,
    batch: usize,
    text_seq: usize,
    image_seq: usize,
) -> DiffusionResult<Option<Vec<bool>>> {
    let Some(mask) = text_attention_mask else {
        return Ok(None);
    };
    validate_text_attention_mask(mask, batch, text_seq, "Qwen text")?;
    let joint_seq = text_seq.checked_add(image_seq).ok_or_else(|| {
        DiffusionError::InvalidRequest("Qwen joint attention sequence length overflow".to_string())
    })?;
    let mut joint_mask = vec![true; batch * joint_seq];
    for b in 0..batch {
        for text_idx in 0..text_seq {
            joint_mask[b * joint_seq + text_idx] = mask.data[b * text_seq + text_idx] > 0.5;
        }
    }
    Ok(Some(joint_mask))
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

fn classifier_free_guidance_is_identity(cfg_scale: f32) -> bool {
    (cfg_scale - 1.0).abs() <= f32::EPSILON
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
    {
        runtime_context.with_rocm_gpu(|gpu| {
            maybe_center_unet_input_hip_on_gpu(gpu, sample, center_input_sample)
        })
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
    {
        runtime_context.with_rocm_gpu(|gpu| {
            timestep_embedding_hip_on_gpu(gpu, timesteps, dim, flip_sin_to_cos, freq_shift)
        })
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
    {
        let data = runtime_context
            .with_rocm_gpu(|gpu| scale_model_input_hip_on_gpu(gpu, &input.data, scale))?;
        Ok(CpuTensor {
            shape: input.shape.clone(),
            data,
        })
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
    {
        runtime_context.with_rocm_gpu_weighted(|gpu, cache| {
            linear_optional_bias_hip_on_gpu(gpu, cache, input, weight, bias)
        })
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
    {
        runtime_context.with_rocm_gpu(|gpu| silu_hip_on_gpu(gpu, input))
    }
}

fn quick_gelu_with_runtime_context(
    input: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return Ok(tensor_map(input, quick_gelu));
    };
    {
        runtime_context.with_rocm_gpu(|gpu| quick_gelu_hip_on_gpu(gpu, input))
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
}

fn tensor_add_with_runtime_context(
    a: &CpuTensor,
    b: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return tensor_add(a, b);
    };
    {
        runtime_context.with_rocm_gpu(|gpu| tensor_add_hip_on_gpu(gpu, a, b))
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
    {
        runtime_context.with_rocm_gpu(|gpu| concat_last_dim_2d_hip_on_gpu(gpu, a, b))
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
    {
        runtime_context.with_rocm_gpu(|gpu| concat_last_dim_3d_hip_on_gpu(gpu, a, b))
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
    {
        runtime_context.with_rocm_gpu_weighted(|gpu, cache| {
            conv2d_nchw_hip_on_gpu(gpu, cache, input, weight, bias, padding, stride)
        })
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
    {
        runtime_context
            .with_rocm_gpu(|gpu| group_norm_nchw_hip_on_gpu(gpu, input, weight, bias, groups, eps))
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
    {
        *input = runtime_context
            .with_rocm_gpu(|gpu| add_channel_bias_nchw_hip_on_gpu(gpu, input, bias))?;
        Ok(())
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
    {
        runtime_context.with_rocm_gpu(|gpu| concat_channels_nchw_hip_on_gpu(gpu, a, b))
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
    {
        runtime_context.with_rocm_gpu(|gpu| upsample_nearest2d_nchw_hip_on_gpu(gpu, input, scale))
    }
}

fn nchw_to_bsc_with_runtime_context(
    input: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return nchw_to_bsc(input);
    };
    {
        runtime_context.with_rocm_gpu(|gpu| nchw_to_bsc_hip_on_gpu(gpu, input))
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
    {
        runtime_context
            .with_rocm_gpu(|gpu| bsc_to_nchw_hip_on_gpu(gpu, input, batch, channels, height, width))
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
    {
        runtime_context.with_rocm_gpu(|gpu| layer_norm_hip_on_gpu(gpu, input, weight, bias, eps))
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
    {
        runtime_context
            .with_rocm_gpu(|gpu| scaled_dot_product_attention_hip_on_gpu(gpu, q, k, v, heads))
    }
}

fn scaled_dot_product_attention_with_key_mask_and_runtime_context(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    heads: usize,
    key_mask: Option<&[bool]>,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    if key_mask.is_none() {
        return scaled_dot_product_attention_with_runtime_context(q, k, v, heads, runtime_context);
    }
    scaled_dot_product_attention_with_key_mask(q, k, v, heads, key_mask)
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
    {
        runtime_context
            .with_rocm_gpu(|gpu| clip_causal_self_attention_hip_on_gpu(gpu, q, k, v, n_heads))
    }
}

fn geglu_gate_3d_with_runtime_context(
    projected: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return geglu_gate_3d(projected);
    };
    {
        runtime_context.with_rocm_gpu(|gpu| geglu_gate_3d_hip_on_gpu(gpu, projected))
    }
}

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

mod quant_decode;
use quant_decode::*;

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
        let runtime_support_error = native_runtime_support_error(&hfq, &metadata)?;
        let (native_runtime, native_runtime_error) = if let Some(error) = runtime_support_error {
            (None, Some(error))
        } else {
            match NativeDiffusionRuntime::from_hfq(&hfq, &metadata, &config) {
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
            &mut RocmWeightCache::default(),
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
            &mut RocmWeightCache::default(),
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
        let vae_moments_to_latents_cpu_reference =
            vae_moments_to_latents(&vae_moments, &VaeLatentNorm::scalar(0.18215))?;
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
            plan.conditioning.prompt_attention_mask.as_ref(),
            plan.conditioning.negative_attention_mask.as_ref(),
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
        let request_seeds = request
            .batch
            .prompts
            .iter()
            .map(|prompt| prompt.seed)
            .collect::<Vec<_>>();
        let init_encode_seeds = vae_encode_seeds(&request_seeds, VAE_INIT_ENCODE_SEED_SALT);
        let (encoded_init_latents, init_encode_kind) = encode_to_latents_with_runtime_context(
            encoder,
            &init_image,
            Some(&init_encode_seeds),
            &mut runtime_context,
        )?;
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
                &request_seeds,
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
                plan.conditioning.prompt_attention_mask.as_ref(),
                plan.conditioning.negative_attention_mask.as_ref(),
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
        let latent_shape = latent_shape_for_request(&self.config, request)?;
        let conditioning =
            self.prepare_conditioning_batch_with_runtime_context(request, runtime_context)?;
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
        if let Some(conditioning) = request.conditioning.as_ref() {
            return Ok(diffusion_conditioning_from_external_batch(
                conditioning,
                request.prompts.len(),
            ));
        }
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
        let cfg_is_identity = classifier_free_guidance_is_identity(request.cfg_scale);
        let negative_tokens = if cfg_is_identity {
            prompt_tokens.clone()
        } else {
            request
                .prompts
                .iter()
                .map(|prompt| tokenizer.encode_padded(&prompt.negative_prompt))
                .collect::<Vec<_>>()
        };
        let (prompt_tokens_2, negative_tokens_2) =
            if let Some(tokenizer_2) = self.tokenizer_2.as_ref() {
                let prompt_tokens_2 = request
                    .prompts
                    .iter()
                    .map(|prompt| tokenizer_2.encode_padded(&prompt.prompt))
                    .collect::<Vec<_>>();
                let negative_tokens_2 = if cfg_is_identity {
                    prompt_tokens_2.clone()
                } else {
                    request
                        .prompts
                        .iter()
                        .map(|prompt| tokenizer_2.encode_padded(&prompt.negative_prompt))
                        .collect::<Vec<_>>()
                };
                (Some(prompt_tokens_2), Some(negative_tokens_2))
            } else {
                (None, None)
            };
        let prompt_embeddings = self
            .text_encoder
            .as_ref()
            .map(|text_encoder| {
                encode_token_batch_with_runtime_context(
                    text_encoder,
                    &prompt_tokens,
                    runtime_context,
                )
            })
            .transpose()?;
        let negative_embeddings = if cfg_is_identity {
            prompt_embeddings.clone()
        } else if let Some(text_encoder) = self.text_encoder.as_ref() {
            Some(encode_token_batch_with_runtime_context(
                text_encoder,
                &negative_tokens,
                runtime_context,
            )?)
        } else {
            None
        };
        let (
            prompt_embeddings_2,
            negative_embeddings_2,
            prompt_cross_attention_embeddings,
            negative_cross_attention_embeddings,
            prompt_pooled_embeddings,
            negative_pooled_embeddings,
        ) = if let (Some(text_encoder_2), Some(tokenizer_2), Some(prompt_tokens_2)) = (
            self.text_encoder_2.as_ref(),
            self.tokenizer_2.as_ref(),
            prompt_tokens_2.as_ref(),
        ) {
            let (prompt_embeddings_2, prompt_pooled_embeddings) =
                encode_token_batch_with_pooled_and_runtime_context(
                    text_encoder_2,
                    prompt_tokens_2,
                    tokenizer_2.end_token_id(),
                    runtime_context,
                )?;
            let (negative_embeddings_2, negative_pooled_embeddings) = if cfg_is_identity {
                (
                    prompt_embeddings_2.clone(),
                    prompt_pooled_embeddings.clone(),
                )
            } else {
                let negative_tokens_2 = negative_tokens_2.as_ref().ok_or_else(|| {
                    DiffusionError::InvalidRequest(
                        "secondary negative prompt tokens are missing".to_string(),
                    )
                })?;
                encode_token_batch_with_pooled_and_runtime_context(
                    text_encoder_2,
                    negative_tokens_2,
                    tokenizer_2.end_token_id(),
                    runtime_context,
                )?
            };
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
            let negative_cross_attention_embeddings = if cfg_is_identity {
                prompt_cross_attention_embeddings.clone()
            } else {
                negative_embeddings
                    .as_ref()
                    .map(|negative_embeddings| {
                        concat_last_dim_3d_with_runtime_context(
                            negative_embeddings,
                            &negative_embeddings_2,
                            runtime_context,
                        )
                    })
                    .transpose()?
            };
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
            prompt_attention_mask: None,
            negative_attention_mask: None,
            prompt_pooled_embeddings,
            negative_pooled_embeddings,
        })
    }
}

fn diffusion_conditioning_from_external_batch(
    conditioning: &DiffusionExternalConditioningBatch,
    batch: usize,
) -> DiffusionConditioningBatch {
    let prompt_cross_attention_embeddings = conditioning
        .prompt_pooled_embeddings
        .as_ref()
        .map(|_| conditioning.prompt_embeddings.clone());
    let negative_cross_attention_embeddings = conditioning
        .negative_pooled_embeddings
        .as_ref()
        .map(|_| conditioning.negative_embeddings.clone());
    DiffusionConditioningBatch {
        prompt_tokens: vec![Vec::new(); batch],
        negative_tokens: vec![Vec::new(); batch],
        prompt_tokens_2: None,
        negative_tokens_2: None,
        prompt_embeddings: Some(conditioning.prompt_embeddings.clone()),
        negative_embeddings: Some(conditioning.negative_embeddings.clone()),
        prompt_embeddings_2: None,
        negative_embeddings_2: None,
        prompt_cross_attention_embeddings,
        negative_cross_attention_embeddings,
        prompt_attention_mask: conditioning.prompt_attention_mask.clone(),
        negative_attention_mask: conditioning.negative_attention_mask.clone(),
        prompt_pooled_embeddings: conditioning.prompt_pooled_embeddings.clone(),
        negative_pooled_embeddings: conditioning.negative_pooled_embeddings.clone(),
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
        positive_attention_mask: Option<&CpuTensor>,
        negative_attention_mask: Option<&CpuTensor>,
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
        positive_attention_mask: Option<&CpuTensor>,
        negative_attention_mask: Option<&CpuTensor>,
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
            positive_attention_mask,
            negative_attention_mask,
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
    seeds: Option<&[i64]>,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<(LatentBatch, DiffusionRuntimeKind)> {
    let latents = match seeds {
        Some(seeds) => {
            encoder.encode_to_latents_sampled_with_runtime_context(image, seeds, runtime_context)?
        }
        None => encoder.encode_to_latents_with_runtime_context(image, runtime_context)?,
    };
    Ok((latents, runtime_kind_for_context(runtime_context)))
}

fn latent_mask_weights_with_runtime_context(
    mask: &RgbImageBatch,
    latents: &LatentBatch,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<(Vec<f32>, DiffusionRuntimeKind)> {
    if let Some(_device_id) = runtime_context.rocm_device_id() {
        {
            let weights = runtime_context.with_rocm_gpu(|gpu| {
                latent_mask_weights_from_rgb_batch_hip_on_gpu(gpu, mask, latents)
            })?;
            return Ok((weights, DiffusionRuntimeKind::RocmHybridReference));
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
        {
            let masked = runtime_context
                .with_rocm_gpu(|gpu| masked_rgb_batch_for_inpaint_hip_on_gpu(gpu, image, mask))?;
            return Ok((masked, DiffusionRuntimeKind::RocmHybridReference));
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
        {
            *generated = runtime_context.with_rocm_gpu(|gpu| {
                blend_latents_with_mask_hip_on_gpu(gpu, generated, init, mask_weights)
            })?;
            return Ok(DiffusionRuntimeKind::RocmHybridReference);
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
    fn from_hfq(
        hfq: &HfqFile,
        metadata: &DiffusionHfqMetadata,
        config: &StableDiffusionConfig,
    ) -> DiffusionResult<Self> {
        let noise: Box<dyn DiffusionNoiseBackend> =
            if let Some(transformer) = metadata.components.get("transformer") {
                let topology = transformer_denoiser_weight_topology(transformer);
                Box::new(NativeTransformerDenoiser::from_hfq(hfq, config, &topology)?)
            } else {
                Box::new(NativeUnet2DConditionModel::from_hfq(hfq, &config.unet)?)
            };
        Ok(Self {
            kind: DiffusionRuntimeKind::CpuSourceReference,
            noise,
            encoder: NativeVaeEncoder::from_hfq(hfq, &config.vae).ok(),
            decoder: Box::new(NativeVaeDecoder::from_hfq(hfq, &config.vae)?),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct NativeTransformerDenoiserIo {
    family: TransformerDenoiserFamily,
    patch_size: usize,
    input_channels: usize,
    output_channels: usize,
    input_token_width: usize,
    hidden_width: usize,
    output_token_width: usize,
    img_in_weight: CpuTensor,
    img_in_bias: CpuTensor,
    output_weight: CpuTensor,
    output_bias: CpuTensor,
    text_norm_weight: Option<CpuTensor>,
    text_in_weight: Option<CpuTensor>,
    text_in_bias: Option<CpuTensor>,
    output_norm_weight: Option<CpuTensor>,
    output_norm_bias: Option<CpuTensor>,
}

#[allow(dead_code)]
impl NativeTransformerDenoiserIo {
    fn from_hfq(
        hfq: &HfqFile,
        config: &StableDiffusionConfig,
        topology: &TransformerDenoiserWeightTopology,
    ) -> DiffusionResult<Self> {
        let transformer = config.transformer.as_ref().ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "transformer denoiser config is required for transformer IO".to_string(),
            )
        })?;
        let patch_size = transformer
            .patch_size
            .or_else(|| default_transformer_patch_size(&transformer.class_name))
            .unwrap_or(1)
            .max(1);
        let patch_feature_width = config
            .latent_channels
            .checked_mul(patch_size)
            .and_then(|value| value.checked_mul(patch_size))
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata(
                    "transformer patch feature width overflow".to_string(),
                )
            })?;
        let input_channels = transformer.in_channels.unwrap_or(patch_feature_width);
        if input_channels < patch_feature_width {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer in_channels {input_channels} is smaller than latent patch feature width {patch_feature_width}"
            )));
        }
        let output_channels = transformer.out_channels.unwrap_or(config.latent_channels);
        let output_patch_feature_width = output_channels
            .checked_mul(patch_size)
            .and_then(|value| value.checked_mul(patch_size))
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata(
                    "transformer output patch feature width overflow".to_string(),
                )
            })?;

        let img_in_weight = CpuTensor::from_hfq(hfq, "transformer/tensors/img_in.weight")?;
        let img_in_bias = CpuTensor::from_hfq(hfq, "transformer/tensors/img_in.bias")?;
        let [hidden_width, input_token_width] = shape2(&img_in_weight)?;
        if input_token_width != input_channels {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer img_in input width {input_token_width} != configured in_channels {input_channels}"
            )));
        }
        if img_in_bias.shape.as_slice() != [hidden_width] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer img_in bias shape {:?} != [{hidden_width}]",
                img_in_bias.shape
            )));
        }

        let (output_weight_entry, output_bias_entry) = match topology.family {
            TransformerDenoiserFamily::QwenImage | TransformerDenoiserFamily::Unknown => (
                "transformer/tensors/proj_out.weight",
                "transformer/tensors/proj_out.bias",
            ),
            TransformerDenoiserFamily::Krea2 => (
                "transformer/tensors/final_layer.linear.weight",
                "transformer/tensors/final_layer.linear.bias",
            ),
        };
        let output_weight = CpuTensor::from_hfq(hfq, output_weight_entry)?;
        let output_bias = CpuTensor::from_hfq(hfq, output_bias_entry)?;
        let [output_token_width, output_hidden_width] = shape2(&output_weight)?;
        if output_hidden_width != hidden_width {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer output projection input width {output_hidden_width} != img_in hidden width {hidden_width}"
            )));
        }
        if output_token_width < output_patch_feature_width {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer output token width {output_token_width} is smaller than output patch feature width {output_patch_feature_width}"
            )));
        }
        if output_bias.shape.as_slice() != [output_token_width] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer output projection bias shape {:?} != [{output_token_width}]",
                output_bias.shape
            )));
        }
        let text_norm_weight = optional_tensor(hfq, "transformer/tensors/txt_norm.weight")?;
        if let Some(weight) = text_norm_weight.as_ref() {
            let text_width = transformer
                .cross_attention_dim
                .or(transformer.text_hidden_dim)
                .unwrap_or_else(|| weight.shape.first().copied().unwrap_or(0));
            if weight.shape.as_slice() != [text_width] {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "transformer txt_norm weight shape {:?} != [{text_width}]",
                    weight.shape
                )));
            }
        }
        let text_in_weight = optional_tensor(hfq, "transformer/tensors/txt_in.weight")?;
        let text_in_bias = if text_in_weight.is_some() {
            Some(CpuTensor::from_hfq(hfq, "transformer/tensors/txt_in.bias")?)
        } else {
            None
        };
        if let Some(weight) = text_in_weight.as_ref() {
            let [text_out_width, text_in_width] = shape2(weight)?;
            if text_out_width != hidden_width {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "transformer txt_in output width {text_out_width} != img_in hidden width {hidden_width}"
                )));
            }
            if let Some(norm_weight) = text_norm_weight.as_ref() {
                if norm_weight.shape.as_slice() != [text_in_width] {
                    return Err(DiffusionError::InvalidMetadata(format!(
                        "transformer txt_norm width {:?} != txt_in input width {text_in_width}",
                        norm_weight.shape
                    )));
                }
            }
            if text_in_bias.as_ref().map(|bias| bias.shape.as_slice()) != Some(&[hidden_width][..])
            {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "transformer txt_in bias shape {:?} != [{hidden_width}]",
                    text_in_bias.as_ref().map(|bias| bias.shape.clone())
                )));
            }
        }
        let output_norm_weight =
            optional_tensor(hfq, "transformer/tensors/norm_out.linear.weight")?;
        let output_norm_bias = if output_norm_weight.is_some() {
            Some(CpuTensor::from_hfq(
                hfq,
                "transformer/tensors/norm_out.linear.bias",
            )?)
        } else {
            None
        };
        if let Some(weight) = output_norm_weight.as_ref() {
            let [norm_rows, norm_cols] = shape2(weight)?;
            if norm_rows != hidden_width * 2 || norm_cols != hidden_width {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "transformer norm_out.linear weight shape {:?} != [{}, {hidden_width}]",
                    weight.shape,
                    hidden_width * 2
                )));
            }
            if output_norm_bias.as_ref().map(|bias| bias.shape.as_slice())
                != Some(&[hidden_width * 2][..])
            {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "transformer norm_out.linear bias shape {:?} != [{}]",
                    output_norm_bias.as_ref().map(|bias| bias.shape.clone()),
                    hidden_width * 2
                )));
            }
        }

        Ok(Self {
            family: topology.family,
            patch_size,
            input_channels,
            output_channels,
            input_token_width,
            hidden_width,
            output_token_width,
            img_in_weight,
            img_in_bias,
            output_weight,
            output_bias,
            text_norm_weight,
            text_in_weight,
            text_in_bias,
            output_norm_weight,
            output_norm_bias,
        })
    }

    fn project_latents_to_hidden_with_runtime_context(
        &self,
        latents: &LatentBatch,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let tokens =
            latent_batch_to_patch_tokens(latents, self.patch_size, self.input_token_width)?;
        linear_3d_with_runtime_context(
            &tokens,
            &self.img_in_weight,
            Some(&self.img_in_bias),
            runtime_context,
        )
    }

    fn project_hidden_to_latents_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        timestep_embedding: &CpuTensor,
        batch: usize,
        height: usize,
        width: usize,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<LatentBatch> {
        let hidden =
            self.output_norm_with_runtime_context(hidden, timestep_embedding, runtime_context)?;
        let tokens = linear_3d_with_runtime_context(
            &hidden,
            &self.output_weight,
            Some(&self.output_bias),
            runtime_context,
        )?;
        patch_tokens_to_latent_batch(
            &tokens,
            batch,
            self.output_channels,
            height,
            width,
            self.patch_size,
        )
    }

    fn project_text_to_hidden_with_runtime_context(
        &self,
        text_hidden: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let [_, _, input_width] = shape3(text_hidden)?;
        let text_hidden = if let Some(weight) = self.text_norm_weight.as_ref() {
            if weight.shape.as_slice() != [input_width] {
                return Err(DiffusionError::InvalidRequest(format!(
                    "transformer text hidden width {input_width} != txt_norm width {}",
                    weight.shape.first().copied().unwrap_or(0)
                )));
            }
            rms_norm_3d_with_runtime_context(text_hidden, weight, 1e-6, runtime_context)?
        } else {
            text_hidden.clone()
        };
        let Some(weight) = self.text_in_weight.as_ref() else {
            if input_width != self.hidden_width {
                return Err(DiffusionError::InvalidRequest(format!(
                    "transformer text hidden width {input_width} != expected hidden width {}; artifact has no txt_in projection",
                    self.hidden_width
                )));
            }
            return Ok(text_hidden);
        };
        let bias = self.text_in_bias.as_ref().ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "transformer txt_in weight is present but bias is missing".to_string(),
            )
        })?;
        linear_3d_with_runtime_context(&text_hidden, weight, Some(bias), runtime_context)
    }

    fn output_norm_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        timestep_embedding: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let Some(weight) = self.output_norm_weight.as_ref() else {
            return Ok(hidden.clone());
        };
        let bias = self.output_norm_bias.as_ref().ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "transformer norm_out.linear weight is present but bias is missing".to_string(),
            )
        })?;
        let [batch, _, width] = shape3(hidden)?;
        if width != self.hidden_width {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer output hidden width {width} != expected {}",
                self.hidden_width
            )));
        }
        let [time_batch, time_width] = shape2(timestep_embedding)?;
        if time_batch != batch || time_width != self.hidden_width {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer output norm timestep shape {:?} != [{batch}, {}]",
                timestep_embedding.shape, self.hidden_width
            )));
        }
        let activated = silu_with_runtime_context(timestep_embedding, runtime_context)?;
        let projected = linear_with_runtime_context(&activated, weight, bias, runtime_context)?;
        let [projected_batch, projected_width] = shape2(&projected)?;
        if projected_batch != batch || projected_width != width * 2 {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer output norm projection shape {:?} != [{batch}, {}]",
                projected.shape,
                width * 2
            )));
        }
        let normalized =
            layer_norm_3d_no_affine_with_runtime_context(hidden, 1e-6, runtime_context)?;
        let mut scale = CpuTensor::zeros(&[batch, width]);
        let mut shift = CpuTensor::zeros(&[batch, width]);
        for b in 0..batch {
            let src = b * projected_width;
            let dst = b * width;
            scale.data[dst..dst + width].copy_from_slice(&projected.data[src..src + width]);
            shift.data[dst..dst + width]
                .copy_from_slice(&projected.data[src + width..src + projected_width]);
        }
        modulate_3d(&normalized, &shift, &scale)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct NativeTransformerTimestepEmbedding {
    family: TransformerDenoiserFamily,
    linear_1_weight: CpuTensor,
    linear_1_bias: CpuTensor,
    linear_2_weight: CpuTensor,
    linear_2_bias: CpuTensor,
    modulation_weight: Option<CpuTensor>,
    modulation_bias: Option<CpuTensor>,
}

#[allow(dead_code)]
impl NativeTransformerTimestepEmbedding {
    fn from_hfq(hfq: &HfqFile, family: TransformerDenoiserFamily) -> DiffusionResult<Self> {
        let prefix = match family {
            TransformerDenoiserFamily::Krea2 => "transformer/tensors/time_embed",
            TransformerDenoiserFamily::QwenImage | TransformerDenoiserFamily::Unknown => {
                "transformer/tensors/time_text_embed.timestep_embedder"
            }
        };
        let modulation_weight = optional_tensor(hfq, "transformer/tensors/time_mod_proj.weight")?;
        let modulation_bias = if modulation_weight.is_some() {
            Some(CpuTensor::from_hfq(
                hfq,
                "transformer/tensors/time_mod_proj.bias",
            )?)
        } else {
            None
        };
        Ok(Self {
            family,
            linear_1_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.linear_1.weight"))?,
            linear_1_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.linear_1.bias"))?,
            linear_2_weight: CpuTensor::from_hfq(hfq, &format!("{prefix}.linear_2.weight"))?,
            linear_2_bias: CpuTensor::from_hfq(hfq, &format!("{prefix}.linear_2.bias"))?,
            modulation_weight,
            modulation_bias,
        })
    }

    fn embedding_dim(&self) -> DiffusionResult<usize> {
        let (_, embedding_dim) = self.linear_1_weight.rows_cols()?;
        Ok(embedding_dim)
    }

    fn forward_with_runtime_context(
        &self,
        timesteps: &[f32],
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let input = timestep_embedding_with_runtime_context(
            timesteps,
            self.embedding_dim()?,
            true,
            0.0,
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

    fn modulation_with_runtime_context(
        &self,
        timestep_embedding: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<Option<CpuTensor>> {
        let Some(weight) = self.modulation_weight.as_ref() else {
            return Ok(None);
        };
        let bias = self.modulation_bias.as_ref().ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "transformer time_mod_proj weight is present but bias is missing".to_string(),
            )
        })?;
        linear_with_runtime_context(timestep_embedding, weight, bias, runtime_context).map(Some)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct TransformerModulationChunks {
    shift_msa: CpuTensor,
    scale_msa: CpuTensor,
    gate_msa: CpuTensor,
    shift_mlp: CpuTensor,
    scale_mlp: CpuTensor,
    gate_mlp: CpuTensor,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct NativeTransformerBlockModulation {
    family: TransformerDenoiserFamily,
    block_index: usize,
    hidden_width: usize,
    img_mod_weight: Option<CpuTensor>,
    img_mod_bias: Option<CpuTensor>,
    txt_mod_weight: Option<CpuTensor>,
    txt_mod_bias: Option<CpuTensor>,
    scale_shift_table: Option<CpuTensor>,
}

#[allow(dead_code)]
impl NativeTransformerBlockModulation {
    fn from_hfq(
        hfq: &HfqFile,
        family: TransformerDenoiserFamily,
        block_index: usize,
    ) -> DiffusionResult<Self> {
        let block_prefix = format!("transformer/tensors/transformer_blocks.{block_index}");
        match family {
            TransformerDenoiserFamily::Krea2 => {
                let scale_shift_table =
                    CpuTensor::from_hfq(hfq, &format!("{block_prefix}.scale_shift_table"))?;
                let [chunks, hidden_width] = shape2(&scale_shift_table)?;
                if chunks == 0 || hidden_width == 0 {
                    return Err(DiffusionError::InvalidMetadata(format!(
                        "Krea transformer block {block_index} scale_shift_table shape {:?} is empty",
                        scale_shift_table.shape
                    )));
                }
                Ok(Self {
                    family,
                    block_index,
                    hidden_width,
                    img_mod_weight: None,
                    img_mod_bias: None,
                    txt_mod_weight: None,
                    txt_mod_bias: None,
                    scale_shift_table: Some(scale_shift_table),
                })
            }
            TransformerDenoiserFamily::QwenImage | TransformerDenoiserFamily::Unknown => {
                let img_mod_weight =
                    CpuTensor::from_hfq(hfq, &format!("{block_prefix}.img_mod.1.weight"))?;
                let img_mod_bias =
                    CpuTensor::from_hfq(hfq, &format!("{block_prefix}.img_mod.1.bias"))?;
                let txt_mod_weight =
                    CpuTensor::from_hfq(hfq, &format!("{block_prefix}.txt_mod.1.weight"))?;
                let txt_mod_bias =
                    CpuTensor::from_hfq(hfq, &format!("{block_prefix}.txt_mod.1.bias"))?;
                let [img_rows, hidden_width] = shape2(&img_mod_weight)?;
                let [txt_rows, txt_hidden_width] = shape2(&txt_mod_weight)?;
                if hidden_width == 0 || img_rows != hidden_width * 6 {
                    return Err(DiffusionError::InvalidMetadata(format!(
                        "Qwen transformer block {block_index} img_mod weight shape {:?} is not [6*hidden, hidden]",
                        img_mod_weight.shape
                    )));
                }
                if txt_hidden_width != hidden_width || txt_rows != hidden_width * 6 {
                    return Err(DiffusionError::InvalidMetadata(format!(
                        "Qwen transformer block {block_index} txt_mod weight shape {:?} does not match hidden width {hidden_width}",
                        txt_mod_weight.shape
                    )));
                }
                if img_mod_bias.shape.as_slice() != [img_rows] {
                    return Err(DiffusionError::InvalidMetadata(format!(
                        "Qwen transformer block {block_index} img_mod bias shape {:?} != [{img_rows}]",
                        img_mod_bias.shape
                    )));
                }
                if txt_mod_bias.shape.as_slice() != [txt_rows] {
                    return Err(DiffusionError::InvalidMetadata(format!(
                        "Qwen transformer block {block_index} txt_mod bias shape {:?} != [{txt_rows}]",
                        txt_mod_bias.shape
                    )));
                }
                Ok(Self {
                    family,
                    block_index,
                    hidden_width,
                    img_mod_weight: Some(img_mod_weight),
                    img_mod_bias: Some(img_mod_bias),
                    txt_mod_weight: Some(txt_mod_weight),
                    txt_mod_bias: Some(txt_mod_bias),
                    scale_shift_table: None,
                })
            }
        }
    }

    fn qwen_image_modulation_with_runtime_context(
        &self,
        timestep_embedding: &CpuTensor,
        stream: TransformerModulationStream,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<TransformerModulationChunks> {
        if !matches!(
            self.family,
            TransformerDenoiserFamily::QwenImage | TransformerDenoiserFamily::Unknown
        ) {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer block {} family {:?} does not use Qwen image/text modulation",
                self.block_index, self.family
            )));
        }
        let (weight, bias) = match stream {
            TransformerModulationStream::Image => {
                (self.img_mod_weight.as_ref(), self.img_mod_bias.as_ref())
            }
            TransformerModulationStream::Text => {
                (self.txt_mod_weight.as_ref(), self.txt_mod_bias.as_ref())
            }
        };
        let weight = weight.ok_or_else(|| {
            DiffusionError::InvalidMetadata("Qwen transformer modulation weight is missing".into())
        })?;
        let bias = bias.ok_or_else(|| {
            DiffusionError::InvalidMetadata("Qwen transformer modulation bias is missing".into())
        })?;
        let activated = silu_with_runtime_context(timestep_embedding, runtime_context)?;
        let projected = linear_with_runtime_context(&activated, weight, bias, runtime_context)?;
        split_modulation_chunks(projected, 6)
    }

    fn krea_scale_shift_with_runtime_context(
        &self,
        time_modulation: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let _ = runtime_context;
        let table = self.scale_shift_table.as_ref().ok_or_else(|| {
            DiffusionError::InvalidMetadata("Krea transformer scale_shift_table is missing".into())
        })?;
        let [chunks, hidden_width] = shape2(table)?;
        let [batch, width] = shape2(time_modulation)?;
        if hidden_width != self.hidden_width {
            return Err(DiffusionError::InvalidMetadata(format!(
                "Krea transformer block {} hidden width drifted from {} to {hidden_width}",
                self.block_index, self.hidden_width
            )));
        }
        if width != chunks * hidden_width {
            return Err(DiffusionError::InvalidMetadata(format!(
                "Krea time modulation width {width} != scale_shift_table chunks*hidden {}",
                chunks * hidden_width
            )));
        }
        let mut out = CpuTensor::zeros(&[batch, chunks, hidden_width]);
        for b in 0..batch {
            for chunk in 0..chunks {
                for hidden in 0..hidden_width {
                    let flat = chunk * hidden_width + hidden;
                    out.data[(b * chunks + chunk) * hidden_width + hidden] = time_modulation.data
                        [b * width + flat]
                        + table.data[chunk * hidden_width + hidden];
                }
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct TransformerAttentionQkv {
    q: CpuTensor,
    k: CpuTensor,
    v: CpuTensor,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct TransformerAttentionStreamProjection {
    stream_label: &'static str,
    q_weight: CpuTensor,
    q_bias: Option<CpuTensor>,
    k_weight: CpuTensor,
    k_bias: Option<CpuTensor>,
    v_weight: CpuTensor,
    v_bias: Option<CpuTensor>,
    norm_q_weight: Option<CpuTensor>,
    norm_k_weight: Option<CpuTensor>,
    out_weight: CpuTensor,
    out_bias: Option<CpuTensor>,
}

#[allow(dead_code)]
impl TransformerAttentionStreamProjection {
    #[allow(clippy::too_many_arguments)]
    fn from_hfq(
        hfq: &HfqFile,
        stream_label: &'static str,
        q_weight_entry: &str,
        q_bias_entry: &str,
        k_weight_entry: &str,
        k_bias_entry: &str,
        v_weight_entry: &str,
        v_bias_entry: &str,
        norm_q_entry: &str,
        norm_k_entry: &str,
        out_weight_entry: &str,
        out_bias_entry: &str,
        required: bool,
        heads: usize,
        expected_hidden_width: Option<usize>,
        expected_inner_width: Option<usize>,
        expected_head_dim: Option<usize>,
    ) -> DiffusionResult<Option<Self>> {
        if hfq.find_tensor_info(q_weight_entry).is_none() {
            if required {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "{stream_label} transformer attention q projection {q_weight_entry:?} is missing"
                )));
            }
            return Ok(None);
        }

        let stream = Self {
            stream_label,
            q_weight: CpuTensor::from_hfq(hfq, q_weight_entry)?,
            q_bias: optional_tensor(hfq, q_bias_entry)?,
            k_weight: CpuTensor::from_hfq(hfq, k_weight_entry)?,
            k_bias: optional_tensor(hfq, k_bias_entry)?,
            v_weight: CpuTensor::from_hfq(hfq, v_weight_entry)?,
            v_bias: optional_tensor(hfq, v_bias_entry)?,
            norm_q_weight: optional_tensor(hfq, norm_q_entry)?,
            norm_k_weight: optional_tensor(hfq, norm_k_entry)?,
            out_weight: CpuTensor::from_hfq(hfq, out_weight_entry)?,
            out_bias: optional_tensor(hfq, out_bias_entry)?,
        };
        stream.validate_shapes(
            heads,
            expected_hidden_width,
            expected_inner_width,
            expected_head_dim,
        )?;
        Ok(Some(stream))
    }

    fn validate_shapes(
        &self,
        heads: usize,
        expected_hidden_width: Option<usize>,
        expected_inner_width: Option<usize>,
        expected_head_dim: Option<usize>,
    ) -> DiffusionResult<(usize, usize, usize)> {
        if heads == 0 {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{} transformer attention heads must be positive",
                self.stream_label
            )));
        }
        let [inner_width, hidden_width] = shape2(&self.q_weight)?;
        if inner_width == 0 || hidden_width == 0 {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{} transformer attention q weight shape {:?} is empty",
                self.stream_label, self.q_weight.shape
            )));
        }
        if let Some(expected) = expected_hidden_width {
            if hidden_width != expected {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "{} transformer attention hidden width {hidden_width} != expected {expected}",
                    self.stream_label
                )));
            }
        }
        if let Some(expected) = expected_inner_width {
            if inner_width != expected {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "{} transformer attention inner width {inner_width} != expected {expected}",
                    self.stream_label
                )));
            }
        }
        let head_dim = self
            .norm_q_weight
            .as_ref()
            .or(self.norm_k_weight.as_ref())
            .map(attention_norm_weight_dim)
            .transpose()?
            .or(expected_head_dim)
            .unwrap_or_else(|| inner_width / heads);
        if head_dim == 0 || inner_width != heads * head_dim {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{} transformer attention inner width {inner_width} is incompatible with heads {heads} and head_dim {head_dim}",
                self.stream_label
            )));
        }
        validate_attention_linear_shape(
            self.stream_label,
            "k",
            &self.k_weight,
            inner_width,
            hidden_width,
        )?;
        validate_attention_linear_shape(
            self.stream_label,
            "v",
            &self.v_weight,
            inner_width,
            hidden_width,
        )?;
        validate_attention_bias_shape(self.stream_label, "q", self.q_bias.as_ref(), inner_width)?;
        validate_attention_bias_shape(self.stream_label, "k", self.k_bias.as_ref(), inner_width)?;
        validate_attention_bias_shape(self.stream_label, "v", self.v_bias.as_ref(), inner_width)?;
        validate_attention_norm_shape(
            self.stream_label,
            "q",
            self.norm_q_weight.as_ref(),
            head_dim,
        )?;
        validate_attention_norm_shape(
            self.stream_label,
            "k",
            self.norm_k_weight.as_ref(),
            head_dim,
        )?;
        if self.out_weight.shape.as_slice() != [hidden_width, inner_width] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{} transformer attention output weight shape {:?} != [{hidden_width}, {inner_width}]",
                self.stream_label, self.out_weight.shape
            )));
        }
        validate_attention_bias_shape(
            self.stream_label,
            "out",
            self.out_bias.as_ref(),
            hidden_width,
        )?;
        Ok((hidden_width, inner_width, head_dim))
    }

    fn project_qkv_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        heads: usize,
        head_dim: usize,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<TransformerAttentionQkv> {
        let q = linear_3d_with_runtime_context(
            hidden,
            &self.q_weight,
            self.q_bias.as_ref(),
            runtime_context,
        )?;
        let k = linear_3d_with_runtime_context(
            hidden,
            &self.k_weight,
            self.k_bias.as_ref(),
            runtime_context,
        )?;
        let v = linear_3d_with_runtime_context(
            hidden,
            &self.v_weight,
            self.v_bias.as_ref(),
            runtime_context,
        )?;
        let q = maybe_rms_norm_attention_heads_3d(
            q,
            self.norm_q_weight.as_ref(),
            heads,
            head_dim,
            1e-6,
        )?;
        let k = maybe_rms_norm_attention_heads_3d(
            k,
            self.norm_k_weight.as_ref(),
            heads,
            head_dim,
            1e-6,
        )?;
        Ok(TransformerAttentionQkv { q, k, v })
    }

    fn project_output_with_runtime_context(
        &self,
        attention: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        linear_3d_with_runtime_context(
            attention,
            &self.out_weight,
            self.out_bias.as_ref(),
            runtime_context,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct NativeTransformerAttentionProjection {
    family: TransformerDenoiserFamily,
    block_index: usize,
    heads: usize,
    head_dim: usize,
    hidden_width: usize,
    inner_width: usize,
    image: TransformerAttentionStreamProjection,
    text: Option<TransformerAttentionStreamProjection>,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct RotaryFrequencies {
    cos: CpuTensor,
    sin: CpuTensor,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct QwenRotaryEmbeddings {
    image: RotaryFrequencies,
    text: RotaryFrequencies,
}

#[allow(dead_code)]
impl NativeTransformerAttentionProjection {
    fn from_hfq(
        hfq: &HfqFile,
        family: TransformerDenoiserFamily,
        block_index: usize,
        heads: usize,
    ) -> DiffusionResult<Self> {
        let block_prefix = format!("transformer/tensors/transformer_blocks.{block_index}.attn");
        let image = TransformerAttentionStreamProjection::from_hfq(
            hfq,
            "image",
            &format!("{block_prefix}.to_q.weight"),
            &format!("{block_prefix}.to_q.bias"),
            &format!("{block_prefix}.to_k.weight"),
            &format!("{block_prefix}.to_k.bias"),
            &format!("{block_prefix}.to_v.weight"),
            &format!("{block_prefix}.to_v.bias"),
            &format!("{block_prefix}.norm_q.weight"),
            &format!("{block_prefix}.norm_k.weight"),
            &format!("{block_prefix}.to_out.0.weight"),
            &format!("{block_prefix}.to_out.0.bias"),
            true,
            heads,
            None,
            None,
            None,
        )?
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata(format!(
                "transformer block {block_index} image attention stream is missing"
            ))
        })?;
        let (hidden_width, inner_width, head_dim) =
            image.validate_shapes(heads, None, None, None)?;

        let text_required = matches!(
            family,
            TransformerDenoiserFamily::QwenImage | TransformerDenoiserFamily::Unknown
        );
        let text = TransformerAttentionStreamProjection::from_hfq(
            hfq,
            "text",
            &format!("{block_prefix}.add_q_proj.weight"),
            &format!("{block_prefix}.add_q_proj.bias"),
            &format!("{block_prefix}.add_k_proj.weight"),
            &format!("{block_prefix}.add_k_proj.bias"),
            &format!("{block_prefix}.add_v_proj.weight"),
            &format!("{block_prefix}.add_v_proj.bias"),
            &format!("{block_prefix}.norm_added_q.weight"),
            &format!("{block_prefix}.norm_added_k.weight"),
            &format!("{block_prefix}.to_add_out.weight"),
            &format!("{block_prefix}.to_add_out.bias"),
            text_required,
            heads,
            Some(hidden_width),
            Some(inner_width),
            Some(head_dim),
        )?;

        Ok(Self {
            family,
            block_index,
            heads,
            head_dim,
            hidden_width,
            inner_width,
            image,
            text,
        })
    }

    fn project_image_qkv_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<TransformerAttentionQkv> {
        self.validate_hidden_input(hidden, TransformerModulationStream::Image)?;
        self.image.project_qkv_with_runtime_context(
            hidden,
            self.heads,
            self.head_dim,
            runtime_context,
        )
    }

    fn project_text_qkv_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<Option<TransformerAttentionQkv>> {
        self.validate_hidden_input(hidden, TransformerModulationStream::Text)?;
        let Some(text) = self.text.as_ref() else {
            return Ok(None);
        };
        text.project_qkv_with_runtime_context(hidden, self.heads, self.head_dim, runtime_context)
            .map(Some)
    }

    fn project_image_output_with_runtime_context(
        &self,
        attention: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        self.validate_attention_input(attention, TransformerModulationStream::Image)?;
        self.image
            .project_output_with_runtime_context(attention, runtime_context)
    }

    fn project_text_output_with_runtime_context(
        &self,
        attention: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<Option<CpuTensor>> {
        self.validate_attention_input(attention, TransformerModulationStream::Text)?;
        let Some(text) = self.text.as_ref() else {
            return Ok(None);
        };
        text.project_output_with_runtime_context(attention, runtime_context)
            .map(Some)
    }

    fn attend_image_text_with_runtime_context(
        &self,
        image_hidden: &CpuTensor,
        text_hidden: Option<&CpuTensor>,
        text_attention_mask: Option<&CpuTensor>,
        qwen_rotary: Option<&QwenRotaryEmbeddings>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<(CpuTensor, Option<CpuTensor>)> {
        let mut image_qkv =
            self.project_image_qkv_with_runtime_context(image_hidden, runtime_context)?;
        if let Some(rotary) = qwen_rotary {
            image_qkv.q = apply_qwen_rotary_embedding(
                &image_qkv.q,
                &rotary.image,
                self.heads,
                self.head_dim,
            )?;
            image_qkv.k = apply_qwen_rotary_embedding(
                &image_qkv.k,
                &rotary.image,
                self.heads,
                self.head_dim,
            )?;
        }
        let Some(text_projection) = self.text.as_ref() else {
            let image_attention = scaled_dot_product_attention_with_runtime_context(
                &image_qkv.q,
                &image_qkv.k,
                &image_qkv.v,
                self.heads,
                runtime_context,
            )?;
            let image_output =
                self.project_image_output_with_runtime_context(&image_attention, runtime_context)?;
            return Ok((image_output, None));
        };

        let text_hidden = text_hidden.ok_or_else(|| {
            DiffusionError::InvalidRequest(format!(
                "transformer block {} {:?} attention requires text hidden states",
                self.block_index, self.family
            ))
        })?;
        self.validate_hidden_input(text_hidden, TransformerModulationStream::Text)?;
        let mut text_qkv = text_projection.project_qkv_with_runtime_context(
            text_hidden,
            self.heads,
            self.head_dim,
            runtime_context,
        )?;
        if let Some(rotary) = qwen_rotary {
            text_qkv.q =
                apply_qwen_rotary_embedding(&text_qkv.q, &rotary.text, self.heads, self.head_dim)?;
            text_qkv.k =
                apply_qwen_rotary_embedding(&text_qkv.k, &rotary.text, self.heads, self.head_dim)?;
        }
        let joint_k = concat_sequence_3d(&text_qkv.k, &image_qkv.k)?;
        let joint_v = concat_sequence_3d(&text_qkv.v, &image_qkv.v)?;
        let [batch, image_seq, _] = shape3(&image_qkv.k)?;
        let [text_batch, text_seq, _] = shape3(&text_qkv.k)?;
        if text_batch != batch {
            return Err(DiffusionError::InvalidMetadata(format!(
                "Qwen joint attention text batch {text_batch} != image batch {batch}"
            )));
        }
        let joint_key_mask = qwen_joint_key_mask(text_attention_mask, batch, text_seq, image_seq)?;
        let image_attention = scaled_dot_product_attention_with_key_mask_and_runtime_context(
            &image_qkv.q,
            &joint_k,
            &joint_v,
            self.heads,
            joint_key_mask.as_deref(),
            runtime_context,
        )?;
        let text_attention = scaled_dot_product_attention_with_key_mask_and_runtime_context(
            &text_qkv.q,
            &joint_k,
            &joint_v,
            self.heads,
            joint_key_mask.as_deref(),
            runtime_context,
        )?;
        let image_output =
            self.project_image_output_with_runtime_context(&image_attention, runtime_context)?;
        let text_output = text_projection
            .project_output_with_runtime_context(&text_attention, runtime_context)?;
        Ok((image_output, Some(text_output)))
    }

    fn validate_hidden_input(
        &self,
        hidden: &CpuTensor,
        stream: TransformerModulationStream,
    ) -> DiffusionResult<()> {
        let [_, _, width] = shape3(hidden)?;
        if width != self.hidden_width {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer block {} {:?} hidden width {width} != expected {}",
                self.block_index, stream, self.hidden_width
            )));
        }
        Ok(())
    }

    fn validate_attention_input(
        &self,
        attention: &CpuTensor,
        stream: TransformerModulationStream,
    ) -> DiffusionResult<()> {
        let [_, _, width] = shape3(attention)?;
        if width != self.inner_width {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer block {} {:?} attention width {width} != expected {}",
                self.block_index, stream, self.inner_width
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum TransformerFeedForwardActivation {
    GeGlu,
    SwiGlu,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct TransformerFeedForwardStream {
    stream_label: &'static str,
    activation: TransformerFeedForwardActivation,
    hidden_width: usize,
    inner_width: usize,
    proj_weight: Option<CpuTensor>,
    proj_bias: Option<CpuTensor>,
    up_weight: Option<CpuTensor>,
    up_bias: Option<CpuTensor>,
    gate_weight: Option<CpuTensor>,
    gate_bias: Option<CpuTensor>,
    down_weight: CpuTensor,
    down_bias: Option<CpuTensor>,
}

#[allow(dead_code)]
impl TransformerFeedForwardStream {
    fn qwen_geglu_from_hfq(
        hfq: &HfqFile,
        stream_label: &'static str,
        prefix: &str,
    ) -> DiffusionResult<Self> {
        let proj_weight = CpuTensor::from_hfq(hfq, &format!("{prefix}.net.0.proj.weight"))?;
        let proj_bias = CpuTensor::from_hfq(hfq, &format!("{prefix}.net.0.proj.bias"))?;
        let down_weight = CpuTensor::from_hfq(hfq, &format!("{prefix}.net.2.weight"))?;
        let down_bias = CpuTensor::from_hfq(hfq, &format!("{prefix}.net.2.bias"))?;
        let [projected_width, hidden_width] = shape2(&proj_weight)?;
        if projected_width == 0 || projected_width % 2 != 0 {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{stream_label} transformer GEGLU projection shape {:?} is not [2*inner, hidden]",
                proj_weight.shape
            )));
        }
        let inner_width = projected_width / 2;
        if proj_bias.shape.as_slice() != [projected_width] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{stream_label} transformer GEGLU projection bias shape {:?} != [{projected_width}]",
                proj_bias.shape
            )));
        }
        validate_transformer_ff_down_shape(
            stream_label,
            &down_weight,
            Some(&down_bias),
            hidden_width,
            inner_width,
        )?;
        Ok(Self {
            stream_label,
            activation: TransformerFeedForwardActivation::GeGlu,
            hidden_width,
            inner_width,
            proj_weight: Some(proj_weight),
            proj_bias: Some(proj_bias),
            up_weight: None,
            up_bias: None,
            gate_weight: None,
            gate_bias: None,
            down_weight,
            down_bias: Some(down_bias),
        })
    }

    fn krea_swiglu_from_hfq(
        hfq: &HfqFile,
        stream_label: &'static str,
        prefix: &str,
    ) -> DiffusionResult<Self> {
        let up_weight = CpuTensor::from_hfq(hfq, &format!("{prefix}.up.weight"))?;
        let gate_weight = CpuTensor::from_hfq(hfq, &format!("{prefix}.gate.weight"))?;
        let down_weight = CpuTensor::from_hfq(hfq, &format!("{prefix}.down.weight"))?;
        let up_bias = optional_tensor(hfq, &format!("{prefix}.up.bias"))?;
        let gate_bias = optional_tensor(hfq, &format!("{prefix}.gate.bias"))?;
        let down_bias = optional_tensor(hfq, &format!("{prefix}.down.bias"))?;
        let [inner_width, hidden_width] = shape2(&up_weight)?;
        if inner_width == 0 || hidden_width == 0 {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{stream_label} transformer SwiGLU up projection shape {:?} is empty",
                up_weight.shape
            )));
        }
        if gate_weight.shape.as_slice() != [inner_width, hidden_width] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{stream_label} transformer SwiGLU gate weight shape {:?} != [{inner_width}, {hidden_width}]",
                gate_weight.shape
            )));
        }
        validate_attention_bias_shape(stream_label, "ff.up", up_bias.as_ref(), inner_width)?;
        validate_attention_bias_shape(stream_label, "ff.gate", gate_bias.as_ref(), inner_width)?;
        validate_transformer_ff_down_shape(
            stream_label,
            &down_weight,
            down_bias.as_ref(),
            hidden_width,
            inner_width,
        )?;
        Ok(Self {
            stream_label,
            activation: TransformerFeedForwardActivation::SwiGlu,
            hidden_width,
            inner_width,
            proj_weight: None,
            proj_bias: None,
            up_weight: Some(up_weight),
            up_bias,
            gate_weight: Some(gate_weight),
            gate_bias,
            down_weight,
            down_bias,
        })
    }

    fn forward_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let [_, _, width] = shape3(hidden)?;
        if width != self.hidden_width {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{} transformer feed-forward hidden width {width} != expected {}",
                self.stream_label, self.hidden_width
            )));
        }
        let activated = match self.activation {
            TransformerFeedForwardActivation::GeGlu => {
                let proj_weight = self.proj_weight.as_ref().ok_or_else(|| {
                    DiffusionError::InvalidMetadata(
                        "GEGLU transformer feed-forward projection weight is missing".into(),
                    )
                })?;
                let proj_bias = self.proj_bias.as_ref().ok_or_else(|| {
                    DiffusionError::InvalidMetadata(
                        "GEGLU transformer feed-forward projection bias is missing".into(),
                    )
                })?;
                let projected = linear_3d_with_runtime_context(
                    hidden,
                    proj_weight,
                    Some(proj_bias),
                    runtime_context,
                )?;
                geglu_gate_3d_with_runtime_context(&projected, runtime_context)?
            }
            TransformerFeedForwardActivation::SwiGlu => {
                let up_weight = self.up_weight.as_ref().ok_or_else(|| {
                    DiffusionError::InvalidMetadata(
                        "SwiGLU transformer feed-forward up weight is missing".into(),
                    )
                })?;
                let gate_weight = self.gate_weight.as_ref().ok_or_else(|| {
                    DiffusionError::InvalidMetadata(
                        "SwiGLU transformer feed-forward gate weight is missing".into(),
                    )
                })?;
                let up = linear_3d_with_runtime_context(
                    hidden,
                    up_weight,
                    self.up_bias.as_ref(),
                    runtime_context,
                )?;
                let gate = linear_3d_with_runtime_context(
                    hidden,
                    gate_weight,
                    self.gate_bias.as_ref(),
                    runtime_context,
                )?;
                swiglu_gate_3d(&up, &gate)?
            }
        };
        linear_3d_with_runtime_context(
            &activated,
            &self.down_weight,
            self.down_bias.as_ref(),
            runtime_context,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct NativeTransformerFeedForward {
    family: TransformerDenoiserFamily,
    block_index: usize,
    hidden_width: usize,
    image: TransformerFeedForwardStream,
    text: Option<TransformerFeedForwardStream>,
}

#[allow(dead_code)]
impl NativeTransformerFeedForward {
    fn from_hfq(
        hfq: &HfqFile,
        family: TransformerDenoiserFamily,
        block_index: usize,
    ) -> DiffusionResult<Self> {
        let block_prefix = format!("transformer/tensors/transformer_blocks.{block_index}");
        match family {
            TransformerDenoiserFamily::Krea2 => {
                let image = TransformerFeedForwardStream::krea_swiglu_from_hfq(
                    hfq,
                    "image",
                    &format!("{block_prefix}.ff"),
                )?;
                Ok(Self {
                    family,
                    block_index,
                    hidden_width: image.hidden_width,
                    image,
                    text: None,
                })
            }
            TransformerDenoiserFamily::QwenImage | TransformerDenoiserFamily::Unknown => {
                let image = TransformerFeedForwardStream::qwen_geglu_from_hfq(
                    hfq,
                    "image",
                    &format!("{block_prefix}.img_mlp"),
                )?;
                let text = TransformerFeedForwardStream::qwen_geglu_from_hfq(
                    hfq,
                    "text",
                    &format!("{block_prefix}.txt_mlp"),
                )?;
                if text.hidden_width != image.hidden_width {
                    return Err(DiffusionError::InvalidMetadata(format!(
                        "Qwen transformer block {block_index} text MLP hidden width {} != image hidden width {}",
                        text.hidden_width, image.hidden_width
                    )));
                }
                Ok(Self {
                    family,
                    block_index,
                    hidden_width: image.hidden_width,
                    image,
                    text: Some(text),
                })
            }
        }
    }

    fn forward_image_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        self.image
            .forward_with_runtime_context(hidden, runtime_context)
    }

    fn forward_text_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<Option<CpuTensor>> {
        let Some(text) = self.text.as_ref() else {
            return Ok(None);
        };
        text.forward_with_runtime_context(hidden, runtime_context)
            .map(Some)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct NativeTransformerBlock {
    family: TransformerDenoiserFamily,
    block_index: usize,
    modulation: NativeTransformerBlockModulation,
    attention: NativeTransformerAttentionProjection,
    feed_forward: NativeTransformerFeedForward,
}

#[allow(dead_code)]
impl NativeTransformerBlock {
    fn from_hfq(
        hfq: &HfqFile,
        family: TransformerDenoiserFamily,
        block_index: usize,
        heads: usize,
    ) -> DiffusionResult<Self> {
        Ok(Self {
            family,
            block_index,
            modulation: NativeTransformerBlockModulation::from_hfq(hfq, family, block_index)?,
            attention: NativeTransformerAttentionProjection::from_hfq(
                hfq,
                family,
                block_index,
                heads,
            )?,
            feed_forward: NativeTransformerFeedForward::from_hfq(hfq, family, block_index)?,
        })
    }

    fn forward_qwen_with_runtime_context(
        &self,
        image_hidden: &CpuTensor,
        text_hidden: &CpuTensor,
        text_attention_mask: Option<&CpuTensor>,
        timestep_embedding: &CpuTensor,
        qwen_rotary: Option<&QwenRotaryEmbeddings>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<(CpuTensor, CpuTensor)> {
        if !matches!(
            self.family,
            TransformerDenoiserFamily::QwenImage | TransformerDenoiserFamily::Unknown
        ) {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer block {} family {:?} is not Qwen-style",
                self.block_index, self.family
            )));
        }

        let image_mod = self.modulation.qwen_image_modulation_with_runtime_context(
            timestep_embedding,
            TransformerModulationStream::Image,
            runtime_context,
        )?;
        let text_mod = self.modulation.qwen_image_modulation_with_runtime_context(
            timestep_embedding,
            TransformerModulationStream::Text,
            runtime_context,
        )?;

        let image_attention_input = modulate_3d(
            &layer_norm_3d_no_affine_with_runtime_context(image_hidden, 1e-6, runtime_context)?,
            &image_mod.shift_msa,
            &image_mod.scale_msa,
        )?;
        let text_attention_input = modulate_3d(
            &layer_norm_3d_no_affine_with_runtime_context(text_hidden, 1e-6, runtime_context)?,
            &text_mod.shift_msa,
            &text_mod.scale_msa,
        )?;
        let (image_attention, text_attention) =
            self.attention.attend_image_text_with_runtime_context(
                &image_attention_input,
                Some(&text_attention_input),
                text_attention_mask,
                qwen_rotary,
                runtime_context,
            )?;
        let text_attention = text_attention.ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "Qwen transformer block attention returned no text stream".to_string(),
            )
        })?;
        let image_after_attention =
            gated_residual_3d(image_hidden, &image_attention, &image_mod.gate_msa)?;
        let text_after_attention =
            gated_residual_3d(text_hidden, &text_attention, &text_mod.gate_msa)?;

        let image_mlp_input = modulate_3d(
            &layer_norm_3d_no_affine_with_runtime_context(
                &image_after_attention,
                1e-6,
                runtime_context,
            )?,
            &image_mod.shift_mlp,
            &image_mod.scale_mlp,
        )?;
        let text_mlp_input = modulate_3d(
            &layer_norm_3d_no_affine_with_runtime_context(
                &text_after_attention,
                1e-6,
                runtime_context,
            )?,
            &text_mod.shift_mlp,
            &text_mod.scale_mlp,
        )?;
        let image_mlp = self
            .feed_forward
            .forward_image_with_runtime_context(&image_mlp_input, runtime_context)?;
        let text_mlp = self
            .feed_forward
            .forward_text_with_runtime_context(&text_mlp_input, runtime_context)?
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata(
                    "Qwen transformer block feed-forward returned no text stream".to_string(),
                )
            })?;
        let image_out = gated_residual_3d(&image_after_attention, &image_mlp, &image_mod.gate_mlp)?;
        let text_out = gated_residual_3d(&text_after_attention, &text_mlp, &text_mod.gate_mlp)?;
        Ok((image_out, text_out))
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
struct NativeTransformerDenoiser {
    family: TransformerDenoiserFamily,
    io: NativeTransformerDenoiserIo,
    timestep_embedding: NativeTransformerTimestepEmbedding,
    blocks: Vec<NativeTransformerBlock>,
    heads: usize,
    qwen_rope_axes: Option<[usize; 3]>,
    qwen_rope_theta: f32,
}

#[allow(dead_code)]
impl NativeTransformerDenoiser {
    fn from_hfq(
        hfq: &HfqFile,
        config: &StableDiffusionConfig,
        topology: &TransformerDenoiserWeightTopology,
    ) -> DiffusionResult<Self> {
        if !matches!(topology.family, TransformerDenoiserFamily::QwenImage) {
            return Err(DiffusionError::InvalidMetadata(format!(
                "native transformer denoiser assembly currently supports Qwen image MMDiT only; got {}",
                topology.diagnostic_label()
            )));
        }
        if topology.block_count == 0 {
            return Err(DiffusionError::InvalidMetadata(
                "transformer denoiser contains no transformer_blocks.* weights".to_string(),
            ));
        }
        let transformer = config.transformer.as_ref().ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "transformer denoiser config is required for native transformer assembly"
                    .to_string(),
            )
        })?;
        if transformer.guidance_embeds.unwrap_or(false) {
            return Err(DiffusionError::InvalidMetadata(
                "Qwen guidance-distilled transformer embeddings are not implemented; guidance_embeds=true needs a separate guidance-scale embedding path, not classifier-free guidance".to_string(),
            ));
        }
        let heads = transformer.num_attention_heads.unwrap_or(1);
        if heads == 0 {
            return Err(DiffusionError::InvalidMetadata(
                "transformer num_attention_heads must be positive".to_string(),
            ));
        }
        let io = NativeTransformerDenoiserIo::from_hfq(hfq, config, topology)?;
        let timestep_embedding =
            NativeTransformerTimestepEmbedding::from_hfq(hfq, topology.family)?;
        let mut blocks = Vec::with_capacity(topology.block_count);
        for block_index in 0..topology.block_count {
            blocks.push(NativeTransformerBlock::from_hfq(
                hfq,
                topology.family,
                block_index,
                heads,
            )?);
        }
        let head_dim = blocks
            .first()
            .map(|block| block.attention.head_dim)
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata(
                    "transformer denoiser contains no attention blocks".to_string(),
                )
            })?;
        let qwen_rope_axes = qwen_rope_axes_from_transformer_config(transformer, head_dim)?;
        let qwen_rope_theta = transformer.rope_theta.unwrap_or(10_000.0);
        Ok(Self {
            family: topology.family,
            io,
            timestep_embedding,
            blocks,
            heads,
            qwen_rope_axes,
            qwen_rope_theta,
        })
    }

    fn forward_qwen_with_runtime_context(
        &self,
        latents: &LatentBatch,
        timesteps: &[f32],
        text_hidden: &CpuTensor,
        text_attention_mask: Option<&CpuTensor>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<LatentBatch> {
        if !matches!(self.family, TransformerDenoiserFamily::QwenImage) {
            return Err(DiffusionError::InvalidMetadata(format!(
                "native transformer denoiser family {:?} is not Qwen image MMDiT",
                self.family
            )));
        }
        if timesteps.len() != latents.batch {
            return Err(DiffusionError::InvalidRequest(format!(
                "transformer timestep batch {} != latent batch {}",
                timesteps.len(),
                latents.batch
            )));
        }
        let [text_batch, _, _] = shape3(text_hidden)?;
        if text_batch != latents.batch {
            return Err(DiffusionError::InvalidRequest(format!(
                "transformer text hidden batch {text_batch} != latent batch {}",
                latents.batch
            )));
        }

        let mut image_hidden = self
            .io
            .project_latents_to_hidden_with_runtime_context(latents, runtime_context)?;
        let mut text_hidden = self
            .io
            .project_text_to_hidden_with_runtime_context(text_hidden, runtime_context)?;
        let [_, text_seq, _] = shape3(&text_hidden)?;
        if let Some(mask) = text_attention_mask {
            validate_text_attention_mask(mask, latents.batch, text_seq, "Qwen text")?;
        }
        let qwen_rotary = self.qwen_rotary_embeddings(latents, text_seq)?;
        let timestep_embedding = self
            .timestep_embedding
            .forward_with_runtime_context(timesteps, runtime_context)?;
        for block in &self.blocks {
            let (next_image_hidden, next_text_hidden) = block.forward_qwen_with_runtime_context(
                &image_hidden,
                &text_hidden,
                text_attention_mask,
                &timestep_embedding,
                qwen_rotary.as_ref(),
                runtime_context,
            )?;
            image_hidden = next_image_hidden;
            text_hidden = next_text_hidden;
        }
        self.io.project_hidden_to_latents_with_runtime_context(
            &image_hidden,
            &timestep_embedding,
            latents.batch,
            latents.height,
            latents.width,
            runtime_context,
        )
    }

    fn qwen_rotary_embeddings(
        &self,
        latents: &LatentBatch,
        text_seq_len: usize,
    ) -> DiffusionResult<Option<QwenRotaryEmbeddings>> {
        let Some(axes) = self.qwen_rope_axes else {
            return Ok(None);
        };
        if latents.height % self.io.patch_size != 0 || latents.width % self.io.patch_size != 0 {
            return Err(DiffusionError::InvalidRequest(format!(
                "Qwen RoPE requires latent dimensions {}x{} to be divisible by patch size {}",
                latents.height, latents.width, self.io.patch_size
            )));
        }
        let grid_height = latents.height / self.io.patch_size;
        let grid_width = latents.width / self.io.patch_size;
        let image_seq_len = grid_height.checked_mul(grid_width).ok_or_else(|| {
            DiffusionError::InvalidRequest("Qwen RoPE image token count overflow".to_string())
        })?;
        if image_seq_len == 0 || text_seq_len == 0 {
            return Err(DiffusionError::InvalidRequest(
                "Qwen RoPE requires non-empty image and text token sequences".to_string(),
            ));
        }
        Ok(Some(qwen_rotary_embeddings_for_grid(
            axes,
            self.qwen_rope_theta,
            self.blocks[0].attention.head_dim,
            1,
            grid_height,
            grid_width,
            text_seq_len,
        )?))
    }
}

impl DiffusionNoiseBackend for NativeTransformerDenoiser {
    fn model_input_channels(&self) -> usize {
        self.io.output_channels
    }

    fn denoise_latents_with_runtime_context(
        &self,
        latents: LatentBatch,
        schedule: &DiffusionSchedule,
        cfg_scale: f32,
        positive_embeddings: &CpuTensor,
        negative_embeddings: &CpuTensor,
        positive_attention_mask: Option<&CpuTensor>,
        negative_attention_mask: Option<&CpuTensor>,
        positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
        inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
        masked_reference: Option<&MaskedDenoiseReference<'_>>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<DenoiseLatentsOutput> {
        if positive_sdxl_conditioning.is_some() || negative_sdxl_conditioning.is_some() {
            return Err(DiffusionError::InvalidRequest(
                "Qwen transformer denoiser does not accept SDXL auxiliary conditioning".to_string(),
            ));
        }
        if inpaint_conditioning.is_some() || masked_reference.is_some() {
            return Err(DiffusionError::InvalidRequest(
                "Qwen transformer denoiser inpaint conditioning is not implemented".to_string(),
            ));
        }
        denoise_latents_with_cfg_progress_and_runtime_context(
            latents,
            schedule,
            cfg_scale,
            positive_embeddings,
            negative_embeddings,
            |sample, timesteps, encoder_states, attention_mask, _sdxl, runtime_context| {
                let model_latents = LatentBatch::from_nchw_tensor(sample.clone())?;
                let prediction = self.forward_qwen_with_runtime_context(
                    &model_latents,
                    timesteps,
                    encoder_states,
                    attention_mask,
                    runtime_context,
                )?;
                Ok(prediction.as_nchw_tensor())
            },
            positive_attention_mask,
            negative_attention_mask,
            None,
            None,
            None,
            None,
            runtime_context,
            progress,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum TransformerModulationStream {
    Image,
    Text,
}

#[allow(dead_code)]
fn attention_norm_weight_dim(weight: &CpuTensor) -> DiffusionResult<usize> {
    match weight.shape.as_slice() {
        [dim] if *dim > 0 => Ok(*dim),
        _ => Err(DiffusionError::InvalidMetadata(format!(
            "transformer attention norm weight shape {:?} is not [head_dim]",
            weight.shape
        ))),
    }
}

#[allow(dead_code)]
fn qwen_rope_axes_from_transformer_config(
    transformer: &TransformerDenoiserConfig,
    head_dim: usize,
) -> DiffusionResult<Option<[usize; 3]>> {
    let axes = if transformer.axes_dims_rope.is_empty() {
        if head_dim == 128 {
            vec![16, 56, 56]
        } else {
            return Ok(None);
        }
    } else {
        transformer.axes_dims_rope.clone()
    };
    if axes.len() != 3 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Qwen axes_dims_rope {:?} must contain exactly 3 axes",
            axes
        )));
    }
    if axes.iter().any(|dim| *dim == 0 || dim % 2 != 0) {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Qwen axes_dims_rope {:?} must contain non-zero even dimensions",
            axes
        )));
    }
    let sum = axes.iter().sum::<usize>();
    if sum != head_dim {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Qwen axes_dims_rope {:?} sum {sum} != attention head_dim {head_dim}",
            axes
        )));
    }
    Ok(Some([axes[0], axes[1], axes[2]]))
}

#[allow(dead_code)]
fn qwen_rotary_embeddings_for_grid(
    axes: [usize; 3],
    theta: f32,
    head_dim: usize,
    frame: usize,
    height: usize,
    width: usize,
    text_seq_len: usize,
) -> DiffusionResult<QwenRotaryEmbeddings> {
    if frame == 0 || height == 0 || width == 0 || text_seq_len == 0 {
        return Err(DiffusionError::InvalidRequest(
            "Qwen RoPE requires non-empty frame, height, width, and text sequence".to_string(),
        ));
    }
    if axes.iter().sum::<usize>() != head_dim || head_dim % 2 != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Qwen RoPE axes {:?} are incompatible with head_dim {head_dim}",
            axes
        )));
    }
    if theta <= 0.0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Qwen rope_theta {theta} must be positive"
        )));
    }

    let freq_width = head_dim / 2;
    let image_seq_len = frame
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("Qwen RoPE image size overflow".to_string())
        })?;
    let mut image_cos = CpuTensor::zeros(&[image_seq_len, freq_width]);
    let mut image_sin = CpuTensor::zeros(&[image_seq_len, freq_width]);
    let max_vid_index = height.max(width);
    let mut token = 0usize;
    for f in 0..frame {
        for y in 0..height {
            for x in 0..width {
                write_qwen_rope_token(
                    &mut image_cos.data,
                    &mut image_sin.data,
                    token,
                    freq_width,
                    axes,
                    theta,
                    [f as isize, y as isize, x as isize],
                );
                token += 1;
            }
        }
    }

    let mut text_cos = CpuTensor::zeros(&[text_seq_len, freq_width]);
    let mut text_sin = CpuTensor::zeros(&[text_seq_len, freq_width]);
    for token in 0..text_seq_len {
        write_qwen_rope_token(
            &mut text_cos.data,
            &mut text_sin.data,
            token,
            freq_width,
            axes,
            theta,
            [
                (max_vid_index + token) as isize,
                (max_vid_index + token) as isize,
                (max_vid_index + token) as isize,
            ],
        );
    }

    Ok(QwenRotaryEmbeddings {
        image: RotaryFrequencies {
            cos: image_cos,
            sin: image_sin,
        },
        text: RotaryFrequencies {
            cos: text_cos,
            sin: text_sin,
        },
    })
}

fn write_qwen_rope_token(
    cos: &mut [f32],
    sin: &mut [f32],
    token: usize,
    freq_width: usize,
    axes: [usize; 3],
    theta: f32,
    positions: [isize; 3],
) {
    let mut dst = token * freq_width;
    for (axis_index, axis_dim) in axes.into_iter().enumerate() {
        let axis_freqs = axis_dim / 2;
        for freq_index in 0..axis_freqs {
            let exponent = (2 * freq_index) as f32 / axis_dim as f32;
            let angle = positions[axis_index] as f32 / theta.powf(exponent);
            cos[dst] = angle.cos();
            sin[dst] = angle.sin();
            dst += 1;
        }
    }
}

#[allow(dead_code)]
fn apply_qwen_rotary_embedding(
    input: &CpuTensor,
    freqs: &RotaryFrequencies,
    heads: usize,
    head_dim: usize,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, width] = shape3(input)?;
    if heads == 0 || head_dim == 0 || head_dim % 2 != 0 || width != heads * head_dim {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Qwen RoPE input width {width} is incompatible with heads {heads} and head_dim {head_dim}"
        )));
    }
    let freq_width = head_dim / 2;
    if freqs.cos.shape.as_slice() != [seq, freq_width]
        || freqs.sin.shape.as_slice() != [seq, freq_width]
    {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Qwen RoPE frequency shapes {:?}/{:?} != [{seq}, {freq_width}]",
            freqs.cos.shape, freqs.sin.shape
        )));
    }
    let mut out = CpuTensor::zeros(&[batch, seq, width]);
    for b in 0..batch {
        for token in 0..seq {
            for head in 0..heads {
                let token_base = (b * seq + token) * width + head * head_dim;
                let freq_base = token * freq_width;
                for pair in 0..freq_width {
                    let real_idx = token_base + pair * 2;
                    let imag_idx = real_idx + 1;
                    let real = input.data[real_idx];
                    let imag = input.data[imag_idx];
                    let cos = freqs.cos.data[freq_base + pair];
                    let sin = freqs.sin.data[freq_base + pair];
                    out.data[real_idx] = real * cos - imag * sin;
                    out.data[imag_idx] = real * sin + imag * cos;
                }
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn validate_attention_linear_shape(
    stream_label: &str,
    name: &str,
    weight: &CpuTensor,
    expected_rows: usize,
    expected_cols: usize,
) -> DiffusionResult<()> {
    if weight.shape.as_slice() != [expected_rows, expected_cols] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "{stream_label} transformer attention {name} weight shape {:?} != [{expected_rows}, {expected_cols}]",
            weight.shape
        )));
    }
    Ok(())
}

#[allow(dead_code)]
fn validate_attention_bias_shape(
    stream_label: &str,
    name: &str,
    bias: Option<&CpuTensor>,
    expected_width: usize,
) -> DiffusionResult<()> {
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [expected_width] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{stream_label} transformer attention {name} bias shape {:?} != [{expected_width}]",
                bias.shape
            )));
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn validate_attention_norm_shape(
    stream_label: &str,
    name: &str,
    weight: Option<&CpuTensor>,
    head_dim: usize,
) -> DiffusionResult<()> {
    if let Some(weight) = weight {
        if weight.shape.as_slice() != [head_dim] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{stream_label} transformer attention {name} norm shape {:?} != [{head_dim}]",
                weight.shape
            )));
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn maybe_rms_norm_attention_heads_3d(
    input: CpuTensor,
    weight: Option<&CpuTensor>,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> DiffusionResult<CpuTensor> {
    let Some(weight) = weight else {
        return Ok(input);
    };
    rms_norm_attention_heads_3d(&input, weight, heads, head_dim, eps)
}

#[allow(dead_code)]
fn rms_norm_attention_heads_3d(
    input: &CpuTensor,
    weight: &CpuTensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, width] = shape3(input)?;
    if heads == 0 || head_dim == 0 || width != heads * head_dim {
        return Err(DiffusionError::InvalidMetadata(format!(
            "attention-head RMSNorm input width {width} is incompatible with heads {heads} and head_dim {head_dim}"
        )));
    }
    if weight.shape.as_slice() != [head_dim] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "attention-head RMSNorm weight shape {:?} != [{head_dim}]",
            weight.shape
        )));
    }
    let mut out = CpuTensor::zeros(&[batch, seq, width]);
    for b in 0..batch {
        for token in 0..seq {
            let token_base = (b * seq + token) * width;
            for head in 0..heads {
                let head_base = token_base + head * head_dim;
                let mut square_sum = 0.0f32;
                for dim in 0..head_dim {
                    let value = input.data[head_base + dim];
                    square_sum += value * value;
                }
                let inv_rms = (square_sum / head_dim as f32 + eps).sqrt().recip();
                for dim in 0..head_dim {
                    out.data[head_base + dim] =
                        input.data[head_base + dim] * inv_rms * weight.data[dim];
                }
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn rms_norm_3d_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    eps: f32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let _ = runtime_context;
    let [batch, seq, width] = shape3(input)?;
    if weight.shape.as_slice() != [width] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "RMSNorm weight shape {:?} != [{width}]",
            weight.shape
        )));
    }
    let mut out = CpuTensor::zeros(&[batch, seq, width]);
    for b in 0..batch {
        for token in 0..seq {
            let token_base = (b * seq + token) * width;
            let mut square_sum = 0.0f32;
            for dim in 0..width {
                let value = input.data[token_base + dim];
                square_sum += value * value;
            }
            let inv_rms = (square_sum / width as f32 + eps).sqrt().recip();
            for dim in 0..width {
                out.data[token_base + dim] =
                    input.data[token_base + dim] * inv_rms * weight.data[dim];
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn validate_transformer_ff_down_shape(
    stream_label: &str,
    down_weight: &CpuTensor,
    down_bias: Option<&CpuTensor>,
    hidden_width: usize,
    inner_width: usize,
) -> DiffusionResult<()> {
    if down_weight.shape.as_slice() != [hidden_width, inner_width] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "{stream_label} transformer feed-forward down weight shape {:?} != [{hidden_width}, {inner_width}]",
            down_weight.shape
        )));
    }
    validate_attention_bias_shape(stream_label, "ff.down", down_bias, hidden_width)
}

#[allow(dead_code)]
fn swiglu_gate_3d(up: &CpuTensor, gate: &CpuTensor) -> DiffusionResult<CpuTensor> {
    if up.shape != gate.shape {
        return Err(DiffusionError::InvalidMetadata(format!(
            "SwiGLU up/gate shape mismatch {:?} vs {:?}",
            up.shape, gate.shape
        )));
    }
    let [batch, seq, width] = shape3(up)?;
    let mut out = CpuTensor::zeros(&[batch, seq, width]);
    for (dst, (up, gate)) in out.data.iter_mut().zip(up.data.iter().zip(&gate.data)) {
        *dst = *up * silu(*gate);
    }
    Ok(out)
}

#[allow(dead_code)]
fn layer_norm_3d_no_affine_with_runtime_context(
    input: &CpuTensor,
    eps: f32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let [_, _, width] = shape3(input)?;
    let weight = CpuTensor {
        shape: vec![width],
        data: vec![1.0; width],
    };
    let bias = CpuTensor {
        shape: vec![width],
        data: vec![0.0; width],
    };
    layer_norm_3d_with_runtime_context(input, &weight, &bias, eps, runtime_context)
}

#[allow(dead_code)]
fn modulate_3d(
    input: &CpuTensor,
    shift: &CpuTensor,
    scale: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, width] = shape3(input)?;
    if shift.shape.as_slice() != [batch, width] || scale.shape.as_slice() != [batch, width] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "modulate_3d input shape {:?} requires shift/scale [{batch}, {width}], got {:?}/{:?}",
            input.shape, shift.shape, scale.shape
        )));
    }
    let mut out = CpuTensor::zeros(&[batch, seq, width]);
    for b in 0..batch {
        for s in 0..seq {
            let token_base = (b * seq + s) * width;
            let mod_base = b * width;
            for col in 0..width {
                out.data[token_base + col] = input.data[token_base + col]
                    * (1.0 + scale.data[mod_base + col])
                    + shift.data[mod_base + col];
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn gated_residual_3d(
    residual: &CpuTensor,
    update: &CpuTensor,
    gate: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, width] = shape3(residual)?;
    if update.shape != residual.shape || gate.shape.as_slice() != [batch, width] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "gated residual shape mismatch residual/update/gate {:?}/{:?}/{:?}",
            residual.shape, update.shape, gate.shape
        )));
    }
    let mut out = CpuTensor::zeros(&[batch, seq, width]);
    for b in 0..batch {
        for s in 0..seq {
            let token_base = (b * seq + s) * width;
            let gate_base = b * width;
            for col in 0..width {
                out.data[token_base + col] = residual.data[token_base + col]
                    + gate.data[gate_base + col] * update.data[token_base + col];
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn concat_sequence_3d(left: &CpuTensor, right: &CpuTensor) -> DiffusionResult<CpuTensor> {
    let [left_batch, left_seq, left_width] = shape3(left)?;
    let [right_batch, right_seq, right_width] = shape3(right)?;
    if left_batch != right_batch || left_width != right_width {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate BSC tensors with shapes {:?} and {:?}",
            left.shape, right.shape
        )));
    }
    let mut out = CpuTensor::zeros(&[left_batch, left_seq + right_seq, left_width]);
    for batch in 0..left_batch {
        let left_src = batch * left_seq * left_width;
        let right_src = batch * right_seq * right_width;
        let dst = batch * (left_seq + right_seq) * left_width;
        out.data[dst..dst + left_seq * left_width]
            .copy_from_slice(&left.data[left_src..left_src + left_seq * left_width]);
        let right_dst = dst + left_seq * left_width;
        out.data[right_dst..right_dst + right_seq * right_width]
            .copy_from_slice(&right.data[right_src..right_src + right_seq * right_width]);
    }
    Ok(out)
}

#[allow(dead_code)]
fn split_modulation_chunks(
    projected: CpuTensor,
    chunk_count: usize,
) -> DiffusionResult<TransformerModulationChunks> {
    if chunk_count != 6 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "expected 6 modulation chunks, got {chunk_count}"
        )));
    }
    let [batch, width] = shape2(&projected)?;
    if width % chunk_count != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "modulation width {width} is not divisible by {chunk_count}"
        )));
    }
    let chunk_width = width / chunk_count;
    let chunk = |chunk_idx: usize| -> CpuTensor {
        let mut data = vec![0.0; batch * chunk_width];
        for b in 0..batch {
            let src = b * width + chunk_idx * chunk_width;
            let dst = b * chunk_width;
            data[dst..dst + chunk_width].copy_from_slice(&projected.data[src..src + chunk_width]);
        }
        CpuTensor {
            shape: vec![batch, chunk_width],
            data,
        }
    };
    Ok(TransformerModulationChunks {
        shift_msa: chunk(0),
        scale_msa: chunk(1),
        gate_msa: chunk(2),
        shift_mlp: chunk(3),
        scale_mlp: chunk(4),
        gate_mlp: chunk(5),
    })
}

fn diffusion_generation_info(
    summary: &DiffusionModelSummary,
    runtime_kind: DiffusionRuntimeKind,
    request: &DiffusionBatchRequest,
    latent_shape: &DiffusionLatentShape,
) -> Value {
    let mut info = json!({
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
    });
    if let Some(scale) = request.distilled_guidance_scale {
        if let Some(map) = info.as_object_mut() {
            map.insert("distilled_guidance_scale".to_string(), json!(scale));
        }
    }
    info
}

impl StableDiffusionConfig {
    pub fn from_hfq(hfq: &HfqFile, metadata: &DiffusionHfqMetadata) -> DiffusionResult<Self> {
        let text_json = component_json(hfq, metadata, "text_encoder")?.unwrap_or_else(|| json!({}));
        let text_2_json = component_json(hfq, metadata, "text_encoder_2")?;
        let unet_json = component_json(hfq, metadata, "unet")?.unwrap_or_else(|| json!({}));
        let transformer_json = component_json(hfq, metadata, "transformer")?;
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
        let transformer =
            transformer_json
                .as_ref()
                .map(|transformer_json| TransformerDenoiserConfig {
                    class_name: json_string(transformer_json, "_class_name"),
                    in_channels: json_usize(transformer_json, "in_channels"),
                    out_channels: json_usize(transformer_json, "out_channels"),
                    patch_size: json_usize(transformer_json, "patch_size").or_else(|| {
                        default_transformer_patch_size(&json_string(
                            transformer_json,
                            "_class_name",
                        ))
                    }),
                    num_layers: json_usize(transformer_json, "num_layers"),
                    num_attention_heads: json_usize(transformer_json, "num_attention_heads"),
                    num_key_value_heads: json_usize(transformer_json, "num_key_value_heads"),
                    attention_head_dim: json_usize(transformer_json, "attention_head_dim"),
                    cross_attention_dim: json_usize(transformer_json, "cross_attention_dim")
                        .or_else(|| json_usize(transformer_json, "joint_attention_dim")),
                    caption_projection_dim: json_usize(transformer_json, "caption_projection_dim"),
                    pooled_projection_dim: json_usize(transformer_json, "pooled_projection_dim"),
                    axes_dims_rope: json_usize_vec(transformer_json, "axes_dims_rope"),
                    guidance_embeds: json_bool(transformer_json, "guidance_embeds"),
                    intermediate_size: json_usize(transformer_json, "intermediate_size"),
                    norm_eps: json_f32(transformer_json, "norm_eps"),
                    text_hidden_dim: json_usize(transformer_json, "text_hidden_dim"),
                    text_intermediate_size: json_usize(transformer_json, "text_intermediate_size"),
                    text_num_attention_heads: json_usize(
                        transformer_json,
                        "text_num_attention_heads",
                    ),
                    text_num_key_value_heads: json_usize(
                        transformer_json,
                        "text_num_key_value_heads",
                    ),
                    num_text_layers: json_usize(transformer_json, "num_text_layers"),
                    num_refiner_text_blocks: json_usize(
                        transformer_json,
                        "num_refiner_text_blocks",
                    ),
                    num_layerwise_text_blocks: json_usize(
                        transformer_json,
                        "num_layerwise_text_blocks",
                    ),
                    timestep_embed_dim: json_usize(transformer_json, "timestep_embed_dim"),
                    rope_theta: json_f32(transformer_json, "rope_theta"),
                });
        let vae = VaeConfig {
            class_name: json_string(&vae_json, "_class_name"),
            latent_channels: json_usize(&vae_json, "latent_channels"),
            z_dim: json_usize(&vae_json, "z_dim"),
            scaling_factor: json_f32(&vae_json, "scaling_factor"),
            shift_factor: json_f32(&vae_json, "shift_factor"),
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
            transformer,
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
    let latent_shape = DiffusionLatentShape {
        batch: request.prompts.len(),
        channels: config.latent_channels,
        height: (request.height / scale) as usize,
        width: (request.width / scale) as usize,
    };
    validate_unet_latent_shape_for_request(config, &latent_shape, scale as usize)?;
    Ok(latent_shape)
}

fn validate_unet_latent_shape_for_request(
    config: &StableDiffusionConfig,
    latent_shape: &DiffusionLatentShape,
    scale: usize,
) -> DiffusionResult<()> {
    if config.transformer.is_some() {
        return Ok(());
    }
    let block_count = unet_down_block_count(&config.unet);
    let min_side = minimum_unet_latent_side(&config.unet);
    if min_side <= 1 {
        return Ok(());
    }
    if latent_shape.width < min_side || latent_shape.height < min_side {
        let min_pixels = min_side.saturating_mul(scale);
        return Err(DiffusionError::InvalidRequest(format!(
            "latent shape {}x{} is too small for UNet downsampling depth {}; request at least {}x{} pixels with VAE scale factor {scale}",
            latent_shape.width,
            latent_shape.height,
            block_count.saturating_sub(1),
            min_pixels,
            min_pixels
        )));
    }
    Ok(())
}

fn unet_down_block_count(config: &UnetConfig) -> usize {
    if config.down_block_types.is_empty() {
        config.block_out_channels.len()
    } else {
        config.down_block_types.len()
    }
}

fn minimum_unet_latent_side(config: &UnetConfig) -> usize {
    let downsample_count = unet_down_block_count(config).saturating_sub(1);
    if downsample_count >= usize::BITS as usize {
        usize::MAX
    } else {
        1usize << downsample_count
    }
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
    let transformer_denoiser = transformer_denoiser_plan(config, &latent_shape)?;
    let denoise_elements = if let Some(plan) = &transformer_denoiser {
        checked_shape_elements(
            "transformer denoise input",
            &[plan.batch, plan.sequence_length, plan.token_width],
        )?
    } else {
        let denoise_channels = config
            .unet
            .in_channels
            .unwrap_or(config.latent_channels)
            .max(config.latent_channels);
        checked_shape_elements(
            "denoise input",
            &[
                latent_shape.batch,
                denoise_channels,
                latent_shape.height,
                latent_shape.width,
            ],
        )?
    };
    let denoise_input_bytes = checked_bytes("denoise input", denoise_elements, 4)?;
    let max_position_embeddings = config
        .text_encoder
        .max_position_embeddings
        .unwrap_or(77)
        .max(1);
    let cross_attention_dim = config
        .transformer
        .as_ref()
        .and_then(|transformer| {
            transformer
                .cross_attention_dim
                .or(transformer.caption_projection_dim)
                .or(transformer.pooled_projection_dim)
        })
        .or(config.unet.cross_attention_dim)
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
        transformer_denoiser,
        latent_bytes,
        denoise_input_bytes,
        conditioning_bytes,
        vae_decode_bytes,
        rgb_bytes,
        scheduler_scratch_bytes,
        total_device_bytes,
    })
}

fn transformer_denoiser_plan(
    config: &StableDiffusionConfig,
    latent_shape: &DiffusionLatentShape,
) -> DiffusionResult<Option<DiffusionTransformerDenoiserPlan>> {
    let Some(transformer) = &config.transformer else {
        return Ok(None);
    };
    let patch_size = transformer.patch_size.unwrap_or(1).max(1);
    if latent_shape.height % patch_size != 0 || latent_shape.width % patch_size != 0 {
        return Err(DiffusionError::InvalidRequest(format!(
            "latent shape {}x{} must be divisible by transformer patch_size {patch_size}",
            latent_shape.width, latent_shape.height
        )));
    }
    let patch_height = latent_shape.height / patch_size;
    let patch_width = latent_shape.width / patch_size;
    let sequence_length = patch_height.checked_mul(patch_width).ok_or_else(|| {
        DiffusionError::InvalidRequest("transformer sequence length overflow".into())
    })?;
    let patch_width_channels = latent_shape
        .channels
        .checked_mul(patch_size)
        .and_then(|value| value.checked_mul(patch_size))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("transformer patch token width overflow".into())
        })?;
    let token_width = transformer
        .in_channels
        .or(transformer.out_channels)
        .unwrap_or(patch_width_channels)
        .max(patch_width_channels);
    Ok(Some(DiffusionTransformerDenoiserPlan {
        representation: "patch_tokens".to_string(),
        batch: latent_shape.batch,
        sequence_length,
        token_width,
        patch_size,
        latent_height: latent_shape.height,
        latent_width: latent_shape.width,
        patch_height,
        patch_width,
        output_channels: transformer.out_channels.unwrap_or(latent_shape.channels),
    }))
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
            None,
            None,
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
        positive_attention_mask: Option<&CpuTensor>,
        negative_attention_mask: Option<&CpuTensor>,
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
            |sample,
             timesteps,
             encoder_states,
             _attention_mask,
             sdxl_conditioning,
             runtime_context| {
                self.forward_with_sdxl_conditioning_and_runtime_context(
                    sample,
                    timesteps,
                    encoder_states,
                    sdxl_conditioning,
                    runtime_context,
                )
            },
            positive_attention_mask,
            negative_attention_mask,
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
        positive_attention_mask: Option<&CpuTensor>,
        negative_attention_mask: Option<&CpuTensor>,
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
            |sample,
             timesteps,
             encoder_states,
             _attention_mask,
             sdxl_conditioning,
             runtime_context| {
                self.forward_with_sdxl_conditioning_and_runtime_context(
                    sample,
                    timesteps,
                    encoder_states,
                    sdxl_conditioning,
                    runtime_context,
                )
            },
            positive_attention_mask,
            negative_attention_mask,
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
    latent_norm: VaeLatentNorm,
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
            latent_norm: VaeLatentNorm::from_config(config)?,
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
        vae_moments_to_latents(&moments, &self.latent_norm)
    }

    #[cfg(test)]
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
        vae_moments_to_latents_with_runtime_context(&moments, &self.latent_norm, runtime_context)
    }

    /// Stochastic counterpart of [`encode_to_latents_with_runtime_context`]: sample
    /// from the VAE's diagonal Gaussian using the supplied per-batch `seeds`.
    fn encode_to_latents_sampled_with_runtime_context(
        &self,
        image: &RgbImageBatch,
        seeds: &[i64],
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<LatentBatch> {
        let image_tensor = rgb_batch_to_vae_tensor_with_runtime_context(image, runtime_context)?;
        let moments =
            self.encode_tensor_moments_with_runtime_context(&image_tensor, runtime_context)?;
        vae_moments_to_latents_sampled(&moments, &self.latent_norm, seeds)
    }
}

fn vae_moments_to_latents(
    moments: &CpuTensor,
    norm: &VaeLatentNorm,
) -> DiffusionResult<LatentBatch> {
    let [batch, channels, height, width] = shape4(moments)?;
    if channels % 2 != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "VAE encoder moments channel count {channels} is not even"
        )));
    }
    let latent_channels = channels / 2;
    let mut data = Vec::with_capacity(batch * latent_channels * height * width);
    for b in 0..batch {
        for c in 0..latent_channels {
            for y in 0..height {
                for x in 0..width {
                    data.push(moments.data[nchw_idx(b, c, y, x, channels, height, width)]);
                }
            }
        }
    }
    norm.apply_encode(&mut data, latent_channels, height * width)?;
    Ok(LatentBatch {
        batch,
        channels: latent_channels,
        height,
        width,
        data,
    })
}

/// Domain-separation salts so VAE-encode Gaussian noise does not alias the
/// initial-latent noise stream (which seeds the request seed directly) or other
/// encode sites. The values are arbitrary fixed constants.
const VAE_INIT_ENCODE_SEED_SALT: u64 = 0x7661_655f_696e_6974; // "vae_init"
const VAE_MASKED_ENCODE_SEED_SALT: u64 = 0x7661_655f_6d61_736b; // "vae_mask"

/// Derive decorrelated per-batch RNG seeds for a specific VAE encode site.
fn vae_encode_seeds(seeds: &[i64], salt: u64) -> Vec<i64> {
    seeds
        .iter()
        .map(|seed| ((*seed as u64) ^ salt) as i64)
        .collect()
}

/// Stochastic VAE encode: sample from the diagonal Gaussian `mean + std * eps`
/// (`std = exp(0.5 * clamp(logvar, -30, 20))`) instead of taking the distribution
/// mode, then apply latent-space normalization. The moments tensor packs the mean
/// in the first half of the channel axis and the log-variance in the second half.
/// Sampling is deterministic given `seeds` and always runs on the CPU (the fused
/// HIP kernel only covers the scalar-scaled mode path).
fn vae_moments_to_latents_sampled(
    moments: &CpuTensor,
    norm: &VaeLatentNorm,
    seeds: &[i64],
) -> DiffusionResult<LatentBatch> {
    let [batch, channels, height, width] = shape4(moments)?;
    if channels % 2 != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "VAE encoder moments channel count {channels} is not even"
        )));
    }
    let latent_channels = channels / 2;
    let plane = height * width;
    let mut data = Vec::with_capacity(batch * latent_channels * plane);
    for b in 0..batch {
        let mut rng = SplitMix64::new(seeds.get(b).copied().unwrap_or(-1) as u64);
        let mut spare: Option<f32> = None;
        for c in 0..latent_channels {
            for y in 0..height {
                for x in 0..width {
                    let mean = moments.data[nchw_idx(b, c, y, x, channels, height, width)];
                    let logvar = moments.data
                        [nchw_idx(b, latent_channels + c, y, x, channels, height, width)];
                    let std = (0.5 * logvar.clamp(-30.0, 20.0)).exp();
                    let noise = match spare.take() {
                        Some(value) => value,
                        None => {
                            let (first, second) = box_muller_pair(&mut rng);
                            spare = Some(second);
                            first
                        }
                    };
                    data.push(mean + std * noise);
                }
            }
        }
    }
    norm.apply_encode(&mut data, latent_channels, plane)?;
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
    {
        runtime_context.with_rocm_gpu(|gpu| rgb_batch_to_vae_tensor_hip_on_gpu(gpu, image))
    }
}

fn vae_moments_to_latents_with_runtime_context(
    moments: &CpuTensor,
    norm: &VaeLatentNorm,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<LatentBatch> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return vae_moments_to_latents(moments, norm);
    };
    // The fused HIP kernel only applies the scalar scaling factor. Per-channel
    // (Qwen-Image) or shifted (Flux/SD3) normalization falls back to the CPU
    // reference, which is cheap on the small latent tensor.
    if !norm.is_scalar_scale_only() {
        return vae_moments_to_latents(moments, norm);
    }
    {
        runtime_context.with_rocm_gpu(|gpu| {
            vae_moments_to_latents_hip_on_gpu(gpu, moments, norm.scaling_factor)
        })
    }
}

/// Map latents (NCHW) back into VAE input space ahead of decoding. The scalar
/// scale-only case routes through the GPU-capable scale kernel; per-channel or
/// shifted normalization is applied on the CPU.
fn denormalize_decode_latents(
    hidden: &CpuTensor,
    norm: &VaeLatentNorm,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    if norm.is_scalar_scale_only() {
        let scale = norm.scaling_factor.max(f32::MIN_POSITIVE);
        return scale_tensor_with_runtime_context(hidden, scale.recip(), runtime_context);
    }
    let [_, channels, height, width] = shape4(hidden)?;
    let mut data = hidden.data.clone();
    norm.apply_decode(&mut data, channels, height * width)?;
    Ok(CpuTensor {
        shape: hidden.shape.clone(),
        data,
    })
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
    latent_norm: VaeLatentNorm,
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
            latent_norm: VaeLatentNorm::from_config(config)?,
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
        hidden = denormalize_decode_latents(&hidden, &self.latent_norm, runtime_context)?;
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
    {
        runtime_context.with_rocm_gpu(|gpu| rgb_tensor_to_u8_hip_on_gpu(gpu, tensor))
    }
}

mod hip_kernels;
use hip_kernels::*;

mod gpu_ops;
use gpu_ops::*;

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

#[allow(dead_code)]
fn latent_batch_to_patch_tokens(
    latents: &LatentBatch,
    patch_size: usize,
    token_width: usize,
) -> DiffusionResult<CpuTensor> {
    if patch_size == 0 {
        return Err(DiffusionError::InvalidRequest(
            "transformer patch_size must be positive".to_string(),
        ));
    }
    if latents.height % patch_size != 0 || latents.width % patch_size != 0 {
        return Err(DiffusionError::InvalidRequest(format!(
            "latent shape {}x{} must be divisible by transformer patch_size {patch_size}",
            latents.width, latents.height
        )));
    }
    let patch_height = latents.height / patch_size;
    let patch_width = latents.width / patch_size;
    let sequence_length = patch_height.checked_mul(patch_width).ok_or_else(|| {
        DiffusionError::InvalidRequest("transformer sequence length overflow".to_string())
    })?;
    let patch_feature_width = latents
        .channels
        .checked_mul(patch_size)
        .and_then(|value| value.checked_mul(patch_size))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("transformer patch token width overflow".to_string())
        })?;
    if token_width < patch_feature_width {
        return Err(DiffusionError::InvalidRequest(format!(
            "transformer token_width {token_width} is smaller than latent patch feature width {patch_feature_width}"
        )));
    }
    let expected = latents
        .batch
        .checked_mul(latents.channels)
        .and_then(|value| value.checked_mul(latents.height))
        .and_then(|value| value.checked_mul(latents.width))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("latent element count overflow".to_string())
        })?;
    if latents.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "latent batch has {} values, expected {expected}",
            latents.data.len()
        )));
    }

    let mut tokens = CpuTensor::zeros(&[latents.batch, sequence_length, token_width]);
    for batch in 0..latents.batch {
        for patch_y in 0..patch_height {
            for patch_x in 0..patch_width {
                let token = patch_y * patch_width + patch_x;
                let token_base = (batch * sequence_length + token) * token_width;
                let mut feature = 0;
                for channel in 0..latents.channels {
                    for local_y in 0..patch_size {
                        for local_x in 0..patch_size {
                            let y = patch_y * patch_size + local_y;
                            let x = patch_x * patch_size + local_x;
                            tokens.data[token_base + feature] = latents.data[nchw_idx(
                                batch,
                                channel,
                                y,
                                x,
                                latents.channels,
                                latents.height,
                                latents.width,
                            )];
                            feature += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(tokens)
}

#[allow(dead_code)]
fn patch_tokens_to_latent_batch(
    tokens: &CpuTensor,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
) -> DiffusionResult<LatentBatch> {
    if patch_size == 0 {
        return Err(DiffusionError::InvalidRequest(
            "transformer patch_size must be positive".to_string(),
        ));
    }
    if height % patch_size != 0 || width % patch_size != 0 {
        return Err(DiffusionError::InvalidRequest(format!(
            "latent shape {width}x{height} must be divisible by transformer patch_size {patch_size}"
        )));
    }
    let patch_height = height / patch_size;
    let patch_width = width / patch_size;
    let expected_sequence = patch_height.checked_mul(patch_width).ok_or_else(|| {
        DiffusionError::InvalidRequest("transformer sequence length overflow".to_string())
    })?;
    let patch_feature_width = channels
        .checked_mul(patch_size)
        .and_then(|value| value.checked_mul(patch_size))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("transformer patch token width overflow".to_string())
        })?;
    let [token_batch, sequence_length, token_width] = shape3(tokens)?;
    if token_batch != batch || sequence_length != expected_sequence {
        return Err(DiffusionError::InvalidMetadata(format!(
            "transformer token shape {:?} cannot unpatchify to [{batch}, {channels}, {height}, {width}]",
            tokens.shape
        )));
    }
    if token_width < patch_feature_width {
        return Err(DiffusionError::InvalidMetadata(format!(
            "transformer token_width {token_width} is smaller than latent patch feature width {patch_feature_width}"
        )));
    }

    let element_count = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_mul(width))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("latent element count overflow".to_string())
        })?;
    let mut latents = LatentBatch {
        batch,
        channels,
        height,
        width,
        data: vec![0.0; element_count],
    };
    for batch_idx in 0..batch {
        for patch_y in 0..patch_height {
            for patch_x in 0..patch_width {
                let token = patch_y * patch_width + patch_x;
                let token_base = (batch_idx * sequence_length + token) * token_width;
                let mut feature = 0;
                for channel in 0..channels {
                    for local_y in 0..patch_size {
                        for local_x in 0..patch_size {
                            let y = patch_y * patch_size + local_y;
                            let x = patch_x * patch_size + local_x;
                            let latent_idx =
                                nchw_idx(batch_idx, channel, y, x, channels, height, width);
                            latents.data[latent_idx] = tokens.data[token_base + feature];
                            feature += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(latents)
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
    scaled_dot_product_attention_with_key_mask(q, k, v, heads, None)
}

fn scaled_dot_product_attention_with_key_mask(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    heads: usize,
    key_mask: Option<&[bool]>,
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
    if let Some(mask) = key_mask {
        let expected = batch * k_seq;
        if mask.len() != expected {
            return Err(DiffusionError::InvalidMetadata(format!(
                "attention key mask has {} entries, expected {expected}",
                mask.len()
            )));
        }
    }
    let head_dim = hidden / heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = CpuTensor::zeros(&[batch, q_seq, hidden]);
    for b in 0..batch {
        for head in 0..heads {
            let head_off = head * head_dim;
            for qi in 0..q_seq {
                let mut scores = vec![0.0f32; k_seq];
                let mut has_active_key = false;
                for ki in 0..k_seq {
                    if let Some(mask) = key_mask {
                        if !mask[b * k_seq + ki] {
                            scores[ki] = f32::NEG_INFINITY;
                            continue;
                        }
                    }
                    has_active_key = true;
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += q.data[((b * q_seq + qi) * hidden) + head_off + d]
                            * k.data[((b * k_seq + ki) * hidden) + head_off + d];
                    }
                    scores[ki] = dot * scale;
                }
                if !has_active_key {
                    continue;
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
    let plane_len = out_h * out_w;
    out.data
        .par_chunks_mut(plane_len)
        .enumerate()
        .for_each(|(plane_idx, out_plane)| {
            let b = plane_idx / out_channels;
            let oc = plane_idx % out_channels;
            if let Some(bias) = bias {
                out_plane.fill(bias.data[oc]);
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
                        let output_row = oy * out_w;
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
                                out_plane[output_row + ox] +=
                                    input.data[input_row + ix] * weight_value;
                            }
                        }
                    }
                }
            }
        });
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

mod tokenizer;
pub use tokenizer::*;

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
    let runtime_support = match native_runtime_support_error(&hfq, &metadata)? {
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

fn default_transformer_patch_size(class_name: &str) -> Option<usize> {
    match class_name {
        "QwenImageTransformer2DModel" | "Krea2Transformer2DModel" => Some(2),
        _ => None,
    }
}

fn transformer_denoiser_weight_topology(
    component: &DiffusionComponentMetadata,
) -> TransformerDenoiserWeightTopology {
    let class_name = component.class_name.as_deref().unwrap_or_default();
    let mut blocks = BTreeSet::new();
    let mut has_input_projection = false;
    let mut has_output_projection = false;
    let mut has_text_modulation = false;
    let mut has_text_fusion = false;

    for entry in &component.weight_entries {
        let name = entry
            .strip_prefix("transformer/tensors/")
            .unwrap_or(entry.as_str());
        has_input_projection |= matches!(name, "img_in.weight" | "img_in.bias");
        has_output_projection |= matches!(
            name,
            "proj_out.weight"
                | "proj_out.bias"
                | "norm_out.linear.weight"
                | "norm_out.linear.bias"
                | "final_layer.linear.weight"
                | "final_layer.linear.bias"
        );
        has_text_modulation |= name.contains(".txt_mod.")
            || name.contains(".txt_mlp.")
            || name.contains(".attn.add_q_proj.")
            || name.contains(".attn.add_k_proj.")
            || name.contains(".attn.add_v_proj.");
        has_text_fusion |= name.starts_with("text_fusion.");
        if let Some(rest) = name.strip_prefix("transformer_blocks.") {
            if let Some((idx, _)) = rest.split_once('.') {
                if let Ok(idx) = idx.parse::<usize>() {
                    blocks.insert(idx);
                }
            }
        }
    }

    let family = match class_name {
        "QwenImageTransformer2DModel" => TransformerDenoiserFamily::QwenImage,
        "Krea2Transformer2DModel" => TransformerDenoiserFamily::Krea2,
        _ if has_text_fusion => TransformerDenoiserFamily::Krea2,
        _ if has_text_modulation => TransformerDenoiserFamily::QwenImage,
        _ => TransformerDenoiserFamily::Unknown,
    };

    TransformerDenoiserWeightTopology {
        family,
        block_count: blocks.len(),
        has_input_projection,
        has_output_projection,
        has_text_modulation,
        has_text_fusion,
    }
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
    if request.distilled_guidance_scale.is_some() {
        return Err(DiffusionError::InvalidRequest(
            "distilled_guidance_scale is not implemented by the native diffusion denoiser yet; it is distinct from cfg_scale and must not be silently ignored".to_string(),
        ));
    }
    if let Some(conditioning) = request.conditioning.as_ref() {
        validate_external_conditioning_batch(conditioning, request.prompts.len())?;
    }
    Ok(())
}

fn validate_external_conditioning_batch(
    conditioning: &DiffusionExternalConditioningBatch,
    batch: usize,
) -> DiffusionResult<()> {
    let prompt_shape = validate_external_conditioning_hidden(
        "prompt_embeddings",
        &conditioning.prompt_embeddings,
        batch,
    )?;
    let negative_shape = validate_external_conditioning_hidden(
        "negative_embeddings",
        &conditioning.negative_embeddings,
        batch,
    )?;
    if prompt_shape != negative_shape {
        return Err(DiffusionError::InvalidRequest(format!(
            "external prompt_embeddings shape {:?} must match negative_embeddings shape {:?}",
            conditioning.prompt_embeddings.shape, conditioning.negative_embeddings.shape
        )));
    }
    match (
        conditioning.prompt_pooled_embeddings.as_ref(),
        conditioning.negative_pooled_embeddings.as_ref(),
    ) {
        (Some(prompt), Some(negative)) => {
            let prompt_shape =
                validate_external_conditioning_pooled("prompt_pooled_embeddings", prompt, batch)?;
            let negative_shape = validate_external_conditioning_pooled(
                "negative_pooled_embeddings",
                negative,
                batch,
            )?;
            if prompt_shape != negative_shape {
                return Err(DiffusionError::InvalidRequest(format!(
                    "external prompt_pooled_embeddings shape {:?} must match negative_pooled_embeddings shape {:?}",
                    prompt.shape, negative.shape
                )));
            }
        }
        (None, None) => {}
        _ => {
            return Err(DiffusionError::InvalidRequest(
                "external pooled conditioning requires both prompt_pooled_embeddings and negative_pooled_embeddings".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_external_conditioning_hidden(
    label: &str,
    tensor: &CpuTensor,
    batch: usize,
) -> DiffusionResult<[usize; 3]> {
    let shape = match tensor.shape.as_slice() {
        [tensor_batch, seq, width] => [*tensor_batch, *seq, *width],
        _ => {
            return Err(DiffusionError::InvalidRequest(format!(
                "external {label} must be a 3-D tensor [batch, sequence, width], got {:?}",
                tensor.shape
            )));
        }
    };
    validate_external_conditioning_shape(label, &tensor.shape, &tensor.data, batch)?;
    if shape[1] == 0 || shape[2] == 0 {
        return Err(DiffusionError::InvalidRequest(format!(
            "external {label} sequence and width must be non-zero, got {:?}",
            tensor.shape
        )));
    }
    Ok(shape)
}

fn validate_external_conditioning_pooled(
    label: &str,
    tensor: &CpuTensor,
    batch: usize,
) -> DiffusionResult<[usize; 2]> {
    let shape = match tensor.shape.as_slice() {
        [tensor_batch, width] => [*tensor_batch, *width],
        _ => {
            return Err(DiffusionError::InvalidRequest(format!(
                "external {label} must be a 2-D tensor [batch, width], got {:?}",
                tensor.shape
            )));
        }
    };
    validate_external_conditioning_shape(label, &tensor.shape, &tensor.data, batch)?;
    if shape[1] == 0 {
        return Err(DiffusionError::InvalidRequest(format!(
            "external {label} width must be non-zero, got {:?}",
            tensor.shape
        )));
    }
    Ok(shape)
}

fn validate_external_conditioning_shape(
    label: &str,
    shape: &[usize],
    data: &[f32],
    batch: usize,
) -> DiffusionResult<()> {
    if shape.first().copied() != Some(batch) {
        return Err(DiffusionError::InvalidRequest(format!(
            "external {label} batch {} must match prompt batch {batch}",
            shape.first().copied().unwrap_or(0)
        )));
    }
    let elements = checked_shape_elements(&format!("external {label}"), shape)?;
    if data.len() != elements {
        return Err(DiffusionError::InvalidRequest(format!(
            "external {label} has {} elements but shape {:?} expects {elements}",
            data.len(),
            shape
        )));
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err(DiffusionError::InvalidRequest(format!(
            "external {label} contains non-finite values"
        )));
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
    seeds: &[i64],
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
    let masked_encode_seeds = vae_encode_seeds(seeds, VAE_MASKED_ENCODE_SEED_SALT);
    let (masked_image_latents, masked_latents_kind) = encode_to_latents_with_runtime_context(
        encoder,
        &masked_image,
        Some(&masked_encode_seeds),
        runtime_context,
    )?;
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
    let transformer_topology = metadata
        .components
        .get("transformer")
        .map(transformer_denoiser_weight_topology);
    let uses_supported_transformer = transformer_topology
        .as_ref()
        .is_some_and(|topology| matches!(topology.family, TransformerDenoiserFamily::QwenImage));
    if !is_native_unet_pipeline_class(&metadata.pipeline.class_name) && !uses_supported_transformer
    {
        let denoiser = transformer_topology
            .as_ref()
            .map(|topology| format!("transformer denoiser ({})", topology.diagnostic_label()))
            .unwrap_or_else(|| "unsupported denoiser".to_string());
        return Some(format!(
            "native diffusion runtime currently supports Stable Diffusion UNet-family pipelines only; artifact pipeline {:?} uses a {denoiser} and requires a matching diffusion runtime",
            metadata.pipeline.class_name
        ));
    }
    if uses_supported_transformer {
        if let Some(topology) = transformer_topology.as_ref() {
            if topology.block_count == 0
                || !topology.has_input_projection
                || !topology.has_output_projection
                || !topology.has_text_modulation
            {
                return Some(format!(
                    "native transformer runtime requires complete Qwen image transformer weights; artifact has {}",
                    topology.diagnostic_label()
                ));
            }
        }
    } else {
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
    }
    if let Some(vae) = metadata
        .components
        .get("vae")
        .and_then(|component| component.class_name.as_deref())
    {
        if vae != "AutoencoderKL" && vae != "AutoencoderKLQwenImage" {
            return Some(format!(
                "native diffusion runtime supports AutoencoderKL-family VAEs only; artifact vae class {vae:?} is unsupported"
            ));
        }
    }
    if !uses_supported_transformer {
        let text_encoder_class = metadata
            .components
            .get("text_encoder")
            .and_then(|component| component.class_name.as_deref());
        if let Some(text_encoder) = text_encoder_class {
            if text_encoder != "CLIPTextModel" && text_encoder != "CLIPTextModelWithProjection" {
                return Some(format!(
                    "native diffusion runtime supports CLIP text encoders only; artifact text_encoder class {text_encoder:?} is unsupported"
                ));
            }
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

fn native_runtime_support_error(
    hfq: &HfqFile,
    metadata: &DiffusionHfqMetadata,
) -> DiffusionResult<Option<String>> {
    if let Some(error) = native_runtime_metadata_support_error(metadata) {
        return Ok(Some(error));
    }
    let transformer_topology = metadata
        .components
        .get("transformer")
        .map(transformer_denoiser_weight_topology);
    let uses_qwen_transformer = transformer_topology
        .as_ref()
        .is_some_and(|topology| matches!(topology.family, TransformerDenoiserFamily::QwenImage));
    if uses_qwen_transformer {
        let transformer_json = component_json(hfq, metadata, "transformer")?;
        if transformer_json
            .as_ref()
            .and_then(|json| json_bool(json, "guidance_embeds"))
            .unwrap_or(false)
        {
            return Ok(Some(
                "native transformer runtime does not support Qwen guidance-distilled transformer embeddings yet; guidance_embeds=true needs a separate guidance-scale embedding path, not classifier-free guidance"
                    .to_string(),
            ));
        }
    }
    Ok(None)
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
mod tests;
