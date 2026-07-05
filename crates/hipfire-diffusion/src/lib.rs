// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Diffusion model support for HFQ-backed Hipfire serving.
//!
//! This crate owns the stable metadata and batched runtime API for diffusion
//! artifacts. The first importer preserves Diffusers component weights as HFQ
//! role entries; later importer phases can replace those entries with decoded
//! and quantized tensors without changing server routing.

use base64::Engine;
use hipfire_runtime::hfq::HfqFile;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

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
/// Opus Quant 4-bit, 256-group, FWHT-rotated (130 B/block: f16 scale + 128
/// nibbles, range [-7,7]). Calibrated via hipfire_quantize::ldlq::oq4_ldlq_pack.
pub const QT_DIFFUSION_TENSOR_OQ4_G256: u8 = 9;
/// Opus Quant 8-bit, 256-group, FWHT-rotated (258 B/block: f16 scale + 256 i8).
pub const QT_DIFFUSION_TENSOR_OQ8_G256: u8 = 10;
pub const QT_DIFFUSION_TENSOR_BF16: u8 = 16;

mod metadata;
pub use metadata::*;
mod scheduler;
pub use scheduler::*;
mod config;
pub use config::*;
// Crate-internal config helpers (pub(crate)) also re-exported so sibling
// modules that used to see them as crate-root-private items keep resolving.
pub(crate) use config::{
    TransformerDenoiserFamily, TransformerDenoiserWeightTopology, VaeLatentNorm,
};

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
/// Per-step activation precision for the resident linear path. The progressive
/// schedule (set per denoise step) walks from cheap/lossy (W4A4) on the early,
/// high-noise steps to full precision (F16) on the final steps. All oq4 rungs
/// consume the same resident oq4-packed weight + FWHT-rotated activation; they
/// only apply to linears with `in % 256 == 0` (others fall back to F16).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinearPrecision {
    /// f16 weight × f16 activation (the Phase-3 WMMA path).
    #[default]
    F16,
    /// oq4 weight dequant→f16 × f16 activation (memory-bandwidth win).
    W4A16,
    /// oq4 weight × int8 (q8_1) activation.
    W4A8,
    /// oq4 weight × int4 activation (2× matrix rate on gfx1103).
    W4A4,
}

#[derive(Default)]
struct RocmWeightCache {
    entries: std::collections::HashMap<(usize, usize), hipfire_rdna::GpuTensor>,
    /// F16 copies of weights, for the Phase 3 WMMA-GEMM convolution path. Keyed
    /// the same way as `entries`; populated lazily by converting the resident
    /// F32 weight once.
    f16_entries: std::collections::HashMap<(usize, usize), hipfire_rdna::GpuTensor>,
    /// oq4 arch-combined device buffers, for the W4A* schedule rungs. Keyed the
    /// same way; built once by quantize_oq4g256 → pack_oq4_arch_combined → upload.
    oq4_entries: std::collections::HashMap<(usize, usize), hipfire_rdna::GpuTensor>,
    /// Active activation precision for the resident linear path this step (the
    /// per-STEP schedule). Used directly unless the per-LAYER policy overrides.
    linear_precision: LinearPrecision,
    /// Per-LAYER schedule policy. When `layer_stride > 0`, the precision of each
    /// resident linear is decided by its call index this forward (not the per-step
    /// `linear_precision`): every `layer_stride`-th linear runs `layer_rung`, the
    /// first `layer_skip_first` and last `layer_skip_last` stay F16.
    layer_stride: usize,
    layer_skip_first: usize,
    layer_skip_last: usize,
    layer_rung: LinearPrecision,
    /// Resident-linear call index within the current forward (reset per step).
    linear_index: usize,
    /// Total resident-linear calls in the previous forward (for `layer_skip_last`;
    /// 0 on the first step, so skip_last activates from step 1).
    linear_total: usize,
}

impl RocmWeightCache {
    /// Resolve the activation precision for the resident linear at call index
    /// `idx` this forward, applying the per-layer policy when active else the
    /// per-step precision.
    fn resolve_linear_precision(&self, idx: usize) -> LinearPrecision {
        if self.layer_stride == 0 {
            return self.linear_precision;
        }
        if idx < self.layer_skip_first {
            return LinearPrecision::F16;
        }
        if self.linear_total > 0 && idx >= self.linear_total.saturating_sub(self.layer_skip_last) {
            return LinearPrecision::F16;
        }
        if (idx - self.layer_skip_first) % self.layer_stride == 0 {
            self.layer_rung
        } else {
            LinearPrecision::F16
        }
    }
}

impl RocmWeightCache {
    /// Return the raw device pointer for `tensor`, uploading it once on first use.
    fn resident_ptr(
        &mut self,
        gpu: &mut hipfire_rdna::Gpu,
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

    /// Return the raw device pointer to an F16 copy of `tensor`, converting the
    /// (resident) F32 weight once on first use. The F16 buffer holds the same
    /// element count as `tensor`; the caller wraps it in a non-owning
    /// [`hipfire_rdna::GpuTensor`] for the GEMM.
    fn resident_f16_ptr(
        &mut self,
        gpu: &mut hipfire_rdna::Gpu,
        tensor: &CpuTensor,
    ) -> DiffusionResult<*mut std::ffi::c_void> {
        let key = (tensor.data.as_ptr() as usize, tensor.data.len());
        if !self.f16_entries.contains_key(&key) {
            let f32_ptr = self.resident_ptr(gpu, tensor)?;
            let n = tensor.data.len();
            let f16 = gpu
                .alloc_tensor(&[n], hipfire_rdna::DType::F16)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            let mut kernargs = hip_bridge::KernargBlob::new();
            kernargs.push_ptr(f32_ptr);
            kernargs.push_ptr(f16.buf.as_ptr());
            kernargs.push_i32(i32_kernel_dim("f32->f16 element count", n)?);
            kernargs.pad_to(16);
            let grid = [((n as u32).saturating_add(255)) / 256, 1, 1];
            // No sync: the convert runs on the same stream as, and is enqueued
            // before, the GEMM that reads the F16 buffer.
            ensure_and_launch_diffusion_kernel(
                gpu,
                "diffusion_f32_to_f16",
                DIFFUSION_F32_TO_F16_HIP_SRC,
                "diffusion_f32_to_f16",
                grid,
                [256, 1, 1],
                0,
                &mut kernargs,
            )?;
            self.f16_entries.insert(key, f16);
        }
        Ok(self
            .f16_entries
            .get(&key)
            .expect("f16 weight just inserted")
            .buf
            .as_ptr())
    }

    /// Return the device pointer to the oq4 arch-combined buffer for `tensor`
    /// (`[m, k]`, `k % 256 == 0`), building it once: quantize the resident f32
    /// weight to oq4g256 (FWHT-rotated), repack to the arch combined layout, and
    /// upload. Consumed by the W4A* GEMM kernels. The returned buffer is
    /// `pack_oq4_arch_combined`-sized.
    fn resident_oq4_ptr(
        &mut self,
        gpu: &mut hipfire_rdna::Gpu,
        tensor: &CpuTensor,
        m: usize,
        k: usize,
    ) -> DiffusionResult<*mut std::ffi::c_void> {
        let key = (tensor.data.as_ptr() as usize, tensor.data.len());
        if !self.oq4_entries.contains_key(&key) {
            let signs1 = hipfire_quantize::gen_fwht_signs(quant_decode::OQ_FWHT_SEED1, 256);
            let signs2 = hipfire_quantize::gen_fwht_signs(quant_decode::OQ_FWHT_SEED2, 256);
            let oq4 = hipfire_quantize::codecs::quantize_oq4g256(&tensor.data, &signs1, &signs2);
            let packed = quant_encode::pack_oq4_arch_combined(&oq4, m, k);
            let resident = gpu
                .upload_raw(&packed, &[packed.len()])
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            self.oq4_entries.insert(key, resident);
        }
        Ok(self
            .oq4_entries
            .get(&key)
            .expect("oq4 weight just inserted")
            .buf
            .as_ptr())
    }
}

struct DiffusionGenerationRuntimeContext {
    options: DiffusionGenerationRuntimeOptions,
    rocm_gpu: Option<hipfire_rdna::Gpu>,
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
            let gpu = hipfire_rdna::Gpu::init_with_device(device_id)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            self.rocm_gpu = Some(gpu);
            self.rocm_gpu_init_count += 1;
        }
        Ok(())
    }

    fn with_rocm_gpu<T>(
        &mut self,
        f: impl FnOnce(&mut hipfire_rdna::Gpu) -> DiffusionResult<T>,
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
        f: impl FnOnce(&mut hipfire_rdna::Gpu, &mut RocmWeightCache) -> DiffusionResult<T>,
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

mod batch;
pub use batch::*;
pub(crate) use batch::DenoiseLatentsOutput;

mod denoise;
pub use denoise::*;

mod ops_dispatch;
pub(crate) use ops_dispatch::*;

fn f32_slices_close(actual: &[f32], expected: &[f32], tolerance: f32) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= tolerance)
}

#[derive(Debug, Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
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

pub(crate) fn box_muller_pair(rng: &mut SplitMix64) -> (f32, f32) {
    let u1 = rng.next_unit().max(f32::MIN_POSITIVE);
    let u2 = rng.next_unit();
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (radius * theta.cos(), radius * theta.sin())
}

// CpuTensor now lives in the hipfire-cpu backend crate (re-exported at the top
// of this file). Its `from_hfq` constructor stays here as a free function since
// it couples to the HFQ format loader (hipfire-runtime), which the low-level
// hipfire-cpu crate does not depend on.

/// Parity-debug tensor dump: when `HIPFIRE_DIFFUSION_DUMP_DIR` is set, write
/// `<dir>/<name>.npy` (numpy v1, `<f4`) so intermediate activations can be diffed
/// against a diffusers reference. No-op (and never errors the run) otherwise.
pub(crate) fn dump_debug_tensor(name: &str, tensor: &CpuTensor) {
    let Ok(dir) = std::env::var("HIPFIRE_DIFFUSION_DUMP_DIR") else {
        return;
    };
    let _ = std::fs::create_dir_all(&dir);
    let shape = tensor
        .shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let shape = if tensor.shape.len() == 1 {
        format!("{shape},")
    } else {
        shape
    };
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape}), }}");
    // Pad so total (magic 8 + len 2 + header) is a multiple of 64, header ends '\n'.
    let mut header = header.into_bytes();
    let total = 10 + header.len() + 1;
    let pad = (64 - total % 64) % 64;
    header.extend(std::iter::repeat_n(b' ', pad));
    header.push(b'\n');
    let mut out = Vec::with_capacity(10 + header.len() + tensor.data.len() * 4);
    out.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    for value in &tensor.data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{name}.npy")), out);
}

/// Decode a raw tensor payload (`quant_type` + bytes) into `f32`. Shared by
/// `cpu_tensor_from_hfq` (decode-at-load) and `ResidentWeight` (decode-on-use).
pub(crate) fn decode_tensor_payload(
    name: &str,
    quant_type: u8,
    bytes: &[u8],
    elem_count: usize,
) -> DiffusionResult<Vec<f32>> {
    Ok(match quant_type {
        QT_DIFFUSION_TENSOR_Q4F16_G64 => decode_q4f16_g64_slice(name, bytes, elem_count)?,
        QT_DIFFUSION_TENSOR_F16 => decode_f16_slice(bytes),
        QT_DIFFUSION_TENSOR_BF16 => decode_bf16_slice(bytes),
        QT_DIFFUSION_TENSOR_F32 => decode_f32_slice(bytes),
        QT_DIFFUSION_TENSOR_Q8F16 => decode_q8f16_slice(name, bytes, elem_count)?,
        QT_DIFFUSION_TENSOR_Q4_K => decode_q4_k_slice(name, bytes, elem_count)?,
        QT_DIFFUSION_TENSOR_HFQ4_G256 => decode_hfq4_slice(name, bytes, elem_count, 256, 136, "HFQ4G256")?,
        QT_DIFFUSION_TENSOR_HFQ4_G128 => decode_hfq4_slice(name, bytes, elem_count, 128, 72, "HFQ4G128")?,
        QT_DIFFUSION_TENSOR_HFQ6_G256 => decode_hfq6_g256_slice(name, bytes, elem_count)?,
        QT_DIFFUSION_TENSOR_OQ4_G256 => decode_oq4g256_slice(name, bytes, elem_count)?,
        QT_DIFFUSION_TENSOR_OQ8_G256 => decode_oq8g256_slice(name, bytes, elem_count)?,
        other => {
            return Err(DiffusionError::InvalidMetadata(format!(
                "tensor {name:?} has unsupported quant_type {other}; native diffusion tensor decoding currently supports Q4F16_G64, f16, bf16, f32, Q8F16, Q4_K, HFQ4G256, HFQ4G128, HFQ6G256, OQ4G256, and OQ8G256 tensor payloads. Other packed or quantized payloads require a diffusion dequantizer/runtime implementation"
            )))
        }
    })
}

/// A weight kept resident in its **on-disk packed form** (bf16 ~2 B/elem, oq4
/// ~0.5 B/elem, etc.) and decoded to `f32` **transiently per forward call**,
/// instead of holding the expanded `f32` (4 B/elem) for the lifetime of the
/// model. This roughly halves resident memory for a bf16 model (and cuts it ~8x
/// for oq4), letting the full Krea2 model fit where the f32 working set would
/// OOM. `decode()` allocates the f32 only for the duration of one op.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResidentWeight {
    name: String,
    quant_type: u8,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

#[allow(dead_code)]
impl ResidentWeight {
    pub(crate) fn from_hfq(hfq: &HfqFile, name: &str) -> DiffusionResult<Self> {
        let (info, bytes) = hfq.tensor_data_vec(name).ok_or_else(|| {
            DiffusionError::InvalidMetadata(format!("tensor {name:?} is missing"))
        })?;
        Ok(Self {
            name: name.to_string(),
            quant_type: info.quant_type,
            shape: info.shape.iter().map(|&dim| dim as usize).collect(),
            bytes,
        })
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Decode the packed payload into an f32 `CpuTensor` (transient; drop it
    /// immediately after the op to keep only one weight expanded at a time).
    pub(crate) fn decode(&self) -> DiffusionResult<CpuTensor> {
        let elem_count = self.shape.iter().product();
        let data = decode_tensor_payload(&self.name, self.quant_type, &self.bytes, elem_count)?;
        // Register this decode's data pointer for activation calibration so the
        // per-linear Hessian is captured for resident-packed weights too (the
        // `cpu_tensor_from_hfq` calibration hook does not see them). No-op unless
        // a calibration run is armed.
        if quant_calib::calib_active() {
            quant_calib::calib_register(data.as_ptr() as usize, &self.name);
        }
        Ok(CpuTensor {
            shape: self.shape.clone(),
            data,
        })
    }
}

/// Load a [`CpuTensor`] from an HFQ tensor by name. Free function (not a
/// `CpuTensor` method) because it couples to the HFQ loader / activation-calib
/// registry, which the low-level `hipfire-cpu` crate that owns `CpuTensor` does
/// not depend on. `CpuTensor::{zeros, rows_cols}` live in hipfire-cpu.
pub(crate) fn cpu_tensor_from_hfq(hfq: &HfqFile, name: &str) -> DiffusionResult<CpuTensor> {
    let (info, bytes) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| DiffusionError::InvalidMetadata(format!("tensor {name:?} is missing")))?;
    let elem_count = info
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim as usize))
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata(format!("tensor {name:?} shape overflows"))
        })?;
    let data = decode_tensor_payload(name, info.quant_type, &bytes, elem_count)?;
    if data.len() != elem_count {
        return Err(DiffusionError::InvalidMetadata(format!(
            "tensor {name:?} decoded {} elements but shape expects {elem_count}",
            data.len()
        )));
    }
    // Register this weight's data pointer for activation calibration (no-op
    // unless a calibration run is armed). The Vec buffer is stable across the
    // move into the tensor, so the pointer matches what the forward sees.
    if quant_calib::calib_active() {
        quant_calib::calib_register(data.as_ptr() as usize, name);
    }
    Ok(CpuTensor {
        shape: info.shape.iter().map(|&dim| dim as usize).collect(),
        data,
    })
}

mod quant_decode;
use quant_decode::*;

mod quant_encode;
pub use quant_encode::{
    open_calib_sidecar, oq4_arch_combined_len, pack_oq4_arch_combined, quantize_diffusion_hfq,
    DiffusionQuantFormat, DiffusionQuantizeSummary, HessianSidecar,
};

mod quant_calib;
pub use quant_calib::{
    calib_active, calib_begin, calib_finish_and_write, calib_observed_count,
    calibrate_diffusion_hfq, CalibrateSummary,
};
#[cfg(test)]
use quant_encode::{encode_q4f16_g64, encode_q4k, encode_q8f16};

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

/// CPU-op errors from hipfire-cpu surface as invalid-metadata (shape/precondition)
/// failures in the diffusion pipeline, so `?` on a `CpuResult` works everywhere.
impl From<hipfire_cpu::CpuError> for DiffusionError {
    fn from(err: hipfire_cpu::CpuError) -> Self {
        DiffusionError::InvalidMetadata(err.0)
    }
}

pub type DiffusionResult<T> = Result<T, DiffusionError>;

pub struct DiffusionPipeline {
    pub(crate) summary: DiffusionModelSummary,
    pub(crate) metadata: DiffusionHfqMetadata,
    pub(crate) config: StableDiffusionConfig,
    pub(crate) tokenizer: Option<ClipTokenizer>,
    pub(crate) tokenizer_2: Option<ClipTokenizer>,
    pub(crate) text_encoder: Option<ClipTextEncoder>,
    pub(crate) text_encoder_2: Option<ClipTextEncoder>,
    pub(crate) native_runtime: Option<NativeDiffusionRuntime>,
    pub(crate) native_runtime_error: Option<String>,
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
    // Krea2 text conditioning (Qwen3-VL encoder + text_fusion). Present only for
    // `Krea2Pipeline`; the pipeline drives it to build the DiT conditioning.
    text_conditioner: Option<Krea2TextConditioner>,
    // Qwen2 byte-level BPE tokenizer for the Krea2 prompt (from `tokenizer.json`).
    krea2_tokenizer: Option<hipfire_model::tokenizer::Tokenizer>,
}

impl NativeDiffusionRuntime {
    fn from_hfq(
        hfq: &HfqFile,
        metadata: &DiffusionHfqMetadata,
        config: &StableDiffusionConfig,
    ) -> DiffusionResult<Self> {
        let transformer_family = metadata
            .components
            .get("transformer")
            .map(transformer_denoiser_weight_topology)
            .map(|topology| topology.family);
        let noise: Box<dyn DiffusionNoiseBackend> =
            if let Some(transformer) = metadata.components.get("transformer") {
                let topology = transformer_denoiser_weight_topology(transformer);
                Box::new(NativeTransformerDenoiser::from_hfq(hfq, config, &topology)?)
            } else {
                Box::new(NativeUnet2DConditionModel::from_hfq(hfq, &config.unet)?)
            };
        let is_krea2 = matches!(transformer_family, Some(TransformerDenoiserFamily::Krea2));
        let text_conditioner = if is_krea2 {
            Self::load_krea2_conditioner(hfq, metadata, config)?
        } else {
            None
        };
        let krea2_tokenizer = if is_krea2 {
            hfq.tensor_data_vec("tokenizer/tokenizer.json")
                .and_then(|(_, bytes)| String::from_utf8(bytes).ok())
                .and_then(|json| hipfire_model::tokenizer::Tokenizer::from_hf_json(&json).ok())
        } else {
            None
        };
        Ok(Self {
            kind: DiffusionRuntimeKind::CpuSourceReference,
            noise,
            encoder: NativeVaeEncoder::from_hfq(hfq, &config.vae).ok(),
            decoder: Box::new(NativeVaeDecoder::from_hfq(hfq, &config.vae)?),
            text_conditioner,
            krea2_tokenizer,
        })
    }

    /// Tokenize a prompt and build the Krea2 DiT conditioning `[1, seq, hidden]`.
    /// `None` for non-Krea2 runtimes or when the tokenizer/conditioner is absent.
    fn krea2_conditioning_from_prompt(
        &self,
        prompt: &str,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<Option<CpuTensor>> {
        let (Some(tokenizer), Some(_)) = (&self.krea2_tokenizer, &self.text_conditioner) else {
            return Ok(None);
        };
        let token_ids = tokenizer.encode(prompt);
        self.krea2_conditioning_from_token_ids(&token_ids, runtime_context)
    }

    /// Krea2 conditioning for a tokenized prompt (`None` for non-Krea2 runtimes).
    /// The pipeline drives this then feeds the result through the transformer
    /// denoiser's external-conditioning seam. (Generate-path call site is the
    /// remaining pipeline glue.)
    #[allow(dead_code)]
    fn krea2_conditioning_from_token_ids(
        &self,
        token_ids: &[u32],
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<Option<CpuTensor>> {
        match &self.text_conditioner {
            Some(conditioner) => conditioner
                .conditioning_from_token_ids(token_ids, runtime_context)
                .map(Some),
            None => Ok(None),
        }
    }

    /// Build the Krea2 text conditioner from the artifact: encoder geometry from
    /// the `text_encoder` (Qwen3-VL `text_config`), the selected layers from
    /// `model_index.text_encoder_select_layers`, and the fusion head count from
    /// the transformer config.
    fn load_krea2_conditioner(
        hfq: &HfqFile,
        metadata: &DiffusionHfqMetadata,
        config: &StableDiffusionConfig,
    ) -> DiffusionResult<Option<Krea2TextConditioner>> {
        let text_encoder =
            component_json(hfq, metadata, "text_encoder")?.unwrap_or_else(|| json!({}));
        let text_config = text_encoder
            .get("text_config")
            .cloned()
            .unwrap_or_else(|| text_encoder.clone());
        let heads = json_usize(&text_config, "num_attention_heads").unwrap_or(32);
        let kv_heads = json_usize(&text_config, "num_key_value_heads").unwrap_or(8);
        let head_dim = json_usize(&text_config, "head_dim").unwrap_or(128);
        let rope_theta = text_config
            .get("rope_parameters")
            .and_then(|params| json_f32(params, "rope_theta"))
            .or_else(|| json_f32(&text_config, "rope_theta"))
            .unwrap_or(5_000_000.0);
        // select_layers live in model_index.json (stored verbatim on import).
        let model_index = hfq
            .tensor_data_vec("diffusers/model_index.json")
            .and_then(|(_, bytes)| String::from_utf8(bytes).ok())
            .and_then(|text| parse_json_lenient(&text).ok())
            .unwrap_or_else(|| json!({}));
        let select_layers = json_usize_vec(&model_index, "text_encoder_select_layers");
        let fusion_heads = config
            .transformer
            .as_ref()
            .and_then(|transformer| transformer.text_num_attention_heads)
            .unwrap_or(20);
        Krea2TextConditioner::from_hfq(
            hfq,
            "text_encoder/tensors/language_model",
            heads,
            kv_heads,
            head_dim,
            rope_theta,
            fusion_heads,
            select_layers,
        )
    }
}

mod transformer;
pub(crate) use transformer::*;

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
            base_shift: json_f32(&scheduler_json, "base_shift"),
            max_shift: json_f32(&scheduler_json, "max_shift"),
            base_image_seq_len: json_usize(&scheduler_json, "base_image_seq_len"),
            max_image_seq_len: json_usize(&scheduler_json, "max_image_seq_len"),
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

mod layers;
pub use layers::*;

mod unet;
pub use unet::*;

mod vae;
pub use vae::*;

/// Domain-separation salts so VAE-encode Gaussian noise does not alias the
/// initial-latent noise stream (which seeds the request seed directly) or other
/// encode sites. The values are arbitrary fixed constants. Consumed by the
/// pipeline/inpaint encode sites via [`vae_encode_seeds`].
const VAE_INIT_ENCODE_SEED_SALT: u64 = 0x7661_655f_696e_6974; // "vae_init"
const VAE_MASKED_ENCODE_SEED_SALT: u64 = 0x7661_655f_6d61_736b; // "vae_mask"

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
        cpu_tensor_from_hfq(hfq, entry).map(Some)
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

// gelu now lives in hipfire-cpu::tensor_ops (re-exported at the crate root).

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
        let token_embedding = cpu_tensor_from_hfq(
            hfq,
            &format!("{component}/tensors/text_model.embeddings.token_embedding.weight"),
        )?;
        let position_embedding = cpu_tensor_from_hfq(
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
                q_proj_weight: cpu_tensor_from_hfq(
                    hfq,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                )?,
                q_proj_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.self_attn.q_proj.bias"))?,
                k_proj_weight: cpu_tensor_from_hfq(
                    hfq,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                )?,
                k_proj_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.self_attn.k_proj.bias"))?,
                v_proj_weight: cpu_tensor_from_hfq(
                    hfq,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                )?,
                v_proj_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.self_attn.v_proj.bias"))?,
                out_proj_weight: cpu_tensor_from_hfq(
                    hfq,
                    &format!("{prefix}.self_attn.out_proj.weight"),
                )?,
                out_proj_bias: cpu_tensor_from_hfq(
                    hfq,
                    &format!("{prefix}.self_attn.out_proj.bias"),
                )?,
                layer_norm1_weight: cpu_tensor_from_hfq(
                    hfq,
                    &format!("{prefix}.layer_norm1.weight"),
                )?,
                layer_norm1_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.layer_norm1.bias"))?,
                fc1_weight: cpu_tensor_from_hfq(hfq, &format!("{prefix}.mlp.fc1.weight"))?,
                fc1_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.mlp.fc1.bias"))?,
                fc2_weight: cpu_tensor_from_hfq(hfq, &format!("{prefix}.mlp.fc2.weight"))?,
                fc2_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.mlp.fc2.bias"))?,
                layer_norm2_weight: cpu_tensor_from_hfq(
                    hfq,
                    &format!("{prefix}.layer_norm2.weight"),
                )?,
                layer_norm2_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.layer_norm2.bias"))?,
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
            final_layer_norm_weight: cpu_tensor_from_hfq(
                hfq,
                &format!("{component}/tensors/text_model.final_layer_norm.weight"),
            )?,
            final_layer_norm_bias: cpu_tensor_from_hfq(
                hfq,
                &format!("{component}/tensors/text_model.final_layer_norm.bias"),
            )?,
            text_projection: cpu_tensor_from_hfq(
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
        // Phase 1b: when a GPU is present, keep the whole transformer stack
        // device-resident — upload the embedded tokens once, run the 12 encoder
        // layers + final layer-norm on-device, download once.
        if runtime_context.rocm_device_id().is_some() {
            return self.encode_resident(tokens, runtime_context);
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

    /// Phase 1b device-resident CLIP encode. The token+position embedding gather
    /// is a cheap host op (and avoids re-uploading the ~vocab×hidden embedding
    /// table to the device every call); its result uploads once, the encoder
    /// layers + final layer-norm run device-resident, and only the final
    /// hidden states download.
    fn encode_resident(
        &self,
        tokens: &[u32],
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let x_host = clip_token_position_embeddings(
            &self.token_embedding,
            &self.position_embedding,
            tokens,
        )?;
        if x_host.shape.as_slice() != [tokens.len(), self.hidden_size] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "CLIP embedding output shape {:?} does not match [{}, {}]",
                x_host.shape,
                tokens.len(),
                self.hidden_size
            )));
        }
        let n_heads = self.n_heads;
        runtime_context.with_rocm_gpu_weighted(|gpu, cache| {
            gpu.bind_thread()
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            let mut x = gpu
                .upload_f32(&x_host.data, &x_host.shape)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            for layer in &self.layers {
                x = layer.forward_resident(x, n_heads, gpu, cache)?;
            }
            let normed = layer_norm_resident(
                gpu,
                cache,
                &x,
                &self.final_layer_norm_weight,
                &self.final_layer_norm_bias,
                1e-5,
            )?;
            free_resident(gpu, x)?;
            let output = download_resident(gpu, &normed)?;
            free_resident(gpu, normed)?;
            Ok(output)
        })
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

    /// Phase 1b device-resident CLIP encoder layer (LN → causal self-attn →
    /// residual → LN → fc1 → QuickGELU → fc2 → residual). Takes ownership of the
    /// resident `x` and frees it once the residual no longer needs it.
    fn forward_resident(
        &self,
        x: hipfire_rdna::GpuTensor,
        n_heads: usize,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        let norm1 = layer_norm_resident(
            gpu,
            cache,
            &x,
            &self.layer_norm1_weight,
            &self.layer_norm1_bias,
            1e-5,
        )?;
        let attn = self.self_attention_resident(&norm1, n_heads, gpu, cache)?;
        free_resident(gpu, norm1)?;
        let residual1 = tensor_add_resident(gpu, &x, &attn)?;
        free_resident(gpu, x)?;
        free_resident(gpu, attn)?;
        let norm2 = layer_norm_resident(
            gpu,
            cache,
            &residual1,
            &self.layer_norm2_weight,
            &self.layer_norm2_bias,
            1e-5,
        )?;
        let hidden = linear_optional_bias_resident(
            gpu,
            cache,
            &norm2,
            &self.fc1_weight,
            Some(&self.fc1_bias),
        )?;
        free_resident(gpu, norm2)?;
        let activated = quick_gelu_resident(gpu, &hidden)?;
        free_resident(gpu, hidden)?;
        let mlp = linear_optional_bias_resident(
            gpu,
            cache,
            &activated,
            &self.fc2_weight,
            Some(&self.fc2_bias),
        )?;
        free_resident(gpu, activated)?;
        let out = tensor_add_resident(gpu, &residual1, &mlp)?;
        free_resident(gpu, residual1)?;
        free_resident(gpu, mlp)?;
        Ok(out)
    }

    /// Phase 1b device-resident CLIP self-attention (q/k/v projections → causal
    /// attention → out projection). `x` is resident (borrowed; caller owns it).
    fn self_attention_resident(
        &self,
        x: &hipfire_rdna::GpuTensor,
        n_heads: usize,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        let q = linear_optional_bias_resident(
            gpu,
            cache,
            x,
            &self.q_proj_weight,
            Some(&self.q_proj_bias),
        )?;
        let k = linear_optional_bias_resident(
            gpu,
            cache,
            x,
            &self.k_proj_weight,
            Some(&self.k_proj_bias),
        )?;
        let v = linear_optional_bias_resident(
            gpu,
            cache,
            x,
            &self.v_proj_weight,
            Some(&self.v_proj_bias),
        )?;
        let context = clip_causal_self_attention_resident(gpu, &q, &k, &v, n_heads)?;
        free_resident(gpu, q)?;
        free_resident(gpu, k)?;
        free_resident(gpu, v)?;
        let out = linear_optional_bias_resident(
            gpu,
            cache,
            &context,
            &self.out_proj_weight,
            Some(&self.out_proj_bias),
        )?;
        free_resident(gpu, context)?;
        Ok(out)
    }
}

mod cpu_ops;
pub(crate) use cpu_ops::*;
mod pipeline_preflight;
mod pipeline_generate;
mod pipeline_plan;
// CpuTensor + the pure CPU-reference tensor ops now live in the hipfire-cpu
// backend crate; re-export them so this crate's ~1,300 CpuTensor references and
// the ops' call sites resolve unchanged.
pub use hipfire_cpu::tensor_ops::*;

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
    if let Some(refine) = request.refine_sigma.as_ref() {
        if !refine.first_sigma.is_finite() || !(0.0 < refine.first_sigma && refine.first_sigma < 1.0)
        {
            return Err(DiffusionError::InvalidRequest(format!(
                "refine_sigma.first_sigma {} must be in (0, 1)",
                refine.first_sigma
            )));
        }
        if refine.steps == 0 {
            return Err(DiffusionError::InvalidRequest(
                "refine_sigma.steps must be greater than zero".to_string(),
            ));
        }
        if request.mask.is_some() {
            return Err(DiffusionError::InvalidRequest(
                "refine_sigma (MrFlow refine) does not support masked/inpaint requests".to_string(),
            ));
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
    let uses_supported_transformer = transformer_topology.as_ref().is_some_and(|topology| {
        matches!(
            topology.family,
            TransformerDenoiserFamily::QwenImage | TransformerDenoiserFamily::Krea2
        )
    });
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
            // QwenImage (double-stream) carries text modulation; Krea2 conditions
            // via the separate text_fusion module instead.
            let text_conditioning_ok = match topology.family {
                TransformerDenoiserFamily::Krea2 => topology.has_text_fusion,
                _ => topology.has_text_modulation,
            };
            if topology.block_count == 0
                || !topology.has_input_projection
                || !topology.has_output_projection
                || !text_conditioning_ok
            {
                return Some(format!(
                    "native transformer runtime requires complete Qwen image / Krea2 transformer weights; artifact has {}",
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

#[cfg(test)]
mod tests;
