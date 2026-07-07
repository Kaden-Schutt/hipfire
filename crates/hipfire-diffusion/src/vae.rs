// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Native VAE encoder/decoder: attention/resnet blocks, moments-to-latents
//! sampling, latent-space (de)normalization, and image<->latent conversion.

use super::*;

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

    pub(crate) fn forward_with_runtime_options(
        &self,
        input: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(input, &mut runtime_context)
    }

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident VAE self-attention block. Borrows the resident
    /// `input` (the caller owns it) and returns a resident output.
    pub(crate) fn forward_resident(
        &self,
        input: &hipfire_rdna::GpuTensor,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        let (batch, channels, height, width) = match input.shape.as_slice() {
            [b, c, h, w] => (*b, *c, *h, *w),
            other => {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "VAE attention expected a 4D NCHW tensor, got shape {other:?}"
                )))
            }
        };
        let normed = self.norm.forward_resident(input, gpu, cache)?;
        let bsc = nchw_to_bsc_resident(gpu, &normed)?;
        free_resident(gpu, normed)?;
        let attended = self.attention.forward_resident(&bsc, None, gpu, cache)?;
        free_resident(gpu, bsc)?;
        let back = bsc_to_nchw_resident(gpu, &attended, batch, channels, height, width)?;
        free_resident(gpu, attended)?;
        let out = tensor_add_resident(gpu, &back, input)?;
        free_resident(gpu, back)?;
        Ok(out)
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

    pub(crate) fn forward_with_runtime_options(
        &self,
        hidden: CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(hidden, &mut runtime_context)
    }

    pub(crate) fn forward_with_runtime_context(
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
    pub conv_in: Option<Conv2dLayer>,
    pub down_blocks: Vec<VaeEncoderDownBlock>,
    pub mid_resnet_0: Option<ResnetBlock2D>,
    pub mid_attention: Option<VaeAttentionBlock>,
    pub mid_resnet_1: Option<ResnetBlock2D>,
    pub conv_norm_out: Option<GroupNormLayer>,
    pub conv_out: Option<Conv2dLayer>,
    pub quant_conv: Option<Conv2dLayer>,
    // Wan / Qwen-Image (`AutoencoderKLQwenImage`) encoder: 3D causal convs +
    // RMSNorm, a distinct encode path. When present the SD body is unused.
    wan_encoder: Option<WanImageEncoder>,
    latent_norm: VaeLatentNorm,
}

impl NativeVaeEncoder {
    pub fn from_hfq(hfq: &HfqFile, config: &VaeConfig) -> DiffusionResult<Self> {
        let groups = config.norm_num_groups.unwrap_or(32);
        let eps = config.norm_eps.unwrap_or(1e-6);
        // Wan / Qwen-Image encoder takes a distinct path; the SD body is skipped.
        if let Some(wan) = WanImageEncoder::from_hfq(hfq, "vae/tensors/encoder")? {
            return Ok(Self {
                conv_in: None,
                down_blocks: Vec::new(),
                mid_resnet_0: None,
                mid_attention: None,
                mid_resnet_1: None,
                conv_norm_out: None,
                conv_out: None,
                quant_conv: None,
                wan_encoder: Some(wan),
                latent_norm: VaeLatentNorm::from_config(config)?,
            });
        }
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
            conv_in: Some(Conv2dLayer::from_hfq(
                hfq,
                "vae/tensors/encoder.conv_in.weight",
                Some("vae/tensors/encoder.conv_in.bias"),
                1,
            )?),
            down_blocks,
            mid_resnet_0,
            mid_attention,
            mid_resnet_1,
            conv_norm_out: Some(GroupNormLayer::from_hfq(
                hfq,
                "vae/tensors/encoder.conv_norm_out.weight",
                "vae/tensors/encoder.conv_norm_out.bias",
                groups,
                eps,
            )?),
            conv_out: Some(Conv2dLayer::from_hfq(
                hfq,
                "vae/tensors/encoder.conv_out.weight",
                Some("vae/tensors/encoder.conv_out.bias"),
                1,
            )?),
            quant_conv,
            wan_encoder: None,
            latent_norm: VaeLatentNorm::from_config(config)?,
        })
    }

    pub fn encode_tensor_moments(&self, image: &CpuTensor) -> DiffusionResult<CpuTensor> {
        self.encode_tensor_moments_with_runtime_options(
            image,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    pub(crate) fn encode_tensor_moments_with_runtime_options(
        &self,
        image: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.encode_tensor_moments_with_runtime_context(image, &mut runtime_context)
    }

    pub(crate) fn encode_tensor_moments_with_runtime_context(
        &self,
        image: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        // Wan / Qwen-Image path (3D causal, CPU — mirrors the wan_decoder path).
        if let Some(wan) = &self.wan_encoder {
            let _ = runtime_context;
            return wan.encode(image);
        }
        let conv_in = self.conv_in.as_ref().ok_or_else(|| {
            DiffusionError::InvalidMetadata("VAE encoder conv_in missing".to_string())
        })?;
        let conv_norm_out = self.conv_norm_out.as_ref().ok_or_else(|| {
            DiffusionError::InvalidMetadata("VAE encoder conv_norm_out missing".to_string())
        })?;
        let conv_out = self.conv_out.as_ref().ok_or_else(|| {
            DiffusionError::InvalidMetadata("VAE encoder conv_out missing".to_string())
        })?;
        let mut hidden = conv_in.forward_with_runtime_context(image, runtime_context)?;
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
        hidden = conv_norm_out.forward_with_runtime_context(&hidden, runtime_context)?;
        hidden = silu_with_runtime_context(&hidden, runtime_context)?;
        hidden = conv_out.forward_with_runtime_context(&hidden, runtime_context)?;
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
    pub(crate) fn encode_to_latents_with_runtime_options(
        &self,
        image: &RgbImageBatch,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<LatentBatch> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.encode_to_latents_with_runtime_context(image, &mut runtime_context)
    }

    pub(crate) fn encode_to_latents_with_runtime_context(
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
    pub(crate) fn encode_to_latents_sampled_with_runtime_context(
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

pub(crate) fn vae_moments_to_latents(
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

/// Derive decorrelated per-batch RNG seeds for a specific VAE encode site.
pub(crate) fn vae_encode_seeds(seeds: &[i64], salt: u64) -> Vec<i64> {
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
pub(crate) fn vae_moments_to_latents_sampled(
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

    pub(crate) fn forward_with_runtime_options(
        &self,
        hidden: CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(hidden, &mut runtime_context)
    }

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident up block. Takes ownership of the resident
    /// `hidden`, freeing each intermediate as the chain advances.
    pub(crate) fn forward_resident(
        &self,
        mut hidden: hipfire_rdna::GpuTensor,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        for resnet in &self.resnets {
            let next = resnet.forward_resident(&hidden, gpu, cache)?;
            free_resident(gpu, hidden)?;
            hidden = next;
        }
        if let Some(upsampler) = &self.upsampler {
            let upsampled = upsample_nearest2d_nchw_resident(gpu, &hidden, 2)?;
            free_resident(gpu, hidden)?;
            let convolved = upsampler.forward_resident(&upsampled, gpu, cache)?;
            free_resident(gpu, upsampled)?;
            hidden = convolved;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeVaeDecoder {
    pub post_quant_conv: Option<Conv2dLayer>,
    // SD (`AutoencoderKL`) decoder body. `None` when the artifact is a Wan /
    // Qwen-Image decoder (`wan_decoder` is used instead).
    pub conv_in: Option<Conv2dLayer>,
    pub mid_resnet_0: Option<ResnetBlock2D>,
    pub mid_attention: Option<VaeAttentionBlock>,
    pub mid_resnet_1: Option<ResnetBlock2D>,
    pub up_blocks: Vec<VaeDecoderUpBlock>,
    pub conv_norm_out: Option<GroupNormLayer>,
    pub conv_out: Option<Conv2dLayer>,
    // Wan / Qwen-Image (`AutoencoderKLQwenImage`) decoder; takes over the whole
    // decode when present.
    wan_decoder: Option<WanImageDecoder>,
    latent_norm: VaeLatentNorm,
}

impl NativeVaeDecoder {
    pub fn from_hfq(hfq: &HfqFile, config: &VaeConfig) -> DiffusionResult<Self> {
        let groups = config.norm_num_groups.unwrap_or(32);
        let eps = config.norm_eps.unwrap_or(1e-6);
        // Wan / Qwen-Image (`AutoencoderKLQwenImage`) decoders use 3D causal convs
        // and RMSNorm rather than the SD `AutoencoderKL` Conv2d/GroupNorm layout,
        // so they take a distinct decode path. Detect and load it here; the SD
        // body below is skipped entirely.
        if let Some(wan) = WanImageDecoder::from_hfq(hfq, "vae/tensors/decoder")? {
            return Ok(Self {
                post_quant_conv: None,
                conv_in: None,
                mid_resnet_0: None,
                mid_attention: None,
                mid_resnet_1: None,
                up_blocks: Vec::new(),
                conv_norm_out: None,
                conv_out: None,
                wan_decoder: Some(wan),
                latent_norm: VaeLatentNorm::from_config(config)?,
            });
        }
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
            conv_in: Some(Conv2dLayer::from_hfq(
                hfq,
                "vae/tensors/decoder.conv_in.weight",
                Some("vae/tensors/decoder.conv_in.bias"),
                1,
            )?),
            mid_resnet_0,
            mid_attention,
            mid_resnet_1,
            up_blocks,
            conv_norm_out: Some(GroupNormLayer::from_hfq(
                hfq,
                "vae/tensors/decoder.conv_norm_out.weight",
                "vae/tensors/decoder.conv_norm_out.bias",
                groups,
                eps,
            )?),
            conv_out: Some(Conv2dLayer::from_hfq(
                hfq,
                "vae/tensors/decoder.conv_out.weight",
                Some("vae/tensors/decoder.conv_out.bias"),
                1,
            )?),
            wan_decoder: None,
            latent_norm: VaeLatentNorm::from_config(config)?,
        })
    }

    pub fn decode_latents(&self, latents: &LatentBatch) -> DiffusionResult<CpuTensor> {
        self.decode_latents_with_runtime_options(
            latents,
            DiffusionGenerationRuntimeOptions::default(),
        )
    }

    pub(crate) fn decode_latents_with_runtime_options(
        &self,
        latents: &LatentBatch,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.decode_latents_with_runtime_context(latents, &mut runtime_context)
    }

    pub(crate) fn decode_latents_with_runtime_context(
        &self,
        latents: &LatentBatch,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let mut hidden = latents.as_nchw_tensor();
        hidden = denormalize_decode_latents(&hidden, &self.latent_norm, runtime_context)?;
        // Wan / Qwen-Image decoder: 3D-causal CPU path (no resident kernels yet).
        if let Some(wan) = &self.wan_decoder {
            return wan.decode(&hidden);
        }
        // Phase 1b: when a GPU is present, keep the whole decode device-resident —
        // upload once here, run the op chain on-device, download once at the end —
        // instead of round-tripping every activation through the host.
        if runtime_context.rocm_device_id().is_some() {
            return self.decode_latents_resident(hidden, runtime_context);
        }
        if let Some(post_quant_conv) = &self.post_quant_conv {
            hidden = post_quant_conv.forward_with_runtime_context(&hidden, runtime_context)?;
        }
        hidden = self
            .conv_in
            .as_ref()
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata("SD VAE decoder conv_in missing".into())
            })?
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
            .as_ref()
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata("SD VAE decoder conv_norm_out missing".into())
            })?
            .forward_with_runtime_context(&hidden, runtime_context)?;
        hidden = silu_with_runtime_context(&hidden, runtime_context)?;
        self.conv_out
            .as_ref()
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata("SD VAE decoder conv_out missing".into())
            })?
            .forward_with_runtime_context(&hidden, runtime_context)
    }

    /// Phase 1b device-resident decode. `hidden_host` is the denormalized latent
    /// (already on host); it is uploaded once, the full decoder runs with every
    /// activation staying on-device, and only the final RGB-space tensor is
    /// downloaded. Every resident intermediate is freed back to the pool.
    fn decode_latents_resident(
        &self,
        hidden_host: CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        runtime_context.with_rocm_gpu_weighted(|gpu, cache| {
            gpu.bind_thread()
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            let mut hidden = gpu
                .upload_f32(&hidden_host.data, &hidden_host.shape)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            if let Some(post_quant_conv) = &self.post_quant_conv {
                let next = post_quant_conv.forward_resident(&hidden, gpu, cache)?;
                free_resident(gpu, hidden)?;
                hidden = next;
            }
            let next = self
                .conv_in
                .as_ref()
                .ok_or_else(|| {
                    DiffusionError::InvalidMetadata("SD VAE decoder conv_in missing".into())
                })?
                .forward_resident(&hidden, gpu, cache)?;
            free_resident(gpu, hidden)?;
            hidden = next;
            if let Some(resnet) = &self.mid_resnet_0 {
                let next = resnet.forward_resident(&hidden, gpu, cache)?;
                free_resident(gpu, hidden)?;
                hidden = next;
            }
            if let Some(attention) = &self.mid_attention {
                let next = attention.forward_resident(&hidden, gpu, cache)?;
                free_resident(gpu, hidden)?;
                hidden = next;
            }
            if let Some(resnet) = &self.mid_resnet_1 {
                let next = resnet.forward_resident(&hidden, gpu, cache)?;
                free_resident(gpu, hidden)?;
                hidden = next;
            }
            for block in &self.up_blocks {
                hidden = block.forward_resident(hidden, gpu, cache)?;
            }
            let next = self
                .conv_norm_out
                .as_ref()
                .ok_or_else(|| {
                    DiffusionError::InvalidMetadata("SD VAE decoder conv_norm_out missing".into())
                })?
                .forward_resident(&hidden, gpu, cache)?;
            free_resident(gpu, hidden)?;
            hidden = next;
            let next = silu_resident(gpu, &hidden)?;
            free_resident(gpu, hidden)?;
            hidden = next;
            let next = self
                .conv_out
                .as_ref()
                .ok_or_else(|| {
                    DiffusionError::InvalidMetadata("SD VAE decoder conv_out missing".into())
                })?
                .forward_resident(&hidden, gpu, cache)?;
            free_resident(gpu, hidden)?;
            hidden = next;
            let output = download_resident(gpu, &hidden)?;
            free_resident(gpu, hidden)?;
            Ok(output)
        })
    }

    pub fn decode_to_rgb8(&self, latents: &LatentBatch) -> DiffusionResult<RgbImageBatch> {
        let decoded = self.decode_latents(latents)?;
        rgb_tensor_to_u8(&decoded)
    }
}

/// Wan / Qwen-Image VAE channel RMSNorm (`AutoencoderKLQwenImage` `norm.gamma`).
///
/// Normalizes each spatial position across the channel axis and rescales by the
/// per-channel `gamma`. Input is `[batch, channels, height, width]` (the T=1
/// still-image collapse of the VAE's `[B, C, T, H, W]`); `gamma` is length
/// `channels`. This is the decoder's normalization throughout — the Wan VAE uses
/// RMSNorm rather than the SD VAE's GroupNorm.
#[allow(dead_code)]
pub(crate) fn wan_rms_norm_nchw(
    input: &CpuTensor,
    gamma: &[f32],
    eps: f32,
) -> DiffusionResult<CpuTensor> {
    let [batch, channels, height, width] = match input.shape.as_slice() {
        [b, c, h, w] => [*b, *c, *h, *w],
        _ => {
            return Err(DiffusionError::InvalidMetadata(format!(
                "wan_rms_norm expects [batch, channels, height, width], got {:?}",
                input.shape
            )))
        }
    };
    if gamma.len() != channels {
        return Err(DiffusionError::InvalidMetadata(format!(
            "wan_rms_norm gamma length {} != channels {channels}",
            gamma.len()
        )));
    }
    let spatial = height * width;
    let mut out = CpuTensor::zeros(&input.shape);
    for b in 0..batch {
        for pos in 0..spatial {
            let mut square_sum = 0.0f32;
            for c in 0..channels {
                let value = input.data[(b * channels + c) * spatial + pos];
                square_sum += value * value;
            }
            let inv_rms = (square_sum / channels as f32 + eps).sqrt().recip();
            for c in 0..channels {
                let idx = (b * channels + c) * spatial + pos;
                out.data[idx] = input.data[idx] * inv_rms * gamma[c];
            }
        }
    }
    Ok(out)
}

/// Wan / Qwen-Image causal `Conv3d` specialized for the still-image (T=1) case.
///
/// The Wan VAE uses causal temporal convolutions (past-padded, no future tap).
/// For a single latent frame every past temporal position is zero-padded, so
/// only the **last** temporal kernel tap contributes — the `Conv3d`
/// `[O, I, KT, KH, KW]` collapses to a `Conv2d` `[O, I, KH, KW]` using
/// `weight[:, :, KT-1, :, :]`. Spatial padding is `(KH-1)/2` (same-size for the
/// odd 3x3 / 1x1 kernels the decoder uses). This lets the 3D decoder reuse the
/// 2D conv path for image generation; a full T>1 video path would iterate the
/// temporal taps with a causal feature cache.
#[allow(dead_code)]
pub(crate) fn wan_causal_conv2d(
    input: &CpuTensor,
    weight3d: &CpuTensor,
    bias: Option<&CpuTensor>,
) -> DiffusionResult<CpuTensor> {
    let [out_c, in_c, kt, kh, kw] = match weight3d.shape.as_slice() {
        [o, i, t, h, w] => [*o, *i, *t, *h, *w],
        _ => {
            return Err(DiffusionError::InvalidMetadata(format!(
                "wan_causal_conv2d expects a 5-D Conv3d weight [O,I,KT,KH,KW], got {:?}",
                weight3d.shape
            )))
        }
    };
    if kt == 0 || kh == 0 || kw == 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "wan_causal_conv2d kernel dims must be positive, got {:?}",
            weight3d.shape
        )));
    }
    if kh != kw {
        return Err(DiffusionError::InvalidMetadata(format!(
            "wan_causal_conv2d only supports square spatial kernels, got {kh}x{kw}"
        )));
    }
    // T=1 causal-conv temporal collapse. QwenImageCausalConv3d pads the temporal
    // axis CAUSALLY with ZEROS: `_padding = (.., 2*padding[0], 0)` and F.pad's
    // default constant(0) mode. So a single frame is padded to `[0, .., 0, x]`
    // and the Conv3d output is `sum_t weight[t] * padded[t] = weight[KT-1] * x` --
    // only the LAST temporal tap survives (the earlier taps multiply zeros). The
    // replication `cache_x` path only applies in video mode with a feat_cache,
    // not still-image decode. (Summing all taps over-weights every kt=3 conv ~3x
    // and biases/brightens the whole decode -> a per-channel color cast.)
    let last_t = kt - 1;
    let mut weight2d = CpuTensor::zeros(&[out_c, in_c, kh, kw]);
    for o in 0..out_c {
        for i in 0..in_c {
            for y in 0..kh {
                for x in 0..kw {
                    let dst = ((o * in_c + i) * kh + y) * kw + x;
                    weight2d.data[dst] =
                        weight3d.data[(((o * in_c + i) * kt + last_t) * kh + y) * kw + x];
                }
            }
        }
    }
    conv2d_nchw(input, &weight2d, bias, (kh - 1) / 2).map_err(Into::into)
}

/// SiLU activation over a flat tensor (`x * sigmoid(x)`).
#[allow(dead_code)]
pub(crate) fn wan_silu(input: &CpuTensor) -> CpuTensor {
    CpuTensor {
        shape: input.shape.clone(),
        data: input.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
    }
}

fn wan_add(a: &CpuTensor, b: &CpuTensor) -> DiffusionResult<CpuTensor> {
    if a.shape != b.shape {
        return Err(DiffusionError::InvalidMetadata(format!(
            "wan_add shape mismatch {:?}/{:?}",
            a.shape, b.shape
        )));
    }
    Ok(CpuTensor {
        shape: a.shape.clone(),
        data: a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect(),
    })
}

/// Wan / Qwen-Image VAE residual block (still-image T=1 path):
/// `RMSNorm -> SiLU -> CausalConv3d -> RMSNorm -> SiLU -> CausalConv3d`, added to
/// the input (through a 1x1 `conv_shortcut` when the channel count changes).
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct WanResnetBlock {
    norm1_gamma: Vec<f32>,
    conv1_weight: CpuTensor,
    conv1_bias: CpuTensor,
    norm2_gamma: Vec<f32>,
    conv2_weight: CpuTensor,
    conv2_bias: CpuTensor,
    conv_shortcut: Option<(CpuTensor, CpuTensor)>,
}

#[allow(dead_code)]
impl WanResnetBlock {
    const EPS: f32 = 1e-6;

    pub(crate) fn from_hfq(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Self> {
        let conv_shortcut = if hfq
            .find_tensor_info(&format!("{prefix}.conv_shortcut.weight"))
            .is_some()
        {
            Some((
                cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv_shortcut.weight"))?,
                cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv_shortcut.bias"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            norm1_gamma: cpu_tensor_from_hfq(hfq, &format!("{prefix}.norm1.gamma"))?.data,
            conv1_weight: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv1.weight"))?,
            conv1_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv1.bias"))?,
            norm2_gamma: cpu_tensor_from_hfq(hfq, &format!("{prefix}.norm2.gamma"))?.data,
            conv2_weight: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv2.weight"))?,
            conv2_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv2.bias"))?,
            conv_shortcut,
        })
    }

    pub(crate) fn forward(&self, input: &CpuTensor) -> DiffusionResult<CpuTensor> {
        let hidden = wan_silu(&wan_rms_norm_nchw(input, &self.norm1_gamma, Self::EPS)?);
        let hidden = wan_causal_conv2d(&hidden, &self.conv1_weight, Some(&self.conv1_bias))?;
        let hidden = wan_silu(&wan_rms_norm_nchw(&hidden, &self.norm2_gamma, Self::EPS)?);
        let hidden = wan_causal_conv2d(&hidden, &self.conv2_weight, Some(&self.conv2_bias))?;
        let shortcut = match &self.conv_shortcut {
            Some((weight, bias)) => wan_causal_conv2d(input, weight, Some(bias))?,
            None => input.clone(),
        };
        wan_add(&hidden, &shortcut)
    }
}

/// Wan / Qwen-Image VAE mid-block spatial self-attention. RMSNorm, a 1x1 `to_qkv`
/// projection to `3C` channels, single-head scaled-dot-product attention over the
/// `H*W` spatial positions, a 1x1 output projection, and a residual add.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct WanMidAttention {
    norm_gamma: Vec<f32>,
    qkv_weight: CpuTensor,
    qkv_bias: CpuTensor,
    proj_weight: CpuTensor,
    proj_bias: CpuTensor,
}

#[allow(dead_code)]
impl WanMidAttention {
    const EPS: f32 = 1e-6;

    pub(crate) fn from_hfq(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Self> {
        Ok(Self {
            norm_gamma: cpu_tensor_from_hfq(hfq, &format!("{prefix}.norm.gamma"))?.data,
            qkv_weight: cpu_tensor_from_hfq(hfq, &format!("{prefix}.to_qkv.weight"))?,
            qkv_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.to_qkv.bias"))?,
            proj_weight: cpu_tensor_from_hfq(hfq, &format!("{prefix}.proj.weight"))?,
            proj_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.proj.bias"))?,
        })
    }

    pub(crate) fn forward(&self, input: &CpuTensor) -> DiffusionResult<CpuTensor> {
        let [batch, channels, height, width] = match input.shape.as_slice() {
            [b, c, h, w] => [*b, *c, *h, *w],
            _ => {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "wan mid attention expects [batch, channels, height, width], got {:?}",
                    input.shape
                )))
            }
        };
        let normed = wan_rms_norm_nchw(input, &self.norm_gamma, Self::EPS)?;
        // 1x1 conv to [batch, 3*channels, h, w]; channels [0,C) = q, [C,2C) = k,
        // [2C,3C) = v.
        let qkv = conv2d_nchw(&normed, &self.qkv_weight, Some(&self.qkv_bias), 0)?;
        let spatial = height * width;
        let scale = (channels as f32).sqrt().recip();
        let qkv_stride = 3 * channels;
        // q/k/v accessor: value for (batch, channel, position) within the split.
        let get = |b: usize, split: usize, ch: usize, pos: usize| -> f32 {
            qkv.data[((b * qkv_stride) + split * channels + ch) * spatial + pos]
        };
        let mut attended = CpuTensor::zeros(&[batch, channels, height, width]);
        let mut scores = vec![0.0f32; spatial];
        for b in 0..batch {
            for i in 0..spatial {
                // scores[j] = scale * sum_ch q[i,ch] * k[j,ch]
                let mut max_score = f32::NEG_INFINITY;
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for ch in 0..channels {
                        acc += get(b, 0, ch, i) * get(b, 1, ch, j);
                    }
                    *score = acc * scale;
                    max_score = max_score.max(*score);
                }
                let mut denom = 0.0f32;
                for score in scores.iter_mut() {
                    *score = (*score - max_score).exp();
                    denom += *score;
                }
                let inv_denom = denom.recip();
                // out[i,ch] = sum_j softmax[j] * v[j,ch]
                for ch in 0..channels {
                    let mut acc = 0.0f32;
                    for (j, score) in scores.iter().enumerate() {
                        acc += score * get(b, 2, ch, j);
                    }
                    attended.data[(b * channels + ch) * spatial + i] = acc * inv_denom;
                }
            }
        }
        let projected = conv2d_nchw(&attended, &self.proj_weight, Some(&self.proj_bias), 0)?;
        wan_add(input, &projected)
    }
}

/// Wan / Qwen-Image VAE spatial upsampler (still-image path): 2x nearest-neighbour
/// upsampling followed by the `resample` 3x3 conv. The temporal `time_conv` is a
/// no-op for T=1 and is skipped.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct WanUpsample {
    resample_weight: CpuTensor,
    resample_bias: CpuTensor,
}

#[allow(dead_code)]
impl WanUpsample {
    pub(crate) fn from_hfq(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Option<Self>> {
        // `resample` is a small Sequential(Upsample, Conv2d); the conv lives at
        // one of the low indices. Find whichever index carries the weight.
        for index in 0..4 {
            let weight_entry = format!("{prefix}.resample.{index}.weight");
            if hfq.find_tensor_info(&weight_entry).is_some() {
                return Ok(Some(Self {
                    resample_weight: cpu_tensor_from_hfq(hfq, &weight_entry)?,
                    resample_bias: cpu_tensor_from_hfq(
                        hfq,
                        &format!("{prefix}.resample.{index}.bias"),
                    )?,
                }));
            }
        }
        Ok(None)
    }

    pub(crate) fn forward(&self, input: &CpuTensor) -> DiffusionResult<CpuTensor> {
        let upsampled = upsample_nearest2d_nchw(input, 2)?;
        let [_, _, kh, kw] = shape4(&self.resample_weight)?;
        let padding = if kh == kw {
            (kh.saturating_sub(1)) / 2
        } else {
            0
        };
        conv2d_nchw(
            &upsampled,
            &self.resample_weight,
            Some(&self.resample_bias),
            padding,
        )
        .map_err(Into::into)
    }
}

/// One decoder up-block: a run of residual blocks followed by an optional
/// spatial upsampler.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct WanUpBlock {
    resnets: Vec<WanResnetBlock>,
    upsampler: Option<WanUpsample>,
}

/// Full Wan / Qwen-Image (`AutoencoderKLQwenImage`) VAE decoder, still-image
/// (T=1) path: `conv_in -> mid(resnet, attn, resnet) -> up_blocks -> norm_out ->
/// SiLU -> conv_out`, mapping a `[B, z_dim, H, W]` latent to `[B, 3, 8H, 8W]`
/// pixels. Reuses the tested Wan building blocks; the causal 3D convs collapse to
/// 2D for a single frame.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct WanImageDecoder {
    conv_in_weight: CpuTensor,
    conv_in_bias: CpuTensor,
    mid_resnet0: WanResnetBlock,
    mid_attention: WanMidAttention,
    mid_resnet1: WanResnetBlock,
    up_blocks: Vec<WanUpBlock>,
    norm_out_gamma: Vec<f32>,
    conv_out_weight: CpuTensor,
    conv_out_bias: CpuTensor,
}

#[allow(dead_code)]
impl WanImageDecoder {
    const EPS: f32 = 1e-6;

    fn count(hfq: &HfqFile, entry: impl Fn(usize) -> String) -> usize {
        let mut n = 0;
        while hfq.find_tensor_info(&entry(n)).is_some() {
            n += 1;
        }
        n
    }

    /// Load the decoder from an hfq under `prefix` (e.g. `vae/tensors/decoder`
    /// in an imported artifact, or `decoder` for in-memory fixtures). Returns
    /// `None` when this is not a Wan / Qwen-Image decoder. The discriminator is
    /// `{prefix}.norm_out.gamma`: the Wan decoder uses RMSNorm (`gamma`) where the
    /// SD `AutoencoderKL` decoder uses GroupNorm (`conv_norm_out.weight`), so both
    /// share `conv_in.weight` and only the output norm distinguishes them.
    pub(crate) fn from_hfq(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Option<Self>> {
        if hfq
            .find_tensor_info(&format!("{prefix}.norm_out.gamma"))
            .is_none()
        {
            return Ok(None);
        }
        let up_block_count = Self::count(hfq, |n| {
            format!("{prefix}.up_blocks.{n}.resnets.0.conv1.weight")
        });
        let mut up_blocks = Vec::with_capacity(up_block_count);
        for ub in 0..up_block_count {
            let resnet_count = Self::count(hfq, |r| {
                format!("{prefix}.up_blocks.{ub}.resnets.{r}.conv1.weight")
            });
            let resnets = (0..resnet_count)
                .map(|r| {
                    WanResnetBlock::from_hfq(hfq, &format!("{prefix}.up_blocks.{ub}.resnets.{r}"))
                })
                .collect::<DiffusionResult<Vec<_>>>()?;
            let upsampler =
                WanUpsample::from_hfq(hfq, &format!("{prefix}.up_blocks.{ub}.upsamplers.0"))?;
            up_blocks.push(WanUpBlock { resnets, upsampler });
        }
        Ok(Some(Self {
            conv_in_weight: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv_in.weight"))?,
            conv_in_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv_in.bias"))?,
            mid_resnet0: WanResnetBlock::from_hfq(hfq, &format!("{prefix}.mid_block.resnets.0"))?,
            mid_attention: WanMidAttention::from_hfq(
                hfq,
                &format!("{prefix}.mid_block.attentions.0"),
            )?,
            mid_resnet1: WanResnetBlock::from_hfq(hfq, &format!("{prefix}.mid_block.resnets.1"))?,
            up_blocks,
            norm_out_gamma: cpu_tensor_from_hfq(hfq, &format!("{prefix}.norm_out.gamma"))?.data,
            conv_out_weight: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv_out.weight"))?,
            conv_out_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv_out.bias"))?,
        }))
    }

    pub(crate) fn decode(&self, latent: &CpuTensor) -> DiffusionResult<CpuTensor> {
        let mut hidden = wan_causal_conv2d(latent, &self.conv_in_weight, Some(&self.conv_in_bias))?;
        hidden = self.mid_resnet0.forward(&hidden)?;
        hidden = self.mid_attention.forward(&hidden)?;
        hidden = self.mid_resnet1.forward(&hidden)?;
        for up_block in &self.up_blocks {
            for resnet in &up_block.resnets {
                hidden = resnet.forward(&hidden)?;
            }
            if let Some(upsampler) = &up_block.upsampler {
                hidden = upsampler.forward(&hidden)?;
            }
        }
        hidden = wan_silu(&wan_rms_norm_nchw(
            &hidden,
            &self.norm_out_gamma,
            Self::EPS,
        )?);
        wan_causal_conv2d(&hidden, &self.conv_out_weight, Some(&self.conv_out_bias))
    }
}

/// Wan / Qwen-Image VAE spatial downsampler (still-image path): asymmetric
/// zero-pad (right+bottom) then the `resample` 3x3 stride-2 conv, halving H/W.
/// The temporal `time_conv` is a no-op for T=1 and is skipped.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct WanDownsample {
    resample_weight: CpuTensor,
    resample_bias: CpuTensor,
}

#[allow(dead_code)]
impl WanDownsample {
    pub(crate) fn from_hfq(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Option<Self>> {
        for index in 0..4 {
            let weight_entry = format!("{prefix}.resample.{index}.weight");
            if hfq.find_tensor_info(&weight_entry).is_some() {
                return Ok(Some(Self {
                    resample_weight: cpu_tensor_from_hfq(hfq, &weight_entry)?,
                    resample_bias: cpu_tensor_from_hfq(
                        hfq,
                        &format!("{prefix}.resample.{index}.bias"),
                    )?,
                }));
            }
        }
        Ok(None)
    }

    pub(crate) fn forward(&self, input: &CpuTensor) -> DiffusionResult<CpuTensor> {
        // 3x3 stride-2 conv, symmetric padding 1. NB: the exact WanVAE downsample
        // padding convention (this vs the asymmetric ZeroPad2d((0,1,0,1))) is not
        // yet pinned against the diffusers reference; symmetric gives the better
        // encode->decode round-trip so far but full fidelity is unverified.
        conv2d_nchw_with_stride(input, &self.resample_weight, Some(&self.resample_bias), 1, 2)
            .map_err(Into::into)
    }
}

/// One encoder down-step: either a residual block or a spatial downsampler
/// (the flat `down_blocks` list mixes them).
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) enum WanEncoderDownStep {
    Resnet(WanResnetBlock),
    Downsample(WanDownsample),
}

/// Full Wan / Qwen-Image (`AutoencoderKLQwenImage`) VAE encoder, still-image
/// (T=1) path — the mirror of `WanImageDecoder`: `conv_in -> down_blocks ->
/// mid(resnet, attn, resnet) -> norm_out -> SiLU -> conv_out -> quant_conv`,
/// mapping `[B, 3, H, W]` pixels to `[B, 2*z_dim, H/8, W/8]` diagonal-Gaussian
/// moments. Reuses the tested Wan building blocks.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct WanImageEncoder {
    conv_in_weight: CpuTensor,
    conv_in_bias: CpuTensor,
    down_steps: Vec<WanEncoderDownStep>,
    mid_resnet0: WanResnetBlock,
    mid_attention: WanMidAttention,
    mid_resnet1: WanResnetBlock,
    norm_out_gamma: Vec<f32>,
    conv_out_weight: CpuTensor,
    conv_out_bias: CpuTensor,
    quant_conv_weight: CpuTensor,
    quant_conv_bias: CpuTensor,
}

#[allow(dead_code)]
impl WanImageEncoder {
    const EPS: f32 = 1e-6;

    /// Load from an hfq under `prefix` (`vae/tensors/encoder`). Returns `None`
    /// when this is not a Wan / Qwen-Image encoder. Discriminated by
    /// `{prefix}.norm_out.gamma` (RMSNorm), like the decoder.
    pub(crate) fn from_hfq(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Option<Self>> {
        if hfq
            .find_tensor_info(&format!("{prefix}.norm_out.gamma"))
            .is_none()
        {
            return Ok(None);
        }
        // Parse the flat down_blocks list: each index is a resnet (conv1) or a
        // downsample (resample). Stop at the first missing index.
        let mut down_steps = Vec::new();
        let mut idx = 0;
        loop {
            let block_prefix = format!("{prefix}.down_blocks.{idx}");
            if hfq
                .find_tensor_info(&format!("{block_prefix}.conv1.weight"))
                .is_some()
            {
                down_steps.push(WanEncoderDownStep::Resnet(WanResnetBlock::from_hfq(
                    hfq,
                    &block_prefix,
                )?));
            } else if let Some(down) = WanDownsample::from_hfq(hfq, &block_prefix)? {
                down_steps.push(WanEncoderDownStep::Downsample(down));
            } else {
                break;
            }
            idx += 1;
        }
        Ok(Some(Self {
            conv_in_weight: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv_in.weight"))?,
            conv_in_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv_in.bias"))?,
            down_steps,
            mid_resnet0: WanResnetBlock::from_hfq(hfq, &format!("{prefix}.mid_block.resnets.0"))?,
            mid_attention: WanMidAttention::from_hfq(
                hfq,
                &format!("{prefix}.mid_block.attentions.0"),
            )?,
            mid_resnet1: WanResnetBlock::from_hfq(hfq, &format!("{prefix}.mid_block.resnets.1"))?,
            norm_out_gamma: cpu_tensor_from_hfq(hfq, &format!("{prefix}.norm_out.gamma"))?.data,
            conv_out_weight: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv_out.weight"))?,
            conv_out_bias: cpu_tensor_from_hfq(hfq, &format!("{prefix}.conv_out.bias"))?,
            quant_conv_weight: cpu_tensor_from_hfq(hfq, "vae/tensors/quant_conv.weight")?,
            quant_conv_bias: cpu_tensor_from_hfq(hfq, "vae/tensors/quant_conv.bias")?,
        }))
    }

    /// Encode `[B, 3, H, W]` pixels to `[B, 2*z_dim, H/8, W/8]` moments.
    pub(crate) fn encode(&self, image: &CpuTensor) -> DiffusionResult<CpuTensor> {
        let mut hidden = wan_causal_conv2d(image, &self.conv_in_weight, Some(&self.conv_in_bias))?;
        for step in &self.down_steps {
            hidden = match step {
                WanEncoderDownStep::Resnet(resnet) => resnet.forward(&hidden)?,
                WanEncoderDownStep::Downsample(down) => down.forward(&hidden)?,
            };
        }
        hidden = self.mid_resnet0.forward(&hidden)?;
        hidden = self.mid_attention.forward(&hidden)?;
        hidden = self.mid_resnet1.forward(&hidden)?;
        hidden = wan_silu(&wan_rms_norm_nchw(&hidden, &self.norm_out_gamma, Self::EPS)?);
        hidden = wan_causal_conv2d(&hidden, &self.conv_out_weight, Some(&self.conv_out_bias))?;
        // quant_conv is a 1x1x1 Conv3d (per-channel affine); wan_causal_conv2d
        // handles it directly.
        wan_causal_conv2d(&hidden, &self.quant_conv_weight, Some(&self.quant_conv_bias))
    }
}

#[cfg(test)]
mod wan_vae_tests {
    use super::*;

    #[test]
    fn wan_upsample_doubles_spatial_dims() {
        // 1x1 resample conv (identity-scaled) so the output is a clean 2x nearest
        // upsample of a single-channel input.
        let upsampler = WanUpsample {
            resample_weight: CpuTensor {
                shape: vec![1, 1, 1, 1],
                data: vec![1.0],
            },
            resample_bias: CpuTensor {
                shape: vec![1],
                data: vec![0.0],
            },
        };
        let input = CpuTensor {
            shape: vec![1, 1, 1, 2],
            data: vec![5.0, 9.0],
        };
        let out = upsampler.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 4]);
        // nearest 2x of [[5, 9]] -> [[5,5,9,9],[5,5,9,9]].
        assert_eq!(out.data, vec![5.0, 5.0, 9.0, 9.0, 5.0, 5.0, 9.0, 9.0]);
    }

    #[test]
    fn wan_causal_conv2d_uses_last_temporal_tap() {
        // 1x1 spatial, 3 temporal taps [5, 7, 11]. QwenImageCausalConv3d pads the
        // temporal axis causally with ZEROS, so a single (T=1) frame sees
        // [0, 0, x] and only the LAST tap (11) survives: 11 * input.
        let weight3d = CpuTensor {
            shape: vec![1, 1, 3, 1, 1],
            data: vec![5.0, 7.0, 11.0],
        };
        let input = CpuTensor {
            shape: vec![1, 1, 1, 2],
            data: vec![2.0, 3.0],
        };
        let out = wan_causal_conv2d(&input, &weight3d, None).unwrap();
        assert_eq!(out.shape, vec![1, 1, 1, 2]);
        assert_eq!(out.data, vec![22.0, 33.0]);
    }

    #[test]
    fn wan_causal_conv2d_3x3_same_size_with_bias() {
        // 3x3x3 kernel: last temporal tap is an identity-center 3x3, +bias.
        let mut w = vec![0.0f32; 27];
        w[2 * 9 + 4] = 1.0; // tap=2, center of the 3x3
        let weight3d = CpuTensor {
            shape: vec![1, 1, 3, 3, 3],
            data: w,
        };
        let input = CpuTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let bias = CpuTensor {
            shape: vec![1],
            data: vec![10.0],
        };
        let out = wan_causal_conv2d(&input, &weight3d, Some(&bias)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert_eq!(out.data, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn wan_rms_norm_normalizes_across_channels_and_scales_by_gamma() {
        // NCHW [1,2,1,2]: channel 0 occupies data[0..2], channel 1 data[2..4].
        // So spatial position 0 has channels [data[0], data[2]] = [3, 4], and
        // position 1 has [data[1], data[3]] = [6, 8].
        let input = CpuTensor {
            shape: vec![1, 2, 1, 2],
            data: vec![3.0, 6.0, 4.0, 8.0],
        };
        let gamma = [2.0f32, 0.5];
        let out = wan_rms_norm_nchw(&input, &gamma, 0.0).unwrap();
        // pos0 channels [3,4], rms = sqrt((9+16)/2) = sqrt(12.5)
        let rms0 = (12.5f32).sqrt();
        assert!((out.data[0] - 3.0 / rms0 * 2.0).abs() < 1e-5);
        assert!((out.data[2] - 4.0 / rms0 * 0.5).abs() < 1e-5);
        // pos1 channels [6,8], rms = sqrt((36+64)/2) = sqrt(50)
        let rms1 = (50.0f32).sqrt();
        assert!((out.data[1] - 6.0 / rms1 * 2.0).abs() < 1e-5);
        assert!((out.data[3] - 8.0 / rms1 * 0.5).abs() < 1e-5);
    }
}
