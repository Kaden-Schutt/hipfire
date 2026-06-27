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
        input: &rdna_compute::GpuTensor,
        gpu: &mut rdna_compute::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<rdna_compute::GpuTensor> {
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
        mut hidden: rdna_compute::GpuTensor,
        gpu: &mut rdna_compute::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<rdna_compute::GpuTensor> {
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
            let next = self.conv_in.forward_resident(&hidden, gpu, cache)?;
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
            let next = self.conv_norm_out.forward_resident(&hidden, gpu, cache)?;
            free_resident(gpu, hidden)?;
            hidden = next;
            let next = silu_resident(gpu, &hidden)?;
            free_resident(gpu, hidden)?;
            hidden = next;
            let next = self.conv_out.forward_resident(&hidden, gpu, cache)?;
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
