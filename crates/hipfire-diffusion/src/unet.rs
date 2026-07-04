// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Native UNet2DConditionModel assembly: the spatial Transformer2DModel and the
//! down/up/mid block paths that make up the SD/SDXL denoiser.

use super::*;

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

    pub(crate) fn forward_with_runtime_options(
        &self,
        input: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(input, encoder_states, &mut runtime_context)
    }

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident spatial transformer. `input` and `encoder` are
    /// resident (borrowed; the caller owns them).
    pub(crate) fn forward_resident(
        &self,
        input: &hipfire_rdna::GpuTensor,
        encoder_states: &hipfire_rdna::GpuTensor,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        let normed = self.norm.forward_resident(input, gpu, cache)?;
        let projected = self.proj_in.forward_resident(&normed, gpu, cache)?;
        free_resident(gpu, normed)?;
        let (batch, channels, height, width) = match projected.shape.as_slice() {
            [b, c, h, w] => (*b, *c, *h, *w),
            other => {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "Transformer2D expected a 4D NCHW tensor, got shape {other:?}"
                )))
            }
        };
        let bsc = nchw_to_bsc_resident(gpu, &projected)?;
        free_resident(gpu, projected)?;
        let blocked = self
            .block
            .forward_resident(&bsc, encoder_states, gpu, cache)?;
        free_resident(gpu, bsc)?;
        let nchw = bsc_to_nchw_resident(gpu, &blocked, batch, channels, height, width)?;
        free_resident(gpu, blocked)?;
        let proj_out = self.proj_out.forward_resident(&nchw, gpu, cache)?;
        free_resident(gpu, nchw)?;
        let out = tensor_add_resident(gpu, &proj_out, input)?;
        free_resident(gpu, proj_out)?;
        Ok(out)
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

    pub(crate) fn forward_with_runtime_options(
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

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident down block. Takes ownership of `hidden`, pushes a
    /// resident snapshot per layer (and after the downsampler) onto `skips`, and
    /// returns the resident output. The caller owns and must free every skip.
    pub(crate) fn forward_resident(
        &self,
        mut hidden: hipfire_rdna::GpuTensor,
        time_embedding: &hipfire_rdna::GpuTensor,
        encoder_states: &hipfire_rdna::GpuTensor,
        skips: &mut Vec<hipfire_rdna::GpuTensor>,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        for (idx, resnet) in self.resnets.iter().enumerate() {
            let next = resnet.forward_resident(&hidden, time_embedding, gpu, cache)?;
            free_resident(gpu, hidden)?;
            hidden = next;
            if let Some(attention) = self.attentions.get(idx) {
                let next = attention.forward_resident(&hidden, encoder_states, gpu, cache)?;
                free_resident(gpu, hidden)?;
                hidden = next;
            }
            skips.push(clone_resident(gpu, &hidden)?);
        }
        if let Some(downsampler) = &self.downsampler {
            let next = downsampler.forward_resident(&hidden, gpu, cache)?;
            free_resident(gpu, hidden)?;
            hidden = next;
            skips.push(clone_resident(gpu, &hidden)?);
        }
        Ok(hidden)
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
            // diffusers semantics: the `attention_head_dim` config value is the
            // NUMBER of attention heads (num_attention_heads), not the per-head
            // dimension. SD1.5 ships attention_head_dim=8 meaning 8 heads, so the
            // per-head dim is channels/8 (40/80/160 here). The downstream SDPA
            // derives head_dim = hidden/heads, so we pass the head COUNT.
            let heads = config
                .attention_head_dim
                .get(block_idx)
                .copied()
                .or_else(|| config.attention_head_dim.first().copied())
                .filter(|&h| h > 0)
                .unwrap_or(8)
                .min(channels.max(1));
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

    pub(crate) fn forward_with_runtime_options(
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

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident down path. Uploads `conv_in`'s output as the
    /// first skip, then threads the resident hidden through every down block,
    /// accumulating resident skip snapshots. The caller owns and must free
    /// every returned skip.
    pub(crate) fn forward_resident(
        &self,
        sample: &hipfire_rdna::GpuTensor,
        time_embedding: &hipfire_rdna::GpuTensor,
        encoder_states: &hipfire_rdna::GpuTensor,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<(hipfire_rdna::GpuTensor, Vec<hipfire_rdna::GpuTensor>)> {
        let mut hidden = self.conv_in.forward_resident(sample, gpu, cache)?;
        let mut skips = vec![clone_resident(gpu, &hidden)?];
        for block in &self.blocks {
            hidden = block.forward_resident(
                hidden,
                time_embedding,
                encoder_states,
                &mut skips,
                gpu,
                cache,
            )?;
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

    pub(crate) fn forward_with_runtime_options(
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

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident up block. Takes ownership of `hidden`, pops and
    /// frees a resident skip per layer (concatenating it on the channel axis),
    /// and returns the resident output.
    pub(crate) fn forward_resident(
        &self,
        mut hidden: hipfire_rdna::GpuTensor,
        skips: &mut Vec<hipfire_rdna::GpuTensor>,
        time_embedding: &hipfire_rdna::GpuTensor,
        encoder_states: &hipfire_rdna::GpuTensor,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        for (idx, resnet) in self.resnets.iter().enumerate() {
            let skip = skips.pop().ok_or_else(|| {
                DiffusionError::InvalidMetadata("UNet up block ran out of skip tensors".to_string())
            })?;
            let concatenated = concat_channels_nchw_resident(gpu, &hidden, &skip)?;
            free_resident(gpu, hidden)?;
            free_resident(gpu, skip)?;
            hidden = concatenated;
            let next = resnet.forward_resident(&hidden, time_embedding, gpu, cache)?;
            free_resident(gpu, hidden)?;
            hidden = next;
            if let Some(attention) = self.attentions.get(idx) {
                let next = attention.forward_resident(&hidden, encoder_states, gpu, cache)?;
                free_resident(gpu, hidden)?;
                hidden = next;
            }
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
            // `attention_head_dim` config value is the head COUNT (see down path).
            let heads = config
                .attention_head_dim
                .iter()
                .rev()
                .nth(block_idx)
                .copied()
                .or_else(|| config.attention_head_dim.first().copied())
                .filter(|&h| h > 0)
                .unwrap_or(8)
                .min(channels.max(1));
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

    pub(crate) fn forward_with_runtime_options(
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

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident up path. Threads the resident hidden through
    /// every up block, consuming the resident skips accumulated by the down path.
    pub(crate) fn forward_resident(
        &self,
        mut hidden: hipfire_rdna::GpuTensor,
        skips: &mut Vec<hipfire_rdna::GpuTensor>,
        time_embedding: &hipfire_rdna::GpuTensor,
        encoder_states: &hipfire_rdna::GpuTensor,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        for block in &self.blocks {
            hidden = block.forward_resident(
                hidden,
                skips,
                time_embedding,
                encoder_states,
                gpu,
                cache,
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
        // `attention_head_dim` config value is the head COUNT (see down path).
        let heads = config
            .attention_head_dim
            .last()
            .copied()
            .or_else(|| config.attention_head_dim.first().copied())
            .filter(|&h| h > 0)
            .unwrap_or(8)
            .min(channels.max(1));
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

    pub(crate) fn forward_with_runtime_options(
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

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident mid block (resnet → optional cross-attn →
    /// optional resnet). Takes ownership of `hidden`.
    pub(crate) fn forward_resident(
        &self,
        mut hidden: hipfire_rdna::GpuTensor,
        time_embedding: &hipfire_rdna::GpuTensor,
        encoder_states: &hipfire_rdna::GpuTensor,
        gpu: &mut hipfire_rdna::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<hipfire_rdna::GpuTensor> {
        let next = self
            .resnet_0
            .forward_resident(&hidden, time_embedding, gpu, cache)?;
        free_resident(gpu, hidden)?;
        hidden = next;
        if let Some(attention) = &self.attention {
            let next = attention.forward_resident(&hidden, encoder_states, gpu, cache)?;
            free_resident(gpu, hidden)?;
            hidden = next;
        }
        if let Some(resnet) = &self.resnet_1 {
            let next = resnet.forward_resident(&hidden, time_embedding, gpu, cache)?;
            free_resident(gpu, hidden)?;
            hidden = next;
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

    pub(crate) fn forward_with_sdxl_conditioning_and_runtime_options(
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

    pub(crate) fn forward_with_sdxl_conditioning_and_runtime_context(
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
        // Phase 1b: when a GPU is present, run the whole denoiser device-resident
        // — upload the (host-computed) sample, time embedding, and conditioning
        // once, keep every activation on-device through the down/mid/up paths,
        // and download only the final noise prediction.
        if runtime_context.rocm_device_id().is_some() {
            return self.forward_resident(
                &sample,
                &time_embedding,
                encoder_states,
                runtime_context,
            );
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

    /// Phase 1b device-resident UNet forward. `sample_host` and
    /// `time_embedding_host` are computed on the host by the caller (centering +
    /// the small time/SDXL embedding, once per forward); everything from
    /// `conv_in` to `conv_out` runs on-device with no intermediate host
    /// round-trips. Only the final noise prediction is downloaded.
    fn forward_resident(
        &self,
        sample_host: &CpuTensor,
        time_embedding_host: &CpuTensor,
        encoder_states_host: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        runtime_context.with_rocm_gpu_weighted(|gpu, cache| {
            gpu.bind_thread()
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            let sample = gpu
                .upload_f32(&sample_host.data, &sample_host.shape)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            let time_embedding = gpu
                .upload_f32(&time_embedding_host.data, &time_embedding_host.shape)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            let encoder_states = gpu
                .upload_f32(&encoder_states_host.data, &encoder_states_host.shape)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            let (mut hidden, mut skips) = self.down_path.forward_resident(
                &sample,
                &time_embedding,
                &encoder_states,
                gpu,
                cache,
            )?;
            free_resident(gpu, sample)?;
            if let Some(mid_block) = &self.mid_block {
                hidden = mid_block.forward_resident(
                    hidden,
                    &time_embedding,
                    &encoder_states,
                    gpu,
                    cache,
                )?;
            }
            hidden = self.up_path.forward_resident(
                hidden,
                &mut skips,
                &time_embedding,
                &encoder_states,
                gpu,
                cache,
            )?;
            // Any skip not consumed by the up path (unbalanced configs) must
            // still be returned to the pool.
            for skip in skips.drain(..) {
                free_resident(gpu, skip)?;
            }
            let normed = self.conv_norm_out.forward_resident(&hidden, gpu, cache)?;
            free_resident(gpu, hidden)?;
            let silued = silu_resident(gpu, &normed)?;
            free_resident(gpu, normed)?;
            let predicted = self.conv_out.forward_resident(&silued, gpu, cache)?;
            free_resident(gpu, silued)?;
            let output = download_resident(gpu, &predicted)?;
            free_resident(gpu, predicted)?;
            free_resident(gpu, time_embedding)?;
            free_resident(gpu, encoder_states)?;
            Ok(output)
        })
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

    pub(crate) fn denoise_latents_with_runtime_options(
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

    pub(crate) fn denoise_latents_with_runtime_context(
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

pub(crate) fn maybe_center_unet_input(sample: &CpuTensor, center_input_sample: bool) -> CpuTensor {
    if center_input_sample {
        CpuTensor {
            shape: sample.shape.clone(),
            data: sample.data.iter().map(|value| value * 2.0 - 1.0).collect(),
        }
    } else {
        sample.clone()
    }
}
