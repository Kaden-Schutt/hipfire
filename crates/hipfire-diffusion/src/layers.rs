// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Reusable neural-network building blocks shared across the VAE, UNet, and
//! transformer denoiser: conv/groupnorm/resnet, time embeddings, attention,
//! GeGLU feed-forward, and the basic transformer block.

use super::*;

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

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b/3 device-resident conv: consumes a resident input, returns a
    /// resident output. Does not free `input` (the caller owns it). Uses the
    /// Phase 3 im2col + WMMA-GEMM path (with an internal direct-conv fallback on
    /// architectures without wave32 WMMA).
    pub(crate) fn forward_resident(
        &self,
        input: &rdna_compute::GpuTensor,
        gpu: &mut rdna_compute::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<rdna_compute::GpuTensor> {
        conv2d_nchw_wmma_resident(
            gpu,
            cache,
            input,
            &self.weight,
            self.bias.as_ref(),
            self.padding,
            self.stride,
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

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident group-norm. Does not free `input`.
    pub(crate) fn forward_resident(
        &self,
        input: &rdna_compute::GpuTensor,
        gpu: &mut rdna_compute::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<rdna_compute::GpuTensor> {
        group_norm_nchw_resident(
            gpu,
            cache,
            input,
            &self.weight,
            &self.bias,
            self.groups,
            self.eps,
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

    /// Phase 1b device-resident resnet block. Consumes a resident `input`
    /// (borrowed; the caller owns it) and returns a resident output, freeing
    /// every internal intermediate back to the pool.
    pub(crate) fn forward_resident(
        &self,
        input: &rdna_compute::GpuTensor,
        gpu: &mut rdna_compute::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<rdna_compute::GpuTensor> {
        let hidden = self.norm1.forward_resident(input, gpu, cache)?;
        let silued = silu_resident(gpu, &hidden)?;
        free_resident(gpu, hidden)?;
        let conv1 = self.conv1.forward_resident(&silued, gpu, cache)?;
        free_resident(gpu, silued)?;
        let hidden = self.norm2.forward_resident(&conv1, gpu, cache)?;
        free_resident(gpu, conv1)?;
        let silued = silu_resident(gpu, &hidden)?;
        free_resident(gpu, hidden)?;
        let conv2 = self.conv2.forward_resident(&silued, gpu, cache)?;
        free_resident(gpu, silued)?;
        let out = match &self.shortcut {
            Some(shortcut) => {
                let residual = shortcut.forward_resident(input, gpu, cache)?;
                let sum = tensor_add_resident(gpu, &conv2, &residual)?;
                free_resident(gpu, residual)?;
                sum
            }
            None => tensor_add_resident(gpu, &conv2, input)?,
        };
        free_resident(gpu, conv2)?;
        Ok(out)
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

    pub(crate) fn forward_with_runtime_options(
        &self,
        input: &CpuTensor,
        time_embedding: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(input, time_embedding, &mut runtime_context)
    }

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident UNet resnet block. `input` and `time_embedding`
    /// are resident (borrowed; the caller owns them). `time_embedding` is the
    /// raw `[batch, time_dim]` embedding — SiLU + the time projection happen
    /// here, mirroring the host path.
    pub(crate) fn forward_resident(
        &self,
        input: &rdna_compute::GpuTensor,
        time_embedding: &rdna_compute::GpuTensor,
        gpu: &mut rdna_compute::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<rdna_compute::GpuTensor> {
        let normed = self.norm1.forward_resident(input, gpu, cache)?;
        let silued = silu_resident(gpu, &normed)?;
        free_resident(gpu, normed)?;
        let hidden = self.conv1.forward_resident(&silued, gpu, cache)?;
        free_resident(gpu, silued)?;
        let time_silu = silu_resident(gpu, time_embedding)?;
        let projected_time = linear_optional_bias_resident(
            gpu,
            cache,
            &time_silu,
            &self.time_emb_proj_weight,
            Some(&self.time_emb_proj_bias),
        )?;
        free_resident(gpu, time_silu)?;
        let hidden_biased = add_channel_bias_nchw_resident(gpu, &hidden, &projected_time)?;
        free_resident(gpu, projected_time)?;
        free_resident(gpu, hidden)?;
        let normed = self.norm2.forward_resident(&hidden_biased, gpu, cache)?;
        free_resident(gpu, hidden_biased)?;
        let silued = silu_resident(gpu, &normed)?;
        free_resident(gpu, normed)?;
        let conv2 = self.conv2.forward_resident(&silued, gpu, cache)?;
        free_resident(gpu, silued)?;
        let out = match &self.shortcut {
            Some(shortcut) => {
                let residual = shortcut.forward_resident(input, gpu, cache)?;
                let sum = tensor_add_resident(gpu, &conv2, &residual)?;
                free_resident(gpu, residual)?;
                sum
            }
            None => tensor_add_resident(gpu, &conv2, input)?,
        };
        free_resident(gpu, conv2)?;
        Ok(out)
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

    pub(crate) fn forward_with_runtime_options(
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

    pub(crate) fn forward_with_runtime_context(
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

    pub(crate) fn forward_with_runtime_options(
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

    pub(crate) fn forward_with_runtime_context(
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

    pub(crate) fn forward_with_runtime_options(
        &self,
        hidden_states: &CpuTensor,
        encoder_states: Option<&CpuTensor>,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(hidden_states, encoder_states, &mut runtime_context)
    }

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident attention. `hidden`/`encoder` are resident 3D
    /// `[b, seq, hidden]` tensors (borrowed; caller owns them). Self-attention
    /// when `encoder` is `None`.
    pub(crate) fn forward_resident(
        &self,
        hidden_states: &rdna_compute::GpuTensor,
        encoder_states: Option<&rdna_compute::GpuTensor>,
        gpu: &mut rdna_compute::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<rdna_compute::GpuTensor> {
        let context = encoder_states.unwrap_or(hidden_states);
        let q = linear_optional_bias_resident(
            gpu,
            cache,
            hidden_states,
            &self.to_q_weight,
            self.to_q_bias.as_ref(),
        )?;
        let k = linear_optional_bias_resident(
            gpu,
            cache,
            context,
            &self.to_k_weight,
            self.to_k_bias.as_ref(),
        )?;
        let v = linear_optional_bias_resident(
            gpu,
            cache,
            context,
            &self.to_v_weight,
            self.to_v_bias.as_ref(),
        )?;
        let attended = scaled_dot_product_attention_resident(gpu, &q, &k, &v, self.heads)?;
        free_resident(gpu, q)?;
        free_resident(gpu, k)?;
        free_resident(gpu, v)?;
        let out = linear_optional_bias_resident(
            gpu,
            cache,
            &attended,
            &self.to_out_weight,
            self.to_out_bias.as_ref(),
        )?;
        free_resident(gpu, attended)?;
        Ok(out)
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

    pub(crate) fn forward_with_runtime_options(
        &self,
        hidden_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(hidden_states, &mut runtime_context)
    }

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident GeGLU feed-forward over a resident 3D
    /// `[b, seq, in]` input (borrowed; caller owns it).
    pub(crate) fn forward_resident(
        &self,
        hidden_states: &rdna_compute::GpuTensor,
        gpu: &mut rdna_compute::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<rdna_compute::GpuTensor> {
        let projected = linear_optional_bias_resident(
            gpu,
            cache,
            hidden_states,
            &self.proj_weight,
            Some(&self.proj_bias),
        )?;
        let gated = geglu_gate_3d_resident(gpu, &projected)?;
        free_resident(gpu, projected)?;
        let out =
            linear_optional_bias_resident(gpu, cache, &gated, &self.out_weight, Some(&self.out_bias))?;
        free_resident(gpu, gated)?;
        Ok(out)
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

    pub(crate) fn forward_with_runtime_options(
        &self,
        hidden_states: &CpuTensor,
        encoder_states: &CpuTensor,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<CpuTensor> {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        self.forward_with_runtime_context(hidden_states, encoder_states, &mut runtime_context)
    }

    pub(crate) fn forward_with_runtime_context(
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

    /// Phase 1b device-resident basic transformer block (self-attn → cross-attn
    /// → GeGLU FF, each with a layer-norm + residual). `hidden_states` and
    /// `encoder_states` are resident (borrowed; the caller owns them).
    pub(crate) fn forward_resident(
        &self,
        hidden_states: &rdna_compute::GpuTensor,
        encoder_states: &rdna_compute::GpuTensor,
        gpu: &mut rdna_compute::Gpu,
        cache: &mut RocmWeightCache,
    ) -> DiffusionResult<rdna_compute::GpuTensor> {
        let normed =
            layer_norm_resident(gpu, cache, hidden_states, &self.norm1_weight, &self.norm1_bias, 1e-5)?;
        let attn = self.attn1.forward_resident(&normed, None, gpu, cache)?;
        free_resident(gpu, normed)?;
        let mut hidden = tensor_add_resident(gpu, hidden_states, &attn)?;
        free_resident(gpu, attn)?;

        let normed =
            layer_norm_resident(gpu, cache, &hidden, &self.norm2_weight, &self.norm2_bias, 1e-5)?;
        let attn = self
            .attn2
            .forward_resident(&normed, Some(encoder_states), gpu, cache)?;
        free_resident(gpu, normed)?;
        let next = tensor_add_resident(gpu, &hidden, &attn)?;
        free_resident(gpu, attn)?;
        free_resident(gpu, hidden)?;
        hidden = next;

        let normed =
            layer_norm_resident(gpu, cache, &hidden, &self.norm3_weight, &self.norm3_bias, 1e-5)?;
        let ff = self.feed_forward.forward_resident(&normed, gpu, cache)?;
        free_resident(gpu, normed)?;
        let next = tensor_add_resident(gpu, &hidden, &ff)?;
        free_resident(gpu, ff)?;
        free_resident(gpu, hidden)?;
        Ok(next)
    }
}

