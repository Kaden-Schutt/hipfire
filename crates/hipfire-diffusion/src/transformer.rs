// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Native transformer diffusion denoiser (Qwen-Image / Flux / Krea family):
//! IO projection, timestep + modulation embeddings, attention/feed-forward
//! blocks, RoPE, and the 3D norm/residual ops the blocks use.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct NativeTransformerDenoiserIo {
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
    pub(crate) fn from_hfq(
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

    pub(crate) fn project_latents_to_hidden_with_runtime_context(
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

    pub(crate) fn project_hidden_to_latents_with_runtime_context(
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

    pub(crate) fn project_text_to_hidden_with_runtime_context(
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

    pub(crate) fn output_norm_with_runtime_context(
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
pub(crate) struct NativeTransformerTimestepEmbedding {
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
    pub(crate) fn from_hfq(hfq: &HfqFile, family: TransformerDenoiserFamily) -> DiffusionResult<Self> {
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

    pub(crate) fn embedding_dim(&self) -> DiffusionResult<usize> {
        let (_, embedding_dim) = self.linear_1_weight.rows_cols()?;
        Ok(embedding_dim)
    }

    pub(crate) fn forward_with_runtime_context(
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

    pub(crate) fn modulation_with_runtime_context(
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
pub(crate) struct TransformerModulationChunks {
    pub(crate) shift_msa: CpuTensor,
    pub(crate) scale_msa: CpuTensor,
    pub(crate) gate_msa: CpuTensor,
    pub(crate) shift_mlp: CpuTensor,
    pub(crate) scale_mlp: CpuTensor,
    pub(crate) gate_mlp: CpuTensor,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct NativeTransformerBlockModulation {
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
    pub(crate) fn from_hfq(
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

    pub(crate) fn qwen_image_modulation_with_runtime_context(
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

    pub(crate) fn krea_scale_shift_with_runtime_context(
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
pub(crate) struct TransformerAttentionQkv {
    pub(crate) q: CpuTensor,
    pub(crate) k: CpuTensor,
    pub(crate) v: CpuTensor,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct TransformerAttentionStreamProjection {
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
    pub(crate) fn from_hfq(
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

    pub(crate) fn validate_shapes(
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

    pub(crate) fn project_qkv_with_runtime_context(
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

    pub(crate) fn project_output_with_runtime_context(
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
pub(crate) struct NativeTransformerAttentionProjection {
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
pub(crate) struct RotaryFrequencies {
    cos: CpuTensor,
    sin: CpuTensor,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct QwenRotaryEmbeddings {
    pub(crate) image: RotaryFrequencies,
    pub(crate) text: RotaryFrequencies,
}

#[allow(dead_code)]
impl NativeTransformerAttentionProjection {
    pub(crate) fn from_hfq(
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

    pub(crate) fn project_image_qkv_with_runtime_context(
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

    pub(crate) fn project_text_qkv_with_runtime_context(
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

    pub(crate) fn project_image_output_with_runtime_context(
        &self,
        attention: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        self.validate_attention_input(attention, TransformerModulationStream::Image)?;
        self.image
            .project_output_with_runtime_context(attention, runtime_context)
    }

    pub(crate) fn project_text_output_with_runtime_context(
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

    pub(crate) fn attend_image_text_with_runtime_context(
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

    pub(crate) fn validate_hidden_input(
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

    pub(crate) fn validate_attention_input(
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
pub(crate) enum TransformerFeedForwardActivation {
    GeGlu,
    SwiGlu,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct TransformerFeedForwardStream {
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
    pub(crate) fn qwen_geglu_from_hfq(
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

    pub(crate) fn krea_swiglu_from_hfq(
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

    pub(crate) fn forward_with_runtime_context(
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
pub(crate) struct NativeTransformerFeedForward {
    family: TransformerDenoiserFamily,
    block_index: usize,
    hidden_width: usize,
    image: TransformerFeedForwardStream,
    text: Option<TransformerFeedForwardStream>,
}

#[allow(dead_code)]
impl NativeTransformerFeedForward {
    pub(crate) fn from_hfq(
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

    pub(crate) fn forward_image_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        self.image
            .forward_with_runtime_context(hidden, runtime_context)
    }

    pub(crate) fn forward_text_with_runtime_context(
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
pub(crate) struct NativeTransformerBlock {
    family: TransformerDenoiserFamily,
    block_index: usize,
    modulation: NativeTransformerBlockModulation,
    attention: NativeTransformerAttentionProjection,
    feed_forward: NativeTransformerFeedForward,
}

#[allow(dead_code)]
impl NativeTransformerBlock {
    pub(crate) fn from_hfq(
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

    pub(crate) fn forward_qwen_with_runtime_context(
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
pub(crate) struct NativeTransformerDenoiser {
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
    pub(crate) fn from_hfq(
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

    pub(crate) fn forward_qwen_with_runtime_context(
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

    pub(crate) fn qwen_rotary_embeddings(
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
pub(crate) enum TransformerModulationStream {
    Image,
    Text,
}

#[allow(dead_code)]
pub(crate) fn attention_norm_weight_dim(weight: &CpuTensor) -> DiffusionResult<usize> {
    match weight.shape.as_slice() {
        [dim] if *dim > 0 => Ok(*dim),
        _ => Err(DiffusionError::InvalidMetadata(format!(
            "transformer attention norm weight shape {:?} is not [head_dim]",
            weight.shape
        ))),
    }
}

#[allow(dead_code)]
pub(crate) fn qwen_rope_axes_from_transformer_config(
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
pub(crate) fn qwen_rotary_embeddings_for_grid(
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

pub(crate) fn write_qwen_rope_token(
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
pub(crate) fn apply_qwen_rotary_embedding(
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
pub(crate) fn validate_attention_linear_shape(
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
pub(crate) fn validate_attention_bias_shape(
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
pub(crate) fn validate_attention_norm_shape(
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
pub(crate) fn maybe_rms_norm_attention_heads_3d(
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
pub(crate) fn rms_norm_attention_heads_3d(
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
pub(crate) fn rms_norm_3d_with_runtime_context(
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
pub(crate) fn validate_transformer_ff_down_shape(
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
pub(crate) fn swiglu_gate_3d(up: &CpuTensor, gate: &CpuTensor) -> DiffusionResult<CpuTensor> {
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
pub(crate) fn layer_norm_3d_no_affine_with_runtime_context(
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
pub(crate) fn modulate_3d(
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
pub(crate) fn gated_residual_3d(
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
pub(crate) fn concat_sequence_3d(left: &CpuTensor, right: &CpuTensor) -> DiffusionResult<CpuTensor> {
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
pub(crate) fn split_modulation_chunks(
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
