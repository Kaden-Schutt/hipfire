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
    // Krea2 final adaLN: weighted RMSNorm + a `[2, hidden]` scale/shift table
    // combined with the timestep embedding. QwenImage uses `norm_out.linear`
    // (above) instead, so these are None there.
    krea_final_norm_weight: Option<CpuTensor>,
    krea_final_scale_shift: Option<CpuTensor>,
    // Krea2 text-input projection: RMSNorm(`txt_in.norm`) -> `txt_in.linear_1`
    // (text_hidden -> hidden) -> `txt_in.linear_2` (hidden -> hidden). QwenImage
    // uses the single `txt_in.weight` above instead.
    krea_txt_norm_weight: Option<CpuTensor>,
    krea_txt_linear1_weight: Option<CpuTensor>,
    krea_txt_linear1_bias: Option<CpuTensor>,
    krea_txt_linear2_weight: Option<CpuTensor>,
    krea_txt_linear2_bias: Option<CpuTensor>,
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

        // Krea2 final adaLN: RMSNorm weight + a [2, hidden] scale/shift table.
        let krea_final_norm_weight =
            optional_tensor(hfq, "transformer/tensors/final_layer.norm.weight")?;
        let krea_final_scale_shift =
            optional_tensor(hfq, "transformer/tensors/final_layer.scale_shift_table")?;
        if let Some(table) = krea_final_scale_shift.as_ref() {
            let [chunks, table_hidden] = shape2(table)?;
            if chunks != 2 || table_hidden != hidden_width {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "Krea final_layer scale_shift_table shape {:?} != [2, {hidden_width}]",
                    table.shape
                )));
            }
        }
        if let Some(weight) = krea_final_norm_weight.as_ref() {
            if weight.shape.as_slice() != [hidden_width] {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "Krea final_layer norm weight shape {:?} != [{hidden_width}]",
                    weight.shape
                )));
            }
        }

        // Krea2 two-layer text-input projection (`txt_in.norm`/`linear_1`/`linear_2`).
        let krea_txt_norm_weight = optional_tensor(hfq, "transformer/tensors/txt_in.norm.weight")?;
        let krea_txt_linear1_weight =
            optional_tensor(hfq, "transformer/tensors/txt_in.linear_1.weight")?;
        let (krea_txt_linear1_bias, krea_txt_linear2_weight, krea_txt_linear2_bias) =
            if krea_txt_linear1_weight.is_some() {
                (
                    optional_tensor(hfq, "transformer/tensors/txt_in.linear_1.bias")?,
                    optional_tensor(hfq, "transformer/tensors/txt_in.linear_2.weight")?,
                    optional_tensor(hfq, "transformer/tensors/txt_in.linear_2.bias")?,
                )
            } else {
                (None, None, None)
            };

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
            krea_final_norm_weight,
            krea_final_scale_shift,
            krea_txt_norm_weight,
            krea_txt_linear1_weight,
            krea_txt_linear1_bias,
            krea_txt_linear2_weight,
            krea_txt_linear2_bias,
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
        // Krea2: RMSNorm(txt_in.norm) -> linear_1 -> linear_2 (text_hidden -> hidden).
        if let Some(linear1) = self.krea_txt_linear1_weight.as_ref() {
            let normed = match self.krea_txt_norm_weight.as_ref() {
                Some(norm) => {
                    rms_norm_3d_with_runtime_context(text_hidden, norm, 1e-6, runtime_context)?
                }
                None => text_hidden.clone(),
            };
            let hidden = linear_3d_with_runtime_context(
                &normed,
                linear1,
                self.krea_txt_linear1_bias.as_ref(),
                runtime_context,
            )?;
            let linear2 = self.krea_txt_linear2_weight.as_ref().ok_or_else(|| {
                DiffusionError::InvalidMetadata("Krea txt_in.linear_2 weight is missing".into())
            })?;
            return linear_3d_with_runtime_context(
                &hidden,
                linear2,
                self.krea_txt_linear2_bias.as_ref(),
                runtime_context,
            );
        }
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
        // Krea2 final adaLN: RMSNorm(hidden) then modulate by the [2, hidden]
        // scale/shift table combined with the timestep embedding. NOTE: the
        // (scale, shift) chunk order and the time-embedding modulation source
        // follow the per-block adaLN convention; verify against the Krea2
        // reference during numerical parity before trusting decoded pixels.
        if matches!(self.family, TransformerDenoiserFamily::Krea2) {
            let Some(norm_weight) = self.krea_final_norm_weight.as_ref() else {
                return Ok(hidden.clone());
            };
            let table = self.krea_final_scale_shift.as_ref().ok_or_else(|| {
                DiffusionError::InvalidMetadata(
                    "Krea final_layer norm weight present but scale_shift_table is missing"
                        .to_string(),
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
            if time_batch != batch || time_width != width {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "Krea final adaLN timestep shape {:?} != [{batch}, {width}]",
                    timestep_embedding.shape
                )));
            }
            let normalized =
                rms_norm_3d_with_runtime_context(hidden, norm_weight, 1e-5, runtime_context)?;
            let mut scale = CpuTensor::zeros(&[batch, width]);
            let mut shift = CpuTensor::zeros(&[batch, width]);
            for b in 0..batch {
                for col in 0..width {
                    let temb = timestep_embedding.data[b * width + col];
                    scale.data[b * width + col] = table.data[col] + temb;
                    shift.data[b * width + col] = table.data[width + col] + temb;
                }
            }
            return modulate_3d(&normalized, &shift, &scale);
        }
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
    pub(crate) fn from_hfq(
        hfq: &HfqFile,
        family: TransformerDenoiserFamily,
    ) -> DiffusionResult<Self> {
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

/// Load a `ResidentWeight` only if the tensor is present.
pub(crate) fn optional_resident(
    hfq: &HfqFile,
    name: &str,
) -> DiffusionResult<Option<ResidentWeight>> {
    if hfq.find_tensor_info(name).is_some() {
        Ok(Some(ResidentWeight::from_hfq(hfq, name)?))
    } else {
        Ok(None)
    }
}

/// Parse a `[rows, cols]` shape slice (for `ResidentWeight`, which exposes its
/// shape without decoding).
pub(crate) fn shape2_slice(shape: &[usize]) -> DiffusionResult<[usize; 2]> {
    match shape {
        [a, b] => Ok([*a, *b]),
        _ => Err(DiffusionError::InvalidMetadata(format!(
            "expected 2-D tensor, got {shape:?}"
        ))),
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct TransformerAttentionStreamProjection {
    stream_label: &'static str,
    // Large GEMM weights kept resident in packed form (bf16/oq4), decoded to f32
    // transiently per forward; biases/norms are tiny and stay f32.
    q_weight: ResidentWeight,
    q_bias: Option<CpuTensor>,
    k_weight: ResidentWeight,
    k_bias: Option<CpuTensor>,
    v_weight: ResidentWeight,
    v_bias: Option<CpuTensor>,
    norm_q_weight: Option<CpuTensor>,
    norm_k_weight: Option<CpuTensor>,
    out_weight: ResidentWeight,
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
            q_weight: ResidentWeight::from_hfq(hfq, q_weight_entry)?,
            q_bias: optional_tensor(hfq, q_bias_entry)?,
            k_weight: ResidentWeight::from_hfq(hfq, k_weight_entry)?,
            k_bias: optional_tensor(hfq, k_bias_entry)?,
            v_weight: ResidentWeight::from_hfq(hfq, v_weight_entry)?,
            v_bias: optional_tensor(hfq, v_bias_entry)?,
            norm_q_weight: optional_tensor(hfq, norm_q_entry)?,
            norm_k_weight: optional_tensor(hfq, norm_k_entry)?,
            out_weight: ResidentWeight::from_hfq(hfq, out_weight_entry)?,
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
        let [inner_width, hidden_width] = shape2_slice(self.q_weight.shape())?;
        if inner_width == 0 || hidden_width == 0 {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{} transformer attention q weight shape {:?} is empty",
                self.stream_label,
                self.q_weight.shape()
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
        // Grouped-query attention: K/V may have fewer heads than Q (Krea2 uses
        // 12 KV heads to 48 Q heads). Derive kv_heads from the K projection rows
        // and require it to divide the Q head count. When kv_heads == heads this
        // is ordinary multi-head attention (QwenImage), so the QwenImage path is
        // unchanged. V must match K, and the K/V biases follow the kv inner width.
        let [kv_inner_width, kv_hidden_width] = shape2_slice(self.k_weight.shape())?;
        if kv_hidden_width != hidden_width {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{} transformer attention k weight hidden width {kv_hidden_width} != q hidden width {hidden_width}",
                self.stream_label
            )));
        }
        if kv_inner_width == 0 || kv_inner_width % head_dim != 0 {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{} transformer attention k inner width {kv_inner_width} is not a positive multiple of head_dim {head_dim}",
                self.stream_label
            )));
        }
        let kv_heads = kv_inner_width / head_dim;
        if heads % kv_heads != 0 {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{} transformer attention q heads {heads} is not a multiple of kv heads {kv_heads}",
                self.stream_label
            )));
        }
        if self.v_weight.shape() != [kv_inner_width, hidden_width] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{} transformer attention v weight shape {:?} != [{kv_inner_width}, {hidden_width}]",
                self.stream_label,
                self.v_weight.shape()
            )));
        }
        validate_attention_bias_shape(self.stream_label, "q", self.q_bias.as_ref(), inner_width)?;
        validate_attention_bias_shape(
            self.stream_label,
            "k",
            self.k_bias.as_ref(),
            kv_inner_width,
        )?;
        validate_attention_bias_shape(
            self.stream_label,
            "v",
            self.v_bias.as_ref(),
            kv_inner_width,
        )?;
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
        if self.out_weight.shape() != [hidden_width, inner_width] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{} transformer attention output weight shape {:?} != [{hidden_width}, {inner_width}]",
                self.stream_label,
                self.out_weight.shape()
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
        // Decode the packed weights to f32 transiently for this op; each is
        // dropped before the next so only one is expanded at a time.
        let q = linear_3d_with_runtime_context(
            hidden,
            &self.q_weight.decode()?,
            self.q_bias.as_ref(),
            runtime_context,
        )?;
        let k = linear_3d_with_runtime_context(
            hidden,
            &self.k_weight.decode()?,
            self.k_bias.as_ref(),
            runtime_context,
        )?;
        let v = linear_3d_with_runtime_context(
            hidden,
            &self.v_weight.decode()?,
            self.v_bias.as_ref(),
            runtime_context,
        )?;
        // Grouped-query attention: K/V carry kv_heads (<= heads). QK-norm is
        // applied per kv head, then K/V are expanded to the full Q head count so
        // the downstream RoPE / SDPA path stays multi-head. kv_heads == heads
        // (QwenImage) makes the expansion an identity.
        let [_, _, kv_width] = shape3(&k)?;
        let kv_heads = if head_dim == 0 {
            heads
        } else {
            kv_width / head_dim
        };
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
            kv_heads,
            head_dim,
            1e-6,
        )?;
        let k = repeat_kv_heads_3d(&k, heads, kv_heads, head_dim)?;
        let v = repeat_kv_heads_3d(&v, heads, kv_heads, head_dim)?;
        Ok(TransformerAttentionQkv { q, k, v })
    }

    pub(crate) fn project_output_with_runtime_context(
        &self,
        attention: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        linear_3d_with_runtime_context(
            attention,
            &self.out_weight.decode()?,
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
    // Krea2 single-stream sigmoid attention gate (`attn.to_gate`), applied to the
    // flattened SDPA output before the output projection. Absent for QwenImage.
    gate_weight: Option<ResidentWeight>,
    gate_bias: Option<CpuTensor>,
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

        // Krea2 single-stream attention carries a sigmoid output gate; QwenImage
        // does not. `to_gate` is a square [hidden, hidden] projection over the
        // (modulated) block input.
        let gate_weight = optional_resident(hfq, &format!("{block_prefix}.to_gate.weight"))?;
        let gate_bias = if gate_weight.is_some() {
            optional_tensor(hfq, &format!("{block_prefix}.to_gate.bias"))?
        } else {
            None
        };
        if let Some(weight) = gate_weight.as_ref() {
            if weight.shape.as_slice() != [inner_width, hidden_width] {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "transformer block {block_index} to_gate weight shape {:?} != [{inner_width}, {hidden_width}]",
                    weight.shape
                )));
            }
        }

        Ok(Self {
            family,
            block_index,
            heads,
            head_dim,
            hidden_width,
            inner_width,
            image,
            text,
            gate_weight,
            gate_bias,
        })
    }

    /// Load a single-stream (Krea-style) attention from an explicit `attn`
    /// tensor prefix, e.g. `transformer/tensors/text_fusion.refiner_blocks.0.attn`.
    /// Always image-stream-only with an optional sigmoid `to_gate`; reused by the
    /// text-fusion refinement blocks which share the block attention shape.
    pub(crate) fn single_stream_from_prefix(
        hfq: &HfqFile,
        attn_prefix: &str,
        heads: usize,
    ) -> DiffusionResult<Self> {
        let image = TransformerAttentionStreamProjection::from_hfq(
            hfq,
            "text_fusion",
            &format!("{attn_prefix}.to_q.weight"),
            &format!("{attn_prefix}.to_q.bias"),
            &format!("{attn_prefix}.to_k.weight"),
            &format!("{attn_prefix}.to_k.bias"),
            &format!("{attn_prefix}.to_v.weight"),
            &format!("{attn_prefix}.to_v.bias"),
            &format!("{attn_prefix}.norm_q.weight"),
            &format!("{attn_prefix}.norm_k.weight"),
            &format!("{attn_prefix}.to_out.0.weight"),
            &format!("{attn_prefix}.to_out.0.bias"),
            true,
            heads,
            None,
            None,
            None,
        )?
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata(format!("attention stream {attn_prefix:?} is missing"))
        })?;
        let (hidden_width, inner_width, head_dim) =
            image.validate_shapes(heads, None, None, None)?;
        let gate_weight = optional_resident(hfq, &format!("{attn_prefix}.to_gate.weight"))?;
        let gate_bias = if gate_weight.is_some() {
            optional_tensor(hfq, &format!("{attn_prefix}.to_gate.bias"))?
        } else {
            None
        };
        if let Some(weight) = gate_weight.as_ref() {
            if weight.shape.as_slice() != [inner_width, hidden_width] {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "attention {attn_prefix:?} to_gate weight shape {:?} != [{inner_width}, {hidden_width}]",
                    weight.shape
                )));
            }
        }
        Ok(Self {
            family: TransformerDenoiserFamily::Krea2,
            block_index: 0,
            heads,
            head_dim,
            hidden_width,
            inner_width,
            image,
            text: None,
            gate_weight,
            gate_bias,
        })
    }

    /// Krea2 single-stream self-attention with GQA, per-token RoPE and the
    /// sigmoid output gate. `hidden` is the modulated, normalized joint sequence
    /// `[text; image]`; `rotary` (if present) already covers the joint token
    /// order. Returns the projected attention output (pre gated-residual).
    pub(crate) fn attend_krea_self_gated_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        rotary: Option<&RotaryFrequencies>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let mut qkv = self.project_image_qkv_with_runtime_context(hidden, runtime_context)?;
        if let Some(freqs) = rotary {
            qkv.q = apply_qwen_rotary_embedding(&qkv.q, freqs, self.heads, self.head_dim)?;
            qkv.k = apply_qwen_rotary_embedding(&qkv.k, freqs, self.heads, self.head_dim)?;
        }
        let attention = scaled_dot_product_attention_with_runtime_context(
            &qkv.q,
            &qkv.k,
            &qkv.v,
            self.heads,
            runtime_context,
        )?;
        let gated = match self.gate_weight.as_ref() {
            Some(weight) => {
                let gate = linear_3d_with_runtime_context(
                    hidden,
                    &weight.decode()?,
                    self.gate_bias.as_ref(),
                    runtime_context,
                )?;
                sigmoid_gate_3d(&attention, &gate)?
            }
            None => attention,
        };
        self.project_image_output_with_runtime_context(&gated, runtime_context)
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
    proj_weight: Option<ResidentWeight>,
    proj_bias: Option<CpuTensor>,
    up_weight: Option<ResidentWeight>,
    up_bias: Option<CpuTensor>,
    gate_weight: Option<ResidentWeight>,
    gate_bias: Option<CpuTensor>,
    down_weight: ResidentWeight,
    down_bias: Option<CpuTensor>,
}

#[allow(dead_code)]
impl TransformerFeedForwardStream {
    pub(crate) fn qwen_geglu_from_hfq(
        hfq: &HfqFile,
        stream_label: &'static str,
        prefix: &str,
    ) -> DiffusionResult<Self> {
        let proj_weight = ResidentWeight::from_hfq(hfq, &format!("{prefix}.net.0.proj.weight"))?;
        let proj_bias = CpuTensor::from_hfq(hfq, &format!("{prefix}.net.0.proj.bias"))?;
        let down_weight = ResidentWeight::from_hfq(hfq, &format!("{prefix}.net.2.weight"))?;
        let down_bias = CpuTensor::from_hfq(hfq, &format!("{prefix}.net.2.bias"))?;
        let [projected_width, hidden_width] = shape2_slice(proj_weight.shape())?;
        if projected_width == 0 || projected_width % 2 != 0 {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{stream_label} transformer GEGLU projection shape {:?} is not [2*inner, hidden]",
                proj_weight.shape()
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
            down_weight.shape(),
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
        let up_weight = ResidentWeight::from_hfq(hfq, &format!("{prefix}.up.weight"))?;
        let gate_weight = ResidentWeight::from_hfq(hfq, &format!("{prefix}.gate.weight"))?;
        let down_weight = ResidentWeight::from_hfq(hfq, &format!("{prefix}.down.weight"))?;
        let up_bias = optional_tensor(hfq, &format!("{prefix}.up.bias"))?;
        let gate_bias = optional_tensor(hfq, &format!("{prefix}.gate.bias"))?;
        let down_bias = optional_tensor(hfq, &format!("{prefix}.down.bias"))?;
        let [inner_width, hidden_width] = shape2_slice(up_weight.shape())?;
        if inner_width == 0 || hidden_width == 0 {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{stream_label} transformer SwiGLU up projection shape {:?} is empty",
                up_weight.shape()
            )));
        }
        if gate_weight.shape() != [inner_width, hidden_width] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "{stream_label} transformer SwiGLU gate weight shape {:?} != [{inner_width}, {hidden_width}]",
                gate_weight.shape()
            )));
        }
        validate_attention_bias_shape(stream_label, "ff.up", up_bias.as_ref(), inner_width)?;
        validate_attention_bias_shape(stream_label, "ff.gate", gate_bias.as_ref(), inner_width)?;
        validate_transformer_ff_down_shape(
            stream_label,
            down_weight.shape(),
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
                    &proj_weight.decode()?,
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
                    &up_weight.decode()?,
                    self.up_bias.as_ref(),
                    runtime_context,
                )?;
                let gate = linear_3d_with_runtime_context(
                    hidden,
                    &gate_weight.decode()?,
                    self.gate_bias.as_ref(),
                    runtime_context,
                )?;
                swiglu_gate_3d(&up, &gate)?
            }
        };
        linear_3d_with_runtime_context(
            &activated,
            &self.down_weight.decode()?,
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
    // Krea2 blocks apply weighted RMSNorm (`norm1`/`norm2`) around attention and
    // the feed-forward; QwenImage uses affine-free LayerNorm so these are None.
    norm1_weight: Option<CpuTensor>,
    norm2_weight: Option<CpuTensor>,
}

#[allow(dead_code)]
impl NativeTransformerBlock {
    pub(crate) fn from_hfq(
        hfq: &HfqFile,
        family: TransformerDenoiserFamily,
        block_index: usize,
        heads: usize,
    ) -> DiffusionResult<Self> {
        let block_prefix = format!("transformer/tensors/transformer_blocks.{block_index}");
        let (norm1_weight, norm2_weight) = match family {
            TransformerDenoiserFamily::Krea2 => (
                Some(CpuTensor::from_hfq(
                    hfq,
                    &format!("{block_prefix}.norm1.weight"),
                )?),
                Some(CpuTensor::from_hfq(
                    hfq,
                    &format!("{block_prefix}.norm2.weight"),
                )?),
            ),
            TransformerDenoiserFamily::QwenImage | TransformerDenoiserFamily::Unknown => {
                (None, None)
            }
        };
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
            norm1_weight,
            norm2_weight,
        })
    }

    /// Krea2 single-stream block forward on the joint `[text; image]` sequence.
    ///
    /// `time_modulation` is the shared `time_mod_proj(time_embed)` output
    /// (`[batch, 6 * hidden]`); it is combined with this block's
    /// `scale_shift_table` into the six adaLN chunks
    /// `[prescale, preshift, pregate, postscale, postshift, postgate]`. `rotary`
    /// (if present) already covers the joint token order.
    pub(crate) fn forward_krea_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        time_modulation: &CpuTensor,
        rotary: Option<&RotaryFrequencies>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        if !matches!(self.family, TransformerDenoiserFamily::Krea2) {
            return Err(DiffusionError::InvalidMetadata(format!(
                "transformer block {} family {:?} is not Krea2-style",
                self.block_index, self.family
            )));
        }
        let norm1_weight = self.norm1_weight.as_ref().ok_or_else(|| {
            DiffusionError::InvalidMetadata("Krea transformer block norm1 weight is missing".into())
        })?;
        let norm2_weight = self.norm2_weight.as_ref().ok_or_else(|| {
            DiffusionError::InvalidMetadata("Krea transformer block norm2 weight is missing".into())
        })?;
        // adaLN chunks (Krea order): pre = attention, post = feed-forward.
        let modulation = self
            .modulation
            .krea_scale_shift_with_runtime_context(time_modulation, runtime_context)?;
        let prescale = extract_modulation_chunk_2d(&modulation, 0)?;
        let preshift = extract_modulation_chunk_2d(&modulation, 1)?;
        let pregate = extract_modulation_chunk_2d(&modulation, 2)?;
        let postscale = extract_modulation_chunk_2d(&modulation, 3)?;
        let postshift = extract_modulation_chunk_2d(&modulation, 4)?;
        let postgate = extract_modulation_chunk_2d(&modulation, 5)?;

        let attn_input = modulate_3d(
            &rms_norm_3d_with_runtime_context(hidden, norm1_weight, 1e-5, runtime_context)?,
            &preshift,
            &prescale,
        )?;
        let attention = self.attention.attend_krea_self_gated_with_runtime_context(
            &attn_input,
            rotary,
            runtime_context,
        )?;
        let hidden = gated_residual_3d(hidden, &attention, &pregate)?;

        let ff_input = modulate_3d(
            &rms_norm_3d_with_runtime_context(&hidden, norm2_weight, 1e-5, runtime_context)?,
            &postshift,
            &postscale,
        )?;
        let feed_forward = self
            .feed_forward
            .forward_image_with_runtime_context(&ff_input, runtime_context)?;
        gated_residual_3d(&hidden, &feed_forward, &postgate)
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

/// Krea2 text-fusion refinement block: a plain pre-norm transformer block
/// (weighted RMSNorm -> gated GQA self-attention -> residual -> RMSNorm ->
/// SwiGLU -> residual) with NO timestep/adaLN modulation. Runs at the text
/// hidden width for both the layerwise (attends the 12-layer axis) and refiner
/// (attends the token axis) stacks.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct NativeTextFusionBlock {
    norm1_weight: CpuTensor,
    norm2_weight: CpuTensor,
    attention: NativeTransformerAttentionProjection,
    feed_forward: TransformerFeedForwardStream,
}

#[allow(dead_code)]
impl NativeTextFusionBlock {
    pub(crate) fn from_hfq(
        hfq: &HfqFile,
        block_prefix: &str,
        heads: usize,
    ) -> DiffusionResult<Self> {
        Ok(Self {
            norm1_weight: CpuTensor::from_hfq(hfq, &format!("{block_prefix}.norm1.weight"))?,
            norm2_weight: CpuTensor::from_hfq(hfq, &format!("{block_prefix}.norm2.weight"))?,
            attention: NativeTransformerAttentionProjection::single_stream_from_prefix(
                hfq,
                &format!("{block_prefix}.attn"),
                heads,
            )?,
            feed_forward: TransformerFeedForwardStream::krea_swiglu_from_hfq(
                hfq,
                "text_fusion",
                &format!("{block_prefix}.ff"),
            )?,
        })
    }

    pub(crate) fn forward_with_runtime_context(
        &self,
        hidden: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let attention = self.attention.attend_krea_self_gated_with_runtime_context(
            &rms_norm_3d_with_runtime_context(hidden, &self.norm1_weight, 1e-5, runtime_context)?,
            None,
            runtime_context,
        )?;
        let hidden = residual_add_3d(hidden, &attention)?;
        let feed_forward = self.feed_forward.forward_with_runtime_context(
            &rms_norm_3d_with_runtime_context(&hidden, &self.norm2_weight, 1e-5, runtime_context)?,
            runtime_context,
        )?;
        residual_add_3d(&hidden, &feed_forward)
    }
}

/// Krea2 `text_fusion` module: fuses the selected Qwen3-VL encoder layers into a
/// single conditioning stream. `layerwise_blocks` attend across the per-token
/// layer axis, `projector` (`[1, num_layers]`) collapses the layers to one
/// representation, then `refiner_blocks` attend across the token axis.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct NativeTextFusion {
    layerwise_blocks: Vec<NativeTextFusionBlock>,
    projector: CpuTensor,
    refiner_blocks: Vec<NativeTextFusionBlock>,
    num_layers: usize,
    text_hidden_width: usize,
    heads: usize,
}

#[allow(dead_code)]
impl NativeTextFusion {
    fn count_blocks(hfq: &HfqFile, stack_prefix: &str) -> usize {
        let mut count = 0;
        while hfq
            .find_tensor_info(&format!(
                "transformer/tensors/{stack_prefix}.{count}.norm1.weight"
            ))
            .is_some()
        {
            count += 1;
        }
        count
    }

    pub(crate) fn from_hfq(hfq: &HfqFile, heads: usize) -> DiffusionResult<Option<Self>> {
        let Some(projector) =
            optional_tensor(hfq, "transformer/tensors/text_fusion.projector.weight")?
        else {
            return Ok(None);
        };
        let [projector_rows, num_layers] = shape2(&projector)?;
        if projector_rows != 1 || num_layers == 0 {
            return Err(DiffusionError::InvalidMetadata(format!(
                "text_fusion projector shape {:?} != [1, num_layers>0]",
                projector.shape
            )));
        }
        let load_stack = |stack_prefix: &str| -> DiffusionResult<Vec<NativeTextFusionBlock>> {
            let count = Self::count_blocks(hfq, stack_prefix);
            (0..count)
                .map(|index| {
                    NativeTextFusionBlock::from_hfq(
                        hfq,
                        &format!("transformer/tensors/{stack_prefix}.{index}"),
                        heads,
                    )
                })
                .collect()
        };
        let layerwise_blocks = load_stack("text_fusion.layerwise_blocks")?;
        let refiner_blocks = load_stack("text_fusion.refiner_blocks")?;
        if layerwise_blocks.is_empty() && refiner_blocks.is_empty() {
            return Err(DiffusionError::InvalidMetadata(
                "text_fusion has a projector but no layerwise/refiner blocks".to_string(),
            ));
        }
        let text_hidden_width = layerwise_blocks
            .first()
            .or_else(|| refiner_blocks.first())
            .map(|block| block.norm1_weight.data.len())
            .unwrap_or(0);
        Ok(Some(Self {
            layerwise_blocks,
            projector,
            refiner_blocks,
            num_layers,
            text_hidden_width,
            heads,
        }))
    }

    /// `layer_stack` is `[batch, seq, num_layers, dim]` (the stacked selected
    /// encoder hidden states). Returns the fused `[batch, seq, dim]` conditioning.
    pub(crate) fn forward_with_runtime_context(
        &self,
        layer_stack: &CpuTensor,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let [batch, seq, layers, dim] = match layer_stack.shape.as_slice() {
            [b, s, l, d] => [*b, *s, *l, *d],
            _ => {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "text_fusion input {:?} is not [batch, seq, num_layers, dim]",
                    layer_stack.shape
                )))
            }
        };
        if layers != self.num_layers || dim != self.text_hidden_width {
            return Err(DiffusionError::InvalidRequest(format!(
                "text_fusion input [.., {layers}, {dim}] != expected [.., {}, {}]",
                self.num_layers, self.text_hidden_width
            )));
        }
        // Layerwise: treat the layer axis as the sequence, one row per token.
        let mut per_layer = CpuTensor {
            shape: vec![batch * seq, layers, dim],
            data: layer_stack.data.clone(),
        };
        for block in &self.layerwise_blocks {
            per_layer = block.forward_with_runtime_context(&per_layer, runtime_context)?;
        }
        // Projector: collapse the layer axis with the [1, num_layers] weights.
        let mut fused = CpuTensor::zeros(&[batch, seq, dim]);
        for token in 0..(batch * seq) {
            for d in 0..dim {
                let mut acc = 0.0f32;
                for l in 0..layers {
                    acc += self.projector.data[l] * per_layer.data[(token * layers + l) * dim + d];
                }
                fused.data[token * dim + d] = acc;
            }
        }
        // Refiner: attend across the token axis.
        for block in &self.refiner_blocks {
            fused = block.forward_with_runtime_context(&fused, runtime_context)?;
        }
        Ok(fused)
    }

    /// Adapter from the text encoder to text_fusion: stack the `num_layers`
    /// selected Qwen3-VL hidden states (each `[batch, seq, dim]`, in
    /// `text_encoder_select_layers` order) into the `[batch, seq, num_layers,
    /// dim]` layer axis and run the fusion. This is the seam the pipeline drives
    /// once the encoder has produced the per-layer hidden states.
    pub(crate) fn encode_from_layers_with_runtime_context(
        &self,
        layers: &[CpuTensor],
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        if layers.len() != self.num_layers {
            return Err(DiffusionError::InvalidRequest(format!(
                "text_fusion expects {} encoder layers, got {}",
                self.num_layers,
                layers.len()
            )));
        }
        let [batch, seq, dim] = shape3(&layers[0])?;
        if dim != self.text_hidden_width {
            return Err(DiffusionError::InvalidRequest(format!(
                "text_fusion encoder dim {dim} != expected {}",
                self.text_hidden_width
            )));
        }
        let num_layers = self.num_layers;
        let mut stacked = CpuTensor::zeros(&[batch, seq, num_layers, dim]);
        for (layer_index, layer) in layers.iter().enumerate() {
            if layer.shape.as_slice() != [batch, seq, dim] {
                return Err(DiffusionError::InvalidRequest(format!(
                    "text_fusion encoder layer {layer_index} shape {:?} != [{batch}, {seq}, {dim}]",
                    layer.shape
                )));
            }
            for token in 0..(batch * seq) {
                let src = token * dim;
                let dst = (token * num_layers + layer_index) * dim;
                stacked.data[dst..dst + dim].copy_from_slice(&layer.data[src..src + dim]);
            }
        }
        self.forward_with_runtime_context(&stacked, runtime_context)
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
        if !matches!(
            topology.family,
            TransformerDenoiserFamily::QwenImage | TransformerDenoiserFamily::Krea2
        ) {
            return Err(DiffusionError::InvalidMetadata(format!(
                "native transformer denoiser assembly currently supports Qwen image / Krea2 MMDiT only; got {}",
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

    /// Krea2 single-stream MMDiT forward: text and image tokens are concatenated
    /// into one sequence, run through the `transformer_blocks` with a shared
    /// timestep modulation and a joint `[text; image]` RoPE, then the image tail
    /// is projected back to latents through the final adaLN layer.
    pub(crate) fn forward_krea_with_runtime_context(
        &self,
        latents: &LatentBatch,
        timesteps: &[f32],
        text_hidden: &CpuTensor,
        text_attention_mask: Option<&CpuTensor>,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<LatentBatch> {
        if !matches!(self.family, TransformerDenoiserFamily::Krea2) {
            return Err(DiffusionError::InvalidMetadata(format!(
                "native transformer denoiser family {:?} is not Krea2 MMDiT",
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

        let image_hidden = self
            .io
            .project_latents_to_hidden_with_runtime_context(latents, runtime_context)?;
        let text_hidden = self
            .io
            .project_text_to_hidden_with_runtime_context(text_hidden, runtime_context)?;
        let [_, text_seq, _] = shape3(&text_hidden)?;
        let [_, image_seq, _] = shape3(&image_hidden)?;
        if let Some(mask) = text_attention_mask {
            validate_text_attention_mask(mask, latents.batch, text_seq, "Krea text")?;
        }
        // Joint RoPE covers the concatenated [text; image] token order.
        let rotary = self.qwen_rotary_embeddings(latents, text_seq)?;
        let joint_rotary = match rotary.as_ref() {
            Some(freqs) => Some(combined_joint_rotary(freqs)?),
            None => None,
        };
        // Shared timestep modulation: time_embed -> time_mod_proj -> [batch, 6*hidden].
        let time_embedding = self
            .timestep_embedding
            .forward_with_runtime_context(timesteps, runtime_context)?;
        let time_modulation = self
            .timestep_embedding
            .modulation_with_runtime_context(&time_embedding, runtime_context)?
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata(
                    "Krea transformer requires a time_mod_proj modulation projection".to_string(),
                )
            })?;

        let mut joint = concat_sequence_3d(&text_hidden, &image_hidden)?;
        for block in &self.blocks {
            joint = block.forward_krea_with_runtime_context(
                &joint,
                &time_modulation,
                joint_rotary.as_ref(),
                runtime_context,
            )?;
        }
        let image_hidden = slice_sequence_3d(&joint, text_seq, image_seq)?;
        self.io.project_hidden_to_latents_with_runtime_context(
            &image_hidden,
            &time_embedding,
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
                let prediction = match self.family {
                    TransformerDenoiserFamily::Krea2 => self.forward_krea_with_runtime_context(
                        &model_latents,
                        timesteps,
                        encoder_states,
                        attention_mask,
                        runtime_context,
                    )?,
                    TransformerDenoiserFamily::QwenImage | TransformerDenoiserFamily::Unknown => {
                        self.forward_qwen_with_runtime_context(
                            &model_latents,
                            timesteps,
                            encoder_states,
                            attention_mask,
                            runtime_context,
                        )?
                    }
                };
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

/// Expand grouped-query K/V heads up to the full query head count.
///
/// Input is `[batch, seq, kv_heads * head_dim]`; output is
/// `[batch, seq, heads * head_dim]`. Each KV head serves a contiguous group of
/// `heads / kv_heads` query heads (the PyTorch `repeat_kv` ordering: query head
/// `h` reads KV head `h / (heads / kv_heads)`). When `kv_heads == heads` this is
/// a cheap clone (ordinary multi-head attention), so callers can invoke it
/// unconditionally.
#[allow(dead_code)]
pub(crate) fn repeat_kv_heads_3d(
    input: &CpuTensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, width] = shape3(input)?;
    if kv_heads == 0 || head_dim == 0 || width != kv_heads * head_dim {
        return Err(DiffusionError::InvalidMetadata(format!(
            "GQA expand input width {width} is incompatible with kv_heads {kv_heads} and head_dim {head_dim}"
        )));
    }
    if heads == 0 || heads % kv_heads != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "GQA expand target heads {heads} is not a positive multiple of kv_heads {kv_heads}"
        )));
    }
    if heads == kv_heads {
        return Ok(input.clone());
    }
    let group = heads / kv_heads;
    let out_width = heads * head_dim;
    let mut out = CpuTensor::zeros(&[batch, seq, out_width]);
    for b in 0..batch {
        for token in 0..seq {
            let in_base = (b * seq + token) * width;
            let out_base = (b * seq + token) * out_width;
            for head in 0..heads {
                let kv_head = head / group;
                let src = in_base + kv_head * head_dim;
                let dst = out_base + head * head_dim;
                out.data[dst..dst + head_dim].copy_from_slice(&input.data[src..src + head_dim]);
            }
        }
    }
    Ok(out)
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
    down_shape: &[usize],
    down_bias: Option<&CpuTensor>,
    hidden_width: usize,
    inner_width: usize,
) -> DiffusionResult<()> {
    if down_shape != [hidden_width, inner_width] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "{stream_label} transformer feed-forward down weight shape {down_shape:?} != [{hidden_width}, {inner_width}]"
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

/// Slice one adaLN chunk out of a `[batch, chunks, hidden]` modulation tensor,
/// returning `[batch, hidden]` for `modulate_3d` / `gated_residual_3d`.
#[allow(dead_code)]
pub(crate) fn extract_modulation_chunk_2d(
    modulation: &CpuTensor,
    chunk_index: usize,
) -> DiffusionResult<CpuTensor> {
    let [batch, chunks, hidden] = shape3(modulation)?;
    if chunk_index >= chunks {
        return Err(DiffusionError::InvalidMetadata(format!(
            "modulation chunk {chunk_index} out of range for {chunks} chunks"
        )));
    }
    let mut out = CpuTensor::zeros(&[batch, hidden]);
    for b in 0..batch {
        let src = (b * chunks + chunk_index) * hidden;
        let dst = b * hidden;
        out.data[dst..dst + hidden].copy_from_slice(&modulation.data[src..src + hidden]);
    }
    Ok(out)
}

/// Elementwise `a + b` for matching `[batch, seq, width]` tensors (the plain,
/// ungated residual used by the text-fusion refinement blocks).
#[allow(dead_code)]
pub(crate) fn residual_add_3d(a: &CpuTensor, b: &CpuTensor) -> DiffusionResult<CpuTensor> {
    if a.shape != b.shape {
        return Err(DiffusionError::InvalidMetadata(format!(
            "residual add shape mismatch {:?}/{:?}",
            a.shape, b.shape
        )));
    }
    let mut out = CpuTensor::zeros(&a.shape);
    for (idx, slot) in out.data.iter_mut().enumerate() {
        *slot = a.data[idx] + b.data[idx];
    }
    Ok(out)
}

/// Elementwise sigmoid gate: `value * sigmoid(gate)`, same `[batch, seq, width]`
/// shape for both (Krea2 `attn.to_gate`).
#[allow(dead_code)]
pub(crate) fn sigmoid_gate_3d(value: &CpuTensor, gate: &CpuTensor) -> DiffusionResult<CpuTensor> {
    if value.shape != gate.shape {
        return Err(DiffusionError::InvalidMetadata(format!(
            "sigmoid gate shape mismatch value/gate {:?}/{:?}",
            value.shape, gate.shape
        )));
    }
    let mut out = CpuTensor::zeros(&value.shape);
    for (idx, slot) in out.data.iter_mut().enumerate() {
        let g = 1.0f32 / (1.0 + (-gate.data[idx]).exp());
        *slot = value.data[idx] * g;
    }
    Ok(out)
}

/// Build one RoPE frequency table for the Krea2 joint `[text; image]` sequence
/// by row-concatenating the text and image `cos`/`sin` tables (`[seq, dim/2]`).
/// The token order matches `concat_sequence_3d(text, image)`.
#[allow(dead_code)]
pub(crate) fn combined_joint_rotary(
    rotary: &QwenRotaryEmbeddings,
) -> DiffusionResult<RotaryFrequencies> {
    Ok(RotaryFrequencies {
        cos: concat_rows_2d(&rotary.text.cos, &rotary.image.cos)?,
        sin: concat_rows_2d(&rotary.text.sin, &rotary.image.sin)?,
    })
}

/// Concatenate two `[rows, cols]` tensors along the row axis.
#[allow(dead_code)]
pub(crate) fn concat_rows_2d(top: &CpuTensor, bottom: &CpuTensor) -> DiffusionResult<CpuTensor> {
    let [top_rows, top_cols] = shape2(top)?;
    let [bottom_rows, bottom_cols] = shape2(bottom)?;
    if top_cols != bottom_cols {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot row-concat tensors with shapes {:?} and {:?}",
            top.shape, bottom.shape
        )));
    }
    let mut out = CpuTensor::zeros(&[top_rows + bottom_rows, top_cols]);
    out.data[..top.data.len()].copy_from_slice(&top.data);
    out.data[top.data.len()..].copy_from_slice(&bottom.data);
    Ok(out)
}

/// Slice `[start, start + len)` tokens out of a `[batch, seq, width]` sequence.
#[allow(dead_code)]
pub(crate) fn slice_sequence_3d(
    input: &CpuTensor,
    start: usize,
    len: usize,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, width] = shape3(input)?;
    if start + len > seq {
        return Err(DiffusionError::InvalidMetadata(format!(
            "sequence slice [{start}, {}) out of range for seq {seq}",
            start + len
        )));
    }
    let mut out = CpuTensor::zeros(&[batch, len, width]);
    for b in 0..batch {
        let src = (b * seq + start) * width;
        let dst = b * len * width;
        out.data[dst..dst + len * width].copy_from_slice(&input.data[src..src + len * width]);
    }
    Ok(out)
}

#[allow(dead_code)]
pub(crate) fn concat_sequence_3d(
    left: &CpuTensor,
    right: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
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

// ---------------------------------------------------------------------------
// Native Qwen3-VL text encoder (Krea2 conditioning source)
//
// Runs the `language_model` text tower of the Qwen3-VL encoder prefill-only and
// captures the selected mid-layer hidden states that feed `NativeTextFusion`.
// Reuses the diffusion crate's own primitives (RMSNorm, GQA head-expand, causal
// SDPA, SwiGLU) — the same self-contained pattern as `ClipTextEncoder` — so it
// needs neither the runtime model-load path nor a GPU to run on CPU.
//
// PARITY NOTE: the structure/shapes are grounded in the real tensor manifest,
// but three conventions must be validated against a diffusers reference before
// trusting decoded pixels: (1) RoPE is applied half-split (Qwen/Llama
// `rotate_half`) with theta from config — vs the DiT's interleaved RoPE;
// (2) attention is causal; (3) `select_layers` index the per-layer outputs
// (1-based over the layer stack). These are the standard Qwen3 conventions but
// are unverified numerically here.
// ---------------------------------------------------------------------------

/// Build 1-D RoPE cos/sin tables `[seq, head_dim/2]` for `theta`.
#[allow(dead_code)]
pub(crate) fn rope_1d_cos_sin(seq: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for pos in 0..seq {
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = pos as f32 * inv_freq;
            cos[pos * half + i] = angle.cos();
            sin[pos * half + i] = angle.sin();
        }
    }
    (cos, sin)
}

/// Apply half-split RoPE (`rotate_half`) to `[batch, seq, heads*head_dim]`.
#[allow(dead_code)]
pub(crate) fn apply_rope_halfsplit_3d(
    input: &CpuTensor,
    cos: &[f32],
    sin: &[f32],
    heads: usize,
    head_dim: usize,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, width] = shape3(input)?;
    if head_dim % 2 != 0 || width != heads * head_dim {
        return Err(DiffusionError::InvalidMetadata(format!(
            "rope input width {width} incompatible with heads {heads} head_dim {head_dim}"
        )));
    }
    let half = head_dim / 2;
    let mut out = CpuTensor::zeros(&input.shape);
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                let base = ((b * seq + s) * heads + h) * head_dim;
                let freq = s * half;
                for i in 0..half {
                    let x1 = input.data[base + i];
                    let x2 = input.data[base + half + i];
                    let c = cos[freq + i];
                    let sn = sin[freq + i];
                    out.data[base + i] = x1 * c - x2 * sn;
                    out.data[base + half + i] = x2 * c + x1 * sn;
                }
            }
        }
    }
    Ok(out)
}

/// One Qwen3 decoder layer: RMSNorm -> GQA attention (QK-norm, RoPE, causal) ->
/// residual -> RMSNorm -> SwiGLU -> residual.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct Qwen3EncoderLayer {
    input_norm: Vec<f32>,
    q_proj: ResidentWeight,
    k_proj: ResidentWeight,
    v_proj: ResidentWeight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    o_proj: ResidentWeight,
    post_norm: Vec<f32>,
    gate_proj: ResidentWeight,
    up_proj: ResidentWeight,
    down_proj: ResidentWeight,
}

#[allow(dead_code)]
impl Qwen3EncoderLayer {
    const EPS: f32 = 1e-6;

    pub(crate) fn from_hfq(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Self> {
        Ok(Self {
            input_norm: CpuTensor::from_hfq(hfq, &format!("{prefix}.input_layernorm.weight"))?.data,
            q_proj: ResidentWeight::from_hfq(hfq, &format!("{prefix}.self_attn.q_proj.weight"))?,
            k_proj: ResidentWeight::from_hfq(hfq, &format!("{prefix}.self_attn.k_proj.weight"))?,
            v_proj: ResidentWeight::from_hfq(hfq, &format!("{prefix}.self_attn.v_proj.weight"))?,
            q_norm: CpuTensor::from_hfq(hfq, &format!("{prefix}.self_attn.q_norm.weight"))?.data,
            k_norm: CpuTensor::from_hfq(hfq, &format!("{prefix}.self_attn.k_norm.weight"))?.data,
            o_proj: ResidentWeight::from_hfq(hfq, &format!("{prefix}.self_attn.o_proj.weight"))?,
            post_norm: CpuTensor::from_hfq(
                hfq,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?
            .data,
            gate_proj: ResidentWeight::from_hfq(hfq, &format!("{prefix}.mlp.gate_proj.weight"))?,
            up_proj: ResidentWeight::from_hfq(hfq, &format!("{prefix}.mlp.up_proj.weight"))?,
            down_proj: ResidentWeight::from_hfq(hfq, &format!("{prefix}.mlp.down_proj.weight"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_runtime_context(
        &self,
        input: &CpuTensor,
        cos: &[f32],
        sin: &[f32],
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let normed = rms_norm_3d_with_runtime_context(
            &CpuTensor {
                shape: input.shape.clone(),
                data: input.data.clone(),
            },
            &CpuTensor {
                shape: vec![self.input_norm.len()],
                data: self.input_norm.clone(),
            },
            Self::EPS,
            runtime_context,
        )?;
        let mut q =
            linear_3d_with_runtime_context(&normed, &self.q_proj.decode()?, None, runtime_context)?;
        let mut k =
            linear_3d_with_runtime_context(&normed, &self.k_proj.decode()?, None, runtime_context)?;
        let v =
            linear_3d_with_runtime_context(&normed, &self.v_proj.decode()?, None, runtime_context)?;
        q = rms_norm_attention_heads_3d(
            &q,
            &CpuTensor {
                shape: vec![head_dim],
                data: self.q_norm.clone(),
            },
            heads,
            head_dim,
            Self::EPS,
        )?;
        k = rms_norm_attention_heads_3d(
            &k,
            &CpuTensor {
                shape: vec![head_dim],
                data: self.k_norm.clone(),
            },
            kv_heads,
            head_dim,
            Self::EPS,
        )?;
        q = apply_rope_halfsplit_3d(&q, cos, sin, heads, head_dim)?;
        k = apply_rope_halfsplit_3d(&k, cos, sin, kv_heads, head_dim)?;
        let k = repeat_kv_heads_3d(&k, heads, kv_heads, head_dim)?;
        let v = repeat_kv_heads_3d(&v, heads, kv_heads, head_dim)?;
        // The causal SDPA operates on `[seq, heads*head_dim]` (batch 1 encoder).
        let [_, seq, inner] = shape3(&q)?;
        let squeeze = |t: CpuTensor| CpuTensor {
            shape: vec![seq, inner],
            data: t.data,
        };
        let attention = clip_causal_self_attention_with_runtime_context(
            &squeeze(q),
            &squeeze(k),
            &squeeze(v),
            heads,
            runtime_context,
        )?;
        let attention = CpuTensor {
            shape: vec![1, seq, inner],
            data: attention.data,
        };
        let attention = linear_3d_with_runtime_context(
            &attention,
            &self.o_proj.decode()?,
            None,
            runtime_context,
        )?;
        let hidden = residual_add_3d(input, &attention)?;

        let normed2 = rms_norm_3d_with_runtime_context(
            &hidden,
            &CpuTensor {
                shape: vec![self.post_norm.len()],
                data: self.post_norm.clone(),
            },
            Self::EPS,
            runtime_context,
        )?;
        let gate = linear_3d_with_runtime_context(
            &normed2,
            &self.gate_proj.decode()?,
            None,
            runtime_context,
        )?;
        let up = linear_3d_with_runtime_context(
            &normed2,
            &self.up_proj.decode()?,
            None,
            runtime_context,
        )?;
        let mut swish = CpuTensor::zeros(&gate.shape);
        for (idx, slot) in swish.data.iter_mut().enumerate() {
            let g = gate.data[idx];
            *slot = (g / (1.0 + (-g).exp())) * up.data[idx];
        }
        let ff = linear_3d_with_runtime_context(
            &swish,
            &self.down_proj.decode()?,
            None,
            runtime_context,
        )?;
        residual_add_3d(&hidden, &ff)
    }
}

/// Native Qwen3-VL text encoder (`language_model` tower). Prefill-only; returns
/// the hidden states at `select_layers` for `NativeTextFusion`.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct NativeQwen3TextEncoder {
    embed_tokens: CpuTensor,
    layers: Vec<Qwen3EncoderLayer>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    rope_theta: f32,
}

#[allow(dead_code)]
impl NativeQwen3TextEncoder {
    pub(crate) fn from_hfq(
        hfq: &HfqFile,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> DiffusionResult<Option<Self>> {
        let embed_entry = format!("{prefix}.embed_tokens.weight");
        if hfq.find_tensor_info(&embed_entry).is_none() {
            return Ok(None);
        }
        let embed_tokens = CpuTensor::from_hfq(hfq, &embed_entry)?;
        let [_, hidden] = shape2(&embed_tokens)?;
        let mut layers = Vec::new();
        let mut idx = 0;
        while hfq
            .find_tensor_info(&format!("{prefix}.layers.{idx}.input_layernorm.weight"))
            .is_some()
        {
            layers.push(Qwen3EncoderLayer::from_hfq(
                hfq,
                &format!("{prefix}.layers.{idx}"),
            )?);
            idx += 1;
        }
        Ok(Some(Self {
            embed_tokens,
            layers,
            heads,
            kv_heads,
            head_dim,
            hidden,
            rope_theta,
        }))
    }

    /// Run the encoder over `token_ids` and return the hidden state after each
    /// layer index in `select_layers` (1-based over the layer stack), each
    /// `[1, seq, hidden]` — the stack `NativeTextFusion::encode_from_layers`
    /// expects.
    pub(crate) fn encode(
        &self,
        token_ids: &[u32],
        select_layers: &[usize],
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<Vec<CpuTensor>> {
        let seq = token_ids.len();
        if seq == 0 {
            return Err(DiffusionError::InvalidRequest(
                "text encoder requires at least one token".to_string(),
            ));
        }
        let mut hidden = CpuTensor::zeros(&[1, seq, self.hidden]);
        for (pos, &token) in token_ids.iter().enumerate() {
            let row = token as usize * self.hidden;
            if row + self.hidden > self.embed_tokens.data.len() {
                return Err(DiffusionError::InvalidRequest(format!(
                    "token id {token} out of vocab range"
                )));
            }
            hidden.data[pos * self.hidden..(pos + 1) * self.hidden]
                .copy_from_slice(&self.embed_tokens.data[row..row + self.hidden]);
        }
        let (cos, sin) = rope_1d_cos_sin(seq, self.head_dim, self.rope_theta);
        let mut captured = Vec::with_capacity(select_layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_with_runtime_context(
                &hidden,
                &cos,
                &sin,
                self.heads,
                self.kv_heads,
                self.head_dim,
                runtime_context,
            )?;
            if select_layers.contains(&(index + 1)) {
                captured.push(hidden.clone());
            }
        }
        Ok(captured)
    }
}

/// Krea2 text conditioning: the Qwen3-VL encoder + text_fusion, producing the
/// `[1, seq, text_hidden]` conditioning the DiT denoiser's `txt_in` consumes.
/// This is the object the pipeline drives for a `Krea2Pipeline` (tokenize ->
/// this -> external-conditioning seam -> denoiser).
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct Krea2TextConditioner {
    encoder: NativeQwen3TextEncoder,
    fusion: NativeTextFusion,
    select_layers: Vec<usize>,
}

#[allow(dead_code)]
impl Krea2TextConditioner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_hfq(
        hfq: &HfqFile,
        encoder_prefix: &str,
        encoder_heads: usize,
        encoder_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        fusion_heads: usize,
        select_layers: Vec<usize>,
    ) -> DiffusionResult<Option<Self>> {
        let Some(encoder) = NativeQwen3TextEncoder::from_hfq(
            hfq,
            encoder_prefix,
            encoder_heads,
            encoder_kv_heads,
            head_dim,
            rope_theta,
        )?
        else {
            return Ok(None);
        };
        let Some(fusion) = NativeTextFusion::from_hfq(hfq, fusion_heads)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            encoder,
            fusion,
            select_layers,
        }))
    }

    /// Encode already-tokenized prompt ids into DiT conditioning.
    pub(crate) fn conditioning_from_token_ids(
        &self,
        token_ids: &[u32],
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<CpuTensor> {
        let layers = self
            .encoder
            .encode(token_ids, &self.select_layers, runtime_context)?;
        // Parity-debug: dump the selected encoder hidden states + the fused
        // conditioning when HIPFIRE_DIFFUSION_DUMP_DIR is set.
        for (index, layer) in layers.iter().enumerate() {
            let sel = self.select_layers.get(index).copied().unwrap_or(index);
            dump_debug_tensor(&format!("encoder_layer_{sel}"), layer);
        }
        let fused = self
            .fusion
            .encode_from_layers_with_runtime_context(&layers, runtime_context)?;
        dump_debug_tensor("text_fusion_out", &fused);
        Ok(fused)
    }
}

#[cfg(test)]
mod gqa_tests {
    use super::*;

    #[test]
    fn repeat_kv_heads_is_identity_when_kv_equals_q() {
        // Multi-head attention (QwenImage): kv_heads == heads must not change data.
        let input = CpuTensor {
            shape: vec![1, 2, 4],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let out = repeat_kv_heads_3d(&input, 2, 2, 2).unwrap();
        assert_eq!(out.shape, input.shape);
        assert_eq!(out.data, input.data);
    }

    #[test]
    fn repeat_kv_heads_expands_grouped_query_heads() {
        // GQA: 2 kv heads -> 4 query heads (group = 2), head_dim = 2, seq = 1.
        // kv head 0 = [10,11], kv head 1 = [20,21]. Query heads 0,1 read kv 0;
        // query heads 2,3 read kv 1 (PyTorch repeat_kv ordering).
        let input = CpuTensor {
            shape: vec![1, 1, 4],
            data: vec![10.0, 11.0, 20.0, 21.0],
        };
        let out = repeat_kv_heads_3d(&input, 4, 2, 2).unwrap();
        assert_eq!(out.shape, vec![1, 1, 8]);
        assert_eq!(
            out.data,
            vec![10.0, 11.0, 10.0, 11.0, 20.0, 21.0, 20.0, 21.0]
        );
    }

    #[test]
    fn repeat_kv_heads_rejects_incompatible_shapes() {
        let input = CpuTensor {
            shape: vec![1, 1, 4],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        // width 4 with head_dim 2 => kv_heads 2; target heads 3 is not a multiple.
        assert!(repeat_kv_heads_3d(&input, 3, 2, 2).is_err());
        // width 4 not divisible by head_dim 3.
        assert!(repeat_kv_heads_3d(&input, 2, 2, 3).is_err());
    }
}
