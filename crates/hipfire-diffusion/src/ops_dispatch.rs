// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Runtime-context op dispatch.
//!
//! The `*_with_runtime_context` family: each op runs the CPU-reference path
//! (hipfire-cpu ops, re-exported at the crate root) or offloads to the ROCm GPU
//! (`gpu_ops::*_hip_on_gpu`) via `DiffusionGenerationRuntimeContext`. Extracted
//! from lib.rs (3.8 Part 2). Uses `super::*` for the crate's types/helpers.

use super::*;
use crate::gpu_ops::*;

pub(crate) fn scale_model_input_with_runtime_context(
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

pub(crate) fn cfg_guidance_with_runtime_context(
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

pub(crate) fn scheduler_step_with_runtime_context(
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

pub(crate) fn maybe_center_unet_input_with_runtime_context(
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

pub(crate) fn timestep_embedding_with_runtime_context(
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

pub(crate) fn scale_tensor_with_runtime_context(
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

pub(crate) fn linear_optional_bias_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        calib_observe_linear(weight, input);
        return linear_optional_bias(input, weight, bias).map_err(Into::into);
    };
    {
        runtime_context.with_rocm_gpu_weighted(|gpu, cache| {
            linear_optional_bias_hip_on_gpu(gpu, cache, input, weight, bias)
        })
    }
}

/// Fold a linear layer's input activations into the calibration accumulators
/// (no-op unless a calibration run is armed). `weight` is `[out, in]`; the input
/// is row-major `[rows, in]`.
pub(crate) fn calib_observe_linear(weight: &CpuTensor, input: &CpuTensor) {
    if !quant_calib::calib_active() {
        return;
    }
    let Some(&k) = weight.shape.get(1) else {
        return;
    };
    if k == 0 || input.data.is_empty() || input.data.len() % k != 0 {
        return;
    }
    let rows = input.data.len() / k;
    quant_calib::calib_observe_matrix(weight.data.as_ptr() as usize, &input.data, rows, k);
}

pub(crate) fn linear_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    if runtime_context.rocm_device_id().is_none() {
        calib_observe_linear(weight, input);
        return linear(input, weight, bias).map_err(Into::into);
    }
    linear_optional_bias_with_runtime_context(input, weight, Some(bias), runtime_context)
}

pub(crate) fn silu_with_runtime_context(
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

pub(crate) fn quick_gelu_with_runtime_context(
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

pub(crate) fn clip_token_position_embeddings(
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

pub(crate) fn clip_token_position_embeddings_with_runtime_context(
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

pub(crate) fn tensor_add_with_runtime_context(
    a: &CpuTensor,
    b: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return tensor_add(a, b).map_err(Into::into);
    };
    {
        runtime_context.with_rocm_gpu(|gpu| tensor_add_hip_on_gpu(gpu, a, b))
    }
}

pub(crate) fn concat_last_dim_2d_with_runtime_context(
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

pub(crate) fn concat_last_dim_3d_with_runtime_context(
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

pub(crate) fn conv2d_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    padding: usize,
    stride: usize,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return conv2d_nchw_with_stride(input, weight, bias, padding, stride).map_err(Into::into);
    };
    {
        runtime_context.with_rocm_gpu_weighted(|gpu, cache| {
            conv2d_nchw_hip_on_gpu(gpu, cache, input, weight, bias, padding, stride)
        })
    }
}

pub(crate) fn group_norm_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    groups: usize,
    eps: f32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return group_norm_nchw(input, weight, bias, groups, eps).map_err(Into::into);
    };
    {
        runtime_context
            .with_rocm_gpu(|gpu| group_norm_nchw_hip_on_gpu(gpu, input, weight, bias, groups, eps))
    }
}

pub(crate) fn add_channel_bias_nchw_with_runtime_context(
    input: &mut CpuTensor,
    bias: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<()> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return add_channel_bias_nchw(input, bias).map_err(Into::into);
    };
    {
        *input = runtime_context
            .with_rocm_gpu(|gpu| add_channel_bias_nchw_hip_on_gpu(gpu, input, bias))?;
        Ok(())
    }
}

pub(crate) fn concat_channels_nchw_with_runtime_context(
    a: &CpuTensor,
    b: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return concat_channels_nchw(a, b).map_err(Into::into);
    };
    {
        runtime_context.with_rocm_gpu(|gpu| concat_channels_nchw_hip_on_gpu(gpu, a, b))
    }
}

pub(crate) fn upsample_nearest2d_nchw_with_runtime_context(
    input: &CpuTensor,
    scale: usize,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return upsample_nearest2d_nchw(input, scale).map_err(Into::into);
    };
    {
        runtime_context.with_rocm_gpu(|gpu| upsample_nearest2d_nchw_hip_on_gpu(gpu, input, scale))
    }
}

pub(crate) fn nchw_to_bsc_with_runtime_context(
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

pub(crate) fn bsc_to_nchw_with_runtime_context(
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

pub(crate) fn linear_3d_with_runtime_context(
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

/// Linear over a source-reference [`ResidentWeight`]: on the GPU path the weight
/// is uploaded once (bf16, keyed by name) and reused across forward steps —
/// avoiding the per-step decode-to-f32 + re-upload that a decoded `CpuTensor`
/// weight incurs. Falls back to decoding + the CPU-reference linear when no GPU
/// is bound.
pub(crate) fn linear_optional_bias_resident_with_runtime_context(
    input: &CpuTensor,
    weight: &ResidentWeight,
    bias: Option<&CpuTensor>,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    if runtime_context.rocm_device_id().is_none() {
        let decoded = weight.decode()?;
        calib_observe_linear(&decoded, input);
        return linear_optional_bias(input, &decoded, bias).map_err(Into::into);
    }
    runtime_context.with_rocm_gpu_weighted(|gpu, cache| {
        linear_resident_weight_hip_on_gpu(gpu, cache, input, weight, bias)
    })
}

/// 3-D (`[batch, seq, in]`) linear over a source-reference [`ResidentWeight`].
pub(crate) fn linear_3d_resident_with_runtime_context(
    input: &CpuTensor,
    weight: &ResidentWeight,
    bias: Option<&CpuTensor>,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    if runtime_context.rocm_device_id().is_none() {
        return linear_3d(input, &weight.decode()?, bias);
    }
    let [batch, seq, in_features] = shape3(input)?;
    let flat = CpuTensor {
        shape: vec![batch * seq, in_features],
        data: input.data.clone(),
    };
    let out =
        linear_optional_bias_resident_with_runtime_context(&flat, weight, bias, runtime_context)?;
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

pub(crate) fn layer_norm_with_runtime_context(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    eps: f32,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return layer_norm(input, weight, bias, eps).map_err(Into::into);
    };
    {
        runtime_context.with_rocm_gpu(|gpu| layer_norm_hip_on_gpu(gpu, input, weight, bias, eps))
    }
}

pub(crate) fn layer_norm_3d_with_runtime_context(
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

pub(crate) fn scaled_dot_product_attention_with_runtime_context(
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

pub(crate) fn scaled_dot_product_attention_with_key_mask_and_runtime_context(
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

pub(crate) fn clip_causal_self_attention_with_runtime_context(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    n_heads: usize,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return clip_causal_self_attention(q, k, v, n_heads).map_err(Into::into);
    };
    {
        runtime_context
            .with_rocm_gpu(|gpu| clip_causal_self_attention_hip_on_gpu(gpu, q, k, v, n_heads))
    }
}

pub(crate) fn geglu_gate_3d_with_runtime_context(
    projected: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<CpuTensor> {
    let Some(_device_id) = runtime_context.rocm_device_id() else {
        return geglu_gate_3d(projected).map_err(Into::into);
    };
    {
        runtime_context.with_rocm_gpu(|gpu| geglu_gate_3d_hip_on_gpu(gpu, projected))
    }
}
