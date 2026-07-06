// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Denoise / CFG driver + conditioning validation.
//!
//! The `denoise_latents_with_cfg*` loop, seed/subseed latent init, classifier-
//! free guidance, batched-CFG concat/split, layer-policy selection, and the
//! inpaint/masked/text conditioning validators. Drives the op-dispatch layer
//! ([`super::ops_dispatch`]). Extracted from lib.rs (3.8 Part 2).

use super::*;

pub(crate) fn seeded_latents_for_request(
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

pub(crate) fn blend_subseed_latents(
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

pub(crate) fn denoise_latents_with_cfg_progress_and_runtime_options(
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

/// Concatenate two tensors along the leading (batch) dimension. The trailing
/// dims must match; the data is simply appended (both are batch-major
/// row-major). Used to fuse the CFG uncond/cond passes into one batch-2N forward.
pub(crate) fn concat_batch_dim(a: &CpuTensor, b: &CpuTensor) -> DiffusionResult<CpuTensor> {
    if a.shape.len() != b.shape.len() || a.shape.is_empty() || a.shape[1..] != b.shape[1..] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate along batch: shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let mut shape = a.shape.clone();
    shape[0] = a.shape[0] + b.shape[0];
    let mut data = Vec::with_capacity(a.data.len() + b.data.len());
    data.extend_from_slice(&a.data);
    data.extend_from_slice(&b.data);
    Ok(CpuTensor { shape, data })
}

/// Split a batched CFG prediction `[2N, ...]` back into the positive `[0..N]`
/// and negative `[N..2N]` halves.
pub(crate) fn split_batched_cfg_prediction(batched: &CpuTensor) -> DiffusionResult<(CpuTensor, CpuTensor)> {
    if batched.shape.first().copied().unwrap_or(0) % 2 != 0 || batched.shape.is_empty() {
        return Err(DiffusionError::InvalidMetadata(format!(
            "batched CFG prediction must have an even leading dim, got {:?}",
            batched.shape
        )));
    }
    let mut half_shape = batched.shape.clone();
    half_shape[0] = batched.shape[0] / 2;
    let half = batched.data.len() / 2;
    let positive = CpuTensor {
        shape: half_shape.clone(),
        data: batched.data[..half].to_vec(),
    };
    let negative = CpuTensor {
        shape: half_shape,
        data: batched.data[half..].to_vec(),
    };
    Ok((positive, negative))
}

/// Resolve the resident-linear activation precision for denoise `step` of
/// `total`, from the progressive schedule. Env-driven (opt-in; default is all
/// F16, so behavior is unchanged unless set):
///   `HIPFIRE_DIFFUSION_W4A4_UNTIL` — fraction of steps (0..1) to run at W4A4
///   `HIPFIRE_DIFFUSION_W4A8_UNTIL` — fraction (0..1) to run at W4A8 (after W4A4)
/// e.g. `W4A4_UNTIL=0.5 W4A8_UNTIL=0.8` → first 50% W4A4, next 30% W4A8, last
/// 20% F16. Only affects linears with `in % 256 == 0`; others stay F16.
pub(crate) fn linear_precision_for_step(step: usize, total: usize) -> LinearPrecision {
    let frac_env = |name: &str| -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0)
    };
    let w4a4_until = frac_env("HIPFIRE_DIFFUSION_W4A4_UNTIL");
    let w4a8_until = frac_env("HIPFIRE_DIFFUSION_W4A8_UNTIL").max(w4a4_until);
    let frac = if total <= 1 {
        0.0
    } else {
        step as f32 / total as f32
    };
    if frac < w4a4_until {
        LinearPrecision::W4A4
    } else if frac < w4a8_until {
        LinearPrecision::W4A8
    } else {
        LinearPrecision::F16
    }
}

/// Configure the per-layer precision policy on the weight cache from env (opt-in;
/// `HIPFIRE_DIFFUSION_LAYER_STRIDE=0` = off, the default). When active, every
/// `STRIDE`-th resident linear runs `RUNG` (default W4A4), except the first
/// `SKIP_FIRST` and last `SKIP_LAST` linears (kept F16). This is orthogonal to the
/// per-step schedule and overrides it when `STRIDE > 0`.
///   `HIPFIRE_DIFFUSION_LAYER_STRIDE`     — N (every Nth linear; 0=off)
///   `HIPFIRE_DIFFUSION_LAYER_SKIP_FIRST` — keep the first X linears F16
///   `HIPFIRE_DIFFUSION_LAYER_SKIP_LAST`  — keep the last Y linears F16 (from step 1)
///   `HIPFIRE_DIFFUSION_LAYER_RUNG`       — w4a4 | w4a8 | w4a16 (default w4a4)
pub(crate) fn configure_layer_policy(cache: &mut RocmWeightCache) {
    let usize_env = |name: &str| -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    };
    cache.layer_stride = usize_env("HIPFIRE_DIFFUSION_LAYER_STRIDE");
    cache.layer_skip_first = usize_env("HIPFIRE_DIFFUSION_LAYER_SKIP_FIRST");
    cache.layer_skip_last = usize_env("HIPFIRE_DIFFUSION_LAYER_SKIP_LAST");
    cache.layer_rung = match std::env::var("HIPFIRE_DIFFUSION_LAYER_RUNG")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "w4a8" => LinearPrecision::W4A8,
        "w4a16" => LinearPrecision::W4A16,
        _ => LinearPrecision::W4A4,
    };
    cache.linear_total = 0;
}

pub(crate) fn denoise_latents_with_cfg_progress_and_runtime_context(
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
    configure_layer_policy(&mut runtime_context.rocm_weights);
    let total_steps = schedule.timesteps.len();
    for step in 0..total_steps {
        // Progressive precision schedule: pick the resident-linear activation
        // precision for this step (early/high-noise steps tolerate cheaper rungs).
        // `linear_total` carries the previous forward's linear count (for the
        // per-layer skip_last); reset the per-forward index before this step.
        if step > 0 {
            runtime_context.rocm_weights.linear_total = runtime_context.rocm_weights.linear_index;
        }
        runtime_context.rocm_weights.linear_index = 0;
        runtime_context.rocm_weights.linear_precision =
            linear_precision_for_step(step, total_steps);
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
        // Batched CFG: when guidance is active and neither pass needs SDXL
        // conditioning (and the attention masks are batchable), run the uncond
        // and cond passes as one batch-2N forward — `[positive; negative]` —
        // instead of two sequential forwards. Halves launches and feeds bigger
        // GEMMs. SDXL / mixed-mask cases fall back to the sequential path.
        let masks_batchable =
            positive_attention_mask.is_none() == negative_attention_mask.is_none();
        let batched_cfg = !cfg_is_identity
            && positive_sdxl_conditioning.is_none()
            && negative_sdxl_conditioning.is_none()
            && masks_batchable;
        let guided = if cfg_is_identity {
            let positive_pred = predict_noise(
                &model_sample,
                &timesteps,
                positive_embeddings,
                positive_attention_mask,
                positive_sdxl_conditioning,
                runtime_context,
            )?;
            validate_noise_prediction(&latents, &positive_pred)?;
            positive_pred
        } else if batched_cfg {
            let batched_sample = concat_batch_dim(&model_sample, &model_sample)?;
            let mut batched_timesteps = timesteps.clone();
            batched_timesteps.extend_from_slice(&timesteps);
            let batched_encoder = concat_batch_dim(positive_embeddings, negative_embeddings)?;
            let batched_mask = match (positive_attention_mask, negative_attention_mask) {
                (Some(p), Some(n)) => Some(concat_batch_dim(p, n)?),
                _ => None,
            };
            let batched_pred = predict_noise(
                &batched_sample,
                &batched_timesteps,
                &batched_encoder,
                batched_mask.as_ref(),
                None,
                runtime_context,
            )?;
            let (positive_pred, negative_pred) = split_batched_cfg_prediction(&batched_pred)?;
            validate_noise_prediction(&latents, &positive_pred)?;
            validate_noise_prediction(&latents, &negative_pred)?;
            let (guided, guidance_runtime_kind) = cfg_guidance_with_runtime_context(
                &negative_pred,
                &positive_pred,
                cfg_scale,
                runtime_context,
            )?;
            runtime_kind = merge_runtime_kind(runtime_kind, guidance_runtime_kind);
            guided
        } else {
            let positive_pred = predict_noise(
                &model_sample,
                &timesteps,
                positive_embeddings,
                positive_attention_mask,
                positive_sdxl_conditioning,
                runtime_context,
            )?;
            validate_noise_prediction(&latents, &positive_pred)?;
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
        // Debug hook: HIPFIRE_DUMP_VELOCITY=<dir> writes the per-step model
        // velocity (the flow-match prediction, before the scheduler integrates
        // it) as <dir>/vel_<step>.bin (4x u32 LE header [b,c,h,w] + f32 data).
        // Lets us see whether a single forward already carries the token-grid
        // (attention not mixing) or whether the grid only builds up over steps.
        if let Ok(dir) = std::env::var("HIPFIRE_DUMP_VELOCITY") {
            if !dir.is_empty() {
                let mut bytes = Vec::with_capacity(16 + guided.data.len() * 4);
                for dim in [latents.batch, latents.channels, latents.height, latents.width] {
                    bytes.extend_from_slice(&(dim as u32).to_le_bytes());
                }
                for v in &guided.data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                let path = format!("{dir}/vel_{step}.bin");
                match std::fs::write(&path, &bytes) {
                    Ok(()) => eprintln!("[dump] step {step} velocity -> {path}"),
                    Err(e) => eprintln!("[dump] velocity write to {path} failed: {e}"),
                }
            }
        }
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

pub(crate) fn validate_conditioning_for_latents(
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

pub(crate) fn validate_text_attention_mask(
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

pub(crate) fn qwen_joint_key_mask(
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

pub(crate) fn validate_inpaint_denoise_conditioning(
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

pub(crate) fn append_inpaint_conditioning(
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

pub(crate) fn validate_masked_denoise_reference(
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
pub(crate) fn apply_masked_denoise_reference(
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

pub(crate) fn apply_masked_denoise_reference_with_runtime_context(
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

pub(crate) fn validate_noise_prediction(latents: &LatentBatch, noise: &CpuTensor) -> DiffusionResult<()> {
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

pub(crate) fn cfg_guidance(
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

pub(crate) fn classifier_free_guidance_is_identity(cfg_scale: f32) -> bool {
    (cfg_scale - 1.0).abs() <= f32::EPSILON
}

/// Stack per-prompt Krea2 conditioning `[1, seq, hidden]` tensors into a
/// `[n_prompts, seq, hidden]` batch. Prompts must share `seq`/`hidden` (batching
/// unequal prompt lengths needs padding + attention masks — not yet supported).
pub(crate) fn stack_krea2_conditioning(items: &[CpuTensor]) -> DiffusionResult<CpuTensor> {
    let first = items.first().ok_or_else(|| {
        DiffusionError::InvalidRequest("Krea2 conditioning batch is empty".to_string())
    })?;
    let [_, seq, hidden] = shape3(first)?;
    let mut data = Vec::with_capacity(items.len() * seq * hidden);
    for item in items {
        let [batch, item_seq, item_hidden] = shape3(item)?;
        if batch != 1 || item_seq != seq || item_hidden != hidden {
            return Err(DiffusionError::InvalidRequest(format!(
                "Krea2 conditioning batch requires equal prompt lengths; got {:?} vs [1, {seq}, {hidden}]",
                item.shape
            )));
        }
        data.extend_from_slice(&item.data);
    }
    Ok(CpuTensor {
        shape: vec![items.len(), seq, hidden],
        data,
    })
}
