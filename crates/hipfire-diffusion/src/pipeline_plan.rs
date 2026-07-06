// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `DiffusionPipeline::plan` — extracted from the pipeline
//! god-impl in lib.rs into its own `impl` block (3.8 Part 2). Uses `super::*`
//! so the pipeline's helpers + types resolve unchanged; the struct's fields are
//! pub(crate) so this block can read them.

use super::*;

impl DiffusionPipeline {
    pub(crate) fn native_runtime(&self) -> DiffusionResult<&NativeDiffusionRuntime> {
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

    pub(crate) fn prepare_run_plan_with_runtime_context(
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
        // Packed image token count drives the FlowMatchEuler dynamic-shift mu.
        // The transformer patchifies the latent by patch_size, so the token grid
        // is (latent_h / p) x (latent_w / p). Passing this (vs the config base
        // seq len) gives the correct, resolution-dependent sigma schedule.
        let patch_size = self
            .config
            .transformer
            .as_ref()
            .and_then(|t| t.patch_size)
            .unwrap_or(1)
            .max(1);
        let image_seq_len =
            (latent_shape.height / patch_size).max(1) * (latent_shape.width / patch_size).max(1);
        let schedule = DiffusionSchedule::from_config_with_image_seq_len(
            &scheduler_config,
            request.steps,
            Some(image_seq_len),
        )?;
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

    /// Krea2 conditioning: tokenize each prompt with the Qwen2 tokenizer, run the
    /// Qwen3-VL encoder + text_fusion, and stack the per-prompt `[1, seq, hidden]`
    /// conditioning into the batch the transformer denoiser consumes as its text
    /// embeddings. (Numerically unvalidated — see encoder/DiT parity caveats.)
    fn prepare_krea2_conditioning_batch(
        &self,
        request: &DiffusionBatchRequest,
        runtime_context: &mut DiffusionGenerationRuntimeContext,
    ) -> DiffusionResult<DiffusionConditioningBatch> {
        let runtime = self.native_runtime()?;
        let missing =
            || DiffusionError::BackendUnavailable("Krea2 text conditioner unavailable".to_string());
        let cfg_is_identity = classifier_free_guidance_is_identity(request.cfg_scale);
        let mut prompt_conds = Vec::with_capacity(request.prompts.len());
        let mut negative_conds = Vec::with_capacity(request.prompts.len());
        for prompt in &request.prompts {
            let cond = runtime
                .krea2_conditioning_from_prompt(&prompt.prompt, runtime_context)?
                .ok_or_else(missing)?;
            if cfg_is_identity {
                negative_conds.push(cond.clone());
            } else {
                negative_conds.push(
                    runtime
                        .krea2_conditioning_from_prompt(&prompt.negative_prompt, runtime_context)?
                        .ok_or_else(missing)?,
                );
            }
            prompt_conds.push(cond);
        }
        let empty_tokens = vec![Vec::new(); request.prompts.len()];
        Ok(DiffusionConditioningBatch {
            prompt_tokens: empty_tokens.clone(),
            negative_tokens: empty_tokens,
            prompt_tokens_2: None,
            negative_tokens_2: None,
            prompt_embeddings: Some(stack_krea2_conditioning(&prompt_conds)?),
            negative_embeddings: Some(stack_krea2_conditioning(&negative_conds)?),
            prompt_embeddings_2: None,
            negative_embeddings_2: None,
            prompt_cross_attention_embeddings: None,
            negative_cross_attention_embeddings: None,
            prompt_attention_mask: None,
            negative_attention_mask: None,
            prompt_pooled_embeddings: None,
            negative_pooled_embeddings: None,
        })
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
        // Krea2: the Qwen3-VL encoder + text_fusion produce the DiT conditioning
        // in-runtime (no CLIP). NOTE: numerically unvalidated against a diffusers
        // reference — see the parity caveats on the encoder/DiT.
        if self
            .native_runtime
            .as_ref()
            .is_some_and(|runtime| runtime.text_conditioner.is_some())
        {
            return self.prepare_krea2_conditioning_batch(request, runtime_context);
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
