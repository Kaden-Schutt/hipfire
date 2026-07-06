#![allow(unused_imports)]
use super::*;
use std::collections::BTreeMap;
use std::path::PathBuf;
// Import tooling now lives in the offline hipfire-diffusion-coexist crate.
use hipfire_diffusion_coexist::{
    import_diffusers_to_hfq, import_realesrgan_to_hfq, ldm_unet_native_tensor_name,
    ldm_vae_native_tensor_name, parse_pytorch_state_dict, pytorch_tensor_is_contiguous,
    reorder_pytorch_storage_to_contiguous, DiffusersImportOptions, RealesrganImportOptions,
};
use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
use std::fs;
use super::*;

#[test]
fn hip_memory_plan_accounts_for_diffusion_buffers() {
    let mut config = StableDiffusionConfig {
        pipeline_class: "StableDiffusionPipeline".into(),
        text_encoder: TextEncoderConfig {
            hidden_size: Some(32),
            max_position_embeddings: Some(4),
            ..TextEncoderConfig::default()
        },
        text_encoder_2: None,
        unet: UnetConfig {
            in_channels: Some(9),
            cross_attention_dim: Some(48),
            ..UnetConfig::default()
        },
        transformer: None,
        vae: VaeConfig::default(),
        scheduler: SchedulerConfig::default(),
        latent_channels: 4,
        latent_height: None,
        latent_width: None,
        vae_scale_factor: 8,
    };
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![
            DiffusionPrompt {
                prompt: "a".into(),
                negative_prompt: String::new(),
                seed: 1,
                subseed: None,
            },
            DiffusionPrompt {
                prompt: "b".into(),
                negative_prompt: String::new(),
                seed: 2,
                subseed: None,
            },
        ],
        width: 64,
        height: 32,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: 2,
        cfg_scale: 1.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };

    let plan = diffusion_hip_memory_plan(&config, &request).unwrap();

    assert_eq!(
        plan.latent_shape,
        DiffusionLatentShape {
            batch: 2,
            channels: 4,
            height: 4,
            width: 8
        }
    );
    assert_eq!(plan.latent_bytes, 2 * 4 * 4 * 8 * 4);
    assert_eq!(plan.denoise_input_bytes, 2 * 9 * 4 * 8 * 4);
    assert_eq!(plan.conditioning_bytes, 2 * 2 * 1 * 4 * 48 * 4);
    assert_eq!(plan.vae_decode_bytes, plan.latent_bytes);
    assert_eq!(plan.rgb_bytes, 2 * 32 * 64 * 3);
    assert_eq!(
        plan.total_device_bytes,
        plan.latent_bytes
            + plan.denoise_input_bytes
            + plan.conditioning_bytes
            + plan.vae_decode_bytes
            + plan.rgb_bytes
            + plan.scheduler_scratch_bytes
    );

    config.text_encoder_2 = Some(TextEncoderConfig::default());
    let sdxl_plan = diffusion_hip_memory_plan(&config, &request).unwrap();
    assert_eq!(sdxl_plan.conditioning_bytes, plan.conditioning_bytes * 2);
}

#[test]
fn hip_memory_plan_uses_transformer_denoiser_dimensions() {
    let config = StableDiffusionConfig {
        pipeline_class: "QwenImagePipeline".into(),
        text_encoder: TextEncoderConfig {
            hidden_size: Some(32),
            max_position_embeddings: Some(4),
            ..TextEncoderConfig::default()
        },
        text_encoder_2: None,
        unet: UnetConfig::default(),
        transformer: Some(TransformerDenoiserConfig {
            class_name: "QwenImageTransformer2DModel".into(),
            in_channels: Some(64),
            out_channels: Some(16),
            patch_size: Some(2),
            caption_projection_dim: Some(128),
            ..TransformerDenoiserConfig::default()
        }),
        vae: VaeConfig {
            z_dim: Some(16),
            ..VaeConfig::default()
        },
        scheduler: SchedulerConfig {
            class_name: "FlowMatchEulerDiscreteScheduler".into(),
            ..SchedulerConfig::default()
        },
        latent_channels: 16,
        latent_height: None,
        latent_width: None,
        vae_scale_factor: 8,
    };
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![DiffusionPrompt {
            prompt: "a".into(),
            negative_prompt: String::new(),
            seed: 1,
            subseed: None,
        }],
        width: 64,
        height: 64,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: 2,
        cfg_scale: 1.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };

    let plan = diffusion_hip_memory_plan(&config, &request).unwrap();

    assert_eq!(
        plan.latent_shape,
        DiffusionLatentShape {
            batch: 1,
            channels: 16,
            height: 8,
            width: 8
        }
    );
    assert_eq!(plan.latent_bytes, 16 * 8 * 8 * 4);
    assert_eq!(
        plan.transformer_denoiser,
        Some(DiffusionTransformerDenoiserPlan {
            representation: "patch_tokens".into(),
            batch: 1,
            sequence_length: 16,
            token_width: 64,
            patch_size: 2,
            latent_height: 8,
            latent_width: 8,
            patch_height: 4,
            patch_width: 4,
            output_channels: 16,
        })
    );
    assert_eq!(plan.denoise_input_bytes, 16 * 64 * 4);
    assert_eq!(plan.conditioning_bytes, 2 * 4 * 128 * 4);
}

#[test]
fn hip_memory_plan_rejects_transformer_patch_misalignment() {
    let config = StableDiffusionConfig {
        pipeline_class: "QwenImagePipeline".into(),
        text_encoder: TextEncoderConfig::default(),
        text_encoder_2: None,
        unet: UnetConfig::default(),
        transformer: Some(TransformerDenoiserConfig {
            class_name: "QwenImageTransformer2DModel".into(),
            in_channels: Some(64),
            out_channels: Some(16),
            patch_size: Some(2),
            ..TransformerDenoiserConfig::default()
        }),
        vae: VaeConfig {
            z_dim: Some(16),
            ..VaeConfig::default()
        },
        scheduler: SchedulerConfig::default(),
        latent_channels: 16,
        latent_height: None,
        latent_width: None,
        vae_scale_factor: 8,
    };
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![DiffusionPrompt {
            prompt: "a".into(),
            negative_prompt: String::new(),
            seed: 1,
            subseed: None,
        }],
        width: 72,
        height: 64,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: 2,
        cfg_scale: 1.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };

    let error = diffusion_hip_memory_plan(&config, &request)
        .unwrap_err()
        .to_string();
    assert!(error.contains("patch_size 2"));
}

#[test]
fn hip_memory_plan_rejects_invalid_latent_dimensions() {
    let config = StableDiffusionConfig {
        pipeline_class: "StableDiffusionPipeline".into(),
        text_encoder: TextEncoderConfig::default(),
        text_encoder_2: None,
        unet: UnetConfig::default(),
        transformer: None,
        vae: VaeConfig::default(),
        scheduler: SchedulerConfig::default(),
        latent_channels: 4,
        latent_height: None,
        latent_width: None,
        vae_scale_factor: 8,
    };
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![DiffusionPrompt {
            prompt: "a".into(),
            negative_prompt: String::new(),
            seed: 1,
            subseed: None,
        }],
        width: 63,
        height: 64,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: 1,
        cfg_scale: 1.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };

    let error = diffusion_hip_memory_plan(&config, &request)
        .unwrap_err()
        .to_string();
    assert!(error.contains("VAE scale factor 8"));
}

/// Phase 3: the flash-attention kernel (online softmax, no seq×seq matrix)
/// must match the naive SDPA / CPU reference. F32 throughout, so the tolerance
/// is tight. Covers self-attn, cross-attn (q_seq != k_seq), a head_dim that is
/// not a multiple of the wave width, and a single-head VAE-style shape.
#[test]
fn flash_attention_resident_matches_cpu_reference() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for flash attention test: {error}");
            return;
        }
    };
    let fill = |n: usize, seed: f32| -> Vec<f32> {
        (0..n)
            .map(|kk| (((kk as f32 + seed) % 17.0) - 8.0) / 8.0)
            .collect()
    };
    // (batch, heads, head_dim, q_seq, k_seq)
    let cases = [
        (2usize, 2usize, 40usize, 16usize, 16usize), // self-attn, head_dim 40 (>32)
        (2, 2, 24, 10, 5),                           // cross-attn, q_seq != k_seq
        (1, 4, 20, 12, 12),                          // head_dim 20 (< wave width)
        (2, 1, 64, 9, 9),                            // single head, head_dim 64
    ];
    for (idx, (b, heads, head_dim, q_seq, k_seq)) in cases.into_iter().enumerate() {
        let hidden = heads * head_dim;
        let q = CpuTensor {
            shape: vec![b, q_seq, hidden],
            data: fill(b * q_seq * hidden, idx as f32 * 7.0 + 1.0),
        };
        let k = CpuTensor {
            shape: vec![b, k_seq, hidden],
            data: fill(b * k_seq * hidden, idx as f32 * 5.0 + 2.0),
        };
        let v = CpuTensor {
            shape: vec![b, k_seq, hidden],
            data: fill(b * k_seq * hidden, idx as f32 * 3.0 + 3.0),
        };
        let cpu = scaled_dot_product_attention(&q, &k, &v, heads).unwrap();

        let q_gpu = gpu.upload_f32(&q.data, &q.shape).unwrap();
        let k_gpu = gpu.upload_f32(&k.data, &k.shape).unwrap();
        let v_gpu = gpu.upload_f32(&v.data, &v.shape).unwrap();
        let out_gpu =
            scaled_dot_product_attention_resident(&mut gpu, &q_gpu, &k_gpu, &v_gpu, heads).unwrap();
        let hip = download_resident(&mut gpu, &out_gpu).unwrap();
        free_resident(&mut gpu, out_gpu).unwrap();
        free_resident(&mut gpu, q_gpu).unwrap();
        free_resident(&mut gpu, k_gpu).unwrap();
        free_resident(&mut gpu, v_gpu).unwrap();

        assert_eq!(hip.shape, cpu.shape, "case {idx} shape");
        assert!(
            f32_slices_close(&hip.data, &cpu.data, 1e-3),
            "case {idx}: flash {:?} != cpu {:?}",
            &hip.data[..hip.data.len().min(8)],
            &cpu.data[..cpu.data.len().min(8)]
        );
    }
}

#[test]
fn hip_rgb_tensor_to_u8_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for RGB kernel parity test: {error}");
            return;
        }
    };
    let tensor = CpuTensor {
        shape: vec![1, 3, 2, 2],
        data: vec![
            -1.0, 0.0, 1.0, 0.25, -0.5, 0.5, -0.25, 0.75, 1.0, -1.0, 0.1, -0.1,
        ],
    };
    let cpu = rgb_tensor_to_u8(&tensor).unwrap();
    let hip = rgb_tensor_to_u8_hip_on_gpu(&mut gpu, &tensor).unwrap();

    assert_eq!(hip, cpu);
}

#[test]
fn hip_vae_boundary_transforms_match_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for VAE boundary kernel parity test: {error}");
            return;
        }
    };
    let image = RgbImageBatch {
        batch: 2,
        width: 2,
        height: 2,
        data: vec![
            0, 128, 255, 255, 0, 128, 32, 64, 96, 192, 224, 16, 10, 20, 30, 40, 50, 60, 70, 80, 90,
            100, 110, 120,
        ],
    };
    let cpu_tensor = rgb_batch_to_vae_tensor(&image).unwrap();
    let hip_tensor = rgb_batch_to_vae_tensor_hip_on_gpu(&mut gpu, &image).unwrap();

    assert_eq!(hip_tensor.shape, cpu_tensor.shape);
    for (index, (actual, expected)) in hip_tensor.data.iter().zip(&cpu_tensor.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "RGB-to-VAE mismatch at {index}: hip={actual} cpu={expected}"
        );
    }

    let moments = CpuTensor {
        shape: vec![2, 4, 2, 2],
        data: (0..32)
            .map(|idx| idx as f32 / 9.0 - 1.5)
            .collect::<Vec<_>>(),
    };
    let cpu_latents = vae_moments_to_latents(&moments, &VaeLatentNorm::scalar(0.18215)).unwrap();
    let hip_latents = vae_moments_to_latents_hip_on_gpu(&mut gpu, &moments, 0.18215).unwrap();

    assert_eq!(hip_latents.batch, cpu_latents.batch);
    assert_eq!(hip_latents.channels, cpu_latents.channels);
    assert_eq!(hip_latents.height, cpu_latents.height);
    assert_eq!(hip_latents.width, cpu_latents.width);
    for (index, (actual, expected)) in hip_latents.data.iter().zip(&cpu_latents.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "VAE moments-to-latents mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

#[test]
fn hip_inpaint_mask_ops_match_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for inpaint mask kernel parity test: {error}");
            return;
        }
    };
    let image = RgbImageBatch {
        batch: 2,
        width: 4,
        height: 4,
        data: (0..96)
            .map(|idx| ((idx * 19 + 5) % 256) as u8)
            .collect::<Vec<_>>(),
    };
    let mask = RgbImageBatch {
        batch: 2,
        width: 4,
        height: 4,
        data: (0..96)
            .map(|idx| ((idx * 37 + 11) % 256) as u8)
            .collect::<Vec<_>>(),
    };
    let init_latents = LatentBatch {
        batch: 2,
        channels: 2,
        height: 2,
        width: 2,
        data: (0..16)
            .map(|idx| idx as f32 / 7.0 - 1.0)
            .collect::<Vec<_>>(),
    };
    let generated_latents = LatentBatch {
        batch: 2,
        channels: 2,
        height: 2,
        width: 2,
        data: (0..16)
            .map(|idx| (idx as f32 % 9.0 - 4.0) / 3.0)
            .collect::<Vec<_>>(),
    };

    let cpu_weights = latent_mask_weights_from_rgb_batch(&mask, &init_latents).unwrap();
    let hip_weights =
        latent_mask_weights_from_rgb_batch_hip_on_gpu(&mut gpu, &mask, &init_latents).unwrap();
    for (index, (actual, expected)) in hip_weights.iter().zip(&cpu_weights).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "latent mask mismatch at {index}: hip={actual} cpu={expected}"
        );
    }

    let cpu_masked = masked_rgb_batch_for_inpaint(&image, &mask).unwrap();
    let hip_masked = masked_rgb_batch_for_inpaint_hip_on_gpu(&mut gpu, &image, &mask).unwrap();
    assert_eq!(hip_masked, cpu_masked);

    let mut cpu_blend = generated_latents.clone();
    blend_latents_with_mask(&mut cpu_blend, &init_latents, &cpu_weights).unwrap();
    let hip_blend = blend_latents_with_mask_hip_on_gpu(
        &mut gpu,
        &generated_latents,
        &init_latents,
        &cpu_weights,
    )
    .unwrap();
    assert_eq!(hip_blend.batch, cpu_blend.batch);
    assert_eq!(hip_blend.channels, cpu_blend.channels);
    assert_eq!(hip_blend.height, cpu_blend.height);
    assert_eq!(hip_blend.width, cpu_blend.width);
    for (index, (actual, expected)) in hip_blend.data.iter().zip(&cpu_blend.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "latent blend mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

#[test]
fn hip_euler_step_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for Euler kernel parity test: {error}");
            return;
        }
    };
    let sample = vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125];
    let model_output = vec![0.5, -0.5, 0.25, -0.25, 1.5, -1.0, 0.75, -0.125];
    let sigma = 1.0;
    let next_sigma = 0.5;
    for prediction_type in [
        SchedulerPredictionType::Epsilon,
        SchedulerPredictionType::Sample,
        SchedulerPredictionType::VPrediction,
    ] {
        let cpu = sample
            .iter()
            .zip(&model_output)
            .map(|(sample, model_output)| {
                sample
                    + scheduler_derivative(*sample, *model_output, sigma, prediction_type)
                        * (next_sigma - sigma)
            })
            .collect::<Vec<_>>();
        let hip = euler_step_hip_on_gpu(
            &mut gpu,
            &sample,
            &model_output,
            sigma,
            next_sigma,
            prediction_type,
        )
        .unwrap();

        for (index, (actual, expected)) in hip.iter().zip(&cpu).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "{prediction_type:?} mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }
}

#[test]
fn hip_denoise_vector_ops_match_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for denoise vector parity test: {error}");
            return;
        }
    };
    let sample = vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125];
    let scale = 0.5;
    let cpu_scaled = sample
        .iter()
        .map(|sample| sample * scale)
        .collect::<Vec<_>>();
    let hip_scaled = scale_model_input_hip_on_gpu(&mut gpu, &sample, scale).unwrap();
    assert!(f32_slices_close(&hip_scaled, &cpu_scaled, 1e-6));

    let negative = sample;
    let positive = vec![0.5, -0.5, 0.25, -0.25, 1.5, -1.0, 0.75, -0.125];
    let cfg_scale = 7.5;
    let cpu_guided = cfg_guidance(
        &CpuTensor {
            shape: vec![1, 2, 2, 2],
            data: negative.clone(),
        },
        &CpuTensor {
            shape: vec![1, 2, 2, 2],
            data: positive.clone(),
        },
        cfg_scale,
    )
    .unwrap();
    let hip_guided = cfg_guidance_hip_on_gpu(&mut gpu, &negative, &positive, cfg_scale).unwrap();
    assert!(f32_slices_close(&hip_guided, &cpu_guided.data, 1e-6));
}

#[test]
fn hip_denoise_loop_runtime_options_route_vector_stages_when_gpu_is_available() {
    if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
        eprintln!("skip: ROCm GPU unavailable for denoise loop routing test: {error}");
        return;
    }
    let schedule = DiffusionSchedule {
        timesteps: vec![1.0, 0.0],
        sigmas: vec![1.0, 0.5, 0.0],
        prediction_type: SchedulerPredictionType::Epsilon,
        input_scaling: SchedulerInputScaling::Sigma,
        solver: SchedulerSolver::Euler,
        train_timesteps: Vec::new(),
        alpha_t: Vec::new(),
        sigma_t: Vec::new(),
        lambda_t: Vec::new(),
    };
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 2,
        width: 2,
        data: vec![-0.75, -0.25, 0.5, 1.25],
    };
    let positive_embeddings = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![0.2],
    };
    let negative_embeddings = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![-0.1],
    };
    let predict_noise =
        |sample: &CpuTensor,
         _timesteps: &[f32],
         encoder_states: &CpuTensor,
         _attention_mask: Option<&CpuTensor>,
         _sdxl: Option<&SdxlDenoiseConditioning<'_>>,
         _runtime_context: &mut DiffusionGenerationRuntimeContext| {
            let bias = encoder_states.data[0];
            Ok(CpuTensor {
                shape: sample.shape.clone(),
                data: sample
                    .data
                    .iter()
                    .map(|value| value * 0.25 + bias)
                    .collect(),
            })
        };
    let cpu = denoise_latents_with_cfg_progress_and_runtime_options(
        latents.clone(),
        &schedule,
        2.0,
        &positive_embeddings,
        &negative_embeddings,
        predict_noise,
        None,
        None,
        None,
        None,
        None,
        None,
        DiffusionGenerationRuntimeOptions::default(),
        None,
    )
    .unwrap();
    let hip = denoise_latents_with_cfg_progress_and_runtime_options(
        latents,
        &schedule,
        2.0,
        &positive_embeddings,
        &negative_embeddings,
        predict_noise,
        None,
        None,
        None,
        None,
        None,
        None,
        DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
        None,
    )
    .unwrap();

    assert_eq!(cpu.runtime_kind, DiffusionRuntimeKind::CpuSourceReference);
    assert_eq!(hip.runtime_kind, DiffusionRuntimeKind::RocmHybridReference);
    assert_eq!(hip.latents.batch, cpu.latents.batch);
    assert_eq!(hip.latents.channels, cpu.latents.channels);
    assert_eq!(hip.latents.height, cpu.latents.height);
    assert_eq!(hip.latents.width, cpu.latents.width);
    assert!(f32_slices_close(&hip.latents.data, &cpu.latents.data, 1e-5));
}

#[test]
fn hip_denoise_vector_runtime_context_reuses_single_gpu() {
    if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
        eprintln!("skip: ROCm GPU unavailable for denoise context reuse test: {error}");
        return;
    }
    let schedule = DiffusionSchedule {
        timesteps: vec![1.0],
        sigmas: vec![1.0, 0.5],
        prediction_type: SchedulerPredictionType::Epsilon,
        input_scaling: SchedulerInputScaling::Sigma,
        solver: SchedulerSolver::Euler,
        train_timesteps: Vec::new(),
        alpha_t: Vec::new(),
        sigma_t: Vec::new(),
        lambda_t: Vec::new(),
    };
    let sample = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![-0.75, -0.25, 0.5, 1.25],
    };
    let negative_pred = CpuTensor {
        shape: sample.shape.clone(),
        data: vec![0.1, -0.2, 0.3, -0.4],
    };
    let positive_pred = CpuTensor {
        shape: sample.shape.clone(),
        data: vec![0.4, -0.1, 0.6, -0.2],
    };
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 2,
        width: 2,
        data: sample.data.clone(),
    };
    let mut scheduler_state = SchedulerStepState::default();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::rocm_hybrid(0));

    let (_scaled, scale_kind) =
        scale_model_input_with_runtime_context(&schedule, &sample, 0, &mut runtime_context)
            .unwrap();
    let (guided, guidance_kind) = cfg_guidance_with_runtime_context(
        &negative_pred,
        &positive_pred,
        2.0,
        &mut runtime_context,
    )
    .unwrap();
    let step_kind = scheduler_step_with_runtime_context(
        &schedule,
        &mut latents,
        &guided.data,
        0,
        &mut scheduler_state,
        &mut runtime_context,
    )
    .unwrap();

    assert_eq!(scale_kind, DiffusionRuntimeKind::RocmHybridReference);
    assert_eq!(guidance_kind, DiffusionRuntimeKind::RocmHybridReference);
    assert_eq!(step_kind, DiffusionRuntimeKind::RocmHybridReference);
    assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
    assert!(latents.data.iter().all(|value| value.is_finite()));
}

#[test]
fn hip_center_unet_input_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for centered UNet input parity test: {error}");
            return;
        }
    };
    let input = CpuTensor {
        shape: vec![2, 2, 2, 2],
        data: (0..16)
            .map(|idx| idx as f32 / 7.0 - 1.0)
            .collect::<Vec<_>>(),
    };

    let cpu_centered = maybe_center_unet_input(&input, true);
    let hip_centered = maybe_center_unet_input_hip_on_gpu(&mut gpu, &input, true).unwrap();
    assert_eq!(hip_centered, cpu_centered);

    let cpu_passthrough = maybe_center_unet_input(&input, false);
    let hip_passthrough = maybe_center_unet_input_hip_on_gpu(&mut gpu, &input, false).unwrap();
    assert_eq!(hip_passthrough, cpu_passthrough);
}

#[test]
fn hip_timestep_embedding_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for timestep embedding parity test: {error}");
            return;
        }
    };
    let timesteps = [999.0, 500.5, 0.25];
    for (dim, flip_sin_to_cos, freq_shift) in [(7, true, 1.0), (6, false, 0.0), (1, true, 0.0)] {
        let cpu = timestep_embedding(&timesteps, dim, flip_sin_to_cos, freq_shift).unwrap();
        let hip =
            timestep_embedding_hip_on_gpu(&mut gpu, &timesteps, dim, flip_sin_to_cos, freq_shift)
                .unwrap();

        assert_eq!(hip.shape, cpu.shape);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                    (actual - expected).abs() <= 1e-5,
                    "timestep embedding mismatch at {index}: dim={dim} flip={flip_sin_to_cos} shift={freq_shift} hip={actual} cpu={expected}"
                );
        }
        if dim % 2 == 1 {
            for row in 0..timesteps.len() {
                assert_eq!(hip.data[row * dim + dim - 1], 0.0);
            }
        }
    }
}

#[test]
fn hip_conv2d_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for Conv2D kernel parity test: {error}");
            return;
        }
    };
    let input = CpuTensor {
        shape: vec![2, 2, 3, 4],
        data: (0..48)
            .map(|idx| idx as f32 / 11.0 - 2.0)
            .collect::<Vec<_>>(),
    };
    let weight = CpuTensor {
        shape: vec![3, 2, 3, 2],
        data: (0..36)
            .map(|idx| (idx as f32 % 9.0 - 4.0) / 6.0)
            .collect::<Vec<_>>(),
    };
    let bias = CpuTensor {
        shape: vec![3],
        data: vec![0.25, -0.5, 0.75],
    };
    for bias in [Some(&bias), None] {
        let cpu = conv2d_nchw_with_stride(&input, &weight, bias, 1, 2).unwrap();
        let hip = conv2d_nchw_hip_on_gpu(
            &mut gpu,
            &mut RocmWeightCache::default(),
            &input,
            &weight,
            bias,
            1,
            2,
        )
        .unwrap();

        assert_eq!(hip.shape, cpu.shape);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "Conv2D mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }
}

#[test]
fn hip_group_norm_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for GroupNorm kernel parity test: {error}");
            return;
        }
    };
    let input = CpuTensor {
        shape: vec![2, 4, 2, 3],
        data: (0..48)
            .map(|idx| idx as f32 / 13.0 - 1.75)
            .collect::<Vec<_>>(),
    };
    let weight = CpuTensor {
        shape: vec![4],
        data: vec![1.0, 0.5, -1.0, 1.5],
    };
    let bias = CpuTensor {
        shape: vec![4],
        data: vec![0.0, 0.25, -0.5, 0.75],
    };

    let cpu = group_norm_nchw(&input, &weight, &bias, 2, 1e-5).unwrap();
    let hip = group_norm_nchw_hip_on_gpu(&mut gpu, &input, &weight, &bias, 2, 1e-5).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "GroupNorm mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

#[test]
fn hip_silu_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for SiLU kernel parity test: {error}");
            return;
        }
    };
    let input = CpuTensor {
        shape: vec![2, 2, 2, 2],
        data: vec![
            -8.0, -4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 8.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0,
            0.125,
        ],
    };

    let cpu = tensor_map(&input, silu);
    let hip = silu_hip_on_gpu(&mut gpu, &input).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "SiLU mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

#[test]
fn hip_leaky_relu_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for LeakyReLU kernel parity test: {error}");
            return;
        }
    };
    let input = CpuTensor {
        shape: vec![2, 2, 2, 2],
        data: vec![
            -8.0, -4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 8.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0,
            0.125,
        ],
    };
    // RealESRGAN / RRDBNet negative slope.
    let alpha = 0.2_f32;

    let cpu = tensor_map(&input, |value| if value >= 0.0 { value } else { alpha * value });
    let hip = leaky_relu_hip_on_gpu(&mut gpu, &input, alpha).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "LeakyReLU mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

/// CPU reference for space-to-depth (inverse of pixel-shuffle), matching
/// PyTorch/basicsr pixel_unshuffle channel ordering.
fn pixel_unshuffle_cpu_reference(input: &CpuTensor, scale: usize) -> CpuTensor {
    let (n, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let (oh, ow) = (h / scale, w / scale);
    let oc = c * scale * scale;
    let mut out = vec![0.0f32; n * oc * oh * ow];
    for nn in 0..n {
        for cc in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let (dy, dx) = (y % scale, x % scale);
                    let (by, bx) = (y / scale, x / scale);
                    let c_out = cc * scale * scale + dy * scale + dx;
                    let in_idx = ((nn * c + cc) * h + y) * w + x;
                    let out_idx = ((nn * oc + c_out) * oh + by) * ow + bx;
                    out[out_idx] = input.data[in_idx];
                }
            }
        }
    }
    CpuTensor {
        shape: vec![n, oc, oh, ow],
        data: out,
    }
}

#[test]
fn hip_pixel_unshuffle_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for pixel_unshuffle kernel parity test: {error}");
            return;
        }
    };
    // N=1, C=2, H=4, W=4, scale=2 -> [1, 8, 2, 2]. Sequential fill so any
    // channel-ordering or index error is obvious.
    let input = CpuTensor {
        shape: vec![1, 2, 4, 4],
        data: (0..32).map(|v| v as f32).collect(),
    };
    let scale = 2;

    let cpu = pixel_unshuffle_cpu_reference(&input, scale);
    let hip = pixel_unshuffle_nchw_hip_on_gpu(&mut gpu, &input, scale).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    assert_eq!(hip.shape, vec![1, 8, 2, 2]);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "pixel_unshuffle mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

#[test]
fn scaled_add_resident_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for scaled_add kernel parity test: {error}");
            return;
        }
    };
    let a = CpuTensor {
        shape: vec![2, 2, 2, 2],
        data: (0..16).map(|v| v as f32 - 8.0).collect(),
    };
    let b = CpuTensor {
        shape: vec![2, 2, 2, 2],
        data: (0..16).map(|v| (v as f32) * 0.5).collect(),
    };
    // RRDBNet residual scaling.
    let scale = 0.2_f32;
    let cpu: Vec<f32> = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(av, bv)| av + scale * bv)
        .collect();

    let a_gpu = gpu.upload_f32(&a.data, &a.shape).unwrap();
    let b_gpu = gpu.upload_f32(&b.data, &b.shape).unwrap();
    let out_gpu = scaled_add_resident(&mut gpu, &a_gpu, &b_gpu, scale).unwrap();
    let hip = download_resident(&mut gpu, &out_gpu).unwrap();
    free_resident(&mut gpu, out_gpu).unwrap();
    free_resident(&mut gpu, a_gpu).unwrap();
    free_resident(&mut gpu, b_gpu).unwrap();

    assert_eq!(hip.shape, a.shape);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "scaled_add mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

/// Build a 3x3 Conv2dLayer with deterministic small weights for RDB tests.
fn rdb_test_conv(out_channels: usize, in_channels: usize, seed: f32) -> Conv2dLayer {
    let fill = |n: usize, seed: f32| -> Vec<f32> {
        (0..n)
            .map(|k| (((k as f32 + seed) % 13.0) - 6.0) / 12.0)
            .collect()
    };
    let weight = CpuTensor {
        shape: vec![out_channels, in_channels, 3, 3],
        data: fill(out_channels * in_channels * 9, seed),
    };
    let bias = CpuTensor {
        shape: vec![out_channels],
        data: fill(out_channels, seed + 100.0),
    };
    Conv2dLayer {
        weight,
        bias: Some(bias),
        padding: 1,
        stride: 1,
    }
}

#[test]
fn superres_residual_dense_block_resident_matches_cpu_reference() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for RDB parity test: {error}");
            return;
        }
    };
    // num_feat = 4, num_grow_ch = 2: conv{k} in = num_feat + (k-1)*grow.
    let block = SuperResResidualDenseBlock {
        conv1: rdb_test_conv(2, 4, 1.0),
        conv2: rdb_test_conv(2, 6, 2.0),
        conv3: rdb_test_conv(2, 8, 3.0),
        conv4: rdb_test_conv(2, 10, 4.0),
        conv5: rdb_test_conv(4, 12, 5.0),
    };
    let input = CpuTensor {
        shape: vec![1, 4, 4, 4],
        data: (0..64).map(|v| ((v as f32) % 9.0 - 4.0) / 4.0).collect(),
    };

    let cpu = block.forward(&input).unwrap();

    let input_gpu = gpu.upload_f32(&input.data, &input.shape).unwrap();
    let mut cache = RocmWeightCache::default();
    let out_gpu = block
        .forward_resident(&input_gpu, &mut gpu, &mut cache)
        .unwrap();
    let hip = download_resident(&mut gpu, &out_gpu).unwrap();
    free_resident(&mut gpu, out_gpu).unwrap();
    free_resident(&mut gpu, input_gpu).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    assert_eq!(hip.shape, vec![1, 4, 4, 4]);
    let max_diff = hip
        .data
        .iter()
        .zip(&cpu.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff <= 2e-2,
        "RDB resident vs cpu max_diff {max_diff} too large; hip={:?} cpu={:?}",
        &hip.data[..hip.data.len().min(8)],
        &cpu.data[..cpu.data.len().min(8)]
    );
}

fn rdb_test_block(feat_seed: f32) -> SuperResResidualDenseBlock {
    // num_feat = 4, num_grow_ch = 2.
    SuperResResidualDenseBlock {
        conv1: rdb_test_conv(2, 4, feat_seed + 1.0),
        conv2: rdb_test_conv(2, 6, feat_seed + 2.0),
        conv3: rdb_test_conv(2, 8, feat_seed + 3.0),
        conv4: rdb_test_conv(2, 10, feat_seed + 4.0),
        conv5: rdb_test_conv(4, 12, feat_seed + 5.0),
    }
}

#[test]
fn superres_rrdb_resident_matches_cpu_reference() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for RRDB parity test: {error}");
            return;
        }
    };
    let block = SuperResRrdb {
        rdb1: rdb_test_block(10.0),
        rdb2: rdb_test_block(20.0),
        rdb3: rdb_test_block(30.0),
    };
    let input = CpuTensor {
        shape: vec![1, 4, 4, 4],
        data: (0..64).map(|v| ((v as f32) % 7.0 - 3.0) / 4.0).collect(),
    };

    let cpu = block.forward(&input).unwrap();

    let input_gpu = gpu.upload_f32(&input.data, &input.shape).unwrap();
    let mut cache = RocmWeightCache::default();
    let out_gpu = block
        .forward_resident(&input_gpu, &mut gpu, &mut cache)
        .unwrap();
    let hip = download_resident(&mut gpu, &out_gpu).unwrap();
    free_resident(&mut gpu, out_gpu).unwrap();
    free_resident(&mut gpu, input_gpu).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    let max_diff = hip
        .data
        .iter()
        .zip(&cpu.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Three chained RDBs (15 convs) accumulate a bit more WMMA f16 error.
    assert!(
        max_diff <= 3e-2,
        "RRDB resident vs cpu max_diff {max_diff} too large; hip={:?} cpu={:?}",
        &hip.data[..hip.data.len().min(8)],
        &cpu.data[..cpu.data.len().min(8)]
    );
}

#[test]
fn superres_rrdbnet_x2_resident_matches_cpu_reference() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for RRDBNet parity test: {error}");
            return;
        }
    };
    // Tiny x2 RRDBNet: in=3, num_feat=4, num_grow_ch=2, num_block=1, scale=2.
    // Input [1,3,4,4] -> unshuffle2 -> [1,12,2,2] -> conv_first -> [1,4,2,2]
    // -> body -> conv_body(+res) -> up x2 x2 -> [1,4,8,8] -> hr -> last -> [1,3,8,8].
    let net = SuperResRrdbNet {
        scale: 2,
        conv_first: rdb_test_conv(4, 12, 1.0),
        body: vec![SuperResRrdb {
            rdb1: rdb_test_block(40.0),
            rdb2: rdb_test_block(50.0),
            rdb3: rdb_test_block(60.0),
        }],
        conv_body: rdb_test_conv(4, 4, 7.0),
        conv_up1: rdb_test_conv(4, 4, 8.0),
        conv_up2: rdb_test_conv(4, 4, 9.0),
        conv_hr: rdb_test_conv(4, 4, 11.0),
        conv_last: rdb_test_conv(3, 4, 12.0),
    };
    let input = CpuTensor {
        shape: vec![1, 3, 4, 4],
        data: (0..48).map(|v| ((v as f32) % 11.0 - 5.0) / 6.0).collect(),
    };

    let cpu = net.forward(&input).unwrap();
    assert_eq!(cpu.shape, vec![1, 3, 8, 8], "x2 output should double spatial");

    let input_gpu = gpu.upload_f32(&input.data, &input.shape).unwrap();
    let mut cache = RocmWeightCache::default();
    let out_gpu = net
        .forward_resident(&input_gpu, &mut gpu, &mut cache)
        .unwrap();
    let hip = download_resident(&mut gpu, &out_gpu).unwrap();
    free_resident(&mut gpu, out_gpu).unwrap();
    free_resident(&mut gpu, input_gpu).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    let max_diff = hip
        .data
        .iter()
        .zip(&cpu.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Full net (~20 convs deep) over the WMMA f16 resident conv path.
    assert!(
        max_diff <= 5e-2,
        "RRDBNet resident vs cpu max_diff {max_diff} too large; hip={:?} cpu={:?}",
        &hip.data[..hip.data.len().min(8)],
        &cpu.data[..cpu.data.len().min(8)]
    );
}

/// End-to-end step-4 validation: import a real RealESRGAN RRDBNet checkpoint to
/// .hfq, load it via the metadata, and run the model forward on gfx1103.
/// Skips when the checkpoint is not present on this host.
#[test]
fn realesrgan_import_loads_and_runs_x4_anime_6b() {
    let checkpoint = std::path::Path::new("/home/sadara/RealESRGAN_x4plus_anime_6B.pth");
    if !checkpoint.exists() {
        eprintln!("skip: RealESRGAN_x4plus_anime_6B.pth not present on this host");
        return;
    }
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for RealESRGAN import/run test: {error}");
            return;
        }
    };

    let out_hfq = std::env::temp_dir().join("hipfire_test_realesrgan_x4_anime_6b.hfq");
    let summary = import_realesrgan_to_hfq(RealesrganImportOptions {
        source: checkpoint.to_path_buf(),
        output: out_hfq.clone(),
        model_name: Some("RealESRGAN-x4plus-anime-6B".to_string()),
    })
    .unwrap();
    // anime_6B: x4, 6 RRDBs, feat 64, grow 32, RGB — no input pixel-unshuffle.
    assert_eq!(summary.scale, 4, "anime_6B is an x4 model");
    assert_eq!(summary.num_block, 6);
    assert_eq!(summary.num_feat, 64);
    assert_eq!(summary.num_grow_ch, 32);
    assert_eq!(summary.num_in_ch, 3);
    assert_eq!(summary.num_out_ch, 3);

    let net = SuperResRrdbNet::open_hfq(&out_hfq).unwrap();
    assert_eq!(net.scale, 4);
    assert_eq!(net.body.len(), 6);

    // 16x16 RGB -> x4 -> 64x64.
    let input = CpuTensor {
        shape: vec![1, 3, 16, 16],
        data: (0..(3 * 16 * 16))
            .map(|v| ((v % 23) as f32 - 11.0) / 12.0)
            .collect(),
    };
    let input_gpu = gpu.upload_f32(&input.data, &input.shape).unwrap();
    let mut cache = RocmWeightCache::default();
    let out_gpu = net
        .forward_resident(&input_gpu, &mut gpu, &mut cache)
        .unwrap();
    let hip = download_resident(&mut gpu, &out_gpu).unwrap();
    free_resident(&mut gpu, out_gpu).unwrap();
    free_resident(&mut gpu, input_gpu).unwrap();

    assert_eq!(hip.shape, vec![1, 3, 64, 64], "x4 upscale of 16x16");
    assert!(
        hip.data.iter().all(|v| v.is_finite()),
        "super-res output has non-finite values"
    );

    let _ = std::fs::remove_file(&out_hfq);
}

/// Exercise the public DiffusionSuperResModel surface (RGB<->tensor conversion,
/// batch handling) with the x2plus model, which also covers the input
/// pixel-unshuffle branch. Skips when the checkpoint is not present.
#[test]
fn diffusion_superres_model_upscales_rgb_batch_x2() {
    let checkpoint = std::path::Path::new("/home/sadara/RealESRGAN_x2plus.pth");
    if !checkpoint.exists() {
        eprintln!("skip: RealESRGAN_x2plus.pth not present on this host");
        return;
    }
    if hipfire_rdna::Gpu::init_with_device(0).is_err() {
        eprintln!("skip: ROCm GPU unavailable for super-res upscale test");
        return;
    }

    let out_hfq = std::env::temp_dir().join("hipfire_test_realesrgan_x2plus.hfq");
    let summary = import_realesrgan_to_hfq(RealesrganImportOptions {
        source: checkpoint.to_path_buf(),
        output: out_hfq.clone(),
        model_name: None,
    })
    .unwrap();
    assert_eq!(summary.scale, 2, "x2plus is an x2 model");
    assert_eq!(summary.num_block, 23);

    let model = DiffusionSuperResModel::open_hfq(&out_hfq).unwrap();
    assert_eq!(model.scale(), 2);

    // A 2-image u8 RGB batch at 12x8 -> x2 -> 24x16.
    let (w, h, batch) = (12usize, 8usize, 2usize);
    let input = RgbImageBatch {
        batch,
        width: w,
        height: h,
        data: (0..(batch * w * h * 3))
            .map(|v| (v % 256) as u8)
            .collect(),
    };
    let upscaled = model.upscale_rgb_batch(&input, Some(0)).unwrap();

    assert_eq!(upscaled.batch, batch);
    assert_eq!(upscaled.width, w * 2);
    assert_eq!(upscaled.height, h * 2);
    assert_eq!(upscaled.data.len(), batch * (w * 2) * (h * 2) * 3);

    let _ = std::fs::remove_file(&out_hfq);
}

#[test]
fn bf16_resident_linear_matches_cpu_reference() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for bf16 linear parity test: {error}");
            return;
        }
    };
    if !gpu.arch_caps.has_wmma_w32() {
        eprintln!("skip: bf16 WMMA linear needs wave32 WMMA");
        return;
    }
    // in_features multiple of 16 -> exercises the bf16-native GEMM path.
    let (rows, in_features, out_features) = (4usize, 32usize, 8usize);
    let fill = |n: usize, seed: f32, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|k| (((k as f32 + seed) % 13.0) - 6.0) / 6.0 * scale)
            .collect()
    };
    let input = CpuTensor {
        shape: vec![rows, in_features],
        data: fill(rows * in_features, 1.0, 1.0),
    };
    let weight = CpuTensor {
        shape: vec![out_features, in_features],
        data: fill(out_features * in_features, 3.0, 0.5),
    };
    let bias = CpuTensor {
        shape: vec![out_features],
        data: fill(out_features, 7.0, 0.25),
    };

    // CPU f32 reference: Y[r,o] = sum_k input[r,k]*weight[o,k] + bias[o].
    let mut cpu = vec![0.0f32; rows * out_features];
    for r in 0..rows {
        for o in 0..out_features {
            let mut acc = bias.data[o];
            for k in 0..in_features {
                acc += input.data[r * in_features + k] * weight.data[o * in_features + k];
            }
            cpu[r * out_features + o] = acc;
        }
    }

    let input_gpu = gpu.upload_f32(&input.data, &input.shape).unwrap();
    let mut cache = RocmWeightCache::default();
    let out_gpu =
        linear_optional_bias_resident(&mut gpu, &mut cache, &input_gpu, &weight, Some(&bias))
            .unwrap();
    let hip = download_resident(&mut gpu, &out_gpu).unwrap();
    free_resident(&mut gpu, out_gpu).unwrap();
    free_resident(&mut gpu, input_gpu).unwrap();

    assert_eq!(hip.shape, vec![rows, out_features]);
    // bf16 inputs (~8-bit mantissa) accumulated in f32 over K=32.
    let max_rel = hip
        .data
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs() / (b.abs().max(1.0)))
        .fold(0.0f32, f32::max);
    assert!(
        max_rel <= 3e-2,
        "bf16 linear vs cpu max rel error {max_rel} too large; hip={:?} cpu={:?}",
        hip.data,
        cpu
    );
}

#[test]
fn bf16_hip_on_gpu_linear_matches_cpu_reference() {
    // The transformer (Krea2 DiT + Qwen3-VL text encoder) linears route through
    // linear_optional_bias_hip_on_gpu; verify its bf16-WMMA path.
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for bf16 hip_on_gpu linear test: {error}");
            return;
        }
    };
    if !gpu.arch_caps.has_wmma_w32() {
        eprintln!("skip: bf16 WMMA linear needs wave32 WMMA");
        return;
    }
    let (rows, in_features, out_features) = (6usize, 48usize, 16usize);
    let fill = |n: usize, seed: f32, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|k| (((k as f32 + seed) % 17.0) - 8.0) / 8.0 * scale)
            .collect()
    };
    let input = CpuTensor {
        shape: vec![rows, in_features],
        data: fill(rows * in_features, 2.0, 1.0),
    };
    let weight = CpuTensor {
        shape: vec![out_features, in_features],
        data: fill(out_features * in_features, 5.0, 0.5),
    };
    let bias = CpuTensor {
        shape: vec![out_features],
        data: fill(out_features, 9.0, 0.25),
    };

    let mut cpu = vec![0.0f32; rows * out_features];
    for r in 0..rows {
        for o in 0..out_features {
            let mut acc = bias.data[o];
            for k in 0..in_features {
                acc += input.data[r * in_features + k] * weight.data[o * in_features + k];
            }
            cpu[r * out_features + o] = acc;
        }
    }

    let mut cache = RocmWeightCache::default();
    let hip = linear_optional_bias_hip_on_gpu(&mut gpu, &mut cache, &input, &weight, Some(&bias))
        .unwrap();

    assert_eq!(hip.shape, vec![rows, out_features]);
    let max_rel = hip
        .data
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs() / (b.abs().max(1.0)))
        .fold(0.0f32, f32::max);
    assert!(
        max_rel <= 3e-2,
        "bf16 hip_on_gpu linear vs cpu max rel error {max_rel} too large; hip={:?} cpu={:?}",
        hip.data,
        cpu
    );
}

#[test]
fn resident_weight_bf16_linear_matches_cpu_and_caches_by_name() {
    // The transformer's per-step path: a source-reference ResidentWeight
    // uploaded once (raw bf16 bytes, keyed by name) and reused. Verify the
    // raw-bytes path matches a CPU reference over the SAME bf16-rounded weights,
    // and that a second call hits the name cache (no new upload).
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for resident-weight linear test: {error}");
            return;
        }
    };
    if !gpu.arch_caps.has_wmma_w32() {
        eprintln!("skip: needs wave32 WMMA");
        return;
    }
    let (rows, in_features, out_features) = (5usize, 64usize, 24usize);
    let fill = |n: usize, seed: f32, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|k| (((k as f32 + seed) % 19.0) - 9.0) / 9.0 * scale)
            .collect()
    };
    let input = CpuTensor {
        shape: vec![rows, in_features],
        data: fill(rows * in_features, 2.0, 1.0),
    };
    let weight_f32 = fill(out_features * in_features, 4.0, 0.5);
    let weight = ResidentWeight::from_bf16_parts(
        "test.weight",
        vec![out_features, in_features],
        &weight_f32,
    );
    // Reference uses the weight AS the GPU sees it: bf16-rounded back to f32.
    let bf16_round = |v: f32| f32::from_bits((((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u32) << 16);
    let wq: Vec<f32> = weight_f32.iter().map(|&v| bf16_round(v)).collect();
    let mut cpu = vec![0.0f32; rows * out_features];
    for r in 0..rows {
        for o in 0..out_features {
            let mut acc = 0.0f32;
            for k in 0..in_features {
                acc += input.data[r * in_features + k] * wq[o * in_features + k];
            }
            cpu[r * out_features + o] = acc;
        }
    }

    let mut cache = RocmWeightCache::default();
    let hip = linear_resident_weight_hip_on_gpu(&mut gpu, &mut cache, &input, &weight, None).unwrap();
    // second call: same result, served from the name cache.
    let hip2 =
        linear_resident_weight_hip_on_gpu(&mut gpu, &mut cache, &input, &weight, None).unwrap();

    assert_eq!(hip.shape, vec![rows, out_features]);
    assert_eq!(hip.data, hip2.data, "cached second call must be identical");
    let max_rel = hip
        .data
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs() / (b.abs().max(1.0)))
        .fold(0.0f32, f32::max);
    assert!(
        max_rel <= 3e-2,
        "resident bf16 linear vs cpu max rel error {max_rel} too large; hip={:?} cpu={:?}",
        hip.data,
        cpu
    );
}

#[test]
fn rms_norm_resident_matches_cpu_reference() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for rms_norm resident test: {error}");
            return;
        }
    };
    let (rows, width) = (5usize, 320usize);
    let eps = 1e-5f32;
    let fill = |n: usize, seed: f32, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|k| (((k as f32 + seed) % 13.0) - 6.0) / 6.0 * scale)
            .collect()
    };
    let input = CpuTensor {
        shape: vec![rows, width],
        data: fill(rows * width, 1.0, 1.5),
    };
    let weight = CpuTensor {
        shape: vec![width],
        data: fill(width, 4.0, 1.0),
    };
    // CPU reference: out = x / sqrt(mean(x^2)+eps) * weight.
    let mut cpu = vec![0.0f32; rows * width];
    for r in 0..rows {
        let base = r * width;
        let mut ss = 0.0f32;
        for c in 0..width {
            ss += input.data[base + c] * input.data[base + c];
        }
        let inv_rms = (ss / width as f32 + eps).sqrt().recip();
        for c in 0..width {
            cpu[base + c] = input.data[base + c] * inv_rms * weight.data[c];
        }
    }

    let input_gpu = gpu.upload_f32(&input.data, &input.shape).unwrap();
    let mut cache = RocmWeightCache::default();
    let out_gpu = rms_norm_resident(&mut gpu, &mut cache, &input_gpu, &weight, eps).unwrap();
    let hip = download_resident(&mut gpu, &out_gpu).unwrap();
    free_resident(&mut gpu, out_gpu).unwrap();
    free_resident(&mut gpu, input_gpu).unwrap();

    assert_eq!(hip.shape, vec![rows, width]);
    let max_abs = hip
        .data
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 1e-4,
        "rms_norm resident vs cpu max abs error {max_abs} too large"
    );
}

#[test]
fn linear_resident_weight_resident_matches_cpu_reference() {
    // The fully-resident linear: resident activation x streaming bf16 weight,
    // no host round-trip. Compare to a bf16-rounded-weight CPU reference.
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for resident linear test: {error}");
            return;
        }
    };
    if !gpu.arch_caps.has_wmma_w32() {
        eprintln!("skip: needs wave32 WMMA");
        return;
    }
    let (batch, seq, in_features, out_features) = (2usize, 3usize, 64usize, 32usize);
    let rows = batch * seq;
    let fill = |n: usize, seed: f32, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|k| (((k as f32 + seed) % 17.0) - 8.0) / 8.0 * scale)
            .collect()
    };
    let input = CpuTensor {
        shape: vec![batch, seq, in_features],
        data: fill(rows * in_features, 2.0, 1.0),
    };
    let weight_f32 = fill(out_features * in_features, 5.0, 0.5);
    let weight = ResidentWeight::from_bf16_parts(
        "resident.weight",
        vec![out_features, in_features],
        &weight_f32,
    );
    let bf16_round =
        |v: f32| f32::from_bits((((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u32) << 16);
    let wq: Vec<f32> = weight_f32.iter().map(|&v| bf16_round(v)).collect();
    let mut cpu = vec![0.0f32; rows * out_features];
    for r in 0..rows {
        for o in 0..out_features {
            let mut acc = 0.0f32;
            for k in 0..in_features {
                acc += input.data[r * in_features + k] * wq[o * in_features + k];
            }
            cpu[r * out_features + o] = acc;
        }
    }

    let input_gpu = gpu.upload_f32(&input.data, &input.shape).unwrap();
    let mut cache = RocmWeightCache::default();
    let out_gpu =
        linear_resident_weight_resident(&mut gpu, &mut cache, &input_gpu, &weight, None).unwrap();
    let hip = download_resident(&mut gpu, &out_gpu).unwrap();
    free_resident(&mut gpu, out_gpu).unwrap();
    free_resident(&mut gpu, input_gpu).unwrap();

    // Output keeps the leading dims: [batch, seq, out_features].
    assert_eq!(hip.shape, vec![batch, seq, out_features]);
    let max_rel = hip
        .data
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs() / (b.abs().max(1.0)))
        .fold(0.0f32, f32::max);
    assert!(
        max_rel <= 3e-2,
        "resident-activation linear vs cpu max rel error {max_rel} too large"
    );
}

#[test]
fn resident_swiglu_and_sigmoid_gates_match_cpu_reference() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for resident gate test: {error}");
            return;
        }
    };
    let shape = vec![2usize, 3, 8];
    let n = 2 * 3 * 8;
    let fill = |seed: f32| -> Vec<f32> {
        (0..n)
            .map(|k| (((k as f32 + seed) % 11.0) - 5.0) / 3.0)
            .collect()
    };
    let a = CpuTensor {
        shape: shape.clone(),
        data: fill(1.0),
    };
    let g = CpuTensor {
        shape: shape.clone(),
        data: fill(4.0),
    };
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let swiglu_cpu: Vec<f32> = a.data.iter().zip(&g.data).map(|(u, x)| u * silu(*x)).collect();
    let sigmoid_cpu: Vec<f32> = a
        .data
        .iter()
        .zip(&g.data)
        .map(|(v, x)| v * sigmoid(*x))
        .collect();

    let a_gpu = gpu.upload_f32(&a.data, &a.shape).unwrap();
    let g_gpu = gpu.upload_f32(&g.data, &g.shape).unwrap();
    let sw = swiglu_gate_3d_resident(&mut gpu, &a_gpu, &g_gpu).unwrap();
    let sw_h = download_resident(&mut gpu, &sw).unwrap();
    free_resident(&mut gpu, sw).unwrap();
    let sg = sigmoid_gate_3d_resident(&mut gpu, &a_gpu, &g_gpu).unwrap();
    let sg_h = download_resident(&mut gpu, &sg).unwrap();
    free_resident(&mut gpu, sg).unwrap();
    free_resident(&mut gpu, a_gpu).unwrap();
    free_resident(&mut gpu, g_gpu).unwrap();

    assert_eq!(sw_h.shape, shape);
    for (i, (h, c)) in sw_h.data.iter().zip(&swiglu_cpu).enumerate() {
        assert!((h - c).abs() <= 1e-5, "swiglu mismatch at {i}: {h} vs {c}");
    }
    for (i, (h, c)) in sg_h.data.iter().zip(&sigmoid_cpu).enumerate() {
        assert!((h - c).abs() <= 1e-5, "sigmoid gate mismatch at {i}: {h} vs {c}");
    }
}

#[test]
fn rope_qwen_resident_matches_cpu_reference() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for resident RoPE test: {error}");
            return;
        }
    };
    let (batch, seq, heads, head_dim) = (2usize, 4, 3, 8);
    let width = heads * head_dim;
    let freq_width = head_dim / 2;
    let fill = |n: usize, seed: f32, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|k| (((k as f32 + seed) % 13.0) - 6.0) / 6.0 * scale)
            .collect()
    };
    let input = CpuTensor {
        shape: vec![batch, seq, width],
        data: fill(batch * seq * width, 1.0, 1.0),
    };
    // cos/sin as normalized cos(theta)/sin(theta) so |out| stays sane.
    let mut cos_d = vec![0.0f32; seq * freq_width];
    let mut sin_d = vec![0.0f32; seq * freq_width];
    for t in 0..seq {
        for p in 0..freq_width {
            let theta = (t as f32 + 1.0) * 0.1 * (p as f32 + 1.0);
            cos_d[t * freq_width + p] = theta.cos();
            sin_d[t * freq_width + p] = theta.sin();
        }
    }
    let cos = CpuTensor { shape: vec![seq, freq_width], data: cos_d };
    let sin = CpuTensor { shape: vec![seq, freq_width], data: sin_d };

    // CPU reference (interleaved pairs).
    let mut cpu = vec![0.0f32; batch * seq * width];
    for b in 0..batch {
        for t in 0..seq {
            for h in 0..heads {
                let tb = (b * seq + t) * width + h * head_dim;
                for p in 0..freq_width {
                    let ri = tb + p * 2;
                    let (re, im) = (input.data[ri], input.data[ri + 1]);
                    let (c, s) = (cos.data[t * freq_width + p], sin.data[t * freq_width + p]);
                    cpu[ri] = re * c - im * s;
                    cpu[ri + 1] = re * s + im * c;
                }
            }
        }
    }

    let input_gpu = gpu.upload_f32(&input.data, &input.shape).unwrap();
    let mut cache = RocmWeightCache::default();
    let out_gpu =
        rope_qwen_resident(&mut gpu, &mut cache, &input_gpu, &cos, &sin, heads, head_dim).unwrap();
    let hip = download_resident(&mut gpu, &out_gpu).unwrap();
    free_resident(&mut gpu, out_gpu).unwrap();
    free_resident(&mut gpu, input_gpu).unwrap();

    assert_eq!(hip.shape, vec![batch, seq, width]);
    let max_abs = hip
        .data
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs <= 1e-5, "rope resident vs cpu max abs {max_abs}");
}

#[test]
fn hip_tensor_add_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for tensor-add kernel parity test: {error}");
            return;
        }
    };
    let left = CpuTensor {
        shape: vec![2, 2, 2, 3],
        data: (0..24)
            .map(|idx| idx as f32 / 9.0 - 1.25)
            .collect::<Vec<_>>(),
    };
    let right = CpuTensor {
        shape: vec![2, 2, 2, 3],
        data: (0..24)
            .map(|idx| (idx as f32 % 7.0 - 3.0) / 5.0)
            .collect::<Vec<_>>(),
    };

    let cpu = tensor_add(&left, &right).unwrap();
    let hip = tensor_add_hip_on_gpu(&mut gpu, &left, &right).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "tensor-add mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

#[test]
fn hip_add_channel_bias_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for channel-bias kernel parity test: {error}");
            return;
        }
    };
    let input = CpuTensor {
        shape: vec![2, 3, 2, 3],
        data: (0..36)
            .map(|idx| idx as f32 / 13.0 - 1.5)
            .collect::<Vec<_>>(),
    };
    let bias = CpuTensor {
        shape: vec![2, 3],
        data: vec![0.25, -0.5, 0.75, -1.0, 0.5, -0.25],
    };
    let mut cpu = input.clone();
    add_channel_bias_nchw(&mut cpu, &bias).unwrap();
    let hip = add_channel_bias_nchw_hip_on_gpu(&mut gpu, &input, &bias).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "channel-bias mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

#[test]
fn hip_nchw_bsc_layout_transforms_match_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for layout kernel parity test: {error}");
            return;
        }
    };
    let input = CpuTensor {
        shape: vec![2, 3, 2, 4],
        data: (0..48)
            .map(|idx| idx as f32 / 17.0 - 1.25)
            .collect::<Vec<_>>(),
    };

    let cpu_bsc = nchw_to_bsc(&input).unwrap();
    let hip_bsc = nchw_to_bsc_hip_on_gpu(&mut gpu, &input).unwrap();
    assert_eq!(hip_bsc, cpu_bsc);

    let cpu_nchw = bsc_to_nchw(&cpu_bsc, 2, 3, 2, 4).unwrap();
    let hip_nchw = bsc_to_nchw_hip_on_gpu(&mut gpu, &cpu_bsc, 2, 3, 2, 4).unwrap();
    assert_eq!(hip_nchw, cpu_nchw);
    assert_eq!(hip_nchw, input);
}

#[test]
fn hip_concat_channels_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for channel-concat kernel parity test: {error}");
            return;
        }
    };
    let left = CpuTensor {
        shape: vec![2, 2, 2, 3],
        data: (0..24)
            .map(|idx| idx as f32 / 9.0 - 1.0)
            .collect::<Vec<_>>(),
    };
    let right = CpuTensor {
        shape: vec![2, 3, 2, 3],
        data: (0..36)
            .map(|idx| (idx as f32 % 13.0 - 6.0) / 7.0)
            .collect::<Vec<_>>(),
    };

    let cpu = concat_channels_nchw(&left, &right).unwrap();
    let hip = concat_channels_nchw_hip_on_gpu(&mut gpu, &left, &right).unwrap();

    assert_eq!(hip, cpu);
}

#[test]
fn hip_concat_last_dim_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for last-dim concat kernel parity test: {error}");
            return;
        }
    };
    let left_2d = CpuTensor {
        shape: vec![4, 2],
        data: (0..8).map(|idx| idx as f32 / 5.0 - 0.75).collect(),
    };
    let right_2d = CpuTensor {
        shape: vec![4, 3],
        data: (0..12).map(|idx| (idx as f32 % 7.0 - 3.0) / 4.0).collect(),
    };
    let cpu_2d = concat_last_dim_2d(&left_2d, &right_2d).unwrap();
    let hip_2d = concat_last_dim_2d_hip_on_gpu(&mut gpu, &left_2d, &right_2d).unwrap();
    assert_eq!(hip_2d, cpu_2d);

    let left_3d = CpuTensor {
        shape: vec![2, 3, 2],
        data: (0..12).map(|idx| idx as f32 / 6.0 - 1.0).collect(),
    };
    let right_3d = CpuTensor {
        shape: vec![2, 3, 4],
        data: (0..24).map(|idx| (idx as f32 % 11.0 - 5.0) / 8.0).collect(),
    };
    let cpu_3d = concat_last_dim_3d(&left_3d, &right_3d).unwrap();
    let hip_3d = concat_last_dim_3d_hip_on_gpu(&mut gpu, &left_3d, &right_3d).unwrap();
    assert_eq!(hip_3d, cpu_3d);
}

#[test]
fn hip_upsample_nearest2d_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!(
                "skip: ROCm GPU unavailable for nearest-upsample kernel parity test: {error}"
            );
            return;
        }
    };
    let input = CpuTensor {
        shape: vec![2, 2, 2, 3],
        data: (0..24)
            .map(|idx| idx as f32 / 7.0 - 1.25)
            .collect::<Vec<_>>(),
    };

    for scale in [2, 3] {
        let cpu = upsample_nearest2d_nchw(&input, scale).unwrap();
        let hip = upsample_nearest2d_nchw_hip_on_gpu(&mut gpu, &input, scale).unwrap();

        assert_eq!(hip.shape, cpu.shape);
        assert_eq!(hip.data, cpu.data);
    }
}

#[test]
fn hip_linear_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for linear kernel parity test: {error}");
            return;
        }
    };
    let input = CpuTensor {
        shape: vec![4, 3],
        data: (0..12)
            .map(|idx| idx as f32 / 5.0 - 1.1)
            .collect::<Vec<_>>(),
    };
    let weight = CpuTensor {
        shape: vec![5, 3],
        data: (0..15)
            .map(|idx| (idx as f32 % 7.0 - 3.0) / 4.0)
            .collect::<Vec<_>>(),
    };
    let bias = CpuTensor {
        shape: vec![5],
        data: vec![0.25, -0.5, 0.75, -1.0, 1.25],
    };

    for bias in [Some(&bias), None] {
        let cpu = linear_optional_bias(&input, &weight, bias).unwrap();
        let hip = linear_optional_bias_hip_on_gpu(
            &mut gpu,
            &mut RocmWeightCache::default(),
            &input,
            &weight,
            bias,
        )
        .unwrap();

        assert_eq!(hip.shape, cpu.shape);
        for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "linear mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }
}

#[test]
fn hip_layer_norm_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for LayerNorm kernel parity test: {error}");
            return;
        }
    };
    let input = CpuTensor {
        shape: vec![4, 5],
        data: (0..20)
            .map(|idx| idx as f32 / 6.0 - 1.75)
            .collect::<Vec<_>>(),
    };
    let weight = CpuTensor {
        shape: vec![5],
        data: vec![1.0, 0.5, -1.0, 1.5, -0.25],
    };
    let bias = CpuTensor {
        shape: vec![5],
        data: vec![0.0, 0.25, -0.5, 0.75, -1.0],
    };

    let cpu = layer_norm(&input, &weight, &bias, 1e-5).unwrap();
    let hip = layer_norm_hip_on_gpu(&mut gpu, &input, &weight, &bias, 1e-5).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "LayerNorm mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

#[test]
fn hip_softmax_rows_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for softmax kernel parity test: {error}");
            return;
        }
    };
    let input = CpuTensor {
        shape: vec![4, 5],
        data: vec![
            1.0, 2.0, 3.0, 4.0, 5.0, -3.0, -1.0, -0.5, 0.25, 2.5, 10.0, 9.5, 8.0, 7.25, 6.0, 100.0,
            99.0, 98.0, 97.0, 96.0,
        ],
    };
    let mut cpu = input.clone();
    for row in cpu.data.chunks_mut(5) {
        softmax_in_place(row);
    }
    let hip = softmax_rows_hip_on_gpu(&mut gpu, &input).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "softmax mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
    for row in hip.data.chunks(5) {
        let sum = row.iter().sum::<f32>();
        assert!((sum - 1.0).abs() <= 1e-6, "softmax row sum {sum}");
    }
}

#[test]
fn hip_sdpa_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for SDPA kernel parity test: {error}");
            return;
        }
    };
    let q = CpuTensor {
        shape: vec![2, 2, 4],
        data: vec![
            0.2, -0.4, 0.6, 1.0, -1.2, 0.3, 0.5, -0.7, 1.0, 0.0, -0.5, 0.25, -0.25, 0.75, -1.0, 0.5,
        ],
    };
    let k = CpuTensor {
        shape: vec![2, 3, 4],
        data: vec![
            -0.5, 0.25, 0.75, -1.0, 1.25, -0.75, 0.5, 0.0, 0.1, 0.9, -0.3, 0.4, 0.7, -0.2, 0.3,
            -0.8, -0.6, 1.1, 0.2, 0.9, 1.5, -1.0, 0.4, -0.1,
        ],
    };
    let v = CpuTensor {
        shape: vec![2, 3, 4],
        data: vec![
            0.5, -1.0, 0.25, 0.75, -0.4, 0.6, -0.8, 1.2, 1.0, 0.2, -0.5, -0.1, -0.9, 0.3, 0.8,
            -0.2, 0.4, -0.7, 1.1, 0.6, -1.3, 0.5, 0.0, 0.9,
        ],
    };

    let cpu = scaled_dot_product_attention(&q, &k, &v, 2).unwrap();
    let hip = scaled_dot_product_attention_hip_on_gpu(&mut gpu, &q, &k, &v, 2).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "SDPA mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

#[test]
fn hip_geglu_gate_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for GeGLU gate parity test: {error}");
            return;
        }
    };
    let projected = CpuTensor {
        shape: vec![2, 3, 6],
        data: (0..36)
            .map(|idx| (idx as f32 % 13.0 - 6.0) / 4.0)
            .collect::<Vec<_>>(),
    };

    let cpu = geglu_gate_3d(&projected).unwrap();
    let hip = geglu_gate_3d_hip_on_gpu(&mut gpu, &projected).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "GeGLU gate mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

#[test]
fn hip_clip_causal_attention_matches_cpu_reference_when_gpu_is_available() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for CLIP causal attention parity test: {error}");
            return;
        }
    };
    let q = CpuTensor {
        shape: vec![4, 4],
        data: vec![
            0.2, -0.4, 0.6, 1.0, -1.2, 0.3, 0.5, -0.7, 1.0, 0.0, -0.5, 0.25, -0.25, 0.75, -1.0, 0.5,
        ],
    };
    let k = CpuTensor {
        shape: vec![4, 4],
        data: vec![
            -0.5, 0.25, 0.75, -1.0, 1.25, -0.75, 0.5, 0.0, 0.1, 0.9, -0.3, 0.4, 0.7, -0.2, 0.3,
            -0.8,
        ],
    };
    let v = CpuTensor {
        shape: vec![4, 4],
        data: vec![
            0.5, -1.0, 0.25, 0.75, -0.4, 0.6, -0.8, 1.2, 1.0, 0.2, -0.5, -0.1, -0.9, 0.3, 0.8, -0.2,
        ],
    };

    let cpu = clip_causal_self_attention(&q, &k, &v, 2).unwrap();
    let hip = clip_causal_self_attention_hip_on_gpu(&mut gpu, &q, &k, &v, 2).unwrap();

    assert_eq!(hip.shape, cpu.shape);
    assert_eq!(&cpu.data[0..4], &v.data[0..4]);
    for (index, (actual, expected)) in hip.data.iter().zip(&cpu.data).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "CLIP causal attention mismatch at {index}: hip={actual} cpu={expected}"
        );
    }
}

#[test]
fn hip_img2img_boundary_helpers_reuse_single_runtime_context() {
    if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
        eprintln!("skip: ROCm GPU unavailable for img2img boundary reuse test: {error}");
        return;
    }
    let image = RgbImageBatch {
        batch: 1,
        width: 4,
        height: 4,
        data: (0..48).map(|idx| (idx * 3 % 256) as u8).collect(),
    };
    let mask = RgbImageBatch {
        batch: 1,
        width: 4,
        height: 4,
        data: (0..16)
            .flat_map(|idx| {
                let value = if idx % 3 == 0 { 255 } else { 64 };
                [value, value, value]
            })
            .collect(),
    };
    let init = LatentBatch {
        batch: 1,
        channels: 2,
        height: 2,
        width: 2,
        data: (0..8).map(|idx| idx as f32 / 8.0).collect(),
    };
    let mut generated = LatentBatch {
        batch: 1,
        channels: 2,
        height: 2,
        width: 2,
        data: (0..8).map(|idx| 1.0 - idx as f32 / 8.0).collect(),
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::rocm_hybrid(0));

    let (weights, mask_kind) =
        latent_mask_weights_with_runtime_context(&mask, &init, &mut runtime_context).unwrap();
    let (masked, masked_kind) =
        masked_rgb_batch_for_inpaint_with_runtime_context(&image, &mask, &mut runtime_context)
            .unwrap();
    let blend_kind = blend_latents_with_mask_with_runtime_context(
        &mut generated,
        &init,
        &weights,
        &mut runtime_context,
    )
    .unwrap();

    assert_eq!(mask_kind, DiffusionRuntimeKind::RocmHybridReference);
    assert_eq!(masked_kind, DiffusionRuntimeKind::RocmHybridReference);
    assert_eq!(blend_kind, DiffusionRuntimeKind::RocmHybridReference);
    assert_eq!(masked.batch, image.batch);
    assert_eq!(masked.width, image.width);
    assert_eq!(masked.height, image.height);
    assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
}

#[test]
fn hip_preflight_reports_clip_token_position_embedding_probe() {
    if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
        eprintln!("skip: ROCm GPU unavailable for diffusion preflight test: {error}");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-preflight-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-preflight.hfq");
    let metadata = tiny_runtime_metadata();
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tiny_complete_runtime_tensors(),
    )
    .unwrap();
    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![DiffusionPrompt {
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            seed: 9,
            subseed: None,
        }],
        width: 2,
        height: 2,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: 1,
        cfg_scale: 1.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: false,
        save_images: false,
    };

    let preflight = pipeline
        .preflight_hip_runtime(&request, DiffusionHipRuntimeOptions { device_id: 0 })
        .unwrap();

    assert_eq!(
        preflight.clip_token_position_embedding_kernel_probe.name,
        "diffusion_clip_token_position_embedding_f32"
    );
    assert!(
        preflight
            .clip_token_position_embedding_kernel_probe
            .matched_cpu_reference
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn hip_clip_text_encoder_runtime_context_reuses_single_gpu() {
    if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
        eprintln!("skip: ROCm GPU unavailable for CLIP context reuse test: {error}");
        return;
    }
    let hidden = 12usize;
    let encoder = ClipTextEncoder {
        token_embedding: CpuTensor {
            shape: vec![3, hidden],
            data: (0..3 * hidden).map(|idx| idx as f32 * 0.01).collect(),
        },
        position_embedding: CpuTensor {
            shape: vec![2, hidden],
            data: vec![0.0; 2 * hidden],
        },
        layers: vec![zero_clip_layer(hidden)],
        final_layer_norm_weight: CpuTensor {
            shape: vec![hidden],
            data: vec![1.0; hidden],
        },
        final_layer_norm_bias: CpuTensor {
            shape: vec![hidden],
            data: vec![0.0; hidden],
        },
        text_projection: Some(CpuTensor {
            shape: vec![hidden, hidden],
            data: (0..hidden * hidden)
                .map(|idx| {
                    let row = idx / hidden;
                    let col = idx % hidden;
                    if row == col {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        }),
        hidden_size: hidden,
        max_length: 2,
        n_heads: 3,
    };
    let (cpu_hidden, cpu_pooled) = encoder.encode_tokens_with_pooled(&[0, 1], 1).unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::rocm_hybrid(0));
    let (hip_hidden, hip_pooled) = encoder
        .encode_tokens_with_pooled_and_runtime_context(&[0, 1], 1, &mut runtime_context)
        .unwrap();

    assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
    assert_eq!(hip_hidden.shape, cpu_hidden.shape);
    assert!(f32_slices_close(&hip_hidden.data, &cpu_hidden.data, 1e-5));
    assert!(f32_slices_close(
        &hip_pooled.unwrap(),
        &cpu_pooled.unwrap(),
        1e-5
    ));
}

/// Phase 1b: validate the device-resident CLIP encode against the CPU
/// reference with *non-trivial* weights, so the resident causal self-attention
/// and QuickGELU are exercised numerically (the `zero_clip_layer` routing
/// tests above act as identity layers and don't reach those paths).
#[test]
fn hip_clip_text_encoder_resident_matches_cpu_reference_with_nonzero_weights() {
    const HIDDEN: usize = 12;
    const HEADS: usize = 3;
    // Deterministic small finite [r, c] matrix.
    let mat = |r: usize, c: usize, seed: f32| -> CpuTensor {
        CpuTensor {
            shape: vec![r, c],
            data: (0..r * c)
                .map(|k| 0.05 * (((k as f32 + seed) % 7.0) - 3.0))
                .collect(),
        }
    };
    let vecf = |n: usize, seed: f32| -> CpuTensor {
        CpuTensor {
            shape: vec![n],
            data: (0..n)
                .map(|k| 0.02 * ((k as f32 + seed) % 5.0 - 2.0))
                .collect(),
        }
    };
    let ones = CpuTensor {
        shape: vec![HIDDEN],
        data: vec![1.0; HIDDEN],
    };
    let zeros = CpuTensor {
        shape: vec![HIDDEN],
        data: vec![0.0; HIDDEN],
    };
    let layer = |seed: f32| ClipEncoderLayer {
        q_proj_weight: mat(HIDDEN, HIDDEN, seed),
        q_proj_bias: vecf(HIDDEN, seed + 1.0),
        k_proj_weight: mat(HIDDEN, HIDDEN, seed + 2.0),
        k_proj_bias: vecf(HIDDEN, seed + 3.0),
        v_proj_weight: mat(HIDDEN, HIDDEN, seed + 4.0),
        v_proj_bias: vecf(HIDDEN, seed + 5.0),
        out_proj_weight: mat(HIDDEN, HIDDEN, seed + 6.0),
        out_proj_bias: vecf(HIDDEN, seed + 7.0),
        layer_norm1_weight: ones.clone(),
        layer_norm1_bias: zeros.clone(),
        fc1_weight: mat(HIDDEN, HIDDEN, seed + 8.0),
        fc1_bias: vecf(HIDDEN, seed + 9.0),
        fc2_weight: mat(HIDDEN, HIDDEN, seed + 10.0),
        fc2_bias: vecf(HIDDEN, seed + 11.0),
        layer_norm2_weight: ones.clone(),
        layer_norm2_bias: zeros.clone(),
    };
    let encoder = ClipTextEncoder {
        token_embedding: CpuTensor {
            shape: vec![4, HIDDEN],
            data: (0..4 * HIDDEN)
                .map(|idx| (idx as f32 % 9.0 - 4.0) * 0.1)
                .collect(),
        },
        position_embedding: CpuTensor {
            shape: vec![3, HIDDEN],
            data: (0..3 * HIDDEN)
                .map(|idx| (idx as f32 % 5.0 - 2.0) * 0.05)
                .collect(),
        },
        layers: vec![layer(0.0), layer(13.0)],
        final_layer_norm_weight: ones.clone(),
        final_layer_norm_bias: zeros.clone(),
        text_projection: None,
        hidden_size: HIDDEN,
        max_length: 3,
        n_heads: HEADS,
    };
    let tokens = [2u32, 0, 3];
    let cpu = encoder.encode_tokens(&tokens).unwrap();
    assert!(cpu.data.iter().all(|value| value.is_finite()));
    // The layers are non-identity: the encoded output must differ from a bare
    // embedding + final-norm, confirming attention/MLP actually contributed.
    assert!(cpu.data.iter().any(|value| value.abs() > 0.01));

    if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
        eprintln!("skip: ROCm GPU unavailable for resident CLIP encode test: {error}");
    } else {
        let resident = encoder
            .encode_tokens_with_runtime_options(
                &tokens,
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            )
            .unwrap();
        assert_eq!(resident.shape, cpu.shape);
        // The encoder linears now run through the F16 WMMA GEMM (Phase 3), so
        // match the F32 reference to F16 tolerance, not 1e-4.
        assert!(
            f32_slices_close(&resident.data, &cpu.data, 5e-3),
            "resident CLIP encode {:?} != cpu reference {:?}",
            resident.data,
            cpu.data
        );
    }
}

#[test]
fn hip_clip_token_position_embeddings_match_cpu_reference() {
    let token_embedding = CpuTensor {
        shape: vec![4, 5],
        data: (0..20).map(|idx| idx as f32 / 13.0 - 0.6).collect(),
    };
    let position_embedding = CpuTensor {
        shape: vec![3, 5],
        data: (0..15).map(|idx| (idx as f32 % 7.0 - 3.0) / 11.0).collect(),
    };
    let tokens = [3, 0, 2];
    let cpu =
        clip_token_position_embeddings(&token_embedding, &position_embedding, &tokens).unwrap();
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: ROCm GPU unavailable for CLIP embedding routing test: {error}");
            return;
        }
    };
    let hip = clip_token_position_embeddings_hip_on_gpu(
        &mut gpu,
        &token_embedding,
        &position_embedding,
        &tokens,
    )
    .unwrap();

    assert_eq!(hip.shape, cpu.shape);
    assert!(f32_slices_close(&hip.data, &cpu.data, 1e-6));
}
