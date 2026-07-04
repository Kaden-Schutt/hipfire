#![allow(unused_imports)]
use super::*;
use std::collections::BTreeMap;
use std::path::PathBuf;
// Import tooling now lives in the offline hipfire-diffusion-coexist crate.
use hipfire_diffusion_coexist::{
    import_diffusers_to_hfq, ldm_unet_native_tensor_name, ldm_vae_native_tensor_name,
    parse_pytorch_state_dict, pytorch_tensor_is_contiguous, reorder_pytorch_storage_to_contiguous,
    DiffusersImportOptions,
};
use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
use std::fs;
use super::*;

#[test]
fn scaled_dot_product_attention_respects_key_mask() {
    let q = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![1.0, 0.0],
    };
    let k = CpuTensor {
        shape: vec![1, 2, 2],
        data: vec![10.0, 0.0, 0.0, 10.0],
    };
    let v = CpuTensor {
        shape: vec![1, 2, 2],
        data: vec![3.0, 5.0, 7.0, 11.0],
    };

    let out =
        scaled_dot_product_attention_with_key_mask(&q, &k, &v, 1, Some(&[false, true])).unwrap();

    assert_eq!(out.shape, vec![1, 1, 2]);
    assert_f32_close(&out.data, &[7.0, 11.0], 1e-6);
}

#[test]
fn rejects_non_diffusion_metadata() {
    let err = parse_diffusion_metadata(
        r#"{"artifact_kind":"llm","schema_version":1,"pipeline":{"class_name":"x","source":"x"}}"#,
    )
    .unwrap_err();
    assert!(err.to_string().contains("artifact_kind"));
}

#[test]
fn lenient_config_json_accepts_diffusers_non_finite_tokens() {
    let parsed = parse_json_lenient(
        r#"{"_class_name":"DPMSolverMultistepScheduler","lambda_min_clipped":-Infinity}"#,
    )
    .unwrap();
    assert_eq!(
        parsed.get("_class_name").and_then(Value::as_str),
        Some("DPMSolverMultistepScheduler")
    );
    assert!(parsed.get("lambda_min_clipped").unwrap().is_null());
}

#[test]
fn validates_batched_request_limits() {
    let metadata = minimal_metadata();
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
        width: 512,
        height: 512,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: 20,
        cfg_scale: 7.0,
        distilled_guidance_scale: None,
        scheduler: "DPM++ 2M".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };
    assert!(validate_batch_request(&metadata, &request).is_ok());

    let mut distilled_guidance_request = request.clone();
    distilled_guidance_request.distilled_guidance_scale = Some(4.0);
    let err = validate_batch_request(&metadata, &distilled_guidance_request).unwrap_err();
    assert!(err.to_string().contains("must not be silently ignored"));
}

#[test]
fn metadata_only_import_skips_weights_and_reports_non_runnable_artifact() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-metadata-only-import-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let source = dir.join("snapshot");
    fs::create_dir_all(source.join("transformer")).unwrap();
    fs::create_dir_all(source.join("vae")).unwrap();
    fs::create_dir_all(source.join("scheduler")).unwrap();
    fs::write(
            source.join("model_index.json"),
            br#"{"_class_name":"QwenImagePipeline","transformer":["diffusers","QwenImageTransformer2DModel"],"vae":["diffusers","AutoencoderKLQwenImage"],"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"]}"#,
        )
        .unwrap();
    fs::write(
        source.join("transformer/config.json"),
        br#"{"_class_name":"QwenImageTransformer2DModel","in_channels":64,"out_channels":16}"#,
    )
    .unwrap();
    fs::write(
        source.join("transformer/diffusion_pytorch_model.safetensors.index.json"),
        br#"{"metadata":{"total_size":4},"weight_map":{"x":"missing.safetensors"}}"#,
    )
    .unwrap();
    fs::write(
        source.join("vae/config.json"),
        br#"{"_class_name":"AutoencoderKLQwenImage","z_dim":16}"#,
    )
    .unwrap();
    fs::write(
            source.join("scheduler/scheduler_config.json"),
            br#"{"_class_name":"FlowMatchEulerDiscreteScheduler","num_train_timesteps":1000,"shift":1.0}"#,
        )
        .unwrap();

    let output = dir.join("qwen-metadata.hfq");
    let summary = import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: Some("qwen-image".into()),
        max_batch: 1,
        metadata_only: true,
    })
    .unwrap();
    let hfq = HfqFile::open_index_only(&output).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let inspection = inspect_hfq_with_runtime_support(&output).unwrap();

    assert_eq!(summary.weight_format, "metadata-only");
    assert_eq!(metadata.quantization.weight_format, "metadata-only");
    assert_eq!(metadata.pipeline.latent_channels, Some(16));
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    let transformer = config.transformer.as_ref().unwrap();
    assert_eq!(transformer.class_name, "QwenImageTransformer2DModel");
    assert_eq!(transformer.in_channels, Some(64));
    assert_eq!(transformer.out_channels, Some(16));
    assert!(metadata.components["transformer"].weight_entries.is_empty());
    assert_eq!(
        metadata.components["transformer"].config_entry.as_deref(),
        Some("transformer/config.json")
    );
    assert!(!inspection.runtime_support.supported);
    assert!(inspection
        .runtime_support
        .reason
        .as_deref()
        .unwrap()
        .contains("metadata only"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn ldm_unet_native_tensor_name_maps_standard_sd_blocks() {
    let cases = [
        (
            "input_blocks.0.0.weight",
            Some("conv_in.weight".to_string()),
        ),
        (
            "input_blocks.2.0.emb_layers.1.bias",
            Some("down_blocks.0.resnets.1.time_emb_proj.bias".to_string()),
        ),
        (
            "input_blocks.6.0.op.bias",
            Some("down_blocks.1.downsamplers.0.conv.bias".to_string()),
        ),
        (
            "middle_block.2.skip_connection.weight",
            Some("mid_block.resnets.1.conv_shortcut.weight".to_string()),
        ),
        (
            "output_blocks.4.0.out_layers.3.weight",
            Some("up_blocks.1.resnets.1.conv2.weight".to_string()),
        ),
        (
            "output_blocks.5.1.op.bias",
            Some("up_blocks.1.upsamplers.0.conv.bias".to_string()),
        ),
        ("input_blocks.3.1.norm.weight", None),
    ];

    for (input, expected) in cases {
        assert_eq!(ldm_unet_native_tensor_name(input), expected, "{input}");
    }
}

#[test]
fn ldm_vae_native_tensor_name_maps_standard_sd_blocks() {
    let cases = [
        (
            "encoder.down.0.block.1.norm1.weight",
            Some("encoder.down_blocks.0.resnets.1.norm1.weight".to_string()),
        ),
        (
            "encoder.down.2.downsample.conv.bias",
            Some("encoder.down_blocks.2.downsamplers.0.conv.bias".to_string()),
        ),
        (
            "encoder.mid.attn_1.proj_out.weight",
            Some("encoder.mid_block.attentions.0.to_out.0.weight".to_string()),
        ),
        (
            "decoder.mid.block_2.nin_shortcut.bias",
            Some("decoder.mid_block.resnets.1.conv_shortcut.bias".to_string()),
        ),
        (
            "decoder.up.3.block.0.conv2.weight",
            Some("decoder.up_blocks.0.resnets.0.conv2.weight".to_string()),
        ),
        (
            "decoder.up.1.upsample.conv.weight",
            Some("decoder.up_blocks.2.upsamplers.0.conv.weight".to_string()),
        ),
        (
            "decoder.norm_out.bias",
            Some("decoder.conv_norm_out.bias".to_string()),
        ),
        ("decoder.up.4.block.0.norm1.weight", None),
    ];

    for (input, expected) in cases {
        assert_eq!(ldm_vae_native_tensor_name(input), expected, "{input}");
    }
}

#[test]
fn cpu_linear_layer_norm_and_softmax_are_stable() {
    let input = CpuTensor {
        shape: vec![2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let weight = CpuTensor {
        shape: vec![2, 2],
        data: vec![1.0, 0.0, 0.0, 1.0],
    };
    let bias = CpuTensor {
        shape: vec![2],
        data: vec![0.5, -0.5],
    };
    let out = linear(&input, &weight, &bias).unwrap();
    assert_eq!(out.data, vec![1.5, 1.5, 3.5, 3.5]);

    let norm_weight = CpuTensor {
        shape: vec![2],
        data: vec![1.0, 1.0],
    };
    let norm_bias = CpuTensor {
        shape: vec![2],
        data: vec![0.0, 0.0],
    };
    let normed = layer_norm(&input, &norm_weight, &norm_bias, 1e-5).unwrap();
    assert!(normed.data[0] < -0.99 && normed.data[1] > 0.99);

    let mut logits = vec![1.0, 2.0, 3.0];
    softmax_in_place(&mut logits);
    let sum = logits.iter().sum::<f32>();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(logits[2] > logits[1] && logits[1] > logits[0]);

    assert_eq!(quick_gelu(0.0), 0.0);
    assert!((quick_gelu(1.0) - 0.845795).abs() < 1e-5);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for QuickGELU routing test: {error}");
        } else {
            let cpu = tensor_map(&input, quick_gelu);
            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let hip = quick_gelu_with_runtime_context(&input, &mut runtime_context).unwrap();
            assert_eq!(hip.shape, cpu.shape);
            assert!(f32_slices_close(&hip.data, &cpu.data, 1e-6));
        }
    }
}

#[test]
fn seed_resize_generates_source_latents_and_resizes_to_target_shape() {
    let config = StableDiffusionConfig {
        pipeline_class: "StableDiffusionPipeline".into(),
        text_encoder: TextEncoderConfig::default(),
        text_encoder_2: None,
        unet: UnetConfig::default(),
        transformer: None,
        vae: VaeConfig::default(),
        scheduler: SchedulerConfig::default(),
        latent_channels: 1,
        latent_height: None,
        latent_width: None,
        vae_scale_factor: 1,
    };
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![DiffusionPrompt {
            prompt: "a".into(),
            negative_prompt: String::new(),
            seed: 123,
            subseed: None,
        }],
        width: 2,
        height: 2,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: Some(1),
        seed_resize_from_height: Some(1),
        crop_x: 0,
        crop_y: 0,
        steps: 2,
        cfg_scale: 7.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };
    let latent_shape = latent_shape_for_request(&config, &request).unwrap();

    let resized = seeded_latents_for_request(&config, &request, &latent_shape, &[123]).unwrap();
    let source = LatentBatch::seeded_normal(1, 1, 1, 1, &[123]);
    let direct = LatentBatch::seeded_normal(1, 1, 2, 2, &[123]);

    assert_eq!(resized, resize_latent_batch_nearest(&source, 2, 2).unwrap());
    assert_ne!(resized, direct);
}

#[test]
fn subseed_strength_blends_only_prompt_latents_with_subseeds() {
    let mut latents = LatentBatch::seeded_normal(2, 1, 1, 2, &[10, 20]);
    let original = latents.clone();
    let subseed = LatentBatch::seeded_normal(2, 1, 1, 2, &[30, 20]);
    let config = StableDiffusionConfig {
        pipeline_class: "StableDiffusionPipeline".into(),
        text_encoder: TextEncoderConfig::default(),
        text_encoder_2: None,
        unet: UnetConfig::default(),
        transformer: None,
        vae: VaeConfig::default(),
        scheduler: SchedulerConfig::default(),
        latent_channels: 1,
        latent_height: None,
        latent_width: None,
        vae_scale_factor: 1,
    };
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![
            DiffusionPrompt {
                prompt: "a".into(),
                negative_prompt: String::new(),
                seed: 10,
                subseed: Some(30),
            },
            DiffusionPrompt {
                prompt: "b".into(),
                negative_prompt: String::new(),
                seed: 20,
                subseed: None,
            },
        ],
        width: 2,
        height: 1,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: 2,
        cfg_scale: 7.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.25,
        send_images: true,
        save_images: false,
    };
    let latent_shape = latent_shape_for_request(&config, &request).unwrap();

    blend_subseed_latents(&config, &mut latents, &request, &latent_shape).unwrap();

    assert_eq!(latents.batch, 2);
    for idx in 0..2 {
        let expected = original.data[idx] * 0.75 + subseed.data[idx] * 0.25;
        assert!((latents.data[idx] - expected).abs() < 1e-6);
    }
    assert_eq!(&latents.data[2..], &original.data[2..]);
}

#[test]
fn linear_scheduler_euler_step_moves_toward_next_sigma() {
    let schedule = DiffusionSchedule::linear(2).unwrap();
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2,
        data: vec![1.0, -1.0],
    };

    schedule.euler_step(&mut latents, &[0.25, -0.5], 0).unwrap();

    assert_eq!(schedule.timesteps, vec![1.0, 0.0]);
    assert_eq!(schedule.sigmas, vec![1.0, 0.0, 0.0]);
    assert_eq!(latents.data, vec![0.75, -0.5]);
}

#[test]
fn flow_match_euler_scheduler_uses_shifted_sigmas_and_terminal_rescale() {
    let config = SchedulerConfig {
        class_name: "FlowMatchEulerDiscreteScheduler".into(),
        num_train_timesteps: Some(1000),
        shift: Some(1.0),
        shift_terminal: Some(0.02),
        invert_sigmas: Some(false),
        ..SchedulerConfig::default()
    };

    let schedule = DiffusionSchedule::from_config(&config, 3).unwrap();

    assert_eq!(schedule.solver, SchedulerSolver::FlowMatchEuler);
    assert_eq!(schedule.input_scaling, SchedulerInputScaling::None);
    assert_eq!(schedule.sigmas, vec![1.0, 0.51, 0.02, 0.0]);
    assert_eq!(schedule.timesteps, vec![1000.0, 510.0, 20.0]);
    assert_eq!(schedule.initial_noise_sigma(), 1.0);
}

#[test]
fn flow_match_euler_step_uses_model_output_as_velocity() {
    let config = SchedulerConfig {
        class_name: "FlowMatchEulerDiscreteScheduler".into(),
        num_train_timesteps: Some(1000),
        shift: Some(1.0),
        ..SchedulerConfig::default()
    };
    let schedule = DiffusionSchedule::from_config(&config, 2).unwrap();
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2,
        data: vec![1.0, -1.0],
    };
    let mut state = SchedulerStepState::default();

    schedule
        .step(&mut latents, &[0.25, -0.5], 0, &mut state)
        .unwrap();

    assert_eq!(schedule.sigmas, vec![1.0, 0.0, 0.0]);
    assert_eq!(latents.data, vec![0.75, -0.5]);
}

#[test]
fn karras_scheduler_uses_power_law_sigmas_and_nearest_train_timesteps() {
    let mut config = SchedulerConfig {
        class_name: "DPMSolverMultistepScheduler".into(),
        beta_start: Some(0.00085),
        beta_end: Some(0.012),
        beta_schedule: Some("scaled_linear".into()),
        num_train_timesteps: Some(1000),
        prediction_type: Some("epsilon".into()),
        algorithm_type: Some("dpmsolver++".into()),
        solver_order: Some(2),
        solver_type: Some("midpoint".into()),
        lower_order_final: Some(true),
        thresholding: Some(false),
        timestep_spacing: Some("linspace".into()),
        steps_offset: Some(1),
        use_karras_sigmas: Some(false),
        set_alpha_to_one: None,
        ..SchedulerConfig::default()
    };
    let normal = DiffusionSchedule::from_config(&config, 4).unwrap();
    config.use_karras_sigmas = Some(true);

    let karras = DiffusionSchedule::from_config(&config, 4).unwrap();

    assert_eq!(karras.sigmas.len(), 5);
    assert!((karras.sigmas[0] - normal.sigmas[0]).abs() / normal.sigmas[0].max(1.0) < 1e-4);
    assert_eq!(karras.sigmas[4], 0.0);
    assert!(karras.sigmas[0] > karras.sigmas[1]);
    assert!(karras.sigmas[1] > karras.sigmas[2]);
    assert!(karras.sigmas[2] > karras.sigmas[3]);
    assert_ne!(karras.sigmas, normal.sigmas);
    assert_eq!(karras.train_timesteps.len(), 4);
    assert!(karras
        .train_timesteps
        .windows(2)
        .all(|pair| pair[0] >= pair[1]));
}

#[test]
fn ddim_scheduler_step_matches_deterministic_epsilon_update() {
    let schedule = DiffusionSchedule {
        timesteps: vec![2.0, 1.0],
        sigmas: vec![0.8, 0.6, 0.0],
        prediction_type: SchedulerPredictionType::Epsilon,
        input_scaling: SchedulerInputScaling::None,
        solver: SchedulerSolver::Ddim {
            set_alpha_to_one: true,
        },
        train_timesteps: vec![2, 1],
        alpha_t: vec![1.0, 0.8, 0.6],
        sigma_t: vec![0.0, 0.6, 0.8],
        lambda_t: Vec::new(),
    };
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
        data: vec![1.4],
    };
    let mut state = SchedulerStepState::default();

    schedule.step(&mut latents, &[0.5], 0, &mut state).unwrap();

    let pred_original = (1.4 - 0.8 * 0.5) / 0.6;
    let expected = 0.8 * pred_original + 0.6 * 0.5;
    assert!((latents.data[0] - expected).abs() < 1e-6);
}

#[test]
fn timestep_embedding_matches_diffusers_ordering_flags() {
    let flipped = timestep_embedding(&[0.0], 4, true, 0.0).unwrap();
    assert_eq!(flipped.shape, vec![1, 4]);
    assert_eq!(flipped.data, vec![1.0, 1.0, 0.0, 0.0]);

    let unflipped = timestep_embedding(&[0.0], 4, false, 0.0).unwrap();
    assert_eq!(unflipped.data, vec![0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn sdxl_time_ids_default_to_requested_size_and_crop() {
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
        width: 768,
        height: 512,
        original_width: Some(1024),
        original_height: Some(768),
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 8,
        crop_y: 16,
        steps: 1,
        cfg_scale: 7.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: false,
        save_images: false,
    };

    let time_ids = sdxl_time_ids_for_request(&request).unwrap();

    assert_eq!(time_ids.shape, vec![2, 6]);
    assert_eq!(
        time_ids.data,
        vec![
            768.0, 1024.0, 16.0, 8.0, 512.0, 768.0, //
            768.0, 1024.0, 16.0, 8.0, 512.0, 768.0,
        ]
    );
}

#[test]
fn conv2d_groupnorm_silu_and_upsample_primitives_work() {
    let input = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let weight = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, 0.0, 0.0, -1.0],
    };
    let bias = CpuTensor {
        shape: vec![1],
        data: vec![0.5],
    };
    let conv = conv2d_nchw(&input, &weight, Some(&bias), 0).unwrap();
    assert_eq!(conv.shape, vec![1, 1, 1, 1]);
    assert_eq!(conv.data, vec![-2.5]);

    let padded = conv2d_nchw(&input, &weight, None, 1).unwrap();
    assert_eq!(padded.shape, vec![1, 1, 3, 3]);
    assert_eq!(padded.data[0], -1.0);

    let gn_input = CpuTensor {
        shape: vec![1, 2, 1, 2],
        data: vec![1.0, 3.0, 10.0, 14.0],
    };
    let affine = CpuTensor {
        shape: vec![2],
        data: vec![1.0, 1.0],
    };
    let zeros = CpuTensor {
        shape: vec![2],
        data: vec![0.0, 0.0],
    };
    let normed = group_norm_nchw(&gn_input, &affine, &zeros, 2, 1e-5).unwrap();
    assert!(normed.data[0] < -0.99 && normed.data[1] > 0.99);
    assert!(normed.data[2] < -0.99 && normed.data[3] > 0.99);

    assert!((silu(1.0) - 0.7310586).abs() < 1e-6);

    let up = upsample_nearest2d_nchw(&input, 2).unwrap();
    assert_eq!(up.shape, vec![1, 1, 4, 4]);
    assert_eq!(&up.data[0..4], &[1.0, 1.0, 2.0, 2.0]);
}

#[test]
fn resnet_block_loads_from_hfq_and_preserves_residual_shape() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-resnet-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("resnet.hfq");
    let prefix = "vae/tensors/decoder.up_blocks.0.resnets.0";
    let metadata = minimal_metadata();
    let tensors = [
        f32_mem_tensor(&format!("{prefix}.norm1.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{prefix}.norm1.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{prefix}.conv1.weight"), &[1, 1, 3, 3], &[0.0; 9]),
        f32_mem_tensor(&format!("{prefix}.conv1.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{prefix}.norm2.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{prefix}.norm2.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{prefix}.conv2.weight"), &[1, 1, 3, 3], &[0.0; 9]),
        f32_mem_tensor(&format!("{prefix}.conv2.bias"), &[1], &[0.0]),
    ];
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
    let block = ResnetBlock2D::from_hfq(&hfq, prefix, 1).unwrap();
    let input = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let output = block.forward(&input).unwrap();
    assert_eq!(output.shape, input.shape);
    assert_eq!(output.data, input.data);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for VAE ResNet context test: {error}");
        } else {
            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let hip = block
                .forward_with_runtime_context(&input, &mut runtime_context)
                .unwrap();
            assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
            assert_eq!(hip.shape, output.shape);
            assert!(f32_slices_close(&hip.data, &output.data, 1e-5));
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn pytorch_contiguous_detection_matches_torch_semantics() {
    // Standard contiguous OIHW conv weight.
    let shape = [2u32, 3, 3, 3];
    let contiguous = [27i64, 9, 3, 1];
    assert!(pytorch_tensor_is_contiguous(&shape, &contiguous));
    // channels_last (OHWI physical order) carries OIHW size with permuted
    // strides; this must be detected as non-contiguous.
    let channels_last = [27i64, 1, 9, 3];
    assert!(!pytorch_tensor_is_contiguous(&shape, &channels_last));
    // Size-1 dims carry arbitrary strides and must not break detection.
    let shape1 = [4u32, 1, 1, 1];
    assert!(pytorch_tensor_is_contiguous(&shape1, &[1i64, 1, 1, 1]));
    assert!(pytorch_tensor_is_contiguous(&shape1, &[1i64, 4, 4, 4]));
    // Missing/empty stride metadata is treated as already contiguous.
    assert!(pytorch_tensor_is_contiguous(&shape, &[]));
}

#[test]
fn channels_last_storage_reorders_to_contiguous_oihw() {
    // Logical OIHW reference values (row-major) we want to recover.
    let (o, i, h, w) = (2usize, 3, 2, 2);
    let oihw: Vec<f32> = (0..(o * i * h * w)).map(|v| v as f32).collect();
    // Build the physical channels_last storage: OHWI element order.
    let mut storage_f32 = vec![0f32; o * i * h * w];
    let mut p = 0usize;
    for oo in 0..o {
        for hh in 0..h {
            for ww in 0..w {
                for ii in 0..i {
                    let logical = ((oo * i + ii) * h + hh) * w + ww;
                    storage_f32[p] = oihw[logical];
                    p += 1;
                }
            }
        }
    }
    let storage: Vec<u8> = storage_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let shape = [o as u32, i as u32, h as u32, w as u32];
    // channels_last strides for an OIHW-sized tensor.
    let stride = [(i * h * w) as i64, 1, (i * w) as i64, i as i64];
    assert!(!pytorch_tensor_is_contiguous(&shape, &stride));
    let bytes = reorder_pytorch_storage_to_contiguous(&storage, &shape, &stride, 0, 4).unwrap();
    let recovered: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(recovered, oihw);
}

/// Quantization fidelity harness (env-gated; not part of normal CI).
///
/// `HIPFIRE_QUANT_SRC=<f16 source.hfq>` and
/// `HIPFIRE_QUANT_CANDS=path1=label1,path2=label2,...` enable it. For each
/// candidate it reports two metrics that — unlike image-space PSNR against a
/// reference image — are NOT confounded by the chaotic multi-step denoise
/// trajectory:
///   (1) global weight SQNR vs the source (the encoder's direct objective),
///   (2) single-pass UNet eps error at a fixed deterministic input (the
///       functional error of the quantized weights, no trajectory amplification).
/// Validate a calibration sidecar (env `HIPFIRE_QUANT_CALIB`): every Hessian
/// must be readable, symmetric, and PSD (non-negative diagonal). Confirms the
/// Phase-1 diffusion CPU collector writes a sidecar the quantizer's
/// `HessianSidecar` reader (in hipfire-quantize) consumes correctly.
#[test]
fn calib_sidecar_is_valid() {
    use hipfire_quantize::hessian_io::HessianSidecar;
    let Ok(path) = std::env::var("HIPFIRE_QUANT_CALIB") else {
        return;
    };
    let sc = HessianSidecar::open(std::path::Path::new(&path)).unwrap();
    let (mut hessians, mut imatrices) = (0usize, 0usize);
    for h in sc.tensors() {
        HessianSidecar::check_symmetry(&h, 1e-4).unwrap();
        HessianSidecar::check_positive_diagonal(&h).unwrap();
        assert_eq!(h.k % 256, 0, "{}: K not 256-aligned", h.name);
        hessians += 1;
    }
    for im in sc.imatrices() {
        assert!(
            im.iter_f32().all(|v| v >= 0.0),
            "{}: negative imatrix",
            im.name
        );
        imatrices += 1;
    }
    eprintln!("[calib-valid] hessians={hessians} imatrices={imatrices} (all symmetric+PSD)");
    assert!(hessians > 0 && imatrices > 0);
}

#[test]
fn quant_fidelity_report() {
    let Ok(src_path) = std::env::var("HIPFIRE_QUANT_SRC") else {
        return;
    };
    let Ok(cands) = std::env::var("HIPFIRE_QUANT_CANDS") else {
        return;
    };
    let src = HfqFile::open(std::path::Path::new(&src_path)).unwrap();
    let weight_names: Vec<String> = src
        .tensors()
        .iter()
        .filter(|t| t.name.ends_with(".weight") && t.shape.len() >= 2)
        .map(|t| t.name.clone())
        .collect();

    // Deterministic UNet input (matches the diffusers reference harness).
    let sample = CpuTensor {
        shape: vec![1, 4, 32, 32],
        data: (0..4 * 32 * 32)
            .map(|i| (0.1 * ((i % 97) as f32)).sin())
            .collect(),
    };
    let enc = CpuTensor {
        shape: vec![1, 77, 768],
        data: (0..77 * 768)
            .map(|i| (0.1 * ((i % 89) as f32)).cos())
            .collect(),
    };
    let run_unet = |hfq: &HfqFile| -> Vec<f32> {
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let config = StableDiffusionConfig::from_hfq(hfq, &metadata).unwrap();
        let unet = NativeUnet2DConditionModel::from_hfq(hfq, &config.unet).unwrap();
        unet.forward_with_runtime_options(
            &sample,
            &[999.0],
            &enc,
            DiffusionGenerationRuntimeOptions::cpu_reference(),
        )
        .unwrap()
        .data
    };
    let src_eps = run_unet(&src);

    for spec in cands.split(',') {
        let (path, label) = spec.split_once('=').unwrap_or((spec, spec));
        let cand = HfqFile::open(std::path::Path::new(path)).unwrap();
        // (1) weight SQNR vs source, aggregated over all weight tensors.
        let (mut sig, mut noise) = (0.0f64, 0.0f64);
        for name in &weight_names {
            let a = cpu_tensor_from_hfq(&src, name).unwrap().data;
            let b = cpu_tensor_from_hfq(&cand, name).unwrap().data;
            for (x, y) in a.iter().zip(b.iter()) {
                sig += (*x as f64) * (*x as f64);
                noise += ((*x - *y) as f64) * ((*x - *y) as f64);
            }
        }
        let sqnr = if noise > 0.0 {
            10.0 * (sig / noise).log10()
        } else {
            f64::INFINITY
        };
        // (2) single-pass eps functional error vs source.
        let cand_eps = run_unet(&cand);
        let (mut dot, mut na, mut nb, mut err) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (x, y) in src_eps.iter().zip(cand_eps.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
            err += ((*x - *y) as f64) * ((*x - *y) as f64);
        }
        let corr = dot / (na.sqrt() * nb.sqrt());
        let rel_l2 = (err / na).sqrt();
        eprintln!(
                "[quant-fidelity] {label:12}: weight_SQNR={sqnr:6.2} dB | eps_corr={corr:.5} eps_relL2={rel_l2:.4}"
            );
    }
}

#[test]
fn q4f16_g64_encoder_round_trips_through_decoder() {
    let data: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) * 0.02).collect();
    let bytes = encode_q4f16_g64(&data);
    assert_eq!(bytes.len(), data.len().div_ceil(64) * 36);
    let decoded = decode_q4f16_g64_slice("t", &bytes, data.len()).unwrap();
    // 4-bit affine over each 64-group: error bounded by half a (range/15) step.
    for group in data.chunks(64) {
        let min = group.iter().copied().fold(f32::INFINITY, f32::min);
        let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let step = ((max - min) / 15.0).max(1e-6);
        let base = data
            .iter()
            .position(|v| (*v - group[0]).abs() < 1e-12)
            .unwrap();
        for (k, &orig) in group.iter().enumerate() {
            assert!((decoded[base + k] - orig).abs() <= step * 0.5 + 1e-2);
        }
    }
}

#[test]
fn cpu_reference_env_toggle_defaults_to_gpu() {
    // Unset / falsy values keep the ROCm (GPU) default.
    assert!(!cpu_reference_env_enabled(None));
    assert!(!cpu_reference_env_enabled(Some("")));
    assert!(!cpu_reference_env_enabled(Some("0")));
    assert!(!cpu_reference_env_enabled(Some("false")));
    assert!(!cpu_reference_env_enabled(Some(" No ")));
    // Any other value opts into the CPU reference oracle.
    assert!(cpu_reference_env_enabled(Some("1")));
    assert!(cpu_reference_env_enabled(Some("true")));
    assert!(cpu_reference_env_enabled(Some("yes")));
}

#[test]
fn for_device_uses_rocm_by_default() {
    // Without the env opt-in, for_device targets the resolved GPU.
    assert_eq!(
        DiffusionGenerationRuntimeOptions::rocm_hybrid(2),
        DiffusionGenerationRuntimeOptions {
            rocm_device_id: Some(2)
        }
    );
}

#[test]
fn masked_rgb_batch_for_inpaint_zeroes_white_mask_pixels() {
    let image = RgbImageBatch {
        batch: 1,
        width: 2,
        height: 1,
        data: vec![10, 20, 30, 100, 120, 140],
    };
    let mask = RgbImageBatch {
        batch: 1,
        width: 2,
        height: 1,
        data: vec![0, 0, 0, 255, 255, 255],
    };

    let masked = masked_rgb_batch_for_inpaint(&image, &mask).unwrap();

    assert_eq!(masked.data, vec![10, 20, 30, 0, 0, 0]);
}

#[test]
fn append_inpaint_conditioning_concatenates_latents_mask_and_masked_latents() {
    let sample = CpuTensor {
        shape: vec![1, 2, 1, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let conditioning = InpaintDenoiseConditioning {
        mask_weights: vec![0.25, 0.75],
        masked_image_latents: LatentBatch {
            batch: 1,
            channels: 2,
            height: 1,
            width: 2,
            data: vec![5.0, 6.0, 7.0, 8.0],
        },
    };

    let conditioned = append_inpaint_conditioning(&sample, &conditioning).unwrap();

    assert_eq!(conditioned.shape, vec![1, 5, 1, 2]);
    assert_eq!(
        conditioned.data,
        vec![1.0, 2.0, 3.0, 4.0, 0.25, 0.75, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn masked_denoise_reference_reprojects_noised_init_latents_per_step() {
    let source_schedule = DiffusionSchedule::linear(3).unwrap();
    let init = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2,
        data: vec![10.0, 20.0],
    };
    let noise = vec![2.0, 4.0];
    let mut generated = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2,
        data: vec![100.0, 200.0],
    };
    let reference = MaskedDenoiseReference {
        init_latents: &init,
        noise: &noise,
        mask_weights: &[0.0, 1.0],
        source_schedule: &source_schedule,
        start_step: 0,
    };

    apply_masked_denoise_reference(&mut generated, &reference, 0).unwrap();

    assert_eq!(generated.data, vec![11.0, 200.0]);
}

#[test]
fn diffusion_pipeline_generate_batch_returns_sdapi_png_images_with_test_backend() {
    let metadata = tiny_runtime_metadata();
    let config = tiny_runtime_config();
    let tokenizer = ClipTokenizer::from_bytes(
        br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "cat</w>": 3
            }"#,
        b"#version: 0.2\n",
        4,
    )
    .unwrap();
    let text_encoder = ClipTextEncoder {
        token_embedding: CpuTensor {
            shape: vec![4, 2],
            data: vec![0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
        },
        position_embedding: CpuTensor {
            shape: vec![4, 2],
            data: vec![0.0; 8],
        },
        layers: Vec::new(),
        final_layer_norm_weight: CpuTensor {
            shape: vec![2],
            data: vec![1.0, 1.0],
        },
        final_layer_norm_bias: CpuTensor {
            shape: vec![2],
            data: vec![0.0, 0.0],
        },
        text_projection: None,
        hidden_size: 2,
        max_length: 4,
        n_heads: 1,
    };
    let pipeline = DiffusionPipeline {
        summary: summarize_hfq(Path::new("/tmp/tiny-runtime.hfq"), &metadata),
        metadata,
        config,
        tokenizer: Some(tokenizer),
        tokenizer_2: None,
        text_encoder: Some(text_encoder),
        text_encoder_2: None,
        native_runtime: Some(NativeDiffusionRuntime {
            kind: DiffusionRuntimeKind::CpuSourceReference,
            noise: Box::new(TestNoiseBackend),
            encoder: None,
            decoder: Box::new(TestImageDecoder),
            text_conditioner: None,
            krea2_tokenizer: None,
        }),
        native_runtime_error: None,
    };
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![
            DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 7,
                subseed: None,
            },
            DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: "blur".into(),
                seed: 8,
                subseed: None,
            },
        ],
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
        steps: 2,
        cfg_scale: 7.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };

    let mut progress_events = Vec::new();
    let output = pipeline
        .generate_batch_with_progress(request, &mut |progress| {
            progress_events.push(progress);
            Ok(())
        })
        .unwrap();

    assert_eq!(output.images.len(), 2);
    assert_eq!(progress_events.len(), 2);
    assert_eq!(progress_events[0].completed_steps, 1);
    assert_eq!(progress_events[0].total_steps, 2);
    let first_preview = progress_events[0].preview_latents.as_ref().unwrap();
    assert_eq!(first_preview.batch, 2);
    assert_eq!(first_preview.channels, 1);
    assert_eq!(first_preview.height, 2);
    assert_eq!(first_preview.width, 2);
    assert_eq!(progress_events[1].completed_steps, 2);
    assert_eq!(progress_events[1].total_steps, 2);
    assert!(progress_events[1].preview_latents.is_some());
    assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
    assert_eq!(output.info["runtime"], "cpu-source-reference");
    assert_eq!(output.info["latent_shape"]["batch"], 2);
    let capabilities = pipeline.runtime_capabilities().unwrap();
    assert_eq!(capabilities.kind, DiffusionRuntimeKind::CpuSourceReference);
    assert_eq!(capabilities.weight_format, "source");
    assert!(!capabilities.supports_img2img);
    for image in output.images {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(image)
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
    }
}

#[test]
fn generate_batch_runtime_options_surface_rocm_hybrid_runtime_when_gpu_is_available() {
    if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
        eprintln!("skip: ROCm GPU unavailable for hybrid generation test: {error}");
        return;
    }
    let pipeline = tiny_txt2img_test_pipeline(Box::new(SolidTensorImageDecoder));
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![DiffusionPrompt {
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            seed: 7,
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
        steps: 2,
        cfg_scale: 7.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };

    let output = pipeline
        .generate_batch_with_runtime_options(
            request,
            DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
        )
        .unwrap();

    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["runtime"], "rocm-hybrid-reference");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    assert_eq!(decoded.get_pixel(0, 0).0, [32, 128, 224]);
}

#[test]
fn diffusion_pipeline_prepares_secondary_clip_conditioning_when_available() {
    let mut metadata = tiny_runtime_metadata();
    metadata.pipeline.class_name = "StableDiffusionXLPipeline".into();
    metadata.tokenizer_2 = Some(DiffusionTokenizerMetadata {
        kind: "clip-bpe".into(),
        max_length: Some(4),
        entries: vec!["tokenizer_2/vocab.json".into()],
    });
    let mut config = tiny_runtime_config();
    config.pipeline_class = "StableDiffusionXLPipeline".into();
    config.text_encoder_2 = Some(TextEncoderConfig {
        class_name: "CLIPTextModelWithProjection".into(),
        hidden_size: Some(2),
        intermediate_size: Some(4),
        num_hidden_layers: Some(0),
        num_attention_heads: Some(1),
        max_position_embeddings: Some(4),
        vocab_size: Some(4),
    });
    let tokenizer = ClipTokenizer::from_bytes(
        br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "cat</w>": 3
            }"#,
        b"#version: 0.2\n",
        4,
    )
    .unwrap();
    let text_encoder = ClipTextEncoder {
        token_embedding: CpuTensor {
            shape: vec![4, 2],
            data: vec![0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
        },
        position_embedding: CpuTensor {
            shape: vec![4, 2],
            data: vec![0.0; 8],
        },
        layers: Vec::new(),
        final_layer_norm_weight: CpuTensor {
            shape: vec![2],
            data: vec![1.0, 1.0],
        },
        final_layer_norm_bias: CpuTensor {
            shape: vec![2],
            data: vec![0.0, 0.0],
        },
        text_projection: Some(CpuTensor {
            shape: vec![2, 2],
            data: vec![1.0, 0.0, 0.0, 1.0],
        }),
        hidden_size: 2,
        max_length: 4,
        n_heads: 1,
    };
    let pipeline = DiffusionPipeline {
        summary: summarize_hfq(Path::new("/tmp/tiny-sdxl-runtime.hfq"), &metadata),
        metadata,
        config,
        tokenizer: Some(tokenizer.clone()),
        tokenizer_2: Some(tokenizer),
        text_encoder: Some(text_encoder.clone()),
        text_encoder_2: Some(text_encoder),
        native_runtime: None,
        native_runtime_error: Some("dual encoder test".into()),
    };
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![DiffusionPrompt {
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            seed: 7,
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
        steps: 2,
        cfg_scale: 7.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: false,
        save_images: false,
    };

    let conditioning = pipeline.prepare_conditioning_batch(&request).unwrap();

    assert!(conditioning.prompt_tokens_2.is_some());
    assert_eq!(
        conditioning.prompt_embeddings_2.as_ref().unwrap().shape,
        vec![1, 4, 2]
    );
    assert_eq!(
        conditioning
            .prompt_pooled_embeddings
            .as_ref()
            .unwrap()
            .shape,
        vec![1, 2]
    );
    assert_eq!(
        conditioning
            .negative_pooled_embeddings
            .as_ref()
            .unwrap()
            .shape,
        vec![1, 2]
    );
}

#[test]
fn diffusion_pipeline_reuses_positive_conditioning_when_cfg_is_identity() {
    let mut metadata = tiny_runtime_metadata();
    metadata.pipeline.class_name = "StableDiffusionXLPipeline".into();
    metadata.tokenizer_2 = Some(DiffusionTokenizerMetadata {
        kind: "clip-bpe".into(),
        max_length: Some(4),
        entries: vec!["tokenizer_2/vocab.json".into()],
    });
    let mut config = tiny_runtime_config();
    config.pipeline_class = "StableDiffusionXLPipeline".into();
    config.text_encoder_2 = Some(TextEncoderConfig {
        class_name: "CLIPTextModelWithProjection".into(),
        hidden_size: Some(2),
        intermediate_size: Some(4),
        num_hidden_layers: Some(0),
        num_attention_heads: Some(1),
        max_position_embeddings: Some(4),
        vocab_size: Some(4),
    });
    let tokenizer = ClipTokenizer::from_bytes(
        br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "cat</w>": 3
            }"#,
        b"#version: 0.2\n",
        4,
    )
    .unwrap();
    let text_encoder = ClipTextEncoder {
        token_embedding: CpuTensor {
            shape: vec![4, 2],
            data: vec![0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
        },
        position_embedding: CpuTensor {
            shape: vec![4, 2],
            data: vec![0.0; 8],
        },
        layers: Vec::new(),
        final_layer_norm_weight: CpuTensor {
            shape: vec![2],
            data: vec![1.0, 1.0],
        },
        final_layer_norm_bias: CpuTensor {
            shape: vec![2],
            data: vec![0.0, 0.0],
        },
        text_projection: Some(CpuTensor {
            shape: vec![2, 2],
            data: vec![1.0, 0.0, 0.0, 1.0],
        }),
        hidden_size: 2,
        max_length: 4,
        n_heads: 1,
    };
    let pipeline = DiffusionPipeline {
        summary: summarize_hfq(Path::new("/tmp/tiny-sdxl-runtime.hfq"), &metadata),
        metadata,
        config,
        tokenizer: Some(tokenizer.clone()),
        tokenizer_2: Some(tokenizer),
        text_encoder: Some(text_encoder.clone()),
        text_encoder_2: Some(text_encoder),
        native_runtime: None,
        native_runtime_error: Some("dual encoder test".into()),
    };
    let request = DiffusionBatchRequest {
        conditioning: None,
        prompts: vec![DiffusionPrompt {
            prompt: "a".into(),
            negative_prompt: "cat".into(),
            seed: 7,
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
        steps: 2,
        cfg_scale: 1.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: false,
        save_images: false,
    };

    let conditioning = pipeline.prepare_conditioning_batch(&request).unwrap();

    assert_eq!(conditioning.negative_tokens, conditioning.prompt_tokens);
    assert_eq!(conditioning.negative_tokens_2, conditioning.prompt_tokens_2);
    assert_eq!(
        conditioning.negative_embeddings,
        conditioning.prompt_embeddings
    );
    assert_eq!(
        conditioning.negative_embeddings_2,
        conditioning.prompt_embeddings_2
    );
    assert_eq!(
        conditioning.negative_cross_attention_embeddings,
        conditioning.prompt_cross_attention_embeddings
    );
    assert_eq!(
        conditioning.negative_pooled_embeddings,
        conditioning.prompt_pooled_embeddings
    );
}

#[test]
fn diffusion_pipeline_rejects_tiny_unet_latents_before_conditioning() {
    let metadata = tiny_runtime_metadata();
    let mut config = tiny_runtime_config();
    config.vae_scale_factor = 8;
    config.unet.down_block_types = vec![
        "CrossAttnDownBlock2D".into(),
        "CrossAttnDownBlock2D".into(),
        "CrossAttnDownBlock2D".into(),
        "DownBlock2D".into(),
    ];
    let pipeline = DiffusionPipeline {
        summary: summarize_hfq(Path::new("/tmp/tiny-runtime.hfq"), &metadata),
        metadata,
        config,
        tokenizer: None,
        tokenizer_2: None,
        text_encoder: None,
        text_encoder_2: None,
        native_runtime: None,
        native_runtime_error: Some("synthetic test".into()),
    };
    let request = DiffusionBatchRequest {
        conditioning: None,
        prompts: vec![DiffusionPrompt {
            prompt: "a".into(),
            negative_prompt: String::new(),
            seed: 7,
            subseed: None,
        }],
        width: 8,
        height: 8,
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
        send_images: false,
        save_images: false,
    };

    let err = pipeline.prepare_run_plan(&request).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("too small for UNet downsampling depth 3"));
    assert!(!message.contains("CLIP tokenizer"));
}

#[test]
fn diffusion_pipeline_passes_sdxl_conditioning_to_noise_backend() {
    let mut metadata = tiny_runtime_metadata();
    metadata.pipeline.class_name = "StableDiffusionXLPipeline".into();
    metadata.tokenizer_2 = Some(DiffusionTokenizerMetadata {
        kind: "clip-bpe".into(),
        max_length: Some(4),
        entries: vec!["tokenizer_2/vocab.json".into()],
    });
    let mut config = tiny_runtime_config();
    config.pipeline_class = "StableDiffusionXLPipeline".into();
    config.text_encoder_2 = Some(TextEncoderConfig {
        class_name: "CLIPTextModelWithProjection".into(),
        hidden_size: Some(2),
        intermediate_size: Some(4),
        num_hidden_layers: Some(0),
        num_attention_heads: Some(1),
        max_position_embeddings: Some(4),
        vocab_size: Some(4),
    });
    let tokenizer = ClipTokenizer::from_bytes(
        br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "cat</w>": 3
            }"#,
        b"#version: 0.2\n",
        4,
    )
    .unwrap();
    let text_encoder = ClipTextEncoder {
        token_embedding: CpuTensor {
            shape: vec![4, 2],
            data: vec![0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
        },
        position_embedding: CpuTensor {
            shape: vec![4, 2],
            data: vec![0.0; 8],
        },
        layers: Vec::new(),
        final_layer_norm_weight: CpuTensor {
            shape: vec![2],
            data: vec![1.0, 1.0],
        },
        final_layer_norm_bias: CpuTensor {
            shape: vec![2],
            data: vec![0.0, 0.0],
        },
        text_projection: Some(CpuTensor {
            shape: vec![2, 2],
            data: vec![1.0, 0.0, 0.0, 1.0],
        }),
        hidden_size: 2,
        max_length: 4,
        n_heads: 1,
    };
    let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let pipeline = DiffusionPipeline {
        summary: summarize_hfq(Path::new("/tmp/tiny-sdxl-runtime.hfq"), &metadata),
        metadata,
        config,
        tokenizer: Some(tokenizer.clone()),
        tokenizer_2: Some(tokenizer),
        text_encoder: Some(text_encoder.clone()),
        text_encoder_2: Some(text_encoder),
        native_runtime: Some(NativeDiffusionRuntime {
            kind: DiffusionRuntimeKind::CpuSourceReference,
            noise: Box::new(TestSdxlNoiseBackend {
                called: called.clone(),
            }),
            encoder: None,
            decoder: Box::new(TestImageDecoder),
            text_conditioner: None,
            krea2_tokenizer: None,
        }),
        native_runtime_error: None,
    };
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![DiffusionPrompt {
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            seed: 7,
            subseed: None,
        }],
        width: 2,
        height: 2,
        original_width: Some(128),
        original_height: Some(256),
        target_width: Some(32),
        target_height: Some(64),
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 8,
        crop_y: 4,
        steps: 2,
        cfg_scale: 7.0,
        distilled_guidance_scale: None,
        scheduler: "Euler".into(),
        subseed_strength: 0.0,
        send_images: false,
        save_images: false,
    };

    let output = pipeline.generate_batch(request).unwrap();

    assert!(called.load(std::sync::atomic::Ordering::SeqCst));
    assert!(output.images.is_empty());
    assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
}

#[test]
fn diffusion_pipeline_img2img_uses_inpaint_conditioning_for_inpaint_channel_model() {
    let (pipeline, called, dir) = tiny_inpaint_test_pipeline(
        "hipfire-diffusion-inpaint-routing-test",
        Box::new(TestImageDecoder),
    );
    let request = DiffusionImg2ImgRequest {
        batch: DiffusionBatchRequest {
            conditioning: None,

            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 7,
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
            steps: 2,
            cfg_scale: 7.0,
            distilled_guidance_scale: None,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        },
        init_image: tiny_rgb_image_batch(1, 2, 2),
        mask: Some(tiny_mask_image_batch(1, 2, 2)),
        inpainting_fill: None,
        resize_mode: DiffusionImg2ImgResizeMode::Image,
        denoising_strength: 1.0,
    };

    let output = pipeline.generate_img2img_batch(request).unwrap();

    assert!(called.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["masked"], true);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_img2img_resizes_init_and_mask_to_request_dimensions() {
    let (pipeline, called, dir) = tiny_inpaint_test_pipeline(
        "hipfire-diffusion-inpaint-resize-routing-test",
        Box::new(TestImageDecoder),
    );
    let request = DiffusionImg2ImgRequest {
        batch: DiffusionBatchRequest {
            conditioning: None,

            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 7,
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
            steps: 2,
            cfg_scale: 7.0,
            distilled_guidance_scale: None,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        },
        init_image: tiny_rgb_image_batch(1, 1, 1),
        mask: Some(tiny_mask_image_batch(1, 1, 1)),
        inpainting_fill: None,
        resize_mode: DiffusionImg2ImgResizeMode::Image,
        denoising_strength: 1.0,
    };

    let output = pipeline.generate_img2img_batch(request).unwrap();

    assert!(called.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["mode"], "img2img");
    assert_eq!(output.info["masked"], true);
    assert_eq!(output.info["width"], 2);
    assert_eq!(output.info["height"], 2);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_img2img_latent_resize_mode_resizes_encoded_latents() {
    let (pipeline, called, dir) = tiny_inpaint_test_pipeline(
        "hipfire-diffusion-inpaint-latent-resize-routing-test",
        Box::new(TestImageDecoder),
    );
    let request = DiffusionImg2ImgRequest {
        batch: DiffusionBatchRequest {
            conditioning: None,

            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 7,
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
            steps: 2,
            cfg_scale: 7.0,
            distilled_guidance_scale: None,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        },
        init_image: tiny_rgb_image_batch(1, 1, 1),
        mask: Some(tiny_mask_image_batch(1, 1, 1)),
        inpainting_fill: None,
        resize_mode: DiffusionImg2ImgResizeMode::Latent,
        denoising_strength: 1.0,
    };

    let output = pipeline.generate_img2img_batch(request).unwrap();

    assert!(called.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["mode"], "img2img");
    assert_eq!(output.info["masked"], true);
    assert_eq!(output.info["resize_mode"], "latent");
    assert_eq!(output.info["latent_resize"], true);
    assert_eq!(output.info["width"], 2);
    assert_eq!(output.info["height"], 2);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_qwen_transformer_accepts_external_conditioning_without_clip() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-qwen-external-conditioning-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-qwen-transformer-external.hfq");
    let mut metadata = tiny_qwen_transformer_runtime_metadata();
    metadata.components.remove("text_encoder");
    let tensors = tiny_qwen_transformer_runtime_tensors()
        .into_iter()
        .filter(|tensor| {
            !tensor.name.starts_with("text_encoder/") && !tensor.name.starts_with("tokenizer/")
        })
        .collect::<Vec<_>>();
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let inspection = inspect_hfq_with_runtime_support(&hfq_path).unwrap();
    assert!(inspection.runtime_support.supported);
    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    let mut request = DiffusionBatchRequest {
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
        send_images: true,
        save_images: false,
    };

    let prompt_error = pipeline.generate_batch(request.clone()).unwrap_err();
    assert!(prompt_error
        .to_string()
        .contains("does not contain a usable CLIP tokenizer"));

    request.conditioning = Some(DiffusionExternalConditioningBatch {
        prompt_embeddings: CpuTensor {
            shape: vec![1, 1, 2],
            data: vec![0.5, -0.5],
        },
        negative_embeddings: CpuTensor {
            shape: vec![1, 1, 2],
            data: vec![0.5, -0.5],
        },
        prompt_attention_mask: None,
        negative_attention_mask: None,
        prompt_pooled_embeddings: None,
        negative_pooled_embeddings: None,
    });

    let output = pipeline.generate_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
    assert_eq!(output.info["pipeline"], "QwenImagePipeline");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_qwen_transformer_projects_external_text_width() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-qwen-external-projection-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-qwen-transformer-external-projection.hfq");
    let mut metadata = tiny_qwen_transformer_runtime_metadata();
    metadata.components.remove("text_encoder");
    let mut tensors = tiny_qwen_transformer_runtime_tensors()
        .into_iter()
        .filter(|tensor| {
            !tensor.name.starts_with("text_encoder/")
                && !tensor.name.starts_with("tokenizer/")
                && tensor.name != "transformer/config.json"
        })
        .collect::<Vec<_>>();
    tensors.push(bytes_mem_tensor(
            "transformer/config.json",
            QT_DIFFUSION_JSON,
            br#"{"_class_name":"QwenImageTransformer2DModel","in_channels":4,"out_channels":1,"patch_size":2,"num_layers":1,"num_attention_heads":1,"attention_head_dim":2,"joint_attention_dim":3}"#,
        ));
    tensors.extend([
        f32_mem_tensor(
            "transformer/tensors/txt_norm.weight",
            &[3],
            &[1.0, 0.5, 2.0],
        ),
        f32_mem_tensor(
            "transformer/tensors/txt_in.weight",
            &[2, 3],
            &[1.0, 0.0, 0.25, 0.0, 1.0, -0.5],
        ),
        f32_mem_tensor("transformer/tensors/txt_in.bias", &[2], &[0.1, -0.2]),
        f32_mem_tensor(
            "transformer/tensors/norm_out.linear.weight",
            &[4, 2],
            &[0.0; 8],
        ),
        f32_mem_tensor(
            "transformer/tensors/norm_out.linear.bias",
            &[4],
            &[0.1, -0.2, 0.3, -0.4],
        ),
    ]);
    metadata
        .components
        .get_mut("transformer")
        .unwrap()
        .weight_entries = tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("transformer/tensors/"))
        .map(|tensor| tensor.name.clone())
        .collect();
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let inspection = inspect_hfq_with_runtime_support(&hfq_path).unwrap();
    assert!(inspection.runtime_support.supported);
    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    let request = DiffusionBatchRequest {
        conditioning: Some(DiffusionExternalConditioningBatch {
            prompt_embeddings: CpuTensor {
                shape: vec![1, 1, 3],
                data: vec![0.5, -0.5, 2.0],
            },
            negative_embeddings: CpuTensor {
                shape: vec![1, 1, 3],
                data: vec![-0.25, 0.75, 1.5],
            },
            prompt_attention_mask: Some(CpuTensor {
                shape: vec![1, 1],
                data: vec![1.0],
            }),
            negative_attention_mask: Some(CpuTensor {
                shape: vec![1, 1],
                data: vec![1.0],
            }),
            prompt_pooled_embeddings: None,
            negative_pooled_embeddings: None,
        }),
        prompts: vec![DiffusionPrompt {
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            seed: 11,
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
        send_images: true,
        save_images: false,
    };

    let output = pipeline.generate_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
    assert_eq!(output.info["pipeline"], "QwenImagePipeline");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn resize_latent_batch_nearest_resizes_spatial_axes_per_channel() {
    let latents = LatentBatch {
        batch: 1,
        channels: 2,
        height: 2,
        width: 2,
        data: vec![
            1.0, 2.0, //
            3.0, 4.0, //
            10.0, 20.0, //
            30.0, 40.0,
        ],
    };

    let resized = resize_latent_batch_nearest(&latents, 1, 4).unwrap();

    assert_eq!(resized.batch, 1);
    assert_eq!(resized.channels, 2);
    assert_eq!(resized.height, 1);
    assert_eq!(resized.width, 4);
    assert_eq!(
        resized.data,
        vec![1.0, 1.0, 2.0, 2.0, 10.0, 10.0, 20.0, 20.0]
    );
}

#[test]
fn inpainting_fill_latent_noise_replaces_masked_latents() {
    let mut init = LatentBatch {
        batch: 1,
        channels: 2,
        height: 1,
        width: 2,
        data: vec![10.0, 20.0, 30.0, 40.0],
    };
    let noise = LatentBatch {
        batch: 1,
        channels: 2,
        height: 1,
        width: 2,
        data: vec![1.0, 2.0, 3.0, 4.0],
    };

    let applied = apply_inpainting_fill_to_latents(&mut init, &noise, &[0.0, 1.0], 2).unwrap();

    assert!(applied);
    assert_eq!(init.data, vec![10.0, 2.0, 30.0, 4.0]);
}

#[test]
fn inpainting_fill_latent_nothing_zeros_masked_latents() {
    let mut init = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2,
        data: vec![10.0, 20.0],
    };
    let noise = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2,
        data: vec![1.0, 2.0],
    };

    let applied = apply_inpainting_fill_to_latents(&mut init, &noise, &[1.0, 0.25], 3).unwrap();

    assert!(applied);
    assert_eq!(init.data, vec![0.0, 15.0]);
}

#[test]
fn generate_img2img_runtime_options_route_vae_mask_boundaries_when_gpu_is_available() {
    if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
        eprintln!("skip: ROCm GPU unavailable for hybrid img2img generation test: {error}");
        return;
    }
    let (pipeline, called, dir) = tiny_inpaint_test_pipeline(
        "hipfire-diffusion-inpaint-hybrid-routing-test",
        Box::new(SolidTensorImageDecoder),
    );
    let request = DiffusionImg2ImgRequest {
        batch: DiffusionBatchRequest {
            conditioning: None,

            prompts: vec![DiffusionPrompt {
                prompt: "a cat".into(),
                negative_prompt: String::new(),
                seed: 7,
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
            steps: 2,
            cfg_scale: 7.0,
            distilled_guidance_scale: None,
            scheduler: "Euler".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        },
        init_image: tiny_rgb_image_batch(1, 2, 2),
        mask: Some(tiny_mask_image_batch(1, 2, 2)),
        inpainting_fill: None,
        resize_mode: DiffusionImg2ImgResizeMode::Image,
        denoising_strength: 1.0,
    };

    let output = pipeline
        .generate_img2img_batch_with_runtime_options(
            request,
            DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
        )
        .unwrap();

    assert!(called.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["runtime"], "rocm-hybrid-reference");
    assert_eq!(output.info["masked"], true);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    assert_eq!(decoded.get_pixel(0, 0).0, [32, 128, 224]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_open_hfq_generates_png_with_native_tiny_components() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-complete-pipeline-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-complete.hfq");
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
        send_images: true,
        save_images: false,
    };

    let output = pipeline.generate_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_open_hfq_generates_png_with_native_tiny_qwen_transformer() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-qwen-transformer-pipeline-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-qwen-transformer.hfq");
    let metadata = tiny_qwen_transformer_runtime_metadata();
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tiny_qwen_transformer_runtime_tensors(),
    )
    .unwrap();
    let inspection = inspect_hfq_with_runtime_support(&hfq_path).unwrap();
    assert!(inspection.runtime_support.supported);
    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    let capabilities = pipeline.runtime_capabilities().unwrap();
    assert_eq!(capabilities.kind, DiffusionRuntimeKind::CpuSourceReference);
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
        send_images: true,
        save_images: false,
    };

    let output = pipeline.generate_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
    assert_eq!(output.info["pipeline"], "QwenImagePipeline");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_runs_quantized_metadata_with_float_tensor_payloads() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-quantized-float-runtime-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-quantized.hfq");
    let mut metadata = tiny_runtime_metadata();
    metadata.quantization.weight_format = "oq4".to_string();
    metadata.quantization.activation_format = "fp16".to_string();
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tiny_complete_runtime_tensors(),
    )
    .unwrap();
    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    assert!(pipeline.native_runtime.is_some());
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
        send_images: true,
        save_images: false,
    };

    let output = pipeline.generate_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["weight_format"], "oq4");
    assert_eq!(output.info["runtime"], "cpu-source-reference");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_runs_with_q8f16_tensor_payloads() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-q8-runtime-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-q8.hfq");
    let mut metadata = tiny_runtime_metadata();
    metadata.quantization.weight_format = "q8".to_string();
    let mut tensors = tiny_complete_runtime_tensors();
    let tensor = tensors
        .iter_mut()
        .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
        .unwrap();
    *tensor = q8f16_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &[0.0; 9]);
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    assert!(pipeline.native_runtime.is_some());
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
        send_images: true,
        save_images: false,
    };

    let output = pipeline.generate_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["weight_format"], "q8");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_runs_with_q4f16_g64_tensor_payloads() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-q4-runtime-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-q4.hfq");
    let mut metadata = tiny_runtime_metadata();
    metadata.quantization.weight_format = "q4f16".to_string();
    let mut tensors = tiny_complete_runtime_tensors();
    let tensor = tensors
        .iter_mut()
        .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
        .unwrap();
    *tensor = q4f16_g64_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &[0.0; 9]);
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    assert!(pipeline.native_runtime.is_some());
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
        send_images: true,
        save_images: false,
    };

    let output = pipeline.generate_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["weight_format"], "q4f16");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_runs_with_q4k_tensor_payloads() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-q4k-runtime-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-q4k.hfq");
    let mut metadata = tiny_runtime_metadata();
    metadata.quantization.weight_format = "q4k".to_string();
    let mut tensors = tiny_complete_runtime_tensors();
    let tensor = tensors
        .iter_mut()
        .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
        .unwrap();
    *tensor = q4k_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &[0; 9]);
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    assert!(pipeline.native_runtime.is_some());
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
        send_images: true,
        save_images: false,
    };

    let output = pipeline.generate_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["weight_format"], "q4k");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_runs_with_hfq4_tensor_payloads() {
    for (label, quant_type, group_size) in [
        ("hfq4g128", QT_DIFFUSION_TENSOR_HFQ4_G128, 128usize),
        ("hfq4g256", QT_DIFFUSION_TENSOR_HFQ4_G256, 256usize),
    ] {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-{label}-runtime-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join(format!("tiny-{label}.hfq"));
        let mut metadata = tiny_runtime_metadata();
        metadata.quantization.weight_format = label.to_string();
        let mut tensors = tiny_complete_runtime_tensors();
        let tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
            .unwrap();
        *tensor = hfq4_mem_tensor(
            "unet/tensors/conv_in.weight",
            quant_type,
            &[1, 1, 3, 3],
            group_size,
            &[0; 9],
        );
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
        let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        assert!(pipeline.native_runtime.is_some());
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
            send_images: true,
            save_images: false,
        };

        let output = pipeline.generate_batch(request).unwrap();

        assert_eq!(output.images.len(), 1);
        assert_eq!(output.info["weight_format"], label);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&output.images[0])
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let _ = fs::remove_dir_all(&dir);
    }
}

#[test]
fn diffusion_pipeline_runs_with_hfq6_tensor_payloads() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-hfq6-runtime-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-hfq6g256.hfq");
    let mut metadata = tiny_runtime_metadata();
    metadata.quantization.weight_format = "hfq6g256".to_string();
    let mut tensors = tiny_complete_runtime_tensors();
    let tensor = tensors
        .iter_mut()
        .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
        .unwrap();
    *tensor = hfq6_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &[0; 9]);
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    assert!(pipeline.native_runtime.is_some());
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
        send_images: true,
        save_images: false,
    };

    let output = pipeline.generate_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["weight_format"], "hfq6g256");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_rejects_packed_quant_tensor_payload_without_dequantizer() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-packed-quant-runtime-boundary-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-packed-quant.hfq");
    let mut metadata = tiny_runtime_metadata();
    metadata.quantization.weight_format = "oq4".to_string();
    metadata.quantization.activation_format = "fp16".to_string();
    let mut tensors = tiny_complete_runtime_tensors();
    let tensor = tensors
        .iter_mut()
        .find(|tensor| tensor.name == "unet/tensors/conv_in.weight")
        .unwrap();
    tensor.quant_type = 99;
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();

    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();

    assert!(pipeline.native_runtime.is_none());
    let error = pipeline.native_runtime_error.as_deref().unwrap();
    assert!(error.contains("unsupported quant_type 99"));
    assert!(error.contains("diffusion dequantizer/runtime"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diffusion_pipeline_open_hfq_generates_img2img_png_with_native_tiny_components() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-complete-img2img-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-complete-img2img.hfq");
    let metadata = tiny_runtime_metadata();
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tiny_complete_runtime_tensors(),
    )
    .unwrap();
    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    assert!(pipeline.supports_img2img());
    let request = DiffusionImg2ImgRequest {
        batch: DiffusionBatchRequest {
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
            send_images: true,
            save_images: false,
        },
        init_image: RgbImageBatch {
            batch: 1,
            width: 2,
            height: 2,
            data: vec![
                255, 0, 0, 128, 0, 0, //
                64, 0, 0, 0, 0, 0,
            ],
        },
        mask: None,
        inpainting_fill: None,
        resize_mode: DiffusionImg2ImgResizeMode::Image,
        denoising_strength: 1.0,
    };

    let output = pipeline.generate_img2img_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
    assert_eq!(output.info["mode"], "img2img");
    assert_eq!(output.info["masked"], false);
    assert_eq!(output.info["denoise_steps"], 1);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.images[0])
        .unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn attention_layer_runs_biasless_self_and_cross_attention() {
    let identity = CpuTensor {
        shape: vec![2, 2],
        data: vec![1.0, 0.0, 0.0, 1.0],
    };
    let attention = AttentionLayer {
        to_q_weight: identity.clone(),
        to_q_bias: None,
        to_k_weight: identity.clone(),
        to_k_bias: None,
        to_v_weight: identity.clone(),
        to_v_bias: None,
        to_out_weight: identity,
        to_out_bias: None,
        heads: 1,
    };
    let hidden = CpuTensor {
        shape: vec![1, 2, 2],
        data: vec![1.0, 0.0, 0.0, 1.0],
    };
    let self_out = attention.forward(&hidden, None).unwrap();
    assert_eq!(self_out.shape, hidden.shape);
    assert!(self_out.data.iter().all(|value| value.is_finite()));

    let encoder = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![0.25, 0.75],
    };
    let cross_out = attention.forward(&hidden, Some(&encoder)).unwrap();
    assert_eq!(cross_out.shape, hidden.shape);
    assert_eq!(cross_out.data, vec![0.25, 0.75, 0.25, 0.75]);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for attention routing test: {error}");
        } else {
            let runtime_options = DiffusionGenerationRuntimeOptions::rocm_hybrid(0);
            let hip_self = attention
                .forward_with_runtime_options(&hidden, None, runtime_options)
                .unwrap();
            assert_eq!(hip_self.shape, self_out.shape);
            assert!(f32_slices_close(&hip_self.data, &self_out.data, 1e-5));

            let hip_cross = attention
                .forward_with_runtime_options(&hidden, Some(&encoder), runtime_options)
                .unwrap();
            assert_eq!(hip_cross.shape, cross_out.shape);
            assert!(f32_slices_close(&hip_cross.data, &cross_out.data, 1e-5));

            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let hip_context_self = attention
                .forward_with_runtime_context(&hidden, None, &mut runtime_context)
                .unwrap();
            let hip_context_cross = attention
                .forward_with_runtime_context(&hidden, Some(&encoder), &mut runtime_context)
                .unwrap();
            assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
            assert_eq!(hip_context_self.shape, self_out.shape);
            assert!(f32_slices_close(
                &hip_context_self.data,
                &self_out.data,
                1e-5
            ));
            assert_eq!(hip_context_cross.shape, cross_out.shape);
            assert!(f32_slices_close(
                &hip_context_cross.data,
                &cross_out.data,
                1e-5
            ));
        }
    }
}

#[test]
fn synthetic_clip_text_encoder_forward_is_finite() {
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
        text_projection: None,
        hidden_size: hidden,
        max_length: 2,
        n_heads: 3,
    };
    let encoded = encoder.encode_tokens(&[0, 1]).unwrap();

    assert_eq!(encoded.shape, vec![2, hidden]);
    assert!(encoded.data.iter().all(|value| value.is_finite()));
    assert!(encoded.data.iter().any(|value| value.abs() > 0.001));

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for CLIP encoder routing test: {error}");
        } else {
            let hip = encoder
                .encode_tokens_with_runtime_options(
                    &[0, 1],
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                )
                .unwrap();
            assert_eq!(hip.shape, encoded.shape);
            assert!(f32_slices_close(&hip.data, &encoded.data, 1e-5));
        }
    }
}
