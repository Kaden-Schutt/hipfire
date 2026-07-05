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
fn runtime_options_default_decode_uses_cpu_rgb_conversion() {
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 2,
        width: 2,
        data: vec![0.0, 0.25, 0.5, 0.75],
    };
    let (rgb, runtime_kind) = decode_to_rgb8_with_runtime_options(
        &SolidTensorImageDecoder,
        &latents,
        DiffusionGenerationRuntimeOptions::default(),
    )
    .unwrap();

    assert_eq!(runtime_kind, DiffusionRuntimeKind::CpuSourceReference);
    assert_eq!(rgb, SolidTensorImageDecoder::expected_rgb(&latents));
}

#[test]
fn runtime_options_rocm_hybrid_decode_matches_cpu_when_gpu_is_available() {
    if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
        eprintln!("skip: ROCm GPU unavailable for hybrid decode test: {error}");
        return;
    }
    let latents = LatentBatch {
        batch: 2,
        channels: 1,
        height: 2,
        width: 3,
        data: vec![0.0; 12],
    };
    let (rgb, runtime_kind) = decode_to_rgb8_with_runtime_options(
        &SolidTensorImageDecoder,
        &latents,
        DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
    )
    .unwrap();

    assert_eq!(runtime_kind, DiffusionRuntimeKind::RocmHybridReference);
    assert_eq!(rgb, SolidTensorImageDecoder::expected_rgb(&latents));
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
        refine_sigma: None,
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
        refine_sigma: None,
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
        refine_sigma: None,
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
        refine_sigma: None,
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
        refine_sigma: None,
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
