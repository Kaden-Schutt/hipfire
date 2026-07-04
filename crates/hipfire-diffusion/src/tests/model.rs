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
fn inspect_hfq_reports_metadata_runtime_support_without_loading_runtime() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-runtime-support-inspect-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let source_path = dir.join("source.hfq");
    let quantized_path = dir.join("quantized.hfq");
    let source_metadata = tiny_runtime_metadata();
    let mut quantized_metadata = tiny_runtime_metadata();
    quantized_metadata.quantization.weight_format = "oq4".to_string();
    write_hfqm_package_mem(
        &source_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&source_metadata).unwrap(),
        &tiny_complete_runtime_tensors(),
    )
    .unwrap();
    write_hfqm_package_mem(
        &quantized_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&quantized_metadata).unwrap(),
        &tiny_complete_runtime_tensors(),
    )
    .unwrap();

    let source = inspect_hfq_with_runtime_support(&source_path).unwrap();
    let quantized = inspect_hfq_with_runtime_support(&quantized_path).unwrap();

    assert!(source.runtime_support.supported);
    assert_eq!(
        source.runtime_support.runtime_kind,
        Some(DiffusionRuntimeKind::CpuSourceReference)
    );
    assert_eq!(source.runtime_support.reason, None);
    assert!(quantized.runtime_support.supported);
    assert_eq!(
        quantized.runtime_support.runtime_kind,
        Some(DiffusionRuntimeKind::CpuSourceReference)
    );
    assert_eq!(quantized.runtime_support.reason, None);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn inspect_hfq_marks_guidance_distilled_qwen_transformer_unsupported() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-qwen-guidance-runtime-support-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-qwen-guidance-distilled.hfq");
    let metadata = tiny_qwen_transformer_runtime_metadata();
    let tensors = tiny_qwen_transformer_runtime_tensors()
            .into_iter()
            .map(|tensor| {
                if tensor.name == "transformer/config.json" {
                    bytes_mem_tensor(
                        "transformer/config.json",
                        QT_DIFFUSION_JSON,
                        br#"{"_class_name":"QwenImageTransformer2DModel","in_channels":4,"out_channels":1,"patch_size":2,"num_layers":1,"num_attention_heads":1,"attention_head_dim":2,"joint_attention_dim":2,"guidance_embeds":true}"#,
                    )
                } else {
                    tensor
                }
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
    assert!(!inspection.runtime_support.supported);
    let reason = inspection.runtime_support.reason.unwrap();
    assert!(reason.contains("guidance_embeds=true"));
    assert!(reason.contains("guidance-scale embedding path"));

    let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    assert!(pipeline.native_runtime.is_none());
    let runtime_error = pipeline.native_runtime_error.unwrap();
    assert!(runtime_error.contains("guidance_embeds=true"));
    assert!(runtime_error.contains("guidance-scale embedding path"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn inspect_hfq_detects_diffusion_container() {
    let dir = std::env::temp_dir().join(format!("hipfire-diffusion-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let config_path = dir.join("config.json");
    fs::write(&config_path, b"{}").unwrap();
    let hfq_path = dir.join("model.hfq");
    let metadata = minimal_metadata();
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &[HfqMemTensor {
            name: "unet/config.json".into(),
            quant_type: QT_DIFFUSION_JSON,
            shape: vec![2],
            group_size: 0,
            data: b"{}".to_vec(),
        }],
    )
    .unwrap();
    let summary = inspect_hfq(&hfq_path).unwrap();
    assert_eq!(summary.pipeline_class, "StableDiffusionPipeline");
    assert!(is_diffusion_hfq(&hfq_path));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn tiny_sd_clip_tokenizer_files_encode_prompt_when_cache_is_present() {
    let root = Path::new("/srv/huggingface/models--segmind--tiny-sd/snapshots/cad0bd7495fa6c4bcca01b19a723dc91627fe84f/tokenizer");
    if !root.exists() {
        eprintln!("skip: tiny-sd tokenizer cache not present");
        return;
    }
    let tokenizer = ClipTokenizer::from_bytes(
        &fs::read(root.join("vocab.json")).unwrap(),
        &fs::read(root.join("merges.txt")).unwrap(),
        77,
    )
    .unwrap();
    let encoded = tokenizer.encode_padded("a red robot");

    assert_eq!(encoded.len(), 77);
    assert_eq!(encoded[0], 49406);
    assert!(encoded[1..10].iter().any(|&token| token != 49407));
    assert!(encoded.contains(&49407));
}

#[test]
fn tiny_sd_unet_down_path_loads_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    let down_path = UnetDownPath::from_hfq(&hfq, &config.unet).unwrap();

    assert_eq!(down_path.conv_in.weight.shape, vec![320, 4, 3, 3]);
    assert_eq!(down_path.blocks.len(), 3);
    assert!(down_path.blocks[0].downsampler.is_some());
    assert!(down_path.blocks[1].downsampler.is_some());
    assert!(down_path.blocks[2].downsampler.is_none());
    assert_eq!(
        down_path.blocks[2].resnets[0].conv2.weight.shape,
        vec![1280, 1280, 3, 3]
    );
}

#[test]
fn tiny_sd_unet_up_path_loads_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    let up_path = UnetUpPath::from_hfq(&hfq, &config.unet).unwrap();

    assert_eq!(up_path.blocks.len(), 3);
    assert_eq!(up_path.blocks[0].resnets.len(), 2);
    assert_eq!(up_path.blocks[1].resnets.len(), 2);
    assert_eq!(up_path.blocks[2].resnets.len(), 2);
    assert!(up_path.blocks[0].upsampler.is_some());
    assert!(up_path.blocks[1].upsampler.is_some());
    assert!(up_path.blocks[2].upsampler.is_none());
    assert_eq!(
        up_path.blocks[0].resnets[0].conv1.weight.shape,
        vec![1280, 2560, 3, 3]
    );
    assert_eq!(
        up_path.blocks[2].resnets[1].conv2.weight.shape,
        vec![320, 320, 3, 3]
    );
}

#[test]
fn tiny_sd_unet_mid_block_loads_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    let Some(mid_block) = UnetMidBlock2DCrossAttn::from_hfq(&hfq, &config.unet).unwrap() else {
        eprintln!("skip: imported tiny-sd artifact has no UNet mid_block tensors");
        return;
    };

    assert!(mid_block.attention.is_some());
    assert!(mid_block.resnet_1.is_some());
    assert_eq!(
        mid_block.resnet_0.conv1.weight.shape,
        vec![1280, 1280, 3, 3]
    );
    assert_eq!(
        mid_block.attention.as_ref().unwrap().proj_in.weight.shape,
        vec![1280, 1280, 1, 1]
    );
}

#[test]
fn tiny_sd_native_unet_loads_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    let unet = NativeUnet2DConditionModel::from_hfq(&hfq, &config.unet).unwrap();

    assert_eq!(unet.down_path.blocks.len(), 3);
    assert_eq!(unet.up_path.blocks.len(), 3);
    assert_eq!(unet.conv_norm_out.weight.shape, vec![320]);
    assert_eq!(unet.conv_out.weight.shape, vec![4, 320, 3, 3]);
}

#[test]
fn tiny_sd_native_vae_decoder_loads_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    let decoder = NativeVaeDecoder::from_hfq(&hfq, &config.vae).unwrap();

    assert_eq!(
        decoder.conv_in.as_ref().unwrap().weight.shape,
        vec![512, 4, 3, 3]
    );
    assert_eq!(decoder.up_blocks.len(), 4);
    assert!(decoder.up_blocks[0].upsampler.is_some());
    assert!(decoder.up_blocks[1].upsampler.is_some());
    assert!(decoder.up_blocks[2].upsampler.is_some());
    assert!(decoder.up_blocks[3].upsampler.is_none());
    assert_eq!(
        decoder.conv_out.as_ref().unwrap().weight.shape,
        vec![3, 128, 3, 3]
    );
}

#[test]
fn tiny_sd_native_vae_encoder_loads_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    let encoder = NativeVaeEncoder::from_hfq(&hfq, &config.vae).unwrap();

    assert_eq!(encoder.conv_in.weight.shape, vec![128, 3, 3, 3]);
    assert_eq!(encoder.down_blocks.len(), 4);
    assert!(encoder.down_blocks[0].downsampler.is_some());
    assert!(encoder.down_blocks[1].downsampler.is_some());
    assert!(encoder.down_blocks[2].downsampler.is_some());
    assert!(encoder.down_blocks[3].downsampler.is_none());
    assert_eq!(encoder.conv_out.weight.shape, vec![8, 512, 3, 3]);
    assert!(encoder.quant_conv.is_some());
}

#[test]
fn tiny_sd_unet_resnet_block_loads_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let block = UnetResnetBlock2D::from_hfq(&hfq, "unet/tensors/down_blocks.0.resnets.0", 32, 1e-5)
        .unwrap();

    assert_eq!(block.conv1.weight.shape, vec![320, 320, 3, 3]);
    assert_eq!(block.time_emb_proj_weight.shape, vec![320, 1280]);
    assert!(block.shortcut.is_none());
}

#[test]
#[ignore = "naive CPU CLIP forward over tiny-sd is a correctness smoke, not a normal unit test"]
fn tiny_sd_clip_text_encoder_loads_and_encodes_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let tokenizer = ClipTokenizer::from_hfq_file(&hfq).unwrap();
    let text_encoder = ClipTextEncoder::from_hfq_file(&hfq).unwrap();
    let tokens = tokenizer.encode_padded("a red robot");
    let encoded = text_encoder.encode_tokens(&tokens).unwrap();

    assert_eq!(encoded.shape, vec![77, 768]);
    assert!(encoded.data.iter().all(|value| value.is_finite()));
    assert!(encoded.data.iter().any(|value| value.abs() > 0.001));
}

#[test]
#[ignore = "real Tiny-SD end-to-end generation is an admission smoke; the naive CPU runtime is slow"]
fn tiny_sd_pipeline_generates_one_step_png_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let pipeline = DiffusionPipeline::open_hfq(&path).unwrap();
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![DiffusionPrompt {
            prompt: "a red robot".into(),
            negative_prompt: String::new(),
            seed: 123,
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
        steps: 1,
        cfg_scale: 1.0,
        distilled_guidance_scale: None,
        scheduler: "DPM++ 2M".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };

    let output = pipeline.generate_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert!(output.images[0].starts_with("iVBORw0KGgo"));
    assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
}

#[test]
#[ignore = "real Tiny-SD HFQ shape guard smoke; requires /tmp/hipfire-tiny-sd-diffusion.hfq"]
fn tiny_sd_pipeline_rejects_too_small_unet_latents_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let pipeline = DiffusionPipeline::open_hfq(&path).unwrap();
    let request = DiffusionBatchRequest {
        conditioning: None,
        prompts: vec![DiffusionPrompt {
            prompt: "a red robot".into(),
            negative_prompt: String::new(),
            seed: 123,
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
        steps: 1,
        cfg_scale: 1.0,
        distilled_guidance_scale: None,
        scheduler: "DPM++ 2M".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };

    let err = pipeline.prepare_run_plan(&request).unwrap_err();
    assert!(err
        .to_string()
        .contains("too small for UNet downsampling depth"));
}

#[test]
#[ignore = "real Tiny-SD img2img is an admission smoke; run in release mode under an external timeout"]
fn tiny_sd_pipeline_generates_one_step_img2img_png_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let pipeline = DiffusionPipeline::open_hfq(&path).unwrap();
    if !pipeline.supports_img2img() {
        eprintln!("skip: {} has no native VAE encoder", path.display());
        return;
    }
    let request = DiffusionImg2ImgRequest {
        batch: DiffusionBatchRequest {
            conditioning: None,

            prompts: vec![DiffusionPrompt {
                prompt: "a red robot".into(),
                negative_prompt: String::new(),
                seed: 123,
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
            steps: 1,
            cfg_scale: 1.0,
            distilled_guidance_scale: None,
            scheduler: "DPM++ 2M".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        },
        init_image: tiny_rgb_image_batch(1, 64, 64),
        mask: Some(tiny_mask_image_batch(1, 64, 64)),
        inpainting_fill: None,
        resize_mode: DiffusionImg2ImgResizeMode::Image,
        denoising_strength: 1.0,
    };

    let output = pipeline.generate_img2img_batch(request).unwrap();

    assert_eq!(output.images.len(), 1);
    assert!(output.images[0].starts_with("iVBORw0KGgo"));
    assert_eq!(output.info["backend"], "hipfire-diffusion-hfq");
    assert_eq!(output.info["mode"], "img2img");
    assert_eq!(output.info["masked"], true);
}

#[test]
#[ignore = "diagnostic real-model phase timing; run with --nocapture under an external timeout"]
fn tiny_sd_pipeline_phase_timings_when_import_exists() {
    let path = tiny_sd_hfq_path();
    if skip_missing_tiny_sd(&path) {
        return;
    }
    let request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![DiffusionPrompt {
            prompt: "a red robot".into(),
            negative_prompt: String::new(),
            seed: 123,
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
        steps: 1,
        cfg_scale: 1.0,
        distilled_guidance_scale: None,
        scheduler: "DPM++ 2M".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };

    let total = std::time::Instant::now();
    let phase = std::time::Instant::now();
    let pipeline = DiffusionPipeline::open_hfq(&path).unwrap();
    eprintln!("phase open_hfq {:?}", phase.elapsed());

    let phase = std::time::Instant::now();
    let plan = pipeline.prepare_run_plan(&request).unwrap();
    eprintln!("phase prepare_run_plan {:?}", phase.elapsed());

    let runtime = pipeline.native_runtime.as_ref().unwrap();
    let positive = plan.conditioning.prompt_embeddings.as_ref().unwrap();
    let negative = plan.conditioning.negative_embeddings.as_ref().unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());
    let phase = std::time::Instant::now();
    let latents = runtime
        .noise
        .denoise_latents_with_runtime_context(
            plan.latents,
            &plan.schedule,
            request.cfg_scale,
            positive,
            negative,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut runtime_context,
            None,
        )
        .unwrap();
    eprintln!("phase denoise {:?}", phase.elapsed());

    let hfq = HfqFile::open_index_only(&path).unwrap();
    let decoder = NativeVaeDecoder::from_hfq(&hfq, &pipeline.config.vae).unwrap();
    let phase = std::time::Instant::now();
    let decoded = decoder.decode_latents(&latents.latents).unwrap();
    eprintln!("phase decode_latents {:?}", phase.elapsed());

    let phase = std::time::Instant::now();
    let rgb = rgb_tensor_to_u8(&decoded).unwrap();
    eprintln!("phase rgb_tensor_to_u8 {:?}", phase.elapsed());

    let phase = std::time::Instant::now();
    let images = encode_rgb_batch_png_base64(&rgb).unwrap();
    eprintln!("phase png_base64 {:?}", phase.elapsed());
    eprintln!("phase total {:?}", total.elapsed());

    assert_eq!(images.len(), 1);
    assert!(images[0].starts_with("iVBORw0KGgo"));
}
