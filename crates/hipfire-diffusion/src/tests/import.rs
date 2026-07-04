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
fn parses_diffusion_metadata() {
    let metadata = minimal_metadata();
    let json = serde_json::to_string(&metadata).unwrap();
    assert_eq!(parse_diffusion_metadata(&json).unwrap(), metadata);
}

#[test]
fn imports_minimal_diffusers_snapshot_to_hfq() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-import-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let source = dir.join("snapshot");
    fs::create_dir_all(source.join("text_encoder")).unwrap();
    fs::create_dir_all(source.join("unet")).unwrap();
    fs::create_dir_all(source.join("vae")).unwrap();
    fs::create_dir_all(source.join("scheduler")).unwrap();
    fs::create_dir_all(source.join("tokenizer")).unwrap();
    fs::write(
        source.join("model_index.json"),
        br#"{"_class_name":"StableDiffusionPipeline"}"#,
    )
    .unwrap();
    fs::write(
        source.join("unet/config.json"),
        br#"{"_class_name":"UNet2DConditionModel","sample_size":64,"in_channels":4}"#,
    )
    .unwrap();
    fs::write(
        source.join("vae/config.json"),
        br#"{"_class_name":"AutoencoderKL","latent_channels":4}"#,
    )
    .unwrap();
    fs::write(
        source.join("text_encoder/config.json"),
        br#"{"_class_name":"CLIPTextModel"}"#,
    )
    .unwrap();
    fs::write(
        source.join("scheduler/scheduler_config.json"),
        br#"{"_class_name":"DPMSolverMultistepScheduler"}"#,
    )
    .unwrap();
    fs::write(source.join("tokenizer/vocab.json"), b"{}").unwrap();
    fs::write(source.join("unet/diffusion_pytorch_model.bin"), b"unet").unwrap();
    fs::write(source.join("vae/diffusion_pytorch_model.bin"), b"vae").unwrap();
    fs::write(source.join("text_encoder/pytorch_model.bin"), b"text").unwrap();

    let output = dir.join("tiny.hfq");
    let summary = import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: Some("tiny-sd".into()),
        max_batch: 3,
        metadata_only: false,
    })
    .unwrap();

    assert_eq!(summary.model_name, "tiny-sd");
    assert_eq!(summary.max_batch, 3);
    assert!(is_diffusion_hfq(&output));

    let hfq = HfqFile::open_index_only(&output).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    assert_eq!(config.pipeline_class, "StableDiffusionPipeline");
    assert_eq!(config.unet.sample_size, Some(64));
    assert_eq!(config.latent_channels, 4);
    assert_eq!(config.scheduler.class_name, "DPMSolverMultistepScheduler");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn imports_transformer_pipeline_metadata_without_marking_runtime_supported() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-transformer-import-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let source = dir.join("snapshot");
    fs::create_dir_all(source.join("text_encoder")).unwrap();
    fs::create_dir_all(source.join("transformer")).unwrap();
    fs::create_dir_all(source.join("vae")).unwrap();
    fs::create_dir_all(source.join("scheduler")).unwrap();
    fs::write(
            source.join("model_index.json"),
            br#"{"_class_name":"Krea2Pipeline","text_encoder":["transformers","Qwen3VLModel"],"transformer":["diffusers","Krea2Transformer2DModel"],"vae":["diffusers","AutoencoderKLQwenImage"],"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"]}"#,
        )
        .unwrap();
    fs::write(
        source.join("text_encoder/config.json"),
        br#"{"_class_name":"Qwen3VLModel","hidden_size":2560}"#,
    )
    .unwrap();
    fs::write(
            source.join("transformer/config.json"),
            br#"{"_class_name":"Krea2Transformer2DModel","in_channels":64,"out_channels":16,"num_layers":28}"#,
        )
        .unwrap();
    fs::write(
            source.join("vae/config.json"),
            br#"{"_class_name":"AutoencoderKLQwenImage","z_dim":16,"latents_mean":[-0.75,0.25],"latents_std":[2.0,1.5]}"#,
        )
        .unwrap();
    fs::write(
            source.join("scheduler/scheduler_config.json"),
            br#"{"_class_name":"FlowMatchEulerDiscreteScheduler","num_train_timesteps":1000,"shift":1.0,"shift_terminal":0.02,"invert_sigmas":false,"use_dynamic_shifting":true,"time_shift_type":"exponential"}"#,
        )
        .unwrap();

    let output = dir.join("krea.hfq");
    let summary = import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: Some("krea".into()),
        max_batch: 1,
        metadata_only: false,
    })
    .unwrap();
    let hfq = HfqFile::open_index_only(&output).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let inspection = inspect_hfq_with_runtime_support(&output).unwrap();

    assert_eq!(summary.pipeline_class, "Krea2Pipeline");
    assert_eq!(metadata.pipeline.latent_channels, Some(16));
    assert!(metadata.components.contains_key("transformer"));
    assert_eq!(
        metadata.components["transformer"].class_name.as_deref(),
        Some("Krea2Transformer2DModel")
    );
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    assert_eq!(
        config.scheduler.class_name,
        "FlowMatchEulerDiscreteScheduler"
    );
    assert_eq!(config.latent_channels, 16);
    let transformer = config.transformer.as_ref().unwrap();
    assert_eq!(transformer.class_name, "Krea2Transformer2DModel");
    assert_eq!(transformer.in_channels, Some(64));
    assert_eq!(transformer.out_channels, Some(16));
    assert_eq!(transformer.patch_size, Some(2));
    assert_eq!(transformer.num_layers, Some(28));
    assert_eq!(config.vae.z_dim, Some(16));
    assert_eq!(config.vae.latents_mean, vec![-0.75, 0.25]);
    assert_eq!(config.vae.latents_std, vec![2.0, 1.5]);
    assert_eq!(config.scheduler.shift, Some(1.0));
    assert_eq!(config.scheduler.shift_terminal, Some(0.02));
    assert_eq!(config.scheduler.invert_sigmas, Some(false));
    assert_eq!(config.scheduler.use_dynamic_shifting, Some(true));
    assert_eq!(
        config.scheduler.time_shift_type.as_deref(),
        Some("exponential")
    );
    // Krea2 is a supported transformer family, but this import carries no
    // transformer weights, so it is rejected as an incomplete artifact.
    assert!(!inspection.runtime_support.supported);
    assert!(inspection
        .runtime_support
        .reason
        .as_deref()
        .unwrap()
        .contains("requires complete"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn imports_qwen_image_edit_transformer_metadata_and_shards() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-qwen-image-edit-import-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let source = dir.join("snapshot");
    fs::create_dir_all(source.join("text_encoder")).unwrap();
    fs::create_dir_all(source.join("tokenizer")).unwrap();
    fs::create_dir_all(source.join("transformer")).unwrap();
    fs::create_dir_all(source.join("vae")).unwrap();
    fs::create_dir_all(source.join("scheduler")).unwrap();
    fs::write(
            source.join("model_index.json"),
            br#"{"_class_name":"QwenImageEditPipeline","processor":["transformers","Qwen2VLProcessor"],"text_encoder":["transformers","Qwen2_5_VLForConditionalGeneration"],"tokenizer":["transformers","Qwen2Tokenizer"],"transformer":["diffusers","QwenImageTransformer2DModel"],"vae":["diffusers","AutoencoderKLQwenImage"],"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"]}"#,
        )
        .unwrap();
    fs::write(
        source.join("text_encoder/config.json"),
        br#"{"_class_name":"Qwen2_5_VLForConditionalGeneration","hidden_size":3584}"#,
    )
    .unwrap();
    fs::write(
        source.join("tokenizer/vocab.json"),
        br#"{"<|endoftext|>":0}"#,
    )
    .unwrap();
    fs::write(source.join("tokenizer/merges.txt"), b"#version: 0.2\n").unwrap();
    fs::write(
            source.join("transformer/config.json"),
            br#"{"_class_name":"QwenImageTransformer2DModel","in_channels":64,"out_channels":16,"num_layers":60,"num_attention_heads":24,"num_key_value_heads":8,"attention_head_dim":128,"joint_attention_dim":3584,"axes_dims_rope":[16,56,56],"guidance_embeds":false,"patch_size":2,"pooled_projection_dim":768}"#,
        )
        .unwrap();
    write_safetensors_fixture(
        &source.join("transformer/diffusion_pytorch_model-00001-of-00002.safetensors"),
        &[(
            "patch_embed.proj.weight",
            "F32",
            &[1],
            &[0x00, 0x00, 0xc0, 0x3f],
        )],
    );
    write_safetensors_fixture(
        &source.join("transformer/diffusion_pytorch_model-00002-of-00002.safetensors"),
        &[("norm_out.weight", "F32", &[1], &[0x00, 0x00, 0x20, 0x40])],
    );
    fs::write(
        source.join("transformer/diffusion_pytorch_model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "metadata": {"total_size": 8},
            "weight_map": {
                "patch_embed.proj.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
                "norm_out.weight": "diffusion_pytorch_model-00002-of-00002.safetensors"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
            source.join("vae/config.json"),
            br#"{"_class_name":"AutoencoderKLQwenImage","z_dim":16,"latents_mean":[-0.7571],"latents_std":[2.8184]}"#,
        )
        .unwrap();
    fs::write(
            source.join("scheduler/scheduler_config.json"),
            br#"{"_class_name":"FlowMatchEulerDiscreteScheduler","num_train_timesteps":1000,"shift":1.0,"shift_terminal":0.02,"use_dynamic_shifting":true,"time_shift_type":"exponential"}"#,
        )
        .unwrap();

    let output = dir.join("qwen-image-edit.hfq");
    let summary = import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: Some("qwen-image-edit".into()),
        max_batch: 2,
        metadata_only: false,
    })
    .unwrap();
    let hfq = HfqFile::open_index_only(&output).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    let transformer = config.transformer.as_ref().unwrap();
    let entries = &metadata.components["transformer"].weight_entries;

    assert_eq!(summary.pipeline_class, "QwenImageEditPipeline");
    assert_eq!(metadata.pipeline.latent_channels, Some(16));
    assert_eq!(metadata.batch.max_batch, 2);
    assert_eq!(metadata.tokenizer.entries.len(), 2);
    assert_eq!(transformer.class_name, "QwenImageTransformer2DModel");
    assert_eq!(transformer.in_channels, Some(64));
    assert_eq!(transformer.out_channels, Some(16));
    assert_eq!(transformer.cross_attention_dim, Some(3584));
    assert_eq!(transformer.patch_size, Some(2));
    assert_eq!(transformer.num_layers, Some(60));
    assert_eq!(transformer.num_attention_heads, Some(24));
    assert_eq!(transformer.num_key_value_heads, Some(8));
    assert_eq!(transformer.attention_head_dim, Some(128));
    assert_eq!(transformer.axes_dims_rope, vec![16, 56, 56]);
    assert_eq!(transformer.guidance_embeds, Some(false));
    assert_eq!(transformer.pooled_projection_dim, Some(768));
    assert_eq!(entries.len(), 2);
    assert!(entries.contains(&"transformer/tensors/patch_embed.proj.weight".to_string()));
    assert!(entries.contains(&"transformer/tensors/norm_out.weight".to_string()));
    let patch_embed =
        cpu_tensor_from_hfq(&hfq, "transformer/tensors/patch_embed.proj.weight").unwrap();
    let norm_out = cpu_tensor_from_hfq(&hfq, "transformer/tensors/norm_out.weight").unwrap();
    assert_eq!(patch_embed.data, vec![1.5]);
    assert_eq!(norm_out.data, vec![2.5]);
    assert!(hfq
        .tensor_data_vec("transformer/diffusion_pytorch_model.safetensors.index.json")
        .is_none());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn imports_sdxl_secondary_text_encoder_and_tokenizer_metadata() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-sdxl-import-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let source = dir.join("snapshot");
    fs::create_dir_all(source.join("text_encoder")).unwrap();
    fs::create_dir_all(source.join("text_encoder_2")).unwrap();
    fs::create_dir_all(source.join("unet")).unwrap();
    fs::create_dir_all(source.join("vae")).unwrap();
    fs::create_dir_all(source.join("scheduler")).unwrap();
    fs::create_dir_all(source.join("tokenizer")).unwrap();
    fs::create_dir_all(source.join("tokenizer_2")).unwrap();
    fs::write(
        source.join("model_index.json"),
        br#"{"_class_name":"StableDiffusionXLPipeline"}"#,
    )
    .unwrap();
    fs::write(
            source.join("unet/config.json"),
            br#"{"_class_name":"UNet2DConditionModel","sample_size":128,"in_channels":4,"addition_embed_type":"text_time"}"#,
        )
        .unwrap();
    fs::write(
        source.join("vae/config.json"),
        br#"{"_class_name":"AutoencoderKL","latent_channels":4}"#,
    )
    .unwrap();
    fs::write(
        source.join("text_encoder/config.json"),
        br#"{"_class_name":"CLIPTextModel","hidden_size":768}"#,
    )
    .unwrap();
    fs::write(
            source.join("text_encoder_2/config.json"),
            br#"{"_class_name":"CLIPTextModelWithProjection","hidden_size":1280,"projection_dim":1280}"#,
        )
        .unwrap();
    fs::write(
        source.join("scheduler/scheduler_config.json"),
        br#"{"_class_name":"EulerDiscreteScheduler"}"#,
    )
    .unwrap();
    fs::write(source.join("tokenizer/vocab.json"), b"{}").unwrap();
    fs::write(source.join("tokenizer_2/vocab.json"), b"{}").unwrap();
    fs::write(source.join("text_encoder/pytorch_model.bin"), b"text").unwrap();
    fs::write(source.join("text_encoder_2/pytorch_model.bin"), b"text2").unwrap();
    fs::write(source.join("unet/diffusion_pytorch_model.bin"), b"unet").unwrap();
    fs::write(source.join("vae/diffusion_pytorch_model.bin"), b"vae").unwrap();

    let output = dir.join("tiny-sdxl.hfq");
    import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: Some("tiny-sdxl".into()),
        max_batch: 2,
        metadata_only: false,
    })
    .unwrap();

    let hfq = HfqFile::open_index_only(&output).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    assert_eq!(metadata.pipeline.class_name, "StableDiffusionXLPipeline");
    assert!(metadata.components.contains_key("text_encoder_2"));
    assert_eq!(
        metadata.components["text_encoder_2"]
            .config_entry
            .as_deref(),
        Some("text_encoder_2/config.json")
    );
    assert_eq!(
        metadata.tokenizer_2.as_ref().unwrap().entries,
        vec!["tokenizer_2/vocab.json"]
    );
    assert!(hfq.find_tensor_info("text_encoder_2/config.json").is_some());
    assert!(hfq.find_tensor_info("tokenizer_2/vocab.json").is_some());

    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    assert_eq!(
        config.text_encoder_2.as_ref().unwrap().class_name,
        "CLIPTextModelWithProjection"
    );
    let pipeline = DiffusionPipeline::open_hfq(&output).unwrap();
    assert!(pipeline.native_runtime.is_none());
    let native_runtime_error = pipeline.native_runtime_error.as_deref().unwrap();
    assert!(!native_runtime_error.contains("dual-text-encoder"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn imports_diffusers_safetensors_as_hfq_tensor_entries() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-safetensors-import-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let source = dir.join("snapshot");
    fs::create_dir_all(source.join("text_encoder")).unwrap();
    fs::create_dir_all(source.join("unet")).unwrap();
    fs::create_dir_all(source.join("vae")).unwrap();
    fs::create_dir_all(source.join("scheduler")).unwrap();
    fs::write(
        source.join("model_index.json"),
        br#"{"_class_name":"StableDiffusionPipeline"}"#,
    )
    .unwrap();
    fs::write(
        source.join("unet/config.json"),
        br#"{"_class_name":"UNet2DConditionModel","sample_size":2,"in_channels":1}"#,
    )
    .unwrap();
    fs::write(
        source.join("vae/config.json"),
        br#"{"_class_name":"AutoencoderKL","latent_channels":1}"#,
    )
    .unwrap();
    fs::write(
        source.join("text_encoder/config.json"),
        br#"{"_class_name":"CLIPTextModel"}"#,
    )
    .unwrap();
    fs::write(
        source.join("scheduler/scheduler_config.json"),
        br#"{"_class_name":"EulerDiscreteScheduler"}"#,
    )
    .unwrap();
    write_safetensors_fixture(
        &source.join("unet/diffusion_pytorch_model.safetensors"),
        &[("conv_in.weight", "F32", &[1, 1], &[0x00, 0x00, 0xc0, 0x3f])],
    );
    write_safetensors_fixture(
        &source.join("vae/diffusion_pytorch_model.safetensors"),
        &[("post_quant_conv.weight", "F16", &[1], &[0x00, 0x3c])],
    );
    write_safetensors_fixture(
        &source.join("text_encoder/model.safetensors"),
        &[(
            "text_model.final_layer_norm.weight",
            "BF16",
            &[1],
            &[0x80, 0x3f],
        )],
    );

    let output = dir.join("safe.hfq");
    import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: Some("safe-sd".into()),
        max_batch: 1,
        metadata_only: false,
    })
    .unwrap();

    let hfq = HfqFile::open_index_only(&output).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    assert_eq!(
        metadata.components["unet"].weight_entries,
        vec!["unet/tensors/conv_in.weight"]
    );
    assert_eq!(metadata.components["unet"].tensor_roles[0].dtype, "F32");
    assert_eq!(metadata.components["vae"].tensor_roles[0].dtype, "F16");
    assert_eq!(
        metadata.components["text_encoder"].tensor_roles[0].dtype,
        "BF16"
    );
    let unet = cpu_tensor_from_hfq(&hfq, "unet/tensors/conv_in.weight").unwrap();
    let vae = cpu_tensor_from_hfq(&hfq, "vae/tensors/post_quant_conv.weight").unwrap();
    let text = cpu_tensor_from_hfq(
        &hfq,
        "text_encoder/tensors/text_model.final_layer_norm.weight",
    )
    .unwrap();
    assert_eq!(unet.shape, vec![1, 1]);
    assert_eq!(unet.data, vec![1.5]);
    assert_eq!(vae.data, vec![1.0]);
    assert_eq!(text.data, vec![1.0]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn importer_prefers_safetensors_over_legacy_bin_when_both_exist() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-safetensors-precedence-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let source = dir.join("snapshot");
    fs::create_dir_all(source.join("unet")).unwrap();
    fs::create_dir_all(source.join("scheduler")).unwrap();
    fs::write(
        source.join("model_index.json"),
        br#"{"_class_name":"StableDiffusionPipeline"}"#,
    )
    .unwrap();
    fs::write(
        source.join("unet/config.json"),
        br#"{"_class_name":"UNet2DConditionModel","sample_size":2,"in_channels":1}"#,
    )
    .unwrap();
    fs::write(
        source.join("scheduler/scheduler_config.json"),
        br#"{"_class_name":"EulerDiscreteScheduler"}"#,
    )
    .unwrap();
    fs::write(source.join("unet/diffusion_pytorch_model.bin"), b"opaque").unwrap();
    write_safetensors_fixture(
        &source.join("unet/diffusion_pytorch_model.safetensors"),
        &[("conv_in.bias", "F32", &[1], &[0x00, 0x00, 0x20, 0x40])],
    );

    let output = dir.join("precedence.hfq");
    import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: None,
        max_batch: 1,
        metadata_only: false,
    })
    .unwrap();

    let hfq = HfqFile::open_index_only(&output).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    assert_eq!(
        metadata.components["unet"].weight_entries,
        vec!["unet/tensors/conv_in.bias"]
    );
    let tensor = cpu_tensor_from_hfq(&hfq, "unet/tensors/conv_in.bias").unwrap();
    assert_eq!(tensor.data, vec![2.5]);
    assert!(hfq
        .tensor_data_vec("unet/diffusion_pytorch_model.bin")
        .is_none());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn imports_diffusers_sharded_safetensors_index_as_hfq_tensor_entries() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-sharded-safetensors-import-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let source = dir.join("snapshot");
    fs::create_dir_all(source.join("unet")).unwrap();
    fs::create_dir_all(source.join("scheduler")).unwrap();
    fs::write(
        source.join("model_index.json"),
        br#"{"_class_name":"StableDiffusionPipeline"}"#,
    )
    .unwrap();
    fs::write(
        source.join("unet/config.json"),
        br#"{"_class_name":"UNet2DConditionModel","sample_size":2,"in_channels":1}"#,
    )
    .unwrap();
    fs::write(
        source.join("scheduler/scheduler_config.json"),
        br#"{"_class_name":"EulerDiscreteScheduler"}"#,
    )
    .unwrap();
    write_safetensors_fixture(
        &source.join("unet/diffusion_pytorch_model-00001-of-00002.safetensors"),
        &[("conv_in.weight", "F32", &[1], &[0x00, 0x00, 0xc0, 0x3f])],
    );
    write_safetensors_fixture(
        &source.join("unet/diffusion_pytorch_model-00002-of-00002.safetensors"),
        &[("conv_out.bias", "F32", &[1], &[0x00, 0x00, 0x20, 0x40])],
    );
    fs::write(
        source.join("unet/diffusion_pytorch_model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "metadata": {"total_size": 8},
            "weight_map": {
                "conv_in.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
                "conv_out.bias": "diffusion_pytorch_model-00002-of-00002.safetensors"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let output = dir.join("sharded.hfq");
    import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: None,
        max_batch: 1,
        metadata_only: false,
    })
    .unwrap();

    let hfq = HfqFile::open_index_only(&output).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let entries = &metadata.components["unet"].weight_entries;
    assert_eq!(entries.len(), 2);
    assert!(entries.contains(&"unet/tensors/conv_in.weight".to_string()));
    assert!(entries.contains(&"unet/tensors/conv_out.bias".to_string()));
    let conv_in = cpu_tensor_from_hfq(&hfq, "unet/tensors/conv_in.weight").unwrap();
    let conv_out = cpu_tensor_from_hfq(&hfq, "unet/tensors/conv_out.bias").unwrap();
    assert_eq!(conv_in.data, vec![1.5]);
    assert_eq!(conv_out.data, vec![2.5]);
    assert!(hfq
        .tensor_data_vec("unet/diffusion_pytorch_model.safetensors.index.json")
        .is_none());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn imports_single_file_safetensors_checkpoint_as_component_tensors() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-single-file-import-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::create_dir_all(dir.join("tokenizer")).unwrap();
    fs::write(
        dir.join("tokenizer/vocab.json"),
        br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "cat": 3
            }"#,
    )
    .unwrap();
    fs::write(dir.join("tokenizer/merges.txt"), b"#version: 0.2\n").unwrap();
    let source = dir.join("webui-model.safetensors");
    write_safetensors_fixture(
        &source,
        &[
            (
                "model.diffusion_model.input_blocks.0.0.weight",
                "F32",
                &[1, 4, 1, 1],
                &[
                    0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00,
                    0x00, 0x80, 0x40,
                ],
            ),
            (
                "first_stage_model.decoder.conv_in.weight",
                "F16",
                &[1],
                &[0x00, 0x3c],
            ),
            (
                "cond_stage_model.transformer.text_model.final_layer_norm.weight",
                "BF16",
                &[1],
                &[0x80, 0x3f],
            ),
            (
                "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
                "F32",
                &[1],
                &[0x00, 0x00, 0x80, 0x3f],
            ),
            (
                "model.diffusion_model.input_blocks.1.1.norm.weight",
                "F32",
                &[1],
                &[0x00, 0x00, 0x00, 0x40],
            ),
            (
                "model.diffusion_model.input_blocks.3.0.op.weight",
                "F32",
                &[1],
                &[0x00, 0x00, 0x40, 0x40],
            ),
            (
                "model.diffusion_model.middle_block.0.out_layers.3.bias",
                "F32",
                &[1],
                &[0x00, 0x00, 0x80, 0x40],
            ),
            (
                "model.diffusion_model.middle_block.1.proj_in.weight",
                "F32",
                &[1],
                &[0x00, 0x00, 0xa0, 0x40],
            ),
            (
                "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
                "F32",
                &[1],
                &[0x00, 0x00, 0xc0, 0x40],
            ),
            (
                "model.diffusion_model.output_blocks.2.2.conv.weight",
                "F32",
                &[1],
                &[0x00, 0x00, 0xe0, 0x40],
            ),
        ],
    );

    let output = dir.join("webui-model.hfq");
    let summary = import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: None,
        max_batch: 2,
        metadata_only: false,
    })
    .unwrap();

    assert_eq!(summary.model_name, "webui-model");
    assert_eq!(summary.pipeline_class, "StableDiffusionPipeline");
    assert_eq!(summary.max_batch, 2);
    let hfq = HfqFile::open_index_only(&output).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    assert_eq!(metadata.pipeline.latent_channels, Some(4));
    assert_eq!(config.pipeline_class, "StableDiffusionPipeline");
    assert_eq!(config.unet.sample_size, Some(64));
    assert_eq!(config.unet.block_out_channels, vec![320, 640, 1280, 1280]);
    assert_eq!(config.vae.scaling_factor, Some(0.18215));
    assert_eq!(config.scheduler.class_name, "EulerDiscreteScheduler");
    assert_eq!(
        metadata.components["unet"].config_entry.as_deref(),
        Some("unet/config.json")
    );
    assert_eq!(
        metadata.tokenizer.entries,
        vec!["tokenizer/vocab.json", "tokenizer/merges.txt"]
    );
    assert!(metadata.components.contains_key("unet"));
    assert!(metadata.components.contains_key("vae"));
    assert!(metadata.components.contains_key("text_encoder"));
    assert!(metadata.components["unet"].weight_entries.contains(
        &"unet/checkpoint_tensors/model.diffusion_model.input_blocks.0.0.weight".to_string()
    ));
    assert!(metadata.components["unet"]
        .weight_entries
        .contains(&"unet/tensors/conv_in.weight".to_string()));
    assert!(metadata.components["vae"]
        .weight_entries
        .contains(&"vae/tensors/decoder.conv_in.weight".to_string()));
    assert!(metadata.components["text_encoder"]
        .weight_entries
        .contains(&"text_encoder/tensors/text_model.final_layer_norm.weight".to_string()));
    for expected in [
        "unet/tensors/down_blocks.0.resnets.0.norm1.weight",
        "unet/tensors/down_blocks.0.attentions.0.norm.weight",
        "unet/tensors/down_blocks.0.downsamplers.0.conv.weight",
        "unet/tensors/mid_block.resnets.0.conv2.bias",
        "unet/tensors/mid_block.attentions.0.proj_in.weight",
        "unet/tensors/up_blocks.0.resnets.0.conv_shortcut.weight",
        "unet/tensors/up_blocks.0.upsamplers.0.conv.weight",
    ] {
        assert!(
            metadata.components["unet"]
                .weight_entries
                .contains(&expected.to_string()),
            "missing projected native entry {expected}"
        );
    }
    let checkpoint_tensor = cpu_tensor_from_hfq(
        &hfq,
        "unet/checkpoint_tensors/model.diffusion_model.input_blocks.0.0.weight",
    )
    .unwrap();
    let native_tensor = cpu_tensor_from_hfq(&hfq, "unet/tensors/conv_in.weight").unwrap();
    let tokenizer = ClipTokenizer::from_hfq_file(&hfq).unwrap();
    let tokens = tokenizer.encode_padded("a cat");
    let down_resnet =
        cpu_tensor_from_hfq(&hfq, "unet/tensors/down_blocks.0.resnets.0.norm1.weight").unwrap();
    let upsample =
        cpu_tensor_from_hfq(&hfq, "unet/tensors/up_blocks.0.upsamplers.0.conv.weight").unwrap();
    assert_eq!(checkpoint_tensor.shape, vec![1, 4, 1, 1]);
    assert_eq!(checkpoint_tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(native_tensor.shape, checkpoint_tensor.shape);
    assert_eq!(native_tensor.data, checkpoint_tensor.data);
    assert_eq!(&tokens[..4], &[0, 2, 3, 1]);
    assert_eq!(down_resnet.data, vec![1.0]);
    assert_eq!(upsample.data, vec![7.0]);
    let pipeline = DiffusionPipeline::open_hfq(&output).unwrap();
    assert!(pipeline.native_runtime.is_none());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn imports_single_file_sdxl_safetensors_checkpoint_metadata() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-single-file-sdxl-import-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::create_dir_all(dir.join("tokenizer_2")).unwrap();
    fs::write(
        dir.join("tokenizer_2/vocab.json"),
        br#"{
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "wide": 2
            }"#,
    )
    .unwrap();
    fs::write(dir.join("tokenizer_2/merges.txt"), b"#version: 0.2\n").unwrap();
    let source = dir.join("webui-sdxl.safetensors");
    write_safetensors_fixture(
        &source,
        &[
            (
                "model.diffusion_model.input_blocks.0.0.weight",
                "F32",
                &[1, 4, 1, 1],
                &[
                    0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00,
                    0x00, 0x80, 0x40,
                ],
            ),
            (
                "conditioner.embedders.1.model.text_model.final_layer_norm.weight",
                "F32",
                &[1],
                &[0x00, 0x00, 0x80, 0x3f],
            ),
        ],
    );

    let output = dir.join("webui-sdxl.hfq");
    let summary = import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: Some("webui-sdxl".into()),
        max_batch: 1,
        metadata_only: false,
    })
    .unwrap();

    assert_eq!(summary.pipeline_class, "StableDiffusionXLPipeline");
    let hfq = HfqFile::open_index_only(&output).unwrap();
    let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
    let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
    assert_eq!(
        config.unet.addition_embed_type.as_deref(),
        Some("text_time")
    );
    assert_eq!(
        config.text_encoder_2.as_ref().unwrap().hidden_size,
        Some(1280)
    );
    assert!(metadata.components.contains_key("text_encoder_2"));
    assert_eq!(
        metadata.tokenizer_2.as_ref().unwrap().entries,
        vec!["tokenizer_2/vocab.json", "tokenizer_2/merges.txt"]
    );
    assert!(metadata.components["text_encoder_2"].weight_entries.contains(
            &"text_encoder_2/checkpoint_tensors/conditioner.embedders.1.model.text_model.final_layer_norm.weight".to_string()
        ));
    assert!(metadata.components["text_encoder_2"]
        .weight_entries
        .contains(&"text_encoder_2/tensors/text_model.final_layer_norm.weight".to_string()));
    let tokenizer_2 = ClipTokenizer::from_hfq_file_with_prefix(&hfq, "tokenizer_2").unwrap();
    assert_eq!(&tokenizer_2.encode_padded("wide")[..3], &[0, 2, 1]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn single_file_checkpoint_projection_loads_tiny_native_unet() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-single-file-native-unet-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let source = dir.join("tiny-ldm.safetensors");
    write_tiny_ldm_unet_safetensors(&source);

    let output = dir.join("tiny-ldm.hfq");
    import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: Some("tiny-ldm".into()),
        max_batch: 1,
        metadata_only: false,
    })
    .unwrap();

    let hfq = HfqFile::open_index_only(&output).unwrap();
    let config = tiny_runtime_config();
    let unet = NativeUnet2DConditionModel::from_hfq(&hfq, &config.unet).unwrap();
    let encoder = NativeVaeEncoder::from_hfq(&hfq, &config.vae).unwrap();
    let decoder = NativeVaeDecoder::from_hfq(&hfq, &config.vae).unwrap();
    let sample = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![0.25, -0.25, 0.5, -0.5],
    };
    let encoder_states = CpuTensor {
        shape: vec![1, 4, 2],
        data: vec![0.0; 8],
    };

    let output = unet.forward(&sample, &[0.0], &encoder_states).unwrap();

    assert_eq!(output.shape, sample.shape);
    assert!(output.data.iter().all(|value| value.is_finite()));
    let latents = encoder
        .encode_to_latents(&RgbImageBatch {
            batch: 1,
            width: 2,
            height: 2,
            data: vec![255; 12],
        })
        .unwrap();
    assert_eq!(latents.batch, 1);
    assert_eq!(latents.channels, 1);
    assert_eq!(latents.height, 2);
    assert_eq!(latents.width, 2);
    assert!(latents.data.iter().all(|value| value.is_finite()));
    let rgb = decoder
        .decode_to_rgb8(&LatentBatch {
            batch: 1,
            channels: 1,
            height: 2,
            width: 2,
            data: output.data,
        })
        .unwrap();
    assert_eq!(rgb.batch, 1);
    assert_eq!(rgb.width, 2);
    assert_eq!(rgb.height, 2);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn single_file_checkpoint_projection_loads_tiny_text_conditioning() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-single-file-text-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("tokenizer")).unwrap();
    fs::write(
        dir.join("tokenizer/vocab.json"),
        br#"{"<|startoftext|>":0,"<|endoftext|>":1,"cat":2}"#,
    )
    .unwrap();
    fs::write(dir.join("tokenizer/merges.txt"), b"#version: 0.2\n").unwrap();
    let source = dir.join("tiny-ldm.safetensors");
    write_tiny_ldm_unet_safetensors(&source);

    let output = dir.join("tiny-ldm.hfq");
    import_diffusers_to_hfq(DiffusersImportOptions {
        source,
        output: output.clone(),
        model_name: Some("tiny-ldm".into()),
        max_batch: 1,
        metadata_only: false,
    })
    .unwrap();

    let hfq = HfqFile::open_index_only(&output).unwrap();
    let tokenizer = ClipTokenizer::from_hfq_file(&hfq).unwrap();
    let tokens = tokenizer.encode_padded("cat");
    assert_eq!(&tokens[..3], &[0, 2, 1]);
    let text_encoder = ClipTextEncoder::from_hfq_file_with_heads(&hfq, 1).unwrap();
    let hidden_states = text_encoder.encode_tokens(&tokens).unwrap();
    assert_eq!(hidden_states.shape, vec![77, 2]);
    assert!(hidden_states.data.iter().all(|value| value.is_finite()));
    let (hidden_states, pooled) = text_encoder
        .encode_tokens_with_pooled(&tokens, tokenizer.end_token_id())
        .unwrap();
    assert_eq!(hidden_states.shape, vec![77, 2]);
    let pooled = pooled.unwrap();
    assert_eq!(pooled.len(), 2);
    assert!(pooled.iter().all(|value| value.is_finite()));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn parses_tiny_sd_pytorch_tensor_indexes_when_cache_is_present() {
    let root = Path::new("/srv/huggingface/models--segmind--tiny-sd/snapshots/cad0bd7495fa6c4bcca01b19a723dc91627fe84f");
    if !root.exists() {
        eprintln!("skip: tiny-sd cache not present");
        return;
    }

    let unet = parse_pytorch_state_dict(&root.join("unet/diffusion_pytorch_model.bin")).unwrap();
    let vae = parse_pytorch_state_dict(&root.join("vae/diffusion_pytorch_model.bin")).unwrap();
    let text = parse_pytorch_state_dict(&root.join("text_encoder/pytorch_model.bin")).unwrap();

    assert!(unet
        .iter()
        .any(|tensor| tensor.name == "conv_in.weight" && tensor.shape == [320, 4, 3, 3]));
    assert!(vae
        .iter()
        .any(|tensor| tensor.name == "decoder.conv_out.weight"));
    assert!(text
        .iter()
        .any(|tensor| tensor.name == "text_model.embeddings.token_embedding.weight"));
}
