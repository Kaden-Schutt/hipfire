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
fn native_runtime_metadata_support_reports_runtime_boundaries() {
    let mut metadata = minimal_metadata();
    assert!(native_runtime_metadata_support_error(&metadata).is_none());

    metadata.quantization.weight_format = "metadata-only".to_string();
    let error = native_runtime_metadata_support_error(&metadata).unwrap();
    assert!(error.contains("metadata only"));

    metadata.quantization.weight_format = "oq4".to_string();
    assert!(native_runtime_metadata_support_error(&metadata).is_none());

    metadata.quantization.activation_format = "fp8".to_string();
    let error = native_runtime_metadata_support_error(&metadata).unwrap();
    assert!(error.contains("activation_format"));
    assert!(error.contains("fp8"));

    metadata.quantization.activation_format = "fp16".to_string();
    metadata.quantization.tensor_roles_version = 2;
    let error = native_runtime_metadata_support_error(&metadata).unwrap();
    assert!(error.contains("tensor_roles_version 2"));
}

#[test]
fn native_source_runtime_support_rejects_incomplete_transformer_weights() {
    // A Krea2 pipeline with no transformer weights is a recognized family but
    // an incomplete artifact, so it is rejected as incomplete (not as an
    // unsupported family — Krea2/QwenImage transformers are supported).
    let mut metadata = minimal_metadata();
    metadata.pipeline.class_name = "Krea2Pipeline".to_string();
    metadata.components.remove("unet");
    metadata.components.insert(
        "transformer".to_string(),
        DiffusionComponentMetadata {
            class_name: Some("Krea2Transformer2DModel".to_string()),
            config_entry: Some("transformer/config.json".to_string()),
            weight_entries: Vec::new(),
            tensor_roles: Vec::new(),
        },
    );

    let error = native_runtime_metadata_support_error(&metadata).unwrap();

    assert!(error.contains("requires complete"));
    assert!(error.contains("krea2-mmdit"));
}

#[test]
fn native_transformer_io_projects_qwen_patch_tokens() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-io-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-io.hfq");
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(
                "transformer/tensors/img_in.weight",
                &[2, 4],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
            f32_mem_tensor("transformer/tensors/img_in.bias", &[2], &[0.5, -0.5]),
            f32_mem_tensor(
                "transformer/tensors/proj_out.weight",
                &[4, 2],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ),
            f32_mem_tensor(
                "transformer/tensors/proj_out.bias",
                &[4],
                &[0.0, 0.0, 1.0, -1.0],
            ),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let mut config = tiny_runtime_config();
    config.transformer = Some(TransformerDenoiserConfig {
        class_name: "QwenImageTransformer2DModel".into(),
        in_channels: Some(4),
        out_channels: Some(1),
        patch_size: Some(2),
        ..TransformerDenoiserConfig::default()
    });
    config.latent_channels = 1;
    let topology = transformer_denoiser_weight_topology(&DiffusionComponentMetadata {
        class_name: Some("QwenImageTransformer2DModel".to_string()),
        weight_entries: vec![
            "transformer/tensors/img_in.weight".to_string(),
            "transformer/tensors/proj_out.weight".to_string(),
            "transformer/tensors/transformer_blocks.0.txt_mod.1.weight".to_string(),
        ],
        ..DiffusionComponentMetadata::default()
    });
    let io = NativeTransformerDenoiserIo::from_hfq(&hfq, &config, &topology).unwrap();
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 2,
        width: 2,
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let hidden = io
        .project_latents_to_hidden_with_runtime_context(&latents, &mut runtime_context)
        .unwrap();
    assert_eq!(hidden.shape, vec![1, 1, 2]);
    assert_eq!(hidden.data, vec![1.5, 1.5]);
    let timestep_embedding = CpuTensor {
        shape: vec![1, 2],
        data: vec![0.0, 0.0],
    };
    let output = io
        .project_hidden_to_latents_with_runtime_context(
            &hidden,
            &timestep_embedding,
            1,
            2,
            2,
            &mut runtime_context,
        )
        .unwrap();
    assert_eq!(output.batch, 1);
    assert_eq!(output.channels, 1);
    assert_eq!(output.height, 2);
    assert_eq!(output.width, 2);
    assert_eq!(output.data, vec![1.5, 1.5, 4.0, -1.0]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_io_projects_krea_final_layer_tokens() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-krea-transformer-io-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("krea-transformer-io.hfq");
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(
                "transformer/tensors/img_in.weight",
                &[2, 4],
                &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor("transformer/tensors/img_in.bias", &[2], &[0.0, 0.25]),
            f32_mem_tensor(
                "transformer/tensors/final_layer.linear.weight",
                &[4, 2],
                &[1.0, 1.0, -1.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            ),
            f32_mem_tensor(
                "transformer/tensors/final_layer.linear.bias",
                &[4],
                &[0.0, 1.0, 0.0, -1.0],
            ),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let mut config = tiny_runtime_config();
    config.transformer = Some(TransformerDenoiserConfig {
        class_name: "Krea2Transformer2DModel".into(),
        in_channels: Some(4),
        out_channels: None,
        patch_size: Some(2),
        ..TransformerDenoiserConfig::default()
    });
    config.latent_channels = 1;
    let topology = transformer_denoiser_weight_topology(&DiffusionComponentMetadata {
        class_name: Some("Krea2Transformer2DModel".to_string()),
        weight_entries: vec![
            "transformer/tensors/img_in.weight".to_string(),
            "transformer/tensors/final_layer.linear.weight".to_string(),
            "transformer/tensors/text_fusion.projector.weight".to_string(),
        ],
        ..DiffusionComponentMetadata::default()
    });
    let io = NativeTransformerDenoiserIo::from_hfq(&hfq, &config, &topology).unwrap();
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 2,
        width: 2,
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let hidden = io
        .project_latents_to_hidden_with_runtime_context(&latents, &mut runtime_context)
        .unwrap();
    assert_eq!(hidden.shape, vec![1, 1, 2]);
    assert_eq!(hidden.data, vec![3.0, 4.25]);
    let timestep_embedding = CpuTensor {
        shape: vec![1, 2],
        data: vec![0.0, 0.0],
    };
    let output = io
        .project_hidden_to_latents_with_runtime_context(
            &hidden,
            &timestep_embedding,
            1,
            2,
            2,
            &mut runtime_context,
        )
        .unwrap();
    assert_eq!(output.data, vec![7.25, 2.25, 6.0, 7.5]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_krea_block_zero_gate_preserves_residual() {
    // A Krea2 single-stream block with an all-zero scale_shift_table and zero
    // time modulation drives every adaLN gate (pregate/postgate) to zero, so
    // the gated residuals collapse to the identity regardless of the
    // attention / feed-forward weights. This exercises the full Krea block
    // forward (RMSNorm -> adaLN -> GQA+gate attention -> SwiGLU) and asserts
    // the residual structure is wired correctly.
    let dir = std::env::temp_dir().join(format!(
        "hipfire-krea-transformer-block-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("krea-transformer-block.hfq");
    let zeros4 = [0.0f32; 4];
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            // hidden = 2, heads = 1, head_dim = 2, ffn = 2.
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.scale_shift_table",
                &[6, 2],
                &[0.0f32; 12],
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.norm1.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.norm2.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.attn.to_q.weight",
                &[2, 2],
                &zeros4,
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.attn.to_k.weight",
                &[2, 2],
                &zeros4,
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.attn.to_v.weight",
                &[2, 2],
                &zeros4,
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.attn.to_gate.weight",
                &[2, 2],
                &zeros4,
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.attn.to_out.0.weight",
                &[2, 2],
                &zeros4,
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.ff.gate.weight",
                &[2, 2],
                &zeros4,
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.ff.up.weight",
                &[2, 2],
                &zeros4,
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.ff.down.weight",
                &[2, 2],
                &zeros4,
            ),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let block =
        NativeTransformerBlock::from_hfq(&hfq, TransformerDenoiserFamily::Krea2, 0, 1).unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());
    // Joint [text; image] sequence of 3 tokens, hidden width 2.
    let hidden = CpuTensor {
        shape: vec![1, 3, 2],
        data: vec![1.0, 2.0, -3.0, 0.5, 4.0, -1.0],
    };
    let time_modulation = CpuTensor {
        shape: vec![1, 12],
        data: vec![0.0f32; 12],
    };
    let output = block
        .forward_krea_with_runtime_context(&hidden, &time_modulation, None, &mut runtime_context)
        .unwrap();
    assert_eq!(output.shape, hidden.shape);
    assert_eq!(output.data, hidden.data);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_timestep_embedding_loads_qwen_layout() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-time-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-time.hfq");
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(
                "transformer/tensors/time_text_embed.timestep_embedder.linear_1.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor(
                "transformer/tensors/time_text_embed.timestep_embedder.linear_1.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "transformer/tensors/time_text_embed.timestep_embedder.linear_2.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor(
                "transformer/tensors/time_text_embed.timestep_embedder.linear_2.bias",
                &[2],
                &[0.0, 0.0],
            ),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let embedding =
        NativeTransformerTimestepEmbedding::from_hfq(&hfq, TransformerDenoiserFamily::QwenImage)
            .unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let output = embedding
        .forward_with_runtime_context(&[0.0], &mut runtime_context)
        .unwrap();
    assert_eq!(output.shape, vec![1, 2]);
    assert!((output.data[0] - silu(1.0)).abs() < 1e-6);
    assert!(output.data[1].abs() < 1e-6);
    assert!(embedding
        .modulation_with_runtime_context(&output, &mut runtime_context)
        .unwrap()
        .is_none());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_timestep_embedding_loads_krea_mod_projection() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-krea-transformer-time-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("krea-transformer-time.hfq");
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(
                "transformer/tensors/time_embed.linear_1.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor(
                "transformer/tensors/time_embed.linear_1.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "transformer/tensors/time_embed.linear_2.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 2.0],
            ),
            f32_mem_tensor(
                "transformer/tensors/time_embed.linear_2.bias",
                &[2],
                &[0.25, -0.5],
            ),
            f32_mem_tensor(
                "transformer/tensors/time_mod_proj.weight",
                &[3, 2],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            f32_mem_tensor(
                "transformer/tensors/time_mod_proj.bias",
                &[3],
                &[0.0, 1.0, -1.0],
            ),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let embedding =
        NativeTransformerTimestepEmbedding::from_hfq(&hfq, TransformerDenoiserFamily::Krea2)
            .unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let output = embedding
        .forward_with_runtime_context(&[0.0], &mut runtime_context)
        .unwrap();
    let expected = [silu(1.0) + 0.25, -0.5];
    assert_eq!(output.shape, vec![1, 2]);
    assert!((output.data[0] - expected[0]).abs() < 1e-6);
    assert!((output.data[1] - expected[1]).abs() < 1e-6);
    let modulation = embedding
        .modulation_with_runtime_context(&output, &mut runtime_context)
        .unwrap()
        .unwrap();
    assert_eq!(modulation.shape, vec![1, 3]);
    let expected_modulation = [
        expected[0],
        expected[1] + 1.0,
        expected[0] + expected[1] - 1.0,
    ];
    for (actual, expected) in modulation.data.iter().zip(expected_modulation) {
        assert!((actual - expected).abs() < 1e-6);
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_block_modulation_splits_qwen_image_and_text_chunks() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-mod-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-mod.hfq");
    let mut image_weight = Vec::new();
    let mut text_weight = Vec::new();
    for row in 0..12 {
        image_weight.extend_from_slice(if row % 2 == 0 {
            &[1.0, 0.0]
        } else {
            &[0.0, 1.0]
        });
        text_weight.extend_from_slice(if row % 2 == 0 {
            &[2.0, 0.0]
        } else {
            &[0.0, 2.0]
        });
    }
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.img_mod.1.weight",
                &[12, 2],
                &image_weight,
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.img_mod.1.bias",
                &[12],
                &[0.0; 12],
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.txt_mod.1.weight",
                &[12, 2],
                &text_weight,
            ),
            f32_mem_tensor(
                "transformer/tensors/transformer_blocks.0.txt_mod.1.bias",
                &[12],
                &[1.0; 12],
            ),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let modulation =
        NativeTransformerBlockModulation::from_hfq(&hfq, TransformerDenoiserFamily::QwenImage, 0)
            .unwrap();
    let timestep = CpuTensor {
        shape: vec![1, 2],
        data: vec![1.0, 2.0],
    };
    let silu_values = [silu(1.0), silu(2.0)];
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let image = modulation
        .qwen_image_modulation_with_runtime_context(
            &timestep,
            TransformerModulationStream::Image,
            &mut runtime_context,
        )
        .unwrap();
    assert_eq!(image.shift_msa.shape, vec![1, 2]);
    assert!((image.shift_msa.data[0] - silu_values[0]).abs() < 1e-6);
    assert!((image.shift_msa.data[1] - silu_values[1]).abs() < 1e-6);
    assert_eq!(image.gate_mlp.shape, vec![1, 2]);
    assert!((image.gate_mlp.data[0] - silu_values[0]).abs() < 1e-6);
    assert!((image.gate_mlp.data[1] - silu_values[1]).abs() < 1e-6);

    let text = modulation
        .qwen_image_modulation_with_runtime_context(
            &timestep,
            TransformerModulationStream::Text,
            &mut runtime_context,
        )
        .unwrap();
    assert!((text.scale_msa.data[0] - (2.0 * silu_values[0] + 1.0)).abs() < 1e-6);
    assert!((text.scale_msa.data[1] - (2.0 * silu_values[1] + 1.0)).abs() < 1e-6);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_block_modulation_applies_krea_scale_shift_table() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-krea-transformer-mod-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("krea-transformer-mod.hfq");
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[f32_mem_tensor(
            "transformer/tensors/transformer_blocks.0.scale_shift_table",
            &[3, 2],
            &[0.5, -0.5, 1.0, 2.0, -1.0, 0.25],
        )],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let modulation =
        NativeTransformerBlockModulation::from_hfq(&hfq, TransformerDenoiserFamily::Krea2, 0)
            .unwrap();
    let time_modulation = CpuTensor {
        shape: vec![2, 6],
        data: vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
        ],
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let out = modulation
        .krea_scale_shift_with_runtime_context(&time_modulation, &mut runtime_context)
        .unwrap();

    assert_eq!(out.shape, vec![2, 3, 2]);
    assert_eq!(
        out.data,
        vec![1.5, 1.5, 4.0, 6.0, 4.0, 6.25, -0.5, -2.5, -2.0, -2.0, -6.0, -5.75]
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_attention_projects_qwen_image_and_text_qkv() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-attn-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-attn.hfq");
    let prefix = "transformer/tensors/transformer_blocks.0.attn";
    let identity4 = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let double_identity4 = [
        2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0,
    ];
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(&format!("{prefix}.to_q.weight"), &[4, 4], &identity4),
            f32_mem_tensor(&format!("{prefix}.to_q.bias"), &[4], &[0.0, 0.0, 0.0, 0.0]),
            f32_mem_tensor(&format!("{prefix}.to_k.weight"), &[4, 4], &identity4),
            f32_mem_tensor(&format!("{prefix}.to_v.weight"), &[4, 4], &identity4),
            f32_mem_tensor(&format!("{prefix}.norm_q.weight"), &[2], &[1.0, 2.0]),
            f32_mem_tensor(&format!("{prefix}.norm_k.weight"), &[2], &[0.5, 1.5]),
            f32_mem_tensor(&format!("{prefix}.to_out.0.weight"), &[4, 4], &identity4),
            f32_mem_tensor(
                &format!("{prefix}.to_out.0.bias"),
                &[4],
                &[0.25, -0.25, 1.0, -1.0],
            ),
            f32_mem_tensor(
                &format!("{prefix}.add_q_proj.weight"),
                &[4, 4],
                &double_identity4,
            ),
            f32_mem_tensor(
                &format!("{prefix}.add_k_proj.weight"),
                &[4, 4],
                &double_identity4,
            ),
            f32_mem_tensor(
                &format!("{prefix}.add_v_proj.weight"),
                &[4, 4],
                &double_identity4,
            ),
            f32_mem_tensor(&format!("{prefix}.norm_added_q.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.norm_added_k.weight"), &[2], &[2.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.to_add_out.weight"), &[4, 4], &identity4),
            f32_mem_tensor(
                &format!("{prefix}.to_add_out.bias"),
                &[4],
                &[1.0, 0.0, -1.0, 0.5],
            ),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let attention = NativeTransformerAttentionProjection::from_hfq(
        &hfq,
        TransformerDenoiserFamily::QwenImage,
        0,
        2,
    )
    .unwrap();
    let hidden = CpuTensor {
        shape: vec![1, 2, 4],
        data: vec![3.0, 4.0, 0.0, 5.0, 0.0, 0.0, 6.0, 8.0],
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let image = attention
        .project_image_qkv_with_runtime_context(&hidden, &mut runtime_context)
        .unwrap();
    assert_eq!(image.q.shape, vec![1, 2, 4]);
    assert_f32_close(
        &image.q.data,
        &rms_norm_heads_reference(&hidden.data, 2, 2, &[1.0, 2.0]),
        1e-5,
    );
    assert_f32_close(
        &image.k.data,
        &rms_norm_heads_reference(&hidden.data, 2, 2, &[0.5, 1.5]),
        1e-5,
    );
    assert_eq!(image.v.data, hidden.data);
    let image_out = attention
        .project_image_output_with_runtime_context(&image.v, &mut runtime_context)
        .unwrap();
    assert_eq!(image_out.shape, vec![1, 2, 4]);
    assert_eq!(
        image_out.data,
        vec![3.25, 3.75, 1.0, 4.0, 0.25, -0.25, 7.0, 7.0]
    );

    let text = attention
        .project_text_qkv_with_runtime_context(&hidden, &mut runtime_context)
        .unwrap()
        .unwrap();
    let doubled = hidden
        .data
        .iter()
        .map(|value| value * 2.0)
        .collect::<Vec<_>>();
    assert_f32_close(
        &text.q.data,
        &rms_norm_heads_reference(&doubled, 2, 2, &[1.0, 1.0]),
        1e-5,
    );
    assert_f32_close(
        &text.k.data,
        &rms_norm_heads_reference(&doubled, 2, 2, &[2.0, 1.0]),
        1e-5,
    );
    assert_eq!(text.v.data, doubled);
    let text_out = attention
        .project_text_output_with_runtime_context(&text.v, &mut runtime_context)
        .unwrap()
        .unwrap();
    assert_eq!(
        text_out.data,
        vec![7.0, 8.0, -1.0, 10.5, 1.0, 0.0, 11.0, 16.5]
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_attention_projects_krea_image_qkv_without_text_path() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-krea-transformer-attn-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("krea-transformer-attn.hfq");
    let prefix = "transformer/tensors/transformer_blocks.0.attn";
    let identity4 = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(&format!("{prefix}.to_q.weight"), &[4, 4], &identity4),
            f32_mem_tensor(&format!("{prefix}.to_k.weight"), &[4, 4], &identity4),
            f32_mem_tensor(&format!("{prefix}.to_v.weight"), &[4, 4], &identity4),
            f32_mem_tensor(&format!("{prefix}.to_out.0.weight"), &[4, 4], &identity4),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let attention = NativeTransformerAttentionProjection::from_hfq(
        &hfq,
        TransformerDenoiserFamily::Krea2,
        0,
        2,
    )
    .unwrap();
    let hidden = CpuTensor {
        shape: vec![1, 1, 4],
        data: vec![1.0, -2.0, 3.0, -4.0],
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let image = attention
        .project_image_qkv_with_runtime_context(&hidden, &mut runtime_context)
        .unwrap();
    assert_eq!(image.q.data, hidden.data);
    assert_eq!(image.k.data, hidden.data);
    assert_eq!(image.v.data, hidden.data);
    assert!(attention
        .project_text_qkv_with_runtime_context(&hidden, &mut runtime_context)
        .unwrap()
        .is_none());
    let image_out = attention
        .project_image_output_with_runtime_context(&image.v, &mut runtime_context)
        .unwrap();
    assert_eq!(image_out.data, hidden.data);
    assert!(attention
        .project_text_output_with_runtime_context(&image.v, &mut runtime_context)
        .unwrap()
        .is_none());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_attention_runs_qwen_joint_image_text_attention() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-joint-attn-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-joint-attn.hfq");
    let prefix = "transformer/tensors/transformer_blocks.0.attn";
    let identity2 = [1.0, 0.0, 0.0, 1.0];
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(&format!("{prefix}.to_q.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.to_k.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.to_v.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.to_out.0.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.add_q_proj.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.add_k_proj.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.add_v_proj.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.to_add_out.weight"), &[2, 2], &identity2),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let attention = NativeTransformerAttentionProjection::from_hfq(
        &hfq,
        TransformerDenoiserFamily::QwenImage,
        0,
        1,
    )
    .unwrap();
    let image_hidden = CpuTensor {
        shape: vec![1, 2, 2],
        data: vec![1.0, 0.0, 0.0, 1.0],
    };
    let text_hidden = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![1.0, 1.0],
    };
    let joint = concat_sequence_3d(&text_hidden, &image_hidden).unwrap();
    let expected_image = scaled_dot_product_attention(&image_hidden, &joint, &joint, 1).unwrap();
    let expected_text = scaled_dot_product_attention(&text_hidden, &joint, &joint, 1).unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let (image_out, text_out) = attention
        .attend_image_text_with_runtime_context(
            &image_hidden,
            Some(&text_hidden),
            None,
            None,
            &mut runtime_context,
        )
        .unwrap();

    assert_f32_close(&image_out.data, &expected_image.data, 1e-6);
    assert_f32_close(&text_out.unwrap().data, &expected_text.data, 1e-6);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_attention_masks_qwen_text_keys_but_keeps_image_keys() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-mask-attn-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-mask-attn.hfq");
    let prefix = "transformer/tensors/transformer_blocks.0.attn";
    let identity2 = [1.0, 0.0, 0.0, 1.0];
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(&format!("{prefix}.to_q.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.to_k.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.to_v.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.to_out.0.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.add_q_proj.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.add_k_proj.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.add_v_proj.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.to_add_out.weight"), &[2, 2], &identity2),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let attention = NativeTransformerAttentionProjection::from_hfq(
        &hfq,
        TransformerDenoiserFamily::QwenImage,
        0,
        1,
    )
    .unwrap();
    let image_hidden = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![0.0, 1.0],
    };
    let text_hidden = CpuTensor {
        shape: vec![1, 2, 2],
        data: vec![1.0, 0.0, 8.0, 0.0],
    };
    let text_attention_mask = CpuTensor {
        shape: vec![1, 2],
        data: vec![1.0, 0.0],
    };
    let joint = concat_sequence_3d(&text_hidden, &image_hidden).unwrap();
    let expected_mask = [true, false, true];
    let expected_image = scaled_dot_product_attention_with_key_mask(
        &image_hidden,
        &joint,
        &joint,
        1,
        Some(&expected_mask),
    )
    .unwrap();
    let expected_text = scaled_dot_product_attention_with_key_mask(
        &text_hidden,
        &joint,
        &joint,
        1,
        Some(&expected_mask),
    )
    .unwrap();
    let unmasked_image = scaled_dot_product_attention(&image_hidden, &joint, &joint, 1).unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let (image_out, text_out) = attention
        .attend_image_text_with_runtime_context(
            &image_hidden,
            Some(&text_hidden),
            Some(&text_attention_mask),
            None,
            &mut runtime_context,
        )
        .unwrap();

    assert_f32_close(&image_out.data, &expected_image.data, 1e-6);
    assert_f32_close(&text_out.unwrap().data, &expected_text.data, 1e-6);
    assert!(image_out
        .data
        .iter()
        .zip(unmasked_image.data.iter())
        .any(|(a, b)| (a - b).abs() > 1e-5));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_attention_applies_qwen_rope_to_image_and_text_qk() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-rope-attn-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-rope-attn.hfq");
    let prefix = "transformer/tensors/transformer_blocks.0.attn";
    let identity6 = (0..36)
        .map(|idx| {
            let row = idx / 6;
            let col = idx % 6;
            if row == col {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(&format!("{prefix}.to_q.weight"), &[6, 6], &identity6),
            f32_mem_tensor(&format!("{prefix}.to_k.weight"), &[6, 6], &identity6),
            f32_mem_tensor(&format!("{prefix}.to_v.weight"), &[6, 6], &identity6),
            f32_mem_tensor(&format!("{prefix}.to_out.0.weight"), &[6, 6], &identity6),
            f32_mem_tensor(&format!("{prefix}.add_q_proj.weight"), &[6, 6], &identity6),
            f32_mem_tensor(&format!("{prefix}.add_k_proj.weight"), &[6, 6], &identity6),
            f32_mem_tensor(&format!("{prefix}.add_v_proj.weight"), &[6, 6], &identity6),
            f32_mem_tensor(&format!("{prefix}.to_add_out.weight"), &[6, 6], &identity6),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let attention = NativeTransformerAttentionProjection::from_hfq(
        &hfq,
        TransformerDenoiserFamily::QwenImage,
        0,
        1,
    )
    .unwrap();
    let image_hidden = CpuTensor {
        shape: vec![1, 2, 6],
        data: vec![
            1.0, 0.0, 0.0, 1.0, 0.5, -0.5, //
            -0.25, 0.75, 1.5, -1.0, 2.0, 0.25,
        ],
    };
    let text_hidden = CpuTensor {
        shape: vec![1, 1, 6],
        data: vec![0.25, -0.5, 0.75, 1.0, -1.25, 0.5],
    };
    let rotary = qwen_rotary_embeddings_for_grid([2, 2, 2], 10_000.0, 6, 1, 1, 2, 1).unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let mut image_qkv = attention
        .project_image_qkv_with_runtime_context(&image_hidden, &mut runtime_context)
        .unwrap();
    image_qkv.q = apply_qwen_rotary_embedding(&image_qkv.q, &rotary.image, 1, 6).unwrap();
    image_qkv.k = apply_qwen_rotary_embedding(&image_qkv.k, &rotary.image, 1, 6).unwrap();
    let mut text_qkv = attention
        .project_text_qkv_with_runtime_context(&text_hidden, &mut runtime_context)
        .unwrap()
        .unwrap();
    text_qkv.q = apply_qwen_rotary_embedding(&text_qkv.q, &rotary.text, 1, 6).unwrap();
    text_qkv.k = apply_qwen_rotary_embedding(&text_qkv.k, &rotary.text, 1, 6).unwrap();
    let joint_k = concat_sequence_3d(&text_qkv.k, &image_qkv.k).unwrap();
    let joint_v = concat_sequence_3d(&text_qkv.v, &image_qkv.v).unwrap();
    let expected_image = scaled_dot_product_attention(&image_qkv.q, &joint_k, &joint_v, 1).unwrap();
    let expected_text = scaled_dot_product_attention(&text_qkv.q, &joint_k, &joint_v, 1).unwrap();

    let (image_out, text_out) = attention
        .attend_image_text_with_runtime_context(
            &image_hidden,
            Some(&text_hidden),
            None,
            Some(&rotary),
            &mut runtime_context,
        )
        .unwrap();
    let (no_rope_image, _) = attention
        .attend_image_text_with_runtime_context(
            &image_hidden,
            Some(&text_hidden),
            None,
            None,
            &mut runtime_context,
        )
        .unwrap();

    assert_f32_close(&image_out.data, &expected_image.data, 1e-6);
    assert_f32_close(&text_out.unwrap().data, &expected_text.data, 1e-6);
    assert!(image_out
        .data
        .iter()
        .zip(no_rope_image.data.iter())
        .any(|(a, b)| (a - b).abs() > 1e-5));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn qwen_rope_grid_uses_centered_scale_rope_coordinates() {
    // Golden convention pin: Qwen-Image-family RoPE uses scale_rope=True, i.e.
    // CENTERED height/width grid coordinates (diffusers hardcodes this in every
    // Qwen*Transformer). For an HxW grid, spatial axis positions run
    // [-(N-N/2) .. N/2-1], so the grid-center token sits at spatial position 0
    // (identity rotation), while corner tokens sit at non-zero positions.
    // A 0-based (scale_rope=False) implementation would put the center token at
    // position N/2 != 0 and fail this test -- that bug scrambles image attention
    // into high-frequency noise. axes=[4,4,4] over head_dim=12 -> freq_width=6:
    // band layout [frame(2), height(2), width(2)].
    let (frame, height, width) = (1usize, 4usize, 4usize);
    let rotary =
        qwen_rotary_embeddings_for_grid([4, 4, 4], 1000.0, 12, frame, height, width, 2).unwrap();
    let fw = 6usize;
    let (img_cos, img_sin) = (rotary.image.cos_data(), rotary.image.sin_data());
    let row = |t: usize| -> (&[f32], &[f32]) {
        (&img_cos[t * fw..t * fw + fw], &img_sin[t * fw..t * fw + fw])
    };
    // Grid center (y=2, x=2): centered h/w position = 2 - (4 - 2) = 0, frame = 0,
    // so every axis is at position 0 -> cos == 1, sin == 0 across all bands.
    let center = 2 * width + 2;
    let (ccos, csin) = row(center);
    for (i, (&c, &s)) in ccos.iter().zip(csin.iter()).enumerate() {
        assert!(
            (c - 1.0).abs() < 1e-6 && s.abs() < 1e-6,
            "center token freq {i}: cos={c} sin={s}, expected identity (position 0) \
             -- rope is not using centered scale_rope=True coordinates"
        );
    }
    // Corner (y=0, x=0): centered h/w position = -2 (non-zero) -> height & width
    // bands (freqs 2..6) must carry real rotation (some non-zero sin).
    let (_, corner_sin) = row(0);
    assert!(
        corner_sin[2..6].iter().any(|&s| s.abs() > 1e-6),
        "corner token has no spatial rotation; centered coordinates not applied"
    );
}

#[test]
fn native_transformer_attention_runs_krea_image_self_attention() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-krea-transformer-self-attn-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("krea-transformer-self-attn.hfq");
    let prefix = "transformer/tensors/transformer_blocks.0.attn";
    let identity2 = [1.0, 0.0, 0.0, 1.0];
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(&format!("{prefix}.to_q.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.to_k.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.to_v.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.to_out.0.weight"), &[2, 2], &identity2),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let attention = NativeTransformerAttentionProjection::from_hfq(
        &hfq,
        TransformerDenoiserFamily::Krea2,
        0,
        1,
    )
    .unwrap();
    let image_hidden = CpuTensor {
        shape: vec![1, 2, 2],
        data: vec![1.0, 0.0, 0.0, 1.0],
    };
    let expected =
        scaled_dot_product_attention(&image_hidden, &image_hidden, &image_hidden, 1).unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let (image_out, text_out) = attention
        .attend_image_text_with_runtime_context(
            &image_hidden,
            None,
            None,
            None,
            &mut runtime_context,
        )
        .unwrap();

    assert_f32_close(&image_out.data, &expected.data, 1e-6);
    assert!(text_out.is_none());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_feed_forward_runs_qwen_image_and_text_geglu() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-ff-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-ff.hfq");
    let block = "transformer/tensors/transformer_blocks.0";
    let image_proj = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let text_proj = [2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0];
    let down = [1.0, 0.0, 0.0, 1.0];
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(
                &format!("{block}.img_mlp.net.0.proj.weight"),
                &[4, 2],
                &image_proj,
            ),
            f32_mem_tensor(
                &format!("{block}.img_mlp.net.0.proj.bias"),
                &[4],
                &[0.0, 0.0, 0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{block}.img_mlp.net.2.weight"), &[2, 2], &down),
            f32_mem_tensor(&format!("{block}.img_mlp.net.2.bias"), &[2], &[0.5, -0.5]),
            f32_mem_tensor(
                &format!("{block}.txt_mlp.net.0.proj.weight"),
                &[4, 2],
                &text_proj,
            ),
            f32_mem_tensor(
                &format!("{block}.txt_mlp.net.0.proj.bias"),
                &[4],
                &[0.0, 0.0, 0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{block}.txt_mlp.net.2.weight"), &[2, 2], &down),
            f32_mem_tensor(&format!("{block}.txt_mlp.net.2.bias"), &[2], &[0.0, 0.25]),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let ff = NativeTransformerFeedForward::from_hfq(&hfq, TransformerDenoiserFamily::QwenImage, 0)
        .unwrap();
    let hidden = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![1.0, 2.0],
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let image = ff
        .forward_image_with_runtime_context(&hidden, &mut runtime_context)
        .unwrap();
    let text = ff
        .forward_text_with_runtime_context(&hidden, &mut runtime_context)
        .unwrap()
        .unwrap();

    assert_eq!(image.shape, vec![1, 1, 2]);
    assert_f32_close(&image.data, &[gelu(1.0) + 0.5, 2.0 * gelu(2.0) - 0.5], 1e-6);
    assert_f32_close(&text.data, &[2.0 * gelu(1.0), 4.0 * gelu(2.0) + 0.25], 1e-6);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_feed_forward_runs_krea_image_swiglu_without_text_stream() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-krea-transformer-ff-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("krea-transformer-ff.hfq");
    let prefix = "transformer/tensors/transformer_blocks.0.ff";
    let identity2 = [1.0, 0.0, 0.0, 1.0];
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(&format!("{prefix}.up.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.gate.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.down.weight"), &[2, 2], &identity2),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let ff =
        NativeTransformerFeedForward::from_hfq(&hfq, TransformerDenoiserFamily::Krea2, 0).unwrap();
    let hidden = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![1.0, 2.0],
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let image = ff
        .forward_image_with_runtime_context(&hidden, &mut runtime_context)
        .unwrap();
    let text = ff
        .forward_text_with_runtime_context(&hidden, &mut runtime_context)
        .unwrap();

    assert_eq!(image.shape, vec![1, 1, 2]);
    assert_f32_close(&image.data, &[silu(1.0), 2.0 * silu(2.0)], 1e-6);
    assert!(text.is_none());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_block_runs_qwen_attention_and_mlp_residuals() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-block-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-block.hfq");
    let block = "transformer/tensors/transformer_blocks.0";
    let attn = format!("{block}.attn");
    let identity2 = [1.0, 0.0, 0.0, 1.0];
    let geglu_identity = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let silu_one = silu(1.0);
    let mut modulation_weight = vec![0.0f32; 12 * 2];
    modulation_weight[10 * 2] = silu_one.recip();
    modulation_weight[11 * 2] = silu_one.recip();
    let mut tensors = vec![
        f32_mem_tensor(
            &format!("{block}.img_mod.1.weight"),
            &[12, 2],
            &modulation_weight,
        ),
        f32_mem_tensor(&format!("{block}.img_mod.1.bias"), &[12], &[0.0; 12]),
        f32_mem_tensor(
            &format!("{block}.txt_mod.1.weight"),
            &[12, 2],
            &modulation_weight,
        ),
        f32_mem_tensor(&format!("{block}.txt_mod.1.bias"), &[12], &[0.0; 12]),
        f32_mem_tensor(&format!("{attn}.to_q.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{attn}.to_k.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{attn}.to_v.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{attn}.to_out.0.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{attn}.add_q_proj.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{attn}.add_k_proj.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{attn}.add_v_proj.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{attn}.to_add_out.weight"), &[2, 2], &identity2),
        f32_mem_tensor(
            &format!("{block}.img_mlp.net.0.proj.weight"),
            &[4, 2],
            &geglu_identity,
        ),
        f32_mem_tensor(&format!("{block}.img_mlp.net.0.proj.bias"), &[4], &[0.0; 4]),
        f32_mem_tensor(
            &format!("{block}.img_mlp.net.2.weight"),
            &[2, 2],
            &identity2,
        ),
        f32_mem_tensor(&format!("{block}.img_mlp.net.2.bias"), &[2], &[0.0; 2]),
        f32_mem_tensor(
            &format!("{block}.txt_mlp.net.0.proj.weight"),
            &[4, 2],
            &geglu_identity,
        ),
        f32_mem_tensor(&format!("{block}.txt_mlp.net.0.proj.bias"), &[4], &[0.0; 4]),
        f32_mem_tensor(
            &format!("{block}.txt_mlp.net.2.weight"),
            &[2, 2],
            &identity2,
        ),
        f32_mem_tensor(&format!("{block}.txt_mlp.net.2.bias"), &[2], &[0.0; 2]),
    ];
    tensors.shrink_to_fit();
    write_hfqm_package_mem(&path, HFQ_ARCH_DIFFUSION, "{}", &tensors).unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let transformer_block =
        NativeTransformerBlock::from_hfq(&hfq, TransformerDenoiserFamily::QwenImage, 0, 1).unwrap();
    let image_hidden = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![1.0, -1.0],
    };
    let text_hidden = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![0.5, -0.5],
    };
    let timestep_embedding = CpuTensor {
        shape: vec![1, 2],
        data: vec![1.0, 0.0],
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let (image_out, text_out) = transformer_block
        .forward_qwen_with_runtime_context(
            &image_hidden,
            &text_hidden,
            None,
            &timestep_embedding,
            None,
            &mut runtime_context,
        )
        .unwrap();

    let expected_image = qwen_block_expected_mlp_only(&image_hidden);
    let expected_text = qwen_block_expected_mlp_only(&text_hidden);
    assert_f32_close(&image_out.data, &expected_image, 1e-5);
    assert_f32_close(&text_out.data, &expected_text, 1e-5);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_denoiser_runs_qwen_tiny_single_block_roundtrip() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-denoiser-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-denoiser.hfq");
    let tensors = qwen_tiny_transformer_denoiser_tensors();
    let weight_entries = tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    write_hfqm_package_mem(&path, HFQ_ARCH_DIFFUSION, "{}", &tensors).unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let mut config = tiny_runtime_config();
    config.latent_channels = 1;
    config.transformer = Some(TransformerDenoiserConfig {
        class_name: "QwenImageTransformer2DModel".into(),
        in_channels: Some(4),
        out_channels: Some(1),
        patch_size: Some(2),
        num_attention_heads: Some(1),
        ..TransformerDenoiserConfig::default()
    });
    let topology = transformer_denoiser_weight_topology(&DiffusionComponentMetadata {
        class_name: Some("QwenImageTransformer2DModel".to_string()),
        weight_entries,
        ..DiffusionComponentMetadata::default()
    });
    let denoiser = NativeTransformerDenoiser::from_hfq(&hfq, &config, &topology).unwrap();
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 2,
        width: 2,
        data: vec![1.0, -1.0, 0.5, -0.5],
    };
    let text_hidden = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![0.5, -0.5],
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let output = denoiser
        .forward_qwen_with_runtime_context(
            &latents,
            &[0.0],
            &text_hidden,
            None,
            &mut runtime_context,
        )
        .unwrap();

    assert_eq!(output.batch, 1);
    assert_eq!(output.channels, 1);
    assert_eq!(output.height, 2);
    assert_eq!(output.width, 2);
    let expected_hidden = qwen_block_expected_mlp_only(&CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![1.0, -1.0],
    });
    let expected = vec![
        expected_hidden[0],
        expected_hidden[1],
        expected_hidden[0] + expected_hidden[1],
        expected_hidden[0] - expected_hidden[1],
    ];
    assert_f32_close(&output.data, &expected, 1e-5);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_denoiser_rejects_qwen_guidance_embeds() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-guidance-embeds-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-guidance-embeds.hfq");
    let tensors = qwen_tiny_transformer_denoiser_tensors();
    let weight_entries = tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    write_hfqm_package_mem(&path, HFQ_ARCH_DIFFUSION, "{}", &tensors).unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let mut config = tiny_runtime_config();
    config.latent_channels = 1;
    config.transformer = Some(TransformerDenoiserConfig {
        class_name: "QwenImageTransformer2DModel".into(),
        in_channels: Some(4),
        out_channels: Some(1),
        patch_size: Some(2),
        num_attention_heads: Some(1),
        guidance_embeds: Some(true),
        ..TransformerDenoiserConfig::default()
    });
    let topology = transformer_denoiser_weight_topology(&DiffusionComponentMetadata {
        class_name: Some("QwenImageTransformer2DModel".to_string()),
        weight_entries,
        ..DiffusionComponentMetadata::default()
    });

    let err = NativeTransformerDenoiser::from_hfq(&hfq, &config, &topology).unwrap_err();
    assert!(err
        .to_string()
        .contains("guidance-distilled transformer embeddings are not implemented"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_denoiser_projects_qwen_text_and_output_norm() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-projection-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-projection.hfq");
    let mut tensors = qwen_tiny_transformer_denoiser_tensors();
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
    let weight_entries = tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    write_hfqm_package_mem(&path, HFQ_ARCH_DIFFUSION, "{}", &tensors).unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let mut config = tiny_runtime_config();
    config.latent_channels = 1;
    config.transformer = Some(TransformerDenoiserConfig {
        class_name: "QwenImageTransformer2DModel".into(),
        in_channels: Some(4),
        out_channels: Some(1),
        patch_size: Some(2),
        num_attention_heads: Some(1),
        attention_head_dim: Some(2),
        cross_attention_dim: Some(3),
        ..TransformerDenoiserConfig::default()
    });
    let topology = transformer_denoiser_weight_topology(&DiffusionComponentMetadata {
        class_name: Some("QwenImageTransformer2DModel".to_string()),
        weight_entries,
        ..DiffusionComponentMetadata::default()
    });
    let denoiser = NativeTransformerDenoiser::from_hfq(&hfq, &config, &topology).unwrap();
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 2,
        width: 2,
        data: vec![1.0, -1.0, 0.5, -0.5],
    };
    let text_hidden = CpuTensor {
        shape: vec![1, 1, 3],
        data: vec![0.5, -0.5, 2.0],
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());

    let output = denoiser
        .forward_qwen_with_runtime_context(
            &latents,
            &[1.0],
            &text_hidden,
            None,
            &mut runtime_context,
        )
        .unwrap();

    assert_eq!(output.batch, 1);
    assert_eq!(output.channels, 1);
    assert_eq!(output.height, 2);
    assert_eq!(output.width, 2);
    assert!(output.data.iter().all(|value| value.is_finite()));
    assert_ne!(output.data, vec![0.0; 4]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_denoiser_runs_qwen_cfg_scheduler_path() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-qwen-transformer-denoise-loop-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen-transformer-denoise-loop.hfq");
    let tensors = qwen_tiny_transformer_denoiser_tensors();
    let weight_entries = tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    write_hfqm_package_mem(&path, HFQ_ARCH_DIFFUSION, "{}", &tensors).unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let mut config = tiny_runtime_config();
    config.latent_channels = 1;
    config.transformer = Some(TransformerDenoiserConfig {
        class_name: "QwenImageTransformer2DModel".into(),
        in_channels: Some(4),
        out_channels: Some(1),
        patch_size: Some(2),
        num_attention_heads: Some(1),
        ..TransformerDenoiserConfig::default()
    });
    let topology = transformer_denoiser_weight_topology(&DiffusionComponentMetadata {
        class_name: Some("QwenImageTransformer2DModel".to_string()),
        weight_entries,
        ..DiffusionComponentMetadata::default()
    });
    let denoiser = NativeTransformerDenoiser::from_hfq(&hfq, &config, &topology).unwrap();
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 2,
        width: 2,
        data: vec![1.0, -1.0, 0.5, -0.5],
    };
    let positive = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![0.5, -0.5],
    };
    let negative = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![0.5, -0.5],
    };
    let schedule = DiffusionSchedule::linear(1).unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());
    let mut progress_calls = 0usize;

    let output = denoiser
        .denoise_latents_with_runtime_context(
            latents,
            &schedule,
            1.0,
            &positive,
            &negative,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut runtime_context,
            Some(&mut |_progress| {
                progress_calls += 1;
                Ok(())
            }),
        )
        .unwrap();

    assert_eq!(output.latents.batch, 1);
    assert_eq!(output.latents.channels, 1);
    assert_eq!(output.latents.height, 2);
    assert_eq!(output.latents.width, 2);
    assert!(output.latents.data.iter().all(|value| value.is_finite()));
    assert_eq!(
        output.runtime_kind,
        DiffusionRuntimeKind::CpuSourceReference
    );
    assert_eq!(progress_calls, 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_unet_forward_runs_synthetic_graph() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-native-unet-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("native-unet.hfq");
    let metadata = minimal_metadata();
    let identity1 = center_identity_conv(1);
    let down_prefix = "unet/tensors/down_blocks.0.resnets.0";
    let mid0_prefix = "unet/tensors/mid_block.resnets.0";
    let mid1_prefix = "unet/tensors/mid_block.resnets.1";
    let up_prefix = "unet/tensors/up_blocks.0.resnets.0";
    let tensors = [
        f32_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &identity1),
        f32_mem_tensor("unet/tensors/conv_in.bias", &[1], &[0.0]),
        f32_mem_tensor(
            "unet/tensors/time_embedding.linear_1.weight",
            &[2, 2],
            &[1.0, 0.0, 0.0, 1.0],
        ),
        f32_mem_tensor(
            "unet/tensors/time_embedding.linear_1.bias",
            &[2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(
            "unet/tensors/time_embedding.linear_2.weight",
            &[2, 2],
            &[1.0, 0.0, 0.0, 1.0],
        ),
        f32_mem_tensor(
            "unet/tensors/time_embedding.linear_2.bias",
            &[2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(&format!("{down_prefix}.norm1.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{down_prefix}.norm1.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{down_prefix}.conv1.weight"),
            &[1, 1, 3, 3],
            &identity1,
        ),
        f32_mem_tensor(&format!("{down_prefix}.conv1.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{down_prefix}.time_emb_proj.weight"),
            &[1, 2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(&format!("{down_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{down_prefix}.norm2.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{down_prefix}.norm2.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{down_prefix}.conv2.weight"),
            &[1, 1, 3, 3],
            &[0.0; 9],
        ),
        f32_mem_tensor(&format!("{down_prefix}.conv2.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{mid0_prefix}.norm1.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{mid0_prefix}.norm1.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{mid0_prefix}.conv1.weight"),
            &[1, 1, 3, 3],
            &identity1,
        ),
        f32_mem_tensor(&format!("{mid0_prefix}.conv1.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{mid0_prefix}.time_emb_proj.weight"),
            &[1, 2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(&format!("{mid0_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{mid0_prefix}.norm2.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{mid0_prefix}.norm2.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{mid0_prefix}.conv2.weight"),
            &[1, 1, 3, 3],
            &[0.0; 9],
        ),
        f32_mem_tensor(&format!("{mid0_prefix}.conv2.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{mid1_prefix}.norm1.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{mid1_prefix}.norm1.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{mid1_prefix}.conv1.weight"),
            &[1, 1, 3, 3],
            &identity1,
        ),
        f32_mem_tensor(&format!("{mid1_prefix}.conv1.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{mid1_prefix}.time_emb_proj.weight"),
            &[1, 2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(&format!("{mid1_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{mid1_prefix}.norm2.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{mid1_prefix}.norm2.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{mid1_prefix}.conv2.weight"),
            &[1, 1, 3, 3],
            &[0.0; 9],
        ),
        f32_mem_tensor(&format!("{mid1_prefix}.conv2.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{up_prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{up_prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{up_prefix}.conv1.weight"),
            &[1, 2, 3, 3],
            &[0.0; 18],
        ),
        f32_mem_tensor(&format!("{up_prefix}.conv1.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{up_prefix}.time_emb_proj.weight"),
            &[1, 2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(&format!("{up_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{up_prefix}.norm2.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{up_prefix}.norm2.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{up_prefix}.conv2.weight"),
            &[1, 1, 3, 3],
            &[0.0; 9],
        ),
        f32_mem_tensor(&format!("{up_prefix}.conv2.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{up_prefix}.conv_shortcut.weight"),
            &[1, 2, 1, 1],
            &[1.0, 0.0],
        ),
        f32_mem_tensor(&format!("{up_prefix}.conv_shortcut.bias"), &[1], &[0.0]),
        f32_mem_tensor("unet/tensors/conv_norm_out.weight", &[1], &[1.0]),
        f32_mem_tensor("unet/tensors/conv_norm_out.bias", &[1], &[0.0]),
        f32_mem_tensor("unet/tensors/conv_out.weight", &[1, 1, 3, 3], &identity1),
        f32_mem_tensor("unet/tensors/conv_out.bias", &[1], &[0.0]),
    ];
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
    let config = UnetConfig {
        class_name: "UNet2DConditionModel".into(),
        sample_size: Some(2),
        in_channels: Some(1),
        out_channels: Some(1),
        cross_attention_dim: Some(1),
        attention_head_dim: vec![1],
        block_out_channels: vec![1],
        down_block_types: vec!["DownBlock2D".into()],
        up_block_types: vec!["UpBlock2D".into()],
        layers_per_block: Some(1),
        norm_num_groups: Some(1),
        norm_eps: Some(1e-5),
        center_input_sample: false,
        flip_sin_to_cos: true,
        freq_shift: 0.0,
        addition_embed_type: None,
        addition_time_embed_dim: None,
        projection_class_embeddings_input_dim: None,
    };
    let unet = NativeUnet2DConditionModel::from_hfq(&hfq, &config).unwrap();
    assert!(unet.mid_block.is_some());
    let sample = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let encoder = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![0.0],
    };
    let output = unet.forward(&sample, &[0.0], &encoder).unwrap();
    assert_eq!(output.shape, vec![1, 1, 2, 2]);
    assert!(output.data.iter().all(|value| value.is_finite()));

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for UNet forward routing test: {error}");
        } else {
            let hip = unet
                .forward_with_runtime_options(
                    &sample,
                    &[0.0],
                    &encoder,
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                )
                .unwrap();
            assert_eq!(hip.shape, output.shape);
            // F16 WMMA-GEMM conv (Phase 3) → F16 tolerance.
            assert!(f32_slices_close(&hip.data, &output.data, 5e-3));
        }
    }

    let bad_encoder = CpuTensor {
        shape: vec![2, 1, 1],
        data: vec![0.0, 0.0],
    };
    assert!(unet.forward(&sample, &[0.0], &bad_encoder).is_err());
    let _ = fs::remove_dir_all(&dir);
}

/// Phase 1b: validate the device-resident UNet forward against the CPU
/// reference on a 2-channel UNet whose mid block carries a cross-attention
/// `Transformer2DModel`. This exercises the resident transformer path the
/// `native_unet_forward_runs_synthetic_graph` test does not reach:
/// `proj_in` → `nchw_to_bsc` → layer-norm → self-attn → cross-attn → GeGLU
/// (`geglu_gate`) → `bsc_to_nchw` → `proj_out`, plus the up-path channel
/// concat with two resnets consuming two skips.
#[test]
fn native_unet_resident_path_matches_cpu_reference_with_cross_attention() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-resident-unet-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("resident-unet.hfq");
    let metadata = minimal_metadata();

    const CH: usize = 2;
    const TIME_DIM: usize = 2;
    const CROSS: usize = 2;
    const INNER: usize = 2;

    // [out, in, 3, 3] center-tap (near-identity) conv.
    let conv3 = |out_ch: usize, in_ch: usize| -> Vec<f32> {
        let mut d = vec![0.0f32; out_ch * in_ch * 9];
        for c in 0..out_ch.min(in_ch) {
            d[((c * in_ch + c) * 3 + 1) * 3 + 1] = 1.0;
        }
        d
    };
    // [out, in, 1, 1] near-identity 1x1 conv.
    let conv1 = |out_ch: usize, in_ch: usize| -> Vec<f32> {
        let mut d = vec![0.0f32; out_ch * in_ch];
        for c in 0..out_ch.min(in_ch) {
            d[c * in_ch + c] = 1.0;
        }
        d
    };
    // Deterministic small finite [r, c] matrix for linear projections.
    let mat = |r: usize, c: usize| -> Vec<f32> {
        (0..r * c).map(|k| 0.1 * ((k as f32 % 5.0) - 2.0)).collect()
    };

    // Capture-free tensor builder (so the resnet helper below can take
    // `&mut Vec` and avoid a closure-capture borrow conflict).
    let mk = |name: String, shape: Vec<u32>, data: Vec<f32>| -> HfqMemTensor {
        f32_mem_tensor(&name, &shape, &data)
    };

    // A UnetResnetBlock2D with zeroed time projection (still exercises the
    // resident linear + add_channel_bias) and an optional shortcut.
    let resnet =
        |v: &mut Vec<HfqMemTensor>, prefix: &str, in_ch: usize, out_ch: usize, shortcut: bool| {
            v.push(mk(
                format!("{prefix}.norm1.weight"),
                vec![in_ch as u32],
                vec![1.0; in_ch],
            ));
            v.push(mk(
                format!("{prefix}.norm1.bias"),
                vec![in_ch as u32],
                vec![0.0; in_ch],
            ));
            v.push(mk(
                format!("{prefix}.conv1.weight"),
                vec![out_ch as u32, in_ch as u32, 3, 3],
                conv3(out_ch, in_ch),
            ));
            v.push(mk(
                format!("{prefix}.conv1.bias"),
                vec![out_ch as u32],
                vec![0.0; out_ch],
            ));
            v.push(mk(
                format!("{prefix}.time_emb_proj.weight"),
                vec![out_ch as u32, TIME_DIM as u32],
                vec![0.0; out_ch * TIME_DIM],
            ));
            v.push(mk(
                format!("{prefix}.time_emb_proj.bias"),
                vec![out_ch as u32],
                vec![0.0; out_ch],
            ));
            v.push(mk(
                format!("{prefix}.norm2.weight"),
                vec![out_ch as u32],
                vec![1.0; out_ch],
            ));
            v.push(mk(
                format!("{prefix}.norm2.bias"),
                vec![out_ch as u32],
                vec![0.0; out_ch],
            ));
            v.push(mk(
                format!("{prefix}.conv2.weight"),
                vec![out_ch as u32, out_ch as u32, 3, 3],
                conv3(out_ch, out_ch),
            ));
            v.push(mk(
                format!("{prefix}.conv2.bias"),
                vec![out_ch as u32],
                vec![0.0; out_ch],
            ));
            if shortcut {
                v.push(mk(
                    format!("{prefix}.conv_shortcut.weight"),
                    vec![out_ch as u32, in_ch as u32, 1, 1],
                    conv1(out_ch, in_ch),
                ));
                v.push(mk(
                    format!("{prefix}.conv_shortcut.bias"),
                    vec![out_ch as u32],
                    vec![0.0; out_ch],
                ));
            }
        };

    let mut tensors: Vec<HfqMemTensor> = Vec::new();
    // conv_in.
    tensors.push(mk(
        "unet/tensors/conv_in.weight".into(),
        vec![CH as u32, CH as u32, 3, 3],
        conv3(CH, CH),
    ));
    tensors.push(mk(
        "unet/tensors/conv_in.bias".into(),
        vec![CH as u32],
        vec![0.0; CH],
    ));
    // time embedding (dim = TIME_DIM).
    tensors.push(mk(
        "unet/tensors/time_embedding.linear_1.weight".into(),
        vec![TIME_DIM as u32, TIME_DIM as u32],
        conv1(TIME_DIM, TIME_DIM),
    ));
    tensors.push(mk(
        "unet/tensors/time_embedding.linear_1.bias".into(),
        vec![TIME_DIM as u32],
        vec![0.0; TIME_DIM],
    ));
    tensors.push(mk(
        "unet/tensors/time_embedding.linear_2.weight".into(),
        vec![TIME_DIM as u32, TIME_DIM as u32],
        conv1(TIME_DIM, TIME_DIM),
    ));
    tensors.push(mk(
        "unet/tensors/time_embedding.linear_2.bias".into(),
        vec![TIME_DIM as u32],
        vec![0.0; TIME_DIM],
    ));

    // Down block 0: one resnet, no attention, no downsampler.
    resnet(
        &mut tensors,
        "unet/tensors/down_blocks.0.resnets.0",
        CH,
        CH,
        false,
    );

    // Mid block: resnet0 + cross-attention transformer + resnet1.
    resnet(
        &mut tensors,
        "unet/tensors/mid_block.resnets.0",
        CH,
        CH,
        false,
    );
    let attn = "unet/tensors/mid_block.attentions.0";
    tensors.push(mk(
        format!("{attn}.norm.weight"),
        vec![CH as u32],
        vec![1.0; CH],
    ));
    tensors.push(mk(
        format!("{attn}.norm.bias"),
        vec![CH as u32],
        vec![0.0; CH],
    ));
    tensors.push(mk(
        format!("{attn}.proj_in.weight"),
        vec![CH as u32, CH as u32, 1, 1],
        conv1(CH, CH),
    ));
    tensors.push(mk(
        format!("{attn}.proj_in.bias"),
        vec![CH as u32],
        vec![0.0; CH],
    ));
    let tb = format!("{attn}.transformer_blocks.0");
    tensors.push(mk(
        format!("{tb}.norm1.weight"),
        vec![CH as u32],
        vec![1.0; CH],
    ));
    tensors.push(mk(
        format!("{tb}.norm1.bias"),
        vec![CH as u32],
        vec![0.0; CH],
    ));
    tensors.push(mk(
        format!("{tb}.attn1.to_q.weight"),
        vec![CH as u32, CH as u32],
        mat(CH, CH),
    ));
    tensors.push(mk(
        format!("{tb}.attn1.to_k.weight"),
        vec![CH as u32, CH as u32],
        mat(CH, CH),
    ));
    tensors.push(mk(
        format!("{tb}.attn1.to_v.weight"),
        vec![CH as u32, CH as u32],
        mat(CH, CH),
    ));
    tensors.push(mk(
        format!("{tb}.attn1.to_out.0.weight"),
        vec![CH as u32, CH as u32],
        mat(CH, CH),
    ));
    tensors.push(mk(
        format!("{tb}.attn1.to_out.0.bias"),
        vec![CH as u32],
        vec![0.0; CH],
    ));
    tensors.push(mk(
        format!("{tb}.norm2.weight"),
        vec![CH as u32],
        vec![1.0; CH],
    ));
    tensors.push(mk(
        format!("{tb}.norm2.bias"),
        vec![CH as u32],
        vec![0.0; CH],
    ));
    tensors.push(mk(
        format!("{tb}.attn2.to_q.weight"),
        vec![CH as u32, CH as u32],
        mat(CH, CH),
    ));
    tensors.push(mk(
        format!("{tb}.attn2.to_k.weight"),
        vec![CH as u32, CROSS as u32],
        mat(CH, CROSS),
    ));
    tensors.push(mk(
        format!("{tb}.attn2.to_v.weight"),
        vec![CH as u32, CROSS as u32],
        mat(CH, CROSS),
    ));
    tensors.push(mk(
        format!("{tb}.attn2.to_out.0.weight"),
        vec![CH as u32, CH as u32],
        mat(CH, CH),
    ));
    tensors.push(mk(
        format!("{tb}.attn2.to_out.0.bias"),
        vec![CH as u32],
        vec![0.0; CH],
    ));
    tensors.push(mk(
        format!("{tb}.norm3.weight"),
        vec![CH as u32],
        vec![1.0; CH],
    ));
    tensors.push(mk(
        format!("{tb}.norm3.bias"),
        vec![CH as u32],
        vec![0.0; CH],
    ));
    tensors.push(mk(
        format!("{tb}.ff.net.0.proj.weight"),
        vec![(2 * INNER) as u32, CH as u32],
        mat(2 * INNER, CH),
    ));
    tensors.push(mk(
        format!("{tb}.ff.net.0.proj.bias"),
        vec![(2 * INNER) as u32],
        vec![0.0; 2 * INNER],
    ));
    tensors.push(mk(
        format!("{tb}.ff.net.2.weight"),
        vec![CH as u32, INNER as u32],
        mat(CH, INNER),
    ));
    tensors.push(mk(
        format!("{tb}.ff.net.2.bias"),
        vec![CH as u32],
        vec![0.0; CH],
    ));
    tensors.push(mk(
        format!("{attn}.proj_out.weight"),
        vec![CH as u32, CH as u32, 1, 1],
        conv1(CH, CH),
    ));
    tensors.push(mk(
        format!("{attn}.proj_out.bias"),
        vec![CH as u32],
        vec![0.0; CH],
    ));
    resnet(
        &mut tensors,
        "unet/tensors/mid_block.resnets.1",
        CH,
        CH,
        false,
    );

    // Up block 0: two resnets (consume the two skips), each concatenating a
    // skip onto the channel axis (in = 2*CH), with a shortcut. No upsampler.
    resnet(
        &mut tensors,
        "unet/tensors/up_blocks.0.resnets.0",
        2 * CH,
        CH,
        true,
    );
    resnet(
        &mut tensors,
        "unet/tensors/up_blocks.0.resnets.1",
        2 * CH,
        CH,
        true,
    );

    tensors.push(mk(
        "unet/tensors/conv_norm_out.weight".into(),
        vec![CH as u32],
        vec![1.0; CH],
    ));
    tensors.push(mk(
        "unet/tensors/conv_norm_out.bias".into(),
        vec![CH as u32],
        vec![0.0; CH],
    ));
    tensors.push(mk(
        "unet/tensors/conv_out.weight".into(),
        vec![CH as u32, CH as u32, 3, 3],
        conv3(CH, CH),
    ));
    tensors.push(mk(
        "unet/tensors/conv_out.bias".into(),
        vec![CH as u32],
        vec![0.0; CH],
    ));

    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
    let config = UnetConfig {
        class_name: "UNet2DConditionModel".into(),
        sample_size: Some(2),
        in_channels: Some(CH),
        out_channels: Some(CH),
        cross_attention_dim: Some(CROSS),
        attention_head_dim: vec![CH],
        block_out_channels: vec![CH],
        down_block_types: vec!["DownBlock2D".into()],
        up_block_types: vec!["UpBlock2D".into()],
        layers_per_block: Some(1),
        norm_num_groups: Some(1),
        norm_eps: Some(1e-5),
        center_input_sample: false,
        flip_sin_to_cos: true,
        freq_shift: 0.0,
        addition_embed_type: None,
        addition_time_embed_dim: None,
        projection_class_embeddings_input_dim: None,
    };
    let unet = NativeUnet2DConditionModel::from_hfq(&hfq, &config).unwrap();
    assert!(unet.mid_block.is_some(), "mid block not loaded");
    assert!(
        unet.mid_block.as_ref().unwrap().attention.is_some(),
        "mid-block cross-attention not loaded"
    );

    let sample = CpuTensor {
        shape: vec![1, CH, 2, 2],
        data: vec![1.0, -2.0, 0.5, 3.0, -0.25, 1.5, 2.0, -1.0],
    };
    let encoder = CpuTensor {
        shape: vec![1, 2, CROSS],
        data: vec![0.3, -0.6, 0.9, 0.1],
    };
    let cpu_output = unet.forward(&sample, &[0.5], &encoder).unwrap();
    assert!(cpu_output.data.iter().all(|value| value.is_finite()));

    if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
        eprintln!("skip: ROCm GPU unavailable for resident UNet cross-attention test: {error}");
    } else {
        let resident = unet
            .forward_with_runtime_options(
                &sample,
                &[0.5],
                &encoder,
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            )
            .unwrap();
        assert_eq!(resident.shape, cpu_output.shape);
        // F16 WMMA-GEMM conv (Phase 3) → F16 tolerance, not 1e-4.
        assert!(
            f32_slices_close(&resident.data, &cpu_output.data, 5e-3),
            "resident UNet {:?} != cpu reference {:?}",
            resident.data,
            cpu_output.data
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_vae_decoder_decodes_synthetic_latents_to_rgb8() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-native-vae-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("native-vae.hfq");
    let metadata = minimal_metadata();
    let identity1 = center_identity_conv(1);
    let resnet_prefix = "vae/tensors/decoder.up_blocks.0.resnets.0";
    let tensors = [
        f32_mem_tensor("vae/tensors/post_quant_conv.weight", &[1, 1, 1, 1], &[1.0]),
        f32_mem_tensor("vae/tensors/post_quant_conv.bias", &[1], &[0.0]),
        f32_mem_tensor(
            "vae/tensors/decoder.conv_in.weight",
            &[1, 1, 3, 3],
            &identity1,
        ),
        f32_mem_tensor("vae/tensors/decoder.conv_in.bias", &[1], &[0.0]),
        f32_mem_tensor(&format!("{resnet_prefix}.norm1.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{resnet_prefix}.norm1.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{resnet_prefix}.conv1.weight"),
            &[1, 1, 3, 3],
            &[0.0; 9],
        ),
        f32_mem_tensor(&format!("{resnet_prefix}.conv1.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{resnet_prefix}.norm2.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{resnet_prefix}.norm2.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{resnet_prefix}.conv2.weight"),
            &[1, 1, 3, 3],
            &[0.0; 9],
        ),
        f32_mem_tensor(&format!("{resnet_prefix}.conv2.bias"), &[1], &[0.0]),
        f32_mem_tensor("vae/tensors/decoder.conv_norm_out.weight", &[1], &[1.0]),
        f32_mem_tensor("vae/tensors/decoder.conv_norm_out.bias", &[1], &[0.0]),
        f32_mem_tensor(
            "vae/tensors/decoder.conv_out.weight",
            &[3, 1, 3, 3],
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        f32_mem_tensor("vae/tensors/decoder.conv_out.bias", &[3], &[0.0, 0.0, 0.0]),
    ];
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
    let config = VaeConfig {
        class_name: "AutoencoderKL".into(),
        latent_channels: Some(1),
        z_dim: None,
        scaling_factor: Some(1.0),
        shift_factor: None,
        latents_mean: Vec::new(),
        latents_std: Vec::new(),
        block_out_channels: vec![1],
        down_block_types: Vec::new(),
        up_block_types: vec!["UpDecoderBlock2D".into()],
        norm_num_groups: Some(1),
        norm_eps: Some(1e-6),
    };
    let decoder = NativeVaeDecoder::from_hfq(&hfq, &config).unwrap();
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 2,
        width: 2,
        data: vec![0.0, 0.5, -0.5, 1.0],
    };
    let decoded = decoder.decode_latents(&latents).unwrap();
    assert_eq!(decoded.shape, vec![1, 3, 2, 2]);
    assert!(decoded.data.iter().all(|value| value.is_finite()));
    let image = decoder.decode_to_rgb8(&latents).unwrap();
    assert_eq!(image.batch, 1);
    assert_eq!(image.width, 2);
    assert_eq!(image.height, 2);
    assert_eq!(image.data.len(), 12);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for VAE decoder routing test: {error}");
        } else {
            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let hip_context_decoded = decoder
                .decode_latents_with_runtime_context(&latents, &mut runtime_context)
                .unwrap();
            assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
            assert_eq!(hip_context_decoded.shape, decoded.shape);
            // F16 WMMA-GEMM conv (Phase 3): match the F32 reference to F16
            // tolerance, not 1e-5.
            assert!(f32_slices_close(
                &hip_context_decoded.data,
                &decoded.data,
                5e-3
            ));
            let hip_decoded = decoder
                .decode_latents_with_runtime_options(
                    &latents,
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                )
                .unwrap();
            assert_eq!(hip_decoded.shape, decoded.shape);
            assert!(f32_slices_close(&hip_decoded.data, &decoded.data, 5e-3));
            let (hip_image, runtime_kind) = decode_to_rgb8_with_runtime_options(
                &decoder,
                &latents,
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            )
            .unwrap();
            assert_eq!(runtime_kind, DiffusionRuntimeKind::RocmHybridReference);
            // The F16 conv may shift a u8 pixel by ±1 vs the F32 reference.
            assert_eq!(hip_image.batch, image.batch);
            assert_eq!(hip_image.width, image.width);
            assert_eq!(hip_image.height, image.height);
            assert_eq!(hip_image.data.len(), image.data.len());
            for (h, c) in hip_image.data.iter().zip(image.data.iter()) {
                assert!((*h as i16 - *c as i16).abs() <= 2, "rgb8 pixel {h} vs {c}");
            }
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

/// Phase 1b: exercise the full device-resident VAE decode path against the
/// CPU reference on a decoder that hits every resident op — including the
/// ones the basic decoder test above does not: the mid-block self-attention
/// (`nchw_to_bsc` → linear q/k/v → SDPA → out-proj → `bsc_to_nchw`), a resnet
/// `conv_shortcut`, and an up-block nearest-neighbour upsampler.
#[test]
fn native_vae_decoder_resident_path_matches_cpu_reference() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-resident-vae-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("resident-vae.hfq");
    let metadata = minimal_metadata();

    // 2-channel decoder so group-norm (1 group over 2 channels) and the
    // attention projections are non-trivial.
    let conv33 = center_identity_conv(2); // [2,2,3,3] identity
    let conv11 = vec![1.0, 0.0, 0.0, 1.0]; // [2,2,1,1] identity
                                           // conv_out maps 2 -> 3 channels; each output channel reads input
                                           // channel (o % 2) center tap. [3,2,3,3].
    let mut conv_out = vec![0.0f32; 3 * 2 * 3 * 3];
    for o in 0..3usize {
        let i = o % 2;
        conv_out[(((o * 2 + i) * 3 + 1) * 3) + 1] = 1.0;
    }
    // Distinct, finite attention projections (not identity) so q/k/v/out are
    // all genuinely exercised; correctness only needs CPU and GPU to agree.
    let proj_q = vec![0.5, 0.1, -0.2, 0.7];
    let proj_k = vec![0.3, -0.4, 0.6, 0.2];
    let proj_v = vec![0.8, 0.05, 0.15, -0.3];
    let proj_out = vec![0.4, 0.2, -0.1, 0.9];

    let mid_r0 = "vae/tensors/decoder.mid_block.resnets.0";
    let mid_attn = "vae/tensors/decoder.mid_block.attentions.0";
    let mid_r1 = "vae/tensors/decoder.mid_block.resnets.1";
    let up_r0 = "vae/tensors/decoder.up_blocks.0.resnets.0";

    let tensors = vec![
        f32_mem_tensor("vae/tensors/post_quant_conv.weight", &[2, 2, 1, 1], &conv11),
        f32_mem_tensor("vae/tensors/post_quant_conv.bias", &[2], &[0.0, 0.0]),
        f32_mem_tensor("vae/tensors/decoder.conv_in.weight", &[2, 2, 3, 3], &conv33),
        f32_mem_tensor("vae/tensors/decoder.conv_in.bias", &[2], &[0.0, 0.0]),
        // mid resnet 0 — WITH a conv_shortcut to exercise the shortcut path.
        f32_mem_tensor(&format!("{mid_r0}.norm1.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{mid_r0}.norm1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{mid_r0}.conv1.weight"), &[2, 2, 3, 3], &conv33),
        f32_mem_tensor(&format!("{mid_r0}.conv1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{mid_r0}.norm2.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{mid_r0}.norm2.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{mid_r0}.conv2.weight"), &[2, 2, 3, 3], &conv33),
        f32_mem_tensor(&format!("{mid_r0}.conv2.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{mid_r0}.conv_shortcut.weight"),
            &[2, 2, 1, 1],
            &conv11,
        ),
        f32_mem_tensor(&format!("{mid_r0}.conv_shortcut.bias"), &[2], &[0.0, 0.0]),
        // mid attention.
        f32_mem_tensor(&format!("{mid_attn}.group_norm.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{mid_attn}.group_norm.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{mid_attn}.to_q.weight"), &[2, 2], &proj_q),
        f32_mem_tensor(&format!("{mid_attn}.to_k.weight"), &[2, 2], &proj_k),
        f32_mem_tensor(&format!("{mid_attn}.to_v.weight"), &[2, 2], &proj_v),
        f32_mem_tensor(&format!("{mid_attn}.to_out.0.weight"), &[2, 2], &proj_out),
        f32_mem_tensor(&format!("{mid_attn}.to_out.0.bias"), &[2], &[0.0, 0.0]),
        // mid resnet 1 — no shortcut.
        f32_mem_tensor(&format!("{mid_r1}.norm1.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{mid_r1}.norm1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{mid_r1}.conv1.weight"), &[2, 2, 3, 3], &conv33),
        f32_mem_tensor(&format!("{mid_r1}.conv1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{mid_r1}.norm2.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{mid_r1}.norm2.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{mid_r1}.conv2.weight"), &[2, 2, 3, 3], &conv33),
        f32_mem_tensor(&format!("{mid_r1}.conv2.bias"), &[2], &[0.0, 0.0]),
        // up block 0 — one resnet plus an upsampler conv.
        f32_mem_tensor(&format!("{up_r0}.norm1.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{up_r0}.norm1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{up_r0}.conv1.weight"), &[2, 2, 3, 3], &conv33),
        f32_mem_tensor(&format!("{up_r0}.conv1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{up_r0}.norm2.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{up_r0}.norm2.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{up_r0}.conv2.weight"), &[2, 2, 3, 3], &conv33),
        f32_mem_tensor(&format!("{up_r0}.conv2.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            "vae/tensors/decoder.up_blocks.0.upsamplers.0.conv.weight",
            &[2, 2, 3, 3],
            &conv33,
        ),
        f32_mem_tensor(
            "vae/tensors/decoder.up_blocks.0.upsamplers.0.conv.bias",
            &[2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(
            "vae/tensors/decoder.conv_norm_out.weight",
            &[2],
            &[1.0, 1.0],
        ),
        f32_mem_tensor("vae/tensors/decoder.conv_norm_out.bias", &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            "vae/tensors/decoder.conv_out.weight",
            &[3, 2, 3, 3],
            &conv_out,
        ),
        f32_mem_tensor("vae/tensors/decoder.conv_out.bias", &[3], &[0.0, 0.0, 0.0]),
    ];
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
    let config = VaeConfig {
        class_name: "AutoencoderKL".into(),
        latent_channels: Some(2),
        z_dim: None,
        scaling_factor: Some(1.0),
        shift_factor: None,
        latents_mean: Vec::new(),
        latents_std: Vec::new(),
        block_out_channels: vec![2],
        down_block_types: Vec::new(),
        up_block_types: vec!["UpDecoderBlock2D".into()],
        norm_num_groups: Some(1),
        norm_eps: Some(1e-6),
    };
    let decoder = NativeVaeDecoder::from_hfq(&hfq, &config).unwrap();
    // Confirm the fixture actually built the optional blocks we mean to test.
    assert!(decoder.mid_attention.is_some(), "mid attention not loaded");
    assert!(decoder.mid_resnet_0.is_some(), "mid resnet 0 not loaded");
    assert!(
        decoder.up_blocks[0].upsampler.is_some(),
        "up-block upsampler not loaded"
    );

    let latents = LatentBatch {
        batch: 1,
        channels: 2,
        height: 2,
        width: 2,
        data: vec![0.0, 0.5, -0.5, 1.0, 0.25, -0.75, 0.9, -0.1],
    };
    let cpu_decoded = decoder.decode_latents(&latents).unwrap();
    assert!(cpu_decoded.data.iter().all(|value| value.is_finite()));

    if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
        eprintln!("skip: ROCm GPU unavailable for resident VAE decode test: {error}");
    } else {
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(
            DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
        );
        let resident_decoded = decoder
            .decode_latents_with_runtime_context(&latents, &mut runtime_context)
            .unwrap();
        assert_eq!(resident_decoded.shape, cpu_decoded.shape);
        // F16 WMMA-GEMM conv (Phase 3) → F16 tolerance, not 1e-4.
        assert!(
            f32_slices_close(&resident_decoded.data, &cpu_decoded.data, 5e-3),
            "resident decode {:?} != cpu reference {:?}",
            resident_decoded.data,
            cpu_decoded.data
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_vae_encoder_encodes_synthetic_image_to_latents() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-native-vae-encoder-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("native-vae-encoder.hfq");
    let metadata = minimal_metadata();
    let prefix = "vae/tensors/encoder.down_blocks.0.resnets.0";
    let identity1 = center_identity_conv(1);
    let mut conv_in = vec![0.0; 1 * 3 * 3 * 3];
    conv_in[1 * 3 + 1] = 1.0;
    let mut conv_out = vec![0.0; 2 * 1 * 3 * 3];
    conv_out[1 * 3 + 1] = 1.0;
    let tensors = vec![
        f32_mem_tensor(
            "vae/tensors/encoder.conv_in.weight",
            &[1, 3, 3, 3],
            &conv_in,
        ),
        f32_mem_tensor("vae/tensors/encoder.conv_in.bias", &[1], &[0.0]),
        f32_mem_tensor(&format!("{prefix}.norm1.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{prefix}.norm1.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{prefix}.conv1.weight"), &[1, 1, 3, 3], &identity1),
        f32_mem_tensor(&format!("{prefix}.conv1.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{prefix}.norm2.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{prefix}.norm2.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{prefix}.conv2.weight"), &[1, 1, 3, 3], &[0.0; 9]),
        f32_mem_tensor(&format!("{prefix}.conv2.bias"), &[1], &[0.0]),
        f32_mem_tensor("vae/tensors/encoder.conv_norm_out.weight", &[1], &[1.0]),
        f32_mem_tensor("vae/tensors/encoder.conv_norm_out.bias", &[1], &[0.0]),
        f32_mem_tensor(
            "vae/tensors/encoder.conv_out.weight",
            &[2, 1, 3, 3],
            &conv_out,
        ),
        f32_mem_tensor("vae/tensors/encoder.conv_out.bias", &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            "vae/tensors/quant_conv.weight",
            &[2, 2, 1, 1],
            &[1.0, 0.0, 0.0, 1.0],
        ),
        f32_mem_tensor("vae/tensors/quant_conv.bias", &[2], &[0.0, 0.0]),
    ];
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
    let config = VaeConfig {
        class_name: "AutoencoderKL".into(),
        latent_channels: Some(1),
        z_dim: None,
        scaling_factor: Some(0.5),
        shift_factor: None,
        latents_mean: Vec::new(),
        latents_std: Vec::new(),
        block_out_channels: vec![1],
        down_block_types: vec!["DownEncoderBlock2D".into()],
        up_block_types: Vec::new(),
        norm_num_groups: Some(1),
        norm_eps: Some(1e-6),
    };
    let encoder = NativeVaeEncoder::from_hfq(&hfq, &config).unwrap();
    let image = RgbImageBatch {
        batch: 1,
        width: 2,
        height: 2,
        data: vec![255; 12],
    };

    let latents = encoder.encode_to_latents(&image).unwrap();

    assert_eq!(latents.batch, 1);
    assert_eq!(latents.channels, 1);
    assert_eq!(latents.height, 2);
    assert_eq!(latents.width, 2);
    assert!(latents.data.iter().all(|value| value.is_finite()));

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for VAE encoder routing test: {error}");
        } else {
            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let hip_context_latents = encoder
                .encode_to_latents_with_runtime_context(&image, &mut runtime_context)
                .unwrap();
            assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
            assert_eq!(hip_context_latents.batch, latents.batch);
            assert_eq!(hip_context_latents.channels, latents.channels);
            assert_eq!(hip_context_latents.height, latents.height);
            assert_eq!(hip_context_latents.width, latents.width);
            assert!(f32_slices_close(
                &hip_context_latents.data,
                &latents.data,
                1e-5
            ));
            let hip_latents = encoder
                .encode_to_latents_with_runtime_options(
                    &image,
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                )
                .unwrap();
            assert_eq!(hip_latents.batch, latents.batch);
            assert_eq!(hip_latents.channels, latents.channels);
            assert_eq!(hip_latents.height, latents.height);
            assert_eq!(hip_latents.width, latents.width);
            assert!(f32_slices_close(&hip_latents.data, &latents.data, 1e-5));
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_transformer_denoiser_runs_krea_tiny_single_block() {
    // Full Krea2 single-stream denoiser assembly + forward on a tiny fixture:
    // img_in -> txt_in -> concat[text;image] -> block(forward_krea) -> split
    // image -> final adaLN -> latents. Zero-gate blocks keep it stable; we
    // assert the round-trip shape and that every output is finite.
    let dir = std::env::temp_dir().join(format!(
        "hipfire-krea-transformer-denoiser-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("krea-transformer-denoiser.hfq");
    let tensors = krea_tiny_transformer_denoiser_tensors();
    let weight_entries = tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>();
    write_hfqm_package_mem(&path, HFQ_ARCH_DIFFUSION, "{}", &tensors).unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let mut config = tiny_runtime_config();
    config.latent_channels = 1;
    config.transformer = Some(TransformerDenoiserConfig {
        class_name: "Krea2Transformer2DModel".into(),
        in_channels: Some(4),
        out_channels: Some(1),
        patch_size: Some(2),
        num_attention_heads: Some(1),
        ..TransformerDenoiserConfig::default()
    });
    let topology = transformer_denoiser_weight_topology(&DiffusionComponentMetadata {
        class_name: Some("Krea2Transformer2DModel".to_string()),
        weight_entries,
        ..DiffusionComponentMetadata::default()
    });
    let denoiser = NativeTransformerDenoiser::from_hfq(&hfq, &config, &topology).unwrap();
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 2,
        width: 2,
        data: vec![1.0, -1.0, 0.5, -0.5],
    };
    let text_hidden = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![0.5, -0.5],
    };
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());
    let output = denoiser
        .forward_krea_with_runtime_context(
            &latents,
            &[0.0],
            &text_hidden,
            None,
            &mut runtime_context,
        )
        .unwrap();
    assert_eq!(output.batch, 1);
    assert_eq!(output.channels, 1);
    assert_eq!(output.height, 2);
    assert_eq!(output.width, 2);
    assert_eq!(output.data.len(), 4);
    assert!(
        output.data.iter().all(|value| value.is_finite()),
        "krea denoiser produced non-finite latents: {:?}",
        output.data
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_text_fusion_projector_collapses_encoder_layers() {
    // With zero attention output projections and zero SwiGLU down weights,
    // each layerwise/refiner block is the identity, so text_fusion reduces to
    // the projector's weighted sum over the selected encoder layers.
    let dir = std::env::temp_dir().join(format!(
        "hipfire-krea-text-fusion-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("krea-text-fusion.hfq");
    let identity2 = [1.0, 0.0, 0.0, 1.0];
    let zeros4 = [0.0f32; 4];
    let identity_block = |prefix: &str| {
        vec![
            f32_mem_tensor(&format!("{prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.attn.to_q.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.attn.to_k.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.attn.to_v.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.attn.to_gate.weight"), &[2, 2], &zeros4),
            f32_mem_tensor(&format!("{prefix}.attn.to_out.0.weight"), &[2, 2], &zeros4),
            f32_mem_tensor(&format!("{prefix}.ff.gate.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.ff.up.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.ff.down.weight"), &[2, 2], &zeros4),
        ]
    };
    let mut tensors = vec![f32_mem_tensor(
        "transformer/tensors/text_fusion.projector.weight",
        &[1, 2],
        &[0.5, 0.5],
    )];
    tensors.extend(identity_block(
        "transformer/tensors/text_fusion.layerwise_blocks.0",
    ));
    tensors.extend(identity_block(
        "transformer/tensors/text_fusion.refiner_blocks.0",
    ));
    write_hfqm_package_mem(&path, HFQ_ARCH_DIFFUSION, "{}", &tensors).unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let fusion = NativeTextFusion::from_hfq(&hfq, 1).unwrap().unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());
    // [batch=1, seq=1, layers=2, dim=2]: layer0 = [1,2], layer1 = [3,4].
    let layer_stack = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let fused = fusion
        .forward_with_runtime_context(&layer_stack, &mut runtime_context)
        .unwrap();
    assert_eq!(fused.shape, vec![1, 1, 2]);
    // 0.5 * [1,2] + 0.5 * [3,4] = [2,3].
    assert_eq!(fused.data, vec![2.0, 3.0]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_text_fusion_encode_from_layers_stacks_and_fuses() {
    // Same identity-block fixture as the projector test, but driven through
    // the encoder adapter with two separate per-layer hidden states. The
    // stack + projector([0.5,0.5]) must give 0.5*layer0 + 0.5*layer1.
    let dir = std::env::temp_dir().join(format!(
        "hipfire-krea-text-fusion-adapter-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("krea-text-fusion-adapter.hfq");
    let identity2 = [1.0, 0.0, 0.0, 1.0];
    let zeros4 = [0.0f32; 4];
    let identity_block = |prefix: &str| {
        vec![
            f32_mem_tensor(&format!("{prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{prefix}.attn.to_q.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.attn.to_k.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.attn.to_v.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.attn.to_gate.weight"), &[2, 2], &zeros4),
            f32_mem_tensor(&format!("{prefix}.attn.to_out.0.weight"), &[2, 2], &zeros4),
            f32_mem_tensor(&format!("{prefix}.ff.gate.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.ff.up.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{prefix}.ff.down.weight"), &[2, 2], &zeros4),
        ]
    };
    let mut tensors = vec![f32_mem_tensor(
        "transformer/tensors/text_fusion.projector.weight",
        &[1, 2],
        &[0.5, 0.5],
    )];
    tensors.extend(identity_block(
        "transformer/tensors/text_fusion.layerwise_blocks.0",
    ));
    tensors.extend(identity_block(
        "transformer/tensors/text_fusion.refiner_blocks.0",
    ));
    write_hfqm_package_mem(&path, HFQ_ARCH_DIFFUSION, "{}", &tensors).unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let fusion = NativeTextFusion::from_hfq(&hfq, 1).unwrap().unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());
    let layer0 = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![1.0, 2.0],
    };
    let layer1 = CpuTensor {
        shape: vec![1, 1, 2],
        data: vec![3.0, 4.0],
    };
    let fused = fusion
        .encode_from_layers_with_runtime_context(&[layer0, layer1], &mut runtime_context)
        .unwrap();
    assert_eq!(fused.shape, vec![1, 1, 2]);
    assert_eq!(fused.data, vec![2.0, 3.0]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_qwen3_text_encoder_runs_and_captures_selected_layers() {
    // Tiny 2-layer Qwen3 encoder (hidden 4, heads 2, kv_heads 1, head_dim 2).
    // Zero o_proj and mlp.down_proj make every layer the identity, so the
    // captured hidden states equal the token embeddings. Exercises embed ->
    // RMSNorm -> GQA(QK-norm, RoPE, causal) -> SwiGLU -> residual + capture.
    let dir =
        std::env::temp_dir().join(format!("hipfire-qwen3-encoder-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen3-encoder.hfq");
    let ident4 = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let layer = |idx: usize| {
        let p = format!("language_model.layers.{idx}");
        vec![
            f32_mem_tensor(&format!("{p}.input_layernorm.weight"), &[4], &[1.0; 4]),
            f32_mem_tensor(&format!("{p}.self_attn.q_proj.weight"), &[4, 4], &ident4),
            f32_mem_tensor(&format!("{p}.self_attn.k_proj.weight"), &[2, 4], &[0.0; 8]),
            f32_mem_tensor(&format!("{p}.self_attn.v_proj.weight"), &[2, 4], &[0.0; 8]),
            f32_mem_tensor(&format!("{p}.self_attn.q_norm.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{p}.self_attn.k_norm.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{p}.self_attn.o_proj.weight"), &[4, 4], &[0.0; 16]),
            f32_mem_tensor(
                &format!("{p}.post_attention_layernorm.weight"),
                &[4],
                &[1.0; 4],
            ),
            f32_mem_tensor(&format!("{p}.mlp.gate_proj.weight"), &[4, 4], &ident4),
            f32_mem_tensor(&format!("{p}.mlp.up_proj.weight"), &[4, 4], &ident4),
            f32_mem_tensor(&format!("{p}.mlp.down_proj.weight"), &[4, 4], &[0.0; 16]),
        ]
    };
    let mut tensors = vec![f32_mem_tensor(
        "language_model.embed_tokens.weight",
        &[8, 4],
        &[
            0.0, 0.0, 0.0, 0.0, // token 0
            1.0, 2.0, 3.0, 4.0, // token 1
            5.0, 6.0, 7.0, 8.0, // token 2
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ],
    )];
    tensors.extend(layer(0));
    tensors.extend(layer(1));
    write_hfqm_package_mem(&path, HFQ_ARCH_DIFFUSION, "{}", &tensors).unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let encoder = NativeQwen3TextEncoder::from_hfq(&hfq, "language_model", 2, 1, 2, 5_000_000.0)
        .unwrap()
        .unwrap();
    let mut runtime_context =
        DiffusionGenerationRuntimeContext::new(DiffusionGenerationRuntimeOptions::default());
    let captured = encoder
        .encode(&[1, 2], &[1, 2], &mut runtime_context)
        .unwrap();
    assert_eq!(captured.len(), 2);
    for hidden in &captured {
        assert_eq!(hidden.shape, vec![1, 2, 4]);
        assert_eq!(hidden.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
    let _ = fs::remove_dir_all(&dir);
}
