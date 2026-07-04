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
fn transformer_topology_detects_qwen_image_layout() {
    let topology = transformer_denoiser_weight_topology(&DiffusionComponentMetadata {
        class_name: Some("QwenImageTransformer2DModel".to_string()),
        config_entry: Some("transformer/config.json".to_string()),
        weight_entries: vec![
            "transformer/tensors/img_in.weight".to_string(),
            "transformer/tensors/proj_out.weight".to_string(),
            "transformer/tensors/transformer_blocks.0.attn.add_q_proj.weight".to_string(),
            "transformer/tensors/transformer_blocks.0.txt_mod.1.weight".to_string(),
            "transformer/tensors/transformer_blocks.1.img_mlp.net.0.proj.weight".to_string(),
        ],
        tensor_roles: Vec::new(),
    });

    assert_eq!(topology.family, TransformerDenoiserFamily::QwenImage);
    assert_eq!(topology.block_count, 2);
    assert!(topology.has_input_projection);
    assert!(topology.has_output_projection);
    assert!(topology.has_text_modulation);
    assert!(!topology.has_text_fusion);
    assert!(topology
        .diagnostic_label()
        .contains("qwen-image-mmdit blocks=2"));
}

#[test]
fn transformer_topology_detects_krea2_layout() {
    let topology = transformer_denoiser_weight_topology(&DiffusionComponentMetadata {
        class_name: Some("Krea2Transformer2DModel".to_string()),
        config_entry: Some("transformer/config.json".to_string()),
        weight_entries: vec![
            "transformer/tensors/img_in.weight".to_string(),
            "transformer/tensors/final_layer.linear.weight".to_string(),
            "transformer/tensors/text_fusion.projector.weight".to_string(),
            "transformer/tensors/transformer_blocks.0.attn.to_q.weight".to_string(),
            "transformer/tensors/transformer_blocks.27.ff.down.weight".to_string(),
        ],
        tensor_roles: Vec::new(),
    });

    assert_eq!(topology.family, TransformerDenoiserFamily::Krea2);
    assert_eq!(topology.block_count, 2);
    assert!(topology.has_input_projection);
    assert!(topology.has_output_projection);
    assert!(!topology.has_text_modulation);
    assert!(topology.has_text_fusion);
    assert!(topology.diagnostic_label().contains("krea2-mmdit"));
}

#[test]
fn clip_tokenizer_pads_and_keeps_special_tokens() {
    let vocab = br#"{
            "<|startoftext|>": 49406,
            "<|endoftext|>": 49407,
            "a</w>": 10,
            "cat</w>": 11
        }"#;
    let merges = b"#version: 0.2\nc a\nca t</w>\n";
    let tokenizer = ClipTokenizer::from_bytes(vocab, merges, 6).unwrap();
    let encoded = tokenizer.encode_padded("a cat");

    assert_eq!(encoded[0], 49406);
    assert_eq!(encoded[1], 10);
    assert_eq!(encoded[2], 11);
    assert_eq!(encoded[3], 49407);
    assert_eq!(encoded[4], 49407);
    assert_eq!(encoded[5], 49407);
}

#[test]
fn transformer_block_loads_from_hfq_and_preserves_residual_with_zero_weights() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-transformer-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("transformer.hfq");
    let metadata = minimal_metadata();
    let prefix = "unet/tensors/down_blocks.0.attentions.0.transformer_blocks.0";
    let mut tensors = vec![
        f32_mem_tensor(&format!("{prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{prefix}.norm2.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{prefix}.norm2.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{prefix}.norm3.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{prefix}.norm3.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{prefix}.ff.net.0.proj.weight"),
            &[4, 2],
            &[0.0; 8],
        ),
        f32_mem_tensor(&format!("{prefix}.ff.net.0.proj.bias"), &[4], &[0.0; 4]),
        f32_mem_tensor(&format!("{prefix}.ff.net.2.weight"), &[2, 2], &[0.0; 4]),
        f32_mem_tensor(&format!("{prefix}.ff.net.2.bias"), &[2], &[0.0; 2]),
    ];
    push_zero_attention_tensors(&mut tensors, &format!("{prefix}.attn1"), 2, 2);
    push_zero_attention_tensors(&mut tensors, &format!("{prefix}.attn2"), 2, 3);
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
    let block = BasicTransformerBlock::from_hfq(&hfq, prefix, 1).unwrap();
    let hidden = CpuTensor {
        shape: vec![1, 2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let encoder = CpuTensor {
        shape: vec![1, 1, 3],
        data: vec![0.5, 0.25, -0.5],
    };
    let output = block.forward(&hidden, &encoder).unwrap();
    assert_eq!(output.shape, hidden.shape);
    assert_eq!(output.data, hidden.data);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for transformer block routing test: {error}");
        } else {
            let hip = block
                .forward_with_runtime_options(
                    &hidden,
                    &encoder,
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                )
                .unwrap();
            assert_eq!(hip.shape, output.shape);
            assert!(f32_slices_close(&hip.data, &output.data, 1e-5));

            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let hip_context = block
                .forward_with_runtime_context(&hidden, &encoder, &mut runtime_context)
                .unwrap();
            assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
            assert_eq!(hip_context.shape, output.shape);
            assert!(f32_slices_close(&hip_context.data, &output.data, 1e-5));
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn transformer2d_model_loads_from_hfq_and_preserves_residual_with_zero_weights() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-transformer2d-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("transformer2d.hfq");
    let metadata = minimal_metadata();
    let prefix = "unet/tensors/down_blocks.0.attentions.0";
    let block = format!("{prefix}.transformer_blocks.0");
    let mut tensors = vec![
        f32_mem_tensor(&format!("{prefix}.norm.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{prefix}.norm.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{prefix}.proj_in.weight"),
            &[2, 2, 1, 1],
            &[0.0; 4],
        ),
        f32_mem_tensor(&format!("{prefix}.proj_in.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{prefix}.proj_out.weight"),
            &[2, 2, 1, 1],
            &[0.0; 4],
        ),
        f32_mem_tensor(&format!("{prefix}.proj_out.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{block}.norm1.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{block}.norm1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{block}.norm2.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{block}.norm2.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{block}.norm3.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{block}.norm3.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{block}.ff.net.0.proj.weight"), &[4, 2], &[0.0; 8]),
        f32_mem_tensor(&format!("{block}.ff.net.0.proj.bias"), &[4], &[0.0; 4]),
        f32_mem_tensor(&format!("{block}.ff.net.2.weight"), &[2, 2], &[0.0; 4]),
        f32_mem_tensor(&format!("{block}.ff.net.2.bias"), &[2], &[0.0; 2]),
    ];
    push_zero_attention_tensors(&mut tensors, &format!("{block}.attn1"), 2, 2);
    push_zero_attention_tensors(&mut tensors, &format!("{block}.attn2"), 2, 3);
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
    let model = Transformer2DModel::from_hfq(&hfq, prefix, 1, 1, 1e-5).unwrap();
    let input = CpuTensor {
        shape: vec![1, 2, 2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
    };
    let encoder = CpuTensor {
        shape: vec![1, 1, 3],
        data: vec![0.5, 0.25, -0.5],
    };
    let output = model.forward(&input, &encoder).unwrap();
    assert_eq!(output.shape, input.shape);
    assert_eq!(output.data, input.data);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for transformer2d routing test: {error}");
        } else {
            let hip = model
                .forward_with_runtime_options(
                    &input,
                    &encoder,
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                )
                .unwrap();
            assert_eq!(hip.shape, output.shape);
            assert!(f32_slices_close(&hip.data, &output.data, 1e-5));

            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let hip_context = model
                .forward_with_runtime_context(&input, &encoder, &mut runtime_context)
                .unwrap();
            assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
            assert_eq!(hip_context.shape, output.shape);
            assert!(f32_slices_close(&hip_context.data, &output.data, 1e-5));
        }
    }
    let _ = fs::remove_dir_all(&dir);
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

#[test]
fn clip_text_encoder_pools_eos_hidden_state_and_applies_projection() {
    let encoder = ClipTextEncoder {
        token_embedding: CpuTensor {
            shape: vec![3, 2],
            data: vec![0.0, 0.0, 1.0, -1.0, 0.5, 0.5],
        },
        position_embedding: CpuTensor {
            shape: vec![3, 2],
            data: vec![0.0; 6],
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
            data: vec![2.0, 0.0, 0.0, 3.0],
        }),
        hidden_size: 2,
        max_length: 3,
        n_heads: 1,
    };

    let (hidden, pooled) = encoder.encode_tokens_with_pooled(&[0, 1, 2], 1).unwrap();
    let pooled = pooled.unwrap();

    assert_eq!(hidden.shape, vec![3, 2]);
    assert_eq!(pooled.len(), 2);
    assert!((pooled[0] - 2.0).abs() < 1e-4);
    assert!((pooled[1] + 3.0).abs() < 1e-4);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for CLIP pooled routing test: {error}");
        } else {
            let (hip_hidden, hip_pooled) = encoder
                .encode_tokens_with_pooled_and_runtime_options(
                    &[0, 1, 2],
                    1,
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                )
                .unwrap();
            let hip_pooled = hip_pooled.unwrap();
            assert_eq!(hip_hidden.shape, hidden.shape);
            assert!(f32_slices_close(&hip_hidden.data, &hidden.data, 1e-5));
            assert!(f32_slices_close(&hip_pooled, &pooled, 1e-5));
        }
    }
}

#[test]
fn wan_resnet_block_zero_conv_preserves_residual() {
    // conv2 weight/bias zero => the block contributes zero => output == input
    // (same in/out channels, no shortcut). Exercises the Wan resnet path:
    // RMSNorm -> SiLU -> causal conv -> RMSNorm -> SiLU -> causal conv -> add.
    let dir = std::env::temp_dir().join(format!("hipfire-wan-resnet-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("wan-resnet.hfq");
    let prefix = "decoder.mid_block.resnets.0";
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(&format!("{prefix}.norm1.gamma"), &[1, 1, 1, 1], &[1.0]),
            f32_mem_tensor(
                &format!("{prefix}.conv1.weight"),
                &[1, 1, 3, 3, 3],
                &[0.0; 27],
            ),
            f32_mem_tensor(&format!("{prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.gamma"), &[1, 1, 1, 1], &[1.0]),
            f32_mem_tensor(
                &format!("{prefix}.conv2.weight"),
                &[1, 1, 3, 3, 3],
                &[0.0; 27],
            ),
            f32_mem_tensor(&format!("{prefix}.conv2.bias"), &[1], &[0.0]),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let block = WanResnetBlock::from_hfq(&hfq, prefix).unwrap();
    let input = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, -2.0, 3.0, 0.5],
    };
    let out = block.forward(&input).unwrap();
    assert_eq!(out.shape, input.shape);
    assert_eq!(out.data, input.data);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn wan_mid_attention_zero_proj_preserves_residual() {
    // A zero output projection makes the attention contribution zero, so the
    // block is the identity. Exercises RMSNorm -> 1x1 qkv -> spatial softmax
    // attention -> 1x1 proj -> residual add.
    let dir =
        std::env::temp_dir().join(format!("hipfire-wan-mid-attn-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("wan-mid-attn.hfq");
    let prefix = "decoder.mid_block.attentions.0";
    write_hfqm_package_mem(
        &path,
        HFQ_ARCH_DIFFUSION,
        "{}",
        &[
            f32_mem_tensor(&format!("{prefix}.norm.gamma"), &[1, 1, 1], &[1.0]),
            f32_mem_tensor(
                &format!("{prefix}.to_qkv.weight"),
                &[3, 1, 1, 1],
                &[1.0, 1.0, 1.0],
            ),
            f32_mem_tensor(&format!("{prefix}.to_qkv.bias"), &[3], &[0.0, 0.0, 0.0]),
            f32_mem_tensor(&format!("{prefix}.proj.weight"), &[1, 1, 1, 1], &[0.0]),
            f32_mem_tensor(&format!("{prefix}.proj.bias"), &[1], &[0.0]),
        ],
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let attn = WanMidAttention::from_hfq(&hfq, prefix).unwrap();
    let input = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, -2.0, 3.0, 0.5],
    };
    let out = attn.forward(&input).unwrap();
    assert_eq!(out.shape, input.shape);
    assert_eq!(out.data, input.data);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn wan_image_decoder_assembles_and_runs_end_to_end() {
    // Minimal 1-channel decoder (conv_in -> mid{resnet,attn,resnet} -> one
    // up_block{resnet, upsampler} -> norm_out -> conv_out). All conv weights
    // are zero, so zeros propagate through the residual/attention/upsample
    // path and conv_out emits its per-channel bias -> deterministic output.
    let dir = std::env::temp_dir().join(format!("hipfire-wan-decoder-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("wan-decoder.hfq");
    let conv3d = |name: &str| f32_mem_tensor(name, &[1, 1, 3, 3, 3], &[0.0; 27]);
    let resnet = |prefix: &str| {
        vec![
            f32_mem_tensor(&format!("{prefix}.norm1.gamma"), &[1, 1, 1, 1], &[1.0]),
            conv3d(&format!("{prefix}.conv1.weight")),
            f32_mem_tensor(&format!("{prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{prefix}.norm2.gamma"), &[1, 1, 1, 1], &[1.0]),
            conv3d(&format!("{prefix}.conv2.weight")),
            f32_mem_tensor(&format!("{prefix}.conv2.bias"), &[1], &[0.0]),
        ]
    };
    let mut tensors = vec![
        conv3d("decoder.conv_in.weight"),
        f32_mem_tensor("decoder.conv_in.bias", &[1], &[0.0]),
        f32_mem_tensor(
            "decoder.mid_block.attentions.0.norm.gamma",
            &[1, 1, 1],
            &[1.0],
        ),
        f32_mem_tensor(
            "decoder.mid_block.attentions.0.to_qkv.weight",
            &[3, 1, 1, 1],
            &[0.0, 0.0, 0.0],
        ),
        f32_mem_tensor(
            "decoder.mid_block.attentions.0.to_qkv.bias",
            &[3],
            &[0.0; 3],
        ),
        f32_mem_tensor(
            "decoder.mid_block.attentions.0.proj.weight",
            &[1, 1, 1, 1],
            &[0.0],
        ),
        f32_mem_tensor("decoder.mid_block.attentions.0.proj.bias", &[1], &[0.0]),
        f32_mem_tensor(
            "decoder.up_blocks.0.upsamplers.0.resample.1.weight",
            &[1, 1, 3, 3],
            &[0.0; 9],
        ),
        f32_mem_tensor(
            "decoder.up_blocks.0.upsamplers.0.resample.1.bias",
            &[1],
            &[0.0],
        ),
        f32_mem_tensor("decoder.norm_out.gamma", &[1, 1, 1, 1], &[1.0]),
        f32_mem_tensor("decoder.conv_out.weight", &[3, 1, 3, 3, 3], &[0.0; 81]),
        f32_mem_tensor("decoder.conv_out.bias", &[3], &[0.1, 0.2, 0.3]),
    ];
    tensors.extend(resnet("decoder.mid_block.resnets.0"));
    tensors.extend(resnet("decoder.mid_block.resnets.1"));
    tensors.extend(resnet("decoder.up_blocks.0.resnets.0"));
    write_hfqm_package_mem(&path, HFQ_ARCH_DIFFUSION, "{}", &tensors).unwrap();
    let hfq = HfqFile::open_index_only(&path).unwrap();
    let decoder = WanImageDecoder::from_hfq(&hfq, "decoder").unwrap().unwrap();
    let latent = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let out = decoder.decode(&latent).unwrap();
    // one up_block upsamples 2x2 -> 4x4; conv_out has 3 channels.
    assert_eq!(out.shape, vec![1, 3, 4, 4]);
    assert!(out.data.iter().all(|v| v.is_finite()));
    // Each channel is filled with its conv_out bias.
    for (channel, bias) in [0.1f32, 0.2, 0.3].iter().enumerate() {
        for pos in 0..16 {
            assert!((out.data[channel * 16 + pos] - bias).abs() < 1e-6);
        }
    }
    let _ = fs::remove_dir_all(&dir);
}
