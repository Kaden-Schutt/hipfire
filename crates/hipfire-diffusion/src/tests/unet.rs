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
fn unet_input_centering_matches_diffusers_config() {
    let sample = CpuTensor {
        shape: vec![1, 1, 1, 3],
        data: vec![0.0, 0.5, 1.0],
    };

    let centered = maybe_center_unet_input(&sample, true);
    let unchanged = maybe_center_unet_input(&sample, false);

    assert_eq!(centered.shape, sample.shape);
    assert_eq!(centered.data, vec![-1.0, 0.0, 1.0]);
    assert_eq!(unchanged, sample);
}

#[test]
fn unet_text_time_embedding_projects_pooled_text_and_time_ids() {
    let add_embedding = UnetTextTimeEmbedding {
        addition_time_embed_dim: 2,
        linear_1_weight: CpuTensor {
            shape: vec![2, 14],
            data: vec![0.0; 28],
        },
        linear_1_bias: CpuTensor {
            shape: vec![2],
            data: vec![1.0, -1.0],
        },
        linear_2_weight: CpuTensor {
            shape: vec![2, 2],
            data: vec![1.0, 0.0, 0.0, 1.0],
        },
        linear_2_bias: CpuTensor {
            shape: vec![2],
            data: vec![0.0, 0.0],
        },
    };
    let text_embeds = CpuTensor {
        shape: vec![1, 2],
        data: vec![0.5, -0.25],
    };
    let time_ids = CpuTensor {
        shape: vec![1, 6],
        data: vec![512.0, 512.0, 0.0, 0.0, 512.0, 512.0],
    };

    let output = add_embedding
        .forward(&text_embeds, &time_ids, true, 0.0)
        .unwrap();

    assert_eq!(output.shape, vec![1, 2]);
    assert!((output.data[0] - silu(1.0)).abs() < 1e-6);
    assert!((output.data[1] - silu(-1.0)).abs() < 1e-6);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!(
                "skip: ROCm GPU unavailable for UNet text-time embedding routing test: {error}"
            );
        } else {
            let hip = add_embedding
                .forward_with_runtime_options(
                    &text_embeds,
                    &time_ids,
                    true,
                    0.0,
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                )
                .unwrap();
            assert_eq!(hip.shape, output.shape);
            assert!(f32_slices_close(&hip.data, &output.data, 1e-5));
        }
    }
}

#[test]
fn unet_resnet_block_loads_time_projection_from_hfq() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-unet-resnet-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("unet-resnet.hfq");
    let prefix = "unet/tensors/down_blocks.0.resnets.0";
    let metadata = minimal_metadata();
    let identity_conv = center_identity_conv2(2);
    let tensors = [
        f32_mem_tensor(&format!("{prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{prefix}.conv1.weight"),
            &[2, 2, 3, 3],
            &identity_conv,
        ),
        f32_mem_tensor(&format!("{prefix}.conv1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{prefix}.time_emb_proj.weight"),
            &[2, 2],
            &[1.0, 0.0, 0.0, 1.0],
        ),
        f32_mem_tensor(&format!("{prefix}.time_emb_proj.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(&format!("{prefix}.norm2.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{prefix}.norm2.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{prefix}.conv2.weight"),
            &[2, 2, 3, 3],
            &identity_conv,
        ),
        f32_mem_tensor(&format!("{prefix}.conv2.bias"), &[2], &[0.0, 0.0]),
    ];
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
    let block = UnetResnetBlock2D::from_hfq(&hfq, prefix, 1, 1e-5).unwrap();
    let input = CpuTensor {
        shape: vec![1, 2, 1, 1],
        data: vec![0.0, 2.0],
    };
    let time_a = CpuTensor {
        shape: vec![1, 2],
        data: vec![0.0, 0.0],
    };
    let time_b = CpuTensor {
        shape: vec![1, 2],
        data: vec![2.0, 0.0],
    };
    let out_a = block.forward(&input, &time_a).unwrap();
    let out_b = block.forward(&input, &time_b).unwrap();
    assert_eq!(out_a.shape, input.shape);
    assert_eq!(out_b.shape, input.shape);
    assert_ne!(out_a.data, out_b.data);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for UNet ResNet context test: {error}");
        } else {
            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let hip = block
                .forward_with_runtime_context(&input, &time_b, &mut runtime_context)
                .unwrap();
            assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
            assert_eq!(hip.shape, out_b.shape);
            assert!(f32_slices_close(&hip.data, &out_b.data, 1e-5));
        }
    }

    let bad_time = CpuTensor {
        shape: vec![2, 2],
        data: vec![0.0; 4],
    };
    assert!(block.forward(&input, &bad_time).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn unet_down_block_forward_collects_skips_and_downsamples() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-down-block-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("down-block.hfq");
    let metadata = minimal_metadata();
    let block_prefix = "unet/tensors/down_blocks.0";
    let resnet_prefix = format!("{block_prefix}.resnets.0");
    let attention_prefix = format!("{block_prefix}.attentions.0");
    let block = format!("{attention_prefix}.transformer_blocks.0");
    let identity_conv = center_identity_conv2(2);
    let mut tensors = vec![
        f32_mem_tensor(&format!("{resnet_prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{resnet_prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{resnet_prefix}.conv1.weight"),
            &[2, 2, 3, 3],
            &identity_conv,
        ),
        f32_mem_tensor(&format!("{resnet_prefix}.conv1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{resnet_prefix}.time_emb_proj.weight"),
            &[2, 2],
            &[0.0; 4],
        ),
        f32_mem_tensor(
            &format!("{resnet_prefix}.time_emb_proj.bias"),
            &[2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(&format!("{resnet_prefix}.norm2.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{resnet_prefix}.norm2.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{resnet_prefix}.conv2.weight"),
            &[2, 2, 3, 3],
            &[0.0; 36],
        ),
        f32_mem_tensor(&format!("{resnet_prefix}.conv2.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{block_prefix}.downsamplers.0.conv.weight"),
            &[2, 2, 3, 3],
            &identity_conv,
        ),
        f32_mem_tensor(
            &format!("{block_prefix}.downsamplers.0.conv.bias"),
            &[2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(
            &format!("{attention_prefix}.norm.weight"),
            &[2],
            &[1.0, 1.0],
        ),
        f32_mem_tensor(&format!("{attention_prefix}.norm.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{attention_prefix}.proj_in.weight"),
            &[2, 2, 1, 1],
            &[0.0; 4],
        ),
        f32_mem_tensor(
            &format!("{attention_prefix}.proj_in.bias"),
            &[2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(
            &format!("{attention_prefix}.proj_out.weight"),
            &[2, 2, 1, 1],
            &[0.0; 4],
        ),
        f32_mem_tensor(
            &format!("{attention_prefix}.proj_out.bias"),
            &[2],
            &[0.0, 0.0],
        ),
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
    let block = UnetDownBlock2D::from_hfq(&hfq, 0, 1, 1, 1, 1e-5).unwrap();
    let input = CpuTensor {
        shape: vec![1, 2, 4, 4],
        data: (0..32).map(|value| value as f32).collect(),
    };
    let time = CpuTensor {
        shape: vec![1, 2],
        data: vec![0.0, 0.0],
    };
    let encoder = CpuTensor {
        shape: vec![1, 1, 3],
        data: vec![0.0; 3],
    };
    let input_for_hip = input.clone();
    let (hidden, skips) = block.forward(input, &time, &encoder).unwrap();
    assert_eq!(skips.len(), 2);
    assert_eq!(skips[0].shape, vec![1, 2, 4, 4]);
    assert_eq!(skips[1].shape, vec![1, 2, 2, 2]);
    assert_eq!(hidden.shape, vec![1, 2, 2, 2]);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for UNet down block context test: {error}");
        } else {
            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let (hip_hidden, hip_skips) = block
                .forward_with_runtime_context(input_for_hip, &time, &encoder, &mut runtime_context)
                .unwrap();
            assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
            assert_eq!(hip_hidden.shape, hidden.shape);
            assert!(f32_slices_close(&hip_hidden.data, &hidden.data, 1e-5));
            assert_eq!(hip_skips.len(), skips.len());
            for (hip_skip, cpu_skip) in hip_skips.iter().zip(&skips) {
                assert_eq!(hip_skip.shape, cpu_skip.shape);
                assert!(f32_slices_close(&hip_skip.data, &cpu_skip.data, 1e-5));
            }
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn unet_up_block_pops_skip_and_upsamples() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-up-block-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("up-block.hfq");
    let metadata = minimal_metadata();
    let prefix = "unet/tensors/up_blocks.0";
    let resnet_prefix = format!("{prefix}.resnets.0");
    let tensors = [
        f32_mem_tensor(&format!("{resnet_prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{resnet_prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            &format!("{resnet_prefix}.conv1.weight"),
            &[1, 2, 3, 3],
            &[0.0; 18],
        ),
        f32_mem_tensor(&format!("{resnet_prefix}.conv1.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{resnet_prefix}.time_emb_proj.weight"),
            &[1, 2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(&format!("{resnet_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{resnet_prefix}.norm2.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{resnet_prefix}.norm2.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{resnet_prefix}.conv2.weight"),
            &[1, 1, 3, 3],
            &[0.0; 9],
        ),
        f32_mem_tensor(&format!("{resnet_prefix}.conv2.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{resnet_prefix}.conv_shortcut.weight"),
            &[1, 2, 1, 1],
            &[1.0, 0.0],
        ),
        f32_mem_tensor(&format!("{resnet_prefix}.conv_shortcut.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{prefix}.upsamplers.0.conv.weight"),
            &[1, 1, 3, 3],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        f32_mem_tensor(&format!("{prefix}.upsamplers.0.conv.bias"), &[1], &[0.0]),
    ];
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
    let block = UnetUpBlock2D::from_hfq(&hfq, 0, 1, 1, 1e-5).unwrap();
    let hidden = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let mut skips = vec![CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![10.0, 20.0, 30.0, 40.0],
    }];
    let time = CpuTensor {
        shape: vec![1, 2],
        data: vec![0.0, 0.0],
    };
    let encoder = CpuTensor {
        shape: vec![1, 1, 3],
        data: vec![0.0; 3],
    };
    let hidden_for_hip = hidden.clone();
    let mut skips_for_hip = skips.clone();
    let output = block.forward(hidden, &mut skips, &time, &encoder).unwrap();
    assert!(skips.is_empty());
    assert_eq!(output.shape, vec![1, 1, 4, 4]);
    assert_eq!(&output.data[0..4], &[1.0, 1.0, 2.0, 2.0]);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for UNet up block context test: {error}");
        } else {
            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let hip_output = block
                .forward_with_runtime_context(
                    hidden_for_hip,
                    &mut skips_for_hip,
                    &time,
                    &encoder,
                    &mut runtime_context,
                )
                .unwrap();
            assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
            assert!(skips_for_hip.is_empty());
            assert_eq!(hip_output.shape, output.shape);
            assert!(f32_slices_close(&hip_output.data, &output.data, 1e-5));
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn unet_mid_block_loads_attention_and_resnets_from_hfq() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-mid-block-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("mid-block.hfq");
    let metadata = minimal_metadata();
    let identity1 = center_identity_conv(1);
    let mid0_prefix = "unet/tensors/mid_block.resnets.0";
    let mid1_prefix = "unet/tensors/mid_block.resnets.1";
    let attention_prefix = "unet/tensors/mid_block.attentions.0";
    let block_prefix = format!("{attention_prefix}.transformer_blocks.0");
    let mut tensors = vec![
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
        f32_mem_tensor(&format!("{attention_prefix}.norm.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{attention_prefix}.norm.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{attention_prefix}.proj_in.weight"),
            &[1, 1, 1, 1],
            &[0.0],
        ),
        f32_mem_tensor(&format!("{attention_prefix}.proj_in.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{attention_prefix}.proj_out.weight"),
            &[1, 1, 1, 1],
            &[0.0],
        ),
        f32_mem_tensor(&format!("{attention_prefix}.proj_out.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{block_prefix}.norm1.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{block_prefix}.norm1.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{block_prefix}.norm2.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{block_prefix}.norm2.bias"), &[1], &[0.0]),
        f32_mem_tensor(&format!("{block_prefix}.norm3.weight"), &[1], &[1.0]),
        f32_mem_tensor(&format!("{block_prefix}.norm3.bias"), &[1], &[0.0]),
        f32_mem_tensor(
            &format!("{block_prefix}.ff.net.0.proj.weight"),
            &[2, 1],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(
            &format!("{block_prefix}.ff.net.0.proj.bias"),
            &[2],
            &[0.0, 0.0],
        ),
        f32_mem_tensor(&format!("{block_prefix}.ff.net.2.weight"), &[1, 1], &[0.0]),
        f32_mem_tensor(&format!("{block_prefix}.ff.net.2.bias"), &[1], &[0.0]),
    ];
    push_zero_attention_tensors(&mut tensors, &format!("{block_prefix}.attn1"), 1, 1);
    push_zero_attention_tensors(&mut tensors, &format!("{block_prefix}.attn2"), 1, 1);
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
        center_input_sample: true,
        flip_sin_to_cos: true,
        freq_shift: 0.0,
        addition_embed_type: None,
        addition_time_embed_dim: None,
        projection_class_embeddings_input_dim: None,
    };
    let mid_block = UnetMidBlock2DCrossAttn::from_hfq(&hfq, &config)
        .unwrap()
        .unwrap();
    let input = CpuTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let time = CpuTensor {
        shape: vec![1, 2],
        data: vec![0.0, 0.0],
    };
    let encoder = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![0.0],
    };
    let input_for_hip = input.clone();

    let output = mid_block.forward(input, &time, &encoder).unwrap();

    assert!(mid_block.attention.is_some());
    assert!(mid_block.resnet_1.is_some());
    assert_eq!(output.shape, vec![1, 1, 2, 2]);
    assert!(output.data.iter().all(|value| value.is_finite()));

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for UNet mid block context test: {error}");
        } else {
            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let hip_output = mid_block
                .forward_with_runtime_context(input_for_hip, &time, &encoder, &mut runtime_context)
                .unwrap();
            assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
            assert_eq!(hip_output.shape, output.shape);
            assert!(f32_slices_close(&hip_output.data, &output.data, 1e-5));
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn unet_time_embedding_loads_from_hfq() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-time-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("time.hfq");
    let metadata = minimal_metadata();
    let identity = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    let tensors = [
        f32_mem_tensor(
            "unet/tensors/time_embedding.linear_1.weight",
            &[4, 4],
            &identity,
        ),
        f32_mem_tensor("unet/tensors/time_embedding.linear_1.bias", &[4], &[0.0; 4]),
        f32_mem_tensor(
            "unet/tensors/time_embedding.linear_2.weight",
            &[4, 4],
            &identity,
        ),
        f32_mem_tensor("unet/tensors/time_embedding.linear_2.bias", &[4], &[0.0; 4]),
    ];
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tensors,
    )
    .unwrap();
    let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
    let time_embedding = UnetTimeEmbedding::from_hfq(&hfq).unwrap();
    let output = time_embedding.forward(&[0.0, 1.0], true, 0.0).unwrap();
    assert_eq!(output.shape, vec![2, 4]);
    assert!(output.data.iter().all(|value| value.is_finite()));
    assert!(output.data[0] > 0.73 && output.data[2] == 0.0);

    {
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
            eprintln!("skip: ROCm GPU unavailable for UNet time embedding routing test: {error}");
        } else {
            let hip = time_embedding
                .forward_with_runtime_options(
                    &[0.0, 1.0],
                    true,
                    0.0,
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                )
                .unwrap();
            assert_eq!(hip.shape, output.shape);
            assert!(f32_slices_close(&hip.data, &output.data, 1e-5));

            let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
            );
            let hip_context = time_embedding
                .forward_with_runtime_context(&[0.0, 1.0], true, 0.0, &mut runtime_context)
                .unwrap();
            assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
            assert_eq!(hip_context.shape, output.shape);
            assert!(f32_slices_close(&hip_context.data, &output.data, 1e-5));
        }
    }
    let _ = fs::remove_dir_all(&dir);
}
