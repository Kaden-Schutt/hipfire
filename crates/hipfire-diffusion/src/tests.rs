    use super::*;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

    const DEFAULT_TINY_SD_HFQ: &str = "/tmp/hipfire-tiny-sd-diffusion.hfq";

    fn tiny_sd_hfq_path() -> PathBuf {
        std::env::var_os("HIPFIRE_TINY_SD_HFQ")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_TINY_SD_HFQ))
    }

    fn skip_missing_tiny_sd(path: &Path) -> bool {
        if path.exists() {
            false
        } else {
            eprintln!(
                "skip: set HIPFIRE_TINY_SD_HFQ or create {}",
                DEFAULT_TINY_SD_HFQ
            );
            true
        }
    }

    #[test]
    fn parses_diffusion_metadata() {
        let metadata = minimal_metadata();
        let json = serde_json::to_string(&metadata).unwrap();
        assert_eq!(parse_diffusion_metadata(&json).unwrap(), metadata);
    }

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
    fn native_source_runtime_support_rejects_transformer_pipeline_classes() {
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

        assert!(error.contains("Stable Diffusion UNet-family"));
        assert!(error.contains("Krea2Pipeline"));
        assert!(error.contains("transformer denoiser"));
        assert!(error.contains("krea2-mmdit"));
    }

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
        let embedding = NativeTransformerTimestepEmbedding::from_hfq(
            &hfq,
            TransformerDenoiserFamily::QwenImage,
        )
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
        let modulation = NativeTransformerBlockModulation::from_hfq(
            &hfq,
            TransformerDenoiserFamily::QwenImage,
            0,
        )
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
        let expected_image =
            scaled_dot_product_attention(&image_hidden, &joint, &joint, 1).unwrap();
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

        let out = scaled_dot_product_attention_with_key_mask(&q, &k, &v, 1, Some(&[false, true]))
            .unwrap();

        assert_eq!(out.shape, vec![1, 1, 2]);
        assert_f32_close(&out.data, &[7.0, 11.0], 1e-6);
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
        let unmasked_image =
            scaled_dot_product_attention(&image_hidden, &joint, &joint, 1).unwrap();
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
        let expected_image =
            scaled_dot_product_attention(&image_qkv.q, &joint_k, &joint_v, 1).unwrap();
        let expected_text =
            scaled_dot_product_attention(&text_qkv.q, &joint_k, &joint_v, 1).unwrap();

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
        let ff =
            NativeTransformerFeedForward::from_hfq(&hfq, TransformerDenoiserFamily::QwenImage, 0)
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
        let ff = NativeTransformerFeedForward::from_hfq(&hfq, TransformerDenoiserFamily::Krea2, 0)
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
            NativeTransformerBlock::from_hfq(&hfq, TransformerDenoiserFamily::QwenImage, 0, 1)
                .unwrap();
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
    fn rejects_non_diffusion_metadata() {
        let err = parse_diffusion_metadata(r#"{"artifact_kind":"llm","schema_version":1,"pipeline":{"class_name":"x","source":"x"}}"#)
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
    fn inspect_hfq_detects_diffusion_container() {
        let dir =
            std::env::temp_dir().join(format!("hipfire-diffusion-test-{}", std::process::id()));
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
        assert!(!inspection.runtime_support.supported);
        assert!(inspection
            .runtime_support
            .reason
            .as_deref()
            .unwrap()
            .contains("transformer denoiser"));
        let _ = fs::remove_dir_all(&dir);
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
            CpuTensor::from_hfq(&hfq, "transformer/tensors/patch_embed.proj.weight").unwrap();
        let norm_out = CpuTensor::from_hfq(&hfq, "transformer/tensors/norm_out.weight").unwrap();
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
        let unet = CpuTensor::from_hfq(&hfq, "unet/tensors/conv_in.weight").unwrap();
        let vae = CpuTensor::from_hfq(&hfq, "vae/tensors/post_quant_conv.weight").unwrap();
        let text = CpuTensor::from_hfq(
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
        let tensor = CpuTensor::from_hfq(&hfq, "unet/tensors/conv_in.bias").unwrap();
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
        let conv_in = CpuTensor::from_hfq(&hfq, "unet/tensors/conv_in.weight").unwrap();
        let conv_out = CpuTensor::from_hfq(&hfq, "unet/tensors/conv_out.bias").unwrap();
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
                        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40,
                        0x00, 0x00, 0x80, 0x40,
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
        let checkpoint_tensor = CpuTensor::from_hfq(
            &hfq,
            "unet/checkpoint_tensors/model.diffusion_model.input_blocks.0.0.weight",
        )
        .unwrap();
        let native_tensor = CpuTensor::from_hfq(&hfq, "unet/tensors/conv_in.weight").unwrap();
        let tokenizer = ClipTokenizer::from_hfq_file(&hfq).unwrap();
        let tokens = tokenizer.encode_padded("a cat");
        let down_resnet =
            CpuTensor::from_hfq(&hfq, "unet/tensors/down_blocks.0.resnets.0.norm1.weight").unwrap();
        let upsample =
            CpuTensor::from_hfq(&hfq, "unet/tensors/up_blocks.0.upsamplers.0.conv.weight").unwrap();
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
                        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40,
                        0x00, 0x00, 0x80, 0x40,
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

        let unet =
            parse_pytorch_state_dict(&root.join("unet/diffusion_pytorch_model.bin")).unwrap();
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
    fn cpu_tensor_loads_supported_source_and_packed_formats_from_hfq() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-tensor-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tensors.hfq");
        let metadata = minimal_metadata();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &[
                HfqMemTensor {
                    name: "unet/config.json".into(),
                    quant_type: QT_DIFFUSION_JSON,
                    shape: vec![2],
                    group_size: 0,
                    data: b"{}".to_vec(),
                },
                HfqMemTensor {
                    name: "f16".into(),
                    quant_type: QT_DIFFUSION_TENSOR_F16,
                    shape: vec![2],
                    group_size: 0,
                    data: [
                        f32_to_f16_bits(1.5).to_le_bytes(),
                        f32_to_f16_bits(-2.0).to_le_bytes(),
                    ]
                    .concat(),
                },
                HfqMemTensor {
                    name: "bf16".into(),
                    quant_type: QT_DIFFUSION_TENSOR_BF16,
                    shape: vec![1],
                    group_size: 0,
                    data: (((3.0f32).to_bits() >> 16) as u16).to_le_bytes().to_vec(),
                },
                HfqMemTensor {
                    name: "f32".into(),
                    quant_type: QT_DIFFUSION_TENSOR_F32,
                    shape: vec![1],
                    group_size: 0,
                    data: 4.25f32.to_le_bytes().to_vec(),
                },
                HfqMemTensor {
                    name: "q8".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q8F16,
                    shape: vec![3],
                    group_size: 32,
                    data: [
                        f32_to_f16_bits(0.5).to_le_bytes().as_slice(),
                        &[2u8, (-4i8) as u8, 7u8],
                        &[0u8; 29],
                    ]
                    .concat(),
                },
                HfqMemTensor {
                    name: "q4".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q4F16_G64,
                    shape: vec![4],
                    group_size: 64,
                    data: [
                        f32_to_f16_bits(0.25).to_le_bytes().as_slice(),
                        f32_to_f16_bits(-1.0).to_le_bytes().as_slice(),
                        &[0x00u8, 0x08u8, 0x04u8, 0x0bu8],
                        &[0u8; 28],
                    ]
                    .concat(),
                },
                HfqMemTensor {
                    name: "q4k".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q4_K,
                    shape: vec![4],
                    group_size: 256,
                    data: q4k_test_block(&[4, 8, 0, 7]),
                },
                hfq4_mem_tensor(
                    "hfq4g128",
                    QT_DIFFUSION_TENSOR_HFQ4_G128,
                    &[4],
                    128,
                    &[0, 8, 4, 11],
                ),
                hfq4_mem_tensor(
                    "hfq4g256",
                    QT_DIFFUSION_TENSOR_HFQ4_G256,
                    &[4],
                    256,
                    &[0, 8, 4, 11],
                ),
                hfq6_mem_tensor("hfq6g256", &[4], &[0, 8, 4, 11]),
            ],
        )
        .unwrap();

        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "f16").unwrap().data,
            vec![1.5, -2.0]
        );
        assert_eq!(CpuTensor::from_hfq(&hfq, "bf16").unwrap().data, vec![3.0]);
        assert_eq!(CpuTensor::from_hfq(&hfq, "f32").unwrap().data, vec![4.25]);
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "q8").unwrap().data,
            vec![1.0, -2.0, 3.5]
        );
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "q4").unwrap().data,
            vec![-1.0, 1.0, 0.0, 1.75]
        );
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "q4k").unwrap().data,
            vec![1.0, 2.0, 0.0, 1.75]
        );
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "hfq4g128").unwrap().data,
            vec![-1.0, 1.0, 0.0, 1.75]
        );
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "hfq4g256").unwrap().data,
            vec![-1.0, 1.0, 0.0, 1.75]
        );
        assert_eq!(
            CpuTensor::from_hfq(&hfq, "hfq6g256").unwrap().data,
            vec![-1.0, 1.0, 0.0, 1.75]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cpu_tensor_rejects_truncated_packed_payloads() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-truncated-packed-tensor-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("truncated-tensors.hfq");
        let metadata = minimal_metadata();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &[
                bytes_mem_tensor("unet/config.json", QT_DIFFUSION_JSON, b"{}"),
                HfqMemTensor {
                    name: "bad_q4".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q4F16_G64,
                    shape: vec![64],
                    group_size: 64,
                    data: vec![0u8; 35],
                },
                HfqMemTensor {
                    name: "bad_q8".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q8F16,
                    shape: vec![32],
                    group_size: 32,
                    data: vec![0u8; 33],
                },
                HfqMemTensor {
                    name: "bad_q4k".into(),
                    quant_type: QT_DIFFUSION_TENSOR_Q4_K,
                    shape: vec![256],
                    group_size: 256,
                    data: vec![0u8; 143],
                },
                HfqMemTensor {
                    name: "bad_hfq4g128".into(),
                    quant_type: QT_DIFFUSION_TENSOR_HFQ4_G128,
                    shape: vec![128],
                    group_size: 128,
                    data: vec![0u8; 71],
                },
                HfqMemTensor {
                    name: "bad_hfq4g256".into(),
                    quant_type: QT_DIFFUSION_TENSOR_HFQ4_G256,
                    shape: vec![256],
                    group_size: 256,
                    data: vec![0u8; 135],
                },
                HfqMemTensor {
                    name: "bad_hfq6g256".into(),
                    quant_type: QT_DIFFUSION_TENSOR_HFQ6_G256,
                    shape: vec![256],
                    group_size: 256,
                    data: vec![0u8; 199],
                },
            ],
        )
        .unwrap();

        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let q4_error = CpuTensor::from_hfq(&hfq, "bad_q4").unwrap_err();
        assert!(q4_error.to_string().contains("Q4F16_G64"));
        assert!(q4_error.to_string().contains("requires at least 36"));
        let q8_error = CpuTensor::from_hfq(&hfq, "bad_q8").unwrap_err();
        assert!(q8_error.to_string().contains("Q8F16"));
        assert!(q8_error.to_string().contains("requires at least 34"));
        let q4k_error = CpuTensor::from_hfq(&hfq, "bad_q4k").unwrap_err();
        assert!(q4k_error.to_string().contains("Q4_K"));
        assert!(q4k_error.to_string().contains("requires at least 144"));
        let hfq4g128_error = CpuTensor::from_hfq(&hfq, "bad_hfq4g128").unwrap_err();
        assert!(hfq4g128_error.to_string().contains("HFQ4G128"));
        assert!(hfq4g128_error.to_string().contains("requires at least 72"));
        let hfq4g256_error = CpuTensor::from_hfq(&hfq, "bad_hfq4g256").unwrap_err();
        assert!(hfq4g256_error.to_string().contains("HFQ4G256"));
        assert!(hfq4g256_error.to_string().contains("requires at least 136"));
        let hfq6g256_error = CpuTensor::from_hfq(&hfq, "bad_hfq6g256").unwrap_err();
        assert!(hfq6g256_error.to_string().contains("HFQ6G256"));
        assert!(hfq6g256_error.to_string().contains("requires at least 200"));
        let _ = fs::remove_dir_all(&dir);
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
    fn seeded_latents_are_deterministic_and_batched() {
        let a = LatentBatch::seeded_normal(2, 4, 2, 2, &[123, 456]);
        let b = LatentBatch::seeded_normal(2, 4, 2, 2, &[123, 456]);
        let c = LatentBatch::seeded_normal(2, 4, 2, 2, &[123, 789]);

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.batch, 2);
        assert_eq!(a.len_per_batch(), 16);
        assert!(a.data.iter().all(|value| value.is_finite()));
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
    fn scheduler_config_uses_diffusers_beta_sigmas_and_train_timesteps() {
        let config = SchedulerConfig {
            class_name: "EulerDiscreteScheduler".into(),
            beta_start: Some(0.0001),
            beta_end: Some(0.02),
            beta_schedule: Some("linear".into()),
            num_train_timesteps: Some(10),
            prediction_type: Some("epsilon".into()),
            ..SchedulerConfig::default()
        };

        let schedule = DiffusionSchedule::from_config(&config, 3).unwrap();

        assert_eq!(schedule.timesteps, vec![9.0, 5.0, 0.0]);
        assert_eq!(schedule.sigmas.len(), 4);
        assert!(schedule.sigmas[0] > schedule.sigmas[1]);
        assert!(schedule.sigmas[1] > schedule.sigmas[2]);
        assert_eq!(schedule.sigmas[3], 0.0);
    }

    #[test]
    fn dpm_solver_config_uses_diffusers_linspace_timesteps() {
        let config = SchedulerConfig {
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

        let schedule = DiffusionSchedule::from_config(&config, 3).unwrap();

        assert_eq!(schedule.train_timesteps, vec![999, 666, 333]);
        assert_eq!(schedule.timesteps, vec![999.0, 666.0, 333.0]);
        assert_eq!(
            schedule.solver,
            SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 2,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: true,
                thresholding: false,
                dynamic_thresholding_ratio: 0.995,
                sample_max_value: 1.0,
            }
        );
        assert_eq!(schedule.input_scaling, SchedulerInputScaling::None);
        assert_eq!(schedule.initial_noise_sigma(), 1.0);
    }

    #[test]
    fn dpm_solver_config_preserves_dynamic_thresholding_settings() {
        let config = SchedulerConfig {
            class_name: "DPMSolverMultistepScheduler".into(),
            beta_start: Some(0.00085),
            beta_end: Some(0.012),
            beta_schedule: Some("scaled_linear".into()),
            num_train_timesteps: Some(1000),
            prediction_type: Some("epsilon".into()),
            algorithm_type: Some("dpmsolver++".into()),
            solver_order: Some(2),
            solver_type: Some("midpoint".into()),
            thresholding: Some(true),
            dynamic_thresholding_ratio: Some(0.9),
            sample_max_value: Some(2.0),
            ..SchedulerConfig::default()
        };

        let schedule = DiffusionSchedule::from_config(&config, 2).unwrap();

        assert_eq!(
            schedule.solver,
            SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 2,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: true,
                thresholding: true,
                dynamic_thresholding_ratio: 0.9,
                sample_max_value: 2.0,
            }
        );
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
    fn scheduler_request_aliases_select_actual_sampler_config() {
        let config = tiny_sd_scheduler_config_for_tests();

        let dpm = config.resolve_request_scheduler("DPM++ 2M").unwrap();
        let dpm_karras = config.resolve_request_scheduler("DPM++ 2M Karras").unwrap();
        let dpm3 = config.resolve_request_scheduler("DPM++ 3M").unwrap();
        let dpm3_karras = config.resolve_request_scheduler("DPM++ 3M Karras").unwrap();
        let euler = config.resolve_request_scheduler("Euler").unwrap();
        let euler_karras = config.resolve_request_scheduler("Euler Karras").unwrap();
        let euler_a = config.resolve_request_scheduler("Euler a").unwrap();
        let ddim = config.resolve_request_scheduler("DDIM").unwrap();

        assert_eq!(dpm.class_name, "DPMSolverMultistepScheduler");
        assert_eq!(dpm_karras.class_name, "DPMSolverMultistepScheduler");
        assert_eq!(dpm_karras.use_karras_sigmas, Some(true));
        assert_eq!(dpm3.class_name, "DPMSolverMultistepScheduler");
        assert_eq!(dpm3.algorithm_type.as_deref(), Some("dpmsolver++"));
        assert_eq!(dpm3.solver_order, Some(3));
        assert_eq!(dpm3_karras.solver_order, Some(3));
        assert_eq!(dpm3_karras.use_karras_sigmas, Some(true));
        assert_eq!(euler.class_name, "EulerDiscreteScheduler");
        assert_eq!(euler.algorithm_type, None);
        assert_eq!(euler_karras.class_name, "EulerDiscreteScheduler");
        assert_eq!(euler_karras.use_karras_sigmas, Some(true));
        assert_eq!(euler_a.class_name, "EulerAncestralDiscreteScheduler");
        assert_eq!(ddim.class_name, "DDIMScheduler");
        assert!(config.resolve_request_scheduler("not a sampler").is_err());
    }

    #[test]
    fn scheduler_request_alias_changes_run_plan_schedule() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-scheduler-alias-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-scheduler-alias.hfq");
        let metadata = tiny_runtime_metadata();
        write_hfqm_package_mem(
            &hfq_path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tiny_complete_runtime_tensors(),
        )
        .unwrap();
        let mut pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        pipeline.config.scheduler = tiny_sd_scheduler_config_for_tests();
        let mut request = DiffusionBatchRequest {
            conditioning: None,

            prompts: vec![DiffusionPrompt {
                prompt: "a".into(),
                negative_prompt: String::new(),
                seed: 1,
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
            scheduler: "DPM++ 2M".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let dpm_plan = pipeline.prepare_run_plan(&request).unwrap();
        request.scheduler = "Euler".into();
        let euler_plan = pipeline.prepare_run_plan(&request).unwrap();
        request.scheduler = "DDIM".into();
        let ddim_plan = pipeline.prepare_run_plan(&request).unwrap();

        assert!(matches!(
            dpm_plan.schedule.solver,
            SchedulerSolver::DpmSolverMultistep { .. }
        ));
        assert_eq!(euler_plan.schedule.solver, SchedulerSolver::Euler);
        assert_eq!(
            euler_plan.schedule.input_scaling,
            SchedulerInputScaling::Sigma
        );
        assert_eq!(
            ddim_plan.schedule.solver,
            SchedulerSolver::Ddim {
                set_alpha_to_one: true
            }
        );
        assert_eq!(
            ddim_plan.schedule.input_scaling,
            SchedulerInputScaling::None
        );
        assert_ne!(dpm_plan.latents.data, euler_plan.latents.data);
        let _ = fs::remove_dir_all(&dir);
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
    fn dpm_solver_multistep_updates_with_model_output_history() {
        let lambda = |alpha: f32, sigma: f32| alpha.ln() - sigma.ln();
        let schedule = DiffusionSchedule {
            timesteps: vec![2.0, 1.0],
            sigmas: vec![0.3, 0.2, 0.0],
            prediction_type: SchedulerPredictionType::Epsilon,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 2,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: false,
                thresholding: false,
                dynamic_thresholding_ratio: 0.995,
                sample_max_value: 1.0,
            },
            train_timesteps: vec![2, 1],
            alpha_t: vec![0.9, 0.8, 0.7],
            sigma_t: vec![0.1, 0.2, 0.3],
            lambda_t: vec![lambda(0.9, 0.1), lambda(0.8, 0.2), lambda(0.7, 0.3)],
        };
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![1.0],
        };
        let mut state = SchedulerStepState::default();

        schedule.step(&mut latents, &[0.5], 0, &mut state).unwrap();
        let first = latents.data[0];
        schedule.step(&mut latents, &[0.25], 1, &mut state).unwrap();

        assert_eq!(state.lower_order_nums, 2);
        assert_eq!(state.model_outputs.len(), 2);
        assert!(first.is_finite());
        assert!(latents.data[0].is_finite());
        assert_ne!(latents.data[0], first);
    }

    #[test]
    fn dpm_solver_dynamic_thresholding_clips_predicted_original_sample() {
        let schedule = DiffusionSchedule {
            timesteps: vec![0.0],
            sigmas: vec![0.0, 0.0],
            prediction_type: SchedulerPredictionType::Sample,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 2,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: true,
                thresholding: true,
                dynamic_thresholding_ratio: 1.0,
                sample_max_value: 4.0,
            },
            train_timesteps: vec![0],
            alpha_t: vec![1.0],
            sigma_t: vec![0.0],
            lambda_t: vec![0.0],
        };
        let sample = CpuTensor {
            shape: vec![2, 1, 1, 4],
            data: vec![0.0; 8],
        };
        let model_output = [-0.5, 0.5, 2.0, -4.0, 0.2, -3.0, 6.0, -9.0];

        let output = schedule
            .dpm_convert_model_output(&model_output, 0, &sample)
            .unwrap();

        assert_eq!(
            output,
            vec![-0.125, 0.125, 0.5, -1.0, 0.05, -0.75, 1.0, -1.0]
        );
    }

    #[test]
    fn dpm_solver_third_order_update_matches_diffusers_formula() {
        let lambda = |alpha: f32, sigma: f32| alpha.ln() - sigma.ln();
        let schedule = DiffusionSchedule {
            timesteps: vec![3.0, 2.0, 1.0],
            sigmas: vec![0.4, 0.3, 0.2, 0.0],
            prediction_type: SchedulerPredictionType::Sample,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 3,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: false,
                thresholding: false,
                dynamic_thresholding_ratio: 0.995,
                sample_max_value: 1.0,
            },
            train_timesteps: vec![3, 2, 1],
            alpha_t: vec![0.95, 0.85, 0.75, 0.65],
            sigma_t: vec![0.10, 0.20, 0.30, 0.40],
            lambda_t: vec![
                lambda(0.95, 0.10),
                lambda(0.85, 0.20),
                lambda(0.75, 0.30),
                lambda(0.65, 0.40),
            ],
        };
        let sample = CpuTensor {
            shape: vec![1, 1, 1, 1],
            data: vec![1.25],
        };
        let state = SchedulerStepState {
            model_outputs: vec![vec![0.20], vec![0.40], vec![0.70]],
            lower_order_nums: 2,
        };

        let next = schedule
            .dpm_third_order_update(3, 2, 1, 0, &sample, &state)
            .unwrap();

        let lambda_t = schedule.scheduler_lambda(0).unwrap();
        let lambda_s0 = schedule.scheduler_lambda(1).unwrap();
        let lambda_s1 = schedule.scheduler_lambda(2).unwrap();
        let lambda_s2 = schedule.scheduler_lambda(3).unwrap();
        let h = lambda_t - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        let h1 = lambda_s1 - lambda_s2;
        let r0 = h0 / h;
        let r1 = h1 / h;
        let m0 = 0.70;
        let m1 = 0.40;
        let m2 = 0.20;
        let d1_0 = (m0 - m1) / r0;
        let d1_1 = (m1 - m2) / r1;
        let d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1);
        let d2 = (d1_0 - d1_1) / (r0 + r1);
        let exp_neg_h = (-h).exp();
        let expected = (schedule.scheduler_sigma(0).unwrap()
            / schedule.scheduler_sigma(1).unwrap())
            * sample.data[0]
            - (schedule.scheduler_alpha(0).unwrap() * (exp_neg_h - 1.0)) * m0
            + (schedule.scheduler_alpha(0).unwrap() * ((exp_neg_h - 1.0) / h + 1.0)) * d1
            - (schedule.scheduler_alpha(0).unwrap() * ((exp_neg_h - 1.0 + h) / (h * h) - 0.5)) * d2;

        assert!((next.data[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn dpm_solver_order_three_step_uses_third_order_history() {
        let lambda = |alpha: f32, sigma: f32| alpha.ln() - sigma.ln();
        let schedule = DiffusionSchedule {
            timesteps: vec![3.0, 2.0, 1.0],
            sigmas: vec![0.4, 0.3, 0.2, 0.0],
            prediction_type: SchedulerPredictionType::Sample,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::DpmSolverMultistep {
                algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
                solver_order: 3,
                solver_type: DpmSolverType::Midpoint,
                lower_order_final: false,
                thresholding: false,
                dynamic_thresholding_ratio: 0.995,
                sample_max_value: 1.0,
            },
            train_timesteps: vec![3, 2, 1],
            alpha_t: vec![0.95, 0.85, 0.75, 0.65],
            sigma_t: vec![0.10, 0.20, 0.30, 0.40],
            lambda_t: vec![
                lambda(0.95, 0.10),
                lambda(0.85, 0.20),
                lambda(0.75, 0.30),
                lambda(0.65, 0.40),
            ],
        };
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![1.0],
        };
        let mut state = SchedulerStepState::default();

        schedule.step(&mut latents, &[0.20], 0, &mut state).unwrap();
        schedule.step(&mut latents, &[0.40], 1, &mut state).unwrap();
        let second = latents.data[0];
        schedule.step(&mut latents, &[0.70], 2, &mut state).unwrap();

        assert_eq!(state.lower_order_nums, 3);
        assert_eq!(state.model_outputs.len(), 3);
        assert!(latents.data[0].is_finite());
        assert_ne!(latents.data[0], second);
    }

    #[test]
    fn scheduler_config_falls_back_to_linear_when_beta_metadata_is_missing() {
        let schedule = DiffusionSchedule::from_config(&SchedulerConfig::default(), 2).unwrap();

        assert_eq!(schedule.timesteps, vec![1.0, 0.0]);
        assert_eq!(schedule.sigmas, vec![1.0, 0.0, 0.0]);
        assert_eq!(schedule.prediction_type, SchedulerPredictionType::Epsilon);
        assert_eq!(schedule.input_scaling, SchedulerInputScaling::None);
    }

    #[test]
    fn denoise_progress_callback_can_interrupt_generation() {
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![0.0],
        };
        let schedule = DiffusionSchedule::from_config(&SchedulerConfig::default(), 2).unwrap();
        let positive = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.0],
        };
        let negative = positive.clone();
        let mut events = Vec::new();

        let error = denoise_latents_with_cfg_progress(
            latents,
            &schedule,
            1.0,
            &positive,
            &negative,
            |sample, _timesteps, _encoder_states, _attention_mask, _sdxl_conditioning| {
                Ok(CpuTensor {
                    shape: sample.shape.clone(),
                    data: vec![0.0; sample.data.len()],
                })
            },
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&mut |progress| {
                events.push(progress);
                Err(DiffusionError::Interrupted("test interrupt".to_string()))
            }),
        )
        .unwrap_err();

        assert!(matches!(error, DiffusionError::Interrupted(_)));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].completed_steps, 1);
        assert_eq!(events[0].total_steps, 2);
    }

    #[test]
    fn scheduler_scales_model_input_for_euler_class() {
        let config = SchedulerConfig {
            class_name: "EulerDiscreteScheduler".into(),
            beta_start: Some(0.0001),
            beta_end: Some(0.02),
            beta_schedule: Some("linear".into()),
            num_train_timesteps: Some(1000),
            prediction_type: Some("epsilon".into()),
            ..SchedulerConfig::default()
        };
        let schedule = DiffusionSchedule::from_config(&config, 1).unwrap();
        let sample = CpuTensor {
            shape: vec![1, 1, 1, 1],
            data: vec![2.0],
        };

        let scaled = schedule.scale_model_input(&sample, 0).unwrap();

        assert_eq!(schedule.input_scaling, SchedulerInputScaling::Sigma);
        assert!(scaled.data[0] < sample.data[0]);
    }

    #[test]
    fn scheduler_scales_initial_latents_for_euler_class() {
        let config = SchedulerConfig {
            class_name: "EulerDiscreteScheduler".into(),
            beta_start: Some(0.0001),
            beta_end: Some(0.02),
            beta_schedule: Some("linear".into()),
            num_train_timesteps: Some(1000),
            prediction_type: Some("epsilon".into()),
            ..SchedulerConfig::default()
        };
        let schedule = DiffusionSchedule::from_config(&config, 2).unwrap();
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            data: vec![1.0, -2.0],
        };
        let sigma = schedule.initial_noise_sigma();

        schedule.scale_initial_latents(&mut latents);

        assert!(sigma > 1.0);
        assert_eq!(latents.data, vec![sigma, -2.0 * sigma]);
    }

    #[test]
    fn scheduler_step_supports_sample_prediction_type() {
        let mut schedule = DiffusionSchedule::linear(1).unwrap();
        schedule.prediction_type = SchedulerPredictionType::Sample;
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![2.0],
        };

        schedule.euler_step(&mut latents, &[1.5], 0).unwrap();

        assert_eq!(latents.data, vec![1.5]);
    }

    #[test]
    fn scheduler_step_supports_v_prediction_type() {
        let mut schedule = DiffusionSchedule::linear(1).unwrap();
        schedule.prediction_type = SchedulerPredictionType::VPrediction;
        let mut latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![2.0],
        };

        schedule.euler_step(&mut latents, &[0.5], 0).unwrap();

        let expected =
            2.0 - scheduler_derivative(2.0, 0.5, 1.0, SchedulerPredictionType::VPrediction);
        assert!((latents.data[0] - expected).abs() < 1e-6);
    }

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
    fn denoise_loop_applies_classifier_free_guidance() {
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            data: vec![1.0, -1.0],
        };
        let schedule = DiffusionSchedule::linear(1).unwrap();
        let positive = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![1.0],
        };
        let negative = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.0],
        };
        let out = denoise_latents_with_cfg(
            latents,
            &schedule,
            2.0,
            &positive,
            &negative,
            // Batch-aware mock: batched CFG fuses the uncond/cond passes into one
            // call with encoder rows [positive; negative], so emit a prediction
            // per batch row (negative row value 0.0 -> [0.25,-0.25], else
            // positive -> [0.75,0.25]). Same per-row predictions as before.
            |_sample, _timesteps, encoder| {
                let rows = encoder.shape[0];
                let mut data = Vec::with_capacity(rows * 2);
                for r in 0..rows {
                    if encoder.data[r] == 0.0 {
                        data.extend_from_slice(&[0.25, -0.25]);
                    } else {
                        data.extend_from_slice(&[0.75, 0.25]);
                    }
                }
                Ok(CpuTensor {
                    shape: vec![rows, 1, 1, 2],
                    data,
                })
            },
        )
        .unwrap();

        assert_eq!(out.data, vec![-0.25, -1.75]);
    }

    #[test]
    fn denoise_loop_skips_negative_prediction_when_cfg_is_identity() {
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            data: vec![1.0, -1.0],
        };
        let schedule = DiffusionSchedule::linear(1).unwrap();
        let positive = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![1.0],
        };
        let negative = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.0],
        };
        let mut calls = 0usize;
        let out = denoise_latents_with_cfg(
            latents,
            &schedule,
            1.0,
            &positive,
            &negative,
            |_sample, _timesteps, encoder| {
                calls += 1;
                assert_eq!(encoder.data, positive.data);
                Ok(CpuTensor {
                    shape: vec![1, 1, 1, 2],
                    data: vec![0.75, 0.25],
                })
            },
        )
        .unwrap();

        assert_eq!(calls, 1);
        assert_eq!(out.batch, 1);
        assert_eq!(out.channels, 1);
        assert_eq!(out.height, 1);
        assert_eq!(out.width, 2);
    }

    #[test]
    fn denoise_loop_uses_scheduler_model_input_scaling() {
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![2.0],
        };
        let mut schedule = DiffusionSchedule::linear(1).unwrap();
        schedule.input_scaling = SchedulerInputScaling::Sigma;
        let positive = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![1.0],
        };
        let negative = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.0],
        };
        let mut seen_sample = None;
        let _ = denoise_latents_with_cfg(
            latents,
            &schedule,
            1.0,
            &positive,
            &negative,
            |sample, _timesteps, _encoder| {
                seen_sample.get_or_insert(sample.data[0]);
                Ok(CpuTensor {
                    shape: vec![1, 1, 1, 1],
                    data: vec![0.0],
                })
            },
        )
        .unwrap();

        assert!((seen_sample.unwrap() - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn denoise_loop_rejects_bad_conditioning_and_noise_shapes() {
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
            data: vec![0.0],
        };
        let schedule = DiffusionSchedule::linear(1).unwrap();
        let positive = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![1.0],
        };
        let negative_bad_batch = CpuTensor {
            shape: vec![2, 1, 1],
            data: vec![0.0, 0.0],
        };
        assert!(denoise_latents_with_cfg(
            latents.clone(),
            &schedule,
            1.0,
            &positive,
            &negative_bad_batch,
            |_sample, _timesteps, _encoder| unreachable!(),
        )
        .is_err());

        let negative = CpuTensor {
            shape: vec![1, 1, 1],
            data: vec![0.0],
        };
        assert!(denoise_latents_with_cfg(
            latents,
            &schedule,
            1.0,
            &positive,
            &negative,
            |_sample, _timesteps, _encoder| Ok(CpuTensor {
                shape: vec![1, 1, 1, 2],
                data: vec![0.0, 0.0],
            }),
        )
        .is_err());
    }

    #[test]
    fn latent_shape_uses_vae_scale_factor() {
        let mut config = StableDiffusionConfig {
            pipeline_class: "StableDiffusionPipeline".into(),
            text_encoder: TextEncoderConfig::default(),
            text_encoder_2: None,
            unet: UnetConfig::default(),
            transformer: None,
            vae: VaeConfig::default(),
            scheduler: SchedulerConfig::default(),
            latent_channels: 4,
            latent_height: Some(64),
            latent_width: Some(64),
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
        let shape = latent_shape_for_request(&config, &request).unwrap();
        assert_eq!(
            shape,
            DiffusionLatentShape {
                batch: 1,
                channels: 4,
                height: 64,
                width: 64
            }
        );

        config.vae_scale_factor = 7;
        assert!(latent_shape_for_request(&config, &request).is_err());
    }

    #[test]
    fn latent_shape_rejects_unet_latents_too_small_for_downsampling_depth() {
        let mut config = StableDiffusionConfig {
            pipeline_class: "StableDiffusionPipeline".into(),
            text_encoder: TextEncoderConfig::default(),
            text_encoder_2: None,
            unet: UnetConfig {
                down_block_types: vec![
                    "CrossAttnDownBlock2D".into(),
                    "CrossAttnDownBlock2D".into(),
                    "CrossAttnDownBlock2D".into(),
                    "DownBlock2D".into(),
                ],
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
        let mut request = DiffusionBatchRequest {
            conditioning: None,
            prompts: vec![DiffusionPrompt {
                prompt: "a".into(),
                negative_prompt: String::new(),
                seed: 1,
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
            steps: 20,
            cfg_scale: 7.0,
            distilled_guidance_scale: None,
            scheduler: "DPM++ 2M".into(),
            subseed_strength: 0.0,
            send_images: true,
            save_images: false,
        };

        let err = latent_shape_for_request(&config, &request).unwrap_err();
        assert!(err
            .to_string()
            .contains("too small for UNet downsampling depth 3"));

        request.width = 64;
        request.height = 64;
        let shape = latent_shape_for_request(&config, &request).unwrap();
        assert_eq!(shape.width, 8);
        assert_eq!(shape.height, 8);

        config.transformer = Some(TransformerDenoiserConfig::default());
        request.width = 8;
        request.height = 8;
        let shape = latent_shape_for_request(&config, &request).unwrap();
        assert_eq!(shape.width, 1);
        assert_eq!(shape.height, 1);
    }

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
    fn latent_patch_tokens_roundtrip_and_zero_pad_extra_width() {
        let latents = LatentBatch {
            batch: 1,
            channels: 2,
            height: 4,
            width: 4,
            data: (0..32).map(|idx| idx as f32).collect(),
        };

        let tokens = latent_batch_to_patch_tokens(&latents, 2, 10).unwrap();

        assert_eq!(tokens.shape, vec![1, 4, 10]);
        assert_eq!(
            &tokens.data[0..8],
            &[0.0, 1.0, 4.0, 5.0, 16.0, 17.0, 20.0, 21.0]
        );
        assert_eq!(&tokens.data[8..10], &[0.0, 0.0]);
        let roundtrip = patch_tokens_to_latent_batch(&tokens, 1, 2, 4, 4, 2).unwrap();
        assert_eq!(roundtrip, latents);
    }

    #[test]
    fn latent_patch_tokens_reject_narrow_token_width() {
        let latents = LatentBatch {
            batch: 1,
            channels: 2,
            height: 4,
            width: 4,
            data: vec![0.0; 32],
        };

        let error = latent_batch_to_patch_tokens(&latents, 2, 7)
            .unwrap_err()
            .to_string();

        assert!(error.contains("token_width 7"));
        assert!(error.contains("patch feature width 8"));
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for UNet down block context test: {error}");
            } else {
                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let (hip_hidden, hip_skips) = block
                    .forward_with_runtime_context(
                        input_for_hip,
                        &time,
                        &encoder,
                        &mut runtime_context,
                    )
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!("skip: ROCm GPU unavailable for UNet mid block context test: {error}");
            } else {
                let mut runtime_context = DiffusionGenerationRuntimeContext::new(
                    DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
                );
                let hip_output = mid_block
                    .forward_with_runtime_context(
                        input_for_hip,
                        &time,
                        &encoder,
                        &mut runtime_context,
                    )
                    .unwrap();
                assert_eq!(runtime_context.rocm_gpu_init_count(), 1);
                assert_eq!(hip_output.shape, output.shape);
                assert!(f32_slices_close(&hip_output.data, &output.data, 1e-5));
            }
        }
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
            (0..r * c)
                .map(|k| 0.1 * ((k as f32 % 5.0) - 2.0))
                .collect()
        };

        // Capture-free tensor builder (so the resnet helper below can take
        // `&mut Vec` and avoid a closure-capture borrow conflict).
        let mk = |name: String, shape: Vec<u32>, data: Vec<f32>| -> HfqMemTensor {
            f32_mem_tensor(&name, &shape, &data)
        };

        // A UnetResnetBlock2D with zeroed time projection (still exercises the
        // resident linear + add_channel_bias) and an optional shortcut.
        let resnet = |v: &mut Vec<HfqMemTensor>,
                      prefix: &str,
                      in_ch: usize,
                      out_ch: usize,
                      shortcut: bool| {
            v.push(mk(format!("{prefix}.norm1.weight"), vec![in_ch as u32], vec![1.0; in_ch]));
            v.push(mk(format!("{prefix}.norm1.bias"), vec![in_ch as u32], vec![0.0; in_ch]));
            v.push(mk(
                format!("{prefix}.conv1.weight"),
                vec![out_ch as u32, in_ch as u32, 3, 3],
                conv3(out_ch, in_ch),
            ));
            v.push(mk(format!("{prefix}.conv1.bias"), vec![out_ch as u32], vec![0.0; out_ch]));
            v.push(mk(
                format!("{prefix}.time_emb_proj.weight"),
                vec![out_ch as u32, TIME_DIM as u32],
                vec![0.0; out_ch * TIME_DIM],
            ));
            v.push(mk(format!("{prefix}.time_emb_proj.bias"), vec![out_ch as u32], vec![0.0; out_ch]));
            v.push(mk(format!("{prefix}.norm2.weight"), vec![out_ch as u32], vec![1.0; out_ch]));
            v.push(mk(format!("{prefix}.norm2.bias"), vec![out_ch as u32], vec![0.0; out_ch]));
            v.push(mk(
                format!("{prefix}.conv2.weight"),
                vec![out_ch as u32, out_ch as u32, 3, 3],
                conv3(out_ch, out_ch),
            ));
            v.push(mk(format!("{prefix}.conv2.bias"), vec![out_ch as u32], vec![0.0; out_ch]));
            if shortcut {
                v.push(mk(
                    format!("{prefix}.conv_shortcut.weight"),
                    vec![out_ch as u32, in_ch as u32, 1, 1],
                    conv1(out_ch, in_ch),
                ));
                v.push(mk(format!("{prefix}.conv_shortcut.bias"), vec![out_ch as u32], vec![0.0; out_ch]));
            }
        };

        let mut tensors: Vec<HfqMemTensor> = Vec::new();
        // conv_in.
        tensors.push(mk("unet/tensors/conv_in.weight".into(), vec![CH as u32, CH as u32, 3, 3], conv3(CH, CH)));
        tensors.push(mk("unet/tensors/conv_in.bias".into(), vec![CH as u32], vec![0.0; CH]));
        // time embedding (dim = TIME_DIM).
        tensors.push(mk(
            "unet/tensors/time_embedding.linear_1.weight".into(),
            vec![TIME_DIM as u32, TIME_DIM as u32],
            conv1(TIME_DIM, TIME_DIM),
        ));
        tensors.push(mk("unet/tensors/time_embedding.linear_1.bias".into(), vec![TIME_DIM as u32], vec![0.0; TIME_DIM]));
        tensors.push(mk(
            "unet/tensors/time_embedding.linear_2.weight".into(),
            vec![TIME_DIM as u32, TIME_DIM as u32],
            conv1(TIME_DIM, TIME_DIM),
        ));
        tensors.push(mk("unet/tensors/time_embedding.linear_2.bias".into(), vec![TIME_DIM as u32], vec![0.0; TIME_DIM]));

        // Down block 0: one resnet, no attention, no downsampler.
        resnet(&mut tensors, "unet/tensors/down_blocks.0.resnets.0", CH, CH, false);

        // Mid block: resnet0 + cross-attention transformer + resnet1.
        resnet(&mut tensors, "unet/tensors/mid_block.resnets.0", CH, CH, false);
        let attn = "unet/tensors/mid_block.attentions.0";
        tensors.push(mk(format!("{attn}.norm.weight"), vec![CH as u32], vec![1.0; CH]));
        tensors.push(mk(format!("{attn}.norm.bias"), vec![CH as u32], vec![0.0; CH]));
        tensors.push(mk(format!("{attn}.proj_in.weight"), vec![CH as u32, CH as u32, 1, 1], conv1(CH, CH)));
        tensors.push(mk(format!("{attn}.proj_in.bias"), vec![CH as u32], vec![0.0; CH]));
        let tb = format!("{attn}.transformer_blocks.0");
        tensors.push(mk(format!("{tb}.norm1.weight"), vec![CH as u32], vec![1.0; CH]));
        tensors.push(mk(format!("{tb}.norm1.bias"), vec![CH as u32], vec![0.0; CH]));
        tensors.push(mk(format!("{tb}.attn1.to_q.weight"), vec![CH as u32, CH as u32], mat(CH, CH)));
        tensors.push(mk(format!("{tb}.attn1.to_k.weight"), vec![CH as u32, CH as u32], mat(CH, CH)));
        tensors.push(mk(format!("{tb}.attn1.to_v.weight"), vec![CH as u32, CH as u32], mat(CH, CH)));
        tensors.push(mk(format!("{tb}.attn1.to_out.0.weight"), vec![CH as u32, CH as u32], mat(CH, CH)));
        tensors.push(mk(format!("{tb}.attn1.to_out.0.bias"), vec![CH as u32], vec![0.0; CH]));
        tensors.push(mk(format!("{tb}.norm2.weight"), vec![CH as u32], vec![1.0; CH]));
        tensors.push(mk(format!("{tb}.norm2.bias"), vec![CH as u32], vec![0.0; CH]));
        tensors.push(mk(format!("{tb}.attn2.to_q.weight"), vec![CH as u32, CH as u32], mat(CH, CH)));
        tensors.push(mk(format!("{tb}.attn2.to_k.weight"), vec![CH as u32, CROSS as u32], mat(CH, CROSS)));
        tensors.push(mk(format!("{tb}.attn2.to_v.weight"), vec![CH as u32, CROSS as u32], mat(CH, CROSS)));
        tensors.push(mk(format!("{tb}.attn2.to_out.0.weight"), vec![CH as u32, CH as u32], mat(CH, CH)));
        tensors.push(mk(format!("{tb}.attn2.to_out.0.bias"), vec![CH as u32], vec![0.0; CH]));
        tensors.push(mk(format!("{tb}.norm3.weight"), vec![CH as u32], vec![1.0; CH]));
        tensors.push(mk(format!("{tb}.norm3.bias"), vec![CH as u32], vec![0.0; CH]));
        tensors.push(mk(
            format!("{tb}.ff.net.0.proj.weight"),
            vec![(2 * INNER) as u32, CH as u32],
            mat(2 * INNER, CH),
        ));
        tensors.push(mk(format!("{tb}.ff.net.0.proj.bias"), vec![(2 * INNER) as u32], vec![0.0; 2 * INNER]));
        tensors.push(mk(format!("{tb}.ff.net.2.weight"), vec![CH as u32, INNER as u32], mat(CH, INNER)));
        tensors.push(mk(format!("{tb}.ff.net.2.bias"), vec![CH as u32], vec![0.0; CH]));
        tensors.push(mk(format!("{attn}.proj_out.weight"), vec![CH as u32, CH as u32, 1, 1], conv1(CH, CH)));
        tensors.push(mk(format!("{attn}.proj_out.bias"), vec![CH as u32], vec![0.0; CH]));
        resnet(&mut tensors, "unet/tensors/mid_block.resnets.1", CH, CH, false);

        // Up block 0: two resnets (consume the two skips), each concatenating a
        // skip onto the channel axis (in = 2*CH), with a shortcut. No upsampler.
        resnet(&mut tensors, "unet/tensors/up_blocks.0.resnets.0", 2 * CH, CH, true);
        resnet(&mut tensors, "unet/tensors/up_blocks.0.resnets.1", 2 * CH, CH, true);

        tensors.push(mk("unet/tensors/conv_norm_out.weight".into(), vec![CH as u32], vec![1.0; CH]));
        tensors.push(mk("unet/tensors/conv_norm_out.bias".into(), vec![CH as u32], vec![0.0; CH]));
        tensors.push(mk("unet/tensors/conv_out.weight".into(), vec![CH as u32, CH as u32, 3, 3], conv3(CH, CH)));
        tensors.push(mk("unet/tensors/conv_out.bias".into(), vec![CH as u32], vec![0.0; CH]));

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

        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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

    /// Phase 3: the im2col + WMMA-GEMM conv must match the F32 direct-conv CPU
    /// reference to F16 tolerance, across a 3x3 stride-1 pad-1 conv (batch 2, so
    /// the per-batch GEMM offset logic is exercised) and a 1x1 conv (the
    /// post_quant/proj/shortcut shape, K = in_channels).
    #[test]
    fn wmma_conv2d_resident_matches_cpu_reference_to_f16_tolerance() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for WMMA conv test: {error}");
                return;
            }
        };
        if !gpu.arch_caps.has_wmma_w32() {
            eprintln!("skip: device has no wave32 WMMA; WMMA conv falls back to direct conv");
            return;
        }

        // Deterministic small finite tensor filler in [-1, 1].
        let fill = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|k| (((k as f32 + seed) % 13.0) - 6.0) / 6.0)
                .collect()
        };

        // case = (batch, in_ch, ih, iw, out_ch, kh, kw, padding, stride)
        let cases = [
            (2usize, 4usize, 5usize, 5usize, 6usize, 3usize, 3usize, 1usize, 1usize),
            (1, 8, 4, 4, 8, 1, 1, 0, 1),
            (2, 3, 6, 6, 5, 3, 3, 1, 2),
        ];
        for (case_idx, (b, ic, ih, iw, oc, kh, kw, pad, stride)) in cases.into_iter().enumerate() {
            let input = CpuTensor {
                shape: vec![b, ic, ih, iw],
                data: fill(b * ic * ih * iw, case_idx as f32 * 7.0 + 1.0),
            };
            let weight = CpuTensor {
                shape: vec![oc, ic, kh, kw],
                data: fill(oc * ic * kh * kw, case_idx as f32 * 3.0 + 2.0),
            };
            let bias = CpuTensor {
                shape: vec![oc],
                data: fill(oc, case_idx as f32 + 0.5),
            };
            let cpu = conv2d_nchw_with_stride(&input, &weight, Some(&bias), pad, stride).unwrap();

            let mut cache = RocmWeightCache::default();
            let input_gpu = gpu.upload_f32(&input.data, &input.shape).unwrap();
            let out_gpu = conv2d_nchw_wmma_resident(
                &mut gpu,
                &mut cache,
                &input_gpu,
                &weight,
                Some(&bias),
                pad,
                stride,
            )
            .unwrap();
            let hip = download_resident(&mut gpu, &out_gpu).unwrap();
            free_resident(&mut gpu, out_gpu).unwrap();
            free_resident(&mut gpu, input_gpu).unwrap();

            assert_eq!(hip.shape, cpu.shape, "case {case_idx} shape");
            // F16 inputs (F32 accumulate): tolerance scales with output magnitude.
            let max_mag = cpu
                .data
                .iter()
                .fold(0.0f32, |acc, value| acc.max(value.abs()))
                .max(1.0);
            let tol = 1e-2 * max_mag;
            for (i, (h, c)) in hip.data.iter().zip(cpu.data.iter()).enumerate() {
                assert!(
                    (h - c).abs() <= tol,
                    "case {case_idx} elem {i}: wmma {h} vs cpu {c} (tol {tol})"
                );
            }
        }
    }

    /// Phase 3: the WMMA linear (`linear_optional_bias_resident`) must match the
    /// F32 CPU reference to F16 tolerance, across 2D and 3D inputs and with/without
    /// bias. This isolates the op the chain tests exercise only indirectly.
    #[test]
    fn wmma_linear_resident_matches_cpu_reference_to_f16_tolerance() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for WMMA linear test: {error}");
                return;
            }
        };
        if !gpu.arch_caps.has_wmma_w32() {
            eprintln!("skip: device has no wave32 WMMA; linear falls back to naive path");
            return;
        }
        let fill = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|k| (((k as f32 + seed) % 11.0) - 5.0) / 5.0)
                .collect()
        };
        // (input_shape, in_features, out_features, with_bias)
        let cases: [(Vec<usize>, usize, usize, bool); 4] = [
            (vec![20, 16], 16, 24, true),
            (vec![20, 16], 16, 24, false),
            (vec![2, 10, 32], 32, 48, true),
            (vec![3, 7, 48], 48, 16, true),
        ];
        for (idx, (in_shape, in_f, out_f, with_bias)) in cases.into_iter().enumerate() {
            let total: usize = in_shape.iter().product();
            let input = CpuTensor {
                shape: in_shape.clone(),
                data: fill(total, idx as f32 * 5.0 + 1.0),
            };
            let weight = CpuTensor {
                shape: vec![out_f, in_f],
                data: fill(out_f * in_f, idx as f32 * 3.0 + 2.0),
            };
            let bias = CpuTensor {
                shape: vec![out_f],
                data: fill(out_f, idx as f32 + 0.5),
            };
            let bias_ref = if with_bias { Some(&bias) } else { None };
            // CPU reference works on 2D [rows, in]; the resident op accepts N-D and
            // flattens internally, so compare flat data against a flattened ref.
            let flat_input = CpuTensor {
                shape: vec![total / in_f, in_f],
                data: input.data.clone(),
            };
            let cpu = linear_optional_bias(&flat_input, &weight, bias_ref).unwrap();
            let mut expected_shape = in_shape.clone();
            *expected_shape.last_mut().unwrap() = out_f;

            let mut cache = RocmWeightCache::default();
            let input_gpu = gpu.upload_f32(&input.data, &input.shape).unwrap();
            let out_gpu =
                linear_optional_bias_resident(&mut gpu, &mut cache, &input_gpu, &weight, bias_ref)
                    .unwrap();
            let hip = download_resident(&mut gpu, &out_gpu).unwrap();
            free_resident(&mut gpu, out_gpu).unwrap();
            free_resident(&mut gpu, input_gpu).unwrap();

            assert_eq!(hip.shape, expected_shape, "case {idx} shape");
            assert_eq!(hip.data.len(), cpu.data.len(), "case {idx} len");
            let max_mag = cpu
                .data
                .iter()
                .fold(0.0f32, |acc, value| acc.max(value.abs()))
                .max(1.0);
            let tol = 1e-2 * max_mag;
            for (i, (h, c)) in hip.data.iter().zip(cpu.data.iter()).enumerate() {
                assert!(
                    (h - c).abs() <= tol,
                    "case {idx} elem {i}: wmma {h} vs cpu {c} (tol {tol})"
                );
            }
        }
    }

    /// Phase 3: the flash-attention kernel (online softmax, no seq×seq matrix)
    /// must match the naive SDPA / CPU reference. F32 throughout, so the tolerance
    /// is tight. Covers self-attn, cross-attn (q_seq != k_seq), a head_dim that is
    /// not a multiple of the wave width, and a single-head VAE-style shape.
    #[test]
    fn flash_attention_resident_matches_cpu_reference() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
            (2, 2, 24, 10, 5),                            // cross-attn, q_seq != k_seq
            (1, 4, 20, 12, 12),                           // head_dim 20 (< wave width)
            (2, 1, 64, 9, 9),                             // single head, head_dim 64
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
                scaled_dot_product_attention_resident(&mut gpu, &q_gpu, &k_gpu, &v_gpu, heads)
                    .unwrap();
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
        let stride = [
            (i * h * w) as i64,
            1,
            (i * w) as i64,
            i as i64,
        ];
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
            assert!(im.iter_f32().all(|v| v >= 0.0), "{}: negative imatrix", im.name);
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
            data: (0..4 * 32 * 32).map(|i| (0.1 * ((i % 97) as f32)).sin()).collect(),
        };
        let enc = CpuTensor {
            shape: vec![1, 77, 768],
            data: (0..77 * 768).map(|i| (0.1 * ((i % 89) as f32)).cos()).collect(),
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
                let a = CpuTensor::from_hfq(&src, name).unwrap().data;
                let b = CpuTensor::from_hfq(&cand, name).unwrap().data;
                for (x, y) in a.iter().zip(b.iter()) {
                    sig += (*x as f64) * (*x as f64);
                    noise += ((*x - *y) as f64) * ((*x - *y) as f64);
                }
            }
            let sqnr = if noise > 0.0 { 10.0 * (sig / noise).log10() } else { f64::INFINITY };
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
    fn oq4_oq8_round_trip_through_diffusion_decoder() {
        // Encode with the hipfire-quantize oq codecs, decode with the diffusion
        // CPU decoders. Guards that the diffusion decode (incl. inverse FWHT with
        // the regenerated deterministic sign vectors) matches the encoder layout.
        let signs1 = hipfire_quantize::gen_fwht_signs(OQ_FWHT_SEED1, 256);
        let signs2 = hipfire_quantize::gen_fwht_signs(OQ_FWHT_SEED2, 256);
        let data: Vec<f32> = (0..512)
            .map(|i| ((i as f32 - 256.0) * 0.013).sin() * (1.0 + (i % 13) as f32 * 0.2))
            .collect();
        let sqnr = |orig: &[f32], rec: &[f32]| {
            let (mut s, mut e) = (0.0f64, 0.0f64);
            for (x, y) in orig.iter().zip(rec) {
                s += (*x as f64) * (*x as f64);
                e += ((*x - *y) as f64) * ((*x - *y) as f64);
            }
            10.0 * (s / e).log10()
        };

        let oq4 = hipfire_quantize::codecs::quantize_oq4g256(&data, &signs1, &signs2);
        assert_eq!(oq4.len(), data.len().div_ceil(256) * 130);
        let dec4 = decode_oq4g256_slice("t", &oq4, data.len()).unwrap();
        let s4 = sqnr(&data, &dec4);
        assert!(s4 > 15.0, "oq4 round-trip SQNR too low ({s4:.1} dB) — layout mismatch?");

        let oq8 = hipfire_quantize::codecs::quantize_oq8g256(&data, &signs1, &signs2);
        assert_eq!(oq8.len(), data.len().div_ceil(256) * 258);
        let dec8 = decode_oq8g256_slice("t", &oq8, data.len()).unwrap();
        let s8 = sqnr(&data, &dec8);
        assert!(s8 > 30.0, "oq8 round-trip SQNR too low ({s8:.1} dB)");
        assert!(s8 > s4, "oq8 ({s8:.1}) should beat oq4 ({s4:.1})");
    }

    #[test]
    fn q4k_encoder_round_trips_through_diffusion_decoder() {
        // The Q4_K encoder is ported from hipfire-quantize but the decoder is
        // hipfire_runtime::quant::dequantize_q4_k (a different crate) — this guards
        // that their byte layouts agree, otherwise reused Q4_K weights are garbage.
        let data: Vec<f32> = (0..512)
            .map(|i| ((i as f32 - 256.0) * 0.011).sin() * (1.0 + (i % 11) as f32 * 0.3))
            .collect();
        let bytes = encode_q4k(&data);
        assert_eq!(bytes.len(), data.len().div_ceil(256) * 144);
        let decoded = decode_q4_k_slice("t", &bytes, data.len()).unwrap();
        let mut sig = 0.0f64;
        let mut noise = 0.0f64;
        for (x, y) in data.iter().zip(decoded.iter()) {
            sig += (*x as f64) * (*x as f64);
            noise += ((*x - *y) as f64) * ((*x - *y) as f64);
        }
        let sqnr = 10.0 * (sig / noise).log10();
        // A correctly-laid-out 4-bit k-quant lands ~20+ dB on this data; a layout
        // mismatch would be near 0 dB (uncorrelated). 15 dB cleanly separates them.
        assert!(sqnr > 15.0, "Q4_K round-trip SQNR too low ({sqnr:.1} dB) — layout mismatch?");
    }

    #[test]
    fn q8f16_encoder_round_trips_through_decoder() {
        // Mixed-magnitude data spanning >1 group (32) with negatives and zeros.
        let data: Vec<f32> = (0..100)
            .map(|i| ((i as f32 - 50.0) * 0.013).sin() * (1.0 + (i % 7) as f32))
            .collect();
        let bytes = encode_q8f16(&data);
        assert_eq!(bytes.len(), data.len().div_ceil(32) * 34);
        let decoded = decode_q8f16_slice("t", &bytes, data.len()).unwrap();
        // q8_0 step is max_abs/127; per-group error is bounded by half a step.
        for group in data.chunks(32) {
            let max_abs = group.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let step = (max_abs / 127.0).max(1e-6);
            let base = data.iter().position(|v| (*v - group[0]).abs() < 1e-12).unwrap();
            for (k, &orig) in group.iter().enumerate() {
                assert!((decoded[base + k] - orig).abs() <= step * 0.5 + 1e-4);
            }
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
            let base = data.iter().position(|v| (*v - group[0]).abs() < 1e-12).unwrap();
            for (k, &orig) in group.iter().enumerate() {
                assert!((decoded[base + k] - orig).abs() <= step * 0.5 + 1e-2);
            }
        }
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
    fn rgb_tensor_to_u8_maps_model_range_to_pixels() {
        let tensor = CpuTensor {
            shape: vec![1, 3, 1, 2],
            data: vec![-1.0, 1.0, 0.0, 2.0, -2.0, 0.5],
        };
        let image = rgb_tensor_to_u8(&tensor).unwrap();
        assert_eq!(image.batch, 1);
        assert_eq!(image.width, 2);
        assert_eq!(image.height, 1);
        assert_eq!(image.data, vec![0, 128, 0, 255, 255, 191]);
    }

    #[test]
    fn hip_rgb_tensor_to_u8_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for VAE boundary kernel parity test: {error}"
                );
                return;
            }
        };
        let image = RgbImageBatch {
            batch: 2,
            width: 2,
            height: 2,
            data: vec![
                0, 128, 255, 255, 0, 128, 32, 64, 96, 192, 224, 16, 10, 20, 30, 40, 50, 60, 70, 80,
                90, 100, 110, 120,
            ],
        };
        let cpu_tensor = rgb_batch_to_vae_tensor(&image).unwrap();
        let hip_tensor = rgb_batch_to_vae_tensor_hip_on_gpu(&mut gpu, &image).unwrap();

        assert_eq!(hip_tensor.shape, cpu_tensor.shape);
        for (index, (actual, expected)) in hip_tensor.data.iter().zip(&cpu_tensor.data).enumerate()
        {
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
        for (index, (actual, expected)) in
            hip_latents.data.iter().zip(&cpu_latents.data).enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "VAE moments-to-latents mismatch at {index}: hip={actual} cpu={expected}"
            );
        }
    }

    #[test]
    fn hip_inpaint_mask_ops_match_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for inpaint mask kernel parity test: {error}"
                );
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
        let hip_guided =
            cfg_guidance_hip_on_gpu(&mut gpu, &negative, &positive, cfg_scale).unwrap();
        assert!(f32_slices_close(&hip_guided, &cpu_guided.data, 1e-6));
    }

    #[test]
    fn hip_denoise_loop_runtime_options_route_vector_stages_when_gpu_is_available() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(
            DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
        );

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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for centered UNet input parity test: {error}"
                );
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for timestep embedding parity test: {error}");
                return;
            }
        };
        let timesteps = [999.0, 500.5, 0.25];
        for (dim, flip_sin_to_cos, freq_shift) in [(7, true, 1.0), (6, false, 0.0), (1, true, 0.0)]
        {
            let cpu = timestep_embedding(&timesteps, dim, flip_sin_to_cos, freq_shift).unwrap();
            let hip = timestep_embedding_hip_on_gpu(
                &mut gpu,
                &timesteps,
                dim,
                flip_sin_to_cos,
                freq_shift,
            )
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
            let hip =
                conv2d_nchw_hip_on_gpu(&mut gpu, &mut RocmWeightCache::default(), &input, &weight, bias, 1, 2)
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
    fn hip_tensor_add_matches_cpu_reference_when_gpu_is_available() {
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for channel-bias kernel parity test: {error}"
                );
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for channel-concat kernel parity test: {error}"
                );
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for last-dim concat kernel parity test: {error}"
                );
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
            let hip =
                linear_optional_bias_hip_on_gpu(&mut gpu, &mut RocmWeightCache::default(), &input, &weight, bias)
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for softmax kernel parity test: {error}");
                return;
            }
        };
        let input = CpuTensor {
            shape: vec![4, 5],
            data: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, -3.0, -1.0, -0.5, 0.25, 2.5, 10.0, 9.5, 8.0, 7.25, 6.0,
                100.0, 99.0, 98.0, 97.0, 96.0,
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: ROCm GPU unavailable for SDPA kernel parity test: {error}");
                return;
            }
        };
        let q = CpuTensor {
            shape: vec![2, 2, 4],
            data: vec![
                0.2, -0.4, 0.6, 1.0, -1.2, 0.3, 0.5, -0.7, 1.0, 0.0, -0.5, 0.25, -0.25, 0.75, -1.0,
                0.5,
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!(
                    "skip: ROCm GPU unavailable for CLIP causal attention parity test: {error}"
                );
                return;
            }
        };
        let q = CpuTensor {
            shape: vec![4, 4],
            data: vec![
                0.2, -0.4, 0.6, 1.0, -1.2, 0.3, 0.5, -0.7, 1.0, 0.0, -0.5, 0.25, -0.25, 0.75, -1.0,
                0.5,
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
                0.5, -1.0, 0.25, 0.75, -0.4, 0.6, -0.8, 1.2, 1.0, 0.2, -0.5, -0.1, -0.9, 0.3, 0.8,
                -0.2,
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
    fn rgb_batch_to_vae_tensor_maps_pixels_to_model_range() {
        let image = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 1,
            data: vec![0, 128, 255, 255, 0, 128],
        };

        let tensor = rgb_batch_to_vae_tensor(&image).unwrap();

        assert_eq!(tensor.shape, vec![1, 3, 1, 2]);
        assert!((tensor.data[nchw_idx(0, 0, 0, 0, 3, 1, 2)] + 1.0).abs() < 1e-6);
        assert!((tensor.data[nchw_idx(0, 1, 0, 0, 3, 1, 2)] - 0.003921628).abs() < 1e-6);
        assert!((tensor.data[nchw_idx(0, 2, 0, 0, 3, 1, 2)] - 1.0).abs() < 1e-6);
        assert!((tensor.data[nchw_idx(0, 0, 0, 1, 3, 1, 2)] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vae_moments_to_latents_selects_mean_channels_and_scales() {
        let moments = CpuTensor {
            shape: vec![1, 4, 1, 2],
            data: vec![1.0, -2.0, 3.0, -4.0, 10.0, 20.0, 30.0, 40.0],
        };

        let latents = vae_moments_to_latents(&moments, &VaeLatentNorm::scalar(0.5)).unwrap();

        assert_eq!(latents.batch, 1);
        assert_eq!(latents.channels, 2);
        assert_eq!(latents.height, 1);
        assert_eq!(latents.width, 2);
        assert_eq!(latents.data, vec![0.5, -1.0, 1.5, -2.0]);
    }

    #[test]
    fn vae_per_channel_norm_overrides_scalar_scaling() {
        // AutoencoderKLQwenImage publishes per-channel latents_mean/std and no
        // scaling_factor. The per-channel statistics must take precedence over the
        // legacy 0.18215 default rather than being silently ignored.
        let config = VaeConfig {
            class_name: "AutoencoderKLQwenImage".into(),
            latent_channels: None,
            z_dim: Some(2),
            scaling_factor: None,
            shift_factor: None,
            latents_mean: vec![1.0, -2.0],
            latents_std: vec![2.0, 4.0],
            block_out_channels: Vec::new(),
            down_block_types: Vec::new(),
            up_block_types: Vec::new(),
            norm_num_groups: None,
            norm_eps: None,
        };
        let norm = VaeLatentNorm::from_config(&config).unwrap();
        assert!(norm.is_per_channel());
        assert!(!norm.is_scalar_scale_only());

        // Two channels, 1x2 spatial: encode applies (z - mean[c]) / std[c].
        let moments = CpuTensor {
            shape: vec![1, 4, 1, 2],
            data: vec![3.0, 5.0, 2.0, 6.0, 100.0, 200.0, 300.0, 400.0],
        };
        let latents = vae_moments_to_latents(&moments, &norm).unwrap();
        assert_eq!(latents.channels, 2);
        // channel 0: (3-1)/2=1, (5-1)/2=2 ; channel 1: (2-(-2))/4=1, (6-(-2))/4=2
        assert_eq!(latents.data, vec![1.0, 2.0, 1.0, 2.0]);

        // Decode inverts encode (z * std + mean) exactly.
        let mut roundtrip = latents.data.clone();
        norm.apply_decode(&mut roundtrip, 2, 2).unwrap();
        assert_eq!(roundtrip, vec![3.0, 5.0, 2.0, 6.0]);
    }

    #[test]
    fn vae_stochastic_encode_samples_distribution_deterministically() {
        // 1 batch, 1 latent channel, 1x2 spatial. Channels: [mean(0), logvar(1)].
        // logvar = 0 -> std = 1, so sample = mean + N(0,1) noise.
        let moments = CpuTensor {
            shape: vec![1, 2, 1, 2],
            data: vec![5.0, -5.0, 0.0, 0.0],
        };
        let norm = VaeLatentNorm::scalar(1.0);

        // Deterministic given the seed.
        let a = vae_moments_to_latents_sampled(&moments, &norm, &[42]).unwrap();
        let b = vae_moments_to_latents_sampled(&moments, &norm, &[42]).unwrap();
        assert_eq!(a.data, b.data);

        // A different seed yields different noise.
        let c = vae_moments_to_latents_sampled(&moments, &norm, &[43]).unwrap();
        assert_ne!(a.data, c.data);

        // Sampling perturbs around the mode rather than returning it exactly.
        let mode = vae_moments_to_latents(&moments, &norm).unwrap();
        assert_eq!(mode.data, vec![5.0, -5.0]);
        assert_ne!(a.data, mode.data);
        // Noise has unit std here, so samples stay in a sane neighborhood of the mean.
        assert!((a.data[0] - 5.0).abs() < 8.0);
        assert!((a.data[1] + 5.0).abs() < 8.0);
    }

    #[test]
    fn vae_encode_seed_salts_decorrelate_streams() {
        let seeds = vec![1_i64, 2, 3];
        let init = vae_encode_seeds(&seeds, VAE_INIT_ENCODE_SEED_SALT);
        let masked = vae_encode_seeds(&seeds, VAE_MASKED_ENCODE_SEED_SALT);
        assert_eq!(init.len(), seeds.len());
        // Distinct salts must not collide with each other or the raw seeds.
        assert_ne!(init, masked);
        assert_ne!(init, seeds);
    }

    #[test]
    fn vae_stochastic_encode_honors_log_variance() {
        // A large negative logvar collapses std toward 0, so the sample tracks the
        // mean almost exactly regardless of the drawn noise.
        let moments = CpuTensor {
            shape: vec![1, 2, 1, 1],
            data: vec![3.0, -60.0],
        };
        let norm = VaeLatentNorm::scalar(1.0);
        let sampled = vae_moments_to_latents_sampled(&moments, &norm, &[7]).unwrap();
        assert!((sampled.data[0] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn vae_scalar_shift_norm_round_trips() {
        // Flux/SD3-class scalar normalization: encode (z - shift) * scaling,
        // decode z / scaling + shift.
        let norm = VaeLatentNorm {
            scaling_factor: 0.5,
            shift_factor: 0.25,
            latents_mean: Vec::new(),
            latents_std: Vec::new(),
        };
        assert!(!norm.is_scalar_scale_only());
        let mut data = vec![1.0_f32, -3.0, 0.25];
        norm.apply_encode(&mut data, 1, 3).unwrap();
        assert_eq!(data, vec![(1.0 - 0.25) * 0.5, (-3.0 - 0.25) * 0.5, 0.0]);
        norm.apply_decode(&mut data, 1, 3).unwrap();
        assert_eq!(data, vec![1.0, -3.0, 0.25]);
    }

    #[test]
    fn rgb_batch_encodes_to_decodeable_png_base64_images() {
        let batch = RgbImageBatch {
            batch: 2,
            width: 1,
            height: 1,
            data: vec![255, 0, 0, 0, 255, 0],
        };

        let images = encode_rgb_batch_png_base64(&batch).unwrap();

        assert_eq!(images.len(), 2);
        for image in images {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(image)
                .unwrap();
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
            let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
            assert_eq!(decoded.dimensions(), (1, 1));
        }
    }

    #[test]
    fn rgb_batch_resize_nearest_preserves_batch_items() {
        let image = RgbImageBatch {
            batch: 2,
            width: 1,
            height: 2,
            data: vec![
                10, 20, 30, //
                40, 50, 60, //
                70, 80, 90, //
                100, 110, 120,
            ],
        };

        let resized = resize_rgb_batch_nearest(&image, 2, 4).unwrap();

        assert_eq!(resized.batch, 2);
        assert_eq!(resized.width, 2);
        assert_eq!(resized.height, 4);
        assert_eq!(
            resized.data,
            vec![
                10, 20, 30, 10, 20, 30, //
                10, 20, 30, 10, 20, 30, //
                40, 50, 60, 40, 50, 60, //
                40, 50, 60, 40, 50, 60, //
                70, 80, 90, 70, 80, 90, //
                70, 80, 90, 70, 80, 90, //
                100, 110, 120, 100, 110, 120, //
                100, 110, 120, 100, 110, 120,
            ]
        );
    }

    #[test]
    fn rgb_batch_resize_to_cover_center_crops_aspect_mismatch() {
        let image = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 4,
            data: vec![
                10, 10, 10, 10, 10, 10, //
                20, 20, 20, 20, 20, 20, //
                30, 30, 30, 30, 30, 30, //
                40, 40, 40, 40, 40, 40, //
            ],
        };

        let resized = resize_rgb_batch_to_cover_nearest(&image, 4, 4).unwrap();

        assert_eq!(resized.batch, 1);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
        assert_eq!(
            resized.data,
            vec![
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
                30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, //
                30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, //
            ]
        );
    }

    #[test]
    fn rgb_batch_resize_to_contain_fill_extends_edges() {
        let image = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 4,
            data: vec![
                10, 10, 10, 11, 11, 11, //
                20, 20, 20, 21, 21, 21, //
                30, 30, 30, 31, 31, 31, //
                40, 40, 40, 41, 41, 41, //
            ],
        };

        let resized = resize_rgb_batch_to_contain_fill_nearest(&image, 4, 4).unwrap();

        assert_eq!(resized.batch, 1);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
        assert_eq!(
            resized.data,
            vec![
                10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, //
                20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, //
                30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, //
                40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, //
            ]
        );
    }

    #[test]
    fn latent_mask_weights_downsample_rgb_luma_to_latent_shape() {
        let mask = RgbImageBatch {
            batch: 1,
            width: 4,
            height: 4,
            data: vec![
                0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, //
                0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, //
                128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64, //
                128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64,
            ],
        };
        let latents = LatentBatch {
            batch: 1,
            channels: 1,
            height: 2,
            width: 2,
            data: vec![0.0; 4],
        };

        let weights = latent_mask_weights_from_rgb_batch(&mask, &latents).unwrap();

        assert_eq!(weights.len(), 4);
        assert_eq!(weights[0], 0.0);
        assert_eq!(weights[1], 1.0);
        assert!((weights[2] - (128.0 / 255.0)).abs() < 1e-6);
        assert!((weights[3] - (64.0 / 255.0)).abs() < 1e-6);
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
    fn blend_latents_with_mask_preserves_black_and_uses_generated_white() {
        let mut generated = LatentBatch {
            batch: 1,
            channels: 2,
            height: 1,
            width: 2,
            data: vec![10.0, 20.0, 30.0, 40.0],
        };
        let init = LatentBatch {
            batch: 1,
            channels: 2,
            height: 1,
            width: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };

        blend_latents_with_mask(&mut generated, &init, &[0.0, 1.0]).unwrap();

        assert_eq!(generated.data, vec![1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn hip_img2img_boundary_helpers_reuse_single_runtime_context() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(
            DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
        );

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
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
    fn hip_preflight_reports_clip_token_position_embedding_probe() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
                    assert!(
                        (*h as i16 - *c as i16).abs() <= 2,
                        "rgb8 pixel {h} vs {c}"
                    );
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
            f32_mem_tensor("vae/tensors/decoder.conv_norm_out.weight", &[2], &[1.0, 1.0]),
            f32_mem_tensor("vae/tensors/decoder.conv_norm_out.bias", &[2], &[0.0, 0.0]),
            f32_mem_tensor("vae/tensors/decoder.conv_out.weight", &[3, 2, 3, 3], &conv_out),
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

        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
    fn tiny_sd_native_vae_decoder_loads_when_import_exists() {
        let path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&path) {
            return;
        }
        let hfq = HfqFile::open_index_only(&path).unwrap();
        let metadata = parse_diffusion_metadata(&hfq.metadata_json).unwrap();
        let config = StableDiffusionConfig::from_hfq(&hfq, &metadata).unwrap();
        let decoder = NativeVaeDecoder::from_hfq(&hfq, &config.vae).unwrap();

        assert_eq!(decoder.conv_in.weight.shape, vec![512, 4, 3, 3]);
        assert_eq!(decoder.up_blocks.len(), 4);
        assert!(decoder.up_blocks[0].upsampler.is_some());
        assert!(decoder.up_blocks[1].upsampler.is_some());
        assert!(decoder.up_blocks[2].upsampler.is_some());
        assert!(decoder.up_blocks[3].upsampler.is_none());
        assert_eq!(decoder.conv_out.weight.shape, vec![3, 128, 3, 3]);
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
        let block =
            UnetResnetBlock2D::from_hfq(&hfq, "unet/tensors/down_blocks.0.resnets.0", 32, 1e-5)
                .unwrap();

        assert_eq!(block.conv1.weight.shape, vec![320, 320, 3, 3]);
        assert_eq!(block.time_emb_proj_weight.shape, vec![320, 1280]);
        assert!(block.shortcut.is_none());
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
                eprintln!(
                    "skip: ROCm GPU unavailable for UNet time embedding routing test: {error}"
                );
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
    fn hip_clip_text_encoder_runtime_context_reuses_single_gpu() {
        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(
            DiffusionGenerationRuntimeOptions::rocm_hybrid(0),
        );
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
                data: (0..n).map(|k| 0.02 * ((k as f32 + seed) % 5.0 - 2.0)).collect(),
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
                data: (0..4 * HIDDEN).map(|idx| (idx as f32 % 9.0 - 4.0) * 0.1).collect(),
            },
            position_embedding: CpuTensor {
                shape: vec![3, HIDDEN],
                data: (0..3 * HIDDEN).map(|idx| (idx as f32 % 5.0 - 2.0) * 0.05).collect(),
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

        if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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
        let mut gpu = match rdna_compute::Gpu::init_with_device(0) {
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
            if let Err(error) = rdna_compute::Gpu::init_with_device(0) {
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

    fn zero_clip_layer(hidden: usize) -> ClipEncoderLayer {
        let square = CpuTensor {
            shape: vec![hidden, hidden],
            data: vec![0.0; hidden * hidden],
        };
        let bias = CpuTensor {
            shape: vec![hidden],
            data: vec![0.0; hidden],
        };
        let norm_weight = CpuTensor {
            shape: vec![hidden],
            data: vec![1.0; hidden],
        };
        let norm_bias = bias.clone();
        ClipEncoderLayer {
            q_proj_weight: square.clone(),
            q_proj_bias: bias.clone(),
            k_proj_weight: square.clone(),
            k_proj_bias: bias.clone(),
            v_proj_weight: square.clone(),
            v_proj_bias: bias.clone(),
            out_proj_weight: square.clone(),
            out_proj_bias: bias.clone(),
            layer_norm1_weight: norm_weight.clone(),
            layer_norm1_bias: norm_bias.clone(),
            fc1_weight: square.clone(),
            fc1_bias: bias.clone(),
            fc2_weight: square,
            fc2_bias: bias,
            layer_norm2_weight: norm_weight,
            layer_norm2_bias: norm_bias,
        }
    }

    fn tiny_rgb_image_batch(batch: usize, width: usize, height: usize) -> RgbImageBatch {
        let mut data = Vec::with_capacity(batch * width * height * 3);
        for batch_idx in 0..batch {
            for y in 0..height {
                for x in 0..width {
                    let red = ((x * 255) / width.max(1)) as u8;
                    let green = ((y * 255) / height.max(1)) as u8;
                    let blue = if batch_idx % 2 == 0 { 32 } else { 96 };
                    data.extend_from_slice(&[red, green, blue]);
                }
            }
        }
        RgbImageBatch {
            batch,
            width,
            height,
            data,
        }
    }

    fn tiny_mask_image_batch(batch: usize, width: usize, height: usize) -> RgbImageBatch {
        let mut data = Vec::with_capacity(batch * width * height * 3);
        for _ in 0..batch {
            for y in 0..height {
                for x in 0..width {
                    let value = if (x + y) % 2 == 0 { 255 } else { 0 };
                    data.extend_from_slice(&[value, value, value]);
                }
            }
        }
        RgbImageBatch {
            batch,
            width,
            height,
            data,
        }
    }

    fn minimal_metadata() -> DiffusionHfqMetadata {
        let mut components = BTreeMap::new();
        components.insert(
            "unet".to_string(),
            DiffusionComponentMetadata {
                class_name: Some("UNet2DConditionModel".into()),
                config_entry: Some("unet/config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        DiffusionHfqMetadata {
            artifact_kind: DIFFUSION_ARTIFACT_KIND.to_string(),
            schema_version: DIFFUSION_SCHEMA_VERSION,
            pipeline: DiffusionPipelineMetadata {
                class_name: "StableDiffusionPipeline".into(),
                source: "/tmp/model".into(),
                model_name: "tiny-sd".into(),
                latent_channels: Some(4),
                latent_height: Some(64),
                latent_width: Some(64),
                supported_widths: vec![512],
                supported_heights: vec![512],
            },
            tokenizer: DiffusionTokenizerMetadata::default(),
            tokenizer_2: None,
            batch: DiffusionBatchMetadata {
                max_batch: 2,
                batched_runtime: true,
            },
            quantization: DiffusionQuantizationMetadata::default(),
            components,
        }
    }

    fn tiny_sd_scheduler_config_for_tests() -> SchedulerConfig {
        SchedulerConfig {
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
        }
    }

    fn tiny_runtime_metadata() -> DiffusionHfqMetadata {
        let mut metadata = minimal_metadata();
        metadata.pipeline.model_name = "tiny-runtime".into();
        metadata.pipeline.latent_channels = Some(1);
        metadata.pipeline.latent_height = Some(2);
        metadata.pipeline.latent_width = Some(2);
        metadata.pipeline.supported_widths = vec![2];
        metadata.pipeline.supported_heights = vec![2];
        metadata.batch.max_batch = 4;
        metadata.components.insert(
            "text_encoder".into(),
            DiffusionComponentMetadata {
                class_name: Some("CLIPTextModel".into()),
                config_entry: Some("text_encoder/config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        metadata.components.insert(
            "vae".into(),
            DiffusionComponentMetadata {
                class_name: Some("AutoencoderKL".into()),
                config_entry: Some("vae/config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        metadata.components.insert(
            "scheduler".into(),
            DiffusionComponentMetadata {
                class_name: Some("EulerDiscreteScheduler".into()),
                config_entry: Some("scheduler/scheduler_config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        metadata
    }

    fn tiny_qwen_transformer_runtime_metadata() -> DiffusionHfqMetadata {
        let mut metadata = tiny_runtime_metadata();
        metadata.pipeline.class_name = "QwenImagePipeline".into();
        metadata.pipeline.model_name = "tiny-qwen-transformer-runtime".into();
        metadata.components.remove("unet");
        metadata.components.insert(
            "transformer".into(),
            DiffusionComponentMetadata {
                class_name: Some("QwenImageTransformer2DModel".into()),
                config_entry: Some("transformer/config.json".into()),
                weight_entries: qwen_tiny_transformer_denoiser_tensors()
                    .iter()
                    .map(|tensor| tensor.name.clone())
                    .collect(),
                tensor_roles: Vec::new(),
            },
        );
        metadata
    }

    fn tiny_runtime_config() -> StableDiffusionConfig {
        StableDiffusionConfig {
            pipeline_class: "StableDiffusionPipeline".into(),
            text_encoder: TextEncoderConfig {
                class_name: "CLIPTextModel".into(),
                hidden_size: Some(2),
                intermediate_size: Some(4),
                num_hidden_layers: Some(0),
                num_attention_heads: Some(1),
                max_position_embeddings: Some(4),
                vocab_size: Some(4),
            },
            text_encoder_2: None,
            unet: UnetConfig {
                class_name: "UNet2DConditionModel".into(),
                sample_size: Some(2),
                in_channels: Some(1),
                out_channels: Some(1),
                cross_attention_dim: Some(2),
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
            },
            transformer: None,
            vae: VaeConfig {
                class_name: "AutoencoderKL".into(),
                latent_channels: Some(1),
                z_dim: None,
                scaling_factor: Some(1.0),
                shift_factor: None,
                latents_mean: Vec::new(),
                latents_std: Vec::new(),
                block_out_channels: vec![1],
                down_block_types: vec!["DownEncoderBlock2D".into()],
                up_block_types: vec!["UpDecoderBlock2D".into()],
                norm_num_groups: Some(1),
                norm_eps: Some(1e-6),
            },
            scheduler: SchedulerConfig::default(),
            latent_channels: 1,
            latent_height: Some(2),
            latent_width: Some(2),
            vae_scale_factor: 1,
        }
    }

    fn tiny_qwen_transformer_runtime_tensors() -> Vec<HfqMemTensor> {
        let mut tensors = tiny_complete_runtime_tensors()
            .into_iter()
            .filter(|tensor| !tensor.name.starts_with("unet/"))
            .collect::<Vec<_>>();
        tensors.push(bytes_mem_tensor(
            "transformer/config.json",
            QT_DIFFUSION_JSON,
            br#"{"_class_name":"QwenImageTransformer2DModel","in_channels":4,"out_channels":1,"patch_size":2,"num_layers":1,"num_attention_heads":1,"attention_head_dim":2,"joint_attention_dim":2}"#,
        ));
        tensors.extend(qwen_tiny_transformer_denoiser_tensors());
        tensors
    }

    fn tiny_complete_runtime_tensors() -> Vec<HfqMemTensor> {
        let identity1 = center_identity_conv(1);
        let mut vae_encoder_conv_in = vec![0.0; 1 * 3 * 3 * 3];
        vae_encoder_conv_in[1 * 3 + 1] = 1.0;
        let mut vae_encoder_conv_out = vec![0.0; 2 * 1 * 3 * 3];
        vae_encoder_conv_out[1 * 3 + 1] = 1.0;
        let down_prefix = "unet/tensors/down_blocks.0.resnets.0";
        let mid0_prefix = "unet/tensors/mid_block.resnets.0";
        let mid1_prefix = "unet/tensors/mid_block.resnets.1";
        let up_prefix = "unet/tensors/up_blocks.0.resnets.0";
        let vae_resnet_prefix = "vae/tensors/decoder.up_blocks.0.resnets.0";
        let vae_encoder_resnet_prefix = "vae/tensors/encoder.down_blocks.0.resnets.0";
        vec![
            bytes_mem_tensor(
                "text_encoder/config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"CLIPTextModel","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"max_position_embeddings":77,"vocab_size":4}"#,
            ),
            bytes_mem_tensor(
                "unet/config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"UNet2DConditionModel","sample_size":2,"in_channels":1,"out_channels":1,"cross_attention_dim":2,"attention_head_dim":[1],"block_out_channels":[1],"down_block_types":["DownBlock2D"],"up_block_types":["UpBlock2D"],"layers_per_block":1,"norm_num_groups":1,"norm_eps":0.00001,"flip_sin_to_cos":true,"freq_shift":0.0}"#,
            ),
            bytes_mem_tensor(
                "vae/config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"AutoencoderKL","latent_channels":1,"scaling_factor":1.0,"block_out_channels":[1],"down_block_types":["DownEncoderBlock2D"],"up_block_types":["UpDecoderBlock2D"],"norm_num_groups":1,"norm_eps":0.000001}"#,
            ),
            bytes_mem_tensor(
                "scheduler/scheduler_config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"EulerDiscreteScheduler"}"#,
            ),
            bytes_mem_tensor(
                "tokenizer/vocab.json",
                QT_DIFFUSION_TOKENIZER,
                br#"{"<|startoftext|>":0,"<|endoftext|>":1,"a</w>":2,"cat</w>":3}"#,
            ),
            bytes_mem_tensor("tokenizer/merges.txt", QT_DIFFUSION_TOKENIZER, b"#version: 0.2\n"),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.embeddings.token_embedding.weight",
                &[4, 2],
                &[0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.embeddings.position_embedding.weight",
                &[77, 2],
                &[0.0; 154],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.final_layer_norm.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.final_layer_norm.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.q_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.q_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.k_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.k_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.v_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.v_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.out_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.out_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm1.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm1.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc1.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc1.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc2.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc2.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm2.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm2.bias",
                &[2],
                &[0.0, 0.0],
            ),
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
            f32_mem_tensor(
                "vae/tensors/encoder.conv_in.weight",
                &[1, 3, 3, 3],
                &vae_encoder_conv_in,
            ),
            f32_mem_tensor("vae/tensors/encoder.conv_in.bias", &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm1.weight"),
                &[1],
                &[1.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm1.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv1.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm2.weight"),
                &[1],
                &[1.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm2.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv2.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor("vae/tensors/encoder.conv_norm_out.weight", &[1], &[1.0]),
            f32_mem_tensor("vae/tensors/encoder.conv_norm_out.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/encoder.conv_out.weight",
                &[2, 1, 3, 3],
                &vae_encoder_conv_out,
            ),
            f32_mem_tensor(
                "vae/tensors/encoder.conv_out.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "vae/tensors/quant_conv.weight",
                &[2, 2, 1, 1],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor("vae/tensors/quant_conv.bias", &[2], &[0.0, 0.0]),
            f32_mem_tensor("vae/tensors/post_quant_conv.weight", &[1, 1, 1, 1], &[1.0]),
            f32_mem_tensor("vae/tensors/post_quant_conv.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/decoder.conv_in.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor("vae/tensors/decoder.conv_in.bias", &[1], &[0.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{vae_resnet_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{vae_resnet_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.conv2.bias"), &[1], &[0.0]),
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
        ]
    }

    struct TestNoiseBackend;

    impl DiffusionNoiseBackend for TestNoiseBackend {
        fn model_input_channels(&self) -> usize {
            1
        }

        fn denoise_latents_with_runtime_context(
            &self,
            mut latents: LatentBatch,
            schedule: &DiffusionSchedule,
            cfg_scale: f32,
            positive_embeddings: &CpuTensor,
            negative_embeddings: &CpuTensor,
            _positive_attention_mask: Option<&CpuTensor>,
            _negative_attention_mask: Option<&CpuTensor>,
            _positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            _negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            _inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
            _masked_reference: Option<&MaskedDenoiseReference<'_>>,
            _runtime_context: &mut DiffusionGenerationRuntimeContext,
            mut progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
        ) -> DiffusionResult<DenoiseLatentsOutput> {
            assert_eq!(schedule.timesteps.len(), 2);
            assert_eq!(cfg_scale, 7.0);
            assert_eq!(positive_embeddings.shape[0], latents.batch);
            assert_eq!(negative_embeddings.shape[0], latents.batch);
            for (idx, value) in latents.data.iter_mut().enumerate() {
                *value = (idx as f32 % 4.0) / 3.0;
            }
            for step in 0..schedule.timesteps.len() {
                if let Some(progress) = progress.as_deref_mut() {
                    progress(DiffusionProgress {
                        completed_steps: step + 1,
                        total_steps: schedule.timesteps.len(),
                        timestep: schedule.timesteps[step].round().max(0.0) as usize,
                        preview_latents: Some(latents.clone()),
                    })?;
                }
            }
            Ok(DenoiseLatentsOutput {
                latents,
                runtime_kind: DiffusionRuntimeKind::CpuSourceReference,
            })
        }
    }

    struct TestSdxlNoiseBackend {
        called: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl DiffusionNoiseBackend for TestSdxlNoiseBackend {
        fn model_input_channels(&self) -> usize {
            1
        }

        fn denoise_latents_with_runtime_context(
            &self,
            latents: LatentBatch,
            schedule: &DiffusionSchedule,
            cfg_scale: f32,
            positive_embeddings: &CpuTensor,
            negative_embeddings: &CpuTensor,
            _positive_attention_mask: Option<&CpuTensor>,
            _negative_attention_mask: Option<&CpuTensor>,
            positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            _inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
            _masked_reference: Option<&MaskedDenoiseReference<'_>>,
            _runtime_context: &mut DiffusionGenerationRuntimeContext,
            _progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
        ) -> DiffusionResult<DenoiseLatentsOutput> {
            assert_eq!(schedule.timesteps.len(), 2);
            assert_eq!(cfg_scale, 7.0);
            assert_eq!(positive_embeddings.shape, vec![1, 4, 4]);
            assert_eq!(negative_embeddings.shape, vec![1, 4, 4]);
            let positive = positive_sdxl_conditioning.expect("positive SDXL conditioning");
            let negative = negative_sdxl_conditioning.expect("negative SDXL conditioning");
            assert_eq!(positive.text_embeds.shape, vec![1, 2]);
            assert_eq!(negative.text_embeds.shape, vec![1, 2]);
            assert_eq!(positive.time_ids.shape, vec![1, 6]);
            assert_eq!(negative.time_ids.shape, vec![1, 6]);
            assert_eq!(
                positive.time_ids.data,
                vec![256.0, 128.0, 4.0, 8.0, 64.0, 32.0]
            );
            assert_eq!(negative.time_ids.data, positive.time_ids.data);
            self.called.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(DenoiseLatentsOutput {
                latents,
                runtime_kind: DiffusionRuntimeKind::CpuSourceReference,
            })
        }
    }

    struct TestInpaintNoiseBackend {
        called: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl DiffusionNoiseBackend for TestInpaintNoiseBackend {
        fn model_input_channels(&self) -> usize {
            3
        }

        fn denoise_latents_with_runtime_context(
            &self,
            latents: LatentBatch,
            schedule: &DiffusionSchedule,
            cfg_scale: f32,
            positive_embeddings: &CpuTensor,
            negative_embeddings: &CpuTensor,
            _positive_attention_mask: Option<&CpuTensor>,
            _negative_attention_mask: Option<&CpuTensor>,
            _positive_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            _negative_sdxl_conditioning: Option<&SdxlDenoiseConditioning<'_>>,
            inpaint_conditioning: Option<&InpaintDenoiseConditioning>,
            _masked_reference: Option<&MaskedDenoiseReference<'_>>,
            _runtime_context: &mut DiffusionGenerationRuntimeContext,
            mut progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
        ) -> DiffusionResult<DenoiseLatentsOutput> {
            assert_eq!(schedule.timesteps.len(), 2);
            assert_eq!(cfg_scale, 7.0);
            assert_eq!(positive_embeddings.shape[0], latents.batch);
            assert_eq!(negative_embeddings.shape[0], latents.batch);
            let conditioning = inpaint_conditioning.expect("inpaint conditioning is required");
            assert_eq!(
                conditioning.mask_weights.len(),
                latents.batch * latents.height * latents.width
            );
            assert_eq!(conditioning.masked_image_latents.batch, latents.batch);
            assert_eq!(conditioning.masked_image_latents.channels, latents.channels);
            assert_eq!(conditioning.masked_image_latents.height, latents.height);
            assert_eq!(conditioning.masked_image_latents.width, latents.width);
            self.called.store(true, std::sync::atomic::Ordering::SeqCst);
            for step in 0..schedule.timesteps.len() {
                if let Some(progress) = progress.as_deref_mut() {
                    progress(DiffusionProgress {
                        completed_steps: step + 1,
                        total_steps: schedule.timesteps.len(),
                        timestep: schedule.timesteps[step].round().max(0.0) as usize,
                        preview_latents: Some(latents.clone()),
                    })?;
                }
            }
            Ok(DenoiseLatentsOutput {
                latents,
                runtime_kind: DiffusionRuntimeKind::CpuSourceReference,
            })
        }
    }

    struct TestImageDecoder;

    impl DiffusionImageDecoder for TestImageDecoder {
        fn decode_to_rgb_tensor(&self, latents: &LatentBatch) -> DiffusionResult<CpuTensor> {
            let mut data = Vec::with_capacity(latents.batch * latents.height * latents.width * 3);
            let image_len = latents.len_per_batch();
            for batch in 0..latents.batch {
                let mut red = Vec::with_capacity(latents.height * latents.width);
                let mut green = Vec::with_capacity(latents.height * latents.width);
                let mut blue = Vec::with_capacity(latents.height * latents.width);
                for pixel in 0..(latents.height * latents.width) {
                    let value = (latents.data[batch * image_len + pixel] * 255.0).round() as u8;
                    red.push(rgb_byte_to_model_value(value));
                    green.push(rgb_byte_to_model_value(255u8.saturating_sub(value)));
                    blue.push(rgb_byte_to_model_value(value / 2));
                }
                data.extend(red);
                data.extend(green);
                data.extend(blue);
            }
            Ok(CpuTensor {
                shape: vec![latents.batch, 3, latents.height, latents.width],
                data,
            })
        }
    }

    fn rgb_byte_to_model_value(value: u8) -> f32 {
        (value as f32) / 127.5 - 1.0
    }

    struct SolidTensorImageDecoder;

    impl DiffusionImageDecoder for SolidTensorImageDecoder {
        fn decode_to_rgb_tensor(&self, latents: &LatentBatch) -> DiffusionResult<CpuTensor> {
            let pixels = latents.batch * latents.height * latents.width;
            let mut data = Vec::with_capacity(pixels * 3);
            let pixels_per_batch = latents.height * latents.width;
            for _ in 0..latents.batch {
                data.extend(std::iter::repeat(rgb_byte_to_model_value(32)).take(pixels_per_batch));
                data.extend(std::iter::repeat(rgb_byte_to_model_value(128)).take(pixels_per_batch));
                data.extend(std::iter::repeat(rgb_byte_to_model_value(224)).take(pixels_per_batch));
            }
            Ok(CpuTensor {
                shape: vec![latents.batch, 3, latents.height, latents.width],
                data,
            })
        }
    }

    impl SolidTensorImageDecoder {
        fn expected_rgb(latents: &LatentBatch) -> RgbImageBatch {
            let pixels = latents.batch * latents.height * latents.width;
            let mut data = Vec::with_capacity(pixels * 3);
            for _ in 0..pixels {
                data.extend_from_slice(&[32, 128, 224]);
            }
            RgbImageBatch {
                batch: latents.batch,
                width: latents.width,
                height: latents.height,
                data,
            }
        }
    }

    fn tiny_txt2img_test_pipeline(decoder: Box<dyn DiffusionImageDecoder>) -> DiffusionPipeline {
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
        DiffusionPipeline {
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
                decoder,
            }),
            native_runtime_error: None,
        }
    }

    fn tiny_inpaint_test_pipeline(
        temp_label: &str,
        decoder: Box<dyn DiffusionImageDecoder>,
    ) -> (
        DiffusionPipeline,
        std::sync::Arc<std::sync::atomic::AtomicBool>,
        PathBuf,
    ) {
        let dir = std::env::temp_dir().join(format!("{temp_label}-{}", std::process::id()));
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
        let hfq = HfqFile::open_index_only(&hfq_path).unwrap();
        let config = tiny_runtime_config();
        let encoder = NativeVaeEncoder::from_hfq(&hfq, &config.vae).unwrap();
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
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pipeline = DiffusionPipeline {
            summary: summarize_hfq(Path::new("/tmp/tiny-inpaint.hfq"), &metadata),
            metadata,
            config,
            tokenizer: Some(tokenizer),
            tokenizer_2: None,
            text_encoder: Some(text_encoder),
            text_encoder_2: None,
            native_runtime: Some(NativeDiffusionRuntime {
                kind: DiffusionRuntimeKind::CpuSourceReference,
                noise: Box::new(TestInpaintNoiseBackend {
                    called: called.clone(),
                }),
                encoder: Some(encoder),
                decoder,
            }),
            native_runtime_error: None,
        };
        (pipeline, called, dir)
    }

    fn f32_mem_tensor(name: &str, shape: &[u32], data: &[f32]) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_DIFFUSION_TENSOR_F32,
            shape: shape.to_vec(),
            group_size: 0,
            data: data
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        }
    }

    fn qwen_tiny_transformer_denoiser_tensors() -> Vec<HfqMemTensor> {
        let block = "transformer/tensors/transformer_blocks.0";
        let attn = format!("{block}.attn");
        let time = "transformer/tensors/time_text_embed.timestep_embedder";
        let identity2 = [1.0, 0.0, 0.0, 1.0];
        let geglu_identity = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let silu_one = silu(1.0);
        let mut modulation_weight = vec![0.0f32; 12 * 2];
        modulation_weight[10 * 2] = silu_one.recip();
        modulation_weight[11 * 2] = silu_one.recip();
        vec![
            f32_mem_tensor(
                "transformer/tensors/img_in.weight",
                &[2, 4],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
            f32_mem_tensor("transformer/tensors/img_in.bias", &[2], &[0.0, 0.0]),
            f32_mem_tensor(
                "transformer/tensors/proj_out.weight",
                &[4, 2],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ),
            f32_mem_tensor("transformer/tensors/proj_out.bias", &[4], &[0.0; 4]),
            f32_mem_tensor(&format!("{time}.linear_1.weight"), &[2, 2], &identity2),
            f32_mem_tensor(&format!("{time}.linear_1.bias"), &[2], &[0.0; 2]),
            f32_mem_tensor(
                &format!("{time}.linear_2.weight"),
                &[2, 2],
                &[silu_one.recip(), 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor(&format!("{time}.linear_2.bias"), &[2], &[0.0; 2]),
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
        ]
    }

    fn assert_f32_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "mismatch at {index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }

    fn rms_norm_heads_reference(
        data: &[f32],
        heads: usize,
        head_dim: usize,
        weight: &[f32],
    ) -> Vec<f32> {
        assert_eq!(weight.len(), head_dim);
        let width = heads * head_dim;
        assert_eq!(data.len() % width, 0);
        let mut out = vec![0.0; data.len()];
        for token in 0..(data.len() / width) {
            let token_base = token * width;
            for head in 0..heads {
                let head_base = token_base + head * head_dim;
                let mut square_sum = 0.0f32;
                for dim in 0..head_dim {
                    let value = data[head_base + dim];
                    square_sum += value * value;
                }
                let inv_rms = (square_sum / head_dim as f32 + 1e-6).sqrt().recip();
                for dim in 0..head_dim {
                    out[head_base + dim] = data[head_base + dim] * inv_rms * weight[dim];
                }
            }
        }
        out
    }

    fn qwen_block_expected_mlp_only(hidden: &CpuTensor) -> Vec<f32> {
        assert_eq!(hidden.shape.as_slice(), &[1, 1, 2]);
        let mean = (hidden.data[0] + hidden.data[1]) * 0.5;
        let var = ((hidden.data[0] - mean).powi(2) + (hidden.data[1] - mean).powi(2)) * 0.5;
        let inv_std = (var + 1e-6).sqrt().recip();
        let norm0 = (hidden.data[0] - mean) * inv_std;
        let norm1 = (hidden.data[1] - mean) * inv_std;
        vec![
            hidden.data[0] + norm0 * gelu(norm0),
            hidden.data[1] + norm1 * gelu(norm1),
        ]
    }

    fn q4f16_g64_mem_tensor(name: &str, shape: &[u32], data: &[f32]) -> HfqMemTensor {
        let mut bytes = Vec::new();
        for group in data.chunks(64) {
            let min = group.iter().copied().fold(f32::INFINITY, f32::min);
            let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = if max > min { (max - min) / 15.0 } else { 1.0 };
            bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
            bytes.extend_from_slice(&f32_to_f16_bits(min).to_le_bytes());
            for idx in 0..32 {
                let lo = group.get(idx).copied().unwrap_or(min);
                let hi = group.get(idx + 32).copied().unwrap_or(min);
                let lo_q = ((lo - min) / scale).round().clamp(0.0, 15.0) as u8;
                let hi_q = ((hi - min) / scale).round().clamp(0.0, 15.0) as u8;
                bytes.push(lo_q | (hi_q << 4));
            }
        }
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_DIFFUSION_TENSOR_Q4F16_G64,
            shape: shape.to_vec(),
            group_size: 64,
            data: bytes,
        }
    }

    fn q4k_mem_tensor(name: &str, shape: &[u32], low_nibbles: &[u8]) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_DIFFUSION_TENSOR_Q4_K,
            shape: shape.to_vec(),
            group_size: 256,
            data: q4k_test_block(low_nibbles),
        }
    }

    fn q4k_test_block(low_nibbles: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; 144];
        bytes[0..2].copy_from_slice(&f32_to_f16_bits(0.25).to_le_bytes());
        bytes[2..4].copy_from_slice(&f32_to_f16_bits(0.0).to_le_bytes());
        bytes[4] = 1;
        bytes[5] = 1;
        for (idx, value) in low_nibbles.iter().copied().take(32).enumerate() {
            bytes[16 + idx] = value.min(15);
        }
        bytes
    }

    fn hfq4_mem_tensor(
        name: &str,
        quant_type: u8,
        shape: &[u32],
        group_size: usize,
        low_nibbles: &[u8],
    ) -> HfqMemTensor {
        let block_bytes = match group_size {
            128 => 72,
            256 => 136,
            _ => panic!("unsupported test HFQ4 group size {group_size}"),
        };
        let mut bytes = vec![0u8; block_bytes];
        bytes[0..4].copy_from_slice(&0.25f32.to_le_bytes());
        bytes[4..8].copy_from_slice(&(-1.0f32).to_le_bytes());
        for idx in 0..(group_size / 2) {
            let lo = low_nibbles.get(idx * 2).copied().unwrap_or(0).min(15);
            let hi = low_nibbles.get(idx * 2 + 1).copied().unwrap_or(0).min(15);
            bytes[8 + idx] = lo | (hi << 4);
        }
        HfqMemTensor {
            name: name.to_string(),
            quant_type,
            shape: shape.to_vec(),
            group_size: group_size as u32,
            data: bytes,
        }
    }

    fn hfq6_mem_tensor(name: &str, shape: &[u32], values: &[u8]) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_DIFFUSION_TENSOR_HFQ6_G256,
            shape: shape.to_vec(),
            group_size: 256,
            data: hfq6_test_block(values),
        }
    }

    fn hfq6_test_block(values: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; 200];
        bytes[0..4].copy_from_slice(&0.25f32.to_le_bytes());
        bytes[4..8].copy_from_slice(&(-1.0f32).to_le_bytes());
        for i in (0..256).step_by(4) {
            let q0 = values.get(i).copied().unwrap_or(0).min(63);
            let q1 = values.get(i + 1).copied().unwrap_or(0).min(63);
            let q2 = values.get(i + 2).copied().unwrap_or(0).min(63);
            let q3 = values.get(i + 3).copied().unwrap_or(0).min(63);
            let offset = 8 + (i / 4) * 3;
            bytes[offset] = q0 | (q1 << 6);
            bytes[offset + 1] = (q1 >> 2) | (q2 << 4);
            bytes[offset + 2] = (q2 >> 4) | (q3 << 2);
        }
        bytes
    }

    fn q8f16_mem_tensor(name: &str, shape: &[u32], data: &[f32]) -> HfqMemTensor {
        let mut bytes = Vec::new();
        for group in data.chunks(32) {
            let max_abs = group.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
            for idx in 0..32 {
                let value = group.get(idx).copied().unwrap_or(0.0);
                let quantized = (value / scale).round().clamp(-128.0, 127.0) as i8;
                bytes.push(quantized as u8);
            }
        }
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_DIFFUSION_TENSOR_Q8F16,
            shape: shape.to_vec(),
            group_size: 32,
            data: bytes,
        }
    }

    fn bytes_mem_tensor(name: &str, quant_type: u8, data: &[u8]) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type,
            shape: vec![data.len() as u32],
            group_size: 0,
            data: data.to_vec(),
        }
    }

    fn write_safetensors_fixture(path: &Path, tensors: &[(&str, &str, &[u64], &[u8])]) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut offset = 0u64;
        for (name, dtype, shape, data) in tensors {
            let end = offset + data.len() as u64;
            header.insert(
                (*name).to_string(),
                json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, end],
                }),
            );
            payload.extend_from_slice(data);
            offset = end;
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        fs::write(path, bytes).unwrap();
    }

    fn write_safetensors_fixture_owned(
        path: &Path,
        tensors: &[(String, String, Vec<u64>, Vec<u8>)],
    ) {
        let borrowed = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    name.as_str(),
                    dtype.as_str(),
                    shape.as_slice(),
                    data.as_slice(),
                )
            })
            .collect::<Vec<_>>();
        write_safetensors_fixture(path, &borrowed);
    }

    fn f32_safetensors_tensor(
        name: &str,
        shape: &[u64],
        data: &[f32],
    ) -> (String, String, Vec<u64>, Vec<u8>) {
        (
            name.to_string(),
            "F32".to_string(),
            shape.to_vec(),
            data.iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }

    fn write_tiny_ldm_unet_safetensors(path: &Path) {
        let identity1 = center_identity_conv(1);
        let mut vae_encoder_conv_in = vec![0.0; 1 * 3 * 3 * 3];
        vae_encoder_conv_in[1 * 3 + 1] = 1.0;
        let mut vae_encoder_conv_out = vec![0.0; 2 * 1 * 3 * 3];
        vae_encoder_conv_out[1 * 3 + 1] = 1.0;
        let mut tensors = vec![
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.0.0.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_safetensors_tensor("model.diffusion_model.input_blocks.0.0.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "model.diffusion_model.time_embed.0.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor("model.diffusion_model.time_embed.0.bias", &[2], &[0.0, 0.0]),
            f32_safetensors_tensor(
                "model.diffusion_model.time_embed.2.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor("model.diffusion_model.time_embed.2.bias", &[2], &[0.0, 0.0]),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.in_layers.0.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.in_layers.2.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.in_layers.2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight",
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.emb_layers.1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.out_layers.0.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.out_layers.0.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.out_layers.3.weight",
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.input_blocks.1.0.out_layers.3.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.in_layers.0.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.in_layers.0.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.in_layers.2.weight",
                &[1, 2, 3, 3],
                &[0.0; 18],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.in_layers.2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.emb_layers.1.weight",
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.emb_layers.1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.out_layers.0.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.out_layers.0.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.out_layers.3.weight",
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.out_layers.3.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
                &[1, 2, 1, 1],
                &[1.0, 0.0],
            ),
            f32_safetensors_tensor(
                "model.diffusion_model.output_blocks.0.0.skip_connection.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor("model.diffusion_model.out.0.weight", &[1], &[1.0]),
            f32_safetensors_tensor("model.diffusion_model.out.0.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "model.diffusion_model.out.2.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_safetensors_tensor("model.diffusion_model.out.2.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.post_quant_conv.weight",
                &[1, 1, 1, 1],
                &[1.0],
            ),
            f32_safetensors_tensor("first_stage_model.post_quant_conv.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.encoder.conv_in.weight",
                &[1, 3, 3, 3],
                &vae_encoder_conv_in,
            ),
            f32_safetensors_tensor("first_stage_model.encoder.conv_in.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.norm1.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.norm1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.conv1.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.conv1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.norm2.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.norm2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.conv2.weight",
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_safetensors_tensor(
                "first_stage_model.encoder.down.0.block.0.conv2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor("first_stage_model.encoder.norm_out.weight", &[1], &[1.0]),
            f32_safetensors_tensor("first_stage_model.encoder.norm_out.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.encoder.conv_out.weight",
                &[2, 1, 3, 3],
                &vae_encoder_conv_out,
            ),
            f32_safetensors_tensor("first_stage_model.encoder.conv_out.bias", &[2], &[0.0, 0.0]),
            f32_safetensors_tensor(
                "first_stage_model.quant_conv.weight",
                &[2, 2, 1, 1],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor("first_stage_model.quant_conv.bias", &[2], &[0.0, 0.0]),
            f32_safetensors_tensor(
                "first_stage_model.decoder.conv_in.weight",
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_safetensors_tensor("first_stage_model.decoder.conv_in.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.norm1.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.norm1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.conv1.weight",
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.conv1.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.norm2.weight",
                &[1],
                &[1.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.norm2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.conv2.weight",
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.up.3.block.0.conv2.bias",
                &[1],
                &[0.0],
            ),
            f32_safetensors_tensor("first_stage_model.decoder.norm_out.weight", &[1], &[1.0]),
            f32_safetensors_tensor("first_stage_model.decoder.norm_out.bias", &[1], &[0.0]),
            f32_safetensors_tensor(
                "first_stage_model.decoder.conv_out.weight",
                &[3, 1, 3, 3],
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            f32_safetensors_tensor(
                "first_stage_model.decoder.conv_out.bias",
                &[3],
                &[0.0, 0.0, 0.0],
            ),
        ];
        push_tiny_ldm_clip_text_encoder_tensors(&mut tensors);
        write_safetensors_fixture_owned(path, &tensors);
    }

    fn push_tiny_ldm_clip_text_encoder_tensors(
        tensors: &mut Vec<(String, String, Vec<u64>, Vec<u8>)>,
    ) {
        let prefix = "cond_stage_model.transformer.text_model";
        tensors.extend([
            f32_safetensors_tensor(
                &format!("{prefix}.embeddings.token_embedding.weight"),
                &[3, 2],
                &[0.0, 0.0, 0.5, -0.5, 1.0, 0.25],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.embeddings.position_embedding.weight"),
                &[77, 2],
                &vec![0.0; 77 * 2],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.q_proj.weight"),
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.q_proj.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.k_proj.weight"),
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.k_proj.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.v_proj.weight"),
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.v_proj.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.out_proj.weight"),
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.self_attn.out_proj.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.layer_norm1.weight"),
                &[2],
                &[1.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.layer_norm1.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.mlp.fc1.weight"),
                &[4, 2],
                &[0.0; 8],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.mlp.fc1.bias"),
                &[4],
                &[0.0; 4],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.mlp.fc2.weight"),
                &[2, 4],
                &[0.0; 8],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.mlp.fc2.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.layer_norm2.weight"),
                &[2],
                &[1.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.encoder.layers.0.layer_norm2.bias"),
                &[2],
                &[0.0, 0.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.final_layer_norm.weight"),
                &[2],
                &[1.0, 1.0],
            ),
            f32_safetensors_tensor(
                &format!("{prefix}.final_layer_norm.bias"),
                &[2],
                &[0.0, 0.0],
            ),
        ]);
    }

    fn center_identity_conv2(channels: usize) -> Vec<f32> {
        center_identity_conv(channels)
    }

    fn center_identity_conv(channels: usize) -> Vec<f32> {
        let mut data = vec![0.0; channels * channels * 3 * 3];
        for channel in 0..channels {
            data[(((channel * channels + channel) * 3 + 1) * 3) + 1] = 1.0;
        }
        data
    }

    fn push_zero_attention_tensors(
        tensors: &mut Vec<HfqMemTensor>,
        prefix: &str,
        hidden: u32,
        context: u32,
    ) {
        tensors.push(f32_mem_tensor(
            &format!("{prefix}.to_q.weight"),
            &[hidden, hidden],
            &vec![0.0; (hidden * hidden) as usize],
        ));
        tensors.push(f32_mem_tensor(
            &format!("{prefix}.to_k.weight"),
            &[hidden, context],
            &vec![0.0; (hidden * context) as usize],
        ));
        tensors.push(f32_mem_tensor(
            &format!("{prefix}.to_v.weight"),
            &[hidden, context],
            &vec![0.0; (hidden * context) as usize],
        ));
        tensors.push(f32_mem_tensor(
            &format!("{prefix}.to_out.0.weight"),
            &[hidden, hidden],
            &vec![0.0; (hidden * hidden) as usize],
        ));
        tensors.push(f32_mem_tensor(
            &format!("{prefix}.to_out.0.bias"),
            &[hidden],
            &vec![0.0; hidden as usize],
        ));
    }
