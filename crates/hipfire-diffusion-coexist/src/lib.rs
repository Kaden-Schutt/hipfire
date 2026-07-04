// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Offline diffusion import/conversion tooling.
//!
//! `import_diffusers_to_hfq` + single-file checkpoint import, the safetensors /
//! PyTorch state-dict parsers, the hand-rolled PyTorch **pickle interpreter**,
//! and a from-scratch **zip reader**. Extracted out of `hipfire-diffusion`
//! (which `hipfire-server` links) so this untrusted-format parsing no longer
//! ships in the serving binary — AGENTS.md mandates conversion tooling live
//! outside the inference/serving crates. Mirrors the `hipfire-gguf` split.
//!
//! Shared HFQ metadata types and `inspect_hfq` stay in `hipfire-diffusion`
//! (the pipeline reads them); this crate consumes them to emit `.hfq` artifacts.

use hipfire_diffusion::{
    inspect_hfq, DiffusionBatchMetadata, DiffusionComponentMetadata, DiffusionHfqMetadata,
    DiffusionModelSummary, DiffusionPipelineMetadata, DiffusionQuantizationMetadata,
    DiffusionTensorRole, DiffusionTokenizerMetadata, DIFFUSION_ARTIFACT_KIND,
    DIFFUSION_SCHEMA_VERSION, HFQ_ARCH_DIFFUSION, QT_DIFFUSION_JSON, QT_DIFFUSION_SOURCE_WEIGHTS,
    QT_DIFFUSION_TENSOR_BF16, QT_DIFFUSION_TENSOR_F16, QT_DIFFUSION_TENSOR_F32,
    QT_DIFFUSION_TOKENIZER,
};
use hipfire_runtime::hfq::{write_hfqm_package_streaming, HfqStreamEntry};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct DiffusersImportOptions {
    pub source: PathBuf,
    pub output: PathBuf,
    pub model_name: Option<String>,
    pub max_batch: u32,
    pub metadata_only: bool,
}

pub fn import_diffusers_to_hfq(
    options: DiffusersImportOptions,
) -> anyhow::Result<DiffusionModelSummary> {
    let source = options.source.canonicalize()?;
    if source.is_file() {
        if options.metadata_only {
            anyhow::bail!("--metadata-only is only supported for Diffusers snapshot directories");
        }
        return import_single_file_checkpoint_to_hfq(
            source,
            options.output,
            options.model_name,
            options.max_batch,
        );
    }
    let output = options.output;
    let model_index = read_json(source.join("model_index.json"))?;
    let class_name = model_index
        .get("_class_name")
        .and_then(Value::as_str)
        .unwrap_or("DiffusionPipeline")
        .to_string();

    let mut entries = Vec::new();
    let mut components = BTreeMap::new();
    let mut tokenizer_entries = Vec::new();
    let mut tokenizer_2_entries = Vec::new();
    let weight_files_enabled = !options.metadata_only;

    push_import_file_entry(
        &mut entries,
        "diffusers/model_index.json",
        QT_DIFFUSION_JSON,
        source.join("model_index.json"),
    )?;
    add_component(
        &source,
        &mut entries,
        &mut components,
        "text_encoder",
        if weight_files_enabled {
            &[
                "model.safetensors",
                "pytorch_model.safetensors",
                "pytorch_model.bin",
            ]
        } else {
            &[]
        },
    )?;
    add_component(
        &source,
        &mut entries,
        &mut components,
        "text_encoder_2",
        if weight_files_enabled {
            &[
                "model.safetensors",
                "pytorch_model.safetensors",
                "pytorch_model.bin",
            ]
        } else {
            &[]
        },
    )?;
    add_component(
        &source,
        &mut entries,
        &mut components,
        "unet",
        if weight_files_enabled {
            &[
                "diffusion_pytorch_model.safetensors",
                "model.safetensors",
                "diffusion_pytorch_model.bin",
            ]
        } else {
            &[]
        },
    )?;
    add_component(
        &source,
        &mut entries,
        &mut components,
        "transformer",
        if weight_files_enabled {
            &[
                "diffusion_pytorch_model.safetensors",
                "model.safetensors",
                "diffusion_pytorch_model.bin",
            ]
        } else {
            &[]
        },
    )?;
    add_component(
        &source,
        &mut entries,
        &mut components,
        "vae",
        if weight_files_enabled {
            &[
                "diffusion_pytorch_model.safetensors",
                "model.safetensors",
                "diffusion_pytorch_model.bin",
            ]
        } else {
            &[]
        },
    )?;
    add_component(&source, &mut entries, &mut components, "scheduler", &[])?;

    // `vocab.json`/`merges.txt` are the CLIP-BPE layout; `tokenizer.json` is the
    // HF fast-tokenizer bundle (Qwen2 and friends pack vocab+merges into it);
    // `chat_template.jinja` carries the prompt wrapping some text encoders need.
    const TOKENIZER_FILES: [&str; 6] = [
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ];
    for name in TOKENIZER_FILES {
        let path = source.join("tokenizer").join(name);
        if path.is_file() {
            let entry_name = format!("tokenizer/{name}");
            push_import_file_entry(&mut entries, &entry_name, QT_DIFFUSION_TOKENIZER, path)?;
            tokenizer_entries.push(entry_name);
        }
    }
    for name in TOKENIZER_FILES {
        let path = source.join("tokenizer_2").join(name);
        if path.is_file() {
            let entry_name = format!("tokenizer_2/{name}");
            push_import_file_entry(&mut entries, &entry_name, QT_DIFFUSION_TOKENIZER, path)?;
            tokenizer_2_entries.push(entry_name);
        }
    }
    let (tokenizer_kind, tokenizer_max_length) = tokenizer_descriptor(&source.join("tokenizer"));
    let (tokenizer_2_kind, tokenizer_2_max_length) =
        tokenizer_descriptor(&source.join("tokenizer_2"));

    let unet_config = read_json(source.join("unet/config.json")).unwrap_or_else(|_| json!({}));
    let transformer_config =
        read_json(source.join("transformer/config.json")).unwrap_or_else(|_| json!({}));
    let vae_config = read_json(source.join("vae/config.json")).unwrap_or_else(|_| json!({}));
    // Latent channels = the VAE latent space (what the scheduler denoises).
    // Prefer the VAE's own channel count: for patchified DiTs the transformer
    // `in_channels` is the patch-flattened width (e.g. Krea2 64 = z_dim 16 x a
    // 2x2 patch), and for inpaint UNets `in_channels` folds in mask/masked-latent
    // concat, so the denoiser input width is not the latent width. Fall back to
    // the denoiser channels only when the VAE config lacks a latent-channel field.
    let latent_channels = vae_config
        .get("latent_channels")
        .and_then(Value::as_u64)
        .or_else(|| vae_config.get("z_dim").and_then(Value::as_u64))
        .or_else(|| unet_config.get("in_channels").and_then(Value::as_u64))
        .or_else(|| {
            transformer_config
                .get("out_channels")
                .and_then(Value::as_u64)
        })
        .or_else(|| {
            transformer_config
                .get("in_channels")
                .and_then(Value::as_u64)
        })
        .map(|value| value as u32);
    let latent_size = unet_config
        .get("sample_size")
        .and_then(Value::as_u64)
        .map(|value| value as u32);
    let model_name = options
        .model_name
        .or_else(|| {
            source
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "diffusion".to_string());
    let max_batch = options.max_batch.max(1);
    let metadata = DiffusionHfqMetadata {
        artifact_kind: DIFFUSION_ARTIFACT_KIND.to_string(),
        schema_version: DIFFUSION_SCHEMA_VERSION,
        pipeline: DiffusionPipelineMetadata {
            class_name,
            source: source.to_string_lossy().into_owned(),
            model_name,
            latent_channels,
            latent_height: latent_size,
            latent_width: latent_size,
            supported_widths: Vec::new(),
            supported_heights: Vec::new(),
        },
        tokenizer: DiffusionTokenizerMetadata {
            kind: tokenizer_kind,
            max_length: tokenizer_max_length,
            entries: tokenizer_entries,
        },
        tokenizer_2: (!tokenizer_2_entries.is_empty()).then_some(DiffusionTokenizerMetadata {
            kind: tokenizer_2_kind,
            max_length: tokenizer_2_max_length,
            entries: tokenizer_2_entries,
        }),
        batch: DiffusionBatchMetadata {
            max_batch,
            batched_runtime: true,
        },
        quantization: if options.metadata_only {
            DiffusionQuantizationMetadata {
                weight_format: "metadata-only".to_string(),
                ..DiffusionQuantizationMetadata::default()
            }
        } else {
            DiffusionQuantizationMetadata::default()
        },
        components,
    };
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    write_import_entries_to_hfq(&output, &metadata_json, &entries)?;
    inspect_hfq(output).map_err(anyhow::Error::from)
}

fn import_single_file_checkpoint_to_hfq(
    source: PathBuf,
    output: PathBuf,
    model_name: Option<String>,
    max_batch: u32,
) -> anyhow::Result<DiffusionModelSummary> {
    let parsed_safetensors = source
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"));
    let tensors = if parsed_safetensors {
        parse_safetensors_state_dict(&source).unwrap_or_default()
    } else {
        Vec::new()
    };
    let mut entries = Vec::new();
    let mut components = BTreeMap::<String, DiffusionComponentMetadata>::new();
    let mut tokenizer_entries = Vec::new();
    let mut tokenizer_2_entries = Vec::new();
    let pipeline_class = infer_single_file_pipeline_class(&tensors);
    let latent_channels = infer_single_file_latent_channels(&tensors);

    if tensors.is_empty() {
        push_import_file_entry(
            &mut entries,
            "checkpoint/source_weights",
            QT_DIFFUSION_SOURCE_WEIGHTS,
            source.clone(),
        )?;
        components.insert(
            "checkpoint".to_string(),
            DiffusionComponentMetadata {
                class_name: Some("SingleFileCheckpoint".to_string()),
                config_entry: None,
                weight_entries: vec!["checkpoint/source_weights".to_string()],
                tensor_roles: Vec::new(),
            },
        );
    } else {
        for tensor in tensors {
            let component = single_file_tensor_component(&tensor.name);
            let entry_name = format!("{component}/checkpoint_tensors/{}", tensor.name);
            let native_entry_name = native_entry_for_single_file_tensor(&tensor.name);
            let metadata = components.entry(component.to_string()).or_insert_with(|| {
                DiffusionComponentMetadata {
                    class_name: Some(single_file_component_class_name(component).to_string()),
                    config_entry: None,
                    weight_entries: Vec::new(),
                    tensor_roles: Vec::new(),
                }
            });
            metadata.tensor_roles.push(DiffusionTensorRole {
                role: tensor.name.clone(),
                entry: entry_name.clone(),
                dtype: tensor.dtype.clone(),
                quant_format: None,
            });
            metadata.weight_entries.push(entry_name.clone());
            if let Some(native_entry_name) = native_entry_name {
                metadata.tensor_roles.push(DiffusionTensorRole {
                    role: tensor.name.clone(),
                    entry: native_entry_name.clone(),
                    dtype: tensor.dtype.clone(),
                    quant_format: None,
                });
                metadata.weight_entries.push(native_entry_name.clone());
                entries.push(DiffusionImportEntry {
                    name: native_entry_name,
                    quant_type: tensor.quant_type,
                    shape: tensor.shape.clone(),
                    group_size: 0,
                    source: DiffusionImportSource::FileSlice {
                        path: tensor.source_path.clone(),
                        offset: tensor.data_offset,
                        len: tensor.data_len,
                    },
                });
            }
            entries.push(DiffusionImportEntry {
                name: entry_name,
                quant_type: tensor.quant_type,
                shape: tensor.shape,
                group_size: 0,
                source: DiffusionImportSource::FileSlice {
                    path: tensor.source_path,
                    offset: tensor.data_offset,
                    len: tensor.data_len,
                },
            });
        }
    }

    add_single_file_generated_configs(&mut entries, &mut components, &pipeline_class)?;
    add_single_file_tokenizer_sidecars(
        &source,
        &mut entries,
        &mut tokenizer_entries,
        &mut tokenizer_2_entries,
    )?;

    let model_name = model_name
        .or_else(|| {
            source
                .file_stem()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "diffusion-checkpoint".to_string());
    let metadata = DiffusionHfqMetadata {
        artifact_kind: DIFFUSION_ARTIFACT_KIND.to_string(),
        schema_version: DIFFUSION_SCHEMA_VERSION,
        pipeline: DiffusionPipelineMetadata {
            class_name: pipeline_class,
            source: source.to_string_lossy().into_owned(),
            model_name,
            latent_channels,
            latent_height: None,
            latent_width: None,
            supported_widths: Vec::new(),
            supported_heights: Vec::new(),
        },
        tokenizer: DiffusionTokenizerMetadata {
            kind: "clip-bpe".to_string(),
            max_length: Some(77),
            entries: tokenizer_entries,
        },
        tokenizer_2: (!tokenizer_2_entries.is_empty()).then_some(DiffusionTokenizerMetadata {
            kind: "clip-bpe".to_string(),
            max_length: Some(77),
            entries: tokenizer_2_entries,
        }),
        batch: DiffusionBatchMetadata {
            max_batch: max_batch.max(1),
            batched_runtime: true,
        },
        quantization: DiffusionQuantizationMetadata::default(),
        components,
    };
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    write_import_entries_to_hfq(&output, &metadata_json, &entries)?;
    inspect_hfq(output).map_err(anyhow::Error::from)
}

fn add_single_file_generated_configs(
    entries: &mut Vec<DiffusionImportEntry>,
    components: &mut BTreeMap<String, DiffusionComponentMetadata>,
    pipeline_class: &str,
) -> anyhow::Result<()> {
    let is_sdxl = pipeline_class == "StableDiffusionXLPipeline";
    if components.contains_key("unet") {
        push_single_file_component_config(
            entries,
            components,
            "unet",
            "config.json",
            if is_sdxl {
                json!({
                    "_class_name": "UNet2DConditionModel",
                    "sample_size": 128,
                    "in_channels": 4,
                    "out_channels": 4,
                    "cross_attention_dim": 2048,
                    "attention_head_dim": [5, 10, 20],
                    "block_out_channels": [320, 640, 1280],
                    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
                    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
                    "layers_per_block": 2,
                    "norm_num_groups": 32,
                    "norm_eps": 1e-5,
                    "center_input_sample": false,
                    "flip_sin_to_cos": true,
                    "freq_shift": 0,
                    "addition_embed_type": "text_time",
                    "addition_time_embed_dim": 256,
                    "projection_class_embeddings_input_dim": 2816
                })
            } else {
                json!({
                    "_class_name": "UNet2DConditionModel",
                    "sample_size": 64,
                    "in_channels": 4,
                    "out_channels": 4,
                    "cross_attention_dim": 768,
                    "attention_head_dim": 8,
                    "block_out_channels": [320, 640, 1280, 1280],
                    "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
                    "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
                    "layers_per_block": 2,
                    "norm_num_groups": 32,
                    "norm_eps": 1e-5,
                    "center_input_sample": false,
                    "flip_sin_to_cos": true,
                    "freq_shift": 0
                })
            },
        )?;
    }
    if components.contains_key("vae") {
        push_single_file_component_config(
            entries,
            components,
            "vae",
            "config.json",
            json!({
                "_class_name": "AutoencoderKL",
                "latent_channels": 4,
                "scaling_factor": 0.18215,
                "block_out_channels": [128, 256, 512, 512],
                "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
                "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
                "norm_num_groups": 32,
                "norm_eps": 1e-6
            }),
        )?;
    }
    if components.contains_key("text_encoder") {
        push_single_file_component_config(
            entries,
            components,
            "text_encoder",
            "config.json",
            json!({
                "_class_name": "CLIPTextModel",
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "max_position_embeddings": 77,
                "vocab_size": 49408
            }),
        )?;
    }
    if components.contains_key("text_encoder_2") {
        push_single_file_component_config(
            entries,
            components,
            "text_encoder_2",
            "config.json",
            json!({
                "_class_name": "CLIPTextModelWithProjection",
                "hidden_size": 1280,
                "intermediate_size": 5120,
                "num_hidden_layers": 32,
                "num_attention_heads": 20,
                "max_position_embeddings": 77,
                "vocab_size": 49408
            }),
        )?;
    }
    push_single_file_component_config(
        entries,
        components,
        "scheduler",
        "scheduler_config.json",
        json!({
            "_class_name": "EulerDiscreteScheduler",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "timestep_spacing": "linspace",
            "steps_offset": 1
        }),
    )?;
    Ok(())
}

fn push_single_file_component_config(
    entries: &mut Vec<DiffusionImportEntry>,
    components: &mut BTreeMap<String, DiffusionComponentMetadata>,
    component: &str,
    config_name: &str,
    config: Value,
) -> anyhow::Result<()> {
    let entry_name = format!("{component}/{config_name}");
    let data = serde_json::to_vec_pretty(&config)?;
    push_import_inline_entry(entries, &entry_name, QT_DIFFUSION_JSON, data);
    let metadata = components.entry(component.to_string()).or_default();
    metadata.class_name = config
        .get("_class_name")
        .and_then(Value::as_str)
        .map(str::to_string);
    metadata.config_entry = Some(entry_name);
    Ok(())
}

fn add_single_file_tokenizer_sidecars(
    source: &Path,
    entries: &mut Vec<DiffusionImportEntry>,
    tokenizer_entries: &mut Vec<String>,
    tokenizer_2_entries: &mut Vec<String>,
) -> anyhow::Result<()> {
    let Some(parent) = source.parent() else {
        return Ok(());
    };
    add_single_file_tokenizer_sidecar(parent, "tokenizer", entries, tokenizer_entries)?;
    add_single_file_tokenizer_sidecar(parent, "tokenizer_2", entries, tokenizer_2_entries)?;
    Ok(())
}

fn add_single_file_tokenizer_sidecar(
    parent: &Path,
    sidecar_dir: &str,
    entries: &mut Vec<DiffusionImportEntry>,
    tokenizer_entries: &mut Vec<String>,
) -> anyhow::Result<()> {
    for name in [
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let path = parent.join(sidecar_dir).join(name);
        if path.is_file() {
            let entry_name = format!("{sidecar_dir}/{name}");
            push_import_file_entry(entries, &entry_name, QT_DIFFUSION_TOKENIZER, path)?;
            tokenizer_entries.push(entry_name);
        }
    }
    Ok(())
}

fn infer_single_file_pipeline_class(tensors: &[SafetensorsTensorEntry]) -> String {
    if tensors.iter().any(|tensor| {
        tensor.name.starts_with("conditioner.embedders.1.")
            || tensor.name.starts_with("conditioner.embedders.0.model.")
    }) {
        "StableDiffusionXLPipeline".to_string()
    } else {
        "StableDiffusionPipeline".to_string()
    }
}

fn infer_single_file_latent_channels(tensors: &[SafetensorsTensorEntry]) -> Option<u32> {
    tensors
        .iter()
        .find(|tensor| {
            (tensor.name == "model.diffusion_model.input_blocks.0.0.weight"
                || tensor.name == "diffusion_model.input_blocks.0.0.weight"
                || tensor.name == "conv_in.weight")
                && tensor.shape.len() == 4
                && tensor.shape[1] > 0
        })
        .map(|tensor| tensor.shape[1])
}

fn single_file_tensor_component(name: &str) -> &'static str {
    if name.starts_with("model.diffusion_model.") || name.starts_with("diffusion_model.") {
        "unet"
    } else if name.starts_with("first_stage_model.") {
        "vae"
    } else if name.starts_with("cond_stage_model.") || name.starts_with("conditioner.embedders.0.")
    {
        "text_encoder"
    } else if name.starts_with("conditioner.embedders.1.") {
        "text_encoder_2"
    } else {
        "checkpoint"
    }
}

fn single_file_component_class_name(component: &str) -> &'static str {
    match component {
        "unet" => "UNet2DConditionModel",
        "vae" => "AutoencoderKL",
        "text_encoder" => "CLIPTextModel",
        "text_encoder_2" => "CLIPTextModelWithProjection",
        _ => "SingleFileCheckpoint",
    }
}

fn native_entry_for_single_file_tensor(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("first_stage_model.") {
        let mapped = ldm_vae_native_tensor_name(rest).unwrap_or_else(|| rest.to_string());
        return Some(format!("vae/tensors/{mapped}"));
    }
    if let Some(rest) = name.strip_prefix("cond_stage_model.transformer.") {
        return Some(format!("text_encoder/tensors/{rest}"));
    }
    if let Some(rest) = name.strip_prefix("conditioner.embedders.0.transformer.") {
        return Some(format!("text_encoder/tensors/{rest}"));
    }
    if let Some(rest) = name.strip_prefix("conditioner.embedders.1.model.") {
        return Some(format!("text_encoder_2/tensors/{rest}"));
    }
    let unet_name = name
        .strip_prefix("model.diffusion_model.")
        .or_else(|| name.strip_prefix("diffusion_model."))?;
    ldm_unet_native_tensor_name(unet_name).map(|mapped| format!("unet/tensors/{mapped}"))
}

pub fn ldm_vae_native_tensor_name(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("encoder.down.") {
        return ldm_vae_encoder_down_tensor_name(rest);
    }
    if let Some(rest) = name.strip_prefix("encoder.mid.") {
        return ldm_vae_mid_tensor_name("encoder", rest);
    }
    if let Some(rest) = name.strip_prefix("decoder.mid.") {
        return ldm_vae_mid_tensor_name("decoder", rest);
    }
    if let Some(rest) = name.strip_prefix("decoder.up.") {
        return ldm_vae_decoder_up_tensor_name(rest);
    }
    let mapped = match name {
        "encoder.norm_out.weight" => "encoder.conv_norm_out.weight",
        "encoder.norm_out.bias" => "encoder.conv_norm_out.bias",
        "decoder.norm_out.weight" => "decoder.conv_norm_out.weight",
        "decoder.norm_out.bias" => "decoder.conv_norm_out.bias",
        _ => return None,
    };
    Some(mapped.to_string())
}

fn ldm_vae_encoder_down_tensor_name(rest: &str) -> Option<String> {
    let (block_idx, rest) = split_usize_prefix(rest)?;
    if let Some(rest) = rest.strip_prefix("block.") {
        let (layer_idx, rest) = split_usize_prefix(rest)?;
        return map_ldm_vae_resnet_suffix(rest)
            .map(|suffix| format!("encoder.down_blocks.{block_idx}.resnets.{layer_idx}.{suffix}"));
    }
    rest.strip_prefix("downsample.conv.")
        .map(|suffix| format!("encoder.down_blocks.{block_idx}.downsamplers.0.conv.{suffix}"))
}

fn ldm_vae_decoder_up_tensor_name(rest: &str) -> Option<String> {
    const STANDARD_LDM_VAE_MAX_LEVEL: usize = 3;
    let (ldm_block_idx, rest) = split_usize_prefix(rest)?;
    let block_idx = STANDARD_LDM_VAE_MAX_LEVEL.checked_sub(ldm_block_idx)?;
    if let Some(rest) = rest.strip_prefix("block.") {
        let (layer_idx, rest) = split_usize_prefix(rest)?;
        return map_ldm_vae_resnet_suffix(rest)
            .map(|suffix| format!("decoder.up_blocks.{block_idx}.resnets.{layer_idx}.{suffix}"));
    }
    rest.strip_prefix("upsample.conv.")
        .map(|suffix| format!("decoder.up_blocks.{block_idx}.upsamplers.0.conv.{suffix}"))
}

fn ldm_vae_mid_tensor_name(side: &str, rest: &str) -> Option<String> {
    if let Some(rest) = rest.strip_prefix("block_1.") {
        return map_ldm_vae_resnet_suffix(rest)
            .map(|suffix| format!("{side}.mid_block.resnets.0.{suffix}"));
    }
    if let Some(rest) = rest.strip_prefix("attn_1.") {
        return map_ldm_vae_attention_suffix(rest)
            .map(|suffix| format!("{side}.mid_block.attentions.0.{suffix}"));
    }
    if let Some(rest) = rest.strip_prefix("block_2.") {
        return map_ldm_vae_resnet_suffix(rest)
            .map(|suffix| format!("{side}.mid_block.resnets.1.{suffix}"));
    }
    None
}

fn map_ldm_vae_resnet_suffix(rest: &str) -> Option<String> {
    let mapped = match rest {
        "norm1.weight" => "norm1.weight",
        "norm1.bias" => "norm1.bias",
        "conv1.weight" => "conv1.weight",
        "conv1.bias" => "conv1.bias",
        "norm2.weight" => "norm2.weight",
        "norm2.bias" => "norm2.bias",
        "conv2.weight" => "conv2.weight",
        "conv2.bias" => "conv2.bias",
        "nin_shortcut.weight" => "conv_shortcut.weight",
        "nin_shortcut.bias" => "conv_shortcut.bias",
        _ => return None,
    };
    Some(mapped.to_string())
}

fn map_ldm_vae_attention_suffix(rest: &str) -> Option<String> {
    let mapped = match rest {
        "norm.weight" => "group_norm.weight",
        "norm.bias" => "group_norm.bias",
        "q.weight" => "to_q.weight",
        "q.bias" => "to_q.bias",
        "k.weight" => "to_k.weight",
        "k.bias" => "to_k.bias",
        "v.weight" => "to_v.weight",
        "v.bias" => "to_v.bias",
        "proj_out.weight" => "to_out.0.weight",
        "proj_out.bias" => "to_out.0.bias",
        _ => return None,
    };
    Some(mapped.to_string())
}

pub fn ldm_unet_native_tensor_name(name: &str) -> Option<String> {
    let mapped = match name {
        "input_blocks.0.0.weight" => return Some("conv_in.weight".to_string()),
        "input_blocks.0.0.bias" => return Some("conv_in.bias".to_string()),
        "time_embed.0.weight" => return Some("time_embedding.linear_1.weight".to_string()),
        "time_embed.0.bias" => return Some("time_embedding.linear_1.bias".to_string()),
        "time_embed.2.weight" => return Some("time_embedding.linear_2.weight".to_string()),
        "time_embed.2.bias" => return Some("time_embedding.linear_2.bias".to_string()),
        "out.0.weight" => return Some("conv_norm_out.weight".to_string()),
        "out.0.bias" => return Some("conv_norm_out.bias".to_string()),
        "out.2.weight" => return Some("conv_out.weight".to_string()),
        "out.2.bias" => return Some("conv_out.bias".to_string()),
        _ => name,
    };
    ldm_input_block_native_tensor_name(mapped)
        .or_else(|| ldm_middle_block_native_tensor_name(mapped))
        .or_else(|| ldm_output_block_native_tensor_name(mapped))
}

fn ldm_input_block_native_tensor_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("input_blocks.")?;
    let (block_idx, rest) = split_usize_prefix(rest)?;
    if block_idx == 0 {
        return None;
    }
    if [3, 6, 9].contains(&block_idx) {
        let down_block = (block_idx / 3).saturating_sub(1);
        let sampler_rest = rest
            .strip_prefix("0.op.")
            .or_else(|| rest.strip_prefix("0.conv."))?;
        return Some(format!(
            "down_blocks.{down_block}.downsamplers.0.conv.{sampler_rest}"
        ));
    }
    let down_block = (block_idx - 1) / 3;
    let layer_idx = (block_idx - 1) % 3;
    if layer_idx >= 2 {
        return None;
    }
    let (submodule_idx, rest) = split_usize_prefix(rest)?;
    match submodule_idx {
        0 => map_ldm_resnet_suffix(rest)
            .map(|suffix| format!("down_blocks.{down_block}.resnets.{layer_idx}.{suffix}")),
        1 => Some(format!(
            "down_blocks.{down_block}.attentions.{layer_idx}.{rest}"
        )),
        _ => None,
    }
}

fn ldm_middle_block_native_tensor_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("middle_block.")?;
    let (block_idx, rest) = split_usize_prefix(rest)?;
    match block_idx {
        0 => map_ldm_resnet_suffix(rest).map(|suffix| format!("mid_block.resnets.0.{suffix}")),
        1 => Some(format!("mid_block.attentions.0.{rest}")),
        2 => map_ldm_resnet_suffix(rest).map(|suffix| format!("mid_block.resnets.1.{suffix}")),
        _ => None,
    }
}

fn ldm_output_block_native_tensor_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("output_blocks.")?;
    let (block_idx, rest) = split_usize_prefix(rest)?;
    let up_block = block_idx / 3;
    let layer_idx = block_idx % 3;
    let (submodule_idx, rest) = split_usize_prefix(rest)?;
    if layer_idx == 2 {
        if let Some(sampler_rest) = rest
            .strip_prefix("op.")
            .or_else(|| rest.strip_prefix("conv."))
        {
            return Some(format!(
                "up_blocks.{up_block}.upsamplers.0.conv.{sampler_rest}"
            ));
        }
    }
    match submodule_idx {
        0 => map_ldm_resnet_suffix(rest)
            .map(|suffix| format!("up_blocks.{up_block}.resnets.{layer_idx}.{suffix}")),
        1 => Some(format!(
            "up_blocks.{up_block}.attentions.{layer_idx}.{rest}"
        )),
        2 if layer_idx == 2 => rest
            .strip_prefix("op.")
            .or_else(|| rest.strip_prefix("conv."))
            .map(|sampler_rest| format!("up_blocks.{up_block}.upsamplers.0.conv.{sampler_rest}")),
        _ => None,
    }
}

fn map_ldm_resnet_suffix(rest: &str) -> Option<String> {
    let mapped = match rest {
        "in_layers.0.weight" => "norm1.weight",
        "in_layers.0.bias" => "norm1.bias",
        "in_layers.2.weight" => "conv1.weight",
        "in_layers.2.bias" => "conv1.bias",
        "emb_layers.1.weight" => "time_emb_proj.weight",
        "emb_layers.1.bias" => "time_emb_proj.bias",
        "out_layers.0.weight" => "norm2.weight",
        "out_layers.0.bias" => "norm2.bias",
        "out_layers.3.weight" => "conv2.weight",
        "out_layers.3.bias" => "conv2.bias",
        "skip_connection.weight" => "conv_shortcut.weight",
        "skip_connection.bias" => "conv_shortcut.bias",
        _ => return None,
    };
    Some(mapped.to_string())
}

fn split_usize_prefix(value: &str) -> Option<(usize, &str)> {
    let (head, tail) = value.split_once('.')?;
    Some((head.parse().ok()?, tail))
}

/// Describe a Diffusers tokenizer directory: `(kind, max_length)`.
///
/// `kind` is a coarse family tag the runtime uses to pick a tokenizer backend
/// (`clip-bpe` for CLIP, `qwen2-bpe` for the Qwen2 fast tokenizer that packs its
/// vocab/merges into `tokenizer.json`). `max_length` is the tokenizer's declared
/// `model_max_length` (the conditioning path may cap lower). Missing or
/// unreadable configs fall back to the CLIP defaults (`clip-bpe`, 77).
fn tokenizer_descriptor(dir: &Path) -> (String, Option<u32>) {
    let config = read_json(dir.join("tokenizer_config.json")).unwrap_or_else(|_| json!({}));
    let class = config
        .get("tokenizer_class")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let kind = if class.contains("Qwen") {
        "qwen2-bpe"
    } else if class.contains("CLIP") || class.is_empty() {
        "clip-bpe"
    } else if class.contains("Llama") || class.contains("T5") {
        "sentencepiece"
    } else {
        "bpe"
    }
    .to_string();
    let max_length = config
        .get("model_max_length")
        .and_then(Value::as_u64)
        .filter(|&value| value > 0 && value <= 10_000_000)
        .map(|value| value as u32)
        .or(Some(77));
    (kind, max_length)
}

fn add_component(
    source: &Path,
    entries: &mut Vec<DiffusionImportEntry>,
    components: &mut BTreeMap<String, DiffusionComponentMetadata>,
    component: &str,
    weight_files: &[&str],
) -> anyhow::Result<()> {
    let component_dir = source.join(component);
    let config_name =
        if component == "scheduler" && component_dir.join("scheduler_config.json").is_file() {
            "scheduler_config.json"
        } else {
            "config.json"
        };
    let config_path = component_dir.join(config_name);
    let mut metadata = DiffusionComponentMetadata::default();
    if config_path.is_file() {
        let entry_name = format!("{component}/{config_name}");
        let config = read_json(&config_path).unwrap_or_else(|_| json!({}));
        // Diffusers components use `_class_name`; transformers text encoders
        // (e.g. Qwen3VLModel) declare `architectures: [..]` instead.
        metadata.class_name = config
            .get("_class_name")
            .and_then(Value::as_str)
            .or_else(|| {
                config
                    .get("architectures")
                    .and_then(Value::as_array)
                    .and_then(|arr| arr.first())
                    .and_then(Value::as_str)
            })
            .map(str::to_string);
        metadata.config_entry = Some(entry_name.clone());
        push_import_file_entry(entries, &entry_name, QT_DIFFUSION_JSON, config_path)?;
    }
    if let Some(weight_file) = weight_files
        .iter()
        .filter(|candidate| candidate.ends_with(".safetensors"))
        .map(|candidate| format!("{candidate}.index.json"))
        .find(|candidate| component_dir.join(candidate).is_file())
    {
        let index_path = component_dir.join(&weight_file);
        match parse_sharded_safetensors_state_dict(&component_dir, &index_path) {
            Ok(tensors) if !tensors.is_empty() => {
                for tensor in tensors {
                    let entry_name = format!("{component}/tensors/{}", tensor.name);
                    metadata.tensor_roles.push(DiffusionTensorRole {
                        role: tensor.name.clone(),
                        entry: entry_name.clone(),
                        dtype: tensor.dtype.clone(),
                        quant_format: None,
                    });
                    metadata.weight_entries.push(entry_name.clone());
                    entries.push(DiffusionImportEntry {
                        name: entry_name,
                        quant_type: tensor.quant_type,
                        shape: tensor.shape,
                        group_size: 0,
                        source: DiffusionImportSource::FileSlice {
                            path: tensor.source_path,
                            offset: tensor.data_offset,
                            len: tensor.data_len,
                        },
                    });
                }
            }
            _ => {
                let entry_name = format!("{component}/{weight_file}");
                metadata.weight_entries.push(entry_name.clone());
                push_import_file_entry(
                    entries,
                    &entry_name,
                    QT_DIFFUSION_SOURCE_WEIGHTS,
                    index_path,
                )?;
            }
        }
    } else if let Some(weight_file) = weight_files
        .iter()
        .find(|candidate| component_dir.join(candidate).is_file())
    {
        let weight_path = component_dir.join(weight_file);
        if weight_path.is_file() {
            if weight_file.ends_with(".safetensors") {
                match parse_safetensors_state_dict(&weight_path) {
                    Ok(tensors) if !tensors.is_empty() => {
                        for tensor in tensors {
                            let entry_name = format!("{component}/tensors/{}", tensor.name);
                            metadata.tensor_roles.push(DiffusionTensorRole {
                                role: tensor.name.clone(),
                                entry: entry_name.clone(),
                                dtype: tensor.dtype.clone(),
                                quant_format: None,
                            });
                            metadata.weight_entries.push(entry_name.clone());
                            entries.push(DiffusionImportEntry {
                                name: entry_name,
                                quant_type: tensor.quant_type,
                                shape: tensor.shape,
                                group_size: 0,
                                source: DiffusionImportSource::FileSlice {
                                    path: tensor.source_path,
                                    offset: tensor.data_offset,
                                    len: tensor.data_len,
                                },
                            });
                        }
                    }
                    _ => {
                        let entry_name = format!("{component}/{weight_file}");
                        metadata.weight_entries.push(entry_name.clone());
                        push_import_file_entry(
                            entries,
                            &entry_name,
                            QT_DIFFUSION_SOURCE_WEIGHTS,
                            weight_path,
                        )?;
                    }
                }
            } else {
                match parse_pytorch_state_dict(&weight_path) {
                    Ok(tensors) if !tensors.is_empty() => {
                        for tensor in tensors {
                            let entry_name = format!("{component}/tensors/{}", tensor.name);
                            metadata.tensor_roles.push(DiffusionTensorRole {
                                role: tensor.name.clone(),
                                entry: entry_name.clone(),
                                dtype: tensor.dtype.clone(),
                                quant_format: None,
                            });
                            metadata.weight_entries.push(entry_name.clone());
                            let source =
                                if pytorch_tensor_is_contiguous(&tensor.shape, &tensor.stride) {
                                    DiffusionImportSource::ZipMember {
                                        archive_path: weight_path.clone(),
                                        member_name: tensor.member_name,
                                    }
                                } else {
                                    // Non-contiguous storage (e.g. channels_last conv
                                    // weights). Materialize the tensor in contiguous
                                    // row-major order so downstream layout matches the
                                    // logical shape.
                                    let archive = MiniZipArchive::open(&weight_path)?;
                                    let storage = archive.read_entry(&tensor.member_name)?;
                                    let data = reorder_pytorch_storage_to_contiguous(
                                        &storage,
                                        &tensor.shape,
                                        &tensor.stride,
                                        tensor.storage_offset,
                                        pytorch_dtype_elem_size(&tensor.dtype),
                                    )?;
                                    DiffusionImportSource::Inline(data)
                                };
                            entries.push(DiffusionImportEntry {
                                name: entry_name,
                                quant_type: tensor.quant_type,
                                shape: tensor.shape,
                                group_size: 0,
                                source,
                            });
                        }
                    }
                    _ => {
                        let entry_name = format!("{component}/{weight_file}");
                        metadata.weight_entries.push(entry_name.clone());
                        push_import_file_entry(
                            entries,
                            &entry_name,
                            QT_DIFFUSION_SOURCE_WEIGHTS,
                            weight_path,
                        )?;
                    }
                }
            }
        }
    }
    if metadata.config_entry.is_some() || !metadata.weight_entries.is_empty() {
        components.insert(component.to_string(), metadata);
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct DiffusionImportEntry {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    source: DiffusionImportSource,
}

#[derive(Debug, Clone)]
enum DiffusionImportSource {
    Inline(Vec<u8>),
    File(PathBuf),
    FileSlice {
        path: PathBuf,
        offset: u64,
        len: u64,
    },
    ZipMember {
        archive_path: PathBuf,
        member_name: String,
    },
}

fn push_import_file_entry(
    entries: &mut Vec<DiffusionImportEntry>,
    name: &str,
    quant_type: u8,
    source_path: PathBuf,
) -> anyhow::Result<()> {
    let data_size = fs::metadata(&source_path)?.len();
    entries.push(DiffusionImportEntry {
        name: name.to_string(),
        quant_type,
        shape: vec![data_size.min(u32::MAX as u64) as u32],
        group_size: 0,
        source: DiffusionImportSource::File(source_path),
    });
    Ok(())
}

fn push_import_inline_entry(
    entries: &mut Vec<DiffusionImportEntry>,
    name: &str,
    quant_type: u8,
    data: Vec<u8>,
) {
    entries.push(DiffusionImportEntry {
        name: name.to_string(),
        quant_type,
        shape: vec![data.len().min(u32::MAX as usize) as u32],
        group_size: 0,
        source: DiffusionImportSource::Inline(data),
    });
}

fn write_import_entries_to_hfq(
    output: &Path,
    metadata_json: &str,
    entries: &[DiffusionImportEntry],
) -> anyhow::Result<()> {
    let stream_entries = entries
        .iter()
        .map(|entry| {
            let data_len = match &entry.source {
                DiffusionImportSource::Inline(data) => data.len() as u64,
                DiffusionImportSource::File(path) => fs::metadata(path)?.len(),
                DiffusionImportSource::FileSlice { len, .. } => *len,
                DiffusionImportSource::ZipMember {
                    archive_path,
                    member_name,
                } => {
                    MiniZipArchive::open(archive_path)?
                        .entry(member_name)?
                        .uncompressed_size
                }
            };
            Ok(HfqStreamEntry {
                name: entry.name.clone(),
                quant_type: entry.quant_type,
                shape: entry.shape.clone(),
                group_size: entry.group_size,
                data_len,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    write_hfqm_package_streaming(
        output,
        HFQ_ARCH_DIFFUSION,
        metadata_json,
        &stream_entries,
        |i, writer| write_import_entry_payload(&entries[i], writer),
    )?;
    Ok(())
}

fn write_import_entry_payload(
    entry: &DiffusionImportEntry,
    writer: &mut dyn Write,
) -> std::io::Result<()> {
    match &entry.source {
        DiffusionImportSource::Inline(data) => {
            writer.write_all(data)?;
        }
        DiffusionImportSource::File(path) => {
            let mut file = fs::File::open(path)?;
            std::io::copy(&mut file, writer)?;
        }
        DiffusionImportSource::FileSlice { path, offset, len } => {
            let mut file = fs::File::open(path)?;
            file.seek(SeekFrom::Start(*offset))?;
            std::io::copy(&mut file.take(*len), writer)?;
        }
        DiffusionImportSource::ZipMember {
            archive_path,
            member_name,
        } => {
            let archive = MiniZipArchive::open(archive_path).map_err(anyhow_to_io)?;
            archive.copy_entry_to(member_name, writer)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
/// A tensor entry parsed from a PyTorch `.bin` state dict. Public so the
/// importer's parser can be exercised directly by tests.
pub struct PytorchTensorEntry {
    pub name: String,
    pub member_name: String,
    pub dtype: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub stride: Vec<i64>,
    pub storage_offset: i64,
}

#[derive(Debug, Clone)]
struct SafetensorsTensorEntry {
    name: String,
    dtype: String,
    quant_type: u8,
    shape: Vec<u32>,
    source_path: PathBuf,
    data_offset: u64,
    data_len: u64,
}

fn parse_sharded_safetensors_state_dict(
    component_dir: &Path,
    index_path: &Path,
) -> anyhow::Result<Vec<SafetensorsTensorEntry>> {
    let index = read_json(index_path)?;
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("safetensors index missing weight_map"))?;
    let mut shard_to_tensors: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (tensor, shard) in weight_map {
        let shard = shard
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("safetensors shard for {tensor:?} is not a string"))?;
        shard_to_tensors
            .entry(shard.to_string())
            .or_default()
            .push(tensor.clone());
    }
    let mut tensors = Vec::new();
    for (shard, wanted) in shard_to_tensors {
        let shard_path = component_dir.join(&shard);
        let parsed = parse_safetensors_state_dict(&shard_path)?;
        let parsed_by_name = parsed
            .into_iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect::<BTreeMap<_, _>>();
        for name in wanted {
            let tensor = parsed_by_name.get(&name).ok_or_else(|| {
                anyhow::anyhow!("safetensors index references missing tensor {name:?} in {shard:?}")
            })?;
            tensors.push(tensor.clone());
        }
    }
    Ok(tensors)
}

fn parse_safetensors_state_dict(path: &Path) -> anyhow::Result<Vec<SafetensorsTensorEntry>> {
    let mut file = fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    let data_start = 8u64
        .checked_add(header_len)
        .ok_or_else(|| anyhow::anyhow!("safetensors header length overflow"))?;
    if data_start > file_len {
        anyhow::bail!(
            "safetensors header extends past end of file: header bytes {header_len}, file bytes {file_len}"
        );
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let header: Value = serde_json::from_slice(&header)?;
    let object = header
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("safetensors header must be a JSON object"))?;
    let mut tensors = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let tensor = value.as_object().ok_or_else(|| {
            anyhow::anyhow!("safetensors tensor {name:?} metadata is not an object")
        })?;
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} missing dtype"))?;
        let (dtype, quant_type, byte_width) = safetensors_dtype_info(dtype)
            .ok_or_else(|| anyhow::anyhow!("unsupported safetensors dtype {dtype:?}"))?;
        let shape_values = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} missing shape"))?;
        let shape = shape_values
            .iter()
            .map(|dim| {
                let dim = dim.as_u64().ok_or_else(|| {
                    anyhow::anyhow!("safetensors tensor {name:?} has non-u64 shape dim")
                })?;
                u32::try_from(dim).map_err(|_| {
                    anyhow::anyhow!("safetensors tensor {name:?} shape dim {dim} exceeds u32")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let offsets = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} missing data_offsets"))?;
        if offsets.len() != 2 {
            anyhow::bail!("safetensors tensor {name:?} data_offsets must have two entries");
        }
        let start = offsets[0].as_u64().ok_or_else(|| {
            anyhow::anyhow!("safetensors tensor {name:?} start offset is not u64")
        })?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} end offset is not u64"))?;
        if end < start {
            anyhow::bail!("safetensors tensor {name:?} end offset precedes start offset");
        }
        let data_len = end - start;
        let absolute_start = data_start.checked_add(start).ok_or_else(|| {
            anyhow::anyhow!("safetensors tensor {name:?} absolute offset overflow")
        })?;
        let absolute_end = data_start
            .checked_add(end)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} absolute end overflow"))?;
        if absolute_end > file_len {
            anyhow::bail!(
                "safetensors tensor {name:?} extends past end of file: end {absolute_end}, file {file_len}"
            );
        }
        let elem_count = shape.iter().try_fold(1u64, |acc, &dim| {
            acc.checked_mul(dim as u64)
                .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} shape overflows"))
        })?;
        let expected_len = elem_count
            .checked_mul(byte_width)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} byte length overflows"))?;
        if data_len != expected_len {
            anyhow::bail!(
                "safetensors tensor {name:?} has {data_len} data bytes but shape/dtype expect {expected_len}"
            );
        }
        tensors.push(SafetensorsTensorEntry {
            name: name.clone(),
            dtype: dtype.to_string(),
            quant_type,
            shape,
            source_path: path.to_path_buf(),
            data_offset: absolute_start,
            data_len,
        });
    }
    Ok(tensors)
}

fn safetensors_dtype_info(dtype: &str) -> Option<(&'static str, u8, u64)> {
    match dtype {
        "F16" => Some(("F16", QT_DIFFUSION_TENSOR_F16, 2)),
        "BF16" => Some(("BF16", QT_DIFFUSION_TENSOR_BF16, 2)),
        "F32" => Some(("F32", QT_DIFFUSION_TENSOR_F32, 4)),
        _ => None,
    }
}

fn pytorch_dtype_elem_size(dtype: &str) -> usize {
    match dtype {
        "F32" => 4,
        "F16" | "BF16" => 2,
        _ => 4,
    }
}

/// Mirrors PyTorch's `is_contiguous`: a tensor is contiguous when, walking the
/// dims from innermost to outermost, each stride equals the running product of
/// the inner sizes. Size-1 (and empty) dims carry arbitrary strides and are
/// skipped, exactly as PyTorch does.
pub fn pytorch_tensor_is_contiguous(shape: &[u32], stride: &[i64]) -> bool {
    if stride.is_empty() {
        return true;
    }
    if stride.len() != shape.len() {
        return true;
    }
    let mut expected: i64 = 1;
    for dim in (0..shape.len()).rev() {
        let size = shape[dim] as i64;
        if size <= 1 {
            continue;
        }
        if stride[dim] != expected {
            return false;
        }
        expected *= size;
    }
    true
}

/// Gather a strided PyTorch storage into a contiguous row-major byte buffer for
/// `shape`. Handles any stride layout (channels_last conv weights in practice).
pub fn reorder_pytorch_storage_to_contiguous(
    storage: &[u8],
    shape: &[u32],
    stride: &[i64],
    storage_offset: i64,
    elem_size: usize,
) -> anyhow::Result<Vec<u8>> {
    let ndim = shape.len();
    if stride.len() != ndim {
        anyhow::bail!(
            "pytorch tensor stride rank {} != shape rank {ndim}",
            stride.len()
        );
    }
    let count: usize = shape.iter().map(|&dim| dim as usize).product();
    let mut out = vec![0u8; count * elem_size];
    let mut index = vec![0usize; ndim];
    for linear in 0..count {
        let mut physical = storage_offset;
        for dim in 0..ndim {
            physical += index[dim] as i64 * stride[dim];
        }
        if physical < 0 {
            anyhow::bail!("pytorch tensor negative element offset");
        }
        let src = physical as usize * elem_size;
        let dst = linear * elem_size;
        let src_end = src
            .checked_add(elem_size)
            .filter(|end| *end <= storage.len())
            .ok_or_else(|| anyhow::anyhow!("pytorch tensor element out of storage bounds"))?;
        out[dst..dst + elem_size].copy_from_slice(&storage[src..src_end]);
        for dim in (0..ndim).rev() {
            index[dim] += 1;
            if index[dim] < shape[dim] as usize {
                break;
            }
            index[dim] = 0;
        }
    }
    Ok(out)
}

pub fn parse_pytorch_state_dict(path: &Path) -> anyhow::Result<Vec<PytorchTensorEntry>> {
    let archive = MiniZipArchive::open(path)?;
    let data_pkl_name = archive
        .entries
        .keys()
        .find(|name| name.ends_with("/data.pkl") || name.as_str() == "data.pkl")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("PyTorch archive missing data.pkl"))?;
    let root = data_pkl_name
        .strip_suffix("data.pkl")
        .unwrap_or("")
        .trim_end_matches('/');
    let pickle = archive.read_entry(&data_pkl_name)?;
    let mut tensors = parse_pytorch_pickle_tensor_index(&pickle)?;
    tensors.retain(|tensor| {
        archive
            .entry(&format!("{root}/data/{}", tensor.storage_key))
            .is_ok()
    });
    Ok(tensors
        .into_iter()
        .map(|tensor| PytorchTensorEntry {
            name: tensor.name,
            member_name: format!("{root}/data/{}", tensor.storage_key),
            dtype: tensor.dtype,
            quant_type: tensor.quant_type,
            shape: tensor.shape,
            stride: tensor.stride,
            storage_offset: tensor.storage_offset,
        })
        .collect())
}

#[derive(Debug, Clone)]
struct ParsedPytorchTensor {
    name: String,
    storage_key: String,
    dtype: String,
    quant_type: u8,
    shape: Vec<u32>,
    stride: Vec<i64>,
    storage_offset: i64,
}

#[derive(Debug, Clone)]
enum PickleValue {
    Mark,
    Str(String),
    Int(i64),
    Bool(()),
    Tuple(Vec<PickleValue>),
    Global {
        module: String,
        name: String,
    },
    StorageRef {
        key: String,
        dtype: String,
        quant_type: u8,
    },
    Tensor(ParsedPytorchTensor),
    Other,
}

fn parse_pytorch_pickle_tensor_index(data: &[u8]) -> anyhow::Result<Vec<ParsedPytorchTensor>> {
    let mut pos = 0usize;
    let mut stack: Vec<PickleValue> = Vec::new();
    let mut memo: BTreeMap<u32, PickleValue> = BTreeMap::new();
    let mut tensors = Vec::new();
    while pos < data.len() {
        let op = data[pos];
        pos += 1;
        match op {
            0x80 => pos += 1, // PROTO
            b'c' => {
                let module = read_pickle_line(data, &mut pos)?;
                let name = read_pickle_line(data, &mut pos)?;
                stack.push(PickleValue::Global { module, name });
            }
            b'q' => {
                let idx = read_u8(data, &mut pos)? as u32;
                if let Some(value) = stack.last().cloned() {
                    memo.insert(idx, value);
                }
            }
            b'r' => {
                let idx = read_u32(data, &mut pos)?;
                if let Some(value) = stack.last().cloned() {
                    memo.insert(idx, value);
                }
            }
            b'h' => {
                let idx = read_u8(data, &mut pos)? as u32;
                stack.push(memo.get(&idx).cloned().unwrap_or(PickleValue::Other));
            }
            b'j' => {
                let idx = read_u32(data, &mut pos)?;
                stack.push(memo.get(&idx).cloned().unwrap_or(PickleValue::Other));
            }
            b'(' => stack.push(PickleValue::Mark),
            b')' => stack.push(PickleValue::Tuple(Vec::new())),
            b'X' => {
                let len = read_u32(data, &mut pos)? as usize;
                let bytes = read_bytes(data, &mut pos, len)?;
                stack.push(PickleValue::Str(String::from_utf8_lossy(bytes).to_string()));
            }
            b'U' => {
                let len = read_u8(data, &mut pos)? as usize;
                let bytes = read_bytes(data, &mut pos, len)?;
                stack.push(PickleValue::Str(String::from_utf8_lossy(bytes).to_string()));
            }
            b'K' => stack.push(PickleValue::Int(read_u8(data, &mut pos)? as i64)),
            b'M' => stack.push(PickleValue::Int(read_u16(data, &mut pos)? as i64)),
            b'J' => stack.push(PickleValue::Int(read_i32(data, &mut pos)? as i64)),
            0x88 => stack.push(PickleValue::Bool(())),
            0x89 => stack.push(PickleValue::Bool(())),
            b'N' => stack.push(PickleValue::Other),
            b'}' | b']' => stack.push(PickleValue::Other),
            b't' => {
                let values = pop_to_mark(&mut stack);
                stack.push(PickleValue::Tuple(values));
            }
            0x85 => {
                let Some(value) = stack.pop() else {
                    anyhow::bail!("pickle TUPLE1 stack underflow");
                };
                stack.push(PickleValue::Tuple(vec![value]));
            }
            0x86 => {
                let b = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle TUPLE2 stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle TUPLE2 stack underflow"))?;
                stack.push(PickleValue::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle TUPLE3 stack underflow"))?;
                let b = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle TUPLE3 stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle TUPLE3 stack underflow"))?;
                stack.push(PickleValue::Tuple(vec![a, b, c]));
            }
            b'Q' => {
                let value = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("pickle BINPERSID stack underflow"))?;
                stack.push(storage_ref_from_pickle_value(value));
            }
            b'R' => {
                let args = stack.pop().unwrap_or(PickleValue::Other);
                let callable = stack.pop().unwrap_or(PickleValue::Other);
                let reduced = reduce_pickle_value(callable, args);
                if let PickleValue::Tensor(mut tensor) = reduced.clone() {
                    if let Some(name) = previous_stack_string(&stack) {
                        tensor.name = name;
                        tensors.push(tensor.clone());
                    }
                    stack.push(PickleValue::Tensor(tensor));
                } else {
                    stack.push(reduced);
                }
            }
            b'b' => {
                let _ = stack.pop();
            }
            b's' => {
                let _ = stack.pop();
                let _ = stack.pop();
            }
            b'u' | b'e' => {
                let _ = pop_to_mark(&mut stack);
            }
            b'.' => break,
            _ => {
                anyhow::bail!("unsupported pickle opcode 0x{op:02x} at {}", pos - 1);
            }
        }
    }
    Ok(tensors)
}

fn reduce_pickle_value(callable: PickleValue, args: PickleValue) -> PickleValue {
    match callable {
        PickleValue::Global { module, name }
            if module == "torch._utils" && name == "_rebuild_tensor_v2" =>
        {
            tensor_from_rebuild_args(args)
                .map(PickleValue::Tensor)
                .unwrap_or(PickleValue::Other)
        }
        _ => PickleValue::Other,
    }
}

fn tensor_from_rebuild_args(args: PickleValue) -> Option<ParsedPytorchTensor> {
    let PickleValue::Tuple(items) = args else {
        return None;
    };
    let storage = match items.first()? {
        PickleValue::StorageRef {
            key,
            dtype,
            quant_type,
        } => (key.clone(), dtype.clone(), *quant_type),
        _ => return None,
    };
    let shape = tuple_ints(items.get(2)?)?
        .into_iter()
        .map(|dim| dim as u32)
        .collect();
    // `_rebuild_tensor_v2(storage, storage_offset, size, stride, ...)`. We must
    // honor `storage_offset` and `stride`: tensors saved in a non-contiguous
    // memory format (e.g. channels_last conv weights) keep a logical OIHW `size`
    // while the underlying storage is physically reordered. Ignoring the stride
    // loads such weights transposed.
    let storage_offset = match items.get(1) {
        Some(PickleValue::Int(value)) => *value,
        _ => 0,
    };
    let stride = items.get(3).and_then(tuple_ints).unwrap_or_default();
    Some(ParsedPytorchTensor {
        name: String::new(),
        storage_key: storage.0,
        dtype: storage.1,
        quant_type: storage.2,
        shape,
        stride,
        storage_offset,
    })
}

fn storage_ref_from_pickle_value(value: PickleValue) -> PickleValue {
    let PickleValue::Tuple(items) = value else {
        return PickleValue::Other;
    };
    if items.len() < 5 {
        return PickleValue::Other;
    }
    let PickleValue::Global { module, name } = &items[1] else {
        return PickleValue::Other;
    };
    if module != "torch" {
        return PickleValue::Other;
    }
    let Some((dtype, quant_type)) = torch_storage_dtype(name) else {
        return PickleValue::Other;
    };
    let PickleValue::Str(key) = &items[2] else {
        return PickleValue::Other;
    };
    PickleValue::StorageRef {
        key: key.clone(),
        dtype: dtype.to_string(),
        quant_type,
    }
}

fn torch_storage_dtype(storage: &str) -> Option<(&'static str, u8)> {
    match storage {
        "HalfStorage" => Some(("F16", QT_DIFFUSION_TENSOR_F16)),
        "FloatStorage" => Some(("F32", QT_DIFFUSION_TENSOR_F32)),
        "BFloat16Storage" => Some(("BF16", QT_DIFFUSION_TENSOR_BF16)),
        _ => None,
    }
}

fn previous_stack_string(stack: &[PickleValue]) -> Option<String> {
    stack.iter().rev().find_map(|value| match value {
        PickleValue::Str(value) => Some(value.clone()),
        _ => None,
    })
}

fn tuple_ints(value: &PickleValue) -> Option<Vec<i64>> {
    let PickleValue::Tuple(items) = value else {
        return None;
    };
    items
        .iter()
        .map(|item| match item {
            PickleValue::Int(value) => Some(*value),
            _ => None,
        })
        .collect()
}

fn pop_to_mark(stack: &mut Vec<PickleValue>) -> Vec<PickleValue> {
    let mut values = Vec::new();
    while let Some(value) = stack.pop() {
        if matches!(value, PickleValue::Mark) {
            values.reverse();
            return values;
        }
        values.push(value);
    }
    values.reverse();
    values
}

fn read_pickle_line(data: &[u8], pos: &mut usize) -> anyhow::Result<String> {
    let start = *pos;
    while *pos < data.len() && data[*pos] != b'\n' {
        *pos += 1;
    }
    if *pos >= data.len() {
        anyhow::bail!("pickle line extends past end of data");
    }
    let line = String::from_utf8_lossy(&data[start..*pos]).to_string();
    *pos += 1;
    Ok(line)
}

fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, len: usize) -> anyhow::Result<&'a [u8]> {
    if *pos + len > data.len() {
        anyhow::bail!("read past end");
    }
    let out = &data[*pos..*pos + len];
    *pos += len;
    Ok(out)
}

fn read_u8(data: &[u8], pos: &mut usize) -> anyhow::Result<u8> {
    Ok(read_bytes(data, pos, 1)?[0])
}

fn read_u16(data: &[u8], pos: &mut usize) -> anyhow::Result<u16> {
    Ok(u16::from_le_bytes(read_bytes(data, pos, 2)?.try_into()?))
}

fn read_u32(data: &[u8], pos: &mut usize) -> anyhow::Result<u32> {
    Ok(u32::from_le_bytes(read_bytes(data, pos, 4)?.try_into()?))
}

fn read_i32(data: &[u8], pos: &mut usize) -> anyhow::Result<i32> {
    Ok(i32::from_le_bytes(read_bytes(data, pos, 4)?.try_into()?))
}

#[derive(Debug, Clone)]
struct MiniZipEntry {
    compressed_size: u64,
    uncompressed_size: u64,
    data_offset: u64,
    compression_method: u16,
}

#[derive(Debug, Clone)]
struct MiniZipArchive {
    path: PathBuf,
    entries: BTreeMap<String, MiniZipEntry>,
}

impl MiniZipArchive {
    fn open(path: &Path) -> anyhow::Result<Self> {
        let mut file = fs::File::open(path)?;
        let len = file.metadata()?.len();
        let tail_len = len.min(66_000) as usize;
        file.seek(SeekFrom::End(-(tail_len as i64)))?;
        let mut tail = vec![0u8; tail_len];
        file.read_exact(&mut tail)?;
        let eocd_pos = tail
            .windows(4)
            .rposition(|window| window == b"PK\x05\x06")
            .ok_or_else(|| anyhow::anyhow!("zip EOCD not found in {}", path.display()))?;
        let eocd = &tail[eocd_pos..];
        if eocd.len() < 22 {
            anyhow::bail!("truncated zip EOCD");
        }
        let central_size = u32::from_le_bytes(eocd[12..16].try_into()?) as u64;
        let central_offset = u32::from_le_bytes(eocd[16..20].try_into()?) as u64;
        let mut central = vec![0u8; central_size as usize];
        file.seek(SeekFrom::Start(central_offset))?;
        file.read_exact(&mut central)?;
        let mut pos = 0usize;
        let mut entries = BTreeMap::new();
        while pos + 46 <= central.len() {
            if &central[pos..pos + 4] != b"PK\x01\x02" {
                break;
            }
            let compression_method = u16::from_le_bytes(central[pos + 10..pos + 12].try_into()?);
            let compressed_size =
                u32::from_le_bytes(central[pos + 20..pos + 24].try_into()?) as u64;
            let uncompressed_size =
                u32::from_le_bytes(central[pos + 24..pos + 28].try_into()?) as u64;
            let name_len = u16::from_le_bytes(central[pos + 28..pos + 30].try_into()?) as usize;
            let extra_len = u16::from_le_bytes(central[pos + 30..pos + 32].try_into()?) as usize;
            let comment_len = u16::from_le_bytes(central[pos + 32..pos + 34].try_into()?) as usize;
            let local_offset = u32::from_le_bytes(central[pos + 42..pos + 46].try_into()?) as u64;
            let name_start = pos + 46;
            let name_end = name_start + name_len;
            if name_end > central.len() {
                anyhow::bail!("truncated zip central directory name");
            }
            let name = String::from_utf8_lossy(&central[name_start..name_end]).to_string();
            let data_offset = local_data_offset(&mut file, local_offset)?;
            entries.insert(
                name,
                MiniZipEntry {
                    compressed_size,
                    uncompressed_size,
                    data_offset,
                    compression_method,
                },
            );
            pos = name_end + extra_len + comment_len;
        }
        Ok(Self {
            path: path.to_path_buf(),
            entries,
        })
    }

    fn entry(&self, name: &str) -> anyhow::Result<&MiniZipEntry> {
        self.entries
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("zip entry {name:?} not found"))
    }

    fn read_entry(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        let entry = self.entry(name)?;
        if entry.compression_method != 0 {
            anyhow::bail!("zip entry {name:?} is compressed; only stored entries are supported");
        }
        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(entry.data_offset))?;
        let mut data = vec![0u8; entry.uncompressed_size as usize];
        file.read_exact(&mut data)?;
        Ok(data)
    }

    fn copy_entry_to(&self, name: &str, writer: &mut dyn Write) -> std::io::Result<()> {
        let entry = self.entry(name).map_err(anyhow_to_io)?;
        if entry.compression_method != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("zip entry {name:?} is compressed"),
            ));
        }
        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(entry.data_offset))?;
        let mut limited = file.take(entry.compressed_size);
        std::io::copy(&mut limited, writer)?;
        Ok(())
    }
}

fn local_data_offset(file: &mut fs::File, local_offset: u64) -> anyhow::Result<u64> {
    let mut header = [0u8; 30];
    file.seek(SeekFrom::Start(local_offset))?;
    file.read_exact(&mut header)?;
    if &header[0..4] != b"PK\x03\x04" {
        anyhow::bail!("invalid zip local header");
    }
    let name_len = u16::from_le_bytes(header[26..28].try_into()?) as u64;
    let extra_len = u16::from_le_bytes(header[28..30].try_into()?) as u64;
    Ok(local_offset + 30 + name_len + extra_len)
}

fn anyhow_to_io(error: anyhow::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
}

fn read_json(path: impl AsRef<Path>) -> anyhow::Result<Value> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}
