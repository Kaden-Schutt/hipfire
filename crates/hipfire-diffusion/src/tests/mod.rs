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

mod encoder;
mod hip;
mod import;
mod latent;
mod misc;
mod model;
mod native;
mod pipeline;
mod quant;
mod scheduler;
mod unet;
mod vae;

pub(crate) const DEFAULT_TINY_SD_HFQ: &str = "/tmp/hipfire-tiny-sd-diffusion.hfq";

pub(crate) fn tiny_sd_hfq_path() -> PathBuf {
    std::env::var_os("HIPFIRE_TINY_SD_HFQ")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_TINY_SD_HFQ))
}

pub(crate) fn skip_missing_tiny_sd(path: &Path) -> bool {
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

// Shared scaffold for the oq4 activation-precision ladder parity tests:
// builds W[m,k]/X[batch,k], the full-precision CPU Y=X@Wᵀ, and the GPU
// oq4-packed weight + rotated activation. Returns None if no GPU.
pub(crate) fn oq4_gpu_parity_fixture(
    m: usize,
    k: usize,
    batch: usize,
) -> Option<(
    hipfire_rdna::Gpu,
    hipfire_rdna::GpuTensor,
    hipfire_rdna::GpuTensor,
    Vec<f32>,
)> {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: ROCm GPU unavailable: {e}");
            return None;
        }
    };
    if !gpu.arch_caps.has_wmma_w32() {
        eprintln!("skip: no wave32 WMMA");
        return None;
    }
    let w: Vec<f32> = (0..m * k)
        .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.01)
        .collect();
    let x: Vec<f32> = (0..batch * k)
        .map(|i| (((i * 13) % 97) as f32 - 48.0) * 0.02)
        .collect();
    let mut yref = vec![0f32; batch * m];
    for b in 0..batch {
        for o in 0..m {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += x[b * k + kk] * w[o * k + kk];
            }
            yref[b * m + o] = acc;
        }
    }
    let signs1 = hipfire_quantize::gen_fwht_signs(OQ_FWHT_SEED1, 256);
    let signs2 = hipfire_quantize::gen_fwht_signs(OQ_FWHT_SEED2, 256);
    let oq4 = hipfire_quantize::codecs::quantize_oq4g256(&w, &signs1, &signs2);
    let packed = pack_oq4_arch_combined(&oq4, m, k);
    let w_dev = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
    let x_dev = gpu.upload_f32(&x, &[batch * k]).unwrap();
    let x_rot = gpu
        .alloc_tensor(&[batch * k], hipfire_rdna::DType::F32)
        .unwrap();
    gpu.rotate_x_mq_batched(&x_dev, &x_rot, k, batch).unwrap();
    Some((gpu, w_dev, x_rot, yref))
}

pub(crate) fn corr_rel_l2(reference: &[f32], got: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut err) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in reference.iter().zip(got) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
        err += ((*a - *b) as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt()), (err / na).sqrt())
}

pub(crate) fn zero_clip_layer(hidden: usize) -> ClipEncoderLayer {
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

pub(crate) fn tiny_rgb_image_batch(batch: usize, width: usize, height: usize) -> RgbImageBatch {
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

pub(crate) fn tiny_mask_image_batch(batch: usize, width: usize, height: usize) -> RgbImageBatch {
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

pub(crate) fn minimal_metadata() -> DiffusionHfqMetadata {
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

pub(crate) fn tiny_sd_scheduler_config_for_tests() -> SchedulerConfig {
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

pub(crate) fn tiny_runtime_metadata() -> DiffusionHfqMetadata {
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

pub(crate) fn tiny_qwen_transformer_runtime_metadata() -> DiffusionHfqMetadata {
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

pub(crate) fn tiny_runtime_config() -> StableDiffusionConfig {
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

pub(crate) fn tiny_qwen_transformer_runtime_tensors() -> Vec<HfqMemTensor> {
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

pub(crate) fn tiny_complete_runtime_tensors() -> Vec<HfqMemTensor> {
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

pub(crate) struct TestNoiseBackend;

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

pub(crate) struct TestSdxlNoiseBackend {
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

pub(crate) struct TestInpaintNoiseBackend {
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

pub(crate) struct TestImageDecoder;

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

pub(crate) fn rgb_byte_to_model_value(value: u8) -> f32 {
    (value as f32) / 127.5 - 1.0
}

pub(crate) struct SolidTensorImageDecoder;

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

pub(crate) fn tiny_txt2img_test_pipeline(decoder: Box<dyn DiffusionImageDecoder>) -> DiffusionPipeline {
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
            text_conditioner: None,
            krea2_tokenizer: None,
        }),
        native_runtime_error: None,
    }
}

pub(crate) fn tiny_inpaint_test_pipeline(
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
            text_conditioner: None,
            krea2_tokenizer: None,
        }),
        native_runtime_error: None,
    };
    (pipeline, called, dir)
}

pub(crate) fn f32_mem_tensor(name: &str, shape: &[u32], data: &[f32]) -> HfqMemTensor {
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

pub(crate) fn qwen_tiny_transformer_denoiser_tensors() -> Vec<HfqMemTensor> {
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

pub(crate) fn krea_tiny_transformer_denoiser_tensors() -> Vec<HfqMemTensor> {
    let block = "transformer/tensors/transformer_blocks.0";
    let attn = format!("{block}.attn");
    let identity2 = [1.0, 0.0, 0.0, 1.0];
    vec![
        f32_mem_tensor(
            "transformer/tensors/img_in.weight",
            &[2, 4],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ),
        f32_mem_tensor("transformer/tensors/img_in.bias", &[2], &[0.0, 0.0]),
        f32_mem_tensor(
            "transformer/tensors/final_layer.linear.weight",
            &[4, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0],
        ),
        f32_mem_tensor(
            "transformer/tensors/final_layer.linear.bias",
            &[4],
            &[0.0; 4],
        ),
        f32_mem_tensor(
            "transformer/tensors/final_layer.norm.weight",
            &[2],
            &[1.0, 1.0],
        ),
        f32_mem_tensor(
            "transformer/tensors/final_layer.scale_shift_table",
            &[2, 2],
            &[0.0; 4],
        ),
        f32_mem_tensor(
            "transformer/tensors/time_embed.linear_1.weight",
            &[2, 2],
            &identity2,
        ),
        f32_mem_tensor(
            "transformer/tensors/time_embed.linear_1.bias",
            &[2],
            &[0.0; 2],
        ),
        f32_mem_tensor(
            "transformer/tensors/time_embed.linear_2.weight",
            &[2, 2],
            &identity2,
        ),
        f32_mem_tensor(
            "transformer/tensors/time_embed.linear_2.bias",
            &[2],
            &[0.0; 2],
        ),
        // Zero time modulation + zero block scale_shift_table => all adaLN
        // gates are zero => each block is the identity (stable smoke test).
        f32_mem_tensor(
            "transformer/tensors/time_mod_proj.weight",
            &[12, 2],
            &[0.0; 24],
        ),
        f32_mem_tensor("transformer/tensors/time_mod_proj.bias", &[12], &[0.0; 12]),
        f32_mem_tensor("transformer/tensors/txt_in.weight", &[2, 2], &identity2),
        f32_mem_tensor("transformer/tensors/txt_in.bias", &[2], &[0.0; 2]),
        f32_mem_tensor(&format!("{block}.scale_shift_table"), &[6, 2], &[0.0; 12]),
        f32_mem_tensor(&format!("{block}.norm1.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{block}.norm2.weight"), &[2], &[1.0, 1.0]),
        f32_mem_tensor(&format!("{attn}.to_q.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{attn}.to_k.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{attn}.to_v.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{attn}.to_gate.weight"), &[2, 2], &[0.0; 4]),
        f32_mem_tensor(&format!("{attn}.to_out.0.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{block}.ff.gate.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{block}.ff.up.weight"), &[2, 2], &identity2),
        f32_mem_tensor(&format!("{block}.ff.down.weight"), &[2, 2], &identity2),
    ]
}

pub(crate) fn assert_f32_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "mismatch at {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}

pub(crate) fn rms_norm_heads_reference(
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

pub(crate) fn qwen_block_expected_mlp_only(hidden: &CpuTensor) -> Vec<f32> {
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

pub(crate) fn q4f16_g64_mem_tensor(name: &str, shape: &[u32], data: &[f32]) -> HfqMemTensor {
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

pub(crate) fn q4k_mem_tensor(name: &str, shape: &[u32], low_nibbles: &[u8]) -> HfqMemTensor {
    HfqMemTensor {
        name: name.to_string(),
        quant_type: QT_DIFFUSION_TENSOR_Q4_K,
        shape: shape.to_vec(),
        group_size: 256,
        data: q4k_test_block(low_nibbles),
    }
}

pub(crate) fn q4k_test_block(low_nibbles: &[u8]) -> Vec<u8> {
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

pub(crate) fn hfq4_mem_tensor(
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

pub(crate) fn hfq6_mem_tensor(name: &str, shape: &[u32], values: &[u8]) -> HfqMemTensor {
    HfqMemTensor {
        name: name.to_string(),
        quant_type: QT_DIFFUSION_TENSOR_HFQ6_G256,
        shape: shape.to_vec(),
        group_size: 256,
        data: hfq6_test_block(values),
    }
}

pub(crate) fn hfq6_test_block(values: &[u8]) -> Vec<u8> {
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

pub(crate) fn q8f16_mem_tensor(name: &str, shape: &[u32], data: &[f32]) -> HfqMemTensor {
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

pub(crate) fn bytes_mem_tensor(name: &str, quant_type: u8, data: &[u8]) -> HfqMemTensor {
    HfqMemTensor {
        name: name.to_string(),
        quant_type,
        shape: vec![data.len() as u32],
        group_size: 0,
        data: data.to_vec(),
    }
}

pub(crate) fn write_safetensors_fixture(path: &Path, tensors: &[(&str, &str, &[u64], &[u8])]) {
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

pub(crate) fn write_safetensors_fixture_owned(path: &Path, tensors: &[(String, String, Vec<u64>, Vec<u8>)]) {
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

pub(crate) fn f32_safetensors_tensor(
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

pub(crate) fn write_tiny_ldm_unet_safetensors(path: &Path) {
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

pub(crate) fn push_tiny_ldm_clip_text_encoder_tensors(tensors: &mut Vec<(String, String, Vec<u64>, Vec<u8>)>) {
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

pub(crate) fn center_identity_conv2(channels: usize) -> Vec<f32> {
    center_identity_conv(channels)
}

pub(crate) fn center_identity_conv(channels: usize) -> Vec<f32> {
    let mut data = vec![0.0; channels * channels * 3 * 3];
    for channel in 0..channels {
        data[(((channel * channels + channel) * 3 + 1) * 3) + 1] = 1.0;
    }
    data
}

pub(crate) fn push_zero_attention_tensors(
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
