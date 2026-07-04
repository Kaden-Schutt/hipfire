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
