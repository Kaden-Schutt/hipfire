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
