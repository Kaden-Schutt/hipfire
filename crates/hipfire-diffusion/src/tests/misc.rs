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

    let out =
        scaled_dot_product_attention_with_key_mask(&q, &k, &v, 1, Some(&[false, true])).unwrap();

    assert_eq!(out.shape, vec![1, 1, 2]);
    assert_f32_close(&out.data, &[7.0, 11.0], 1e-6);
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
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
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
fn timestep_embedding_matches_diffusers_ordering_flags() {
    let flipped = timestep_embedding(&[0.0], 4, true, 0.0).unwrap();
    assert_eq!(flipped.shape, vec![1, 4]);
    assert_eq!(flipped.data, vec![1.0, 1.0, 0.0, 0.0]);

    let unflipped = timestep_embedding(&[0.0], 4, false, 0.0).unwrap();
    assert_eq!(unflipped.data, vec![0.0, 0.0, 1.0, 1.0]);
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
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
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
    let stride = [(i * h * w) as i64, 1, (i * w) as i64, i as i64];
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
        assert!(
            im.iter_f32().all(|v| v >= 0.0),
            "{}: negative imatrix",
            im.name
        );
        imatrices += 1;
    }
    eprintln!("[calib-valid] hessians={hessians} imatrices={imatrices} (all symmetric+PSD)");
    assert!(hessians > 0 && imatrices > 0);
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
        if let Err(error) = hipfire_rdna::Gpu::init_with_device(0) {
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
