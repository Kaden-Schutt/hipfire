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
        cpu_tensor_from_hfq(&hfq, "f16").unwrap().data,
        vec![1.5, -2.0]
    );
    assert_eq!(cpu_tensor_from_hfq(&hfq, "bf16").unwrap().data, vec![3.0]);
    assert_eq!(cpu_tensor_from_hfq(&hfq, "f32").unwrap().data, vec![4.25]);
    assert_eq!(
        cpu_tensor_from_hfq(&hfq, "q8").unwrap().data,
        vec![1.0, -2.0, 3.5]
    );
    assert_eq!(
        cpu_tensor_from_hfq(&hfq, "q4").unwrap().data,
        vec![-1.0, 1.0, 0.0, 1.75]
    );
    assert_eq!(
        cpu_tensor_from_hfq(&hfq, "q4k").unwrap().data,
        vec![1.0, 2.0, 0.0, 1.75]
    );
    assert_eq!(
        cpu_tensor_from_hfq(&hfq, "hfq4g128").unwrap().data,
        vec![-1.0, 1.0, 0.0, 1.75]
    );
    assert_eq!(
        cpu_tensor_from_hfq(&hfq, "hfq4g256").unwrap().data,
        vec![-1.0, 1.0, 0.0, 1.75]
    );
    assert_eq!(
        cpu_tensor_from_hfq(&hfq, "hfq6g256").unwrap().data,
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
    let q4_error = cpu_tensor_from_hfq(&hfq, "bad_q4").unwrap_err();
    assert!(q4_error.to_string().contains("Q4F16_G64"));
    assert!(q4_error.to_string().contains("requires at least 36"));
    let q8_error = cpu_tensor_from_hfq(&hfq, "bad_q8").unwrap_err();
    assert!(q8_error.to_string().contains("Q8F16"));
    assert!(q8_error.to_string().contains("requires at least 34"));
    let q4k_error = cpu_tensor_from_hfq(&hfq, "bad_q4k").unwrap_err();
    assert!(q4k_error.to_string().contains("Q4_K"));
    assert!(q4k_error.to_string().contains("requires at least 144"));
    let hfq4g128_error = cpu_tensor_from_hfq(&hfq, "bad_hfq4g128").unwrap_err();
    assert!(hfq4g128_error.to_string().contains("HFQ4G128"));
    assert!(hfq4g128_error.to_string().contains("requires at least 72"));
    let hfq4g256_error = cpu_tensor_from_hfq(&hfq, "bad_hfq4g256").unwrap_err();
    assert!(hfq4g256_error.to_string().contains("HFQ4G256"));
    assert!(hfq4g256_error.to_string().contains("requires at least 136"));
    let hfq6g256_error = cpu_tensor_from_hfq(&hfq, "bad_hfq6g256").unwrap_err();
    assert!(hfq6g256_error.to_string().contains("HFQ6G256"));
    assert!(hfq6g256_error.to_string().contains("requires at least 200"));
    let _ = fs::remove_dir_all(&dir);
}

/// Phase 3: the im2col + WMMA-GEMM conv must match the F32 direct-conv CPU
/// reference to F16 tolerance, across a 3x3 stride-1 pad-1 conv (batch 2, so
/// the per-batch GEMM offset logic is exercised) and a 1x1 conv (the
/// post_quant/proj/shortcut shape, K = in_channels).
#[test]
fn wmma_conv2d_resident_matches_cpu_reference_to_f16_tolerance() {
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
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
        (
            2usize, 4usize, 5usize, 5usize, 6usize, 3usize, 3usize, 1usize, 1usize,
        ),
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
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
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

#[test]
fn oq4_w4a16_gpu_matches_cpu_reference_when_gpu_is_available() {
    // Phase 4a: validate the W4A16 quantized-compute chain on-device:
    // oq4g256 weight -> pack_oq4_arch_combined -> rotate_x_mq_batched(act)
    // -> gemm_oq4_grouped_f16_wmma, vs the full-precision CPU Y = X @ Wᵀ.
    let mut gpu = match hipfire_rdna::Gpu::init_with_device(0) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: ROCm GPU unavailable for oq4 W4A16 parity: {e}");
            return;
        }
    };
    if !gpu.arch_caps.has_wmma_w32() {
        eprintln!("skip: no wave32 WMMA");
        return;
    }
    let (m, k, batch) = (256usize, 512usize, 8usize); // out, in, rows; k%256==0
    let w: Vec<f32> = (0..m * k)
        .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.01)
        .collect();
    let x: Vec<f32> = (0..batch * k)
        .map(|i| (((i * 13) % 97) as f32 - 48.0) * 0.02)
        .collect();
    // CPU reference Y[batch, m] = X @ Wᵀ.
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
    let y_dev = gpu
        .alloc_tensor(&[batch * m], hipfire_rdna::DType::F32)
        .unwrap();
    gpu.gemm_oq4_grouped_f16_wmma(&w_dev, &x_rot, &y_dev, m, k, batch, 256)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let y = gpu.download_f32(&y_dev).unwrap();
    // 4-bit weight quant: expect high correlation + small relative L2.
    let (mut dot, mut na, mut nb, mut err) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in yref.iter().zip(&y) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
        err += ((*a - *b) as f64).powi(2);
    }
    let corr = dot / (na.sqrt() * nb.sqrt());
    let rel_l2 = (err / na).sqrt();
    eprintln!("[oq4-w4a16] corr={corr:.5} relL2={rel_l2:.4}");
    assert!(
        corr > 0.99,
        "oq4 W4A16 corr too low ({corr:.4}) — rotation/layout bug?"
    );
    assert!(rel_l2 < 0.06, "oq4 W4A16 relL2 too high ({rel_l2:.4})");
}

#[test]
fn oq4_w4a8_gpu_matches_cpu_reference_when_gpu_is_available() {
    // W4A8: oq4 weight, activation quantized to q8_1 (int8 WMMA over 4-bit).
    let (m, k, batch) = (256usize, 512usize, 8usize);
    let Some((mut gpu, w_dev, x_rot, yref)) = oq4_gpu_parity_fixture(m, k, batch) else {
        return;
    };
    let y_dev = gpu
        .alloc_tensor(&[batch * m], hipfire_rdna::DType::F32)
        .unwrap();
    gpu.gemm_oq4_residual_mmq(&w_dev, &x_rot, &y_dev, m, k, batch, false)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let y = gpu.download_f32(&y_dev).unwrap();
    let (corr, rel_l2) = corr_rel_l2(&yref, &y);
    eprintln!("[oq4-w4a8] corr={corr:.5} relL2={rel_l2:.4}");
    assert!(corr > 0.99, "W4A8 corr too low ({corr:.4})");
    assert!(rel_l2 < 0.10, "W4A8 relL2 too high ({rel_l2:.4})");
}

#[test]
fn oq4_w4a4_gpu_matches_cpu_reference_when_gpu_is_available() {
    // W4A4: oq4 weight, activation quantized to int4 (int4 WMMA, 2x on gfx1103).
    // Lossiest rung — validate the kernel RUNS and is roughly correct (a
    // rotation/layout bug gives corr~0), not high-fidelity.
    let (m, k, batch) = (256usize, 512usize, 8usize);
    let Some((mut gpu, w_dev, x_rot, yref)) = oq4_gpu_parity_fixture(m, k, batch) else {
        return;
    };
    let y_dev = gpu
        .alloc_tensor(&[batch * m], hipfire_rdna::DType::F32)
        .unwrap();
    gpu.gemm_oq4_grouped_act_batched(&w_dev, &x_rot, &y_dev, m, k, batch)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let y = gpu.download_f32(&y_dev).unwrap();
    let (corr, rel_l2) = corr_rel_l2(&yref, &y);
    eprintln!("[oq4-w4a4] corr={corr:.5} relL2={rel_l2:.4}");
    assert!(
        corr > 0.9,
        "W4A4 corr too low ({corr:.4}) — rotation/layout bug?"
    );
}

#[test]
fn oq4_arch_combined_pack_layout_is_correct() {
    // Pack canonical oq4g256 into the W4A16 arch-combined device layout and
    // verify the byte regions (nibbles + f32 scales) match the source — guards
    // the layout fed to gemm_oq4_grouped_f16_wmma (Phase 4 W4A16).
    let signs1 = hipfire_quantize::gen_fwht_signs(OQ_FWHT_SEED1, 256);
    let signs2 = hipfire_quantize::gen_fwht_signs(OQ_FWHT_SEED2, 256);
    let (m, k) = (2usize, 512usize);
    let ng = k / 256;
    let data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.02)
        .collect();
    let oq4 = hipfire_quantize::codecs::quantize_oq4g256(&data, &signs1, &signs2);
    assert_eq!(oq4.len(), m * ng * 130);
    let combined = pack_oq4_arch_combined(&oq4, m, k);
    let packed_bytes = m * (k / 2);
    let scales_bytes = m * ng * 4;
    assert_eq!(combined.len(), packed_bytes + scales_bytes + m * ng * 132);
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * 130;
            // nibble region
            let dst = r * (k / 2) + g * 128;
            assert_eq!(&combined[dst..dst + 128], &oq4[src + 2..src + 130]);
            // f32 scale region == f16->f32 of the source f16 scale
            let want =
                crate::quant_decode::f16_bits_to_f32(u16::from_le_bytes([oq4[src], oq4[src + 1]]));
            let so = packed_bytes + (r * ng + g) * 4;
            let got = f32::from_le_bytes([
                combined[so],
                combined[so + 1],
                combined[so + 2],
                combined[so + 3],
            ]);
            assert_eq!(got, want);
        }
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
    assert!(
        s4 > 15.0,
        "oq4 round-trip SQNR too low ({s4:.1} dB) — layout mismatch?"
    );

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
    // hipfire_runtime::quant::dequant_q4k (a different crate) — this guards
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
    assert!(
        sqnr > 15.0,
        "Q4_K round-trip SQNR too low ({sqnr:.1} dB) — layout mismatch?"
    );
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
        let base = data
            .iter()
            .position(|v| (*v - group[0]).abs() < 1e-12)
            .unwrap();
        for (k, &orig) in group.iter().enumerate() {
            assert!((decoded[base + k] - orig).abs() <= step * 0.5 + 1e-4);
        }
    }
}
