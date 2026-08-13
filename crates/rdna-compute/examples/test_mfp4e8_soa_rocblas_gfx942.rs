// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx942 qt=35 staged-rocBLAS correctness and production-shape screen.

use rdna_compute::{DType, Gpu, GpuTensor};

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn bytes_of_f32(values: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain data and the byte slice cannot outlive `values`.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 };
    }
    let new_exp = exp - 127 + 15;
    if new_exp < 1 {
        return sign << 15;
    }
    if new_exp > 30 {
        return (sign << 15) | (0x1f << 10);
    }
    let discarded = mant & 0x1fff;
    let mut new_mant = (mant >> 13) as u16;
    if discarded > 0x1000 || (discarded == 0x1000 && new_mant & 1 != 0) {
        new_mant += 1;
    }
    let mut exp_bits = new_exp as u16;
    if new_mant == 0x400 {
        new_mant = 0;
        exp_bits += 1;
    }
    (sign << 15) | (exp_bits << 10) | new_mant
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = -1i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((e + 127 - 14) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn e4m3_scale(byte: u8) -> f32 {
    let exp = (byte >> 3) & 0xf;
    let mant = byte & 7;
    if exp == 0 {
        return 0.015625 * mant as f32 * 0.125;
    }
    if exp == 0xf && mant == 7 {
        return 448.0;
    }
    f32::from_bits((exp as u32 + 120) << 23) * (1.0 + mant as f32 * 0.125)
}

fn decode_e8(codeword: u32) -> [f32; 8] {
    let coset = (codeword >> 31) & 1;
    let mut e = [0u32; 8];
    let mut sum = 0u32;
    for (i, slot) in e.iter_mut().enumerate().take(7) {
        *slot = (codeword >> (4 * i)) & 0xf;
        sum += *slot;
    }
    let p7 = ((codeword >> 28) & 7) << 1;
    e[7] = p7 | ((sum + p7) & 1);
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        out[i] = (e[i] as i32 - 7) as f32 + if coset != 0 { 0.5 } else { 0.0 };
    }
    out
}

fn build_e8_soa(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let row_bytes = 16 + scale_padded + blocks * 16;
    let mut data = vec![0u8; m * row_bytes];
    let mut state = seed;
    const ROW_SCALES: [u16; 4] = [0x3400, 0x3800, 0x3c00, 0x4000];
    for row in 0..m {
        let off = row * row_bytes;
        data[off..off + 2].copy_from_slice(&ROW_SCALES[row % ROW_SCALES.len()].to_le_bytes());
        data[off + 4..off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        data[off + 6] = 0x06;
        for block in 0..blocks {
            data[off + 16 + block] = 0x38 | (lcg(&mut state) as u8 & 7);
            let cw_off = off + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                data[cw_off + slot * 4..cw_off + slot * 4 + 4]
                    .copy_from_slice(&lcg(&mut state).to_le_bytes());
            }
        }
    }
    data
}

fn expected_shadow_bits(packed: &[u8], m: usize, k: usize) -> Vec<u16> {
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let row_bytes = 16 + scale_padded + blocks * 16;
    let mut out = vec![0u16; m * k];
    for row in 0..m {
        let row_off = row * row_bytes;
        let row_scale = f16_bits_to_f32(u16::from_le_bytes([
            packed[row_off],
            packed[row_off + 1],
        ]));
        for block in 0..blocks {
            let scale = row_scale * e4m3_scale(packed[row_off + 16 + block]) * 0.88;
            let cw_off = row_off + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                let base = cw_off + slot * 4;
                let codeword = u32::from_le_bytes([
                    packed[base],
                    packed[base + 1],
                    packed[base + 2],
                    packed[base + 3],
                ]);
                for (i, value) in decode_e8(codeword).iter().enumerate() {
                    let dst = row * k + block * 32 + slot * 8 + i;
                    out[dst] = f32_to_f16_bits(scale * value);
                }
            }
        }
    }
    out
}

fn upload_weight(gpu: &Gpu, packed: &[u8], m: usize, k: usize) -> GpuTensor {
    let mut weight = gpu
        .upload_raw(packed, &[packed.len()])
        .expect("upload qt35 SoA weight");
    weight.shape = vec![m, k];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn check_shadow(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    packed: &[u8],
    m: usize,
    k: usize,
) -> GpuTensor {
    let expanded = gpu
        .alloc_tensor(&[m * k], DType::F16)
        .expect("allocate FP16 shadow oracle");
    gpu.dequantize_mfp4g32_e8_soa_to_f16_gfx942(&weight.buf, &expanded.buf, m, k)
        .expect("launch qt35 SoA dequant");
    gpu.hip.device_synchronize().expect("shadow sync");
    let expected = expected_shadow_bits(packed, m, k);
    let mut raw = vec![0u8; m * k * 2];
    gpu.hip
        .memcpy_dtoh(&mut raw, &expanded.buf)
        .expect("download FP16 shadow");
    let mut mismatches = 0usize;
    let mut first = None;
    for (i, &want) in expected.iter().enumerate() {
        let got = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
        if got != want {
            mismatches += 1;
            first.get_or_insert((i, got, want));
        }
    }
    assert_eq!(
        mismatches, 0,
        "FP16 shadow mismatch M={m} K={k}: count={mismatches} first={first:?}"
    );
    println!("PASS shadow raw bits M={m} K={k}: {} values", m * k);
    expanded
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.5)
        .collect()
}

fn direct_batch(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch: usize,
) {
    for b in 0..batch {
        let xb = x.sub_offset(b * k, k);
        let yb = y.sub_offset(b * m, m);
        gpu.gemv_mfp4g32_e8_soa(weight, &xb, &yb, m, k)
            .expect("incumbent qt35 GEMV");
    }
}

fn direct_rocblas(
    gpu: &mut Gpu,
    shadow: &GpuTensor,
    x: &GpuTensor,
    x_f16: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch: usize,
) {
    gpu.deepseek4_convert_f32_to_f16(x, x_f16, (batch * k) as i64)
        .expect("direct F32-to-F16 conversion");
    gpu.rocblas_gemm_hfq4_prefill(&shadow.buf, &x_f16.buf, &y.buf, m, batch, k)
        .expect("direct FP16 rocBLAS GEMM");
}

fn time_gpu_ms<F>(gpu: &mut Gpu, trials: usize, mut launch: F) -> Vec<f64>
where
    F: FnMut(&mut Gpu),
{
    let mut values = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = gpu.hip.event_create().expect("create start event");
        let stop = gpu.hip.event_create().expect("create stop event");
        gpu.hip.event_record(&start, None).expect("record start event");
        launch(gpu);
        gpu.hip.event_record(&stop, None).expect("record stop event");
        gpu.hip.event_synchronize(&stop).expect("wait stop event");
        values.push(
            gpu.hip
                .event_elapsed_ms(&start, &stop)
                .expect("elapsed GPU time") as f64,
        );
        gpu.hip.event_destroy(start).expect("destroy start event");
        gpu.hip.event_destroy(stop).expect("destroy stop event");
    }
    values
}

fn median_ms(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn hash_bits(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325u64, |mut h, v| {
        for byte in v.to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    })
}

fn assert_bits_equal(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    let first = got
        .iter()
        .zip(want)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits());
    assert!(first.is_none(), "{label} bit mismatch: {first:?}");
    println!("PASS {label}: {} f32 values bit-identical", got.len());
}

fn compare_outputs(candidate: &[f32], reference: &[f32], m: usize, batch: usize) {
    let mut max_abs = 0.0f64;
    let mut err2 = 0.0f64;
    let mut ref2 = 0.0f64;
    let mut cand2 = 0.0f64;
    let mut dot = 0.0f64;
    let mut nonfinite = 0usize;
    let mut tolerance_failures = 0usize;
    let mut first_tolerance_failure = None;
    for (i, (&got, &want)) in candidate.iter().zip(reference).enumerate() {
        if !got.is_finite() || !want.is_finite() {
            nonfinite += 1;
            continue;
        }
        let g = got as f64;
        let w = want as f64;
        let d = g - w;
        max_abs = max_abs.max(d.abs());
        let tolerance = 2.0e-3 + 1.0e-3 * w.abs();
        if d.abs() > tolerance {
            tolerance_failures += 1;
            first_tolerance_failure.get_or_insert((i, got, want, d.abs(), tolerance));
        }
        err2 += d * d;
        ref2 += w * w;
        cand2 += g * g;
        dot += g * w;
    }
    let nrmse = (err2 / ref2.max(1.0e-30)).sqrt();
    let rms_ref = (ref2 / candidate.len() as f64).sqrt();
    let cosine = dot / (ref2 * cand2).sqrt().max(1.0e-30);
    let mut top1_matches = 0usize;
    for b in 0..batch {
        let lo = b * m;
        let hi = lo + m;
        let ref_top = reference[lo..hi]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let got_top = candidate[lo..hi]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        top1_matches += usize::from(ref_top == got_top);
    }
    let top1 = top1_matches as f64 / batch as f64;
    println!(
        "NUMERICS max_abs={max_abs:.6e} rms_ref={rms_ref:.6e} rel_l2={nrmse:.6e} cosine={cosine:.9} top1_diag={top1:.6} nonfinite={nonfinite} strict_local_diag_failures={tolerance_failures} first_local={first_tolerance_failure:?}"
    );
    assert_eq!(nonfinite, 0, "non-finite candidate/reference values");
    assert!(max_abs <= 0.1, "max absolute error {max_abs:.6e} exceeds 0.1");
    assert!(nrmse <= 1.0e-3, "relative L2 {nrmse:.6e} exceeds 1e-3");
    assert!(cosine >= 0.999999, "cosine {cosine:.9} below 0.999999");
}

fn main() {
    const M: usize = 32768;
    const K: usize = 1024;
    const B: usize = 1024;
    const TRIALS: usize = 3;

    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx942", "this oracle requires gfx942");
    assert!(gpu.rocblas.is_some(), "rocBLAS must be enabled");
    // Geometry tail: the staging grid must cover 24 blocks, not floor(K/512).
    let tail_packed = build_e8_soa(17, 768, 0x7140);
    let tail_weight = upload_weight(&gpu, &tail_packed, 17, 768);
    let _tail_shadow = check_shadow(&mut gpu, &tail_weight, &tail_packed, 17, 768);

    let packed = build_e8_soa(M, K, 0x9420);
    let weight = upload_weight(&gpu, &packed, M, K);
    let shadow = check_shadow(&mut gpu, &weight, &packed, M, K);

    let x1_host = make_x(B * K, 0xa001);
    let x2_host = make_x(B * K, 0xa002);
    let x = gpu.upload_f32(&x1_host, &[B, K]).expect("upload x1");
    let candidate = gpu.zeros(&[B * M], DType::F32).expect("candidate y");
    let reference = gpu.zeros(&[B * M], DType::F32).expect("reference y");
    let direct_fp16 = gpu.zeros(&[B * M], DType::F32).expect("direct rocBLAS y");
    let x_f16 = gpu
        .alloc_tensor(&[B * K], DType::F16)
        .expect("direct FP16 x");

    let cold_ms = time_gpu_ms(&mut gpu, 1, |gpu| {
        let used = gpu
            .rocblas_gemm_mfp4e8_soa_prefill_auto(&weight, &x, &candidate, M, K, B)
            .expect("first staged rocBLAS call");
        assert!(
            used,
            "route did not engage; set HIPFIRE_DEEPSEEK4_GFX942_E8_ROCBLAS=1 and HIPFIRE_ROCBLAS_OFF=0"
        );
    })[0];
    println!("COLD staged shadow+convert+rocBLAS: {cold_ms:.3} ms");
    let y1_first = gpu
        .download_f32(&candidate.sub_offset(0, M))
        .expect("download first x1 row");

    // Rewrite the exact same allocation. A pointer-keyed X cache would now be stale.
    gpu.hip
        .memcpy_htod(&x.buf, bytes_of_f32(&x2_host))
        .expect("overwrite x allocation");
    assert!(
        gpu.rocblas_gemm_mfp4e8_soa_prefill_auto(&weight, &x, &candidate, M, K, B)
            .expect("second staged rocBLAS call")
    );
    direct_rocblas(&mut gpu, &shadow, &x, &x_f16, &direct_fp16, M, K, B);
    direct_batch(&mut gpu, &weight, &x, &reference, M, K, B);
    gpu.hip.device_synchronize().expect("correctness warm sync");
    let y2_first = gpu
        .download_f32(&candidate.sub_offset(0, M))
        .expect("download first x2 row");
    assert_ne!(
        hash_bits(&y1_first),
        hash_bits(&y2_first),
        "rewriting the stable X allocation did not change output"
    );
    println!(
        "PASS stable-pointer rewrite: x1={:016x} x2={:016x}",
        hash_bits(&y1_first),
        hash_bits(&y2_first)
    );

    let candidate_times = time_gpu_ms(&mut gpu, TRIALS, |gpu| {
        assert!(
            gpu.rocblas_gemm_mfp4e8_soa_prefill_auto(&weight, &x, &candidate, M, K, B)
                .expect("timed staged rocBLAS call")
        );
    });
    let direct_fp16_times = time_gpu_ms(&mut gpu, TRIALS, |gpu| {
        direct_rocblas(gpu, &shadow, &x, &x_f16, &direct_fp16, M, K, B);
    });
    let direct_times = time_gpu_ms(&mut gpu, TRIALS, |gpu| {
        direct_batch(gpu, &weight, &x, &reference, M, K, B);
    });

    let candidate_host = gpu.download_f32(&candidate).expect("download candidate");
    let direct_fp16_host = gpu
        .download_f32(&direct_fp16)
        .expect("download direct rocBLAS");
    let reference_host = gpu.download_f32(&reference).expect("download reference");
    assert_bits_equal("auto vs assembled FP16 rocBLAS", &candidate_host, &direct_fp16_host);
    compare_outputs(&candidate_host, &reference_host, M, B);

    let candidate_ms = median_ms(candidate_times.clone());
    let direct_fp16_ms = median_ms(direct_fp16_times.clone());
    let direct_ms = median_ms(direct_times.clone());
    let speedup = direct_ms / candidate_ms;
    let assembly_ratio = candidate_ms / direct_fp16_ms;
    println!(
        "PERF M={M} K={K} B={B} incumbent_ms={direct_times:?} assembled_fp16_ms={direct_fp16_times:?} auto_ms={candidate_times:?} median_speedup={speedup:.3}x auto_over_assembled={assembly_ratio:.3}x"
    );
    assert!(
        (0.90..=1.10).contains(&assembly_ratio),
        "auto/assembled hot-time ratio {assembly_ratio:.3} outside 10%"
    );
    assert!(speedup >= 2.0, "staged rocBLAS speedup {speedup:.3}x below 2x screen");
    println!("PASS gfx942 qt35 staged-rocBLAS oracle");
}
