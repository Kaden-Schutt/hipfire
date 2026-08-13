// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx942 native two-job/LDS-X qt35 mechanism channel.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

const SRC: &str = include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_twojob_lds8.gfx942.hip");
const MODULE: &str = "gemv_mfp4g32_e8_soa_twojob_lds8_gfx942_candidate";
const SYMBOL: &str = "gemv_mfp4g32_e8_soa_twojob_lds8_gfx942_candidate";
const M: usize = 2048;
const K: usize = 4096;
const GUARDS: usize = 32;
const POISON: f32 = 12345.625;
const TRIALS: usize = 7;
const REPEATS: usize = 24;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn row_bytes() -> usize {
    let blocks = K / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    16 + scale_padded + blocks * 16
}

fn build_e8_soa(seed: u64) -> Vec<u8> {
    let blocks = K / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let stride = row_bytes();
    let mut packed = vec![0u8; M * stride];
    let mut state = seed;

    for row in 0..M {
        let off = row * stride;
        let row_scale = [0x3400u16, 0x3800, 0x3c00, 0x4000][row & 3];
        packed[off..off + 2].copy_from_slice(&row_scale.to_le_bytes());
        packed[off + 4..off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        packed[off + 6] = 0x06;
        for block in 0..blocks {
            packed[off + 16 + block] = [0x01, 0x07, 0x38, 0x7f][block & 3];
            let cw = off + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                let codeword = match (block, slot) {
                    (0, 0) => 0x0000_0000,
                    (0, 1) => 0x8000_0000,
                    (0, 2) => 0x7777_7777,
                    (0, 3) => 0xffff_ffff,
                    _ => lcg(&mut state),
                };
                packed[cw + slot * 4..cw + slot * 4 + 4].copy_from_slice(&codeword.to_le_bytes());
            }
        }
    }
    packed
}

fn upload_weight(gpu: &Gpu, bytes: &[u8]) -> GpuTensor {
    let mut weight = gpu
        .upload_raw(bytes, &[bytes.len()])
        .expect("upload qt35 weight");
    weight.shape = vec![M, K];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn make_x() -> Vec<f32> {
    let mut state = 0x9420_1d58_cafe_u64;
    (0..K)
        .map(|i| match i & 31 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            4 => 1.0,
            5 => -1.0,
            _ => (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.25,
        })
        .collect()
}

fn guarded_output(gpu: &mut Gpu, label: &str) -> (GpuTensor, GpuTensor) {
    let poisoned = vec![POISON; M + GUARDS];
    let backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .unwrap_or_else(|e| panic!("allocate {label} + guard: {e:?}"));
    let view = backing.sub_offset(0, M);
    (backing, view)
}

fn download_checked(gpu: &Gpu, label: &str, backing: &GpuTensor) -> Vec<f32> {
    let host = gpu
        .download_f32(backing)
        .unwrap_or_else(|e| panic!("download {label}: {e:?}"));
    assert!(
        host[..M]
            .iter()
            .all(|&value| value.to_bits() != POISON.to_bits()),
        "{label} left poisoned rows"
    );
    assert!(
        host[M..]
            .iter()
            .all(|&value| value.to_bits() == POISON.to_bits()),
        "{label} overwrote output guard"
    );
    host[..M].to_vec()
}

fn launch_candidate(
    gpu: &Gpu,
    weight0: &GpuTensor,
    weight1: &GpuTensor,
    x: &GpuTensor,
    y0: &GpuTensor,
    y1: &GpuTensor,
) {
    let mut kb = KernargBlob::new();
    kb.push_ptr(weight0.buf.as_ptr() as *const c_void);
    kb.push_ptr(weight1.buf.as_ptr() as *const c_void);
    kb.push_ptr(x.buf.as_ptr() as *const c_void);
    kb.push_ptr(y0.buf.as_ptr() as *const c_void);
    kb.push_ptr(y1.buf.as_ptr() as *const c_void);
    kb.pad_to(16);
    gpu.launch_kernel_blob(SYMBOL, [512, 1, 1], [512, 1, 1], 0, kb.as_mut_slice())
        .expect("launch native two-job candidate");
}

fn launch_reference(
    gpu: &mut Gpu,
    weight0: &GpuTensor,
    weight1: &GpuTensor,
    x: &GpuTensor,
    y0: &GpuTensor,
    y1: &GpuTensor,
) {
    gpu.gemv_mfp4g32_e8_soa(weight0, x, y0, M, K)
        .expect("reference w1");
    gpu.gemv_mfp4g32_e8_soa(weight1, x, y1, M, K)
        .expect("reference w3");
}

fn elapsed_ms<F>(gpu: &mut Gpu, mut launch: F) -> f64
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("create start event");
    let stop = gpu.hip.event_create().expect("create stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..REPEATS {
        launch(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("wait stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed ms") as f64 / REPEATS as f64;
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    ms
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn compare_tight(label: &str, reference: &[f32], candidate: &[f32]) {
    let mut raw_bit_mismatches = 0usize;
    let mut numerical_violations = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut first_bad = None;
    for (row, (&a, &b)) in reference.iter().zip(candidate).enumerate() {
        raw_bit_mismatches += usize::from(a.to_bits() != b.to_bits());
        let abs = (b - a).abs();
        let rel = abs / a.abs().max(1.0e-12);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        // Predeclared before the run: different full-wave reduction order is
        // accepted only inside this tight FP32 accumulation envelope.
        let tolerance = 5.0e-3 + 1.0e-4 * a.abs();
        if !a.is_finite() || !b.is_finite() || abs > tolerance {
            numerical_violations += 1;
            first_bad.get_or_insert((row, a, b, abs, tolerance));
        }
    }
    println!(
        "CHANNEL {label} values={M} raw_bit_mismatches={raw_bit_mismatches} \
         max_abs={max_abs:.9e} max_rel={max_rel:.9e} \
         numerical_violations={numerical_violations} first_bad={first_bad:?}"
    );
    assert_eq!(
        numerical_violations, 0,
        "{label} failed predeclared tolerance: {first_bad:?}"
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(
        gpu.arch, "gfx942",
        "this mechanism channel requires exact gfx942"
    );
    gpu.ensure_kernel_public(MODULE, SRC, SYMBOL)
        .expect("compile native gfx942 scheduler");

    assert_eq!(row_bytes(), 2192);
    let packed0 = build_e8_soa(0x9420_a001_0000_u64);
    let packed1 = build_e8_soa(0x9420_b003_0000_u64);
    assert_eq!(packed0.len(), 4_489_216);
    assert_eq!(packed1.len(), 4_489_216);
    let weight0 = upload_weight(&gpu, &packed0);
    let weight1 = upload_weight(&gpu, &packed1);
    let x = gpu.upload_f32(&make_x(), &[K]).expect("upload shared x");

    let (ref0_backing, ref0) = guarded_output(&mut gpu, "reference w1");
    let (ref1_backing, ref1) = guarded_output(&mut gpu, "reference w3");
    let (cand0_backing, cand0) = guarded_output(&mut gpu, "candidate w1");
    let (cand1_backing, cand1) = guarded_output(&mut gpu, "candidate w3");

    launch_reference(&mut gpu, &weight0, &weight1, &x, &ref0, &ref1);
    launch_candidate(&gpu, &weight0, &weight1, &x, &cand0, &cand1);
    gpu.hip.device_synchronize().expect("channel sync");

    let ref0_host = download_checked(&gpu, "reference w1", &ref0_backing);
    let ref1_host = download_checked(&gpu, "reference w3", &ref1_backing);
    let cand0_host = download_checked(&gpu, "candidate w1", &cand0_backing);
    let cand1_host = download_checked(&gpu, "candidate w3", &cand1_backing);
    compare_tight("w1", &ref0_host, &cand0_host);
    compare_tight("w3", &ref1_host, &cand1_host);

    for _ in 0..3 {
        launch_reference(&mut gpu, &weight0, &weight1, &x, &ref0, &ref1);
        launch_candidate(&gpu, &weight0, &weight1, &x, &cand0, &cand1);
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    let mut reference_ms = Vec::with_capacity(TRIALS);
    let mut candidate_ms = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        if trial & 1 == 0 {
            reference_ms.push(elapsed_ms(&mut gpu, |gpu| {
                launch_reference(gpu, &weight0, &weight1, &x, &ref0, &ref1)
            }));
            candidate_ms.push(elapsed_ms(&mut gpu, |gpu| {
                launch_candidate(gpu, &weight0, &weight1, &x, &cand0, &cand1)
            }));
        } else {
            candidate_ms.push(elapsed_ms(&mut gpu, |gpu| {
                launch_candidate(gpu, &weight0, &weight1, &x, &cand0, &cand1)
            }));
            reference_ms.push(elapsed_ms(&mut gpu, |gpu| {
                launch_reference(gpu, &weight0, &weight1, &x, &ref0, &ref1)
            }));
        }
    }

    let reference_median = median(&mut reference_ms);
    let candidate_median = median(&mut candidate_ms);
    let semantic_bytes = packed0.len() + packed1.len() + K * 4 + 2 * M * 4;
    println!(
        "MICRO shape=two_job_shared_x M={M} K={K} trials={TRIALS} repeats={REPEATS} \
         reference_two_launch_ms={reference_median:.6} \
         candidate_one_launch_ms={candidate_median:.6} speedup={:.4}x \
         reference_effective_GBps={:.2} candidate_effective_GBps={:.2} \
         grid=512 block=512 waves_per_block=8 x_global_load_reduction=8x \
         launches_saved_per_token=43 weight_bytes_per_token=386072576",
        reference_median / candidate_median,
        semantic_bytes as f64 / reference_median / 1.0e6,
        semantic_bytes as f64 / candidate_median / 1.0e6,
    );
}
