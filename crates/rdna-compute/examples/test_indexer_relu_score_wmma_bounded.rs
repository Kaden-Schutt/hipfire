// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Live-score parity and production-shape microbench for bounding the gfx1151
//! DeepSeek4 batched indexer WMMA grid by live compressed history rather than
//! its VMM allocation stride.

use rdna_compute::{DType, Gpu, GpuTensor};

const H: usize = 64;
const D: usize = 128;
const POISON_BITS: u32 = 0x7fc0_1234;

struct Case {
    name: &'static str,
    batch: usize,
    n_iter: usize,
    n_stride: usize,
}

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let tensor = gpu
        .alloc_tensor(&[values.len() * 4], DType::Raw)
        .expect("alloc i32 tensor");
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .expect("upload i32 tensor");
    tensor
}

fn deterministic_f32(len: usize, mul: usize, add: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let centered = ((i.wrapping_mul(mul).wrapping_add(add)) % 257) as i32 - 128;
            centered as f32 * scale
        })
        .collect()
}

fn n_per_batch(case: &Case) -> Vec<i32> {
    (0..case.batch)
        .map(|b| {
            // Ratio-4 compressed history grows through a 1024-token prefill
            // chunk. The final row reaches n_iter; earlier rows retain their
            // real causal suffix.
            case.n_iter.saturating_sub((case.batch - 1 - b).div_ceil(4)) as i32
        })
        .collect()
}

fn launch_reference(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    weights: &GpuTensor,
    n_valid: &GpuTensor,
    scores: &GpuTensor,
    case: &Case,
) {
    gpu.indexer_relu_score_wmma_batched_f32(
        q,
        k,
        weights,
        n_valid,
        scores,
        H as i32,
        D as i32,
        case.n_stride as i32,
        case.batch as i32,
    )
    .expect("launch capacity-grid reference");
}

fn launch_candidate(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    weights: &GpuTensor,
    n_valid: &GpuTensor,
    scores: &GpuTensor,
    case: &Case,
) {
    gpu.indexer_relu_score_wmma_batched_bounded_gfx1151(
        q,
        k,
        weights,
        n_valid,
        scores,
        H as i32,
        D as i32,
        case.n_stride as i32,
        case.n_iter as i32,
        case.batch as i32,
    )
    .expect("launch live-grid candidate");
}

#[allow(clippy::too_many_arguments)]
fn time_arm(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    weights: &GpuTensor,
    n_valid: &GpuTensor,
    scores: &GpuTensor,
    case: &Case,
    candidate: bool,
) -> f64 {
    let iterations = if case.n_stride <= 8192 { 3 } else { 2 };
    if candidate {
        launch_candidate(gpu, q, k, weights, n_valid, scores, case);
    } else {
        launch_reference(gpu, q, k, weights, n_valid, scores, case);
    }
    gpu.hip.device_synchronize().expect("sync warmup");
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        if candidate {
            launch_candidate(gpu, q, k, weights, n_valid, scores, case);
        } else {
            launch_reference(gpu, q, k, weights, n_valid, scores, case);
        }
    }
    gpu.hip.device_synchronize().expect("sync timing");
    start.elapsed().as_secs_f64() * 1e3 / iterations as f64
}

fn first_live_diff(reference: &[f32], candidate: &[f32], case: &Case) -> Option<(usize, u32, u32)> {
    let n_valid = n_per_batch(case);
    for (b, &valid) in n_valid.iter().enumerate() {
        for n in 0..valid as usize {
            let slot = b * case.n_stride + n;
            let reference_bits = reference[slot].to_bits();
            let candidate_bits = candidate[slot].to_bits();
            if reference_bits != candidate_bits {
                return Some((slot, reference_bits, candidate_bits));
            }
        }
    }
    None
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(
        gpu.arch, "gfx1151",
        "refuse: this product-candidate probe is gfx1151-only"
    );
    eprintln!("detected_arch={} arch_gate=PASS", gpu.arch);

    let cases = [
        Case {
            name: "capacity2048_live512_b1024",
            batch: 1024,
            n_iter: 512,
            n_stride: 2048,
        },
        Case {
            name: "ctx21k_capacity8192_live5338_b1024",
            batch: 1024,
            n_iter: 5338,
            n_stride: 8192,
        },
        Case {
            name: "ctx85k_capacity32768_live21423_b1024",
            batch: 1024,
            n_iter: 21423,
            n_stride: 32768,
        },
    ];

    let mut failed = false;
    for case in &cases {
        let q_host = deterministic_f32(case.batch * H * D, 17, 3, 1.0 / 512.0);
        let k_host = deterministic_f32(case.n_stride * D, 29, 11, 1.0 / 512.0);
        let weights_host = deterministic_f32(case.batch * H, 13, 7, 1.0 / 128.0);
        let n_valid_host = n_per_batch(case);
        let poison = f32::from_bits(POISON_BITS);
        let score_slots = case.batch * case.n_stride;

        let q = gpu
            .upload_f32(&q_host, &[case.batch, H, D])
            .expect("upload q");
        let k = gpu
            .upload_f32(&k_host, &[case.n_stride, D])
            .expect("upload k");
        let weights = gpu
            .upload_f32(&weights_host, &[case.batch, H])
            .expect("upload weights");
        let n_valid = upload_i32(&mut gpu, &n_valid_host);
        let reference = gpu
            .upload_f32(&vec![poison; score_slots], &[case.batch, case.n_stride])
            .expect("upload reference scores");
        let candidate = gpu
            .upload_f32(&vec![poison; score_slots], &[case.batch, case.n_stride])
            .expect("upload candidate scores");

        launch_reference(&mut gpu, &q, &k, &weights, &n_valid, &reference, case);
        launch_candidate(&mut gpu, &q, &k, &weights, &n_valid, &candidate, case);
        gpu.hip.device_synchronize().expect("sync parity");

        let reference_host = gpu.download_f32(&reference).expect("download reference");
        let candidate_host = gpu.download_f32(&candidate).expect("download candidate");
        let live_diff = first_live_diff(&reference_host, &candidate_host, case);
        let untouched_tail = candidate_host
            .iter()
            .filter(|value| value.to_bits() == POISON_BITS)
            .count();

        // ABBA order avoids giving either grid the same thermal/order slot in
        // both measurements while keeping this probe to seconds.
        let reference_a = time_arm(
            &mut gpu, &q, &k, &weights, &n_valid, &reference, case, false,
        );
        let candidate_a = time_arm(&mut gpu, &q, &k, &weights, &n_valid, &candidate, case, true);
        let candidate_b = time_arm(&mut gpu, &q, &k, &weights, &n_valid, &candidate, case, true);
        let reference_b = time_arm(
            &mut gpu, &q, &k, &weights, &n_valid, &reference, case, false,
        );
        let reference_ms = (reference_a + reference_b) * 0.5;
        let candidate_ms = (candidate_a + candidate_b) * 0.5;
        let pass = live_diff.is_none();
        failed |= !pass;
        eprintln!(
            "CASE name={} batch={} n_iter={} n_stride={} empty_grid_fraction={:.3} \
             live_raw_bits_equal={} untouched_tail={} reference_ms={:.6} \
             candidate_ms={:.6} speedup={:.3}x verdict={}",
            case.name,
            case.batch,
            case.n_iter,
            case.n_stride,
            1.0 - case.n_iter as f64 / case.n_stride as f64,
            live_diff.is_none(),
            untouched_tail,
            reference_ms,
            candidate_ms,
            reference_ms / candidate_ms,
            if pass { "PASS" } else { "FAIL" },
        );
        if let Some((slot, reference_bits, candidate_bits)) = live_diff {
            eprintln!(
                "  LIVE_DIFF slot={slot} reference=0x{reference_bits:08x} candidate=0x{candidate_bits:08x}"
            );
        }
    }

    if failed {
        eprintln!("OVERALL=FAIL");
        std::process::exit(1);
    }
    eprintln!("OVERALL=PASS");
}
