// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Raw-order parity and shape-distributed microbench for the gfx1151 bounded
//! batched indexer top-K used by DeepSeek V4 prefill.
//!
//! The portable O(N^2) rank-count and gfx1151 bounded O(N log^2 K) kernels run
//! in the same process against the same frozen score buffer. Every ordered i32
//! slot must match both the incumbent and a host oracle. Cases include the
//! identity boundary, tied finite scores, real `-inf` padding inside each row,
//! storage stride larger than the live iteration bound, and long-context
//! product shapes.

use rdna_compute::{DType, Gpu, GpuTensor};

const K: usize = 512;
const POISON_REF: i32 = -7777;
const POISON_CAND: i32 = -8888;

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

fn download_i32(gpu: &Gpu, tensor: &GpuTensor, len: usize) -> Vec<i32> {
    let mut values = vec![0i32; len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("download i32 tensor");
    values
}

fn splitmix_score(row: usize, index: usize) -> f32 {
    let mut z = (row as u64)
        .wrapping_mul(0xD6E8_FEB8_6659_FD93)
        .wrapping_add(index as u64)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Deliberately retain only 12 score bits. Exact ties exercise the stable
    // source-index tiebreak heavily, just as quantized model scores can.
    ((z >> 40) & 0xfff) as f32 / 4096.0
}

struct Case {
    name: &'static str,
    batch: usize,
    n_iter: usize,
    n_stride: usize,
    growing_valid_rows: bool,
}

fn build_scores(case: &Case) -> Vec<f32> {
    let mut scores = vec![f32::NEG_INFINITY; case.batch * case.n_stride];
    for batch in 0..case.batch {
        let valid = if case.growing_valid_rows {
            // Compressed history grows by roughly one row per four prompt
            // tokens. Earlier rows in the same chunk therefore have a real
            // -inf suffix even though the launch uses the chunk-wide n_iter.
            case.n_iter.saturating_sub((case.batch - 1 - batch).div_ceil(4))
        } else {
            case.n_iter
        };
        for index in 0..valid {
            scores[batch * case.n_stride + index] = splitmix_score(batch, index);
        }
    }
    scores
}

fn oracle(scores: &[f32], case: &Case) -> Vec<i32> {
    let mut out = vec![-1i32; case.batch * K];
    for batch in 0..case.batch {
        if case.n_iter <= K {
            for rank in 0..case.n_iter {
                out[batch * K + rank] = rank as i32;
            }
            continue;
        }
        let row = &scores[batch * case.n_stride..batch * case.n_stride + case.n_iter];
        let mut indices: Vec<usize> = (0..case.n_iter).collect();
        indices.sort_unstable_by(|&lhs, &rhs| {
            row[rhs]
                .total_cmp(&row[lhs])
                .then_with(|| lhs.cmp(&rhs))
        });
        for (rank, index) in indices.into_iter().take(K).enumerate() {
            out[batch * K + rank] = index as i32;
        }
    }
    out
}

fn launch_reference(gpu: &mut Gpu, scores: &GpuTensor, out: &GpuTensor, case: &Case) {
    gpu.indexer_top_k_batched(
        scores,
        out,
        1,
        case.n_stride as i32,
        case.n_iter as i32,
        K as i32,
        K.min(case.n_iter) as i32,
        case.batch as i32,
    )
    .expect("launch portable batched top-K");
}

fn launch_candidate(gpu: &mut Gpu, scores: &GpuTensor, out: &GpuTensor, case: &Case) {
    gpu.indexer_top_k_batched_bounded_gfx1151(
        scores,
        out,
        1,
        case.n_stride as i32,
        case.n_iter as i32,
        K as i32,
        K.min(case.n_iter) as i32,
        case.batch as i32,
    )
    .expect("launch gfx1151 bounded batched top-K");
}

fn first_diff(lhs: &[i32], rhs: &[i32]) -> Option<(usize, i32, i32)> {
    lhs.iter()
        .zip(rhs)
        .enumerate()
        .find_map(|(slot, (&a, &b))| (a != b).then_some((slot, a, b)))
}

fn timing_plan(case: &Case) -> (usize, usize) {
    if case.batch * case.n_iter <= 16 * 2048 {
        (3, 10)
    } else {
        (1, 3)
    }
}

fn time_arm(
    gpu: &mut Gpu,
    scores: &GpuTensor,
    out: &GpuTensor,
    case: &Case,
    candidate: bool,
) -> f64 {
    let (warmups, iterations) = timing_plan(case);
    for _ in 0..warmups {
        if candidate {
            launch_candidate(gpu, scores, out, case);
        } else {
            launch_reference(gpu, scores, out, case);
        }
    }
    gpu.hip.device_synchronize().expect("sync warmup");
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        if candidate {
            launch_candidate(gpu, scores, out, case);
        } else {
            launch_reference(gpu, scores, out, case);
        }
    }
    gpu.hip.device_synchronize().expect("sync timing");
    start.elapsed().as_secs_f64() * 1e3 / iterations as f64
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
            name: "identity_n512_b7_stride529",
            batch: 7,
            n_iter: 512,
            n_stride: 529,
            growing_valid_rows: false,
        },
        Case {
            name: "boundary_n513_b7_stride544",
            batch: 7,
            n_iter: 513,
            n_stride: 544,
            growing_valid_rows: true,
        },
        Case {
            name: "mid_n2048_b7_stride2112",
            batch: 7,
            n_iter: 2048,
            n_stride: 2112,
            growing_valid_rows: true,
        },
        Case {
            name: "ctx21k_shape_n5338_b32_stride5376",
            batch: 32,
            n_iter: 5338,
            n_stride: 5376,
            growing_valid_rows: true,
        },
        Case {
            name: "ctx32k_shape_n8192_b16_stride8192",
            batch: 16,
            n_iter: 8192,
            n_stride: 8192,
            growing_valid_rows: true,
        },
    ];

    let mut failed = false;
    for case in &cases {
        let host_scores = build_scores(case);
        let scores = gpu
            .upload_f32(&host_scores, &[host_scores.len()])
            .expect("upload scores");
        let slots = case.batch * K;
        let reference = upload_i32(&mut gpu, &vec![POISON_REF; slots]);
        let candidate = upload_i32(&mut gpu, &vec![POISON_CAND; slots]);

        launch_reference(&mut gpu, &scores, &reference, case);
        launch_candidate(&mut gpu, &scores, &candidate, case);
        gpu.hip.device_synchronize().expect("sync parity");

        let reference_host = download_i32(&gpu, &reference, slots);
        let candidate_host = download_i32(&gpu, &candidate, slots);
        let oracle_host = oracle(&host_scores, case);
        let ref_diff = first_diff(&reference_host, &oracle_host);
        let cand_diff = first_diff(&candidate_host, &reference_host);
        let poison_ref = reference_host.iter().filter(|&&v| v == POISON_REF).count();
        let poison_cand = candidate_host.iter().filter(|&&v| v == POISON_CAND).count();

        let reference_ms = time_arm(&mut gpu, &scores, &reference, case, false);
        let candidate_ms = time_arm(&mut gpu, &scores, &candidate, case, true);
        let speedup = reference_ms / candidate_ms;
        let pass = ref_diff.is_none()
            && cand_diff.is_none()
            && poison_ref == 0
            && poison_cand == 0;
        failed |= !pass;
        eprintln!(
            "CASE name={} batch={} n_iter={} n_stride={} raw_i32_equal={} \
             reference_vs_oracle={} poison_ref={} poison_candidate={} \
             reference_ms={:.6} candidate_ms={:.6} speedup={:.3}x verdict={}",
            case.name,
            case.batch,
            case.n_iter,
            case.n_stride,
            cand_diff.is_none(),
            ref_diff.is_none(),
            poison_ref,
            poison_cand,
            reference_ms,
            candidate_ms,
            speedup,
            if pass { "PASS" } else { "FAIL" },
        );
        if let Some((slot, expected, actual)) = ref_diff {
            eprintln!(
                "  REFERENCE_DIFF slot={slot} oracle={actual} reference={expected}"
            );
        }
        if let Some((slot, reference_value, candidate_value)) = cand_diff {
            eprintln!(
                "  CANDIDATE_DIFF slot={slot} reference={reference_value} candidate={candidate_value}"
            );
        }
    }

    if failed {
        eprintln!("OVERALL=FAIL");
        std::process::exit(1);
    }
    eprintln!("OVERALL=PASS");
}
