// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — occurrence-shaped gfx1201 shared-activation E8 pair screen.

use rdna_compute::{DType, Gpu, GpuTensor};

const TRIALS: usize = 7;
const WORKING_SET_BYTES: usize = 160 * 1024 * 1024;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn row_bytes(k: usize) -> usize {
    let blocks = k / 32;
    16 + blocks.div_ceil(16) * 16 + blocks * 16
}

fn build_weight(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let stride = row_bytes(k);
    let mut packed = vec![0u8; rows * stride];
    let mut state = seed;
    for row in 0..rows {
        let off = row * stride;
        let row_scale = [0x3400u16, 0x3800, 0x3c00, 0x4000][row & 3];
        packed[off..off + 2].copy_from_slice(&row_scale.to_le_bytes());
        packed[off + 4..off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        packed[off + 6] = 0x06;
        for block in 0..blocks {
            packed[off + 16 + block] = [0x01, 0x07, 0x38, 0x7e][block & 3];
            let codewords = off + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                let word = lcg(&mut state);
                let dst = codewords + slot * 4;
                packed[dst..dst + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
    }
    packed
}

fn upload_weight(gpu: &Gpu, rows: usize, k: usize, seed: u64) -> GpuTensor {
    let bytes = build_weight(rows, k, seed);
    let mut weight = gpu
        .upload_raw(&bytes, &[bytes.len()])
        .expect("upload weight");
    weight.shape = vec![rows, k];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn sequential(
    gpu: &mut Gpu,
    weights: &[GpuTensor; 2],
    x: &GpuTensor,
    outputs: &[GpuTensor; 2],
    m0: usize,
    m1: usize,
    k: usize,
) {
    gpu.gemv_mfp4g32_e8_soa(&weights[0], x, &outputs[0], m0, k)
        .expect("sequential 0");
    gpu.gemv_mfp4g32_e8_soa(&weights[1], x, &outputs[1], m1, k)
        .expect("sequential 1");
}

fn paired(
    gpu: &mut Gpu,
    weights: &[GpuTensor; 2],
    x: &GpuTensor,
    outputs: &[GpuTensor; 2],
    m0: usize,
    m1: usize,
    k: usize,
) {
    gpu.gemv_mfp4g32_e8_soa_shared_pair_gfx1201(
        &weights[0],
        &weights[1],
        x,
        &outputs[0],
        &outputs[1],
        m0,
        m1,
        k,
    )
    .expect("paired");
}

fn event_us<F>(gpu: &mut Gpu, repeats: usize, mut launch: F) -> f64
where
    F: FnMut(&mut Gpu, usize),
{
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for repeat in 0..repeats {
        launch(gpu, repeat);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("sync stop");
    let us =
        gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64 * 1000.0 / repeats as f64;
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    us
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn run_case(
    gpu: &mut Gpu,
    label: &str,
    m0: usize,
    m1: usize,
    k: usize,
    occurrences: usize,
    seed: u64,
) {
    let pair_bytes = (m0 + m1) * row_bytes(k);
    let replicas = (WORKING_SET_BYTES / pair_bytes).max(3);
    let weight_sets = (0..replicas)
        .map(|replica| {
            [
                upload_weight(gpu, m0, k, seed + replica as u64 * 2),
                upload_weight(gpu, m1, k, seed + replica as u64 * 2 + 1),
            ]
        })
        .collect::<Vec<_>>();
    let mut rng = seed ^ 0x1201_5a5a;
    let x_host = (0..k)
        .map(|index| match index & 31 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            _ => (lcg(&mut rng) as f32 / u32::MAX as f32 - 0.5) * 0.25,
        })
        .collect::<Vec<_>>();
    let x = gpu.upload_f32(&x_host, &[k]).expect("upload x");
    let reference = [
        gpu.alloc_tensor(&[m0], DType::F32).expect("reference 0"),
        gpu.alloc_tensor(&[m1], DType::F32).expect("reference 1"),
    ];
    let candidate = [
        gpu.alloc_tensor(&[m0], DType::F32).expect("candidate 0"),
        gpu.alloc_tensor(&[m1], DType::F32).expect("candidate 1"),
    ];

    sequential(gpu, &weight_sets[0], &x, &reference, m0, m1, k);
    paired(gpu, &weight_sets[0], &x, &candidate, m0, m1, k);
    gpu.hip.device_synchronize().expect("correctness sync");
    let mut comparisons = 0usize;
    for job in 0..2 {
        let expected = gpu
            .download_f32(&reference[job])
            .expect("reference download");
        let actual = gpu
            .download_f32(&candidate[job])
            .expect("candidate download");
        for (row, (a, b)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label} raw-bit mismatch job={job} row={row}"
            );
            comparisons += 1;
        }
    }

    for weights in &weight_sets {
        sequential(gpu, weights, &x, &reference, m0, m1, k);
        paired(gpu, weights, &x, &candidate, m0, m1, k);
    }
    gpu.hip.device_synchronize().expect("warm sync");

    let mut sequential_us = Vec::with_capacity(TRIALS);
    let mut paired_us = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        let seq = |gpu: &mut Gpu, repeat: usize| {
            sequential(
                gpu,
                &weight_sets[repeat % replicas],
                &x,
                &reference,
                m0,
                m1,
                k,
            )
        };
        let pair = |gpu: &mut Gpu, repeat: usize| {
            paired(
                gpu,
                &weight_sets[repeat % replicas],
                &x,
                &candidate,
                m0,
                m1,
                k,
            )
        };
        if trial & 1 == 0 {
            sequential_us.push(event_us(gpu, replicas, seq));
            paired_us.push(event_us(gpu, replicas, pair));
        } else {
            paired_us.push(event_us(gpu, replicas, pair));
            sequential_us.push(event_us(gpu, replicas, seq));
        }
    }
    let sequential_us = median(&mut sequential_us);
    let paired_us = median(&mut paired_us);
    println!(
        "CASE label={label} M0={m0} M1={m1} K={k} occurrences={occurrences} replicas={replicas} working_set_bytes={} raw_bit_comparisons={comparisons} sequential_us={sequential_us:.6} paired_us={paired_us:.6} speedup_x={:.4} saved_us_per_occurrence={:.6} saved_ms_per_rank_token={:.6}",
        pair_bytes * replicas,
        sequential_us / paired_us,
        sequential_us - paired_us,
        (sequential_us - paired_us) * occurrences as f64 / 1000.0,
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "exact gfx1201 required");
    run_case(&mut gpu, "shared-rank01", 768, 768, 4096, 43, 0x1201_0100);
    run_case(&mut gpu, "shared-rank2", 512, 512, 4096, 43, 0x1201_0200);
    run_case(
        &mut gpu,
        "wq-indexer-rank01",
        12288,
        8192,
        1024,
        21,
        0x1201_0300,
    );
    run_case(
        &mut gpu,
        "wq-indexer-rank2",
        8192,
        8192,
        1024,
        21,
        0x1201_0400,
    );
}
