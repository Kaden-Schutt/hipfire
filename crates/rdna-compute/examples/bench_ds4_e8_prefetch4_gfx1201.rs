// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — occurrence-weighted gfx1201 E8 four-group prefetch screen.

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

fn run_case(gpu: &mut Gpu, label: &str, m: usize, k: usize, occurrences: usize, seed: u64) {
    let weight_bytes = m * row_bytes(k);
    let replicas = (WORKING_SET_BYTES / weight_bytes).clamp(1, 64);
    let weights = (0..replicas)
        .map(|replica| upload_weight(gpu, m, k, seed + replica as u64))
        .collect::<Vec<_>>();
    let mut rng = seed ^ 0x1201_a5a5;
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
    let reference = gpu.alloc_tensor(&[m], DType::F32).expect("reference");
    let candidate = gpu.alloc_tensor(&[m], DType::F32).expect("candidate");

    gpu.gemv_mfp4g32_e8_soa(&weights[0], &x, &reference, m, k)
        .expect("reference launch");
    gpu.gemv_mfp4g32_e8_soa_prefetch4_gfx1201(&weights[0], &x, &candidate, m, k)
        .expect("candidate launch");
    gpu.hip.device_synchronize().expect("correctness sync");
    let expected = gpu.download_f32(&reference).expect("reference download");
    let actual = gpu.download_f32(&candidate).expect("candidate download");
    for (row, (a, b)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label} raw-bit mismatch row={row}"
        );
    }

    for weight in &weights {
        gpu.gemv_mfp4g32_e8_soa(weight, &x, &reference, m, k)
            .expect("reference warm");
        gpu.gemv_mfp4g32_e8_soa_prefetch4_gfx1201(weight, &x, &candidate, m, k)
            .expect("candidate warm");
    }
    gpu.hip.device_synchronize().expect("warm sync");

    let mut reference_us = Vec::with_capacity(TRIALS);
    let mut candidate_us = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        let incumbent = |gpu: &mut Gpu, repeat: usize| {
            gpu.gemv_mfp4g32_e8_soa(&weights[repeat % replicas], &x, &reference, m, k)
                .expect("reference timed");
        };
        let prefetch = |gpu: &mut Gpu, repeat: usize| {
            gpu.gemv_mfp4g32_e8_soa_prefetch4_gfx1201(
                &weights[repeat % replicas],
                &x,
                &candidate,
                m,
                k,
            )
            .expect("candidate timed");
        };
        if trial & 1 == 0 {
            reference_us.push(event_us(gpu, replicas, incumbent));
            candidate_us.push(event_us(gpu, replicas, prefetch));
        } else {
            candidate_us.push(event_us(gpu, replicas, prefetch));
            reference_us.push(event_us(gpu, replicas, incumbent));
        }
    }
    let reference_us = median(&mut reference_us);
    let candidate_us = median(&mut candidate_us);
    println!(
        "CASE label={label} M={m} K={k} occurrences={occurrences} replicas={replicas} working_set_bytes={} raw_bit_comparisons={m} incumbent_us={reference_us:.6} prefetch4_us={candidate_us:.6} speedup_x={:.4} saved_us_per_call={:.6} saved_ms_per_rank_token={:.6}",
        weight_bytes * replicas,
        reference_us / candidate_us,
        reference_us - candidate_us,
        (reference_us - candidate_us) * occurrences as f64 / 1000.0,
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "exact gfx1201 required");
    run_case(&mut gpu, "router", 256, 4096, 43, 0x1201_1000);
    run_case(&mut gpu, "shared-up-rank01", 768, 4096, 86, 0x1201_1100);
    run_case(&mut gpu, "shared-up-rank2", 512, 4096, 86, 0x1201_1200);
    run_case(&mut gpu, "shared-down-rank01", 4096, 768, 43, 0x1201_1300);
    run_case(&mut gpu, "shared-down-rank2", 4096, 512, 43, 0x1201_1400);
    run_case(&mut gpu, "olora-down", 4096, 8192, 43, 0x1201_1500);
    run_case(&mut gpu, "wq-rank01", 12288, 1024, 43, 0x1201_1600);
    run_case(&mut gpu, "wq-or-index-rank2", 8192, 1024, 64, 0x1201_1700);
    run_case(&mut gpu, "index-rank01", 8192, 1024, 21, 0x1201_1800);
    run_case(&mut gpu, "lm-head-rank0", 129280, 4096, 1, 0x1201_1900);
}
