// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1100 shared-input E8 projection micro screen.

use rdna_compute::{DType, Gpu, GpuTensor};

const K: usize = 4096;
const TRIALS: usize = 7;
const L3_BYTES: usize = 96 * 1024 * 1024;
const POISON: f32 = 12345.625;
const GUARDS: usize = 32;
const PRODUCT_MS: f64 = 1000.0 / 32.00291295314136;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn row_bytes() -> usize {
    let blocks = K / 32;
    16 + blocks.div_ceil(16) * 16 + blocks * 16
}

fn build_weight(m: usize, seed: u64) -> Vec<u8> {
    let blocks = K / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let stride = row_bytes();
    let mut packed = vec![0_u8; m * stride];
    let mut state = seed;
    for row in 0..m {
        let off = row * stride;
        let row_scale = [0x3400_u16, 0x3800, 0x3c00, 0x4000][row & 3];
        packed[off..off + 2].copy_from_slice(&row_scale.to_le_bytes());
        packed[off + 4..off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        packed[off + 6] = 0x06;
        for block in 0..blocks {
            packed[off + 16 + block] = [0x01, 0x07, 0x38, 0x7f][block & 3];
            let codewords = off + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                let word = if block == 0 {
                    [0x0000_0000, 0x8000_0000, 0x7777_7777, 0xffff_ffff][slot]
                } else {
                    lcg(&mut state)
                };
                let dst = codewords + slot * 4;
                packed[dst..dst + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
    }
    packed
}

fn make_x(seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..K)
        .map(|i| match i & 31 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            _ => (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.25,
        })
        .collect()
}

fn upload_weight(gpu: &Gpu, m: usize, seed: u64) -> GpuTensor {
    let packed = build_weight(m, seed);
    let mut weight = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
    weight.shape = vec![m, K];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn guarded(gpu: &mut Gpu, jobs: usize, m: usize) -> (GpuTensor, Vec<GpuTensor>) {
    let total = jobs * m;
    let backing = gpu
        .upload_f32(&vec![POISON; total + GUARDS], &[total + GUARDS])
        .unwrap();
    let outputs = (0..jobs)
        .map(|job| backing.sub_offset(job * m, m))
        .collect();
    (backing, outputs)
}

fn check(gpu: &Gpu, backing: &GpuTensor, total: usize) -> Vec<f32> {
    let host = gpu.download_f32(backing).unwrap();
    assert!(host[..total]
        .iter()
        .all(|v| v.to_bits() != POISON.to_bits()));
    assert!(host[total..]
        .iter()
        .all(|v| v.to_bits() == POISON.to_bits()));
    host[..total].to_vec()
}

fn sequential(
    gpu: &mut Gpu,
    weights: &[GpuTensor],
    x: &GpuTensor,
    outputs: &[GpuTensor],
    m: usize,
) {
    for (weight, output) in weights.iter().zip(outputs) {
        gpu.gemv_mfp4g32_e8_soa_buffer_gfx1100(weight, x, output, m, K)
            .unwrap();
    }
}

fn shared(gpu: &mut Gpu, weights: &[GpuTensor], x: &GpuTensor, outputs: &[GpuTensor], m: usize) {
    let weight_refs: Vec<&GpuTensor> = weights.iter().collect();
    let output_refs: Vec<&GpuTensor> = outputs.iter().collect();
    gpu.gemv_mfp4g32_e8_soa_shared_jobs_buffer_gfx1100(&weight_refs, x, &output_refs, m, K)
        .unwrap();
}

fn event_ms<F>(gpu: &mut Gpu, repeats: usize, mut launch: F) -> f64
where
    F: FnMut(&mut Gpu, usize),
{
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for repeat in 0..repeats {
        launch(gpu, repeat);
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let elapsed = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 / repeats as f64;
    gpu.hip.event_destroy(start).unwrap();
    gpu.hip.event_destroy(stop).unwrap();
    elapsed
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn family(gpu: &mut Gpu, label: &str, jobs: usize, m: usize, layers: usize, seed: u64) -> f64 {
    let set_bytes = jobs * m * row_bytes();
    let replicas = ((L3_BYTES * 3 / 2) / set_bytes).max(2) + 1;
    let weight_sets: Vec<Vec<GpuTensor>> = (0..replicas)
        .map(|replica| {
            (0..jobs)
                .map(|job| upload_weight(gpu, m, seed + (replica * jobs + job) as u64))
                .collect()
        })
        .collect();
    let x = gpu.upload_f32(&make_x(seed ^ 0x55aa), &[K]).unwrap();
    let (seq_backing, seq_y) = guarded(gpu, jobs, m);
    let (shared_backing, shared_y) = guarded(gpu, jobs, m);

    sequential(gpu, &weight_sets[0], &x, &seq_y, m);
    shared(gpu, &weight_sets[0], &x, &shared_y, m);
    gpu.hip.device_synchronize().unwrap();
    let reference = check(gpu, &seq_backing, jobs * m);
    let candidate = check(gpu, &shared_backing, jobs * m);
    assert_eq!(
        reference
            .iter()
            .zip(&candidate)
            .position(|(a, b)| a.to_bits() != b.to_bits()),
        None
    );

    let mut seq_ms = Vec::with_capacity(TRIALS);
    let mut shared_ms = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        let seq = |gpu: &mut Gpu, repeat: usize| {
            sequential(gpu, &weight_sets[repeat % replicas], &x, &seq_y, m)
        };
        let shr = |gpu: &mut Gpu, repeat: usize| {
            shared(gpu, &weight_sets[repeat % replicas], &x, &shared_y, m)
        };
        if trial & 1 == 0 {
            seq_ms.push(event_ms(gpu, replicas, seq));
            shared_ms.push(event_ms(gpu, replicas, shr));
        } else {
            shared_ms.push(event_ms(gpu, replicas, shr));
            seq_ms.push(event_ms(gpu, replicas, seq));
        }
    }
    let seq_ms = median(&mut seq_ms);
    let shared_ms = median(&mut shared_ms);
    let saved = (seq_ms - shared_ms) * layers as f64;
    println!(
        "FAMILY label={label} jobs={jobs} M={m} K={K} layers={layers} replicas={replicas} working_set_bytes={} trials={TRIALS} raw_bits={} sequential_ms={seq_ms:.6} shared_ms={shared_ms:.6} speedup={:.4}x saved_ms_per_token={saved:.6} projected_e2e_percent={:.3} product_evidence=false",
        set_bytes * replicas,
        jobs * m,
        seq_ms / shared_ms,
        saved / PRODUCT_MS * 100.0,
    );
    saved
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    assert_eq!(gpu.arch, "gfx1100");
    let triple512 = family(&mut gpu, "ratio128", 3, 512, 20, 0x1100_1280);
    let pair1024 = family(&mut gpu, "ratio4-main", 2, 1024, 21, 0x1100_4001);
    let pair256 = family(&mut gpu, "ratio4-indexer", 2, 256, 21, 0x1100_4002);
    let total = triple512 + pair1024 + pair256;
    println!(
        "TOTAL saved_ms_per_token={total:.6} projected_e2e_percent={:.3} product_evidence=false",
        total / PRODUCT_MS * 100.0
    );
}
