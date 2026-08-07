// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — full-route exact-gfx1100 E8 raw-buffer micro screen.

use rdna_compute::{DType, Gpu, GpuTensor};

const CACHE_BYTES: usize = 96 * 1024 * 1024;
const SAMPLES: usize = 7;
const ROUNDS: usize = 4;
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("M64K4096", 64, 4096, 21),
    ("M256K4096", 256, 4096, 85),
    ("M512K4096", 512, 4096, 83),
    ("M1024K4096", 1024, 4096, 85),
    ("M2048K4096", 2048, 4096, 86),
    ("M4096K2048", 4096, 2048, 43),
    ("M4096K8192", 4096, 8192, 43),
    ("M8192K1024", 8192, 1024, 21),
    ("M32768K1024", 32768, 1024, 43),
    ("M129280K4096", 129280, 4096, 1),
];

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn build_e8_soa(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let row_bytes = 16 + scale_padded + blocks * 16;
    let mut packed = vec![0u8; m * row_bytes];
    let mut state = seed;
    for row in 0..m {
        let offset = row * row_bytes;
        let row_scale = [0x3000u16, 0x3400, 0x3800, 0x3c00][row & 3];
        packed[offset..offset + 2].copy_from_slice(&row_scale.to_le_bytes());
        packed[offset + 4..offset + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        packed[offset + 6] = 0x06;
        for block in 0..blocks {
            packed[offset + 16 + block] = [0x01, 0x07, 0x38, 0x7f][block & 3];
            let codewords = offset + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                packed[codewords + slot * 4..codewords + slot * 4 + 4]
                    .copy_from_slice(&lcg(&mut state).to_le_bytes());
            }
        }
    }
    packed
}

fn make_x(k: usize) -> Vec<f32> {
    let mut state = 0x1100_b0ffe8u64;
    (0..k)
        .map(|index| match index & 31 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            _ => (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.25,
        })
        .collect()
}

fn upload_weight(gpu: &Gpu, packed: &[u8], m: usize, row_bytes: usize) -> GpuTensor {
    let mut weight = gpu
        .upload_raw(packed, &[m, row_bytes])
        .expect("upload E8 weight");
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn elapsed_us<F>(gpu: &mut Gpu, buffers: &[GpuTensor], mut launch: F) -> f64
where
    F: FnMut(&mut Gpu, &GpuTensor),
{
    let start = gpu.hip.event_create().expect("create start event");
    let stop = gpu.hip.event_create().expect("create stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..ROUNDS {
        for weight in buffers {
            launch(gpu, weight);
        }
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("wait stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    f64::from(ms) * 1000.0 / (ROUNDS * buffers.len()) as f64
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1100", "this micro requires exact gfx1100");
    println!("gfx1100 E8 raw-buffer full-route screen");
    println!(
        "{:<14} {:>5} {:>10} {:>10} {:>9} {:>10}",
        "shape", "occ", "base us", "cand us", "speedup", "cand GB/s"
    );
    let mut weighted_base_us = 0.0;
    let mut weighted_candidate_us = 0.0;
    let mut covered_launches = 0usize;
    let mut covered_bytes = 0usize;

    for &(name, m, k, occurrences) in SHAPES {
        let packed = build_e8_soa(m, k, 0xe8_1100_b0ff ^ m as u64 ^ k as u64);
        let row_bytes = packed.len() / m;
        let replicas = (CACHE_BYTES * 3 / 2).div_ceil(packed.len()).max(2);
        let buffers: Vec<GpuTensor> = (0..replicas)
            .map(|_| upload_weight(&gpu, &packed, m, row_bytes))
            .collect();
        let x = gpu.upload_f32(&make_x(k), &[k]).expect("upload activation");
        let baseline = gpu.alloc_tensor(&[m], DType::F32).expect("baseline y");
        let candidate = gpu.alloc_tensor(&[m], DType::F32).expect("candidate y");

        gpu.gemv_mfp4g32_e8_soa(&buffers[0], &x, &baseline, m, k)
            .expect("baseline correctness launch");
        gpu.gemv_mfp4g32_e8_soa_buffer_gfx1100(&buffers[0], &x, &candidate, m, k)
            .expect("candidate correctness launch");
        gpu.hip.device_synchronize().expect("correctness sync");
        let expected = gpu.download_f32(&baseline).expect("download baseline");
        let observed = gpu.download_f32(&candidate).expect("download candidate");
        let mismatch = expected
            .iter()
            .zip(&observed)
            .position(|(left, right)| left.to_bits() != right.to_bits());
        assert_eq!(mismatch, None, "{name}: raw-bit mismatch at {mismatch:?}");

        for weight in &buffers {
            gpu.gemv_mfp4g32_e8_soa(weight, &x, &baseline, m, k)
                .expect("baseline warmup");
            gpu.gemv_mfp4g32_e8_soa_buffer_gfx1100(weight, &x, &candidate, m, k)
                .expect("candidate warmup");
        }
        gpu.hip.device_synchronize().expect("warmup sync");

        let mut base_samples = Vec::with_capacity(SAMPLES);
        let mut candidate_samples = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let base = |gpu: &mut Gpu| {
                elapsed_us(gpu, &buffers, |gpu, weight| {
                    gpu.gemv_mfp4g32_e8_soa(weight, &x, &baseline, m, k)
                        .expect("timed baseline")
                })
            };
            let cand = |gpu: &mut Gpu| {
                elapsed_us(gpu, &buffers, |gpu, weight| {
                    gpu.gemv_mfp4g32_e8_soa_buffer_gfx1100(weight, &x, &candidate, m, k)
                        .expect("timed candidate")
                })
            };
            if sample & 1 == 0 {
                base_samples.push(base(&mut gpu));
                candidate_samples.push(cand(&mut gpu));
            } else {
                candidate_samples.push(cand(&mut gpu));
                base_samples.push(base(&mut gpu));
            }
        }
        let base_us = median(&mut base_samples);
        let candidate_us = median(&mut candidate_samples);
        println!(
            "{name:<14} {occurrences:>5} {base_us:>10.3} {candidate_us:>10.3} {:>8.4}x {:>10.1}",
            base_us / candidate_us,
            packed.len() as f64 / candidate_us / 1000.0,
        );
        weighted_base_us += base_us * occurrences as f64;
        weighted_candidate_us += candidate_us * occurrences as f64;
        covered_launches += occurrences;
        covered_bytes += packed.len() * occurrences;
    }

    println!(
        "WEIGHTED covered_launches={covered_launches}/511 covered_bytes_gb={:.3} baseline_ms={:.3} candidate_ms={:.3} saved_ms={:.3} speedup={:.4}x projected_e2e_percent={:.3} raw_bits=true",
        covered_bytes as f64 / 1e9,
        weighted_base_us / 1000.0,
        weighted_candidate_us / 1000.0,
        (weighted_base_us - weighted_candidate_us) / 1000.0,
        weighted_base_us / weighted_candidate_us,
        (weighted_base_us - weighted_candidate_us) / 1000.0 / 36.6 * 100.0,
    );
}
