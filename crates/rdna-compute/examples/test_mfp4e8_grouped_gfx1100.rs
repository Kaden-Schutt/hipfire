// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1100 grouped O-LoRA E8 micro screen.

use rdna_compute::{DType, Gpu, GpuTensor};

const GROUPS: usize = 8;
const M: usize = 1024;
const K: usize = 4096;
const TRIALS: usize = 7;
const L3_BYTES: usize = 96 * 1024 * 1024;
const POISON: f32 = 12345.625;
const GUARDS: usize = 32;

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

fn build_weights(seed: u64) -> Vec<u8> {
    let blocks = K / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let stride = row_bytes();
    let mut packed = vec![0u8; GROUPS * M * stride];
    let mut state = seed;
    for row in 0..GROUPS * M {
        let off = row * stride;
        let row_scale = [0x3400u16, 0x3800, 0x3c00, 0x4000][row & 3];
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
    (0..GROUPS * K)
        .map(|i| match i & 31 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            _ => (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.25,
        })
        .collect()
}

fn upload_weight(gpu: &Gpu, packed: &[u8]) -> GpuTensor {
    let mut weight = gpu
        .upload_raw(packed, &[packed.len()])
        .expect("upload weight");
    weight.shape = vec![GROUPS, M, K];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn guarded(gpu: &mut Gpu) -> (GpuTensor, GpuTensor) {
    let poison = vec![POISON; GROUPS * M + GUARDS];
    let backing = gpu
        .upload_f32(&poison, &[poison.len()])
        .expect("guarded output");
    let view = backing.sub_offset(0, GROUPS * M);
    (backing, view)
}

fn check(gpu: &Gpu, label: &str, backing: &GpuTensor) -> Vec<f32> {
    let host = gpu.download_f32(backing).expect("download output");
    assert!(
        host[..GROUPS * M]
            .iter()
            .all(|v| v.to_bits() != POISON.to_bits()),
        "{label} left poisoned values"
    );
    assert!(
        host[GROUPS * M..]
            .iter()
            .all(|v| v.to_bits() == POISON.to_bits()),
        "{label} overwrote guard"
    );
    host[..GROUPS * M].to_vec()
}

fn sequential(gpu: &mut Gpu, weight: &GpuTensor, x: &GpuTensor, y: &GpuTensor) {
    let group_weight_bytes = M * row_bytes();
    for group in 0..GROUPS {
        let weight_view = weight.sub_offset(group * group_weight_bytes, group_weight_bytes);
        let x_view = x.sub_offset(group * K, K);
        let y_view = y.sub_offset(group * M, M);
        gpu.gemv_mfp4g32_e8_soa(&weight_view, &x_view, &y_view, M, K)
            .expect("sequential E8 launch");
    }
}

fn grouped(gpu: &mut Gpu, weight: &GpuTensor, x: &GpuTensor, y: &GpuTensor) {
    gpu.gemv_mfp4g32_e8_soa_grouped_gfx1100_candidate(weight, x, y, GROUPS, M, K)
        .expect("grouped E8 launch");
}

fn event_ms<F>(gpu: &mut Gpu, repeats: usize, mut launch: F) -> f64
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
    let elapsed = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64 / repeats as f64;
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    elapsed
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1100", "exact gfx1100 required");
    let first = build_weights(0x1100_7000);
    let replicas = ((L3_BYTES * 3 / 2) / first.len()).max(2) + 1;
    let mut weights = Vec::with_capacity(replicas);
    for replica in 0..replicas {
        weights.push(upload_weight(
            &gpu,
            &build_weights(0x1100_7000 + replica as u64),
        ));
    }
    let x = gpu
        .upload_f32(&make_x(0x1100_8000), &[GROUPS, K])
        .expect("upload x");
    let (sequential_backing, sequential_y) = guarded(&mut gpu);
    let (grouped_backing, grouped_y) = guarded(&mut gpu);

    sequential(&mut gpu, &weights[0], &x, &sequential_y);
    grouped(&mut gpu, &weights[0], &x, &grouped_y);
    gpu.hip.device_synchronize().expect("correctness sync");
    let reference = check(&gpu, "sequential", &sequential_backing);
    let candidate = check(&gpu, "grouped", &grouped_backing);
    let mismatch = reference
        .iter()
        .zip(&candidate)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(mismatch, None, "grouped raw-bit mismatch at {mismatch:?}");

    for weight in &weights {
        sequential(&mut gpu, weight, &x, &sequential_y);
        grouped(&mut gpu, weight, &x, &grouped_y);
    }
    gpu.hip.device_synchronize().expect("warm sync");
    let mut sequential_ms = Vec::with_capacity(TRIALS);
    let mut grouped_ms = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        let seq = |gpu: &mut Gpu, repeat: usize| {
            sequential(gpu, &weights[repeat % weights.len()], &x, &sequential_y)
        };
        let grp = |gpu: &mut Gpu, repeat: usize| {
            grouped(gpu, &weights[repeat % weights.len()], &x, &grouped_y)
        };
        if trial & 1 == 0 {
            sequential_ms.push(event_ms(&mut gpu, replicas, seq));
            grouped_ms.push(event_ms(&mut gpu, replicas, grp));
        } else {
            grouped_ms.push(event_ms(&mut gpu, replicas, grp));
            sequential_ms.push(event_ms(&mut gpu, replicas, seq));
        }
    }
    let sequential_ms = median(&mut sequential_ms);
    let grouped_ms = median(&mut grouped_ms);
    let saved_per_token = (sequential_ms - grouped_ms) * 43.0;
    const PRODUCT_MS: f64 = 1000.0 / 30.04391048548806;
    println!(
        "MICRO groups={GROUPS} M={M} K={K} replicas={replicas} working_set_bytes={} trials={TRIALS} raw_bits={} sequential_ms={sequential_ms:.6} grouped_ms={grouped_ms:.6} speedup={:.4}x saved_ms_per_token={saved_per_token:.6} projected_e2e_percent={:.3} sequential_GBps={:.2} grouped_GBps={:.2} product_evidence=false",
        first.len() * replicas,
        GROUPS * M,
        sequential_ms / grouped_ms,
        saved_per_token / PRODUCT_MS * 100.0,
        first.len() as f64 / sequential_ms / 1.0e6,
        first.len() as f64 / grouped_ms / 1.0e6,
    );
}
