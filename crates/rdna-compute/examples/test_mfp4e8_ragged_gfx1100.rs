// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1100 ragged shared-input E8 projection micro screen.

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
    let mut weight = gpu
        .upload_raw(&packed, &[packed.len()])
        .expect("upload weight");
    weight.shape = vec![m, K];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn guarded(gpu: &mut Gpu, rows: &[usize]) -> (GpuTensor, Vec<GpuTensor>) {
    let total: usize = rows.iter().sum();
    let poison = vec![POISON; total + GUARDS];
    let backing = gpu
        .upload_f32(&poison, &[poison.len()])
        .expect("guarded output");
    let mut offset = 0;
    let mut views = Vec::with_capacity(rows.len());
    for &m in rows {
        views.push(backing.sub_offset(offset, m));
        offset += m;
    }
    (backing, views)
}

fn check(gpu: &Gpu, label: &str, backing: &GpuTensor, total: usize) -> Vec<f32> {
    let host = gpu.download_f32(backing).expect("download output");
    assert!(
        host[..total]
            .iter()
            .all(|v| v.to_bits() != POISON.to_bits()),
        "{label} left poisoned values"
    );
    assert!(
        host[total..]
            .iter()
            .all(|v| v.to_bits() == POISON.to_bits()),
        "{label} overwrote guard"
    );
    host[..total].to_vec()
}

fn sequential(
    gpu: &mut Gpu,
    weights: &[GpuTensor],
    x: &GpuTensor,
    outputs: &[GpuTensor],
    rows: &[usize],
) {
    for ((weight, output), &m) in weights.iter().zip(outputs).zip(rows) {
        gpu.gemv_mfp4g32_e8_soa(weight, x, output, m, K)
            .expect("sequential E8 launch");
    }
}

fn ragged(
    gpu: &mut Gpu,
    weights: &[GpuTensor],
    x: &GpuTensor,
    outputs: &[GpuTensor],
    rows: &[usize],
) {
    let weight_refs: Vec<&GpuTensor> = weights.iter().collect();
    let output_refs: Vec<&GpuTensor> = outputs.iter().collect();
    gpu.gemv_mfp4g32_e8_soa_ragged_gfx1100(&weight_refs, x, &output_refs, rows, K)
        .expect("ragged E8 launch");
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

fn run_family(
    gpu: &mut Gpu,
    label: &str,
    rows: &[usize],
    layers_per_token: usize,
    seed: u64,
) -> f64 {
    let total_rows: usize = rows.iter().sum();
    let set_bytes = total_rows * row_bytes();
    let replicas = ((L3_BYTES * 3 / 2) / set_bytes).max(2) + 1;
    let mut weight_sets: Vec<Vec<GpuTensor>> = Vec::with_capacity(replicas);
    for replica in 0..replicas {
        let weights = rows
            .iter()
            .enumerate()
            .map(|(projection, &m)| {
                upload_weight(gpu, m, seed + (replica * rows.len() + projection) as u64)
            })
            .collect();
        weight_sets.push(weights);
    }
    let x = gpu
        .upload_f32(&make_x(seed ^ 0x55aa_aa55), &[K])
        .expect("upload x");
    let (sequential_backing, sequential_y) = guarded(gpu, rows);
    let (ragged_backing, ragged_y) = guarded(gpu, rows);

    sequential(gpu, &weight_sets[0], &x, &sequential_y, rows);
    ragged(gpu, &weight_sets[0], &x, &ragged_y, rows);
    gpu.hip.device_synchronize().expect("correctness sync");
    let reference = check(gpu, "sequential", &sequential_backing, total_rows);
    let candidate = check(gpu, "ragged", &ragged_backing, total_rows);
    let mismatch = reference
        .iter()
        .zip(&candidate)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(mismatch, None, "ragged raw-bit mismatch at {mismatch:?}");

    for weights in &weight_sets {
        sequential(gpu, weights, &x, &sequential_y, rows);
        ragged(gpu, weights, &x, &ragged_y, rows);
    }
    gpu.hip.device_synchronize().expect("warm sync");

    let mut sequential_ms = Vec::with_capacity(TRIALS);
    let mut ragged_ms = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        let seq = |gpu: &mut Gpu, repeat: usize| {
            sequential(
                gpu,
                &weight_sets[repeat % weight_sets.len()],
                &x,
                &sequential_y,
                rows,
            )
        };
        let rag = |gpu: &mut Gpu, repeat: usize| {
            ragged(
                gpu,
                &weight_sets[repeat % weight_sets.len()],
                &x,
                &ragged_y,
                rows,
            )
        };
        if trial & 1 == 0 {
            sequential_ms.push(event_ms(gpu, replicas, seq));
            ragged_ms.push(event_ms(gpu, replicas, rag));
        } else {
            ragged_ms.push(event_ms(gpu, replicas, rag));
            sequential_ms.push(event_ms(gpu, replicas, seq));
        }
    }

    let sequential_ms = median(&mut sequential_ms);
    let ragged_ms = median(&mut ragged_ms);
    let saved_ms_per_token = (sequential_ms - ragged_ms) * layers_per_token as f64;
    println!(
        "FAMILY label={label} rows={rows:?} K={K} layers_per_token={layers_per_token} replicas={replicas} working_set_bytes={} trials={TRIALS} raw_bits={total_rows} sequential_ms={sequential_ms:.6} ragged_ms={ragged_ms:.6} speedup={:.4}x saved_ms_per_token={saved_ms_per_token:.6} projected_e2e_percent={:.3} sequential_GBps={:.2} ragged_GBps={:.2} product_evidence=false",
        set_bytes * replicas,
        sequential_ms / ragged_ms,
        saved_ms_per_token / PRODUCT_MS * 100.0,
        set_bytes as f64 / sequential_ms / 1.0e6,
        set_bytes as f64 / ragged_ms / 1.0e6,
    );
    saved_ms_per_token
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1100", "exact gfx1100 required");
    let ratio128 = run_family(&mut gpu, "ratio128", &[512, 512, 512], 20, 0x1100_1280);
    let ratio4 = run_family(
        &mut gpu,
        "ratio4",
        &[512, 1024, 1024, 256, 256],
        21,
        0x1100_0004,
    );
    let total = ratio128 + ratio4;
    println!(
        "TOTAL saved_ms_per_token={total:.6} projected_e2e_percent={:.3} product_evidence=false",
        total / PRODUCT_MS * 100.0,
    );
}
