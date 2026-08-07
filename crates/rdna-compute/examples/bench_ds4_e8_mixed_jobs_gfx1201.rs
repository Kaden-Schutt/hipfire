// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — DS4 gfx1201 attention-input E8 pack channel screen.

use rdna_compute::{DType, Gpu, GpuTensor};

const K: usize = 4096;
const TRIALS: usize = 7;
const L3_BYTES: usize = 96 * 1024 * 1024;

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

fn build_weight(rows: usize, seed: u64) -> Vec<u8> {
    let blocks = K / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let stride = row_bytes();
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

fn upload_weight(gpu: &Gpu, rows: usize, seed: u64) -> GpuTensor {
    let bytes = build_weight(rows, seed);
    let mut weight = gpu
        .upload_raw(&bytes, &[bytes.len()])
        .expect("upload weight");
    weight.shape = vec![rows, K];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn sequential(
    gpu: &mut Gpu,
    weights: &[GpuTensor],
    x: &GpuTensor,
    outputs: &[GpuTensor],
    rows: &[usize],
    sqrt_softplus_job: Option<usize>,
) {
    for index in 0..weights.len() {
        gpu.gemv_mfp4g32_e8_soa(&weights[index], x, &outputs[index], rows[index], K)
            .expect("sequential E8");
        if sqrt_softplus_job == Some(index) {
            gpu.sqrt_softplus_f32(&outputs[index])
                .expect("sequential sqrt_softplus");
        }
    }
}

fn packed(
    gpu: &mut Gpu,
    weights: &[GpuTensor],
    x: &GpuTensor,
    outputs: &[GpuTensor],
    rows: &[usize],
    sqrt_softplus_job: Option<usize>,
) {
    let weight_refs: Vec<&GpuTensor> = weights.iter().collect();
    let output_refs: Vec<&GpuTensor> = outputs.iter().collect();
    if let Some(job) = sqrt_softplus_job {
        gpu.gemv_mfp4g32_e8_soa_mixed_jobs_sqrt_softplus_gfx1201(
            &weight_refs,
            x,
            &output_refs,
            rows,
            K,
            job,
        )
        .expect("packed E8 post-op");
    } else {
        gpu.gemv_mfp4g32_e8_soa_mixed_jobs_gfx1201(&weight_refs, x, &output_refs, rows, K)
            .expect("packed E8");
    }
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
    rows: &[usize],
    layers: usize,
    seed: u64,
    sqrt_softplus_job: Option<usize>,
) {
    let bytes_per_pack = rows.iter().sum::<usize>() * row_bytes();
    let replicas = ((L3_BYTES * 3 / 2) / bytes_per_pack).max(2) + 1;
    let mut weight_sets = Vec::with_capacity(replicas);
    for replica in 0..replicas {
        let weights = rows
            .iter()
            .enumerate()
            .map(|(job, &m)| upload_weight(gpu, m, seed + replica as u64 * 17 + job as u64))
            .collect::<Vec<_>>();
        weight_sets.push(weights);
    }
    let mut rng = seed ^ 0xabc0_1201;
    let x_host = (0..K)
        .map(|index| match index & 31 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            _ => (lcg(&mut rng) as f32 / u32::MAX as f32 - 0.5) * 0.25,
        })
        .collect::<Vec<_>>();
    let x = gpu.upload_f32(&x_host, &[K]).expect("upload x");
    let reference = rows
        .iter()
        .map(|&m| {
            gpu.alloc_tensor(&[m], DType::F32)
                .expect("reference output")
        })
        .collect::<Vec<_>>();
    let candidate = rows
        .iter()
        .map(|&m| {
            gpu.alloc_tensor(&[m], DType::F32)
                .expect("candidate output")
        })
        .collect::<Vec<_>>();

    sequential(
        gpu,
        &weight_sets[0],
        &x,
        &reference,
        rows,
        sqrt_softplus_job,
    );
    packed(
        gpu,
        &weight_sets[0],
        &x,
        &candidate,
        rows,
        sqrt_softplus_job,
    );
    gpu.hip.device_synchronize().expect("correctness sync");
    let mut comparisons = 0usize;
    for job in 0..rows.len() {
        let expected = gpu
            .download_f32(&reference[job])
            .expect("download reference");
        let actual = gpu
            .download_f32(&candidate[job])
            .expect("download candidate");
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
        sequential(gpu, weights, &x, &reference, rows, sqrt_softplus_job);
        packed(gpu, weights, &x, &candidate, rows, sqrt_softplus_job);
    }
    gpu.hip.device_synchronize().expect("warm sync");
    let mut sequential_us = Vec::with_capacity(TRIALS);
    let mut packed_us = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        let seq = |gpu: &mut Gpu, repeat: usize| {
            sequential(
                gpu,
                &weight_sets[repeat % weight_sets.len()],
                &x,
                &reference,
                rows,
                sqrt_softplus_job,
            )
        };
        let pack = |gpu: &mut Gpu, repeat: usize| {
            packed(
                gpu,
                &weight_sets[repeat % weight_sets.len()],
                &x,
                &candidate,
                rows,
                sqrt_softplus_job,
            )
        };
        if trial & 1 == 0 {
            sequential_us.push(event_us(gpu, replicas, seq));
            packed_us.push(event_us(gpu, replicas, pack));
        } else {
            packed_us.push(event_us(gpu, replicas, pack));
            sequential_us.push(event_us(gpu, replicas, seq));
        }
    }
    let sequential_us = median(&mut sequential_us);
    let packed_us = median(&mut packed_us);
    println!(
        "CASE label={label} jobs={} rows={rows:?} K={K} replicas={replicas} working_set_bytes={} raw_bit_comparisons={comparisons} sequential_us={sequential_us:.6} packed_us={packed_us:.6} speedup_x={:.4} saved_us_per_layer={:.6} saved_ms_per_rank_token={:.6}",
        rows.len(),
        bytes_per_pack * replicas,
        sequential_us / packed_us,
        sequential_us - packed_us,
        (sequential_us - packed_us) * layers as f64 / 1000.0,
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "exact gfx1201 required");
    // Ratio-4: q_a, joint KV, main compressor pair, indexer weights,
    // indexer-compressor pair. The seven row counts sum to 4096.
    run_case(
        &mut gpu,
        "ratio4-attention-input",
        &[1024, 512, 1024, 1024, 64, 256, 256],
        21,
        0x1201_4000,
        None,
    );
    // Ratio-128: q_a, joint KV, and the two main-compressor projections.
    run_case(
        &mut gpu,
        "ratio128-attention-input",
        &[1024, 512, 512, 512],
        21,
        0x1201_8000,
        None,
    );
    // TP3 rank 0/1 own 768 shared-expert rows, rank 2 owns 512. The third
    // 256-row job is the replicated router projection and receives the exact
    // sqrt(softplus) post-op.
    run_case(
        &mut gpu,
        "tp3-ffn-rank01",
        &[768, 768, 256],
        43,
        0x1201_f301,
        Some(2),
    );
    run_case(
        &mut gpu,
        "tp3-ffn-rank2",
        &[512, 512, 256],
        43,
        0x1201_f302,
        Some(2),
    );
}
