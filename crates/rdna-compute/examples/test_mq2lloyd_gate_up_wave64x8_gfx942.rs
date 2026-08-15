// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx942 native MQ2-Lloyd gate/up channel.

use rdna_compute::{Gpu, GpuTensor};

const TOP_K: usize = 6;
const M: usize = 4096;
const MI: usize = M / 2;
const K: usize = 4096;
const GROUP_BYTES: usize = 72;
const ROW_BYTES: usize = (K / 256) * GROUP_BYTES;
const VALUES: usize = TOP_K * M;
const GUARDS: usize = 32;
const POISON: f32 = 8193.25;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn upload_i32(gpu: &Gpu, values: &[i32]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.upload_raw(bytes, &[bytes.len()]).expect("upload i32")
}

fn upload_u64(gpu: &Gpu, values: &[u64]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.upload_raw(bytes, &[bytes.len()]).expect("upload u64")
}

fn build_expert(seed: u64) -> Vec<u8> {
    const CODEBOOK: [u16; 4] = [0xbc00, 0xb400, 0x3400, 0x3c00]; // -1,-.25,.25,1
    let mut state = seed;
    let mut packed = vec![0u8; M * ROW_BYTES];
    for row in 0..M {
        for group in 0..K / 256 {
            let base = row * ROW_BYTES + group * GROUP_BYTES;
            for (slot, bits) in CODEBOOK.iter().enumerate() {
                packed[base + 2 * slot..base + 2 * slot + 2].copy_from_slice(&bits.to_le_bytes());
            }
            for byte in &mut packed[base + 8..base + GROUP_BYTES] {
                *byte = lcg(&mut state) as u8;
            }
        }
    }
    packed
}

fn build_x() -> Vec<f32> {
    let mut state = 0x9420_cafe_2a11_u64;
    (0..K)
        .map(|i| match i & 31 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            4 => 1.0,
            5 => -1.0,
            _ => (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.5,
        })
        .collect()
}

fn cpu_row(weight: &[u8], row: usize, x: &[f32]) -> f32 {
    const CODEBOOK: [f32; 4] = [-1.0, -0.25, 0.25, 1.0];
    let mut sum = 0.0f32;
    for group in 0..K / 256 {
        let base = row * ROW_BYTES + group * GROUP_BYTES + 8;
        for lane in 0..64 {
            let packed = weight[base + lane];
            let x_base = group * 256 + lane * 4;
            sum += CODEBOOK[(packed & 3) as usize] * x[x_base]
                + CODEBOOK[((packed >> 2) & 3) as usize] * x[x_base + 1]
                + CODEBOOK[((packed >> 4) & 3) as usize] * x[x_base + 2]
                + CODEBOOK[((packed >> 6) & 3) as usize] * x[x_base + 3];
        }
    }
    sum
}

fn elapsed_ms<F>(gpu: &mut Gpu, repeats: usize, mut launch: F) -> f32
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("event start");
    let stop = gpu.hip.event_create().expect("event stop");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..repeats {
        launch(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("wait stop");
    let elapsed = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed ms") / repeats as f32;
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    elapsed
}

fn guards_intact(gpu: &Gpu, tensor: &GpuTensor) -> bool {
    gpu.download_f32(tensor).expect("download guarded output")[TOP_K * MI..]
        .iter()
        .all(|value| value.to_bits() == POISON.to_bits())
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx942", "this channel requires exact gfx942");

    let host_weights: Vec<Vec<u8>> = (0..TOP_K)
        .map(|expert| build_expert(0x9420_0000_1000 + expert as u64))
        .collect();
    let device_weights: Vec<GpuTensor> = host_weights
        .iter()
        .map(|packed| {
            gpu.upload_raw(packed, &[packed.len()])
                .expect("upload MQ2-Lloyd expert")
        })
        .collect();
    let expert_ptrs: Vec<u64> = device_weights
        .iter()
        .map(|weight| weight.buf.as_ptr() as usize as u64)
        .collect();
    let expert_ptrs = upload_u64(&gpu, &expert_ptrs);
    let topk = upload_i32(&gpu, &[0, 1, 2, 3, 4, 5]);
    let host_x = build_x();
    let x = gpu.upload_f32(&host_x, &[K]).expect("upload x");

    let poisoned = vec![POISON; TOP_K * MI + GUARDS];
    let baseline_gate_backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .expect("baseline gate + guard");
    let baseline_up_backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .expect("baseline up + guard");
    let candidate_gate_backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .expect("candidate gate + guard");
    let candidate_up_backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .expect("candidate up + guard");
    let baseline_gate = baseline_gate_backing.sub_offset(0, TOP_K * MI);
    let baseline_up = baseline_up_backing.sub_offset(0, TOP_K * MI);
    let candidate_gate = candidate_gate_backing.sub_offset(0, TOP_K * MI);
    let candidate_up = candidate_up_backing.sub_offset(0, TOP_K * MI);

    gpu.try_gfx942()
        .expect("exact gfx942")
        .mq2_lloyd_moe_gate_up_wave64(
            &expert_ptrs,
            &topk,
            &x,
            &baseline_gate,
            &baseline_up,
            M,
            K,
            TOP_K,
        )
        .expect("wave64 baseline");
    gpu.try_gfx942()
        .expect("exact gfx942")
        .mq2_lloyd_moe_gate_up_wave64x8_candidate(
            &expert_ptrs,
            &topk,
            &x,
            &candidate_gate,
            &candidate_up,
            M,
            K,
            TOP_K,
        )
        .expect("wave64x8 candidate");
    gpu.hip.device_synchronize().expect("channel sync");

    let baseline_gate_host = gpu
        .download_f32(&baseline_gate)
        .expect("download baseline gate");
    let baseline_up_host = gpu
        .download_f32(&baseline_up)
        .expect("download baseline up");
    let candidate_gate_host = gpu
        .download_f32(&candidate_gate)
        .expect("download candidate gate");
    let candidate_up_host = gpu
        .download_f32(&candidate_up)
        .expect("download candidate up");
    let baseline: Vec<f32> = baseline_gate_host
        .iter()
        .zip(&baseline_up_host)
        .flat_map(|(&gate, &up)| [gate, up])
        .collect();
    let candidate: Vec<f32> = candidate_gate_host
        .iter()
        .zip(&candidate_up_host)
        .flat_map(|(&gate, &up)| [gate, up])
        .collect();
    let mut raw_bit_mismatches = 0usize;
    let mut numerical_violations = 0usize;
    let mut max_abs = 0.0f32;
    let mut first_raw = None;
    for (index, (&reference, &observed)) in baseline.iter().zip(&candidate).enumerate() {
        if reference.to_bits() != observed.to_bits() {
            raw_bit_mismatches += 1;
            first_raw.get_or_insert((index, reference.to_bits(), observed.to_bits()));
        }
        let abs = (observed - reference).abs();
        max_abs = max_abs.max(abs);
        if !reference.is_finite() || !observed.is_finite() || abs > 1.0e-5 {
            numerical_violations += 1;
        }
    }

    let sentinel_rows = [0usize, 257, 1023, 2047, 2048, 3071, 4095];
    let mut cpu_violations = 0usize;
    let mut max_cpu_abs = 0.0f32;
    for expert in 0..TOP_K {
        for &row in &sentinel_rows {
            let expected = cpu_row(&host_weights[expert], row, &host_x);
            let observed = if row < MI {
                candidate_gate_host[expert * MI + row]
            } else {
                candidate_up_host[expert * MI + row - MI]
            };
            let abs = (observed - expected).abs();
            max_cpu_abs = max_cpu_abs.max(abs);
            let tolerance = 5.0e-3 + 5.0e-4 * expected.abs();
            if !observed.is_finite() || abs > tolerance {
                cpu_violations += 1;
            }
        }
    }

    let guards = [
        guards_intact(&gpu, &baseline_gate_backing),
        guards_intact(&gpu, &baseline_up_backing),
        guards_intact(&gpu, &candidate_gate_backing),
        guards_intact(&gpu, &candidate_up_backing),
    ];
    println!(
        "CHANNEL topk={TOP_K} M={M} K={K} values={VALUES} raw_bit_mismatches={} \
         numerical_violations={} max_abs={max_abs:.9e} first_raw={first_raw:?} \
         cpu_sentinels={} cpu_violations={} max_cpu_abs={max_cpu_abs:.9e} guards={guards:?}",
        raw_bit_mismatches,
        numerical_violations,
        TOP_K * sentinel_rows.len(),
        cpu_violations,
    );
    assert!(guards.into_iter().all(|ok| ok), "output guard overwritten");
    assert_eq!(
        raw_bit_mismatches, 0,
        "wave64x8 changed incumbent arithmetic: {first_raw:?}"
    );
    assert_eq!(cpu_violations, 0, "CPU sentinel check failed");

    for _ in 0..3 {
        gpu.try_gfx942()
            .expect("exact gfx942")
            .mq2_lloyd_moe_gate_up_wave64(
                &expert_ptrs,
                &topk,
                &x,
                &baseline_gate,
                &baseline_up,
                M,
                K,
                TOP_K,
            )
            .expect("warm baseline");
        gpu.try_gfx942()
            .expect("exact gfx942")
            .mq2_lloyd_moe_gate_up_wave64x8_candidate(
                &expert_ptrs,
                &topk,
                &x,
                &candidate_gate,
                &candidate_up,
                M,
                K,
                TOP_K,
            )
            .expect("warm candidate");
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    const REPEATS: usize = 20;
    let baseline_ms = elapsed_ms(&mut gpu, REPEATS, |gpu| {
        gpu.try_gfx942()
            .expect("exact gfx942")
            .mq2_lloyd_moe_gate_up_wave64(
                &expert_ptrs,
                &topk,
                &x,
                &baseline_gate,
                &baseline_up,
                M,
                K,
                TOP_K,
            )
            .expect("timed baseline");
    });
    let candidate_ms = elapsed_ms(&mut gpu, REPEATS, |gpu| {
        gpu.try_gfx942()
            .expect("exact gfx942")
            .mq2_lloyd_moe_gate_up_wave64x8_candidate(
                &expert_ptrs,
                &topk,
                &x,
                &candidate_gate,
                &candidate_up,
                M,
                K,
                TOP_K,
            )
            .expect("timed candidate");
    });
    println!(
        "MICRO repeats={REPEATS} wave64_ms={baseline_ms:.6} \
         wave64x8_shared_x_ms={candidate_ms:.6} speedup={:.4}x \
         baseline_grid=4096x6x64 candidate_grid=512x6x512",
        baseline_ms / candidate_ms
    );
}
