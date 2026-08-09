// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx942 native-wave64 qt35 projection candidate channel.

use rdna_compute::{DType, Gpu, GpuTensor};

const GROUPS: usize = 8;
const M: usize = 1024;
const K: usize = 4096;
const GUARDS: usize = 32;
const POISON: f32 = 12345.625;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn build_e8_soa(rows: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let row_bytes = 16 + scale_padded + blocks * 16;
    assert_eq!(row_bytes, 2192, "unexpected production qt35 row stride");
    let mut packed = vec![0u8; rows * row_bytes];
    let mut state = 0x9420_a11c_64f0_u64;

    for row in 0..rows {
        let off = row * row_bytes;
        let row_scale = [0x3400u16, 0x3800, 0x3c00, 0x4000][row & 3];
        packed[off..off + 2].copy_from_slice(&row_scale.to_le_bytes());
        packed[off + 4..off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        packed[off + 6] = 0x06;
        for block in 0..blocks {
            packed[off + 16 + block] = [0x01, 0x07, 0x38, 0x7f][block & 3];
            let codewords = off + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                let word = match (block, slot) {
                    (0, 0) => 0x0000_0000,
                    (0, 1) => 0x8000_0000,
                    (0, 2) => 0x7777_7777,
                    (0, 3) => 0xffff_ffff,
                    _ => lcg(&mut state),
                };
                packed[codewords + slot * 4..codewords + slot * 4 + 4]
                    .copy_from_slice(&word.to_le_bytes());
            }
        }
    }
    packed
}

fn upload_weight(gpu: &Gpu, packed: &[u8]) -> GpuTensor {
    let mut weight = gpu
        .upload_raw(packed, &[packed.len()])
        .expect("upload grouped qt35 weight");
    weight.shape = vec![packed.len()];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn make_x() -> Vec<f32> {
    let mut state = 0x9420_cafe_f00d_u64;
    (0..GROUPS * K)
        .map(|i| match i & 31 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            4 => 1.0,
            5 => -1.0,
            _ => (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.25,
        })
        .collect()
}

fn elapsed_ms<F>(gpu: &mut Gpu, repeats: usize, mut launch: F) -> f32
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("create start event");
    let stop = gpu.hip.event_create().expect("create stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..repeats {
        launch(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("wait stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed ms");
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    ms / repeats as f32
}

fn guards_intact(gpu: &Gpu, tensor: &GpuTensor) -> bool {
    let all = gpu.download_f32(tensor).expect("download guarded output");
    all[GROUPS * M..]
        .iter()
        .all(|&v| v.to_bits() == POISON.to_bits())
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx942", "this channel requires exact gfx942");

    let packed = build_e8_soa(GROUPS * M, K);
    assert_eq!(packed.len(), 17_956_864);
    let weight = upload_weight(&gpu, &packed);
    let x = gpu.upload_f32(&make_x(), &[GROUPS, K]).expect("upload x");
    let poisoned = vec![POISON; GROUPS * M + GUARDS];
    let baseline_backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .expect("baseline y + guards");
    let candidate_backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .expect("candidate y + guards");
    let baseline = baseline_backing.sub_offset(0, GROUPS * M);
    let candidate = candidate_backing.sub_offset(0, GROUPS * M);

    gpu.try_gfx942()
        .expect("exact gfx942 capability")
        .grouped_olora_e8(&weight, &x, &baseline, GROUPS, M, K)
        .expect("retained grouped baseline");
    gpu.try_gfx942()
        .expect("exact gfx942 capability")
        .grouped_olora_e8_wave64x4_candidate(&weight, &x, &candidate, GROUPS, M, K)
        .expect("native wave64x4 candidate");
    gpu.hip.device_synchronize().expect("channel sync");

    let reference = gpu.download_f32(&baseline).expect("download baseline");
    let observed = gpu.download_f32(&candidate).expect("download candidate");
    let mut raw_bit_mismatches = 0usize;
    let mut numerical_violations = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut first_raw = None;
    let mut first_bad = None;

    for (index, (&a, &b)) in reference.iter().zip(&observed).enumerate() {
        if a.to_bits() != b.to_bits() {
            raw_bit_mismatches += 1;
            first_raw.get_or_insert((index, a.to_bits(), b.to_bits()));
        }
        let abs = (b - a).abs();
        let rel = abs / a.abs().max(1.0e-12);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        let tolerance = 5.0e-3 + 1.0e-4 * a.abs();
        if !a.is_finite() || !b.is_finite() || abs > tolerance {
            numerical_violations += 1;
            first_bad.get_or_insert((index, a, b, abs, tolerance));
        }
    }

    let baseline_guards = guards_intact(&gpu, &baseline_backing);
    let candidate_guards = guards_intact(&gpu, &candidate_backing);
    println!(
        "CHANNEL G={GROUPS} M={M} K={K} values={} raw_bit_mismatches={} \
         max_abs={max_abs:.9e} max_rel={max_rel:.9e} numerical_violations={} \
         first_raw={first_raw:?} first_bad={first_bad:?} \
         baseline_guards={} candidate_guards={}",
        GROUPS * M,
        raw_bit_mismatches,
        numerical_violations,
        baseline_guards,
        candidate_guards,
    );
    assert!(baseline_guards, "retained baseline overwrote output guard");
    assert!(candidate_guards, "wave64x4 candidate overwrote output guard");
    assert_eq!(
        numerical_violations, 0,
        "wave64x4 candidate failed grouped baseline tolerance: {first_bad:?}"
    );

    // Shape-local HIP-event screen only. Both arms use the same allocations,
    // input, process, and stream; this is not a model/product benchmark.
    for _ in 0..3 {
        gpu.try_gfx942()
            .expect("exact gfx942 capability")
            .grouped_olora_e8(&weight, &x, &baseline, GROUPS, M, K)
            .expect("warm baseline");
        gpu.try_gfx942()
            .expect("exact gfx942 capability")
            .grouped_olora_e8_wave64x4_candidate(&weight, &x, &candidate, GROUPS, M, K)
            .expect("warm candidate");
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    const REPEATS: usize = 40;
    let baseline_ms = elapsed_ms(&mut gpu, REPEATS, |gpu| {
        gpu.try_gfx942()
            .expect("exact gfx942 capability")
            .grouped_olora_e8(&weight, &x, &baseline, GROUPS, M, K)
            .expect("timed baseline");
    });
    let candidate_ms = elapsed_ms(&mut gpu, REPEATS, |gpu| {
        gpu.try_gfx942()
            .expect("exact gfx942 capability")
            .grouped_olora_e8_wave64x4_candidate(&weight, &x, &candidate, GROUPS, M, K)
            .expect("timed candidate");
    });
    println!(
        "MICRO repeats={REPEATS} retained_grouped_ms={baseline_ms:.6} \
         wave64x4_candidate_ms={candidate_ms:.6} speedup={:.4}x \
         retained_grid=512x8x64 candidate_grid=256x8x256",
        baseline_ms / candidate_ms
    );
}
