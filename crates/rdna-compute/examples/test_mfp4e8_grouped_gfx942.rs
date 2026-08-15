// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — raw-bit admission gate for the gfx942 grouped DS4 O-LoRA GEMV.

use rdna_compute::{DType, Gpu, GpuTensor};

const GROUPS: usize = 8;
const M: usize = 1024;
const K: usize = 4096;

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
    let mut state = 0x9420_0a11_5eed_u64;

    for row in 0..rows {
        let off = row * row_bytes;
        // Valid qt35 SoA row header: fp16 row scale, block count, layout 0x06.
        let row_scale = [0x3400u16, 0x3800, 0x3c00, 0x4000][row & 3];
        packed[off..off + 2].copy_from_slice(&row_scale.to_le_bytes());
        packed[off + 4..off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        packed[off + 6] = 0x06;
        for block in 0..blocks {
            // Exercise subnormal, ordinary, and max-finite/clamped E4M3 scales.
            packed[off + 16 + block] = [0x01, 0x07, 0x38, 0x7f][block & 3];
            let cw = off + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                let codeword = match (block, slot) {
                    (0, 0) => 0x0000_0000,
                    (0, 1) => 0x8000_0000,
                    (0, 2) => 0x7777_7777,
                    (0, 3) => 0xffff_ffff,
                    _ => lcg(&mut state),
                };
                packed[cw + slot * 4..cw + slot * 4 + 4].copy_from_slice(&codeword.to_le_bytes());
            }
        }
    }
    packed
}

fn upload_weight(gpu: &Gpu, packed: &[u8]) -> GpuTensor {
    let mut weight = gpu
        .upload_raw(packed, &[packed.len()])
        .expect("upload grouped qt35 weight");
    // Keep metadata byte-true: packed qt35 is not a dense M*K allocation.
    weight.shape = vec![packed.len()];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn make_x() -> Vec<f32> {
    let mut state = 0x9420_cafe_f00d_u64;
    (0..GROUPS * K)
        .map(|i| match i & 15 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            _ => (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.25,
        })
        .collect()
}

fn serial_wo_a(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    group_bytes: usize,
) {
    for group in 0..GROUPS {
        let mut wg = weight.sub_offset(group * group_bytes, group_bytes);
        wg.shape = vec![M, K];
        let xg = x.sub_offset(group * K, K);
        let yg = y.sub_offset(group * M, M);
        gpu.gemv_mfp4g32_e8_soa(&wg, &xg, &yg, M, K)
            .expect("serial gfx942 qt35 GEMV");
    }
}

fn elapsed_ms<F>(gpu: &mut Gpu, mut launch: F) -> f32
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("create start event");
    let stop = gpu.hip.event_create().expect("create stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    launch(gpu);
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("wait stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed ms");
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    ms
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx942", "this admission gate requires gfx942");

    let packed = build_e8_soa(GROUPS * M, K);
    let group_bytes = packed.len() / GROUPS;
    assert_eq!(group_bytes, 2_244_608);
    assert_eq!(packed.len(), 17_956_864);
    let weight = upload_weight(&gpu, &packed);
    let x = gpu.upload_f32(&make_x(), &[GROUPS, K]).expect("upload x");
    const GUARDS: usize = 16;
    const POISON: f32 = 12345.625;
    let poisoned = vec![POISON; GROUPS * M + GUARDS];
    let serial_backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .expect("serial y + guards");
    let grouped_backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .expect("grouped y + guards");
    let serial = serial_backing.sub_offset(0, GROUPS * M);
    let grouped = grouped_backing.sub_offset(0, GROUPS * M);

    serial_wo_a(&mut gpu, &weight, &x, &serial, group_bytes);
    gpu.try_gfx942()
        .expect("exact gfx942 capability")
        .grouped_olora_e8(&weight, &x, &grouped, GROUPS, M, K)
        .expect("grouped gfx942 qt35 GEMV");
    gpu.hip.device_synchronize().expect("parity sync");

    let serial_host = gpu.download_f32(&serial).expect("download serial");
    let grouped_host = gpu.download_f32(&grouped).expect("download grouped");
    assert!(
        serial_host.iter().all(|&v| v.to_bits() != POISON.to_bits()),
        "serial path left poisoned output rows"
    );
    assert!(
        grouped_host
            .iter()
            .all(|&v| v.to_bits() != POISON.to_bits()),
        "grouped path left poisoned output rows"
    );
    let first = serial_host
        .iter()
        .zip(&grouped_host)
        .enumerate()
        .find_map(|(flat, (a, b))| {
            (a.to_bits() != b.to_bits()).then_some((
                flat,
                flat / M,
                flat % M,
                a.to_bits(),
                b.to_bits(),
            ))
        });
    assert!(
        first.is_none(),
        "grouped raw-bit mismatch (flat, group, row, serial_bits, grouped_bits): {first:?}"
    );
    for (label, backing) in [("serial", &serial_backing), ("grouped", &grouped_backing)] {
        let all = gpu.download_f32(backing).expect("download guarded output");
        assert!(
            all[GROUPS * M..]
                .iter()
                .all(|&v| v.to_bits() == POISON.to_bits()),
            "{label} path overwrote output guard"
        );
    }
    println!(
        "PASS raw bits G={GROUPS} M={M} K={K}: {} f32 values byte-identical",
        GROUPS * M
    );

    // A tiny shape-local screen only: one event interval per arm, no product-loop
    // claim. It answers whether removing seven launches moves this exact block.
    const REPEATS: usize = 8;
    let serial_ms = elapsed_ms(&mut gpu, |gpu| {
        for _ in 0..REPEATS {
            serial_wo_a(gpu, &weight, &x, &serial, group_bytes);
        }
    }) / REPEATS as f32;
    let grouped_ms = elapsed_ms(&mut gpu, |gpu| {
        for _ in 0..REPEATS {
            gpu.try_gfx942()
                .expect("exact gfx942 capability")
                .grouped_olora_e8(&weight, &x, &grouped, GROUPS, M, K)
                .expect("timed grouped GEMV");
        }
    }) / REPEATS as f32;
    println!(
        "MICRO serial_8_launch_ms={serial_ms:.6} grouped_1_launch_ms={grouped_ms:.6} speedup={:.3}x launches_saved=7",
        serial_ms / grouped_ms
    );
}
