// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — raw-bit/guard/event screen for the gfx942 two-job qt35 GEMV.

use rdna_compute::{DType, Gpu, GpuTensor};

const K: usize = 4096;
const GUARDS: usize = 16;
const POISON: f32 = 12345.625;
const TRIALS: usize = 7;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn row_bytes(k: usize) -> usize {
    assert_eq!(k % 256, 0);
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    16 + scale_padded + blocks * 16
}

fn build_e8_soa(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let stride = row_bytes(k);
    let mut packed = vec![0u8; m * stride];
    let mut state = seed;

    for row in 0..m {
        let off = row * stride;
        let row_scale = [0x3400u16, 0x3800, 0x3c00, 0x4000][row & 3];
        packed[off..off + 2].copy_from_slice(&row_scale.to_le_bytes());
        packed[off + 4..off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        packed[off + 6] = 0x06;
        for block in 0..blocks {
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

fn make_x(k: usize) -> Vec<f32> {
    let mut state = 0x9420_2e80_cafe_u64;
    (0..k)
        .map(|i| match i & 15 {
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
        .expect("upload qt35 weight");
    weight.shape = vec![packed.len()];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn guarded_output(gpu: &mut Gpu, m: usize, label: &str) -> (GpuTensor, GpuTensor) {
    let poisoned = vec![POISON; m + GUARDS];
    let backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .unwrap_or_else(|e| panic!("allocate {label} output + guard: {e:?}"));
    let view = backing.sub_offset(0, m);
    (backing, view)
}

fn check_output(gpu: &Gpu, label: &str, backing: &GpuTensor, m: usize) -> Vec<f32> {
    let host = gpu
        .download_f32(backing)
        .unwrap_or_else(|e| panic!("download {label}: {e:?}"));
    assert!(
        host[..m]
            .iter()
            .all(|&value| value.to_bits() != POISON.to_bits()),
        "{label} left poisoned output rows"
    );
    assert!(
        host[m..]
            .iter()
            .all(|&value| value.to_bits() == POISON.to_bits()),
        "{label} overwrote its output guard"
    );
    host[..m].to_vec()
}

fn elapsed_ms<F>(gpu: &mut Gpu, repeats: usize, mut launch: F) -> f64
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
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed ms") as f64 / repeats as f64;
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    ms
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn assert_raw_bits(label: &str, reference: &[f32], candidate: &[f32]) {
    let first = reference
        .iter()
        .zip(candidate)
        .enumerate()
        .find_map(|(row, (a, b))| {
            (a.to_bits() != b.to_bits()).then_some((row, a.to_bits(), b.to_bits()))
        });
    assert!(
        first.is_none(),
        "raw-bit mismatch for {label} (row, reference_bits, pair_bits): {first:?}"
    );
}

fn run_shape(gpu: &mut Gpu, m: usize, ordinal: usize) {
    let packed0 = build_e8_soa(m, K, 0x9420_a000_0000_u64 + ordinal as u64);
    let packed1 = build_e8_soa(m, K, 0x9420_b000_0000_u64 + ordinal as u64);
    assert_eq!(packed0.len(), m * row_bytes(K));
    assert_eq!(packed1.len(), m * row_bytes(K));
    let weight0 = upload_weight(gpu, &packed0);
    let weight1 = upload_weight(gpu, &packed1);
    let x = gpu.upload_f32(&make_x(K), &[K]).expect("upload shared x");

    let (ref0_backing, ref0) = guarded_output(gpu, m, "reference job 0");
    let (ref1_backing, ref1) = guarded_output(gpu, m, "reference job 1");
    let (pair0_backing, pair0) = guarded_output(gpu, m, "pair job 0");
    let (pair1_backing, pair1) = guarded_output(gpu, m, "pair job 1");

    gpu.gemv_mfp4g32_e8_soa(&weight0, &x, &ref0, m, K)
        .expect("launch reference job 0");
    gpu.gemv_mfp4g32_e8_soa(&weight1, &x, &ref1, m, K)
        .expect("launch reference job 1");
    gpu.gemv_mfp4g32_e8_soa_pair_gfx942(&weight0, &weight1, &x, &pair0, &pair1, m, K)
        .expect("launch pair candidate");
    gpu.hip.device_synchronize().expect("parity sync");

    let ref0_host = check_output(gpu, "reference job 0", &ref0_backing, m);
    let ref1_host = check_output(gpu, "reference job 1", &ref1_backing, m);
    let pair0_host = check_output(gpu, "pair job 0", &pair0_backing, m);
    let pair1_host = check_output(gpu, "pair job 1", &pair1_backing, m);
    assert_raw_bits("job 0", &ref0_host, &pair0_host);
    assert_raw_bits("job 1", &ref1_host, &pair1_host);

    for _ in 0..3 {
        gpu.gemv_mfp4g32_e8_soa(&weight0, &x, &ref0, m, K)
            .expect("warm reference job 0");
        gpu.gemv_mfp4g32_e8_soa(&weight1, &x, &ref1, m, K)
            .expect("warm reference job 1");
        gpu.gemv_mfp4g32_e8_soa_pair_gfx942(&weight0, &weight1, &x, &pair0, &pair1, m, K)
            .expect("warm pair candidate");
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    let bytes = packed0.len() + packed1.len() + (K + 2 * m) * std::mem::size_of::<f32>();
    let repeats = (256_000_000usize / bytes.max(1)).clamp(8, 64);
    let mut serial_ms = Vec::with_capacity(TRIALS);
    let mut pair_ms = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        let serial = |gpu: &mut Gpu| {
            gpu.gemv_mfp4g32_e8_soa(&weight0, &x, &ref0, m, K)
                .expect("timed reference job 0");
            gpu.gemv_mfp4g32_e8_soa(&weight1, &x, &ref1, m, K)
                .expect("timed reference job 1");
        };
        let pair = |gpu: &mut Gpu| {
            gpu.gemv_mfp4g32_e8_soa_pair_gfx942(&weight0, &weight1, &x, &pair0, &pair1, m, K)
                .expect("timed pair candidate");
        };
        if trial & 1 == 0 {
            serial_ms.push(elapsed_ms(gpu, repeats, serial));
            pair_ms.push(elapsed_ms(gpu, repeats, pair));
        } else {
            pair_ms.push(elapsed_ms(gpu, repeats, pair));
            serial_ms.push(elapsed_ms(gpu, repeats, serial));
        }
    }

    let serial_median = median(&mut serial_ms);
    let pair_median = median(&mut pair_ms);
    let serial_gbps = bytes as f64 / serial_median / 1.0e6;
    let pair_gbps = bytes as f64 / pair_median / 1.0e6;
    println!(
        "PASS M={m} K={K} raw_bits={} guards=4x{GUARDS} repeats={repeats} trials={TRIALS} \
         serial_2_ms={serial_median:.6} pair_1_ms={pair_median:.6} speedup={:.4}x \
         serial_effective_GBps={serial_gbps:.2} pair_effective_GBps={pair_gbps:.2}",
        2 * m,
        serial_median / pair_median
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx942", "this micro requires exact gfx942");

    for (ordinal, m) in [2048, 1024, 512, 256].into_iter().enumerate() {
        run_shape(&mut gpu, m, ordinal);
    }
}
