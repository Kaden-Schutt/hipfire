// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — raw-bit/event screen for the gfx942 two-wave qt35 GEMV.

use rdna_compute::{DType, Gpu, GpuTensor};

const GUARDS: usize = 16;
const POISON: f32 = 12345.625;
const TRIALS: usize = 7;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    m: usize,
    k: usize,
}

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

fn make_x(k: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
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
    // Keep metadata byte-true: qt35 storage is packed, not a dense M*K buffer.
    weight.shape = vec![packed.len()];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn elapsed_ms<F>(gpu: &mut Gpu, repeats: usize, mut launch: F) -> f64
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("create start event");
    let stop = gpu.hip.event_create().expect("create stop event");
    gpu.hip
        .event_record(&start, None)
        .expect("record start event");
    for _ in 0..repeats {
        launch(gpu);
    }
    gpu.hip
        .event_record(&stop, None)
        .expect("record stop event");
    gpu.hip.event_synchronize(&stop).expect("wait stop event");
    let ms = gpu
        .hip
        .event_elapsed_ms(&start, &stop)
        .expect("elapsed event time") as f64
        / repeats as f64;
    gpu.hip.event_destroy(start).expect("destroy start event");
    gpu.hip.event_destroy(stop).expect("destroy stop event");
    ms
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn check_guard(gpu: &Gpu, label: &str, backing: &GpuTensor, m: usize) {
    let host = gpu.download_f32(backing).expect("download guarded output");
    assert!(
        host[m..].iter().all(|&v| v.to_bits() == POISON.to_bits()),
        "{label} overwrote its output guard"
    );
}

fn run_shape(gpu: &mut Gpu, shape: Shape, ordinal: usize) {
    let Shape { name, m, k } = shape;
    let packed = build_e8_soa(m, k, 0x9420_e800_0000_u64 + ordinal as u64);
    assert_eq!(packed.len(), m * row_bytes(k));
    let weight = upload_weight(gpu, &packed);
    let x = gpu
        .upload_f32(&make_x(k, 0x9420_0128_0000_u64 + ordinal as u64), &[k])
        .expect("upload x");
    let poisoned = vec![POISON; m + GUARDS];
    let w64_backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .expect("allocate w64 output + guard");
    let w128_backing = gpu
        .upload_f32(&poisoned, &[poisoned.len()])
        .expect("allocate w128 output + guard");
    let w64 = w64_backing.sub_offset(0, m);
    let w128 = w128_backing.sub_offset(0, m);

    gpu.gemv_mfp4g32_e8_soa(&weight, &x, &w64, m, k)
        .expect("launch incumbent w64 GEMV");
    gpu.gemv_mfp4g32_e8_soa_w128_gfx942(&weight, &x, &w128, m, k)
        .expect("launch candidate w128 GEMV");
    gpu.hip.device_synchronize().expect("parity sync");

    let w64_host = gpu.download_f32(&w64).expect("download w64 output");
    let w128_host = gpu.download_f32(&w128).expect("download w128 output");
    assert!(
        w64_host
            .iter()
            .all(|&value| value.to_bits() != POISON.to_bits()),
        "w64 left poisoned rows for {name}"
    );
    assert!(
        w128_host
            .iter()
            .all(|&value| value.to_bits() != POISON.to_bits()),
        "w128 left poisoned rows for {name}"
    );
    let first = w64_host
        .iter()
        .zip(&w128_host)
        .enumerate()
        .find_map(|(row, (a, b))| {
            (a.to_bits() != b.to_bits()).then_some((row, a.to_bits(), b.to_bits()))
        });
    assert!(
        first.is_none(),
        "raw-bit mismatch for {name} M={m} K={k} (row, w64_bits, w128_bits): {first:?}"
    );
    check_guard(gpu, "w64", &w64_backing, m);
    check_guard(gpu, "w128", &w128_backing, m);

    // Warm both symbols before recording. This remains a shape-local screen;
    // repeated weights may become cache-resident and are not a product claim.
    for _ in 0..3 {
        gpu.gemv_mfp4g32_e8_soa(&weight, &x, &w64, m, k)
            .expect("warm w64");
        gpu.gemv_mfp4g32_e8_soa_w128_gfx942(&weight, &x, &w128, m, k)
            .expect("warm w128");
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    let bytes = packed.len() + (k + m) * std::mem::size_of::<f32>();
    let repeats = (256_000_000usize / bytes.max(1)).clamp(8, 64);
    let mut w64_ms = Vec::with_capacity(TRIALS);
    let mut w128_ms = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        let time_w64 = |gpu: &mut Gpu| {
            gpu.gemv_mfp4g32_e8_soa(&weight, &x, &w64, m, k)
                .expect("timed w64 GEMV");
        };
        let time_w128 = |gpu: &mut Gpu| {
            gpu.gemv_mfp4g32_e8_soa_w128_gfx942(&weight, &x, &w128, m, k)
                .expect("timed w128 GEMV");
        };
        if trial & 1 == 0 {
            w64_ms.push(elapsed_ms(gpu, repeats, time_w64));
            w128_ms.push(elapsed_ms(gpu, repeats, time_w128));
        } else {
            w128_ms.push(elapsed_ms(gpu, repeats, time_w128));
            w64_ms.push(elapsed_ms(gpu, repeats, time_w64));
        }
    }
    let w64_median = median(&mut w64_ms);
    let w128_median = median(&mut w128_ms);
    let w64_gbps = bytes as f64 / w64_median / 1.0e6;
    let w128_gbps = bytes as f64 / w128_median / 1.0e6;
    println!(
        "PASS name={name} M={m} K={k} raw_bits={m} repeats={repeats} trials={TRIALS} \
         w64_ms={w64_median:.6} w128_ms={w128_median:.6} speedup={:.4}x \
         w64_effective_GBps={w64_gbps:.2} w128_effective_GBps={w128_gbps:.2}",
        w64_median / w128_median
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx942", "this micro requires exact gfx942");

    let mut shapes = vec![
        Shape {
            name: "wq_a_or_wo_a_group",
            m: 1024,
            k: 4096,
        },
        Shape {
            name: "shared_w1_w3",
            m: 2048,
            k: 4096,
        },
        Shape {
            name: "wq_b",
            m: 32768,
            k: 1024,
        },
        Shape {
            name: "wo_b",
            m: 4096,
            k: 8192,
        },
    ];
    if std::env::args().any(|arg| arg == "--full") {
        shapes.push(Shape {
            name: "lm_head",
            m: 129280,
            k: 4096,
        });
    }

    for (ordinal, shape) in shapes.into_iter().enumerate() {
        run_shape(&mut gpu, shape, ordinal);
    }
}
