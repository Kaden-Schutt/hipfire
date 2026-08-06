// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1100 E8-SoA four-wave workgroup micro screen.

use rdna_compute::{DType, Gpu, GpuTensor};

const TRIALS: usize = 7;
const GUARDS: usize = 32;
const POISON: f32 = 12345.625;

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    m: usize,
    k: usize,
    calls_per_token: usize,
}

// Occurrences come from the selected-decode gfx1100 trace at the canonical
// 2,048/512 fixture. M=4096 is split between its two K families.
const SHAPES: &[Shape] = &[
    Shape {
        label: "olora_wq_a_wo_a",
        m: 1024,
        k: 4096,
        calls_per_token: 429,
    },
    Shape {
        label: "shared_w1_w3",
        m: 2048,
        k: 4096,
        calls_per_token: 86,
    },
    Shape {
        label: "main_wq_b",
        m: 32768,
        k: 1024,
        calls_per_token: 43,
    },
    Shape {
        label: "shared_w2",
        m: 4096,
        k: 2048,
        calls_per_token: 43,
    },
    Shape {
        label: "wo_b",
        m: 4096,
        k: 8192,
        calls_per_token: 43,
    },
    Shape {
        label: "wkv",
        m: 512,
        k: 4096,
        calls_per_token: 83,
    },
    Shape {
        label: "router",
        m: 256,
        k: 4096,
        calls_per_token: 85,
    },
    Shape {
        label: "indexer_wq_b",
        m: 8192,
        k: 1024,
        calls_per_token: 21,
    },
    Shape {
        label: "indexer_wq_a",
        m: 64,
        k: 4096,
        calls_per_token: 21,
    },
    Shape {
        label: "lm_head",
        m: 129280,
        k: 4096,
        calls_per_token: 1,
    },
];

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn row_bytes(k: usize) -> usize {
    let blocks = k / 32;
    16 + blocks.div_ceil(16) * 16 + blocks * 16
}

fn build_e8_soa(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
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

fn make_x(k: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..k)
        .map(|i| match i & 31 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            _ => (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.25,
        })
        .collect()
}

fn upload_weight(gpu: &Gpu, packed: &[u8], m: usize, k: usize) -> GpuTensor {
    let mut weight = gpu
        .upload_raw(packed, &[packed.len()])
        .expect("upload weight");
    weight.shape = vec![m, k];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn guarded(gpu: &mut Gpu, n: usize) -> (GpuTensor, GpuTensor) {
    let poison = vec![POISON; n + GUARDS];
    let backing = gpu
        .upload_f32(&poison, &[poison.len()])
        .expect("guarded output");
    let view = backing.sub_offset(0, n);
    (backing, view)
}

fn checked(gpu: &Gpu, label: &str, backing: &GpuTensor, n: usize) -> Vec<f32> {
    let host = gpu.download_f32(backing).expect("download output");
    assert!(
        host[..n].iter().all(|v| v.to_bits() != POISON.to_bits()),
        "{label} left poisoned values"
    );
    assert!(
        host[n..].iter().all(|v| v.to_bits() == POISON.to_bits()),
        "{label} overwrote output guard"
    );
    host[..n].to_vec()
}

fn event_ms<F>(gpu: &mut Gpu, repeats: usize, mut launch: F) -> f64
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..repeats {
        launch(gpu);
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

fn run_shape(gpu: &mut Gpu, shape: Shape, ordinal: usize) -> (f64, f64) {
    let packed = build_e8_soa(shape.m, shape.k, 0x1100_5000 + ordinal as u64);
    let bytes = packed.len();
    let weight = upload_weight(gpu, &packed, shape.m, shape.k);
    let x = gpu
        .upload_f32(&make_x(shape.k, 0x1100_6000 + ordinal as u64), &[shape.k])
        .expect("upload x");
    let (reference_backing, reference) = guarded(gpu, shape.m);
    let (candidate_backing, candidate) = guarded(gpu, shape.m);
    gpu.gemv_mfp4g32_e8_soa(&weight, &x, &reference, shape.m, shape.k)
        .expect("reference launch");
    gpu.gemv_mfp4g32_e8_soa_pack4_gfx1100(&weight, &x, &candidate, shape.m, shape.k)
        .expect("candidate launch");
    gpu.hip.device_synchronize().expect("correctness sync");
    let reference_host = checked(gpu, "reference", &reference_backing, shape.m);
    let candidate_host = checked(gpu, "candidate", &candidate_backing, shape.m);
    let mismatch = reference_host
        .iter()
        .zip(&candidate_host)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(
        mismatch, None,
        "{} raw-bit mismatch at {mismatch:?}",
        shape.label
    );

    for _ in 0..4 {
        gpu.gemv_mfp4g32_e8_soa(&weight, &x, &reference, shape.m, shape.k)
            .unwrap();
        gpu.gemv_mfp4g32_e8_soa_pack4_gfx1100(&weight, &x, &candidate, shape.m, shape.k)
            .unwrap();
    }
    gpu.hip.device_synchronize().expect("warm sync");
    let repeats = (256_000_000usize / bytes).clamp(1, 64);
    let mut incumbent = Vec::with_capacity(TRIALS);
    let mut pack4 = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        let incumbent_launch = |gpu: &mut Gpu| {
            gpu.gemv_mfp4g32_e8_soa(&weight, &x, &reference, shape.m, shape.k)
                .unwrap()
        };
        let pack4_launch = |gpu: &mut Gpu| {
            gpu.gemv_mfp4g32_e8_soa_pack4_gfx1100(&weight, &x, &candidate, shape.m, shape.k)
                .unwrap()
        };
        if trial & 1 == 0 {
            incumbent.push(event_ms(gpu, repeats, incumbent_launch));
            pack4.push(event_ms(gpu, repeats, pack4_launch));
        } else {
            pack4.push(event_ms(gpu, repeats, pack4_launch));
            incumbent.push(event_ms(gpu, repeats, incumbent_launch));
        }
    }
    let incumbent = median(&mut incumbent);
    let pack4 = median(&mut pack4);
    println!(
        "MICRO {} M={} K={} calls_per_token={} bytes={} repeats={} trials={} raw_bits={} incumbent_ms={incumbent:.6} pack4_ms={pack4:.6} speedup={:.4}x incumbent_GBps={:.2} pack4_GBps={:.2} saved_ms_per_token={:.6}",
        shape.label,
        shape.m,
        shape.k,
        shape.calls_per_token,
        bytes,
        repeats,
        TRIALS,
        shape.m,
        incumbent / pack4,
        bytes as f64 / incumbent / 1.0e6,
        bytes as f64 / pack4 / 1.0e6,
        (incumbent - pack4) * shape.calls_per_token as f64,
    );
    (incumbent, pack4)
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1100", "exact gfx1100 required");
    let mut projected_saved_ms = 0.0f64;
    for (ordinal, &shape) in SHAPES.iter().enumerate() {
        let (incumbent, pack4) = run_shape(&mut gpu, shape, ordinal);
        projected_saved_ms += (incumbent - pack4) * shape.calls_per_token as f64;
    }
    const PRODUCT_MS: f64 = 512.0 / 30.04391048548806 * 1000.0 / 512.0;
    println!(
        "BUNDLE projected_saved_ms_per_token={projected_saved_ms:.6} projected_e2e_percent={:.3} product_ms_per_token={PRODUCT_MS:.6} raw_bit_exact=true product_evidence=false",
        projected_saved_ms / PRODUCT_MS * 100.0,
    );
}
