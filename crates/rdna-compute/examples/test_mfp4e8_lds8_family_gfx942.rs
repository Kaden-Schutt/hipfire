// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx942 qt35 LDS8 family numerical/performance mechanism channel.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

const SRC: &str = include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_lds8_family.gfx942.hip");
const MODULE: &str = "gemv_mfp4g32_e8_soa_lds8_family_gfx942_candidate";
const GUARDS: usize = 32;
const POISON: f32 = 12345.625;
const TRIALS: usize = 5;

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    m: usize,
    k: usize,
    symbol: &'static str,
    bytes_per_token: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        label: "wq_b",
        m: 32768,
        k: 1024,
        symbol: "gemv_mfp4g32_e8_soa_lds8_k1024_gfx942_candidate",
        bytes_per_token: 789_053_440,
    },
    Shape {
        label: "indexer_wq_b",
        m: 8192,
        k: 1024,
        symbol: "gemv_mfp4g32_e8_soa_lds8_k1024_gfx942_candidate",
        bytes_per_token: 96_337_920,
    },
    Shape {
        label: "shared_w2",
        m: 4096,
        k: 2048,
        symbol: "gemv_mfp4g32_e8_soa_lds8_k2048_gfx942_candidate",
        bytes_per_token: 194_527_232,
    },
    Shape {
        label: "wo_b",
        m: 4096,
        k: 8192,
        symbol: "gemv_mfp4g32_e8_soa_lds8_k8192_gfx942_candidate",
        bytes_per_token: 769_294_336,
    },
    Shape {
        label: "lm_head",
        m: 129280,
        k: 4096,
        symbol: "gemv_mfp4g32_e8_soa_lds8_k4096_gfx942_candidate",
        bytes_per_token: 283_381_760,
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
            let cw = off + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                let word = if block == 0 {
                    [0x0000_0000, 0x8000_0000, 0x7777_7777, 0xffff_ffff][slot]
                } else {
                    lcg(&mut state)
                };
                packed[cw + slot * 4..cw + slot * 4 + 4].copy_from_slice(&word.to_le_bytes());
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
        "{label} overwrote guard"
    );
    host[..n].to_vec()
}

fn launch_candidate(
    gpu: &Gpu,
    symbol: &str,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
) {
    let mut kb = KernargBlob::new();
    kb.push_ptr(w.buf.as_ptr() as *const c_void);
    kb.push_ptr(x.buf.as_ptr() as *const c_void);
    kb.push_ptr(y.buf.as_ptr() as *const c_void);
    kb.pad_to(16);
    gpu.launch_kernel_blob(
        symbol,
        [(m / 8) as u32, 1, 1],
        [512, 1, 1],
        0,
        kb.as_mut_slice(),
    )
    .expect("candidate launch");
}

fn event_ms<F>(gpu: &mut Gpu, repeats: usize, mut f: F) -> f64
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..repeats {
        f(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("sync stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64 / repeats as f64;
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    ms
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn compare(label: &str, a: &[f32], b: &[f32]) {
    let mut raw = 0usize;
    let mut bad = 0usize;
    let mut max_abs = 0.0f32;
    let mut first = None;
    for (i, (&r, &c)) in a.iter().zip(b).enumerate() {
        raw += usize::from(r.to_bits() != c.to_bits());
        let abs = (r - c).abs();
        max_abs = max_abs.max(abs);
        let tol = 5.0e-3 + 1.0e-4 * r.abs();
        if !r.is_finite() || !c.is_finite() || abs > tol {
            bad += 1;
            first.get_or_insert((i, r, c, abs, tol));
        }
    }
    println!("CHANNEL {label} values={} raw_bit_mismatches={raw} max_abs={max_abs:.9e} numerical_violations={bad} first_bad={first:?}", a.len());
    assert_eq!(bad, 0, "{label} tolerance failure: {first:?}");
}

fn run_shape(gpu: &mut Gpu, shape: Shape, ordinal: usize) -> (usize, f64, f64) {
    let packed = build_e8_soa(shape.m, shape.k, 0x9420_5000_u64 + ordinal as u64);
    let weight_bytes = packed.len();
    let weight = upload_weight(gpu, &packed, shape.m, shape.k);
    let x = gpu
        .upload_f32(&make_x(shape.k, 0x9420_6000 + ordinal as u64), &[shape.k])
        .expect("upload x");
    let (ref_back, ref_y) = guarded(gpu, shape.m);
    let (cand_back, cand_y) = guarded(gpu, shape.m);
    gpu.gemv_mfp4g32_e8_soa(&weight, &x, &ref_y, shape.m, shape.k)
        .expect("reference launch");
    launch_candidate(gpu, shape.symbol, &weight, &x, &cand_y, shape.m);
    gpu.hip.device_synchronize().expect("channel sync");
    compare(
        shape.label,
        &checked(gpu, "reference", &ref_back, shape.m),
        &checked(gpu, "candidate", &cand_back, shape.m),
    );

    for _ in 0..3 {
        gpu.gemv_mfp4g32_e8_soa(&weight, &x, &ref_y, shape.m, shape.k)
            .unwrap();
        launch_candidate(gpu, shape.symbol, &weight, &x, &cand_y, shape.m);
    }
    gpu.hip.device_synchronize().expect("warm sync");
    let repeats = (256_000_000usize / weight_bytes).clamp(1, 32);
    let mut reference = Vec::with_capacity(TRIALS);
    let mut candidate = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        let r = |gpu: &mut Gpu| {
            gpu.gemv_mfp4g32_e8_soa(&weight, &x, &ref_y, shape.m, shape.k)
                .unwrap()
        };
        let c = |gpu: &mut Gpu| launch_candidate(gpu, shape.symbol, &weight, &x, &cand_y, shape.m);
        if trial & 1 == 0 {
            reference.push(event_ms(gpu, repeats, r));
            candidate.push(event_ms(gpu, repeats, c));
        } else {
            candidate.push(event_ms(gpu, repeats, c));
            reference.push(event_ms(gpu, repeats, r));
        }
    }
    let r = median(&mut reference);
    let c = median(&mut candidate);
    println!("MICRO {} M={} K={} bytes={} repeats={} trials={} reference_ms={r:.6} candidate_ms={c:.6} speedup={:.4}x reference_GBps={:.2} candidate_GBps={:.2} bytes_per_token={}",
        shape.label, shape.m, shape.k, weight_bytes, repeats, TRIALS, r/c,
        weight_bytes as f64/r/1.0e6, weight_bytes as f64/c/1.0e6,
        shape.bytes_per_token);
    (shape.bytes_per_token, r, c)
}

fn run_grouped(gpu: &mut Gpu) -> (usize, f64, f64) {
    const G: usize = 8;
    const M: usize = 1024;
    const K: usize = 4096;
    const SYMBOL: &str = "gemv_mfp4g32_e8_soa_grouped_lds8_gfx942_candidate";
    let packed = build_e8_soa(G * M, K, 0x9420_7000);
    let weight = upload_weight(gpu, &packed, G * M, K);
    let mut xh = Vec::with_capacity(G * K);
    for g in 0..G {
        xh.extend(make_x(K, 0x9420_7100 + g as u64));
    }
    let x = gpu.upload_f32(&xh, &[G, K]).expect("grouped x");
    let (ref_back, ref_y) = guarded(gpu, G * M);
    let (cand_back, cand_y) = guarded(gpu, G * M);
    let launch = |gpu: &Gpu, y: &GpuTensor| {
        let mut kb = KernargBlob::new();
        kb.push_ptr(weight.buf.as_ptr() as *const c_void);
        kb.push_ptr(x.buf.as_ptr() as *const c_void);
        kb.push_ptr(y.buf.as_ptr() as *const c_void);
        kb.pad_to(16);
        gpu.launch_kernel_blob(
            SYMBOL,
            [M as u32 / 8, G as u32, 1],
            [512, 1, 1],
            0,
            kb.as_mut_slice(),
        )
        .unwrap();
    };
    gpu.try_gfx942()
        .unwrap()
        .grouped_olora_e8(&weight, &x, &ref_y, G, M, K)
        .unwrap();
    launch(gpu, &cand_y);
    gpu.hip.device_synchronize().unwrap();
    compare(
        "wo_a_grouped",
        &checked(gpu, "grouped ref", &ref_back, G * M),
        &checked(gpu, "grouped cand", &cand_back, G * M),
    );
    for _ in 0..3 {
        gpu.try_gfx942()
            .unwrap()
            .grouped_olora_e8(&weight, &x, &ref_y, G, M, K)
            .unwrap();
        launch(gpu, &cand_y);
    }
    gpu.hip.device_synchronize().unwrap();
    let repeats = 16;
    let mut reference = Vec::new();
    let mut candidate = Vec::new();
    for trial in 0..TRIALS {
        let r = |gpu: &mut Gpu| {
            gpu.try_gfx942()
                .unwrap()
                .grouped_olora_e8(&weight, &x, &ref_y, G, M, K)
                .unwrap()
        };
        let c = |gpu: &mut Gpu| launch(gpu, &cand_y);
        if trial & 1 == 0 {
            reference.push(event_ms(gpu, repeats, r));
            candidate.push(event_ms(gpu, repeats, c));
        } else {
            candidate.push(event_ms(gpu, repeats, c));
            reference.push(event_ms(gpu, repeats, r));
        }
    }
    let r = median(&mut reference);
    let c = median(&mut candidate);
    const BPT: usize = 772_239_360;
    println!("MICRO wo_a_grouped G={G} M={M} K={K} bytes={} repeats={repeats} trials={TRIALS} reference_ms={r:.6} candidate_ms={c:.6} speedup={:.4}x reference_GBps={:.2} candidate_GBps={:.2} bytes_per_token={BPT}", packed.len(), r/c, packed.len() as f64/r/1.0e6, packed.len() as f64/c/1.0e6);
    (BPT, r, c)
}

fn main() {
    let grouped_only = std::env::args().any(|arg| arg == "--grouped-only");
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx942", "exact gfx942 required");
    if !grouped_only {
        for s in SHAPES {
            gpu.ensure_kernel_public(MODULE, SRC, s.symbol).unwrap();
        }
    }
    gpu.ensure_kernel_public(
        MODULE,
        SRC,
        "gemv_mfp4g32_e8_soa_grouped_lds8_gfx942_candidate",
    )
    .unwrap();
    let mut weighted_saved_ms = 0.0f64;
    let mut covered_bytes = 0usize;
    // Run the structurally distinct O-LoRA group first so a later family
    // member's numerical rejection cannot hide this independent channel.
    let (b, r, c) = run_grouped(&mut gpu);
    covered_bytes += b;
    weighted_saved_ms += (r - c) * 43.0;
    if grouped_only {
        println!(
            "BUNDLE grouped_only=true covered_bytes_per_token={covered_bytes} projected_saved_ms_per_token={weighted_saved_ms:.6} projected_e2e_percent_at_36ms={:.3} arithmetic_only_not_product_evidence=true",
            weighted_saved_ms / 36.0 * 100.0
        );
        return;
    }
    for (i, &shape) in SHAPES.iter().enumerate() {
        let (b, r, c) = run_shape(&mut gpu, shape, i);
        covered_bytes += b;
        weighted_saved_ms += (r - c) * (b as f64 / (shape.m * row_bytes(shape.k)) as f64);
    }
    println!("BUNDLE covered_bytes_per_token={covered_bytes} projected_saved_ms_per_token={weighted_saved_ms:.6} projected_e2e_percent_at_36ms={:.3} arithmetic_only_not_product_evidence=true", weighted_saved_ms/36.0*100.0);
}
