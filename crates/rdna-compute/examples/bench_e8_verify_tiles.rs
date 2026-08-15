// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dense E8-SoA GEMM variant sweep at **speculative-verify batch sizes**, across
//! the **real DeepSeek-V4 dense shape mix**.
//!
//! Why this exists: `gemm_mfp4g32_e8_soa_wmma` (the B1 tile) is 57% of DSpark's
//! `verify_block` on gfx1151 at 53.5 GiB/s, while MoE GEMVs in the same capture
//! hit 167-181 GiB/s. `e8_prefill_batch_tiles` (deepseek4 forward.rs:1266) only
//! selects the b2 tile above batch 16 and b4 above batch 32, so every verify
//! (B<=6) structurally gets tiles=1 and wastes 15/16 of each 16-wide WMMA tile.
//!
//! A single-shape bench (M=4096 K=4096) previously suggested the batched GEMV
//! was 1.4-3.7x faster here — enabling it end-to-end REGRESSED verify 144->202 ms.
//! That bench measured one shape; real verify spans M=1024..32768, K=1024..8192.
//! So this sweeps the ACTUAL mix and reports a per-(shape,batch) winner.
//!
//! Weights are cycled across replicas so no buffer stays cache-resident —
//! matching the real model, where each layer's weights are touched once per
//! forward.
//!
//! Run: HIP_VISIBLE_DEVICES=<gfx1151> cargo run --release -p rdna-compute \
//!        --example bench_e8_verify_tiles

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 20;
/// Exceed gfx1151's cache so we measure the DRAM-resident regime the model sees.
const MIN_WORKING_SET: usize = 160 * 1024 * 1024;

/// Real DS4 dense E8 shapes (from the live verify profile). K%256==0 for all,
/// which every b2/b4 kernel asserts.
const SHAPES: &[(&str, usize, usize)] = &[
    ("wq_a/wo_a", 1024, 4096),
    ("wq_b", 32768, 1024),
    ("wo_b", 4096, 8192),
    ("shared_up", 2048, 4096),
    ("shared_dn", 4096, 2048),
];

fn synth_e8_aos(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = k / 32;
    let row_bytes = 16 + blocks_per_row * 17;
    let mut out = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut rng = || -> u32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        let roff = row * row_bytes;
        out[roff..roff + 2].copy_from_slice(&0x2400u16.to_le_bytes());
        out[roff + 4..roff + 6].copy_from_slice(&(blocks_per_row as u16).to_le_bytes());
        out[roff + 6] = 0x05;
        for b in 0..blocks_per_row {
            let bp = roff + 16 + b * 17;
            out[bp] = 120u8.wrapping_add((rng() & 0x3F) as u8);
            for w in 0..4 {
                let cw = rng();
                out[bp + 1 + w * 4..bp + 1 + w * 4 + 4].copy_from_slice(&cw.to_le_bytes());
            }
        }
    }
    out
}

fn aos_to_soa_full(aos: &[u8], m: usize, k: usize) -> Vec<u8> {
    let n_blocks = k / 32;
    let aos_row_bytes = 16 + n_blocks * 17;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
    let mut out = Vec::with_capacity(m * soa_row_bytes);
    for r in 0..m {
        let row = &aos[r * aos_row_bytes..(r + 1) * aos_row_bytes];
        let mut o = vec![0u8; soa_row_bytes];
        o[..16].copy_from_slice(&row[..16]);
        o[6] = 0x06; // SoA flag
        for b in 0..n_blocks {
            o[16 + b] = row[16 + b * 17];
        }
        let cw_start = 16 + scale_padded;
        for b in 0..n_blocks {
            let src = 16 + b * 17 + 1;
            let dst = cw_start + b * 16;
            o[dst..dst + 16].copy_from_slice(&row[src..src + 16]);
        }
        out.extend_from_slice(&o);
    }
    out
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) * 2.3e-10 - 0.5
        })
        .collect()
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("GPU: {}  has_wmma={}", gpu.arch, gpu.arch_caps.has_wmma());
    if gpu.arch != "gfx1151" {
        println!("NOTE: b2/b4 dense E8 tiles are gfx1151 kernels; results elsewhere may not load.");
    }
    println!(
        "\nDense E8-SoA at verify batch, DRAM-resident (weights cycled over replicas).\n\
         b1/b2/b4 = gemm_mfp4g32_e8_soa_wmma{{,_b2,_b4}}; gemv = gemv_mfp4g32_e8_soa_batched_gfx1151.\n\
         Selector today (forward.rs:1266): tiles=1 for every batch <= 16, so verify always uses b1.\n"
    );

    for &(name, m, k) in SHAPES {
        let soa = aos_to_soa_full(&synth_e8_aos(m, k, 0xE8_5EED), m, k);
        let row_bytes = soa.len() / m;
        let one = soa.len();
        let reps = (MIN_WORKING_SET / one).max(2);
        let mut wbufs = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = gpu.alloc_tensor(&[m, row_bytes], DType::MFP4G32E8SOA).expect("alloc w");
            gpu.hip.memcpy_htod(&t.buf, &soa).expect("htod w");
            wbufs.push(t);
        }
        println!(
            "  {name}  M={m} K={k}  weights {:.1} MB x {reps} = {:.0} MB working set",
            one as f64 / 1e6,
            (one * reps) as f64 / 1e6
        );
        println!(
            "    {:>3} {:>10} {:>10} {:>10} {:>10}   {:>9} {:>9} {:>9} {:>9}   {}",
            "B", "b1 us", "b2 us", "b4 us", "gemv us", "b1 GB/s", "b2 GB/s", "b4 GB/s", "gemv GB/s", "winner"
        );

        for b in [1usize, 2, 3, 4, 5, 6, 8] {
            let x_h = make_x(b * k, 0x1234 + b as u64);
            let x = gpu.alloc_tensor(&[b, k], DType::F32).expect("alloc x");
            gpu.hip.memcpy_htod(&x.buf, bytes_of(&x_h)).expect("htod x");
            let y = gpu.alloc_tensor(&[b, m], DType::F32).expect("alloc y");

            let mut time_variant = |which: usize, gpu: &mut Gpu| -> Option<f64> {
                let run = |g: &mut Gpu, w: &rdna_compute::GpuTensor| -> Result<(), String> {
                    let r = match which {
                        0 => g.gemm_mfp4g32_e8_soa_wmma(w, &x, &y, m, k, b),
                        1 => g.gemm_mfp4g32_e8_soa_wmma_b2(w, &x, &y, m, k, b),
                        2 => g.gemm_mfp4g32_e8_soa_wmma_b4(w, &x, &y, m, k, b),
                        _ => g.gemv_mfp4g32_e8_soa_batched_gfx1151(w, &x, &y, b, m, k),
                    };
                    r.map_err(|e| format!("{e:?}"))
                };
                for i in 0..WARMUP {
                    if run(gpu, &wbufs[i % reps]).is_err() {
                        return None;
                    }
                }
                let _ = gpu.hip.device_synchronize();
                let t = Instant::now();
                for i in 0..ITERS {
                    if run(gpu, &wbufs[i % reps]).is_err() {
                        return None;
                    }
                }
                let _ = gpu.hip.device_synchronize();
                Some(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
            };

            let us: Vec<Option<f64>> = (0..4).map(|w| time_variant(w, &mut gpu)).collect();
            let gbs = |o: Option<f64>| o.map(|u| one as f64 / 1e9 / (u / 1e6));
            let fmt_us = |o: Option<f64>| o.map(|u| format!("{u:.1}")).unwrap_or_else(|| "-".into());
            let fmt_gb = |o: Option<f64>| gbs(o).map(|g| format!("{g:.1}")).unwrap_or_else(|| "-".into());
            let names = ["b1", "b2", "b4", "gemv"];
            let winner = us
                .iter()
                .enumerate()
                .filter_map(|(i, o)| o.map(|u| (i, u)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let wtxt = match (winner, us[0]) {
                (Some((i, u)), Some(b1)) => format!("{} ({:.2}x vs b1)", names[i], b1 / u),
                _ => "-".into(),
            };
            println!(
                "    {:>3} {:>10} {:>10} {:>10} {:>10}   {:>9} {:>9} {:>9} {:>9}   {}",
                b,
                fmt_us(us[0]), fmt_us(us[1]), fmt_us(us[2]), fmt_us(us[3]),
                fmt_gb(us[0]), fmt_gb(us[1]), fmt_gb(us[2]), fmt_gb(us[3]),
                wtxt
            );
        }
        println!();
    }
}
