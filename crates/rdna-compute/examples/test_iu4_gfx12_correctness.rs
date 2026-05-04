//! gfx12 (RDNA4) HFQ4-G256 iu4 K=32 GEMM correctness test (issue #136 part B).
//!
//! Compares the new gfx12 iu4 K=32 residual GEMM kernel
//! (`gemm_hfq4g256_residual_iu4_gfx12`, commit faace73) + the Q4_1 activation
//! quantizer against the existing gfx12 FP16 dequant→WMMA reference path.
//!
//! Expected error sources:
//!   - Q4_1 activation quant: signed INT4 in [-7,7] = ~14% precision per
//!     element pre-accumulation. Bigger noise floor than Q8_1 (~0.8%) by ~8×.
//!   - Per-K=32 sub-block d/sum metadata round-trip.
//!   - HFQ4 weight is consumed AS-IS as iu4 — no FP8/FP16 cast loss on the
//!     weight side.
//!
//! PASS thresholds (looser than MMQ + tighter than FP8; calibrated for Q4_1):
//!   max abs err        < 0.30   (E4M3 was 0.20 — Q4_1 has bigger range hits)
//!   mean abs err       < 0.03
//!   max rel err†       < 1.0    (Q4_1 worst-case on near-floor outputs)
//!   mean rel err†      < 0.15   (15% per-element noise — Q4_1 reality)
//!   pct rel-err > 10%† < 50%    (Q4_1 has substantial elements above 10% rel)
//!
//! † Rel-err only on elements where |fp16_output| > REL_FLOOR (0.1).
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example test_iu4_gfx12_correctness

use rdna_compute::{DType, Gpu};

fn main() {
    let m: usize = std::env::var("M").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let k: usize = std::env::var("K").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(128);

    assert!(k % 256 == 0);
    assert!(m % 16 == 0);
    assert!(n % 16 == 0);

    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");

    if !(arch == "gfx1200" || arch == "gfx1201") {
        eprintln!("SKIP: requires gfx1200/gfx1201 (RDNA4). Current: {arch}");
        std::process::exit(0);
    }

    eprintln!("=== gfx12 iu4 K=32 vs FP16-WMMA correctness test ===");
    eprintln!("M={m}, K={k}, N={n}");
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    let weight_bytes: Vec<u8> = synth_hfq4g256_weights(m, groups_per_row, 0xC0DE_FACEu64);
    let a_raw = gpu.upload_raw(&weight_bytes, &[m * row_bytes]).expect("upload weights");

    let x_host: Vec<f32> = (0..n * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();
    let y_init_host: Vec<f32> = (0..n * m)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(2147483647).wrapping_add(7)) as f32;
            (v * 1e-7) % 1.0
        })
        .collect();

    let x_gpu = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
    let y_fp16 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_fp16");
    let y_iu4 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_iu4");

    gpu.hip.memcpy_htod(&x_gpu.buf, bytes_of(&x_host)).unwrap();

    // Path A: FP16 dequant → WMMA gfx12 reference.
    gpu.hip.memcpy_htod(&y_fp16.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.gemm_hfq4g256_residual_wmma_gfx12(&a_raw, &x_gpu, &y_fp16, m, k, n)
        .expect("fp16 wmma gfx12");
    gpu.hip.device_synchronize().unwrap();
    let y_fp16_host: Vec<f32> = gpu.download_f32(&y_fp16).expect("download y_fp16");

    // Path B: iu4 K=32 gfx12 (Q4_1 prequant + iu4 wmma).
    gpu.hip.memcpy_htod(&y_iu4.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let xq_ptr = gpu.ensure_q4_1_x(&x_gpu, n, k).expect("ensure_q4_1_x");
    gpu.gemm_hfq4g256_residual_iu4_gfx12(&a_raw, xq_ptr, &y_iu4, m, k, n, true)
        .expect("iu4 gfx12");
    gpu.hip.device_synchronize().unwrap();
    let y_iu4_host: Vec<f32> = gpu.download_f32(&y_iu4).expect("download y_iu4");

    assert_eq!(y_fp16_host.len(), n * m);
    assert_eq!(y_iu4_host.len(), n * m);

    const REL_FLOOR: f32 = 0.1;
    let mut max_abs_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    let mut sum_abs_err: f64 = 0.0;
    let mut sum_rel_err: f64 = 0.0;
    let mut max_loc: (usize, usize) = (0, 0);
    let mut samples_above_10pct: usize = 0;
    let mut rel_eligible: usize = 0;

    for col in 0..n {
        for row in 0..m {
            let idx = col * m + row;
            let a = y_fp16_host[idx];
            let b = y_iu4_host[idx];
            let err = (a - b).abs();
            if err > max_abs_err {
                max_abs_err = err;
                max_loc = (col, row);
            }
            sum_abs_err += err as f64;
            if a.abs() > REL_FLOOR {
                let rel = err / a.abs();
                if rel > max_rel_err { max_rel_err = rel; }
                sum_rel_err += rel as f64;
                rel_eligible += 1;
                if rel > 0.10 { samples_above_10pct += 1; }
            }
        }
    }

    let total = (n * m) as f64;
    let mean_abs_err = sum_abs_err / total;
    let mean_rel_err = if rel_eligible > 0 { sum_rel_err / (rel_eligible as f64) } else { 0.0 };
    let pct_above = 100.0 * samples_above_10pct as f32 / rel_eligible.max(1) as f32;

    eprintln!("\n--- per-channel error (n*m = {} elements) ---", n * m);
    eprintln!("  max abs err:                       {:.6}  at (col={}, row={})",
              max_abs_err, max_loc.0, max_loc.1);
    eprintln!("  mean abs err:                      {:.6}", mean_abs_err);
    eprintln!("  rel-err eligible (|out| > {:.2}):    {} / {} ({:.1}%)",
              REL_FLOOR, rel_eligible, n * m,
              100.0 * rel_eligible as f32 / (n * m) as f32);
    eprintln!("  max rel err†:                      {:.4}", max_rel_err);
    eprintln!("  mean rel err†:                     {:.4}", mean_rel_err);
    eprintln!("  samples > 10% rel†:                {} / {} ({:.3}%)",
              samples_above_10pct, rel_eligible.max(1), pct_above);
    eprintln!("  † counted only on non-near-zero outputs (|out| > {REL_FLOOR})");

    eprintln!("\n--- sample triples (col=0..2, row=0..4) ---");
    for col in 0..2.min(n) {
        for row in 0..4.min(m) {
            let idx = col * m + row;
            let a = y_fp16_host[idx];
            let b = y_iu4_host[idx];
            eprintln!("  col={col} row={row}: fp16={a:>10.4}  iu4={b:>10.4}  err={:.4}",
                      (a - b).abs());
        }
    }

    let max_abs_thresh = 0.30;
    let mean_abs_thresh = 0.03;
    let max_rel_thresh = 1.0;
    let mean_rel_thresh = 0.15;
    let pct_thresh = 50.0;

    let max_abs_ok = max_abs_err < max_abs_thresh;
    let mean_abs_ok = (mean_abs_err as f32) < mean_abs_thresh;
    let max_rel_ok = max_rel_err < max_rel_thresh;
    let mean_rel_ok = (mean_rel_err as f32) < mean_rel_thresh;
    let pct_ok = pct_above < pct_thresh;

    eprintln!("\n--- PASS criteria (Q4_1 activation = ~14% per-elem precision) ---");
    eprintln!("  max abs err   < {max_abs_thresh}:   {}", if max_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  mean abs err  < {mean_abs_thresh}:  {}", if mean_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  max rel err†  < {max_rel_thresh}:   {}", if max_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  mean rel err† < {mean_rel_thresh}:  {}", if mean_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  pct >10% rel† < {pct_thresh}%: {}", if pct_ok { "OK" } else { "FAIL" });

    if max_abs_ok && mean_abs_ok && max_rel_ok && mean_rel_ok && pct_ok {
        eprintln!("\nPASS: gfx12 iu4 K=32 GEMM is numerically equivalent to FP16 WMMA \
                   reference within Q4_1 tolerance.");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL: gfx12 iu4 K=32 GEMM diverges beyond Q4_1 tolerance.");
        std::process::exit(1);
    }
}

fn synth_hfq4g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            let scale_bits = 0x3a000000u32 | (next() & 0x007F_FFFF);
            let zp_bits = ((next() & 0x80) << 24) | 0x39000000u32 | (next() & 0x007F_FFFF);
            let scale = f32::from_bits(scale_bits);
            let zp = f32::from_bits(zp_bits);
            let scale_ok = if scale.is_finite() && scale.abs() < 1e-2 && scale > 0.0 { scale } else { 1e-3 };
            let zp_ok = if zp.is_finite() && zp.abs() < 1.0 { zp } else { -0.5 };
            out[gp..gp + 4].copy_from_slice(&scale_ok.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp_ok.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
