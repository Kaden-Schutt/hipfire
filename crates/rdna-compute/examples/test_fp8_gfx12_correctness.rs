//! gfx12 (RDNA4) HFQ4-G256 FP8 GEMM correctness test (issue #136 part B).
//!
//! Compares the new gfx12 FP8 (E4M3) residual GEMM kernel
//! (`gemm_hfq4g256_residual_fp8_gfx12`, commit c77e1d1) against the existing
//! gfx12 FP16 dequant→WMMA reference path. Same shape as the iu8 MMQ
//! correctness test (test_mmq_gfx12_correctness.rs) but with looser
//! tolerances since FP8 has ~3-bit mantissa precision (vs Q8_1's ~7 bits).
//!
//! Expected error sources:
//!   - FP8 weight cast: HFQ4 dequant → FP32 → FP8 introduces E4M3 rounding
//!     (~3% relative per weight element).
//!   - FP8 activation cast: FP32 → FP8 with per-(tile_y col) dynamic scale.
//!     Adds another rounding pass.
//!   - Saturation: if any activation × scale exceeds E4M3_MAX (=448), it
//!     saturates. Synthetic test inputs are bounded to avoid this.
//!
//! PASS thresholds (FP8-realistic, calibrated against R9700 measurements):
//!   max abs err        < 0.20   (catches algorithmic bugs: layout, saturation, bounds)
//!   mean abs err       < 0.02   (average drift bound)
//!   max rel err†       < 0.75   (E4M3 worst-case on small-magnitude outputs)
//!   mean rel err†      < 0.10   (per-element distribution drift)
//!   pct rel-err > 10%† < 25%    (thin-tail bound — FP8 has more than MMQ)
//!
//! Rel-err metrics are looser than the MMQ test by an order of magnitude
//! because FP8 has materially worse per-element precision than Q8_1.
//! Empirically at K=512, mean_rel_err≈1.6%; at K=2048 it grows to ~6%
//! (super-√K — likely amax-statistic noise widening with K in the per-kb
//! scale calc). Production K runs 2K-27K, so thresholds are sized for that.
//!
//! † Rel-err only on elements where |fp16_output| > REL_FLOOR (0.1).
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example test_fp8_gfx12_correctness

use rdna_compute::{DType, Gpu};

fn main() {
    let m: usize = std::env::var("M").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let k: usize = std::env::var("K").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(128);

    assert!(k % 256 == 0, "K must be a multiple of 256 for HFQ4-G256");
    assert!(m % 16 == 0);
    assert!(n % 16 == 0);

    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");

    if !(arch == "gfx1200" || arch == "gfx1201") {
        eprintln!(
            "SKIP: this test requires gfx1200/gfx1201 (RDNA4). \
             Current arch: {arch}. The gfx12 FP8 GEMM kernel only exists on RDNA4."
        );
        std::process::exit(0);
    }

    eprintln!("=== gfx12 FP8 vs FP16-WMMA correctness test ===");
    eprintln!("M={m}, K={k}, N={n}");
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    // Random HFQ4-G256 weights (deterministic).
    let weight_bytes: Vec<u8> = synth_hfq4g256_weights(m, groups_per_row, 0xC0DE_FACEu64);
    let a_raw = gpu.upload_raw(&weight_bytes, &[m * row_bytes]).expect("upload weights");

    // Random FP32 activations (deterministic) bounded to ±1 (well under E4M3_MAX=448).
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
    let y_fp8 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_fp8");

    gpu.hip.memcpy_htod(&x_gpu.buf, bytes_of(&x_host)).unwrap();

    // Path A: FP16 dequant → WMMA gfx12 reference.
    gpu.hip.memcpy_htod(&y_fp16.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.gemm_hfq4g256_residual_wmma_gfx12(&a_raw, &x_gpu, &y_fp16, m, k, n)
        .expect("fp16 wmma gfx12");
    gpu.hip.device_synchronize().unwrap();
    let y_fp16_host: Vec<f32> = gpu.download_f32(&y_fp16).expect("download y_fp16");

    // Path B: FP8 GEMM gfx12 (in-kernel HFQ4 dequant → FP8 → wmma).
    gpu.hip.memcpy_htod(&y_fp8.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.gemm_hfq4g256_residual_fp8_gfx12(&a_raw, &x_gpu, &y_fp8, m, k, n)
        .expect("fp8 gfx12");
    gpu.hip.device_synchronize().unwrap();
    let y_fp8_host: Vec<f32> = gpu.download_f32(&y_fp8).expect("download y_fp8");

    assert_eq!(y_fp16_host.len(), n * m);
    assert_eq!(y_fp8_host.len(), n * m);

    // Compare per channel; rel-err floor at |out| > 0.1 (looser than the MMQ
    // test's 0.05 floor since FP8 noise is larger).
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
            let b = y_fp8_host[idx];
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

    // Sample triples for human eyeball.
    eprintln!("\n--- sample triples (col=0..2, row=0..4) ---");
    for col in 0..2.min(n) {
        for row in 0..4.min(m) {
            let idx = col * m + row;
            let a = y_fp16_host[idx];
            let b = y_fp8_host[idx];
            eprintln!("  col={col} row={row}: fp16={a:>10.4}  fp8={b:>10.4}  err={:.4}", (a - b).abs());
        }
    }

    // PASS thresholds. Substantially looser than the MMQ test because:
    //   - E4M3 has ~3-bit mantissa (vs Q8_1's effective ~7 bits)
    //   - Per-tile dynamic scale on activations adds another rounding pass
    //   - Empirically: at K=512, mean_rel_err is ~1.6%; at K=2048 it grows to
    //     ~6% (super-√K scaling — likely amax-statistic noise widening with K
    //     in the per-kb scale calculation). Production K runs 2K-27K.
    // The max-abs metric is the meaningful "did the kernel work" signal; rel
    // metrics catch distribution drift but need to tolerate FP8 reality.
    let max_abs_thresh = 0.20;     // catches algorithmic bugs (saturation, layout, bounds)
    let mean_abs_thresh = 0.02;    // average output drift bound
    let max_rel_thresh = 0.75;     // E4M3 worst-case on small-magnitude outputs
    let mean_rel_thresh = 0.10;    // per-element distribution drift (10% accepted for FP8)
    let pct_thresh = 25.0;         // up to 25% of elements may exceed 10% rel-err on FP8

    let max_abs_ok = max_abs_err < max_abs_thresh;
    let mean_abs_ok = (mean_abs_err as f32) < mean_abs_thresh;
    let max_rel_ok = max_rel_err < max_rel_thresh;
    let mean_rel_ok = (mean_rel_err as f32) < mean_rel_thresh;
    let pct_ok = pct_above < pct_thresh;

    eprintln!("\n--- PASS criteria (FP8 E4M3 expected ~3-bit mantissa precision) ---");
    eprintln!("  max abs err   < {max_abs_thresh}:   {}", if max_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  mean abs err  < {mean_abs_thresh}:   {}", if mean_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  max rel err†  < {max_rel_thresh}:   {}", if max_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  mean rel err† < {mean_rel_thresh}:   {}", if mean_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  pct >10% rel† < {pct_thresh}%: {}", if pct_ok { "OK" } else { "FAIL" });

    if max_abs_ok && mean_abs_ok && max_rel_ok && mean_rel_ok && pct_ok {
        eprintln!("\nPASS: gfx12 FP8 GEMM is numerically equivalent to FP16 WMMA reference \
                   within E4M3 precision tolerance.");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL: gfx12 FP8 GEMM diverges from FP16 WMMA reference beyond E4M3 \
                   tolerance. Investigate before flipping any default routing.");
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
