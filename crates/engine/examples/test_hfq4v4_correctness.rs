//! gfx12 (RDNA4) HFQ4v4 iu4 K=32 GEMM correctness test.
//!
//! Sister of `test_iu4_gfx12_correctness.rs` (the v1 / PR #140 test). Tests
//! the new `gemm_hfq4v4_residual_iu4_gfx12` kernel against the FP16 dequant→
//! WMMA reference on synthetic weights:
//!
//!   1. Generate a random HFQ4-G256 weight blob (the same shape used by the
//!      v1 test, so we can rerun back-to-back).
//!   2. Dequantize to FP32 → run FP16 WMMA reference (path A).
//!   3. Apply the offline HFQ4-G256 → HFQ4v4 conversion (per-row mu absorbed,
//!      K=32 groups with FP16 d, with or without FWHT-32 rotation).
//!   4. Quantize activations to Q4_1 (existing quantizer).
//!   5. Run `gemm_hfq4v4_residual_iu4_gfx12` (path B).
//!   6. Compare path A vs path B per element.
//!
//! PASS thresholds (tighter than v1 because the mu correction recovers the
//! per-channel signal that v1's symmetric Q4 activations clipped):
//!
//!   max abs err        < 0.30
//!   mean abs err       < 0.03
//!   mean rel err†      < 0.10    (vs v1's 0.15)
//!   pct rel-err > 10%† < 40%     (vs v1's 50)
//!
//! † Rel-err only on elements where |fp16_output| > REL_FLOOR (0.1).
//!
//! Run on gfx1201 (hiptrx):
//!   cargo run --release -p rdna-compute --example test_hfq4v4_correctness
//!
//! Env knobs:
//!   M, K, N    — matrix dims (default 256 / 512 / 128)
//!   ROTATE=1   — apply FWHT-32 rotation (mq4v4 variant)

use engine::hfq4v4::{convert_hfq4g256_to_hfq4v4, MuStrategy};
use rdna_compute::{DType, Gpu};

fn main() {
    let m: usize = std::env::var("M").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let k: usize = std::env::var("K").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let rotate = std::env::var("ROTATE").ok().as_deref() == Some("1");

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

    eprintln!("=== gfx12 HFQ4v4 iu4 K=32 vs FP16-WMMA correctness test ===");
    eprintln!("M={m}, K={k}, N={n}, rotate={rotate}");

    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    let weight_bytes_v1: Vec<u8> = synth_hfq4g256_weights(m, groups_per_row, 0xC0DE_FACEu64);
    let a_raw_v1 = gpu.upload_raw(&weight_bytes_v1, &[m * row_bytes]).expect("upload v1 weights");

    // Convert to v4 offline.
    let (w_v4, mu_v4) = convert_hfq4g256_to_hfq4v4(
        &weight_bytes_v1, m, k, rotate, &MuStrategy::WeightMean,
    );
    let v4_weight_size = w_v4.len();
    let v4_mu_size = mu_v4.len();
    eprintln!(
        "  v4 weight blob: {} bytes ({:.2} bits/weight)",
        v4_weight_size,
        (v4_weight_size as f32 * 8.0) / (m * k) as f32
    );
    eprintln!(
        "  v4 mu sidecar:  {} bytes ({} FP16 values)",
        v4_mu_size,
        m
    );
    let a_raw_v4 = gpu.upload_raw(&w_v4, &[v4_weight_size]).expect("upload v4 weights");
    let mu_t = gpu.upload_raw(&mu_v4, &[v4_mu_size]).expect("upload v4 mu");

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

    // For mq4v4 (rotate=true), we must FWHT-32 the activations before the
    // Q4_1 quantizer sees them, because the weight side was rotated. The
    // Q4_1 quantizer doesn't know about that — we apply the rotation on the
    // CPU here for the test path. (Production dispatch will need a GPU
    // FWHT-32 kernel, but that's a follow-up; for the correctness gate we
    // emulate it on host.)
    let x_for_q4_1: Vec<f32> = if rotate {
        let mut buf = x_host.clone();
        for col in 0..n {
            // Each col is a "token" with K elements laid out as buf[col*k..col*k+k].
            // Wait: x_host layout — looking at the v1 test it's organized as
            // [n][k] flat: idx = col * k + ki (col-major batch). Confirm by
            // looking at how the wmma reference uses it...
            // The v1 test uploads x_host as length n*k and the reference
            // path (gemm_hfq4g256_residual_wmma_gfx12) treats it as F32 of
            // shape [n*k]. Inside the kernel it interprets as [N][K]
            // row-major (col c at offset c*k). So idx = col*k + ki, which
            // matches our assumption.
            let off = col * k;
            for g in 0..(k / 32) {
                let go = off + g * 32;
                let mut grp = [0f32; 32];
                grp.copy_from_slice(&buf[go..go + 32]);
                engine::hfq4v4::fwht_32(&mut grp);
                buf[go..go + 32].copy_from_slice(&grp);
            }
        }
        buf
    } else {
        x_host.clone()
    };

    let x_gpu = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
    let x_gpu_for_q4_1 = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x rot");
    let y_fp16 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_fp16");
    let y_v4 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_v4");

    gpu.hip.memcpy_htod(&x_gpu.buf, bytes_of(&x_host)).unwrap();
    gpu.hip.memcpy_htod(&x_gpu_for_q4_1.buf, bytes_of(&x_for_q4_1)).unwrap();

    // Path A: FP16 dequant → WMMA gfx12 reference (using the ORIGINAL,
    // un-rotated weights). This computes the ground truth y[col, row] = sum_k
    // W[row, k] * X[col, k] + y_init[col, row].
    gpu.hip.memcpy_htod(&y_fp16.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.gemm_hfq4g256_residual_wmma_gfx12(&a_raw_v1, &x_gpu, &y_fp16, m, k, n)
        .expect("fp16 wmma gfx12");
    gpu.hip.device_synchronize().unwrap();
    let y_fp16_host: Vec<f32> = gpu.download_f32(&y_fp16).expect("download y_fp16");

    // Path B: HFQ4v4 + iu4 K=32 with mu correction.
    gpu.hip.memcpy_htod(&y_v4.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let xq_ptr = gpu.ensure_q4_1_x(&x_gpu_for_q4_1, n, k).expect("ensure_q4_1_x");
    gpu.gemm_hfq4v4_residual_iu4_gfx12(&a_raw_v4, &mu_t, xq_ptr, &y_v4, m, k, n, true)
        .expect("hfq4v4 iu4 gfx12");
    gpu.hip.device_synchronize().unwrap();
    let y_v4_host: Vec<f32> = gpu.download_f32(&y_v4).expect("download y_v4");

    assert_eq!(y_fp16_host.len(), n * m);
    assert_eq!(y_v4_host.len(), n * m);

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
            let b = y_v4_host[idx];
            let err = (a - b).abs();
            if err > max_abs_err {
                max_abs_err = err;
                max_loc = (col, row);
            }
            sum_abs_err += err as f64;
            if a.abs() > REL_FLOOR {
                let rel = err / a.abs();
                if rel > max_rel_err {
                    max_rel_err = rel;
                }
                sum_rel_err += rel as f64;
                rel_eligible += 1;
                if rel > 0.10 {
                    samples_above_10pct += 1;
                }
            }
        }
    }

    let total = (n * m) as f64;
    let mean_abs_err = sum_abs_err / total;
    let mean_rel_err = if rel_eligible > 0 { sum_rel_err / (rel_eligible as f64) } else { 0.0 };
    let pct_above = 100.0 * samples_above_10pct as f32 / rel_eligible.max(1) as f32;

    eprintln!("\n--- per-channel error (n*m = {} elements) ---", n * m);
    eprintln!(
        "  max abs err:                       {:.6}  at (col={}, row={})",
        max_abs_err, max_loc.0, max_loc.1
    );
    eprintln!("  mean abs err:                      {:.6}", mean_abs_err);
    eprintln!(
        "  rel-err eligible (|out| > {:.2}):    {} / {} ({:.1}%)",
        REL_FLOOR,
        rel_eligible,
        n * m,
        100.0 * rel_eligible as f32 / (n * m) as f32
    );
    eprintln!("  max rel err†:                      {:.4}", max_rel_err);
    eprintln!("  mean rel err†:                     {:.4}", mean_rel_err);
    eprintln!(
        "  samples > 10% rel†:                {} / {} ({:.3}%)",
        samples_above_10pct,
        rel_eligible.max(1),
        pct_above
    );
    eprintln!("  † counted only on non-near-zero outputs (|out| > {REL_FLOOR})");

    eprintln!("\n--- sample triples (col=0..2, row=0..4) ---");
    for col in 0..2.min(n) {
        for row in 0..4.min(m) {
            let idx = col * m + row;
            let a = y_fp16_host[idx];
            let b = y_v4_host[idx];
            eprintln!(
                "  col={col} row={row}: fp16={a:>10.4}  v4={b:>10.4}  err={:.4}",
                (a - b).abs()
            );
        }
    }

    let max_abs_thresh = 0.30;
    let mean_abs_thresh = 0.03;
    let mean_rel_thresh = 0.10;
    let pct_thresh = 40.0;

    let max_abs_ok = max_abs_err < max_abs_thresh;
    let mean_abs_ok = (mean_abs_err as f32) < mean_abs_thresh;
    let mean_rel_ok = (mean_rel_err as f32) < mean_rel_thresh;
    let pct_ok = pct_above < pct_thresh;

    eprintln!("\n--- PASS criteria (Q4_1 + per-row mu correction) ---");
    eprintln!("  max abs err   < {max_abs_thresh}:   {}", if max_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  mean abs err  < {mean_abs_thresh}:  {}", if mean_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  mean rel err† < {mean_rel_thresh}:  {}", if mean_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  pct >10% rel† < {pct_thresh}%: {}", if pct_ok { "OK" } else { "FAIL" });

    if max_abs_ok && mean_abs_ok && mean_rel_ok && pct_ok {
        eprintln!(
            "\nPASS: gfx12 HFQ4v4 GEMM stays within tolerance vs FP16 WMMA reference."
        );
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL: gfx12 HFQ4v4 GEMM diverges beyond tolerance.");
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
            let scale_ok = if scale.is_finite() && scale.abs() < 1e-2 && scale > 0.0 {
                scale
            } else {
                1e-3
            };
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
