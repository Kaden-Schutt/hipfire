//! Numerical correctness test for the gfx906 dp4a MMQ kernel.
//!
//! Modes (selected via `MMQ_TEST_MODE` env var):
//!   residual (default): `gemm_hfq4g256_residual_mmq_gfx906` (add=1) vs
//!                       `gemm_hfq4g256_residual_fp16_wave64`. Both start
//!                       from the same non-zero Y; differences come from
//!                       Q8_1 vs FP16 quantization noise.
//!   set:                `gemm_hfq4g256_mmq_set_gfx906` (add=0) vs the
//!                       same FP16 wave64 reference started from Y=0.
//!                       Both produce Y = A·X^T.
//!
//! Usage: cargo run --release -p rdna-compute --example test_gfx906_mmq_correctness \
//!        -- [M] [K] [N]
//!
//! Defaults: M=128, K=256, N=64 (one full MMQ tile).

use rdna_compute::{DType, Gpu};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    assert!(k % 256 == 0, "K must be a multiple of 256");
    // M does not need to be a multiple of 128 — the bounds-checked
    // gfx906 MMQ entry guards on row index. Allow partial-M runs to
    // exercise the bounds-checked path.

    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    eprintln!("=== gfx906 MMQ correctness test ===");
    eprintln!("M={m} K={k} N={n}");
    eprintln!("groups_per_row={groups_per_row}");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("arch: {}", gpu.arch);

    if gpu.arch != "gfx906" {
        eprintln!("WARNING: this test is only meaningful on gfx906; skipping");
        std::process::exit(0);
    }

    // ── Random HFQ4-G256 weights (deterministic).
    let weight_bytes = synth_hfq4g256_weights(m, groups_per_row, 0xC0DE_FACEu64);
    let a_raw = gpu.upload_raw(&weight_bytes, &[m * row_bytes]).expect("upload weights");

    // ── Random activations.
    let x_host: Vec<f32> = (0..n * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();

    let x_tensor = gpu.upload_f32(&x_host, &[n * k]).expect("upload x");

    let mode = std::env::var("MMQ_TEST_MODE").unwrap_or_else(|_| "residual".to_string());
    let set_mode = mode == "set";
    eprintln!("test mode: {mode}");

    // residual: Y += A·X^T, both kernels start from non-zero y_init.
    // set:      Y  = A·X^T, FP16 ref starts from zero, MMQ from garbage.
    let y_init_host: Vec<f32> = if set_mode {
        vec![0.0f32; n * m]
    } else {
        (0..n * m)
            .map(|i| {
                let v = ((i as i64).wrapping_mul(2147483647).wrapping_add(7)) as f32;
                (v * 1e-7) % 1.0
            })
            .collect()
    };
    // For set-mode, prefill the MMQ output with garbage so we can verify
    // it actually overwrites (catches a "write-back skipped" bug).
    let y_mmq_init: Vec<f32> = if set_mode {
        (0..n * m).map(|i| 1e3 * ((i as f32) * 0.123).sin()).collect()
    } else {
        y_init_host.clone()
    };
    let y_mmq = gpu.upload_f32(&y_mmq_init, &[n * m]).expect("alloc y_mmq");
    let y_fp16 = gpu.upload_f32(&y_init_host, &[n * m]).expect("alloc y_fp16");

    let n_iter = std::env::var("HFQ_TEST_N_ITER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    eprintln!("running {n_iter} GEMM iterations on the same Y buffer");

    eprintln!("\n--- Running gemm_hfq4g256_residual_fp16_wave64 (reference) ---");
    for _ in 0..n_iter {
        gpu.gemm_hfq4g256_residual_fp16_wave64(&a_raw, &x_tensor, &y_fp16, m, k, n)
            .expect("fp16 wave64 launch");
    }
    gpu.hip.device_synchronize().expect("sync after fp16");

    if set_mode {
        eprintln!("--- Running gemm_hfq4g256_mmq_set_gfx906 (set, add=0) ---");
        // gemm_hfq4g256_mmq_set_gfx906 takes a pre-quantized Q8_1 X pointer.
        let xq_ptr = gpu.ensure_q8_1_mmq_x(&x_tensor, n, k).expect("quantize x → q8_1");
        for _ in 0..n_iter {
            gpu.gemm_hfq4g256_mmq_set_gfx906(&a_raw, xq_ptr, &y_mmq, m, k, n)
                .expect("mmq set gfx906 launch");
        }
    } else {
        eprintln!("--- Running gemm_hfq4g256_residual_mmq_gfx906 ---");
        for _ in 0..n_iter {
            gpu.gemm_hfq4g256_residual_mmq_gfx906(&a_raw, &x_tensor, &y_mmq, m, k, n)
                .expect("mmq gfx906 launch");
        }
    }
    gpu.hip.device_synchronize().expect("sync after mmq");

    let fp16_out = gpu.download_f32(&y_fp16).expect("download fp16");
    let mmq_out = gpu.download_f32(&y_mmq).expect("download mmq");

    eprintln!("\n--- Comparing outputs ---");
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut worst_idx = 0usize;
    let mut worst_pair = (0.0f32, 0.0f32);
    for i in 0..n * m {
        let r = fp16_out[i];
        let q = mmq_out[i];
        let err = (r - q).abs();
        if err > max_abs_err {
            max_abs_err = err;
            worst_idx = i;
            worst_pair = (r, q);
        }
        let rel = if r.abs() > 1e-6 { err / r.abs() } else { 0.0 };
        if rel > max_rel_err {
            max_rel_err = rel;
        }
        sum_sq_err += (err as f64).powi(2);
        sum_sq_ref += (r as f64).powi(2);
    }

    let rms_err = (sum_sq_err / (n * m) as f64).sqrt() as f32;
    let rms_ref = (sum_sq_ref / (n * m) as f64).sqrt() as f32;
    let nrmse = rms_err / rms_ref.max(1e-12);

    let ref_min = fp16_out.iter().copied().fold(f32::INFINITY, f32::min);
    let ref_max = fp16_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mmq_min = mmq_out.iter().copied().fold(f32::INFINITY, f32::min);
    let mmq_max = mmq_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let worst_col = worst_idx / m;
    let worst_row = worst_idx % m;
    eprintln!("max_abs_err  = {:.6e}", max_abs_err);
    eprintln!("max_rel_err  = {:.4}%", max_rel_err * 100.0);
    eprintln!("rms_err      = {:.6e}", rms_err);
    eprintln!("rms_ref      = {:.6e}", rms_ref);
    eprintln!("NRMSE        = {:.4}%", nrmse * 100.0);
    eprintln!("worst (col,row) = ({worst_col}, {worst_row})");
    eprintln!("                  fp16={:.6e}  mmq={:.6e}", worst_pair.0, worst_pair.1);
    eprintln!("ref range:  [{ref_min:.4e}, {ref_max:.4e}]");
    eprintln!("mmq range:  [{mmq_min:.4e}, {mmq_max:.4e}]");

    eprintln!("\n--- First 16 output cells (col=0, rows=0..15) ---");
    for i in 0..16.min(m) {
        eprintln!("  row {i}: fp16={:.6e}  mmq={:.6e}  diff={:.6e}",
            fp16_out[i], mmq_out[i], (fp16_out[i] - mmq_out[i]).abs());
    }

    // Pass criteria:
    //  - NRMSE < 1e-2 (1%) for Q8_1×HFQ4 quantization noise
    //  - MMQ output is non-zero (catches "kernel did nothing" bug)
    let mmq_nonzero = mmq_out.iter().any(|&v| v.abs() > 1e-12);
    let pass = nrmse < 1e-2 && mmq_nonzero;
    if pass {
        eprintln!("\nPASS (NRMSE within tolerance)");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL");
        if !mmq_nonzero {
            eprintln!("  mmq output is all-zero — kernel may not have run, or wrote to wrong location");
        }
        if nrmse >= 1e-2 {
            eprintln!("  NRMSE {:.4}% exceeds 1% threshold", nrmse * 100.0);
        }
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

    // Scale magnitude controlled by HFQ_TEST_SCALE_LOG10 env var (default -3
    // = scale ~1e-3). Real Qwen weights have scale magnitudes typically in
    // 1e-3 to 1e-1 and zp magnitudes in 1e-2 to 10. Use this to sweep.
    let scale_log10 = std::env::var("HFQ_TEST_SCALE_LOG10")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(-3.0);
    let zp_max = std::env::var("HFQ_TEST_ZP_MAX")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    let scale_target = 10.0f32.powf(scale_log10);
    eprintln!("synth weights: scale ~ {scale_target:.2e}, zp_max = {zp_max:.2}");

    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            // Random scale in [scale_target/2, scale_target*2], always positive
            let scale = scale_target * (0.5 + (next() & 0xFFFF) as f32 / 65535.0 * 1.5);
            // Random zp in [-zp_max, +zp_max]
            let zp = ((next() & 0xFFFF) as f32 / 65535.0) * 2.0 * zp_max - zp_max;
            out[gp..gp + 4].copy_from_slice(&scale.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}
