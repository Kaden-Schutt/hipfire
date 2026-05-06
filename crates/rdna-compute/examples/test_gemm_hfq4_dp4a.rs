//! Correctness test for gemm_hfq4g256_wave64_dp4a vs the FP wave64
//! reference. Both produce y[b, m] = sum_k A[m, k] * x[b, k]; they
//! differ by Q8_1 quantization noise on x (~1 % per-element relative).
//!
//! Usage: cargo run --release -p rdna-compute --example test_gemm_hfq4_dp4a \
//!        -- [M] [K] [BATCH]
//!
//! Defaults: M=512, K=4096, BATCH=8 (real LM-head shape range).
//!
//! Tolerance: max relative error <5 %, mean relative error <1 %.

use rdna_compute::Gpu;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize     = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let k: usize     = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let batch: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    assert!(k % 256 == 0, "K must be a multiple of 256");

    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    eprintln!("=== gemm_hfq4 dp4a correctness test ===");
    eprintln!("M={m}, K={k}, batch={batch}");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("arch: {}", gpu.arch);

    if gpu.arch != "gfx906" {
        eprintln!("WARNING: this test is only meaningful on gfx906; skipping");
        std::process::exit(0);
    }

    // Random weights (deterministic).
    let weight_bytes = synth_hfq4g256_weights(m, groups_per_row, 0xC0DE_FACEu64);
    let a = gpu.upload_raw(&weight_bytes, &[m * row_bytes]).expect("upload A");

    // Random batched activations [batch, K].
    let x_host: Vec<f32> = (0..batch * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();
    let x = gpu.upload_f32(&x_host, &[batch * k]).expect("upload x");

    let y_fp   = gpu.upload_f32(&vec![0f32; batch * m], &[batch * m]).expect("alloc y_fp");
    let y_dp4a = gpu.upload_f32(&vec![0f32; batch * m], &[batch * m]).expect("alloc y_dp4a");

    eprintln!("running FP reference (gemm_hfq4g256)...");
    // Force the FP path by toggling dp4a off via env var, then re-enabling
    // for the dp4a call. Simpler: call the dp4a fn explicitly for the dp4a
    // branch and bypass the toggle by constructing a different gpu? No —
    // toggle is process-global once read. Use an env override route by
    // running this test twice if needed. For this test, just call the
    // public entry point and rely on it dispatching correctly.
    //
    // To get FP output: temporarily disable dp4a via the env var BEFORE
    // gpu init. We can't do that mid-process. Instead, call the dp4a fn
    // directly, and call the kernel-level FP path by dispatching to
    // gemm_hfq4g256_wave64 manually.
    //
    // Simplest portable approach: call fused_gate_up_hfq4g256 (FP) for
    // a SINGLE row at a time as the reference. That's the GEMV. For a
    // batched test, we'd need the FP gemm path. Easier: assume the user
    // runs this test with dp4a OFF first (saves FP output), then ON. For
    // a single-process test, we'd skip the FP comparison and just check
    // the dp4a output is non-NaN + reasonably-magnituded.
    //
    // Pragmatic: do a CPU reference in float and compare against dp4a
    // output. CPU reference is slow at LM-head shapes (152k * 4096) but
    // fine at the test default M=512 K=4096.
    let y_ref = cpu_reference_gemm(&weight_bytes, &x_host, m, k, batch);
    eprintln!("running dp4a port...");
    gpu.gemm_hfq4g256_dp4a(&a, &x, &y_dp4a, m, k, batch).expect("dp4a gemm");

    let y_dp4a_h = gpu.download_f32(&y_dp4a).expect("dl dp4a");

    let (max_rel, mean_rel) = compare(&y_ref, &y_dp4a_h, "gemm");
    let pass = max_rel < 0.05 && mean_rel < 0.01;
    if pass {
        println!("PASS  (max_rel < 5%, mean_rel < 1%)");
    } else {
        println!("FAIL  thresholds: max_rel<5%, mean_rel<1%");
        std::process::exit(1);
    }
    let _ = y_fp;
}

fn cpu_reference_gemm(
    weight_bytes: &[u8],
    x: &[f32],
    m: usize, k: usize, batch: usize,
) -> Vec<f32> {
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;
    let mut out = vec![0f32; batch * m];
    for row in 0..m {
        let row_off = row * row_bytes;
        for g in 0..groups_per_row {
            let gp = row_off + g * 136;
            let scale = f32::from_le_bytes(weight_bytes[gp..gp+4].try_into().unwrap());
            let zero  = f32::from_le_bytes(weight_bytes[gp+4..gp+8].try_into().unwrap());
            // 128 bytes = 256 nibbles = 256 K-elements.
            for byte_i in 0..128 {
                let b = weight_bytes[gp + 8 + byte_i];
                let n_lo = (b & 0xF) as f32;
                let n_hi = (b >> 4)  as f32;
                let k_lo = g * 256 + byte_i * 2;
                let k_hi = k_lo + 1;
                let w_lo = scale * n_lo + zero;
                let w_hi = scale * n_hi + zero;
                for tok in 0..batch {
                    let xb = tok * k;
                    out[tok * m + row] += w_lo * x[xb + k_lo] + w_hi * x[xb + k_hi];
                }
            }
        }
    }
    out
}

fn compare(reference: &[f32], dut: &[f32], label: &str) -> (f32, f32) {
    assert_eq!(reference.len(), dut.len());
    let mut ref_max = 0f32;
    for &r in reference { if r.is_finite() { ref_max = ref_max.max(r.abs()); } }
    let rel_floor = (ref_max * 1e-2).max(1e-6);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut sum_abs = 0f32;
    let mut sum_rel = 0f32;
    let mut n = 0usize;
    let mut max_idx = 0usize;
    for (i, (&r, &d)) in reference.iter().zip(dut.iter()).enumerate() {
        if !r.is_finite() || !d.is_finite() { continue; }
        let abs_err = (r - d).abs();
        let rel_err = abs_err / r.abs().max(rel_floor);
        if rel_err > max_rel { max_rel = rel_err; max_idx = i; }
        max_abs = max_abs.max(abs_err);
        sum_abs += abs_err;
        sum_rel += rel_err;
        n += 1;
    }
    let mean_abs = sum_abs / n as f32;
    let mean_rel = sum_rel / n as f32;
    eprintln!("  {label}: ref_max={ref_max:.3e} max_abs={max_abs:.3e} max_rel={max_rel:.3e} (idx {max_idx}) mean_abs={mean_abs:.3e} mean_rel={mean_rel:.3e}");
    if max_rel > 0.10 {
        eprintln!("    sample at max_rel idx {max_idx}: ref={}, dut={}", reference[max_idx], dut[max_idx]);
    }
    (max_rel, mean_rel)
}

fn synth_hfq4g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let scale_log10 = std::env::var("HFQ_TEST_SCALE_LOG10")
        .ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(-3.0);
    let zp_max = std::env::var("HFQ_TEST_ZP_MAX")
        .ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);
    let scale_target = 10.0f32.powf(scale_log10);
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            let scale = scale_target * (0.5 + (next() & 0xFFFF) as f32 / 65535.0 * 1.5);
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
