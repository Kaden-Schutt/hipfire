//! Correctness test for fused_gate_up_hfq4g256_wave64_dp4a vs the
//! FP wave64 reference. Both produce y_gate = A_gate · x and
//! y_up = A_up · x; they differ by Q8_1 quantization noise on x
//! (~1 % per-element relative).
//!
//! Usage: cargo run --release -p rdna-compute --example test_fused_gate_up_dp4a \
//!        -- [GATE_M] [UP_M] [K]
//!
//! Defaults: GATE_M=128, UP_M=128, K=4096 (small but real-shaped).
//!
//! Tolerance: max relative error <5 %, mean relative error <1 %.

use rdna_compute::Gpu;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gate_m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let up_m: usize   = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize      = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4096);

    assert!(k % 256 == 0, "K must be a multiple of 256");

    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    eprintln!("=== fused_gate_up dp4a correctness test ===");
    eprintln!("gate_m={gate_m}, up_m={up_m}, K={k}");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("arch: {}", gpu.arch);

    if gpu.arch != "gfx906" {
        eprintln!("WARNING: this test is only meaningful on gfx906; skipping");
        std::process::exit(0);
    }

    // Random weights (deterministic, two distinct seeds for gate/up).
    let gate_bytes = synth_hfq4g256_weights(gate_m, groups_per_row, 0xC0DE_FACEu64);
    let up_bytes   = synth_hfq4g256_weights(up_m,   groups_per_row, 0xDEAD_BEEFu64);

    let a_gate = gpu.upload_raw(&gate_bytes, &[gate_m * row_bytes]).expect("upload gate");
    let a_up   = gpu.upload_raw(&up_bytes,   &[up_m   * row_bytes]).expect("upload up");

    // Random activations.
    let x_host: Vec<f32> = (0..k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();
    let x = gpu.upload_f32(&x_host, &[k]).expect("upload x");

    // Allocate output tensors.
    let y_gate_fp   = gpu.upload_f32(&vec![0f32; gate_m], &[gate_m]).expect("alloc y_gate_fp");
    let y_up_fp     = gpu.upload_f32(&vec![0f32; up_m],   &[up_m]).expect("alloc y_up_fp");
    let y_gate_dp4a = gpu.upload_f32(&vec![0f32; gate_m], &[gate_m]).expect("alloc y_gate_dp4a");
    let y_up_dp4a   = gpu.upload_f32(&vec![0f32; up_m],   &[up_m]).expect("alloc y_up_dp4a");

    // Reference: FP wave64 path.
    eprintln!("running FP reference...");
    gpu.fused_gate_up_hfq4g256(&a_gate, &a_up, &x, &y_gate_fp, &y_up_fp, gate_m, up_m, k)
        .expect("FP fused_gate_up");

    // dp4a path.
    eprintln!("running dp4a port...");
    gpu.fused_gate_up_hfq4g256_dp4a(&a_gate, &a_up, &x, &y_gate_dp4a, &y_up_dp4a, gate_m, up_m, k)
        .expect("dp4a fused_gate_up");

    let yg_fp   = gpu.download_f32(&y_gate_fp).expect("dl yg_fp");
    let yu_fp   = gpu.download_f32(&y_up_fp).expect("dl yu_fp");
    let yg_dp4a = gpu.download_f32(&y_gate_dp4a).expect("dl yg_dp4a");
    let yu_dp4a = gpu.download_f32(&y_up_dp4a).expect("dl yu_dp4a");

    let (gate_max_rel, gate_mean_rel) = compare(&yg_fp, &yg_dp4a, "gate");
    let (up_max_rel,   up_mean_rel)   = compare(&yu_fp, &yu_dp4a, "up");

    let pass = gate_max_rel < 0.05 && up_max_rel < 0.05
            && gate_mean_rel < 0.01 && up_mean_rel < 0.01;
    if pass {
        println!("PASS  (max_rel < 5%, mean_rel < 1% on both gate and up)");
    } else {
        println!("FAIL  thresholds: max_rel<5%, mean_rel<1%");
        std::process::exit(1);
    }
}

fn compare(reference: &[f32], dut: &[f32], label: &str) -> (f32, f32) {
    assert_eq!(reference.len(), dut.len());
    // Compute the dynamic range of the reference so we can set an
    // absolute floor for the relative-error denominator. Q8_1 gives ~1%
    // per-element error on activations; near-zero outputs blow up
    // pure relative error. Floor the denominator at 1e-2 of the max
    // |reference| to keep the metric meaningful.
    let mut ref_max = 0f32;
    for &r in reference { if r.is_finite() { ref_max = ref_max.max(r.abs()); } }
    let rel_floor = (ref_max * 1e-2).max(1e-6);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut sum_abs = 0f32;
    let mut sum_rel = 0f32;
    let mut n_finite = 0usize;
    let mut max_idx = 0usize;
    for (i, (&r, &d)) in reference.iter().zip(dut.iter()).enumerate() {
        if !r.is_finite() || !d.is_finite() { continue; }
        let abs_err = (r - d).abs();
        let rel_err = abs_err / r.abs().max(rel_floor);
        if rel_err > max_rel { max_rel = rel_err; max_idx = i; }
        max_abs = max_abs.max(abs_err);
        sum_abs += abs_err;
        sum_rel += rel_err;
        n_finite += 1;
    }
    let mean_abs = sum_abs / n_finite as f32;
    let mean_rel = sum_rel / n_finite as f32;
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
