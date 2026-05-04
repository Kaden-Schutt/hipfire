//! gfx12 (RDNA4) HFQ4-G256 iu4 K=32 GEMM with SmoothQuant-style activation
//! calibration — correctness test.
//!
//! This is the calibrated companion to `test_iu4_gfx12_correctness.rs`. It
//! exercises the calibrated dispatch path:
//!
//!   1. activation_preshift_gfx12     (x_centered = (x - mu) * inv_s)
//!   2. quantize_q4_1_mmq_ds4_gfx12   (Q4_1 of x_centered, unchanged)
//!   3. iu4_bake_weight_scales_gfx12  (W * s_group baked into a clone)
//!   4. gemm_hfq4g256_residual_iu4    (unchanged kernel; consumes baked W)
//!   5. bias_add_f32                   (broadcast W·mu_a bias on output)
//!
//! Compares against the existing FP16 dequant→WMMA reference. Calibrated
//! iu4 should be SUBSTANTIALLY tighter than raw iu4 (since the math
//! closes at group resolution + the activation-distribution shape is now
//! mean-zero / unit-variance per group rather than raw FP16).
//!
//! Calibration in this test is SYNTHETIC — we compute mu from the same
//! random activation matrix the test uses, then derive s_a from the same
//! data. That's the best-case "perfect" calibration; production calibration
//! has corpus-vs-runtime distribution drift on top.
//!
//! PASS thresholds (substantially tighter than raw-iu4's 0.30/0.15):
//!   max abs err        < 0.20
//!   mean abs err       < 0.02
//!   max rel err†       < 0.5
//!   mean rel err†      < 0.10
//!   pct rel-err > 10%† < 30%
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example test_iu4_calibration_correctness

use rdna_compute::{
    iu4_calibration::{GpuIu4CalSite, GpuIu4Calibration},
    DType, Gpu,
};

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

    eprintln!("=== gfx12 calibrated iu4 K=32 vs FP16-WMMA correctness test ===");
    eprintln!("M={m}, K={k}, N={n}");
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    let weight_bytes: Vec<u8> = synth_hfq4g256_weights(m, groups_per_row, 0xC0DE_FACEu64);
    let a_raw = gpu.upload_raw(&weight_bytes, &[m * row_bytes]).expect("upload weights");

    // Synthesize activation. Heavy-tail flavoring: 99% Gaussian-ish in
    // [-1, 1], 1% in [-5, 5] — this is what Q4_1 raw fails on and what
    // SmoothQuant should fix.
    let mut state: u64 = 0xDEAD_BEEF;
    let mut nrand = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32) / (u32::MAX as f32)
    };
    // Two regimes:
    //   - REGIME=normal (default): synthetic data with ±1 typical, ±5 outliers.
    //     Tests the math at the favorable activation scale.
    //   - REGIME=tiny: synthetic data scaled to ±0.03 range (mimicking real
    //     post-rmsnorm activations entering wo). Tests the bake math when
    //     inv_s ≈ 33 (huge per-channel scale).
    let regime = std::env::var("REGIME").unwrap_or_else(|_| "normal".to_string());
    let act_scale: f32 = if regime == "tiny" { 0.03 } else { 1.0 };
    let x_host: Vec<f32> = (0..n * k)
        .map(|_| {
            let u = nrand();
            let v = nrand() * 2.0 - 1.0;
            // 1% outliers at ±5×, the rest within ±1.
            let mag = if u > 0.99 { 5.0 } else { 1.0 };
            v * mag * act_scale
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

    // Path A: FP16 dequant → WMMA gfx12 reference (with the residual init).
    gpu.hip.memcpy_htod(&y_fp16.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.gemm_hfq4g256_residual_wmma_gfx12(&a_raw, &x_gpu, &y_fp16, m, k, n)
        .expect("fp16 wmma gfx12");
    gpu.hip.device_synchronize().unwrap();
    let y_fp16_host: Vec<f32> = gpu.download_f32(&y_fp16).expect("download y_fp16");

    // Build a synthetic calibration FROM THE TEST DATA. Compute per-channel
    // mu and s_a (99-pctile-abs after centering). Compute W·mu_a bias by
    // CPU-dequanting the synthetic weight and dotting against mu_a.
    eprintln!("--- building synthetic calibration from test data ---");
    let mu_a: Vec<f32> = compute_mu(&x_host, n, k);
    let s_a_per_channel: Vec<f32> = compute_p99_abs(&x_host, &mu_a, n, k);
    // Expand to group-shared form via geomean within each K=256 group.
    let s_group: Vec<f32> = s_group_geomean(&s_a_per_channel);
    let s_a_grouped: Vec<f32> = broadcast_s_group(&s_group);
    let inv_s_a_grouped: Vec<f32> = s_a_grouped
        .iter()
        .map(|&v| if v > 1e-6 { 1.0 / v } else { 1.0 })
        .collect();
    let bias: Vec<f32> = compute_bias(&weight_bytes, &mu_a, m, k);

    // Upload calibration to GPU.
    let mu_fp16: Vec<u16> = mu_a.iter().map(|&v| f32_to_f16(v)).collect();
    let inv_s_fp16: Vec<u16> = inv_s_a_grouped.iter().map(|&v| f32_to_f16(v)).collect();
    let s_group_fp16: Vec<u16> = s_group.iter().map(|&v| f32_to_f16(v)).collect();
    let mu_buf = gpu.hip.malloc(mu_fp16.len() * 2).unwrap();
    gpu.hip.memcpy_htod(&mu_buf, bytes_of_u16(&mu_fp16)).unwrap();
    let inv_s_buf = gpu.hip.malloc(inv_s_fp16.len() * 2).unwrap();
    gpu.hip.memcpy_htod(&inv_s_buf, bytes_of_u16(&inv_s_fp16)).unwrap();
    let s_group_buf = gpu.hip.malloc(s_group_fp16.len() * 2).unwrap();
    gpu.hip.memcpy_htod(&s_group_buf, bytes_of_u16(&s_group_fp16)).unwrap();
    let bias_buf = gpu.hip.malloc(bias.len() * 4).unwrap();
    gpu.hip.memcpy_htod(&bias_buf, bytes_of(&bias)).unwrap();

    let cal = GpuIu4Calibration {
        sites: vec![GpuIu4CalSite {
            layer_idx: 0,
            proj_id: 0,
            n_channels: k as u32,
            n_output_rows: m as u32,
            groups_per_row: groups_per_row as u32,
            mu_a: mu_buf,
            inv_s_a: inv_s_buf,
            s_group: s_group_buf,
            w_mu_bias_f32: bias_buf,
        }],
    };
    gpu.load_iu4_calibration(cal);

    // Path B: calibrated iu4. Set the env var so dispatch takes the path,
    // pre-init y, then call gemm_hfq4g256_residual which in turn dispatches
    // through the calibrated branch.
    std::env::set_var("HIPFIRE_GFX12_IU4_CALIBRATED", "1");
    gpu.hip.memcpy_htod(&y_iu4.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.reset_iu4_dispatch_counter();
    gpu.gemm_hfq4g256_residual(&a_raw, &x_gpu, &y_iu4, m, k, n)
        .expect("calibrated iu4 dispatch");
    gpu.hip.device_synchronize().unwrap();
    std::env::remove_var("HIPFIRE_GFX12_IU4_CALIBRATED");

    let y_iu4_host: Vec<f32> = gpu.download_f32(&y_iu4).expect("download y_iu4");

    assert_eq!(y_fp16_host.len(), n * m);
    assert_eq!(y_iu4_host.len(), n * m);

    // Compare.
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
    let mean_rel_err = if rel_eligible > 0 {
        sum_rel_err / rel_eligible as f64
    } else {
        0.0
    };
    let pct_above = 100.0 * samples_above_10pct as f32 / rel_eligible.max(1) as f32;

    eprintln!("\n--- per-channel error (n*m = {} elements) ---", n * m);
    eprintln!("  max abs err:           {:.6}  at (col={}, row={})", max_abs_err, max_loc.0, max_loc.1);
    eprintln!("  mean abs err:          {:.6}", mean_abs_err);
    eprintln!("  rel-err eligible:      {} / {} ({:.1}%)", rel_eligible, n * m,
        100.0 * rel_eligible as f32 / (n * m) as f32);
    eprintln!("  max rel err†:          {:.4}", max_rel_err);
    eprintln!("  mean rel err†:         {:.4}", mean_rel_err);
    eprintln!("  samples > 10% rel†:    {} / {} ({:.1}%)",
        samples_above_10pct, rel_eligible.max(1), pct_above);

    eprintln!("\n--- sample triples ---");
    for col in 0..2.min(n) {
        for row in 0..4.min(m) {
            let idx = col * m + row;
            let a = y_fp16_host[idx];
            let b = y_iu4_host[idx];
            eprintln!(
                "  col={col} row={row}: fp16={a:>10.4}  cal_iu4={b:>10.4}  err={:.4}",
                (a - b).abs()
            );
        }
    }

    // Tighter thresholds than raw iu4 — calibration should shrink errors
    // ~3-5× depending on activation tail mass.
    let max_abs_thresh = 0.20;
    let mean_abs_thresh = 0.02;
    let max_rel_thresh = 0.5;
    let mean_rel_thresh = 0.10;
    let pct_thresh = 30.0;

    let max_abs_ok = max_abs_err < max_abs_thresh;
    let mean_abs_ok = (mean_abs_err as f32) < mean_abs_thresh;
    let max_rel_ok = max_rel_err < max_rel_thresh;
    let mean_rel_ok = (mean_rel_err as f32) < mean_rel_thresh;
    let pct_ok = pct_above < pct_thresh;

    eprintln!("\n--- PASS criteria (calibrated iu4 — tighter than raw) ---");
    eprintln!("  max abs err   < {max_abs_thresh}:  {}", if max_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  mean abs err  < {mean_abs_thresh}: {}", if mean_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  max rel err†  < {max_rel_thresh}:  {}", if max_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  mean rel err† < {mean_rel_thresh}: {}", if mean_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  pct >10% rel† < {pct_thresh}%:    {}", if pct_ok { "OK" } else { "FAIL" });

    if max_abs_ok && mean_abs_ok && max_rel_ok && mean_rel_ok && pct_ok {
        eprintln!("\nPASS: gfx12 calibrated iu4 GEMM tighter than raw iu4 — SmoothQuant working.");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL: calibrated iu4 not converging within expected tolerance.");
        std::process::exit(1);
    }
}

fn compute_mu(x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut sum = vec![0.0f64; k];
    for t in 0..n {
        for c in 0..k {
            sum[c] += x[t * k + c] as f64;
        }
    }
    sum.into_iter().map(|s| (s / n as f64) as f32).collect()
}

fn compute_p99_abs(x: &[f32], mu: &[f32], n: usize, k: usize) -> Vec<f32> {
    // Not a true percentile — sample max-abs after centering, multiply by
    // 0.85 as a proxy for "above-99-pctile clipping". Fast and good enough
    // for synthetic test data where the distribution is symmetric.
    let mut max_abs = vec![0.0f32; k];
    for t in 0..n {
        for c in 0..k {
            let v = (x[t * k + c] - mu[c]).abs();
            if v > max_abs[c] {
                max_abs[c] = v;
            }
        }
    }
    max_abs.into_iter().map(|v| (v * 0.85).max(1e-4)).collect()
}

fn s_group_geomean(s_a: &[f32]) -> Vec<f32> {
    const GROUP: usize = 256;
    assert!(s_a.len() % GROUP == 0);
    let n_groups = s_a.len() / GROUP;
    let mut out = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let slice = &s_a[g * GROUP..(g + 1) * GROUP];
        let log_sum: f64 = slice.iter().map(|&v| (v as f64).ln()).sum();
        let g_mean = ((log_sum / GROUP as f64).exp()) as f32;
        out.push(g_mean.max(1e-6));
    }
    out
}

fn broadcast_s_group(s_group: &[f32]) -> Vec<f32> {
    const GROUP: usize = 256;
    let mut out = Vec::with_capacity(s_group.len() * GROUP);
    for &v in s_group {
        for _ in 0..GROUP {
            out.push(v);
        }
    }
    out
}

fn compute_bias(weight_bytes: &[u8], mu: &[f32], m: usize, k: usize) -> Vec<f32> {
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;
    let mut bias = vec![0.0f32; m];
    for r in 0..m {
        let row_off = r * row_bytes;
        let mut acc = 0.0f64;
        for g in 0..groups_per_row {
            let off = row_off + g * 136;
            let scale = f32::from_le_bytes([
                weight_bytes[off + 0],
                weight_bytes[off + 1],
                weight_bytes[off + 2],
                weight_bytes[off + 3],
            ]);
            let zero = f32::from_le_bytes([
                weight_bytes[off + 4],
                weight_bytes[off + 5],
                weight_bytes[off + 6],
                weight_bytes[off + 7],
            ]);
            for i in 0..256 {
                let nibbles = &weight_bytes[off + 8..off + 136];
                let byte_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    nibbles[byte_idx] & 0xF
                } else {
                    nibbles[byte_idx] >> 4
                };
                let w = scale * (nibble as f32) + zero;
                acc += (w as f64) * (mu[g * 256 + i] as f64);
            }
        }
        bias[r] = acc as f32;
    }
    bias
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
            let zp_ok = if zp.is_finite() && zp.abs() < 1.0 {
                zp
            } else {
                -0.5
            };
            out[gp..gp + 4].copy_from_slice(&scale_ok.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp_ok.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

// Local FP16 conversion (no engine-side dependency to keep this test
// self-contained inside rdna-compute).
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let frac_with_implicit = frac | (1 << 23);
        let shift = 14 - new_exp as u32;
        let f16_frac = (frac_with_implicit >> shift) as u16;
        return ((sign << 15) as u16) | f16_frac;
    }
    let f16_frac = (frac >> 13) as u16;
    ((sign << 15) as u16) | ((new_exp as u16) << 10) | f16_frac
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytes_of_u16(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}
