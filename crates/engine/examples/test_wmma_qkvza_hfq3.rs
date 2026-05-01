//! Channel-test for `gemm_qkvza_hfq3g256_wmma` (gfx11 K2 variant).
//!
//! 4-output sister of test_wmma_qkvza_gfx12 — exercises qkv/z/beta/alpha
//! routing for the DeltaNet LinearAttention preamble at HFQ3 storage.
//! Compares the new WMMA kernel against a CPU reference that decodes the
//! same packed HFQ3 bytes via the unpack pattern from gemv_hfq3g256.hip.
//!
//! Run: cargo run --release --features deltanet -p engine \
//!         --example test_wmma_qkvza_hfq3

use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {} ({:.1} GB VRAM)", arch, {
        let (_, total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
        total as f64 / 1e9
    });

    let supported = matches!(arch.as_str(), "gfx1100" | "gfx1101" | "gfx1102");
    if !supported {
        eprintln!(
            "SKIP: gemm_qkvza_hfq3g256_wmma is the gfx11 K2 variant. \
             Current arch: {arch}. (gfx12 K4 variant: follow-up commit.)"
        );
        std::process::exit(0);
    }

    let mut total_pass = 0;
    let mut total_fail = 0;

    // (qkv_m, z_m, beta_m, alpha_m, K, N)
    let shapes: &[(usize, usize, usize, usize, usize, usize)] = &[
        // Single 16-row tile per projection — minimum to exercise routing.
        (16, 16, 16, 16, 256, 16),
        // K = 2 groups
        (16, 16, 16, 16, 512, 16),
        // batch = 2 tiles
        (16, 16, 16, 16, 256, 32),
        // qkv has 2 row-tiles
        (32, 16, 16, 16, 256, 16),
        // multi-tile in every dim
        (32, 32, 16, 16, 512, 32),
        // Realistic Qwen3.5-9B DeltaNet LA preamble shape (qkv_m=8192,
        // z_m=4096, alpha=beta=32 — but those don't divide 16 so use the
        // smallest that exercises the boundary straddle).
        (48, 16, 16, 16, 256, 16),
    ];

    for &(qkv_m, z_m, beta_m, alpha_m, k, n) in shapes {
        let label = format!("qkv={qkv_m} z={z_m} b={beta_m} a={alpha_m} K={k} N={n}");
        eprintln!("\n--- {label} ---");
        match run_one(&mut gpu, qkv_m, z_m, beta_m, alpha_m, k, n) {
            Ok(()) => {
                total_pass += 1;
                eprintln!("  {label:60} OK");
            }
            Err(e) => {
                total_fail += 1;
                eprintln!("  {label:60} FAIL");
                eprintln!("{e}");
            }
        }
    }

    eprintln!("\n--- Summary ---");
    eprintln!("  Passed: {total_pass}");
    eprintln!("  Failed: {total_fail}");
    if total_fail > 0 {
        std::process::exit(1);
    }
}

fn run_one(
    gpu: &mut Gpu,
    qkv_m: usize,
    z_m: usize,
    beta_m: usize,
    alpha_m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    assert_eq!(qkv_m % 16, 0);
    assert_eq!(z_m % 16, 0);
    assert_eq!(beta_m % 16, 0);
    assert_eq!(alpha_m % 16, 0);
    assert_eq!(k % 256, 0);
    assert_eq!(n % 16, 0);

    let aq_bytes = build_hfq3g256(qkv_m, k, 0xA1);
    let az_bytes = build_hfq3g256(z_m, k, 0xB2);
    let ab_bytes = build_hfq3g256(beta_m, k, 0xC3);
    let aa_bytes = build_hfq3g256(alpha_m, k, 0xD4);

    // CPU reference: per (batch, row), dequant the HFQ3 bytes and dot
    // against x.
    let x_f32: Vec<f32> = (0..(n * k))
        .map(|i| {
            let b = (i / k) as i32;
            let kk = (i % k) as i32;
            ((b * 7 + kk * 11) % 31 - 15) as f32 * 0.05
        })
        .collect();

    let ref_q = cpu_gemm(&aq_bytes, qkv_m, k, &x_f32, n);
    let ref_z = cpu_gemm(&az_bytes, z_m, k, &x_f32, n);
    let ref_b = cpu_gemm(&ab_bytes, beta_m, k, &x_f32, n);
    let ref_a = cpu_gemm(&aa_bytes, alpha_m, k, &x_f32, n);

    let aq = gpu.upload_raw(&aq_bytes, &[qkv_m, k]).map_err(|e| format!("upload aq: {e}"))?;
    let az = gpu.upload_raw(&az_bytes, &[z_m, k]).map_err(|e| format!("upload az: {e}"))?;
    let ab = gpu.upload_raw(&ab_bytes, &[beta_m, k]).map_err(|e| format!("upload ab: {e}"))?;
    let aa = gpu.upload_raw(&aa_bytes, &[alpha_m, k]).map_err(|e| format!("upload aa: {e}"))?;
    let x = gpu.upload_f32(&x_f32, &[n, k]).map_err(|e| format!("upload x: {e}"))?;

    let yq = gpu.alloc_tensor(&[n, qkv_m], DType::F32).map_err(|e| format!("alloc yq: {e}"))?;
    let yz = gpu.alloc_tensor(&[n, z_m], DType::F32).map_err(|e| format!("alloc yz: {e}"))?;
    let yb = gpu.alloc_tensor(&[n, beta_m], DType::F32).map_err(|e| format!("alloc yb: {e}"))?;
    let ya = gpu.alloc_tensor(&[n, alpha_m], DType::F32).map_err(|e| format!("alloc ya: {e}"))?;

    gpu.gemm_qkvza_hfq3g256_wmma(
        &aq, &az, &ab, &aa, &x,
        &yq, &yz, &yb, &ya,
        qkv_m, z_m, beta_m, alpha_m, k, n,
    )
    .map_err(|e| format!("wmma: {e}"))?;

    let cand_q = gpu.download_f32(&yq).map_err(|e| format!("download yq: {e}"))?;
    let cand_z = gpu.download_f32(&yz).map_err(|e| format!("download yz: {e}"))?;
    let cand_b = gpu.download_f32(&yb).map_err(|e| format!("download yb: {e}"))?;
    let cand_a = gpu.download_f32(&ya).map_err(|e| format!("download ya: {e}"))?;

    // Intentionally NOT freeing tensors here — the GPU allocator can
    // recycle freed addresses, and `ensure_fp16_x` keys its conversion
    // cache by source pointer. If a recycled X address matches the
    // previous test's pointer with the same total `n_elems`, the cache
    // would hit and return stale fp16 data with the prior test's stride.
    // (Real production code paths don't hit this because each layer's X
    // is a unique tensor with stable identity for its full lifetime.)
    let _keep_alive = (aq, az, ab, aa, x, yq, yz, yb, ya);

    let mut report = String::new();
    let ok_q = compare_proj("Y_qkv", n, qkv_m, &cand_q, &ref_q, &mut report);
    let ok_z = compare_proj("Y_z", n, z_m, &cand_z, &ref_z, &mut report);
    let ok_b = compare_proj("Y_beta", n, beta_m, &cand_b, &ref_b, &mut report);
    let ok_a = compare_proj("Y_alpha", n, alpha_m, &cand_a, &ref_a, &mut report);

    if ok_q && ok_z && ok_b && ok_a {
        Ok(())
    } else {
        // Dump first 4 batches × first 4 rows for Y_qkv to localize.
        use std::fmt::Write;
        writeln!(&mut report, "    Y_qkv head dump (cand vs ref):").ok();
        for b in 0..n.min(4) {
            for r in 0..qkv_m.min(4) {
                let idx = b * qkv_m + r;
                writeln!(
                    &mut report,
                    "      [b={b} r={r}] cand={:.4}  ref={:.4}  diff={:.4e}",
                    cand_q[idx], ref_q[idx], cand_q[idx] - ref_q[idx]
                ).ok();
            }
        }
        Err(report)
    }
}

/// CPU reference: replicates `gemv_hfq3g256.hip` per-row for each of the
/// N batch elements, and stores Y row-major as [batch × m].
fn cpu_gemm(a_bytes: &[u8], m: usize, k: usize, x: &[f32], n: usize) -> Vec<f32> {
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 104;
    let mut y = vec![0f32; n * m];

    for b in 0..n {
        let x_row = &x[b * k..(b + 1) * k];
        for row in 0..m {
            let row_off = row * row_bytes;
            let mut acc = 0f32;
            for g in 0..groups_per_row {
                let goff = row_off + g * 104;
                let scale_bits = u32::from_le_bytes([
                    a_bytes[goff],
                    a_bytes[goff + 1],
                    a_bytes[goff + 2],
                    a_bytes[goff + 3],
                ]);
                let zero_bits = u32::from_le_bytes([
                    a_bytes[goff + 4],
                    a_bytes[goff + 5],
                    a_bytes[goff + 6],
                    a_bytes[goff + 7],
                ]);
                let scale = f32::from_bits(scale_bits);
                let zero = f32::from_bits(zero_bits);
                // 32 chunks × 8 weights × 3 bytes = 96 bytes data.
                for chunk in 0..32 {
                    let bo = goff + 8 + chunk * 3;
                    let b0 = a_bytes[bo] as u32;
                    let b1 = a_bytes[bo + 1] as u32;
                    let b2 = a_bytes[bo + 2] as u32;
                    let q0 = b0 & 7;
                    let q1 = (b0 >> 3) & 7;
                    let q2 = ((b0 >> 6) | (b1 << 2)) & 7;
                    let q3 = (b1 >> 1) & 7;
                    let q4 = (b1 >> 4) & 7;
                    let q5 = ((b1 >> 7) | (b2 << 1)) & 7;
                    let q6 = (b2 >> 2) & 7;
                    let q7 = (b2 >> 5) & 7;
                    let base = g * 256 + chunk * 8;
                    acc += (scale * (q0 as f32) + zero) * x_row[base];
                    acc += (scale * (q1 as f32) + zero) * x_row[base + 1];
                    acc += (scale * (q2 as f32) + zero) * x_row[base + 2];
                    acc += (scale * (q3 as f32) + zero) * x_row[base + 3];
                    acc += (scale * (q4 as f32) + zero) * x_row[base + 4];
                    acc += (scale * (q5 as f32) + zero) * x_row[base + 5];
                    acc += (scale * (q6 as f32) + zero) * x_row[base + 6];
                    acc += (scale * (q7 as f32) + zero) * x_row[base + 7];
                }
            }
            y[b * m + row] = acc;
        }
    }
    y
}

fn compare_proj(
    name: &str,
    n: usize,
    m: usize,
    cand: &[f32],
    refr: &[f32],
    report: &mut String,
) -> bool {
    assert_eq!(cand.len(), refr.len());
    assert_eq!(cand.len(), n * m);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut n_bad = 0usize;
    let mut first_bad: Option<(usize, usize, f32, f32)> = None;
    let abs_tol = 5e-2_f32;
    let rel_tol = 1e-2_f32;
    let mut hist_row_mod16 = [0usize; 16];

    for batch in 0..n {
        for row in 0..m {
            let idx = batch * m + row;
            let a = cand[idx];
            let b = refr[idx];
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-3);
            if abs > max_abs { max_abs = abs; }
            if rel > max_rel { max_rel = rel; }
            if abs > abs_tol && rel > rel_tol {
                n_bad += 1;
                hist_row_mod16[row % 16] += 1;
                if first_bad.is_none() {
                    first_bad = Some((batch, row, a, b));
                }
            }
        }
    }

    use std::fmt::Write;
    let _ = write!(
        report,
        "    {name}: max_abs={max_abs:.4e} max_rel={max_rel:.4e} bad={n_bad}/{}",
        n * m
    );
    if n_bad > 0 {
        let _ = writeln!(report);
        if let Some((b, r, a, ref_v)) = first_bad {
            let _ = writeln!(
                report,
                "      first mismatch at (batch={b}, row={r}): cand={a:.4} ref={ref_v:.4} diff={:.4e}",
                a - ref_v
            );
        }
        let _ = writeln!(report, "      mismatches by (row % 16): {hist_row_mod16:?}");
        false
    } else {
        let _ = writeln!(report);
        true
    }
}

/// Deterministic HFQ3-G256 weight bytes for testing.
/// Layout: 8B header (fp32 scale + fp32 zero, both bit-cast from fp16) +
/// 96B data (32 chunks × 3 bytes, each chunk encoding 8 × 3-bit indices
/// per the same packing as `quantize_mq3g256` in hipfire-quantize).
fn build_hfq3g256(m: usize, k: usize, seed: u8) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 104;
    let mut out = vec![0u8; m * bytes_per_row];

    let mix = |x: u64| {
        let h = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
    };
    let s0 = seed as u64;

    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 104;
            let r1 = mix(s0 ^ ((row as u64) << 16) ^ (g as u64));
            let r2 = mix(s0 ^ ((row as u64) * 7 + g as u64));
            let scale = 0.01 + (((r1 as u32) % 4001) as f32) * 1e-5;
            let zero = (((r2 as u32) % 1500) as f32) * 1e-4 - 0.075;

            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());

            for byte_i in 0..96 {
                let r = mix(s0 ^ ((row as u64) << 24) ^ ((g as u64) << 12) ^ (byte_i as u64));
                out[off + 8 + byte_i] = (r & 0xff) as u8;
            }
        }
    }
    out
}
