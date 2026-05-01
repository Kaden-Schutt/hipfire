//! Channel-test for `gemm_hfq3g256_residual_wmma` (gfx11 K1 variant).
//! Note: residual kernel does Y += W*X (fused residual add) and uses a
//! TRANSPOSED C-output convention vs qkvza/gate_up (lane covers 1 row,
//! 8 batches; lanes 0..15 → batches 0..7, lanes 16..31 → batches 8..15).

use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {}", arch);

    if !matches!(arch.as_str(), "gfx1100" | "gfx1101" | "gfx1102") {
        eprintln!("SKIP: gfx11 K1 only — current arch {arch}");
        std::process::exit(0);
    }

    let mut total_pass = 0;
    let mut total_fail = 0;

    let shapes: &[(usize, usize, usize)] = &[
        // (M, K, N)
        (16, 256, 16),
        (32, 256, 16),
        (16, 512, 16),
        (16, 256, 32),
        (32, 512, 32),
        (64, 256, 16),
    ];

    for &(m, k, n) in shapes {
        let label = format!("M={m} K={k} N={n}");
        match run_one(&mut gpu, m, k, n) {
            Ok(()) => { total_pass += 1; eprintln!("  {label:30} OK"); }
            Err(e) => { total_fail += 1; eprintln!("  {label:30} FAIL\n{e}"); }
        }
    }

    eprintln!("\nPassed: {total_pass}  Failed: {total_fail}");
    if total_fail > 0 { std::process::exit(1); }
}

fn run_one(gpu: &mut Gpu, m: usize, k: usize, n: usize) -> Result<(), String> {
    assert_eq!(m % 16, 0);
    assert_eq!(k % 256, 0);
    assert_eq!(n % 16, 0);

    let a_bytes = build_hfq3g256(m, k, 0xC3);
    let x_f32: Vec<f32> = (0..(n * k))
        .map(|i| {
            let b = (i / k) as i32;
            let kk = (i % k) as i32;
            ((b * 7 + kk * 11) % 31 - 15) as f32 * 0.05
        })
        .collect();

    // Pre-residual Y data (kernel does Y += W*X).
    let y_pre: Vec<f32> = (0..(n * m))
        .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
        .collect();
    let mut ref_y = y_pre.clone();
    let wx = cpu_gemm(&a_bytes, m, k, &x_f32, n);
    for i in 0..(n * m) { ref_y[i] += wx[i]; }

    let a = gpu.upload_raw(&a_bytes, &[m, k]).map_err(|e| format!("upload a: {e}"))?;
    let x = gpu.upload_f32(&x_f32, &[n, k]).map_err(|e| format!("upload x: {e}"))?;
    let y = gpu.upload_f32(&y_pre, &[n, m]).map_err(|e| format!("upload y_pre: {e}"))?;

    gpu.gemm_hfq3g256_residual_wmma(&a, &x, &y, m, k, n)
        .map_err(|e| format!("wmma: {e}"))?;

    let cand_y = gpu.download_f32(&y).map_err(|e| format!("download y: {e}"))?;

    let _keep_alive = (a, x, y);

    let mut report = String::new();
    let ok = compare("Y", n, m, &cand_y, &ref_y, &mut report);
    if ok { Ok(()) } else { Err(report) }
}

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
                let scale = f32::from_bits(u32::from_le_bytes([a_bytes[goff], a_bytes[goff+1], a_bytes[goff+2], a_bytes[goff+3]]));
                let zero = f32::from_bits(u32::from_le_bytes([a_bytes[goff+4], a_bytes[goff+5], a_bytes[goff+6], a_bytes[goff+7]]));
                for chunk in 0..32 {
                    let bo = goff + 8 + chunk * 3;
                    let b0 = a_bytes[bo] as u32;
                    let b1 = a_bytes[bo + 1] as u32;
                    let b2 = a_bytes[bo + 2] as u32;
                    let qs = [
                        b0 & 7, (b0 >> 3) & 7, ((b0 >> 6) | (b1 << 2)) & 7, (b1 >> 1) & 7,
                        (b1 >> 4) & 7, ((b1 >> 7) | (b2 << 1)) & 7, (b2 >> 2) & 7, (b2 >> 5) & 7,
                    ];
                    let base = g * 256 + chunk * 8;
                    for i in 0..8 { acc += (scale * (qs[i] as f32) + zero) * x_row[base + i]; }
                }
            }
            y[b * m + row] = acc;
        }
    }
    y
}

fn compare(name: &str, n: usize, m: usize, cand: &[f32], refr: &[f32], report: &mut String) -> bool {
    let mut max_abs = 0f32;
    let mut bad = 0usize;
    for batch in 0..n {
        for row in 0..m {
            let idx = batch * m + row;
            let abs = (cand[idx] - refr[idx]).abs();
            if abs > max_abs { max_abs = abs; }
            if abs > 5e-2 && abs / refr[idx].abs().max(1e-3) > 1e-2 { bad += 1; }
        }
    }
    use std::fmt::Write;
    let _ = writeln!(report, "    {name}: max_abs={max_abs:.4e}  bad={bad}/{}", n * m);
    bad == 0
}

fn build_hfq3g256(m: usize, k: usize, seed: u8) -> Vec<u8> {
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 104;
    let mut out = vec![0u8; m * bytes_per_row];
    let mix = |x: u64| {
        let h = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
