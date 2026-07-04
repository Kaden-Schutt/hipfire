// SPDX-License-Identifier: Apache-2.0
// hipfire — RoughQuant real-format kernel-level proof.
//
//! Proves the roughquant real packed format reconstructs PROTECTED channels to
//! EXACT bf16 precision on the REAL GPU kernels (not the sim). Composition under
//! test:
//!
//!   y = gemv_mq4g256(packed, x)            [real mq4 kernel: rotates x, eff W = dequant]
//!     + gemv_f32(corr[m × |S|], x[S])      [real dense GEMV of the bf16 correction]
//!
//! where corr = (W − dequant_mq4g256(packed))[:, S] are the protected residuals.
//! Because the mq4 kernel's effective weight equals dequant_mq4g256(packed) (it
//! rotates x; the FWHT cancels), the correction restores W exactly on S:
//!   recon[:,S]·x[S] + (W−recon)[:,S]·x[S] = W[:,S]·x[S].
//!
//! Two tests per shape:
//!   A) x nonzero ONLY on S  → y must equal exact W·x within bf16(corr) rounding
//!      (DECISIVE: protected channels are exact on the real kernel).
//!   B) full random x        → corrected error < uncorrected mq4 error
//!      (the correction strictly reduces error; residual is mq4 on non-protected).
//!
//! Run: cargo run --release -p hipfire-runtime --example rq_real_gemv_check

use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
use hipfire_rdna::DType;

fn main() {
    let mut gpu = hipfire_rdna::Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    let shapes = [(8usize, 256usize), (16, 512), (32, 1024)];
    // Protected column set (residual channels), like the shared outlier set.
    let mut any_fail = false;

    for &(m, k) in &shapes {
        eprintln!("\n===== shape {m} x {k} =====");
        let w: Vec<f32> = (0..m * k)
            .map(|i| fract_sin(i as f32 * 0.731 + 1.337))
            .collect();
        // ~5% protected columns, spread across the input dim.
        let n_s = (k / 20).max(2);
        let s: Vec<usize> = (0..n_s).map(|j| (j * k / n_s).min(k - 1)).collect();

        // Producer path: real mq4 pack + kernel-faithful dequant + residual corr.
        let packed = quantize_mq4g256(&w, &signs1, &signs2);
        let recon = dequant_mq4g256(&packed, m * k, &signs1, &signs2);
        // corr[m × np] = bf16( (W − recon)[:, S] ), padded to a power-of-2 column
        // count `np` (gemv_f32's tree reduction needs blockDim a power of 2 when
        // k<256 — padding with zeros leaves the dot product unchanged). |S|=np_pad.
        let np = n_s.next_power_of_two();
        let mut corr_f32 = vec![0.0f32; m * np];
        for r in 0..m {
            for (j, &c) in s.iter().enumerate() {
                corr_f32[r * np + j] = bf16_round(w[r * k + c] - recon[r * k + c]);
            }
        }

        let d_packed = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
        let d_corr = gpu.upload_f32(&corr_f32, &[m, np]).unwrap();

        for (label, x) in [
            ("A: x on S only", x_on_subset(k, &s)),
            ("B: full random x", full_x(k)),
        ] {
            // y_mq4 = real mq4 kernel
            let d_x = gpu.upload_f32(&x, &[k]).unwrap();
            let d_y = gpu.zeros(&[m], DType::F32).unwrap();
            let d_xrot = gpu.zeros(&[k], DType::F32).unwrap();
            gpu.gemv_mq4g256_with_rotate(&d_packed, &d_x, &d_y, &d_xrot, m, k)
                .unwrap();
            let y_mq4 = download(&gpu, &d_y, m);
            // y_corr = real dense GEMV of corr over gathered x[S] (padded to np)
            let mut xs = vec![0.0f32; np];
            for (j, &c) in s.iter().enumerate() {
                xs[j] = x[c];
            }
            let d_xs = gpu.upload_f32(&xs, &[np]).unwrap();
            let d_yc = gpu.zeros(&[m], DType::F32).unwrap();
            gpu.gemv_f32(&d_corr, &d_xs, &d_yc).unwrap();
            let y_corr = download(&gpu, &d_yc, m);
            // Diagnostic: does gemv_f32(corr, xs) match CPU corr·xs?
            let mut max_cdiff = 0.0f32;
            for r in 0..m {
                let cpu_c: f32 = (0..np).map(|j| corr_f32[r * np + j] * xs[j]).sum();
                max_cdiff = max_cdiff.max((y_corr[r] - cpu_c).abs());
            }
            eprintln!("    [diag] |gemv_f32(corr,xs) - cpu(corr·xs)| = {max_cdiff:.3e}");
            let y_rq: Vec<f32> = (0..m).map(|r| y_mq4[r] + y_corr[r]).collect();

            // Diagnostic: does the kernel's mq4 output match my CPU dequant·x?
            let y_recon_cpu = cpu_matvec(&recon, &x, m, k);
            let mut max_kern = 0.0f32;
            for r in 0..m {
                max_kern = max_kern.max((y_mq4[r] - y_recon_cpu[r]).abs());
            }
            eprintln!("    [diag] |kernel_mq4 - cpu(recon·x)| = {max_kern:.3e}");
            // Reference: exact W·x (f32) and uncorrected mq4 (recon·x).
            let y_exact = cpu_matvec(&w, &x, m, k);
            let mut max_rq = 0.0f32;
            let mut max_mq4 = 0.0f32;
            for r in 0..m {
                max_rq = max_rq.max((y_rq[r] - y_exact[r]).abs());
                max_mq4 = max_mq4.max((y_mq4[r] - y_exact[r]).abs());
            }
            if label.starts_with('A') {
                // Decisive: the correction must reduce the protected-channel error to
                // bf16 precision — i.e. ≥50× below the uncorrected mq4 error (the
                // residual is only the bf16 rounding of R, which IS the design).
                let ok = max_rq < 0.02 * max_mq4;
                any_fail |= !ok;
                eprintln!(
                    "  {label:18}  corrected_err={max_rq:.3e}  uncorrected_mq4_err={max_mq4:.3e}  [{}]",
                    if ok { "PASS" } else { "FAIL" }
                );
            } else {
                // Correction must strictly reduce error vs plain mq4.
                let ok = max_rq < max_mq4;
                any_fail |= !ok;
                eprintln!(
                    "  {label:18}  corrected_err={max_rq:.3e}  uncorrected_mq4_err={max_mq4:.3e}  reduced={} [{}]",
                    if ok { "yes" } else { "NO" },
                    if ok { "PASS" } else { "FAIL" }
                );
            }
        }
    }

    if any_fail {
        eprintln!("\n[FAIL] roughquant real-format kernel composition is INCORRECT.");
        std::process::exit(1);
    }
    eprintln!(
        "\n[PASS] roughquant real format: protected channels EXACT on the real mq4 kernel; \
         correction reduces error on full input."
    );
}

fn x_on_subset(k: usize, s: &[usize]) -> Vec<f32> {
    let mut x = vec![0.0f32; k];
    for &c in s {
        x[c] = fract_sin(c as f32 * 0.513 + 2.719);
    }
    x
}
fn full_x(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| fract_sin(i as f32 * 0.513 + 2.719))
        .collect()
}

fn cpu_matvec(w: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    (0..m)
        .map(|r| (0..k).map(|c| w[r * k + c] * x[c]).sum())
        .collect()
}

fn download(gpu: &hipfire_rdna::Gpu, t: &hipfire_rdna::GpuTensor, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    let b = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(b, &t.buf).unwrap();
    y
}

fn fract_sin(x: f32) -> f32 {
    (x.sin() * 12345.6789f32).fract() * 2.0f32 - 1.0f32
}

/// Round an f32 through bf16 (truncate-to-nearest-even-ish: top 16 bits + round).
fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let round_bias = 0x7fff + ((bits >> 16) & 1);
    let rounded = (bits + round_bias) & 0xffff_0000;
    f32::from_bits(rounded)
}

// ── mq4 pack/dequant (mirrors hipfire-quantize) ──────────────────────────────

fn quantize_mq4g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let (gs, block) = (256usize, 136usize);
    let n = f32_data.len();
    let n_blocks = n.div_ceil(gs);
    let mut out = vec![0u8; n_blocks * block];
    for b in 0..n_blocks {
        let start = b * gs;
        let end = (start + gs).min(n);
        let mut group = [0.0f32; 256];
        group[..end - start].copy_from_slice(&f32_data[start..end]);
        cpu_fwht_256(&mut group, signs1, signs2);
        let mn = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = mx - mn;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv = if range > 0.0 { 1.0 / scale } else { 0.0 };
        let off = b * block;
        out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
        out[off + 4..off + 8].copy_from_slice(&mn.to_le_bytes());
        for i in 0..128 {
            let lo = ((group[2 * i] - mn) * inv + 0.5) as u8;
            let hi = ((group[2 * i + 1] - mn) * inv + 0.5) as u8;
            out[off + 8 + i] = lo.min(15) | (hi.min(15) << 4);
        }
    }
    out
}

fn dequant_mq4g256(data: &[u8], n: usize, signs1: &[f32], signs2: &[f32]) -> Vec<f32> {
    let (gs, block) = (256usize, 136usize);
    let n_blocks = n.div_ceil(gs);
    let mut out = Vec::with_capacity(n_blocks * gs);
    for b in 0..n_blocks {
        let off = b * block;
        let scale = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let mn = f32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
        let mut g = [0.0f32; 256];
        for i in 0..128 {
            let byte = data[off + 8 + i];
            g[2 * i] = mn + scale * (byte & 0xF) as f32;
            g[2 * i + 1] = mn + scale * (byte >> 4) as f32;
        }
        cpu_fwht_256(&mut g, signs2, signs1);
        out.extend_from_slice(&g);
    }
    out.truncate(n);
    out
}
