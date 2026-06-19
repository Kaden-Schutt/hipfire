//! Finite-difference gradcheck for the gated linear-recurrence scan (fwd+bwd).
//! `h[t]=g[t]*h[t-1]+(1-g[t])*u[t]`. Checks dL/dg and dL/du against FD.
//! Never train on an unverified gradient.
//!
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   source ./scripts/gpu-lock.sh && gpu_acquire "gated-scan-gradcheck"
//!   cargo run -p hipfire-train --release --example gradcheck_gated_scan

use hipfire_train::ops::gated_scan::{gated_scan_backward, gated_scan_forward};
use rdna_compute::Gpu;

const SEQ: usize = 12;
const D: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    let n = SEQ * D;
    println!("gradcheck gated_scan: seq={SEQ} D={D} (n={n})");

    // deterministic pseudo-random g∈(0,1), u∈[-1,1), loss weights w∈[-1,1)
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut rng = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (1u64 << 31) as f32 // ~[0,1)
    };
    let mut g: Vec<f32> = (0..n).map(|_| 0.05 + 0.9 * rng()).collect(); // keep off the edges
    let mut u: Vec<f32> = (0..n).map(|_| 2.0 * rng() - 1.0).collect();
    let w: Vec<f32> = (0..n).map(|_| 2.0 * rng() - 1.0).collect();

    let d_hout = gpu.upload_f32(&w, &[n])?;

    // L(g,u) = Σ_i w_i * h_out_i
    let loss = |gpu: &mut Gpu, g: &[f32], u: &[f32]| -> f32 {
        let gd = gpu.upload_f32(g, &[n]).unwrap();
        let ud = gpu.upload_f32(u, &[n]).unwrap();
        let h = gated_scan_forward(gpu, &gd, &ud, SEQ, D).unwrap();
        let hh = gpu.download_f32(&h).unwrap();
        let _ = gpu.free_tensor(h);
        let _ = gpu.free_tensor(gd);
        let _ = gpu.free_tensor(ud);
        hh.iter().zip(&w).map(|(a, b)| a * b).sum()
    };

    // analytic d_g, d_u
    let gd = gpu.upload_f32(&g, &[n])?;
    let ud = gpu.upload_f32(&u, &[n])?;
    let h = gated_scan_forward(&mut gpu, &gd, &ud, SEQ, D)?;
    let (dg, du) = gated_scan_backward(&mut gpu, &gd, &ud, &h, &d_hout, SEQ, D)?;
    let dg_host = gpu.download_f32(&dg)?;
    let du_host = gpu.download_f32(&du)?;

    let h_step = 1e-3f32;
    let (atol, rtol) = (1e-3f32, 2e-2f32);
    // spread across early/mid/late timesteps and channels
    let idxs = [0usize, 3, D, D + 5, 5 * D + 2, 7 * D, (SEQ - 1) * D + 1, (SEQ - 1) * D + 7];
    let mut max_abs = 0.0f32;
    let mut all_ok = true;

    println!("\n-- dL/dg --\n  idx      analytic         fd        abs_err   tol      ok");
    for &i in &idxs {
        let orig = g[i];
        g[i] = orig + h_step;
        let lp = loss(&mut gpu, &g, &u);
        g[i] = orig - h_step;
        let lm = loss(&mut gpu, &g, &u);
        g[i] = orig;
        let fd = (lp - lm) / (2.0 * h_step);
        let a = dg_host[i];
        let abs = (a - fd).abs();
        let tol = atol + rtol * fd.abs();
        let ok = abs <= tol;
        all_ok &= ok;
        max_abs = max_abs.max(abs);
        println!("  {i:>5} {a:>14.6} {fd:>12.6} {abs:>10.2e} {tol:>8.2e}   {}", if ok { "✓" } else { "✗" });
    }

    println!("\n-- dL/du --\n  idx      analytic         fd        abs_err   tol      ok");
    for &i in &idxs {
        let orig = u[i];
        u[i] = orig + h_step;
        let lp = loss(&mut gpu, &g, &u);
        u[i] = orig - h_step;
        let lm = loss(&mut gpu, &g, &u);
        u[i] = orig;
        let fd = (lp - lm) / (2.0 * h_step);
        let a = du_host[i];
        let abs = (a - fd).abs();
        let tol = atol + rtol * fd.abs();
        let ok = abs <= tol;
        all_ok &= ok;
        max_abs = max_abs.max(abs);
        println!("  {i:>5} {a:>14.6} {fd:>12.6} {abs:>10.2e} {tol:>8.2e}   {}", if ok { "✓" } else { "✗" });
    }

    println!("\n  max_abs_err={max_abs:.2e}  (atol={atol:.0e} rtol={rtol:.0e})");
    if all_ok {
        println!("  PASS ✓ (analytic gradient matches finite differences)");
        Ok(())
    } else {
        Err("gradcheck FAILED: some index exceeded atol+rtol·|fd|".into())
    }
}
