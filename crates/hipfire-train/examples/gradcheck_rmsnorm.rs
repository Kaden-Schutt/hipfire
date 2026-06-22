//! Finite-difference gradient check for the `rmsnorm` op (Phase 0, M1).
//!
//! Scalar loss L = Σ Y∘G (fixed random G ⇒ dL/dY = G). Checks analytic dX and
//! dW against central differences over both x and w.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "gradcheck-rmsnorm"
//!   cargo run -p hipfire-train --release --example gradcheck_rmsnorm
//!   hipfire gpu-lock release

use hipfire_train::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

const ROWS: usize = 3;
const H: usize = 8;
const EPS: f32 = 1e-6;

fn loss(gpu: &mut Gpu, x: &GpuTensor, w: &GpuTensor, g: &[f32]) -> HipResult<f32> {
    let y = gpu.zeros(&[ROWS * H], DType::F32)?;
    let rinv = gpu.zeros(&[ROWS], DType::F32)?;
    rmsnorm_forward(gpu, x, w, &y, &rinv, ROWS, H, EPS)?;
    let yv = gpu.download_f32(&y)?;
    Ok(yv.iter().zip(g).map(|(a, b)| a * b).sum())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let x_host: Vec<f32> = (0..ROWS * H)
        .map(|i| ((i * 17 % 11) as f32) * 0.2 - 0.9)
        .collect();
    let w_host: Vec<f32> = (0..H).map(|i| 0.5 + ((i * 7 % 5) as f32) * 0.1).collect();
    let g_host: Vec<f32> = (0..ROWS * H)
        .map(|i| ((i * 13 % 7) as f32) * 0.1 - 0.3)
        .collect();

    let x = gpu.upload_f32(&x_host, &[ROWS * H])?;
    let w = gpu.upload_f32(&w_host, &[H])?;

    // Analytic: forward to get rinv, then backward with dy=G.
    let y = gpu.zeros(&[ROWS * H], DType::F32)?;
    let rinv = gpu.zeros(&[ROWS], DType::F32)?;
    rmsnorm_forward(&mut gpu, &x, &w, &y, &rinv, ROWS, H, EPS)?;
    let dy = gpu.upload_f32(&g_host, &[ROWS * H])?;
    let dx = gpu.zeros(&[ROWS * H], DType::F32)?;
    let dw = gpu.zeros(&[H], DType::F32)?; // atomic-accumulated; starts zero
    rmsnorm_backward(&mut gpu, &dy, &x, &w, &rinv, &dx, &dw, ROWS, H)?;
    let dx_analytic = gpu.download_f32(&dx)?;
    let dw_analytic = gpu.download_f32(&dw)?;

    let eps = 1e-3f32;
    let mut max_err_x = 0.0f32;
    for i in 0..ROWS * H {
        let mut xp = x_host.clone();
        xp[i] += eps;
        let xpd = gpu.upload_f32(&xp, &[ROWS * H])?;
        let lp = loss(&mut gpu, &xpd, &w, &g_host)?;
        let mut xm = x_host.clone();
        xm[i] -= eps;
        let xmd = gpu.upload_f32(&xm, &[ROWS * H])?;
        let lm = loss(&mut gpu, &xmd, &w, &g_host)?;
        max_err_x = max_err_x.max(((lp - lm) / (2.0 * eps) - dx_analytic[i]).abs());
    }

    let mut max_err_w = 0.0f32;
    for i in 0..H {
        let mut wp = w_host.clone();
        wp[i] += eps;
        let wpd = gpu.upload_f32(&wp, &[H])?;
        let lp = loss(&mut gpu, &x, &wpd, &g_host)?;
        let mut wm = w_host.clone();
        wm[i] -= eps;
        let wmd = gpu.upload_f32(&wm, &[H])?;
        let lm = loss(&mut gpu, &x, &wmd, &g_host)?;
        max_err_w = max_err_w.max(((lp - lm) / (2.0 * eps) - dw_analytic[i]).abs());
    }

    println!("rmsnorm dX  max|analytic-numeric| = {max_err_x:.2e}");
    println!("rmsnorm dW  max|analytic-numeric| = {max_err_w:.2e}");

    let tol = 5e-2f32; // rmsnorm is nonlinear ⇒ central-diff truncation larger
    if max_err_x < tol && max_err_w < tol {
        println!("\nGRADCHECK PASS — rmsnorm backward matches finite differences.");
        Ok(())
    } else {
        Err(
            format!("gradcheck FAIL (tol {tol:.0e}): dX {max_err_x:.2e}, dW {max_err_w:.2e}")
                .into(),
        )
    }
}
