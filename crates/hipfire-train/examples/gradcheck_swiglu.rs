//! Finite-difference gradient check for the `swiglu` op (Phase 0, M1).
//!
//! Loss L = Σ OUT∘G ⇒ dL/dOUT = G. Checks analytic d_gate and d_up against
//! central differences over gate and up.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "gradcheck-swiglu"
//!   cargo run -p hipfire-train --release --example gradcheck_swiglu
//!   hipfire gpu-lock release

use hipfire_train::ops::swiglu::{swiglu_backward, swiglu_forward};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

const N: usize = 16;

fn loss(gpu: &mut Gpu, gate: &GpuTensor, up: &GpuTensor, g: &[f32]) -> HipResult<f32> {
    let out = gpu.zeros(&[N], DType::F32)?;
    swiglu_forward(gpu, gate, up, &out, N)?;
    let ov = gpu.download_f32(&out)?;
    Ok(ov.iter().zip(g).map(|(a, b)| a * b).sum())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let gate_host: Vec<f32> = (0..N).map(|i| ((i * 17 % 11) as f32) * 0.3 - 1.5).collect();
    let up_host: Vec<f32> = (0..N).map(|i| ((i * 13 % 7) as f32) * 0.25 - 0.8).collect();
    let g_host: Vec<f32> = (0..N).map(|i| ((i * 7 % 5) as f32) * 0.4 - 0.6).collect();

    let gate = gpu.upload_f32(&gate_host, &[N])?;
    let up = gpu.upload_f32(&up_host, &[N])?;
    let d_out = gpu.upload_f32(&g_host, &[N])?;
    let d_gate = gpu.zeros(&[N], DType::F32)?;
    let d_up = gpu.zeros(&[N], DType::F32)?;
    swiglu_backward(&mut gpu, &d_out, &gate, &up, &d_gate, &d_up, N)?;
    let dg_analytic = gpu.download_f32(&d_gate)?;
    let du_analytic = gpu.download_f32(&d_up)?;

    let eps = 1e-3f32;
    let mut max_err_g = 0.0f32;
    let mut max_err_u = 0.0f32;
    for i in 0..N {
        // gate
        let mut gp = gate_host.clone();
        gp[i] += eps;
        let gpd = gpu.upload_f32(&gp, &[N])?;
        let lp = loss(&mut gpu, &gpd, &up, &g_host)?;
        let mut gm = gate_host.clone();
        gm[i] -= eps;
        let gmd = gpu.upload_f32(&gm, &[N])?;
        let lm = loss(&mut gpu, &gmd, &up, &g_host)?;
        max_err_g = max_err_g.max(((lp - lm) / (2.0 * eps) - dg_analytic[i]).abs());
        // up
        let mut upp = up_host.clone();
        upp[i] += eps;
        let uppd = gpu.upload_f32(&upp, &[N])?;
        let lp = loss(&mut gpu, &gate, &uppd, &g_host)?;
        let mut upm = up_host.clone();
        upm[i] -= eps;
        let upmd = gpu.upload_f32(&upm, &[N])?;
        let lm = loss(&mut gpu, &gate, &upmd, &g_host)?;
        max_err_u = max_err_u.max(((lp - lm) / (2.0 * eps) - du_analytic[i]).abs());
    }

    println!("swiglu d_gate max|analytic-numeric| = {max_err_g:.2e}");
    println!("swiglu d_up   max|analytic-numeric| = {max_err_u:.2e}");
    let tol = 1e-2f32;
    if max_err_g < tol && max_err_u < tol {
        println!("\nGRADCHECK PASS — swiglu backward matches finite differences.");
        Ok(())
    } else {
        Err(
            format!("gradcheck FAIL (tol {tol:.0e}): d_gate {max_err_g:.2e}, d_up {max_err_u:.2e}")
                .into(),
        )
    }
}
