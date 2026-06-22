//! Finite-difference gradient check for the `rope` op (Phase 0, M1).
//!
//! Loss L = Σ OUT∘G ⇒ d_out = G; analytic dx (rotation by −angle) vs central
//! differences. Also asserts RoPE is norm-preserving (rotation) per position.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "gradcheck-rope"
//!   cargo run -p hipfire-train --release --example gradcheck_rope
//!   hipfire gpu-lock release

use hipfire_train::ops::rope::{rope_backward, rope_forward};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

const SEQ: usize = 3;
const NH: usize = 2;
const D: usize = 8;
const ROWS: usize = SEQ * NH;
const BASE: f32 = 10000.0;

fn loss(gpu: &mut Gpu, x: &GpuTensor, pos: &GpuTensor, g: &[f32]) -> HipResult<f32> {
    let out = gpu.zeros(&[ROWS * D], DType::F32)?;
    rope_forward(gpu, x, &out, pos, ROWS, NH, D, BASE)?;
    let ov = gpu.download_f32(&out)?;
    Ok(ov.iter().zip(g).map(|(a, b)| a * b).sum())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let x_host: Vec<f32> = (0..ROWS * D)
        .map(|i| ((i * 19 % 13) as f32) * 0.2 - 1.1)
        .collect();
    let g_host: Vec<f32> = (0..ROWS * D)
        .map(|i| ((i * 7 % 5) as f32) * 0.3 - 0.5)
        .collect();
    let pos_host: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();

    let x = gpu.upload_f32(&x_host, &[ROWS * D])?;
    let pos = gpu.upload_f32(&pos_host, &[SEQ])?;

    // Norm-preservation sanity (each row's L2 norm unchanged by rotation).
    let out = gpu.zeros(&[ROWS * D], DType::F32)?;
    rope_forward(&mut gpu, &x, &out, &pos, ROWS, NH, D, BASE)?;
    let ov = gpu.download_f32(&out)?;
    let mut max_norm_err = 0.0f32;
    for r in 0..ROWS {
        let n_in: f32 = x_host[r * D..(r + 1) * D]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        let n_out: f32 = ov[r * D..(r + 1) * D]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        max_norm_err = max_norm_err.max((n_in - n_out).abs());
    }

    let dy = gpu.upload_f32(&g_host, &[ROWS * D])?;
    let dx = gpu.zeros(&[ROWS * D], DType::F32)?;
    rope_backward(&mut gpu, &dy, &dx, &pos, ROWS, NH, D, BASE)?;
    let dx_analytic = gpu.download_f32(&dx)?;

    let eps = 1e-3f32;
    let mut max_err = 0.0f32;
    for i in 0..ROWS * D {
        let mut xp = x_host.clone();
        xp[i] += eps;
        let xpd = gpu.upload_f32(&xp, &[ROWS * D])?;
        let lp = loss(&mut gpu, &xpd, &pos, &g_host)?;
        let mut xm = x_host.clone();
        xm[i] -= eps;
        let xmd = gpu.upload_f32(&xm, &[ROWS * D])?;
        let lm = loss(&mut gpu, &xmd, &pos, &g_host)?;
        max_err = max_err.max(((lp - lm) / (2.0 * eps) - dx_analytic[i]).abs());
    }

    println!("rope norm-preservation max err = {max_norm_err:.2e}");
    println!("rope dX max|analytic-numeric|  = {max_err:.2e}");
    let tol = 1e-2f32;
    if max_err < tol && max_norm_err < 1e-4 {
        println!(
            "\nGRADCHECK PASS — rope backward matches finite differences (and is a rotation)."
        );
        Ok(())
    } else {
        Err(format!("gradcheck FAIL: dX {max_err:.2e}, norm {max_norm_err:.2e}").into())
    }
}
