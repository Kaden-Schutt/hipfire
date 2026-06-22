//! Finite-difference gradient check for the `softmax` op (Phase 0, M1).
//!
//! Loss L = Σ P∘G (fixed random G ⇒ dL/dP = G). Checks analytic dS against
//! central differences over the logits s.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "gradcheck-softmax"
//!   cargo run -p hipfire-train --release --example gradcheck_softmax
//!   hipfire gpu-lock release

use hipfire_train::ops::softmax::{softmax_backward, softmax_forward};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

const ROWS: usize = 3;
const N: usize = 7;

fn loss(gpu: &mut Gpu, s: &GpuTensor, g: &[f32]) -> HipResult<f32> {
    let y = gpu.zeros(&[ROWS * N], DType::F32)?;
    softmax_forward(gpu, s, &y, ROWS, N)?;
    let yv = gpu.download_f32(&y)?;
    Ok(yv.iter().zip(g).map(|(a, b)| a * b).sum())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let s_host: Vec<f32> = (0..ROWS * N)
        .map(|i| ((i * 19 % 13) as f32) * 0.3 - 1.5)
        .collect();
    let g_host: Vec<f32> = (0..ROWS * N)
        .map(|i| ((i * 11 % 5) as f32) * 0.4 - 0.7)
        .collect();

    let s = gpu.upload_f32(&s_host, &[ROWS * N])?;
    let p = gpu.zeros(&[ROWS * N], DType::F32)?;
    softmax_forward(&mut gpu, &s, &p, ROWS, N)?;
    let dy = gpu.upload_f32(&g_host, &[ROWS * N])?;
    let ds = gpu.zeros(&[ROWS * N], DType::F32)?;
    softmax_backward(&mut gpu, &dy, &p, &ds, ROWS, N)?;
    let ds_analytic = gpu.download_f32(&ds)?;

    let eps = 1e-3f32;
    let mut max_err = 0.0f32;
    for i in 0..ROWS * N {
        let mut sp = s_host.clone();
        sp[i] += eps;
        let spd = gpu.upload_f32(&sp, &[ROWS * N])?;
        let lp = loss(&mut gpu, &spd, &g_host)?;
        let mut sm = s_host.clone();
        sm[i] -= eps;
        let smd = gpu.upload_f32(&sm, &[ROWS * N])?;
        let lm = loss(&mut gpu, &smd, &g_host)?;
        max_err = max_err.max(((lp - lm) / (2.0 * eps) - ds_analytic[i]).abs());
    }

    println!("softmax dS  max|analytic-numeric| = {max_err:.2e}");
    let tol = 1e-2f32;
    if max_err < tol {
        println!("\nGRADCHECK PASS — softmax backward matches finite differences.");
        Ok(())
    } else {
        Err(format!("gradcheck FAIL (tol {tol:.0e}): dS {max_err:.2e}").into())
    }
}
