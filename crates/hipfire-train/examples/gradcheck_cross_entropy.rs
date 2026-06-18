//! Finite-difference gradient check for the `cross_entropy` op (Phase 0, M1).
//!
//! Sum-reduction loss L = Σ_rows loss_row; analytic d_logits = softmax − onehot
//! must match central differences. One row uses target = ignore_index to verify
//! masking (its loss and grad must be exactly zero).
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   source ./scripts/gpu-lock.sh && gpu_acquire "gradcheck-ce"
//!   cargo run -p hipfire-train --release --example gradcheck_cross_entropy
//!   gpu_release

use hipfire_train::ops::cross_entropy::cross_entropy;
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

const ROWS: usize = 4;
const V: usize = 9;
const IGNORE: i32 = -100;

fn total_loss(gpu: &mut Gpu, logits: &GpuTensor, targets: &GpuTensor) -> HipResult<f32> {
    let loss = gpu.zeros(&[ROWS], DType::F32)?;
    let dl = gpu.zeros(&[ROWS * V], DType::F32)?;
    cross_entropy(gpu, logits, targets, &loss, &dl, ROWS, V, IGNORE)?;
    Ok(gpu.download_f32(&loss)?.iter().sum())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let logits_host: Vec<f32> = (0..ROWS * V)
        .map(|i| ((i * 23 % 17) as f32) * 0.25 - 2.0)
        .collect();
    // Row 2 is the ignored row.
    let targets_host: Vec<f32> = vec![3.0, 7.0, IGNORE as f32, 1.0];

    let logits = gpu.upload_f32(&logits_host, &[ROWS * V])?;
    let targets = gpu.upload_f32(&targets_host, &[ROWS])?;

    let loss = gpu.zeros(&[ROWS], DType::F32)?;
    let dl = gpu.zeros(&[ROWS * V], DType::F32)?;
    cross_entropy(&mut gpu, &logits, &targets, &loss, &dl, ROWS, V, IGNORE)?;
    let loss_host = gpu.download_f32(&loss)?;
    let dl_analytic = gpu.download_f32(&dl)?;

    // Masking: ignored row must be exactly zero in loss and grad.
    if loss_host[2] != 0.0 {
        return Err(format!("ignored row loss = {} != 0", loss_host[2]).into());
    }
    if dl_analytic[2 * V..3 * V].iter().any(|x| *x != 0.0) {
        return Err("ignored row gradient not all zero".into());
    }

    let eps = 1e-3f32;
    let mut max_err = 0.0f32;
    for i in 0..ROWS * V {
        let mut lp = logits_host.clone();
        lp[i] += eps;
        let lpd = gpu.upload_f32(&lp, &[ROWS * V])?;
        let hp = total_loss(&mut gpu, &lpd, &targets)?;
        let mut lm = logits_host.clone();
        lm[i] -= eps;
        let lmd = gpu.upload_f32(&lm, &[ROWS * V])?;
        let hm = total_loss(&mut gpu, &lmd, &targets)?;
        max_err = max_err.max(((hp - hm) / (2.0 * eps) - dl_analytic[i]).abs());
    }

    println!("cross_entropy d_logits max|analytic-numeric| = {max_err:.2e}");
    println!("(ignored-row masking verified: loss[2]=0, grad row 2 all zero)");
    let tol = 1e-2f32;
    if max_err < tol {
        println!("\nGRADCHECK PASS — cross_entropy backward matches finite differences.");
        Ok(())
    } else {
        Err(format!("gradcheck FAIL (tol {tol:.0e}): {max_err:.2e}").into())
    }
}
