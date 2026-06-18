//! Finite-difference gradient check for the KL distillation loss (Phase 2 Q1).
//!
//! Sum-reduction L = Σ_rows KL(teacher_p ‖ softmax(student)); analytic
//! d_logits = softmax(student) − teacher_p must match central differences over
//! the student logits. teacher_p is a fixed random distribution (softmax of
//! random logits).
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   source ./scripts/gpu-lock.sh && gpu_acquire "gradcheck-distill"
//!   cargo run -p hipfire-train --release --example gradcheck_distill
//!   gpu_release

use hipfire_train::ops::distill::distill_kl;
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

const ROWS: usize = 3;
const V: usize = 9;

fn softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().cloned().fold(f32::MIN, f32::max);
    let ex: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = ex.iter().sum();
    ex.iter().map(|x| x / s).collect()
}

fn total_loss(gpu: &mut Gpu, student: &GpuTensor, tp: &GpuTensor) -> HipResult<f32> {
    let loss = gpu.zeros(&[ROWS], DType::F32)?;
    let dl = gpu.zeros(&[ROWS * V], DType::F32)?;
    distill_kl(gpu, student, tp, &loss, &dl, ROWS, V)?;
    Ok(gpu.download_f32(&loss)?.iter().sum())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let s_host: Vec<f32> = (0..ROWS * V)
        .map(|i| ((i * 23 % 17) as f32) * 0.2 - 1.0)
        .collect();
    // teacher_p: row-wise softmax of a different deterministic logit field.
    let mut tp_host = vec![0.0f32; ROWS * V];
    for r in 0..ROWS {
        let tl: Vec<f32> = (0..V)
            .map(|i| (((r * V + i) * 13 % 11) as f32) * 0.3 - 0.5)
            .collect();
        tp_host[r * V..(r + 1) * V].copy_from_slice(&softmax(&tl));
    }

    let student = gpu.upload_f32(&s_host, &[ROWS * V])?;
    let tp = gpu.upload_f32(&tp_host, &[ROWS * V])?;
    let loss = gpu.zeros(&[ROWS], DType::F32)?;
    let dl = gpu.zeros(&[ROWS * V], DType::F32)?;
    distill_kl(&mut gpu, &student, &tp, &loss, &dl, ROWS, V)?;
    let dl_analytic = gpu.download_f32(&dl)?;

    let eps = 1e-3f32;
    let mut max_err = 0.0f32;
    for i in 0..ROWS * V {
        let mut sp = s_host.clone();
        sp[i] += eps;
        let spd = gpu.upload_f32(&sp, &[ROWS * V])?;
        let hp = total_loss(&mut gpu, &spd, &tp)?;
        let mut sm = s_host.clone();
        sm[i] -= eps;
        let smd = gpu.upload_f32(&sm, &[ROWS * V])?;
        let hm = total_loss(&mut gpu, &smd, &tp)?;
        max_err = max_err.max(((hp - hm) / (2.0 * eps) - dl_analytic[i]).abs());
    }

    println!("distill d_logits max|analytic-numeric| = {max_err:.2e}");
    let tol = 1e-2f32;
    if max_err < tol {
        println!("\nGRADCHECK PASS — KL distillation backward matches finite differences.");
        Ok(())
    } else {
        Err(format!("gradcheck FAIL (tol {tol:.0e}): {max_err:.2e}").into())
    }
}
