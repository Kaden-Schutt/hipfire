//! Finite-difference gradient check for the `lora` op (Phase 0, toward M2).
//!
//! LoRA-adapted linear y = x·Wᵀ + scale·(x·Aᵀ)·Bᵀ, base W frozen. Loss
//! L = Σ Y∘G ⇒ dL/dY = G. Checks analytic dA, dB, dX against central
//! differences (W is frozen, so no dW check).
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "gradcheck-lora"
//!   cargo run -p hipfire-train --release --example gradcheck_lora
//!   hipfire gpu-lock release

use hipfire_train::ops::lora::{lora_backward, lora_forward};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

const M: usize = 4; // tokens
const K: usize = 6; // in
const N: usize = 5; // out
const R: usize = 2; // rank
const SCALE: f32 = 1.5; // alpha/r

#[allow(clippy::too_many_arguments)]
fn loss(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &GpuTensor,
    a: &GpuTensor,
    b: &GpuTensor,
    g: &[f32],
) -> HipResult<f32> {
    let h = gpu.zeros(&[M * R], DType::F32)?;
    let lora = gpu.zeros(&[M * N], DType::F32)?;
    let y = gpu.zeros(&[M * N], DType::F32)?;
    lora_forward(gpu, x, w, a, b, &h, &lora, &y, M, K, N, R, SCALE)?;
    let yv = gpu.download_f32(&y)?;
    Ok(yv.iter().zip(g).map(|(p, q)| p * q).sum())
}

fn maxerr(num: f32, ana: f32, acc: &mut f32) {
    *acc = acc.max((num - ana).abs());
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let xh: Vec<f32> = (0..M * K)
        .map(|i| ((i * 31 % 17) as f32) * 0.07 - 0.4)
        .collect();
    let wh: Vec<f32> = (0..N * K)
        .map(|i| ((i * 23 % 13) as f32) * 0.05 - 0.25)
        .collect();
    let ah: Vec<f32> = (0..R * K)
        .map(|i| ((i * 7 % 5) as f32) * 0.1 - 0.2)
        .collect();
    let bh: Vec<f32> = (0..N * R)
        .map(|i| ((i * 13 % 11) as f32) * 0.08 - 0.3)
        .collect();
    let gh: Vec<f32> = (0..M * N)
        .map(|i| ((i * 11 % 7) as f32) * 0.1 - 0.2)
        .collect();

    let x = gpu.upload_f32(&xh, &[M * K])?;
    let w = gpu.upload_f32(&wh, &[N * K])?;
    let a = gpu.upload_f32(&ah, &[R * K])?;
    let b = gpu.upload_f32(&bh, &[N * R])?;

    // Analytic
    let h = gpu.zeros(&[M * R], DType::F32)?;
    let lora = gpu.zeros(&[M * N], DType::F32)?;
    let y = gpu.zeros(&[M * N], DType::F32)?;
    lora_forward(&mut gpu, &x, &w, &a, &b, &h, &lora, &y, M, K, N, R, SCALE)?;
    let dy = gpu.upload_f32(&gh, &[M * N])?;
    let dyl = gpu.zeros(&[M * N], DType::F32)?;
    let dh = gpu.zeros(&[M * R], DType::F32)?;
    let da = gpu.zeros(&[R * K], DType::F32)?;
    let db = gpu.zeros(&[N * R], DType::F32)?;
    let dx = gpu.zeros(&[M * K], DType::F32)?;
    lora_backward(
        &mut gpu, &dy, &x, &w, &a, &b, &h, &dyl, &dh, &da, &db, &dx, M, K, N, R, SCALE, false,
    )?;
    let da_a = gpu.download_f32(&da)?;
    let db_a = gpu.download_f32(&db)?;
    let dx_a = gpu.download_f32(&dx)?;

    let eps = 1e-3f32;
    let (mut ea, mut eb, mut ex) = (0.0f32, 0.0f32, 0.0f32);

    for i in 0..R * K {
        let mut p = ah.clone();
        p[i] += eps;
        let pd = gpu.upload_f32(&p, &[R * K])?;
        let lp = loss(&mut gpu, &x, &w, &pd, &b, &gh)?;
        let mut m = ah.clone();
        m[i] -= eps;
        let md = gpu.upload_f32(&m, &[R * K])?;
        let lm = loss(&mut gpu, &x, &w, &md, &b, &gh)?;
        maxerr((lp - lm) / (2.0 * eps), da_a[i], &mut ea);
    }
    for i in 0..N * R {
        let mut p = bh.clone();
        p[i] += eps;
        let pd = gpu.upload_f32(&p, &[N * R])?;
        let lp = loss(&mut gpu, &x, &w, &a, &pd, &gh)?;
        let mut m = bh.clone();
        m[i] -= eps;
        let md = gpu.upload_f32(&m, &[N * R])?;
        let lm = loss(&mut gpu, &x, &w, &a, &md, &gh)?;
        maxerr((lp - lm) / (2.0 * eps), db_a[i], &mut eb);
    }
    for i in 0..M * K {
        let mut p = xh.clone();
        p[i] += eps;
        let pd = gpu.upload_f32(&p, &[M * K])?;
        let lp = loss(&mut gpu, &pd, &w, &a, &b, &gh)?;
        let mut m = xh.clone();
        m[i] -= eps;
        let md = gpu.upload_f32(&m, &[M * K])?;
        let lm = loss(&mut gpu, &md, &w, &a, &b, &gh)?;
        maxerr((lp - lm) / (2.0 * eps), dx_a[i], &mut ex);
    }

    println!("lora dA max|analytic-numeric| = {ea:.2e}");
    println!("lora dB max|analytic-numeric| = {eb:.2e}");
    println!("lora dX max|analytic-numeric| = {ex:.2e}");
    let tol = 1e-2f32;
    if ea < tol && eb < tol && ex < tol {
        println!("\nGRADCHECK PASS — lora backward matches finite differences.");
        Ok(())
    } else {
        Err(format!("gradcheck FAIL: dA {ea:.2e}, dB {eb:.2e}, dX {ex:.2e}").into())
    }
}
