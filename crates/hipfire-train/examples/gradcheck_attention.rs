//! Finite-difference gradient check for single-head causal SDPA (Phase 0, M2).
//!
//! Loss L = Σ CTX∘G ⇒ d_ctx = G. Checks analytic dQ, dK, dV vs central
//! differences. Causal masking is exercised (seq>1).
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "gradcheck-attn"
//!   cargo run -p hipfire-train --release --example gradcheck_attention
//!   hipfire gpu-lock release

use hipfire_train::ops::attention::{sdpa_backward, sdpa_forward};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

const SEQ: usize = 5;
const D: usize = 8;

fn scale() -> f32 {
    1.0 / (D as f32).sqrt()
}

fn loss(gpu: &mut Gpu, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, g: &[f32]) -> HipResult<f32> {
    let scores = gpu.zeros(&[SEQ * SEQ], DType::F32)?;
    let p = gpu.zeros(&[SEQ * SEQ], DType::F32)?;
    let ctx = gpu.zeros(&[SEQ * D], DType::F32)?;
    sdpa_forward(gpu, q, k, v, &scores, &p, &ctx, SEQ, D, scale())?;
    let cv = gpu.download_f32(&ctx)?;
    Ok(cv.iter().zip(g).map(|(a, b)| a * b).sum())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let qh: Vec<f32> = (0..SEQ * D)
        .map(|i| ((i * 17 % 13) as f32) * 0.15 - 0.8)
        .collect();
    let kh: Vec<f32> = (0..SEQ * D)
        .map(|i| ((i * 23 % 11) as f32) * 0.12 - 0.5)
        .collect();
    let vh: Vec<f32> = (0..SEQ * D)
        .map(|i| ((i * 7 % 9) as f32) * 0.2 - 0.7)
        .collect();
    let gh: Vec<f32> = (0..SEQ * D)
        .map(|i| ((i * 13 % 5) as f32) * 0.25 - 0.4)
        .collect();

    let q = gpu.upload_f32(&qh, &[SEQ * D])?;
    let k = gpu.upload_f32(&kh, &[SEQ * D])?;
    let v = gpu.upload_f32(&vh, &[SEQ * D])?;

    // Analytic
    let scores = gpu.zeros(&[SEQ * SEQ], DType::F32)?;
    let p = gpu.zeros(&[SEQ * SEQ], DType::F32)?;
    let ctx = gpu.zeros(&[SEQ * D], DType::F32)?;
    sdpa_forward(&mut gpu, &q, &k, &v, &scores, &p, &ctx, SEQ, D, scale())?;
    let d_ctx = gpu.upload_f32(&gh, &[SEQ * D])?;
    let dp = gpu.zeros(&[SEQ * SEQ], DType::F32)?;
    let dsc = gpu.zeros(&[SEQ * SEQ], DType::F32)?;
    let dq = gpu.zeros(&[SEQ * D], DType::F32)?;
    let dk = gpu.zeros(&[SEQ * D], DType::F32)?;
    let dv = gpu.zeros(&[SEQ * D], DType::F32)?;
    sdpa_backward(
        &mut gpu,
        &d_ctx,
        &q,
        &k,
        &v,
        &p,
        &dp,
        &dsc,
        &dq,
        &dk,
        &dv,
        SEQ,
        D,
        scale(),
    )?;
    let dq_a = gpu.download_f32(&dq)?;
    let dk_a = gpu.download_f32(&dk)?;
    let dv_a = gpu.download_f32(&dv)?;

    let eps = 1e-3f32;
    let check = |gpu: &mut Gpu,
                 host: &[f32],
                 which: u8,
                 ana: &[f32],
                 q: &GpuTensor,
                 k: &GpuTensor,
                 v: &GpuTensor|
     -> HipResult<f32> {
        let mut e = 0.0f32;
        for i in 0..host.len() {
            let mut hp = host.to_vec();
            hp[i] += eps;
            let mut hm = host.to_vec();
            hm[i] -= eps;
            let pd = gpu.upload_f32(&hp, &[host.len()])?;
            let md = gpu.upload_f32(&hm, &[host.len()])?;
            let (lp, lm) = match which {
                0 => (loss(gpu, &pd, k, v, &gh)?, loss(gpu, &md, k, v, &gh)?),
                1 => (loss(gpu, q, &pd, v, &gh)?, loss(gpu, q, &md, v, &gh)?),
                _ => (loss(gpu, q, k, &pd, &gh)?, loss(gpu, q, k, &md, &gh)?),
            };
            e = e.max(((lp - lm) / (2.0 * eps) - ana[i]).abs());
        }
        Ok(e)
    };

    let eq = check(&mut gpu, &qh, 0, &dq_a, &q, &k, &v)?;
    let ek = check(&mut gpu, &kh, 1, &dk_a, &q, &k, &v)?;
    let ev = check(&mut gpu, &vh, 2, &dv_a, &q, &k, &v)?;

    println!("attn dQ max|analytic-numeric| = {eq:.2e}");
    println!("attn dK max|analytic-numeric| = {ek:.2e}");
    println!("attn dV max|analytic-numeric| = {ev:.2e}");
    let tol = 1e-2f32;
    if eq < tol && ek < tol && ev < tol {
        println!("\nGRADCHECK PASS — single-head causal SDPA backward matches finite differences.");
        Ok(())
    } else {
        Err(format!("gradcheck FAIL: dQ {eq:.2e}, dK {ek:.2e}, dV {ev:.2e}").into())
    }
}
