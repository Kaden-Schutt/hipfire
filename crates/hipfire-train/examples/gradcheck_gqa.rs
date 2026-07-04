//! Finite-difference gradient check for GQA multi-head attention (Phase 0, M2).
//!
//! n_heads=4, n_kv=2 (group=2) exercises kv-head broadcast and dK/dV grad
//! accumulation. Loss L = Σ CTX∘G ⇒ d_ctx = G; checks dQ, dK, dV.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "gradcheck-gqa"
//!   cargo run -p hipfire-train --release --example gradcheck_gqa
//!   hipfire gpu-lock release

use hipfire_train::ops::attention::{gqa_backward, gqa_forward};
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

const SEQ: usize = 4;
const NH: usize = 4;
const NKV: usize = 2;
const D: usize = 4;
const QDIM: usize = NH * D;
const KVDIM: usize = NKV * D;

fn scale() -> f32 {
    1.0 / (D as f32).sqrt()
}

fn loss(gpu: &mut Gpu, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, g: &[f32]) -> HipResult<f32> {
    let p_all = gpu.zeros(&[NH * SEQ * SEQ], DType::F32)?;
    let ctx = gpu.zeros(&[SEQ * QDIM], DType::F32)?;
    gqa_forward(gpu, q, k, v, &p_all, &ctx, SEQ, NH, NKV, D, scale())?;
    let cv = gpu.download_f32(&ctx)?;
    Ok(cv.iter().zip(g).map(|(a, b)| a * b).sum())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let qh: Vec<f32> = (0..SEQ * QDIM)
        .map(|i| ((i * 17 % 13) as f32) * 0.12 - 0.6)
        .collect();
    let kh: Vec<f32> = (0..SEQ * KVDIM)
        .map(|i| ((i * 23 % 11) as f32) * 0.1 - 0.4)
        .collect();
    let vh: Vec<f32> = (0..SEQ * KVDIM)
        .map(|i| ((i * 7 % 9) as f32) * 0.15 - 0.5)
        .collect();
    let gh: Vec<f32> = (0..SEQ * QDIM)
        .map(|i| ((i * 13 % 5) as f32) * 0.2 - 0.3)
        .collect();

    let q = gpu.upload_f32(&qh, &[SEQ * QDIM])?;
    let k = gpu.upload_f32(&kh, &[SEQ * KVDIM])?;
    let v = gpu.upload_f32(&vh, &[SEQ * KVDIM])?;

    let p_all = gpu.zeros(&[NH * SEQ * SEQ], DType::F32)?;
    let ctx = gpu.zeros(&[SEQ * QDIM], DType::F32)?;
    gqa_forward(&mut gpu, &q, &k, &v, &p_all, &ctx, SEQ, NH, NKV, D, scale())?;
    let d_ctx = gpu.upload_f32(&gh, &[SEQ * QDIM])?;
    let dq = gpu.zeros(&[SEQ * QDIM], DType::F32)?;
    let dk = gpu.zeros(&[SEQ * KVDIM], DType::F32)?; // zeroed (scatter-add target)
    let dv = gpu.zeros(&[SEQ * KVDIM], DType::F32)?;
    gqa_backward(
        &mut gpu,
        &d_ctx,
        &q,
        &k,
        &v,
        &p_all,
        &dq,
        &dk,
        &dv,
        SEQ,
        NH,
        NKV,
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

    println!("gqa dQ max|analytic-numeric| = {eq:.2e}");
    println!(
        "gqa dK max|analytic-numeric| = {ek:.2e}  (grad-accum across {} q-heads/kv)",
        NH / NKV
    );
    println!("gqa dV max|analytic-numeric| = {ev:.2e}");
    let tol = 1e-2f32;
    if eq < tol && ek < tol && ev < tol {
        println!("\nGRADCHECK PASS — GQA attention backward matches finite differences.");
        Ok(())
    } else {
        Err(format!("gradcheck FAIL: dQ {eq:.2e}, dK {ek:.2e}, dV {ev:.2e}").into())
    }
}
