#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

//! End-to-end finite-difference gradient check for one transformer block
//! (Phase 0, M2). Tiny synthetic config + random base weights. Loss
//! L = Σ X_OUT∘G ⇒ d_x_out = G; checks dAq, dBq, dAv, dBv (LoRA params) —
//! gradients flowing back through MLP, residuals, attention, rope, and norms.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "gradcheck-block"
//!   cargo run -p hipfire-train --release --example gradcheck_block
//!   hipfire gpu-lock release

use hipfire_train::block::{block_backward, block_forward, BlockDims, BlockLora, BlockWeights};
use hipfire_rdna::{Gpu, GpuTensor, HipResult};

const SEQ: usize = 3;
const H: usize = 8;
const NH: usize = 2;
const NKV: usize = 1;
const HD: usize = 4;
const INTER: usize = 16;
const R: usize = 2;
const QD: usize = NH * HD; // 8
const KVD: usize = NKV * HD; // 4

fn dims() -> BlockDims {
    BlockDims {
        seq: SEQ,
        h: H,
        n_heads: NH,
        n_kv: NKV,
        head_dim: HD,
        inter: INTER,
        rope_base: 10000.0,
        eps: 1e-6,
        lora_scale: 1.0,
        lora_rank: R,
    }
}

fn rnd(n: usize, a: usize, b: usize, scale: f32, off: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * a % b) as f32) * scale + off).collect()
}

#[allow(clippy::too_many_arguments)]
fn loss(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &BlockWeights,
    aq: &GpuTensor,
    bq: &GpuTensor,
    av: &GpuTensor,
    bv: &GpuTensor,
    g: &[f32],
    pos: &[f32],
) -> HipResult<f32> {
    let lora = BlockLora { aq, bq, av, bv };
    let (x_out, _) = block_forward(gpu, x, w, &lora, &dims(), pos)?;
    let ov = gpu.download_f32(&x_out)?;
    Ok(ov.iter().zip(g).map(|(p, q)| p * q).sum())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let xh = rnd(SEQ * H, 17, 13, 0.1, -0.5);
    let gh = rnd(SEQ * H, 7, 5, 0.2, -0.3);

    // Base weights (random, frozen). norms near 1.0.
    let up = |gpu: &mut Gpu, v: &[f32]| gpu.upload_f32(v, &[v.len()]);
    let norm1 = up(&mut gpu, &rnd(H, 3, 4, 0.05, 0.9))?;
    let wq = up(&mut gpu, &rnd(QD * H, 11, 7, 0.06, -0.2))?;
    let wk = up(&mut gpu, &rnd(KVD * H, 13, 9, 0.06, -0.2))?;
    let wv = up(&mut gpu, &rnd(KVD * H, 5, 11, 0.06, -0.2))?;
    let wo = up(&mut gpu, &rnd(H * QD, 7, 13, 0.06, -0.2))?;
    let norm2 = up(&mut gpu, &rnd(H, 5, 4, 0.05, 0.9))?;
    let wgate = up(&mut gpu, &rnd(INTER * H, 9, 7, 0.05, -0.15))?;
    let wup = up(&mut gpu, &rnd(INTER * H, 11, 5, 0.05, -0.15))?;
    let wdown = up(&mut gpu, &rnd(H * INTER, 13, 7, 0.05, -0.15))?;
    let w = BlockWeights {
        norm1: &norm1,
        wq: &wq,
        wk: &wk,
        wv: &wv,
        wo: &wo,
        norm2: &norm2,
        wgate: &wgate,
        wup: &wup,
        wdown: &wdown,
    };

    // LoRA params (random, trainable).
    let aqh = rnd(R * H, 7, 5, 0.1, -0.2);
    let bqh = rnd(QD * R, 11, 7, 0.1, -0.2);
    let avh = rnd(R * H, 13, 9, 0.1, -0.2);
    let bvh = rnd(KVD * R, 5, 11, 0.1, -0.2);
    let aq = up(&mut gpu, &aqh)?;
    let bq = up(&mut gpu, &bqh)?;
    let av = up(&mut gpu, &avh)?;
    let bv = up(&mut gpu, &bvh)?;

    let x = up(&mut gpu, &xh)?;

    // Analytic
    let lora = BlockLora {
        aq: &aq,
        bq: &bq,
        av: &av,
        bv: &bv,
    };
    let (_xo, acts) = block_forward(&mut gpu, &x, &w, &lora, &dims(), &pos)?;
    let d_x_out = up(&mut gpu, &gh)?;
    let (_dx, grads) = block_backward(&mut gpu, &d_x_out, &x, &w, &lora, &acts, &dims())?;
    let daq = gpu.download_f32(&grads.daq)?;
    let dbq = gpu.download_f32(&grads.dbq)?;
    let dav = gpu.download_f32(&grads.dav)?;
    let dbv = gpu.download_f32(&grads.dbv)?;

    let eps = 1e-3f32;
    // which: 0=aq,1=bq,2=av,3=bv
    let check = |gpu: &mut Gpu, host: &[f32], which: u8, ana: &[f32]| -> HipResult<f32> {
        let mut e = 0.0f32;
        for i in 0..host.len() {
            let mut hp = host.to_vec();
            hp[i] += eps;
            let mut hm = host.to_vec();
            hm[i] -= eps;
            let pd = gpu.upload_f32(&hp, &[host.len()])?;
            let md = gpu.upload_f32(&hm, &[host.len()])?;
            let (lp, lm) = match which {
                0 => (
                    loss(gpu, &x, &w, &pd, &bq, &av, &bv, &gh, &pos)?,
                    loss(gpu, &x, &w, &md, &bq, &av, &bv, &gh, &pos)?,
                ),
                1 => (
                    loss(gpu, &x, &w, &aq, &pd, &av, &bv, &gh, &pos)?,
                    loss(gpu, &x, &w, &aq, &md, &av, &bv, &gh, &pos)?,
                ),
                2 => (
                    loss(gpu, &x, &w, &aq, &bq, &pd, &bv, &gh, &pos)?,
                    loss(gpu, &x, &w, &aq, &bq, &md, &bv, &gh, &pos)?,
                ),
                _ => (
                    loss(gpu, &x, &w, &aq, &bq, &av, &pd, &gh, &pos)?,
                    loss(gpu, &x, &w, &aq, &bq, &av, &md, &gh, &pos)?,
                ),
            };
            e = e.max(((lp - lm) / (2.0 * eps) - ana[i]).abs());
        }
        Ok(e)
    };

    let eaq = check(&mut gpu, &aqh, 0, &daq)?;
    let ebq = check(&mut gpu, &bqh, 1, &dbq)?;
    let eav = check(&mut gpu, &avh, 2, &dav)?;
    let ebv = check(&mut gpu, &bvh, 3, &dbv)?;

    println!("block dAq max|analytic-numeric| = {eaq:.2e}");
    println!("block dBq max|analytic-numeric| = {ebq:.2e}");
    println!("block dAv max|analytic-numeric| = {eav:.2e}");
    println!("block dBv max|analytic-numeric| = {ebv:.2e}");
    let tol = 2e-2f32;
    if eaq < tol && ebq < tol && eav < tol && ebv < tol {
        println!("\nGRADCHECK PASS — full block LoRA backward matches finite differences.");
        Ok(())
    } else {
        Err(
            format!("gradcheck FAIL: dAq {eaq:.2e}, dBq {ebq:.2e}, dAv {eav:.2e}, dBv {ebv:.2e}")
                .into(),
        )
    }
}
