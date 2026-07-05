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

//! Finite-difference gradcheck for ONE DSpark drafter block
//! (`dspark_block_forward`/`dspark_block_backward`): qk-norm + bidirectional
//! ctx++block attention + SwiGLU MLP, all fp32. Loss L = Σ X_OUT∘G ⇒
//! d_x_out = G. Checks analytic grads vs central differences for x_block, ctx
//! (main_x), and every layer weight.
//!
//! Run (LDS-wedge hazard on gfx1103 — do NOT run on nix2):
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire lock acquire "gradcheck-dspark-block"
//!   cargo run -p hipfire-train --release --example gradcheck_dspark_block
//!   hipfire lock release

use hipfire_rdna::{Gpu, GpuTensor, HipResult};
use hipfire_train::dspark_drafter::{
    dspark_block_backward, dspark_block_forward, DsparkBlockWeights, DsparkDims,
};

const H: usize = 16;
const NH: usize = 2;
const NKV: usize = 1;
const HD: usize = 8;
const INTER: usize = 32;
const BLOCK: usize = 3;
const CTX: usize = 4;
const QD: usize = NH * HD; // 16
const KVD: usize = NKV * HD; // 8
const KV_ROWS: usize = CTX + BLOCK; // 7

fn dims() -> DsparkDims {
    DsparkDims {
        h: H,
        n_heads: NH,
        n_kv: NKV,
        head_dim: HD,
        inter: INTER,
        rope_base: 10000.0,
        eps: 1e-6,
        qk_norm: true,
    }
}

fn rnd(n: usize, a: usize, b: usize, scale: f32, off: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * a % b) as f32) * scale + off).collect()
}

/// Fixed weight order (matches the closure indices below):
/// 0 input_ln, 1 wq, 2 wk, 3 wv, 4 wo, 5 q_norm, 6 k_norm, 7 post_ln,
/// 8 wgate, 9 wup, 10 wdown.
fn weight_hosts() -> Vec<Vec<f32>> {
    vec![
        rnd(H, 3, 4, 0.05, 0.9),            // input_ln
        rnd(QD * H, 11, 7, 0.06, -0.2),     // wq
        rnd(KVD * H, 13, 9, 0.06, -0.2),    // wk
        rnd(KVD * H, 5, 11, 0.06, -0.2),    // wv
        rnd(H * QD, 7, 13, 0.06, -0.2),     // wo
        rnd(HD, 7, 5, 0.05, 0.9),           // q_norm
        rnd(HD, 5, 6, 0.05, 0.9),           // k_norm
        rnd(H, 5, 4, 0.05, 0.9),            // post_ln
        rnd(INTER * H, 9, 7, 0.05, -0.15),  // wgate
        rnd(INTER * H, 11, 5, 0.05, -0.15), // wup
        rnd(H * INTER, 13, 7, 0.05, -0.15), // wdown
    ]
}

fn upload_all(gpu: &mut Gpu, w: &[Vec<f32>]) -> HipResult<Vec<GpuTensor>> {
    w.iter().map(|v| gpu.upload_f32(v, &[v.len()])).collect()
}

fn view(t: &[GpuTensor]) -> DsparkBlockWeights<'_> {
    DsparkBlockWeights {
        input_ln: &t[0],
        wq: &t[1],
        wk: &t[2],
        wv: &t[3],
        wo: &t[4],
        q_norm: &t[5],
        k_norm: &t[6],
        post_ln: &t[7],
        wgate: &t[8],
        wup: &t[9],
        wdown: &t[10],
    }
}

fn loss(
    gpu: &mut Gpu,
    xh: &[f32],
    ctxh: &[f32],
    wh: &[Vec<f32>],
    q_pos: &[f32],
    k_pos: &[f32],
    g: &[f32],
) -> HipResult<f32> {
    let x = gpu.upload_f32(xh, &[BLOCK * H])?;
    let ctx = gpu.upload_f32(ctxh, &[CTX * H])?;
    let wt = upload_all(gpu, wh)?;
    let (x_out, _acts) =
        dspark_block_forward(gpu, &x, &ctx, &view(&wt), &dims(), q_pos, k_pos, None)?;
    let ov = gpu.download_f32(&x_out)?;
    Ok(ov.iter().zip(g).map(|(p, q)| p * q).sum())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let q_pos: Vec<f32> = (0..BLOCK).map(|t| (CTX + t) as f32).collect();
    let k_pos: Vec<f32> = (0..KV_ROWS).map(|t| t as f32).collect();
    let xh = rnd(BLOCK * H, 17, 13, 0.1, -0.5);
    let ctxh = rnd(CTX * H, 19, 11, 0.1, -0.4);
    let gh = rnd(BLOCK * H, 7, 5, 0.2, -0.3);
    let wh = weight_hosts();

    // Analytic
    let x = gpu.upload_f32(&xh, &[BLOCK * H])?;
    let ctx = gpu.upload_f32(&ctxh, &[CTX * H])?;
    let wt = upload_all(&mut gpu, &wh)?;
    let (_xo, acts) = dspark_block_forward(
        &mut gpu,
        &x,
        &ctx,
        &view(&wt),
        &dims(),
        &q_pos,
        &k_pos,
        None,
    )?;
    let d_x_out = gpu.upload_f32(&gh, &[BLOCK * H])?;
    let (d_x, d_ctx, wg) =
        dspark_block_backward(&mut gpu, &d_x_out, &x, &ctx, &view(&wt), &acts, &dims())?;
    let d_x_a = gpu.download_f32(&d_x)?;
    let d_ctx_a = gpu.download_f32(&d_ctx)?;
    let wg_a: Vec<Vec<f32>> = vec![
        gpu.download_f32(&wg.dinput_ln)?,
        gpu.download_f32(&wg.dwq)?,
        gpu.download_f32(&wg.dwk)?,
        gpu.download_f32(&wg.dwv)?,
        gpu.download_f32(&wg.dwo)?,
        gpu.download_f32(&wg.dq_norm)?,
        gpu.download_f32(&wg.dk_norm)?,
        gpu.download_f32(&wg.dpost_ln)?,
        gpu.download_f32(&wg.dwgate)?,
        gpu.download_f32(&wg.dwup)?,
        gpu.download_f32(&wg.dwdown)?,
    ];

    let eps = 1e-3f32;
    let tol = 2e-2f32;
    let mut worst = 0.0f32;

    // Perturb x_block.
    {
        let mut e = 0.0f32;
        for i in 0..xh.len() {
            let mut hp = xh.clone();
            hp[i] += eps;
            let mut hm = xh.clone();
            hm[i] -= eps;
            let lp = loss(&mut gpu, &hp, &ctxh, &wh, &q_pos, &k_pos, &gh)?;
            let lm = loss(&mut gpu, &hm, &ctxh, &wh, &q_pos, &k_pos, &gh)?;
            e = e.max(((lp - lm) / (2.0 * eps) - d_x_a[i]).abs());
        }
        println!("d_x_block  {e:.2e}");
        worst = worst.max(e);
    }

    // Perturb ctx (main_x).
    {
        let mut e = 0.0f32;
        for i in 0..ctxh.len() {
            let mut hp = ctxh.clone();
            hp[i] += eps;
            let mut hm = ctxh.clone();
            hm[i] -= eps;
            let lp = loss(&mut gpu, &xh, &hp, &wh, &q_pos, &k_pos, &gh)?;
            let lm = loss(&mut gpu, &xh, &hm, &wh, &q_pos, &k_pos, &gh)?;
            e = e.max(((lp - lm) / (2.0 * eps) - d_ctx_a[i]).abs());
        }
        println!("d_ctx      {e:.2e}");
        worst = worst.max(e);
    }

    // Perturb each weight.
    let names = [
        "input_ln", "wq", "wk", "wv", "wo", "q_norm", "k_norm", "post_ln", "wgate", "wup", "wdown",
    ];
    for wi in 0..wh.len() {
        let mut e = 0.0f32;
        for i in 0..wh[wi].len() {
            let mut whp = wh.clone();
            whp[wi][i] += eps;
            let mut whm = wh.clone();
            whm[wi][i] -= eps;
            let lp = loss(&mut gpu, &xh, &ctxh, &whp, &q_pos, &k_pos, &gh)?;
            let lm = loss(&mut gpu, &xh, &ctxh, &whm, &q_pos, &k_pos, &gh)?;
            e = e.max(((lp - lm) / (2.0 * eps) - wg_a[wi][i]).abs());
        }
        println!("d{:<9} {e:.2e}", names[wi]);
        worst = worst.max(e);
    }

    println!("\nworst = {worst:.2e} (tol {tol:.2e})");
    if worst < tol {
        println!("GRADCHECK PASS — DSpark drafter block backward matches finite differences.");
        Ok(())
    } else {
        Err(format!("gradcheck FAIL: worst {worst:.2e}").into())
    }
}
