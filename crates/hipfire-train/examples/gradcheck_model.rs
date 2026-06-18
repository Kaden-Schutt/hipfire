//! End-to-end finite-difference gradient check for the FULL model (Phase 0, M2
//! completion). 2-layer tiny LLaMA, tied embeddings, CE loss with one masked
//! target. Checks LoRA grads in BOTH layers — exercising the reverse block loop,
//! the tied-logit head, and the final-norm/embed bookends.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   source ./scripts/gpu-lock.sh && gpu_acquire "gradcheck-model"
//!   cargo run -p hipfire-train --release --example gradcheck_model
//!   gpu_release

use hipfire_train::block::BlockDims;
use hipfire_train::model::{model_forward, model_loss_backward};
use hipfire_train::model::{LayerLora, LayerWeights, LlamaModel};
use rdna_compute::{Gpu, HipResult};

const NL: usize = 2;
const SEQ: usize = 3;
const H: usize = 8;
const NH: usize = 2;
const NKV: usize = 1;
const HD: usize = 4;
const INTER: usize = 16;
const R: usize = 2;
const VOCAB: usize = 10;
const QD: usize = NH * HD;
const KVD: usize = NKV * HD;
const IGNORE: i32 = -100;

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

// Deterministic base weights (identical across rebuilds), varied by layer l.
fn base_layer(l: usize) -> [Vec<f32>; 9] {
    let s = l + 1;
    [
        rnd(H, 3 * s, 4, 0.05, 0.9),            // norm1
        rnd(QD * H, 11 * s, 7, 0.06, -0.2),     // wq
        rnd(KVD * H, 13 * s, 9, 0.06, -0.2),    // wk
        rnd(KVD * H, 5 * s, 11, 0.06, -0.2),    // wv
        rnd(H * QD, 7 * s, 13, 0.06, -0.2),     // wo
        rnd(H, 5 * s, 4, 0.05, 0.9),            // norm2
        rnd(INTER * H, 9 * s, 7, 0.05, -0.15),  // wgate
        rnd(INTER * H, 11 * s, 5, 0.05, -0.15), // wup
        rnd(H * INTER, 13 * s, 7, 0.05, -0.15), // wdown
    ]
}

// LoRA host params per layer: [aq, bq, av, bv].
fn lora_init(l: usize) -> [Vec<f32>; 4] {
    let s = l + 1;
    [
        rnd(R * H, 7 * s, 5, 0.1, -0.2),
        rnd(QD * R, 11 * s, 7, 0.1, -0.2),
        rnd(R * H, 13 * s, 9, 0.1, -0.2),
        rnd(KVD * R, 5 * s, 11, 0.1, -0.2),
    ]
}

fn build_model(gpu: &mut Gpu, lora: &[[Vec<f32>; 4]]) -> HipResult<LlamaModel> {
    let embed = gpu.upload_f32(&rnd(VOCAB * H, 7, 13, 0.05, -0.1), &[VOCAB * H])?;
    let final_norm = gpu.upload_f32(&rnd(H, 3, 4, 0.05, 0.9), &[H])?;
    let mut layers = Vec::with_capacity(NL);
    for l in 0..NL {
        let b = base_layer(l);
        let lw = LayerWeights {
            norm1: gpu.upload_f32(&b[0], &[H])?,
            wq: gpu.upload_f32(&b[1], &[QD * H])?,
            wk: gpu.upload_f32(&b[2], &[KVD * H])?,
            wv: gpu.upload_f32(&b[3], &[KVD * H])?,
            wo: gpu.upload_f32(&b[4], &[H * QD])?,
            norm2: gpu.upload_f32(&b[5], &[H])?,
            wgate: gpu.upload_f32(&b[6], &[INTER * H])?,
            wup: gpu.upload_f32(&b[7], &[INTER * H])?,
            wdown: gpu.upload_f32(&b[8], &[H * INTER])?,
        };
        let ll = LayerLora {
            aq: gpu.upload_f32(&lora[l][0], &[R * H])?,
            bq: gpu.upload_f32(&lora[l][1], &[QD * R])?,
            av: gpu.upload_f32(&lora[l][2], &[R * H])?,
            bv: gpu.upload_f32(&lora[l][3], &[KVD * R])?,
        };
        layers.push((lw, ll));
    }
    Ok(LlamaModel {
        embed,
        final_norm,
        layers,
        dims: dims(),
        vocab: VOCAB,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let tokens: Vec<u32> = vec![2, 5, 1];
    let targets: Vec<f32> = vec![5.0, 1.0, IGNORE as f32]; // last masked
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();

    let lora: Vec<[Vec<f32>; 4]> = (0..NL).map(lora_init).collect();

    // Analytic
    let model = build_model(&mut gpu, &lora)?;
    let acts = model_forward(&mut gpu, &model, &tokens, &pos)?;
    let (loss0, grads) = model_loss_backward(&mut gpu, &model, &acts, &targets, IGNORE)?;
    println!("loss (sum over non-masked) = {loss0:.5}");

    let loss_only = |gpu: &mut Gpu, lora: &[[Vec<f32>; 4]]| -> HipResult<f32> {
        let m = build_model(gpu, lora)?;
        let a = model_forward(gpu, &m, &tokens, &pos)?;
        let (l, _) = model_loss_backward(gpu, &m, &a, &targets, IGNORE)?;
        Ok(l)
    };

    let names = ["aq", "bq", "av", "bv"];
    let eps = 1e-3f32;
    let tol = 3e-2f32;
    let mut worst = 0.0f32;
    for l in 0..NL {
        for which in 0..4usize {
            let ana = match which {
                0 => gpu.download_f32(&grads[l].daq)?,
                1 => gpu.download_f32(&grads[l].dbq)?,
                2 => gpu.download_f32(&grads[l].dav)?,
                _ => gpu.download_f32(&grads[l].dbv)?,
            };
            let mut e = 0.0f32;
            for i in 0..lora[l][which].len() {
                let mut lp = lora.to_vec();
                lp[l][which][i] += eps;
                let mut lm = lora.to_vec();
                lm[l][which][i] -= eps;
                let hp = loss_only(&mut gpu, &lp)?;
                let hm = loss_only(&mut gpu, &lm)?;
                e = e.max(((hp - hm) / (2.0 * eps) - ana[i]).abs());
            }
            worst = worst.max(e);
            println!("layer{l} {} max|analytic-numeric| = {e:.2e}", names[which]);
        }
    }

    if worst < tol {
        println!("\nGRADCHECK PASS — full 2-layer model LoRA backward matches finite differences.");
        Ok(())
    } else {
        Err(format!("gradcheck FAIL (tol {tol:.0e}): worst {worst:.2e}").into())
    }
}
