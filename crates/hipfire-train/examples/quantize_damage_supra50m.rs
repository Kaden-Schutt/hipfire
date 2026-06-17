//! Phase 2 Q0b — measure how much QTIP quantization damages Supra-50M.
//!
//! Loads the model twice: a teacher (fp32 base) and a student whose 7 linears
//! per layer are QTIP-quantized (decode→fp32). Runs both forwards on a fixed
//! sequence and reports the gap the recovery FT must close: per-token CE for
//! each, the mean KL(teacher‖student), and top-1 argmax agreement.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   source ./scripts/gpu-lock.sh && gpu_acquire "qtip-damage"
//!   cargo run -p hipfire-train --release --example quantize_damage_supra50m
//!   gpu_release

use hipfire_train::loader::{load_llama_fp32, LlamaWeightsF32};
use hipfire_train::model::{model_forward, LlamaModel};
use hipfire_train::qtip_quant::qtip_quantize_dequant;
use rdna_compute::{Gpu, GpuTensor};
use std::path::Path;

const MODEL_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const BITS: u32 = 3;
const BEAM: usize = 32;

fn quantize_inplace(gpu: &mut Gpu, t: &mut GpuTensor) -> Result<(), Box<dyn std::error::Error>> {
    let host = gpu.download_f32(t)?;
    let q = qtip_quantize_dequant(&host, BITS, BEAM);
    *t = gpu.upload_f32(&q, &t.shape.clone())?;
    Ok(())
}

fn quantize_linears(gpu: &mut Gpu, w: &mut LlamaWeightsF32) -> Result<usize, Box<dyn std::error::Error>> {
    let mut nparams = 0;
    for l in w.layers.iter_mut() {
        for t in [
            &mut l.q_proj, &mut l.k_proj, &mut l.v_proj, &mut l.o_proj,
            &mut l.gate_proj, &mut l.up_proj, &mut l.down_proj,
        ] {
            nparams += t.shape.iter().product::<usize>();
            quantize_inplace(gpu, t)?;
        }
    }
    Ok(nparams)
}

fn softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().cloned().fold(f32::MIN, f32::max);
    let ex: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = ex.iter().sum();
    ex.iter().map(|x| x / s).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(MODEL_DIR);
    if !dir.exists() {
        return Err(format!("model dir not found: {MODEL_DIR}").into());
    }
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    // Teacher (fp32) and student (to be quantized).
    let (cfg, w_teacher) = load_llama_fp32(&mut gpu, dir)?;
    let (_, mut w_student) = load_llama_fp32(&mut gpu, dir)?;
    let vocab = cfg.vocab_size;

    println!("quantizing student linears to QTIP-{BITS} (beam {BEAM})...");
    let t0 = std::time::Instant::now();
    let nq = quantize_linears(&mut gpu, &mut w_student)?;
    println!("quantized {nq} linear params in {:.1}s", t0.elapsed().as_secs_f32());

    let teacher = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_teacher, SEQ, 4, 8.0)?;
    let student = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_student, SEQ, 4, 8.0)?;

    // Fixed input + next-token targets.
    let tokens: Vec<u32> = (0..SEQ).map(|t| (((t + 1) * 2654435761) % vocab) as u32).collect();
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();

    let at = model_forward(&mut gpu, &teacher, &tokens, &pos)?;
    let as_ = model_forward(&mut gpu, &student, &tokens, &pos)?;
    let lt = gpu.download_f32(&at.logits)?;
    let ls = gpu.download_f32(&as_.logits)?;

    let (mut ce_t, mut ce_s, mut kl, mut agree) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    let npos = SEQ - 1;
    for t in 0..npos {
        let pt = softmax(&lt[t * vocab..(t + 1) * vocab]);
        let ps = softmax(&ls[t * vocab..(t + 1) * vocab]);
        let tgt = tokens[t + 1] as usize;
        ce_t += -(pt[tgt].max(1e-12) as f64).ln();
        ce_s += -(ps[tgt].max(1e-12) as f64).ln();
        for v in 0..vocab {
            if pt[v] > 1e-12 {
                kl += pt[v] as f64 * ((pt[v].max(1e-12) / ps[v].max(1e-12)) as f64).ln();
            }
        }
        let amax = |p: &[f32]| p.iter().enumerate().fold((0, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0;
        if amax(&pt) == amax(&ps) {
            agree += 1;
        }
    }

    println!("\n── QTIP-{BITS} damage on Supra-50M (linears only, embed/norms fp) ──");
    println!("teacher per-token CE = {:.4}", ce_t / npos as f64);
    println!("student per-token CE = {:.4}", ce_s / npos as f64);
    println!("mean KL(teacher‖student) = {:.4} nats/token", kl / npos as f64);
    println!("top-1 argmax agreement = {}/{}", agree, npos);
    println!("\n→ recovery FT (Q2) must shrink the KL / restore agreement.");
    Ok(())
}
