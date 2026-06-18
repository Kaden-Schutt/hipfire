//! Phase 2 Q2 — QTIP recovery fine-tuning. Distill a QTIP-quantized Supra-50M
//! student toward the fp32 teacher, training LoRA + RMSNorm weights (codes
//! frozen). Shows the mean KL(teacher‖student) gap shrinking — the recovery FT
//! mechanism working end to end.
//!
//! Note: distills on a small FIXED set of sequences (no tokenizer/real text
//! yet). This demonstrates the mechanism (KL drops); broad coherence needs a
//! real calibration corpus — that's Q3.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   source ./scripts/gpu-lock.sh && gpu_acquire "qtip-recovery"
//!   cargo run -p hipfire-train --release --example recovery_ft_supra50m
//!   gpu_release

use hipfire_train::loader::{load_llama_fp32, LlamaWeightsF32};
use hipfire_train::model::{
    flatten_recovery_grads, model_distill_backward, model_forward, LlamaModel,
};
use hipfire_train::ops::softmax::softmax_forward;
use hipfire_train::optim::AdamW;
use hipfire_train::qtip_quant::qtip_quantize_dequant;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

const MODEL_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const N_SEQS: usize = 4;
const BITS: u32 = 3;
const BEAM: usize = 32;
const RANK: usize = 16;
const ALPHA: f32 = 32.0;
const LR: f32 = 1e-3;
const STEPS: usize = 120;

fn quantize_linears(
    gpu: &mut Gpu,
    w: &mut LlamaWeightsF32,
) -> Result<(), Box<dyn std::error::Error>> {
    for l in w.layers.iter_mut() {
        for t in [
            &mut l.q_proj,
            &mut l.k_proj,
            &mut l.v_proj,
            &mut l.o_proj,
            &mut l.gate_proj,
            &mut l.up_proj,
            &mut l.down_proj,
        ] {
            let host = gpu.download_f32(t)?;
            let q = qtip_quantize_dequant(&host, BITS, BEAM);
            *t = gpu.upload_f32(&q, &t.shape.clone())?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(MODEL_DIR);
    if !dir.exists() {
        return Err(format!("model dir not found: {MODEL_DIR}").into());
    }
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let (cfg, w_teacher) = load_llama_fp32(&mut gpu, dir)?;
    let (_, mut w_student) = load_llama_fp32(&mut gpu, dir)?;
    let vocab = cfg.vocab_size;
    println!("quantizing student to QTIP-{BITS}...");
    quantize_linears(&mut gpu, &mut w_student)?;

    let teacher = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_teacher, SEQ, RANK, ALPHA)?;
    let student = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_student, SEQ, RANK, ALPHA)?;

    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let batch: Vec<Vec<u32>> = (0..N_SEQS)
        .map(|s| {
            (0..SEQ)
                .map(|t| (((t + 1) * 2654435761 + s * 40503) % vocab) as u32)
                .collect()
        })
        .collect();

    // Precompute frozen teacher distributions (teacher never changes).
    let mut teacher_p: Vec<GpuTensor> = Vec::with_capacity(N_SEQS);
    for toks in &batch {
        let at = model_forward(&mut gpu, &teacher, toks, &pos)?;
        let p = gpu.zeros(&[SEQ * vocab], DType::F32)?;
        softmax_forward(&mut gpu, &at.logits, &p, SEQ, vocab)?;
        teacher_p.push(p);
    }

    let sizes = student.recovery_param_sizes();
    let mut opt = AdamW::new(&mut gpu, &sizes, LR, 0.9, 0.999, 1e-8, 0.0)?;
    println!(
        "recovery FT: {} trainable tensors (LoRA + layernorms)\n",
        sizes.len()
    );

    let mut last = 0.0f32;
    for step in 0..=STEPS {
        let mut total = 0.0f32;
        for (si, toks) in batch.iter().enumerate() {
            let acts = model_forward(&mut gpu, &student, toks, &pos)?;
            let (kl, grads, d_final) =
                model_distill_backward(&mut gpu, &student, &acts, &teacher_p[si])?;
            total += kl;
            if step < STEPS {
                let params = student.recovery_params();
                let gflat = flatten_recovery_grads(&grads, &d_final);
                opt.step(&mut gpu, &params, &gflat)?;
            }
        }
        last = total / (N_SEQS * SEQ) as f32;
        if step % 15 == 0 || step == STEPS {
            println!("step {step:4}: mean KL(teacher‖student) = {last:.4} nats/token");
        }
    }

    println!("\nfinal mean KL = {last:.4} nats/token (started near the QTIP-3 gap ~0.87)");
    if last < 0.2 {
        println!("WIN — recovery FT shrank the quantization gap. QTIP-style tuning works.");
        Ok(())
    } else if last < 0.7 {
        println!("PARTIAL — gap shrank; more steps / capacity / real calibration data would help.");
        Ok(())
    } else {
        Err(format!("gap did not shrink (KL {last:.4}) — investigate").into())
    }
}
