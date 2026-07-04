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

//! OQ+ (Opus Plus, W4A8) recovery fine-tuning probe. Distill an OQ+-quantized
//! Supra-50M student toward the fp32 teacher, training LoRA (q/v) + RMSNorm
//! (weight codes frozen). Reports the mean KL(teacher‖student) gap BEFORE vs
//! AFTER recovery — i.e. how much of the OQ+ weight-quant loss is recoverable.
//!
//! Mirrors `recovery_ft_supra50m.rs` (QTIP) but swaps the codec for OQ+
//! sim-quant (`oqplus_quant::oqplus_simquant`: sym-int4 + FWHT-256 + clip-search).
//! Per the user's choice this trains LoRA + norm (ceiling-measuring; LoRA can't
//! ship into OQ+ without a fold-then-requant step — that's deferred).
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "oqplus-recovery" --watch-pid $$
//!   cargo run -p hipfire-train --release --example oqplus_recovery_supra50m
//!   hipfire gpu-lock release

use hipfire_train::loader::{load_llama_fp32, LlamaWeightsF32};
use hipfire_train::model::{
    flatten_recovery_grads, model_distill_backward, model_forward, LlamaModel,
};
use hipfire_train::ops::softmax::softmax_forward;
use hipfire_train::optim::AdamW;
use hipfire_train::oqplus_quant::oqplus_simquant;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::path::Path;

const MODEL_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const N_SEQS: usize = 4;
const RANK: usize = 16;
const ALPHA: f32 = 32.0;
const LR: f32 = 1e-3;
const STEPS: usize = 120;

/// OQ+ sim-quant the 7 linears per layer (same set as the QTIP probe).
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
            let q = oqplus_simquant(&host);
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
    println!("quantizing student to OQ+ (W4A8, sym-int4 + FWHT-256 + clip-search)...");
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

    let mut start = 0.0f32;
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
        if step == 0 {
            start = last;
        }
        if step % 15 == 0 || step == STEPS {
            println!("step {step:4}: mean KL(teacher‖student) = {last:.4} nats/token");
        }
    }

    let recovered = if start > 1e-6 {
        100.0 * (start - last) / start
    } else {
        0.0
    };
    println!(
        "\nOQ+ start gap = {start:.4} → final = {last:.4} nats/token  ({recovered:.1}% recovered)"
    );
    if recovered > 40.0 {
        println!("WIN — recovery FT meaningfully closes the OQ+ W4A8 weight-quant gap.");
        Ok(())
    } else if recovered > 10.0 {
        println!("PARTIAL — gap shrank; more steps / capacity / real calibration data would help.");
        Ok(())
    } else {
        Err(format!("gap barely moved ({recovered:.1}% recovered) — investigate").into())
    }
}
