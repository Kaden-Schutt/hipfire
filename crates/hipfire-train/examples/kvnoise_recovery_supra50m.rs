//! KVarN + CASK-merge KV-noise recovery probe. Isolates the *KV-compression*
//! noise (NOT weight quant): teacher and student share full-precision fp32
//! weights; the student's attention forward runs post-RoPE K and V through
//! KVarN-4bit quant + CASK cold-token merge (the RoPE-phase blur the
//! kv-compression study isolates as the ~+3 PPL "inherent lossy-merge floor").
//!
//! Question: is that merge floor trainable-away? We recovery-FT LoRA(q/v) + norm
//! with a straight-through estimator on the KV perturbation and watch
//! KL(clean-teacher‖compressed-student) before vs after.
//!
//! Knobs (the noise is forced ON for the student here regardless of preset env):
//!   HIPFIRE_KVNOISE_HOT (exact recent window, default 4)
//!   HIPFIRE_KVNOISE_FOLD (cold merge group, default 4)
//!   HIPFIRE_KVNOISE_BITS (KVarN bits, default 4; 0 = merge only)
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "kvnoise-recovery" --watch-pid $$
//!   cargo run -p hipfire-train --release --example kvnoise_recovery_supra50m
//!   hipfire gpu-lock release

use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{
    flatten_recovery_grads, model_distill_backward, model_forward, LlamaModel,
};
use hipfire_train::ops::softmax::softmax_forward;
use hipfire_train::optim::AdamW;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

const MODEL_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const N_SEQS: usize = 4;
const RANK: usize = 16;
const ALPHA: f32 = 32.0;
const STEPS: usize = 120;
// LR is env-tunable (HIPFIRE_KVNOISE_LR) — aggressive merge+quant via STE can
// need a lower LR to stay stable.
fn lr() -> f32 {
    std::env::var("HIPFIRE_KVNOISE_LR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3e-4)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(MODEL_DIR);
    if !dir.exists() {
        return Err(format!("model dir not found: {MODEL_DIR}").into());
    }
    // Teacher must run CLEAN: ensure KV-noise is off during its precompute.
    std::env::remove_var("HIPFIRE_KVNOISE");

    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let (cfg, w_teacher) = load_llama_fp32(&mut gpu, dir)?;
    let (_, w_student) = load_llama_fp32(&mut gpu, dir)?;
    let vocab = cfg.vocab_size;

    // Same full-precision weights for both — this probe isolates KV noise only.
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

    // Frozen CLEAN teacher distributions (KV-noise still off).
    let mut teacher_p: Vec<GpuTensor> = Vec::with_capacity(N_SEQS);
    for toks in &batch {
        let at = model_forward(&mut gpu, &teacher, toks, &pos)?;
        let p = gpu.zeros(&[SEQ * vocab], DType::F32)?;
        softmax_forward(&mut gpu, &at.logits, &p, SEQ, vocab)?;
        teacher_p.push(p);
    }

    // Now turn KV-noise ON for every subsequent (student) forward.
    std::env::set_var("HIPFIRE_KVNOISE", "1");
    let hot = std::env::var("HIPFIRE_KVNOISE_HOT").unwrap_or_else(|_| "4".into());
    let fold = std::env::var("HIPFIRE_KVNOISE_FOLD").unwrap_or_else(|_| "4".into());
    let bits = std::env::var("HIPFIRE_KVNOISE_BITS").unwrap_or_else(|_| "4".into());
    println!("KV-noise: KVarN-{bits}bit + CASK merge (hot={hot}, fold={fold}) on K & V\n");

    let sizes = student.recovery_param_sizes();
    let mut opt = AdamW::new(&mut gpu, &sizes, lr(), 0.9, 0.999, 1e-8, 0.0)?;

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
            println!("step {step:4}: mean KL(clean‖kv-compressed) = {last:.4} nats/token");
        }
    }

    let recovered = if start > 1e-6 {
        100.0 * (start - last) / start
    } else {
        0.0
    };
    println!(
        "\nKV-noise start gap = {start:.4} → final = {last:.4} nats/token  ({recovered:.1}% recovered)"
    );
    if recovered > 40.0 {
        println!("WIN — recovery FT absorbs most of the KVarN+CASK merge noise.");
        Ok(())
    } else if recovered > 10.0 {
        println!("PARTIAL — some of the merge floor is trainable; rest is inherent position blur.");
        Ok(())
    } else {
        println!("FLOOR — KV merge noise is largely NOT recoverable by q/v+norm FT (as theory predicts: RoPE-phase blur is information loss, not a learnable bias).");
        Ok(())
    }
}
