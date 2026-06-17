//! Phase 2 Q3.2 — the coherence-recovery demonstration. Quantize Supra-50M to
//! QTIP-3, generate BEFORE recovery, distill the student on a real text corpus
//! against the fp32 teacher (LoRA + layernorms, codes frozen), generate AFTER,
//! and compare all three (teacher / student-before / student-after).
//!
//! One fixed seq length serves both the distillation chunks and generation
//! (causal buffer: future slots don't affect earlier logits).
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   source ./scripts/gpu-lock.sh && gpu_acquire "qtip-coherence"
//!   cargo run -p hipfire-train --release --example coherence_recovery_supra50m
//!   gpu_release

use hipfire_model::tokenizer::Tokenizer;
use hipfire_train::loader::{load_llama_fp32, LlamaWeightsF32};
use hipfire_train::model::{
    flatten_norm_grads, flatten_recovery_grads, model_distill_backward, model_forward, LlamaModel,
};
use hipfire_train::ops::softmax::softmax_forward;
use hipfire_train::optim::AdamW;
use hipfire_train::qtip_quant::qtip_quantize_dequant;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

const MODEL_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const L: usize = 32; // seq length for both distill chunks and generation
const BEAM: usize = 32;
const RANK: usize = 16;
const ALPHA: f32 = 32.0;
const LR: f32 = 1e-3;
const PROMPT: &str = "The Roman Empire was";
// Bit-width and step count are env-tunable (QTIP-2 needs more recovery than -3):
//   HIPFIRE_QTIP_BITS (default 3), HIPFIRE_QTIP_STEPS (default 120).
fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

const CORPUS: &str = "The Roman Empire was one of the largest empires in ancient history. At its \
height it controlled vast territories across Europe, North Africa, and the Middle East. Roman \
engineers built roads, aqueducts, and public buildings that still stand today. The empire was \
ruled by a series of emperors, beginning with Augustus. Latin, the language of Rome, became the \
foundation of many modern European languages. Over the centuries the empire faced invasions, \
economic troubles, and political instability. The western half eventually fell, while the eastern \
half continued as the Byzantine Empire for another thousand years. Roman law, architecture, and \
culture continue to influence the modern world to this day in countless ways.";

fn quantize_linears(
    gpu: &mut Gpu,
    w: &mut LlamaWeightsF32,
    bits: u32,
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
            let q = qtip_quantize_dequant(&host, bits, BEAM);
            *t = gpu.upload_f32(&q, &t.shape.clone())?;
        }
    }
    Ok(())
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .fold(
            (0u32, f32::MIN),
            |a, (i, &x)| if x > a.1 { (i as u32, x) } else { a },
        )
        .0
}

fn generate(
    gpu: &mut Gpu,
    m: &LlamaModel,
    prompt: &[u32],
    vocab: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let plen = prompt.len();
    let mut tokens = vec![0u32; L];
    tokens[..plen].copy_from_slice(prompt);
    let pos: Vec<f32> = (0..L).map(|t| t as f32).collect();
    for cur in plen..L {
        let acts = model_forward(gpu, m, &tokens, &pos)?;
        let logits = gpu.download_f32(&acts.logits)?;
        tokens[cur] = argmax(&logits[(cur - 1) * vocab..cur * vocab]);
    }
    Ok(tokens)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(MODEL_DIR);
    if !dir.exists() {
        return Err(format!("model dir not found: {MODEL_DIR}").into());
    }
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let tok = Tokenizer::from_hf_json(&std::fs::read_to_string(dir.join("tokenizer.json"))?)
        .map_err(|e| format!("tokenizer: {e:?}"))?;
    let prompt_ids = tok.encode(PROMPT);

    let (cfg, w_teacher) = load_llama_fp32(&mut gpu, dir)?;
    let (_, mut w_student) = load_llama_fp32(&mut gpu, dir)?;
    let vocab = cfg.vocab_size;
    let bits = env_u32("HIPFIRE_QTIP_BITS", 3);
    let steps = env_u32("HIPFIRE_QTIP_STEPS", 120) as usize;
    println!("quantizing student to QTIP-{bits}...");
    quantize_linears(&mut gpu, &mut w_student, bits)?;

    let teacher = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_teacher, L, RANK, ALPHA)?;
    let student = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_student, L, RANK, ALPHA)?;

    // Generations BEFORE recovery.
    let gen_teacher = generate(&mut gpu, &teacher, &prompt_ids, vocab)?;
    let gen_before = generate(&mut gpu, &student, &prompt_ids, vocab)?;

    // Distillation corpus → L-token chunks → frozen teacher distributions.
    let corpus_ids = tok.encode(CORPUS);
    let n_chunks = corpus_ids.len() / L;
    println!(
        "corpus {} tokens → {n_chunks} chunks of {L}",
        corpus_ids.len()
    );
    let pos: Vec<f32> = (0..L).map(|t| t as f32).collect();
    let mut chunks: Vec<Vec<u32>> = Vec::new();
    let mut teacher_p: Vec<GpuTensor> = Vec::new();
    for c in 0..n_chunks {
        let toks = corpus_ids[c * L..(c + 1) * L].to_vec();
        let at = model_forward(&mut gpu, &teacher, &toks, &pos)?;
        let p = gpu.zeros(&[L * vocab], DType::F32)?;
        softmax_forward(&mut gpu, &at.logits, &p, L, vocab)?;
        teacher_p.push(p);
        chunks.push(toks);
    }

    // Recovery FT. Mode selects trainable params:
    //   HIPFIRE_RECOVER_MODE=norms      → layernorm-only (faithful QTIP, Path A export)
    //   HIPFIRE_RECOVER_MODE=lora+norms → LoRA + layernorms (default, more capacity)
    let norms_only = std::env::var("HIPFIRE_RECOVER_MODE").as_deref() == Ok("norms");
    let sizes = if norms_only { student.norm_param_sizes() } else { student.recovery_param_sizes() };
    let mut opt = AdamW::new(&mut gpu, &sizes, LR, 0.9, 0.999, 1e-8, 0.0)?;
    println!(
        "recovery FT [{}] ({} trainable tensors)...",
        if norms_only { "norms-only" } else { "lora+norms" },
        sizes.len()
    );
    let mut last = 0.0f32;
    for step in 0..steps {
        let mut total = 0.0f32;
        for (ci, toks) in chunks.iter().enumerate() {
            let acts = model_forward(&mut gpu, &student, toks, &pos)?;
            let (kl, grads, d_final) =
                model_distill_backward(&mut gpu, &student, &acts, &teacher_p[ci])?;
            total += kl;
            if norms_only {
                let params = student.norm_params();
                let gflat = flatten_norm_grads(&grads, &d_final);
                opt.step(&mut gpu, &params, &gflat)?;
            } else {
                let params = student.recovery_params();
                let gflat = flatten_recovery_grads(&grads, &d_final);
                opt.step(&mut gpu, &params, &gflat)?;
            }
        }
        last = total / (n_chunks * L) as f32;
        if step % 20 == 0 {
            println!("  step {step:3}: corpus KL = {last:.4} nats/token");
        }
    }
    println!("  final corpus KL = {last:.4} nats/token");

    // Generation AFTER recovery.
    let gen_after = generate(&mut gpu, &student, &prompt_ids, vocab)?;

    let plen = prompt_ids.len();
    println!("\n══ greedy continuations of {PROMPT:?}  (QTIP-{bits}) ══");
    println!(
        "TEACHER (fp32):        {}",
        tok.decode(&gen_teacher[plen..L])
    );
    println!(
        "STUDENT before recov:  {}",
        tok.decode(&gen_before[plen..L])
    );
    println!("STUDENT after  recov:  {}", tok.decode(&gen_after[plen..L]));
    println!("\n(corpus KL dropped over FT; compare whether 'after' reads closer to the");
    println!(
        " teacher / more coherent than 'before' — eyeball test for a 50M @ {bits}-bit model.)"
    );
    Ok(())
}
