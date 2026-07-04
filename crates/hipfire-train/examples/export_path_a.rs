//! Phase 3 Path A — export tool (layer-1 runtime-unified). The student base is
//! loaded DIRECTLY from the served `.hfq` (decoded to fp32 via
//! `load_llama_from_hfq`), so it IS the served model — no re-quantize, no
//! beam/grouping/format-matching guesswork. Recover RMSNorms (codes/weights
//! frozen) against the fp32 teacher, patch the tuned norms back into the `.hfq`.
//!
//! Works for any `.hfq` the loader can decode (bf16 / qtip2-sim today; qtip3
//! real-packed once Qtip3 decode lands).
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "export-path-a"
//!   cargo run -p hipfire-train --release --example export_path_a -- \
//!       <served.hfq> <recovered.hfq>
//!   hipfire gpu-lock release

use hipfire_model::tokenizer::Tokenizer;
use hipfire_train::hfq_patch::{is_norm, parse_hfq, patch_norms_inplace};
use hipfire_train::loader::{load_llama_fp32, load_llama_from_hfq};
use hipfire_train::model::{flatten_norm_grads, model_distill_backward, model_forward, LlamaModel};
use hipfire_train::ops::softmax::softmax_forward;
use hipfire_train::optim::AdamW;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::path::Path;

const MODEL_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const L: usize = 32;
const LR: f32 = 1e-3;
const STEPS: usize = 200;

const CORPUS: &str = "The Roman Empire was one of the largest empires in ancient history. At its \
height it controlled vast territories across Europe, North Africa, and the Middle East. Roman \
engineers built roads, aqueducts, and public buildings that still stand today. The empire was \
ruled by a series of emperors, beginning with Augustus. Latin, the language of Rome, became the \
foundation of many modern European languages. Over the centuries the empire faced invasions, \
economic troubles, and political instability. The western half eventually fell, while the eastern \
half continued as the Byzantine Empire for another thousand years. Roman law, architecture, and \
culture continue to influence the modern world to this day in countless ways.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let in_hfq = args
        .next()
        .ok_or("usage: export_path_a <in.hfq> <out.hfq>")?;
    let out_hfq = args
        .next()
        .ok_or("usage: export_path_a <in.hfq> <out.hfq>")?;
    let dir = Path::new(MODEL_DIR);

    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let tok = Tokenizer::from_hf_json(&std::fs::read_to_string(dir.join("tokenizer.json"))?)
        .map_err(|e| format!("tokenizer: {e:?}"))?;

    // Teacher = fp32 original (distillation target). Student = the served .hfq,
    // decoded to fp32 — it IS the served model, no re-quantize/matching.
    let (cfg, w_teacher) = load_llama_fp32(&mut gpu, dir)?;
    let (scfg, w_student) = load_llama_from_hfq(&mut gpu, Path::new(&in_hfq))?;
    let vocab = cfg.vocab_size;
    println!(
        "student loaded directly from served {in_hfq} (decoded), {} layers",
        scfg.num_hidden_layers
    );

    let teacher = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_teacher, L, 16, 32.0)?;
    let student = LlamaModel::from_f32_weights(&mut gpu, &scfg, w_student, L, 16, 32.0)?;

    // teacher distributions over the corpus chunks
    let corpus_ids = tok.encode(CORPUS);
    let n_chunks = corpus_ids.len() / L;
    let pos: Vec<f32> = (0..L).map(|t| t as f32).collect();
    let mut chunks = Vec::new();
    let mut teacher_p: Vec<GpuTensor> = Vec::new();
    for c in 0..n_chunks {
        let toks = corpus_ids[c * L..(c + 1) * L].to_vec();
        let at = model_forward(&mut gpu, &teacher, &toks, &pos)?;
        let p = gpu.zeros(&[L * vocab], DType::F32)?;
        softmax_forward(&mut gpu, &at.logits, &p, L, vocab)?;
        teacher_p.push(p);
        chunks.push(toks);
    }

    // norms-only recovery
    let sizes = student.norm_param_sizes();
    let mut opt = AdamW::new(&mut gpu, &sizes, LR, 0.9, 0.999, 1e-8, 0.0)?;
    println!(
        "norms-only recovery ({} norm tensors, {n_chunks} chunks)...",
        sizes.len()
    );
    let mut last = 0.0f32;
    for step in 0..STEPS {
        let mut total = 0.0f32;
        for (ci, toks) in chunks.iter().enumerate() {
            let acts = model_forward(&mut gpu, &student, toks, &pos)?;
            let (kl, grads, d_final) =
                model_distill_backward(&mut gpu, &student, &acts, &teacher_p[ci])?;
            total += kl;
            let params = student.norm_params();
            let gflat = flatten_norm_grads(&grads, &d_final);
            opt.step(&mut gpu, &params, &gflat)?;
        }
        last = total / (n_chunks * L) as f32;
        if step % 40 == 0 {
            println!("  step {step:3}: corpus KL = {last:.4}");
        }
    }
    println!("  final corpus KL = {last:.4} nats/token");

    // collect tuned norms → name map
    let mut tuned: HashMap<String, Vec<f32>> = HashMap::new();
    for (i, (w, _)) in student.layers.iter().enumerate() {
        tuned.insert(
            format!("model.layers.{i}.input_layernorm.weight"),
            gpu.download_f32(&w.norm1)?,
        );
        tuned.insert(
            format!("model.layers.{i}.post_attention_layernorm.weight"),
            gpu.download_f32(&w.norm2)?,
        );
    }
    tuned.insert(
        "model.norm.weight".to_string(),
        gpu.download_f32(&student.final_norm)?,
    );
    // sanity: every tuned name must be a norm
    assert!(tuned.keys().all(|k| is_norm(k)));

    // patch the .hfq
    let mut bytes = std::fs::read(&in_hfq)?;
    let (entries, _meta) = parse_hfq(&bytes)?;
    let n = patch_norms_inplace(&mut bytes, &entries, &tuned)?;
    std::fs::write(&out_hfq, &bytes)?;
    println!("\npatched {n}/{} norm tensors → {out_hfq}", tuned.len());
    if n != tuned.len() {
        return Err(format!(
            "patched {n} but had {} tuned norms — name mismatch",
            tuned.len()
        )
        .into());
    }
    println!("OK — recovered .hfq written (codes/weights unchanged, norms tuned).");
    Ok(())
}
