//! Phase 2 Q3 (step 1) — greedy generation from the fp32 Supra-50M teacher.
//!
//! Verifies the tokenizer + generation plumbing on the full-precision model
//! before any student/coherence comparison. Uses a fixed-length token buffer of
//! size prompt+gen: under causal masking, position i's logits depend only on
//! 0..i, so unfilled future slots (left as token 0) don't affect the next-token
//! logit we read. No KV cache — re-forwards the whole buffer each step (fine for
//! a short demo).
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   source ./scripts/gpu-lock.sh && gpu_acquire "gen-supra"
//!   cargo run -p hipfire-train --release --example generate_supra50m
//!   gpu_release

use hipfire_model::tokenizer::Tokenizer;
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use rdna_compute::Gpu;
use std::path::Path;

const MODEL_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const PROMPT: &str = "The history of the Roman Empire is";
const GEN: usize = 32;

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().fold((0u32, f32::MIN), |a, (i, &x)| if x > a.1 { (i as u32, x) } else { a }).0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(MODEL_DIR);
    if !dir.exists() {
        return Err(format!("model dir not found: {MODEL_DIR}").into());
    }
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let tok_json = std::fs::read_to_string(dir.join("tokenizer.json"))?;
    let tok = Tokenizer::from_hf_json(&tok_json).map_err(|e| format!("tokenizer: {e:?}"))?;
    let prompt_ids = tok.encode(PROMPT);
    let plen = prompt_ids.len();
    let l = plen + GEN;
    println!("prompt: {PROMPT:?} -> {plen} tokens, generating {GEN} (seq {l})");

    let (cfg, w) = load_llama_fp32(&mut gpu, dir)?;
    let vocab = cfg.vocab_size;
    let teacher = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, l, 4, 8.0)?;

    let mut tokens: Vec<u32> = vec![0u32; l];
    tokens[..plen].copy_from_slice(&prompt_ids);
    let pos: Vec<f32> = (0..l).map(|t| t as f32).collect();

    for cur in plen..l {
        let acts = model_forward(&mut gpu, &teacher, &tokens, &pos)?;
        let logits = gpu.download_f32(&acts.logits)?;
        let next = argmax(&logits[(cur - 1) * vocab..cur * vocab]);
        tokens[cur] = next;
    }

    let full = tok.decode(&tokens[..l]);
    let cont = tok.decode(&tokens[plen..l]);
    println!("\n── teacher greedy generation ──");
    println!("full:        {full}");
    println!("continuation:{cont}");
    println!("\n(pipeline check — coherence of a 50M model is its own limit; this only");
    println!(" confirms tokenizer + forward + generation produce real text.)");
    Ok(())
}
