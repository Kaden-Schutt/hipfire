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

//! Phase 0 M3 — THE WIN: load Supra-50M (frozen fp32 base), attach LoRA on
//! q/v of every layer, and overfit a tiny fixed batch with AdamW. If the
//! mean per-token cross-entropy collapses toward ~0, the entire training loop
//! (un-fused fp32 forward → hand-written backward → AdamW) is proven to learn.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "overfit-supra"
//!   cargo run -p hipfire-train --release --example overfit_supra50m
//!   hipfire gpu-lock release

use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{flatten_lora_grads, model_forward, model_loss_backward, LlamaModel};
use hipfire_train::optim::AdamW;
use hipfire_rdna::Gpu;
use std::path::Path;

const MODEL_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";

const SEQ: usize = 8;
const N_SEQS: usize = 3;
const RANK: usize = 16;
const ALPHA: f32 = 32.0;
const LR: f32 = 5e-3;
const STEPS: usize = 300;
const IGNORE: i32 = -100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(MODEL_DIR);
    if !dir.exists() {
        return Err(format!("model dir not found: {MODEL_DIR}").into());
    }
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let (cfg, weights) = load_llama_fp32(&mut gpu, dir)?;
    let vocab = cfg.vocab_size;
    let model = LlamaModel::from_f32_weights(&mut gpu, &cfg, weights, SEQ, RANK, ALPHA)?;
    println!(
        "Supra-50M loaded: {} layers, LoRA r={RANK} a={ALPHA} on q/v ({} trainable tensors)",
        model.layers.len(),
        model.lora_params().len()
    );

    // Fixed overfit batch: N_SEQS deterministic token sequences; targets are the
    // next token (causal LM), last position masked.
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let batch: Vec<(Vec<u32>, Vec<f32>)> = (0..N_SEQS)
        .map(|s| {
            let toks: Vec<u32> = (0..SEQ)
                .map(|t| (((t + 1) * 2654435761 + s * 40503) % vocab) as u32)
                .collect();
            let mut tgts: Vec<f32> = (0..SEQ).map(|t| toks[(t + 1) % SEQ] as f32).collect();
            tgts[SEQ - 1] = IGNORE as f32; // no target for last position
            (toks, tgts)
        })
        .collect();
    let target_tokens = (N_SEQS * (SEQ - 1)) as f32;

    let sizes = model.lora_param_sizes();
    let mut opt = AdamW::new(&mut gpu, &sizes, LR, 0.9, 0.999, 1e-8, 0.0)?;

    let mut last_per_tok = 0.0f32;
    for step in 0..=STEPS {
        let mut total = 0.0f32;
        for (toks, tgts) in &batch {
            let acts = model_forward(&mut gpu, &model, toks, &pos)?;
            let (loss, grads) = model_loss_backward(&mut gpu, &model, &acts, tgts, IGNORE)?;
            total += loss;
            if step < STEPS {
                let params = model.lora_params();
                let gflat = flatten_lora_grads(&grads);
                opt.step(&mut gpu, &params, &gflat)?;
            }
        }
        last_per_tok = total / target_tokens;
        if step % 25 == 0 || step == STEPS {
            println!("step {step:4}: mean per-token CE = {last_per_tok:.4}");
        }
    }

    println!(
        "\nfinal mean per-token CE = {last_per_tok:.4}  (random-init baseline ≈ ln(vocab) = {:.2})",
        (vocab as f32).ln()
    );
    if last_per_tok < 0.1 {
        println!("WIN — LoRA overfit the batch to ~0 loss. Training loop proven end-to-end.");
        Ok(())
    } else if last_per_tok < 2.0 {
        println!("PARTIAL — loss dropped sharply but did not reach ~0; loop learns (may need more steps / capacity).");
        Ok(())
    } else {
        Err(format!("loss did not collapse (per-token CE {last_per_tok:.4}) — investigate").into())
    }
}
