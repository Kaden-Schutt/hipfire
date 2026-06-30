// SPDX-License-Identifier: Apache-2.0
//! GuidedQuant down_proj calibration-backward driver (first move, LLaMA-dense).
//!
//! Loads an fp32 LLaMA (safetensors dir), runs forward + cross-entropy backward
//! over calibration token sequences, capturing each layer's down_proj
//! **Fisher-weighted** Hessian H̄ = Σ wₙ·xₙxₙᵀ (wₙ from the down output-grad),
//! and writes a `.calib.hfq` whose `model.layers.{l}.mlp.down_proj.hessian`
//! entries the quantizer's LDLQ consumes unchanged (point `--hessian` /
//! `HIPFIRE_QTIP_HESSIAN` at it). This is the in-engine GuidedQuant Hessian:
//! the end-loss gradients come from hipfire-train's autograd, no external oracle.
//!
//!   calib_guided <model_dir> <out.calib.hfq> [seq] [n_seq] [seed]
//!
//! Tokens are seeded-synthetic here (proves the pipeline + quantizer
//! consumption); real calibration text is the quality refinement.

use hipfire_runtime::calibration::CalibCollector;
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{free_model_acts, model_calib_down_backward, model_forward, LlamaModel};
use rdna_compute::Gpu;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: calib_guided <model_dir> <out.calib.hfq> [seq] [n_seq]");
    let out = args.next().expect("missing <out.calib.hfq>");
    let seq: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let n_seq: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1234);

    let mut gpu = Gpu::init().expect("Gpu::init");
    let (cfg, w) = load_llama_fp32(&mut gpu, Path::new(&dir))?;
    let vocab = cfg.vocab_size;
    // rank-1 LoRA (B=0 ⇒ zero contribution); the backward computes + discards
    // its grads — this path drives the weighted-Hessian capture, not training.
    let model = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, seq, 1, 1.0)?;
    let collector = CalibCollector::new();

    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
    let mut s = seed.max(1);
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as usize
    };

    let mut total = 0.0f32;
    for si in 0..n_seq {
        let toks: Vec<u32> = (0..seq).map(|_| (next() % vocab) as u32).collect();
        // CE targets = next token; the final position is ignored (-1).
        let mut targets = vec![0.0f32; seq];
        for t in 0..seq - 1 {
            targets[t] = toks[t + 1] as f32;
        }
        targets[seq - 1] = -1.0;
        let acts = model_forward(&mut gpu, &model, &toks, &pos)?;
        let loss = model_calib_down_backward(&mut gpu, &model, &acts, &targets, -1, &collector)?;
        free_model_acts(&mut gpu, acts)?;
        total += loss;
        eprintln!("seq {}/{}  ce/tok {:.3}", si + 1, n_seq, loss / seq as f32);
    }
    eprintln!(
        "mean ce/tok {:.4} over {} tensors",
        total / (n_seq * seq) as f32,
        collector.len()
    );

    let consistency = collector.write_streaming(&mut gpu, Path::new(&out), 0, "{}", &[])?;
    eprintln!("wrote {out}  (down_proj guided Hessians, diag-vs-H consistency {consistency:.2e})");
    Ok(())
}
