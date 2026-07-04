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
//!   calib_guided <model_dir> <out.calib.hfq> [seq] [n_seq] [--text <file>]
//!
//! With `--text`, the model's `tokenizer.json` encodes the file into real
//! calibration sequences (meaningful Fisher weights). Without it, tokens are
//! seeded-synthetic (proves the pipeline only).

use hipfire_model::tokenizer::Tokenizer;
use hipfire_runtime::calibration::CalibCollector;
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{free_model_acts, model_calib_down_backward, model_forward, LlamaModel};
use hipfire_rdna::Gpu;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Positional: <model_dir> <out.calib.hfq> [seq] [n_seq]; flag: --text <file>.
    let mut text_path: Option<String> = None;
    let mut plain = false; // --plain ⇒ w≡1 baseline (plain XᵀX over the same tokens)
    let mut skip_seq = 0usize; // --skip N ⇒ drop the first N sequences (held-out split)
    let mut pos_args: Vec<String> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--text" => text_path = it.next(),
            "--plain" => plain = true,
            "--skip" => skip_seq = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            _ => pos_args.push(a),
        }
    }
    let fisher = !plain;
    let dir = pos_args
        .first()
        .cloned()
        .expect("usage: calib_guided <model_dir> <out.calib.hfq> [seq] [n_seq] [--text <file>]");
    let out = pos_args.get(1).cloned().expect("missing <out.calib.hfq>");
    let seq: usize = pos_args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let n_seq: usize = pos_args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    let mut gpu = Gpu::init().expect("Gpu::init");
    let (cfg, w) = load_llama_fp32(&mut gpu, Path::new(&dir))?;
    let vocab = cfg.vocab_size;
    // rank-1 LoRA (B=0 ⇒ zero contribution); the backward computes + discards
    // its grads — this path drives the weighted-Hessian capture, not training.
    let model = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, seq, 1, 1.0)?;
    let collector = CalibCollector::new();

    // Build calibration sequences.
    let sequences: Vec<Vec<u32>> = if let Some(tp) = &text_path {
        let tok = Tokenizer::from_tokenizer_json(&Path::new(&dir).join("tokenizer.json"))?
            .ok_or("no tokenizer.json in model dir")?;
        let text = std::fs::read_to_string(tp)?;
        let ids = tok.encode(&text);
        eprintln!("tokenized {} chars -> {} tokens", text.len(), ids.len());
        ids.chunks(seq)
            .filter(|c| c.len() == seq)
            .skip(skip_seq)
            .take(n_seq)
            .map(|c| c.to_vec())
            .collect()
    } else {
        let mut s = 0x1234_5678u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as usize
        };
        (0..n_seq)
            .map(|_| (0..seq).map(|_| (next() % vocab) as u32).collect())
            .collect()
    };
    if sequences.is_empty() {
        return Err("no calibration sequences (text too short for one seq?)".into());
    }
    eprintln!(
        "calibrating on {} sequence(s) × {} tokens ({}, {})",
        sequences.len(),
        seq,
        if text_path.is_some() {
            "real text"
        } else {
            "synthetic"
        },
        if fisher {
            "Fisher-weighted"
        } else {
            "plain XᵀX control"
        }
    );

    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
    let mut total = 0.0f32;
    for (si, toks) in sequences.iter().enumerate() {
        // CE targets = next token; the final position is ignored (-1).
        let mut targets = vec![0.0f32; seq];
        for t in 0..seq - 1 {
            targets[t] = toks[t + 1] as f32;
        }
        targets[seq - 1] = -1.0;
        let acts = model_forward(&mut gpu, &model, toks, &pos)?;
        let loss =
            model_calib_down_backward(&mut gpu, &model, &acts, &targets, -1, &collector, fisher)?;
        free_model_acts(&mut gpu, acts)?;
        total += loss;
        eprintln!(
            "seq {}/{}  ce/tok {:.3}",
            si + 1,
            sequences.len(),
            loss / seq as f32
        );
    }
    eprintln!(
        "mean ce/tok {:.4} over {} tensors",
        total / (sequences.len() * seq) as f32,
        collector.len()
    );

    let consistency = collector.write_streaming(&mut gpu, Path::new(&out), 0, "{}", &[])?;
    eprintln!("wrote {out}  (down_proj guided Hessians, diag-vs-H consistency {consistency:.2e})");
    Ok(())
}
