// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KLD A/B for ZAYA1: teacher-force a bf16 reference and one or two quantized
//! candidates over the same corpus tokens, reporting mean KL(P_ref ‖ P_cand)
//! per next-token position (full-vocab, fp64 reduction). Lower = closer to the
//! reference. All models are resident at once (one bf16 forward shared), so
//! `KLD(ref‖A)` and `KLD(ref‖B)` are measured on identical inputs.
//!
//! Run: cargo run --release -p hipfire-arch-zaya --example kld -- \
//!        <ref.bf16.hfq> <candA.hfq> [candB.hfq] --corpus <txt> [--ntok N]

use hipfire_arch_zaya::arch::ZayaModel;
use hipfire_arch_zaya::ZayaConfig;
use hipfire_model::tokenizer::Tokenizer;
use hipfire_runtime::arch::SimpleAr;
use hipfire_runtime::hfq::HfqFile;
use hipfire_rdna::Gpu;
use std::path::Path;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

/// fp64 log-softmax of a logit row.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&x| (x as f64 - m).exp()).sum();
    let logz = m + sum.ln();
    logits.iter().map(|&x| x as f64 - logz).collect()
}

/// KL(P ‖ Q) = Σ P_i (logP_i − logQ_i), inputs are log-probs.
fn kl_div(logp: &[f64], logq: &[f64]) -> f64 {
    logp.iter()
        .zip(logq)
        .map(|(&lp, &lq)| {
            let p = lp.exp();
            if p > 0.0 {
                p * (lp - lq)
            } else {
                0.0
            }
        })
        .sum()
}

struct Model {
    label: String,
    model: ZayaModel,
}

fn load(gpu: &mut Gpu, path: &str, max_seq: usize) -> Model {
    let hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).expect("metadata");
    let cfg = ZayaConfig::from_json(meta.get("config").unwrap_or(&meta)).expect("config");
    eprintln!("loading {path} ...");
    let model = ZayaModel::from_hfq(gpu, &hfq, cfg, max_seq).expect("load");
    Model {
        label: path.rsplit('/').next().unwrap_or(path).to_string(),
        model,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Positionals = args that are neither a `--flag` nor a flag's value.
    let mut positionals: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" | "--ntok" => i += 2,
            s if s.starts_with("--") => i += 1,
            s => {
                positionals.push(s.to_string());
                i += 1;
            }
        }
    }
    let ref_path = positionals.first().expect("ref hfq required").clone();
    let ref_path = ref_path.as_str();
    let cand_paths: Vec<&str> = positionals[1..].iter().map(|s| s.as_str()).collect();
    assert!(
        !cand_paths.is_empty(),
        "at least one candidate hfq required"
    );
    let corpus = flag(&args, "--corpus").expect("--corpus required");
    let ntok: usize = flag(&args, "--ntok")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    // Tokenize the corpus from the reference's embedded tokenizer.
    let ref_hfq = HfqFile::open(Path::new(ref_path)).expect("open ref");
    let tok = Tokenizer::from_hfq_metadata(&ref_hfq.metadata_json).expect("tokenizer");
    let raw = std::fs::read(&corpus).expect("read corpus");
    let take = (ntok * 8).min(raw.len());
    let text = String::from_utf8_lossy(&raw[..take]).to_string();
    let all = tok.encode(&text);
    let n = all.len().min(ntok);
    let ids = &all[..n];
    eprintln!("KLD over {n} positions; ref={ref_path}");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);
    let max_seq = n + 16;

    let mut reference = load(&mut gpu, ref_path, max_seq);
    let mut cands: Vec<Model> = cand_paths
        .iter()
        .map(|p| load(&mut gpu, p, max_seq))
        .collect();
    eprintln!("all models resident; teacher-forcing ...\n");

    // Prime every model on the first token.
    reference
        .model
        .prefill(&mut gpu, &ids[..1])
        .expect("ref prefill");
    for c in &mut cands {
        c.model.prefill(&mut gpu, &ids[..1]).expect("cand prefill");
    }

    let mut sum_kl = vec![0.0f64; cands.len()];
    let mut count = 0usize;
    for (i, &tok_i) in ids.iter().enumerate().skip(1) {
        // Logits now predict token i given the true prefix 0..i-1.
        let ref_lg = gpu
            .download_f32(reference.model.logits())
            .expect("ref logits");
        let ref_logp = log_softmax(&ref_lg);
        for (ci, c) in cands.iter().enumerate() {
            let c_lg = gpu.download_f32(c.model.logits()).expect("cand logits");
            let c_logp = log_softmax(&c_lg);
            sum_kl[ci] += kl_div(&ref_logp, &c_logp);
        }
        count += 1;
        // Feed the true token to every model.
        reference
            .model
            .decode_step(&mut gpu, tok_i, i)
            .expect("ref decode");
        for c in &mut cands {
            c.model
                .decode_step(&mut gpu, tok_i, i)
                .expect("cand decode");
        }
    }

    println!("\n=== KLD(ref ‖ candidate), mean over {count} positions ===");
    println!("ref: {}", reference.label);
    let results: Vec<(String, f64)> = cands
        .iter()
        .zip(&sum_kl)
        .map(|(c, &s)| (c.label.clone(), s / count.max(1) as f64))
        .collect();
    for (label, mean) in &results {
        println!("  {mean:.6}  {label}");
    }
    if results.len() == 2 {
        let delta = results[0].1 - results[1].1;
        let better = if delta > 0.0 {
            &results[1].0
        } else {
            &results[0].0
        };
        println!(
            "\nΔ = {:.6} ({:+.2}%); lower-KLD (better) = {better}",
            delta.abs(),
            100.0 * (results[0].1 - results[1].1) / results[1].1.max(1e-12)
        );
    }
}
