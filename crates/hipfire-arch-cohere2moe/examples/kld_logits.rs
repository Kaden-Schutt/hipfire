// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! kld_logits (cohere2moe) — per-position next-token logits for a fixed token
//! list, for the BF16/Q8/MQ6/MQ4 KLD + PPL sweep of North-Mini-Code-1.0.
//!
//! Two modes:
//!   * `--dump <out.bin>`  : run ONE model over the token list (per-token
//!     prefill via `decode_step`) and write its per-position logits to a binary
//!     (u32 n_pos, u32 vocab, then n_pos*vocab f32 LE). Run this once per tier;
//!     KL(oracle‖tier) and wikitext PPL are then computed offline from the
//!     dumps + the token list (each 61 GB model loads only once).
//!   * `--model-a <ref> --model-b <cand>` : load both and print the per-position
//!     KL(softmax(ref) || softmax(cand)) distribution directly.
//!
//! Usage:
//!   kld_logits --model-a <model.hfq> --dump <out.bin> --tokens <tokens.json> [--max N]
//!   kld_logits --model-a <ref.hfq> --model-b <cand.hfq> --tokens <tokens.json> [--max N]
//!
//!   --tokens : JSON array of u32 token ids, e.g. [504, 2849, 8868, ...]

use hipfire_arch_cohere2moe::cohere2moe::{Cohere2MoeState, Cohere2MoeWeights};
use hipfire_arch_cohere2moe::config::Cohere2MoeConfig;
use hipfire_arch_cohere2moe::forward::decode_step;
use hipfire_runtime::hfq::HfqFile;
use std::fs;
use std::io::Write;
use std::path::Path;

struct Args {
    model_a: String,
    model_b: Option<String>,
    tokens: String,
    max: usize,
    dump: Option<String>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_a = None;
    let mut model_b = None;
    let mut tokens = None;
    let mut max = usize::MAX;
    let mut dump = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model-a" => {
                model_a = Some(argv[i + 1].clone());
                i += 2;
            }
            "--model-b" => {
                model_b = Some(argv[i + 1].clone());
                i += 2;
            }
            "--tokens" => {
                tokens = Some(argv[i + 1].clone());
                i += 2;
            }
            "--max" => {
                max = argv[i + 1].parse().expect("--max");
                i += 2;
            }
            "--dump" => {
                dump = Some(argv[i + 1].clone());
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    Args {
        model_a: model_a.expect("--model-a required"),
        model_b,
        tokens: tokens.expect("--tokens required"),
        max,
        dump,
    }
}

fn load_tokens(path: &str) -> Vec<u32> {
    let raw = fs::read_to_string(path).expect("read tokens json");
    serde_json::from_str(&raw).expect("parse tokens json (expected JSON array of u32)")
}

/// Run `path` over the token list, returning the per-position next-token logits.
fn run_model(path: &str, args: &Args) -> Vec<Vec<f32>> {
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(Path::new(path)).expect("open model");
    assert_eq!(
        hfq.arch_id, 12,
        "kld_logits(cohere2moe): expected arch_id 12, got {}",
        hfq.arch_id
    );
    let cfg = Cohere2MoeConfig::from_hfq(&hfq).expect("config");
    eprintln!(
        "[{}] cohere2moe hidden={} layers={} experts={}/{} dense_prefix={} vocab={}",
        path,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.first_k_dense_replace,
        cfg.vocab_size,
    );
    let weights = Cohere2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");

    let tokens = load_tokens(&args.tokens);
    let n = tokens.len().min(args.max);
    let max_seq = n + 16;
    let mut state = Cohere2MoeState::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state");

    let mut all = Vec::with_capacity(n);
    for (pos, &tok) in tokens.iter().take(n).enumerate() {
        let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos as u32)
            .expect("decode_step");
        all.push(logits);
        if pos % 64 == 0 {
            eprintln!("  pos {pos}/{n}");
        }
    }
    all
}

/// Per-position logits → binary: u32 n_pos, u32 vocab, then n_pos*vocab f32 LE.
fn dump_logits(path: &str, logits: &[Vec<f32>]) {
    let n = logits.len() as u32;
    let vocab = logits.first().map(|r| r.len()).unwrap_or(0) as u32;
    let mut f = std::io::BufWriter::new(fs::File::create(path).expect("create dump"));
    f.write_all(&n.to_le_bytes()).unwrap();
    f.write_all(&vocab.to_le_bytes()).unwrap();
    for row in logits {
        for &v in row {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    eprintln!("dumped {n} positions × {vocab} vocab → {path}");
}

/// KL(softmax(ref) || softmax(cand)) in nats (max-subtracted for stability).
fn compute_kl(r: &[f32], c: &[f32]) -> f64 {
    assert_eq!(r.len(), c.len());
    let rmax = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let cmax = c.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let (mut rs, mut cs) = (0.0f64, 0.0f64);
    for i in 0..r.len() {
        rs += ((r[i] as f64) - rmax).exp();
        cs += ((c[i] as f64) - cmax).exp();
    }
    let (lrs, lcs) = (rs.ln() + rmax, cs.ln() + cmax);
    let mut kl = 0.0f64;
    for i in 0..r.len() {
        let lp = (r[i] as f64) - lrs;
        let p = lp.exp();
        if p > 0.0 {
            kl += p * (lp - ((c[i] as f64) - lcs));
        }
    }
    if kl < 0.0 && kl > -1e-9 {
        kl = 0.0;
    }
    kl
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_summary(kls: &[f64]) {
    let n = kls.len();
    if n == 0 {
        eprintln!("no positions");
        return;
    }
    let mut s = kls.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = kls.iter().sum::<f64>() / n as f64;
    println!("=== KL(ref || cand), nats ===");
    println!("positions : {n}");
    println!("mean      : {mean:.6}");
    println!("median    : {:.6}", percentile(&s, 0.50));
    println!("p99       : {:.6}", percentile(&s, 0.99));
    println!("max       : {:.6}", s.last().unwrap());
    println!(
        "frac>0.1  : {:.4}",
        kls.iter().filter(|&&k| k > 0.1).count() as f64 / n as f64
    );
}

fn main() {
    let args = parse_args();
    if let Some(ref dump_path) = args.dump {
        eprintln!("=== dump model: {} ===", args.model_a);
        let logits = run_model(&args.model_a, &args);
        dump_logits(dump_path, &logits);
        return;
    }
    eprintln!("=== model-a (reference): {} ===", args.model_a);
    let r = run_model(&args.model_a, &args);
    let mb = args
        .model_b
        .clone()
        .expect("--model-b required (or use --dump)");
    eprintln!("=== model-b (candidate): {mb} ===");
    let c = run_model(&mb, &args);
    let n = r.len().min(c.len());
    let kls: Vec<f64> = (0..n).map(|i| compute_kl(&r[i], &c[i])).collect();
    print_summary(&kls);
}
