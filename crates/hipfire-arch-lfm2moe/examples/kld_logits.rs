// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! kld_logits — per-position KL-divergence between two models' next-token
//! logits over a fixed token-id list.
//!
//! For each position i in the token list, both model-a (reference) and
//! model-b (candidate) are fed tokens[0..=i] (per-token prefill via
//! `decode_step`), and the next-token logits at position i are captured.
//! We then compute KL(softmax(ref_i) || softmax(cand_i)) per position and
//! report the distribution (mean / median / p99 / max / frac>0.1).
//!
//! Currently wired for arch_id 11 (LFM2.5-MoE). The `run_model` dispatch
//! reads arch_id from the HFQ metadata; other arches panic with a clear
//! message (extend the match to add them).
//!
//! Usage:
//!   kld_logits --model-a <ref.hfq> --model-b <cand.hfq> \
//!              --tokens <tokens.json> [--max N]
//!
//!   --tokens : JSON array of u32 token ids, e.g. [504, 2849, 8868, ...]
//!   --max    : cap on the number of positions evaluated (default: all)

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
use std::fs;

#[cfg(feature = "deltanet")]
struct Args {
    model_a: String,
    model_b: Option<String>,
    tokens: String,
    max: usize,
    dump: Option<String>,
}

#[cfg(feature = "deltanet")]
fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_a: Option<String> = None;
    let mut model_b: Option<String> = None;
    let mut tokens: Option<String> = None;
    let mut max: usize = usize::MAX;
    let mut dump: Option<String> = None;
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
            // --dump <file>: run model-a only, write its per-position logits to a
            // binary (u32 n_pos, u32 vocab, then n_pos*vocab f32 LE), and exit.
            // Used to capture each variant's logits for offline KL vs a bf16
            // reference (the true ground truth — Q8 has its own quant error so it
            // must NOT be the reference).
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

/// Write per-position logits as: u32 n_pos, u32 vocab, then n_pos*vocab f32 LE.
#[cfg(feature = "deltanet")]
fn dump_logits(path: &str, logits: &[Vec<f32>]) {
    use std::io::Write;
    let n = logits.len() as u32;
    let vocab = if logits.is_empty() {
        0
    } else {
        logits[0].len()
    } as u32;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("create dump"));
    f.write_all(&n.to_le_bytes()).unwrap();
    f.write_all(&vocab.to_le_bytes()).unwrap();
    for row in logits {
        for &v in row {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    eprintln!("dumped {n} positions × {vocab} vocab → {path}");
}

#[cfg(feature = "deltanet")]
fn detect_arch_id(path: &str) -> Result<u32, String> {
    use hipfire_runtime::hfq::HfqFile;
    use std::path::Path;
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("open {path}: {e}"))?;
    Ok(hfq.arch_id)
}

#[cfg(feature = "deltanet")]
fn load_tokens(path: &str) -> Vec<u32> {
    let raw = fs::read_to_string(path).expect("read tokens json");
    serde_json::from_str(&raw).expect("parse tokens json (expected JSON array of u32)")
}

/// Run `path` over the token list, returning the per-position next-token
/// logits vector (one Vec<f32> per evaluated position).
#[cfg(feature = "deltanet")]
fn run_model(path: &str, args: &Args) -> Vec<Vec<f32>> {
    let arch_id = detect_arch_id(path).expect("read arch_id");
    match arch_id {
        11 => run_lfm2moe(path, args),
        other => panic!("kld_logits: unsupported arch_id {other} (only 11/lfm2moe wired)"),
    }
}

#[cfg(feature = "deltanet")]
fn run_lfm2moe(path: &str, args: &Args) -> Vec<Vec<f32>> {
    use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
    use hipfire_arch_lfm2moe::forward::decode_step;
    use hipfire_arch_lfm2moe::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_runtime::hfq::HfqFile;
    use std::path::Path;

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(Path::new(path)).expect("open model");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    eprintln!(
        "[{}] lfm2moe hidden={} layers={} experts={}/{} vocab={}",
        path,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.vocab_size
    );
    let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");

    let tokens = load_tokens(&args.tokens);
    let n = tokens.len().min(args.max);
    let max_seq = n + 16;
    let mut state = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state");

    let mut all_logits = Vec::with_capacity(n);
    for (pos, &tok) in tokens.iter().take(n).enumerate() {
        let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos as u32)
            .expect("decode_step");
        all_logits.push(logits);
    }
    all_logits
}

/// KL(softmax(ref) || softmax(cand)) in nats, computed in a numerically
/// stable way (subtract per-vector max before exp).
#[cfg(feature = "deltanet")]
fn compute_kl(ref_logits: &[f32], cand_logits: &[f32]) -> f64 {
    assert_eq!(
        ref_logits.len(),
        cand_logits.len(),
        "logit vector length mismatch"
    );

    let ref_max = ref_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let cand_max = cand_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max) as f64;

    let mut ref_sum = 0.0f64;
    let mut cand_sum = 0.0f64;
    for i in 0..ref_logits.len() {
        ref_sum += ((ref_logits[i] as f64) - ref_max).exp();
        cand_sum += ((cand_logits[i] as f64) - cand_max).exp();
    }
    let log_ref_sum = ref_sum.ln() + ref_max;
    let log_cand_sum = cand_sum.ln() + cand_max;

    // KL = sum_i p_i * (log p_i - log q_i)
    //    = sum_i p_i * ((ref_i - log_ref_sum) - (cand_i - log_cand_sum))
    let mut kl = 0.0f64;
    for i in 0..ref_logits.len() {
        let log_p = (ref_logits[i] as f64) - log_ref_sum;
        let p = log_p.exp();
        if p <= 0.0 {
            continue;
        }
        let log_q = (cand_logits[i] as f64) - log_cand_sum;
        kl += p * (log_p - log_q);
    }
    // Guard tiny negative values from FP rounding.
    if kl < 0.0 && kl > -1e-9 {
        kl = 0.0;
    }
    kl
}

#[cfg(feature = "deltanet")]
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(feature = "deltanet")]
fn print_summary(kls: &[f64]) {
    let n = kls.len();
    if n == 0 {
        eprintln!("no positions evaluated");
        return;
    }
    let mut sorted = kls.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = kls.iter().sum::<f64>() / n as f64;
    let median = percentile(&sorted, 0.50);
    let p99 = percentile(&sorted, 0.99);
    let max = *sorted.last().unwrap();
    let frac_gt_0_1 = kls.iter().filter(|&&k| k > 0.1).count() as f64 / n as f64;

    println!("=== KL-divergence summary (ref || cand), nats ===");
    println!("positions   : {n}");
    println!("mean        : {mean:.6}");
    println!("median      : {median:.6}");
    println!("p99         : {p99:.6}");
    println!("max         : {max:.6}");
    println!("frac > 0.1  : {frac_gt_0_1:.4}");
}

#[cfg(feature = "deltanet")]
fn main() {
    let args = parse_args();

    // --dump mode: run model-a only, write its logits, exit (no model-b / KL).
    if let Some(ref dump_path) = args.dump {
        eprintln!("=== dump model: {} ===", args.model_a);
        let logits = run_model(&args.model_a, &args);
        dump_logits(dump_path, &logits);
        return;
    }

    eprintln!("=== model-a (reference): {} ===", args.model_a);
    let ref_logits = run_model(&args.model_a, &args);
    let model_b = args
        .model_b
        .clone()
        .expect("--model-b required (or use --dump)");
    eprintln!("=== model-b (candidate): {} ===", model_b);
    let cand_logits = run_model(&model_b, &args);

    let n = ref_logits.len().min(cand_logits.len());
    let mut kls = Vec::with_capacity(n);
    for i in 0..n {
        kls.push(compute_kl(&ref_logits[i], &cand_logits[i]));
    }

    print_summary(&kls);
}
