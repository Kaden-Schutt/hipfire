// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Offline KLD + PPL over per-position logit dumps (no numpy). Reads the
//! `kld_logits --dump` binaries (u32 n_pos, u32 vocab, then n_pos*vocab f32 LE)
//! + the token-id list, and prints, per tier: KL(ref || tier) over next-token
//! distributions (mean/median/p99) and wikitext PPL = exp(mean NLL of the true
//! next token). First --dump (or --ref) is the KLD reference (the bf16 oracle).
//!
//! Usage: analyze_dumps --tokens t.json --ref bf16 \
//!          --dump bf16=bf16.logits --dump q8=q8.logits ...

use std::collections::HashMap;
use std::fs;

fn load_dump(path: &str) -> (usize, usize, Vec<f32>) {
    let b = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let n = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let v = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut data = Vec::with_capacity(n * v);
    for c in b[8..8 + n * v * 4].chunks_exact(4) {
        data.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    (n, v, data)
}

fn logsumexp(row: &[f32]) -> f64 {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = row.iter().map(|&x| ((x as f64) - m).exp()).sum();
    s.ln() + m
}

/// KL(softmax(r) || softmax(c)) in nats.
fn kl(r: &[f32], c: &[f32]) -> f64 {
    let lr = logsumexp(r);
    let lc = logsumexp(c);
    let mut kl = 0.0f64;
    for i in 0..r.len() {
        let lp = (r[i] as f64) - lr;
        let p = lp.exp();
        if p > 0.0 {
            kl += p * (lp - ((c[i] as f64) - lc));
        }
    }
    kl.max(0.0)
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[(((sorted.len() - 1) as f64) * q).round() as usize]
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut tokens_path = String::new();
    let mut refn: Option<String> = None;
    let mut dumps: Vec<(String, String)> = Vec::new();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--tokens" => {
                tokens_path = argv[i + 1].clone();
                i += 2;
            }
            "--ref" => {
                refn = Some(argv[i + 1].clone());
                i += 2;
            }
            "--dump" => {
                let (n, p) = argv[i + 1].split_once('=').expect("name=path");
                dumps.push((n.to_string(), p.to_string()));
                i += 2;
            }
            o => {
                eprintln!("unknown arg {o}");
                std::process::exit(1);
            }
        }
    }
    let toks: Vec<usize> =
        serde_json::from_str::<Vec<i64>>(&fs::read_to_string(&tokens_path).expect("read tokens"))
            .expect("parse tokens")
            .into_iter()
            .map(|t| t as usize)
            .collect();

    let loaded: HashMap<String, (usize, usize, Vec<f32>)> = dumps
        .iter()
        .map(|(n, p)| (n.clone(), load_dump(p)))
        .collect();
    let ref_name = refn.unwrap_or_else(|| dumps[0].0.clone());
    let (n_pos, vocab, ref_d) = &loaded[&ref_name];
    let n_pos = *n_pos;
    let vocab = *vocab;
    eprintln!(
        "ref={ref_name} n_pos={n_pos} vocab={vocab} tokens={}",
        toks.len()
    );

    println!(
        "{:>6} | {:>9} {:>9} {:>9} | {:>9}",
        "tier", "KL.mean", "KL.med", "KL.p99", "PPL"
    );
    println!("{}", "-".repeat(56));
    for (name, _) in &dumps {
        let (_, _, d) = &loaded[name];
        // KL(ref || tier) per position.
        let mut kls: Vec<f64> = (0..n_pos)
            .map(|p| {
                kl(
                    &ref_d[p * vocab..(p + 1) * vocab],
                    &d[p * vocab..(p + 1) * vocab],
                )
            })
            .collect();
        let mean = kls.iter().sum::<f64>() / kls.len() as f64;
        kls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // PPL: NLL of the true next token. pos i predicts token[i+1].
        let ppl_n = (n_pos - 1).min(toks.len() - 1);
        let mut nll = 0.0f64;
        for p in 0..ppl_n {
            let row = &d[p * vocab..(p + 1) * vocab];
            nll += logsumexp(row) - row[toks[p + 1]] as f64;
        }
        let ppl = (nll / ppl_n as f64).exp();
        println!(
            "{:>6} | {:>9.5} {:>9.5} {:>9.5} | {:>9.3}",
            name,
            mean,
            pct(&kls, 0.5),
            pct(&kls, 0.99),
            ppl
        );
    }
}
