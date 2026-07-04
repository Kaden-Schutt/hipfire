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

//! PFlash M0 feasibility probe (hipfire-native, Supra-50M).
//!
//! Core question: does a SHALLOW layer's K-cosine block ranking track a DEEPER
//! layer's? PFlash scores importance = cosine(block_mean_K, last_token_K) at the
//! *shallowest* full-attn layer, and the drafter (or a "first-few-layers"
//! teacher) is exactly that cheap shallow computation. If shallow-K ranking
//! correlates with deep-K ranking, the cheap signal carries the importance →
//! the drafter concept is viable. If not, the premise is shaky.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "pflash-m0"
//!   cargo run -p hipfire-train --release --example pflash_m0_probe
//!   hipfire gpu-lock release

#![allow(clippy::needless_range_loop)]

use hipfire_model::tokenizer::Tokenizer;
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_block_activations, LlamaModel};
use hipfire_rdna::Gpu;
use std::path::Path;

const MODEL_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 1024; // Supra max_position
const BLOCK: usize = 64;
const SHALLOW: usize = 1;
const DEEP: usize = 8;

fn long_text() -> String {
    let mut files: Vec<_> = glob_md("docs");
    files.sort();
    let mut s = String::new();
    for f in files.iter().take(20) {
        if let Ok(t) = std::fs::read_to_string(f) {
            s.push_str(&t);
            s.push_str("\n\n");
        }
    }
    s
}

fn glob_md(dir: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(dir)];
    while let Some(d) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "md") {
                    out.push(p.to_string_lossy().into_owned());
                }
            }
        }
    }
    out
}

/// Per-block importance = head-averaged cosine(block_mean_K, last_token_K).
/// k: flat [seq * n_kv * hd]. Returns n_blocks scores.
fn block_scores(k: &[f32], seq: usize, n_kv: usize, hd: usize, block: usize) -> Vec<f32> {
    let kvd = n_kv * hd;
    let nb = seq / block;
    let last = &k[(seq - 1) * kvd..seq * kvd]; // [n_kv*hd]
    let cos = |a: &[f32], b: &[f32]| {
        let (mut d, mut na, mut nb_) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..a.len() {
            d += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb_ += (b[i] as f64).powi(2);
        }
        if na == 0.0 || nb_ == 0.0 {
            0.0
        } else {
            d / (na.sqrt() * nb_.sqrt())
        }
    };
    let mut scores = vec![0.0f32; nb];
    for b in 0..nb {
        let mut mean = vec![0.0f32; kvd];
        for t in b * block..(b + 1) * block {
            for j in 0..kvd {
                mean[j] += k[t * kvd + j];
            }
        }
        for v in mean.iter_mut() {
            *v /= block as f32;
        }
        // per kv-head cosine, averaged
        let mut s = 0.0f64;
        for h in 0..n_kv {
            s += cos(&mean[h * hd..(h + 1) * hd], &last[h * hd..(h + 1) * hd]);
        }
        scores[b] = (s / n_kv as f64) as f32;
    }
    scores
}

fn rank(a: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..a.len()).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap());
    let mut r = vec![0.0f32; a.len()];
    for (pos, &i) in idx.iter().enumerate() {
        r[i] = pos as f32;
    }
    r
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().sum::<f32>() as f64 / n,
        b.iter().sum::<f32>() as f64 / n,
    );
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
        c += da * db;
        va += da * da;
        vb += db * db;
    }
    if va == 0.0 || vb == 0.0 {
        0.0
    } else {
        (c / (va.sqrt() * vb.sqrt())) as f32
    }
}

fn topk_recall(small: &[f32], big: &[f32], frac: f32) -> f32 {
    let k = ((big.len() as f32 * frac).round() as usize).max(1);
    let top = |a: &[f32]| {
        let mut idx: Vec<usize> = (0..a.len()).collect();
        idx.sort_by(|&i, &j| a[j].partial_cmp(&a[i]).unwrap());
        idx.into_iter()
            .take(k)
            .collect::<std::collections::HashSet<_>>()
    };
    let (s, g) = (top(small), top(big));
    s.intersection(&g).count() as f32 / k as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(MODEL_DIR);
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let tok = Tokenizer::from_hf_json(&std::fs::read_to_string(dir.join("tokenizer.json"))?)
        .map_err(|e| format!("tokenizer: {e:?}"))?;
    let mut ids = tok.encode(&long_text());
    ids.truncate(SEQ);
    if ids.len() < SEQ {
        return Err(format!("only {} tokens; need {SEQ}", ids.len()).into());
    }
    println!("input: {SEQ} tokens → {} blocks of {BLOCK}", SEQ / BLOCK);

    let (cfg, w) = load_llama_fp32(&mut gpu, dir)?;
    let (n_kv, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    let model = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, SEQ, 4, 8.0)?;
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();

    println!("partial forward to layer {DEEP} ...");
    let acts = model_block_activations(&mut gpu, &model, &ids, &pos, DEEP)?;
    let k_shallow = gpu.download_f32(&acts[SHALLOW].k_r)?;
    let k_deep = gpu.download_f32(&acts[DEEP].k_r)?;

    println!("\n── M0: shallow (L{SHALLOW}) vs deep (L{DEEP}) K-cosine block ranking ──");
    println!(
        "  recency baseline = block index (recent blocks trivially resemble last token).\n  \
         shallow is only a real signal if it beats recency at tracking deep.\n"
    );
    for &block in &[16usize, 32, 64] {
        let nb = SEQ / block;
        let s_shallow = block_scores(&k_shallow, SEQ, n_kv, hd, block);
        let s_deep = block_scores(&k_deep, SEQ, n_kv, hd, block);
        // recency score: higher block index = more recent.
        let s_recency: Vec<f32> = (0..nb).map(|b| b as f32).collect();

        let r_sd = pearson(&rank(&s_shallow), &rank(&s_deep));
        let r_rd = pearson(&rank(&s_recency), &rank(&s_deep));
        let r_rs = pearson(&rank(&s_recency), &rank(&s_shallow));
        println!("  block={block:>3} ({nb:>2} blocks):");
        println!("    shallow↔deep    Spearman {r_sd:+.3}");
        println!("    recency↔deep    Spearman {r_rd:+.3}   (confound: how much is just recency)");
        println!("    recency↔shallow Spearman {r_rs:+.3}");
        for frac in [0.10f32, 0.25] {
            let r = topk_recall(&s_shallow, &s_deep, frac);
            println!(
                "    top-{:>2}% recall  {r:.2}  (random ≈ {frac:.2})",
                (frac * 100.0) as i32
            );
        }
    }
    println!("\nInterpret: shallow↔deep ≫ recency↔deep → shallow K carries a real importance");
    println!("signal beyond recency → first-few-layers teacher / tiny drafter is viable.");
    println!(
        "If shallow↔deep ≈ recency↔deep → the ranking is mostly recency; a drafter buys little."
    );
    Ok(())
}
