// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Perplexity / NLL eval for DeepSeek V4 Flash (ds4, arch_id=9).
//!
//! The qwen `perplexity` example hardcodes `qwen35::forward_scratch`; this is
//! the ds4 analog. ds4 manages its own KV/SWA/indexer/compressor cache inside
//! `DeepseekV4State`, so there is no external `KvCache` — we just loop
//! `decode_step`, which returns the host logits for each position.
//!
//! Usage:
//!   deepseek4_perplexity <model.hfq> <corpus.txt> \
//!       [--ctx 2048] [--warmup 8] [--offset 0] [--dump-logits <path>] \
//!       [--dump-dense-acts <directory>]
//!
//! Set `HIPFIRE_DEEPSEEK4_REAP_KEEPMAP=<dir>` to evaluate the REAP-pruned
//! (e.g. 162B / 144-expert) variant of the SAME quant — the loader keeps only
//! the mapped experts. Run twice (off / on) for a full-vs-pruned comparison.
//!
//! `--dump-logits <path>` writes per-scored-position full-vocab logits for an
//! offline KLD comparison between two runs (same corpus/offset/ctx/warmup):
//!   magic "DS4PPL01"(8) | vocab:u32 | n_scored:u32 |
//!   then n_scored × { pos:u32, target:u32, logits:[f32; vocab] }

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: deepseek4_perplexity <model> <corpus> [--ctx N] [--warmup N] [--offset N] [--dump-logits PATH]");
    let corpus_path = args
        .next()
        .expect("usage: deepseek4_perplexity <model> <corpus> [--ctx N] [--warmup N] [--offset N] [--dump-logits PATH]");

    let mut ctx_len: usize = 2048;
    let mut warmup: usize = 8;
    let mut offset: usize = 0;
    let mut dump_logits: Option<String> = None;
    let mut dump_dense_acts: Option<String> = None;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--ctx" => ctx_len = val.parse().unwrap(),
            "--warmup" => warmup = val.parse().unwrap(),
            "--offset" => offset = val.parse().unwrap(),
            "--dump-logits" => dump_logits = Some(val),
            "--dump-dense-acts" => dump_dense_acts = Some(val),
            _ => panic!("unknown flag: {flag}"),
        }
    }
    assert!(
        ctx_len > warmup + 4,
        "ctx must exceed warmup by enough to score"
    );
    if let Some(path) = dump_dense_acts.as_ref() {
        // Single-threaded process initialization, before the forward path's
        // OnceLock snapshots the calibration destination.
        unsafe {
            std::env::set_var("HIPFIRE_DS4_DENSE_ACT_DIR", path);
        }
        eprintln!("Dumping DeepSeek P1 dense activations to {path}");
    }

    let want_bytes = (offset + ctx_len) * 8;
    let raw = std::fs::read(&corpus_path).expect("read corpus");
    let take = want_bytes.min(raw.len());
    let corpus = String::from_utf8_lossy(&raw[..take]).to_string();
    eprintln!(
        "Corpus: {} bytes (of {}) from {corpus_path}",
        corpus.len(),
        raw.len()
    );

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let cfg = DeepseekV4::config_from_hfq(&hfq).expect("config");
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer from HFQ metadata");
    eprintln!(
        "Model arch_id={} n_layers={} n_routed_experts={} (keep-map {})",
        DeepseekV4::arch_id(),
        cfg.num_hidden_layers,
        cfg.n_routed_experts,
        if cfg.reap_keep.is_some() {
            "ACTIVE"
        } else {
            "off"
        },
    );

    eprintln!("Tokenizing...");
    let t_tok = Instant::now();
    let all_tokens: Vec<u32> = tokenizer.encode(&corpus);
    eprintln!(
        "Tokenized: {} tokens in {:.2}s",
        all_tokens.len(),
        t_tok.elapsed().as_secs_f64()
    );

    let end = (offset + ctx_len).min(all_tokens.len());
    if end <= offset + warmup + 4 {
        panic!("not enough tokens past offset={offset} for warmup={warmup} + scoring");
    }
    let window = &all_tokens[offset..end];
    eprintln!(
        "Window: offset={offset} ctx={} (warmup {warmup}, scoring {})",
        window.len(),
        window.len() - warmup - 1
    );

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch.clone());
    eprintln!("Loading weights from {model_path}...");
    let t_load = Instant::now();
    let mut state = DeepseekV4::new_state(&mut gpu, &cfg).expect("new_state");
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu).expect("load_weights");
    eprintln!("Loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    // Optional KLD logit dump.
    let mut dump = dump_logits.as_ref().map(|p| {
        let f = std::fs::File::create(p).expect("create dump-logits file");
        let w = std::io::BufWriter::new(f);
        eprintln!("Dumping full-vocab logits to {p}");
        w
    });
    let scored_total = (window.len() - 1).saturating_sub(warmup);
    if let Some(w) = dump.as_mut() {
        w.write_all(b"DS4PPL01").unwrap();
        w.write_all(&(0u32).to_le_bytes()).unwrap(); // vocab placeholder (patched at end)
        w.write_all(&(scored_total as u32).to_le_bytes()).unwrap();
    }

    let mut total_nll: f64 = 0.0;
    let mut scored: usize = 0;
    let mut vocab_seen: u32 = 0;
    let t0 = Instant::now();

    for (pos, &tok) in window.iter().enumerate().take(window.len() - 1) {
        let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos as u32)
            .expect("decode_step");
        if pos < warmup {
            continue;
        }
        let target = window[pos + 1] as usize;
        let nll = neg_log_softmax_at(&logits, target);
        if !nll.is_finite() {
            eprintln!("  warn: non-finite NLL at pos={pos} target={target}, skipping");
            continue;
        }
        total_nll += nll as f64;
        scored += 1;

        if let Some(w) = dump.as_mut() {
            vocab_seen = logits.len() as u32;
            w.write_all(&(pos as u32).to_le_bytes()).unwrap();
            w.write_all(&(target as u32).to_le_bytes()).unwrap();
            let bytes: &[u8] = bytemuck_cast(&logits);
            w.write_all(bytes).unwrap();
        }

        if scored == 1 || scored % 128 == 0 {
            let avg_nll = total_nll / scored as f64;
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = scored as f64 / elapsed.max(1e-9);
            eprintln!(
                "  pos={:5} scored={:5} nll/tok={:.4} ppl={:.3} ({:.2} tok/s)",
                pos,
                scored,
                avg_nll,
                avg_nll.exp(),
                rate,
            );
        }
    }

    if let Some(mut w) = dump {
        w.flush().unwrap();
        drop(w);
        // Patch the vocab field (offset 8) now that we know it.
        if let Some(p) = dump_logits.as_ref() {
            if let Ok(mut f) = std::fs::OpenOptions::new().write(true).open(p) {
                use std::io::Seek;
                f.seek(std::io::SeekFrom::Start(8)).unwrap();
                f.write_all(&vocab_seen.to_le_bytes()).unwrap();
            }
        }
    }
    hipfire_arch_deepseek4::forward::finish_dense_activation_dump()
        .expect("finalize DeepSeek P1 dense activation dump");

    let avg_nll = if scored > 0 {
        total_nll / scored as f64
    } else {
        0.0
    };
    let ppl = avg_nll.exp();
    let elapsed = t0.elapsed().as_secs_f64();
    println!();
    println!("Model:    {model_path}");
    println!("Corpus:   {corpus_path}");
    println!(
        "Variant:  {}",
        if cfg.reap_keep.is_some() {
            format!("REAP keep-map ({} experts/layer)", cfg.n_routed_experts)
        } else {
            format!("full ({} experts/layer)", cfg.n_routed_experts)
        }
    );
    println!(
        "Tokens:   offset={offset} ctx={} warmup={warmup}",
        window.len()
    );
    println!("Scored:   {scored}");
    println!("NLL/tok:  {avg_nll:.10}");
    println!("PPL:      {ppl:.4}");
    println!(
        "Elapsed:  {:.1}s ({:.2} tok/s)",
        elapsed,
        scored as f64 / elapsed.max(1e-9)
    );
}

fn neg_log_softmax_at(logits: &[f32], target: usize) -> f32 {
    if target >= logits.len() {
        return f32::NAN;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v - max) as f64).exp();
    }
    let log_sum = max as f64 + sum.ln();
    (log_sum - logits[target] as f64) as f32
}

/// Reinterpret &[f32] as &[u8] (little-endian host) for the logit dump.
fn bytemuck_cast(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; we only read it as bytes for a binary dump
    // consumed on the same host (little-endian).
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
