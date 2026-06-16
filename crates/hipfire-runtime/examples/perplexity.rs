// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Perplexity / NLL eval on a text corpus (single-window).
//!
//! Usage:
//!   perplexity <model.hfq> <corpus.txt> [--ctx 2048] [--warmup 8] [--offset 0]
//!
//! Tokenizes the corpus, takes a slice [offset, offset+ctx), prefills it
//! position-by-position, and scores -log_softmax(logits)[next_token]
//! for positions in [warmup, ctx-1). Reports total NLL, NLL/token, ppl.
//!
//! For comparing quants: same model class, same corpus, same offset/ctx/warmup.
//! 2K tokens is enough to see sub-4-bit deltas (single decimal of ppl);
//! 8K+ if you want stable second-decimal numbers.

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;
use std::path::Path;
use std::time::Instant;

/// KVarN tile size (tokens per Sinkhorn flush), matching the KVarN spec GROUP.
const KVARN_GROUP: usize = 128;

/// In-place KVarN degrade of a GROUP of K tokens (CPU-flush sim, v1). For each
/// kv_head, form the [head_dim × group_len] tile (channels × tokens), Sinkhorn
/// variance-normalize + per-channel 4-bit + dequant, write back. This is the
/// numerical reference the GPU Sinkhorn+pack kernel will mirror; inlined here
/// (kvarn.rs lives in the quantize bin crate, not importable) to get the KVarN-K
/// PPL verdict. K cache layout: flat [max_seq × (n_kv_heads*head_dim)] f32.
fn degrade_k_group_kvarn(
    gpu: &mut Gpu,
    kv: &KvCache,
    start: usize,
    len: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let kv_dim = n_kv_heads * head_dim;
    for layer in 0..kv.k_gpu.len() {
        let mut k = gpu.download_f32(&kv.k_gpu[layer]).unwrap();
        for h in 0..n_kv_heads {
            // tile[d, j] = K[start+j, h, d]  → [head_dim × len]
            let mut tile = vec![0.0f32; head_dim * len];
            for j in 0..len {
                let base = (start + j) * kv_dim + h * head_dim;
                for d in 0..head_dim {
                    tile[d * len + j] = k[base + d];
                }
            }
            let deq = kvarn_degrade_tile(&tile, head_dim, len);
            for j in 0..len {
                let base = (start + j) * kv_dim + h * head_dim;
                for d in 0..head_dim {
                    k[base + d] = deq[d * len + j];
                }
            }
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len() * 4) };
        gpu.memcpy_htod_auto(&kv.k_gpu[layer].buf, bytes).unwrap();
    }
}

/// Variance-normalize (log-domain Sinkhorn, best-so-far) + per-channel 4-bit +
/// dequant a [r_dim × c_dim] tile. Mirrors the tested `kvarn.rs` reference.
fn kvarn_degrade_tile(tile: &[f32], r_dim: usize, c_dim: usize) -> Vec<f32> {
    let imbalance = |m: &[f32]| -> f64 {
        let (mut cmin, mut cmax, mut rmin, mut rmax) =
            (f64::INFINITY, 0.0f64, f64::INFINITY, 0.0f64);
        for c in 0..c_dim {
            let (mut s, mut sq) = (0.0f64, 0.0f64);
            for r in 0..r_dim {
                let v = m[r * c_dim + c] as f64;
                s += v;
                sq += v * v;
            }
            let n = r_dim as f64;
            let std = (sq / n - (s / n) * (s / n)).max(0.0).sqrt();
            cmin = cmin.min(std);
            cmax = cmax.max(std);
        }
        for r in 0..r_dim {
            let (mut s, mut sq) = (0.0f64, 0.0f64);
            for c in 0..c_dim {
                let v = m[r * c_dim + c] as f64;
                s += v;
                sq += v * v;
            }
            let n = c_dim as f64;
            let std = (sq / n - (s / n) * (s / n)).max(0.0).sqrt();
            rmin = rmin.min(std);
            rmax = rmax.max(std);
        }
        cmax / cmin.max(1e-8) + rmax / rmin.max(1e-8)
    };
    let mut lc = vec![0.0f64; c_dim];
    let mut lr = vec![0.0f64; r_dim];
    let cur = |lc: &[f64], lr: &[f64]| -> Vec<f32> {
        let mut out = vec![0.0f32; r_dim * c_dim];
        for r in 0..r_dim {
            let er = (-lr[r]).exp();
            for c in 0..c_dim {
                out[r * c_dim + c] = (tile[r * c_dim + c] as f64 * er * (-lc[c]).exp()) as f32;
            }
        }
        out
    };
    let clamp = |x: f64| x.clamp(-0.3, 10.0);
    let mut best = cur(&lc, &lr);
    let mut best_imb = imbalance(&best);
    let (mut blc, mut blr) = (lc.clone(), lr.clone());
    for _ in 0..16 {
        let m = cur(&lc, &lr);
        for c in 0..c_dim {
            let (mut s, mut sq) = (0.0f64, 0.0f64);
            for r in 0..r_dim {
                let v = m[r * c_dim + c] as f64;
                s += v;
                sq += v * v;
            }
            let n = r_dim as f64;
            let std = (sq / n - (s / n) * (s / n)).max(0.0).sqrt().clamp(1e-3, 1e3);
            lc[c] = clamp(lc[c] + std.ln());
        }
        let m = cur(&lc, &lr);
        for r in 0..r_dim {
            let (mut s, mut sq) = (0.0f64, 0.0f64);
            for c in 0..c_dim {
                let v = m[r * c_dim + c] as f64;
                s += v;
                sq += v * v;
            }
            let n = c_dim as f64;
            let std = (sq / n - (s / n) * (s / n)).max(0.0).sqrt().clamp(1e-3, 1e3);
            lr[r] = clamp(lr[r] + std.ln());
        }
        let cand = cur(&lc, &lr);
        let imb = imbalance(&cand);
        if imb < best_imb {
            best_imb = imb;
            best = cand;
            blc = lc.clone();
            blr = lr.clone();
        }
    }
    let s_col: Vec<f32> = blc.iter().map(|&x| x.exp() as f32).collect();
    let s_row: Vec<f32> = blr.iter().map(|&x| x.exp() as f32).collect();
    // Per-row (per-channel) 4-bit min/max on the balanced tile, then dequant
    // with the absorbed row scale + per-col scale.
    let mut out = vec![0.0f32; r_dim * c_dim];
    for r in 0..r_dim {
        let row = &best[r * c_dim..r * c_dim + c_dim];
        let lo = row.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = ((hi - lo) / 15.0).max(1e-8);
        let scale_abs = scale * s_row[r];
        let zp_abs = lo * s_row[r];
        for c in 0..c_dim {
            let q = (((best[r * c_dim + c] - lo) / scale).round()).clamp(0.0, 15.0);
            out[r * c_dim + c] = (q * scale_abs + zp_abs) * s_col[c];
        }
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: perplexity <model> <corpus> [--ctx N] [--warmup N] [--offset N]");
    let corpus_path = args
        .next()
        .expect("usage: perplexity <model> <corpus> [--ctx N] [--warmup N] [--offset N]");

    let mut ctx_len: usize = 2048;
    let mut warmup: usize = 8;
    let mut offset: usize = 0;
    let mut kv_mode: String = "q8".to_string();

    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--ctx" => ctx_len = val.parse().unwrap(),
            "--warmup" => warmup = val.parse().unwrap(),
            "--offset" => offset = val.parse().unwrap(),
            "--kv-mode" => kv_mode = val,
            _ => panic!("unknown flag: {flag}"),
        }
    }
    let kvarn_sim = std::env::var("HIPFIRE_KVARN_SIM").as_deref() == Ok("1");
    if kvarn_sim {
        eprintln!("KVarN-K sim ON: degrading K per GROUP={KVARN_GROUP} (Sinkhorn+4bit), V lossless");
    }
    assert!(
        ctx_len > warmup + 4,
        "ctx must exceed warmup by enough to score"
    );

    // Tokenizer.encode is O(N) at best, often slow on multi-MB inputs.
    // Read enough chars to safely cover offset+ctx tokens at ~3 char/token,
    // capped to corpus length. 8x slack covers heavy non-ASCII / wikitext markup.
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
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");

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

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");
    eprintln!("Loading weights from {model_path}...");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");

    let kv_max = window.len() + 16;
    eprintln!("KV mode: {kv_mode}");
    let mut kv_cache = match kv_mode.as_str() {
        // fp32 KV — the substrate for the KVarN sim (HIPFIRE_KVARN_SIM=1
        // degrades K in-place per GROUP; plain f32 is the lossless baseline).
        "f32" | "fp16" => KvCache::new_gpu(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_max,
        )
        .unwrap(),
        "q8" => KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_max,
        )
        .unwrap(),
        "asym4" => KvCache::new_gpu_asym4(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_max,
        )
        .unwrap(),
        "asym3" => KvCache::new_gpu_asym3(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_max,
        )
        .unwrap(),
        "asym2" => {
            let is_kv_layer: Vec<bool> = config
                .layer_types
                .iter()
                .map(|t| *t == qwen35::LayerType::FullAttention)
                .collect();
            KvCache::new_gpu_asym2_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap()
        }
        "fwht4" => {
            let is_kv_layer: Vec<bool> = config
                .layer_types
                .iter()
                .map(|t| *t == qwen35::LayerType::FullAttention)
                .collect();
            KvCache::new_gpu_fwht4_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap()
        }
        "fwht3" => {
            let is_kv_layer: Vec<bool> = config
                .layer_types
                .iter()
                .map(|t| *t == qwen35::LayerType::FullAttention)
                .collect();
            KvCache::new_gpu_fwht3_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap()
        }
        "fwht2" => {
            let is_kv_layer: Vec<bool> = config
                .layer_types
                .iter()
                .map(|t| *t == qwen35::LayerType::FullAttention)
                .collect();
            KvCache::new_gpu_fwht2_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap()
        }
        other => {
            panic!("unknown --kv-mode: {other} (q8, asym4, asym3, asym2, fwht4, fwht3, fwht2)")
        }
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();

    let mut total_nll: f64 = 0.0;
    let mut scored: usize = 0;
    let t0 = Instant::now();

    for (pos, &tok) in window.iter().enumerate().take(window.len() - 1) {
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            tok,
            pos,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
        )
        .expect("forward");

        // KVarN KV sim (HIPFIRE_KVARN_SIM=1): when a GROUP of K finishes, degrade
        // those tokens' K in-place through Sinkhorn variance-norm + 4-bit + dequant
        // (CPU-flush, the v1 strategy). Causal: later positions then attend to the
        // degraded K. fp32 V kept lossless (KVarN's focus is K). Gives the KVarN-K
        // PPL verdict without the GPU Sinkhorn kernel.
        if kvarn_sim && (pos + 1) % KVARN_GROUP == 0 {
            let start = pos + 1 - KVARN_GROUP;
            degrade_k_group_kvarn(
                &mut gpu,
                &kv_cache,
                start,
                KVARN_GROUP,
                config.n_kv_heads,
                config.head_dim,
            );
        }

        if pos < warmup {
            continue;
        }

        let logits = gpu.download_f32(&scratch.logits).unwrap();
        let target = window[pos + 1] as usize;
        let nll = neg_log_softmax_at(&logits, target);
        if !nll.is_finite() {
            eprintln!("  warn: non-finite NLL at pos={pos} target={target}, skipping");
            continue;
        }
        total_nll += nll as f64;
        scored += 1;

        if scored == 1 || scored % 256 == 0 {
            let avg_nll = total_nll / scored as f64;
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = scored as f64 / elapsed.max(1e-9);
            eprintln!(
                "  pos={:5} scored={:5} nll/tok={:.4} ppl={:.3} ({:.1} tok/s)",
                pos,
                scored,
                avg_nll,
                avg_nll.exp(),
                rate,
            );
        }
    }

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
        "Tokens:   offset={offset} ctx={} warmup={warmup}",
        window.len()
    );
    println!("Scored:   {scored}");
    println!("NLL/tok:  {:.10}", avg_nll);
    println!("PPL:      {:.4}", ppl);
    println!(
        "Elapsed:  {:.1}s ({:.1} tok/s)",
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
